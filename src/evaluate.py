"""
WarnaIn - Evaluation and Metric Benchmark
Calculates quantitative metrics (PSNR, SSIM, MAE) on the test split,
and evaluates performance across various saturation degradation levels.
"""

import os
import argparse
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch

from src.model import build_colorizer
from src.utils import (
    image_to_tensor,
    tensor_to_rgb,
    degrade_saturation,
    calculate_psnr,
    calculate_ssim,
    calculate_mae
)


def evaluate_test_set(
    model_path: str = "checkpoints/best_model.pth",
    test_dir: str = "data/splits/test",
    img_size: int = 256,
    device: str = "cpu"
) -> dict:
    device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    print(f"[INFO] Evaluating WarnaIn Model on {device}...")

    if not os.path.exists(model_path):
        print(f"[ERROR] Checkpoint not found at: {model_path}")
        return {}

    model = build_colorizer(checkpoint_path=model_path, device=str(device))
    model.eval()

    valid_exts = (".jpg", ".jpeg", ".png")
    test_files = [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if os.path.splitext(f.lower())[1] in valid_exts
    ] if os.path.exists(test_dir) else []

    if len(test_files) == 0:
        print(f"[WARNING] No test images found in {test_dir}.")
        return {}

    psnr_list, ssim_list, mae_list = [], [], []

    with torch.no_grad():
        for fpath in tqdm(test_files, desc="Benchmarking Test Set"):
            orig_pil = Image.open(fpath).convert("RGB")
            orig_np = np.array(orig_pil.resize((img_size, img_size), Image.Resampling.BILINEAR))

            L_tensor, ab_target, _ = image_to_tensor(orig_np, target_size=(img_size, img_size))
            L_tensor = L_tensor.to(device)

            pred_ab = model(L_tensor)
            colorized_rgb = tensor_to_rgb(L_tensor[0], pred_ab[0])

            # Quantitative evaluation vs Ground Truth
            psnr_list.append(calculate_psnr(orig_np, colorized_rgb))
            ssim_list.append(calculate_ssim(orig_np, colorized_rgb))
            mae_list.append(calculate_mae(orig_np, colorized_rgb))

    results = {
        "num_samples": len(test_files),
        "mean_psnr": float(np.mean(psnr_list)),
        "std_psnr": float(np.std(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "std_ssim": float(np.std(ssim_list)),
        "mean_mae": float(np.mean(mae_list)),
        "std_mae": float(np.std(mae_list)),
    }

    print("\n" + "=" * 50)
    print("           WARNAIN TEST SET METRICS           ")
    print("=" * 50)
    print(f"Total Test Citra : {results['num_samples']}")
    print(f"PSNR (dB)        : {results['mean_psnr']:.2f} ± {results['std_psnr']:.2f} dB (Higher is better)")
    print(f"SSIM             : {results['mean_ssim']:.4f} ± {results['std_ssim']:.4f} (Max 1.0)")
    print(f"MAE              : {results['mean_mae']:.2f} ± {results['std_mae']:.2f} (Lower is better)")
    print("=" * 50 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WarnaIn Model Evaluation")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--test_dir", type=str, default="data/splits/test")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_test_set(args.model_path, args.test_dir, args.img_size, args.device)
