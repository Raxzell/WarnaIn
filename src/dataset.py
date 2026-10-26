"""
WarnaIn - PyTorch Dataset and Split Preparation
Handles loading images, RGB to CIE Lab conversion, data augmentations,
and automatic dataset train/val/test splitting.
"""

import os
import glob
import shutil
import random
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional, Callable
import torch
from torch.utils.data import Dataset
from src.utils import rgb_to_lab, degrade_saturation


class ColorizationDataset(Dataset):
    """
    Custom PyTorch Dataset for CIE Lab Image Colorization.
    
    Returns:
        L: Tensor of shape (1, H, W), float32 in [0, 1]
        ab: Tensor of shape (2, H, W), float32 in [-1, 1]
    """
    def __init__(
        self,
        image_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        is_training: bool = True,
        degrade_prob: float = 0.0,
        max_samples: Optional[int] = None,
    ):
        super(ColorizationDataset, self).__init__()
        self.image_dir = image_dir
        self.image_size = image_size
        self.is_training = is_training
        self.degrade_prob = degrade_prob

        # Collect image paths
        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        self.image_paths: List[str] = []
        for ext in valid_exts:
            self.image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))

        self.image_paths = sorted(list(set(self.image_paths)))
        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]

        if len(self.image_paths) == 0:
            print(f"[WARNING] No images found in directory: {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        """Applies data augmentations (flips, slight rotations)."""
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.7:
            angle = random.uniform(-10.0, 10.0)
            image = image.rotate(angle, resample=Image.BILINEAR)
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Training augmentations
        if self.is_training:
            image = self._apply_augmentations(image)

        # Resize to model standard input size
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        img_np = np.array(image, dtype=np.uint8)

        # Optional saturation degradation experiment / augmentation
        if self.is_training and self.degrade_prob > 0.0 and random.random() < self.degrade_prob:
            factor = random.uniform(0.1, 0.8)
            img_np = degrade_saturation(img_np, factor)

        # Convert to CIE Lab
        L_norm, ab_norm = rgb_to_lab(img_np)

        # Convert to torch.Tensor: (C, H, W)
        L_tensor = torch.from_numpy(L_norm.transpose(2, 0, 1)).float()
        ab_tensor = torch.from_numpy(ab_norm.transpose(2, 0, 1)).float()

        return L_tensor, ab_tensor


def prepare_dataset_splits(
    raw_dir: str = "data/raw",
    output_dir: str = "data/splits",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy_files: bool = True
) -> dict:
    """
    Splits COCO 2017 validation set into train (80%), val (10%), test (10%).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    valid_exts = ("*.jpg", "*.jpeg", "*.png")
    all_images = []
    for ext in valid_exts:
        all_images.extend(glob.glob(os.path.join(raw_dir, ext)))
        all_images.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))

    all_images = sorted(list(set(all_images)))
    total_imgs = len(all_images)
    if total_imgs == 0:
        print(f"[INFO] No images in {raw_dir} yet. Ready to split once downloaded.")
        return {"total": 0, "train": 0, "val": 0, "test": 0}

    random.seed(seed)
    random.shuffle(all_images)

    n_train = int(total_imgs * train_ratio)
    n_val = int(total_imgs * val_ratio)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:]
    }

    for split_name, split_files in splits.items():
        split_folder = os.path.join(output_dir, split_name)
        os.makedirs(split_folder, exist_ok=True)
        if copy_files:
            for file_path in split_files:
                dest = os.path.join(split_folder, os.path.basename(file_path))
                if not os.path.exists(dest):
                    shutil.copy2(file_path, dest)

    summary = {
        "total": total_imgs,
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
    }
    print(f"[SUCCESS] Dataset splits prepared: {summary}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split dataset into train, val, test")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw COCO images")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Destination splits path")
    args = parser.parse_args()
    prepare_dataset_splits(args.raw_dir, args.splits_dir)
