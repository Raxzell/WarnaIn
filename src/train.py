"""
WarnaIn - Training Pipeline
Implements training loop, validation, early stopping, LR scheduling,
and model checkpointing for ResNet-based U-Net Colorizer.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import ColorizationDataset
from src.model import ResNetUNetColorizer
from src.utils import calculate_psnr, calculate_ssim, tensor_to_rgb


class ColorizationLoss(nn.Module):
    """
    Combined Loss for Colorization:
    Smooth L1 / L1 (robust to outliers) + MSE Loss (structural fidelity).
    """
    def __init__(self, l1_weight: float = 1.0, mse_weight: float = 0.5):
        super(ColorizationLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.1)
        self.mse = nn.MSELoss()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight

    def forward(self, pred_ab: torch.Tensor, target_ab: torch.Tensor) -> torch.Tensor:
        loss_l1 = self.smooth_l1(pred_ab, target_ab)
        loss_mse = self.mse(pred_ab, target_ab)
        return (self.l1_weight * loss_l1) + (self.mse_weight * loss_mse)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int
) -> float:
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{total_epochs}] Train", leave=False)
    for L_batch, ab_batch in pbar:
        L_batch = L_batch.to(device)
        ab_batch = ab_batch.to(device)

        optimizer.zero_grad()
        pred_ab = model(L_batch)
        loss = criterion(pred_ab, ab_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * L_batch.size(0)
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    model.eval()
    running_loss = 0.0
    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for L_batch, ab_batch in tqdm(dataloader, desc="Validating", leave=False):
            L_batch = L_batch.to(device)
            ab_batch = ab_batch.to(device)

            pred_ab = model(L_batch)
            loss = criterion(pred_ab, ab_batch)
            running_loss += loss.item() * L_batch.size(0)

            # Sample metrics calculation on a subset of batch for speed
            for i in range(min(4, L_batch.size(0))):
                orig_rgb = tensor_to_rgb(L_batch[i], ab_batch[i])
                pred_rgb = tensor_to_rgb(L_batch[i], pred_ab[i])
                psnr_scores.append(calculate_psnr(orig_rgb, pred_rgb))
                ssim_scores.append(calculate_ssim(orig_rgb, pred_rgb))

    val_loss = running_loss / max(1, len(dataloader.dataset))
    mean_psnr = float(np.mean(psnr_scores)) if psnr_scores else 0.0
    mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    return val_loss, mean_psnr, mean_ssim


def run_training(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[NOTICE] CUDA requested but not available in this PyTorch build. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"[INFO] WarnaIn Training Starting on device: {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Dataset & DataLoaders
    train_dataset = ColorizationDataset(
        image_dir=args.train_dir,
        image_size=(args.img_size, args.img_size),
        is_training=True,
        degrade_prob=args.degrade_prob,
        max_samples=args.max_samples if args.max_samples > 0 else None
    )
    val_dataset = ColorizationDataset(
        image_dir=args.val_dir,
        image_size=(args.img_size, args.img_size),
        is_training=False,
        max_samples=max(50, args.max_samples // 5) if args.max_samples > 0 else None
    )

    if len(train_dataset) == 0:
        print(f"[ERROR] No training images found in '{args.train_dir}'.")
        print("Please place COCO validation images in 'data/raw' and run dataset splitting first.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # 2. Model, Loss, Optimizer
    model = ResNetUNetColorizer(backbone=args.backbone, pretrained=args.pretrained).to(device)
    criterion = ColorizationLoss(l1_weight=args.l1_weight, mse_weight=args.mse_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 3. Training Loop with Checkpointing & Early Stopping
    best_val_loss = float("inf")
    patience = args.patience
    patience_counter = 0

    print(f"[INFO] Commencing {args.epochs} Epochs. Train images: {len(train_dataset)}, Val images: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_psnr": val_psnr,
                "val_ssim": val_ssim,
            }, best_path)
            print(f"  >>> Best model checkpoint saved to: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[EARLY STOPPING] Validation loss stopped improving for {patience} epochs.")
                break

    print("[SUCCESS] Training pipeline completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WarnaIn Training Script")
    parser.add_argument("--train_dir", type=str, default="data/splits/train")
    parser.add_argument("--val_dir", type=str, default="data/splits/val")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--mse_weight", type=float, default=0.5)
    parser.add_argument("--degrade_prob", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=0, help="Limit dataset size for fast testing (0 for all)")

    args = parser.parse_args()
    run_training(args)
