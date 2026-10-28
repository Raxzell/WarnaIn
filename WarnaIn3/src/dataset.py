"""
WarnaIn v3 - PyTorch Dataset and Category-Aware Split Preparation
Handles loading images, RGB to CIE Lab conversion with standardized [-1, 1] range,
and automatic path resolution to the parent workspace's COCO dataset splits.
"""

import os
import glob
import random
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset

from src.utils import rgb_to_lab, degrade_saturation

CATEGORIES = ("general", "animals", "architecture", "landscapes", "portraits", "objects")
VALID_CATEGORIES = CATEGORIES
CATEGORY_SUBFOLDERS = ("animals", "architecture", "landscapes", "portraits", "objects")


def resolve_data_path(relative_path: str = "data/splits/test") -> str:
    """
    Resolves dataset path by checking inside WarnaIn3 first, then falling back
    to the parent d:/WarnaIn directory so large COCO datasets are never duplicated.
    """
    v3_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parent_root = os.path.abspath(os.path.join(v3_root, ".."))

    cand_v3 = os.path.join(v3_root, relative_path)
    if os.path.exists(cand_v3):
        return cand_v3

    cand_parent = os.path.join(parent_root, relative_path)
    if os.path.exists(cand_parent):
        return cand_parent

    return cand_v3


class ColorizationDataset(Dataset):
    """
    Custom PyTorch Dataset for CIE Lab Image Colorization (v3 Specification).
    Returns:
        L: Tensor of shape (1, 224, 224), float32 normalized in [-1, 1]
        ab: Tensor of shape (2, 224, 224), float32 normalized in [-1, 1]
    """

    def __init__(
        self,
        image_dir: str = "data/splits/train",
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = True,
        degrade_prob: float = 0.0,
        max_samples: Optional[int] = None,
        category: str = "general",
    ):
        super(ColorizationDataset, self).__init__()
        self.image_size = image_size
        self.is_training = is_training
        self.degrade_prob = degrade_prob
        self.category = category.lower() if category else "general"

        resolved_dir = image_dir if os.path.exists(image_dir) else resolve_data_path(image_dir)

        if self.category != "general":
            cat_dir = os.path.join(resolved_dir, self.category)
            search_dir = cat_dir if os.path.isdir(cat_dir) else resolved_dir
        else:
            search_dir = resolved_dir

        self.image_dir = search_dir

        valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        self.image_paths: List[str] = []
        for ext in valid_exts:
            self.image_paths.extend(glob.glob(os.path.join(search_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(search_dir, "**", ext), recursive=True))

        self.image_paths = sorted(list(set(self.image_paths)))
        if max_samples is not None and max_samples > 0:
            self.image_paths = self.image_paths[:max_samples]

        if len(self.image_paths) == 0:
            print(f"[WARNING] No images found in directory: {search_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.7:
            angle = random.uniform(-10.0, 10.0)
            image = image.rotate(angle, resample=Image.BILINEAR)
        return image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.is_training:
            image = self._apply_augmentations(image)

        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        img_np = np.array(image, dtype=np.uint8)

        if self.is_training and self.degrade_prob > 0.0 and random.random() < self.degrade_prob:
            factor = random.uniform(0.1, 0.8)
            img_np = degrade_saturation(img_np, factor)

        L_norm, ab_norm = rgb_to_lab(img_np)

        L_tensor = torch.from_numpy(L_norm.transpose(2, 0, 1)).float()
        ab_tensor = torch.from_numpy(ab_norm.transpose(2, 0, 1)).float()

        return L_tensor, ab_tensor
