"""
WarnaIn - PyTorch Dataset and Category-Aware Split Preparation
Handles loading images, RGB to CIE Lab conversion, data augmentations,
and automatic dataset train/val/test splitting into 4 semantic categories:
- animals/
- landscapes/
- portraits/
- objects/
(Plus 'general' mode which aggregates all categories).
"""

import os
import glob
import json
import shutil
import random
from collections import defaultdict
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset
from src.utils import rgb_to_lab, degrade_saturation

CATEGORIES = ("general", "animals", "architecture", "landscapes", "portraits", "objects")
VALID_CATEGORIES = CATEGORIES
CATEGORY_SUBFOLDERS = ("animals", "architecture", "landscapes", "portraits", "objects")


class ColorizationDataset(Dataset):
    """
    Custom PyTorch Dataset for CIE Lab Image Colorization.
    Supports loading all images ('general') or a specific semantic category
    ('animals', 'landscapes', 'portraits', 'objects').

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
        category: str = "general",
    ):
        super(ColorizationDataset, self).__init__()
        self.image_size = image_size
        self.is_training = is_training
        self.degrade_prob = degrade_prob
        self.category = category.lower() if category else "general"

        # If a specific category subfolder exists, scope to that subfolder
        if self.category != "general":
            cat_dir = os.path.join(image_dir, self.category)
            search_dir = cat_dir if os.path.isdir(cat_dir) else image_dir
        else:
            search_dir = image_dir

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


def classify_coco_images(annotations_dir: str = "data/annotations") -> dict:
    """
    Uses official COCO 2017 instances_val2017.json and captions_val2017.json
    to map each image filename (e.g. '000000000139.jpg') to one of:
    'animals', 'landscapes', 'portraits', 'objects'.
    """
    inst_path = os.path.join(annotations_dir, "instances_val2017.json")
    caps_path = os.path.join(annotations_dir, "captions_val2017.json")

    if not os.path.isfile(inst_path):
        print(f"[WARNING] COCO annotation file not found at {inst_path}.")
        return {}

    with open(inst_path, "r", encoding="utf-8") as f:
        inst = json.load(f)

    caps_by_img = defaultdict(list)
    if os.path.isfile(caps_path):
        with open(caps_path, "r", encoding="utf-8") as f:
            caps = json.load(f)
        for c in caps.get("annotations", []):
            caps_by_img[c["image_id"]].append(c.get("caption", "").lower())

    cat_id_to_name = {c["id"]: c["name"] for c in inst.get("categories", [])}
    anns_by_img = defaultdict(list)
    for ann in inst.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)

    animal_names = {
        "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe",
    }
    landscape_keywords = {
        "beach", "ocean", "sea", "wave", "waves", "mountain", "mountains",
        "hill", "hills", "forest", "woods", "lake", "river", "sunset",
        "sunrise", "sky", "clouds", "field", "pasture", "meadow", "shore",
        "coast", "island", "waterfall", "valley", "desert", "canyon",
    }
    architecture_keywords = {
        "building", "buildings", "house", "houses", "church", "cathedral",
        "tower", "castle", "bridge", "bridges", "temple", "monument", "palace",
        "facade", "architecture", "skyscraper", "skyscrapers", "clock",
        "cottage", "barn", "apartment", "apartments", "arch", "ruins",
        "pagoda", "lighthouse", "pavilion", "mansion", "dome", "cathedrals",
        "brick", "stone building"
    }

    filename_to_category = {}
    for img in inst.get("images", []):
        img_id = img["id"]
        fname = img["file_name"]
        img_area = max(1, img.get("width", 1) * img.get("height", 1))
        anns = anns_by_img[img_id]
        cap_text = " ".join(caps_by_img[img_id])
        words = set(cap_text.replace(".", " ").replace(",", " ").split())

        animal_area = sum(a["area"] for a in anns if cat_id_to_name.get(a["category_id"]) in animal_names)
        person_area = sum(a["area"] for a in anns if cat_id_to_name.get(a["category_id"]) == "person")
        object_area = sum(
            a["area"]
            for a in anns
            if cat_id_to_name.get(a["category_id"]) not in animal_names
            and cat_id_to_name.get(a["category_id"]) != "person"
        )

        has_landscape_word = bool(words.intersection(landscape_keywords))
        has_architecture_word = bool(words.intersection(architecture_keywords))

        if animal_area > 0.02 * img_area or (animal_area > 0 and animal_area >= person_area):
            category = "animals"
        elif person_area > 0.08 * img_area and person_area >= object_area:
            category = "portraits"
        elif has_architecture_word and person_area < 0.20 * img_area and animal_area == 0:
            category = "architecture"
        elif has_landscape_word and person_area < 0.20 * img_area and object_area < 0.25 * img_area:
            category = "landscapes"
        else:
            category = "objects"

        filename_to_category[fname] = category

    return filename_to_category


def prepare_dataset_splits(
    raw_dir: str = "data/raw",
    output_dir: str = "data/splits",
    annotations_dir: str = "data/annotations",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Splits COCO 2017 validation set into train (80%), val (10%), test (10%),
    and organizes each split into 4 category subfolders:
      - animals/
      - landscapes/
      - portraits/
      - objects/
    If files already exist flat inside data/splits/{train,val,test}, moves them
    into their category subfolders in-place for instant execution.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    filename_to_cat = classify_coco_images(annotations_dir)

    # Check if splits already exist and just need categorization
    existing_split_files = sum(
        len(glob.glob(os.path.join(output_dir, s, "**", "*.jpg"), recursive=True))
        for s in ("train", "val", "test")
    )

    stats = {s: defaultdict(int) for s in ("train", "val", "test")}

    if existing_split_files > 0 and filename_to_cat:
        print(f"[INFO] Organizing {existing_split_files} existing split images into {len(CATEGORY_SUBFOLDERS)} category folders...")
        for split_name in ("train", "val", "test"):
            split_dir = os.path.join(output_dir, split_name)
            for cat in CATEGORY_SUBFOLDERS:
                os.makedirs(os.path.join(split_dir, cat), exist_ok=True)

            # Check all existing images in split_dir (flat or already inside category subfolders)
            all_files = glob.glob(os.path.join(split_dir, "**", "*.jpg"), recursive=True)
            for fpath in all_files:
                fname = os.path.basename(fpath)
                cat = filename_to_cat.get(fname, "objects")
                dest = os.path.join(split_dir, cat, fname)
                if os.path.abspath(fpath) != os.path.abspath(dest):
                    shutil.move(fpath, dest)

            # Count categorized files
            for cat in CATEGORY_SUBFOLDERS:
                c_count = len(glob.glob(os.path.join(split_dir, cat, "*.jpg")))
                stats[split_name][cat] = c_count

        summary = {s: dict(stats[s]) for s in ("train", "val", "test")}
        print(f"[SUCCESS] Categorized splits ready: {summary}")
        return summary

    # Otherwise, build splits from raw_dir
    valid_exts = ("*.jpg", "*.jpeg", "*.png")
    all_images = []
    for ext in valid_exts:
        all_images.extend(glob.glob(os.path.join(raw_dir, ext)))
        all_images.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))

    all_images = sorted(list(set(all_images)))
    total_imgs = len(all_images)
    if total_imgs == 0:
        print(f"[INFO] No images in {raw_dir} yet.")
        return {}

    random.seed(seed)
    random.shuffle(all_images)

    n_train = int(total_imgs * train_ratio)
    n_val = int(total_imgs * val_ratio)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train : n_train + n_val],
        "test": all_images[n_train + n_val :],
    }

    for split_name, split_files in splits.items():
        for cat in CATEGORY_SUBFOLDERS:
            os.makedirs(os.path.join(output_dir, split_name, cat), exist_ok=True)

        for file_path in split_files:
            fname = os.path.basename(file_path)
            cat = filename_to_cat.get(fname, "objects")
            dest = os.path.join(output_dir, split_name, cat, fname)
            if not os.path.exists(dest):
                shutil.copy2(file_path, dest)
            stats[split_name][cat] += 1

    summary = {s: dict(stats[s]) for s in ("train", "val", "test")}
    print(f"[SUCCESS] Categorized dataset splits prepared: {summary}")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into categorized train, val, test")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Path to raw COCO images")
    parser.add_argument("--splits_dir", type=str, default="data/splits", help="Destination splits path")
    parser.add_argument("--annotations_dir", type=str, default="data/annotations", help="Path to COCO annotations")
    args = parser.parse_args()
    prepare_dataset_splits(args.raw_dir, args.splits_dir, args.annotations_dir)
