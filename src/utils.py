"""
WarnaIn - Utility Functions
Handles high-precision float32 CIE Lab <-> RGB conversions, tensor transformations,
saturation degradation experiments, and quantitative metrics (PSNR, SSIM, MAE).
"""

import numpy as np
import cv2
import torch
from typing import Tuple, Optional, Union
from PIL import Image


def rgb_to_lab(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts an RGB image (uint8 [0, 255]) to normalized float32 CIE Lab channels:
    - L_norm: shape (H, W, 1), float32 in [0, 1] (representing L* in [0, 100])
    - ab_norm: shape (H, W, 2), float32 in [-1, 1] (representing a*, b* in [-128, 128])
    """
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    rgb_float = rgb_image.astype(np.float32) / 255.0
    lab_float = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2Lab)

    L_norm = np.clip(lab_float[:, :, 0:1] / 100.0, 0.0, 1.0).astype(np.float32)
    ab_norm = np.clip(lab_float[:, :, 1:3] / 128.0, -1.0, 1.0).astype(np.float32)

    return L_norm, ab_norm


def lab_to_rgb(L_norm: np.ndarray, ab_norm: np.ndarray) -> np.ndarray:
    """
    Reconstructs an RGB image from normalized CIE Lab channels:
    - L_norm: shape (H, W, 1) or (H, W), float32 in [0, 1]
    - ab_norm: shape (H, W, 2), float32 in [-1, 1]

    Returns:
        rgb: numpy array of shape (H, W, 3), uint8 in [0, 255]
    """
    if L_norm.ndim == 2:
        L_norm = L_norm[:, :, np.newaxis]

    L_true = np.clip(L_norm.astype(np.float32) * 100.0, 0.0, 100.0)
    ab_true = np.clip(ab_norm.astype(np.float32) * 128.0, -127.0, 127.0)

    lab_float = np.concatenate([L_true, ab_true], axis=2).astype(np.float32)
    rgb_float = cv2.cvtColor(lab_float, cv2.COLOR_Lab2RGB)
    rgb_uint8 = np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)
    return rgb_uint8


def tensor_to_rgb(L_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts PyTorch tensors back to RGB numpy array.
    """
    if L_tensor.ndim == 4:
        L_tensor = L_tensor.squeeze(0)
    if ab_tensor.ndim == 4:
        ab_tensor = ab_tensor.squeeze(0)

    L_np = L_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    ab_np = ab_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    return lab_to_rgb(L_np, ab_np)


def image_to_tensor(
    image: Union[np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = (256, 256),
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Preprocesses an input image (RGB) into normalized L and ab PyTorch tensors.
    """
    if isinstance(image, Image.Image):
        original_size = image.size  # (W, H)
        img_np = np.array(image.convert("RGB"))
    else:
        original_size = (image.shape[1], image.shape[0])
        img_np = image.copy()

    if target_size is not None:
        img_np = cv2.resize(img_np, target_size, interpolation=cv2.INTER_AREA)

    L, ab = rgb_to_lab(img_np)

    L_tensor = torch.from_numpy(L.transpose(2, 0, 1)).unsqueeze(0).float()
    ab_tensor = torch.from_numpy(ab.transpose(2, 0, 1)).unsqueeze(0).float()

    return L_tensor, ab_tensor, original_size


def degrade_saturation(rgb_image: np.ndarray, saturation_factor: float = 0.0) -> np.ndarray:
    """
    Degrades color saturation of an RGB image to simulate faded vintage photos or grayscale.
    Uses CIE Lab chrominance scaling so luminance (L*) stays 100% identical.
    """
    saturation_factor = float(np.clip(saturation_factor, 0.0, 1.0))
    if saturation_factor >= 0.999:
        return rgb_image.copy()

    L_norm, ab_norm = rgb_to_lab(rgb_image)
    ab_degraded = ab_norm * saturation_factor
    return lab_to_rgb(L_norm, ab_degraded)


def enhance_luminance_contrast(rgb_or_gray: np.ndarray, clip_limit: float = 1.5) -> np.ndarray:
    """
    Applies subtle CLAHE (Contrast Limited Adaptive Histogram Equalization)
    on the L channel for old/faded vintage photos.
    """
    if rgb_or_gray.ndim == 2:
        rgb = cv2.cvtColor(rgb_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = rgb_or_gray.copy()

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates Mean Absolute Error (MAE) in [0, 255]."""
    return float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32))))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates Structural Similarity Index (SSIM) between two images."""
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        if img1.ndim == 3:
            return float(ssim_fn(img1, img2, channel_axis=2, data_range=255))
        return float(ssim_fn(img1, img2, data_range=255))
    except ImportError:
        pass

    if img1.ndim == 3:
        return float(np.mean([_ssim_single_channel(img1[:, :, i], img2[:, :, i]) for i in range(3)]))
    return float(_ssim_single_channel(img1, img2))


def _ssim_single_channel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(y_true, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(y_pred, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(y_true ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(y_pred ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(y_true * y_pred, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-10
    )
    return float(np.mean(ssim_map))


def simulate_colorization_preview(rgb_or_gray: np.ndarray, warmth: float = 0.0, vibrancy: float = 1.0) -> np.ndarray:
    """Fallback heuristic colorizer if neural weights are unavailable."""
    if rgb_or_gray.ndim == 2 or (rgb_or_gray.ndim == 3 and rgb_or_gray.shape[2] == 1):
        gray = rgb_or_gray if rgb_or_gray.ndim == 2 else rgb_or_gray[:, :, 0]
    else:
        gray = cv2.cvtColor(rgb_or_gray, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    lum = gray.astype(np.float32) / 255.0
    y_grad = np.tile(np.linspace(1.0, 0.0, h)[:, np.newaxis], (1, w))

    a_pred = np.sin(lum * np.pi) * 0.12 + (warmth * 0.15)
    b_pred = (lum - 0.4) * 0.25 + (warmth * 0.12)
    sky_mask = (y_grad > 0.5) & (lum > 0.45)
    b_pred[sky_mask] -= (y_grad[sky_mask] - 0.5) * 0.35

    ab_pred = np.clip(np.stack([a_pred * vibrancy, b_pred * vibrancy], axis=2), -1.0, 1.0).astype(np.float32)
    return lab_to_rgb(lum[:, :, np.newaxis], ab_pred)
