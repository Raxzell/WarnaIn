"""
WarnaIn v3 - Utility Functions (CIE L*a*b* [-1, 1] Standardized Pipeline)
Handles high-precision float32 CIE L*a*b* <-> RGB conversions, standardized
[-1, 1] tensor normalization for (B, 1, 224, 224) inputs and (B, 2, 224, 224)
outputs, saturation degradation experiments, category chrominance priors,
and quantitative metrics (PSNR, SSIM, MAE on ab & RGB).
"""

import numpy as np
import cv2
import torch
from typing import Tuple, Optional, Union
from PIL import Image


def rgb_to_lab(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts an RGB image (uint8 [0, 255]) to normalized float32 CIE Lab channels:
    - L_norm: shape (H, W, 1), float32 in [-1, 1] (mapped from L* in [0, 100])
    - ab_norm: shape (H, W, 2), float32 in [-1, 1] (mapped from a*, b* in [-128, 128])
    """
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    rgb_float = rgb_image.astype(np.float32) / 255.0
    lab_float = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2Lab)

    # Standardize L* [0, 100] -> [-1, 1] and a*b* [-128, 128] -> [-1, 1]
    L_norm = np.clip((lab_float[:, :, 0:1] / 50.0) - 1.0, -1.0, 1.0).astype(np.float32)
    ab_norm = np.clip(lab_float[:, :, 1:3] / 128.0, -1.0, 1.0).astype(np.float32)

    return L_norm, ab_norm


def lab_to_rgb(L_norm: np.ndarray, ab_norm: np.ndarray, l_range: str = "neg1_1") -> np.ndarray:
    """
    Reconstructs an RGB image from normalized CIE Lab channels:
    - L_norm: shape (H, W, 1) or (H, W), float32 in [-1, 1] (or [0, 1] if l_range='0_1')
    - ab_norm: shape (H, W, 2), float32 in [-1, 1]

    Returns:
        rgb: numpy array of shape (H, W, 3), uint8 in [0, 255]
    """
    if L_norm.ndim == 2:
        L_norm = L_norm[:, :, np.newaxis]

    if l_range == "0_1":
        L_true = np.clip(L_norm.astype(np.float32) * 100.0, 0.0, 100.0)
    else:
        L_true = np.clip((L_norm.astype(np.float32) + 1.0) * 50.0, 0.0, 100.0)

    ab_true = np.clip(ab_norm.astype(np.float32) * 128.0, -127.0, 127.0)

    lab_float = np.concatenate([L_true, ab_true], axis=2).astype(np.float32)
    rgb_float = cv2.cvtColor(lab_float, cv2.COLOR_Lab2RGB)
    rgb_uint8 = np.clip(np.round(rgb_float * 255.0), 0, 255).astype(np.uint8)
    return rgb_uint8


def tensor_to_rgb(L_tensor: torch.Tensor, ab_tensor: torch.Tensor) -> np.ndarray:
    """Converts PyTorch tensors L (B,1,H,W) in [-1,1] and ab (B,2,H,W) in [-1,1] to RGB uint8."""
    if L_tensor.ndim == 4:
        L_tensor = L_tensor.squeeze(0)
    if ab_tensor.ndim == 4:
        ab_tensor = ab_tensor.squeeze(0)

    L_np = L_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    ab_np = ab_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    return lab_to_rgb(L_np, ab_np, l_range="neg1_1")


def image_to_tensor(
    image: Union[np.ndarray, Image.Image],
    target_size: Optional[Tuple[int, int]] = (224, 224),
    preserve_aspect: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Preprocesses an input RGB image into normalized L* and a*b* PyTorch tensors:
    - L_tensor: (1, 1, H, W) in [-1, 1] (default 224x224 per v3 specification)
    - ab_tensor: (1, 2, H, W) in [-1, 1]
    - original_size: (orig_W, orig_H)
    """
    if isinstance(image, Image.Image):
        original_size = image.size  # (W, H)
        img_np = np.array(image.convert("RGB"))
    else:
        original_size = (image.shape[1], image.shape[0])
        img_np = image.copy()

    if target_size is not None:
        orig_w, orig_h = original_size
        if preserve_aspect and orig_w > 0 and orig_h > 0:
            short_target = min(target_size[0], target_size[1])
            scale = short_target / float(min(orig_w, orig_h))
            new_w = max(32, int(round((orig_w * scale) / 32.0) * 32))
            new_h = max(32, int(round((orig_h * scale) / 32.0) * 32))
            max_dim = short_target * 2
            if new_w > max_dim or new_h > max_dim:
                cap_scale = max_dim / float(max(new_w, new_h))
                new_w = max(32, int(round((new_w * cap_scale) / 32.0) * 32))
                new_h = max(32, int(round((new_h * cap_scale) / 32.0) * 32))
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_np = cv2.resize(img_np, target_size, interpolation=cv2.INTER_AREA)

    L_norm, ab_norm = rgb_to_lab(img_np)

    L_tensor = torch.from_numpy(L_norm.transpose(2, 0, 1)).unsqueeze(0).float()
    ab_tensor = torch.from_numpy(ab_norm.transpose(2, 0, 1)).unsqueeze(0).float()

    return L_tensor, ab_tensor, original_size


def apply_category_chroma_prior(
    L_norm: np.ndarray,
    ab_norm: np.ndarray,
    category: str = "general",
) -> np.ndarray:
    """
    Applies domain-specific CIE Lab chrominance refinement.
    Handles L_norm in [-1, 1] seamlessly by mapping internal luminance mask to [0, 1].
    """
    cat = (category or "general").lower()
    if cat == "general":
        return ab_norm

    raw_l = L_norm[:, :, 0] if L_norm.ndim == 3 else L_norm
    # Convert L from [-1, 1] to [0, 1] for luminance zone masks
    lum = np.clip((raw_l + 1.0) * 0.5, 0.0, 1.0)
    h, w = lum.shape
    a = ab_norm[:, :, 0].copy()
    b = ab_norm[:, :, 1].copy()

    if cat == "animals":
        false_blue_mask = b < 0.0
        b[false_blue_mask] = b[false_blue_mask] * 0.15 + 0.04 * (1.0 - lum[false_blue_mask])

        fur_mask = np.clip(1.0 - np.abs(lum - 0.45) / 0.38, 0.0, 1.0)
        b = b + fur_mask * 0.08
        a = a + fur_mask * 0.035

        yy, xx = np.ogrid[:h, :w]
        center_weight = np.exp(-(((yy - h * 0.42) / (h * 0.28)) ** 2 + ((xx - w * 0.5) / (w * 0.28)) ** 2))
        nose_tone = (lum > 0.45) & (lum < 0.82)
        a = a + center_weight * nose_tone * 0.045

        white_fur = np.clip((lum - 0.80) / 0.20, 0.0, 1.0)
        a = a * (1.0 - 0.45 * white_fur)
        b = b * (1.0 - 0.35 * white_fur)

    elif cat == "architecture":
        sky_blue = (b < -0.02) & (lum > 0.35)
        b[sky_blue] = b[sky_blue] * 1.15

        facade_lum = np.clip(1.0 - np.abs(lum - 0.48) / 0.35, 0.0, 1.0)
        warm_facade = (a > -0.02) & (b > -0.02)
        a[warm_facade] = a[warm_facade] + facade_lum[warm_facade] * 0.035
        b[warm_facade] = b[warm_facade] + facade_lum[warm_facade] * 0.045

        bright_neutral = lum > 0.78
        a[bright_neutral] = a[bright_neutral] * 0.85
        b[bright_neutral] = b[bright_neutral] * 0.90

    elif cat == "landscapes":
        blue_mask = b < -0.02
        b[blue_mask] = b[blue_mask] * 1.18
        green_mask = (a < 0.0) & (b > 0.0)
        a[green_mask] = a[green_mask] * 1.15
        b[green_mask] = b[green_mask] * 1.12

    elif cat == "portraits":
        cold_mask = b < 0.0
        b[cold_mask] = b[cold_mask] * 0.55
        green_cast = a < -0.02
        a[green_cast] = a[green_cast] * 0.60
        skin_lum = np.clip(1.0 - np.abs(lum - 0.58) / 0.32, 0.0, 1.0)
        a = a + skin_lum * 0.04
        b = b + skin_lum * 0.05

    elif cat == "objects":
        chroma_mag = np.sqrt(a ** 2 + b ** 2) + 1e-6
        boost = np.power(np.clip(chroma_mag / 0.5, 0.0, 2.0), -0.22)
        boost = np.clip(boost, 1.0, 1.45)
        a = a * boost
        b = b * boost

    return np.clip(np.stack([a, b], axis=2), -1.0, 1.0).astype(np.float32)


def degrade_saturation(rgb_image: np.ndarray, saturation_factor: float = 0.0) -> np.ndarray:
    """
    Degrades color saturation of an RGB image to simulate faded vintage photos or grayscale.
    """
    saturation_factor = float(np.clip(saturation_factor, 0.0, 1.0))
    if saturation_factor >= 0.999:
        return rgb_image.copy()

    L_norm, ab_norm = rgb_to_lab(rgb_image)
    ab_degraded = ab_norm * saturation_factor
    return lab_to_rgb(L_norm, ab_degraded)


def enhance_luminance_contrast(rgb_or_gray: np.ndarray, clip_limit: float = 1.5) -> np.ndarray:
    """Applies subtle CLAHE on the L channel."""
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


def calculate_mae(img1: np.ndarray, img2: np.ndarray, on_ab: bool = True) -> float:
    """
    Calculates Mean Absolute Error (MAE).
    If on_ab=True, computes MAE on the CIE a*b* chrominance channels in scale [0, 128]
    as specified in the v3 benchmark guidelines.
    """
    if on_ab and img1.ndim == 3 and img2.ndim == 3:
        _, ab1 = rgb_to_lab(img1)
        _, ab2 = rgb_to_lab(img2)
        # Scale normalized [-1, 1] difference to CIE Lab chrominance units [0, 128]
        return float(np.mean(np.abs(ab1 - ab2)) * 128.0)
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
    lum_01 = gray.astype(np.float32) / 255.0
    lum_neg1_1 = lum_01 * 2.0 - 1.0
    y_grad = np.tile(np.linspace(1.0, 0.0, h)[:, np.newaxis], (1, w))

    a_pred = np.sin(lum_01 * np.pi) * 0.12 + (warmth * 0.15)
    b_pred = (lum_01 - 0.4) * 0.25 + (warmth * 0.12)
    sky_mask = (y_grad > 0.5) & (lum_01 > 0.45)
    b_pred[sky_mask] -= (y_grad[sky_mask] - 0.5) * 0.35

    ab_pred = np.clip(np.stack([a_pred * vibrancy, b_pred * vibrancy], axis=2), -1.0, 1.0).astype(np.float32)
    return lab_to_rgb(lum_neg1_1[:, :, np.newaxis], ab_pred)
