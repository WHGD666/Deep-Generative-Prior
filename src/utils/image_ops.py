"""纯 numpy/PIL 图像操作（可本地单测，不依赖 torch/cv2）。"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def to_float01(img: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 [0,1]。"""
    return img.astype(np.float32) / 255.0


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float [0,1] -> uint8（四舍五入并裁剪，确定性）。"""
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """高斯模糊（float 或 uint8 RGB 均可，返回 float32 [0,1]）。

    用 PIL 实现（半径≈sigma），避免引入 cv2 依赖。
    """
    if sigma <= 0:
        return to_float01(img) if img.dtype == np.uint8 else img.astype(np.float32)
    pil = Image.fromarray(img if img.dtype == np.uint8 else to_uint8(img))
    out = pil.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.asarray(out, dtype=np.float32) / 255.0


def resize_to(img: np.ndarray, height: int, width: int) -> np.ndarray:
    """精确缩放到目标尺寸（LANCZOS；尺寸不符即抛错由调用方守卫）。"""
    pil = Image.fromarray(img if img.dtype == np.uint8 else to_uint8(img))
    out = pil.resize((int(width), int(height)), Image.LANCZOS)
    return np.asarray(out, dtype=np.uint8)


def downscale_int(img: np.ndarray, factor: int) -> np.ndarray:
    """整数倍降采样（用于 ×k 后端的预缩放），尺寸向下取整。"""
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    return resize_to(img, h // factor, w // factor)
