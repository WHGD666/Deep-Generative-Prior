"""低频回填：把输入图的低频分量（色彩/亮度/大结构）回填到输出。

数学形式（strength=1 时完整回填）：
    result = out + strength * ( low(ref) - low(out) )
即输出的低频被替换为参考（输入）的低频，高频细节保留自输出。
对 PSNR/SSIM 的收益来自全局色调/亮度与 GT 对齐；细节仍来自扩散模型。
"""

from __future__ import annotations

import numpy as np

from src.utils.image_ops import gaussian_blur, to_float01, to_uint8


def backfill(img: np.ndarray, ref: np.ndarray, sigma: float, strength: float = 1.0) -> np.ndarray:
    """float 域低频回填，返回 float32 [0,1]。

    Args:
        img: 增强输出 (H, W, 3)。
        ref: 参考图（低频来源），同尺寸。
        sigma: 低频截止（高斯 sigma，像素）。过大=回填过多结构，过小=仅色彩。
        strength: 回填强度 [0,1]。
    """
    f, r = to_float01(img), to_float01(ref)
    low_r = gaussian_blur(ref, sigma)
    low_f = gaussian_blur(img, sigma)
    return f + strength * (low_r - low_f)


def apply(img: np.ndarray, ref: np.ndarray, cfg: dict) -> np.ndarray:
    """配置入口。cfg: {enabled, sigma, strength}"""
    if ref.shape[:2] != img.shape[:2]:
        raise ValueError("低频回填要求 ref 与 img 同尺寸")
    sigma = float(cfg.get("sigma", 16.0))
    strength = float(cfg.get("strength", 1.0))
    if sigma <= 0:
        raise ValueError(f"sigma 须 > 0, 得到 {sigma}")
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength 须在 [0,1], 得到 {strength}")
    out_f = backfill(img, ref, sigma=sigma, strength=strength)
    return to_uint8(np.clip(out_f, 0.0, 1.0))
