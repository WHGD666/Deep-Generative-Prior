"""色彩对齐：把输出的逐通道均值/方差向输入靠拢。

原理：GT 与输入是同一场景的同分辨率干净版，全局色调一致；
扩散输出常见整体色调漂移，而有参考指标（PSNR/SSIM）对色调极其敏感。
"""

from __future__ import annotations

import numpy as np

from src.utils.image_ops import to_float01, to_uint8


def match_mean_std(img: np.ndarray, ref: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """逐通道均值方差对齐（float 域）。

    Args:
        img: 待对齐图 (H, W, 3) float32 [0,1] 或 uint8。
        ref: 参考图（同尺寸）。
        strength: 对齐强度 [0,1]，1=完全对齐。
    """
    f, r = to_float01(img), to_float01(ref)
    out = np.empty_like(f)
    for c in range(3):
        m_i, s_i = f[..., c].mean(), f[..., c].std() + 1e-8
        m_r, s_r = r[..., c].mean(), r[..., c].std() + 1e-8
        aligned = (f[..., c] - m_i) * (s_r / s_i) + m_r
        out[..., c] = (1.0 - strength) * f[..., c] + strength * aligned
    return out


def apply(img: np.ndarray, ref: np.ndarray, cfg: dict) -> np.ndarray:
    """配置入口。cfg: {enabled, strength, mode}"""
    if ref.shape[:2] != img.shape[:2]:
        raise ValueError("色彩对齐要求 ref 与 img 同尺寸")
    mode = cfg.get("mode", "mean_std")
    if mode != "mean_std":
        raise ValueError(f"未知色彩对齐模式: {mode}（当前仅支持 mean_std）")
    strength = float(cfg.get("strength", 1.0))
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength 须在 [0,1], 得到 {strength}")
    return to_uint8(match_mean_std(img, ref, strength))
