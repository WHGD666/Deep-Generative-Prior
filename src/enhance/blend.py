"""羽化融合工具（供分块引擎与人脸分支复用）。"""

from __future__ import annotations

import numpy as np

from src.preprocess.tiling import feather_weight


def feathered_paste(
    canvas: np.ndarray,
    wsum: np.ndarray,
    tile: np.ndarray,
    y0: int,
    x0: int,
    overlap: int,
) -> None:
    """把一块（输出空间的）tile 以羽化权重累加进 canvas（就地修改）。

    Args:
        canvas: float32 (H, W, 3) 累加画布。
        wsum: float32 (H, W) 权重和累加画布。
        tile: uint8/float 块。
        overlap: 该块的羽化宽度（输出空间像素）。
    """
    th, tw = tile.shape[:2]
    w = feather_weight(th, tw, overlap)[..., None]
    tile_f = tile.astype(np.float32) if tile.dtype != np.float32 else tile
    canvas[y0 : y0 + th, x0 : x0 + tw] += tile_f * w
    wsum[y0 : y0 + th, x0 : x0 + tw] += w[..., 0]


def blend_two(base: np.ndarray, patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """按单通道 mask（0~1，1 表示完全取 patch）融合两张同尺寸图。"""
    if base.shape[:2] != patch.shape[:2] or base.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"融合尺寸不一致: base={base.shape[:2]}, patch={patch.shape[:2]}, mask={mask.shape[:2]}"
        )
    m = mask.astype(np.float32)[..., None]
    out = base.astype(np.float32) * (1.0 - m) + patch.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)
