"""CPU 端单元测试通用工具（不依赖 torch/GPU）。"""

from __future__ import annotations

import numpy as np

TINY_H, TINY_W = 96, 128  # 能容纳 tile=64/overlap=16 与 tile=32/overlap=8 的最小整除图


def make_img(h: int = TINY_H, w: int = TINY_W, seed: int = 7) -> np.ndarray:
    """带结构（渐变+色块）的合成 uint8 RGB 图，比随机噪声更能暴露融合/对齐缺陷。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            (xx * 255 // max(w - 1, 1)),
            (yy * 255 // max(h - 1, 1)),
            np.full((h, w), 128),
        ],
        axis=-1,
    ).astype(np.uint8)
    blocks = rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8)
    bh, bw = h // 4, w // 4
    for i in range(4):
        for j in range(4):
            base[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw] = blocks[i, j]
    return base
