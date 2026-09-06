"""轻度去模糊（实验性，默认关闭）。

USM（Unsharp Mask）锐化对轻度发软（case2/case5）有收益，
但过强会放大噪声并在扩散输出上叠加振铃——待 baseline 后评估是否保留。

配置（configs/defaults/preprocess.yaml -> deblur）：
    enabled: false
    usm_radius: 2.0
    usm_percent: 60
    usm_threshold: 2
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def deblur(
    img: np.ndarray, usm_radius: float = 2.0, usm_percent: int = 60, usm_threshold: int = 2
) -> np.ndarray:
    """USM 锐化，返回 uint8 RGB。参数名与 preprocess.yaml 的键一一对应（语义同 PIL UnsharpMask）。"""
    out = Image.fromarray(img).filter(
        ImageFilter.UnsharpMask(radius=usm_radius, percent=usm_percent, threshold=usm_threshold)
    )
    return np.asarray(out, dtype=np.uint8)
