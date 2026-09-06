"""轻量去噪（确定性，无 GPU 依赖）。

定位：扩散模型前的输入净化。val 观察到 case1 有重彩色噪点；
若不抑制，扩散模型会把噪声当"细节"放大成幻觉纹理。
注意：赛题的 Diffusion 架构合规性由 enhance 层承担，本层只是传统预处理。

配置（configs/defaults/preprocess.yaml -> denoise）：
    enabled: true
    gaussian_sigma: 0.8   # 0 关闭；与后端预缩放叠加时注意平滑量
    median_size: 0        # 0 关闭；奇数，如 3（对椒盐类噪点有效）
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from src.utils.image_ops import gaussian_blur, to_float01, to_uint8


def denoise(img: np.ndarray, gaussian_sigma: float = 0.0, median_size: int = 0) -> np.ndarray:
    """按配置执行去噪，返回 uint8 RGB。参数名与 preprocess.yaml 的键一一对应。

    Args:
        img: uint8 RGB。
        gaussian_sigma: 高斯 sigma（像素）；<=0 跳过高斯。
        median_size: 中值滤波核（奇数）；<=1 跳过中值。
    """
    out = img
    if median_size and median_size > 1:
        if median_size % 2 == 0:
            raise ValueError(f"median_size 须为奇数, 得到 {median_size}")
        out = np.asarray(
            Image.fromarray(out).filter(ImageFilter.MedianFilter(median_size)),
            dtype=np.uint8,
        )
    if gaussian_sigma and gaussian_sigma > 0:
        out = to_uint8(gaussian_blur(out, gaussian_sigma))
    return out
