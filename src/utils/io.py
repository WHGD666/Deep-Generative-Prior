"""统一图像读写。

约定（全管线强制）：
- 读图一律返回 uint8、RGB、形状 (H, W, 3) 的 numpy 数组；
- 默认**忽略 EXIF 方向**（respect_exif=False）：按原始像素数据读写，
  保证输出与输入分辨率逐像素严格一致（赛题红线 R4）。
  手机拍摄的 4K 图可能带旋转元数据，GT 与输入来自同一份像素数据，
  若按 EXIF 旋转会导致尺寸/内容错位。确需尊重 EXIF 时显式传参并自行
  验证尺寸一致性。
- 写 jpg 使用 quality=95 且 subsampling=0（4:4:4 无色度下采样），
  避免彩色文字/边缘被色度压缩损伤。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None  # 4K 图 (12.6MP) 远低于默认上限，但显式放开以绝后患

DEFAULT_JPEG_QUALITY = 95


def imread(path: str | Path, respect_exif: bool = False) -> np.ndarray:
    """读取图片为 uint8 RGB (H, W, 3)。

    Args:
        path: 图片路径。
        respect_exif: 是否按 EXIF 方向旋转（默认 False，见模块 docstring）。
    Returns:
        uint8 RGB 数组。
    Raises:
        ValueError: 文件不存在或无法解码。
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"图片不存在: {path}")
    try:
        with Image.open(path) as im:
            if respect_exif:
                im = ImageOps.exif_transpose(im)
            arr = np.array(im.convert("RGB"), dtype=np.uint8, copy=True)
    except OSError as e:
        raise ValueError(f"图片解码失败: {path} ({e})") from e
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"非 RGB 三通道图像: {path}, shape={arr.shape}")
    return arr


def imwrite(
    path: str | Path,
    img: np.ndarray,
    quality: int = DEFAULT_JPEG_QUALITY,
) -> Path:
    """写出 uint8 RGB 数组为 jpg。

    Args:
        path: 目标路径（父目录须已存在）。
        img: uint8 RGB (H, W, 3)。
        quality: JPEG 质量，默认 95。
    Returns:
        实际写出的路径。
    Raises:
        ValueError: 数组类型/形状非法。
    """
    if img.dtype != np.uint8:
        raise ValueError(f"imwrite 要求 uint8, 实际 {img.dtype}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"imwrite 要求 (H, W, 3), 实际 {img.shape}")
    path = Path(path)
    if path.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError(f"赛题要求 jpg 输出, 拒绝写出其他后缀: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(
        path, format="JPEG", quality=quality, subsampling=0
    )
    return path


def assert_same_size(a: np.ndarray, b: np.ndarray, names: tuple[str, str] = ("a", "b")) -> None:
    """断言两图分辨率一致（赛题红线 R4 的程序化守卫）。"""
    if a.shape[:2] != b.shape[:2]:
        raise ValueError(
            f"分辨率不一致: {names[0]}={a.shape[:2]}, {names[1]}={b.shape[:2]}"
        )
