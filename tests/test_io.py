"""图像 IO：像素级保真（EXIF 默认忽略）与尺寸守卫。"""

import pytest
from PIL import Image

from src.utils import io as uio
from tests.helpers import make_img


def test_imread_imwrite_roundtrip(tmp_path):
    img = make_img()
    p = tmp_path / "a.jpg"
    uio.imwrite(p, img, quality=95)
    back = uio.imread(p)
    assert back.shape == img.shape
    # quality=95 + 4:4:4 下，合成图的往返误差应极小（JPEG 有损，不要求逐像素相等）
    import numpy as np

    assert np.abs(back.astype(int) - img.astype(int)).mean() < 2.0


def test_imwrite_rejects_non_jpg(tmp_path):
    with pytest.raises(ValueError):
        uio.imwrite(tmp_path / "a.png", make_img())


def test_imwrite_rejects_bad_shape(tmp_path):
    with pytest.raises(ValueError):
        uio.imwrite(tmp_path / "a.jpg", make_img()[..., :2])


def test_exif_ignored_by_default(tmp_path):
    """带旋转元数据的图：默认读取必须保持原始像素尺寸（红线 R4）。"""
    img = make_img(64, 32)  # 横图
    p = tmp_path / "rot.jpg"
    # 写入 orientation=6（应旋转 90°）的 EXIF
    pil = Image.fromarray(img)
    exif = Image.Exif()
    exif[274] = 6  # Orientation tag
    pil.save(p, exif=exif)
    back = uio.imread(p)
    assert back.shape[:2] == (64, 32), "EXIF 被意外应用，破坏像素级对齐"
    # 显式开启 respect_exif 时才旋转
    rotated = uio.imread(p, respect_exif=True)
    assert rotated.shape[:2] == (32, 64)


def test_assert_same_size():
    a = make_img()
    uio.assert_same_size(a, a)
    with pytest.raises(ValueError):
        uio.assert_same_size(a, a[:10])
