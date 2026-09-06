"""指标实现锚点测试：PSNR/SSIM 的数值正确性是一切评测结论的地基。"""

import numpy as np
import pytest

from src.evaluate.metrics_fr import ms_ssim, psnr, ssim
from tests.helpers import make_img


def test_psnr_identical_capped():
    img = make_img()
    assert psnr(img, img) == 60.0


def test_psnr_known_value():
    """常数 0 vs 常数 10：mse=100，PSNR = 10*log10(255^2/100) ≈ 28.1308。"""
    a = np.full((16, 16, 3), 0, dtype=np.uint8)
    b = np.full((16, 16, 3), 10, dtype=np.uint8)
    expected = 10.0 * np.log10(255.0**2 / 100.0)
    assert psnr(a, b) == pytest.approx(expected, abs=1e-9)


def test_psnr_rejects_size_mismatch():
    with pytest.raises(ValueError):
        psnr(make_img(), make_img()[:50])


def test_ssim_identical_is_one():
    img = make_img()
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-9)


def test_ssim_distorted_lower():
    img = make_img(seed=11)
    noisy = np.clip(img.astype(np.int16) + np.random.default_rng(0).integers(-30, 30, img.shape), 0, 255).astype(np.uint8)
    assert ssim(noisy, img) < 0.95


def test_ms_ssim_identical_is_one():
    img = make_img()
    assert ms_ssim(img, img) == pytest.approx(1.0, abs=1e-6)
