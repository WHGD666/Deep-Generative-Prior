"""综合分：方向折算与权重归一化。"""

import pytest

from src.evaluate import composite
from src.evaluate.metrics_fr import LOWERS_BETTER as FR_LOWERS
from src.evaluate.metrics_nr import LOWERS_BETTER as NR_LOWERS


def test_direction_registry_complete():
    """方向登记自检（新指标加入时必须登记方向）。"""
    composite.sanity_check_directions(NR_LOWERS, FR_LOWERS)


def test_scale_bounds():
    assert composite.scale("psnr", 50.0) == 1.0
    assert composite.scale("psnr", 0.0) == 0.0
    assert composite.scale("lpips", 0.0) == 1.0   # 低好 -> 高好
    assert composite.scale("lpips", 1.0) == 0.0
    assert composite.scale("niqe", 0.0) == 1.0
    assert 0.0 < composite.scale("niqe", 8.0) < 1.0
    assert composite.scale("musiq", 100.0) == 1.0


def test_composite_renormalizes_missing_metrics():
    means = {"psnr": 25.0}  # 只给了 psnr
    weights = {"psnr": 1.0, "musiq": 3.0}
    # 缺失的 musiq 不参与，等价于 psnr 单指标
    assert composite.composite(means, weights) == pytest.approx(composite.scale("psnr", 25.0))


def test_composite_rejects_empty_overlap():
    with pytest.raises(ValueError):
        composite.composite({"ssim": 0.9}, {"musiq": 1.0})


def test_unknown_metric_rejected():
    with pytest.raises(KeyError):
        composite.scale("not_a_metric", 1.0)
