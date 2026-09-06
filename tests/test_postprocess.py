"""后处理层语义：关闭即恒等、对齐数学正确、λ 端点正确、一致性核查拦截。"""

import numpy as np
import pytest

from src.postprocess import color_align, consistency_check, lambda_blend, lowfreq_backfill
from src.utils.image_ops import gaussian_blur, to_float01
from tests.helpers import make_img


def test_color_align_exact_statistics():
    img = make_img(seed=1)
    ref = make_img(seed=2)
    aligned = color_align.apply(img, ref=ref, cfg={"enabled": True, "mode": "mean_std", "strength": 1.0})
    f, r = to_float01(aligned.astype(np.uint8)), to_float01(ref)
    for c in range(3):
        assert abs(f[..., c].mean() - r[..., c].mean()) < 0.01
        assert abs(f[..., c].std() - r[..., c].std()) < 0.02


def test_color_align_strength_zero_is_identity():
    img = make_img(seed=1)
    ref = make_img(seed=2)
    out = color_align.apply(img, ref=ref, cfg={"strength": 0.0})
    assert np.array_equal(out, img)


def test_lowfreq_backfill_strength_zero_is_identity():
    img = make_img(seed=3)
    ref = make_img(seed=4)
    out = lowfreq_backfill.apply(img, ref=ref, cfg={"sigma": 8.0, "strength": 0.0})
    assert np.array_equal(out, img)


def test_lowfreq_backfill_moves_lowfreq_toward_ref():
    img = make_img(seed=3)
    ref = make_img(seed=4)
    sigma, strength = 8.0, 1.0
    out = to_float01(lowfreq_backfill.apply(img, ref=ref, cfg={"sigma": sigma, "strength": strength}))
    low_out = gaussian_blur((out * 255).astype(np.uint8), sigma)
    low_ref = gaussian_blur(ref, sigma)
    drift = np.abs(low_out - low_ref).mean()
    baseline = np.abs(gaussian_blur(img, sigma) - low_ref).mean()
    assert drift < baseline, "回填后低频未向参考靠拢"


def test_lambda_endpoints():
    enh = (make_img(seed=5) // 2 + 64).astype(np.uint8)
    fid = make_img(seed=6)
    assert np.array_equal(lambda_blend.apply(enh, fid, {"lambda": 1.0}), enh)
    assert np.array_equal(lambda_blend.apply(enh, fid, {"lambda": 0.0}), fid)


def test_lambda_rejects_bad_params():
    with pytest.raises(ValueError):
        lambda_blend.blend(make_img(), make_img(), lam=1.5)
    with pytest.raises(ValueError):
        lambda_blend.blend(make_img(), make_img()[:10], lam=0.5)


def test_consistency_checks():
    ref = make_img(seed=7)
    # 尺寸错 -> 立即告警
    assert consistency_check.check(ref[:10], ref, cfg={})
    # 色调漂移 -> 告警
    drifted = np.clip(ref.astype(np.int16) + 40, 0, 255).astype(np.uint8)
    msgs = consistency_check.check(drifted, ref, cfg={"max_color_drift": 0.06, "min_corr": 0.99})
    assert any("漂移" in m for m in msgs)
    # 恒等 -> 无告警
    assert consistency_check.check(ref, ref, cfg={}) == []
