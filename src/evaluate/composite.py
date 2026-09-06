"""综合分：把量纲/方向各异的指标折算到 [0,1] 后加权。

⚠️ 本文件的全部折算与权重都是**本地比较用的启发式**，
官方公式公布后必须切换为官方口径（configs/defaults/evaluate.yaml 注释）。
折算规则：
    psnr   : min(v, 50) / 50            （dB 上限截断）
    ssim   : v
    ms_ssim: v
    lpips  : 1 - v                       （低好 -> 高好）
    dists  : 1 - v
    musiq  : v / 100
    clipiqa: v
    maniqa : v
    niqe   : 1 / (1 + max(v, 0) / 10)    （低好 -> 高好，k=10）
"""

from __future__ import annotations

import math

_SCALE_DOC = __doc__


def scale(metric: str, value: float) -> float:
    if metric == "psnr":
        return min(value, 50.0) / 50.0
    if metric in ("ssim", "ms_ssim", "clipiqa", "maniqa"):
        return value
    if metric in ("lpips", "dists"):
        return 1.0 - value
    if metric == "musiq":
        return value / 100.0
    if metric == "niqe":
        return 1.0 / (1.0 + max(value, 0.0) / 10.0)
    raise KeyError(f"未登记的指标缩放: {metric}")


def composite(means: dict[str, float], weights: dict[str, float]) -> float:
    """加权综合分；缺失指标按剩余权重重归一化。"""
    num = den = 0.0
    for m, w in weights.items():
        if m not in means:
            continue
        num += w * scale(m, float(means[m]))
        den += w
    if den <= 0:
        raise ValueError(f"means 中没有任何加权所需的指标: {sorted(means)} vs {weights}")
    return num / den


def all_composites(means: dict[str, float], metric_sets: dict[str, dict[str, float]]) -> dict[str, float]:
    return {name: composite(means, weights) for name, weights in metric_sets.items()}


def sanity_check_directions(nr_lowers: tuple[str, ...], fr_lowers: tuple[str, ...]) -> None:
    """方向登记自检：NIQE/LPIPS/DISTS 应为低好，其余登记指标为高好。"""
    for m in ("niqe", "lpips", "dists"):
        if m not in nr_lowers + fr_lowers:
            raise RuntimeError(f"指标 {m} 方向登记缺失（应为低好）")
    assert math.isclose(scale("lpips", 0.0), 1.0) and math.isclose(scale("lpips", 1.0), 0.0)
