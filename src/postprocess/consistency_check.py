"""输出-输入一致性核查（赛题红线 R3/R4 的程序化守卫）。

产出 warnings 列表（不阻断，但全部记入 run manifest），
提交前由 src/submit/validate.py 做最终硬校验。
"""

from __future__ import annotations

import numpy as np

from src.utils.image_ops import to_float01


def check(out: np.ndarray, ref: np.ndarray, cfg: dict) -> list[str]:
    """返回告警列表（空列表 = 通过）。cfg: {max_color_drift, min_corr}"""
    warnings: list[str] = []
    if out.shape != ref.shape:
        return [f"尺寸/通道不一致: out={out.shape}, ref={ref.shape}"]

    max_drift = float(cfg.get("max_color_drift", 0.06))  # 逐通道均值漂移上限（0~1 域）
    f, r = to_float01(out), to_float01(ref)
    for c, name in enumerate("RGB"):
        drift = abs(f[..., c].mean() - r[..., c].mean())
        if drift > max_drift:
            warnings.append(f"{name} 通道均值漂移 {drift:.4f} > {max_drift}")

    # 结构代理：4 倍降采样后的相关性（内容被改写时相关度会显著下降）
    min_corr = float(cfg.get("min_corr", 0.80))
    a = f[::4, ::4].reshape(-1, 3).mean(axis=1)
    b = r[::4, ::4].reshape(-1, 3).mean(axis=1)
    if a.std() > 1e-6 and b.std() > 1e-6:
        corr = float(np.corrcoef(a, b)[0, 1])
        if corr < min_corr:
            warnings.append(f"结构相关性 {corr:.3f} < {min_corr}（疑似内容改动）")
    return warnings
