"""无参考指标（NR）：MUSIQ / CLIPIQA / NIQE / MANIQA（pyiqa）。

方向登记（composite 与 report 依赖）：NIQE 越低越好；
pyiqa 其余默认越高越好。口径声明在 configs/defaults/evaluate.yaml，
本模块的 LOWERS_BETTER 与其保持一致（tests/test_composite.py 校验）。
"""

from __future__ import annotations

LOWERS_BETTER = ("niqe",)

DEFAULT_NR_METRICS = ["musiq", "clipiqa", "niqe", "maniqa"]


def compute_nr(img: np.ndarray, metrics: list[str], device: str = "cuda") -> dict:
    from src.evaluate._iqa import get_predictor

    predict = get_predictor(metrics, device)
    return {m: predict(m, img, None) for m in metrics}
