"""pyiqa 惰性封装（单例 predictor，FR/NR 统一入口）。

仅远程 GPU 环境使用；本地单测不触发导入。
调用约定：predictor(metric_name, img[, ref]) -> float，
img/ref 均为 uint8 RGB (H, W, 3)；FR 指标参数顺序 = (输出, 参考)。
"""

from __future__ import annotations

import numpy as np

_created: dict = {}


def _to_tensor(img: np.ndarray, device: str):
    import torch  # 延迟导入

    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return t.to(device)


def get_predictor(names: list[str], device: str = "cuda"):
    """返回统一签名 callable(metric_name, img, ref=None) -> float。"""
    import torch
    import pyiqa

    for name in names:
        if name not in _created:
            _created[name] = pyiqa.create_metric(name, device=device).eval()

    def predict(metric_name: str, img: np.ndarray, ref: np.ndarray | None = None) -> float:
        metric = _created[metric_name]
        with torch.no_grad():
            x = _to_tensor(img, device)
            if ref is None:
                value = metric(x)
            else:
                value = metric(x, _to_tensor(ref, device))
        v = float(value.reshape(-1)[0].item())
        if not np.isfinite(v):
            raise RuntimeError(f"pyiqa 指标 {metric_name} 返回非有限值: {v}")
        return v

    return predict
