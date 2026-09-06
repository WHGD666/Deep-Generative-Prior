"""λ 混合：保真-生成的连续调节旋钮。

    result = lambda * enhanced + (1 - lambda) * fidelity

- fidelity 默认取"预处理后的输入"（去噪等），即几乎纯保真端；
  也可配置为其他来源（预留 fidelity_source 字段）。
- lambda 由 val 扫描确定（见 ADR-001 / README 5.3）。
"""

from __future__ import annotations

import numpy as np

from src.utils.image_ops import to_uint8


def blend(enhanced: np.ndarray, fidelity: np.ndarray, lam: float) -> np.ndarray:
    if enhanced.shape[:2] != fidelity.shape[:2]:
        raise ValueError(
            f"λ混合尺寸不一致: enhanced={enhanced.shape[:2]}, fidelity={fidelity.shape[:2]}"
        )
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda 须在 [0,1], 得到 {lam}")
    out = (
        lam * enhanced.astype(np.float32)
        + (1.0 - lam) * fidelity.astype(np.float32)
    )
    return np.clip(out, 0, 255).astype(np.uint8)


def apply(img: np.ndarray, fidelity: np.ndarray, cfg: dict) -> np.ndarray:
    """配置入口。cfg: {enabled, lambda, fidelity_source}"""
    source = cfg.get("fidelity_source", "preprocessed")
    if source != "preprocessed":
        raise ValueError(
            f"未知 fidelity_source: {source}（当前仅支持 preprocessed）"
        )
    return blend(img, fidelity, float(cfg.get("lambda", 0.7)))
