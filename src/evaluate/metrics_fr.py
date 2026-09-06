"""有参考指标（FR）。

- PSNR / SSIM / MS-SSIM：本项目纯 numpy 实现（确定性、可本地单测、
  与 skimage 口径对齐——见 tests/test_metrics.py）；
- LPIPS / DISTS：通过 pyiqa（惰性导入，远程 GPU 环境使用）。

全部指标在 uint8 RGB (H, W, 3) 上计算，data_range=255。
"""

from __future__ import annotations

import numpy as np

# 标准 MS-SSIM 5 级权重（Wang et al. 2003）
_MS_WEIGHTS = np.array([0.04455, 0.28573, 0.3001, 0.23618, 0.1333], dtype=np.float64)

# pyiqa 中"越低越好"的 FR 指标登记在此
LOWERS_BETTER = ("lpips", "dists")


def psnr(out: np.ndarray, ref: np.ndarray) -> float:
    """PSNR (dB)。两图一致时返回上限 60.0（避免 inf 难以聚合）。"""
    if out.shape != ref.shape:
        raise ValueError(f"PSNR 尺寸不一致: {out.shape} vs {ref.shape}")
    f = out.astype(np.float64) - ref.astype(np.float64)
    mse = float(np.mean(f * f))
    if mse <= 1e-12:
        return 60.0
    return float(10.0 * np.log10(255.0**2 / mse))


def _box_filter(x: np.ndarray, size: int = 7) -> np.ndarray:
    """均匀窗滤波（积分图实现，O(N) 内存）。

    4K 图若用滑动窗口 einsum 会产生 >10GB 中间数组（已在本项目审计中否决）；
    积分图实现内存占用为常数级。边界按零填充处理——口径仅用于本地相对比较。
    """
    pad = size // 2
    p = np.pad(x, pad, mode="constant")
    c = np.cumsum(np.cumsum(p, axis=0, dtype=np.float64), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    # 窗口和 = c[y+size, x+size] - c[y, x+size] - c[y+size, x] + c[y, x]
    s = (
        c[size : size + x.shape[0], size : size + x.shape[1]]
        - c[: x.shape[0], size : size + x.shape[1]]
        - c[size : size + x.shape[0], : x.shape[1]]
        + c[: x.shape[0], : x.shape[1]]
    )
    return s / (size * size)


_SSIM_WIN = 7  # skimage 默认均匀窗宽


def ssim(out: np.ndarray, ref: np.ndarray) -> float:
    """SSIM（灰度化，7x7 均匀窗，C1=(0.01*255)^2，C2=(0.03*255)^2）。"""
    if out.shape != ref.shape:
        raise ValueError(f"SSIM 尺寸不一致: {out.shape} vs {ref.shape}")

    def to_gray(x: np.ndarray) -> np.ndarray:
        w = np.array([0.299, 0.587, 0.114])
        return x.astype(np.float64) @ w

    a, b = to_gray(out), to_gray(ref)
    mu_a, mu_b = _box_filter(a, _SSIM_WIN), _box_filter(b, _SSIM_WIN)
    aa, bb, ab = _box_filter(a * a, _SSIM_WIN), _box_filter(b * b, _SSIM_WIN), _box_filter(a * b, _SSIM_WIN)
    va = aa - mu_a**2
    vb = bb - mu_b**2
    vab = ab - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    )
    return float(s.mean())


def ms_ssim(out: np.ndarray, ref: np.ndarray, levels: int = 5) -> float:
    """MS-SSIM：逐级 2x 下采样，加权几何平均；尺寸不足时截断级数并重归一化。"""
    vals = []
    a, b = out.astype(np.uint8), ref.astype(np.uint8)
    for i in range(levels):
        if min(a.shape[:2]) < _SSIM_WIN * 2 ** (levels - 1 - i):
            w = _MS_WEIGHTS[:i]
            if w.sum() <= 0:
                return ssim(out, ref)
            return float(np.exp(np.sum(w * np.log(np.maximum(vals, 1e-12))) / w.sum()))
        vals.append(max(ssim(a, b), 1e-12))
        a, b = a[::2, ::2], b[::2, ::2]
    return float(
        np.exp(np.sum(_MS_WEIGHTS * np.log(np.maximum(vals, 1e-12))))
    )


def compute_fr(out: np.ndarray, ref: np.ndarray, metrics: list[str], device: str = "cpu") -> dict:
    """按名计算一组 FR 指标。pyiqa 指标名直接透传。"""
    result: dict[str, float] = {}
    own = {"psnr": psnr, "ssim": ssim, "ms_ssim": ms_ssim}
    need_iqa = [m for m in metrics if m not in own]
    predictor = None
    if need_iqa:
        import src.evaluate._iqa as iqa_helper

        predictor = iqa_helper.get_predictor(need_iqa, device)
    for m in metrics:
        if m in own:
            result[m] = own[m](out, ref)
        else:
            result[m] = float(
                predictor(m, out, ref)
            )
    return result
