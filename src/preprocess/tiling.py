"""4K 分块：切块规划、羽化权重、加权融合。

设计保证（tests/test_tiling.py 验证）：
- plan_tiles 覆盖全图且最后一块贴齐图像边缘（不留缝隙）；
- 融合按权重和归一化：重叠区两块权重形成线性过渡（线性 crossfade），
  因此"分块 -> 恒等增强 -> 融合"可以逐像素无损还原原图；
- 图像边界处权重虽衰减，但归一化保证边界不产生人为变暗。
"""

from __future__ import annotations

import numpy as np

Box = tuple[int, int, int, int]  # (y0, x0, y1, x1)，半开区间 [y0,y1) x [x0,x1)


def plan_tiles(height: int, width: int, tile: int, overlap: int) -> list[Box]:
    """规划覆盖 (height, width) 的块坐标。

    Args:
        tile: 期望块尺寸（像素）。
        overlap: 相邻块重叠像素数，须 < tile。
    """
    if tile <= 0 or overlap < 0 or overlap >= tile:
        raise ValueError(f"非法分块参数: tile={tile}, overlap={overlap}")

    def starts(total: int) -> list[int]:
        if total <= tile:
            return [0]
        stride = tile - overlap
        # 常规起点 + 末尾强制贴边
        s = list(range(0, total - tile + 1, stride))
        if s[-1] != total - tile:
            s.append(total - tile)
        return s

    ys, xs = starts(height), starts(width)
    return [(y, x, min(y + tile, height), min(x + tile, width)) for y in ys for x in xs]


def feather_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """生成 (h, w) 线性羽化权重：四周 overlap 像素内 0->1 线性上升，中心为 1。

    图像尺寸小于 2*overlap 时退化为可用权重（仍保证 >=0 且归一化由调用方处理）。
    """
    def ramp(n: int) -> np.ndarray:
        r = np.ones(n, dtype=np.float32)
        o = min(overlap, (n - 1) // 2) if n > 1 else 0
        if o > 0:
            edge = np.linspace(1.0 / o, 1.0, o, dtype=np.float32)
            r[:o] = edge
            r[n - o:] = edge[::-1]
        return r

    return np.outer(ramp(h), ramp(w))


def merge_tiles(
    tiles: list[np.ndarray],
    boxes: list[Box],
    out_height: int,
    out_width: int,
    overlap_in: int,
    upscale: int = 1,
) -> np.ndarray:
    """把（可能被后端放大 k 倍的）tile 加权融合回整图。

    Args:
        tiles: 每块增强结果，uint8 RGB；尺寸 = 块输入尺寸 * upscale。
        boxes: 对应的输入空间块坐标。
        overlap_in: 输入空间的块重叠像素（羽化宽度按输出空间换算）。
        upscale: 后端放大倍率（整数，如 4 或 1）。
    """
    if len(tiles) != len(boxes):
        raise ValueError(f"tiles 数 {len(tiles)} 与 boxes 数 {len(boxes)} 不一致")
    acc = np.zeros((out_height, out_width, 3), dtype=np.float32)
    wsum = np.zeros((out_height, out_width), dtype=np.float32)
    for tile, (y0, x0, y1, x1) in zip(tiles, boxes):
        th, tw = tile.shape[:2]
        if (th, tw) != ((y1 - y0) * upscale, (x1 - x0) * upscale):
            raise ValueError(
                f"tile 尺寸 {tile.shape[:2]} 与期望 {((y1-y0)*upscale, (x1-x0)*upscale)} 不符"
            )
        w = feather_weight(th, tw, overlap_in * upscale)
        ys, xs = y0 * upscale, x0 * upscale
        acc[ys : ys + th, xs : xs + tw] += tile.astype(np.float32) * w[..., None]
        wsum[ys : ys + th, xs : xs + tw] += w
    if np.any(wsum < 1e-6):
        raise RuntimeError("存在未被任何 tile 覆盖的像素（分块规划缺陷）")
    return np.clip(acc / wsum[..., None], 0, 255).astype(np.uint8)
