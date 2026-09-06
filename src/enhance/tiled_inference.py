"""分块推理引擎。

职责：把任意大图按配置切块 -> 交给"批增强器"（backend）一次性处理 ->
按羽化权重融合回整图。引擎不关心后端实现（DiffBIR/ResShift 均为 CLI
批处理驱动），只约定批增强器接口：

    enhance_batch(tiles: list[uint8 RGB]) -> list[uint8 RGB]
    # 输入输出按顺序一一对应；输出尺寸 = 输入尺寸 * upscale（整数）。

放大倍率 k 由 run_pipeline 用"探测块"自动推断（infer_upscale），
不依赖后端文档的正确性——这是对第三方行为不确定性的防御性设计。
"""

from __future__ import annotations

import numpy as np

from src.preprocess.tiling import Box, merge_tiles, plan_tiles


class TiledEngine:
    def __init__(self, tile_size: int, overlap: int):
        if overlap >= tile_size:
            raise ValueError(f"overlap({overlap}) 必须小于 tile_size({tile_size})")
        self.tile_size = tile_size
        self.overlap = overlap

    def plan(self, height: int, width: int) -> list[Box]:
        return plan_tiles(height, width, self.tile_size, self.overlap)

    def run(
        self,
        img: np.ndarray,
        enhance_batch,
        upscale: int = 1,
        progress=None,
    ) -> np.ndarray:
        """整图分块增强。

        Args:
            img: uint8 RGB 整图（若后端为 ×k 超分底座，此处应传入已按 1/k 预缩放的图）。
            enhance_batch: 批增强器，见模块 docstring。
            upscale: 后端对每块的放大倍率（整数，来自探测）。
            progress: 可选回调 progress(done, total)。
        """
        h, w = img.shape[:2]
        boxes = self.plan(h, w)
        tiles_in = [img[y0:y1, x0:x1] for (y0, x0, y1, x1) in boxes]
        tiles_out = enhance_batch(tiles_in)
        if len(tiles_out) != len(tiles_in):
            raise RuntimeError(
                f"后端返回块数 {len(tiles_out)} != 输入块数 {len(tiles_in)}"
            )
        merged = merge_tiles(
            tiles_out, boxes, h * upscale, w * upscale, self.overlap, upscale
        )
        if progress is not None:
            progress(len(tiles_in), len(tiles_in))
        return merged


def infer_upscale(enhance_batch, sample: np.ndarray) -> int:
    """用一块小样本探测后端放大倍率。

    后端行为（是否 ×4、是否 ×1）以实测为准：跑一个 128px 的探测块，
    比较输出尺寸。任何无法整除/不成倍的结果都直接报错，绝不静默假设。
    """
    out = enhance_batch([sample])[0]
    k_h, k_w = out.shape[0] / sample.shape[0], out.shape[1] / sample.shape[1]
    if k_h != k_w or k_h != int(k_h) or int(k_h) < 1:
        raise RuntimeError(
            f"后端放大倍率异常: 输入 {sample.shape[:2]} -> 输出 {out.shape[:2]}"
        )
    return int(k_h)
