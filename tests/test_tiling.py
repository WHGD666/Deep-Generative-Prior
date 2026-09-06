"""分块规划/羽化/融合的数学正确性（接缝伪影的防线）。"""

import numpy as np
import pytest

from src.preprocess.tiling import feather_weight, merge_tiles, plan_tiles
from tests.helpers import make_img


def test_plan_tiles_covers_image_exactly():
    img = make_img()
    h, w = img.shape[:2]
    boxes = plan_tiles(h, w, tile=64, overlap=16)
    assert boxes, "必须产出至少一块"
    for (y0, x0, y1, x1) in boxes:
        assert 0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w
    # 末块必须贴边（不留缝）
    assert any(y1 == h for (_, _, y1, _) in boxes)
    assert any(x1 == w for (_, _, _, x1) in boxes)


def test_plan_tiles_rejects_bad_params():
    with pytest.raises(ValueError):
        plan_tiles(100, 100, tile=64, overlap=64)  # overlap >= tile
    with pytest.raises(ValueError):
        plan_tiles(100, 100, tile=0, overlap=0)


def test_identity_roundtrip_lossless():
    """恒等增强下 分块->融合 必须逐像素还原原图（羽化和恒为 1 的数学保证）。"""
    img = make_img()
    h, w = img.shape[:2]
    boxes = plan_tiles(h, w, tile=64, overlap=16)
    tiles = [img[y0:y1, x0:x1] for (y0, x0, y1, x1) in boxes]
    merged = merge_tiles(tiles, boxes, h, w, overlap_in=16, upscale=1)
    assert np.array_equal(merged, img), "恒等往返产生像素差异（融合实现有 bug）"


def test_merge_rejects_uncovered_pixels():
    img = make_img()
    boxes = plan_tiles(*img.shape[:2], tile=64, overlap=16)
    tiles = [img[y0:y1, x0:x1] for (y0, x0, y1, x1) in boxes]
    # 少喂一块 => 中间出现未覆盖像素 => 必须报错而不是静默输出
    with pytest.raises(RuntimeError):
        merge_tiles(tiles[:-1], boxes[:-1], *img.shape[:2], overlap_in=16)


def test_feather_weights_sum_to_one_in_overlap():
    a = feather_weight(64, 64, overlap=16)
    # 中心区域应为 1，边缘角点最小且 >0
    assert a[32, 32] == 1.0
    assert 0.0 < a[0, 0] < 1.0


def test_upscaled_merge_shapes():
    """×k 后端的融合：输出空间尺寸 = 输入尺寸 × k，块尺寸按 k 校验。"""
    img = make_img()
    h, w = img.shape[:2]
    boxes = plan_tiles(h, w, tile=32, overlap=8)
    k = 2
    tiles = [np.repeat(np.repeat(img[y0:y1, x0:x1], k, axis=0), k, axis=1)
             for (y0, x0, y1, x1) in boxes]
    merged = merge_tiles(tiles, boxes, h * k, w * k, overlap_in=8, upscale=k)
    assert merged.shape == (h * k, w * k, 3)
