"""提交校验器：模拟各种坏包，必须全部被拦截（资格红线零容忍）。"""

import numpy as np
import pytest
from PIL import Image

from src.submit.validate import validate
from src.utils import io as uio
from tests.helpers import make_img

CASES = [1, 2, 3]


def _make_pair(tmp_path, sizes=None, extra=None, drop=None):
    """构造 (input_dir, output_dir)：默认 3 张齐全同尺寸；可注入异常。"""
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    sizes = sizes or {}
    for i in CASES:
        img = make_img(48, 64, seed=i)
        if sizes.get(i):
            h, w = sizes[i]
            img = np.asarray(Image.fromarray(img).resize((w, h)), dtype=np.uint8)
        uio.imwrite(in_dir / f"case{i}.jpg", img)
        if drop != i:
            uio.imwrite(out_dir / f"case{i}.jpg", img)
    for name, arr in (extra or {}).items():
        uio.imwrite(out_dir / name, arr)
    return in_dir, out_dir


def test_valid_package_passes(tmp_path):
    in_dir, out_dir = _make_pair(tmp_path)
    assert validate(out_dir, in_dir, cases=CASES) == []


def test_missing_file_caught(tmp_path):
    in_dir, out_dir = _make_pair(tmp_path, drop=2)
    problems = validate(out_dir, in_dir, cases=CASES)
    assert any("case2" in p and "缺" in p for p in problems)


def test_size_mismatch_caught(tmp_path):
    in_dir, out_dir = _make_pair(tmp_path, sizes={3: (32, 32)})
    problems = validate(out_dir, in_dir, cases=CASES)
    assert any("case3" in p and "分辨率" in p for p in problems)


def test_extra_file_caught(tmp_path):
    in_dir, out_dir = _make_pair(tmp_path, extra={"case4.jpg": make_img(48, 64, seed=9)})
    problems = validate(out_dir, in_dir, cases=CASES)
    assert any("多出" in p for p in problems)


def test_non_ascii_name_caught(tmp_path):
    in_dir, out_dir = _make_pair(tmp_path, extra={"casｅ1.jpg": make_img(48, 64)})
    problems = validate(out_dir, in_dir, cases=CASES)
    # 全角字符文件既算"多出"也算非 ASCII
    assert any("非 ASCII" in p for p in problems)


def test_non_jpg_caught(tmp_path):
    in_dir, out_dir = _make_pair(tmp_path)
    (out_dir / "note.txt").write_text("x", encoding="utf-8")
    problems = validate(out_dir, in_dir, cases=CASES)
    assert any("非 jpg" in p for p in problems)


def test_missing_input_side_reported(tmp_path):
    in_dir, out_dir = _make_pair(tmp_path)
    (in_dir / "case3.jpg").unlink()
    problems = validate(out_dir, in_dir, cases=CASES)
    assert any("输入侧缺少 case3" in p for p in problems)


def test_pack_refuses_placeholder_config(tmp_path, monkeypatch):
    """pack.py 必须拒绝未填写的 submit.yaml（占位符防线）。"""
    from src.submit import pack as pack_mod

    cfg_path = tmp_path / "submit.yaml"
    cfg_path.write_text(
        "track: 赛题一\nwork_name: <填写作品名>\nteam_name: 测试队\nphone: '13000000000'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        pack_mod.load_config(cfg_path)
