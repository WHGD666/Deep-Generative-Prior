"""discover_inputs 模式匹配回归测试。

历史 bug：用 stem（无扩展名）去 endswith 带 .jpg 的 suffix，导致永远匹配不上。
"""
from src.run_pipeline import discover_inputs


def _mk(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"x")


def test_val_pattern_matches_lq_only(tmp_path):
    _mk(tmp_path, ["case1_lq.jpg", "case1_gt.jpg", "case2_lq.jpg", "notes.txt"])
    cases = discover_inputs(tmp_path, "{case}_lq.jpg")
    assert set(cases) == {"case1", "case2"}
    assert cases["case1"].name == "case1_lq.jpg"


def test_test_pattern_plain_jpg(tmp_path):
    _mk(tmp_path, ["case1.jpg", "case10.jpg", "case100.jpg"])
    cases = discover_inputs(tmp_path, "{case}.jpg")
    assert set(cases) == {"case1", "case10", "case100"}


def test_ignores_subdirs_and_extension_case(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "case3_lq.jpg").write_bytes(b"x")  # 子目录不扫
    _mk(tmp_path, ["case4_lq.JPG"])  # 大写扩展名可匹配
    cases = discover_inputs(tmp_path, "{case}_lq.jpg")
    assert set(cases) == {"case4"}
