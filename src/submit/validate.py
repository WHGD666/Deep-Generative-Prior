"""提交前强制校验（资格红线，零容忍）。

校验项：
1. 文件名恰好为 case1.jpg ~ case100.jpg（无多余、无缺失、无中文符号）；
2. 每张分辨率与对应输入逐张一致（横 4096x3072 / 竖 3072x4096，以实际输入为准）；
3. 全部为合法可读 jpg；
4. 数量 = 100。

用法：
    python -m src.submit.validate --output_dir output_dir --input_dir data/test
退出码：0 通过；1 存在问题（并列出全部问题）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

EXPECTED_CASES = list(range(1, 101))


def validate(output_dir: Path, input_dir: Path, cases: list[int] | None = None) -> list[str]:
    """cases 缺省为官方 100 张；测试可传入小子集。"""
    cases = cases if cases is not None else EXPECTED_CASES
    problems: list[str] = []

    def bad_name(p: Path) -> bool:
        return any(not ch.isascii() for ch in p.name)

    files = sorted(output_dir.iterdir()) if output_dir.is_dir() else []
    names = {p.name for p in files if p.is_file()}
    non_jpg = [p.name for p in files if p.is_file() and p.suffix.lower() != ".jpg"]
    if non_jpg:
        problems.append(f"存在非 jpg 文件: {non_jpg[:5]}{'...' if len(non_jpg) > 5 else ''}")

    expected = {f"case{i}.jpg" for i in cases}
    missing = sorted(expected - names)
    extra = sorted(names - expected)
    if missing:
        problems.append(f"缺少 {len(missing)} 张: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        problems.append(f"多出 {len(extra)} 个: {extra[:5]}{'...' if len(extra) > 5 else ''}")
    for name in sorted(names):  # 全量扫描，防止怪名文件混进包里
        if bad_name(Path(name)):
            problems.append(f"文件名含非 ASCII 字符: {name}")

    # 逐张尺寸比对（仅对存在的文件）
    for i in cases:
        out_p, in_p = output_dir / f"case{i}.jpg", input_dir / f"case{i}.jpg"
        if not out_p.is_file():
            continue
        if not in_p.is_file():
            problems.append(f"输入侧缺少 case{i}.jpg，无法比对尺寸")
            continue
        try:
            with Image.open(out_p) as fo, Image.open(in_p) as fi:
                out_size, in_size = fo.size, fi.size
        except OSError as e:
            problems.append(f"case{i}.jpg 无法读取: {e}")
            continue
        if tuple(out_size) != tuple(in_size):
            problems.append(
                f"case{i} 分辨率不一致: 输出 {out_size} vs 输入 {in_size}"
            )
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="提交包强制校验")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--input_dir", required=True, help="data/test（尺寸基准）")
    args = ap.parse_args(argv)

    problems = validate(Path(args.output_dir), Path(args.input_dir))
    if problems:
        print(f"[拒绝] 共 {len(problems)} 个问题：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("[通过] 100 张齐全、命名合规、分辨率与输入逐张一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
