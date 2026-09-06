"""数据集统计核查：数量、分辨率分布、命名完整性。

用法：
    python -m src.data_prep.stats --root data [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def scan_dir(d: Path) -> list[dict]:
    records = []
    for p in sorted(d.glob("*.jpg")):
        with Image.open(p) as im:
            w, h = im.size
        records.append({"name": p.name, "w": w, "h": h, "bytes": p.stat().st_size})
    return records


def check_val(d: Path, records: list[dict]) -> list[str]:
    problems = []
    names = {r["name"] for r in records}
    for i in range(1, 6):
        for tag in ("lq", "gt"):
            if f"case{i}_{tag}.jpg" not in names:
                problems.append(f"val 缺少 case{i}_{tag}.jpg")
    return problems


def check_test(d: Path, records: list[dict]) -> list[str]:
    problems = []
    names = {r["name"] for r in records}
    for i in range(1, 101):
        if f"case{i}.jpg" not in names:
            problems.append(f"test 缺少 case{i}.jpg")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="数据集统计核查")
    ap.add_argument("--root", default="data")
    ap.add_argument("--json", default=None, help="可选：统计结果写出的 json 路径")
    args = ap.parse_args(argv)

    root = Path(args.root)
    all_problems: list[str] = []
    report: dict = {}
    for sub, checker in (("val", check_val), ("test", check_test)):
        d = root / sub
        records = scan_dir(d)
        sizes = Counter(f"{r['w']}x{r['h']}" for r in records)
        problems = checker(d, records)
        all_problems += problems
        report[sub] = {
            "count": len(records),
            "resolutions": dict(sizes),
            "total_mb": round(sum(r["bytes"] for r in records) / 1e6, 1),
            "problems": problems,
        }
        print(f"[{sub}] {len(records)} 张, 分辨率分布 {dict(sizes)}, 问题 {len(problems)}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    if all_problems:
        for p in all_problems:
            print(f"[异常] {p}")
        return 1
    print("[通过] 数据集完整性与预期一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
