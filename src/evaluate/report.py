"""评测报告：多个 eval.json -> 一张对照表（markdown）。

用法：
    python -m src.evaluate.report --evals experiments/*/eval.json --out docs/reports/xxx.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v: float) -> str:
    return f"{v:.4f}"


def build_table(eval_paths: list[Path]) -> str:
    runs = {}
    for p in eval_paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        runs[p.parent.name if p.name == "eval.json" else p.stem] = data
    if not runs:
        raise ValueError("没有可汇总的 eval.json")

    metric_names: list[str] = []
    for data in runs.values():
        for m in data.get("means", {}):
            if m not in metric_names:
                metric_names.append(m)

    lines = ["# 评测对照表", "",
             "> 口径：本地启发式综合分，非官方分数。", "",
             "| run | n | " + " | ".join(metric_names) + " | "
             + " | ".join(f"comp:{k}" for k in next(iter(runs.values()))["composite"]) + " |",
             "|---|---|" + "---|" * (len(metric_names) + len(next(iter(runs.values()))["composite"]))]

    for name, data in runs.items():
        means = data["means"]
        comp = data["composite"]
        lines.append(
            f"| {name} | {data['n_images']} | "
            + " | ".join(fmt(means[m]) if m in means else "-" for m in metric_names)
            + " | " + " | ".join(fmt(v) for v in comp.values()) + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="评测对照表")
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="可选 markdown 输出；缺省打印 stdout")
    args = ap.parse_args(argv)

    table = build_table([Path(p) for p in args.evals])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
        print(f"[完成] {out}")
    else:
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
