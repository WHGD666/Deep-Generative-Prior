"""提交包组装：从 run 输出目录复制到 output_dir（不解压、不改名、不重压缩）。

用法：
    python -m src.submit.assemble --run_outputs experiments/<run_id>/outputs --output_dir output_dir
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def assemble(run_outputs: Path, output_dir: Path) -> int:
    if not run_outputs.is_dir():
        raise FileNotFoundError(f"run 输出目录不存在: {run_outputs}")
    files = sorted(run_outputs.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"{run_outputs} 中没有 jpg")
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        shutil.copy2(p, output_dir / p.name)
    print(f"[完成] 复制 {len(files)} 张 -> {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="组装 output_dir")
    ap.add_argument("--run_outputs", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args(argv)
    return assemble(Path(args.run_outputs), Path(args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
