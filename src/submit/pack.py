"""生成提交 zip（校验通过才允许打包）。

- 压缩结构：zip 根下只有 output_dir/ 单层目录，内含 case1~case100.jpg；
- jpg 已压缩，zip 用 STORED（不二次压缩，快且稳）；
- 命名与元信息来自 configs/defaults/submit.yaml（提交前填好占位字段）；
- 生成同步写入 submissions/<subNN>-<date>/manifest.md 快照。

用法：
    python -m src.submit.pack --output_dir output_dir --config configs/defaults/submit.yaml \
        --sub_id sub01
"""

from __future__ import annotations

import argparse
import zipfile
from datetime import datetime
from pathlib import Path

import yaml

from src.submit.validate import validate

MAX_BYTES = 10 * 1024**3  # 官方上限 10GB


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("track", "work_name", "team_name", "phone"):
        v = str(cfg.get(key, ""))
        if not v or "<" in v:  # 占位符未填
            raise ValueError(f"submit.yaml 的 {key} 未填写（当前: {v!r}）")
    if not str(cfg["phone"]).isdigit():
        raise ValueError("phone 必须是纯数字")
    return cfg


def make_zip_name(cfg: dict) -> str:
    return f"{cfg['track']}_{cfg['work_name']}_{cfg['team_name']}_{cfg['phone']}.zip"


def pack(output_dir: Path, zip_path: Path) -> int:
    output_dir = output_dir.resolve()
    n = len(list(output_dir.glob("case*.jpg")))
    if n != 100:
        raise RuntimeError(f"output_dir 内 case*.jpg 数量为 {n}（应为 100），拒绝打包")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:
        for p in sorted(output_dir.glob("case*.jpg")):
            z.write(p, arcname=f"output_dir/{p.name}")
    size = zip_path.stat().st_size
    if size > MAX_BYTES:
        zip_path.unlink()
        raise RuntimeError(f"zip {size/1e9:.2f}GB 超过官方 10GB 上限，已删除")
    print(f"[完成] {zip_path} ({size/1e6:.1f} MB)")
    return size


def write_snapshot(zip_path: Path, run_id: str, sub_id: str) -> Path:
    sub_dir = Path("submissions") / f"{sub_id}-{datetime.now():%Y%m%d}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "manifest.md").write_text(
        f"# 提交快照 {sub_id}\n\n"
        f"- 日期: {datetime.now():%Y-%m-%d %H:%M}\n"
        f"- 来源 run: {run_id}\n"
        f"- zip: {zip_path.name} ({zip_path.stat().st_size/1e6:.1f} MB)\n"
        f"- 本地评测: 见同目录 local_scores.json（由 run_eval 产物复制，如适用）\n"
        f"- 官方分数: 待提交后回填根 SUBMISSIONS.md\n",
        encoding="utf-8",
    )
    return sub_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成提交 zip")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--config", default="configs/defaults/submit.yaml")
    ap.add_argument("--input_dir", required=True, help="data/test（校验尺寸基准）")
    ap.add_argument("--zip_dir", default=".", help="zip 输出目录")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--sub_id", required=True, help="如 sub01")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    problems = validate(Path(args.output_dir), Path(args.input_dir))
    if problems:
        print("[拒绝] 校验未通过：")
        for p in problems:
            print(f"  - {p}")
        return 1

    zip_dir = Path(args.zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zip_dir / make_zip_name(cfg)
    pack(Path(args.output_dir), zip_path)
    snap = write_snapshot(zip_path, args.run_id, args.sub_id)
    print(f"[快照] {snap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
