"""评测入口：输出图集 -> eval.json（逐图指标 + 均值 + 综合分）。

用法（远程，项目根目录）：
    # val：有 GT，FR + NR 全量
    python -m src.evaluate.run_eval \
        --output_dir experiments/<run_id>/outputs \
        --gt_dir data/val --out experiments/<run_id>/eval.json

    # test：无 GT，仅 NR
    python -m src.evaluate.run_eval \
        --output_dir experiments/<run_id>/outputs \
        --out experiments/<run_id>/eval_test.json --device cuda:0

输出的命名与 data/test 一致（caseN.jpg）；val 的 GT 命名为 caseN_gt.jpg。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate.composite import all_composites
from src.evaluate.metrics_fr import compute_fr
from src.evaluate.metrics_nr import compute_nr
from src.utils import io as uio

DEFAULT_EVAL_CFG = Path(__file__).resolve().parents[2] / "configs/defaults/evaluate.yaml"


def load_cfg(path: Path) -> dict:
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def evaluate_set(output_dir: Path, gt_dir: Path | None, cfg: dict, device: str) -> dict:
    outputs = sorted(output_dir.glob("*.jpg"))
    if not outputs:
        raise FileNotFoundError(f"{output_dir} 中没有可评测的 jpg")

    fr_metrics = list(cfg.get("fr_metrics", []))
    nr_metrics = list(cfg.get("nr_metrics", []))
    if gt_dir is None:
        fr_metrics = []  # 无 GT 时强制关闭 FR
    per_image: dict[str, dict] = {}
    sums: dict[str, float] = {}
    for i, p in enumerate(outputs, 1):
        img = uio.imread(p)
        row: dict[str, float] = {}
        if gt_dir is not None:
            gt_path = gt_dir / f"{p.stem}_gt.jpg"
            if not gt_path.exists():
                raise FileNotFoundError(f"缺少 GT: {gt_path}")
            gt = uio.imread(gt_path)
            uio.assert_same_size(img, gt, names=(p.stem, f"{p.stem}_gt"))
            row.update(compute_fr(img, gt, fr_metrics, device))
        if nr_metrics:
            row.update(compute_nr(img, nr_metrics, device))
        per_image[p.stem] = row
        for k, v in row.items():
            sums[k] = sums.get(k, 0.0) + v
        if i % 10 == 0 or i == len(outputs):
            print(f"[评测] {i}/{len(outputs)}")

    means = {k: v / len(per_image) for k, v in sums.items()}
    result = {
        "n_images": len(per_image),
        "has_gt": gt_dir is not None,
        "metrics_used": {"fr": fr_metrics, "nr": nr_metrics},
        "per_image": per_image,
        "means": means,
        "composite": all_composites(means, cfg.get("metric_sets", {})),
        "note": "本地启发式综合分，非官方分数（官方公式公布前仅用于内部比较）",
    }
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="数据集级评测")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gt_dir", default=None, help="提供则计算 FR 指标（val 模式）")
    ap.add_argument("--out", required=True, help="eval.json 输出路径")
    ap.add_argument("--config", default=str(DEFAULT_EVAL_CFG))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    cfg = load_cfg(Path(args.config))
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    result = evaluate_set(Path(args.output_dir), gt_dir, cfg, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print("[均值]", {k: round(v, 4) for k, v in result["means"].items()})
    print("[综合]", {k: round(v, 4) for k, v in result["composite"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
