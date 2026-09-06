"""端到端增强管线：预处理 -> 分块扩散增强 -> 后处理 -> 一致性核查 -> 写出。

这是本项目的唯一全量推理入口（scripts/run_enhance.sh 调用它）。
本地（Windows）不运行本脚本；运行环境为远程 RTX 5090（Linux）。

用法示例（远程，项目根目录下）：
    python -m src.run_pipeline \
        --run_id 20260907_diffbir_val \
        --input_dir data/val --input_pattern "{case}_lq.jpg" \
        --output_dir experiments/20260907_diffbir_val/outputs

    python -m src.run_pipeline \
        --run_id 20260909_diffbir_test \
        --input_dir data/test --input_pattern "{case}.jpg" \
        --output_dir experiments/20260909_diffbir_test/outputs

输出文件名恒为 <case>.jpg（赛题提交命名），val 模式自动剥离 _lq 后缀。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.enhance.backends import build_backend
from src.enhance.tiled_inference import TiledEngine, infer_upscale
from src.preprocess import deblur as deblur_mod
from src.preprocess import denoise as denoise_mod
from src.postprocess import color_align, consistency_check, lambda_blend, lowfreq_backfill
from src.utils import io as uio
from src.utils.image_ops import downscale_int, resize_to
from src.utils.logging import get_logger
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "preprocess": "configs/defaults/preprocess.yaml",
    "enhance": "configs/defaults/enhance.yaml",
    "postprocess": "configs/defaults/postprocess.yaml",
}


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:  # noqa: BLE001 - git 不可用不应阻断实验
        return {"git_commit": None, "git_dirty": None}


def discover_inputs(input_dir: Path, pattern: str) -> dict[str, Path]:
    """返回 {case_id: 路径}。pattern 中 {case} 为样例编号占位。"""
    prefix, suffix = pattern.split("{case}", 1)
    cases: dict[str, Path] = {}
    for p in sorted(input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        stem = p.stem
        if stem.startswith(prefix) and stem.endswith(suffix):
            case = stem[len(prefix): len(stem) - len(suffix) if suffix else None]
            if case:
                cases[case] = p
    if not cases:
        raise FileNotFoundError(f"{input_dir} 中没有匹配 {pattern} 的图片")
    return cases


def center_probe_tile(img, size: int = 128):
    """从图中心取一块小样本作为后端探测块。"""
    h, w = img.shape[:2]
    if h <= size or w <= size:
        return img
    y0, x0 = h // 2 - size // 2, w // 2 - size // 2
    return img[y0: y0 + size, x0: x0 + size]


def run_image(case, img, cfg, backend, engine, probe_k, prescale, logger):
    """处理单张图，返回 (结果数组, 记录)。"""
    t0 = time.time()

    pre = img
    pcfg = cfg["preprocess"]
    if pcfg.get("denoise", {}).get("enabled"):
        pre = denoise_mod.denoise(
            pre, **{k: v for k, v in pcfg["denoise"].items() if k != "enabled"}
        )
    if pcfg.get("deblur", {}).get("enabled"):
        pre = deblur_mod.deblur(
            pre, **{k: v for k, v in pcfg["deblur"].items() if k != "enabled"}
        )

    work = downscale_int(pre, prescale) if prescale > 1 else pre
    enhanced = engine.run(work, backend.enhance_batch, upscale=probe_k)
    if enhanced.shape[:2] != img.shape[:2]:
        # 预缩放除不尽或后端非严格倍率时，精确还原到输入尺寸（红线 R4）
        enhanced = resize_to(enhanced, img.shape[0], img.shape[1])
    uio.assert_same_size(enhanced, img, names=("enhanced", "input"))

    post_cfg = cfg["postprocess"]
    result = enhanced
    if post_cfg.get("color_align", {}).get("enabled"):
        result = color_align.apply(result, ref=img, cfg=post_cfg["color_align"])
    if post_cfg.get("lowfreq_backfill", {}).get("enabled"):
        result = lowfreq_backfill.apply(result, ref=img, cfg=post_cfg["lowfreq_backfill"])
    if post_cfg.get("lambda_blend", {}).get("enabled"):
        result = lambda_blend.apply(result, fidelity=pre, cfg=post_cfg["lambda_blend"])
    if post_cfg.get("face_branch", {}).get("enabled"):
        from src.postprocess import face_branch  # 延迟导入：依赖 gfpgan/facexlib

        result = face_branch.apply(result, cfg=post_cfg["face_branch"])

    uio.assert_same_size(result, img, names=("final", "input"))

    info = {"seconds": None, "warnings": []}
    ccfg = post_cfg.get("consistency_check", {})
    if ccfg.get("enabled", True):
        warnings = consistency_check.check(result, img, ccfg)
        info["warnings"] = warnings
        for w in warnings:
            logger.warning(f"[{case}] 一致性告警: {w}")

    dt = time.time() - t0
    info["seconds"] = round(dt, 1)
    logger.info(
        f"[{case}] 完成, 耗时 {dt:.1f}s (输出 {result.shape[1]}x{result.shape[0]})"
    )
    return result, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="端到端增强管线")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--input_pattern", default="{case}.jpg",
                    help='如 "{case}.jpg"（test）或 "{case}_lq.jpg"（val）')
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--preprocess", default=DEFAULTS["preprocess"])
    ap.add_argument("--enhance", default=DEFAULTS["enhance"])
    ap.add_argument("--postprocess", default=DEFAULTS["postprocess"])
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张（调试）")
    ap.add_argument("--subset", default="", help="逗号分隔 case 列表（调试）")
    args = ap.parse_args(argv)

    run_dir = PROJECT_ROOT / "experiments" / args.run_id
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("pipeline", args.run_id, log_dir=run_dir)

    cfg = {k: load_yaml(getattr(args, k)) for k in DEFAULTS}
    (run_dir / "config_effective.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    command = (
        f"python -m src.run_pipeline --run_id {args.run_id} "
        f"--input_dir {args.input_dir} --input_pattern {args.input_pattern} "
        f"--output_dir {args.output_dir}"
    )
    manifest = {
        "run_id": args.run_id,
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
        **git_info(),
        "command": command,
        "configs": {k: str(Path(getattr(args, k)).resolve()) for k in DEFAULTS},
        "python_version": sys.version.split()[0],
        "n_images": None,
        "backend": None,
        "per_image": {},
        "failure_type": None,
        "failure_message": None,
    }

    try:
        set_seed(int(cfg["enhance"].get("seed", 0)))
        backend = build_backend(cfg["enhance"]["backend"])
        engine = TiledEngine(
            tile_size=int(cfg["enhance"]["tile_size"]),
            overlap=int(cfg["enhance"]["overlap"]),
        )

        cases = discover_inputs(Path(args.input_dir), args.input_pattern)
        if args.subset:
            keep = {s.strip() for s in args.subset.split(",") if s.strip()}
            cases = {c: p for c, p in cases.items() if c in keep}
        if args.limit:
            cases = dict(list(cases.items())[: args.limit])
        manifest["n_images"] = len(cases)
        logger.info(f"[{args.run_id}] 后端={backend.name}, 待处理 {len(cases)} 张")

        first_img = uio.imread(next(iter(cases.values())))
        probe_k = infer_upscale(backend.enhance_batch, center_probe_tile(first_img))
        prescale_cfg = cfg["enhance"].get("prescale", "auto")
        prescale = probe_k if prescale_cfg == "auto" else int(prescale_cfg)
        manifest["backend"] = {
            "name": backend.name, "probe_upscale": probe_k, "prescale": prescale,
            "tile_size": cfg["enhance"]["tile_size"],
            "overlap": cfg["enhance"]["overlap"],
        }
        logger.info(
            f"[探测] 后端放大倍率 k={probe_k}, 预缩放 p={prescale}, "
            f"tile={cfg['enhance']['tile_size']}(overlap={cfg['enhance']['overlap']})"
        )

        for i, (case, path) in enumerate(cases.items(), 1):
            logger.info(f"[{i}/{len(cases)}] case={case}")
            img = uio.imread(path)
            result, info = run_image(
                case, img, cfg, backend, engine, probe_k, prescale, logger
            )
            uio.imwrite(out_dir / f"{case}.jpg", result)
            manifest["per_image"][case] = info

        manifest["status"] = "completed"
        manifest["end_time"] = datetime.now(timezone.utc).isoformat()
    except Exception as e:  # noqa: BLE001 - 失败实验也要留下完整记录
        manifest["status"] = "failed"
        manifest["failure_type"] = type(e).__name__
        manifest["failure_message"] = str(e)[-1500:]
        manifest["end_time"] = datetime.now(timezone.utc).isoformat()
        logger.error(f"[失败] {type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
    finally:
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    if manifest["status"] != "completed":
        return 1
    logger.info(f"[完成] 输出目录: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
