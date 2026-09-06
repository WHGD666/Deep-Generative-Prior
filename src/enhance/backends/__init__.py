"""第三方扩散模型后端（CLI 批处理驱动）。

设计原则（ADR-002）：本项目代码**不 import 第三方仓库内部 API**，
只通过其官方 CLI 脚本以子进程方式调用——第三方升级不会击穿我们的代码；
集成正确性由 tests/test_smoke_diffbir.py 在远程环境冒烟验证。

统一接口：
    build_backend(cfg) -> driver，driver.enhance_batch(tiles)->tiles
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _run_cli(cmd: list[str], cwd: Path, device: str | None, tag: str) -> None:
    env = os.environ.copy()
    if device:
        # 允许 "cuda:1" 这类写法 -> 映射为可见设备编号
        env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[-1]
    proc = subprocess.run(
        cmd, cwd=str(cwd), env=env, capture_output=True, text=True
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-2000:]
        raise RuntimeError(f"[{tag}] CLI 调用失败 (exit={proc.returncode}):\n{tail}")


def _save_tiles(tiles: list[np.ndarray], in_dir: Path) -> list[str]:
    stems = []
    for i, t in enumerate(tiles):
        stem = f"tile_{i:04d}"
        Image.fromarray(t).save(in_dir / f"{stem}.png")
        stems.append(stem)
    return stems


def _collect_outputs(out_dir: Path, stems: list[str]) -> list[np.ndarray]:
    files = {p.stem: p for p in out_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    outs = []
    for s in stems:
        if s not in files:
            found = [p for p in files.values() if p.stem.startswith(s)]
            if not found:
                raise RuntimeError(f"后端输出缺少块: {s}（目录内容: {list(files)[:5]}...）")
            files[s] = found[0]
        outs.append(np.array(Image.open(files[s]).convert("RGB"), dtype=np.uint8))
    return outs


class DiffBIRBackend:
    """XPixelGroup/DiffBIR 官方 CLI（inference_bsr.py）驱动。

    配置（configs/defaults/enhance.yaml -> backends.diffbir）：
        repo_dir: third_party/DiffBIR      # clone 位置
        version: general                   # general | face
        steps: 50                          # 采样步数
        cfg_scale: 4.0                     # 引导强度
        device: cuda:0
        extra_args: []                     # 追加 CLI 参数（应急通道）
    """

    name = "diffbir"

    def __init__(self, cfg: dict):
        self.repo_dir = Path(cfg["repo_dir"]).resolve()
        self.script = self.repo_dir / "inference_bsr.py"
        if not self.script.is_file():
            raise FileNotFoundError(
                f"未找到 {self.script}，请先按 environment/README.md clone DiffBIR"
            )
        self.version = cfg.get("version", "general")
        self.steps = int(cfg.get("steps", 50))
        self.cfg_scale = float(cfg.get("cfg_scale", 4.0))
        self.device = cfg.get("device", "cuda:0")
        self.extra_args = list(cfg.get("extra_args", []))

    def enhance_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        with tempfile.TemporaryDirectory(prefix="diffbir_") as tmp:
            in_dir, out_dir = Path(tmp) / "in", Path(tmp) / "out"
            in_dir.mkdir()
            stems = _save_tiles(tiles, in_dir)
            cmd = [
                sys.executable, str(self.script),
                "--input", str(in_dir),
                "--output", str(out_dir),
                "--version", self.version,
                "--steps", str(self.steps),
                "--cfg_scale", str(self.cfg_scale),
                *self.extra_args,
            ]
            _run_cli(cmd, self.repo_dir, self.device, self.name)
            return _collect_outputs(out_dir, stems)


class ResShiftBackend:
    """ResShift 官方 CLI（inference_resshift.py）驱动（保底管线）。

    配置（configs/defaults/enhance.yaml -> backends.resshift）：
        repo_dir: third_party/ResShift
        config: ResShift/configs/...yaml   # 其仓库内任务配置（SR×4 或 ×1 修复）
        ckpt: weights/resshift/xxx.pth
        device: cuda:0
        extra_args: []
    """

    name = "resshift"

    def __init__(self, cfg: dict):
        self.repo_dir = Path(cfg["repo_dir"]).resolve()
        self.script = self.repo_dir / "inference_resshift.py"
        if not self.script.is_file():
            raise FileNotFoundError(
                f"未找到 {self.script}，请先 clone ResShift 并按 weights_download.md 下载权重"
            )
        self.task_config = str(cfg["config"])
        self.ckpt = str(cfg["ckpt"])
        self.device = cfg.get("device", "cuda:0")
        self.extra_args = list(cfg.get("extra_args", []))

    def enhance_batch(self, tiles: list[np.ndarray]) -> list[np.ndarray]:
        with tempfile.TemporaryDirectory(prefix="resshift_") as tmp:
            in_dir, out_dir = Path(tmp) / "in", Path(tmp) / "out"
            in_dir.mkdir()
            stems = _save_tiles(tiles, in_dir)
            cmd = [
                sys.executable, str(self.script),
                "--input", str(in_dir),
                "--output", str(out_dir),
                "--config", self.task_config,
                "--ckpt", self.ckpt,
                *self.extra_args,
            ]
            _run_cli(cmd, self.repo_dir, self.device, self.name)
            return _collect_outputs(out_dir, stems)


def build_backend(cfg: dict):
    """按配置构造后端驱动。cfg: {"name": "diffbir"|"resshift", "<name>": {...}}"""
    name = cfg.get("name")
    if name == "diffbir":
        return DiffBIRBackend(cfg["diffbir"])
    if name == "resshift":
        return ResShiftBackend(cfg["resshift"])
    raise ValueError(f"未知后端: {name}（可选 diffbir / resshift）")
