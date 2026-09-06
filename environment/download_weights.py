"""模型权重下载（幂等；已存在且大小>1MB 时跳过）。

下载清单：
1. DiffBIR 两阶段权重（general 通用版）-> weights/diffbir/
   - general_swinir_v1.ckpt  （第一阶段：保真复原）
   - general_full_v1.ckpt    （第二阶段：生成细化）
   默认 HF 仓库 XPixelGroup/DiffBIR；可用 --repo 换源，失败时看
   weights_download.md 的手动备选清单。
2. GFPGAN v1.4（人脸分支，可选）-> weights/gfpgan/
   官方 GitHub release 直链。

用法：python environment/download_weights.py [--repo <hf_repo_id>]
提示：国内环境先 export HF_ENDPOINT=https://hf-mirror.com
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "weights"

DIFFBIR_FILES = ["general_swinir_v1.ckpt", "general_full_v1.ckpt"]
DEFAULT_HF_REPO = "XPixelGroup/DiffBIR"
GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)


def have(path: Path, min_mb: float = 1.0) -> bool:
    return path.is_file() and path.stat().st_size > min_mb * 1e6


def download_hf(repo: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    got = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest_dir)
    return Path(got)


def download_url(url: str, dest: Path) -> Path:
    print(f"[下载] {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_HF_REPO, help="DiffBIR 权重的 HF 仓库 id")
    ap.add_argument("--skip-gfpgan", action="store_true")
    args = ap.parse_args()

    ok = True
    diffbir_dir = WEIGHTS / "diffbir"
    diffbir_dir.mkdir(parents=True, exist_ok=True)
    for f in DIFFBIR_FILES:
        dest = diffbir_dir / f
        if have(dest, 100):
            print(f"[跳过] {dest} 已存在")
            continue
        try:
            got = download_hf(args.repo, f, diffbir_dir)
            print(f"[完成] {got}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"[失败] {f}: {e}\n"
                  f"       手动备选清单见 environment/weights_download.md，"
                  f"下载后放到 {dest}")

    if not args.skip_gfpgan:
        gfp_dir = WEIGHTS / "gfpgan"
        gfp_dir.mkdir(parents=True, exist_ok=True)
        dest = gfp_dir / "GFPGANv1.4.pth"
        if have(dest):
            print(f"[跳过] {dest} 已存在")
        else:
            try:
                download_url(GFPGAN_URL, dest)
                print(f"[完成] {dest}")
            except Exception as e:  # noqa: BLE001
                print(f"[警告] GFPGAN 权重下载失败（人脸分支暂不可用）: {e}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
