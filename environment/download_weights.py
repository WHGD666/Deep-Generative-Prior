"""模型权重下载（幂等；已存在且大小>1MB 时跳过）。

下载清单：
1. GFPGAN v1.4（人脸分支，可选）-> weights/gfpgan/
   官方 GitHub release 直链。
2. DiffBIR v2.x 权重【无需手动下载】：新版代码在首次推理时自动从 HF
   拉取（lxq007/DiffBIR-v2），国内环境由后端自动注入 hf-mirror 端点；
   首次运行冒烟测试时会看到权重下载进度，属正常现象。

用法：python environment/download_weights.py [--skip-gfpgan]
提示：国内环境如 GitHub 直连失败，先 export HF_ENDPOINT 与代理。
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = PROJECT_ROOT / "weights"

GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)


def have(path: Path, min_mb: float = 1.0) -> bool:
    return path.is_file() and path.stat().st_size > min_mb * 1e6


def download_url(url: str, dest: Path) -> Path:
    print(f"[下载] {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gfpgan", action="store_true")
    args = ap.parse_args()

    print("[信息] DiffBIR v2.x 权重无需手动下载：首次推理时自动拉取（hf-mirror）")

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
