#!/usr/bin/env bash
# DiffBIR v2.x 权重预下载（wget -c 断点续传，走 hf-mirror）
# ============================================================
# 背景：DiffBIR 用 torch.hub 直下 huggingface.co 裸 URL（HF_ENDPOINT 不生效），
# 国内机房直连被 reset。本脚本经 hf-mirror 把 3 个权重下到其仓库 weights/ 目录，
# 其 load_file_from_url 检测到本地文件即跳过下载。
# 用法：bash environment/download_diffbir_weights.sh
# 幂等：已存在且非空的文件自动跳过；中断重跑 wget -c 续传。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

DEST="third_party/DiffBIR/weights"
mkdir -p "$DEST"
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"

FILES=(
  "lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth"
  "lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt"
  "lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt"
)

echo "[信息] 镜像: $MIRROR, 目标目录: $ROOT/$DEST"
for f in "${FILES[@]}"; do
  name=$(basename "$f")
  if [[ -s "$DEST/$name" ]]; then
    echo "[跳过] $name 已存在"
    continue
  fi
  echo "[下载] $name"
  wget -c "$MIRROR/$f" -O "$DEST/$name"
done

echo "[清单]"
ls -lh "$DEST"
echo "[完成] 权重就位，可运行冒烟测试：pytest tests/test_smoke_diffbir.py -v -m remote -s"
