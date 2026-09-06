#!/usr/bin/env bash
# pyiqa 评测权重预下载（hf-mirror 版，幂等，断点续传）
# ============================================================
# 背景：pyiqa 的 LPIPS/DISTS/MUSIQ/MANIQA/NIQE 等权重是 huggingface.co
# 裸 URL + torch.hub 直下（HF_ENDPOINT 不生效，排障 #16），国内机房被 reset。
# 本脚本从安装的 pyiqa 包源码里收集全部 HF 裸链，逐个经 hf-mirror 下载到
# pyiqa 默认缓存目录 ~/.cache/torch/hub/pyiqa/（其 load_file_from_url
# 检测到本地文件即跳过下载）。
# 用法：bash environment/download_iqa_weights.sh
set -euo pipefail

CACHE="${TORCH_HOME:-$HOME/.cache/torch}/hub/pyiqa"
mkdir -p "$CACHE"

PKG=$(python - <<'EOF'
import pathlib, pyiqa
print(pathlib.Path(pyiqa.__file__).parent)
EOF
)

echo "[信息] pyiqa 包: $PKG"
echo "[信息] 缓存目录: $CACHE"

# 收集包内全部 huggingface.co 裸链（default_model_urls 常量）
URLS=$(grep -rhoE "https://huggingface\.co/[^'\"[:space:]]+" "$PKG" | sort -u || true)
if [[ -z "$URLS" ]]; then
  echo "[警告] 未在 pyiqa 源码中发现 HF 链接（版本可能已变更），请人工检查"
  exit 0
fi

FAIL=0
for u in $URLS; do
  name=$(basename "$u")
  if [[ -s "$CACHE/$name" ]]; then
    echo "[跳过] $name 已存在"
    continue
  fi
  echo "[下载] $name"
  # 链接替换为 hf-mirror；wget -c 支持断点续传
  if ! wget -c "${u/https:\/\/huggingface.co/https://hf-mirror.com}" -O "$CACHE/$name"; then
    echo "[警告] $name 下载失败（评测用到该指标时会再试）"
    rm -f "$CACHE/$name"
    FAIL=1
  fi
done

echo "[清单]"
ls -lh "$CACHE"
if [[ "$FAIL" == "1" ]]; then
  echo "[完成-有告警] 部分权重失败，可重跑本脚本续传"
  exit 1
fi
echo "[完成] IQA 权重就位"
