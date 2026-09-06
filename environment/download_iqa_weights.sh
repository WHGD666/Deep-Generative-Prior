#!/usr/bin/env bash
# pyiqa 评测权重预下载（hf-mirror 枚举官方权重仓库；幂等 + 断点续传）
# ============================================================
# 背景：pyiqa 的 LPIPS/DISTS/MUSIQ/NIQE/MANIQA 等权重是 huggingface.co
# 裸 URL + torch.hub 直下（HF_ENDPOINT 不生效，排障 #16），国内机房被 reset。
# 原理：经 hf-mirror API 枚举 pyiqa 官方权重仓库 chaofengc/IQA-PyTorch-Weights
# 的全部权重文件，逐个 wget -c 到 pyiqa 默认缓存 ~/.cache/torch/hub/pyiqa/
# （其 load_file_from_url 检测到本地同名文件即跳过下载）。
# 用法：bash environment/download_iqa_weights.sh
set -euo pipefail

CACHE="${TORCH_HOME:-$HOME/.cache/torch}/hub/pyiqa"
REPO="chaofengc/IQA-PyTorch-Weights"
MIRROR="https://hf-mirror.com"
mkdir -p "$CACHE"

# 清理旧版脚本误抓的文档 HTML（markdown 链接误匹配所致）
rm -f "$CACHE/parallelism)" "$CACHE/kv_cache)." "$CACHE/pipeline-cat-chonk.jpeg"

echo "[信息] 枚举权重仓库: $REPO (via $MIRROR)"
LIST=$(curl -fsSL "$MIRROR/api/models/$REPO" | python -c "
import json, sys
data = json.load(sys.stdin)
exts = ('.pth', '.pt', '.ckpt', '.pkl', '.npy', '.npz', '.onnx', '.bin', '.safetensors', '.csv', '.txt')
for f in data.get('siblings', []):
    name = f['rfilename']
    if name.lower().endswith(exts):
        print(name)
") || { echo "[致命] 仓库枚举失败（$MIRROR API 不可达）"; exit 1; }

if [[ -z "$LIST" ]]; then
  echo "[致命] 仓库文件清单为空，请人工检查 $REPO"
  exit 1
fi

echo "[信息] 共 $(echo "$LIST" | wc -l) 个权重文件"
FAIL=0
while IFS= read -r name; do
  [[ -z "$name" ]] && continue
  if [[ -s "$CACHE/$name" ]]; then
    echo "[跳过] $name 已存在"
    continue
  fi
  echo "[下载] $name"
  # 不用 rm：失败保留断点，重跑 -c 续传
  if ! wget -c --tries=3 --timeout=60 "$MIRROR/$REPO/resolve/main/$name" -O "$CACHE/$name"; then
    echo "[警告] $name 下载失败（保留断点，重跑本脚本续传）"
    FAIL=1
  fi
done <<< "$LIST"

echo "[清单]"
ls -lh "$CACHE"
if [[ "$FAIL" == "1" ]]; then
  echo "[完成-有告警] 部分权重未完成，重跑本脚本续传"
  exit 1
fi
echo "[完成] IQA 权重就位，可运行评测"
