#!/usr/bin/env bash
# LQ 不增强对照组：把 val 输入原样当"输出"评测，量化管线真实增益。
# 实验纪律：任何调参结论必须与该控制组同口径对比后才算数。
# 用法: ./scripts/eval_lq_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_DIR="experiments/lq_baseline"
mkdir -p "$BASE_DIR/outputs"
n=0
for f in data/val/*_lq.jpg; do
  case=$(basename "${f%_lq.jpg}")
  cp -f "$f" "$BASE_DIR/outputs/$case.jpg"
  n=$((n + 1))
done
echo "[lq_baseline] 已就位 $n 张（原始 lq，未增强，直接评测）"
exec bash scripts/run_eval.sh lq_baseline "$BASE_DIR/outputs" data/val
