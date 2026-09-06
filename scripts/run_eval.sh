#!/usr/bin/env bash
# 评测（远程 5090 执行；FR 需要 GT，仅 test 无 GT 走 NR-only）
# 用法:
#   val 模式: ./scripts/run_eval.sh <RUN_ID> <OUTPUT_DIR> data/val
#   test 模式: ./scripts/run_eval.sh <RUN_ID> <OUTPUT_DIR> none
set -euo pipefail
cd "$(dirname "$0")/.."

# timm（MANIQA 骨干）等经 huggingface_hub 下载：默认端点国内被墙，默认 hf-mirror（排障 #18）
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

RUN_ID=${1:?缺少 RUN_ID}
OUTPUT_DIR=${2:?缺少 OUTPUT_DIR}
GT_DIR=${3:-none}

RUN_DIR="experiments/$RUN_ID"
if [[ "$GT_DIR" == "none" ]]; then
  python -m src.evaluate.run_eval \
    --output_dir "$OUTPUT_DIR" \
    --out "$RUN_DIR/eval_test.json"
else
  python -m src.evaluate.run_eval \
    --output_dir "$OUTPUT_DIR" \
    --gt_dir "$GT_DIR" \
    --out "$RUN_DIR/eval.json"
fi
echo "[run_eval] 完成: $RUN_DIR/eval*.json"
