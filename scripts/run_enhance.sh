#!/usr/bin/env bash
# 全量增强管线（远程 5090 执行；先 conda activate camera310）
# 用法: ./scripts/run_enhance.sh <RUN_ID> <INPUT_DIR> <INPUT_PATTERN> <OUTPUT_DIR> [额外参数透传给 python -m src.run_pipeline]
# 示例:
#   ./scripts/run_enhance.sh 20260907_diffbir_val  data/val  "{case}_lq.jpg" experiments/20260907_diffbir_val/outputs
#   ./scripts/run_enhance.sh 20260909_diffbir_test data/test "{case}.jpg"   experiments/20260909_diffbir_test/outputs --limit 3
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID=${1:?缺少 RUN_ID}
INPUT_DIR=${2:?缺少 INPUT_DIR}
PATTERN=${3:?缺少 INPUT_PATTERN，如 "{case}.jpg"}
OUTPUT_DIR=${4:?缺少 OUTPUT_DIR}
shift 4

exec python -m src.run_pipeline \
  --run_id "$RUN_ID" \
  --input_dir "$INPUT_DIR" \
  --input_pattern "$PATTERN" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
