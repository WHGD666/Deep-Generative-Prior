#!/usr/bin/env bash
# 打包提交 zip（内含强制校验，校验不过自动拒绝）
# 用法: ./scripts/pack_submission.sh <RUN_ID> <SUB_ID>
# 前置: configs/defaults/submit.yaml 三个占位字段已填写
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID=${1:?缺少 RUN_ID}
SUB_ID=${2:?缺少 SUB_ID，如 sub01}
RUN_DIR="experiments/$RUN_ID"

python -m src.submit.assemble \
  --run_outputs "$RUN_DIR/outputs" \
  --output_dir output_dir

python -m src.submit.pack \
  --output_dir output_dir \
  --input_dir data/test \
  --config configs/defaults/submit.yaml \
  --run_id "$RUN_ID" \
  --sub_id "$SUB_ID"

echo "[pack_submission] 完成。提交前请过一遍根 README 第 9 节检查清单，"
echo "并在上传后当天把官方分数回填 SUBMISSIONS.md。"
