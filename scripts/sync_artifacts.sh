#!/usr/bin/env bash
# 云端产物回传本地（在本机执行，非云端；rsync 示例，按租用平台给的 SSH 信息替换）
# 用法: ./scripts/sync_artifacts.sh <user@host> [远端项目路径]
set -euo pipefail

HOST=${1:?用法: sync_artifacts.sh user@host [remote_path]}
REMOTE=${2:-~/Deep-Generative-Prior}

# 回传实验记录（manifest/eval/log，小文件，全量拿回）
rsync -avz --include="manifest.json" --include="eval*.json" \
      --include="config_effective.yaml" --include="*.log" --exclude="*" \
      "$HOST:$REMOTE/experiments/"  experiments/

# 按需回传某个 run 的输出图片（大文件，指定 run 再拉）
# rsync -avz "$HOST:$REMOTE/experiments/<RUN_ID>/outputs/"  experiments/<RUN_ID>/outputs/

# 提交快照
rsync -avz "$HOST:$REMOTE/submissions/" submissions/

echo "[sync_artifacts] 完成"
