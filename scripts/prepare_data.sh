#!/usr/bin/env bash
# 数据准备：解压官方 zip -> 指纹 -> 统计核查（远程 5090 / 本地均可）
# 前置：官方压缩包已位于项目根（默认 赛题一.zip）
set -euo pipefail
cd "$(dirname "$0")/.."

ZIP=${1:-赛题一.zip}

python -m src.data_prep.extract_dataset --zip "$ZIP" --out data
python -m src.data_prep.fingerprint --root data --out data/manifest.json
python -m src.data_prep.stats --root data --json experiments/data_stats.json
echo "[prepare_data] 全部完成"
