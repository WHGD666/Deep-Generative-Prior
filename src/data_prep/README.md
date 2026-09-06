# src/data_prep — 数据准备

把官方发布的数据变成**可靠、可指纹、可核查**的项目输入。

## 规划模块

- `extract_dataset.py` — 官方 zip 解压（处理 macOS UTF-8 文件名乱码，跳过 `__MACOSX`/`.DS_Store`）
- `fingerprint.py` — 生成数据清单与 MD5 指纹（写入 `data/manifest.json`），供实验 manifest 引用
- `stats.py` — 数据集统计核查（数量、分辨率分布、命名完整性），输出核对报告

## 约定

- 原始数据（`data/` 及官方 zip）**只读**，任何处理产出写到别处
- 解压结果目录结构固定为 `data/val/`（case1~5 的 `_lq`/`_gt`）与 `data/test/`（case1~100）
- 每次正式实验的 manifest 必须引用当天的数据指纹，数据变动可被察觉

## 当前状态

骨架阶段，代码未编写。首个任务：解压脚本（已在本地验证过等价逻辑）。
