# src/submit — 提交打包

从增强结果到可上传 zip 的最后一步，**用程序化校验消灭人为失误**（提交失资格的代价远大于多写一个脚本）。

## 规划模块

- `assemble.py` — 组装 `output_dir/`：100 张结果图就位
- `validate.py` — 提交前强制校验：
  - 文件名 `case1.jpg`~`case100.jpg` 严格对应、无中文符号
  - 每张分辨率与对应输入**逐张比对**（横 4096×3072 / 竖 3072×4096）
  - jpg 格式、无多余文件
- `pack.py` — 生成 zip（命名 `赛题一_<作品名>_<队名>_<手机号>.zip`，≤10GB）并输出文件清单到 `submissions/`

## 约定

- `validate.py` 不通过时**禁止**打包，无旁路开关
- zip 本体不入库（`.gitignore` 已覆盖），入库的是清单 + 分数记录（`submissions/` 与根 `SUBMISSIONS.md`）
