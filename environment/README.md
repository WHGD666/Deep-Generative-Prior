# environment — 云端运行环境（RTX 5090）

所有训练/推理在**远程租用的 RTX 5090（约 60GB 存储）**上进行；本目录固化环境定义，保证"云端全量跑"与"决赛现场复现"环境一致。

## 计划内容

- `setup_5090.sh` — 一键环境搭建脚本（Python 3.10 venv/conda、PyTorch + CUDA、DiffBIR 依赖、pyiqa 评测库）
- `requirements.txt` — 锁定版本的需求清单（安装后用 `pip freeze` 回填）
- `weights_download.md` — 模型权重清单（DiffBIR + SD2.1 约 6~8GB）与官方下载地址、存放路径约定（`weights/`）
- `nvidia_smi_check.md` — 驱动/CUDA 兼容性核查记录

## 执行纪律（重要）

- 环境命令**由用户本人执行**：脚本写好放这里，交付给用户运行，不代跑
- 任何依赖变更：先改 `requirements.txt` 并写明原因，再给出增量安装命令
- 60GB 存储红线：权重只保留主干；每次装完执行 `pip cache purge`、清理 HuggingFace 缓存

## 当前状态

骨架阶段。下一里程碑（9/7）：完成 `setup_5090.sh` 与 requirements 清单。
