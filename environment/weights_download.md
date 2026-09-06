# 模型权重清单与下载（全部走脚本，幂等可重跑）

> 2026-09-07 更新：实测 DiffBIR/pyiqa 的"HuggingFace 裸 URL 运行时自动下载"
> 在国内机房全部被墙（torch.hub 直下不吃 HF_ENDPOINT，排障 #14/#16/#18），
> **一律用仓库脚本预下载**。脚本均为 wget -c 断点续传，中断重跑即续。

## 1. DiffBIR v2.x（主选后端，必下）

```bash
bash environment/download_diffbir_weights.sh    # 走 hf-mirror，约 6.3GB
```

| 文件 | 用途 | 体积 | 落位 |
|---|---|---|---|
| `realesrgan_s4_swinir_100k.pth` | stage1 去噪 cleaner | 87MB | `third_party/DiffBIR/weights/` |
| `sd2.1-base-zsnr-laionaes5.ckpt` | SD2.1 底模 | 4.9GB | 同上 |
| `DiffBIR_v2.1.pt` | ControlNet 主模型 | 1.4GB | 同上 |

DiffBIR 的 `load_file_from_url` 检测到本地同名文件即跳过网络下载，之后推理完全离线。

## 2. pyiqa 评测权重（评测必下）

```bash
bash environment/download_iqa_weights.sh    # 走 hf-mirror，全仓库约 12GB（一次下齐永久离线）
```

枚举官方权重仓库 `chaofengc/IQA-PyTorch-Weights`（含 LPIPS/DISTS/MUSIQ/NIQE/MANIQA
及 .mat 参数文件），下到 pyiqa 默认缓存 `~/.cache/torch/hub/pyiqa/`。

## 3. GFPGAN v1.4（人脸分支，可选；默认关闭）

```bash
python environment/download_weights.py    # github release 直链，约 350MB -> weights/gfpgan/
```

国内直连失败时挂代理（SSH 隧道见根 README 快速开始）或手动下载放置。

## 4. ResShift（保底后端，可选，暂未启用）

按其 Model Zoo 手动下载 `resshift_x4.pth` 到 `weights/resshift/`（github.com/zsyOAOA/ResShift）。

## 验证

```bash
ls -lh third_party/DiffBIR/weights/          # 3 个文件齐全
ls -lh ~/.cache/torch/hub/pyiqa/ | head      # LPIPS/DISTS/MUSIQ/niqe*.mat 等在位
pytest tests/test_smoke_diffbir.py -v -m remote -s   # 离线冒烟，应输出 k=1
```
