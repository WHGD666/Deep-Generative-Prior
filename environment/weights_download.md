# 模型权重手动下载备选清单

> 自动下载失败时（网络/仓库变更），按下面清单手动下载并放到指定位置。
> 下载完成后重新运行 `python environment/download_weights.py` 会自动跳过已存在文件。

## 1. DiffBIR（主选后端，两阶段 general 版）

| 文件 | 用途 | 放置位置（相对项目根） |
|---|---|---|
| `general_swinir_v1.ckpt` | 第一阶段：保真复原（SwinIR） | `weights/diffbir/general_swinir_v1.ckpt` |
| `general_full_v1.ckpt` | 第二阶段：生成细化（SD） | `weights/diffbir/general_full_v1.ckpt` |

来源（按顺序尝试）：

1. HuggingFace 主源：`huggingface.co/XPixelGroup/DiffBIR`（Files 页面下载上面两个文件）
2. HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com` 后重跑 download_weights.py
3. 搜索 "DiffBIR general_full_v1.ckpt"（官方 README 的 Model Zoo 区块给出的链接）

> ⚠️ 文件名以官方 README 为准。若官方改了文件名，请同步修改
> `environment/download_weights.py` 的 `DIFFBIR_FILES` 与
> `configs/defaults/enhance.yaml` 注释，并在 EXP_LOG 记录。

## 2. GFPGAN v1.4（人脸分支，可选）

| 文件 | 来源 | 放置位置 |
|---|---|---|
| `GFPGANv1.4.pth` | github.com/TencentARC/GFPGAN/releases（v1.3.4 release 资产） | `weights/gfpgan/GFPGANv1.4.pth` |

## 3. ResShift（保底后端，可选）

| 文件 | 来源 | 放置位置 |
|---|---|---|
| `resshift_x4.pth`（或其 Model Zoo 提供的等价 ×4 权重） | github.com/zsyOAOA/ResShift 的 README/Model Zoo | `weights/resshift/` |

> ResShift 的任务配置 yaml 需与其权重要求一致（×4 SR 或 ×1 修复），
> 放置于其仓库内并由 `configs/defaults/enhance.yaml -> backends.resshift.config` 指向。
> 后端为 ×1 修复时，探测机制会自动识别（无需改我们的代码）。

## 放好之后的验证

```bash
ls -lh weights/diffbir/ weights/gfpgan/    # 确认文件大小正常（百 MB 级）
pytest tests/test_smoke_diffbir.py -v -m remote
```
