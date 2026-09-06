# 模型权重手动下载备选清单

> 2026-09 更新：DiffBIR 官方重写代码库（v2.x），权重迁移到 `lxq007/DiffBIR-v2`，
> 且支持**首次推理时自动下载**——正常情况下本文件只剩 GFPGAN 需要关心。

## 1. DiffBIR（主选后端，v2.x）

**默认无需手动下载**：`src/enhance/backends` 已注入 `HF_ENDPOINT=hf-mirror`，
首次运行冒烟测试/管线时自动拉取（v2.1.pt 等约 1~2GB + stage1 cleaner）。

手动备选（自动下载失败时）：
- HF 仓库：`huggingface.co/lxq007/DiffBIR-v2`（镜像：hf-mirror.com/lxq007/DiffBIR-v2）
- 相关文件：`v2.1.pt`（推荐）、`v2.pth`、`v1_general.pth`、`codeformer_swinir.ckpt`（stage1）
- 自动下载失败时检查：`HF_ENDPOINT` 是否被代理脚本覆盖、磁盘剩余空间

## 2. GFPGAN v1.4（人脸分支，可选）

| 文件 | 来源 | 放置位置 |
|---|---|---|
| `GFPGANv1.4.pth` | github.com/TencentARC/GFPGAN/releases（v1.3.4 release 资产） | `weights/gfpgan/GFPGANv1.4.pth` |

## 3. ResShift（保底后端，可选）

| 文件 | 来源 | 放置位置 |
|---|---|---|
| `resshift_x4.pth`（或其 Model Zoo 提供的等价 ×4 权重） | github.com/zsyOAOA/ResShift 的 README/Model Zoo | `weights/resshift/` |

> ResShift 的任务配置 yaml 需与其权重要求一致（×4 SR 或 ×1 修复），
> 后端为 ×1 修复时，探测机制会自动识别（无需改我们的代码）。

## 放好之后的验证

```bash
ls -lh weights/gfpgan/        # GFPGANv1.4.pth 约 350MB
pytest tests/test_smoke_diffbir.py -v -m remote   # 首次运行会自动下载 DiffBIR 权重
```
