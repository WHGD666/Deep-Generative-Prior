# GPU/驱动核查（开跑前 2 分钟确认）

## 需要确认的项

| 检查 | 命令 | 合格标准 |
|---|---|---|
| GPU 型号 | `nvidia-smi` | NVIDIA GeForce RTX 5090 |
| 驱动版本 | `nvidia-smi` 第一行 | **≥ 570**（CUDA 12.8 支持 Blackwell） |
| 显存 | `nvidia-smi --query-gpu=memory.total --format=csv` | ≈ 32GB (32607MiB) |
| 磁盘 | `df -h ~` | 可用 ≥ 35GB（环境+权重+产物） |
| CUDA 运行时 | 由 PyTorch 轮子自带，无需系统 CUDA | — |

## PyTorch 侧确认（setup 完成后）

```bash
python - <<'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))  # 期望 (12, 0) = sm_120
print("VRAM(GB):", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
EOF
```

- capability 必须是 `(12, 0)`，否则是装错轮子（见 TROUBLESHOOTING #1）。
- 实测算力基线：一张 512px tile、steps=50，期望秒级完成；
  若单 tile 分钟级，说明仍在用 CPU 跑（检查 `device` 配置）。
