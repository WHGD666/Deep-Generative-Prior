# src/enhance — 扩散增强核心（主战场）

赛题合规性的承担者：**Diffusion 架构图像增强**。主选 DiffBIR v2.1（见 `docs/decisions/ADR-001/003`），ResShift 保底。

## 模块

- `backends/` — 第三方后端 CLI 子进程驱动（ADR-002：不 import 第三方内部 API）。
  `DiffBIRBackend`（主选，`--task sr --upscale 1 --version v2.1`）与 `ResShiftBackend`（保底），
  统一 `enhance_batch(tiles) -> tiles` 批接口
- `tiled_inference.py` — 4K 分块推理调度：按 `preprocess/tiling.py` 的分块方案批量推理，
  含 `infer_upscale` 运行时倍率探测（不信任后端文档，实测为准）
- 融合在 `src/preprocess/tiling.py::merge_tiles`（归一化羽化加权，恒等回环无损有测试）

## 关键设计约束

- 输入/输出**分辨率严格一致**（4K→4K），禁止任何隐式缩放
- **同分辨率修复**：DiffBIR `--upscale 1`（其 upscale 语义是先把输入双三次放大 N 倍再修复，
  ×4 会把 512 瓦片撑到 2048 过 VAE 直接 OOM——排障 #15）
- 瓦片 512（v2.1 原生工作分辨率）+ 64 重叠羽化；`prescale: auto` 探测 k=1 自然不缩放
- 推理超参（步数、CFG、种子）全部进配置，一次实验一份配置快照
- 模型权重不入库（`weights/` 与 `third_party/*/weights/` 已 gitignore），manifest 记录配置快照
- 显存策略：fp16 + 512 瓦片，子进程注入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## 当前状态

全链路已在远程 5090 跑通（首个 run 见根 `EXP_LOG.md`：20260907_val_probe）。
第三方集成采用 CLI 子进程 + 运行时探测（ADR-002）；集成正确性由
`tests/test_smoke_diffbir.py`（remote 标记）把关。
