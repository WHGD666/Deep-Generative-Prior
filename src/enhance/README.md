# src/enhance — 扩散增强核心（主战场）

赛题合规性的承担者：**Diffusion 架构图像增强**。主选 DiffBIR（见 `docs/decisions/ADR-001`），ResShift 保底。

## 规划模块

- `model_zoo.py` — 模型/权重加载封装（DiffBIR、ResShift），统一接口，权重路径走配置
- `diffbir_pipeline.py` — DiffBIR 两阶段推理封装（第一阶段保真复原 / 第二阶段生成细化），阶段与采样参数可配
- `tiled_inference.py` — 4K 分块推理调度：按 `preprocess/tiling.py` 的分块方案逐块推理
- `blend.py` — 分块融合：重叠区羽化、接缝质量控制（接缝伪影是赛题扣分点）
- `resshift_pipeline.py` — 保底管线

## 关键设计约束

- 输入/输出**分辨率严格一致**（4K→4K），禁止任何隐式缩放
- 推理超参（步数、CFG、引导强度、种子）全部进配置，一次实验一份配置快照
- 模型权重不入库（`weights/` 已 gitignore），manifest 记录权重文件名 + 哈希
- 显存策略：fp16 + CPU offload，单卡 5090 为基准环境

## 当前状态

骨架阶段。首任务：DiffBIR 在 val case1 的 512 小图上跑通。
