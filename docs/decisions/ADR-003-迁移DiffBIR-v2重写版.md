# ADR-003：迁移到 DiffBIR v2.x 重写版

- 状态：**已接受**
- 日期：2026-09-07
- 关联：ADR-001（选型）、ADR-002（CLI 子进程集成）

## 背景

初赛首日远程部署时发现：ADR-001 假设的 2023 版 DiffBIR 接口已失效——
官方在 2024-04 **完全重写**了代码库：
- 入口 `inference_bsr.py` → `inference.py`，参数体系全部更换；
- 权重从旧命名（general_swinir_v1.ckpt / general_full_v1.ckpt）迁移到
  HF `lxq007/DiffBIR-v2`（v2.1.pt / v2.pth / v1_general.pth / v1_face.pth 等）；
- 支持权重运行时自动下载（stage1 cleaner 等首次推理时经 HF 拉取）；
- 新增 `--task`（sr / denoise / faceSR）、多采样器、captioner 可选等能力；
- 老权重 HF 地址已不存在（实测 404）。

## 决策

跟随官方 main（v2.x）：
1. 后端 CLI 改为 `inference.py --task sr --upscale 4 --version v2.1 --captioner none ...`；
2. 权重不再预下载，首次推理自动拉取（子进程自动注入 `HF_ENDPOINT=hf-mirror`）；
3. 显式传 `--upscale 4` 与我们的"预缩放 1/4 + 运行时探测"机制互相印证；
4. 冒烟测试作为集成正确性的唯一闸门（ADR-002 原则不变）。

## 理由

- 老 v1 接口/权重已不可获取，回到 2023 commit 属于逆行且无权重可用；
- v2.1 是官方当前推荐版本（效果最好），且新代码基于 torch 2.x，与
  我们的 2.7.1+cu128 兼容风险低于 2023 老代码；
- CLI 集成（ADR-002）让这次迁移只改了后端驱动约 40 行，验证了架构决策。

## 影响

- `cfg_scale` 基准从 4.0 调整为 8（v2.1 示例口径），λ 调参在 val 上重新扫描；
- 首次冒烟会多花几分钟下载权重（约 1~2GB，走 hf-mirror）；
- face 版（v1_face）尚未集成，人脸分支仍默认关闭（GFPGAN 方案不变）。

## 后记（2026-09-07 实测修正，两处决策已被推翻）

1. **"权重运行时自动下载"不可用**：v2.x 经 `torch.hub` 直下 `huggingface.co`
   裸 URL，`HF_ENDPOINT` 对其无效（只影响 huggingface_hub 库），国内机房
   全部 Connection reset（排障 #14）。改为脚本预下载：
   `environment/download_diffbir_weights.sh`（6.3GB）。pyiqa 评测权重与
   timm 骨干同理（排障 #16/#18，`download_iqa_weights.sh` + 脚本注入 HF_ENDPOINT）。
2. **`--upscale 4` 改为 `--upscale 1`**：实测 v2.1 的 upscale 语义是"先把输入
   双三次放大 N 倍再修复"，×4 会把 512 瓦片撑到 2048² 再过 VAE，
   中间层注意力分配 16GiB 直接 OOM（排障 #15）。赛题为同分辨率修复，
   本就无需放大；改后 4K 原图直接 512 瓦片输入，速度约快 3 倍。
