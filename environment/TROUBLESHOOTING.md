# 远程环境排错手册（RTX 5090 / Blackwell）

> 按报错关键词定位。每条先给现象，再给处置。修完重跑 setup_5090.sh 会自动续接。

## 1. torch 报 `sm_120 is not compatible` / `no kernel image`

现象：`import torch` 后调用 cuda 即报架构不兼容。
原因：装成了 cu118/cu121 轮子，不含 Blackwell（sm_120）内核。
处置：
```bash
pip uninstall -y torch torchvision
pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_device_capability(0))"   # 应打印 (12, 0)
```

## 2. `ImportError: cannot import name 'rgb_to_grayscale' from 'torchvision.transforms.functional_tensor'`

原因：basicsr 1.x 旧导入路径（facexlib/gfpgan 的依赖链）。
处置：`python environment/patch_basicsr.py`（幂等）。

## 3. HuggingFace 下载超时/断连

处置：`export HF_ENDPOINT=https://hf-mirror.com` 后重跑 `python environment/download_weights.py`，
或按 weights_download.md 手动下载放位。

## 4. DiffBIR 启动即报 pytorch_lightning 相关 ImportError

现象：`inference_bsr.py` 在 import 阶段崩溃（`pytorch_lightning.utilities` 等模块缺失/改名）。
原因：DiffBIR 的 LDM 代码写于 PL 1.x 时代。
处置（按顺序尝试）：
```bash
pip install "pytorch-lightning>=1.9,<2.0"   # 1.9.x 与 torch 2.x 兼容性最好
```
若仍失败，在报错 traceback 中定位 `third_party/DiffBIR/ldm/` 下引用的
PL 旧符号，做最小替换（如 `pytorch_lightning.utilities.distributed` ->
`pytorch_lightning.utilities.rank_zero`）。**所有补丁必须记录进 EXP_LOG.md**，
并在 `third_party_commits.txt` 旁追加 `third_party_patches.md` 说明改动点——
决赛复现时要用同一套补丁。

## 5. DiffBIR CLI 参数名对不上（`unrecognized arguments`）

原因：官方 CLI 参数可能随版本演进。
处置：
```bash
python third_party/DiffBIR/inference_bsr.py --help
```
把实际参数名映射进 `configs/defaults/enhance.yaml -> backends.diffbir`
（支持 `extra_args` 透传），必要时改 `src/enhance/backends/__init__.py` 的
`DiffBIRBackend.enhance_batch`，改动记 EXP_LOG。

## 6. DiffBIR 找不到权重文件

现象：CLI 报 ckpt 路径不存在。
原因：DiffBIR 约定权重在其仓库内 `weights/` 下。
处置：确认 setup 第 6 步的软链已建立：
```bash
ls -l third_party/DiffBIR/weights/
# 若为空：
ln -sfn ~/Deep-Generative-Prior/weights/diffbir/*.ckpt third_party/DiffBIR/weights/
```
仍失败则读其 `inference_bsr.py` 顶部的路径常量，用 `extra_args` 显式传参。

## 7. 显存 OOM

处置（按代价从小到大）：
1. `configs/defaults/enhance.yaml`：`tile_size: 512 -> 384`；
2. `steps: 50 -> 30`；
3. 确认没有其他进程占卡（`nvidia-smi`）；
4. DiffBIR 内部如支持 fp16/半精度开关，用 `extra_args` 打开。

## 8. pyiqa 首次运行卡在下载指标模型

处置：`export HF_ENDPOINT=https://hf-mirror.com`；或预先
`python -c "import pyiqa; [pyiqa.create_metric(m, device='cuda:0') for m in ['musiq','clipiqa','niqe','maniqa','lpips','dists']]"` 预热。

## 9. 磁盘不足（60GB 红线）

处置：
```bash
pip cache purge
rm -rf ~/.cache/huggingface/hub/models--*/locks   # 只清锁与临时，别删模型目录
df -h .
```
大件占用参考：torch ~3GB、DiffBIR ckpt ~4GB、pyiqa 指标模型 ~2GB、conda env ~8GB。

## 10. `Permission denied` / conda init 问题

处置：`source ~/.bashrc`；或直接用绝对路径 `~/miniconda3/envs/camera310/bin/python`。
