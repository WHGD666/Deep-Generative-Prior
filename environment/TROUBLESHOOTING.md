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

原因：我们集成的是 v2.x 重写版（入口 `inference.py`）；官方参数可能继续演进。
处置：
```bash
python third_party/DiffBIR/inference.py --help
```
把实际参数名映射进 `configs/defaults/enhance.yaml -> backends.diffbir`
（支持 `extra_args` 透传），必要时改 `src/enhance/backends/__init__.py` 的
`DiffBIRBackend.enhance_batch`，改动记 EXP_LOG。
典型差异：`--steps` 若不被识别，把配置里 `steps: 50` 改为 `steps: 0` 即可去掉该参数。

## 6. DiffBIR 权重自动下载失败

现象：首次推理时报 HF 下载错误（404/连接超时）。
处置：
```bash
export HF_ENDPOINT=https://hf-mirror.com     # 我们的子进程默认已注入，可显式覆盖
df -h /data                                   # 确认磁盘够（v2.1.pt 约 1~2GB）
```
仍失败则按 weights_download.md 手动下载后，用 `extra_args` 把权重路径传给
官方 CLI（具体参数名以 `inference.py --help` 输出为准）。

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

## 11. 第三方 requirements 把 torch 拽回旧版（已根治）

现象：安装 DiffBIR 依赖时出现 `Collecting torch==2.2.2+cu118 (from xformers...)`。
原因：其 requirements pin 了 `xformers==0.0.25+cu118`，间接 pin torch 2.2.2+cu118。
处置：setup_5090.sh 已排除 torch/torchvision/xformers/pytorch-lightning 行，
并在安装后强制校验 torch 底座（版本 2.7 + sm_120），破坏即自动重装。
若你在旧版脚本里中断过：`pip cache purge` 清掉已下载的 cu118 轮子（约 2.4GB）再重跑。

## 12. `import cv2` 报 numpy ABI 错误 / pyiqa 导入失败

现象：`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`（或反之）。
原因：DiffBIR 依赖链把 numpy 钉在 1.26，而新版 opencv-python-headless 5.x 是按 numpy 2 编译的。
处置：`pip install "opencv-python-headless==4.9.0.80"`（setup 已自动处理）。
另：transformers 保持 DiffBIR 的 4.37.2（pyiqa 想要 5.0 只是版本声明冲突，
我们用的 musiq/clipiqa/niqe/maniqa/lpips/dists 指标均不 import transformers）。

## 13. DiffBIR 上游代码补丁台账

| 补丁 | 现象 | 固化位置 |
|---|---|---|
| `torch.Tuple/List/Dict/Set` → 内置泛型 | 冒烟时 `AttributeError: module 'torch' has no attribute 'Tuple'`（edm_sampler.py:145 等） | `environment/patch_diffbir.py`（setup 第 4 步自动执行） |
| torchsde 缺失 | `ModuleNotFoundError: No module named 'torchsde'`（setup grep 误过滤所致，已收紧正则 + 补入 requirements） | `environment/requirements.txt` |

若冒烟在权重加载处报 `UnpicklingError / weights_only`：这是 torch>=2.4
默认 `weights_only=True` 所致，在 DiffBIR 的 ckpt 加载点（`torch.load`）
加 `weights_only=False` 后同样登记到本表并固化进 patch 脚本。

## 14. DiffBIR 权重下载 Connection reset（已根治）

现象：冒烟时 stage1 cleaner（或 SD 底模）下载报
`urllib.error.URLError: <urlopen error [Errno 104] Connection reset by peer>`。
原因：DiffBIR 用 `torch.hub.download_url_to_file` 直下 `huggingface.co` 裸 URL，
**`HF_ENDPOINT` 环境变量对它无效**（只影响 huggingface_hub 库），国内直连被墙。
处置：`bash environment/download_diffbir_weights.sh`
（wget -c 走 hf-mirror 预下载 3 个文件到 `third_party/DiffBIR/weights/`，
DiffBIR 检测到本地文件即跳过网络下载）。清单与体积：
- `realesrgan_s4_swinir_100k.pth`（stage1 cleaner，~70MB）
- `sd2.1-base-zsnr-laionaes5.ckpt`（SD2.1 底模，**~5.3GB**）
- `DiffBIR_v2.1.pt`（ControlNet，~2.5GB）

## 15. 512 瓦片 VAE 编码 CUDA OOM（16GiB 分配失败，已根治）

现象：管线处理真实瓦片时 `vae.py mid.attn_1 → F.scaled_dot_product_attention`
尝试分配 16GiB 显存失败（31.4GiB 卡已用 27.6GiB）。冒烟小图不触发，真图必炸。
原因：v2.1 的 `--upscale 4` 语义是**先把输入双三次放大 4 倍再修复**——
512px 瓦片被内部撑到 2048×2048，VAE 中间层注意力矩阵 256² tokens 平方级爆炸。
处置：`enhance.yaml` 后端 `upscale: 4 → 1`（同分辨率修复本就不需要放大），
4K 原图直接按 512+64 重叠分块喂入；`prescale: auto` 探测 k=1 自然不缩放。
附带收益：每图瓦片数 ~190 → ~70，速度约快 3 倍且不丢原始细节。
另在 `_run_cli` 注入 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 防碎片。
