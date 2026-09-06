#!/usr/bin/env bash
# RTX 5090 一键环境搭建（远程 Linux 执行，幂等：失败后可重跑续接）
# ============================================================
# 前置：驱动须支持 CUDA 12.8（Blackwell sm_120）——见 nvidia_smi_check.md
# 用法：bash environment/setup_5090.sh
# 可选环境变量：
#   ENV_NAME=camera310        conda 环境名
#   SKIP_RESSHIFT=1           跳过保底后端 ResShift 的克隆
#   USE_CN_MIRROR=1           使用国内镜像（pip 阿里源 + HF hf-mirror）
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
ENV_NAME=${ENV_NAME:-camera310}

echo "===== [0/7] 前置检查 ====="
nvidia-smi || { echo "[致命] nvidia-smi 不可用"; exit 1; }

if [[ "${USE_CN_MIRROR:-0}" == "1" ]]; then
  export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
  export HF_ENDPOINT="https://hf-mirror.com"
  echo "[镜像] pip=阿里云, HF=hf-mirror"
fi

echo "===== [1/7] Python 3.10 环境 ====="
# 新版 conda 创建环境前需接受频道 ToS（老版本无 tos 子命令，静默忽略）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
if conda env list 2>/dev/null | grep -q "^${ENV_NAME} "; then
  echo "[跳过] conda 环境已存在"
else
  if command -v conda >/dev/null 2>&1; then
    conda create -y -n "$ENV_NAME" python=3.10
  else
    echo "[信息] 无 conda，改用系统 python3.10 venv"
    python3.10 -m venv .venv
  fi
fi
if command -v conda >/dev/null 2>&1 && conda env list | grep -q "^${ENV_NAME} "; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV_NAME"
else
  source .venv/bin/activate
fi
python -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version'

echo "===== [2/7] PyTorch (cu128, Blackwell sm_120) ====="
if python -c "import torch" 2>/dev/null && python -c "
import torch
assert torch.cuda.is_available()
cap = torch.cuda.get_device_capability(0)
assert cap >= (12, 0), f'当前 torch 不支持 {cap}，需重装 cu128 版'
print('torch', torch.__version__, 'sm', cap)
"; then
  echo "[跳过] torch 已就绪"
else
  pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
fi
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA 不可用'
print('[确认] torch', torch.__version__, '| 设备:', torch.cuda.get_device_name(0),
      '| sm:', torch.cuda.get_device_capability(0))
"

echo "===== [3/7] 本项目依赖 ====="
pip install -r environment/requirements.txt

echo "===== [4/7] 克隆第三方仓库 ====="
mkdir -p third_party
if [[ ! -d third_party/DiffBIR ]]; then
  git clone https://github.com/XPixelGroup/DiffBIR.git third_party/DiffBIR
fi
if [[ "${SKIP_RESSHIFT:-0}" != "1" && ! -d third_party/ResShift ]]; then
  git clone https://github.com/zsyOAOA/ResShift.git third_party/ResShift || \
    echo "[警告] ResShift 克隆失败（保底后端，可稍后手动补）"
fi
: > third_party_commits.txt
for d in third_party/*/; do
  n=$(basename "$d")
  c=$(git -C "$d" rev-parse HEAD 2>/dev/null || true)
  [[ -n "$c" ]] && echo "$n $c" >> third_party_commits.txt
done
cat third_party_commits.txt
python environment/patch_diffbir.py || true   # 修上游 torch.Tuple 笔误（ADR-003 期间实测）

echo "===== [5/7] DiffBIR 依赖（剔除 torch/xformers/PL，防止拖垮 cu128 底座） ====="
# xformers==0.0.25+cu118 会把 torch 拽回 2.2.2+cu118（不支持 Blackwell），必须排除；
# pytorch-lightning 由下一步的显式版本约束决定，不吃第三方 pin。
# 注意 ^(torch...) 的前缀匹配曾误杀 torchsde（DiffBIR v2 采样器依赖），已收紧为精确包名。
grep -Eiv "^(torch|torchvision|torchaudio|xformers|pytorch-lightning)(==|>=|<=|~=|>|<|$)" \
  third_party/DiffBIR/requirements.txt > /tmp/diffbir_reqs.txt || true
pip install -r /tmp/diffbir_reqs.txt || {
  echo "[回退] requirements 安装失败，改用核心依赖清单"
  pip install einops omegaconf transformers open-clip-torch \
              kornia timm facexlib gfpgan scipy
}
pip install "pytorch-lightning>=1.9,<2.0" || echo "[警告] PL 版本待排错（见 TROUBLESHOOTING.md #4）"

echo "===== [5c/7] 评测链路兼容性 ====="
# DiffBIR 把 numpy 钉在 1.26.x；opencv-headless 5.x 需要 numpy>=2（ABI 不兼容），
# 统一降级到与 numpy 1.26 兼容的 4.9 版本（与 DiffBIR 的 opencv_python 4.9 同源）
pip install "opencv-python-headless==4.9.0.80" >/dev/null 2>&1 || true
python -c "import cv2; import pyiqa; print('[确认] cv2', cv2.__version__, '| pyiqa', pyiqa.__version__)" \
  || echo "[警告] cv2/pyiqa 导入失败，见 TROUBLESHOOTING.md #12"

echo "===== [5b/7] torch 底座完整性校验 ====="
if ! python -c "
import torch
assert torch.__version__.startswith('2.7'), f'torch 被第三方依赖改成了 {torch.__version__}'
assert torch.cuda.is_available(), 'CUDA 不可用'
cap = torch.cuda.get_device_capability(0)
assert cap >= (12, 0), f'sm{cap[0]}{cap[1]} 不受支持'
"; then
  echo "[修复] torch 底座被破坏，重装 2.7.1+cu128 ..."
  pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128
fi

echo "===== [6/7] basicsr 兼容补丁 + 权重下载 ====="
# basicsr 缺失只影响人脸分支（默认关闭），不阻断主流程
python environment/patch_basicsr.py || echo "[提示] basicsr 未安装，人脸分支（默认关闭）暂不可用，不影响主流程"
python environment/download_weights.py
# DiffBIR v2.x 权重不预下载：首次推理时由官方代码自动拉取（后端已注入 hf-mirror）

echo "===== [7/7] 收尾 ====="
pip cache purge 2>/dev/null || true
df -h . | tail -1
echo "============================================================"
echo "环境搭建完成。下一步："
echo "  1) bash scripts/prepare_data.sh"
echo "  2) pytest tests/ -v -m 'not remote'   # 本地逻辑自检（无需 GPU）"
echo "  3) pytest tests/test_smoke_diffbir.py -v -m remote   # DiffBIR 集成冒烟"
echo "============================================================"
