"""远程冒烟测试：DiffBIR 集成正确性（仅远程 5090 执行）。

运行：pytest tests/test_smoke_diffbir.py -v -m remote
跳过条件：第三方仓库缺失（本地开发机自动 skip）。
注意：权重需先经 environment/download_diffbir_weights.sh 预下载（约 6.3GB），
未就位时会在 cleaner 加载处报连接错误（排障 #14）。

验证点：
1. DiffBIR 新版 CLI（inference.py）可启动并完成一张探测块的推理；
2. 输出尺寸与输入成整数倍（infer_upscale 通过）；
3. 放大倍率在预期集合 {1, 2, 4} 内。
"""

import pytest

from src.enhance.backends import build_backend
from src.enhance.tiled_inference import infer_upscale
from tests.helpers import make_img

pytestmark = pytest.mark.remote


def test_diffbir_smoke():
    import yaml
    from pathlib import Path

    if not Path("third_party/DiffBIR/inference.py").is_file():
        pytest.skip("third_party/DiffBIR 缺失（本地开发机正常现象）")

    cfg = yaml.safe_load(open("configs/defaults/enhance.yaml", encoding="utf-8"))
    backend = build_backend(cfg["backend"])
    sample = make_img(64, 64, seed=42)
    k = infer_upscale(backend.enhance_batch, sample)
    assert k in (1, 2, 4), f"探测到异常放大倍率 {k}"
    print(f"[冒烟] DiffBIR v2.x CLI 正常, 放大倍率 k={k}")
