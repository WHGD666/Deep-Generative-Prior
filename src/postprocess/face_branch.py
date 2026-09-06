"""人脸分支：GFPGAN 全图增强（检测+裁剪增强+回贴一体，默认关闭）。

背景：test 集含小人脸场景而 val 无样例。扩散通用管线对小脸易失真；
本分支用 GFPGAN 对人脸区域做针对性增强后回贴全图。

设计取舍（审计记录）：只依赖 GFPGANer 一个入口（其内部完成人脸检测、
裁剪增强、paste_back），不再二次调用 facexlib 检测 API——把第三方接口
面压到最小，正确性由远程冒烟验证。若未检出人脸，GFPGAN 返回背景图，
效果等价于无操作，不阻断管线。

依赖（远程环境随 DiffBIR 附带）：gfpgan（含 facexlib、basicsr 链）。
缺失时抛出明确错误——开关开着却没生效属于危险状态，禁止静默跳过。

配置（configs/defaults/postprocess.yaml -> face_branch）：
    enabled: false
    model_path: weights/gfpgan/GFPGANv1.4.pth
    upscale: 1        # 保持全图尺寸不变（红线 R4）
"""

from __future__ import annotations

from src.utils import io as uio


def apply(img: "np.ndarray", cfg: dict) -> "np.ndarray":  # noqa: F821 - numpy 后置导入
    try:
        from gfpgan import GFPGANer
    except ImportError as e:
        raise RuntimeError(
            "face_branch 已启用但缺少 gfpgan（远程环境应已随 DiffBIR 安装；"
            "若确实不使用请把 face_branch.enabled 设为 false）"
        ) from e

    import cv2  # gfpgan 依赖链自带，仅此处使用 BGR 语义

    enhancer = GFPGANer(
        model_path=str(cfg["model_path"]),
        upscale=int(cfg.get("upscale", 1)),
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,  # 背景不放大不增强，只动人脸区域
    )
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, _, restored = enhancer.enhance(
        bgr, has_aligned=False, only_center_face=False, paste_back=True
    )
    if restored is None:
        raise RuntimeError("GFPGAN 返回空结果（全图无人脸且无 bg_upsampler）")
    out = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
    uio.assert_same_size(out, img, names=("face_branch_out", "input"))
    return out
