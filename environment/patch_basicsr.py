"""basicsr 兼容性补丁（幂等）。

背景：facexlib/gfpgan 依赖 basicsr；basicsr 1.x 的 degradations.py 使用了
torchvision 旧路径 `torchvision.transforms.functional_tensor`，在新
torchvision（>=0.17，含 cu128 配套的 0.22）下已移除，import 即报错。
官方修复方式就是把导入路径改为 `torchvision.transforms.functional`。

本脚本在 site-packages 中定位 basicsr 并原地修补，重复执行安全。
退出码：0=已修补或本就正常；1=未找到 basicsr（若不使用人脸分支可忽略）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

OLD = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
NEW = "from torchvision.transforms.functional import rgb_to_grayscale"


def locate_basicsr() -> Path | None:
    spec = importlib.util.find_spec("basicsr")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def main() -> int:
    root = locate_basicsr()
    if root is None:
        print("[跳过] 未安装 basicsr（仅当不用 facexlib/gfpgan 人脸链路时才正常）")
        return 1
    target = root / "degradations.py"
    if not target.is_file():
        print(f"[警告] 未找到 {target}，basicsr 版本可能已变，请人工检查")
        return 1
    text = target.read_text(encoding="utf-8")
    if NEW in text:
        print("[跳过] basicsr 已修补过")
        return 0
    if OLD not in text:
        print("[信息] 未发现已知旧导入，可能已由新版修复，无需补丁")
        return 0
    target.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print(f"[完成] 已修补 {target}")
    print("[验证] 重新 import 校验…")
    try:
        for m in list(sys.modules):
            if m.startswith("basicsr"):
                del sys.modules[m]
        import basicsr.degradations  # noqa: F401

        print("[验证] basicsr.degradations 导入成功")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"[失败] 修补后仍导入失败: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
