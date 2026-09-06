"""DiffBIR v2.x 上游代码补丁（幂等）。

已知上游 bug：diffbir/sampler/edm_sampler.py 等文件把类型注解误写为
`torch.Tuple[int]` / `torch.List[...]` 等——torch 模块没有这些属性，
Python 在函数定义时求值注解会直接 AttributeError（实测于 2026-09，
DiffBIR main 5c2d6c1）。修复：替换为 PEP 585 内置泛型
（tuple/list/dict/set），Python 3.10 运行时求值合法。

另做一处兼容：torch >= 2.4 起 `torch.load` 默认 weights_only=True，
DiffBIR 加载含 omegaconf 对象的 ckpt 时会失败，这里不全局改动，
仅记录；若冒烟在权重加载处报 UnpicklingError，见 TROUBLESHOOTING #13。

用法：python environment/patch_diffbir.py
退出码：0=无待修内容或已修补；1=第三方仓库缺失。
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO = PROJECT_ROOT / "third_party" / "DiffBIR"

# 仅替换已知误用的泛型注解，不做其他"顺手修复"
PATTERNS = {
    "torch.Tuple": "tuple",
    "torch.List": "list",
    "torch.Dict": "dict",
    "torch.Set": "set",
}


def main() -> int:
    if not REPO.is_dir():
        print("[跳过] third_party/DiffBIR 未克隆")
        return 1
    rx = re.compile(r"\b(" + "|".join(PATTERNS) + r")\b")
    total = 0
    for py in sorted((REPO / "diffbir").rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        new, n = rx.subn(lambda m: PATTERNS[m.group(0)], text)
        if n:
            py.write_text(new, encoding="utf-8")
            total += n
            print(f"[修补] {py.relative_to(REPO)}: {n} 处")
    if total == 0:
        print("[跳过] 未发现 torch.Tuple/List/Dict/Set 误用（已修补或上游已修复）")
    else:
        print(f"[完成] 共修补 {total} 处；此补丁随 environment/patch_diffbir.py 固化，重装环境自动生效")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
