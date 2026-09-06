"""官方赛题数据 zip 解压。

处理两个已知坑：
1. 官方 zip 在 macOS 打包，文件名为 UTF-8 但未设 UTF-8 标志位，
   Python zipfile 会按 cp437 解码 -> 必须先 encode('cp437') 再 decode('utf-8') 修复；
2. zip 内含 __MACOSX/ 元数据目录与 .DS_Store，全部跳过。

输出目录结构固定（项目约定）：
    <out>/val/caseN_lq.jpg, caseN_gt.jpg   (N=1..5)
    <out>/test/caseN.jpg                   (N=1..100)

用法：
    python -m src.data_prep.extract_dataset --zip 赛题一.zip --out data
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# zip 内中文目录名 -> 项目目录名
DIR_MAP = {"验证集": "val", "测试集": "test"}

EXPECTED_COUNTS = {"val": 10, "test": 100}


def fix_name(name: str) -> str:
    """修复 cp437 误解码的 UTF-8 文件名；已是合法 UTF-8 则原样返回。"""
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def extract(zip_path: Path, out_root: Path) -> dict[str, int]:
    """解压并返回 {目录: 图片数}。原始数据只读，解压是唯一入口。"""
    if not zip_path.is_file():
        raise FileNotFoundError(f"找不到官方 zip: {zip_path}")
    counts: dict[str, int] = {"val": 0, "test": 0}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = fix_name(info.filename)
            if name.startswith("__MACOSX") or "/__MACOSX" in name:
                continue
            if "/." in name or Path(name).name.startswith("._"):  # .DS_Store / ._*
                continue
            if name.endswith("/"):
                continue
            if not name.lower().endswith(".jpg"):
                continue
            parts = Path(name).parts  # 形如 (赛题一, 验证集, case1_lq.jpg)
            if len(parts) != 3:
                raise RuntimeError(f"zip 内出现意外的目录层级: {name}")
            sub = DIR_MAP.get(parts[1])
            if sub is None:
                raise RuntimeError(f"zip 内出现未知的子目录: {parts[1]} (来自 {name})")
            target = out_root / sub / parts[2]
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            counts[sub] += 1
    return counts


def verify(counts: dict[str, int]) -> None:
    for sub, expected in EXPECTED_COUNTS.items():
        if counts.get(sub) != expected:
            raise RuntimeError(
                f"{sub} 解压数量异常: 期望 {expected}, 实际 {counts.get(sub)}"
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="官方数据 zip 解压（修复文件名编码）")
    ap.add_argument("--zip", required=True, help="官方压缩包路径")
    ap.add_argument("--out", default="data", help="输出根目录（默认 data/）")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    if any((out_root / s).exists() for s in EXPECTED_COUNTS):
        # 幂等：重复运行 prepare_data.sh 时不报错，后续指纹/统计步骤照常执行
        print(f"[跳过] {out_root}/ 下已存在 val/test，跳过解压")
        return 0
    counts = extract(Path(args.zip), out_root)
    verify(counts)
    print(f"[完成] {counts} -> {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
