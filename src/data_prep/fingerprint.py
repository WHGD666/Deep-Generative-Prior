"""数据指纹：为每次正式实验提供数据可追溯性。

生成 data/manifest.json：每个文件的 MD5 + 大小 + 总量汇总。
实验 manifest 必须引用本文件的哈希，数据一旦变动即可察觉。

用法：
    python -m src.data_prep.fingerprint --root data --out data/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def fingerprint(root: Path) -> dict:
    files: dict[str, dict] = {}
    for p in sorted(root.rglob("*.jpg")):
        rel = p.relative_to(root).as_posix()
        files[rel] = {"md5": md5_of(p), "bytes": p.stat().st_size}
    return {
        "root": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_files": len(files),
        "total_bytes": sum(v["bytes"] for v in files.values()),
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成数据集指纹清单")
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="data/manifest.json")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {root}")
    manifest = fingerprint(root)
    out = Path(args.out)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    overall = hashlib.md5(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    print(f"[完成] {manifest['n_files']} 个文件, 共 {manifest['total_bytes']/1e6:.1f} MB")
    print(f"[指纹] 数据集整体指纹 = {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
