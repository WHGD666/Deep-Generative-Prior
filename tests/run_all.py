"""无 pytest 环境下的本地自检运行器（发现并执行所有 CPU 测试函数）。

用法（项目根目录）：python tests/run_all.py
远程/正式环境仍推荐 pytest：pytest tests/ -v -m "not remote"
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent


class _Raises:
    """pytest.raises 的最小替代（支持 with 语法）。"""

    def __init__(self, exc_type):
        self.exc_type = exc_type
        self.exc = None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"期望抛出 {self.exc_type.__name__} 但未抛出")
        if not issubclass(et, self.exc_type):
            return False  # 让异常继续抛
        self.exc = ev
        return True


def main() -> int:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(TESTS_DIR))
    # 让测试里的 `import pytest` 与 `pytest.mark.*`/raises 可用
    try:
        import pytest  # noqa: F401
    except ImportError:
        import types

        fake = types.ModuleType("pytest")

        def raises(exc_type):
            return _Raises(exc_type)

        def skip(*a, **k):
            raise RuntimeError("skip")

        class _Mark:
            def __getattr__(self, name):
                return lambda f=None, *a, **k: f if f is not None else (lambda f: f)

        fake.raises, fake.skip, fake.mark = raises, skip, _Mark()
        fake.approx = lambda v, **k: v
        sys.modules["pytest"] = fake

    passed, failed, skipped = 0, 0, 0
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            needs_tmp = "tmp_path" in inspect.signature(fn).parameters
            try:
                if needs_tmp:
                    with tempfile.TemporaryDirectory() as tmp:
                        fn(Path(tmp))
                else:
                    fn()
                print(f"[通过] {path.stem}::{name}")
                passed += 1
            except RuntimeError as e:
                if "skip" in str(e):
                    print(f"[跳过] {path.stem}::{name}")
                    skipped += 1
                else:
                    print(f"[失败] {path.stem}::{name}\n{traceback.format_exc()}")
                    failed += 1
            except Exception:  # noqa: BLE001
                print(f"[失败] {path.stem}::{name}\n{traceback.format_exc()}")
                failed += 1
    print(f"\n合计: 通过 {passed}, 失败 {failed}, 跳过 {skipped}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
