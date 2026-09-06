"""pytest 全局配置：项目根加入 sys.path，注册 remote 标记。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "remote: 需要 GPU + 第三方权重，仅在远程 5090 执行"
    )
