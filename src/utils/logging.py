"""日志工具：run_id 贯穿的统一日志。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, run_id: str | None = None, log_dir: str | Path | None = None) -> logging.Logger:
    """获取 logger；提供 run_id 时同时写入文件。

    Args:
        name: logger 名（一般传模块名）。
        run_id: 正式 run 的唯一标识；为 None 时不写文件。
        log_dir: 日志文件目录（log_dir/<run_id>.log）。
    """
    logger = logging.getLogger(name if run_id is None else f"{name}.{run_id}")
    if logger.handlers:  # 幂等：重复调用不叠加 handler
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(_FORMAT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if run_id is not None and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{run_id}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger
