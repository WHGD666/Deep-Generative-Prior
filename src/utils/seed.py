"""随机种子管理：一切随机性必须可由配置复现。"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """固定 python / numpy / torch(若可用) / 环境变量的随机种子。

    对子进程后端（DiffBIR CLI）无效，后端侧的随机性由其自身参数控制，
    并通过配置快照记录。
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
