"""浏览器版 experiments/_common.py：同样的名字，只有 random，没有 torch。

章节里的「运行」把实验脚本放进浏览器里的 Pyodide 运行时。仓库里的 _common.py 要导入 torch，浏览器里没有，
所以换成这一份：PROJECT_ROOT、SRC_DIR、DATA_DIR、CHECKPOINT_DIR 和 seed_everything 的名字和用法都和仓库一致，
seed_everything 只固定 random 的种子。其余文件逐字来自仓库。
"""

from __future__ import annotations

import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def seed_everything(seed: int = 42) -> None:
    """让教学实验尽可能可复现：浏览器里只有 random。"""

    random.seed(seed)
