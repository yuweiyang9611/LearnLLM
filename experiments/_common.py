from __future__ import annotations

import random
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Windows PowerShell 5 的活动代码页可能不是 UTF-8，而 Codex/现代终端按
# UTF-8 读取输出。显式设置后，中文实验说明在两边都不会变成乱码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def seed_everything(seed: int = 42) -> None:
    """让教学实验尽可能可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
