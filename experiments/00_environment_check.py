"""实验 00：确认 Python 环境隔离、依赖和随机种子。"""

from __future__ import annotations

import sys

import numpy as np
import torch

from _common import PROJECT_ROOT, seed_everything


def main() -> None:
    in_venv = sys.prefix != sys.base_prefix
    interpreter_inside_project = str(PROJECT_ROOT / ".venv") in sys.executable

    seed_everything(42)
    first = torch.rand(4)
    seed_everything(42)
    second = torch.rand(4)

    print("=== LearnLLM 环境检查 ===")
    print(f"项目目录: {PROJECT_ROOT}")
    print(f"解释器:   {sys.executable}")
    print(f"虚拟环境: {in_venv}")
    print(f"项目内解释器: {interpreter_inside_project}")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"NumPy:    {np.__version__}")
    print(f"计算设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"随机数可复现: {torch.equal(first, second)}")

    assert in_venv, "当前不是虚拟环境，请使用 .venv\\Scripts\\python.exe"
    assert interpreter_inside_project, "解释器不在项目 .venv 中"
    assert torch.equal(first, second), "固定种子后结果仍不一致"
    print("PASS: 环境隔离和基础依赖均正常。")


if __name__ == "__main__":
    main()

