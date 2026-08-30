"""实验 07：从同一个 TinyGPT 比较 greedy、temperature 与 top-k。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from _common import CHECKPOINT_DIR, seed_everything
from learn_llm.artifacts import (
    ArtifactValidationError,
    load_lora_model,
    load_tiny_gpt_checkpoint,
)
from learn_llm.model import TinyGPT
from learn_llm.tokenizer import CharTokenizer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "完整 TinyGPT checkpoint；不传任何 artifact 参数时默认使用 "
            "checkpoints/tiny_gpt.pt"
        ),
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        help="LoRA 推理使用的基础 TinyGPT checkpoint（必须与 --adapter 同时传入）",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        help="实验 10 生成的 LoRA adapter-only artifact",
    )
    parser.add_argument("--prompt", default="语言模型")
    parser.add_argument("--tokens", type=int, default=80)
    args = parser.parse_args(argv)
    adapter_mode = args.base_checkpoint is not None or args.adapter is not None
    if args.checkpoint is not None and adapter_mode:
        parser.error("--checkpoint 不能与 --base-checkpoint/--adapter 同时使用")
    if adapter_mode and (args.base_checkpoint is None or args.adapter is None):
        parser.error("LoRA 推理必须同时提供 --base-checkpoint 和 --adapter")
    if not adapter_mode and args.checkpoint is None:
        args.checkpoint = CHECKPOINT_DIR / "tiny_gpt.pt"
    if args.tokens < 0:
        parser.error("--tokens 不能为负数")
    return args


def decode_strategy(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt_ids: torch.Tensor,
    *,
    label: str,
    temperature: float,
    top_k: int | None,
    do_sample: bool,
    seed: int,
    tokens: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    output = model.generate(
        prompt_ids,
        tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=do_sample,
        generator=generator,
    )
    print(f"\n--- {label} ---")
    print(tokenizer.decode(output[0]))


def main() -> None:
    args = parse_args()
    seed_everything()
    try:
        if args.adapter is not None:
            bundle = load_lora_model(
                args.base_checkpoint,
                args.adapter,
                map_location="cpu",
            )
            artifact_description = (
                f"base={bundle.checkpoint_path}, adapter={bundle.adapter_path}"
            )
        else:
            bundle = load_tiny_gpt_checkpoint(args.checkpoint, map_location="cpu")
            artifact_description = f"checkpoint={bundle.checkpoint_path}"
    except FileNotFoundError as error:
        raise SystemExit(
            f"找不到模型 artifact：{error}\n"
            "若尚未训练，请先运行: python experiments/06_train_tiny_gpt.py --quick"
        ) from error
    except ArtifactValidationError as error:
        raise SystemExit(f"模型 artifact 校验失败：{error}") from error

    model = bundle.model
    tokenizer = bundle.tokenizer

    encoded = tokenizer.encode(args.prompt, return_tensor=True)
    assert isinstance(encoded, torch.Tensor)
    prompt_ids = encoded.unsqueeze(0)

    print("模型 artifact:", artifact_description)
    print("提示词:", args.prompt)
    print("注意：采样策略只改变如何从概率分布选 token，不会增加模型知识。")
    decode_strategy(
        model,
        tokenizer,
        prompt_ids,
        label="Greedy",
        temperature=0.0,
        top_k=None,
        do_sample=False,
        seed=1,
        tokens=args.tokens,
    )
    decode_strategy(
        model,
        tokenizer,
        prompt_ids,
        label="Temperature=0.7, top-k=10",
        temperature=0.7,
        top_k=10,
        do_sample=True,
        seed=7,
        tokens=args.tokens,
    )
    decode_strategy(
        model,
        tokenizer,
        prompt_ids,
        label="Temperature=1.3, top-k=30",
        temperature=1.3,
        top_k=30,
        do_sample=True,
        seed=7,
        tokens=args.tokens,
    )
    print("\nPASS: 三种解码策略均完成。请比较重复度、连贯性和多样性。")


if __name__ == "__main__":
    main()
