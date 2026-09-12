"""实验 07：从同一个 TinyGPT 比较 greedy、temperature 与 top-k。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch

from _common import PROJECT_ROOT, seed_everything
from learn_llm.run_management import read_manifest, checked_artifact, latest_base
from learn_llm.sft import format_instruction_prompt
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
            "outputs/pretrain/latest.json 对应的 base.pt"
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
    parser.add_argument("--run-dir", type=Path, help="运行目录；按 manifest 校验相对路径与 SHA-256")
    parser.add_argument("--branch", choices=("base", "random", "full", "lora"), default="base")
    parser.add_argument("--seed", type=int, help="run 中的训练 seed；默认使用首个 seed")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--instruction", help="使用 SFT 用户/助手模板的指令")
    parser.add_argument("--tokens", type=int, default=80)
    args = parser.parse_args(argv)
    if args.run_dir is not None and any(p is not None for p in (args.checkpoint, args.base_checkpoint, args.adapter)):
        parser.error("--run-dir cannot be combined with explicit artifact paths")
    if args.run_dir is None and (args.seed is not None or args.branch != "base"):
        parser.error("--seed/--branch require --run-dir")
    adapter_mode = args.base_checkpoint is not None or args.adapter is not None
    if args.checkpoint is not None and adapter_mode:
        parser.error("--checkpoint 不能与 --base-checkpoint/--adapter 同时使用")
    if adapter_mode and (args.base_checkpoint is None or args.adapter is None):
        parser.error("LoRA 推理必须同时提供 --base-checkpoint 和 --adapter")

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
        eos_token_id=tokenizer.eos_id,
    )
    print(f"\n--- {label} ---")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


def main() -> None:
    args = parse_args()
    seed_everything()
    try:
        if args.run_dir is not None:
            manifest = read_manifest(args.run_dir)
            if args.branch == "base":
                args.checkpoint = checked_artifact(args.run_dir, manifest["artifacts"]["base"])
            else:
                seed = args.seed if args.seed is not None else manifest["runs"][0]["seed"]
                key = f"{seed}.{args.branch}"
                if key not in manifest["artifacts"]:
                    raise ValueError(f"run has no saved artifact for {key}")
                path = checked_artifact(args.run_dir, manifest["artifacts"][key])
                if args.branch == "lora":
                    args.base_checkpoint = checked_artifact(args.run_dir, manifest["artifacts"]["base"])
                    args.adapter = path
                else:
                    args.checkpoint = path
        elif args.checkpoint is None and args.adapter is None:
            args.checkpoint = latest_base(PROJECT_ROOT / "outputs/pretrain")
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
    except (ArtifactValidationError, ValueError, KeyError, IndexError) as error:
        raise SystemExit(f"模型 artifact 校验失败：{error}") from error

    model = bundle.model
    tokenizer = bundle.tokenizer

    prompt = format_instruction_prompt(args.instruction) if args.instruction is not None else args.prompt or "语言模型"
    encoded = tokenizer.encode(prompt, return_tensor=True)
    assert isinstance(encoded, torch.Tensor)
    prompt_ids = encoded.unsqueeze(0)

    print("模型 artifact:", artifact_description)
    print("提示词:", prompt)
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
