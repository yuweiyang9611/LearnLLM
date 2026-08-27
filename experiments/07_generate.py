"""实验 07：从同一个 TinyGPT 比较 greedy、temperature 与 top-k。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import CHECKPOINT_DIR, seed_everything
from learn_llm.model import TinyGPT, TinyGPTConfig
from learn_llm.tokenizer import CharTokenizer
from learn_llm.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "tiny_gpt.pt")
    parser.add_argument("--prompt", default="语言模型")
    parser.add_argument("--tokens", type=int, default=80)
    return parser.parse_args()


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
    if not args.checkpoint.exists():
        raise SystemExit(
            "找不到 checkpoint。请先运行：\n"
            ".\\.venv\\Scripts\\python.exe .\\experiments\\06_train_tiny_gpt.py --quick"
        )

    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = raw["metadata"]
    config = TinyGPTConfig(**metadata["config"])
    tokenizer = CharTokenizer.from_state_dict(metadata["tokenizer"])
    model = TinyGPT(config)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    encoded = tokenizer.encode(args.prompt, return_tensor=True)
    assert isinstance(encoded, torch.Tensor)
    prompt_ids = encoded.unsqueeze(0)

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

