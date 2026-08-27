"""实验 03：不用神经网络，先理解“下一个 token 的条件概率”。"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from _common import DATA_DIR, seed_everything


def train_bigram(text: str) -> tuple[list[str], dict[str, Counter[str]]]:
    vocab = sorted(set(text))
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for current, following in zip(text, text[1:], strict=False):
        counts[current][following] += 1
    return vocab, counts


def probability(
    current: str,
    following: str,
    vocab: list[str],
    counts: dict[str, Counter[str]],
) -> float:
    # 加一平滑避免从未见过的二元组概率为 0。
    return (counts[current][following] + 1) / (sum(counts[current].values()) + len(vocab))


def perplexity(text: str, vocab: list[str], counts: dict[str, Counter[str]]) -> float:
    nll = 0.0
    pairs = list(zip(text, text[1:], strict=False))
    for current, following in pairs:
        nll -= math.log(probability(current, following, vocab, counts))
    return math.exp(nll / len(pairs))


def generate(
    start: str,
    length: int,
    vocab: list[str],
    counts: dict[str, Counter[str]],
) -> str:
    output = [start]
    for _ in range(length - 1):
        weights = [probability(output[-1], char, vocab, counts) for char in vocab]
        output.append(random.choices(vocab, weights=weights, k=1)[0])
    return "".join(output)


def main() -> None:
    seed_everything()
    text = (DATA_DIR / "tiny_corpus.txt").read_text(encoding="utf-8")
    vocab, counts = train_bigram(text)
    model_ppl = perplexity(text, vocab, counts)
    random_baseline = float(len(vocab))

    context = "语"
    likely = counts[context].most_common(5)
    print("词表大小:", len(vocab))
    print(f"在训练语料中，'{context}' 后最常见字符:", likely)
    print(f"Bigram 训练语料困惑度: {model_ppl:.2f}")
    print(f"均匀随机基线困惑度:   {random_baseline:.2f}")
    print("采样文本:", generate("语", 80, vocab, counts))

    assert model_ppl < random_baseline
    print("PASS: 利用一个 token 的上下文，模型优于均匀随机猜测。")
    print("局限：Bigram 只看前一个 token，无法表示长距离依赖。")


if __name__ == "__main__":
    main()

