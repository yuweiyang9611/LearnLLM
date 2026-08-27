"""实验 02：字符 Tokenizer 与一个可观察的迷你 BPE 合并过程。"""

from __future__ import annotations

from collections import Counter

from _common import DATA_DIR
from learn_llm.tokenizer import CharTokenizer


def learn_bpe_merges(text: str, merge_count: int = 8) -> tuple[list[str], list[tuple[str, str]]]:
    """在一条符号序列上演示 BPE；它是教学版，不是生产 Tokenizer。"""

    symbols = list(text.replace(" ", "▁"))
    merges: list[tuple[str, str]] = []
    for _ in range(merge_count):
        pairs = Counter(zip(symbols, symbols[1:], strict=False))
        if not pairs:
            break
        best_pair, frequency = pairs.most_common(1)[0]
        if frequency < 2:
            break
        merged_symbol = "".join(best_pair)
        new_symbols: list[str] = []
        index = 0
        while index < len(symbols):
            if index + 1 < len(symbols) and tuple(symbols[index : index + 2]) == best_pair:
                new_symbols.append(merged_symbol)
                index += 2
            else:
                new_symbols.append(symbols[index])
                index += 1
        symbols = new_symbols
        merges.append(best_pair)
        print(f"merge {len(merges):02d}: {best_pair} (出现 {frequency} 次)")
    return symbols, merges


def main() -> None:
    corpus = (DATA_DIR / "tiny_corpus.txt").read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(corpus)
    sample = "语言模型预测下一个 token。"
    token_ids = tokenizer.encode(sample)
    restored = tokenizer.decode(token_ids)

    print("=== 字符 Tokenizer ===")
    print("原文:", sample)
    print("token ids:", token_ids)
    print("还原:", restored)
    print("词表大小:", tokenizer.vocab_size)
    assert restored == sample

    print("\n=== 迷你 BPE：先预测哪一对会最先合并 ===")
    bpe_text = "low lower lowest low lower"
    symbols, merges = learn_bpe_merges(bpe_text)
    print("最终符号:", symbols)
    print("合并规则:", merges)
    assert merges and len(symbols) < len(bpe_text)

    print("\n结论：token 是 Tokenizer 规则产生的单位，并不天然等于字或词。")
    print("PASS: 字符级往返无损，BPE 重复合并高频相邻符号。")


if __name__ == "__main__":
    main()

