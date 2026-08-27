"""实验 09：从检索扩展到可拒答、可引用、可评测的 Tiny RAG。"""

from __future__ import annotations

import json

from _common import DATA_DIR
from learn_llm.rag import (
    Document,
    ExtractiveRag,
    RagEvaluationCase,
    TfidfRetriever,
    evaluate_grounded_rag,
)


def load_documents() -> list[Document]:
    raw_documents = json.loads(
        (DATA_DIR / "rag_documents.json").read_text(encoding="utf-8")
    )
    return [
        Document(item["text"], doc_id=item["id"], metadata={"title": item["title"]})
        for item in raw_documents
    ]


def main() -> None:
    rag = ExtractiveRag(TfidfRetriever(load_documents()), min_score=0.18, top_k=2)
    cases = [
        RagEvaluationCase(
            "Tokenizer 产生的 token 一定是完整单词吗？",
            expected_source_id="tokenizer",
            expected_text="不是固定意义上的一个词",
        ),
        RagEvaluationCase(
            "Attention 为什么除以根号 d_k？",
            expected_source_id="attention",
            expected_text="缩放用于缓解高维点积过大",
        ),
        RagEvaluationCase(
            "Decoder causal mask 如何防止看到未来 token？",
            expected_source_id="causal-mask",
            expected_text="防止训练时看到未来答案",
        ),
        RagEvaluationCase(
            "LoRA 低秩矩阵 A 和 B 有什么作用？",
            expected_source_id="lora",
            expected_text="权重增量近似为 BA",
        ),
        RagEvaluationCase(
            "RAG 检索会修改模型参数吗？",
            expected_source_id="rag",
            expected_text="不修改模型参数",
        ),
        RagEvaluationCase(
            "Agent 为什么需要工具和控制循环？",
            expected_source_id="agent",
            expected_text="工具、状态和控制循环",
        ),
        RagEvaluationCase("量子纠缠能否实现超光速通信？", expect_refusal=True),
        RagEvaluationCase("今天东京天气怎么样？", expect_refusal=True),
    ]

    print("=== 每道题的回答、来源和支持性 ===")
    for case in cases:
        result = rag.answer(case.question)
        scores = ", ".join(
            f"{item.doc_id}:{item.score:.3f}" for item in result.evidence
        )
        print(f"\n问题：{case.question}")
        print(f"检索：{scores}")
        print(f"回答：{result.answer}")
        print(
            "判定："
            f"refused={result.refused}, sources={result.source_ids}, "
            f"support={result.support.ratio:.0%}"
        )

    evaluation = evaluate_grounded_rag(rag, cases)
    print("\n=== 端到端评测 ===")
    print(f"Retrieval Recall: {evaluation.retrieval_recall:.1%}")
    print(f"拒答准确率:       {evaluation.refusal_accuracy:.1%}")
    print(f"回答支持率:       {evaluation.support_rate:.1%}")
    print(f"端到端成功率:     {evaluation.end_to_end_accuracy:.1%}")

    assert evaluation.retrieval_recall == 1.0
    assert evaluation.refusal_accuracy == 1.0
    assert evaluation.support_rate == 1.0
    assert evaluation.end_to_end_accuracy == 1.0
    print("\nPASS: Tiny RAG 能引用来源、拒绝资料外问题，并用确定性规则检查忠实度。")
    print("说明：回答层是抽取式教学基线，不是 LLM，也没有生成或改写能力。")


if __name__ == "__main__":
    main()
