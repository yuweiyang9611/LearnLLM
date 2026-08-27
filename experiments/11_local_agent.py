"""实验 11：完全本地的工具、状态、控制循环与停止条件。"""

from __future__ import annotations

import json

from _common import DATA_DIR
from learn_llm.agent import (
    AgentAction,
    AgentResult,
    LocalAgent,
    Tool,
    build_local_agent,
    safe_calculate,
)
from learn_llm.rag import Document, ExtractiveRag, TfidfRetriever


def load_rag() -> ExtractiveRag:
    records = json.loads(
        (DATA_DIR / "rag_documents.json").read_text(encoding="utf-8")
    )
    documents = [
        Document(item["text"], doc_id=item["id"], metadata={"title": item["title"]})
        for item in records
    ]
    return ExtractiveRag(TfidfRetriever(documents), min_score=0.18, top_k=2)


def print_trace(label: str, result: AgentResult) -> None:
    print(f"\n=== {label} ===")
    print(f"状态: {result.status}; 停止原因: {result.stop_reason}")
    for step in result.steps:
        observation = step.observation
        observed = "FINISH" if observation is None else (
            observation.output if observation.success else observation.error
        )
        print(
            f"step {step.number}: {step.action.tool_name}"
            f"({step.action.tool_input!r}) -> {observed}"
        )
    print(f"最终回答: {result.answer}")


def main() -> None:
    agent = build_local_agent(load_rag())
    calculation = agent.run("计算：(17 + 5) * 3 - 8 / 2")
    search = agent.run("检索：RAG 会修改模型参数吗？")
    refusal = agent.run("检索：今天东京天气怎么样？")

    print_trace("安全算术工具", calculation)
    print_trace("带引用的知识检索工具", search)
    print_trace("知识库外问题拒答", refusal)

    unknown_agent = LocalAgent(
        [Tool("calculator", "安全算术", safe_calculate)],
        planner=lambda _task, _steps: AgentAction("delete_file", "important.txt"),
    )
    unknown = unknown_agent.run("尝试调用未注册工具")
    print_trace("未知工具被控制器拦截", unknown)

    tool_error = agent.run("计算：1 / 0")
    print_trace("工具异常变成可观察状态", tool_error)

    endless_agent = LocalAgent(
        [Tool("calculator", "安全算术", safe_calculate)],
        planner=lambda _task, _steps: AgentAction("calculator", "1 + 1"),
    )
    limited = endless_agent.run("重复调用", max_steps=2)
    print_trace("达到最大步骤数后停止", limited)

    assert calculation.succeeded and calculation.answer == "62"
    assert search.succeeded and "[rag]" in search.answer
    assert refusal.succeeded and "资料不足" in refusal.answer
    assert unknown.status == "error" and unknown.stop_reason == "unknown_tool"
    assert tool_error.status == "error" and tool_error.stop_reason == "tool_error"
    assert limited.status == "step_limit" and len(limited.steps) == 2
    print("\nPASS: Agent 的工具、状态、错误和停止条件均由本地程序显式控制。")
    print("说明：这里使用确定性规则规划器，不是 LLM Agent，也不访问网络 API。")


if __name__ == "__main__":
    main()
