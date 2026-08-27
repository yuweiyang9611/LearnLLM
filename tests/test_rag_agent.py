from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learn_llm.agent import (  # noqa: E402
    AgentAction,
    LocalAgent,
    Tool,
    build_local_agent,
    safe_calculate,
)
from learn_llm.rag import (  # noqa: E402
    Document,
    ExtractiveRag,
    RagEvaluationCase,
    TfidfRetriever,
    check_answer_support,
    evaluate_grounded_rag,
)


class GroundedRagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(
                "RAG 在回答前检索资料。它通常不会修改模型参数。",
                "rag",
                {"title": "RAG"},
            ),
            Document(
                "注意力根据相关性聚合上下文。缩放可缓解高维点积过大。",
                "attention",
                {"title": "注意力"},
            ),
        ]
        self.rag = ExtractiveRag(
            TfidfRetriever(self.documents), min_score=0.18, top_k=2
        )

    def test_answer_is_extractive_cited_and_supported(self) -> None:
        result = self.rag.answer("RAG 会修改模型参数吗？")

        self.assertFalse(result.refused)
        self.assertEqual(result.source_ids, ("rag",))
        self.assertIn("不会修改模型参数", result.answer)
        self.assertIn("[rag]", result.answer)
        self.assertTrue(result.is_supported)
        self.assertEqual(result.support.ratio, 1.0)

        fabricated = check_answer_support("RAG 会自动保证事实正确。[rag]", result.sources)
        self.assertFalse(fabricated.is_fully_supported)
        self.assertEqual(fabricated.ratio, 0.0)

    def test_out_of_scope_question_is_refused_without_fake_citation(self) -> None:
        result = self.rag.answer("What will the weather be in Tokyo tomorrow?")

        self.assertTrue(result.refused)
        self.assertIn("资料不足", result.answer)
        self.assertEqual(result.source_ids, ())
        self.assertNotIn("[rag]", result.answer)

    def test_end_to_end_evaluation_reports_retrieval_refusal_and_support(self) -> None:
        evaluation = evaluate_grounded_rag(
            self.rag,
            [
                RagEvaluationCase(
                    "为什么缩放注意力点积？",
                    expected_source_id="attention",
                    expected_text="高维点积过大",
                ),
                RagEvaluationCase(
                    "What will the weather be in Tokyo tomorrow?",
                    expect_refusal=True,
                ),
            ],
        )

        self.assertEqual(evaluation.retrieval_recall, 1.0)
        self.assertEqual(evaluation.refusal_accuracy, 1.0)
        self.assertEqual(evaluation.support_rate, 1.0)
        self.assertEqual(evaluation.end_to_end_accuracy, 1.0)


class LocalAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        rag = ExtractiveRag(
            TfidfRetriever(
                [
                    Document(
                        "Agent 由工具、状态与有最大步数的控制循环组成。",
                        "agent",
                    ),
                    Document("RAG 在回答前检索资料。", "rag"),
                ]
            ),
            min_score=0.18,
            top_k=2,
        )
        self.agent = build_local_agent(rag)

    def test_calculator_uses_safe_arithmetic_subset(self) -> None:
        result = self.agent.run("计算：(2 + 3) * 4 - 6 / 2")
        self.assertTrue(result.succeeded)
        self.assertEqual(result.answer, "17")
        self.assertEqual(result.steps[0].action.tool_name, "calculator")
        self.assertEqual(result.stop_reason, "finish_action")
        self.assertEqual(safe_calculate("2 ** 10"), "1024")

        blocked = self.agent.run("计算：__import__('os').getcwd()")
        self.assertEqual(blocked.status, "error")
        self.assertEqual(blocked.stop_reason, "tool_error")

    def test_search_returns_grounded_answer_and_source(self) -> None:
        result = self.agent.run("检索：Agent 为什么需要控制循环？")
        self.assertTrue(result.succeeded)
        self.assertIn("控制循环", result.answer)
        self.assertIn("[agent]", result.answer)
        self.assertEqual(result.steps[0].action.tool_name, "knowledge_search")

    def test_unknown_tool_and_tool_exception_stop_with_error_state(self) -> None:
        unknown = LocalAgent(
            [Tool("calculator", "safe math", safe_calculate)],
            planner=lambda _task, _steps: AgentAction("shell", "dir"),
        ).run("call an unregistered tool")
        self.assertEqual(unknown.status, "error")
        self.assertEqual(unknown.stop_reason, "unknown_tool")
        self.assertIn("未知工具", unknown.answer)
        self.assertFalse(unknown.steps[0].observation.success)  # type: ignore[union-attr]

        def broken_tool(_input: str) -> str:
            raise RuntimeError("demonstration failure")

        failed = LocalAgent(
            [Tool("broken", "always fails", broken_tool)],
            planner=lambda _task, _steps: AgentAction("broken", "input"),
        ).run("exercise error handling")
        self.assertEqual(failed.status, "error")
        self.assertEqual(failed.stop_reason, "tool_error")
        self.assertIn("demonstration failure", failed.answer)

    def test_max_steps_stops_a_non_terminating_planner(self) -> None:
        agent = LocalAgent(
            [Tool("calculator", "safe math", safe_calculate)],
            planner=lambda _task, _steps: AgentAction("calculator", "1 + 1"),
        )
        result = agent.run("repeat forever", max_steps=3)

        self.assertEqual(result.status, "step_limit")
        self.assertEqual(result.stop_reason, "max_steps")
        self.assertEqual(len(result.steps), 3)
        self.assertTrue(all(step.observation.success for step in result.steps))  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
