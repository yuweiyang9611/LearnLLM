from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learn_llm.evaluation import (
    EvaluationExample,
    EvaluationSummary,
    FormatRequirements,
    aggregate_seed_summaries,
    evaluate_prediction,
    evaluate_predictions,
    format_accuracy,
    keyword_accuracy,
    load_evaluation_jsonl,
    normalize_text,
    parse_evaluation_records,
    summarize_by_category,
)


class InstructionDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = load_evaluation_jsonl(DATA_DIR / "tiny_instructions.jsonl")
        cls.dev = load_evaluation_jsonl(DATA_DIR / "tiny_instructions_dev.jsonl")
        cls.test = load_evaluation_jsonl(DATA_DIR / "tiny_instructions_test.jsonl")
        cls.compatibility_eval = load_evaluation_jsonl(
            DATA_DIR / "tiny_instructions_eval.jsonl"
        )

    def test_unique_dataset_has_twelve_structured_examples(self) -> None:
        examples = self.train + self.dev + self.test
        self.assertEqual(len(examples), 12)
        self.assertEqual(len({example.example_id for example in examples}), 12)
        self.assertEqual({example.split for example in self.train}, {"train"})
        self.assertEqual({example.split for example in self.dev}, {"dev"})
        self.assertEqual({example.split for example in self.test}, {"test"})
        self.assertEqual(
            {example.evaluation_category for example in examples},
            {"seen", "paraphrase", "new_intent"},
        )

    def test_intent_families_make_category_meaning_explicit(self) -> None:
        train_families = {example.intent_family for example in self.train}
        paraphrase_families = {
            example.intent_family
            for example in self.dev + self.test
            if example.evaluation_category == "paraphrase"
        }
        new_intent_families = {
            example.intent_family
            for example in self.test
            if example.evaluation_category == "new_intent"
        }
        self.assertEqual(
            {example.evaluation_category for example in self.train}, {"seen"}
        )
        self.assertEqual(
            {example.evaluation_category for example in self.dev}, {"paraphrase"}
        )
        self.assertTrue(paraphrase_families <= train_families)
        self.assertTrue(new_intent_families)
        self.assertTrue(new_intent_families.isdisjoint(train_families))

    def test_compatibility_eval_is_exact_dev_and_test_mirror(self) -> None:
        expected = self.dev + self.test
        self.assertEqual(
            [example.example_id for example in self.compatibility_eval],
            [example.example_id for example in expected],
        )
        self.assertEqual(self.compatibility_eval, expected)

    def test_reference_answers_pass_declared_minimum_criteria(self) -> None:
        for example in self.train + self.dev + self.test:
            with self.subTest(example=example.example_id):
                result = evaluate_prediction(example, example.reference_response)
                self.assertEqual(result.keyword_accuracy, 1.0)
                self.assertEqual(result.format_accuracy, 1.0)
                self.assertEqual(result.task_success, 1.0)

    def test_all_splits_fit_frozen_character_vocab_and_block_size(self) -> None:
        corpus = (DATA_DIR / "tiny_corpus.txt").read_text(encoding="utf-8")
        training_prefix = corpus[: int(0.9 * len(corpus))]
        for example in self.train + self.dev + self.test:
            full_text = (
                f"用户：{example.instruction}\n助手：{example.reference_response}"
            )
            with self.subTest(example=example.example_id):
                self.assertLessEqual(len(full_text) - 1, 48)
                self.assertEqual(set(full_text) - set(training_prefix), set())


class EvaluationMetricTests(unittest.TestCase):
    @staticmethod
    def _example(
        *,
        example_id: str = "example-1",
        category: str = "paraphrase",
    ) -> EvaluationExample:
        return EvaluationExample.from_mapping(
            {
                "id": example_id,
                "split": "dev",
                "intent_family": "tokenization",
                "evaluation_category": category,
                "instruction": "什么是 token？",
                "response": "token 是 Tokenizer 产生的单位。",
                "required_keywords": ["token", "Tokenizer", "单位"],
                "format_requirements": {
                    "must_end_with": "。",
                    "max_sentences": 1,
                    "forbidden_substrings": ["用户：", "助手："],
                },
            }
        )

    def test_normalization_and_keyword_accuracy_are_transparent(self) -> None:
        self.assertEqual(normalize_text("  ToKeN\n  UNIT "), "token unit")
        score = keyword_accuracy("TOKEN 是一个单位。", ["token", "Tokenizer", "单位"])
        self.assertAlmostEqual(score, 2 / 3)

    def test_format_accuracy_requires_every_declared_rule(self) -> None:
        rules = FormatRequirements(
            must_end_with="。",
            max_sentences=1,
            forbidden_substrings=("助手：",),
        )
        self.assertEqual(format_accuracy("这是一个回答。", rules), 1.0)
        self.assertEqual(format_accuracy("这是一个回答", rules), 0.0)
        self.assertEqual(format_accuracy("第一句。第二句。", rules), 0.0)
        self.assertEqual(format_accuracy("助手：这是回答。", rules), 0.0)

    def test_task_success_is_strict_keyword_and_format_composite(self) -> None:
        example = self._example()
        successful = evaluate_prediction(
            example, "TOKEN 是 Tokenizer 产生的单位。"
        )
        missing_keyword = evaluate_prediction(example, "token 是单位。")
        bad_format = evaluate_prediction(
            example, "助手：token 是 Tokenizer 产生的单位。"
        )
        self.assertEqual(successful.task_success, 1.0)
        self.assertEqual(missing_keyword.keyword_accuracy, 2 / 3)
        self.assertEqual(missing_keyword.task_success, 0.0)
        self.assertEqual(bad_format.keyword_accuracy, 1.0)
        self.assertEqual(bad_format.format_accuracy, 0.0)
        self.assertEqual(bad_format.task_success, 0.0)

    def test_reports_are_strict_and_keep_category_scores_visible(self) -> None:
        paraphrase = self._example(example_id="p", category="paraphrase")
        new_intent = self._example(example_id="n", category="new_intent")
        report = evaluate_predictions(
            [paraphrase, new_intent],
            {
                "p": paraphrase.reference_response,
                "n": "token 是单位。",
            },
        )
        self.assertEqual(report.summary.example_count, 2)
        self.assertEqual(report.summary.task_success_rate, 0.5)
        by_category = summarize_by_category(report.results)
        self.assertEqual(by_category["paraphrase"].task_success_rate, 1.0)
        self.assertEqual(by_category["new_intent"].task_success_rate, 0.0)
        with self.assertRaisesRegex(ValueError, "prediction IDs do not match"):
            evaluate_predictions([paraphrase], {"wrong": "answer"})

    def test_multi_seed_summary_reports_population_spread(self) -> None:
        aggregate = aggregate_seed_summaries(
            {
                7: EvaluationSummary(2, 1.0, 0.75, 1.0),
                11: EvaluationSummary(2, 0.5, 0.25, 0.0),
            }
        )
        self.assertEqual(aggregate.seed_count, 2)
        self.assertEqual(aggregate.example_count, 2)
        self.assertEqual(aggregate.task_success_rate.mean, 0.75)
        self.assertEqual(aggregate.task_success_rate.population_std, 0.25)
        self.assertEqual(aggregate.keyword_accuracy.mean, 0.5)
        self.assertEqual(aggregate.format_accuracy.minimum, 0.0)
        self.assertEqual(aggregate.format_accuracy.maximum, 1.0)

    def test_record_parser_and_jsonl_loader_reject_duplicate_ids(self) -> None:
        record = {
            "id": "duplicate",
            "split": "test",
            "intent_family": "x",
            "evaluation_category": "new_intent",
            "instruction": "问题",
            "response": "回答。",
            "required_keywords": ["回答"],
            "format_requirements": {"must_end_with": "。"},
        }
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            parse_evaluation_records([record, record])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text(json.dumps(record) + "\nnot-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "is not valid JSON"):
                load_evaluation_jsonl(path)


if __name__ == "__main__":
    unittest.main()
