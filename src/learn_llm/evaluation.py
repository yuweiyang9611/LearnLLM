"""Pure, small evaluation helpers for the instruction-learning experiments.

The module deliberately uses transparent substring and formatting rules.  It
does not claim to measure semantic equivalence; instead, each dataset row
declares the minimum keywords and output shape needed for a teaching task to
count as successful.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_SPLITS = frozenset({"train", "dev", "test"})
EVALUATION_RULES_VERSION = 2
VALID_EVALUATION_CATEGORIES = frozenset(
    {"seen", "paraphrase", "new_intent"}
)

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"[。！？!?]+")


def normalize_text(text: str) -> str:
    """Case-fold text and collapse whitespace for deterministic matching."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _WHITESPACE.sub(" ", text.strip()).casefold()


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _string_tuple(value: object, *, field: str, allow_empty: bool) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a list of strings")
    items = tuple(
        _non_empty_string(item, field=f"{field} item")
        for item in value
    )
    if not allow_empty and not items:
        raise ValueError(f"{field} cannot be empty")
    if len(set(items)) != len(items):
        raise ValueError(f"{field} cannot contain duplicates")
    return items


@dataclass(frozen=True, slots=True)
class FormatRequirements:
    """Small, explicit output rules stored alongside each evaluation row."""

    must_end_with: str | None = None
    max_sentences: int | None = None
    forbidden_substrings: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: object) -> FormatRequirements:
        if not isinstance(value, Mapping):
            raise ValueError("format_requirements must be an object")
        unknown = set(value) - {
            "must_end_with",
            "max_sentences",
            "forbidden_substrings",
        }
        if unknown:
            raise ValueError(
                f"unknown format requirement(s): {', '.join(sorted(unknown))}"
            )

        ending_value = value.get("must_end_with")
        ending = (
            None
            if ending_value is None
            else _non_empty_string(ending_value, field="must_end_with")
        )
        maximum = value.get("max_sentences")
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise ValueError("max_sentences must be a positive integer")
        forbidden = _string_tuple(
            value.get("forbidden_substrings", ()),
            field="forbidden_substrings",
            allow_empty=True,
        )
        return cls(
            must_end_with=ending,
            max_sentences=maximum,
            forbidden_substrings=forbidden,
        )


@dataclass(frozen=True, slots=True)
class EvaluationExample:
    """One structured train/dev/test row used by the simple evaluator."""

    example_id: str
    split: str
    intent_family: str
    evaluation_category: str
    instruction: str
    reference_response: str
    required_keywords: tuple[str, ...]
    format_requirements: FormatRequirements
    required_concepts: tuple[tuple[str, ...], ...] = ()
    contradictions: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> EvaluationExample:
        concepts = record.get("required_concepts", [])
        if not isinstance(concepts, list):
            raise ValueError("required_concepts must be a list of synonym groups")
        parsed_concepts = tuple(_string_tuple(group, field="concept", allow_empty=False) for group in concepts)
        split = _non_empty_string(record.get("split"), field="split")
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split: {split!r}")
        category = _non_empty_string(
            record.get("evaluation_category"),
            field="evaluation_category",
        )
        if category not in VALID_EVALUATION_CATEGORIES:
            raise ValueError(f"unknown evaluation_category: {category!r}")
        return cls(
            example_id=_non_empty_string(record.get("id"), field="id"),
            split=split,
            intent_family=_non_empty_string(
                record.get("intent_family"), field="intent_family"
            ),
            evaluation_category=category,
            instruction=_non_empty_string(
                record.get("instruction"), field="instruction"
            ),
            reference_response=_non_empty_string(
                record.get("response"), field="response"
            ),
            required_keywords=_string_tuple(
                record.get("required_keywords"),
                field="required_keywords",
                allow_empty=False,
            ),
            format_requirements=FormatRequirements.from_mapping(
                record.get("format_requirements")
            ),
            required_concepts=parsed_concepts,
            contradictions=_string_tuple(record.get("contradictions", []), field="contradictions", allow_empty=True),
        )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    example_id: str
    intent_family: str
    evaluation_category: str
    task_success: float
    keyword_accuracy: float
    format_accuracy: float
    matched_keywords: tuple[str, ...]
    missing_concepts: tuple[tuple[str, ...], ...] = ()
    matched_contradictions: tuple[str, ...] = ()
    format_failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    example_count: int
    task_success_rate: float
    keyword_accuracy: float
    format_accuracy: float


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    results: tuple[EvaluationResult, ...]
    summary: EvaluationSummary


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    mean: float
    population_std: float
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class MultiSeedSummary:
    seed_count: int
    example_count: int
    task_success_rate: MetricAggregate
    keyword_accuracy: MetricAggregate
    format_accuracy: MetricAggregate


def parse_evaluation_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[EvaluationExample, ...]:
    """Validate structured records and reject duplicate example IDs."""

    examples = tuple(EvaluationExample.from_mapping(record) for record in records)
    if not examples:
        raise ValueError("at least one evaluation record is required")
    ids = [example.example_id for example in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("evaluation example IDs must be unique")
    return examples


def load_evaluation_jsonl(path: str | Path) -> tuple[EvaluationExample, ...]:
    """Load one UTF-8 JSONL split; metric computation remains pure."""

    source = Path(path)
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{source}:{line_number} is not valid JSON"
            ) from error
        if not isinstance(record, Mapping):
            raise ValueError(f"{source}:{line_number} must contain an object")
        records.append(record)
    return parse_evaluation_records(records)


def matched_keywords(
    prediction: str,
    required_keywords: Sequence[str],
) -> tuple[str, ...]:
    """Return required keywords found as normalized substrings."""

    keywords = _string_tuple(
        required_keywords,
        field="required_keywords",
        allow_empty=False,
    )
    normalized_prediction = normalize_text(prediction)
    return tuple(
        keyword
        for keyword in keywords
        if normalize_text(keyword) in normalized_prediction
    )


def keyword_accuracy(prediction: str, required_keywords: Sequence[str]) -> float:
    """Return the fraction of declared keywords present in ``prediction``."""

    keywords = _string_tuple(
        required_keywords,
        field="required_keywords",
        allow_empty=False,
    )
    return len(matched_keywords(prediction, keywords)) / len(keywords)


def format_accuracy(
    prediction: str,
    requirements: FormatRequirements,
) -> float:
    """Return 1.0 only when every declared formatting rule passes."""

    normalized = normalize_text(prediction)
    if not normalized:
        return 0.0
    stripped = prediction.strip()
    if requirements.must_end_with is not None and not stripped.endswith(
        requirements.must_end_with
    ):
        return 0.0
    if requirements.max_sentences is not None:
        sentence_count = len(
            [part for part in _SENTENCE_END.split(stripped) if part.strip()]
        )
        if sentence_count > requirements.max_sentences:
            return 0.0
    if any(
        normalize_text(forbidden) in normalized
        for forbidden in requirements.forbidden_substrings
    ):
        return 0.0
    return 1.0


def evaluate_prediction(
    example: EvaluationExample,
    prediction: str,
) -> EvaluationResult:
    """Evaluate one answer using the row's transparent minimum criteria."""

    found = matched_keywords(prediction, example.required_keywords)
    keyword_score = len(found) / len(example.required_keywords)
    format_score = format_accuracy(prediction, example.format_requirements)
    normalized = normalize_text(prediction)
    concepts = example.required_concepts or tuple((word,) for word in example.required_keywords)
    missing = tuple(group for group in concepts if not any(normalize_text(word) in normalized for word in group))
    contradictions = tuple(rule for rule in example.contradictions if normalize_text(rule) in normalized)
    failures: list[str] = []
    rules = example.format_requirements
    if rules.must_end_with and not prediction.strip().endswith(rules.must_end_with):
        failures.append("must_end_with")
    if rules.max_sentences is not None and len([part for part in _SENTENCE_END.split(prediction.strip()) if part.strip()]) > rules.max_sentences:
        failures.append("max_sentences")
    failures.extend(f"forbidden_substring:{word}" for word in rules.forbidden_substrings if normalize_text(word) in normalized)
    task_success = float(not missing and not contradictions and format_score == 1.0)
    return EvaluationResult(
        example_id=example.example_id,
        intent_family=example.intent_family,
        evaluation_category=example.evaluation_category,
        task_success=task_success,
        keyword_accuracy=keyword_score,
        format_accuracy=format_score,
        matched_keywords=found,
        missing_concepts=missing,
        matched_contradictions=contradictions,
        format_failures=tuple(failures),
    )


def summarize_results(results: Sequence[EvaluationResult]) -> EvaluationSummary:
    """Average per-example metrics without hiding the denominator."""

    if not results:
        raise ValueError("at least one evaluation result is required")
    count = len(results)
    return EvaluationSummary(
        example_count=count,
        task_success_rate=sum(item.task_success for item in results) / count,
        keyword_accuracy=sum(item.keyword_accuracy for item in results) / count,
        format_accuracy=sum(item.format_accuracy for item in results) / count,
    )


def evaluate_predictions(
    examples: Sequence[EvaluationExample],
    predictions: Mapping[str, str],
) -> EvaluationReport:
    """Strictly match predictions by example ID and summarize all examples."""

    if not examples:
        raise ValueError("at least one evaluation example is required")
    example_ids = [example.example_id for example in examples]
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("evaluation example IDs must be unique")
    missing = sorted(set(example_ids) - set(predictions))
    unexpected = sorted(set(predictions) - set(example_ids))
    if missing or unexpected:
        raise ValueError(
            f"prediction IDs do not match; missing={missing}, "
            f"unexpected={unexpected}"
        )
    results = tuple(
        evaluate_prediction(example, predictions[example.example_id])
        for example in examples
    )
    return EvaluationReport(results=results, summary=summarize_results(results))


def summarize_by_category(
    results: Sequence[EvaluationResult],
) -> dict[str, EvaluationSummary]:
    """Expose seen/paraphrase/new-intent scores instead of one blended mean."""

    grouped: dict[str, list[EvaluationResult]] = {}
    for result in results:
        grouped.setdefault(result.evaluation_category, []).append(result)
    return {
        category: summarize_results(grouped[category])
        for category in sorted(grouped)
    }


def _aggregate(values: Sequence[float]) -> MetricAggregate:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("seed metrics must be non-empty and finite")
    return MetricAggregate(
        mean=statistics.fmean(values),
        population_std=statistics.pstdev(values),
        minimum=min(values),
        maximum=max(values),
    )


def aggregate_seed_summaries(
    summaries: Mapping[int, EvaluationSummary],
) -> MultiSeedSummary:
    """Aggregate comparable runs and report population spread across seeds."""

    if not summaries:
        raise ValueError("at least one seed summary is required")
    example_counts = {summary.example_count for summary in summaries.values()}
    if len(example_counts) != 1:
        raise ValueError("all seed summaries must use the same example count")
    ordered = [summaries[seed] for seed in sorted(summaries)]
    return MultiSeedSummary(
        seed_count=len(ordered),
        example_count=ordered[0].example_count,
        task_success_rate=_aggregate(
            [summary.task_success_rate for summary in ordered]
        ),
        keyword_accuracy=_aggregate(
            [summary.keyword_accuracy for summary in ordered]
        ),
        format_accuracy=_aggregate(
            [summary.format_accuracy for summary in ordered]
        ),
    )
