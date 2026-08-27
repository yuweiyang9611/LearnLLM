"""Small, local RAG building blocks used by the introductory lessons.

The module deliberately separates retrieval from answering:

* :class:`TfidfRetriever` ranks documents.
* :class:`ExtractiveRag` copies a relevant sentence from a retrieved document
  and attaches its source ID.

The extractive answerer is a deterministic teaching baseline, **not** a
language model.  Its limited behaviour is useful here because every claim can
be checked against the supplied evidence and out-of-scope questions can be
refused explicitly.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


_SEGMENT_PATTERN = re.compile(
    r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+", flags=re.IGNORECASE
)
_CJK_PATTERN = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")
_SENTENCE_PATTERN = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", flags=re.MULTILINE)
_CITATION_PATTERN = re.compile(r"\s*\[[^\[\]]+\]\s*")
_NORMALIZE_PATTERN = re.compile(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", re.IGNORECASE)

REFUSAL_MESSAGE = "资料不足，无法根据当前知识库回答。"


@dataclass(frozen=True, slots=True)
class Document:
    text: str
    doc_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str | None:
        """A short read-only alias for integrations that call IDs simply ``id``."""

        return self.doc_id


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    document: Document
    score: float
    index: int

    @property
    def text(self) -> str:
        return self.document.text

    @property
    def doc_id(self) -> str | None:
        return self.document.doc_id

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.document.metadata


@dataclass(frozen=True, slots=True)
class SupportReport:
    """Result of checking whether answer claims occur in the cited sources."""

    supported_claims: int
    claim_count: int
    unsupported_claims: tuple[str, ...] = ()

    @property
    def ratio(self) -> float:
        return self.supported_claims / self.claim_count if self.claim_count else 1.0

    @property
    def is_fully_supported(self) -> bool:
        return not self.unsupported_claims


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """An answer together with the exact retrieval evidence used to produce it."""

    question: str
    answer: str
    evidence: tuple[RetrievalResult, ...]
    sources: tuple[Document, ...]
    refused: bool
    support: SupportReport

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(
            source.doc_id for source in self.sources if source.doc_id is not None
        )

    @property
    def is_supported(self) -> bool:
        return self.support.is_fully_supported


@dataclass(frozen=True, slots=True)
class RagEvaluationCase:
    """One labelled question for a tiny end-to-end RAG evaluation."""

    question: str
    expected_source_id: str | None = None
    expect_refusal: bool = False
    expected_text: str | None = None

    def __post_init__(self) -> None:
        if self.expect_refusal and self.expected_source_id is not None:
            raise ValueError("a refusal case cannot require a source")


@dataclass(frozen=True, slots=True)
class RagEvaluation:
    """Transparent metrics for retrieval, refusal, support and end-to-end success."""

    total: int
    retrieval_cases: int
    retrieval_hits: int
    refusal_cases: int
    correct_refusals: int
    answered_cases: int
    supported_answers: int
    end_to_end_successes: int

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 1.0

    @property
    def retrieval_recall(self) -> float:
        return self._rate(self.retrieval_hits, self.retrieval_cases)

    @property
    def refusal_accuracy(self) -> float:
        return self._rate(self.correct_refusals, self.refusal_cases)

    @property
    def support_rate(self) -> float:
        return self._rate(self.supported_answers, self.answered_cases)

    @property
    def end_to_end_accuracy(self) -> float:
        return self._rate(self.end_to_end_successes, self.total)


class TfidfRetriever:
    """Rank small local document collections by cosine similarity of TF-IDF."""

    def __init__(self, documents: Iterable[Document | str] | None = None) -> None:
        self.documents: list[Document] = []
        self.vocabulary: set[str] = set()
        self.idf: dict[str, float] = {}
        self._document_vectors: list[dict[str, float]] = []
        self._document_norms: list[float] = []
        if documents is not None:
            self.fit(documents)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens: list[str] = []
        for segment in _SEGMENT_PATTERN.findall(text.casefold()):
            if _CJK_PATTERN.fullmatch(segment):
                # Chinese normally has no spaces between words. Character
                # unigrams and bigrams provide a transparent, dependency-free
                # approximation that works well for a small teaching corpus.
                tokens.extend(segment)
                tokens.extend(
                    segment[index : index + 2]
                    for index in range(max(len(segment) - 1, 0))
                )
            else:
                tokens.append(segment)
        return tokens

    def fit(self, documents: Iterable[Document | str]) -> "TfidfRetriever":
        normalized = [
            document if isinstance(document, Document) else Document(str(document))
            for document in documents
        ]
        if not normalized:
            raise ValueError("at least one document is required")

        tokenized = [self.tokenize(document.text) for document in normalized]
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        count = len(normalized)
        self.documents = normalized
        self.vocabulary = set(document_frequency)
        self.idf = {
            term: math.log((1 + count) / (1 + frequency)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self._document_vectors = [self._vectorize_tokens(tokens) for tokens in tokenized]
        self._document_norms = [self._norm(vector) for vector in self._document_vectors]
        return self

    def _vectorize_tokens(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(token for token in tokens if token in self.idf)
        if not counts:
            return {}
        total = sum(counts.values())
        return {term: count / total * self.idf[term] for term, count in counts.items()}

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in vector.values()))

    def retrieve(self, query: str, *, top_k: int = 3) -> list[RetrievalResult]:
        if not self.documents:
            raise RuntimeError("fit the retriever before calling retrieve")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self._vectorize_tokens(self.tokenize(query))
        query_norm = self._norm(query_vector)
        scored: list[tuple[float, int]] = []
        for index, (document_vector, document_norm) in enumerate(
            zip(self._document_vectors, self._document_norms, strict=True)
        ):
            if query_norm == 0.0 or document_norm == 0.0:
                similarity = 0.0
            else:
                dot_product = sum(
                    value * document_vector.get(term, 0.0)
                    for term, value in query_vector.items()
                )
                similarity = dot_product / (query_norm * document_norm)
            scored.append((similarity, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievalResult(self.documents[index], score, index)
            for score, index in scored[: min(top_k, len(scored))]
        ]

    # ``search`` reads naturally in interactive experiments.
    search = retrieve


def _split_sentences(text: str) -> list[str]:
    return [
        match.group(0).strip()
        for match in _SENTENCE_PATTERN.finditer(text)
        if match.group(0).strip()
    ]


def _normalize_claim(text: str) -> str:
    return _NORMALIZE_PATTERN.sub("", text.casefold())


def check_answer_support(answer: str, sources: Iterable[Document]) -> SupportReport:
    """Check answer sentences by exact normalized containment in cited documents.

    This is intentionally stricter and simpler than an LLM-based faithfulness
    judge.  It works for :class:`ExtractiveRag` because that baseline copies
    source sentences instead of paraphrasing them.
    """

    source_texts = tuple(_normalize_claim(source.text) for source in sources)
    answer_without_citations = _CITATION_PATTERN.sub(" ", answer)
    claims = [
        sentence.rstrip("。！？!?；;").strip()
        for sentence in _split_sentences(answer_without_citations)
        if sentence.rstrip("。！？!?；;").strip()
    ]
    unsupported: list[str] = []
    supported = 0
    for claim in claims:
        normalized = _normalize_claim(claim)
        if normalized and any(normalized in source for source in source_texts):
            supported += 1
        else:
            unsupported.append(claim)
    return SupportReport(supported, len(claims), tuple(unsupported))


class ExtractiveRag:
    """A deterministic, citation-producing answer layer over local retrieval.

    The confidence threshold is a refusal gate.  Above it, up to
    ``max_sentences`` from the best evidence document are copied verbatim;
    below it, the system says that the local material is insufficient.
    """

    def __init__(
        self,
        retriever: TfidfRetriever,
        *,
        min_score: float = 0.18,
        top_k: int = 3,
        max_sentences: int = 2,
    ) -> None:
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        self.retriever = retriever
        self.min_score = min_score
        self.top_k = top_k
        self.max_sentences = max_sentences

    def _sentence_score(self, question: str, sentence: str) -> float:
        query_tokens = self.retriever.tokenize(question)
        if not query_tokens:
            return 0.0
        sentence_tokens = set(self.retriever.tokenize(sentence))
        query_weight = sum(self.retriever.idf.get(token, 1.0) for token in query_tokens)
        shared_weight = sum(
            self.retriever.idf.get(token, 1.0)
            for token in query_tokens
            if token in sentence_tokens
        )
        return shared_weight / query_weight if query_weight else 0.0

    def answer(self, question: str) -> GroundedAnswer:
        evidence = tuple(self.retriever.retrieve(question, top_k=self.top_k))
        if not question.strip() or not evidence or evidence[0].score < self.min_score:
            support = SupportReport(0, 0)
            return GroundedAnswer(
                question=question,
                answer=REFUSAL_MESSAGE,
                evidence=evidence,
                sources=(),
                refused=True,
                support=support,
            )

        candidates: list[tuple[float, float, int, int, str, RetrievalResult]] = []
        for document_rank, result in enumerate(evidence):
            for sentence_rank, sentence in enumerate(_split_sentences(result.text)):
                candidates.append(
                    (
                        self._sentence_score(question, sentence),
                        result.score,
                        -document_rank,
                        -sentence_rank,
                        sentence,
                        result,
                    )
                )
        if not candidates:
            support = SupportReport(0, 0)
            return GroundedAnswer(
                question=question,
                answer=REFUSAL_MESSAGE,
                evidence=evidence,
                sources=(),
                refused=True,
                support=support,
            )

        _, _, _, _, _sentence, selected = max(candidates, key=lambda item: item[:4])
        source = selected.document
        source_label = source.doc_id or f"document-{selected.index + 1}"
        source_sentences = _split_sentences(source.text)
        ranked_sentences = sorted(
            enumerate(source_sentences),
            key=lambda item: (-self._sentence_score(question, item[1]), item[0]),
        )[: self.max_sentences]
        selected_sentences = [
            sentence for _, sentence in sorted(ranked_sentences, key=lambda item: item[0])
        ]
        answer = f"{' '.join(selected_sentences)} [{source_label}]"
        sources = (source,)
        support = check_answer_support(answer, sources)
        return GroundedAnswer(
            question=question,
            answer=answer,
            evidence=evidence,
            sources=sources,
            refused=False,
            support=support,
        )


def evaluate_grounded_rag(
    rag: ExtractiveRag, cases: Iterable[RagEvaluationCase]
) -> RagEvaluation:
    """Run a labelled local evaluation without using a model as the judge."""

    normalized_cases = tuple(cases)
    if not normalized_cases:
        raise ValueError("at least one evaluation case is required")

    retrieval_cases = retrieval_hits = 0
    refusal_cases = correct_refusals = 0
    answered_cases = supported_answers = 0
    end_to_end_successes = 0
    for case in normalized_cases:
        result = rag.answer(case.question)
        source_hit = True
        if case.expected_source_id is not None:
            retrieval_cases += 1
            source_hit = case.expected_source_id in result.source_ids
            retrieval_hits += int(source_hit)

        refusal_ok = True
        if case.expect_refusal:
            refusal_cases += 1
            refusal_ok = result.refused
            correct_refusals += int(refusal_ok)
        else:
            answered_cases += 1
            supported_answers += int(not result.refused and result.is_supported)
            refusal_ok = not result.refused

        text_ok = case.expected_text is None or case.expected_text in result.answer
        support_ok = case.expect_refusal or result.is_supported
        success = source_hit and refusal_ok and text_ok and support_ok
        end_to_end_successes += int(success)

    return RagEvaluation(
        total=len(normalized_cases),
        retrieval_cases=retrieval_cases,
        retrieval_hits=retrieval_hits,
        refusal_cases=refusal_cases,
        correct_refusals=correct_refusals,
        answered_cases=answered_cases,
        supported_answers=supported_answers,
        end_to_end_successes=end_to_end_successes,
    )
