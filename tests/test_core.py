from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learn_llm.attention import scaled_dot_product_attention
from learn_llm.lora import LoRALinear
from learn_llm.math_utils import cross_entropy_from_logits, stable_softmax
from learn_llm.model import TinyGPT, TinyGPTConfig
from learn_llm.rag import Document, TfidfRetriever
from learn_llm.tokenizer import CharTokenizer
from learn_llm.training import load_checkpoint, save_checkpoint, train_steps


class MathTests(unittest.TestCase):
    def test_stable_softmax_handles_large_logits(self) -> None:
        logits = torch.tensor([[10_000.0, 10_001.0, 9_999.0]])
        probabilities = stable_softmax(logits)
        self.assertTrue(torch.isfinite(probabilities).all())
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(1))
        torch.testing.assert_close(probabilities, torch.softmax(logits, dim=-1))

    def test_cross_entropy_matches_pytorch(self) -> None:
        logits = torch.tensor(
            [[2.0, -1.0, 0.5], [10_000.0, 9_999.0, 9_998.0]],
            requires_grad=True,
        )
        targets = torch.tensor([0, 2])
        actual = cross_entropy_from_logits(logits, targets)
        expected = F.cross_entropy(logits, targets)
        torch.testing.assert_close(actual, expected)


class TokenizerTests(unittest.TestCase):
    def test_character_tokenizer_round_trip(self) -> None:
        text = "LLM 入门\nHello!"
        tokenizer = CharTokenizer.from_text(text)
        encoded = tokenizer.encode(text)
        self.assertEqual(tokenizer.decode(encoded), text)
        self.assertEqual(tokenizer.vocab_size, len(set(text)))


class AttentionTests(unittest.TestCase):
    def test_attention_rows_sum_to_one_and_causal_mask_blocks_future(self) -> None:
        query = torch.zeros(1, 1, 4, 2)
        key = torch.zeros_like(query)
        value = torch.arange(8, dtype=torch.float32).view(1, 1, 4, 2)
        _, weights = scaled_dot_product_attention(query, key, value, causal=True)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 1, 4))
        self.assertTrue(torch.equal(torch.triu(weights[0, 0], diagonal=1), torch.zeros(4, 4)))

    def test_decoder_has_no_future_token_leakage(self) -> None:
        torch.manual_seed(7)
        model = TinyGPT(
            TinyGPTConfig(
                vocab_size=10, block_size=5, n_layer=1, n_head=2, n_embd=12
            )
        ).eval()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        changed_future = torch.tensor([[1, 2, 3, 9, 8]])
        first_logits, _ = model(first)
        changed_logits, _ = model(changed_future)
        torch.testing.assert_close(first_logits[:, :3], changed_logits[:, :3])


class ModelTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Small matrix operations are faster and more deterministic without a
        # large CPU thread pool.
        torch.set_num_threads(1)

    @staticmethod
    def _model() -> TinyGPT:
        return TinyGPT(
            TinyGPTConfig(
                vocab_size=4,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=16,
                dropout=0.0,
            )
        )

    def test_tiny_batch_loss_declines(self) -> None:
        torch.manual_seed(11)
        model = self._model()
        inputs = torch.tensor(
            [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [1, 2, 3, 0, 1, 2, 3, 0],
                [2, 3, 0, 1, 2, 3, 0, 1],
                [3, 0, 1, 2, 3, 0, 1, 2],
            ]
        )
        targets = (inputs + 1) % 4
        losses = train_steps(model, inputs, targets, steps=24, learning_rate=0.03)
        self.assertLess(losses[-1], losses[0] * 0.35)

    def test_checkpoint_round_trip(self) -> None:
        torch.manual_seed(23)
        model = self._model().eval()
        inputs = torch.tensor([[0, 1, 2, 3]])
        expected, _ = model(inputs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny-gpt.pt"
            save_checkpoint(path, model, step=17, metadata={"lesson": "checkpoint"})
            restored = self._model().eval()
            information = load_checkpoint(path, restored)
            actual, _ = restored(inputs)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(information["step"], 17)
        self.assertEqual(information["metadata"]["lesson"], "checkpoint")


class AdaptationAndRetrievalTests(unittest.TestCase):
    def test_only_lora_parameters_are_trainable(self) -> None:
        base = nn.Linear(6, 4)
        inputs = torch.randn(3, 6)
        expected = base(inputs).detach()
        layer = LoRALinear(base, rank=2, alpha=4.0)

        trainable = {
            name for name, parameter in layer.named_parameters() if parameter.requires_grad
        }
        self.assertEqual(trainable, {"lora_A", "lora_B"})
        torch.testing.assert_close(layer(inputs), expected)

    def test_tfidf_retriever_recalls_relevant_document(self) -> None:
        documents = [
            Document("Saturn has prominent rings made largely of ice particles.", "saturn"),
            Document("Mars is known for its iron-rich red surface.", "mars"),
            Document("Venus has a dense carbon dioxide atmosphere.", "venus"),
        ]
        retriever = TfidfRetriever(documents)
        result = retriever.retrieve("Which planet has rings made of ice?", top_k=1)[0]
        self.assertEqual(result.document.doc_id, "saturn")
        self.assertGreater(result.score, 0.0)

        chinese = TfidfRetriever(
            [
                Document("注意力机制让模型按相关性聚合上下文。", "attention"),
                Document("分词器把原始文本转换成 token 序列。", "tokenizer"),
            ]
        )
        chinese_result = chinese.retrieve("模型怎样聚合上下文？", top_k=1)[0]
        self.assertEqual(chinese_result.document.doc_id, "attention")


if __name__ == "__main__":
    unittest.main()
