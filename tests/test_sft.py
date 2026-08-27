from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from learn_llm.model import TinyGPT, TinyGPTConfig
from learn_llm.sft import (
    IGNORE_INDEX,
    build_sft_sequence,
    collate_sft_batch,
    evaluate_assistant_loss,
    format_instruction_prompt,
    train_sft_steps,
)
from learn_llm.tokenizer import CharTokenizer


class SFTMaskTests(unittest.TestCase):
    def test_shifted_labels_start_at_first_assistant_token(self) -> None:
        instruction = "问"
        response = "答好"
        prompt = format_instruction_prompt(instruction)
        tokenizer = CharTokenizer.from_text(prompt + response)
        sequence = build_sft_sequence(tokenizer, instruction, response)

        prompt_ids = tokenizer.encode(prompt)
        response_ids = tokenizer.encode(response)
        full_ids = tokenizer.encode(prompt + response)
        self.assertIsInstance(prompt_ids, list)
        self.assertIsInstance(response_ids, list)
        self.assertIsInstance(full_ids, list)

        boundary = len(prompt_ids) - 1
        self.assertEqual(sequence.first_assistant_label_index, boundary)
        self.assertEqual(sequence.input_ids, tuple(full_ids[:-1]))
        self.assertEqual(sequence.labels[:boundary], (IGNORE_INDEX,) * boundary)
        self.assertEqual(sequence.labels[boundary:], tuple(response_ids))
        self.assertEqual(sequence.response_token_count, len(response_ids))

        # Every non-ignored label is genuinely the next token, not the token at
        # the same unshifted position.
        for position, label in enumerate(sequence.labels):
            if label != IGNORE_INDEX:
                self.assertEqual(label, full_ids[position + 1])

    def test_variable_length_batch_masks_right_padding(self) -> None:
        first_prompt = format_instruction_prompt("甲")
        second_prompt = format_instruction_prompt("乙")
        tokenizer = CharTokenizer.from_text(first_prompt + second_prompt + "短较长答案")
        short = build_sft_sequence(tokenizer, "甲", "短")
        long = build_sft_sequence(tokenizer, "乙", "较长答案")
        pad_token_id = tokenizer.stoi["\n"]

        batch = collate_sft_batch([short, long], pad_token_id=pad_token_id)

        self.assertEqual(batch.input_ids.dtype, torch.long)
        self.assertEqual(batch.labels.dtype, torch.long)
        self.assertEqual(batch.lengths.tolist(), [short.sequence_length, long.sequence_length])
        self.assertTrue(
            torch.equal(
                batch.input_ids[0, short.sequence_length :],
                torch.full(
                    (long.sequence_length - short.sequence_length,), pad_token_id
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                batch.labels[0, short.sequence_length :],
                torch.full(
                    (long.sequence_length - short.sequence_length,), IGNORE_INDEX
                ),
            )
        )
        self.assertEqual(
            batch.supervised_token_count,
            short.response_token_count + long.response_token_count,
        )

    def test_sequence_limit_fails_instead_of_silently_cutting_answer(self) -> None:
        prompt = format_instruction_prompt("问题")
        tokenizer = CharTokenizer.from_text(prompt + "答案")
        with self.assertRaisesRegex(ValueError, "exceeds max_sequence_length"):
            build_sft_sequence(
                tokenizer,
                "问题",
                "答案",
                max_sequence_length=2,
            )


class SFTTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_tiny_gpt_assistant_loss_declines(self) -> None:
        torch.manual_seed(17)
        pairs = [("A", "xy"), ("B", "zzzz")]
        corpus = "".join(
            format_instruction_prompt(instruction) + response
            for instruction, response in pairs
        )
        tokenizer = CharTokenizer.from_text(corpus)
        sequences = [
            build_sft_sequence(tokenizer, instruction, response)
            for instruction, response in pairs
        ]
        batch = collate_sft_batch(sequences, pad_token_id=0)
        model = TinyGPT(
            TinyGPTConfig(
                vocab_size=tokenizer.vocab_size,
                block_size=batch.input_ids.shape[1],
                n_layer=1,
                n_head=2,
                n_embd=16,
                dropout=0.0,
            )
        )

        initial_loss = evaluate_assistant_loss(model, batch)
        history = train_sft_steps(
            model,
            batch,
            steps=35,
            learning_rate=0.03,
        )
        final_loss = evaluate_assistant_loss(model, batch)

        self.assertEqual(len(history), 35)
        self.assertLess(final_loss, initial_loss * 0.35)


if __name__ == "__main__":
    unittest.main()
