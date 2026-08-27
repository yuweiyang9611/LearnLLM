"""A deterministic character tokenizer suitable for first-principles lessons."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor


class CharTokenizer:
    """Map each Unicode character to one integer token.

    This tokenizer deliberately does not perform normalization: whitespace,
    punctuation, emoji, and different Unicode code points stay visible to the
    learner.  Real LLMs generally use subword tokenizers instead.
    """

    UNK_TOKEN = "<unk>"

    def __init__(self, characters: Iterable[str], *, add_unk: bool = False) -> None:
        unique = sorted(set(characters))
        if any(len(character) != 1 for character in unique):
            raise ValueError("every vocabulary item must be one Unicode character")
        if not unique:
            raise ValueError("the character vocabulary cannot be empty")

        self._characters = tuple(unique)
        tokens = ([self.UNK_TOKEN] if add_unk else []) + unique
        self.itos: dict[int, str] = dict(enumerate(tokens))
        self.stoi: dict[str, int] = {token: index for index, token in self.itos.items()}
        self.unk_id: int | None = self.stoi.get(self.UNK_TOKEN)

    @classmethod
    def from_text(cls, text: str, *, add_unk: bool = False) -> "CharTokenizer":
        """Build a reproducible vocabulary from all characters in ``text``."""

        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(text, add_unk=add_unk)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, text: str, *, return_tensor: bool = False) -> list[int] | Tensor:
        """Encode text, raising on unseen characters unless ``add_unk`` was used."""

        ids: list[int] = []
        for character in text:
            token_id = self.stoi.get(character, self.unk_id)
            if token_id is None:
                raise ValueError(
                    f"character {character!r} is not in the vocabulary; "
                    "build with add_unk=True to map unseen characters"
                )
            ids.append(token_id)
        if return_tensor:
            return torch.tensor(ids, dtype=torch.long)
        return ids

    def decode(self, token_ids: Sequence[int] | Tensor) -> str:
        """Turn token IDs back into text (``�`` represents an unknown token)."""

        if isinstance(token_ids, Tensor):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        output: list[str] = []
        for token_id in token_ids:
            token = self.itos.get(int(token_id))
            if token is None:
                raise ValueError(f"token id {token_id} is outside the vocabulary")
            output.append("�" if token == self.UNK_TOKEN else token)
        return "".join(output)

    def state_dict(self) -> dict[str, object]:
        """Return JSON-serializable tokenizer state."""

        return {"characters": list(self._characters), "add_unk": self.unk_id is not None}

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "CharTokenizer":
        characters = state.get("characters")
        if not isinstance(characters, list) or not all(isinstance(item, str) for item in characters):
            raise ValueError("invalid tokenizer state: characters must be a list of strings")
        return cls(characters, add_unk=bool(state.get("add_unk", False)))
