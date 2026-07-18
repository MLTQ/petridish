"""Cached character-level Tiny Shakespeare task for corpus language experiments."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import urllib.request

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DEFAULT_CONTEXT_LENGTH = 64


def _download_text(cache_path: Path) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        request = urllib.request.Request(
            TINY_SHAKESPEARE_URL,
            headers={"User-Agent": "neural-petridish/0.1 corpus experiment"},
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
        cache_path.write_bytes(payload)
    return cache_path.read_text(encoding="utf-8")


def _chunk_generator(encoded: torch.Tensor, context_length: int):
    def generate(batch_size: int, generator: torch.Generator) -> SequenceBatch:
        maximum = encoded.numel() - context_length - 1
        starts = torch.randint(0, maximum, (batch_size,), generator=generator)
        rows = torch.stack(
            [encoded[int(start) : int(start) + context_length + 1] for start in starts]
        )
        tokens = rows[:, :-1]
        targets = rows[:, 1:]
        return SequenceBatch(tokens, targets, torch.ones_like(targets, dtype=torch.bool))

    return generate


@lru_cache(maxsize=4)
def load_tiny_shakespeare_task(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    cache_path: str = "data/tinyshakespeare/input.txt",
) -> SequenceTask:
    """Download once, split deterministically, and expose character chunk generators."""

    text = _download_text(Path(cache_path))
    vocabulary = tuple(sorted(set(text))) + ("�",)
    token_by_character = {character: index for index, character in enumerate(vocabulary)}
    unknown = token_by_character["�"]
    encoded = torch.tensor([token_by_character[character] for character in text], dtype=torch.long)
    split = int(encoded.numel() * 0.90)
    train = encoded[:split]
    validation = encoded[split:]

    def encode(value: str) -> list[int]:
        return [token_by_character.get(character, unknown) for character in value]

    def decode(tokens: list[int]) -> str:
        return "".join(vocabulary[token] if 0 <= token < len(vocabulary) else "�" for token in tokens)

    return SequenceTask(
        key="tiny_shakespeare",
        title="Tiny Shakespeare NCA",
        description=(
            "Predict characters from a cached public-domain Shakespeare subset; "
            "the organism receives a rolling corpus context rather than a fixed grammar."
        ),
        vocabulary=vocabulary,
        sequence_length=context_length,
        generator=_chunk_generator(train, context_length),
        evaluation_generator=_chunk_generator(validation, context_length),
        encode=encode,
        decode=decode,
        dataset_name="Tiny Shakespeare",
        dataset_characters=len(text),
        source_url=TINY_SHAKESPEARE_URL,
        training_stream=train,
        evaluation_stream=validation,
    )


__all__ = ["DEFAULT_CONTEXT_LENGTH", "TINY_SHAKESPEARE_URL", "load_tiny_shakespeare_task"]
