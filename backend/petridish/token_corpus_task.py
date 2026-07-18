"""Cached token-level TinyStories task for persistent cellular language experiments."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
import re
import urllib.request

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


TINY_STORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-valid.txt?download=true"
)
DEFAULT_CONTEXT_LENGTH = 64
DEFAULT_VOCAB_SIZE = 2_048
TOKEN_PATTERN = re.compile(r"\n| ?[A-Za-z]+(?:'[A-Za-z]+)?| ?[0-9]+| ?[^\w\s]")


def _download_text(cache_path: Path) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        request = urllib.request.Request(
            TINY_STORIES_URL,
            headers={"User-Agent": "neural-petridish/0.2 token corpus experiment"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            cache_path.write_bytes(response.read())
    return cache_path.read_text(encoding="utf-8")


def _pieces(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.replace("\r\n", "\n"))


def _chunk_generator(encoded: torch.Tensor, context_length: int):
    def generate(batch_size: int, generator: torch.Generator) -> SequenceBatch:
        maximum = encoded.numel() - context_length - 1
        starts = torch.randint(0, maximum, (batch_size,), generator=generator)
        offsets = torch.arange(context_length + 1)
        rows = encoded[starts.unsqueeze(1) + offsets]
        return SequenceBatch(
            rows[:, :-1], rows[:, 1:],
            torch.ones(batch_size, context_length, dtype=torch.bool),
        )

    return generate


def _next_token_baselines(
    train: torch.Tensor, validation: torch.Tensor, vocabulary_size: int
) -> tuple[float, float]:
    """Measure exact validation unigram and train-fitted bigram accuracy."""

    unigram_counts = torch.bincount(train, minlength=vocabulary_size)
    unigram_token = int(unigram_counts.argmax())
    targets = validation[1:]
    unigram_accuracy = float((targets == unigram_token).float().mean())
    pair_ids = train[:-1] * vocabulary_size + train[1:]
    bigram_counts = torch.bincount(
        pair_ids, minlength=vocabulary_size * vocabulary_size
    ).reshape(vocabulary_size, vocabulary_size)
    bigram_prediction = bigram_counts.argmax(dim=1)
    unseen = bigram_counts.sum(dim=1) == 0
    bigram_prediction[unseen] = unigram_token
    bigram_accuracy = float(
        (bigram_prediction[validation[:-1]] == targets).float().mean()
    )
    return unigram_accuracy, bigram_accuracy


def build_token_task(
    text: str,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    vocabulary_size: int = DEFAULT_VOCAB_SIZE,
    source_url: str | None = None,
) -> SequenceTask:
    """Build a deterministic leading-space wordpiece corpus task from text."""

    pieces = _pieces(text)
    if len(pieces) <= context_length + 2:
        raise ValueError("token corpus is shorter than one context window")
    reserved = ("<pad>", "<unk>", "<bos>", "<eos>")
    common = [piece for piece, _ in Counter(pieces).most_common(vocabulary_size - len(reserved))]
    vocabulary = reserved + tuple(common)
    token_by_piece = {piece: index for index, piece in enumerate(vocabulary)}
    unknown = token_by_piece["<unk>"]
    encoded = torch.tensor(
        [token_by_piece.get(piece, unknown) for piece in pieces], dtype=torch.long
    )
    split = max(context_length + 2, int(encoded.numel() * 0.90))
    split = min(split, encoded.numel() - context_length - 2)
    train, validation = encoded[:split], encoded[split:]
    unigram_accuracy, bigram_accuracy = _next_token_baselines(
        train, validation, len(vocabulary)
    )

    def encode(value: str) -> list[int]:
        return [token_by_piece.get(piece, unknown) for piece in _pieces(value)]

    def decode(tokens: list[int]) -> str:
        return "".join(
            vocabulary[token] if 0 <= token < len(vocabulary) and token >= len(reserved)
            else ("�" if token == unknown else "")
            for token in tokens
        )

    return SequenceTask(
        key="tiny_stories",
        title="Token Cellular Language Organism",
        description=(
            "Predict wordpieces from TinyStories while tokens enter and leave as "
            "distributed population codes; no vocabulary item owns a boundary neuron."
        ),
        vocabulary=vocabulary,
        sequence_length=context_length,
        generator=_chunk_generator(train, context_length),
        evaluation_generator=_chunk_generator(validation, context_length),
        encode=encode,
        decode=decode,
        dataset_name="TinyStories V2 GPT-4 validation subset",
        dataset_characters=len(text),
        dataset_tokens=len(pieces),
        tokenizer_name=f"leading-space wordpieces · {len(vocabulary):,} tokens",
        source_url=source_url,
        unigram_baseline_accuracy=unigram_accuracy,
        bigram_baseline_accuracy=bigram_accuracy,
        training_stream=train,
        evaluation_stream=validation,
    )


@lru_cache(maxsize=4)
def load_tiny_stories_task(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    vocabulary_size: int = DEFAULT_VOCAB_SIZE,
    cache_path: str = "data/tinystories/TinyStoriesV2-GPT4-valid.txt",
) -> SequenceTask:
    """Download the bounded 22.5 MB validation corpus once and tokenize it locally."""

    return build_token_task(
        _download_text(Path(cache_path)),
        context_length=context_length,
        vocabulary_size=vocabulary_size,
        source_url=TINY_STORIES_URL,
    )


__all__ = [
    "DEFAULT_CONTEXT_LENGTH", "DEFAULT_VOCAB_SIZE", "TINY_STORIES_URL",
    "build_token_task", "load_tiny_stories_task",
]
