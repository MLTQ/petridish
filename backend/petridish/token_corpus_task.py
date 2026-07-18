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
TOKENIZER_PROFILES = ("wordpiece", "byte")
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
) -> tuple[float, float, float, float]:
    """Measure exact accuracy and smoothed validation loss for n-gram baselines."""

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
    unigram_probability = (
        (unigram_counts.double() + 1)
        / (unigram_counts.sum().double() + vocabulary_size)
    )
    unigram_loss = float(-unigram_probability[targets].log().mean())
    bigram_probability = (
        (bigram_counts.double() + 1)
        / (bigram_counts.sum(dim=1, keepdim=True).double() + vocabulary_size)
    )
    bigram_loss = float(
        -bigram_probability[validation[:-1], targets].log().mean()
    )
    return unigram_accuracy, bigram_accuracy, unigram_loss, bigram_loss


def build_token_task(
    text: str,
    *,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    vocabulary_size: int = DEFAULT_VOCAB_SIZE,
    tokenizer_profile: str = "wordpiece",
    training_shard_tokens: int | None = None,
    source_url: str | None = None,
) -> SequenceTask:
    """Build a deterministic wordpiece or byte-complete corpus task from text."""

    if tokenizer_profile not in TOKENIZER_PROFILES:
        raise ValueError(f"unknown token corpus tokenizer: {tokenizer_profile}")
    reserved: tuple[str, ...]
    unknown: int | None
    if tokenizer_profile == "byte":
        if vocabulary_size != 256:
            raise ValueError("byte tokenization requires a 256-token vocabulary")
        raw = text.encode("utf-8")
        pieces_or_bytes: list[str] | list[int] = list(raw)
        vocabulary = tuple(
            chr(value) if 32 <= value <= 126 else f"<0x{value:02x}>"
            for value in range(256)
        )
        encoded = torch.tensor(pieces_or_bytes, dtype=torch.long)
        reserved = ()
        unknown = None

        def encode(value: str) -> list[int]:
            return list(value.encode("utf-8"))

        def decode(tokens: list[int]) -> str:
            return bytes(token for token in tokens if 0 <= token < 256).decode(
                "utf-8", errors="replace"
            )
    else:
        pieces_or_bytes = _pieces(text)
        reserved = ("<pad>", "<unk>", "<bos>", "<eos>")
        common = [
            piece
            for piece, _ in Counter(pieces_or_bytes).most_common(
                vocabulary_size - len(reserved)
            )
        ]
        vocabulary = reserved + tuple(common)
        token_by_piece = {piece: index for index, piece in enumerate(vocabulary)}
        unknown = token_by_piece["<unk>"]
        encoded = torch.tensor(
            [token_by_piece.get(piece, unknown) for piece in pieces_or_bytes],
            dtype=torch.long,
        )

        def encode(value: str) -> list[int]:
            return [token_by_piece.get(piece, unknown) for piece in _pieces(value)]

        def decode(tokens: list[int]) -> str:
            return "".join(
                vocabulary[token]
                if 0 <= token < len(vocabulary) and token >= len(reserved)
                else ("�" if token == unknown else "")
                for token in tokens
            )

    if encoded.numel() <= context_length + 2:
        raise ValueError("token corpus is shorter than one context window")
    split = max(context_length + 2, int(encoded.numel() * 0.90))
    split = min(split, encoded.numel() - context_length - 2)
    full_train, validation = encoded[:split], encoded[split:]
    if training_shard_tokens == 0:
        training_shard_tokens = None
    if training_shard_tokens is not None:
        if training_shard_tokens <= context_length + 1:
            raise ValueError("training shard must exceed one complete context window")
        if training_shard_tokens > full_train.numel():
            raise ValueError("training shard exceeds the available training stream")
        train = full_train[:training_shard_tokens]
    else:
        train = full_train
    unigram_accuracy, bigram_accuracy, unigram_loss, bigram_loss = _next_token_baselines(
        train, validation, len(vocabulary)
    )

    return SequenceTask(
        key="tiny_stories",
        title="Token Cellular Language Organism",
        description=(
            "Predict TinyStories tokens while they enter and leave as "
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
        dataset_tokens=int(encoded.numel()),
        tokenizer_name=(
            "UTF-8 bytes · 256 tokens · no unknown class"
            if tokenizer_profile == "byte"
            else f"leading-space wordpieces · {len(vocabulary):,} tokens"
        ),
        source_url=source_url,
        unigram_baseline_accuracy=unigram_accuracy,
        bigram_baseline_accuracy=bigram_accuracy,
        unigram_baseline_loss=unigram_loss,
        bigram_baseline_loss=bigram_loss,
        training_stream=train,
        evaluation_stream=validation,
        tokenizer_profile=tokenizer_profile,
        special_token_ids=tuple(range(len(reserved))),
        unknown_token_id=unknown,
        validation_unknown_token_rate=(
            float((validation == unknown).float().mean())
            if unknown is not None else 0.0
        ),
        training_stream_tokens=int(train.numel()),
        full_training_stream_tokens=int(full_train.numel()),
        training_shard_tokens=training_shard_tokens,
    )


@lru_cache(maxsize=16)
def load_tiny_stories_task(
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    vocabulary_size: int = DEFAULT_VOCAB_SIZE,
    cache_path: str = "data/tinystories/TinyStoriesV2-GPT4-valid.txt",
    *,
    tokenizer_profile: str = "wordpiece",
    training_shard_tokens: int | None = None,
) -> SequenceTask:
    """Download the bounded 22.5 MB validation corpus once and tokenize it locally."""

    return build_token_task(
        _download_text(Path(cache_path)),
        context_length=context_length,
        vocabulary_size=vocabulary_size,
        tokenizer_profile=tokenizer_profile,
        training_shard_tokens=training_shard_tokens,
        source_url=TINY_STORIES_URL,
    )


__all__ = [
    "DEFAULT_CONTEXT_LENGTH", "DEFAULT_VOCAB_SIZE", "TINY_STORIES_URL",
    "TOKENIZER_PROFILES",
    "build_token_task", "load_tiny_stories_task",
]
