"""Deterministic synthetic sequence tasks that isolate memory and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """Token inputs, aligned targets, and positions included in the loss."""

    tokens: torch.Tensor
    targets: torch.Tensor
    loss_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class SequenceTask:
    """One sequence distribution and its readable vocabulary."""

    key: str
    title: str
    description: str
    vocabulary: tuple[str, ...]
    sequence_length: int
    generator: Callable[[int, torch.Generator], SequenceBatch]
    evaluation_generator: Callable[[int, torch.Generator], SequenceBatch] | None = None
    encode: Callable[[str], list[int]] | None = None
    decode: Callable[[list[int]], str] | None = None
    dataset_name: str | None = None
    dataset_characters: int = 0
    dataset_tokens: int = 0
    tokenizer_name: str | None = None
    source_url: str | None = None
    unigram_baseline_accuracy: float | None = None
    bigram_baseline_accuracy: float | None = None

    def batch(
        self, batch_size: int, generator: torch.Generator, *, evaluation: bool = False
    ) -> SequenceBatch:
        selected = self.evaluation_generator if evaluation else self.generator
        return (selected or self.generator)(batch_size, generator)


def associative_recall_batch(
    batch_size: int, generator: torch.Generator, pair_count: int = 3
) -> SequenceBatch:
    """Generate one to three bindings followed by a queried key."""

    if pair_count not in {1, 2, 3}:
        raise ValueError("associative recall supports one to three pairs")

    rows: list[list[int]] = []
    targets = torch.full((batch_size, 8), -100, dtype=torch.long)
    for row in range(batch_size):
        keys = torch.randperm(4, generator=generator)[:pair_count]
        values = torch.randperm(4, generator=generator)[:pair_count] + 4
        query_slot = int(torch.randint(0, pair_count, (), generator=generator))
        sequence: list[int] = [9] * 8
        for pair_index, (key, value) in enumerate(
            zip(keys.tolist(), values.tolist(), strict=True)
        ):
            sequence[pair_index * 2 : pair_index * 2 + 2] = (key, value)
        sequence[6:] = (8, int(keys[query_slot]))
        rows.append(sequence)
        targets[row, -1] = int(values[query_slot])
    tokens = torch.tensor(rows, dtype=torch.long)
    return SequenceBatch(tokens, targets, targets >= 0)


def tiny_language_batch(
    batch_size: int, generator: torch.Generator
) -> SequenceBatch:
    """Generate a small compositional grammar with context-dependent verbs."""

    color_bit = torch.randint(0, 2, (batch_size,), generator=generator)
    noun_bit = torch.randint(0, 2, (batch_size,), generator=generator)
    color = color_bit + 1
    noun = noun_bit + 3
    verb = (color_bit ^ noun_bit) + 5
    object_token = noun_bit + 7
    sentence = torch.stack(
        (torch.zeros(batch_size, dtype=torch.long), color, noun, verb, object_token,
         torch.full((batch_size,), 9, dtype=torch.long)),
        dim=1,
    )
    tokens = sentence[:, :-1]
    targets = sentence[:, 1:]
    return SequenceBatch(tokens, targets, torch.ones_like(targets, dtype=torch.bool))


TASKS: dict[str, SequenceTask] = {
    "associative_recall": SequenceTask(
        key="associative_recall",
        title="Associative Recall",
        description=(
            "Bind up to three K→V pairs, receive a query marker and key, then retrieve "
            "the corresponding value after a delay."
        ),
        vocabulary=("K0", "K1", "K2", "K3", "V0", "V1", "V2", "V3", "?", "·"),
        sequence_length=8,
        generator=associative_recall_batch,
    ),
    "tiny_language": SequenceTask(
        key="tiny_language",
        title="Tiny Language Model",
        description=(
            "Predict each next token in a compositional color/noun grammar; the "
            "verb depends on both earlier tokens."
        ),
        vocabulary=("<bos>", "red", "blue", "cat", "dog", "likes", "chases", "fish", "ball", "<eos>"),
        sequence_length=5,
        generator=tiny_language_batch,
    ),
}


def resolve_sequence_task(task: str | SequenceTask) -> SequenceTask:
    if isinstance(task, SequenceTask):
        return task
    if task == "tiny_shakespeare":
        from .corpus_task import load_tiny_shakespeare_task

        return load_tiny_shakespeare_task()
    if task == "tiny_stories":
        from .token_corpus_task import load_tiny_stories_task

        return load_tiny_stories_task()
    try:
        return TASKS[task]
    except KeyError as error:
        raise ValueError(f"unknown sequence task: {task}") from error


__all__ = [
    "SequenceBatch", "SequenceTask", "TASKS", "associative_recall_batch",
    "resolve_sequence_task",
]
