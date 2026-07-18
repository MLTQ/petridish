"""Balanced delayed-copy task for distributed cellular token memory."""

from __future__ import annotations

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


def token_memory_task() -> SequenceTask:
    """Return a task that must retain one context bit across a fixed query token."""

    vocabulary = ("context-0", "context-1", "recall", "target-0", "target-1")

    def generate(batch_size: int, _generator: torch.Generator) -> SequenceBatch:
        context = torch.arange(batch_size, dtype=torch.long) % 2
        tokens = torch.stack((context, torch.full_like(context, 2)), dim=1)
        targets = torch.full_like(tokens, -100)
        targets[:, 1] = 3 + context
        return SequenceBatch(tokens, targets, targets >= 0)

    return SequenceTask(
        key="tiny_stories",
        title="Distributed two-token delayed copy",
        description=(
            "Retain one distributed context bit across a fixed recall token and "
            "route the corresponding target through the production output bank."
        ),
        vocabulary=vocabulary,
        sequence_length=2,
        generator=generate,
        evaluation_generator=generate,
        dataset_name="deterministic balanced delayed-copy control",
        dataset_tokens=2,
        tokenizer_name="two context bits · fixed recall · two targets",
    )


__all__ = ["token_memory_task"]
