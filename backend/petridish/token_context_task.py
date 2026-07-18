"""Balanced two-token XOR task for distributed cellular context memory."""

from __future__ import annotations

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


def token_context_task() -> SequenceTask:
    """Return a task whose final target depends on both context and query tokens."""

    vocabulary = (
        "context-0", "context-1", "query-0", "query-1", "target-0", "target-1",
    )

    def generate(batch_size: int, _generator: torch.Generator) -> SequenceBatch:
        rows = torch.arange(batch_size, dtype=torch.long) % 4
        context = rows // 2
        query = rows % 2
        tokens = torch.stack((context, query + 2), dim=1)
        targets = torch.full_like(tokens, -100)
        targets[:, 1] = 4 + (context ^ query)
        mask = targets >= 0
        return SequenceBatch(tokens, targets, mask)

    return SequenceTask(
        key="tiny_stories",
        title="Distributed two-token context XOR",
        description=(
            "Retain one context bit, combine it with a later query bit, and route "
            "the XOR target through the production distributed token interfaces."
        ),
        vocabulary=vocabulary,
        sequence_length=2,
        generator=generate,
        evaluation_generator=generate,
        dataset_name="deterministic balanced context control",
        dataset_tokens=4,
        tokenizer_name="two context bits · two query bits · two targets",
    )


__all__ = ["token_context_task"]
