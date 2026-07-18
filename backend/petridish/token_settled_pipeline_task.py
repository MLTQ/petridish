"""Settled, latency-aligned token pipeline for the distributed cellular field."""

from __future__ import annotations

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


def token_settled_pipeline_task() -> SequenceTask:
    """Return a decorrelated stream with context settling and output latency."""

    vocabulary = (
        "copy", "invert", "clock", "bit-0", "bit-1", "target-0", "target-1",
    )
    patterns = torch.tensor(
        (
            (0, 0, 0, 0),
            (0, 1, 1, 0),
            (1, 0, 0, 1),
            (1, 1, 1, 1),
        ),
        dtype=torch.long,
    )

    def generate(batch_size: int, _generator: torch.Generator) -> SequenceBatch:
        rows = torch.arange(batch_size, dtype=torch.long) % 8
        rule = rows // 4
        bits = patterns[rows % 4]
        clocks = torch.full((batch_size, 2), 2, dtype=torch.long)
        tokens = torch.cat((rule.unsqueeze(1), clocks, bits + 3, clocks), dim=1)
        targets = torch.full_like(tokens, -100)
        targets[:, 5:] = 5 + (bits ^ rule.unsqueeze(1))
        return SequenceBatch(tokens, targets, targets >= 0)

    return SequenceTask(
        key="tiny_stories",
        title="Distributed settled context pipeline",
        description=(
            "Settle one copy/invert rule for two clocks, stream four decorrelated "
            "bits, and emit their transformed targets two token clocks later."
        ),
        vocabulary=vocabulary,
        sequence_length=9,
        generator=generate,
        evaluation_generator=generate,
        dataset_name="deterministic balanced settled two-clock pipeline",
        dataset_tokens=72,
        tokenizer_name="two rules · clock · decorrelated bits · two targets",
    )


__all__ = ["token_settled_pipeline_task"]
