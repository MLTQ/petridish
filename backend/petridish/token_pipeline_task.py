"""Latency-aligned contextual token stream for a physically extended substrate."""

from __future__ import annotations

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


def token_pipeline_task() -> SequenceTask:
    """Return four contextual predictions scored two token clocks after input."""

    vocabulary = (
        "copy", "invert", "bit-0", "bit-1", "clock", "target-0", "target-1",
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
        clocks = torch.full((batch_size, 2), 4, dtype=torch.long)
        tokens = torch.cat((rule.unsqueeze(1), bits + 2, clocks), dim=1)
        targets = torch.full_like(tokens, -100)
        targets[:, 3:] = 5 + (bits ^ rule.unsqueeze(1))
        return SequenceBatch(tokens, targets, targets >= 0)

    return SequenceTask(
        key="tiny_stories",
        title="Distributed latency-aligned context pipeline",
        description=(
            "Retain one copy/invert rule and emit four transformed bit tokens two "
            "token clocks after their distributed inputs enter the field."
        ),
        vocabulary=vocabulary,
        sequence_length=7,
        generator=generate,
        evaluation_generator=generate,
        dataset_name="deterministic balanced two-clock context pipeline",
        dataset_tokens=56,
        tokenizer_name="two rules · two bits · clock · two targets",
    )


__all__ = ["token_pipeline_task"]
