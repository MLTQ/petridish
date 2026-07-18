"""Context-settling token stream for a physically extended cellular substrate."""

from __future__ import annotations

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


def token_settling_task() -> SequenceTask:
    """Return four same-clock predictions after two unsupervised context clocks."""

    vocabulary = (
        "copy", "invert", "clock", "bit-0", "bit-1", "target-0", "target-1",
    )
    patterns = torch.tensor(
        (
            (0, 0, 0, 0),
            (1, 1, 1, 1),
            (0, 1, 0, 1),
            (1, 0, 1, 0),
        ),
        dtype=torch.long,
    )

    def generate(batch_size: int, _generator: torch.Generator) -> SequenceBatch:
        rows = torch.arange(batch_size, dtype=torch.long) % 8
        rule = rows // 4
        bits = patterns[rows % 4]
        clocks = torch.full((batch_size, 2), 2, dtype=torch.long)
        tokens = torch.cat((rule.unsqueeze(1), clocks, bits + 3), dim=1)
        targets = torch.full_like(tokens, -100)
        targets[:, 3:] = 5 + (bits ^ rule.unsqueeze(1))
        return SequenceBatch(tokens, targets, targets >= 0)

    return SequenceTask(
        key="tiny_stories",
        title="Distributed context-settling stream",
        description=(
            "Allow one copy/invert rule two unsupervised token clocks to establish "
            "a field context, then predict four transformed input bits immediately."
        ),
        vocabulary=vocabulary,
        sequence_length=7,
        generator=generate,
        evaluation_generator=generate,
        dataset_name="deterministic balanced two-clock context settling",
        dataset_tokens=56,
        tokenizer_name="two rules · clock · two bits · two targets",
    )


__all__ = ["token_settling_task"]
