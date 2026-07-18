"""Repeated contextual token predictions on the distributed cellular substrate."""

from __future__ import annotations

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


def token_stream_task() -> SequenceTask:
    """Return a stream whose retained rule transforms four later input bits."""

    vocabulary = ("copy", "invert", "bit-0", "bit-1", "target-0", "target-1")
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
        tokens = torch.cat((rule.unsqueeze(1), bits + 2), dim=1)
        targets = torch.full_like(tokens, -100)
        targets[:, 1:] = 4 + (bits ^ rule.unsqueeze(1))
        return SequenceBatch(tokens, targets, targets >= 0)

    return SequenceTask(
        key="tiny_stories",
        title="Distributed persistent context stream",
        description=(
            "Retain one copy/invert rule and apply it to four later distributed "
            "bit tokens, producing a supervised token after every query."
        ),
        vocabulary=vocabulary,
        sequence_length=5,
        generator=generate,
        evaluation_generator=generate,
        dataset_name="deterministic balanced persistent-context stream",
        dataset_tokens=40,
        tokenizer_name="two rules · two bit inputs · two target tokens",
    )


__all__ = ["token_stream_task"]
