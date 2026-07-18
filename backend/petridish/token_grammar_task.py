"""Balanced autoregressive grammar for the distributed cellular field."""

from __future__ import annotations

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


STATE_COUNT = 32
DIGIT_COUNT = 4
DIGIT_OFFSET = 2


def _sample_rows(batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """Draw balanced 32-state cycles without restricting small GPU batches."""

    cycles = [
        torch.randperm(STATE_COUNT, generator=generator)
        for _ in range((batch_size + STATE_COUNT - 1) // STATE_COUNT)
    ]
    return torch.cat(cycles)[:batch_size]


def token_grammar_task() -> SequenceTask:
    """Return a second-order, four-symbol next-token prediction language."""

    vocabulary = (
        "rule-0", "rule-1", "digit-0", "digit-1", "digit-2", "digit-3",
    )

    def generate(batch_size: int, generator: torch.Generator) -> SequenceBatch:
        rows = _sample_rows(batch_size, generator)
        rule = rows // 16
        first = (rows % 16) // 4
        second = rows % 4
        digits = [first, second]
        for _ in range(7):
            digits.append((digits[-2] + digits[-1] + rule) % DIGIT_COUNT)
        stream = torch.stack(digits, dim=1)
        tokens = torch.cat((rule.unsqueeze(1), stream[:, :8] + DIGIT_OFFSET), dim=1)
        targets = torch.full_like(tokens, -100)
        targets[:, 2:] = stream[:, 2:9] + DIGIT_OFFSET
        return SequenceBatch(tokens, targets, targets >= 0)

    return SequenceTask(
        key="tiny_stories",
        title="Autoregressive cellular grammar",
        description=(
            "Infer a persistent rule and predict each next symbol from the two "
            "preceding symbols through the local cellular field."
        ),
        vocabulary=vocabulary,
        sequence_length=9,
        generator=generate,
        evaluation_generator=generate,
        dataset_name="balanced second-order modular grammar",
        dataset_tokens=STATE_COUNT * 9,
        tokenizer_name="two rules · four symbols · exact next-token targets",
    )


__all__ = ["token_grammar_task"]
