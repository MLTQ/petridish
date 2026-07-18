"""Deterministic one-token mapping that isolates physical sensory-output credit."""

from __future__ import annotations

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


def token_routing_task(mapping_size: int = 8) -> SequenceTask:
    """Return a balanced input-to-target task with no positional shortcut."""

    if mapping_size < 2:
        raise ValueError("token routing requires at least two mappings")
    vocabulary = tuple(
        [f"input-{index}" for index in range(mapping_size)]
        + [f"target-{index}" for index in range(mapping_size)]
    )

    def generate(batch_size: int, _generator: torch.Generator) -> SequenceBatch:
        inputs = torch.arange(batch_size, dtype=torch.long) % mapping_size
        tokens = inputs.unsqueeze(1)
        targets = (inputs + mapping_size).unsqueeze(1)
        return SequenceBatch(tokens, targets, torch.ones_like(tokens, dtype=torch.bool))

    return SequenceTask(
        key="tiny_stories",
        title="One-token physical routing overfit",
        description=(
            "Map balanced sensory token codes to distinct targets after one cellular "
            "routing window; position and output bias cannot distinguish batch rows."
        ),
        vocabulary=vocabulary,
        sequence_length=1,
        generator=generate,
        evaluation_generator=generate,
        dataset_name="deterministic balanced routing control",
        dataset_tokens=mapping_size,
        tokenizer_name=f"{mapping_size} direct mappings",
    )


__all__ = ["token_routing_task"]
