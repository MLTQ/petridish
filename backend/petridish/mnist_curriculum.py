"""Deterministic class-balanced overfit curriculum for MNIST experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Subset


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """One fixed training subset and the accuracy required to leave it."""

    examples: int
    target_accuracy: float | None
    dataset: Dataset


def build_curriculum(dataset: Dataset, *, seed: int) -> list[CurriculumStage]:
    """Build balanced 20, 256, 1k, and full-dataset stages when possible."""

    total = len(dataset)
    requested = [(20, 0.98), (256, 0.95), (1_000, 0.90), (total, None)]
    sizes: list[tuple[int, float | None]] = []
    for count, target in requested:
        bounded = min(total, count)
        if sizes and sizes[-1][0] == bounded:
            sizes[-1] = (bounded, None if bounded == total else target)
        else:
            sizes.append((bounded, None if bounded == total else target))

    labels = _labels(dataset)
    ordering = _balanced_order(labels, seed=seed)
    return [
        CurriculumStage(count, target, Subset(dataset, ordering[:count]))
        for count, target in sizes
    ]


def _labels(dataset: Dataset) -> torch.Tensor:
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return torch.as_tensor(targets, dtype=torch.long)
    tensors = getattr(dataset, "tensors", None)
    if tensors is not None and len(tensors) >= 2:
        return torch.as_tensor(tensors[1], dtype=torch.long)
    return torch.tensor([int(dataset[index][1]) for index in range(len(dataset))])


def _balanced_order(labels: torch.Tensor, *, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    buckets: list[list[int]] = []
    for label in labels.unique(sorted=True).tolist():
        indices = (labels == label).nonzero(as_tuple=False).squeeze(1)
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        buckets.append(shuffled.tolist())

    ordering: list[int] = []
    depth = 0
    while len(ordering) < labels.numel():
        added = False
        for bucket in buckets:
            if depth < len(bucket):
                ordering.append(bucket[depth])
                added = True
        if not added:
            break
        depth += 1
    return ordering


__all__ = ["CurriculumStage", "build_curriculum"]
