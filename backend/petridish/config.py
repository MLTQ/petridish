"""Configuration and device selection for a Petri Dish experiment."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Fixed experiment dimensions and interpretable rule parameters."""

    width: int = 32
    height: int = 32
    channels: int = 16
    edge_slots: int = 4
    growth_interval: int = 12
    candidate_targets: int = 8
    eligibility_decay: float = 0.94
    utility_decay: float = 0.985
    weight_learning_rate: float = 0.006
    weight_decay: float = 0.0015
    max_weight: float = 2.0
    initial_active_slots: int = 2
    initial_weight_scale: float = 0.24
    prune_age: int = 180
    prune_utility: float = 0.018
    cue_ticks: int = 18
    delay_ticks: int = 24
    response_ticks: int = 12
    rest_ticks: int = 10

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    @property
    def trial_ticks(self) -> int:
        return self.cue_ticks + self.delay_ticks + self.response_ticks + self.rest_ticks


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an accelerator preference without making it a hard requirement."""

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
