"""Tensor container for all mutable scientific state."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PetriDishState:
    """Fixed-shape cell and directed-edge tensors for one simulation."""

    cells: torch.Tensor
    edge_destination: torch.Tensor
    edge_weight: torch.Tensor
    edge_gate: torch.Tensor
    edge_eligibility: torch.Tensor
    edge_age: torch.Tensor
    edge_utility: torch.Tensor
    tick: int = 0

    def clone(self) -> "PetriDishState":
        """Return a detached deep copy suitable for deterministic comparisons."""

        return PetriDishState(
            cells=self.cells.clone(),
            edge_destination=self.edge_destination.clone(),
            edge_weight=self.edge_weight.clone(),
            edge_gate=self.edge_gate.clone(),
            edge_eligibility=self.edge_eligibility.clone(),
            edge_age=self.edge_age.clone(),
            edge_utility=self.edge_utility.clone(),
            tick=self.tick,
        )
