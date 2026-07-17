"""Broadcast addressing and persistent episode graph state for MNIST cells."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as functional


@dataclass(slots=True)
class EpisodeGraph:
    """Sparse directed graph assembled independently for every input episode."""

    destination: torch.Tensor
    weight: torch.Tensor
    strength: torch.Tensor
    age: torch.Tensor
    utility: torch.Tensor


@dataclass(slots=True)
class RoutingUpdate:
    """One broadcast match, including structural changes for visualization."""

    graph: EpisodeGraph
    axon_request: torch.Tensor
    receptor_request: torch.Tensor
    replaced: torch.Tensor
    old_destination: torch.Tensor


class BroadcastRouter(nn.Module):
    """Turn cell-advertised keys and receptors into a sparse persistent graph."""

    def __init__(
        self,
        hidden_channels: int,
        route_channels: int,
        edge_slots: int,
        coordinates: torch.Tensor,
        *,
        distance_cost: float,
        persistence_bonus: float,
        broadcast_temperature: float,
        activity_bias: float,
    ) -> None:
        super().__init__()
        self.edge_slots = edge_slots
        self.distance_cost = distance_cost
        self.persistence_bonus = persistence_bonus
        self.broadcast_temperature = broadcast_temperature
        self.activity_bias = activity_bias
        self.key = nn.Linear(hidden_channels, route_channels)
        self.query = nn.Linear(hidden_channels, route_channels)
        self.value = nn.Linear(hidden_channels, hidden_channels)
        self.axon_head = nn.Linear(hidden_channels, 1)
        self.receptor_head = nn.Linear(hidden_channels, 1)
        self.source_weight = nn.Linear(hidden_channels, 1)
        self.target_weight = nn.Linear(hidden_channels, 1)
        distance = torch.cdist(coordinates, coordinates) / math.sqrt(8.0)
        self.register_buffer("distance", distance)

    def empty(self, batch_size: int, cell_count: int, device: torch.device) -> EpisodeGraph:
        shape = (batch_size, cell_count, self.edge_slots)
        return EpisodeGraph(
            destination=torch.full(shape, -1, dtype=torch.long, device=device),
            weight=torch.zeros(shape, device=device),
            strength=torch.zeros(shape, device=device),
            age=torch.zeros(shape, dtype=torch.long, device=device),
            utility=torch.zeros(shape, device=device),
        )

    def signals(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.axon_head(hidden).squeeze(-1)), torch.sigmoid(
            self.receptor_head(hidden).squeeze(-1)
        )

    def _scores(
        self, hidden: torch.Tensor, lesion_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, cell_count, _ = hidden.shape
        keys = functional.normalize(self.key(hidden), dim=-1, eps=1e-6)
        queries = functional.normalize(self.query(hidden), dim=-1, eps=1e-6)
        axon_request, receptor_request = self.signals(hidden)
        scores = torch.matmul(keys, queries.transpose(1, 2))
        scores = scores + 0.55 * axon_request.unsqueeze(2) + 0.55 * receptor_request.unsqueeze(1)
        salience = hidden.square().mean(dim=2).sqrt()
        salience = (salience - salience.mean(dim=1, keepdim=True)) / salience.std(
            dim=1, keepdim=True
        ).clamp_min(0.1)
        scores = scores + self.activity_bias * salience.unsqueeze(2)
        scores = scores - self.distance_cost * self.distance.unsqueeze(0)
        alive = lesion_mask > 0.5
        valid = alive.view(1, cell_count, 1) & alive.view(1, 1, cell_count)
        scores = scores.masked_fill(~valid, -1e4)
        diagonal = torch.eye(cell_count, dtype=torch.bool, device=hidden.device).unsqueeze(0)
        return scores.masked_fill(diagonal, -1e4), axon_request, receptor_request

    def broadcast(
        self, hidden: torch.Tensor, lesion_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deliver a differentiable all-cell broadcast before endpoints harden."""

        scores, axon_request, receptor_request = self._scores(hidden, lesion_mask)
        attention = torch.softmax(scores / self.broadcast_temperature, dim=1)
        message = torch.einsum("bst,bsh->bth", attention, self.value(hidden))
        message = message * lesion_mask.view(1, -1, 1)
        return message, axon_request, receptor_request

    def forward(
        self,
        hidden: torch.Tensor,
        graph: EpisodeGraph,
        lesion_mask: torch.Tensor,
    ) -> RoutingUpdate:
        batch_size, cell_count, _ = hidden.shape
        scores, axon_request, receptor_request = self._scores(hidden, lesion_mask)
        alive = lesion_mask > 0.5

        proposed_score, proposed_destination = torch.topk(scores, self.edge_slots, dim=2)
        old_destination = graph.destination
        has_old = old_destination >= 0
        safe_old = old_destination.clamp_min(0)
        old_score = scores.gather(2, safe_old).masked_fill(~has_old, -1e4)
        keep_old = has_old & (old_score + self.persistence_bonus >= proposed_score)
        destination = torch.where(keep_old, old_destination, proposed_destination)
        selected_score = scores.gather(2, destination.clamp_min(0))

        source_weight = self.source_weight(hidden).expand(-1, -1, self.edge_slots)
        target_weight = self.target_weight(hidden).squeeze(-1).gather(1, destination.reshape(batch_size, -1))
        target_weight = target_weight.reshape(batch_size, cell_count, self.edge_slots)
        weight = torch.tanh(source_weight + target_weight)
        strength = torch.sigmoid(selected_score - 0.7)
        edge_alive = alive.view(1, cell_count, 1) & alive[destination]
        strength = strength * edge_alive
        weight = weight * edge_alive

        same = has_old & (destination == old_destination)
        age = torch.where(same, graph.age + 1, torch.zeros_like(graph.age))
        instantaneous_utility = strength * weight.abs()
        utility = torch.where(
            same,
            0.82 * graph.utility + 0.18 * instantaneous_utility,
            instantaneous_utility,
        )
        return RoutingUpdate(
            graph=EpisodeGraph(destination, weight, strength, age, utility),
            axon_request=axon_request,
            receptor_request=receptor_request,
            replaced=~same,
            old_destination=old_destination,
        )

    def messages(self, hidden: torch.Tensor, graph: EpisodeGraph) -> torch.Tensor:
        """Deliver current source state along the assembled directed graph."""

        batch_size, cell_count, hidden_channels = hidden.shape
        destination = graph.destination.clamp_min(0)
        message = hidden.unsqueeze(2) * (graph.weight * graph.strength).unsqueeze(3)
        incoming = torch.zeros_like(hidden)
        scatter_index = destination.reshape(batch_size, -1, 1).expand(-1, -1, hidden_channels)
        incoming.scatter_add_(1, scatter_index, message.reshape(batch_size, -1, hidden_channels))
        normalizer = torch.zeros(batch_size, cell_count, device=hidden.device)
        normalizer.scatter_add_(1, destination.reshape(batch_size, -1), graph.strength.reshape(batch_size, -1))
        return incoming / normalizer.clamp_min(1.0).unsqueeze(2)
