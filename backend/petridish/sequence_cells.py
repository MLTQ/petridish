"""Shared recurrent cell rules for homogeneous sequence-organism controls."""

from __future__ import annotations

import math

import torch
from torch import nn


CELL_ARCHITECTURES = ("gru", "lstm", "esn", "transformer")


class SequenceCellRule(nn.Module):
    """Apply one selected shared rule while neurons retain private state."""

    def __init__(self, architecture: str, hidden_channels: int) -> None:
        super().__init__()
        if architecture not in CELL_ARCHITECTURES:
            raise ValueError(f"unknown sequence cell architecture: {architecture}")
        self.architecture = architecture
        self.hidden_channels = hidden_channels
        self.memory_slots = 4
        if architecture == "gru":
            self.rule = nn.GRUCell(hidden_channels * 2, hidden_channels)
            nn.init.zeros_(self.rule.bias_ih)
            nn.init.zeros_(self.rule.bias_hh)
        elif architecture == "lstm":
            self.rule = nn.LSTMCell(hidden_channels * 2, hidden_channels)
            nn.init.zeros_(self.rule.bias_ih)
            nn.init.zeros_(self.rule.bias_hh)
            with torch.no_grad():
                self.rule.bias_ih[hidden_channels : hidden_channels * 2].fill_(1.0)
        elif architecture == "esn":
            self.input_projection = nn.Linear(hidden_channels * 2, hidden_channels)
            self.output_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
            reservoir = torch.linalg.qr(torch.randn(hidden_channels, hidden_channels)).Q
            self.register_buffer("reservoir", reservoir * 0.90)
            self.register_buffer("leak", torch.tensor(0.35))
            nn.init.eye_(self.output_projection.weight)
        else:
            self.input_projection = nn.Linear(hidden_channels * 2, hidden_channels)
            self.attention_query = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.attention_key = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.attention_value = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.memory_write = nn.Linear(hidden_channels * 3, hidden_channels)
            self.attention_output = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.attention_norm = nn.LayerNorm(hidden_channels)
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.GELU(),
                nn.Linear(hidden_channels * 2, hidden_channels),
            )
            self.output_norm = nn.LayerNorm(hidden_channels)

    def initial_memory(self, hidden: torch.Tensor) -> torch.Tensor | None:
        """Allocate architecture-private state for one sequence forward pass."""

        if self.architecture == "lstm":
            return torch.zeros_like(hidden)
        if self.architecture == "transformer":
            return hidden.new_zeros(*hidden.shape[:-1], self.memory_slots, hidden.shape[-1])
        return None

    def forward(
        self,
        rule_input: torch.Tensor,
        hidden: torch.Tensor,
        incoming: torch.Tensor,
        memory: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance all living neurons once using the selected shared rule."""

        if self.architecture == "gru":
            updated = self.rule(
                rule_input.flatten(0, 1), hidden.flatten(0, 1)
            ).reshape_as(hidden)
            return updated, None
        if self.architecture == "lstm":
            if memory is None:
                raise RuntimeError("LSTM cell memory was not initialized")
            updated, next_memory = self.rule(
                rule_input.flatten(0, 1),
                (hidden.flatten(0, 1), memory.flatten(0, 1)),
            )
            return updated.reshape_as(hidden), next_memory.reshape_as(hidden)
        if self.architecture == "esn":
            proposal = torch.tanh(
                self.input_projection(rule_input)
                + torch.nn.functional.linear(hidden, self.reservoir.to(hidden.dtype))
            )
            proposal = self.output_projection(proposal)
            leak = self.leak.to(hidden.dtype)
            return (1 - leak) * hidden + leak * proposal, None
        if memory is None:
            raise RuntimeError("transformer memory slots were not initialized")
        query = self.attention_query(hidden)
        key = self.attention_key(memory)
        value = self.attention_value(memory)
        score = torch.einsum("bnh,bnsh->bns", query, key) / math.sqrt(
            self.hidden_channels
        )
        attention = torch.softmax(score, dim=2)
        recalled = torch.einsum("bns,bnsh->bnh", attention, value)
        recalled = self.attention_output(recalled)
        proposal = self.attention_norm(
            hidden + self.input_projection(rule_input) + recalled
        )
        updated = self.output_norm(proposal + self.feed_forward(proposal))
        written = self.memory_write(torch.cat((rule_input, incoming), dim=2))
        next_memory = torch.cat((memory[:, :, 1:], written.unsqueeze(2)), dim=2)
        return updated, next_memory


__all__ = ["CELL_ARCHITECTURES", "SequenceCellRule"]
