"""Differentiable recurrent computation over a persistent spatial connectome."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .mnist_config import MnistModelConfig
from .mnist_substrate import GraphSnapshot, SpatialSubstrate


@dataclass(slots=True)
class MnistFrame:
    """One evidence-bearing phase of an MNIST trial."""

    sites: torch.Tensor
    state: torch.Tensor
    stimulation: torch.Tensor
    load: torch.Tensor
    credit: torch.Tensor
    input_signal: torch.Tensor
    graph: GraphSnapshot
    stage: str
    step: int
    events: list[dict[str, Any]]


@dataclass(slots=True)
class MnistForward:
    """Differentiable outputs and measured local quantities from one trial."""

    logits: torch.Tensor
    trajectory_logits: torch.Tensor
    sites: torch.Tensor
    final_state: torch.Tensor
    retained_states: list[torch.Tensor]
    frames: list[MnistFrame]
    stimulation: torch.Tensor
    load: torch.Tensor
    edge_flow: torch.Tensor
    advertised_query: torch.Tensor
    advertised_key: torch.Tensor
    emission: torch.Tensor
    mean_attention_entropy: float


class CellularGraphClassifier(nn.Module):
    """Run one shared recurrent neuron rule over the currently living graph."""

    def __init__(self, config: MnistModelConfig | None = None, *, seed: int = 1) -> None:
        super().__init__()
        self.config = config or MnistModelConfig()
        cfg = self.config
        if cfg.width < 20 or cfg.height < 20:
            raise ValueError("MNIST substrate must be at least 20 by 20")
        torch.manual_seed(seed)
        self.substrate = SpatialSubstrate(cfg, seed=seed)
        self.patch_encoder = nn.Sequential(
            nn.Linear(16, cfg.hidden_channels),
            nn.LayerNorm(cfg.hidden_channels),
            nn.GELU(),
        )
        persistent_features = 2 + 3 + 4
        self.context_rule = nn.Sequential(
            nn.Linear(persistent_features, cfg.hidden_channels),
            nn.Tanh(),
        )
        self.genotype_context = nn.Linear(cfg.genotype_channels, cfg.hidden_channels)
        self.genotype_film = nn.Linear(cfg.genotype_channels, cfg.hidden_channels * 2)
        nn.init.normal_(self.genotype_context.weight, std=0.04)
        nn.init.normal_(self.genotype_film.weight, std=0.025)
        nn.init.zeros_(self.genotype_context.bias)
        nn.init.zeros_(self.genotype_film.bias)
        self.input_identity = nn.Embedding(49, cfg.hidden_channels)
        nn.init.normal_(self.input_identity.weight, std=0.12)
        self.output_identity = nn.Embedding(10, cfg.hidden_channels)
        nn.init.normal_(self.output_identity.weight, std=0.2)
        self.message_norm = nn.LayerNorm(cfg.hidden_channels, elementwise_affine=False)
        self.message_query = nn.Linear(cfg.hidden_channels, cfg.hidden_channels, bias=False)
        self.message_key = nn.Linear(cfg.hidden_channels, cfg.hidden_channels, bias=False)
        self.message_value = nn.Linear(cfg.hidden_channels, cfg.hidden_channels, bias=False)
        self.emit_gate = nn.Linear(cfg.hidden_channels, 1)
        with torch.no_grad():
            self.message_value.weight.copy_(torch.eye(cfg.hidden_channels))
            self.emit_gate.weight.zero_()
            self.emit_gate.bias.fill_(cfg.emit_gate_bias)
        self.message_gain_raw = nn.Parameter(torch.tensor(math.log(math.expm1(cfg.message_gain))))
        self.cell_rule = nn.GRUCell(cfg.hidden_channels * 2, cfg.hidden_channels)
        nn.init.zeros_(self.cell_rule.bias_ih)
        nn.init.zeros_(self.cell_rule.bias_hh)
        with torch.no_grad():
            self.cell_rule.weight_ih[:, cfg.hidden_channels :].zero_()
        self.state_norm = nn.LayerNorm(cfg.hidden_channels)
        self.output_key = nn.Linear(cfg.hidden_channels, cfg.hidden_channels, bias=False)
        self.output_value = nn.Linear(cfg.hidden_channels, cfg.hidden_channels, bias=False)
        with torch.no_grad():
            self.output_key.weight.copy_(torch.eye(cfg.hidden_channels))
            self.output_value.weight.copy_(torch.eye(cfg.hidden_channels))
        self.output_readout = nn.Linear(cfg.hidden_channels, 1)
        self.output_bank_readout = nn.Linear(cfg.hidden_channels * 10, 10)
        nn.init.normal_(self.output_bank_readout.weight, std=0.025)
        nn.init.zeros_(self.output_bank_readout.bias)
        self.class_bias = nn.Parameter(torch.zeros(10))
        self.readout_scale = nn.Parameter(torch.tensor(0.0))

    def shared_parameters(self) -> Iterable[nn.Parameter]:
        """Return learned rule parameters, excluding individually plastic synapses."""

        for name, parameter in self.named_parameters():
            if name != "substrate.synapse_weight":
                yield parameter

    def readout_parameters(self) -> Iterable[nn.Parameter]:
        """Return all output-bank parameters assigned the faster readout rate."""

        modules = (
            self.output_identity,
            self.output_key,
            self.output_value,
            self.output_readout,
            self.output_bank_readout,
        )
        for module in modules:
            yield from module.parameters()
        yield self.class_bias
        yield self.readout_scale

    def probe_parameters(self) -> Iterable[nn.Parameter]:
        """Return only the linear bank probe, keeping reservoir features frozen."""

        yield from self.output_bank_readout.parameters()
        yield self.class_bias

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        patches = images.unfold(2, 4, 4).unfold(3, 4, 4)
        return patches[:, 0].reshape(images.shape[0], 49, 16)

    def _persistent_features(self, sites: torch.Tensor, batch_size: int) -> torch.Tensor:
        substrate = self.substrate
        slow = torch.stack(
            (
                substrate.energy[sites],
                substrate.stimulation_ema[sites],
                substrate.load_ema[sites],
                substrate.neuron_utility[sites].tanh(),
            ),
            dim=1,
        )
        features = torch.cat((substrate.coordinates[sites], substrate.roles[sites], slow), dim=1)
        return features.unsqueeze(0).expand(batch_size, -1, -1)

    def _compact_graph(
        self, sites: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        substrate = self.substrate
        site_to_compact = torch.full(
            (self.config.site_count,), -1, dtype=torch.long, device=sites.device
        )
        site_to_compact[sites] = torch.arange(sites.numel(), device=sites.device)
        target_site, slot, source_site = substrate.edge_list()
        return (
            target_site,
            slot,
            source_site,
            site_to_compact[target_site],
            site_to_compact[source_site],
        )

    def forward(self, images: torch.Tensor, *, capture_trace: bool = True) -> MnistForward:
        """Classify a batch while retaining real traffic and gradient endpoints."""

        cfg = self.config
        substrate = self.substrate
        batch_size = images.shape[0]
        sites = substrate.living_sites
        site_to_compact = torch.full(
            (cfg.site_count,), -1, dtype=torch.long, device=images.device
        )
        site_to_compact[sites] = torch.arange(sites.numel(), device=images.device)
        persistent = self._persistent_features(sites, batch_size).to(images.dtype)
        genotype = substrate.genotype[sites]
        context = self.context_rule(persistent) + torch.tanh(
            self.genotype_context(genotype)
        ).unsqueeze(0)
        film_scale, film_bias = (
            0.18 * torch.tanh(self.genotype_film(genotype))
        ).chunk(2, dim=1)
        output_compact = site_to_compact[substrate.output_sites]
        output_alive = output_compact >= 0
        if output_alive.any():
            identity = self.output_identity.weight[output_alive]
            context = context.index_add(
                1,
                output_compact[output_alive],
                identity.unsqueeze(0).expand(batch_size, -1, -1),
            )

        patches = self._patchify(images)
        encoded = self.patch_encoder(patches)
        external = torch.zeros_like(context)
        input_compact = site_to_compact[substrate.input_sites]
        input_alive = input_compact >= 0
        if input_alive.any():
            input_drive = encoded + self.input_identity.weight.unsqueeze(0)
            external[:, input_compact[input_alive]] = input_drive[:, input_alive]
        hidden = external

        input_signal = torch.zeros(sites.numel(), device=images.device, dtype=images.dtype)
        if input_alive.any():
            patch_strength = patches[:, input_alive].abs().mean(dim=(0, 2))
            input_signal[input_compact[input_alive]] = patch_strength

        target_site, slot, source_site = substrate.edge_list()
        target_compact = site_to_compact[target_site]
        source_compact = site_to_compact[source_site]
        weights = substrate.synapse_weight[target_site, slot]
        edge_flow = torch.zeros_like(substrate.synapse_weight)
        stimulation = torch.zeros(sites.numel(), device=images.device)
        load = torch.zeros(sites.numel(), device=images.device)
        advertised_query = torch.zeros(
            sites.numel(), cfg.hidden_channels, device=images.device
        )
        advertised_key = torch.zeros_like(advertised_query)
        emission = torch.zeros(sites.numel(), device=images.device)
        attention_entropy = 0.0
        retained_states: list[torch.Tensor] = []
        trajectory: list[torch.Tensor] = []
        frames: list[MnistFrame] = []
        if capture_trace:
            frames.append(
                self.make_frame(
                    sites,
                    hidden[0],
                    input_signal=input_signal,
                    stage="input",
                    step=0,
                    edge_flow=edge_flow,
                )
            )

        for step in range(cfg.message_steps):
            normalized = self.message_norm(hidden)
            query = self.message_query(normalized)
            key = self.message_key(normalized)
            value = self.message_value(normalized)
            emit = torch.sigmoid(self.emit_gate(normalized))
            advertised_query += query.detach().mean(dim=0) / cfg.message_steps
            advertised_key += key.detach().mean(dim=0) / cfg.message_steps
            emission += emit.detach().mean(dim=(0, 2)) / cfg.message_steps

            compatibility = (
                query[:, target_compact] * key[:, source_compact]
            ).sum(dim=2) / (math.sqrt(cfg.hidden_channels) * cfg.attention_temperature)
            compatibility += weights.abs().clamp_min(1e-5).log().unsqueeze(0)
            attention_score = compatibility.clamp(-12, 12).exp()
            attention_denominator = torch.zeros(
                batch_size, sites.numel(), device=images.device, dtype=images.dtype
            )
            attention_denominator.index_add_(1, target_compact, attention_score)
            attention = attention_score / attention_denominator[:, target_compact].clamp_min(1e-9)
            edge_message = (
                value[:, source_compact]
                * emit[:, source_compact]
                * attention.unsqueeze(2)
                * weights.view(1, -1, 1)
                * F.softplus(self.message_gain_raw)
            )
            incoming = torch.zeros_like(hidden)
            incoming.index_add_(1, target_compact, edge_message)
            indegree = torch.zeros(sites.numel(), device=images.device, dtype=images.dtype)
            indegree.index_add_(0, target_compact, torch.ones_like(target_compact, dtype=images.dtype))
            if (indegree > 1).any():
                target_entropy = torch.zeros_like(attention_denominator)
                target_entropy.index_add_(
                    1,
                    target_compact,
                    -(attention * attention.clamp_min(1e-9).log()),
                )
                valid = indegree > 1
                normalized_entropy = target_entropy[:, valid] / indegree[valid].log().unsqueeze(0)
                attention_entropy += float(normalized_entropy.detach().mean()) / cfg.message_steps

            step_edge_flow = edge_message.detach().abs().mean(dim=(0, 2))
            step_flow_matrix = torch.zeros_like(edge_flow)
            step_flow_matrix[target_site, slot] = step_edge_flow
            edge_flow += step_flow_matrix / cfg.message_steps
            step_load = incoming.detach().abs().mean(dim=(0, 2))
            variable_edge = edge_message.detach().std(dim=0, correction=0).mean(dim=1)
            step_stimulation = torch.zeros(sites.numel(), device=images.device)
            step_stimulation.index_add_(0, target_compact, variable_edge)
            step_stimulation /= indegree.clamp_min(1).sqrt()
            step_stimulation += external.detach().std(dim=0, correction=0).mean(dim=1)
            load += step_load / cfg.message_steps
            stimulation += step_stimulation / cfg.message_steps

            rule_input = torch.cat((external, 0.1 * context), dim=2)
            updated = self.cell_rule(
                rule_input.reshape(-1, rule_input.shape[-1]),
                hidden.reshape(-1, cfg.hidden_channels),
            ).reshape(batch_size, sites.numel(), cfg.hidden_channels)
            updated = updated * (1 + film_scale.unsqueeze(0)) + film_bias.unsqueeze(0)
            hidden = self.state_norm(updated + incoming)
            if hidden.requires_grad:
                hidden.retain_grad()
                retained_states.append(hidden)
            trajectory.append(self._read_logits(hidden, site_to_compact))
            if capture_trace:
                frames.append(
                    self.make_frame(
                        sites,
                        hidden[0],
                        stimulation=step_stimulation,
                        load=step_load,
                        input_signal=input_signal,
                        stage="forward",
                        step=step + 1,
                        edge_flow=step_flow_matrix,
                    )
                )

        logits = self._read_logits(hidden, site_to_compact)
        return MnistForward(
            logits,
            torch.stack(trajectory, dim=1),
            sites,
            hidden,
            retained_states,
            frames,
            stimulation,
            load,
            edge_flow,
            advertised_query,
            advertised_key,
            emission,
            attention_entropy,
        )

    def _read_logits(self, hidden: torch.Tensor, site_to_compact: torch.Tensor) -> torch.Tensor:
        compact = site_to_compact[self.substrate.output_sites]
        output_state = torch.zeros(
            hidden.shape[0], 10, self.config.hidden_channels,
            device=hidden.device, dtype=hidden.dtype,
        )
        alive = compact >= 0
        if alive.any():
            output_state[:, alive] = hidden[:, compact[alive]]
        query = self.output_identity.weight
        attention = torch.softmax(
            torch.einsum("kh,bnh->bkn", query, self.output_key(output_state))
            / math.sqrt(self.config.hidden_channels),
            dim=2,
        )
        pooled = torch.einsum(
            "bkn,bnh->bkh", attention, self.output_value(output_state)
        ) + output_state
        shared = self.output_readout(pooled).squeeze(-1)
        identity_readout = (
            pooled * query.unsqueeze(0)
        ).sum(dim=2) / math.sqrt(self.config.hidden_channels)
        bank_readout = self.output_bank_readout(output_state.flatten(1))
        return self.readout_scale * (shared + identity_readout) + bank_readout + self.class_bias

    def make_frame(
        self,
        sites: torch.Tensor,
        state: torch.Tensor,
        *,
        stimulation: torch.Tensor | None = None,
        load: torch.Tensor | None = None,
        credit: torch.Tensor | None = None,
        input_signal: torch.Tensor | None = None,
        stage: str,
        step: int,
        edge_flow: torch.Tensor | None = None,
        edge_credit: torch.Tensor | None = None,
        events: list[dict[str, Any]] | None = None,
    ) -> MnistFrame:
        """Capture only measured values; no presentation-only signal is synthesized."""

        zero = torch.zeros(sites.numel(), device=state.device)
        graph = self.substrate.graph_snapshot()
        if edge_flow is not None:
            graph.flow = edge_flow.detach().clone()
        if edge_credit is not None:
            graph.credit = edge_credit.detach().clone()
        return MnistFrame(
            sites.detach().clone(),
            state.detach().clone(),
            (stimulation if stimulation is not None else zero).detach().clone(),
            (load if load is not None else zero).detach().clone(),
            (credit if credit is not None else zero).detach().clone(),
            (input_signal if input_signal is not None else zero).detach().clone(),
            graph,
            stage,
            step,
            list(events or []),
        )

    def regularization(self) -> torch.Tensor:
        active = self.substrate.active_edge_mask
        if not active.any():
            return self.substrate.synapse_weight.sum() * 0
        return self.config.weight_decay * self.substrate.synapse_weight[active].square().mean()

    @torch.no_grad()
    def lesion(self, x: float, y: float, radius: float) -> int:
        return self.substrate.lesion(x, y, radius)


__all__ = [
    "CellularGraphClassifier",
    "MnistForward",
    "MnistFrame",
    "MnistModelConfig",
]
