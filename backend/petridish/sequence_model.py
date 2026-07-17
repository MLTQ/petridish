"""Token-stream computation in a persistent spatial neural graph."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .graph_layout import GraphLayout, resolve_layout
from .mnist_config import MnistModelConfig
from .mnist_substrate import GraphSnapshot, SpatialSubstrate
from .sequence_cells import SequenceCellRule


@dataclass(slots=True)
class SequenceFrame:
    """Measured graph state after consuming one token."""

    sites: torch.Tensor
    state: torch.Tensor
    stimulation: torch.Tensor
    load: torch.Tensor
    credit: torch.Tensor
    input_signal: torch.Tensor
    graph: GraphSnapshot
    stage: str
    step: int
    token_position: int
    events: list[dict[str, Any]]


@dataclass(slots=True)
class SequenceForward:
    """Differentiable sequence predictions and local measurements."""

    logits: torch.Tensor
    sites: torch.Tensor
    final_state: torch.Tensor
    retained_states: list[torch.Tensor]
    frames: list[SequenceFrame]
    stimulation: torch.Tensor
    load: torch.Tensor
    edge_flow: torch.Tensor
    advertised_query: torch.Tensor
    advertised_key: torch.Tensor
    emission: torch.Tensor
    mean_attention_entropy: float


SequenceFrameCallback = Callable[[SequenceFrame, torch.Tensor], None]


class CellularSequenceModel(nn.Module):
    """Apply one genotype-modulated recurrent rule at every living neuron."""

    def __init__(
        self,
        config: MnistModelConfig,
        *,
        layout: str | GraphLayout,
        vocab_size: int = 10,
        max_length: int = 12,
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.layout = resolve_layout(layout)
        self.vocab_size = vocab_size
        if self.layout.input_count != vocab_size or self.layout.output_count != vocab_size:
            raise ValueError("sequence layouts require one input and output port per token")
        torch.manual_seed(seed)
        self.substrate = SpatialSubstrate(config, layout=self.layout, seed=seed)
        hidden = config.hidden_channels
        persistent_features = 2 + 3 + 4
        self.context_rule = nn.Sequential(nn.Linear(persistent_features, hidden), nn.Tanh())
        self.genotype_context = nn.Linear(config.genotype_channels, hidden)
        self.genotype_film = nn.Linear(config.genotype_channels, hidden * 2)
        self.token_identity = nn.Embedding(vocab_size, hidden)
        self.position_identity = nn.Embedding(max_length, hidden)
        self.output_identity = nn.Embedding(vocab_size, hidden)
        self.message_norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.message_query = nn.Linear(hidden, hidden, bias=False)
        self.message_key = nn.Linear(hidden, hidden, bias=False)
        self.message_value = nn.Linear(hidden, hidden, bias=False)
        self.emit_gate = nn.Linear(hidden, 1)
        self.broadcast_key = nn.Linear(hidden, hidden, bias=False)
        self.broadcast_query = nn.Linear(hidden, hidden, bias=False)
        self.broadcast_value = nn.Linear(hidden, hidden, bias=False)
        self.broadcast_slot_keys = nn.Parameter(
            torch.randn(config.broadcast_slots, hidden) / math.sqrt(hidden)
        )
        self.broadcast_gain = nn.Parameter(torch.tensor(config.broadcast_gain))
        self.fast_key = nn.Linear(hidden, hidden, bias=False)
        self.fast_query = nn.Linear(hidden, hidden, bias=False)
        self.fast_value = nn.Linear(hidden, hidden, bias=False)
        self.fast_write_gate = nn.Linear(hidden, 1)
        self.fast_weight_gain = nn.Parameter(torch.tensor(config.fast_weight_gain))
        self.binding_owner_address: nn.Linear | None = None
        self.binding_token_key: nn.Linear | None = None
        self.binding_value: nn.Linear | None = None
        self.binding_read: nn.Linear | None = None
        self.binding_gain: nn.Parameter | None = None
        if config.binding_memory_gain > 0:
            self.binding_owner_address = nn.Linear(
                config.genotype_channels, hidden, bias=False
            )
            self.binding_token_key = nn.Linear(hidden, hidden, bias=False)
            self.binding_value = nn.Linear(hidden, hidden, bias=False)
            self.binding_read = nn.Linear(hidden, hidden, bias=False)
            self.binding_gain = nn.Parameter(torch.tensor(config.binding_memory_gain))
        self.cell_rule = SequenceCellRule(config.cell_architecture, hidden)
        self.state_norm = nn.LayerNorm(hidden)
        self.output_bank_readout = nn.Linear(hidden * vocab_size, vocab_size)
        self.class_bias = nn.Parameter(torch.zeros(vocab_size))
        self.message_gain_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(config.message_gain)))
        )
        nn.init.normal_(self.token_identity.weight, std=0.18)
        nn.init.normal_(self.position_identity.weight, std=0.08)
        nn.init.normal_(self.output_identity.weight, std=0.18)
        nn.init.normal_(self.output_bank_readout.weight, std=0.025)
        nn.init.zeros_(self.output_bank_readout.bias)
        with torch.no_grad():
            self.message_value.weight.copy_(torch.eye(hidden))
            self.emit_gate.weight.zero_()
            self.emit_gate.bias.fill_(config.emit_gate_bias)

    def shared_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if name != "substrate.synapse_weight":
                yield parameter

    def _persistent_features(self, sites: torch.Tensor, batch: int) -> torch.Tensor:
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
        values = torch.cat((substrate.coordinates[sites], substrate.roles[sites], slow), dim=1)
        return values.unsqueeze(0).expand(batch, -1, -1)

    def _read_logits(
        self,
        hidden: torch.Tensor,
        output_compact: torch.Tensor,
        output_alive: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros(
            hidden.shape[0], self.vocab_size, self.config.hidden_channels,
            device=hidden.device, dtype=hidden.dtype,
        )
        if output_alive.any():
            output[:, output_alive] = hidden[:, output_compact[output_alive]]
        return self.output_bank_readout(output.flatten(1)) + self.class_bias

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        capture_trace: bool = True,
        frame_callback: SequenceFrameCallback | None = None,
    ) -> SequenceForward:
        """Consume discrete tokens one at a time without clearing neuron state."""

        if frame_callback is not None and not capture_trace:
            raise ValueError("frame callbacks require trace capture")

        cfg = self.config
        substrate = self.substrate
        batch, length = tokens.shape
        sites = substrate.living_sites
        compact = torch.full((cfg.site_count,), -1, dtype=torch.long, device=tokens.device)
        compact[sites] = torch.arange(sites.numel(), device=tokens.device)
        genotype = substrate.genotype[sites]
        context = self.context_rule(self._persistent_features(sites, batch))
        context = context + torch.tanh(self.genotype_context(genotype)).unsqueeze(0)
        film_scale, film_bias = (0.18 * torch.tanh(self.genotype_film(genotype))).chunk(2, dim=1)
        output_compact = compact[substrate.output_sites]
        output_alive = output_compact >= 0
        if output_alive.any():
            context = context.index_add(
                1,
                output_compact[output_alive],
                self.output_identity.weight[output_alive]
                .to(context.dtype)
                .unsqueeze(0)
                .expand(batch, -1, -1),
            )
        hidden = context.clone()
        cell_memory = self.cell_rule.initial_memory(hidden)
        target_site, slot, source_site = substrate.edge_list()
        target_compact = compact[target_site]
        source_compact = compact[source_site]
        weights = substrate.synapse_weight[target_site, slot]
        compute_weights = weights.to(hidden.dtype)
        input_compact = compact[substrate.input_sites]
        batch_rows = torch.arange(batch, device=tokens.device)
        edge_weight_log = compute_weights.abs().clamp_min(1e-5).log().unsqueeze(0)
        edge_weight_view = compute_weights.view(1, -1, 1)
        target_ones = torch.ones_like(target_compact, dtype=torch.float32)
        indegree = torch.zeros(sites.numel(), device=tokens.device, dtype=torch.float32)
        indegree.index_add_(0, target_compact, target_ones)
        entropy_valid = indegree > 1
        entropy_normalizer = indegree[entropy_valid].log()
        slot_keys = F.normalize(self.broadcast_slot_keys, dim=1)
        message_scale = F.softplus(self.message_gain_raw)
        broadcast_gain = self.broadcast_gain.clamp_min(0)
        fast_weight_gain = self.fast_weight_gain.clamp_min(0)
        hidden_scale = math.sqrt(cfg.hidden_channels)
        attention_scale = hidden_scale * cfg.attention_temperature
        site_count = sites.numel()
        edge_flow = torch.zeros_like(substrate.synapse_weight)
        stimulation = torch.zeros(site_count, device=tokens.device, dtype=torch.float32)
        load = torch.zeros_like(stimulation)
        advertised_query = torch.zeros(
            site_count, cfg.hidden_channels, device=tokens.device, dtype=torch.float32
        )
        advertised_key = torch.zeros_like(advertised_query)
        emission = torch.zeros_like(stimulation)
        frames: list[SequenceFrame] = []
        retained: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        entropy_total = torch.zeros((), device=tokens.device)
        observations = max(1, length * cfg.message_steps)
        workspace_state = torch.zeros(
            batch, cfg.broadcast_slots, cfg.hidden_channels,
            device=tokens.device, dtype=hidden.dtype,
        )
        use_fast_weights = cfg.fast_weight_gain > 0
        fast_memory = (
            torch.zeros(
                batch, cfg.hidden_channels, cfg.hidden_channels,
                device=tokens.device, dtype=hidden.dtype,
            )
            if use_fast_weights
            else None
        )
        use_binding_memory = self.binding_owner_address is not None
        binding_owner_addresses = (
            F.normalize(self.binding_owner_address(genotype), dim=1)
            if self.binding_owner_address is not None else None
        )
        binding_memory = (
            torch.zeros(
                batch, site_count, cfg.hidden_channels,
                device=tokens.device, dtype=hidden.dtype,
            )
            if use_binding_memory else None
        )
        previous_binding_key: torch.Tensor | None = None

        for position in range(length):
            external = torch.zeros_like(hidden)
            token_compact = input_compact[tokens[:, position]]
            alive_batch = token_compact >= 0
            rows = batch_rows[alive_batch]
            token_embedding = self.token_identity(tokens[:, position])
            drive = token_embedding[alive_batch]
            drive = drive + self.position_identity.weight[position]
            current_binding_key: torch.Tensor | None = None
            if use_binding_memory:
                assert self.binding_token_key is not None
                assert self.binding_read is not None
                assert self.binding_gain is not None
                assert binding_owner_addresses is not None
                assert binding_memory is not None
                current_binding_key = F.normalize(
                    self.binding_token_key(token_embedding), dim=1
                )
                read_attention = torch.softmax(
                    torch.einsum(
                        "bh,nh->bn", current_binding_key, binding_owner_addresses
                    ) / cfg.binding_memory_temperature,
                    dim=1,
                )
                retrieved = torch.einsum(
                    "bn,bnh->bh", read_attention, binding_memory
                )
                drive = drive + self.binding_read(retrieved[alive_batch]) * self.binding_gain
            drive = drive.to(external.dtype)
            external[rows, token_compact[alive_batch]] = drive
            input_signal = torch.zeros(sites.numel(), device=tokens.device)
            if alive_batch.any():
                input_signal.index_add_(
                    0, token_compact[alive_batch],
                    torch.ones_like(token_compact[alive_batch], dtype=input_signal.dtype),
                )
                input_signal /= batch
            if capture_trace:
                position_stimulation = torch.zeros_like(stimulation)
                position_load = torch.zeros_like(load)
                position_edge_flow = torch.zeros_like(edge_flow)

            for _ in range(cfg.message_steps):
                normalized = self.message_norm(hidden)
                query = self.message_query(normalized)
                key = self.message_key(normalized)
                value = self.message_value(normalized)
                emit = torch.sigmoid(self.emit_gate(normalized))
                write_score = torch.einsum(
                    "bnh,sh->bns", self.broadcast_key(normalized), slot_keys
                ) / hidden_scale
                write_attention = torch.softmax(write_score, dim=1)
                proposed_workspace = torch.einsum(
                    "bns,bnh->bsh", write_attention, self.broadcast_value(normalized)
                )
                workspace_state = (
                    cfg.broadcast_decay * workspace_state
                    + (1 - cfg.broadcast_decay) * proposed_workspace
                )
                read_score = torch.einsum(
                    "bnh,sh->bns", self.broadcast_query(normalized), slot_keys
                ) / hidden_scale
                read_attention = torch.softmax(read_score, dim=2)
                broadcast_message = torch.einsum(
                    "bns,bsh->bnh", read_attention, workspace_state
                ) * broadcast_gain
                if use_fast_weights:
                    assert fast_memory is not None
                    fast_key = F.normalize(self.fast_key(normalized), dim=2)
                    fast_value = self.fast_value(normalized)
                    fast_gate = torch.sigmoid(self.fast_write_gate(normalized))
                    proposed_fast_memory = torch.einsum(
                        "bni,bnj->bij", fast_key * fast_gate, fast_value
                    ) / site_count
                    fast_memory = (
                        cfg.fast_weight_decay * fast_memory
                        + (1 - cfg.fast_weight_decay) * proposed_fast_memory
                    )
                    fast_query = F.normalize(self.fast_query(normalized), dim=2)
                    fast_message = torch.einsum(
                        "bni,bij->bnj", fast_query, fast_memory
                    ) * fast_weight_gain
                else:
                    fast_message = 0.0
                advertised_query += query.detach().float().mean(dim=0) / observations
                advertised_key += key.detach().float().mean(dim=0) / observations
                emission += emit.detach().float().mean(dim=(0, 2)) / observations
                compatibility = (query[:, target_compact] * key[:, source_compact]).sum(dim=2)
                compatibility /= attention_scale
                compatibility += edge_weight_log
                score = compatibility.clamp(-12, 12).exp()
                score_float = score.float()
                denominator = torch.zeros(
                    batch, site_count, device=tokens.device, dtype=torch.float32
                )
                denominator.index_add_(1, target_compact, score_float)
                attention = (
                    score_float / denominator[:, target_compact].clamp_min(1e-9)
                ).to(hidden.dtype)
                edge_message = (
                    value[:, source_compact]
                    * emit[:, source_compact]
                    * attention.unsqueeze(2)
                    * edge_weight_view
                    * message_scale
                )
                incoming = torch.zeros_like(hidden)
                incoming.index_add_(1, target_compact, edge_message)
                incoming = incoming + broadcast_message + fast_message
                if entropy_valid.any():
                    target_entropy = torch.zeros_like(denominator)
                    target_entropy.index_add_(
                        1, target_compact,
                        -(attention.float() * attention.float().clamp_min(1e-9).log()),
                    )
                    entropy_total += (
                        target_entropy[:, entropy_valid] / entropy_normalizer
                    ).detach().mean() / observations
                flow = edge_message.detach().float().abs().mean(dim=(0, 2))
                step_load = incoming.detach().float().abs().mean(dim=(0, 2))
                variable = (
                    edge_message.detach().float().std(dim=0, correction=0).mean(dim=1)
                )
                step_stimulation = torch.zeros_like(stimulation)
                step_stimulation.index_add_(0, target_compact, variable)
                step_stimulation /= indegree.clamp_min(1).sqrt()
                step_stimulation += (
                    external.detach().float().std(dim=0, correction=0).mean(dim=1)
                )
                edge_flow[target_site, slot] += flow / observations
                stimulation += step_stimulation / observations
                load += step_load / observations
                if capture_trace:
                    position_edge_flow[target_site, slot] += flow / cfg.message_steps
                    position_stimulation += step_stimulation / cfg.message_steps
                    position_load += step_load / cfg.message_steps
                rule_input = torch.cat((external, 0.1 * context), dim=2)
                updated, cell_memory = self.cell_rule(
                    rule_input, hidden, incoming, cell_memory
                )
                updated = updated * (1 + film_scale.unsqueeze(0)) + film_bias.unsqueeze(0)
                hidden = self.state_norm(updated + incoming)
            if use_binding_memory:
                assert self.binding_value is not None
                assert binding_owner_addresses is not None
                assert binding_memory is not None
                assert current_binding_key is not None
                if previous_binding_key is not None:
                    write_attention = torch.softmax(
                        torch.einsum(
                            "bh,nh->bn", previous_binding_key,
                            binding_owner_addresses,
                        ) / cfg.binding_memory_temperature,
                        dim=1,
                    )
                    write_value = torch.zeros(
                        batch, cfg.hidden_channels,
                        device=tokens.device, dtype=hidden.dtype,
                    )
                    if alive_batch.any():
                        value_source = (
                            token_embedding[alive_batch]
                            if cfg.binding_token_values
                            else hidden[rows, token_compact[alive_batch]]
                        )
                        write_value[rows] = self.binding_value(value_source)
                    ownership = write_attention.unsqueeze(2)
                    binding_memory = (
                        binding_memory * (1 - ownership)
                        + ownership * write_value.unsqueeze(1)
                    )
                previous_binding_key = current_binding_key
            if hidden.requires_grad:
                hidden.retain_grad()
                retained.append(hidden)
            position_logits = self._read_logits(hidden, output_compact, output_alive)
            logits.append(position_logits)
            if capture_trace:
                frame = self.make_frame(
                    sites, hidden[0], stimulation=position_stimulation,
                    load=position_load, input_signal=input_signal,
                    edge_flow=position_edge_flow, stage="token", step=position,
                    token_position=position,
                )
                frames.append(frame)
                if frame_callback is not None:
                    frame_callback(frame, position_logits.detach())

        return SequenceForward(
            torch.stack(logits, dim=1), sites, hidden, retained, frames,
            stimulation, load, edge_flow, advertised_query, advertised_key,
            emission, float(entropy_total),
        )

    def make_frame(
        self,
        sites: torch.Tensor,
        state: torch.Tensor,
        *,
        stimulation: torch.Tensor | None = None,
        load: torch.Tensor | None = None,
        credit: torch.Tensor | None = None,
        input_signal: torch.Tensor | None = None,
        edge_flow: torch.Tensor | None = None,
        edge_credit: torch.Tensor | None = None,
        stage: str,
        step: int,
        token_position: int = -1,
        events: list[dict[str, Any]] | None = None,
    ) -> SequenceFrame:
        zero = torch.zeros(sites.numel(), device=state.device)
        graph = self.substrate.graph_snapshot()
        if edge_flow is not None:
            graph.flow = edge_flow.detach().clone()
        if edge_credit is not None:
            graph.credit = edge_credit.detach().clone()
        return SequenceFrame(
            sites.detach().clone(), state.detach().clone(),
            (stimulation if stimulation is not None else zero).detach().clone(),
            (load if load is not None else zero).detach().clone(),
            (credit if credit is not None else zero).detach().clone(),
            (input_signal if input_signal is not None else zero).detach().clone(),
            graph, stage, step, token_position, list(events or []),
        )

    def regularization(self) -> torch.Tensor:
        active = self.substrate.active_edge_mask
        if not active.any():
            return self.substrate.synapse_weight.sum() * 0
        return self.config.weight_decay * self.substrate.synapse_weight[active].square().mean()


__all__ = [
    "CellularSequenceModel", "SequenceForward", "SequenceFrame",
    "SequenceFrameCallback",
]
