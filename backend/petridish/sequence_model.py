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
    runtime_state: "SequenceRuntimeState"


@dataclass(slots=True)
class SequenceRuntimeState:
    """Persistent organism state carried between incremental token calls."""

    sites: torch.Tensor
    hidden: torch.Tensor
    cell_memory: torch.Tensor | None
    workspace: torch.Tensor
    fast_memory: torch.Tensor | None
    binding_memory: torch.Tensor | None
    previous_binding_key: torch.Tensor | None
    position: int

    def detached(self) -> "SequenceRuntimeState":
        """Cut autograd history while preserving every piece of organism state."""

        return SequenceRuntimeState(
            self.sites.detach(), self.hidden.detach(),
            self.cell_memory.detach() if self.cell_memory is not None else None,
            self.workspace.detach(),
            self.fast_memory.detach() if self.fast_memory is not None else None,
            self.binding_memory.detach() if self.binding_memory is not None else None,
            (
                self.previous_binding_key.detach()
                if self.previous_binding_key is not None else None
            ),
            self.position,
        )

    def cloned_detached(self) -> "SequenceRuntimeState":
        """Copy every electrical channel for a mutation-isolated evaluation branch."""

        def clone(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().clone()

        sites = clone(self.sites)
        hidden = clone(self.hidden)
        workspace = clone(self.workspace)
        assert sites is not None and hidden is not None and workspace is not None
        return SequenceRuntimeState(
            sites,
            hidden,
            clone(self.cell_memory),
            workspace,
            clone(self.fast_memory),
            clone(self.binding_memory),
            clone(self.previous_binding_key),
            self.position,
        )


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
        self.distributed_io = (
            self.layout.input_count != vocab_size
            or self.layout.output_count != vocab_size
        )
        torch.manual_seed(seed)
        self.substrate = SpatialSubstrate(config, layout=self.layout, seed=seed)
        hidden = config.hidden_channels
        persistent_features = 2 + 3 + (6 if self.distributed_io else 4)
        self.context_rule = nn.Sequential(nn.Linear(persistent_features, hidden), nn.Tanh())
        self.genotype_context = nn.Linear(config.genotype_channels, hidden)
        self.genotype_film = nn.Linear(config.genotype_channels, hidden * 2)
        self.token_identity = nn.Embedding(vocab_size, hidden)
        self.position_identity = nn.Embedding(max_length, hidden)
        self.input_port_identity: nn.Embedding | None = None
        self.input_value: nn.Linear | None = None
        if self.distributed_io:
            self.input_port_identity = nn.Embedding(self.layout.input_count, hidden)
            self.input_value = nn.Linear(hidden, hidden, bias=False)
        self.output_identity = nn.Embedding(self.layout.output_count, hidden)
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
        readout_width = self.layout.output_count * hidden
        self.output_bank_readout = nn.Linear(
            readout_width, hidden if self.distributed_io else vocab_size
        )
        self.logit_scale: nn.Parameter | None = (
            nn.Parameter(torch.tensor(math.sqrt(hidden))) if self.distributed_io else None
        )
        self.class_bias = nn.Parameter(torch.zeros(vocab_size))
        self.message_gain_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(config.message_gain)))
        )
        nn.init.normal_(self.token_identity.weight, std=0.18)
        nn.init.normal_(self.position_identity.weight, std=0.08)
        nn.init.normal_(self.output_identity.weight, std=0.18)
        if self.input_port_identity is not None:
            nn.init.normal_(self.input_port_identity.weight, std=0.18)
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
        features = [
            substrate.energy[sites],
            substrate.stimulation_ema[sites],
            substrate.load_ema[sites],
            substrate.neuron_utility[sites].tanh(),
        ]
        if self.distributed_io:
            features.extend(
                (substrate.stunned[sites].float(), substrate.excitotoxic_damage[sites])
            )
        slow = torch.stack(features, dim=1)
        values = torch.cat((substrate.coordinates[sites], substrate.roles[sites], slow), dim=1)
        return values.unsqueeze(0).expand(batch, -1, -1)

    def _read_logits(
        self,
        hidden: torch.Tensor,
        output_compact: torch.Tensor,
        output_alive: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros(
            hidden.shape[0], self.layout.output_count, self.config.hidden_channels,
            device=hidden.device, dtype=hidden.dtype,
        )
        if output_alive.any():
            output[:, output_alive] = hidden[:, output_compact[output_alive]]
        decoded = self.output_bank_readout(output.flatten(1))
        if not self.distributed_io:
            return decoded + self.class_bias
        assert self.logit_scale is not None
        code = F.normalize(decoded.float(), dim=1)
        vocabulary = F.normalize(self.token_identity.weight.float(), dim=1)
        return code @ vocabulary.T * self.logit_scale.clamp(1, 20) + self.class_bias

    def _resting_hidden(
        self, sites: torch.Tensor, batch: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """Construct each physical neuron's genotype/role/homeostasis resting state."""

        genotype = self.substrate.genotype[sites]
        hidden = self.context_rule(self._persistent_features(sites, batch))
        hidden = hidden + torch.tanh(self.genotype_context(genotype)).unsqueeze(0)
        compact = torch.full(
            (self.config.site_count,), -1, dtype=torch.long, device=sites.device
        )
        compact[sites] = torch.arange(sites.numel(), device=sites.device)
        output_compact = compact[self.substrate.output_sites]
        output_alive = (
            (output_compact >= 0)
            & ~self.substrate.stunned[self.substrate.output_sites]
        )
        if output_alive.any():
            hidden = hidden.index_add(
                1,
                output_compact[output_alive],
                self.output_identity.weight[output_alive]
                .to(hidden.dtype)
                .unsqueeze(0)
                .expand(batch, -1, -1),
            )
        return hidden.to(dtype)

    @torch.no_grad()
    def reconcile_runtime_state(
        self, runtime_state: SequenceRuntimeState
    ) -> SequenceRuntimeState:
        """Carry survivor state through cell birth/death without resetting the field."""

        state = runtime_state.detached()
        sites = self.substrate.living_sites
        if torch.equal(state.sites, sites):
            return state
        batch = state.hidden.shape[0]
        hidden = self._resting_hidden(sites, batch, state.hidden.dtype)
        compact = torch.full(
            (self.config.site_count,), -1, dtype=torch.long, device=sites.device
        )
        compact[state.sites] = torch.arange(state.sites.numel(), device=sites.device)
        previous = compact[sites]
        survivors = previous >= 0
        hidden[:, survivors] = state.hidden[:, previous[survivors]]
        cell_memory = self.cell_rule.initial_memory(hidden)
        if cell_memory is not None and state.cell_memory is not None:
            cell_memory[:, survivors] = state.cell_memory[:, previous[survivors]]
        binding_memory: torch.Tensor | None = None
        if state.binding_memory is not None:
            binding_memory = state.binding_memory.new_zeros(
                batch, sites.numel(), state.binding_memory.shape[-1]
            )
            binding_memory[:, survivors] = state.binding_memory[:, previous[survivors]]
        return SequenceRuntimeState(
            sites.detach(), hidden.detach(),
            cell_memory.detach() if cell_memory is not None else None,
            state.workspace, state.fast_memory, binding_memory,
            state.previous_binding_key, state.position,
        )

    @torch.no_grad()
    def relax_runtime_state(
        self, runtime_state: SequenceRuntimeState, retention: float
    ) -> SequenceRuntimeState:
        """Relax electrical memory toward physical resting state without a reset."""

        if not 0 <= retention <= 1:
            raise ValueError("state retention must be between zero and one")
        state = self.reconcile_runtime_state(runtime_state)
        if retention == 1:
            return state
        resting = self._resting_hidden(
            state.sites, state.hidden.shape[0], state.hidden.dtype
        )
        hidden = retention * state.hidden + (1 - retention) * resting

        def decay(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value * retention

        return SequenceRuntimeState(
            state.sites, hidden, decay(state.cell_memory),
            state.workspace * retention, decay(state.fast_memory),
            decay(state.binding_memory), state.previous_binding_key, state.position,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        capture_trace: bool = True,
        frame_callback: SequenceFrameCallback | None = None,
        runtime_state: SequenceRuntimeState | None = None,
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
        output_alive = (output_compact >= 0) & ~substrate.stunned[substrate.output_sites]
        if output_alive.any():
            context = context.index_add(
                1,
                output_compact[output_alive],
                self.output_identity.weight[output_alive]
                .to(context.dtype)
                .unsqueeze(0)
                .expand(batch, -1, -1),
            )
        if runtime_state is not None:
            if not torch.equal(runtime_state.sites, sites):
                raise ValueError("incremental state is invalid after a population change")
            if runtime_state.hidden.shape[0] != batch:
                raise ValueError("incremental state batch size does not match tokens")
            hidden = runtime_state.hidden
            cell_memory = runtime_state.cell_memory
            start_position = runtime_state.position
        else:
            hidden = context.clone()
            cell_memory = self.cell_rule.initial_memory(hidden)
            start_position = 0
        target_site, slot, source_site = substrate.conducting_edge_list()
        target_compact = compact[target_site]
        source_compact = compact[source_site]
        weights = substrate.synapse_weight[target_site, slot]
        compute_weights = weights.to(hidden.dtype)
        input_compact = compact[substrate.input_sites]
        responsive = ~substrate.stunned[sites]
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
        use_broadcast = cfg.broadcast_gain > 0
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
        workspace_state = (
            runtime_state.workspace
            if runtime_state is not None
            else torch.zeros(
                batch, cfg.broadcast_slots, cfg.hidden_channels,
                device=tokens.device, dtype=hidden.dtype,
            )
        )
        use_fast_weights = cfg.fast_weight_gain > 0
        fast_memory = (
            runtime_state.fast_memory
            if runtime_state is not None and use_fast_weights
            else torch.zeros(
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
            runtime_state.binding_memory
            if runtime_state is not None and use_binding_memory
            else torch.zeros(
                batch, site_count, cfg.hidden_channels,
                device=tokens.device, dtype=hidden.dtype,
            )
            if use_binding_memory else None
        )
        previous_binding_key = (
            runtime_state.previous_binding_key if runtime_state is not None else None
        )

        for position in range(length):
            external = torch.zeros_like(hidden)
            token_embedding = self.token_identity(tokens[:, position])
            absolute_position = start_position + position
            drive = token_embedding + self.position_identity.weight[
                absolute_position % self.position_identity.num_embeddings
            ]
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
                drive = drive + self.binding_read(retrieved) * self.binding_gain
            input_signal = torch.zeros(sites.numel(), device=tokens.device)
            if self.distributed_io:
                assert self.input_port_identity is not None
                assert self.input_value is not None
                alive_ports = input_compact >= 0
                port_compact = input_compact[alive_ports]
                port_keys = F.normalize(self.input_port_identity.weight[alive_ports], dim=1)
                coefficients = torch.softmax(
                    F.normalize(drive, dim=1) @ port_keys.T
                    * math.sqrt(cfg.hidden_channels), dim=1,
                ) * math.sqrt(max(1, int(alive_ports.sum())))
                external[:, port_compact] = (
                    self.input_value(drive).unsqueeze(1) * coefficients.unsqueeze(2)
                ).to(external.dtype)
                input_signal[port_compact] = coefficients.detach().abs().mean(dim=0)
                token_compact = port_compact[coefficients.argmax(dim=1)]
                alive_batch = torch.ones(batch, device=tokens.device, dtype=torch.bool)
                rows = batch_rows
            else:
                token_compact = input_compact[tokens[:, position]]
                alive_batch = token_compact >= 0
                rows = batch_rows[alive_batch]
                external[rows, token_compact[alive_batch]] = drive[alive_batch].to(
                    external.dtype
                )
                if alive_batch.any():
                    input_signal.index_add_(
                        0, token_compact[alive_batch],
                        torch.ones_like(
                            token_compact[alive_batch], dtype=input_signal.dtype
                        ),
                    )
                    input_signal /= batch
            external[:, ~responsive] = 0
            if capture_trace:
                position_stimulation = torch.zeros_like(stimulation)
                position_load = torch.zeros_like(load)
                position_edge_flow = torch.zeros_like(edge_flow)

            for _ in range(cfg.message_steps):
                normalized = self.message_norm(hidden)
                query = self.message_query(normalized)
                key = self.message_key(normalized)
                value = self.message_value(normalized)
                emit = torch.sigmoid(self.emit_gate(normalized)) * responsive.view(1, -1, 1)
                if use_broadcast:
                    write_score = torch.einsum(
                        "bnh,sh->bns", self.broadcast_key(normalized), slot_keys
                    ) / hidden_scale
                    write_attention = torch.softmax(write_score, dim=1)
                    proposed_workspace = torch.einsum(
                        "bns,bnh->bsh", write_attention,
                        self.broadcast_value(normalized) * responsive.view(1, -1, 1)
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
                else:
                    broadcast_message = 0.0
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
                incoming = incoming * responsive.view(1, -1, 1)
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
                next_hidden = self.state_norm(updated + incoming)
                hidden = torch.where(
                    responsive.view(1, -1, 1), next_hidden, hidden
                )
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

        next_runtime_state = SequenceRuntimeState(
            sites,
            hidden,
            cell_memory,
            workspace_state,
            fast_memory,
            binding_memory,
            previous_binding_key,
            start_position + length,
        )
        return SequenceForward(
            torch.stack(logits, dim=1), sites, hidden, retained, frames,
            stimulation, load, edge_flow, advertised_query, advertised_key,
            emission, float(entropy_total), next_runtime_state,
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
        penalty = (
            self.config.weight_decay * self.substrate.synapse_weight[active].square().mean()
            if active.any() else self.substrate.synapse_weight.sum() * 0
        )
        if (
            self.config.binding_address_regularization > 0
            and self.binding_owner_address is not None
            and self.binding_token_key is not None
        ):
            sites = self.substrate.living_sites
            owner_addresses = F.normalize(
                self.binding_owner_address(self.substrate.genotype[sites]), dim=1
            )
            token_keys = F.normalize(
                self.binding_token_key(self.token_identity.weight), dim=1
            )
            attention = torch.softmax(
                token_keys @ owner_addresses.T
                / self.config.binding_memory_temperature,
                dim=1,
            )
            normalized_attention = F.normalize(attention, dim=1)
            overlap = normalized_attention @ normalized_attention.T
            off_diagonal = ~torch.eye(
                self.vocab_size, dtype=torch.bool, device=overlap.device
            )
            entropy = -(
                attention * attention.clamp_min(1e-9).log()
            ).sum(dim=1) / math.log(max(2, sites.numel()))
            penalty = penalty + self.config.binding_address_regularization * (
                overlap[off_diagonal].square().mean() + 0.1 * entropy.mean()
            )
        return penalty

    @torch.no_grad()
    def binding_memory_diagnostics(self) -> dict[str, float | int] | None:
        """Measure whether vocabulary tokens own separable physical addresses."""

        if self.binding_owner_address is None or self.binding_token_key is None:
            return None
        sites = self.substrate.living_sites
        owner_addresses = F.normalize(
            self.binding_owner_address(self.substrate.genotype[sites]), dim=1
        )
        token_keys = F.normalize(
            self.binding_token_key(self.token_identity.weight), dim=1
        )
        attention = torch.softmax(
            token_keys @ owner_addresses.T / self.config.binding_memory_temperature,
            dim=1,
        )
        entropy = -(
            attention * attention.clamp_min(1e-9).log()
        ).sum(dim=1) / math.log(max(2, sites.numel()))
        normalized_attention = F.normalize(attention, dim=1)
        overlap = normalized_attention @ normalized_attention.T
        off_diagonal = ~torch.eye(
            self.vocab_size, dtype=torch.bool, device=overlap.device
        )
        return {
            "distinctOwners": int(attention.argmax(dim=1).unique().numel()),
            "vocabularySize": self.vocab_size,
            "meanAddressEntropy": float(entropy.mean()),
            "meanAddressOverlap": float(overlap[off_diagonal].mean()),
            "meanPeakOwnership": float(attention.max(dim=1).values.mean()),
        }


__all__ = [
    "CellularSequenceModel", "SequenceForward", "SequenceFrame", "SequenceRuntimeState",
    "SequenceFrameCallback",
]
