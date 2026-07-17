"""Self-assembling recurrent cellular graph classifier for MNIST."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as functional

from .mnist_routing import BroadcastRouter, EpisodeGraph, RoutingUpdate


@dataclass(frozen=True, slots=True)
class MnistModelConfig:
    width: int = 16
    height: int = 16
    hidden_channels: int = 32
    route_channels: int = 12
    edge_slots: int = 4
    sensory_steps: int = 7
    development_steps: int = 5
    inference_steps: int = 4
    routing_interval: int = 2
    batch_size: int = 16
    learning_rate: float = 0.0015
    wiring_cost: float = 0.001
    length_cost: float = 0.0003
    route_distance_cost: float = 0.12
    persistence_bonus: float = 0.22
    broadcast_temperature: float = 0.35
    activity_broadcast_bias: float = 1.0
    evaluation_interval: int = 100
    evaluation_batches: int = 8
    episode_noise: float = 0.0

    @property
    def cell_count(self) -> int:
        return self.width * self.height

    @property
    def total_steps(self) -> int:
        return self.sensory_steps + self.development_steps + self.inference_steps


@dataclass(slots=True)
class MnistFrame:
    state: torch.Tensor
    sensory_signal: torch.Tensor
    graph: EpisodeGraph
    axon_request: torch.Tensor
    receptor_request: torch.Tensor
    stage: str
    step: int
    token_row: int
    events: list[dict[str, Any]]


@dataclass(slots=True)
class MnistForward:
    logits: torch.Tensor
    trajectory_logits: torch.Tensor
    state: torch.Tensor
    patches: torch.Tensor
    graph: EpisodeGraph
    frames: list[MnistFrame]
    routing_cost: torch.Tensor


class CellularGraphClassifier(nn.Module):
    """Let shared recurrent cells assemble a sparse task graph from broadcasts."""

    sensor_count = 7
    motor_count = 10

    def __init__(self, config: MnistModelConfig | None = None, *, seed: int = 1) -> None:
        super().__init__()
        self.config = config or MnistModelConfig()
        cfg = self.config
        if cfg.width < 12 or cfg.height < 12:
            raise ValueError("MNIST assembly field must be at least 12 by 12")
        generator = torch.Generator().manual_seed(seed)
        (
            coordinates,
            sensor_indices,
            motor_indices,
            sensor_identity,
            motor_identity,
            interface_identity,
        ) = self._geometry()
        self.register_buffer("coordinates", coordinates)
        self.register_buffer("sensor_indices", sensor_indices)
        self.register_buffer("motor_indices", motor_indices)
        self.register_buffer("sensor_identity", sensor_identity)
        self.register_buffer("motor_identity", motor_identity)
        self.register_buffer("interface_identity", interface_identity)
        self.register_buffer("cell_noise", torch.randn(cfg.cell_count, 2, generator=generator))

        self.patch_encoder = nn.Sequential(nn.Linear(16, cfg.hidden_channels), nn.Tanh())
        interface_channels = self.sensor_count + self.motor_count
        seed_features = 2 + interface_channels + 2
        self.seed_rule = nn.Sequential(
            nn.Linear(seed_features, cfg.hidden_channels),
            nn.Tanh(),
        )
        rule_input = cfg.hidden_channels * 4 + 2 + interface_channels + 2
        self.cell_rule = nn.GRUCell(rule_input, cfg.hidden_channels)
        self.router = BroadcastRouter(
            cfg.hidden_channels,
            cfg.route_channels,
            cfg.edge_slots,
            coordinates,
            distance_cost=cfg.route_distance_cost,
            persistence_bonus=cfg.persistence_bonus,
            broadcast_temperature=cfg.broadcast_temperature,
            activity_bias=cfg.activity_broadcast_bias,
        )
        self.readout_scale = nn.Parameter(torch.tensor(2.0))
        self.output_readout = nn.Linear(cfg.hidden_channels, 1)
        self.class_bias = nn.Parameter(torch.zeros(10))
        self.register_buffer("lesion_mask", torch.ones(cfg.cell_count))

    def _geometry(self) -> tuple[torch.Tensor, ...]:
        cfg = self.config
        y = torch.arange(cfg.height).repeat_interleave(cfg.width)
        x = torch.arange(cfg.width).repeat(cfg.height)
        coordinates = torch.stack(
            (
                x.float() / max(1, cfg.width - 1) * 2 - 1,
                y.float() / max(1, cfg.height - 1) * 2 - 1,
            ),
            dim=1,
        )
        sensor_y = torch.linspace(3, cfg.height - 4, self.sensor_count).round().long()
        sensor_indices = sensor_y * cfg.width
        motor_y = torch.linspace(2, cfg.height - 3, self.motor_count).round().long()
        motor_indices = motor_y * cfg.width + (cfg.width - 1)
        sensor_identity = torch.zeros(cfg.cell_count)
        sensor_identity[sensor_indices] = 1
        motor_identity = torch.zeros(cfg.cell_count)
        motor_identity[motor_indices] = torch.linspace(0.1, 1.0, self.motor_count)
        interface_identity = torch.zeros(cfg.cell_count, self.sensor_count + self.motor_count)
        interface_identity[sensor_indices, torch.arange(self.sensor_count)] = 1
        interface_identity[motor_indices, self.sensor_count + torch.arange(self.motor_count)] = 1
        return (
            coordinates,
            sensor_indices,
            motor_indices,
            sensor_identity,
            motor_identity,
            interface_identity,
        )

    def _patchify(self, images: torch.Tensor) -> torch.Tensor:
        patches = images.unfold(2, 4, 4).unfold(3, 4, 4)
        return patches[:, 0].reshape(images.shape[0], 7, 7, 16)

    def _initial_state(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        cfg = self.config
        features = torch.cat((self.coordinates, self.interface_identity, 0.2 * self.cell_noise), dim=1)
        hidden = self.seed_rule(features).unsqueeze(0).expand(batch_size, -1, -1)
        if self.training and cfg.episode_noise > 0:
            hidden = hidden + cfg.episode_noise * torch.randn_like(hidden)
        return hidden.to(dtype=dtype) * self.lesion_mask.view(1, cfg.cell_count, 1)

    def forward(self, images: torch.Tensor, *, capture_trace: bool = True) -> MnistForward:
        """Present patch rows to sensory ports and recurrently assemble computation."""

        cfg = self.config
        batch_size = images.shape[0]
        patches = self._patchify(images)
        encoded_patches = self.patch_encoder(patches)
        hidden = self._initial_state(batch_size, images.dtype)
        graph = self.router.empty(batch_size, cfg.cell_count, images.device)
        frames: list[MnistFrame] = []
        trajectory_logits: list[torch.Tensor] = []
        axon_request, receptor_request = self.router.signals(hidden)
        if capture_trace:
            frames.append(
                self._capture_frame(
                    hidden, torch.zeros(batch_size, cfg.cell_count, device=images.device), graph,
                    axon_request, receptor_request, "seed", 0, -1, None,
                )
            )

        coordinates = self.coordinates.view(1, cfg.cell_count, 2).expand(batch_size, -1, -1)
        interfaces = self.interface_identity.view(1, cfg.cell_count, -1).expand(batch_size, -1, -1)
        lesion = self.lesion_mask.view(1, cfg.cell_count, 1)
        for step in range(cfg.total_steps):
            sensory = torch.zeros(
                batch_size, cfg.cell_count, cfg.hidden_channels,
                device=images.device, dtype=images.dtype,
            )
            sensory_signal = torch.zeros(batch_size, cfg.cell_count, device=images.device, dtype=images.dtype)
            token_row = step if step < cfg.sensory_steps else -1
            if token_row >= 0:
                sensory[:, self.sensor_indices] = encoded_patches[:, token_row]
                sensory_signal[:, self.sensor_indices] = patches[:, token_row].mean(dim=2)

            grid = hidden.transpose(1, 2).reshape(batch_size, cfg.hidden_channels, cfg.height, cfg.width)
            local = functional.avg_pool2d(grid, kernel_size=3, stride=1, padding=1)
            local = local.reshape(batch_size, cfg.hidden_channels, cfg.cell_count).transpose(1, 2)
            graph_message = self.router.messages(hidden, graph)
            broadcast_message, _, _ = self.router.broadcast(hidden, self.lesion_mask)
            phase = 2 * math.pi * step / max(1, cfg.total_steps - 1)
            clock = torch.tensor((math.sin(phase), math.cos(phase)), device=images.device, dtype=images.dtype)
            clock = clock.view(1, 1, 2).expand(batch_size, cfg.cell_count, -1)
            rule_input = torch.cat(
                (local, graph_message, broadcast_message, sensory, coordinates, interfaces, clock), dim=2
            )
            hidden = self.cell_rule(
                rule_input.reshape(-1, rule_input.shape[-1]),
                hidden.reshape(-1, cfg.hidden_channels),
            ).reshape(batch_size, cfg.cell_count, cfg.hidden_channels)
            hidden = torch.tanh(
                hidden + sensory + 0.45 * broadcast_message + 0.25 * graph_message
            ) * lesion

            update: RoutingUpdate | None = None
            if step == 0 or (step + 1) % cfg.routing_interval == 0:
                update = self.router(hidden, graph, self.lesion_mask)
                graph = update.graph
                axon_request = update.axon_request
                receptor_request = update.receptor_request
            else:
                axon_request, receptor_request = self.router.signals(hidden)

            if step < cfg.sensory_steps:
                stage = "sensing"
            elif step < cfg.sensory_steps + cfg.development_steps:
                stage = "developing"
            else:
                stage = "reading"
            if capture_trace:
                frames.append(
                    self._capture_frame(
                        hidden, sensory_signal, graph, axon_request, receptor_request,
                        stage, step + 1, token_row, update,
                    )
                )
            if step >= cfg.sensory_steps - 1:
                trajectory_logits.append(self._read_logits(hidden))

        logits = self._read_logits(hidden)
        trajectory = torch.stack(trajectory_logits, dim=1)
        selected_distance = self.router.distance.unsqueeze(0).expand(batch_size, -1, -1)
        selected_distance = selected_distance.gather(2, graph.destination.clamp_min(0))
        routing_cost = cfg.wiring_cost * graph.strength.mean()
        routing_cost = routing_cost + cfg.length_cost * (graph.strength * selected_distance).mean()
        return MnistForward(logits, trajectory, hidden, patches, graph, frames, routing_cost)

    def _read_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.readout_scale * self.output_readout(hidden[:, self.motor_indices]).squeeze(-1)
        return logits + self.class_bias

    def _capture_frame(
        self,
        hidden: torch.Tensor,
        sensory_signal: torch.Tensor,
        graph: EpisodeGraph,
        axon_request: torch.Tensor,
        receptor_request: torch.Tensor,
        stage: str,
        step: int,
        token_row: int,
        update: RoutingUpdate | None,
    ) -> MnistFrame:
        events: list[dict[str, Any]] = []
        if update is not None:
            changed = update.replaced[0].nonzero(as_tuple=False)
            for source, slot in changed[:128].tolist():
                old_destination = int(update.old_destination[0, source, slot])
                new_destination = int(update.graph.destination[0, source, slot])
                if old_destination >= 0:
                    events.append({"type": "pruned", "source": source, "destination": old_destination})
                events.append({"type": "grown", "source": source, "destination": new_destination})
        sample_graph = EpisodeGraph(
            graph.destination[0].detach(), graph.weight[0].detach(), graph.strength[0].detach(),
            graph.age[0].detach(), graph.utility[0].detach(),
        )
        return MnistFrame(
            state=hidden[0].detach(),
            sensory_signal=sensory_signal[0].detach(),
            graph=sample_graph,
            axon_request=axon_request[0].detach(),
            receptor_request=receptor_request[0].detach(),
            stage=stage,
            step=step,
            token_row=token_row,
            events=events,
        )

    def regularization(self, result: MnistForward) -> torch.Tensor:
        return result.routing_cost

    @torch.no_grad()
    def lesion(self, x: float, y: float, radius: float) -> int:
        cfg = self.config
        grid_y = torch.arange(cfg.height, device=self.lesion_mask.device).repeat_interleave(cfg.width)
        grid_x = torch.arange(cfg.width, device=self.lesion_mask.device).repeat(cfg.height)
        damaged = (grid_x.float() - x).square() + (grid_y.float() - y).square() <= radius**2
        previously_alive = self.lesion_mask > 0.5
        self.lesion_mask[damaged] = 0
        return int((damaged & previously_alive).sum())
