"""Core neural cellular-field and self-growing graph dynamics."""

from __future__ import annotations

from collections import deque
import math
from typing import Any

import torch
import torch.nn.functional as functional

from .channels import Channel
from .config import SimulationConfig, resolve_device
from .state import PetriDishState
from .task import DelayedXorTask, TaskObservation


class PetriDishSimulation:
    """Own and advance one deterministic neural petri-dish experiment."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        seed: int = 1,
        device: str = "auto",
    ) -> None:
        self.config = config or SimulationConfig()
        self.seed = seed
        self.device = resolve_device(device)
        self._random = torch.Generator(device="cpu").manual_seed(seed)
        self.task = DelayedXorTask(self.config, self.device, seed)
        self.state = self._create_state()
        self._mark_task_regions()
        self.last_observation = self.task.observe(0, self.state.cells[:, Channel.ACTIVATION])
        self.last_reward = 0.0
        self.reward_history: deque[float] = deque(maxlen=200)
        self.events: deque[dict[str, Any]] = deque(maxlen=256)
        self._manual_stimulus = torch.zeros(self.config.cell_count, device=self.device)
        self._manual_stimulus_ticks = 0
        self._pending_reward = 0.0
        self._kernel = torch.full((self.config.channels, 1, 3, 3), 1 / 9, device=self.device)

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(shape, generator=self._random).to(self.device)

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=self._random).to(self.device)

    def _randint(self, high: int, *shape: int) -> torch.Tensor:
        return torch.randint(high, shape, generator=self._random).to(self.device)

    def _create_state(self) -> PetriDishState:
        cfg = self.config
        n, k = cfg.cell_count, cfg.edge_slots
        cells = torch.zeros((n, cfg.channels), device=self.device)

        y = torch.arange(cfg.height, device=self.device).repeat_interleave(cfg.width)
        x = torch.arange(cfg.width, device=self.device).repeat(cfg.height)
        normalized_x = x.float() / max(1, cfg.width - 1)
        normalized_y = y.float() / max(1, cfg.height - 1)
        radial = torch.sqrt((normalized_x - 0.5) ** 2 + (normalized_y - 0.5) ** 2)

        cells[:, Channel.ALIVE] = (0.93 - 0.12 * radial + 0.025 * self._rand(n)).clamp(0, 1)
        cells[:, Channel.ACTIVATION] = 0.05 * self._randn(n)
        phase = 2 * math.pi * self._rand(n)
        cells[:, Channel.PHASE_SIN] = torch.sin(phase)
        cells[:, Channel.PHASE_COS] = torch.cos(phase)
        cells[:, Channel.MEMORY_0 : Channel.MEMORY_3 + 1] = 0.02 * self._randn(n, 4)
        cells[:, Channel.ENERGY] = (0.75 + 0.1 * self._rand(n)).clamp(0, 1)
        cells[:, Channel.AXON_GROWTH] = 0.5
        cells[:, Channel.DENDRITE_GROWTH] = 0.5
        cells[:, Channel.POSITION_X] = normalized_x * 2 - 1
        cells[:, Channel.POSITION_Y] = normalized_y * 2 - 1

        source = torch.arange(n, device=self.device).view(n, 1).expand(n, k)
        source_x = source % cfg.width
        source_y = source // cfg.width
        offset_x = self._randint(7, n, k) - 3
        offset_y = self._randint(7, n, k) - 3
        destination_x = (source_x + offset_x).clamp(0, cfg.width - 1)
        destination_y = (source_y + offset_y).clamp(0, cfg.height - 1)
        destination = destination_y * cfg.width + destination_x
        if k > 1:
            destination[:, 1] = self._randint(n, n)

        gate = torch.zeros((n, k), device=self.device)
        gate[:, : min(k, cfg.initial_active_slots)] = 1
        weight = cfg.initial_weight_scale * self._randn(n, k) * gate

        return PetriDishState(
            cells=cells,
            edge_destination=destination.long(),
            edge_weight=weight,
            edge_gate=gate,
            edge_eligibility=torch.zeros((n, k), device=self.device),
            edge_age=torch.zeros((n, k), dtype=torch.long, device=self.device),
            edge_utility=torch.zeros((n, k), device=self.device),
        )

    def _mark_task_regions(self) -> None:
        cells = self.state.cells
        cells[:, Channel.SENSOR_ID] = 0
        cells[:, Channel.MOTOR_ID] = 0
        cells[self.task.sensor_a, Channel.SENSOR_ID] = -1
        cells[self.task.sensor_b, Channel.SENSOR_ID] = 1
        cells[self.task.motor_zero, Channel.MOTOR_ID] = -1
        cells[self.task.motor_one, Channel.MOTOR_ID] = 1

    @torch.inference_mode()
    def step(self, count: int = 1) -> None:
        """Advance one or more local-learning ticks without autograd state."""

        for _ in range(count):
            self._step_once()

    def _step_once(self) -> None:
        cfg, state = self.config, self.state
        cells = state.cells
        observation = self.task.observe(state.tick, cells[:, Channel.ACTIVATION])
        stimulus = observation.stimulus + self._manual_stimulus
        reward = observation.reward + self._pending_reward
        self._pending_reward = 0.0
        if self._manual_stimulus_ticks > 0:
            self._manual_stimulus_ticks -= 1
            self._manual_stimulus.mul_(0.91)
        else:
            self._manual_stimulus.zero_()

        grid = cells.transpose(0, 1).reshape(1, cfg.channels, cfg.height, cfg.width)
        local = functional.conv2d(grid, self._kernel, padding=1, groups=cfg.channels)
        local = local.reshape(cfg.channels, cfg.cell_count).transpose(0, 1)

        destination = state.edge_destination
        source_activation = cells[:, Channel.ACTIVATION].unsqueeze(1)
        destination_activation = cells[:, Channel.ACTIVATION][destination]
        source_alive = cells[:, Channel.ALIVE].unsqueeze(1)
        destination_alive = cells[:, Channel.ALIVE][destination]
        phase_alignment = (
            cells[:, Channel.PHASE_SIN].unsqueeze(1) * cells[:, Channel.PHASE_SIN][destination]
            + cells[:, Channel.PHASE_COS].unsqueeze(1) * cells[:, Channel.PHASE_COS][destination]
        )
        signal = (
            state.edge_gate
            * state.edge_weight
            * source_activation
            * source_alive
            * destination_alive
            * (0.7 + 0.3 * phase_alignment)
        )
        graph_input = torch.zeros(cfg.cell_count, device=self.device)
        graph_input.scatter_add_(0, destination.reshape(-1), signal.reshape(-1))
        graph_input = graph_input.clamp(-3, 3)

        alive = cells[:, Channel.ALIVE]
        activation = cells[:, Channel.ACTIVATION]
        next_activation = torch.tanh(
            0.46 * activation
            + 0.27 * local[:, Channel.ACTIVATION]
            + 0.72 * graph_input
            + 1.05 * stimulus
            + 0.13 * cells[:, Channel.MEMORY_0]
        ) * alive

        memory_0 = 0.93 * cells[:, Channel.MEMORY_0] + 0.18 * next_activation
        memory_1 = 0.97 * cells[:, Channel.MEMORY_1] + 0.10 * cells[:, Channel.MEMORY_0]
        memory_2 = 0.96 * cells[:, Channel.MEMORY_2] + 0.08 * local[:, Channel.REWARD_TRACE]
        memory_3 = 0.95 * cells[:, Channel.MEMORY_3] + 0.06 * (next_activation.abs() - 0.2)

        delta_phase = 0.075 + 0.035 * local[:, Channel.ACTIVATION] + 0.025 * graph_input
        phase_sin = cells[:, Channel.PHASE_SIN]
        phase_cos = cells[:, Channel.PHASE_COS]
        sin_delta, cos_delta = torch.sin(delta_phase), torch.cos(delta_phase)
        next_phase_sin = phase_sin * cos_delta + phase_cos * sin_delta
        next_phase_cos = phase_cos * cos_delta - phase_sin * sin_delta

        reward_injection = torch.zeros(cfg.cell_count, device=self.device)
        reward_injection[self.task.motor_zero] = reward
        reward_injection[self.task.motor_one] = reward
        next_reward_trace = (
            0.86 * cells[:, Channel.REWARD_TRACE]
            + 0.12 * local[:, Channel.REWARD_TRACE]
            + 0.48 * reward_injection
        ).clamp(-2, 2)

        outgoing_cost = state.edge_gate.mean(dim=1)
        next_energy = (
            0.996 * cells[:, Channel.ENERGY]
            + 0.007 * (local[:, Channel.ALIVE] - 0.48)
            - 0.0024 * next_activation.abs()
            - 0.0012 * outgoing_cost
            + 0.006 * next_reward_trace.clamp_min(0)
        ).clamp(0, 1)
        viability_pressure = (
            0.026 * (local[:, Channel.ALIVE] - 0.42)
            + 0.006 * (next_energy - 0.25)
        )
        next_alive = (alive + viability_pressure * (0.35 + 0.65 * (1 - alive))).clamp(0, 1)
        next_axon = torch.sigmoid(1.6 * next_activation + 0.8 * memory_0 + 0.5 * next_energy - 0.25)
        next_dendrite = torch.sigmoid(-0.8 * next_activation + 0.7 * memory_1 + 0.6 * next_energy - 0.2)

        cells[:, Channel.ALIVE] = next_alive
        cells[:, Channel.ACTIVATION] = next_activation
        cells[:, Channel.PHASE_SIN] = next_phase_sin
        cells[:, Channel.PHASE_COS] = next_phase_cos
        cells[:, Channel.MEMORY_0] = memory_0
        cells[:, Channel.MEMORY_1] = memory_1
        cells[:, Channel.MEMORY_2] = memory_2
        cells[:, Channel.MEMORY_3] = memory_3
        cells[:, Channel.ENERGY] = next_energy
        cells[:, Channel.AXON_GROWTH] = next_axon
        cells[:, Channel.DENDRITE_GROWTH] = next_dendrite
        cells[:, Channel.REWARD_TRACE] = next_reward_trace

        post_activation = next_activation[destination]
        state.edge_eligibility.mul_(cfg.eligibility_decay).add_(source_activation * post_activation)
        local_third_factor = reward + 0.08 * (
            next_reward_trace.unsqueeze(1) + next_reward_trace[destination]
        )
        state.edge_weight.add_(
            cfg.weight_learning_rate * local_third_factor * state.edge_eligibility
            - cfg.weight_decay * state.edge_weight
        )
        state.edge_weight.clamp_(-cfg.max_weight, cfg.max_weight).mul_(state.edge_gate)
        state.edge_utility.mul_(cfg.utility_decay).add_((1 - cfg.utility_decay) * signal.abs())
        state.edge_age.add_(state.edge_gate.long())

        state.tick += 1
        self.last_observation = observation
        self.last_reward = reward
        self.reward_history.append(reward)
        if state.tick % cfg.growth_interval == 0:
            self._rewire()
        self._expire_events()

    def _rewire(self) -> None:
        cfg, state = self.config, self.state
        cells = state.cells
        active = state.edge_gate > 0.5
        source_dead = cells[:, Channel.ALIVE].unsqueeze(1) < 0.12
        destination_dead = cells[:, Channel.ALIVE][state.edge_destination] < 0.12
        stale = (state.edge_age > cfg.prune_age) & (state.edge_utility < cfg.prune_utility)
        prune = active & (stale | source_dead | destination_dead)
        self._record_edge_events(prune, "pruned")
        state.edge_gate[prune] = 0
        state.edge_weight[prune] = 0
        state.edge_eligibility[prune] = 0
        state.edge_age[prune] = 0
        state.edge_utility[prune] = 0

        empty = state.edge_gate < 0.5
        if not bool(empty.any()):
            return

        n, k, candidates = cfg.cell_count, cfg.edge_slots, cfg.candidate_targets
        source = torch.arange(n, device=self.device).view(n, 1, 1).expand(n, k, candidates)
        source_x, source_y = source % cfg.width, source // cfg.width
        offset_x = self._randint(11, n, k, candidates) - 5
        offset_y = self._randint(11, n, k, candidates) - 5
        target_x = (source_x + offset_x).clamp(0, cfg.width - 1)
        target_y = (source_y + offset_y).clamp(0, cfg.height - 1)
        target = target_y * cfg.width + target_x
        remote_count = min(2, candidates)
        target[:, :, -remote_count:] = self._randint(n, n, k, remote_count)

        source_activation = cells[:, Channel.ACTIVATION].view(n, 1, 1)
        source_growth = cells[:, Channel.AXON_GROWTH].view(n, 1, 1)
        target_activation = cells[:, Channel.ACTIVATION][target]
        target_receptor = cells[:, Channel.DENDRITE_GROWTH][target]
        distance = torch.sqrt((target_x - source_x).float() ** 2 + (target_y - source_y).float() ** 2)
        score = (
            0.9 * source_growth
            + 0.9 * target_receptor
            + 0.55 * source_activation * target_activation
            - 0.018 * distance
            + 0.18 * self._rand(n, k, candidates)
        )
        chosen_index = score.argmax(dim=2, keepdim=True)
        chosen_target = target.gather(2, chosen_index).squeeze(2)
        chosen_score = score.gather(2, chosen_index).squeeze(2)
        grow_probability = 0.13 * torch.sigmoid(chosen_score - 1.15)
        grow = empty & (self._rand(n, k) < grow_probability)
        if not bool(grow.any()):
            return

        state.edge_destination[grow] = chosen_target[grow]
        initial_weight = 0.10 * self._randn(n, k)
        correlated_sign = torch.sign(
            cells[:, Channel.ACTIVATION].unsqueeze(1)
            * cells[:, Channel.ACTIVATION][chosen_target]
        )
        initial_weight = torch.where(correlated_sign == 0, initial_weight, initial_weight.abs() * correlated_sign)
        state.edge_weight[grow] = initial_weight[grow]
        state.edge_gate[grow] = 1
        state.edge_eligibility[grow] = 0
        state.edge_age[grow] = 0
        state.edge_utility[grow] = 0
        self._record_edge_events(grow, "grown")

    def _record_edge_events(self, mask: torch.Tensor, event_type: str) -> None:
        positions = mask.nonzero(as_tuple=False)
        for source, slot in positions[:128].tolist():
            self.events.append(
                {
                    "type": event_type,
                    "source": source,
                    "destination": int(self.state.edge_destination[source, slot]),
                    "tick": self.state.tick,
                }
            )

    def _expire_events(self) -> None:
        while self.events and self.state.tick - int(self.events[0]["tick"]) > 36:
            self.events.popleft()

    @torch.inference_mode()
    def lesion(self, x: float, y: float, radius: float) -> int:
        """Kill cells in a circular brush and remove all incident edges."""

        cfg, state = self.config, self.state
        grid_y = torch.arange(cfg.height, device=self.device).repeat_interleave(cfg.width)
        grid_x = torch.arange(cfg.width, device=self.device).repeat(cfg.height)
        damaged = (grid_x.float() - x) ** 2 + (grid_y.float() - y) ** 2 <= radius**2
        count = int(damaged.sum())
        state.cells[damaged, Channel.ALIVE] = 0
        state.cells[damaged, Channel.ACTIVATION] = 0
        state.cells[damaged, Channel.MEMORY_0 : Channel.MEMORY_3 + 1] = 0
        state.cells[damaged, Channel.ENERGY] = 0
        incident = damaged.unsqueeze(1) | damaged[state.edge_destination]
        self._record_edge_events(incident & (state.edge_gate > 0.5), "pruned")
        state.edge_gate[incident] = 0
        state.edge_weight[incident] = 0
        state.edge_eligibility[incident] = 0
        state.edge_age[incident] = 0
        state.edge_utility[incident] = 0
        return count

    def stimulate(self, region: str, amount: float = 1.2, duration: int = 16) -> None:
        """Inject a temporary signal into a named sensory or motor region."""

        regions = {
            "sensor_a": self.task.sensor_a,
            "sensor_b": self.task.sensor_b,
            "motor_zero": self.task.motor_zero,
            "motor_one": self.task.motor_one,
        }
        if region not in regions:
            raise ValueError(f"unknown region: {region}")
        self._manual_stimulus[regions[region]] = amount
        self._manual_stimulus_ticks = max(self._manual_stimulus_ticks, duration)

    def inject_reward(self, amount: float = 1.0) -> None:
        """Queue a scalar neuromodulatory pulse for the next tick."""

        self._pending_reward += amount

    @property
    def rolling_reward(self) -> float:
        return sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0
