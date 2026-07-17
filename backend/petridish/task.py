"""Delayed-XOR environment embedded at the borders of the cell field."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import torch

from .config import SimulationConfig


@dataclass(frozen=True, slots=True)
class TaskObservation:
    """Inputs and reward presented to the simulation on one tick."""

    stimulus: torch.Tensor
    reward: float
    phase: str
    bit_a: int
    bit_b: int
    target: int
    prediction: int | None
    accuracy: float


class DelayedXorTask:
    """Presents two border cues, waits, then rewards the selected motor region."""

    def __init__(self, config: SimulationConfig, device: torch.device, seed: int) -> None:
        self.config = config
        self.device = device
        self._random = random.Random(seed + 7_919)
        self.sensor_a = self._region(x=2, y=config.height // 3)
        self.sensor_b = self._region(x=2, y=2 * config.height // 3)
        self.motor_zero = self._region(x=config.width - 3, y=config.height // 3)
        self.motor_one = self._region(x=config.width - 3, y=2 * config.height // 3)
        self.bit_a = 0
        self.bit_b = 0
        self.last_prediction: int | None = None
        self._results: deque[float] = deque(maxlen=40)
        self._new_trial()

    def _region(self, x: int, y: int, radius: int = 1) -> torch.Tensor:
        indices = [
            yy * self.config.width + xx
            for yy in range(max(0, y - radius), min(self.config.height, y + radius + 1))
            for xx in range(max(0, x - radius), min(self.config.width, x + radius + 1))
        ]
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def _new_trial(self) -> None:
        self.bit_a = self._random.randint(0, 1)
        self.bit_b = self._random.randint(0, 1)
        self.last_prediction = None

    @property
    def target(self) -> int:
        return self.bit_a ^ self.bit_b

    @property
    def accuracy(self) -> float:
        return sum(self._results) / len(self._results) if self._results else 0.0

    def observe(self, tick: int, activation: torch.Tensor) -> TaskObservation:
        """Return the current cue and evaluate the response at the phase boundary."""

        phase_tick = tick % self.config.trial_ticks
        if phase_tick == 0 and tick > 0:
            self._new_trial()

        stimulus = torch.zeros(self.config.cell_count, device=self.device)
        reward = 0.0
        cue_end = self.config.cue_ticks
        delay_end = cue_end + self.config.delay_ticks
        response_end = delay_end + self.config.response_ticks

        if phase_tick < cue_end:
            phase = "cue"
            stimulus[self.sensor_a] = 1.25 if self.bit_a else -0.85
            stimulus[self.sensor_b] = 1.25 if self.bit_b else -0.85
        elif phase_tick < delay_end:
            phase = "delay"
        elif phase_tick < response_end:
            phase = "response"
            if phase_tick == response_end - 1:
                zero_value = activation[self.motor_zero].mean()
                one_value = activation[self.motor_one].mean()
                self.last_prediction = int(one_value > zero_value)
                correct = self.last_prediction == self.target
                self._results.append(float(correct))
                reward = 1.0 if correct else -0.25
        else:
            phase = "rest"

        return TaskObservation(
            stimulus=stimulus,
            reward=reward,
            phase=phase,
            bit_a=self.bit_a,
            bit_b=self.bit_b,
            target=self.target,
            prediction=self.last_prediction,
            accuracy=self.accuracy,
        )
