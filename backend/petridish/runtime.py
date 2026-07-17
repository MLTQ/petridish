"""Asynchronous experiment control and observer broadcasting."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from fastapi import WebSocket

from .mnist_experiment import MnistExperiment
from .protocol import build_snapshot
from .simulation import PetriDishSimulation


Experiment = PetriDishSimulation | MnistExperiment


class ExperimentRuntime:
    """Run simulation ticks independently from connected WebSocket observers."""

    def __init__(self, *, seed: int = 1, device: str = "auto") -> None:
        xor = PetriDishSimulation(seed=seed, device=device)
        self.experiments: dict[str, Experiment] = {"xor": xor}
        self.experiment_name = "xor"
        self.running = True
        self.steps_per_frame = 2
        self.frame_rate = 15
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop_task: asyncio.Task[None] | None = None

    @property
    def experiment(self) -> Experiment:
        return self.experiments[self.experiment_name]

    @property
    def simulation(self) -> Experiment:
        """Backward-compatible name for the currently selected experiment."""

        return self.experiment

    async def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._run_loop(), name="petridish-runtime")

    async def stop(self) -> None:
        if self._loop_task is None:
            return
        self._loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._loop_task
        self._loop_task = None

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)
        await websocket.send_json(build_snapshot(self.experiment))

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def _run_loop(self) -> None:
        while True:
            started = asyncio.get_running_loop().time()
            if self.running:
                async with self._lock:
                    self.experiment.step(self.steps_per_frame)
                    snapshot = build_snapshot(self.experiment)
                await self.broadcast(snapshot)
            elapsed = asyncio.get_running_loop().time() - started
            await asyncio.sleep(max(0.001, 1 / self.frame_rate - elapsed))

    async def broadcast(self, snapshot: dict[str, Any] | None = None) -> None:
        if not self._clients:
            return
        payload = snapshot or build_snapshot(self.experiment)
        disconnected: list[WebSocket] = []
        for client in tuple(self._clients):
            try:
                await client.send_json(payload)
            except Exception:
                disconnected.append(client)
        for client in disconnected:
            self.disconnect(client)

    async def handle_command(self, message: dict[str, Any]) -> None:
        """Validate and apply one viewer command to the authoritative state."""

        command = message.get("type")
        async with self._lock:
            if command == "play":
                self.running = True
            elif command == "pause":
                self.running = False
            elif command == "step":
                self.experiment.step(max(1, min(100, int(message.get("count", 1)))))
            elif command == "reset":
                seed = int(message.get("seed", self.experiment.seed))
                device = str(self.experiment.device)
                current = self.experiment
                if isinstance(current, MnistExperiment):
                    self.experiments["mnist"] = MnistExperiment(
                        current.config,
                        seed=seed,
                        device=device,
                        train_dataset=current.train_dataset,
                        test_dataset=current.test_dataset,
                    )
                else:
                    self.experiments["xor"] = PetriDishSimulation(
                        current.config, seed=seed, device=device
                    )
            elif command == "experiment":
                name = str(message.get("name", "xor"))
                if name not in {"xor", "mnist"}:
                    raise ValueError(f"unknown experiment: {name}")
                if name not in self.experiments:
                    current = self.experiment
                    self.experiments[name] = MnistExperiment(
                        seed=current.seed,
                        device=str(current.device),
                    )
                self.experiment_name = name
                self.steps_per_frame = 1 if name == "mnist" else 2
            elif command == "speed":
                self.steps_per_frame = max(1, min(64, int(message.get("steps", 2))))
            elif command == "lesion":
                self.experiment.lesion(
                    float(message.get("x", 0)),
                    float(message.get("y", 0)),
                    max(0.5, min(12.0, float(message.get("radius", 2.5)))),
                )
            elif command == "stimulate":
                if not isinstance(self.experiment, PetriDishSimulation):
                    raise ValueError("manual stimuli are only available in the XOR experiment")
                self.experiment.stimulate(
                    str(message.get("region", "sensor_a")),
                    float(message.get("amount", 1.2)),
                    int(message.get("duration", 16)),
                )
            elif command == "reward":
                if not isinstance(self.experiment, PetriDishSimulation):
                    raise ValueError("manual reward is only available in the XOR experiment")
                self.experiment.inject_reward(float(message.get("amount", 1.0)))
            elif command == "evaluate":
                if not isinstance(self.experiment, MnistExperiment):
                    raise ValueError("evaluation is only available in the MNIST experiment")
                self.experiment.evaluate(int(message.get("batches", 5)))
            elif command == "rewire":
                if not isinstance(self.experiment, MnistExperiment):
                    raise ValueError("new assembly is only available in the MNIST experiment")
                self.experiment.rewire_now()
            else:
                raise ValueError(f"unknown command: {command}")
            snapshot = build_snapshot(self.experiment)
        await self.broadcast(snapshot)
