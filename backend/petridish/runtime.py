"""Asynchronous experiment control and observer broadcasting."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from fastapi import WebSocket

from .mnist_experiment import MnistExperiment
from .mnist_hyperparameters import configured
from .protocol import build_snapshot
from .sequence_experiment import SequenceExperiment


LiveExperiment = MnistExperiment | SequenceExperiment


class ExperimentRuntime:
    """Run the MNIST organism independently from connected observers."""

    def __init__(self, *, seed: int = 1, device: str = "auto") -> None:
        self.seed = seed
        self.device = device
        self.experiments: dict[str, LiveExperiment] = {
            "mnist": MnistExperiment(seed=seed, device=device)
        }
        self.experiment: LiveExperiment = self.experiments["mnist"]
        self.experiment_name = "mnist"
        self.running = True
        self.steps_per_frame = 1
        self.frame_rate = 15
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._loop_task: asyncio.Task[None] | None = None

    @property
    def simulation(self) -> LiveExperiment:
        """Backward-compatible name for the selected live experiment."""

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
                    replacement: LiveExperiment = MnistExperiment(
                        current.config, layout=current.layout, seed=seed, device=device,
                        train_dataset=current.train_dataset, test_dataset=current.test_dataset,
                    )
                else:
                    replacement = SequenceExperiment(
                        current.task, current.config, seed=seed, device=device
                    )
                self.experiment = replacement
                self.experiments[current.experiment_name] = replacement
            elif command == "experiment":
                name = str(message.get("name", "mnist"))
                if name not in {"mnist", "associative_recall", "tiny_language"}:
                    raise ValueError(f"unknown experiment: {name}")
                if name not in self.experiments:
                    self.experiments[name] = SequenceExperiment(
                        name, seed=self.seed, device=str(self.experiment.device)
                    )
                self.experiment = self.experiments[name]
                self.experiment_name = name
            elif command == "speed":
                self.steps_per_frame = max(1, min(64, int(message.get("steps", 2))))
            elif command == "lesion":
                self.experiment.lesion(
                    float(message.get("x", 0)),
                    float(message.get("y", 0)),
                    max(0.5, min(12.0, float(message.get("radius", 2.5)))),
                )
            elif command == "evaluate":
                self.experiment.evaluate(int(message.get("batches", 5)))
            elif command == "lifecycle":
                self.experiment.lifecycle_now()
            elif command == "configure":
                current = self.experiment
                values = message.get("values", {})
                if not isinstance(values, dict):
                    raise ValueError("hyperparameter values must be an object")
                next_config = configured(current.config, values)
                if isinstance(current, MnistExperiment):
                    replacement = MnistExperiment(
                        next_config, layout=current.layout, seed=current.seed,
                        device=str(current.device), train_dataset=current.train_dataset,
                        test_dataset=current.test_dataset,
                    )
                else:
                    replacement = SequenceExperiment(
                        current.task, next_config, seed=current.seed,
                        device=str(current.device),
                    )
                self.experiment = replacement
                self.experiments[current.experiment_name] = replacement
            else:
                raise ValueError(f"unknown command: {command}")
            snapshot = build_snapshot(self.experiment)
        await self.broadcast(snapshot)
