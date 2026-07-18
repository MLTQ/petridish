"""Asynchronous experiment control and observer broadcasting."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
from fastapi import WebSocket

from .corpus_task import load_tiny_shakespeare_task
from .graph_layout import sequence_layout
from .mnist_config import MnistModelConfig
from .mnist_experiment import MnistExperiment
from .mnist_hyperparameters import configured
from .protocol import build_snapshot
from .sequence_experiment import SequenceExperiment
from .token_corpus_task import load_tiny_stories_task
from .train_shakespeare import load_checkpoint, restore_checkpoint


LiveExperiment = MnistExperiment | SequenceExperiment


def checkpoint_root_from_environment() -> Path:
    """Resolve the shared trainer run catalog used by the laboratory service."""

    configured = os.environ.get("PETRIDISH_RUN_ROOT")
    return (
        Path(configured)
        if configured
        else Path(__file__).resolve().parents[2] / "runs"
    )


class SequenceUpdateInterrupted(RuntimeError):
    """Stop a visual update before its optimizer mutates model state."""


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
        self.checkpoint_root = checkpoint_root_from_environment()
        self.saved_organisms = self._discover_saved_organisms()
        self.experiment_sources: dict[str, str] = {}
        self.running = True
        self.training_mode = False
        self.steps_per_frame = 1
        self.frame_rate = 15
        self.training_report_interval = 1.0
        self.last_compute_seconds = 0.0
        self.training_updates_per_second = 0.0
        self.training_examples_per_second = 0.0
        self.compute_phase = "idle"
        self.compute_progress = 0
        self.compute_total = 0
        self.control_revision = 0
        self._interrupt_requested = False
        self._last_snapshot: dict[str, Any] | None = None
        self._training_updates_since_report = 0
        self._training_compute_since_report = 0.0
        self._last_training_report = 0.0
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._broadcast_lock = asyncio.Lock()
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
        snapshot = self._cached_snapshot()
        if snapshot is None:
            async with self._lock:
                snapshot = await asyncio.to_thread(self._snapshot)
        await websocket.send_json(snapshot)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def _run_loop(self) -> None:
        while True:
            loop = asyncio.get_running_loop()
            started = loop.time()
            snapshot: dict[str, Any] | None = None
            if self.running:
                async with self._lock:
                    compute_started = loop.time()
                    previous_training_step = getattr(
                        self.experiment, "training_step", None
                    )
                    if self.training_mode:
                        assert isinstance(self.experiment, SequenceExperiment)
                        self.compute_phase = "headless"
                        self.compute_progress = 0
                        self.compute_total = 1
                        await asyncio.to_thread(self.experiment.train_updates, 1)
                    elif isinstance(self.experiment, SequenceExperiment):
                        await self._run_sequence_visual_update(loop)
                    else:
                        await asyncio.to_thread(
                            self.experiment.step, self.steps_per_frame
                        )
                    compute_seconds = loop.time() - compute_started
                    if self.training_mode or getattr(
                        self.experiment, "training_step", None
                    ) != previous_training_step:
                        self.last_compute_seconds = compute_seconds
                    if (
                        not self.training_mode
                        and isinstance(self.experiment, SequenceExperiment)
                        and self.experiment.training_step != previous_training_step
                    ):
                        self.training_updates_per_second = 1 / max(1e-9, compute_seconds)
                        self.training_examples_per_second = (
                            self.training_updates_per_second
                            * self.experiment.config.batch_size
                        )
                    if self.training_mode:
                        self._training_updates_since_report += 1
                        self._training_compute_since_report += compute_seconds
                        if loop.time() - self._last_training_report >= self.training_report_interval:
                            self.training_updates_per_second = (
                                self._training_updates_since_report
                                / max(1e-9, self._training_compute_since_report)
                            )
                            self.training_examples_per_second = (
                                self.training_updates_per_second
                                * self.experiment.config.batch_size
                            )
                            snapshot = await asyncio.to_thread(self._snapshot)
                            self._training_updates_since_report = 0
                            self._training_compute_since_report = 0.0
                            self._last_training_report = loop.time()
                    else:
                        self.compute_phase = "idle"
                        self.compute_progress = 0
                        self.compute_total = 0
                        snapshot = await asyncio.to_thread(self._snapshot)
                if snapshot is not None:
                    await self.broadcast(snapshot)
            elapsed = loop.time() - started
            delay = 0.001 if self.training_mode and self.running else max(
                0.001, 1 / self.frame_rate - elapsed
            )
            await asyncio.sleep(delay)

    async def _run_sequence_visual_update(
        self, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Stream sampled, measured states while one optimizer update runs."""

        assert isinstance(self.experiment, SequenceExperiment)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=2)
        last_emitted_at = 0.0
        last_phase = ""

        def enqueue_latest(payload: dict[str, Any]) -> None:
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(payload)

        def progress(phase: str, current: int, total: int) -> None:
            nonlocal last_emitted_at, last_phase
            if self._should_interrupt_sequence_update(phase, current):
                raise SequenceUpdateInterrupted
            phase_changed = phase != last_phase
            last_phase = phase
            self.compute_phase = phase
            self.compute_progress = current
            self.compute_total = total
            if not self._clients:
                return
            now = time.monotonic()
            stride = max(1, self.steps_per_frame)
            sampled_progress = (
                phase not in {"forward", "backward", "evaluation"}
                or current in {1, total}
                or current % stride == 0
            )
            if not sampled_progress:
                return
            if (
                phase != "backward"
                and not phase_changed
                and current != total
                and now - last_emitted_at < 0.08
            ):
                return
            last_emitted_at = now
            payload = self._snapshot()
            loop.call_soon_threadsafe(enqueue_latest, payload)

        task = asyncio.create_task(
            asyncio.to_thread(self.experiment.train_visual_update, progress)
        )
        while not task.done():
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=0.05)
            except TimeoutError:
                continue
            await self.broadcast(payload)
        with suppress(SequenceUpdateInterrupted):
            await task
        await asyncio.sleep(0)
        latest: dict[str, Any] | None = None
        while not queue.empty():
            latest = queue.get_nowait()
        if latest is not None:
            await self.broadcast(latest)

    def _snapshot(self) -> dict[str, Any]:
        """Attach runtime cadence facts to one scientific snapshot."""

        snapshot = build_snapshot(self.experiment)
        snapshot["runtime"] = self._runtime_payload()
        self._last_snapshot = snapshot
        return snapshot

    def _runtime_payload(self) -> dict[str, Any]:
        """Return control metadata without reading mutable model tensors."""

        return {
            "mode": "headless" if self.training_mode else "visualization",
            "running": self.running,
            "stepsPerFrame": self.steps_per_frame,
            "lastComputeSeconds": round(self.last_compute_seconds, 4),
            "trainingUpdatesPerSecond": round(self.training_updates_per_second, 4),
            "trainingExamplesPerSecond": round(self.training_examples_per_second, 4),
            "reportIntervalSeconds": self.training_report_interval,
            "computePhase": self.compute_phase,
            "computeProgress": self.compute_progress,
            "computeTotal": self.compute_total,
            "controlRevision": self.control_revision,
            "savedOrganisms": getattr(self, "saved_organisms", []),
            "loadedOrganism": getattr(self, "experiment_sources", {}).get(
                getattr(self, "experiment_name", "")
            ),
        }

    def _discover_saved_organisms(self) -> list[dict[str, str]]:
        """List local trainer checkpoints without deserializing trusted payloads."""

        if not self.checkpoint_root.is_dir():
            return []
        return [
            {"id": directory.name, "label": directory.name}
            for directory in sorted(self.checkpoint_root.iterdir())
            if (
                directory.is_dir()
                and (directory / "latest.pt").is_file()
                and self._manifest_has_linear_port_banks(directory)
            )
        ]

    @staticmethod
    def _manifest_has_linear_port_banks(directory: Path) -> bool:
        """Hide known legacy corpus runs whose boundary ports wrap into a stripe."""

        manifest = directory / "manifest.json"
        if not manifest.is_file():
            return True
        try:
            payload = json.loads(manifest.read_text())
            task_key = str(payload.get("task", ""))
            field_size = int(payload.get("configuration", {}).get("fieldSize", 0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return True
        return task_key not in {"tiny_shakespeare", "tiny_stories"} or field_size >= 68

    def _load_saved_organism(self, identifier: str) -> SequenceExperiment:
        """Reconstruct one trusted local checkpoint for testing in the viewer."""

        if not identifier or Path(identifier).name != identifier:
            raise ValueError("invalid saved organism identifier")
        root = self.checkpoint_root.resolve()
        checkpoint = (root / identifier / "latest.pt").resolve()
        if checkpoint.parent.parent != root or not checkpoint.is_file():
            raise ValueError(f"unknown saved organism: {identifier}")

        device = torch.device(str(self.experiment.device))
        payload = load_checkpoint(checkpoint, device)
        task_payload = payload.get("task", {})
        if not isinstance(task_payload, dict):
            raise ValueError("checkpoint task metadata is invalid")
        task_key = str(task_payload.get("key", ""))
        if task_key not in {"tiny_shakespeare", "tiny_stories"}:
            raise ValueError(f"unsupported saved organism task: {task_key or 'missing'}")
        context_length = int(task_payload["context_length"])
        task = (
            load_tiny_stories_task(
                context_length, len(tuple(task_payload.get("vocabulary", ()))),
                tokenizer_profile=str(
                    task_payload.get("tokenizer_profile") or "wordpiece"
                ),
            )
            if task_key == "tiny_stories"
            else load_tiny_shakespeare_task(context_length)
        )
        if tuple(task_payload.get("vocabulary", ())) != task.vocabulary:
            raise ValueError("checkpoint vocabulary does not match the cached corpus")

        config_payload = payload.get("configuration")
        if not isinstance(config_payload, dict):
            raise ValueError("checkpoint configuration is invalid")
        config = MnistModelConfig(**config_payload)
        layout = sequence_layout(task_key, len(task.vocabulary))
        required_height = max(layout.input_count, layout.output_count) + 2
        if config.height < required_height:
            raise ValueError(
                f"checkpoint {config.width}×{config.height} geometry wraps its "
                f"{max(layout.input_count, layout.output_count)}-port boundary bank; "
                f"at least {required_height} rows are required for one linear column"
            )
        saved_amp = str(task_payload.get("amp_mode", "off"))
        amp_mode = saved_amp if device.type == "cuda" else "off"
        experiment = SequenceExperiment(
            task,
            config,
            seed=int(task_payload.get("seed", self.seed)),
            device=str(device),
            amp_mode=amp_mode,
        )
        restore_checkpoint(experiment, payload)
        experiment.refresh_visual_trace()
        return experiment

    def _cached_snapshot(self) -> dict[str, Any] | None:
        """Copy the last serialized scientific state with current controls."""

        if self._last_snapshot is None:
            return None
        snapshot = dict(self._last_snapshot)
        snapshot["runtime"] = self._runtime_payload()
        return snapshot

    def _should_interrupt_sequence_update(self, phase: str, current: int) -> bool:
        """Interrupt only before forward/backward work or optimizer mutation."""

        requested = self._interrupt_requested or not self.running
        return requested and (
            phase == "forward"
            or phase == "optimizer"
            or (phase == "backward" and current == 0)
        )

    async def broadcast(self, snapshot: dict[str, Any] | None = None) -> None:
        if not self._clients:
            return
        payload = snapshot or self._snapshot()
        disconnected: list[WebSocket] = []
        async with self._broadcast_lock:
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
        if command in {"play", "pause"}:
            self.running = command == "play"
            self._interrupt_requested = not self.running
            self.control_revision += 1
            snapshot = self._cached_snapshot()
            if snapshot is not None:
                await self.broadcast(snapshot)
            return
        if command == "speed":
            self.steps_per_frame = max(1, min(64, int(message.get("steps", 2))))
            self.control_revision += 1
            snapshot = self._cached_snapshot()
            if snapshot is not None:
                await self.broadcast(snapshot)
            return

        self._interrupt_requested = True
        async with self._lock:
            self._interrupt_requested = False
            if command == "step":
                if self.training_mode:
                    self.training_mode = False
                    await asyncio.to_thread(self.experiment.refresh_visual_trace)
                self.experiment.step(max(1, min(100, int(message.get("count", 1)))))
            elif command == "reset":
                self.training_mode = False
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
                self.experiment_sources.pop(current.experiment_name, None)
            elif command == "experiment":
                self.training_mode = False
                name = str(message.get("name", "mnist"))
                if name not in {
                    "mnist", "associative_recall", "tiny_language", "tiny_shakespeare",
                    "tiny_stories",
                }:
                    raise ValueError(f"unknown experiment: {name}")
                if name not in self.experiments:
                    self.experiments[name] = SequenceExperiment(
                        name, seed=self.seed, device=str(self.experiment.device)
                    )
                self.experiment = self.experiments[name]
                self.experiment_name = name
            elif command == "load":
                identifier = str(message.get("organism", ""))
                try:
                    replacement = await asyncio.to_thread(
                        self._load_saved_organism, identifier
                    )
                except ValueError:
                    raise
                except Exception as error:
                    raise ValueError(
                        f"could not load saved organism {identifier!r}: {error}"
                    ) from error
                self.training_mode = False
                self.running = False
                self.experiment = replacement
                self.experiment_name = replacement.experiment_name
                self.experiments[replacement.experiment_name] = replacement
                self.experiment_sources[replacement.experiment_name] = identifier
            elif command == "training":
                if not isinstance(self.experiment, SequenceExperiment):
                    raise ValueError("headless training requires a sequence experiment")
                enabled = bool(message.get("enabled", True))
                self.training_mode = enabled
                self.running = enabled
                self._training_updates_since_report = 0
                self._training_compute_since_report = 0.0
                self._last_training_report = asyncio.get_running_loop().time()
                if not enabled:
                    await asyncio.to_thread(self.experiment.refresh_visual_trace)
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
            elif command == "prompt":
                if not isinstance(self.experiment, SequenceExperiment):
                    raise ValueError("prompting requires a sequence experiment")
                text = str(message.get("text", ""))[:4_096]
                self.training_mode = False
                self.running = False
                self.experiment.set_prompt(text)
            elif command == "generate":
                if not isinstance(self.experiment, SequenceExperiment):
                    raise ValueError("generation requires a sequence experiment")
                self.training_mode = False
                self.running = False
                self.experiment.generate_token()
            elif command == "configure":
                self.training_mode = False
                current = self.experiment
                values = message.get("values", {})
                if not isinstance(values, dict):
                    raise ValueError("hyperparameter values must be an object")
                next_config = configured(
                    current.config,
                    values,
                    task_key=(current.task.key if isinstance(current, SequenceExperiment) else None),
                )
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
                self.experiment_sources.pop(current.experiment_name, None)
            else:
                raise ValueError(f"unknown command: {command}")
            snapshot = await asyncio.to_thread(self._snapshot)
        await self.broadcast(snapshot)
