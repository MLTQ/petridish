"""GPU telemetry, run discovery, and bounded trainer process control."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any
import uuid

from .sequence_cells import CELL_ARCHITECTURES
from .sequence_experiment import MAX_STATE_LANES
from .sequence_tasks import STREAM_MODES
from .token_corpus_task import TOKENIZER_PROFILES
from .lifecycle_profiles import LIFECYCLE_PROFILES, resolve_lifecycle_profile
from .topology_profiles import (
    TOPOLOGY_PROFILES,
    resolve_topology_profile,
    topology_mutates,
)


RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
GPU_QUERY = (
    "index,name,uuid,pci.bus_id,memory.total,memory.used,"
    "utilization.gpu,power.draw"
)


@dataclass(frozen=True, slots=True)
class LaunchSpec:
    """Validated arguments for one unattended Tiny Shakespeare run."""

    run_id: str
    gpu_uuid: str
    task: str = "tiny_shakespeare"
    architecture: str = "gru"
    field_size: int = 68
    batch_size: int = 16
    context_length: int = 64
    vocabulary_size: int = 2_048
    tokenizer_profile: str = "wordpiece"
    stream_mode: str = "continuous"
    state_retention: float = 0.9
    state_lanes: int = 1
    message_steps: int = 2
    broadcast_gain: float = 0.3
    updates: int = 100_000
    seed: int = 1
    learning_rate_scale: float = 1.0
    amp: str = "bfloat16"
    lifecycle: bool = False
    lifecycle_profile: str = "off"
    structure: bool = True
    topology_profile: str | None = None


@dataclass(frozen=True, slots=True)
class ContinueSpec:
    """Validated plasticity phase for one existing organism lineage."""

    run_id: str
    gpu_uuid: str
    additional_updates: int = 1_000
    lifecycle: bool = False
    lifecycle_profile: str = "off"
    structure: bool = True
    topology_profile: str | None = None
    phase_name: str | None = None
    training_shard_tokens: int | None = None
    state_lanes: int | None = None


@dataclass(frozen=True, slots=True)
class ForkSpec:
    """Create a counterfactual branch from one stopped organism checkpoint."""

    source_run_id: str
    fork_run_id: str


@dataclass(frozen=True, slots=True)
class RetrySpec:
    """Restart a failed trainer from its own unchanged checkpoint and phase."""

    run_id: str
    gpu_uuid: str


@dataclass(frozen=True, slots=True)
class EvaluateSpec:
    """Read-only held-out evaluation for one checkpointed organism."""

    run_id: str
    gpu_uuid: str
    state_horizons: bool = False
    evaluation_split: str = "validation"
    trajectory_lane: int | None = None


class Laboratory:
    """Expose measured hardware/runs and supervise explicitly enabled jobs."""

    def __init__(
        self,
        repository_root: Path,
        *,
        run_root: Path | None = None,
        benchmark_root: Path | None = None,
        control_enabled: bool = False,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.run_root = (run_root or self.repository_root / "runs").resolve()
        self.benchmark_root = (
            benchmark_root or self.repository_root / "benchmarks" / "lab"
        ).resolve()
        self.control_enabled = control_enabled
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._logs: dict[str, Any] = {}

    def snapshot(self) -> dict[str, Any]:
        """Return current GPU telemetry, run summaries, and server capabilities."""

        gpus, gpu_processes = self._gpu_snapshot()
        active_runs = self._active_run_processes(gpu_processes)
        return {
            "controlEnabled": self.control_enabled,
            "capabilities": {
                "tasks": ["tiny_shakespeare", "tiny_stories"],
                "architectures": list(CELL_ARCHITECTURES),
                "ampModes": ["off", "bfloat16"],
                "lifecycleProfiles": list(LIFECYCLE_PROFILES),
                "topologyProfiles": list(TOPOLOGY_PROFILES),
                "tokenizerProfiles": list(TOKENIZER_PROFILES),
                "checkpointEvaluation": True,
                "trainingShardAudit": True,
                "trainingShardCurriculum": True,
                "stateLaneExpansion": True,
                "stateLaneDomains": True,
                "maximumStateLanes": MAX_STATE_LANES,
                "phaseBalancedLaneExpansion": True,
                "trajectoryLaneAudit": True,
                "checkpointFork": True,
                "sameLineageRetry": True,
            },
            "gpus": gpus,
            "runs": self._discover_runs(active_runs),
            "benchmarks": self._discover_benchmarks(),
            "timestamp": time.time(),
        }

    def metrics(self, run_id: str, *, limit: int = 600) -> list[dict[str, Any]]:
        """Read a bounded tail of append-only JSONL metrics for one local run."""

        directory = self._run_directory(run_id)
        path = directory / "metrics.jsonl"
        if not path.is_file():
            return []
        bounded = max(1, min(2_000, int(limit)))
        lines = path.read_text(encoding="utf-8").splitlines()[-bounded:]
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records

    def launch(self, spec: LaunchSpec) -> dict[str, Any]:
        """Start one trainer pinned to a GPU UUID after strict validation."""

        if not self.control_enabled:
            raise PermissionError("laboratory process control is disabled")
        self._validate_spec(spec)
        available = {gpu["uuid"] for gpu in self._gpu_snapshot()[0]}
        if spec.gpu_uuid not in available:
            raise ValueError("unknown GPU UUID")
        directory = self._run_directory(spec.run_id)
        if directory.exists():
            raise ValueError(f"run already exists: {spec.run_id}")
        directory.mkdir(parents=True)
        organism_id = f"organism-{uuid.uuid4().hex}"
        lifecycle_profile = resolve_lifecycle_profile(
            spec.lifecycle_profile, enabled=spec.lifecycle
        )
        topology_profile = resolve_topology_profile(
            spec.topology_profile, structure=spec.structure
        )
        structure_enabled = topology_mutates(topology_profile)
        phase_name = self._phase_name(topology_profile, lifecycle_profile)
        command = self._trainer_command(
            spec, directory, organism_id=organism_id, phase_name=phase_name
        )
        lifecycle_enabled = lifecycle_profile != "off"
        manifest = {
            "version": 2,
            "runId": spec.run_id,
            "organismId": organism_id,
            "task": spec.task,
            "architecture": spec.architecture,
            "gpuUuid": spec.gpu_uuid,
            "configuration": {
                "fieldSize": spec.field_size,
                "batchSize": spec.batch_size,
                "contextLength": spec.context_length,
                "vocabularySize": spec.vocabulary_size,
                "tokenizerProfile": (
                    spec.tokenizer_profile if spec.task == "tiny_stories" else "character"
                ),
                "streamMode": spec.stream_mode,
                "stateRetention": spec.state_retention,
                "stateLanes": spec.state_lanes,
                "messageSteps": spec.message_steps,
                "broadcastGain": spec.broadcast_gain,
                "updates": spec.updates,
                "seed": spec.seed,
                "learningRateScale": spec.learning_rate_scale,
                "amp": spec.amp,
                "lifecycle": lifecycle_enabled,
                "lifecycleProfile": lifecycle_profile,
                "structure": structure_enabled,
                "topologyProfile": topology_profile,
            },
            "createdAt": time.time(),
            "phaseHistory": [
                {
                    "index": 0,
                    "name": phase_name,
                    "startUpdate": 0,
                    "targetUpdate": spec.updates,
                    "structure": structure_enabled,
                    "topologyProfile": topology_profile,
                    "lifecycleProfile": lifecycle_profile,
                    "startedAt": time.time(),
                }
            ],
            "commit": self._git_commit(),
            "command": command,
        }
        self._write_json(directory / "manifest.json", manifest)
        process = self._start_process(spec.run_id, spec.gpu_uuid, command, directory)
        manifest["pid"] = process.pid
        self._write_json(directory / "manifest.json", manifest)
        return {"runId": spec.run_id, "pid": process.pid, "status": "running"}

    def continue_run(self, spec: ContinueSpec) -> dict[str, Any]:
        """Continue one checkpoint lineage under a new plasticity policy."""

        if not self.control_enabled:
            raise PermissionError("laboratory process control is disabled")
        if spec.additional_updates < 1:
            raise ValueError("additional updates must be positive")
        if (
            spec.state_lanes is not None
            and not 1 <= spec.state_lanes <= MAX_STATE_LANES
        ):
            raise ValueError(
                f"state lanes must be between one and {MAX_STATE_LANES}"
            )
        if spec.lifecycle_profile not in LIFECYCLE_PROFILES:
            raise ValueError("unknown lifecycle profile")
        topology_profile = resolve_topology_profile(
            spec.topology_profile, structure=spec.structure
        )
        structure_enabled = topology_mutates(topology_profile)
        available = {gpu["uuid"] for gpu in self._gpu_snapshot()[0]}
        if spec.gpu_uuid not in available:
            raise ValueError("unknown GPU UUID")
        directory = self._run_directory(spec.run_id)
        manifest = self._read_json(directory / "manifest.json")
        if spec.training_shard_tokens is not None:
            if manifest.get("task") != "tiny_stories":
                raise ValueError("training shards are available only for TinyStories")
            context_length = int(
                dict(manifest.get("configuration", {})).get("contextLength", 64)
            )
            if (
                spec.training_shard_tokens != 0
                and spec.training_shard_tokens <= context_length + 1
            ):
                raise ValueError("training shard must exceed one complete context window")
        if not (directory / "latest.pt").is_file():
            raise ValueError("run has no checkpoint to continue")
        pid = int(manifest.get("pid", 0) or 0)
        if pid > 0 and self._pid_alive(pid):
            raise ValueError("organism is already running")
        records = self.metrics(spec.run_id, limit=2_000)
        latest_train = next(
            (record for record in reversed(records) if record.get("type") == "train"),
            None,
        )
        latest_diagnostic = next(
            (
                record
                for record in reversed(records)
                if record.get("type") == "diagnostic"
            ),
            None,
        )
        start_update = int((latest_train or {}).get("update", 0))
        target_update = start_update + spec.additional_updates
        organism_id = str(manifest.get("organismId") or f"organism-{uuid.uuid4().hex}")
        history = list(manifest.get("phaseHistory") or [])
        if not history:
            configuration = dict(manifest.get("configuration", {}))
            history.append(
                {
                    "index": 0,
                    "name": "legacy training",
                    "startUpdate": 0,
                    "targetUpdate": start_update,
                    "structure": bool(configuration.get("structure", False)),
                    "topologyProfile": str(
                        configuration.get(
                            "topologyProfile",
                            "adaptive" if configuration.get("structure") else "fixed",
                        )
                    ),
                    "lifecycleProfile": str(configuration.get("lifecycleProfile", "off")),
                    "startedAt": manifest.get("createdAt"),
                }
            )
        observed_phase_indices = [
            int(record.get("phaseIndex", 0) or 0)
            for record in records
            if record.get("phaseIndex") is not None
        ]
        phase_index = max(
            max(int(phase.get("index", 0)) for phase in history),
            max(observed_phase_indices, default=0),
        ) + 1
        profile = resolve_lifecycle_profile(
            spec.lifecycle_profile, enabled=spec.lifecycle
        )
        configuration = dict(manifest.get("configuration", {}))
        previous_state_lanes = int(
            (latest_train or {}).get(
                "stateLanes", configuration.get("stateLanes", 1)
            ) or 1
        )
        state_lanes = (
            previous_state_lanes if spec.state_lanes is None else spec.state_lanes
        )
        if state_lanes < previous_state_lanes:
            raise ValueError("state-lane continuation cannot discard existing lanes")
        latest_shard_tokens = (
            (latest_train or {}).get("trainingShardTokens")
            if "trainingShardTokens" in (latest_train or {})
            else configuration.get("trainingShardTokens", 0)
        )
        previous_shard_tokens = int(latest_shard_tokens or 0)
        training_shard_tokens = (
            previous_shard_tokens
            if spec.training_shard_tokens is None
            else spec.training_shard_tokens
        )
        if spec.training_shard_tokens is not None:
            if previous_shard_tokens == 0 and training_shard_tokens != 0:
                raise ValueError(
                    "curriculum cannot shrink a full-stream lane domain"
                )
            if (
                previous_shard_tokens > 0
                and training_shard_tokens > 0
                and training_shard_tokens < previous_shard_tokens
            ):
                raise ValueError(
                    "curriculum cannot shrink a preserved lane stream domain"
                )
            breadth_expands = (
                previous_shard_tokens > 0
                and (
                    training_shard_tokens == 0
                    or training_shard_tokens > previous_shard_tokens
                )
            )
            if breadth_expands and state_lanes == previous_state_lanes:
                raise ValueError(
                    "curriculum breadth expansion requires appending state lanes"
                )
        phase_name = spec.phase_name or self._phase_name(topology_profile, profile)
        if spec.training_shard_tokens is not None and spec.phase_name is None:
            curriculum = (
                "full-stream curriculum"
                if training_shard_tokens == 0
                else f"{training_shard_tokens:,}-token repeated shard"
            )
            phase_name = f"{curriculum} + {phase_name}"
        command = self._continuation_command(
            directory,
            target_update=target_update,
            organism_id=organism_id,
            phase_index=phase_index,
            phase_name=phase_name,
            structure=structure_enabled,
            topology_profile=topology_profile,
            lifecycle_profile=profile,
            training_shard_tokens=spec.training_shard_tokens,
            state_lanes=spec.state_lanes,
        )
        phase = {
            "index": phase_index,
            "name": phase_name,
            "startUpdate": start_update,
            "targetUpdate": target_update,
            "structure": structure_enabled,
            "topologyProfile": topology_profile,
            "lifecycleProfile": profile,
            "trainingShardTokens": training_shard_tokens,
            "stateLanes": state_lanes,
            "startGrownEdges": int(
                (latest_diagnostic or {}).get("cumulativeGrownEdges", 0)
            ),
            "startPrunedEdges": int(
                (latest_diagnostic or {}).get("cumulativePrunedEdges", 0)
            ),
            "startBirths": int((latest_diagnostic or {}).get("cumulativeBirths", 0)),
            "startDeaths": int((latest_diagnostic or {}).get("cumulativeDeaths", 0)),
            "startedAt": time.time(),
        }
        history.append(phase)
        configuration.update(
            {
                "updates": target_update,
                "lifecycle": profile != "off",
                "lifecycleProfile": profile,
                "structure": structure_enabled,
                "topologyProfile": topology_profile,
                "trainingShardTokens": training_shard_tokens,
                "stateLanes": state_lanes,
            }
        )
        manifest.update(
            {
                "version": 2,
                "organismId": organism_id,
                "gpuUuid": spec.gpu_uuid,
                "configuration": configuration,
                "phaseHistory": history,
                "commit": self._git_commit(),
                "command": command,
            }
        )
        self._write_json(directory / "manifest.json", manifest)
        self._append_jsonl(
            directory / "metrics.jsonl",
            {
                "type": "phase",
                "update": start_update,
                "organismId": organism_id,
                "phaseIndex": phase_index,
                "phaseName": phase_name,
                "structure": structure_enabled,
                "topologyProfile": topology_profile,
                "lifecycleProfile": profile,
                "trainingShardTokens": training_shard_tokens,
                "stateLanes": state_lanes,
                "startGrownEdges": phase["startGrownEdges"],
                "startPrunedEdges": phase["startPrunedEdges"],
                "startBirths": phase["startBirths"],
                "startDeaths": phase["startDeaths"],
                "timestamp": time.time(),
            },
        )
        process = self._start_process(spec.run_id, spec.gpu_uuid, command, directory)
        manifest["pid"] = process.pid
        self._write_json(directory / "manifest.json", manifest)
        return {
            "runId": spec.run_id,
            "organismId": organism_id,
            "phaseIndex": phase_index,
            "pid": process.pid,
            "status": "running",
        }

    def fork_run(self, spec: ForkSpec) -> dict[str, Any]:
        """Clone one stopped checkpoint into a traceable counterfactual branch."""

        if not self.control_enabled:
            raise PermissionError("laboratory process control is disabled")
        source = self._run_directory(spec.source_run_id)
        destination = self._run_directory(spec.fork_run_id)
        manifest = self._read_json(source / "manifest.json")
        checkpoint = source / "latest.pt"
        if not checkpoint.is_file():
            raise ValueError("run has no checkpoint to fork")
        pid = int(manifest.get("pid", 0) or 0)
        if pid > 0 and self._pid_alive(pid):
            raise ValueError("organism is already running")
        if destination.exists():
            raise ValueError(f"run already exists: {spec.fork_run_id}")
        organism_id = str(manifest.get("organismId") or "")
        if not organism_id:
            raise ValueError("checkpoint manifest has no organism lineage ID")

        records = self.metrics(spec.source_run_id, limit=2_000)
        latest_train = next(
            (record for record in reversed(records) if record.get("type") == "train"),
            {},
        )
        update = int(latest_train.get("update", 0) or 0)
        checkpoint_sha256 = self._file_sha256(checkpoint)
        parent_checkpoint = {
            "runId": spec.source_run_id,
            "update": update,
            "sha256": checkpoint_sha256,
        }
        temporary = self.run_root / f".{spec.fork_run_id}.fork-{uuid.uuid4().hex}"
        temporary.mkdir(parents=True)
        try:
            shutil.copy2(checkpoint, temporary / "latest.pt")
            if self._file_sha256(temporary / "latest.pt") != checkpoint_sha256:
                raise OSError("forked checkpoint failed SHA-256 verification")
            metrics_path = source / "metrics.jsonl"
            if metrics_path.is_file():
                shutil.copy2(metrics_path, temporary / "metrics.jsonl")
            branch_manifest = dict(manifest)
            branch_manifest.update(
                {
                    "runId": spec.fork_run_id,
                    "pid": 0,
                    "command": [],
                    "commit": self._git_commit(),
                    "parentCheckpoint": parent_checkpoint,
                    "branchRootRunId": manifest.get(
                        "branchRootRunId", spec.source_run_id
                    ),
                    "branchDepth": int(manifest.get("branchDepth", 0) or 0) + 1,
                    "branchedAt": time.time(),
                }
            )
            self._write_json(temporary / "manifest.json", branch_manifest)
            self._append_jsonl(
                temporary / "metrics.jsonl",
                {
                    "type": "branch",
                    "update": update,
                    "organismId": organism_id,
                    "sourceRunId": spec.source_run_id,
                    "forkRunId": spec.fork_run_id,
                    "checkpointSha256": checkpoint_sha256,
                    "timestamp": time.time(),
                },
            )
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {
            "runId": spec.fork_run_id,
            "organismId": organism_id,
            "parentCheckpoint": parent_checkpoint,
            "status": "checkpointed",
        }

    def retry_run(self, spec: RetrySpec) -> dict[str, Any]:
        """Resume a failed phase from its own atomic checkpoint, without mutation."""

        if not self.control_enabled:
            raise PermissionError("laboratory process control is disabled")
        available = {gpu["uuid"] for gpu in self._gpu_snapshot()[0]}
        if spec.gpu_uuid not in available:
            raise ValueError("unknown GPU UUID")
        directory = self._run_directory(spec.run_id)
        manifest_path = directory / "manifest.json"
        manifest = self._read_json(manifest_path)
        checkpoint = directory / "latest.pt"
        if not checkpoint.is_file():
            raise ValueError("run has no checkpoint to retry")
        pid = int(manifest.get("pid", 0) or 0)
        if pid > 0 and self._pid_alive(pid):
            raise ValueError("organism is already running")

        records = self.metrics(spec.run_id, limit=2_000)
        last_boundary = max(
            (
                index for index, record in enumerate(records)
                if record.get("type") in {"phase", "retry"}
            ),
            default=-1,
        )
        if not any(
            record.get("type") == "failure" for record in records[last_boundary + 1 :]
        ):
            raise ValueError("run has no unrecovered trainer failure")

        raw_command = manifest.get("command")
        if not isinstance(raw_command, list) or not all(
            isinstance(argument, str) for argument in raw_command
        ):
            raise ValueError("failed run has no valid trainer command")
        command = list(raw_command)
        if (
            len(command) < 3
            or "petridish.train_shakespeare" not in command
            or "--resume" not in command
            or "--resume-plasticity" not in command
            or "--no-resume" in command
            or "--evaluate-only" in command
        ):
            raise ValueError("retry command is not a same-lineage training continuation")
        try:
            checkpoint_directory = Path(
                command[command.index("--checkpoint-dir") + 1]
            ).resolve()
        except (ValueError, IndexError):
            raise ValueError("retry command has no checkpoint directory") from None
        if checkpoint_directory != directory:
            raise ValueError("retry command points at a different organism directory")
        organism_id = str(manifest.get("organismId") or "")
        try:
            command_organism_id = command[command.index("--organism-id") + 1]
        except (ValueError, IndexError):
            raise ValueError("retry command has no organism lineage ID") from None
        if not organism_id or command_organism_id != organism_id:
            raise ValueError("retry command does not match organism lineage")

        checkpoint_sha256 = self._file_sha256(checkpoint)
        process = self._start_process(
            spec.run_id, spec.gpu_uuid, command, directory
        )
        retry_at = time.time()
        retry_count = int(manifest.get("retryCount", 0) or 0) + 1
        manifest.update(
            {
                "pid": process.pid,
                "gpuUuid": spec.gpu_uuid,
                "commit": self._git_commit(),
                "lastRetryAt": retry_at,
                "retryCount": retry_count,
                "lastRetryCheckpointSha256": checkpoint_sha256,
            }
        )
        self._write_json(manifest_path, manifest)
        self._append_jsonl(
            directory / "metrics.jsonl",
            {
                "type": "retry",
                "update": int(
                    next(
                        (
                            record.get("update", 0) for record in reversed(records)
                            if record.get("type") == "train"
                        ),
                        0,
                    )
                    or 0
                ),
                "organismId": organism_id,
                "checkpointSha256": checkpoint_sha256,
                "retryCount": retry_count,
                "timestamp": retry_at,
            },
        )
        return {
            "runId": spec.run_id,
            "organismId": organism_id,
            "checkpointSha256": checkpoint_sha256,
            "retryCount": retry_count,
            "pid": process.pid,
            "status": "running",
        }

    def stop_run(self, run_id: str) -> dict[str, Any]:
        """Request the trainer's signal-safe checkpoint-and-stop path."""

        if not self.control_enabled:
            raise PermissionError("laboratory process control is disabled")
        directory = self._run_directory(run_id)
        manifest = self._read_json(directory / "manifest.json")
        process = self._processes.get(run_id)
        pid = process.pid if process is not None else int(manifest.get("pid", 0))
        if pid <= 0 or not self._pid_alive(pid):
            return {"runId": run_id, "status": "stopped"}
        os.kill(pid, signal.SIGTERM)
        return {"runId": run_id, "status": "stopping"}

    def evaluate_run(self, spec: EvaluateSpec) -> dict[str, Any]:
        """Evaluate one stopped checkpoint without training or changing its phase."""

        if not self.control_enabled:
            raise PermissionError("laboratory process control is disabled")
        available = {gpu["uuid"] for gpu in self._gpu_snapshot()[0]}
        if spec.gpu_uuid not in available:
            raise ValueError("unknown GPU UUID")
        directory = self._run_directory(spec.run_id)
        manifest = self._read_json(directory / "manifest.json")
        if not (directory / "latest.pt").is_file():
            raise ValueError("run has no checkpoint to evaluate")
        pid = int(manifest.get("pid", 0) or 0)
        if pid > 0 and self._pid_alive(pid):
            raise ValueError("organism is already running")
        history = list(manifest.get("phaseHistory") or [])
        phase = history[-1] if history else {}
        command = self._evaluation_command(
            directory,
            organism_id=manifest.get("organismId"),
            phase_index=int(phase.get("index", 0)),
            phase_name=str(phase.get("name", "training")),
            state_horizons=spec.state_horizons,
            evaluation_split=spec.evaluation_split,
            trajectory_lane=spec.trajectory_lane,
        )
        process = self._start_process(
            spec.run_id, spec.gpu_uuid, command, directory
        )
        manifest.update(
            {
                "pid": process.pid,
                "gpuUuid": spec.gpu_uuid,
                "command": command,
                "commit": self._git_commit(),
                "lastEvaluationRequestedAt": time.time(),
            }
        )
        self._write_json(directory / "manifest.json", manifest)
        return {"runId": spec.run_id, "pid": process.pid, "status": "evaluating"}

    def close(self) -> None:
        """Close server-owned log descriptors without stopping trainers."""

        for log in self._logs.values():
            log.close()
        self._logs.clear()

    def _validate_spec(self, spec: LaunchSpec) -> None:
        if spec.task not in {"tiny_shakespeare", "tiny_stories"}:
            raise ValueError("unknown corpus task")
        if spec.architecture not in CELL_ARCHITECTURES:
            raise ValueError("unknown sequence cell architecture")
        required_field = 68
        if spec.field_size != required_field:
            raise ValueError(f"{spec.task} laboratory runs require a {required_field}×{required_field} field")
        if spec.context_length < 8 or spec.context_length > 256:
            raise ValueError("context length must be between 8 and 256")
        if spec.vocabulary_size not in {64, 128, 256, 512, 1_024, 2_048}:
            raise ValueError("vocabulary size must be a supported power of two from 64 to 2048")
        if spec.task == "tiny_stories":
            if spec.tokenizer_profile not in TOKENIZER_PROFILES:
                raise ValueError("unknown TinyStories tokenizer profile")
            if spec.tokenizer_profile == "byte" and spec.vocabulary_size != 256:
                raise ValueError("byte tokenization requires a 256-token vocabulary")
        if spec.stream_mode not in STREAM_MODES:
            raise ValueError("unknown corpus stream mode")
        if not 0 <= spec.state_retention <= 1:
            raise ValueError("state retention must be between zero and one")
        if not 1 <= spec.state_lanes <= MAX_STATE_LANES:
            raise ValueError(
                f"state lanes must be between one and {MAX_STATE_LANES}"
            )
        if spec.batch_size < 1 or spec.batch_size > 256:
            raise ValueError("batch size must be between 1 and 256")
        if spec.message_steps < 1 or spec.message_steps > 16:
            raise ValueError("message steps must be between 1 and 16")
        if not 0 <= spec.broadcast_gain <= 2.0:
            raise ValueError("broadcast gain must be between 0 and 2")
        if spec.updates < 1:
            raise ValueError("updates must be positive")
        if not 0.01 <= spec.learning_rate_scale <= 1.0:
            raise ValueError("learning-rate scale must be between 0.01 and 1.0")
        if spec.amp not in {"off", "bfloat16"}:
            raise ValueError("unsupported AMP mode")
        if spec.lifecycle_profile not in LIFECYCLE_PROFILES:
            raise ValueError("unknown lifecycle profile")
        resolve_topology_profile(spec.topology_profile, structure=spec.structure)

    def _trainer_command(
        self,
        spec: LaunchSpec,
        directory: Path,
        *,
        organism_id: str | None = None,
        phase_name: str | None = None,
    ) -> list[str]:
        command = [
            sys.executable, "-m", "petridish.train_shakespeare",
            "--task", spec.task,
            "--device", "cuda", "--field-size", str(spec.field_size),
            "--batch-size", str(spec.batch_size),
            "--context-length", str(spec.context_length),
            "--vocabulary-size", str(spec.vocabulary_size),
            "--tokenizer-profile", spec.tokenizer_profile,
            "--stream-mode", spec.stream_mode,
            "--state-retention", str(spec.state_retention),
            "--state-lanes", str(spec.state_lanes),
            "--message-steps", str(spec.message_steps),
            "--broadcast-gain", str(spec.broadcast_gain),
            "--architecture", spec.architecture,
            "--amp", spec.amp, "--compile", "off",
            "--updates", str(spec.updates), "--seed", str(spec.seed),
            "--learning-rate-scale", str(spec.learning_rate_scale),
            "--checkpoint-dir", str(directory), "--checkpoint-interval", "100",
            "--eval-interval", "500", "--eval-batches", "4",
            "--progress-interval", "10", "--no-resume",
        ]
        if organism_id is not None:
            command.extend(("--organism-id", organism_id, "--phase-index", "0"))
        if phase_name is not None:
            command.extend(("--phase-name", phase_name))
        profile = resolve_lifecycle_profile(spec.lifecycle_profile, enabled=spec.lifecycle)
        topology = resolve_topology_profile(
            spec.topology_profile, structure=spec.structure
        )
        command.extend(("--lifecycle-profile", profile))
        command.extend(("--topology-profile", topology))
        command.append("--lifecycle" if profile != "off" else "--no-lifecycle")
        command.append("--structure" if topology_mutates(topology) else "--no-structure")
        return command

    def _continuation_command(
        self,
        directory: Path,
        *,
        target_update: int,
        organism_id: str,
        phase_index: int,
        phase_name: str,
        structure: bool,
        topology_profile: str,
        lifecycle_profile: str,
        training_shard_tokens: int | None,
        state_lanes: int | None,
    ) -> list[str]:
        command = [
            sys.executable, "-m", "petridish.train_shakespeare",
            "--device", "cuda", "--updates", str(target_update),
            "--checkpoint-dir", str(directory), "--checkpoint-interval", "100",
            "--eval-interval", "500", "--eval-batches", "4",
            "--progress-interval", "10", "--resume", "--resume-plasticity",
            "--organism-id", organism_id, "--phase-index", str(phase_index),
            "--phase-name", phase_name,
            "--lifecycle-profile", lifecycle_profile,
            "--topology-profile", topology_profile,
        ]
        command.append("--lifecycle" if lifecycle_profile != "off" else "--no-lifecycle")
        command.append("--structure" if structure else "--no-structure")
        if training_shard_tokens is not None:
            command.extend(("--training-shard-tokens", str(training_shard_tokens)))
        if state_lanes is not None:
            command.extend(("--state-lanes", str(state_lanes)))
        return command

    def _evaluation_command(
        self,
        directory: Path,
        *,
        organism_id: object,
        phase_index: int,
        phase_name: str,
        state_horizons: bool,
        evaluation_split: str,
        trajectory_lane: int | None,
    ) -> list[str]:
        if evaluation_split not in {"validation", "training", "trajectory"}:
            raise ValueError(
                "evaluation split must be validation, training, or trajectory"
            )
        if trajectory_lane is not None and evaluation_split != "trajectory":
            raise ValueError("trajectory lane requires the trajectory evaluation split")
        if (
            trajectory_lane is not None
            and not 0 <= trajectory_lane < MAX_STATE_LANES
        ):
            raise ValueError(
                f"trajectory lane must be between 0 and {MAX_STATE_LANES - 1}"
            )
        command = [
            sys.executable, "-m", "petridish.train_shakespeare",
            "--device", "cuda", "--checkpoint-dir", str(directory),
            "--resume", "--evaluate-only", "--eval-batches", "16",
            "--evaluation-split", evaluation_split,
        ]
        if trajectory_lane is not None:
            command.extend(("--trajectory-lane", str(trajectory_lane)))
        if organism_id:
            command.extend(
                (
                    "--organism-id", str(organism_id),
                    "--phase-index", str(phase_index), "--phase-name", phase_name,
                )
            )
        if state_horizons:
            command.append("--state-horizon-eval")
        return command

    def _start_process(
        self, run_id: str, gpu_uuid: str, command: list[str], directory: Path
    ) -> subprocess.Popen[bytes]:
        previous_log = self._logs.pop(run_id, None)
        if previous_log is not None:
            previous_log.close()
        log = (directory / "trainer.log").open("ab", buffering=0)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        try:
            process = subprocess.Popen(
                command,
                cwd=self.repository_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log.close()
            raise
        self._processes[run_id] = process
        self._logs[run_id] = log
        return process

    @staticmethod
    def _phase_name(topology_profile: str, lifecycle_profile: str) -> str:
        if topology_profile != "fixed" and lifecycle_profile != "off":
            topology = topology_profile.replace("_", "-")
            return f"{topology} topology + {lifecycle_profile} lifecycle"
        if topology_profile != "fixed":
            return f"{topology_profile.replace('_', '-')} topology"
        if lifecycle_profile != "off":
            return f"fixed topology + {lifecycle_profile} lifecycle"
        return "fixed-topology warm-up"

    def _discover_runs(self, active: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.run_root.is_dir():
            return []
        summaries: list[dict[str, Any]] = []
        for directory in sorted(self.run_root.iterdir()):
            if not directory.is_dir():
                continue
            has_metrics = (directory / "metrics.jsonl").is_file()
            has_checkpoint = (directory / "latest.pt").is_file()
            has_manifest = (directory / "manifest.json").is_file()
            if not (has_metrics or has_checkpoint or has_manifest):
                continue
            manifest = self._read_json(directory / "manifest.json")
            records = self.metrics(directory.name, limit=2_000) if has_metrics else []
            latest_phase_offset = max(
                (
                    index for index, record in enumerate(records)
                    if record.get("type") in {"phase", "retry"}
                ),
                default=-1,
            )
            current_phase_records = records[latest_phase_offset + 1 :]
            latest_train = next(
                (record for record in reversed(records) if record.get("type") == "train"), None
            )
            configuration = dict(manifest.get("configuration", {}))
            phase_history = list(manifest.get("phaseHistory", []))
            if latest_train is not None:
                if "stateLanes" in latest_train:
                    configuration["stateLanes"] = int(
                        latest_train.get("stateLanes") or 1
                    )
                if "trainingShardTokens" in latest_train:
                    configuration["trainingShardTokens"] = int(
                        latest_train.get("trainingShardTokens") or 0
                    )
                measured_phase_index = int(latest_train.get("phaseIndex", 0) or 0)
                recorded_phase_index = max(
                    (int(phase.get("index", 0)) for phase in phase_history),
                    default=-1,
                )
                if measured_phase_index > recorded_phase_index:
                    measured_phase_records = [
                        record for record in records
                        if record.get("type") == "train"
                        and int(record.get("phaseIndex", 0) or 0)
                        == measured_phase_index
                    ]
                    first_measured = measured_phase_records[0]
                    topology_profile = str(
                        latest_train.get(
                            "topologyProfile",
                            configuration.get("topologyProfile", "fixed"),
                        )
                    )
                    phase_history.append(
                        {
                            "index": measured_phase_index,
                            "name": latest_train.get("phaseName", "measured phase"),
                            "startUpdate": max(
                                0, int(first_measured.get("update", 0)) - 1
                            ),
                            "targetUpdate": int(latest_train.get("update", 0)),
                            "structure": topology_profile != "fixed",
                            "topologyProfile": topology_profile,
                            "lifecycleProfile": configuration.get(
                                "lifecycleProfile", "off"
                            ),
                            "trainingShardTokens": configuration.get(
                                "trainingShardTokens", 0
                            ),
                            "stateLanes": configuration.get("stateLanes", 1),
                            "startedAt": first_measured.get("timestamp"),
                            "recoveredFromMetrics": True,
                        }
                    )
            latest_held_out = next(
                (record for record in reversed(records) if record.get("type") == "held_out"), None
            )
            latest_training_audit = next(
                (
                    record for record in reversed(records)
                    if record.get("type") == "training_audit"
                ),
                None,
            )
            latest_trajectory_audit = next(
                (
                    record for record in reversed(records)
                    if record.get("type") == "trajectory_audit"
                ),
                None,
            )
            latest_trajectory_audits: list[dict[str, Any]] = []
            seen_trajectory_lanes: set[int | None] = set()
            for record in reversed(records):
                if record.get("type") != "trajectory_audit":
                    continue
                raw_lane = record.get("trajectoryLane")
                lane = int(raw_lane) if isinstance(raw_lane, (int, float)) else None
                if lane in seen_trajectory_lanes:
                    continue
                seen_trajectory_lanes.add(lane)
                latest_trajectory_audits.append(record)
            latest_trajectory_audits.sort(
                key=lambda record: int(record.get("trajectoryLane") or 0)
            )
            latest_diagnostics = next(
                (record for record in reversed(records) if record.get("type") == "diagnostic"), None
            )
            latest_failure = next(
                (
                    record for record in reversed(current_phase_records)
                    if record.get("type") == "failure"
                ),
                None,
            )
            running = active.get(directory.name)
            manifest_pid = int(manifest.get("pid", 0) or 0)
            if running is None and manifest_pid > 0 and self._pid_alive(manifest_pid):
                running = {"pid": manifest_pid, "gpuUuid": manifest.get("gpuUuid")}
            finite = not latest_train or all(
                not isinstance(latest_train.get(key), (int, float))
                or math.isfinite(float(latest_train[key]))
                for key in ("loss", "rollingLoss")
            )
            summaries.append(
                {
                    "id": directory.name,
                    "task": manifest.get("task", "tiny_shakespeare"),
                    "architecture": manifest.get("architecture", "gru"),
                    "gpuUuid": (running or {}).get("gpuUuid", manifest.get("gpuUuid")),
                    "pid": (running or {}).get("pid", manifest_pid or None),
                    "status": (
                        "running" if running else "failed" if latest_failure or not finite
                        else "checkpointed" if has_checkpoint else "stopped"
                    ),
                    "configuration": configuration,
                    "organismId": manifest.get("organismId"),
                    "phaseHistory": phase_history,
                    "parentCheckpoint": manifest.get("parentCheckpoint"),
                    "branchRootRunId": manifest.get("branchRootRunId"),
                    "branchDepth": manifest.get("branchDepth", 0),
                    "commit": manifest.get("commit"),
                    "latestTrain": latest_train,
                    "latestHeldOut": latest_held_out,
                    "latestTrainingAudit": latest_training_audit,
                    "latestTrajectoryAudit": latest_trajectory_audit,
                    "latestTrajectoryAudits": latest_trajectory_audits,
                    "latestDiagnostics": latest_diagnostics,
                    "latestFailure": latest_failure,
                    "hasCheckpoint": has_checkpoint,
                }
            )
        return summaries

    def _discover_benchmarks(self) -> list[dict[str, Any]]:
        """Return bounded, measured stepping-stone benchmark artifacts."""

        if not self.benchmark_root.is_dir():
            return []
        artifacts = sorted(
            self.benchmark_root.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )[:100]
        summaries: list[dict[str, Any]] = []
        for path in artifacts:
            payload = self._read_json(path)
            checkpoints = payload.get("checkpoints")
            if not isinstance(checkpoints, list):
                continue
            valid_checkpoints = [
                checkpoint for checkpoint in checkpoints
                if isinstance(checkpoint, dict)
                and isinstance(checkpoint.get("update"), int)
                and isinstance(checkpoint.get("heldOutAccuracy"), (int, float))
            ][:2_000]
            status = payload.get("status", "complete")
            if not valid_checkpoints and status not in {"running", "failed"}:
                continue
            summaries.append(
                {
                    "id": path.stem,
                    "task": payload.get("task", "unknown"),
                    "profile": payload.get("profile", "unknown"),
                    "architecture": payload.get("architecture", "unknown"),
                    "intervention": payload.get("intervention"),
                    "recallMode": payload.get("recallMode", "adaptive"),
                    "seed": payload.get("seed"),
                    "deterministic": payload.get("deterministic", False),
                    "globalRngMatched": payload.get("globalRngMatched", False),
                    "device": payload.get("device"),
                    "steps": payload.get("steps"),
                    "seconds": payload.get("seconds"),
                    "lesionCount": payload.get("lesionCount", 0),
                    "lesionRadius": payload.get("lesionRadius"),
                    "completedSteps": payload.get(
                        "completedSteps",
                        valid_checkpoints[-1]["update"] if valid_checkpoints else 0,
                    ),
                    "status": status,
                    "failureType": payload.get("failureType"),
                    "failureMessage": payload.get("failureMessage"),
                    "parameterCount": payload.get("parameterCount"),
                    "trainableParameterCount": payload.get("trainableParameterCount"),
                    "cudaAllocatedGiB": payload.get("cudaAllocatedGiB"),
                    "peakCudaAllocatedGiB": payload.get("peakCudaAllocatedGiB"),
                    "bindingDiagnostics": payload.get("bindingDiagnostics"),
                    "livingCells": payload.get("livingCells"),
                    "edgeCount": payload.get("edgeCount"),
                    "minimumOutputHops": payload.get("minimumOutputHops"),
                    "temporallyReachableOutputs": payload.get(
                        "temporallyReachableOutputs"
                    ),
                    "contextReachableOutputs": payload.get("contextReachableOutputs"),
                    "messageSteps": payload.get("messageSteps"),
                    "broadcastGain": payload.get("broadcastGain"),
                    "learningRateScale": payload.get("learningRateScale", 1.0),
                    "outputCount": payload.get("outputCount"),
                    "sequenceLength": payload.get("sequenceLength"),
                    "dependencyTokens": payload.get("dependencyTokens"),
                    "chanceAccuracy": payload.get("chanceAccuracy"),
                    "checkpoints": valid_checkpoints,
                    "artifactMtime": path.stat().st_mtime,
                }
            )
        return summaries

    def _gpu_snapshot(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            output = subprocess.run(
                ["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=3,
            ).stdout
        except (FileNotFoundError, subprocess.SubprocessError):
            return [], []
        gpus: list[dict[str, Any]] = []
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 8:
                continue
            gpus.append(
                {
                    "index": int(fields[0]), "name": fields[1], "uuid": fields[2],
                    "pciBusId": fields[3], "memoryTotalMiB": float(fields[4]),
                    "memoryUsedMiB": float(fields[5]), "utilizationPercent": float(fields[6]),
                    "powerWatts": float(fields[7]), "processes": [],
                }
            )
        try:
            process_output = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=3,
            ).stdout
        except subprocess.SubprocessError:
            process_output = ""
        processes: list[dict[str, Any]] = []
        by_uuid = {gpu["uuid"]: gpu for gpu in gpus}
        for line in process_output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3 or not fields[1].isdigit():
                continue
            process = {
                "gpuUuid": fields[0], "pid": int(fields[1]),
                "memoryMiB": float(fields[2]),
            }
            processes.append(process)
            if fields[0] in by_uuid:
                by_uuid[fields[0]]["processes"].append(process)
        return gpus, processes

    def _active_run_processes(
        self, gpu_processes: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        active: dict[str, dict[str, Any]] = {}
        for process in gpu_processes:
            pid = int(process["pid"])
            try:
                command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except (OSError, UnicodeDecodeError):
                continue
            match = re.search(r"--checkpoint-dir\s+([^\s]+)", command)
            if match is None:
                continue
            directory = Path(match.group(1))
            active[directory.name] = {"pid": pid, "gpuUuid": process["gpuUuid"]}
        return active

    def _run_directory(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("run id must contain lowercase letters, digits, and hyphens")
        directory = (self.run_root / run_id).resolve()
        if directory.parent != self.run_root:
            raise ValueError("run path escapes laboratory root")
        return directory

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _git_commit(self) -> str | None:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=self.repository_root,
                check=True, capture_output=True, text=True, timeout=3,
            ).stdout.strip()
        except subprocess.SubprocessError:
            return None

    @staticmethod
    def _pid_alive(pid: int, *, proc_root: Path = Path("/proc")) -> bool:
        try:
            process_stat = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
        except OSError:
            process_stat = ""
        closing_parenthesis = process_stat.rfind(")")
        if closing_parenthesis >= 0:
            fields = process_stat[closing_parenthesis + 1 :].split()
            if fields and fields[0] == "Z":
                return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


__all__ = ["ContinueSpec", "EvaluateSpec", "ForkSpec", "Laboratory", "LaunchSpec"]
