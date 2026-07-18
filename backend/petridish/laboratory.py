"""GPU telemetry, run discovery, and bounded trainer process control."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

from .sequence_cells import CELL_ARCHITECTURES
from .lifecycle_profiles import LIFECYCLE_PROFILES, resolve_lifecycle_profile


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
    message_steps: int = 2
    broadcast_gain: float = 0.3
    updates: int = 100_000
    seed: int = 1
    learning_rate_scale: float = 1.0
    amp: str = "bfloat16"
    lifecycle: bool = False
    lifecycle_profile: str = "off"
    structure: bool = True


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
        command = self._trainer_command(spec, directory)
        lifecycle_profile = resolve_lifecycle_profile(
            spec.lifecycle_profile, enabled=spec.lifecycle
        )
        lifecycle_enabled = lifecycle_profile != "off"
        manifest = {
            "version": 1,
            "runId": spec.run_id,
            "task": spec.task,
            "architecture": spec.architecture,
            "gpuUuid": spec.gpu_uuid,
            "configuration": {
                "fieldSize": spec.field_size,
                "batchSize": spec.batch_size,
                "contextLength": spec.context_length,
                "messageSteps": spec.message_steps,
                "broadcastGain": spec.broadcast_gain,
                "updates": spec.updates,
                "seed": spec.seed,
                "learningRateScale": spec.learning_rate_scale,
                "amp": spec.amp,
                "lifecycle": lifecycle_enabled,
                "lifecycleProfile": lifecycle_profile,
                "structure": spec.structure,
            },
            "createdAt": time.time(),
            "commit": self._git_commit(),
            "command": command,
        }
        self._write_json(directory / "manifest.json", manifest)
        log = (directory / "trainer.log").open("ab", buffering=0)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = spec.gpu_uuid
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
        self._processes[spec.run_id] = process
        self._logs[spec.run_id] = log
        manifest["pid"] = process.pid
        self._write_json(directory / "manifest.json", manifest)
        return {"runId": spec.run_id, "pid": process.pid, "status": "running"}

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

    def _trainer_command(self, spec: LaunchSpec, directory: Path) -> list[str]:
        command = [
            sys.executable, "-m", "petridish.train_shakespeare",
            "--task", spec.task,
            "--device", "cuda", "--field-size", str(spec.field_size),
            "--batch-size", str(spec.batch_size),
            "--context-length", str(spec.context_length),
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
        profile = resolve_lifecycle_profile(spec.lifecycle_profile, enabled=spec.lifecycle)
        command.extend(("--lifecycle-profile", profile))
        command.append("--lifecycle" if profile != "off" else "--no-lifecycle")
        command.append("--structure" if spec.structure else "--no-structure")
        return command

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
            latest_train = next(
                (record for record in reversed(records) if record.get("type") == "train"), None
            )
            latest_held_out = next(
                (record for record in reversed(records) if record.get("type") == "held_out"), None
            )
            latest_diagnostics = next(
                (record for record in reversed(records) if record.get("type") == "diagnostic"), None
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
                        "running" if running else "failed" if not finite
                        else "checkpointed" if has_checkpoint else "stopped"
                    ),
                    "configuration": manifest.get("configuration", {}),
                    "commit": manifest.get("commit"),
                    "latestTrain": latest_train,
                    "latestHeldOut": latest_held_out,
                    "latestDiagnostics": latest_diagnostics,
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


__all__ = ["Laboratory", "LaunchSpec"]
