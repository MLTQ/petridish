"""GPU telemetry, run discovery, and bounded trainer process control."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


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
    architecture: str = "gru"
    field_size: int = 68
    batch_size: int = 16
    context_length: int = 64
    message_steps: int = 2
    updates: int = 100_000
    seed: int = 1
    amp: str = "bfloat16"
    lifecycle: bool = False


class Laboratory:
    """Expose measured hardware/runs and supervise explicitly enabled jobs."""

    def __init__(
        self,
        repository_root: Path,
        *,
        run_root: Path | None = None,
        control_enabled: bool = False,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.run_root = (run_root or self.repository_root / "runs").resolve()
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
                "tasks": ["tiny_shakespeare"],
                "architectures": ["gru"],
                "ampModes": ["off", "bfloat16"],
            },
            "gpus": gpus,
            "runs": self._discover_runs(active_runs),
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
        manifest = {
            "version": 1,
            "runId": spec.run_id,
            "task": "tiny_shakespeare",
            "architecture": spec.architecture,
            "gpuUuid": spec.gpu_uuid,
            "configuration": {
                "fieldSize": spec.field_size,
                "batchSize": spec.batch_size,
                "contextLength": spec.context_length,
                "messageSteps": spec.message_steps,
                "updates": spec.updates,
                "seed": spec.seed,
                "amp": spec.amp,
                "lifecycle": spec.lifecycle,
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
        if spec.architecture != "gru":
            raise ValueError("architecture is not implemented yet")
        if spec.field_size != 68:
            raise ValueError("Tiny Shakespeare laboratory runs require a 68×68 field")
        if spec.context_length < 8 or spec.context_length > 256:
            raise ValueError("context length must be between 8 and 256")
        if spec.batch_size < 1 or spec.batch_size > 256:
            raise ValueError("batch size must be between 1 and 256")
        if spec.message_steps < 1 or spec.message_steps > 16:
            raise ValueError("message steps must be between 1 and 16")
        if spec.updates < 1:
            raise ValueError("updates must be positive")
        if spec.amp not in {"off", "bfloat16"}:
            raise ValueError("unsupported AMP mode")

    def _trainer_command(self, spec: LaunchSpec, directory: Path) -> list[str]:
        command = [
            sys.executable, "-m", "petridish.train_shakespeare",
            "--device", "cuda", "--field-size", str(spec.field_size),
            "--batch-size", str(spec.batch_size),
            "--context-length", str(spec.context_length),
            "--message-steps", str(spec.message_steps),
            "--amp", spec.amp, "--compile", "off",
            "--updates", str(spec.updates), "--seed", str(spec.seed),
            "--checkpoint-dir", str(directory), "--checkpoint-interval", "100",
            "--eval-interval", "500", "--eval-batches", "4",
            "--progress-interval", "10", "--no-resume",
        ]
        command.append("--lifecycle" if spec.lifecycle else "--no-lifecycle")
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
            running = active.get(directory.name)
            manifest_pid = int(manifest.get("pid", 0) or 0)
            if running is None and manifest_pid > 0 and self._pid_alive(manifest_pid):
                running = {"pid": manifest_pid, "gpuUuid": manifest.get("gpuUuid")}
            summaries.append(
                {
                    "id": directory.name,
                    "task": manifest.get("task", "tiny_shakespeare"),
                    "architecture": manifest.get("architecture", "gru"),
                    "gpuUuid": (running or {}).get("gpuUuid", manifest.get("gpuUuid")),
                    "pid": (running or {}).get("pid", manifest_pid or None),
                    "status": "running" if running else ("checkpointed" if has_checkpoint else "stopped"),
                    "configuration": manifest.get("configuration", {}),
                    "commit": manifest.get("commit"),
                    "latestTrain": latest_train,
                    "latestHeldOut": latest_held_out,
                    "hasCheckpoint": has_checkpoint,
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
    def _pid_alive(pid: int) -> bool:
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
