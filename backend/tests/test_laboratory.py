"""Laboratory discovery and process-control contracts."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from petridish.laboratory import ContinueSpec, Laboratory, LaunchSpec


def test_metrics_are_bounded_and_ignore_partial_json(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "control"
    run.mkdir(parents=True)
    records = [{"type": "train", "update": update} for update in range(6)]
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n{partial",
        encoding="utf-8",
    )
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs")

    assert [record["update"] for record in laboratory.metrics("control", limit=3)] == [4, 5]


def test_run_id_cannot_escape_root(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs")

    with pytest.raises(ValueError, match="run id"):
        laboratory.metrics("../outside")


def test_launch_is_disabled_by_default(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs")

    with pytest.raises(PermissionError, match="disabled"):
        laboratory.launch(LaunchSpec("trial", "GPU-example"))


def test_spec_preserves_single_column_geometry(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)

    with pytest.raises(ValueError, match="68×68"):
        laboratory._validate_spec(LaunchSpec("trial", "GPU-example", field_size=64))
    with pytest.raises(ValueError, match="68×68"):
        laboratory._validate_spec(
            LaunchSpec("trial", "GPU-example", task="tiny_stories", field_size=64)
        )

    laboratory._validate_spec(
        LaunchSpec("trial", "GPU-example", task="tiny_stories", field_size=68)
    )


def test_laboratory_preserves_named_lifecycle_profile(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        lifecycle=True, lifecycle_profile="balanced",
    )

    laboratory._validate_spec(spec)
    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert command[command.index("--lifecycle-profile") + 1] == "balanced"
    assert "--lifecycle" in command


def test_laboratory_can_launch_a_true_fixed_connectome(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        lifecycle=False, structure=False,
    )

    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert "--no-lifecycle" in command
    assert "--no-structure" in command


def test_checkpoint_continuation_preserves_run_lineage_and_changes_only_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "runs" / "trial"
    run.mkdir(parents=True)
    (run / "latest.pt").touch()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "runId": "trial",
                "task": "tiny_stories",
                "architecture": "esn",
                "configuration": {
                    "structure": False, "lifecycleProfile": "off", "updates": 500,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.jsonl").write_text(
        json.dumps({"type": "train", "update": 500, "loss": 3.5}) + "\n",
        encoding="utf-8",
    )
    laboratory = Laboratory(
        tmp_path, run_root=tmp_path / "runs", control_enabled=True
    )
    monkeypatch.setattr(
        laboratory, "_gpu_snapshot",
        lambda: ([{"uuid": "GPU-example"}], []),
    )
    launched: dict[str, object] = {}

    def fake_start(
        run_id: str, gpu_uuid: str, command: list[str], directory: Path
    ) -> SimpleNamespace:
        launched.update(
            run_id=run_id, gpu_uuid=gpu_uuid, command=command, directory=directory
        )
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(laboratory, "_start_process", fake_start)
    monkeypatch.setattr(laboratory, "_git_commit", lambda: "phase-commit")

    result = laboratory.continue_run(
        ContinueSpec(
            "trial", "GPU-example", additional_updates=1_000,
            structure=True, lifecycle_profile="off",
        )
    )

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    command = launched["command"]
    assert isinstance(command, list)
    assert result["runId"] == "trial"
    assert result["organismId"] == manifest["organismId"]
    assert manifest["configuration"]["updates"] == 1_500
    assert manifest["configuration"]["structure"] is True
    assert [phase["startUpdate"] for phase in manifest["phaseHistory"]] == [0, 500]
    assert command[command.index("--updates") + 1] == "1500"
    assert "--resume-plasticity" in command
    assert "--structure" in command
    assert "--no-resume" not in command
    assert records[-1]["type"] == "phase"
    assert records[-1]["organismId"] == manifest["organismId"]


def test_laboratory_preserves_bounded_learning_rate_scale(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        learning_rate_scale=0.25,
    )

    laboratory._validate_spec(spec)
    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert command[command.index("--learning-rate-scale") + 1] == "0.25"
    with pytest.raises(ValueError, match="learning-rate scale"):
        laboratory._validate_spec(
            LaunchSpec("invalid", "GPU-example", learning_rate_scale=0.0)
        )


def test_laboratory_preserves_bounded_broadcast_gain(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        message_steps=16, broadcast_gain=0.0,
    )

    laboratory._validate_spec(spec)
    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert command[command.index("--broadcast-gain") + 1] == "0.0"
    with pytest.raises(ValueError, match="broadcast gain"):
        laboratory._validate_spec(
            LaunchSpec("invalid", "GPU-example", broadcast_gain=2.1)
        )


def test_laboratory_preserves_token_vocabulary_curriculum(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        vocabulary_size=128,
    )

    laboratory._validate_spec(spec)
    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert command[command.index("--vocabulary-size") + 1] == "128"
    with pytest.raises(ValueError, match="vocabulary size"):
        laboratory._validate_spec(
            LaunchSpec("invalid", "GPU-example", vocabulary_size=100)
        )


def test_laboratory_records_continuous_experience_or_cold_control(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        stream_mode="continuous", state_retention=0.9, state_lanes=2,
    )

    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert command[command.index("--stream-mode") + 1] == "continuous"
    assert command[command.index("--state-retention") + 1] == "0.9"
    assert command[command.index("--state-lanes") + 1] == "2"
    with pytest.raises(ValueError, match="stream mode"):
        laboratory._validate_spec(
            LaunchSpec("invalid", "GPU-example", stream_mode="reset-everything")
        )
    with pytest.raises(ValueError, match="state retention"):
        laboratory._validate_spec(
            LaunchSpec("invalid", "GPU-example", state_retention=1.1)
        )
    with pytest.raises(ValueError, match="state lanes"):
        laboratory._validate_spec(
            LaunchSpec("invalid", "GPU-example", state_lanes=17)
        )


def test_legacy_lifecycle_flag_resolves_to_recorded_baseline(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        lifecycle=True,
    )

    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert command[command.index("--lifecycle-profile") + 1] == "baseline"


def test_explicit_profile_is_authoritative_over_compatibility_flag(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        lifecycle=False, lifecycle_profile="balanced",
    )

    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert command[command.index("--lifecycle-profile") + 1] == "balanced"
    assert "--lifecycle" in command


def test_discovery_summarizes_latest_train_and_evaluation(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "trial"
    run.mkdir(parents=True)
    (run / "latest.pt").touch()
    (run / "manifest.json").write_text(
        json.dumps({"task": "tiny_shakespeare", "architecture": "gru"}), encoding="utf-8"
    )
    (run / "metrics.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"type": "train", "update": 10, "loss": 3.5}),
                json.dumps({"type": "diagnostic", "update": 10, "edgeCount": 812}),
                json.dumps({"type": "held_out", "update": 10, "loss": 3.6}),
            )
        ),
        encoding="utf-8",
    )
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs")

    summary = laboratory._discover_runs({})[0]

    assert summary["status"] == "checkpointed"
    assert summary["latestTrain"]["loss"] == 3.5
    assert summary["latestHeldOut"]["loss"] == 3.6
    assert summary["latestDiagnostics"]["edgeCount"] == 812


def test_zombie_trainer_is_not_reported_as_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = tmp_path / "123"
    process.mkdir()
    (process / "stat").write_text(
        "123 (python trainer) Z 1 2 3 4\n", encoding="utf-8"
    )
    kill_called = False

    def unexpected_kill(_pid: int, _signal: int) -> None:
        nonlocal kill_called
        kill_called = True

    monkeypatch.setattr("petridish.laboratory.os.kill", unexpected_kill)

    assert not Laboratory._pid_alive(123, proc_root=tmp_path)
    assert not kill_called


def test_nonfinite_ended_run_is_reported_as_failed(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "diverged"
    run.mkdir(parents=True)
    (run / "latest.pt").touch()
    (run / "metrics.jsonl").write_text(
        json.dumps({"type": "train", "update": 40, "loss": float("nan")}) + "\n",
        encoding="utf-8",
    )

    summary = Laboratory(tmp_path, run_root=tmp_path / "runs")._discover_runs({})[0]

    assert summary["status"] == "failed"


def test_explicit_trainer_failure_is_reported_before_a_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "oom"
    run.mkdir(parents=True)
    failure = {
        "type": "failure", "failureType": "OutOfMemoryError",
        "failureMessage": "CUDA out of memory",
    }
    (run / "metrics.jsonl").write_text(json.dumps(failure) + "\n", encoding="utf-8")

    summary = Laboratory(tmp_path, run_root=tmp_path / "runs")._discover_runs({})[0]

    assert summary["status"] == "failed"
    assert summary["latestFailure"] == failure


def test_snapshot_discovers_bounded_benchmark_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_root = tmp_path / "benchmarks"
    benchmark_root.mkdir()
    (benchmark_root / "recall-gru.json").write_text(
        json.dumps(
            {
                "task": "associative_recall",
                "profile": "compact24",
                "architecture": "gru",
                "recallMode": "fixed_2",
                "seed": 11,
                "steps": 40,
                "seconds": 9.5,
                "status": "complete",
                "completedSteps": 40,
                "peakCudaAllocatedGiB": 1.25,
                "messageSteps": 12,
                "broadcastGain": 0.0,
                "learningRateScale": 0.25,
                "outputCount": 64,
                "sequenceLength": 2,
                "dependencyTokens": 1,
                "contextReachableOutputs": 64,
                "chanceAccuracy": 0.125,
                "checkpoints": [
                    {"update": 20, "heldOutAccuracy": 0.75, "recallPairs": 1},
                    {"update": 40, "heldOutAccuracy": 0.5, "recallPairs": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    laboratory = Laboratory(tmp_path, benchmark_root=benchmark_root)
    monkeypatch.setattr(laboratory, "_gpu_snapshot", lambda: ([], []))

    benchmark = laboratory.snapshot()["benchmarks"][0]

    assert benchmark["id"] == "recall-gru"
    assert benchmark["architecture"] == "gru"
    assert benchmark["recallMode"] == "fixed_2"
    assert benchmark["completedSteps"] == 40
    assert benchmark["peakCudaAllocatedGiB"] == 1.25
    assert benchmark["messageSteps"] == 12
    assert benchmark["broadcastGain"] == 0.0
    assert benchmark["learningRateScale"] == 0.25
    assert benchmark["outputCount"] == 64
    assert benchmark["sequenceLength"] == 2
    assert benchmark["dependencyTokens"] == 1
    assert benchmark["contextReachableOutputs"] == 64
    assert benchmark["chanceAccuracy"] == 0.125
    assert benchmark["checkpoints"][-1]["recallPairs"] == 2


def test_running_benchmark_is_visible_before_first_checkpoint(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    benchmark_root.mkdir()
    (benchmark_root / "running.json").write_text(
        json.dumps(
            {
                "task": "associative_recall", "architecture": "gru",
                "status": "running", "completedSteps": 0, "checkpoints": [],
            }
        ),
        encoding="utf-8",
    )

    benchmark = Laboratory(
        tmp_path, benchmark_root=benchmark_root
    )._discover_benchmarks()[0]

    assert benchmark["status"] == "running"
    assert benchmark["checkpoints"] == []


def test_failed_benchmark_is_visible_before_first_checkpoint(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "benchmarks"
    benchmark_root.mkdir()
    (benchmark_root / "failed.json").write_text(
        json.dumps(
            {
                "task": "token_stream", "architecture": "gru",
                "status": "failed", "completedSteps": 0, "checkpoints": [],
                "failureType": "OutOfMemoryError",
                "failureMessage": "CUDA out of memory",
            }
        ),
        encoding="utf-8",
    )

    benchmark = Laboratory(
        tmp_path, benchmark_root=benchmark_root
    )._discover_benchmarks()[0]

    assert benchmark["status"] == "failed"
    assert benchmark["failureType"] == "OutOfMemoryError"
    assert benchmark["failureMessage"] == "CUDA out of memory"
    assert benchmark["checkpoints"] == []
