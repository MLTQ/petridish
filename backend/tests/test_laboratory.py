"""Laboratory discovery and process-control contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from petridish.laboratory import (
    ContinueSpec, EvaluateSpec, ForkSpec, Laboratory, LaunchSpec,
)


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


def test_training_shard_audit_is_not_mislabeled_as_validation(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "control"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"type": "held_out", "update": 10, "accuracy": 0.2}),
                json.dumps(
                    {
                        "type": "training_audit", "update": 10,
                        "accuracy": 0.8, "evaluationSplit": "training",
                    }
                ),
                json.dumps(
                    {
                        "type": "trajectory_audit", "update": 10,
                        "accuracy": 1.0, "evaluationSplit": "trajectory",
                    }
                ),
            )
        ),
        encoding="utf-8",
    )
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs")

    summary = laboratory._discover_runs({})[0]

    assert summary["latestHeldOut"]["accuracy"] == 0.2
    assert summary["latestTrainingAudit"]["accuracy"] == 0.8
    assert summary["latestTrainingAudit"]["evaluationSplit"] == "training"
    assert summary["latestTrajectoryAudit"]["accuracy"] == 1.0
    assert summary["latestTrajectoryAudit"]["evaluationSplit"] == "trajectory"


def test_run_discovery_repairs_lagging_phase_metadata_from_training_metrics(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "control"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "configuration": {
                    "stateLanes": 1, "trainingShardTokens": 2_048,
                    "topologyProfile": "fixed", "lifecycleProfile": "off",
                },
                "phaseHistory": [
                    {"index": 6, "name": "older phase", "startUpdate": 3_000}
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "train", "update": update, "phaseIndex": 7,
                    "phaseName": "1024-byte recovery curriculum",
                    "stateLanes": 2, "trainingShardTokens": 1_024,
                    "topologyProfile": "fixed", "timestamp": float(update),
                }
            )
            for update in (3_501, 3_502)
        ) + "\n",
        encoding="utf-8",
    )
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs")

    summary = laboratory._discover_runs({})[0]

    assert summary["configuration"]["stateLanes"] == 2
    assert summary["configuration"]["trainingShardTokens"] == 1_024
    assert summary["phaseHistory"][-1] == {
        "index": 7,
        "name": "1024-byte recovery curriculum",
        "startUpdate": 3_500,
        "targetUpdate": 3_502,
        "structure": False,
        "topologyProfile": "fixed",
        "lifecycleProfile": "off",
        "trainingShardTokens": 1_024,
        "stateLanes": 2,
        "startedAt": 3_501.0,
        "recoveredFromMetrics": True,
    }


def test_run_id_cannot_escape_root(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs")

    with pytest.raises(ValueError, match="run id"):
        laboratory.metrics("../outside")


def test_launch_is_disabled_by_default(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs")

    with pytest.raises(PermissionError, match="disabled"):
        laboratory.launch(LaunchSpec("trial", "GPU-example"))


def test_checkpoint_fork_preserves_exact_organism_and_records_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "runs"
    source = run_root / "source"
    source.mkdir(parents=True)
    checkpoint = b"persistent cells, graph, weights, optimizer, rng, and state"
    (source / "latest.pt").write_bytes(checkpoint)
    manifest = {
        "version": 2,
        "runId": "source",
        "organismId": "organism-preserved",
        "pid": 0,
        "phaseHistory": [{"index": 3, "name": "consolidated"}],
        "configuration": {"stateLanes": 16},
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (source / "metrics.jsonl").write_text(
        json.dumps({"type": "train", "update": 5250}) + "\n",
        encoding="utf-8",
    )
    laboratory = Laboratory(tmp_path, run_root=run_root, control_enabled=True)
    monkeypatch.setattr(laboratory, "_git_commit", lambda: "fork-commit")

    result = laboratory.fork_run(ForkSpec("source", "prune-branch"))

    branch = run_root / "prune-branch"
    branch_manifest = json.loads((branch / "manifest.json").read_text())
    branch_records = [
        json.loads(line) for line in (branch / "metrics.jsonl").read_text().splitlines()
    ]
    digest = hashlib.sha256(checkpoint).hexdigest()
    assert result == {
        "runId": "prune-branch",
        "organismId": "organism-preserved",
        "parentCheckpoint": {"runId": "source", "update": 5250, "sha256": digest},
        "status": "checkpointed",
    }
    assert (branch / "latest.pt").read_bytes() == checkpoint
    assert (branch / "latest.pt").stat().st_ino != (source / "latest.pt").stat().st_ino
    assert branch_manifest["organismId"] == manifest["organismId"]
    assert branch_manifest["phaseHistory"] == manifest["phaseHistory"]
    assert branch_manifest["parentCheckpoint"] == result["parentCheckpoint"]
    assert branch_manifest["branchRootRunId"] == "source"
    assert branch_manifest["branchDepth"] == 1
    assert branch_manifest["pid"] == 0
    assert branch_records[-1]["type"] == "branch"
    assert branch_records[-1]["checkpointSha256"] == digest
    assert json.loads((source / "manifest.json").read_text()) == manifest


def test_checkpoint_fork_rejects_a_running_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "runs" / "source"
    source.mkdir(parents=True)
    (source / "latest.pt").write_bytes(b"state")
    (source / "manifest.json").write_text(
        json.dumps({"organismId": "organism-live", "pid": 42}), encoding="utf-8"
    )
    laboratory = Laboratory(
        tmp_path, run_root=tmp_path / "runs", control_enabled=True
    )
    monkeypatch.setattr(laboratory, "_pid_alive", lambda _: True)

    with pytest.raises(ValueError, match="already running"):
        laboratory.fork_run(ForkSpec("source", "unsafe-branch"))

    assert not (tmp_path / "runs" / "unsafe-branch").exists()


def test_snapshot_advertises_checkpoint_evaluation_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs")
    monkeypatch.setattr(laboratory, "_gpu_snapshot", lambda: ([], []))
    monkeypatch.setattr(laboratory, "_discover_runs", lambda active: [])
    monkeypatch.setattr(laboratory, "_discover_benchmarks", lambda: [])

    capabilities = laboratory.snapshot()["capabilities"]
    assert capabilities["checkpointEvaluation"] is True
    assert capabilities["trainingShardAudit"] is True
    assert capabilities["trainingShardCurriculum"] is True
    assert capabilities["stateLaneExpansion"] is True
    assert capabilities["checkpointFork"] is True
    assert capabilities["tokenizerProfiles"] == ["wordpiece", "byte"]


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
                    "stateLanes": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"type": "train", "update": 500, "loss": 3.5}),
                json.dumps(
                    {
                        "type": "diagnostic", "update": 500,
                        "cumulativeGrownEdges": 320,
                        "cumulativePrunedEdges": 448,
                        "cumulativeBirths": 3, "cumulativeDeaths": 5,
                    }
                ),
            )
        ) + "\n",
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
            structure=True, topology_profile="prune_only", lifecycle_profile="off",
            training_shard_tokens=8_192,
            state_lanes=16,
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
    assert manifest["configuration"]["topologyProfile"] == "prune_only"
    assert manifest["configuration"]["trainingShardTokens"] == 8_192
    assert manifest["configuration"]["stateLanes"] == 16
    assert manifest["phaseHistory"][-1]["topologyProfile"] == "prune_only"
    assert manifest["phaseHistory"][-1]["trainingShardTokens"] == 8_192
    assert manifest["phaseHistory"][-1]["stateLanes"] == 16
    assert manifest["phaseHistory"][-1]["startGrownEdges"] == 320
    assert manifest["phaseHistory"][-1]["startPrunedEdges"] == 448
    assert manifest["phaseHistory"][-1]["startBirths"] == 3
    assert manifest["phaseHistory"][-1]["startDeaths"] == 5
    assert [phase["startUpdate"] for phase in manifest["phaseHistory"]] == [0, 500]
    assert command[command.index("--updates") + 1] == "1500"
    assert "--resume-plasticity" in command
    assert "--structure" in command
    assert command[command.index("--topology-profile") + 1] == "prune_only"
    assert command[command.index("--training-shard-tokens") + 1] == "8192"
    assert command[command.index("--state-lanes") + 1] == "16"
    assert "--no-resume" not in command
    assert records[-1]["type"] == "phase"
    assert records[-1]["organismId"] == manifest["organismId"]
    assert records[-1]["trainingShardTokens"] == 8_192
    assert records[-1]["stateLanes"] == 16
    assert records[-1]["startGrownEdges"] == 320
    assert records[-1]["startPrunedEdges"] == 448


def test_continuation_recovers_phase_and_curriculum_from_latest_checkpoint_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "runs" / "trial"
    run.mkdir(parents=True)
    (run / "latest.pt").touch()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "runId": "trial",
                "organismId": "organism-preserved",
                "task": "tiny_stories",
                "configuration": {
                    "updates": 3_500,
                    "stateLanes": 1,
                    "trainingShardTokens": 2_048,
                },
                "phaseHistory": [
                    {
                        "index": 6, "name": "stale manifest phase",
                        "startUpdate": 3_000, "targetUpdate": 3_500,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.jsonl").write_text(
        json.dumps(
            {
                "type": "train", "update": 4_250, "phaseIndex": 7,
                "trainingShardTokens": 1_024, "stateLanes": 2,
            }
        ) + "\n",
        encoding="utf-8",
    )
    laboratory = Laboratory(
        tmp_path, run_root=tmp_path / "runs", control_enabled=True
    )
    monkeypatch.setattr(
        laboratory, "_gpu_snapshot", lambda: ([{"uuid": "GPU-example"}], [])
    )
    monkeypatch.setattr(
        laboratory, "_start_process",
        lambda *_args: SimpleNamespace(pid=1234),
    )

    with pytest.raises(ValueError, match="cannot discard"):
        laboratory.continue_run(
            ContinueSpec(
                "trial", "GPU-example", additional_updates=100,
                structure=False, topology_profile="fixed", lifecycle_profile="off",
                state_lanes=1,
            )
        )

    result = laboratory.continue_run(
        ContinueSpec(
            "trial", "GPU-example", additional_updates=100,
            structure=False, topology_profile="fixed", lifecycle_profile="off",
            state_lanes=16,
        )
    )

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    command = manifest["command"]
    assert result["organismId"] == "organism-preserved"
    assert result["phaseIndex"] == 8
    assert manifest["configuration"]["updates"] == 4_350
    assert manifest["configuration"]["trainingShardTokens"] == 1_024
    assert manifest["configuration"]["stateLanes"] == 16
    assert manifest["phaseHistory"][-1]["index"] == 8
    assert manifest["phaseHistory"][-1]["trainingShardTokens"] == 1_024
    assert "--training-shard-tokens" not in command
    assert command[command.index("--state-lanes") + 1] == "16"


def test_checkpoint_evaluation_is_read_only_and_keeps_the_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "runs" / "trial"
    run.mkdir(parents=True)
    (run / "latest.pt").touch()
    phase_history = [
        {
            "index": 2, "name": "adaptive topology", "startUpdate": 1_000,
            "targetUpdate": 1_500, "structure": True, "lifecycleProfile": "off",
        }
    ]
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "runId": "trial", "organismId": "organism-test",
                "phaseHistory": phase_history,
            }
        ),
        encoding="utf-8",
    )
    laboratory = Laboratory(
        tmp_path, run_root=tmp_path / "runs", control_enabled=True
    )
    monkeypatch.setattr(
        laboratory, "_gpu_snapshot", lambda: ([{"uuid": "GPU-example"}], [])
    )
    launched: dict[str, object] = {}

    def fake_start(
        run_id: str, gpu_uuid: str, command: list[str], directory: Path
    ) -> SimpleNamespace:
        launched.update(command=command)
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(laboratory, "_start_process", fake_start)
    monkeypatch.setattr(laboratory, "_git_commit", lambda: "eval-commit")

    result = laboratory.evaluate_run(
        EvaluateSpec(
            "trial", "GPU-example", state_horizons=True,
            evaluation_split="training",
        )
    )

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    command = launched["command"]
    assert isinstance(command, list)
    assert result["status"] == "evaluating"
    assert "--evaluate-only" in command
    assert "--state-horizon-eval" in command
    assert command[command.index("--evaluation-split") + 1] == "training"
    assert command[command.index("--eval-batches") + 1] == "16"
    assert "--resume-plasticity" not in command
    assert command[command.index("--organism-id") + 1] == "organism-test"
    assert command[command.index("--phase-index") + 1] == "2"
    assert manifest["phaseHistory"] == phase_history


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


def test_laboratory_enforces_byte_complete_tokenizer_contract(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        vocabulary_size=256, tokenizer_profile="byte",
    )

    laboratory._validate_spec(spec)
    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert command[command.index("--tokenizer-profile") + 1] == "byte"
    assert command[command.index("--vocabulary-size") + 1] == "256"
    with pytest.raises(ValueError, match="256-token"):
        laboratory._validate_spec(
            LaunchSpec(
                "invalid", "GPU-example", task="tiny_stories",
                vocabulary_size=128, tokenizer_profile="byte",
            )
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


def test_laboratory_records_prune_only_without_disabling_structure(tmp_path: Path) -> None:
    laboratory = Laboratory(tmp_path, run_root=tmp_path / "runs", control_enabled=True)
    spec = LaunchSpec(
        "trial", "GPU-example", task="tiny_stories", field_size=68,
        topology_profile="prune_only",
    )

    laboratory._validate_spec(spec)
    command = laboratory._trainer_command(spec, tmp_path / "runs" / "trial")

    assert command[command.index("--topology-profile") + 1] == "prune_only"
    assert "--structure" in command
    with pytest.raises(ValueError, match="topology profile"):
        laboratory._validate_spec(
            LaunchSpec("invalid", "GPU-example", topology_profile="regrow-everything")
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
