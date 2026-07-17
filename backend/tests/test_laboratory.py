"""Laboratory discovery and process-control contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from petridish.laboratory import Laboratory, LaunchSpec


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
