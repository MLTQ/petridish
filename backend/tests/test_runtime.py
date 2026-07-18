"""Control-plane responsiveness contracts for the asynchronous runtime."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from petridish.runtime import ExperimentRuntime, checkpoint_root_from_environment


def runtime_shell() -> ExperimentRuntime:
    """Build the control plane without constructing a dataset-backed experiment."""

    runtime = ExperimentRuntime.__new__(ExperimentRuntime)
    runtime.running = True
    runtime.training_mode = False
    runtime.steps_per_frame = 4
    runtime.training_report_interval = 1.0
    runtime.last_compute_seconds = 2.5
    runtime.training_updates_per_second = 0.4
    runtime.training_examples_per_second = 1.6
    runtime.compute_phase = "forward"
    runtime.compute_progress = 8
    runtime.compute_total = 64
    runtime.control_revision = 7
    runtime.experiment_name = "mnist"
    runtime.saved_organisms = []
    runtime.experiment_sources = {}
    runtime._interrupt_requested = False
    runtime._last_snapshot = {"type": "snapshot", "runtime": {"running": True}}
    runtime._clients = set()
    runtime._broadcast_lock = asyncio.Lock()
    return runtime


@pytest.mark.asyncio
async def test_pause_acknowledges_without_waiting_for_scientific_lock() -> None:
    runtime = runtime_shell()

    await runtime.handle_command({"type": "pause"})

    assert runtime.running is False
    assert runtime._interrupt_requested is True
    assert runtime.control_revision == 8
    acknowledgement = runtime._cached_snapshot()
    assert acknowledgement is not None
    assert acknowledgement["runtime"]["running"] is False
    assert acknowledgement["runtime"]["controlRevision"] == 8
    assert runtime._last_snapshot["runtime"]["running"] is True


def test_sequence_interruption_uses_only_safe_boundaries() -> None:
    runtime = runtime_shell()
    runtime._interrupt_requested = True

    assert runtime._should_interrupt_sequence_update("forward", 9)
    assert runtime._should_interrupt_sequence_update("backward", 0)
    assert not runtime._should_interrupt_sequence_update("backward", 1)
    assert runtime._should_interrupt_sequence_update("optimizer", 0)
    assert not runtime._should_interrupt_sequence_update("credit", 0)
    assert not runtime._should_interrupt_sequence_update("lifecycle", 0)


def test_saved_organism_discovery_lists_only_checkpoint_directories(tmp_path) -> None:
    runtime = runtime_shell()
    runtime.checkpoint_root = tmp_path
    (tmp_path / "complete").mkdir()
    (tmp_path / "complete" / "latest.pt").touch()
    (tmp_path / "incomplete").mkdir()

    assert runtime._discover_saved_organisms() == [
        {"id": "complete", "label": "complete"}
    ]


def test_saved_organism_discovery_hides_known_wrapped_corpus_geometry(tmp_path) -> None:
    runtime = runtime_shell()
    runtime.checkpoint_root = tmp_path
    legacy = tmp_path / "legacy-token-64"
    current = tmp_path / "linear-token-68"
    for directory, field_size in ((legacy, 64), (current, 68)):
        directory.mkdir()
        (directory / "latest.pt").touch()
        (directory / "manifest.json").write_text(
            json.dumps({
                "task": "tiny_stories",
                "configuration": {"fieldSize": field_size},
            })
        )

    assert runtime._discover_saved_organisms() == [
        {"id": "linear-token-68", "label": "linear-token-68"}
    ]


def test_checkpoint_catalog_uses_shared_run_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PETRIDISH_RUN_ROOT", str(tmp_path))

    assert checkpoint_root_from_environment() == tmp_path


def test_saved_organism_rejects_paths_outside_run_directory(tmp_path) -> None:
    runtime = runtime_shell()
    runtime.checkpoint_root = tmp_path

    with pytest.raises(ValueError, match="invalid saved organism identifier"):
        runtime._load_saved_organism("../outside")


@pytest.mark.asyncio
async def test_load_selects_and_pauses_saved_organism(monkeypatch) -> None:
    runtime = runtime_shell()
    replacement = SimpleNamespace(experiment_name="tiny_shakespeare")
    runtime.experiment = SimpleNamespace()
    runtime.experiments = {}
    runtime._lock = asyncio.Lock()
    runtime._load_saved_organism = lambda identifier: replacement  # type: ignore[method-assign]
    runtime._snapshot = lambda: {  # type: ignore[method-assign]
        "type": "snapshot", "runtime": runtime._runtime_payload()
    }
    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    await runtime.handle_command({"type": "load", "organism": "shakespeare-4090"})

    assert runtime.experiment is replacement
    assert runtime.experiment_name == "tiny_shakespeare"
    assert runtime.experiments["tiny_shakespeare"] is replacement
    assert runtime.experiment_sources == {"tiny_shakespeare": "shakespeare-4090"}
    assert runtime.running is False
    assert runtime.training_mode is False
