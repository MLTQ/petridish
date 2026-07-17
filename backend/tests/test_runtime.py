"""Control-plane responsiveness contracts for the asynchronous runtime."""

from __future__ import annotations

import asyncio

import pytest

from petridish.runtime import ExperimentRuntime


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
