"""Small, CPU-friendly defaults for token-stream experiments."""

from __future__ import annotations

from dataclasses import replace

from .mnist_config import MnistModelConfig


def sequence_config(task: str | None = None, **overrides: object) -> MnistModelConfig:
    """Return a compact substrate configuration tuned for short sequences."""

    base = MnistModelConfig(
        width=24,
        height=24,
        hidden_channels=12,
        genotype_channels=8,
        initial_density=0.48,
        local_radius=8,
        candidate_probes=28,
        message_steps=6,
        message_gain=2.0,
        batch_size=48,
        learning_rate=0.0015,
        readout_learning_rate=0.006,
        synapse_learning_rate=0.002,
        readout_only_trials=0,
        synapse_unlock_trials=0,
        lifecycle_warmup_trials=160,
        structural_warmup_trials=240,
        structure_accuracy_threshold=0.82,
        evaluation_interval=50,
        evaluation_batches=8,
        births_per_generation=12,
        max_deaths_per_generation=48,
        max_pruned_per_generation=96,
        max_visible_edges=2_000,
    )
    if task == "tiny_shakespeare":
        base = replace(
            base,
            width=68,
            height=68,
            batch_size=16,
            message_steps=2,
            local_radius=16,
            candidate_probes=32,
            initial_density=0.25,
            max_initial_neurons=4_096,
            max_visible_edges=6_000,
            lifecycle_enabled=0,
            lifecycle_warmup_trials=5_000,
            structural_warmup_trials=5_000,
        )
    if task == "tiny_stories":
        base = replace(
            base,
            width=68,
            height=68,
            hidden_channels=32,
            genotype_channels=16,
            edge_slots=8,
            axon_slots=16,
            batch_size=8,
            message_steps=4,
            local_radius=8,
            candidate_probes=48,
            initial_density=0.50,
            max_initial_neurons=2_048,
            max_visible_edges=6_000,
            broadcast_slots=8,
            broadcast_gain=0.35,
            lifecycle_enabled=1,
            lifecycle_warmup_trials=500,
            structural_warmup_trials=1_000,
            structure_plateau_trials=500,
            births_per_generation=16,
            max_deaths_per_generation=64,
            max_pruned_per_generation=256,
            max_grown_per_generation=64,
        )
    return replace(base, **overrides)


__all__ = ["sequence_config"]
