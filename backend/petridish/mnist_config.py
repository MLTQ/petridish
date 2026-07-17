"""Interpretable configuration for the spatial MNIST neural organism."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MnistModelConfig:
    """Dimensions, learning rates, and lifecycle thresholds for one organism."""

    width: int = 64
    height: int = 64
    hidden_channels: int = 16
    genotype_channels: int = 16
    edge_slots: int = 4
    axon_slots: int = 8
    candidate_slots: int = 4
    candidate_probes: int = 32
    initial_density: float = 0.48
    initial_weight_scale: float = 0.25
    local_radius: int = 8
    message_steps: int = 20
    message_gain: float = 1.5
    attention_temperature: float = 1.0
    emit_gate_bias: float = 1.5
    batch_size: int = 16
    learning_rate: float = 0.001
    readout_learning_rate: float = 0.005
    synapse_learning_rate: float = 0.003
    trajectory_loss_weight: float = 0.0
    readout_only_trials: int = 128
    synapse_unlock_trials: int = 256
    curriculum_window_trials: int = 40
    curriculum_min_trials: int = 40
    gradient_clip: float = 1.0
    weight_decay: float = 0.0002
    max_weight: float = 2.0
    structural_interval: int = 16
    structural_warmup_trials: int = 512
    structure_accuracy_threshold: float = 0.70
    structure_plateau_trials: int = 256
    structure_plateau_delta: float = 0.005
    evaluation_interval: int = 100
    evaluation_batches: int = 8
    candidate_decay: float = 0.88
    candidate_threshold: float = 0.11
    emission_threshold: float = 0.018
    edge_utility_decay: float = 0.97
    edge_stat_rate: float = 0.08
    edge_grace_trials: int = 32
    prune_utility: float = 0.012
    max_pruned_per_generation: int = 384
    lifecycle_enabled: int = 1
    lifecycle_warmup_trials: int = 128
    lifecycle_interval: int = 16
    juvenile_trials: int = 64
    target_stimulation_min: float = 0.018
    target_stimulation_max: float = 0.42
    energy_recovery: float = 0.012
    starvation_cost: float = 0.30
    overload_cost: float = 0.30
    maintenance_cost: float = 0.00035
    task_energy_bonus: float = 0.006
    death_energy: float = 0.035
    max_deaths_per_generation: int = 192
    births_per_generation: int = 32
    birth_signal: float = 0.08
    birth_local_density_max: float = 0.38
    birth_energy: float = 0.65
    inheritance_noise: float = 0.035
    max_visible_edges: int = 4_000

    @property
    def site_count(self) -> int:
        return self.width * self.height

    @property
    def trace_steps(self) -> int:
        return self.message_steps + 3
