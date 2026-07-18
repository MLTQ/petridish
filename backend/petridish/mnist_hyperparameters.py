"""Public slider schema and validation for MNIST organism configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math
from typing import Any, Mapping

from .mnist_config import MnistModelConfig


@dataclass(frozen=True, slots=True)
class HyperparameterSpec:
    """One numeric control exposed by the scientific viewer."""

    label: str
    group: str
    minimum: float
    maximum: float
    step: float


FIELD_SIZES = (16, 32, 64, 128, 256, 512, 1024)
TINY_SHAKESPEARE_FIELD_SIZE = 68


def field_sizes(*, include_sequence: bool, task_key: str | None = None) -> tuple[int, ...]:
    """Return task-specific square geometries without weakening global validation."""

    choices = FIELD_SIZES if include_sequence else FIELD_SIZES[1:]
    if task_key == "tiny_shakespeare":
        return tuple(sorted((*choices, TINY_SHAKESPEARE_FIELD_SIZE)))
    return choices


SPECS: dict[str, HyperparameterSpec] = {
    "width": HyperparameterSpec("substrate width", "geometry & compute", 16, 1024, 1),
    "height": HyperparameterSpec("substrate height", "geometry & compute", 16, 1024, 1),
    "hidden_channels": HyperparameterSpec("state channels", "geometry & compute", 4, 64, 4),
    "genotype_channels": HyperparameterSpec("persistent genotype channels", "geometry & compute", 4, 64, 4),
    "edge_slots": HyperparameterSpec("dendrites / neuron", "geometry & compute", 1, 8, 1),
    "axon_slots": HyperparameterSpec("axons / neuron", "geometry & compute", 1, 16, 1),
    "candidate_slots": HyperparameterSpec("candidate IDs", "geometry & compute", 1, 8, 1),
    "candidate_probes": HyperparameterSpec("broadcast probes", "geometry & compute", 8, 128, 4),
    "initial_density": HyperparameterSpec("initial density", "geometry & compute", 0.1, 0.9, 0.01),
    "max_initial_neurons": HyperparameterSpec("initial neuron cap", "geometry & compute", 128, 131_072, 128),
    "initial_weight_scale": HyperparameterSpec("initial weight scale", "geometry & compute", 0.01, 1, 0.01),
    "local_radius": HyperparameterSpec("broadcast radius (cells)", "geometry & compute", 1, 511, 1),
    "message_steps": HyperparameterSpec("message steps", "geometry & compute", 2, 32, 1),
    "message_gain": HyperparameterSpec("message gain", "geometry & compute", 0.05, 2, 0.05),
    "attention_temperature": HyperparameterSpec("attention temperature", "geometry & compute", 0.1, 4, 0.1),
    "emit_gate_bias": HyperparameterSpec("initial emit bias", "geometry & compute", -4, 4, 0.1),
    "broadcast_slots": HyperparameterSpec("broadcast workspace slots", "geometry & compute", 1, 16, 1),
    "broadcast_gain": HyperparameterSpec("broadcast workspace gain", "geometry & compute", 0, 2, 0.05),
    "broadcast_decay": HyperparameterSpec("broadcast memory decay", "geometry & compute", 0, 0.99, 0.01),
    "fast_weight_gain": HyperparameterSpec("fast-weight memory gain", "geometry & compute", 0, 2, 0.05),
    "fast_weight_decay": HyperparameterSpec("fast-weight memory decay", "geometry & compute", 0, 0.999, 0.001),
    "binding_memory_gain": HyperparameterSpec("neuron-owned binding gain", "geometry & compute", 0, 2, 0.05),
    "binding_memory_temperature": HyperparameterSpec("binding address temperature", "geometry & compute", 0.01, 1, 0.01),
    "binding_token_values": HyperparameterSpec("binding stores token values (0/1)", "geometry & compute", 0, 1, 1),
    "binding_address_regularization": HyperparameterSpec("binding address separation", "learning", 0, 0.2, 0.001),
    "batch_size": HyperparameterSpec("batch size", "geometry & compute", 4, 128, 4),
    "max_visible_edges": HyperparameterSpec("rendered edge cap", "geometry & compute", 100, 20_000, 100),
    "learning_rate": HyperparameterSpec("shared-rule learning rate", "learning", 0.0001, 0.01, 0.0001),
    "readout_learning_rate": HyperparameterSpec("readout learning rate", "learning", 0.0001, 0.05, 0.0001),
    "synapse_learning_rate": HyperparameterSpec("synapse learning rate", "learning", 0.001, 0.5, 0.001),
    "trajectory_loss_weight": HyperparameterSpec("early-output loss weight", "learning", 0, 1, 0.01),
    "readout_only_trials": HyperparameterSpec("readout-only trials", "learning", 0, 1_000, 1),
    "synapse_unlock_trials": HyperparameterSpec("synapse unlock trial", "learning", 0, 5_000, 1),
    "curriculum_window_trials": HyperparameterSpec("curriculum accuracy window", "learning", 4, 512, 4),
    "curriculum_min_trials": HyperparameterSpec("minimum trials / curriculum stage", "learning", 1, 2_000, 10),
    "gradient_clip": HyperparameterSpec("gradient clip", "learning", 0.1, 10, 0.1),
    "weight_decay": HyperparameterSpec("synapse weight decay", "learning", 0, 0.01, 0.0001),
    "max_weight": HyperparameterSpec("maximum |weight|", "learning", 0.1, 5, 0.1),
    "evaluation_interval": HyperparameterSpec("evaluation interval", "learning", 10, 1_000, 10),
    "evaluation_batches": HyperparameterSpec("evaluation batches", "learning", 1, 64, 1),
    "structural_interval": HyperparameterSpec("structural interval", "growth & pruning", 1, 128, 1),
    "structural_warmup_trials": HyperparameterSpec("structure warm-up", "growth & pruning", 0, 5_000, 1),
    "structure_accuracy_threshold": HyperparameterSpec("structure competence accuracy", "growth & pruning", 0, 1, 0.01),
    "structure_plateau_trials": HyperparameterSpec("structure plateau wait", "growth & pruning", 1, 5_000, 1),
    "structure_plateau_delta": HyperparameterSpec("accuracy improvement delta", "growth & pruning", 0.0001, 0.1, 0.0001),
    "candidate_decay": HyperparameterSpec("candidate memory decay", "growth & pruning", 0, 1, 0.01),
    "candidate_threshold": HyperparameterSpec("connection threshold", "growth & pruning", 0.01, 2, 0.01),
    "emission_threshold": HyperparameterSpec("emission threshold", "growth & pruning", 0, 0.25, 0.001),
    "edge_utility_decay": HyperparameterSpec("edge utility decay", "growth & pruning", 0, 1, 0.01),
    "edge_stat_rate": HyperparameterSpec("edge statistic rate", "growth & pruning", 0.001, 0.5, 0.001),
    "edge_grace_trials": HyperparameterSpec("edge grace trials", "growth & pruning", 1, 512, 1),
    "prune_utility": HyperparameterSpec("prune utility floor", "growth & pruning", 0, 0.2, 0.001),
    "max_pruned_per_generation": HyperparameterSpec("prune budget", "growth & pruning", 0, 4_096, 32),
    "lifecycle_enabled": HyperparameterSpec("lifecycle enabled (0/1)", "homeostasis", 0, 1, 1),
    "lifecycle_warmup_trials": HyperparameterSpec("lifecycle warm-up", "homeostasis", 0, 5_000, 1),
    "lifecycle_interval": HyperparameterSpec("lifecycle interval", "homeostasis", 1, 128, 1),
    "juvenile_trials": HyperparameterSpec("juvenile grace trials", "homeostasis", 1, 1_024, 1),
    "target_stimulation_min": HyperparameterSpec("stimulation minimum", "homeostasis", 0, 0.5, 0.001),
    "target_stimulation_max": HyperparameterSpec("traffic maximum", "homeostasis", 0.01, 2, 0.01),
    "energy_recovery": HyperparameterSpec("energy recovery", "homeostasis", 0, 0.1, 0.001),
    "starvation_cost": HyperparameterSpec("starvation cost", "homeostasis", 0, 1, 0.01),
    "overload_cost": HyperparameterSpec("excitotoxic damage gain", "homeostasis", 0, 1, 0.01),
    "stun_enabled": HyperparameterSpec("excitotoxic stun enabled (0/1)", "homeostasis", 0, 1, 1),
    "stun_load_threshold": HyperparameterSpec("stun traffic threshold", "homeostasis", 0.01, 4, 0.01),
    "stun_recovery_probability": HyperparameterSpec("stun recovery probability", "homeostasis", 0, 1, 0.01),
    "stun_min_generations": HyperparameterSpec("minimum stunned generations", "homeostasis", 1, 128, 1),
    "excitotoxic_damage_per_stun": HyperparameterSpec("damage per stun", "homeostasis", 0, 2, 0.01),
    "excitotoxic_damage_recovery": HyperparameterSpec("damage recovery / trial", "homeostasis", 0, 0.25, 0.005),
    "excitotoxic_death_threshold": HyperparameterSpec("excitotoxic death threshold", "homeostasis", 0.05, 8, 0.05),
    "maintenance_cost": HyperparameterSpec("edge maintenance cost", "homeostasis", 0, 0.01, 0.00005),
    "task_energy_bonus": HyperparameterSpec("task energy bonus", "homeostasis", 0, 0.05, 0.0005),
    "death_energy": HyperparameterSpec("death energy", "homeostasis", 0, 0.5, 0.005),
    "max_deaths_per_generation": HyperparameterSpec("death budget", "homeostasis", 0, 2_048, 16),
    "births_per_generation": HyperparameterSpec("birth budget", "homeostasis", 0, 1_024, 16),
    "birth_signal": HyperparameterSpec("birth signal", "homeostasis", 0, 0.5, 0.005),
    "birth_local_density_max": HyperparameterSpec("birth local density ceiling", "homeostasis", 0, 1, 0.01),
    "birth_energy": HyperparameterSpec("newborn energy", "homeostasis", 0.05, 1, 0.05),
    "inheritance_noise": HyperparameterSpec("genotype mutation noise", "homeostasis", 0, 0.5, 0.005),
}


def hyperparameter_payload(
    config: MnistModelConfig, *, include_sequence: bool = False,
    task_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return the authoritative ordered slider definitions and current values."""

    values = asdict(config)
    integer_fields = {field.name for field in fields(config) if field.type is int or isinstance(values[field.name], int)}
    field_choices = field_sizes(include_sequence=include_sequence, task_key=task_key)
    payload = [
        {
            "key": "field_size",
            "label": "square field size",
            "group": "geometry & compute",
            "value": config.width,
            "min": field_choices[0],
            "max": field_choices[-1],
            "step": 1,
            "integer": True,
            "choices": list(field_choices),
        }
    ]
    payload.extend(
        [
        {
            "key": key,
            "label": spec.label,
            "group": spec.group,
            "value": values[key],
            "min": spec.minimum,
            "max": (
                min(spec.maximum, min(config.width, config.height) // 2 - 1)
                if key == "local_radius"
                else spec.maximum
            ),
            "step": spec.step,
            "integer": key in integer_fields,
        }
        for key, spec in SPECS.items()
        if key not in {"width", "height"}
        if include_sequence or key not in {
            "broadcast_slots", "broadcast_gain", "broadcast_decay",
            "fast_weight_gain", "fast_weight_decay",
            "binding_memory_gain", "binding_memory_temperature",
            "binding_token_values",
            "binding_address_regularization",
        }
        ]
    )
    return payload


def configured(
    config: MnistModelConfig,
    changes: Mapping[str, Any],
    *,
    task_key: str | None = None,
) -> MnistModelConfig:
    """Validate numeric viewer changes and return a new immutable configuration."""

    unknown = set(changes) - set(SPECS) - {"field_size"}
    if unknown:
        raise ValueError(f"unknown MNIST hyperparameter: {sorted(unknown)[0]}")
    normalized = dict(changes)
    if "field_size" in normalized:
        if "width" in normalized or "height" in normalized:
            raise ValueError("field_size cannot be combined with width or height")
        field_size = normalized.pop("field_size")
        allowed = field_sizes(include_sequence=True, task_key=task_key)
        if isinstance(field_size, bool) or field_size not in allowed:
            if task_key == "tiny_shakespeare":
                raise ValueError(
                    "field_size must be 68 or a power of two from 16 through 1024"
                )
            raise ValueError("field_size must be a power of two from 16 through 1024")
        normalized["width"] = field_size
        normalized["height"] = field_size
    current = asdict(config)
    converted: dict[str, int | float] = {}
    for key, raw in normalized.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise ValueError(f"{key} must be a finite number")
        spec = SPECS[key]
        value = float(raw)
        if value < spec.minimum or value > spec.maximum:
            raise ValueError(f"{key} must be between {spec.minimum:g} and {spec.maximum:g}")
        if isinstance(current[key], int):
            if not value.is_integer():
                raise ValueError(f"{key} must be an integer")
            converted[key] = int(value)
        else:
            converted[key] = value

    result = replace(config, **converted)
    if "field_size" in changes and result.local_radius >= result.width / 2:
        result = replace(result, local_radius=max(1, result.width // 2 - 1))
    if result.candidate_probes < result.edge_slots:
        raise ValueError("candidate_probes must be at least edge_slots")
    if result.local_radius >= min(result.width, result.height) / 2:
        raise ValueError("local_radius must be less than half the smaller field dimension")
    if result.target_stimulation_min >= result.target_stimulation_max:
        raise ValueError("target_stimulation_min must be below target_stimulation_max")
    if result.synapse_unlock_trials < result.readout_only_trials:
        raise ValueError("synapse_unlock_trials must be at least readout_only_trials")
    return result


numeric_config_fields = {
    field.name
    for field in fields(MnistModelConfig)
    if isinstance(getattr(MnistModelConfig(), field.name), (int, float))
}
if set(SPECS) != numeric_config_fields:
    raise RuntimeError("every MNIST model field must have one hyperparameter specification")


__all__ = [
    "FIELD_SIZES", "TINY_SHAKESPEARE_FIELD_SIZE", "HyperparameterSpec", "SPECS",
    "configured", "field_sizes", "hyperparameter_payload",
]
