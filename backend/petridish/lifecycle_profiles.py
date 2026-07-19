"""Named, reproducible homeostasis policies for controlled sequence experiments."""

from __future__ import annotations

from dataclasses import replace

from .mnist_config import MnistModelConfig


LIFECYCLE_PROFILES = (
    "off", "recovery_only", "baseline", "balanced", "replacement",
)


def resolve_lifecycle_profile(profile: str, *, enabled: bool) -> str:
    """Map the legacy enable flag to baseline without overriding named profiles."""

    if profile not in LIFECYCLE_PROFILES:
        raise ValueError(f"unknown lifecycle profile: {profile}")
    return "baseline" if enabled and profile == "off" else profile


def apply_lifecycle_profile(
    config: MnistModelConfig, profile: str
) -> MnistModelConfig:
    """Return a configuration with one explicit lifecycle intervention."""

    if profile not in LIFECYCLE_PROFILES:
        raise ValueError(f"unknown lifecycle profile: {profile}")
    if profile == "off":
        return replace(config, lifecycle_enabled=0)
    if profile == "recovery_only":
        return replace(
            config,
            lifecycle_enabled=1,
            target_stimulation_min=0.0,
            target_stimulation_max=0.80,
            energy_recovery=0.018,
            starvation_cost=0.0,
            maintenance_cost=0.0,
            max_deaths_per_generation=0,
            births_per_generation=0,
            births_replace_deaths=1,
            stun_enabled=1,
            stun_load_threshold=0.80,
            stun_recovery_probability=0.55,
            excitotoxic_damage_per_stun=0.10,
            excitotoxic_damage_recovery=0.03,
            excitotoxic_death_threshold=2.0,
        )
    if profile == "baseline":
        return replace(config, lifecycle_enabled=1)
    balanced = replace(
        config,
        lifecycle_enabled=1,
        target_stimulation_min=0.006,
        target_stimulation_max=0.60,
        energy_recovery=0.018,
        starvation_cost=0.12,
        maintenance_cost=0.00015,
        death_energy=0.015,
        max_deaths_per_generation=16,
        births_per_generation=16,
        birth_local_density_max=0.50,
        overload_cost=0.20,
        stun_load_threshold=0.80,
        stun_recovery_probability=0.55,
        excitotoxic_damage_per_stun=0.10,
        excitotoxic_damage_recovery=0.03,
        excitotoxic_death_threshold=2.0,
    )
    if profile == "balanced":
        return balanced
    return replace(balanced, births_replace_deaths=1)


__all__ = [
    "LIFECYCLE_PROFILES", "apply_lifecycle_profile", "resolve_lifecycle_profile",
]
