"""Named topology-mutation policies for persistent organism phases."""

from __future__ import annotations


TOPOLOGY_PROFILES = ("fixed", "adaptive", "prune_only")


def resolve_topology_profile(
    profile: str | None, *, structure: bool
) -> str:
    """Resolve legacy structure booleans into one explicit phase policy."""

    resolved = profile or ("adaptive" if structure else "fixed")
    if resolved not in TOPOLOGY_PROFILES:
        raise ValueError(f"unknown topology profile: {resolved}")
    return resolved


def topology_mutates(profile: str) -> bool:
    """Return whether a phase may remove or add dendrites."""

    return resolve_topology_profile(profile, structure=True) != "fixed"


def topology_grows(profile: str) -> bool:
    """Return whether a phase may add dendrites."""

    return resolve_topology_profile(profile, structure=True) == "adaptive"


__all__ = [
    "TOPOLOGY_PROFILES",
    "resolve_topology_profile",
    "topology_grows",
    "topology_mutates",
]
