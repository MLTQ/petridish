"""Persistent sparse neurons, dendrites, metabolism, and structural plasticity."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math
from typing import Any

import torch
from torch import nn

from .graph_layout import GraphLayout, resolve_layout
from .mnist_config import MnistModelConfig


@dataclass(slots=True)
class GraphSnapshot:
    """Detached incoming-dendrite state for one visualization frame."""

    source: torch.Tensor
    weight: torch.Tensor
    age: torch.Tensor
    utility: torch.Tensor
    flow: torch.Tensor
    credit: torch.Tensor


@dataclass(slots=True)
class StructuralUpdate:
    """Actual births, deaths, growth, and pruning from one structural cycle."""

    events: list[dict[str, Any]]
    changed_edges: torch.Tensor
    changed_sites: torch.Tensor
    births: int
    deaths: int
    death_causes: dict[str, int]
    recoveries: int = 0


@dataclass(frozen=True, slots=True)
class GraphDiagnostics:
    """Cached sensory-to-output reachability for the current topology."""

    minimum_output_hops: int | None
    median_output_hops: float | None
    reachable_outputs: int
    temporally_reachable_outputs: int


class SpatialSubstrate(nn.Module):
    """Own a persistent population embedded in a fixed positional tensor."""

    def __init__(
        self,
        config: MnistModelConfig,
        *,
        layout: str | GraphLayout = "mnist",
        seed: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.layout = resolve_layout(layout)
        self.input_count = self.layout.input_count
        self.output_count = self.layout.output_count
        generator = torch.Generator().manual_seed(seed)
        site_count = config.site_count
        y = torch.arange(config.height).repeat_interleave(config.width)
        x = torch.arange(config.width).repeat(config.height)
        coordinates = torch.stack(
            (
                x.float() / max(1, config.width - 1) * 2 - 1,
                y.float() / max(1, config.height - 1) * 2 - 1,
            ),
            dim=1,
        )
        input_sites = self._input_sites()
        output_sites = self._output_sites()
        anchor = torch.zeros(site_count, dtype=torch.bool)
        anchor[input_sites] = True
        anchor[output_sites] = True

        interior = (x > 1) & (x < config.width - 2)
        available = max(1, int(interior.sum()))
        effective_density = min(
            config.initial_density,
            config.max_initial_neurons / available,
        )
        occupied = interior & (
            torch.rand(site_count, generator=generator) < effective_density
        )
        occupied[anchor] = True
        roles = torch.zeros(site_count, 3)
        roles[:, 0] = occupied.float()
        roles[input_sites, 1] = 1
        roles[output_sites, 2] = 1

        probe_source, probe_kernel = self._build_probes(x, y, generator)
        dendrite_source = self._initial_dendrites(occupied, input_sites, probe_source, probe_kernel)
        shape = (site_count, config.edge_slots)
        initial_weight = torch.randn(shape, generator=generator) * config.initial_weight_scale
        initial_weight[dendrite_source < 0] = 0
        initial_genotype = torch.randn(
            site_count, config.genotype_channels, generator=generator
        ) * 0.12

        self.register_buffer("coordinates", coordinates)
        self.register_buffer("input_sites", input_sites)
        self.register_buffer("output_sites", output_sites)
        self.register_buffer("anchor_mask", anchor)
        self.register_buffer("roles", roles)
        self.register_buffer("occupied", occupied)
        self.register_buffer("energy", torch.where(occupied, torch.ones(site_count), torch.zeros(site_count)))
        self.register_buffer("stimulation_ema", torch.zeros(site_count))
        self.register_buffer("load_ema", torch.zeros(site_count))
        self.register_buffer("homeostatic_stress", torch.zeros(site_count))
        self.register_buffer("stunned", torch.zeros(site_count, dtype=torch.bool))
        self.register_buffer("stun_generations", torch.zeros(site_count, dtype=torch.long))
        self.register_buffer("stun_episodes", torch.zeros(site_count, dtype=torch.long))
        self.register_buffer("excitotoxic_damage", torch.zeros(site_count))
        self.register_buffer("neuron_credit", torch.zeros(site_count))
        self.register_buffer("neuron_utility", torch.zeros(site_count))
        self.register_buffer("neuron_age", torch.zeros(site_count, dtype=torch.long))
        self.register_buffer("lineage_depth", torch.zeros(site_count, dtype=torch.long))
        self.register_buffer("parent_site", torch.full((site_count,), -1, dtype=torch.long))
        self.register_buffer("dendrite_source", dendrite_source)
        self.register_buffer("edge_age", torch.zeros(shape, dtype=torch.long))
        self.register_buffer(
            "edge_utility",
            torch.where(dendrite_source >= 0, torch.full(shape, 0.05), torch.zeros(shape)),
        )
        self.register_buffer("edge_flow", torch.zeros(shape))
        self.register_buffer("edge_credit", torch.zeros(shape))
        self.register_buffer("query_ema", torch.zeros(site_count, config.hidden_channels))
        self.register_buffer("key_ema", torch.zeros(site_count, config.hidden_channels))
        self.register_buffer("emission_ema", torch.zeros(site_count))
        self.register_buffer("candidate_source", torch.full((site_count, config.candidate_slots), -1, dtype=torch.long))
        self.register_buffer("candidate_counter", torch.zeros(site_count, config.candidate_slots))
        self.register_buffer("probe_source", probe_source)
        self.register_buffer("probe_kernel", probe_kernel)
        self.synapse_weight = nn.Parameter(initial_weight)
        self.genotype = nn.Parameter(initial_genotype)
        self.generation = 0
        self._lifecycle_generator = torch.Generator().manual_seed(seed + 97_531)
        self._diagnostic_cache: GraphDiagnostics | None = None

    def _input_sites(self) -> torch.Tensor:
        cfg = self.config
        if self.input_count == 49:
            rows = torch.linspace(7, cfg.height - 8, 7).round().long()
            sites = [
                int(rows[row]) * cfg.width
                + (1 + column if self.layout.input_side == "left" else cfg.width - 2 - column)
                for row in range(7)
                for column in range(7)
            ]
        else:
            sites = self._boundary_sites(self.input_count, self.layout.input_side)
        order = torch.tensor(self.layout.input_position_order, dtype=torch.long)
        return torch.tensor(sites, dtype=torch.long)[order]

    def _output_sites(self) -> torch.Tensor:
        cfg = self.config
        base = torch.tensor(
            self._boundary_sites(self.output_count, self.layout.output_side),
            dtype=torch.long,
        )
        order = torch.tensor(self.layout.output_position_order, dtype=torch.long)
        return base[order]

    def _boundary_sites(self, count: int, side: str) -> list[int]:
        """Pack unique semantic ports into a boundary stripe on small fields."""

        cfg = self.config
        usable_rows = cfg.height - 2
        stripe_width = math.ceil(count / usable_rows)
        if stripe_width * 2 > cfg.width - 2:
            raise ValueError(
                f"{cfg.width}×{cfg.height} field cannot fit {count} input/output ports"
            )
        sites: list[int] = []
        for index in range(count):
            lane, row_offset = divmod(index, usable_rows)
            row = row_offset + 1
            column = 1 + lane if side == "left" else cfg.width - 2 - lane
            sites.append(row * cfg.width + column)
        return sites

    def _build_probes(
        self, x: torch.Tensor, y: torch.Tensor, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        offsets: list[tuple[int, int]] = []
        while len(offsets) < cfg.candidate_probes:
            if len(offsets) < int(cfg.candidate_probes * 0.75):
                dx = -self.layout.flow_direction * int(
                    torch.randint(1, cfg.local_radius + 1, (), generator=generator)
                )
            else:
                dx = int(torch.randint(-cfg.local_radius, cfg.local_radius + 1, (), generator=generator))
            dy = int(torch.randint(-cfg.local_radius, cfg.local_radius + 1, (), generator=generator))
            if dx == 0 and dy == 0:
                continue
            if dx * dx + dy * dy <= cfg.local_radius * cfg.local_radius:
                offsets.append((dx, dy))
        offset = torch.tensor(offsets, dtype=torch.long)
        source_x = x.unsqueeze(1) + offset[:, 0]
        source_y = y.unsqueeze(1) + offset[:, 1]
        valid = (source_x >= 0) & (source_x < cfg.width) & (source_y >= 0) & (source_y < cfg.height)
        source = source_y * cfg.width + source_x
        source = source.masked_fill(~valid, -1)
        distance = offset.float().square().sum(dim=1).sqrt()
        kernel = torch.exp(-0.5 * (distance / max(1.0, cfg.local_radius * 0.58)).square())
        forward = (
            -self.layout.flow_direction * offset[:, 0]
        ).clamp_min(0).float() / max(1, cfg.local_radius)
        return source, 0.15 * kernel + 1.4 * forward

    def _initial_dendrites(
        self,
        occupied: torch.Tensor,
        input_sites: torch.Tensor,
        probes: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        safe = probes.clamp_min(0)
        score = kernel.unsqueeze(0).expand(cfg.site_count, -1).clone()
        score = score.masked_fill((probes < 0) | ~occupied[safe], -1e4)
        values, slots = torch.topk(score, cfg.edge_slots, dim=1)
        sources = probes.gather(1, slots).masked_fill(values < -100, -1)
        sources[~occupied] = -1
        sources[input_sites] = -1
        for source in sources[sources >= 0].unique().tolist():
            positions = (sources == source).nonzero(as_tuple=False)
            if positions.shape[0] > cfg.axon_slots:
                excess = positions[cfg.axon_slots :]
                sources[excess[:, 0], excess[:, 1]] = -1
        return sources

    @property
    def living_sites(self) -> torch.Tensor:
        return self.occupied.nonzero(as_tuple=False).squeeze(1)

    @property
    def active_edge_mask(self) -> torch.Tensor:
        safe = self.dendrite_source.clamp_min(0)
        target_alive = self.occupied.unsqueeze(1)
        source_alive = self.occupied[safe]
        return (self.dendrite_source >= 0) & target_alive & source_alive

    def edge_list(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target, slot = self.active_edge_mask.nonzero(as_tuple=True)
        return target, slot, self.dendrite_source[target, slot]

    def conducting_edge_list(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return physical edges whose source and target are currently responsive."""

        mask = self.active_edge_mask
        safe = self.dendrite_source.clamp_min(0)
        mask = mask & ~self.stunned.unsqueeze(1) & ~self.stunned[safe]
        target, slot = mask.nonzero(as_tuple=True)
        return target, slot, self.dendrite_source[target, slot]

    def graph_snapshot(self) -> GraphSnapshot:
        return GraphSnapshot(
            self.dendrite_source.detach().clone(),
            self.synapse_weight.detach().clone(),
            self.edge_age.detach().clone(),
            self.edge_utility.detach().clone(),
            self.edge_flow.detach().clone(),
            self.edge_credit.detach().clone(),
        )

    def graph_diagnostics(self) -> GraphDiagnostics:
        """Measure directed sensory-to-output paths, caching until topology changes."""

        if self._diagnostic_cache is not None:
            return self._diagnostic_cache
        targets, _, sources = self.edge_list()
        adjacency: list[list[int]] = [[] for _ in range(self.config.site_count)]
        for source, target in zip(sources.cpu().tolist(), targets.cpu().tolist(), strict=True):
            adjacency[source].append(target)
        distance = [-1] * self.config.site_count
        queue: deque[int] = deque()
        for site in self.input_sites[self.occupied[self.input_sites]].cpu().tolist():
            distance[site] = 0
            queue.append(site)
        while queue:
            source = queue.popleft()
            for target in adjacency[source]:
                if distance[target] < 0:
                    distance[target] = distance[source] + 1
                    queue.append(target)
        output_distance = [
            distance[site]
            for site in self.output_sites[self.occupied[self.output_sites]].cpu().tolist()
            if distance[site] >= 0
        ]
        ordered = sorted(output_distance)
        median = None
        if ordered:
            middle = len(ordered) // 2
            median = (
                float(ordered[middle])
                if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2
            )
        self._diagnostic_cache = GraphDiagnostics(
            min(ordered) if ordered else None,
            median,
            len(ordered),
            sum(value <= self.config.message_steps for value in ordered),
        )
        return self._diagnostic_cache

    @torch.no_grad()
    def record_trial(
        self,
        sites: torch.Tensor,
        stimulation: torch.Tensor,
        load: torch.Tensor,
        neuron_credit: torch.Tensor,
        edge_flow: torch.Tensor,
        edge_credit: torch.Tensor,
        advertised_query: torch.Tensor,
        advertised_key: torch.Tensor,
        emission: torch.Tensor,
        reward: float,
        *,
        homeostasis_active: bool = True,
    ) -> list[dict[str, Any]]:
        """Update slow metabolic and task statistics after one differentiable trial."""

        cfg = self.config
        self.stimulation_ema[sites] = torch.lerp(
            self.stimulation_ema[sites], stimulation, 0.06
        )
        self.load_ema[sites] = torch.lerp(self.load_ema[sites], load, 0.06)
        starvation_stress = (
            (cfg.target_stimulation_min - self.stimulation_ema[sites]).clamp_min(0)
            / max(cfg.target_stimulation_min, 1e-6)
        )
        overload_stress = (
            (self.load_ema[sites] - cfg.target_stimulation_max).clamp_min(0)
            / max(cfg.target_stimulation_max, 1e-6)
        )
        self.homeostatic_stress[sites] = torch.lerp(
            self.homeostatic_stress[sites],
            torch.maximum(starvation_stress, overload_stress).clamp_max(4),
            0.08,
        )
        self.neuron_credit.mul_(0.94)
        self.neuron_credit[sites] = self.neuron_credit[sites] + 0.06 * neuron_credit
        positive_neuron_credit = neuron_credit.clamp_min(0)
        neuron_scale = positive_neuron_credit.quantile(0.95).clamp_min(1e-12)
        neuron_signal = (positive_neuron_credit / neuron_scale).clamp_max(1)
        self.neuron_utility.mul_(0.99)
        self.neuron_utility[sites] = self.neuron_utility[sites] + 0.02 * (
            max(0.0, reward) * stimulation.clamp_max(1) + neuron_signal
        )
        self.edge_flow.lerp_(edge_flow, cfg.edge_stat_rate)
        self.edge_credit.lerp_(edge_credit, cfg.edge_stat_rate)
        self.query_ema[sites] = torch.lerp(
            self.query_ema[sites], advertised_query, 0.06
        )
        self.key_ema[sites] = torch.lerp(
            self.key_ema[sites], advertised_key, 0.06
        )
        self.emission_ema[sites] = torch.lerp(
            self.emission_ema[sites], emission, 0.06
        )
        positive_edge_credit = edge_credit.clamp_min(0)
        active = self.active_edge_mask
        if active.any():
            edge_scale = positive_edge_credit[active].quantile(0.95).clamp_min(1e-12)
            edge_signal = (positive_edge_credit / edge_scale).clamp_max(1)
            self.edge_utility.mul_(cfg.edge_utility_decay).add_(
                (1 - cfg.edge_utility_decay) * edge_signal
            )
            self.edge_utility.masked_fill_(~active, 0)

        events: list[dict[str, Any]] = []
        if homeostasis_active:
            signal = self.stimulation_ema[sites]
            traffic = self.load_ema[sites]
            healthy = (signal >= cfg.target_stimulation_min) & (traffic <= cfg.target_stimulation_max)
            starvation = (cfg.target_stimulation_min - signal).clamp_min(0) * cfg.starvation_cost
            incoming = self.active_edge_mask.sum(dim=1).float()
            _, _, sources = self.edge_list()
            outgoing = torch.bincount(sources, minlength=cfg.site_count).float()
            maintenance = (incoming[sites] + outgoing[sites]) * cfg.maintenance_cost
            task_bonus = self.neuron_utility[sites].clamp(0, 1) * cfg.task_energy_bonus
            delta = (
                healthy.float() * cfg.energy_recovery
                + task_bonus
                - starvation
                - maintenance
            )
            self.energy[sites] = (self.energy[sites] + delta).clamp(0, 1)
            responsive = ~self.stunned[sites]
            newly_stunned = (
                responsive
                & ~self.anchor_mask[sites]
                & (traffic >= cfg.stun_load_threshold)
                & bool(cfg.stun_enabled)
            )
            stunned_sites = sites[newly_stunned]
            if stunned_sites.numel():
                severity = (
                    traffic[newly_stunned] / max(cfg.stun_load_threshold, 1e-6)
                ).clamp(1, 4)
                self.stunned[stunned_sites] = True
                self.stun_generations[stunned_sites] = cfg.stun_min_generations
                self.stun_episodes[stunned_sites] += 1
                self.excitotoxic_damage[stunned_sites] += (
                    cfg.excitotoxic_damage_per_stun * cfg.overload_cost * severity
                )
                for site in stunned_sites[:160].tolist():
                    events.append(
                        {"type": "stunned", "source": site, "destination": site}
                    )
            recovering_damage = self.occupied & ~self.stunned
            self.excitotoxic_damage[recovering_damage] = (
                self.excitotoxic_damage[recovering_damage]
                - cfg.excitotoxic_damage_recovery
            ).clamp_min(0)
        self.energy[self.anchor_mask] = 1
        self.neuron_age[self.occupied] += 1
        self.edge_age[self.active_edge_mask] += 1
        self.synapse_weight.clamp_(-cfg.max_weight, cfg.max_weight)
        self.synapse_weight.masked_fill_(~self.active_edge_mask, 0)
        return events

    @torch.no_grad()
    def structural_step(
        self, *, apply_lifecycle: bool = True, apply_topology: bool = True
    ) -> StructuralUpdate:
        """Run independently gated lifecycle and topology mutations."""

        self.generation += 1
        cfg = self.config
        events: list[dict[str, Any]] = []
        changed = torch.zeros_like(self.dendrite_source, dtype=torch.bool)
        changed_sites = torch.zeros_like(self.occupied)

        invalid = (self.dendrite_source >= 0) & ~self.active_edge_mask
        prune_mask = invalid.clone()
        if apply_topology:
            pruneable = self.active_edge_mask & (self.edge_age >= cfg.edge_grace_trials)
            safe_sources = self.dendrite_source.clamp_min(0)
            pruneable &= ~self.stunned.unsqueeze(1) & ~self.stunned[safe_sources]
            pruneable &= self.edge_utility < cfg.prune_utility
            candidates = pruneable.nonzero(as_tuple=False)
            if candidates.shape[0] > cfg.max_pruned_per_generation:
                score = self.edge_utility[candidates[:, 0], candidates[:, 1]]
                candidates = candidates[
                    torch.topk(score, cfg.max_pruned_per_generation, largest=False).indices
                ]
            if candidates.numel():
                prune_mask[candidates[:, 0], candidates[:, 1]] = True
        for target, slot in prune_mask.nonzero(as_tuple=False)[:160].tolist():
            source = int(self.dendrite_source[target, slot])
            if source >= 0:
                events.append({"type": "pruned", "source": source, "destination": target})
        self._clear_edges(prune_mask)
        changed |= prune_mask

        recovered_sites = self._recover_stunned(events) if apply_lifecycle else []
        dead_sites, death_causes = (
            self._apply_death(events)
            if apply_lifecycle
            else (torch.empty(0, dtype=torch.long, device=self.occupied.device), {})
        )
        changed_sites[dead_sites] = True
        dead_edges = (self.dendrite_source >= 0) & ~self.active_edge_mask
        self._clear_edges(dead_edges)
        changed |= dead_edges
        born_sites, birth_edges = (
            self._apply_birth(events)
            if apply_lifecycle
            else (
                torch.empty(0, dtype=torch.long, device=self.occupied.device),
                torch.zeros_like(self.dendrite_source, dtype=torch.bool),
            )
        )
        changed_sites[born_sites] = True
        changed |= birth_edges
        if apply_topology:
            changed |= self._discover_and_grow(events)
        self.roles[:, 0] = self.occupied.float()
        self._diagnostic_cache = None
        return StructuralUpdate(
            events,
            changed,
            changed_sites,
            int(born_sites.numel()),
            int(dead_sites.numel()),
            death_causes,
            len(recovered_sites),
        )

    @torch.no_grad()
    def _recover_stunned(self, events: list[dict[str, Any]]) -> list[int]:
        """Give stunned cells a seeded chance to resume once their refractory time ends."""

        self.stun_generations[self.stunned] -= 1
        eligible = self.stunned & (self.stun_generations <= 0) & self.occupied
        sites = eligible.nonzero(as_tuple=False).squeeze(1)
        if not sites.numel():
            return []
        draws = torch.rand(sites.numel(), generator=self._lifecycle_generator).to(
            self.occupied.device
        )
        recovered = sites[draws < self.config.stun_recovery_probability]
        self.stunned[recovered] = False
        self.stun_generations[recovered] = 0
        recovered_list = recovered.tolist()
        for site in recovered_list[:160]:
            events.append({"type": "recovered", "source": site, "destination": site})
        return recovered_list

    @torch.no_grad()
    def _clear_edges(self, mask: torch.Tensor) -> None:
        self.dendrite_source[mask] = -1
        self.synapse_weight[mask] = 0
        self.edge_age[mask] = 0
        self.edge_utility[mask] = 0
        self.edge_flow[mask] = 0
        self.edge_credit[mask] = 0

    def _apply_death(
        self, events: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, dict[str, int]]:
        cfg = self.config
        eligible = self.occupied & ~self.anchor_mask
        eligible &= self.neuron_age >= cfg.juvenile_trials
        depleted = self.energy <= cfg.death_energy
        excitotoxic = self.excitotoxic_damage >= cfg.excitotoxic_death_threshold
        eligible &= depleted | excitotoxic
        sites = eligible.nonzero(as_tuple=False).squeeze(1)
        if sites.numel() > cfg.max_deaths_per_generation:
            danger = (
                1 - self.energy[sites]
                + self.excitotoxic_damage[sites]
                / max(cfg.excitotoxic_death_threshold, 1e-6)
            )
            sites = sites[
                torch.topk(danger, cfg.max_deaths_per_generation, largest=True).indices
            ]
        if not sites.numel():
            return sites, {}
        starvation = (cfg.target_stimulation_min - self.stimulation_ema[sites]).clamp_min(0)
        overload = (
            self.excitotoxic_damage[sites] / max(cfg.excitotoxic_death_threshold, 1e-6)
        ).clamp_min(0)
        maintenance = torch.full_like(starvation, cfg.maintenance_cost)
        cause_index = torch.stack((starvation, overload, maintenance), dim=1).argmax(dim=1)
        cause_names = ("starvation", "excitotoxicity", "maintenance")
        death_causes = {
            name: int((cause_index == index).sum())
            for index, name in enumerate(cause_names)
        }
        self.occupied[sites] = False
        self.energy[sites] = 0
        self.stimulation_ema[sites] = 0
        self.load_ema[sites] = 0
        self.homeostatic_stress[sites] = 0
        self.stunned[sites] = False
        self.stun_generations[sites] = 0
        self.stun_episodes[sites] = 0
        self.excitotoxic_damage[sites] = 0
        self.neuron_credit[sites] = 0
        self.neuron_utility[sites] = 0
        self.query_ema[sites] = 0
        self.key_ema[sites] = 0
        self.emission_ema[sites] = 0
        self.neuron_age[sites] = 0
        self.lineage_depth[sites] = 0
        self.parent_site[sites] = -1
        self.candidate_source[sites] = -1
        self.candidate_counter[sites] = 0
        for offset, site in enumerate(sites[:80].tolist()):
            events.append(
                {
                    "type": "died",
                    "source": site,
                    "destination": site,
                    "cause": cause_names[int(cause_index[offset])],
                }
            )
        return sites, death_causes

    def _best_local_source(
        self, *, require_axon_capacity: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe = self.probe_source.clamp_min(0)
        target_query = self.query_ema.unsqueeze(1)
        source_key = self.key_ema[safe]
        compatibility = torch.sigmoid(
            (target_query * source_key).sum(dim=2) / math.sqrt(self.config.hidden_channels)
        )
        active_signal = self.stimulation_ema[safe] + self.emission_ema[safe]
        source_signal = (
            0.55 * self.stimulation_ema[safe]
            + 0.30 * self.emission_ema[safe]
            + 0.15 * compatibility * active_signal
        )
        evidence = source_signal * self.probe_kernel.unsqueeze(0)
        evidence = evidence.masked_fill(
            (self.probe_source < 0) | ~self.occupied[safe] | self.stunned[safe], -1
        )
        if require_axon_capacity:
            _, _, existing_sources = self.edge_list()
            outgoing = torch.bincount(
                existing_sources, minlength=self.config.site_count
            )
            evidence = evidence.masked_fill(
                outgoing[safe] >= self.config.axon_slots, -1
            )
        best_evidence, best_slot = evidence.max(dim=1)
        best_source = self.probe_source.gather(1, best_slot.unsqueeze(1)).squeeze(1)
        return best_source, best_evidence

    def _apply_birth(
        self, events: list[dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        changed = torch.zeros_like(self.dendrite_source, dtype=torch.bool)
        best_source, evidence = self._best_local_source(require_axon_capacity=True)
        safe = self.probe_source.clamp_min(0)
        valid_probe = self.probe_source >= 0
        local_density = (
            (self.occupied[safe] & valid_probe).sum(dim=1).float()
            / valid_probe.sum(dim=1).clamp_min(1)
        )
        eligible = ~self.occupied & ~self.anchor_mask & (evidence >= cfg.birth_signal)
        eligible &= local_density <= cfg.birth_local_density_max
        candidates = eligible.nonzero(as_tuple=False).squeeze(1)
        if not candidates.numel() or cfg.births_per_generation <= 0:
            return candidates[:0], changed
        candidates = candidates[torch.argsort(evidence[candidates], descending=True)]
        _, _, existing_sources = self.edge_list()
        outgoing = torch.bincount(
            existing_sources, minlength=cfg.site_count
        ).cpu().tolist()
        accepted_sites: list[int] = []
        accepted_parents: list[int] = []
        for site in candidates.cpu().tolist():
            parent = int(best_source[site])
            if parent < 0 or outgoing[parent] >= cfg.axon_slots:
                continue
            accepted_sites.append(site)
            accepted_parents.append(parent)
            outgoing[parent] += 1
            if len(accepted_sites) >= cfg.births_per_generation:
                break
        if not accepted_sites:
            return candidates[:0], changed
        sites = torch.tensor(accepted_sites, device=self.occupied.device, dtype=torch.long)
        parents = torch.tensor(accepted_parents, device=self.occupied.device, dtype=torch.long)
        self.occupied[sites] = True
        self.energy[sites] = cfg.birth_energy
        self.neuron_age[sites] = 0
        self.stimulation_ema[sites] = 0
        self.load_ema[sites] = 0
        self.homeostatic_stress[sites] = 0
        self.stunned[sites] = False
        self.stun_generations[sites] = 0
        self.stun_episodes[sites] = 0
        self.excitotoxic_damage[sites] = 0
        self.neuron_credit[sites] = 0
        self.neuron_utility[sites] = 0
        self.query_ema[sites] = 0
        self.key_ema[sites] = 0
        self.emission_ema[sites] = 0
        self.candidate_source[sites] = -1
        self.candidate_counter[sites] = 0
        self.parent_site[sites] = parents
        self.lineage_depth[sites] = self.lineage_depth[parents] + 1
        self.genotype[sites] = (
            self.genotype[parents]
            + torch.randn(
                self.genotype[sites].shape,
                generator=self._lifecycle_generator,
                dtype=self.genotype.dtype,
            ).to(self.genotype.device) * cfg.inheritance_noise
        )
        birth_slot = torch.zeros_like(sites)
        self.dendrite_source[sites, birth_slot] = parents
        self.synapse_weight[sites, birth_slot] = cfg.initial_weight_scale
        self.edge_age[sites, birth_slot] = 0
        self.edge_utility[sites, birth_slot] = 0.04
        changed[sites, birth_slot] = True
        for site, parent in zip(
            sites[:80].tolist(), parents[:80].tolist(), strict=True
        ):
            events.append(
                {"type": "born", "source": parent, "destination": site}
            )
            events.append(
                {"type": "grown", "source": parent, "destination": site}
            )
        return sites, changed

    def _discover_and_grow(self, events: list[dict[str, Any]]) -> torch.Tensor:
        cfg = self.config
        changed = torch.zeros_like(self.dendrite_source, dtype=torch.bool)
        self.candidate_counter.mul_(cfg.candidate_decay)
        best_source, evidence = self._best_local_source()
        free = self.dendrite_source < 0
        receptive = self.occupied & ~self.stunned & (self.roles[:, 1] == 0)
        receptive &= free.any(dim=1) & (evidence >= cfg.emission_threshold)
        targets = receptive.nonzero(as_tuple=False).squeeze(1)
        if not targets.numel():
            return changed

        current = self.candidate_source[targets]
        sources = best_source[targets]
        match = current == sources.unsqueeze(1)
        has_match = match.any(dim=1)
        matched_slot = match.float().argmax(dim=1)
        weakest_slot = self.candidate_counter[targets].argmin(dim=1)
        candidate_slot = torch.where(has_match, matched_slot, weakest_slot)
        self.candidate_source[targets, candidate_slot] = sources
        self.candidate_counter[targets, candidate_slot] += evidence[targets].clamp_min(0)
        ready = self.candidate_counter[targets, candidate_slot] >= cfg.candidate_threshold
        ready_targets = targets[ready]
        ready_sources = sources[ready]
        if not ready_targets.numel():
            return changed
        _, _, existing_sources = self.edge_list()
        outgoing = torch.bincount(existing_sources, minlength=cfg.site_count).cpu().tolist()
        accepted: list[bool] = []
        for source in ready_sources.cpu().tolist():
            has_capacity = outgoing[source] < cfg.axon_slots
            accepted.append(has_capacity)
            if has_capacity:
                outgoing[source] += 1
        accepted_mask = torch.tensor(accepted, device=ready_targets.device, dtype=torch.bool)
        ready_targets = ready_targets[accepted_mask]
        ready_sources = ready_sources[accepted_mask]
        if not ready_targets.numel():
            return changed
        free_slot = free[ready_targets].float().argmax(dim=1)
        self.dendrite_source[ready_targets, free_slot] = ready_sources
        growth_sign = torch.randn(
            ready_targets.shape,
            generator=self._lifecycle_generator,
            dtype=self.synapse_weight.dtype,
        ).to(self.synapse_weight.device)
        self.synapse_weight[ready_targets, free_slot] = (
            cfg.initial_weight_scale * torch.sign(growth_sign + 1e-4)
        )
        self.edge_age[ready_targets, free_slot] = 0
        self.edge_utility[ready_targets, free_slot] = 0.04
        changed[ready_targets, free_slot] = True
        for target, source in zip(ready_targets[:160].tolist(), ready_sources[:160].tolist(), strict=True):
            events.append({"type": "grown", "source": source, "destination": target})
        self.candidate_counter[ready_targets] = 0
        self.candidate_source[ready_targets] = -1
        return changed

    @torch.no_grad()
    def lesion(self, x: float, y: float, radius: float) -> int:
        cfg = self.config
        grid_y = torch.arange(cfg.height, device=self.occupied.device).repeat_interleave(cfg.width)
        grid_x = torch.arange(cfg.width, device=self.occupied.device).repeat(cfg.height)
        damaged = (grid_x.float() - x).square() + (grid_y.float() - y).square() <= radius**2
        killed = damaged & self.occupied
        self.occupied[killed] = False
        self.energy[killed] = 0
        self.homeostatic_stress[killed] = 0
        self.neuron_age[killed] = 0
        self.lineage_depth[killed] = 0
        self.parent_site[killed] = -1
        invalid = (self.dendrite_source >= 0) & ~self.active_edge_mask
        self._clear_edges(invalid)
        self.roles[:, 0] = self.occupied.float()
        self._diagnostic_cache = None
        return int(killed.sum())
