"""Sparse JSON projection for the configurable persistent MNIST organism."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .mnist_experiment import MnistExperiment
from .mnist_hyperparameters import hyperparameter_payload


CHANNEL_NAMES = [
    "alive",
    "activation",
    "energy",
    "stimulation",
    "load",
    "credit",
    "input_signal",
    "sensor_id",
    "motor_id",
    "utility",
    "genotype_norm",
    "emission",
    "age",
    "stress",
    "lineage",
    "parent",
    "stunned",
    "excitotoxic_damage",
]


def build_mnist_snapshot(experiment: MnistExperiment) -> dict[str, Any]:
    """Serialize occupied sites and the most informative real edges."""

    model = experiment.model
    substrate = model.substrate
    cfg = experiment.config
    frame = experiment.last_frame
    sites = frame.sites
    roles = substrate.roles[sites]
    cells = torch.stack(
        (
            torch.ones(sites.numel(), device=experiment.device),
            frame.state[:, 0],
            substrate.energy[sites],
            frame.stimulation,
            frame.load,
            frame.credit,
            frame.input_signal,
            roles[:, 1],
            roles[:, 2],
            substrate.neuron_utility[sites],
            substrate.genotype[sites].norm(dim=1),
            substrate.emission_ema[sites],
            substrate.neuron_age[sites].float(),
            substrate.homeostatic_stress[sites],
            substrate.lineage_depth[sites].float(),
            substrate.parent_site[sites].float(),
            substrate.stunned[sites].float(),
            substrate.excitotoxic_damage[sites],
        ),
        dim=1,
    )

    graph = frame.graph
    occupied = torch.zeros(cfg.site_count, dtype=torch.bool, device=experiment.device)
    occupied[sites] = True
    safe_source = graph.source.clamp_min(0)
    active = (graph.source >= 0) & occupied.unsqueeze(1) & occupied[safe_source]
    target, slot = active.nonzero(as_tuple=True)
    total_edge_count = int(target.numel())
    if total_edge_count:
        weight = graph.weight[target, slot]
        flow = graph.flow[target, slot]
        credit = graph.credit[target, slot]
        utility = graph.utility[target, slot]
        flow_score = flow.abs() / flow.abs().max().clamp_min(1e-9)
        credit_score = credit.abs() / credit.abs().max().clamp_min(1e-12)
        weight_score = weight.abs() / weight.abs().max().clamp_min(1e-9)
        utility_score = utility.abs() / utility.abs().max().clamp_min(1e-9)
        importance = flow_score + credit_score + 0.02 * weight_score + 0.01 * utility_score
        if target.numel() > cfg.max_visible_edges:
            keep = torch.topk(importance, cfg.max_visible_edges).indices
            target, slot = target[keep], slot[keep]
            weight, flow, credit, utility = weight[keep], flow[keep], credit[keep], utility[keep]
        source = graph.source[target, slot]
        edge_payload = {
            "source": source.detach().cpu().tolist(),
            "destination": target.detach().cpu().tolist(),
            "weight": np.round(weight.detach().cpu().numpy(), 5).tolist(),
            "age": graph.age[target, slot].detach().cpu().tolist(),
            "utility": np.round(utility.detach().cpu().numpy(), 6).tolist(),
            "flow": np.round(flow.detach().cpu().numpy(), 6).tolist(),
            "credit": np.round(credit.detach().cpu().numpy(), 7).tolist(),
        }
        mean_weight = float(graph.weight[active].abs().mean())
    else:
        edge_payload = {
            "source": [], "destination": [], "weight": [], "age": [],
            "utility": [], "flow": [], "credit": [],
        }
        mean_weight = 0.0

    diagnostics = substrate.graph_diagnostics()
    living_count = int(substrate.occupied.sum())
    shared_without_site_genotype = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name not in {"substrate.synapse_weight", "substrate.genotype"}
    )
    active_parameters = (
        shared_without_site_genotype
        + living_count * cfg.genotype_channels
        + total_edge_count
    )
    stage = experiment.curriculum_stage
    stage_accuracy = (
        sum(experiment.stage_accuracy_history) / len(experiment.stage_accuracy_history)
        if experiment.stage_accuracy_history else 0.0
    )

    return {
        "type": "snapshot",
        "experiment": "mnist",
        "tick": experiment.tick,
        "field": {
            "width": cfg.width,
            "height": cfg.height,
            "channels": CHANNEL_NAMES,
            "indices": sites.detach().cpu().tolist(),
            "cells": np.round(cells.detach().cpu().numpy(), 6).tolist(),
        },
        "edges": edge_payload,
        "events": experiment.events,
        "task": {
            "kind": "mnist",
            "phase": frame.stage,
            "target": experiment.last_label,
            "prediction": experiment.last_prediction,
            "accuracy": round(experiment.rolling_accuracy, 4),
            "testAccuracy": None if experiment.test_accuracy is None else round(experiment.test_accuracy, 4),
            "confidence": round(experiment.last_confidence, 4),
            "loss": round(experiment.last_loss, 5),
            "reward": round(experiment.last_reward, 5),
            "epoch": round(experiment.epoch, 4),
            "seenExamples": experiment.seen_examples,
            "trainingStep": experiment.training_step,
            "trialStep": frame.step,
            "trialSteps": cfg.trace_steps,
            "generation": substrate.generation,
            "structuralWarmupRemaining": experiment.structural_warmup_remaining,
            "lifecycleWarmupRemaining": experiment.lifecycle_warmup_remaining,
            "lifecycleActive": experiment.lifecycle_active,
            "lifecycleReason": experiment.lifecycle_reason,
            "learningPhase": experiment.learning_phase,
            "structureUnlockReason": experiment.structure_unlock_reason,
            "curriculumStage": experiment.curriculum_stage_index + 1,
            "curriculumStageCount": len(experiment.curriculum),
            "curriculumExamples": stage.examples,
            "curriculumTargetAccuracy": stage.target_accuracy,
            "curriculumStageAccuracy": round(stage_accuracy, 4),
            "curriculumStageUpdates": experiment.curriculum_stage_updates,
            "births": experiment.last_births,
            "deaths": experiment.last_deaths,
            "deathCauses": experiment.last_death_causes,
            "cumulativeBirths": experiment.cumulative_births,
            "cumulativeDeaths": experiment.cumulative_deaths,
            "cumulativeDeathCauses": experiment.cumulative_death_causes,
            "image": np.round(experiment.last_image.reshape(-1).numpy(), 4).tolist(),
        },
        "metrics": {
            "reward": round(experiment.last_reward, 5),
            "rollingReward": round(experiment.rolling_reward, 5),
            "loss": round(experiment.rolling_loss, 5),
            "livingCells": living_count,
            "meanEnergy": round(float(substrate.energy[sites].mean()), 5),
            "meanAge": round(float(substrate.neuron_age[sites].float().mean()), 3),
            "stressedCells": int((substrate.homeostatic_stress[sites] >= 1).sum()),
            "stunnedCells": int(substrate.stunned[sites].sum()),
            "meanExcitotoxicDamage": round(
                float(substrate.excitotoxic_damage[sites].mean()), 5
            ),
            "turnoverEvents": experiment.cumulative_births + experiment.cumulative_deaths,
            "edgeCount": total_edge_count,
            "visibleEdgeCount": len(edge_payload["source"]),
            "meanWeight": round(mean_weight, 5),
            "synapseUpdateRatio": round(experiment.last_synapse_update_ratio, 7),
            "structureLocked": not experiment.structure_unlocked,
            "meanAttentionEntropy": round(experiment.last_mean_attention_entropy, 5),
            "minimumOutputHops": diagnostics.minimum_output_hops,
            "medianOutputHops": diagnostics.median_output_hops,
            "reachableOutputs": diagnostics.reachable_outputs,
            "temporallyReachableOutputs": diagnostics.temporally_reachable_outputs,
            "activeParameters": active_parameters,
            "parametersPerLivingCell": round(active_parameters / max(1, living_count), 3),
            "device": str(experiment.device),
        },
        "configuration": {
            "parameters": hyperparameter_payload(cfg),
        },
    }
