"""JSON projection for live associative-recall and language experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .mnist_hyperparameters import hyperparameter_payload
from .mnist_protocol import CHANNEL_NAMES
from .sequence_experiment import SequenceExperiment


def build_sequence_snapshot(experiment: SequenceExperiment) -> dict[str, Any]:
    """Serialize measured sequence state using the common field protocol."""

    model = experiment.model
    substrate = model.substrate
    cfg = experiment.config
    frame = experiment.last_frame
    sites = frame.sites
    roles = substrate.roles[sites]
    cells = torch.stack(
        (
            torch.ones(sites.numel(), device=experiment.device),
            frame.state[:, 0], substrate.energy[sites], frame.stimulation,
            frame.load, frame.credit, frame.input_signal, roles[:, 1], roles[:, 2],
            substrate.neuron_utility[sites], substrate.genotype[sites].norm(dim=1),
            substrate.emission_ema[sites], substrate.neuron_age[sites].float(),
            substrate.homeostatic_stress[sites], substrate.lineage_depth[sites].float(),
            substrate.parent_site[sites].float(),
        ),
        dim=1,
    )
    graph = frame.graph
    occupied = torch.zeros(cfg.site_count, dtype=torch.bool, device=experiment.device)
    occupied[sites] = True
    safe = graph.source.clamp_min(0)
    active = (graph.source >= 0) & occupied.unsqueeze(1) & occupied[safe]
    target, slot = active.nonzero(as_tuple=True)
    total_edge_count = int(target.numel())
    if total_edge_count:
        weight = graph.weight[target, slot]
        flow = graph.flow[target, slot]
        credit = graph.credit[target, slot]
        utility = graph.utility[target, slot]
        importance = (
            flow.abs() / flow.abs().max().clamp_min(1e-9)
            + credit.abs() / credit.abs().max().clamp_min(1e-12)
            + 0.02 * weight.abs() / weight.abs().max().clamp_min(1e-9)
        )
        if target.numel() > cfg.max_visible_edges:
            keep = torch.topk(importance, cfg.max_visible_edges).indices
            target, slot = target[keep], slot[keep]
            weight, flow, credit, utility = weight[keep], flow[keep], credit[keep], utility[keep]
        source = graph.source[target, slot]
        edge_payload = {
            "source": source.cpu().tolist(),
            "destination": target.cpu().tolist(),
            "weight": np.round(weight.detach().cpu().numpy(), 5).tolist(),
            "age": graph.age[target, slot].cpu().tolist(),
            "utility": np.round(utility.cpu().numpy(), 6).tolist(),
            "flow": np.round(flow.cpu().numpy(), 6).tolist(),
            "credit": np.round(credit.cpu().numpy(), 7).tolist(),
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
    shared = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name not in {"substrate.synapse_weight", "substrate.genotype"}
    )
    active_parameters = shared + living_count * cfg.genotype_channels + total_edge_count
    visible_length = len(experiment.last_tokens)
    position = (
        visible_length - 1
        if frame.token_position < 0
        else min(frame.token_position, visible_length - 1)
    )
    vocabulary = experiment.task.vocabulary
    target_ids = experiment.last_targets.tolist()
    prediction_ids = experiment.last_predictions.tolist()
    confidence = float(experiment.last_confidences[position])
    return {
        "type": "snapshot",
        "experiment": experiment.experiment_name,
        "tick": experiment.tick,
        "field": {
            "width": cfg.width, "height": cfg.height, "channels": CHANNEL_NAMES,
            "indices": sites.cpu().tolist(),
            "cells": np.round(cells.detach().cpu().numpy(), 6).tolist(),
        },
        "edges": edge_payload,
        "events": experiment.events,
        "task": {
            "kind": "sequence", "taskKey": experiment.task.key,
            "title": experiment.task.title, "description": experiment.task.description,
            "phase": frame.stage, "vocabulary": list(vocabulary),
            "tokens": [vocabulary[index] for index in experiment.last_tokens.tolist()],
            "tokenIds": experiment.last_tokens.tolist(),
            "targets": [None if index < 0 else vocabulary[index] for index in target_ids],
            "targetIds": target_ids,
            "predictions": [
                "—" if index < 0 else vocabulary[index] for index in prediction_ids
            ],
            "predictionIds": prediction_ids, "position": position,
            "accuracy": round(experiment.rolling_accuracy, 4),
            "testAccuracy": None if experiment.test_accuracy is None else round(experiment.test_accuracy, 4),
            "confidence": round(confidence, 4), "loss": round(experiment.last_loss, 5),
            "perplexity": round(float(np.exp(min(20.0, max(-20.0, experiment.last_loss)))), 4),
            "recallPairs": experiment.recall_pair_count,
            "recallMaxPairs": 3 if experiment.task.key == "associative_recall" else 0,
            "stageAccuracy": round(
                sum(experiment.stage_accuracy_history) / len(experiment.stage_accuracy_history), 4
            ) if experiment.stage_accuracy_history else 0.0,
            "datasetName": experiment.task.dataset_name,
            "datasetCharacters": experiment.task.dataset_characters,
            "contextLength": experiment.task.sequence_length,
            "interactive": experiment.task.encode is not None,
            "interactivePrompt": experiment.interactive_prompt,
            "generatedText": experiment.generated_text,
            "nextTokenPrediction": experiment.next_token_prediction,
            "sourceUrl": experiment.task.source_url,
            "reward": round(experiment.last_reward, 5),
            "seenExamples": experiment.seen_examples, "trainingStep": experiment.training_step,
            "trialStep": frame.step, "trialSteps": visible_length + 2,
            "generation": substrate.generation,
            "structuralWarmupRemaining": experiment.structural_warmup_remaining,
            "lifecycleWarmupRemaining": experiment.lifecycle_warmup_remaining,
            "lifecycleActive": experiment.lifecycle_active,
            "lifecycleReason": experiment.lifecycle_reason,
            "learningPhase": experiment.learning_phase,
            "structureUnlockReason": experiment.structure_unlock_reason,
            "births": experiment.last_births, "deaths": experiment.last_deaths,
            "deathCauses": experiment.last_death_causes,
            "cumulativeBirths": experiment.cumulative_births,
            "cumulativeDeaths": experiment.cumulative_deaths,
            "cumulativeDeathCauses": experiment.cumulative_death_causes,
        },
        "metrics": {
            "reward": round(experiment.last_reward, 5),
            "rollingReward": round(experiment.rolling_reward, 5),
            "loss": round(experiment.rolling_loss, 5), "livingCells": living_count,
            "meanEnergy": round(float(substrate.energy[sites].mean()), 5),
            "meanAge": round(float(substrate.neuron_age[sites].float().mean()), 3),
            "stressedCells": int((substrate.homeostatic_stress[sites] >= 1).sum()),
            "turnoverEvents": experiment.cumulative_births + experiment.cumulative_deaths,
            "edgeCount": total_edge_count, "visibleEdgeCount": len(edge_payload["source"]),
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
        "configuration": {"parameters": hyperparameter_payload(cfg, include_sequence=True)},
    }


__all__ = ["build_sequence_snapshot"]
