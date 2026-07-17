"""Viewer projection for the live self-assembling MNIST experiment."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .channels import Channel
from .mnist_experiment import MnistExperiment


CHANNEL_NAMES = [channel.name.lower() for channel in Channel]


def build_mnist_snapshot(experiment: MnistExperiment) -> dict[str, Any]:
    """Project the current developmental frame into the common viewer schema."""

    model = experiment.model
    cfg = experiment.config
    device = experiment.device
    frame = experiment.last_frame
    hidden = frame.state
    cells = torch.zeros(cfg.cell_count, len(Channel), device=device)
    cells[:, Channel.ALIVE] = model.lesion_mask
    cells[:, Channel.ACTIVATION] = hidden[:, 0]
    cells[:, Channel.PHASE_SIN] = hidden[:, 1]
    cells[:, Channel.PHASE_COS] = hidden[:, 2]
    cells[:, Channel.MEMORY_0 : Channel.MEMORY_3 + 1] = hidden[:, 3:7]
    cells[:, Channel.ENERGY] = torch.sigmoid(hidden[:, 7])
    cells[:, Channel.AXON_GROWTH] = frame.axon_request
    cells[:, Channel.DENDRITE_GROWTH] = frame.receptor_request
    cells[:, Channel.REWARD_TRACE] = frame.sensory_signal
    cells[:, Channel.POSITION_X : Channel.POSITION_Y + 1] = model.coordinates
    cells[:, Channel.SENSOR_ID] = model.sensor_identity
    cells[:, Channel.MOTOR_ID] = model.motor_identity

    graph = frame.graph
    destination = graph.destination.clamp_min(0)
    edge_alive = model.lesion_mask.unsqueeze(1) * model.lesion_mask[destination]
    active = (graph.destination >= 0) & (graph.strength > 0.18) & (edge_alive > 0.5)
    positions = active.nonzero(as_tuple=False)
    if positions.numel():
        source, slot = positions[:, 0], positions[:, 1]
        selected_destination = graph.destination[source, slot]
        effective_weight = graph.weight[source, slot] * graph.strength[source, slot]
        edge_payload = {
            "source": source.detach().cpu().tolist(),
            "destination": selected_destination.detach().cpu().tolist(),
            "weight": np.round(effective_weight.detach().cpu().numpy(), 4).tolist(),
            "age": graph.age[source, slot].detach().cpu().tolist(),
            "utility": np.round(graph.utility[source, slot].detach().cpu().numpy(), 5).tolist(),
        }
        mean_weight = float(effective_weight.detach().abs().mean())
    else:
        edge_payload = {"source": [], "destination": [], "weight": [], "age": [], "utility": []}
        mean_weight = 0.0

    return {
        "type": "snapshot",
        "experiment": "mnist",
        "tick": experiment.tick,
        "field": {
            "width": cfg.width,
            "height": cfg.height,
            "channels": CHANNEL_NAMES,
            "cells": np.round(cells.detach().cpu().numpy(), 4).tolist(),
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
            "epoch": round(experiment.epoch, 4),
            "seenExamples": experiment.seen_examples,
            "trainingStep": experiment.training_step,
            "assemblyStep": frame.step,
            "assemblySteps": cfg.total_steps,
            "tokenRow": frame.token_row,
            "routingRound": 0 if frame.step == 0 else 1 + (frame.step - 1) // cfg.routing_interval,
            "image": np.round(experiment.last_image.reshape(-1).numpy(), 4).tolist(),
        },
        "metrics": {
            "reward": round(-experiment.last_loss, 4),
            "rollingReward": round(-experiment.rolling_loss, 5),
            "loss": round(experiment.rolling_loss, 5),
            "livingCells": int((model.lesion_mask > 0.5).sum()),
            "edgeCount": len(edge_payload["source"]),
            "meanWeight": round(mean_weight, 5),
            "device": str(device),
        },
    }
