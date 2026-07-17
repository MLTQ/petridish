"""JSON-safe snapshot serialization for live observers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .channels import Channel
from .mnist_experiment import MnistExperiment
from .mnist_protocol import build_mnist_snapshot
from .simulation import PetriDishSimulation


CHANNEL_NAMES = [channel.name.lower() for channel in Channel]


def build_snapshot(simulation: PetriDishSimulation | MnistExperiment) -> dict[str, Any]:
    """Copy the current authoritative state to a compact JSON-safe structure."""

    if isinstance(simulation, MnistExperiment):
        return build_mnist_snapshot(simulation)

    state = simulation.state
    cells = np.round(state.cells.detach().cpu().float().numpy(), 4)
    active = (state.edge_gate > 0.5).nonzero(as_tuple=False)
    if active.numel():
        source = active[:, 0]
        slot = active[:, 1]
        destination = state.edge_destination[source, slot]
        weight = state.edge_weight[source, slot]
        age = state.edge_age[source, slot]
        utility = state.edge_utility[source, slot]
        edge_payload = {
            "source": source.detach().cpu().tolist(),
            "destination": destination.detach().cpu().tolist(),
            "weight": np.round(weight.detach().cpu().float().numpy(), 4).tolist(),
            "age": age.detach().cpu().tolist(),
            "utility": np.round(utility.detach().cpu().float().numpy(), 5).tolist(),
            "flow": np.zeros(len(source), dtype=np.float32).tolist(),
            "credit": np.zeros(len(source), dtype=np.float32).tolist(),
        }
        mean_weight = float(weight.abs().mean())
    else:
        edge_payload = {
            "source": [], "destination": [], "weight": [], "age": [],
            "utility": [], "flow": [], "credit": [],
        }
        mean_weight = 0.0

    observation = simulation.last_observation
    return {
        "type": "snapshot",
        "experiment": "xor",
        "tick": state.tick,
        "field": {
            "width": simulation.config.width,
            "height": simulation.config.height,
            "channels": CHANNEL_NAMES,
            "indices": None,
            "cells": cells.tolist(),
        },
        "edges": edge_payload,
        "events": list(simulation.events),
        "task": {
            "kind": "xor",
            "phase": observation.phase,
            "bitA": observation.bit_a,
            "bitB": observation.bit_b,
            "target": observation.target,
            "prediction": observation.prediction,
            "accuracy": round(observation.accuracy, 4),
        },
        "metrics": {
            "reward": round(simulation.last_reward, 4),
            "rollingReward": round(simulation.rolling_reward, 5),
            "loss": None,
            "livingCells": int((state.cells[:, Channel.ALIVE] > 0.5).sum()),
            "edgeCount": len(edge_payload["source"]),
            "visibleEdgeCount": len(edge_payload["source"]),
            "meanWeight": round(mean_weight, 5),
            "synapseUpdateRatio": None,
            "structureLocked": False,
            "device": str(simulation.device),
        },
    }
