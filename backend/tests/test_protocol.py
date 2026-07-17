from petridish.config import SimulationConfig
from petridish.protocol import build_snapshot
from petridish.simulation import PetriDishSimulation


def test_snapshot_arrays_are_aligned_and_json_safe() -> None:
    config = SimulationConfig(width=10, height=10, edge_slots=2, initial_active_slots=1)
    simulation = PetriDishSimulation(config, seed=4, device="cpu")
    simulation.step(3)
    snapshot = build_snapshot(simulation)

    assert snapshot["type"] == "snapshot"
    assert snapshot["experiment"] == "xor"
    assert len(snapshot["field"]["cells"]) == 100
    assert len(snapshot["field"]["channels"]) == 16
    edge_lengths = {len(values) for values in snapshot["edges"].values()}
    assert len(edge_lengths) == 1
    assert snapshot["metrics"]["edgeCount"] == len(snapshot["edges"]["source"])
    assert snapshot["task"]["phase"] in {"cue", "delay", "response", "rest"}
    assert snapshot["task"]["kind"] == "xor"
