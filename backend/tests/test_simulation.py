import torch

from petridish.channels import Channel
from petridish.config import SimulationConfig
from petridish.simulation import PetriDishSimulation


def small_config() -> SimulationConfig:
    return SimulationConfig(
        width=12,
        height=12,
        edge_slots=3,
        initial_active_slots=2,
        growth_interval=4,
        prune_age=12,
        cue_ticks=4,
        delay_ticks=4,
        response_ticks=3,
        rest_ticks=2,
    )


def test_state_shapes_and_finite_dynamics() -> None:
    simulation = PetriDishSimulation(small_config(), seed=17, device="cpu")
    simulation.step(20)

    state = simulation.state
    assert state.cells.shape == (144, 16)
    assert state.edge_destination.shape == (144, 3)
    assert state.tick == 20
    assert torch.isfinite(state.cells).all()
    assert torch.isfinite(state.edge_weight).all()
    assert int(state.edge_destination.min()) >= 0
    assert int(state.edge_destination.max()) < 144


def test_same_seed_replays_identically() -> None:
    first = PetriDishSimulation(small_config(), seed=9, device="cpu")
    second = PetriDishSimulation(small_config(), seed=9, device="cpu")
    first.stimulate("sensor_a")
    second.stimulate("sensor_a")
    first.inject_reward(0.75)
    second.inject_reward(0.75)
    first.step(18)
    second.step(18)

    assert torch.equal(first.state.edge_destination, second.state.edge_destination)
    assert torch.allclose(first.state.cells, second.state.cells)
    assert torch.allclose(first.state.edge_weight, second.state.edge_weight)


def test_lesion_kills_cells_and_cuts_incident_edges() -> None:
    simulation = PetriDishSimulation(small_config(), seed=3, device="cpu")
    damaged_count = simulation.lesion(x=6, y=6, radius=2.2)
    state = simulation.state
    grid_y = torch.arange(12).repeat_interleave(12)
    grid_x = torch.arange(12).repeat(12)
    damaged = (grid_x.float() - 6) ** 2 + (grid_y.float() - 6) ** 2 <= 2.2**2

    assert damaged_count == int(damaged.sum())
    assert torch.equal(state.cells[damaged, Channel.ALIVE], torch.zeros(damaged_count))
    incident = damaged.unsqueeze(1) | damaged[state.edge_destination]
    assert torch.equal(state.edge_gate[incident], torch.zeros_like(state.edge_gate[incident]))


def test_manual_reward_changes_active_weights() -> None:
    simulation = PetriDishSimulation(small_config(), seed=5, device="cpu")
    simulation.stimulate("sensor_a", amount=1.5, duration=8)
    simulation.step(4)
    before = simulation.state.edge_weight.clone()
    simulation.inject_reward(1.0)
    simulation.step(2)

    assert not torch.allclose(before, simulation.state.edge_weight)
