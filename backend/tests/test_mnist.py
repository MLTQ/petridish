from dataclasses import fields, replace

import pytest
import torch
from torch.utils.data import TensorDataset

from petridish.mnist_experiment import MnistExperiment
from petridish.lifecycle_profiles import apply_lifecycle_profile
from petridish.mnist_curriculum import build_curriculum
from petridish.mnist_model import CellularGraphClassifier, MnistModelConfig
from petridish.mnist_hyperparameters import (
    FIELD_SIZES,
    SPECS,
    configured,
    hyperparameter_payload,
)
from petridish.protocol import build_snapshot


def tiny_config() -> MnistModelConfig:
    return MnistModelConfig(
        width=24,
        height=52,
        hidden_channels=8,
        edge_slots=2,
        candidate_slots=2,
        candidate_probes=12,
        initial_density=0.4,
        local_radius=6,
        message_steps=6,
        batch_size=4,
        structural_interval=2,
        evaluation_interval=2,
        evaluation_batches=1,
        juvenile_trials=1,
    )


def synthetic_digits(count: int = 24) -> TensorDataset:
    generator = torch.Generator().manual_seed(44)
    images = torch.rand(count, 1, 28, 28, generator=generator)
    labels = torch.arange(count) % 10
    return TensorDataset(images, labels)


def easy_spatial_digits() -> TensorDataset:
    images = torch.zeros(16, 1, 28, 28)
    labels = torch.arange(16) % 4
    for index, label in enumerate(labels.tolist()):
        row, column = divmod(label, 2)
        images[index, 0, row * 14 : (row + 1) * 14, column * 14 : (column + 1) * 14] = 1
    return TensorDataset(images, labels)


def test_default_field_is_compact_and_every_hyperparameter_is_controllable() -> None:
    config = MnistModelConfig()

    assert (config.width, config.height, config.local_radius) == (64, 64, 8)
    numeric_fields = {
        field.name
        for field in fields(MnistModelConfig)
        if isinstance(getattr(config, field.name), (int, float))
    }
    assert set(SPECS) == numeric_fields
    assert config.cell_architecture == "gru"
    assert SPECS["local_radius"].minimum == 1
    assert configured(config, {"local_radius": 1}).local_radius == 1
    local_radius = next(
        parameter
        for parameter in hyperparameter_payload(config)
        if parameter["key"] == "local_radius"
    )
    assert local_radius["max"] == 31


def test_hyperparameter_changes_are_typed_and_cross_validated() -> None:
    config = configured(
        MnistModelConfig(),
        {"width": 80, "height": 80, "local_radius": 10, "learning_rate": 0.0025},
    )

    assert config.width == 80
    assert isinstance(config.width, int)
    assert config.learning_rate == pytest.approx(0.0025)
    with pytest.raises(ValueError, match="target_stimulation_min"):
        configured(config, {"target_stimulation_min": 0.5, "target_stimulation_max": 0.1})
    with pytest.raises(ValueError, match="local_radius"):
        configured(config, {"width": 32, "height": 32, "local_radius": 16})

    smallest = configured(config, {"field_size": 16})
    largest = configured(config, {"field_size": 1024})
    assert FIELD_SIZES == (16, 32, 64, 128, 256, 512, 1024)
    assert (smallest.width, smallest.height, smallest.local_radius) == (16, 16, 7)
    assert (largest.width, largest.height) == (1024, 1024)
    with pytest.raises(ValueError, match="power of two"):
        configured(config, {"field_size": 128 + 1})


def test_persistent_spatial_graph_carries_gradients_without_topology_reset() -> None:
    config = tiny_config()
    model = CellularGraphClassifier(config, seed=3)
    input_columns = (model.substrate.input_sites % config.width).unique().tolist()
    input_rows = sorted((model.substrate.input_sites // config.width).tolist())
    images, labels = synthetic_digits(8).tensors
    before = model.substrate.dendrite_source.clone()
    result = model(images[:4])
    loss = torch.nn.functional.cross_entropy(result.logits, labels[:4]) + model.regularization()
    loss.backward()

    assert result.logits.shape == (4, 10)
    assert input_columns == [1]
    assert input_rows == list(range(1, 50))
    assert result.trajectory_logits.shape == (4, config.message_steps, 10)
    assert result.final_state.shape[1] == model.substrate.occupied.sum()
    assert len(result.frames) == config.message_steps + 1
    assert torch.equal(before, model.substrate.dendrite_source)
    assert model.cell_rule.weight_ih.grad is not None
    assert model.patch_encoder[0].weight.grad is not None
    assert model.output_key.weight.grad is not None
    assert model.readout_scale.grad is not None
    assert model.readout_scale.grad.abs() > 0
    assert model.output_bank_readout.weight.grad is not None
    assert model.output_bank_readout.weight.grad.abs().sum() > 0
    assert model.message_query.weight.grad is not None
    assert model.message_query.weight.grad.abs().sum() > 0
    assert model.substrate.genotype.grad is not None
    assert model.substrate.genotype.grad.abs().sum() > 0
    assert model.substrate.synapse_weight.grad is not None
    assert model.substrate.synapse_weight.grad.abs().sum() > 0
    assert result.edge_flow.sum() > 0
    with torch.no_grad():
        blank_logits = model(torch.zeros_like(images[:4]), capture_trace=False).logits
    assert (result.logits.detach() - blank_logits).abs().max() > 1e-5


def test_mnist_experiment_emits_real_feedback_and_sparse_snapshot() -> None:
    dataset = synthetic_digits()
    experiment = MnistExperiment(
        tiny_config(),
        seed=5,
        device="cpu",
        train_dataset=dataset,
        test_dataset=dataset,
    )
    initial_graph = experiment.model.substrate.dendrite_source.clone()
    experiment.step(len(experiment.last_trace))
    assert experiment.training_step == 1
    assert torch.equal(initial_graph, experiment.model.substrate.dendrite_source)
    experiment.step(experiment.config.message_steps + 1)
    snapshot = build_snapshot(experiment)
    held_out = experiment.evaluate(2)

    assert 0 <= held_out <= 1
    assert snapshot["experiment"] == "mnist"
    assert snapshot["task"]["phase"] == "feedback"
    assert snapshot["task"]["generation"] == 0
    assert snapshot["task"]["structuralWarmupRemaining"] == experiment.config.structural_warmup_trials - 1
    assert snapshot["task"]["lifecycleWarmupRemaining"] == experiment.config.lifecycle_warmup_trials - 1
    assert snapshot["task"]["lifecycleActive"] is False
    assert len(snapshot["task"]["image"]) == 784
    assert len(snapshot["field"]["indices"]) == snapshot["metrics"]["livingCells"]
    assert len(snapshot["field"]["cells"]) == len(snapshot["field"]["indices"])
    assert max(abs(row[5]) for row in snapshot["field"]["cells"]) > 0
    assert snapshot["metrics"]["edgeCount"] > 0
    assert snapshot["metrics"]["structureLocked"] is True
    assert snapshot["task"]["learningPhase"] == "readout"
    assert snapshot["task"]["curriculumExamples"] == 20
    assert snapshot["metrics"]["synapseUpdateRatio"] == 0
    assert snapshot["metrics"]["minimumOutputHops"] is not None
    assert snapshot["metrics"]["activeParameters"] > snapshot["metrics"]["edgeCount"]
    assert snapshot["metrics"]["meanEnergy"] > 0
    assert snapshot["metrics"]["turnoverEvents"] == 0
    assert len({len(values) for values in snapshot["edges"].values()}) == 1
    mnist_parameter_keys = {
        parameter["key"] for parameter in snapshot["configuration"]["parameters"]
    }
    expected_parameter_keys = (
        set(SPECS)
        - {"width", "height", "broadcast_slots", "broadcast_gain", "broadcast_decay",
           "fast_weight_gain", "fast_weight_decay", "binding_memory_gain",
           "binding_memory_temperature", "binding_token_values",
           "binding_address_regularization"}
    ) | {"field_size"}
    assert mnist_parameter_keys == expected_parameter_keys
    assert "broadcast_slots" not in mnist_parameter_keys
    assert "broadcast_gain" not in mnist_parameter_keys
    assert "broadcast_decay" not in mnist_parameter_keys
    assert "fast_weight_gain" not in mnist_parameter_keys
    assert "fast_weight_decay" not in mnist_parameter_keys
    assert "field_size" in mnist_parameter_keys
    assert "width" not in mnist_parameter_keys
    assert "height" not in mnist_parameter_keys


def test_fixed_connectome_learns_an_easy_spatial_classification_batch() -> None:
    dataset = easy_spatial_digits()
    experiment = MnistExperiment(
        replace(tiny_config(), readout_only_trials=0, synapse_unlock_trials=0),
        seed=5,
        device="cpu",
        train_dataset=dataset,
        test_dataset=dataset,
    )
    images, labels = dataset.tensors

    with torch.no_grad():
        initial = torch.nn.functional.cross_entropy(
            experiment.model(images, capture_trace=False).logits, labels
        )
    for _ in range(30):
        experiment._train_trial()
    with torch.no_grad():
        final = torch.nn.functional.cross_entropy(
            experiment.model(images, capture_trace=False).logits, labels
        )

    assert float(final) < float(initial) * 0.85
    assert experiment.model.substrate.generation == 0
    assert experiment.last_synapse_update_ratio > 0.001


def test_curriculum_starts_with_a_balanced_overfit_subset() -> None:
    stages = build_curriculum(synthetic_digits(100), seed=7)
    first_labels = torch.tensor([stages[0].dataset[index][1] for index in range(20)])

    assert [stage.examples for stage in stages] == [20, 100]
    assert stages[0].target_accuracy == pytest.approx(0.98)
    assert torch.bincount(first_labels, minlength=10).tolist() == [2] * 10


def test_mnist_lesion_removes_neurons_and_every_incident_dendrite() -> None:
    dataset = synthetic_digits()
    experiment = MnistExperiment(
        tiny_config(),
        seed=8,
        device="cpu",
        train_dataset=dataset,
        test_dataset=dataset,
    )
    before = int(experiment.model.substrate.occupied.sum())
    damaged = experiment.lesion(12, 12, 2.5)
    snapshot = build_snapshot(experiment)
    living = set(snapshot["field"]["indices"])

    assert damaged > 0
    assert snapshot["metrics"]["livingCells"] == before - damaged
    assert all(source in living for source in snapshot["edges"]["source"])
    assert all(destination in living for destination in snapshot["edges"]["destination"])


def test_structural_cycle_kills_depleted_nonanchors_but_preserves_interfaces() -> None:
    model = CellularGraphClassifier(tiny_config(), seed=12)
    substrate = model.substrate
    victims = (substrate.occupied & ~substrate.anchor_mask).nonzero(as_tuple=False).squeeze(1)[:5]
    substrate.neuron_age[victims] = 20
    substrate.energy[victims] = 0
    update = substrate.structural_step()

    assert update.deaths >= 5
    assert update.death_causes["starvation"] >= 5
    assert not substrate.occupied[victims].any()
    assert substrate.occupied[substrate.input_sites].all()
    assert substrate.occupied[substrate.output_sites].all()


def test_prune_only_topology_never_calls_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    substrate = CellularGraphClassifier(tiny_config(), seed=31).substrate

    def unexpected_growth(events: list[dict[str, object]]) -> torch.Tensor:
        raise AssertionError("prune-only phase attempted dendrite growth")

    monkeypatch.setattr(substrate, "_discover_and_grow", unexpected_growth)

    update = substrate.structural_step(
        apply_lifecycle=False, apply_topology=True, allow_growth=False
    )

    assert update.grown_edges == 0


def test_adaptive_growth_accepts_only_the_strongest_bounded_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        tiny_config(), max_grown_per_generation=3,
        max_pruned_per_generation=0, candidate_decay=1,
        candidate_threshold=0.5, emission_threshold=0, axon_slots=512,
    )
    substrate = CellularGraphClassifier(config, seed=32).substrate
    source = int(substrate.input_sites[0])
    best_sources = torch.full(
        (config.site_count,), source, dtype=torch.long,
        device=substrate.occupied.device,
    )
    evidence = torch.linspace(
        0.5, 1.5, config.site_count, device=substrate.occupied.device
    )
    monkeypatch.setattr(
        substrate, "_best_local_source",
        lambda require_axon_capacity=False: (best_sources, evidence),
    )

    update = substrate.structural_step(
        apply_lifecycle=False, apply_topology=True, allow_growth=True
    )

    assert update.grown_edges == 3
    assert int(update.changed_edges.sum()) == 3


def test_zero_growth_budget_defers_ready_proposals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        tiny_config(), max_grown_per_generation=0,
        max_pruned_per_generation=0, candidate_decay=1,
        candidate_threshold=0.5, emission_threshold=0, axon_slots=512,
    )
    substrate = CellularGraphClassifier(config, seed=33).substrate
    source = int(substrate.input_sites[0])
    best_sources = torch.full(
        (config.site_count,), source, dtype=torch.long,
        device=substrate.occupied.device,
    )
    evidence = torch.ones(config.site_count, device=substrate.occupied.device)
    monkeypatch.setattr(
        substrate, "_best_local_source",
        lambda require_axon_capacity=False: (best_sources, evidence),
    )

    update = substrate.structural_step(
        apply_lifecycle=False, apply_topology=True, allow_growth=True
    )

    assert update.grown_edges == 0
    assert float(substrate.candidate_counter.max()) >= config.candidate_threshold


def test_axon_economy_requires_reserves_and_charges_both_neurons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        tiny_config(), max_grown_per_generation=8,
        max_pruned_per_generation=0, candidate_decay=1,
        candidate_threshold=0.5, emission_threshold=0, axon_slots=512,
        axon_growth_cost=0.2, axon_growth_energy_reserve=0.35,
        new_axon_initial_utility=0,
    )
    substrate = CellularGraphClassifier(config, seed=34).substrate
    eligible = (
        substrate.occupied
        & ~substrate.stunned
        & (substrate.roles[:, 1] == 0)
        & (substrate.dendrite_source < 0).any(dim=1)
    ).nonzero(as_tuple=False).squeeze(1)
    target = int(eligible[0])
    source = int(substrate.input_sites[0])
    best_sources = torch.full(
        (config.site_count,), source, dtype=torch.long,
        device=substrate.occupied.device,
    )
    evidence = torch.full(
        (config.site_count,), -1.0, device=substrate.occupied.device
    )
    evidence[target] = 1
    monkeypatch.setattr(
        substrate, "_best_local_source",
        lambda require_axon_capacity=False: (best_sources, evidence),
    )
    substrate.energy[source] = 0.4
    substrate.energy[target] = 0.4

    blocked = substrate.structural_step(
        apply_lifecycle=False, apply_topology=True, allow_growth=True
    )

    assert blocked.grown_edges == 0
    assert int(substrate.last_growth_proposals) == 1
    assert int(substrate.last_growth_energy_blocked) == 1
    assert float(substrate.last_growth_energy_spent) == 0

    substrate.energy[source] = 0.8
    substrate.energy[target] = 0.8
    accepted = substrate.structural_step(
        apply_lifecycle=False, apply_topology=True, allow_growth=True
    )

    assert accepted.grown_edges == 1
    assert float(substrate.energy[source]) == pytest.approx(0.6)
    assert float(substrate.energy[target]) == pytest.approx(0.6)
    assert float(substrate.last_growth_energy_spent) == pytest.approx(0.4)
    assert float(substrate.cumulative_growth_energy_spent) == pytest.approx(0.4)
    slot = (substrate.dendrite_source[target] == source).nonzero(as_tuple=False)[0]
    assert float(substrate.edge_utility[target, slot]) == 0


def test_newborn_inherits_parent_genotype_and_one_real_dendrite() -> None:
    config = replace(
        tiny_config(),
        births_per_generation=1,
        birth_signal=0.001,
        birth_local_density_max=1,
        inheritance_noise=0,
        max_deaths_per_generation=0,
    )
    model = CellularGraphClassifier(config, seed=19)
    substrate = model.substrate
    before = substrate.occupied.clone()
    substrate.stimulation_ema[substrate.occupied] = 1

    update = substrate.structural_step(
        apply_lifecycle=True, apply_topology=False
    )
    newborns = (substrate.occupied & ~before).nonzero(as_tuple=False).squeeze(1)

    assert update.births == 1
    assert newborns.numel() == 1
    child = int(newborns[0])
    parent = int(substrate.parent_site[child])
    assert parent >= 0
    assert torch.equal(substrate.genotype[child], substrate.genotype[parent])
    assert substrate.lineage_depth[child] == substrate.lineage_depth[parent] + 1
    assert parent in substrate.dendrite_source[child].tolist()
    assert update.changed_sites[child]


def test_replacement_birth_policy_cannot_inflate_a_healthy_population() -> None:
    config = replace(
        tiny_config(), births_per_generation=4, births_replace_deaths=1,
        birth_signal=0.001, birth_local_density_max=1,
        max_deaths_per_generation=1, juvenile_trials=1,
    )
    substrate = CellularGraphClassifier(config, seed=29).substrate
    substrate.stimulation_ema[substrate.occupied] = 1
    initial_population = int(substrate.occupied.sum())

    healthy = substrate.structural_step(apply_lifecycle=True, apply_topology=False)

    assert healthy.births == healthy.deaths == 0
    assert int(substrate.occupied.sum()) == initial_population

    victim = int(
        (substrate.occupied & ~substrate.anchor_mask).nonzero(as_tuple=False)[0]
    )
    substrate.neuron_age[victim] = config.juvenile_trials
    substrate.energy[victim] = 0
    replacement = substrate.structural_step(
        apply_lifecycle=True, apply_topology=False
    )

    assert replacement.deaths == 1
    assert replacement.births <= replacement.deaths
    assert int(substrate.occupied.sum()) <= initial_population


def test_recovery_only_restores_stun_without_cell_or_graph_turnover() -> None:
    config = replace(
        apply_lifecycle_profile(tiny_config(), "recovery_only"),
        stun_recovery_probability=1,
    )
    substrate = CellularGraphClassifier(config, seed=35).substrate
    victim = int(
        (substrate.occupied & ~substrate.anchor_mask).nonzero(as_tuple=False)[0]
    )
    substrate.stunned[victim] = True
    substrate.stun_generations[victim] = 0
    substrate.neuron_age[victim] = config.juvenile_trials
    substrate.energy[victim] = 0
    before_population = substrate.occupied.clone()
    before_graph = substrate.dendrite_source.clone()

    update = substrate.structural_step(
        apply_lifecycle=True, apply_topology=False
    )

    assert update.recoveries == 1
    assert update.deaths == 0
    assert update.births == 0
    assert substrate.occupied[victim]
    assert not substrate.stunned[victim]
    assert torch.equal(substrate.occupied, before_population)
    assert torch.equal(substrate.dendrite_source, before_graph)


def test_lifecycle_pressure_activates_before_topology_plasticity() -> None:
    dataset = synthetic_digits()
    config = replace(
        tiny_config(),
        lifecycle_warmup_trials=1,
        lifecycle_interval=1,
        structural_warmup_trials=1_000,
        births_per_generation=0,
        max_deaths_per_generation=0,
        target_stimulation_min=100.0,
        target_stimulation_max=200.0,
    )
    experiment = MnistExperiment(
        config,
        seed=23,
        device="cpu",
        train_dataset=dataset,
        test_dataset=dataset,
    )

    experiment._train_trial()

    assert experiment.lifecycle_active is True
    assert experiment.structure_unlocked is False
    assert experiment.model.substrate.generation == 1
    interior = experiment.model.substrate.occupied & ~experiment.model.substrate.anchor_mask
    assert (experiment.model.substrate.energy[interior] < 1).any()


def test_repeated_local_source_evidence_forms_a_real_dendrite() -> None:
    model = CellularGraphClassifier(tiny_config(), seed=15)
    substrate = model.substrate
    safe_probes = substrate.probe_source.clamp_min(0)
    possible = substrate.occupied.unsqueeze(1) & substrate.occupied[safe_probes]
    possible &= substrate.probe_source >= 0
    possible &= (substrate.roles[:, 1] == 0).unsqueeze(1)
    target = int(possible.any(dim=1).nonzero(as_tuple=False)[0])
    source = int(substrate.probe_source[target][possible[target]][0])
    substrate._clear_edges(torch.nn.functional.one_hot(
        torch.tensor(target * substrate.config.edge_slots),
        num_classes=substrate.config.site_count * substrate.config.edge_slots,
    ).bool().reshape_as(substrate.dendrite_source))
    substrate.stimulation_ema.zero_()
    substrate.stimulation_ema[source] = 1
    substrate.structural_step()

    assert source in substrate.dendrite_source[target].tolist()
