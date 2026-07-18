"""Scientific contracts for token-stream graph experiments."""

from __future__ import annotations

import json
import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from petridish.benchmark_sequences import _scale_learning_rates, _write_result
from petridish.benchmark_recovery import (
    _apply_branch_config,
    _capture_global_rng,
    _restore_global_rng,
)
from petridish.graph_layout import LAYOUTS, sequence_layout
from petridish.lifecycle_profiles import apply_lifecycle_profile
from petridish.mnist_hyperparameters import configured, hyperparameter_payload
from petridish.mnist_substrate import SpatialSubstrate
from petridish.protocol import build_snapshot
from petridish.sequence_config import sequence_config
from petridish.sequence_experiment import SequenceExperiment
from petridish.sequence_model import CellularSequenceModel
from petridish.sequence_tasks import (
    SequenceBatch,
    SequenceTask,
    associative_recall_batch,
    resolve_sequence_task,
)
from petridish.token_corpus_task import build_token_task
from petridish.token_context_task import token_context_task
from petridish.token_grammar_task import token_grammar_task
from petridish.token_memory_task import token_memory_task
from petridish.token_pipeline_task import token_pipeline_task
from petridish.token_routing_task import token_routing_task
from petridish.token_settling_task import token_settling_task
from petridish.token_settled_pipeline_task import token_settled_pipeline_task
from petridish.token_stream_task import token_stream_task
from petridish.train_shakespeare import (
    _fresh_config,
    _held_out_diagnostics,
    _migrate_model_state,
    _scientific_metrics,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)


def small_config():
    return sequence_config(
        width=20, height=20, hidden_channels=8, genotype_channels=6,
        initial_density=0.30, batch_size=3, message_steps=1,
        candidate_probes=12, local_radius=4, max_visible_edges=100,
    )


def corpus_config():
    """Keep distributed corpus fixtures sparse while preserving linear ports."""

    return sequence_config(
        width=68, height=68, hidden_channels=8, genotype_channels=6,
        initial_density=0.03, max_initial_neurons=160,
        batch_size=2, message_steps=1,
        candidate_probes=12, local_radius=4, max_visible_edges=100,
    )


def test_benchmark_artifact_replacement_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"

    _write_result(output, {"status": "running", "completedSteps": 20})
    _write_result(output, {"status": "complete", "completedSteps": 40})

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
    assert list(output.parent.glob(".*.tmp")) == []


def test_recovery_branch_restores_process_global_rng() -> None:
    device = torch.device("cpu")
    torch.manual_seed(808)
    state = _capture_global_rng(device)
    first = torch.rand(12)

    _restore_global_rng(state, device)
    second = torch.rand(12)

    assert torch.equal(first, second)


def test_recovery_branch_clone_is_independent_and_consistently_configured() -> None:
    experiment = SequenceExperiment(
        "associative_recall", small_config(), seed=31, device="cpu",
        recall_pair_count=2, recall_pair_max=2,
    )
    clone = copy.deepcopy(experiment)

    _apply_branch_config(
        clone, lifecycle=True, interval=32,
        births_per_generation=4, max_deaths_per_generation=8,
    )
    killed = clone.model.substrate.lesion(10, 10, 2)

    assert killed > 0
    assert clone.config is clone.model.config
    assert clone.config is clone.model.substrate.config
    assert clone.config.lifecycle_enabled == 1
    assert clone.config.lifecycle_interval == 32
    assert clone.config.structural_interval == 32
    assert clone.config.births_per_generation == 4
    assert clone.config.max_deaths_per_generation == 8
    assert clone.structure_unlocked is True
    assert int(experiment.model.substrate.occupied.sum()) > int(
        clone.model.substrate.occupied.sum()
    )


def test_static_recovery_branch_cannot_reunlock_topology() -> None:
    experiment = SequenceExperiment(
        "associative_recall", small_config(), seed=32, device="cpu",
        recall_pair_count=2, recall_pair_max=2,
    )
    experiment.training_step = 1_200

    _apply_branch_config(experiment, lifecycle=False)

    assert experiment.config.lifecycle_enabled == 0
    assert experiment.structure_unlocked is False
    assert experiment._should_unlock_structure(1_201, accuracy=1.0) is False
    assert experiment.config.structural_warmup_trials > experiment.training_step


def test_sequence_layouts_are_directional_port_permutations() -> None:
    recall = LAYOUTS["associative_recall"]
    language = LAYOUTS["tiny_language"]
    assert recall.input_position_order != tuple(range(10))
    assert recall.output_position_order != tuple(range(10))
    assert language.input_side == "right"
    assert language.output_side == "left"
    assert language.flow_direction == -1


def test_tiny_shakespeare_uses_one_ordered_66_port_column_per_boundary() -> None:
    vocabulary_size = 66
    layout = sequence_layout("tiny_shakespeare", vocabulary_size)
    config = sequence_config("tiny_shakespeare")
    substrate = SpatialSubstrate(config, layout=layout, seed=4)

    assert (config.width, config.height) == (68, 68)
    assert substrate.input_sites.unique().numel() == vocabulary_size
    assert substrate.output_sites.unique().numel() == vocabulary_size
    assert not set(substrate.input_sites.tolist()) & set(substrate.output_sites.tolist())

    input_columns = (substrate.input_sites % config.width).unique().tolist()
    output_columns = (substrate.output_sites % config.width).unique().tolist()
    assert input_columns == [config.width - 2]
    assert output_columns == [1]
    assert sorted((substrate.input_sites // config.width).tolist()) == list(range(1, 67))
    assert sorted((substrate.output_sites // config.width).tolist()) == list(range(1, 67))

    input_base = torch.tensor(substrate._boundary_sites(vocabulary_size, "right"))
    output_base = torch.tensor(substrate._boundary_sites(vocabulary_size, "left"))
    assert torch.equal(
        substrate.input_sites, input_base[torch.tensor(layout.input_position_order)]
    )
    assert torch.equal(
        substrate.output_sites, output_base[torch.tensor(layout.output_position_order)]
    )


def test_boundary_ports_reject_wrap_instead_of_creating_another_column() -> None:
    layout = sequence_layout("tiny_shakespeare", 66)
    undersized = replace(
        sequence_config("tiny_shakespeare"), width=66, height=66
    )

    with pytest.raises(ValueError, match="may not wrap into another column"):
        SpatialSubstrate(undersized, layout=layout, seed=4)


def test_68_field_size_is_available_for_single_column_corpus_tasks() -> None:
    tiny = sequence_config("tiny_shakespeare")
    choices = hyperparameter_payload(
        tiny, include_sequence=True, task_key="tiny_shakespeare"
    )[0]["choices"]
    assert choices == [16, 32, 64, 68, 128, 256, 512, 1024]
    assert configured(tiny, {"field_size": 68}, task_key="tiny_shakespeare").width == 68

    stories = sequence_config("tiny_stories")
    story_choices = hyperparameter_payload(
        stories, include_sequence=True, task_key="tiny_stories"
    )[0]["choices"]
    assert story_choices == [16, 32, 64, 68, 128, 256, 512, 1024]
    assert configured(stories, {"field_size": 68}, task_key="tiny_stories").width == 68

    regular = sequence_config("tiny_language")
    assert 68 not in hyperparameter_payload(
        regular, include_sequence=True, task_key="tiny_language"
    )[0]["choices"]
    with pytest.raises(ValueError, match="power of two"):
        configured(regular, {"field_size": 68}, task_key="tiny_language")


def test_headless_token_launch_preserves_task_specific_warmups() -> None:
    config = _fresh_config(
        "tiny_stories",
        field_size=None,
        batch_size=1,
        message_steps=None,
        architecture="gru",
        lifecycle=True,
    )

    assert (config.width, config.height) == (68, 68)
    assert config.batch_size == 1
    assert config.lifecycle_enabled == 1
    assert config.lifecycle_warmup_trials == 500
    assert config.structural_warmup_trials == 1_000


def test_headless_launch_scales_all_optimizer_rates_together() -> None:
    defaults = sequence_config("tiny_stories")
    scaled = _fresh_config(
        "tiny_stories", field_size=68, batch_size=1, message_steps=4,
        architecture="gru", lifecycle=False, broadcast_gain=0.0,
        learning_rate_scale=0.25,
    )

    assert scaled.broadcast_gain == 0.0
    assert scaled.learning_rate == pytest.approx(defaults.learning_rate * 0.25)
    assert scaled.readout_learning_rate == pytest.approx(
        defaults.readout_learning_rate * 0.25
    )
    assert scaled.synapse_learning_rate == pytest.approx(
        defaults.synapse_learning_rate * 0.25
    )

    benchmark_scaled = _scale_learning_rates(defaults, 0.25)
    assert benchmark_scaled.learning_rate == pytest.approx(
        defaults.learning_rate * 0.25
    )
    assert benchmark_scaled.readout_learning_rate == pytest.approx(
        defaults.readout_learning_rate * 0.25
    )
    assert benchmark_scaled.synapse_learning_rate == pytest.approx(
        defaults.synapse_learning_rate * 0.25
    )
    with pytest.raises(ValueError, match="learning-rate scale"):
        _scale_learning_rates(defaults, 0.0)


def test_balanced_lifecycle_retains_biomimicry_without_death_budget_collapse() -> None:
    baseline = sequence_config("tiny_stories")
    balanced = apply_lifecycle_profile(baseline, "balanced")

    assert balanced.lifecycle_enabled == 1
    assert balanced.lifecycle_warmup_trials == baseline.lifecycle_warmup_trials
    assert balanced.structural_warmup_trials == baseline.structural_warmup_trials
    assert balanced.max_deaths_per_generation == balanced.births_per_generation == 16
    assert balanced.target_stimulation_min < baseline.target_stimulation_min
    assert balanced.stun_load_threshold > baseline.stun_load_threshold
    assert balanced.stun_recovery_probability > baseline.stun_recovery_probability
    assert balanced.excitotoxic_death_threshold > baseline.excitotoxic_death_threshold

    launched = _fresh_config(
        "tiny_stories", field_size=68, batch_size=1, message_steps=4,
        architecture="gru", lifecycle=True, lifecycle_profile="balanced",
    )
    assert launched.max_deaths_per_generation == 16
    assert launched.births_per_generation == 16
    assert launched.starvation_cost == balanced.starvation_cost
    assert launched.lifecycle_enabled == 1

    replacement_profile = apply_lifecycle_profile(baseline, "replacement")
    assert replacement_profile.births_replace_deaths == 1
    assert replacement_profile.births_per_generation == 16
    assert replacement_profile.max_deaths_per_generation == 16

    fixed = _fresh_config(
        "tiny_stories", field_size=68, batch_size=1, message_steps=4,
        architecture="gru", lifecycle=False, structure=False,
    )
    assert fixed.structural_enabled == 0


def test_associative_recall_curriculum_preserves_queried_value() -> None:
    generator = torch.Generator().manual_seed(7)
    for pair_count in (1, 2, 3):
        batch = associative_recall_batch(24, generator, pair_count)
        assert batch.tokens.shape == (24, 8)
        assert torch.equal(batch.loss_mask, batch.targets >= 0)
        for tokens, target in zip(batch.tokens, batch.targets, strict=True):
            query = int(tokens[-1])
            bindings = {
                int(tokens[index]): int(tokens[index + 1])
                for index in range(0, pair_count * 2, 2)
            }
            assert int(target[-1]) == bindings[query]


def test_associative_recall_can_hold_a_fixed_difficulty() -> None:
    experiment = SequenceExperiment(
        "associative_recall", small_config(), seed=7, device="cpu",
        recall_pair_count=2, recall_pair_max=2,
    )
    experiment.stage_accuracy_history.extend([1.0] * 24)

    experiment._maybe_advance_recall_curriculum()

    assert experiment.recall_pair_count == 2


def test_associative_recall_evaluation_reports_each_query_slot() -> None:
    experiment = SequenceExperiment(
        "associative_recall", small_config(), seed=9, device="cpu",
        recall_pair_count=2, recall_pair_max=2,
    )

    metrics = experiment.evaluate_metrics(2)

    assert len(metrics["slotAccuracy"]) == 2
    assert all(0.0 <= accuracy <= 1.0 for accuracy in metrics["slotAccuracy"])
    assert metrics["positionIndices"] == [7]
    assert len(metrics["positionAccuracy"]) == 1
    assert metrics["presentedValueRate"] == pytest.approx(
        metrics["accuracy"] + metrics["distractorRate"]
    )
    assert metrics["presentedValueRate"] + metrics["absentValueRate"] == pytest.approx(1.0)


def test_sequence_model_retains_state_and_backpropagates() -> None:
    task = resolve_sequence_task("tiny_language")
    batch = task.batch(3, torch.Generator().manual_seed(2))
    model = CellularSequenceModel(small_config(), layout=task.key, seed=2)
    result = model(batch.tokens)
    assert result.logits.shape == (3, task.sequence_length, 10)
    assert len(result.frames) == task.sequence_length
    loss = torch.nn.functional.cross_entropy(
        result.logits[batch.loss_mask], batch.targets[batch.loss_mask]
    )
    loss.backward()
    assert model.substrate.synapse_weight.grad is not None
    assert float(model.substrate.synapse_weight.grad.abs().sum()) > 0


def test_nonfinite_loss_is_rejected_before_optimizer_mutation() -> None:
    experiment = SequenceExperiment(
        "tiny_language", small_config(), seed=10, device="cpu"
    )
    parameter = experiment.model.token_identity.weight
    before = parameter.detach().clone()

    def nonfinite_loss(logits, _batch):
        return logits.sum() * float("nan"), 0.0

    experiment._masked_loss_accuracy = nonfinite_loss  # type: ignore[method-assign]

    with pytest.raises(FloatingPointError, match="before backward"):
        experiment.train_updates(1)

    assert experiment.training_step == 0
    assert torch.equal(parameter, before)


def test_neuron_owned_binding_memory_is_optional_and_differentiable() -> None:
    task = resolve_sequence_task("associative_recall")
    config = sequence_config(
        width=20, height=20, hidden_channels=8, genotype_channels=6,
        initial_density=0.30, batch_size=2, message_steps=1,
        candidate_probes=12, local_radius=4, max_visible_edges=100,
        binding_memory_gain=1.0, binding_memory_temperature=0.08,
        binding_token_values=1,
    )
    batch = associative_recall_batch(2, torch.Generator().manual_seed(21), 2)
    model = CellularSequenceModel(config, layout=task.key, seed=21)

    result = model(batch.tokens, capture_trace=False)
    loss = torch.nn.functional.cross_entropy(
        result.logits[batch.loss_mask], batch.targets[batch.loss_mask]
    )
    loss.backward()

    assert model.binding_owner_address is not None
    assert model.binding_owner_address.weight.grad is not None
    assert float(model.binding_owner_address.weight.grad.abs().sum()) > 0
    baseline = CellularSequenceModel(small_config(), layout=task.key, seed=21)
    assert baseline.binding_owner_address is None
    assert config.binding_token_values == 1
    diagnostics = model.binding_memory_diagnostics()
    assert diagnostics is not None
    assert 1 <= diagnostics["distinctOwners"] <= len(task.vocabulary)
    assert 0 <= diagnostics["meanAddressEntropy"] <= 1
    assert 0 <= diagnostics["meanAddressOverlap"] <= 1
    assert baseline.binding_memory_diagnostics() is None


def test_binding_address_separation_penalty_reaches_owner_map() -> None:
    task = resolve_sequence_task("associative_recall")
    config = sequence_config(
        width=20, height=20, hidden_channels=8, genotype_channels=6,
        initial_density=0.30, batch_size=2, message_steps=1,
        candidate_probes=12, local_radius=4, max_visible_edges=100,
        binding_memory_gain=1.0, binding_token_values=1,
        binding_address_regularization=0.02,
    )
    model = CellularSequenceModel(config, layout=task.key, seed=22)

    model.regularization().backward()

    assert model.binding_owner_address is not None
    assert model.binding_owner_address.weight.grad is not None
    assert float(model.binding_owner_address.weight.grad.abs().sum()) > 0


@pytest.mark.parametrize("architecture", ("gru", "lstm", "esn", "transformer"))
def test_sequence_cell_architectures_share_graph_and_gradient_contract(
    architecture: str,
) -> None:
    task = resolve_sequence_task("tiny_language")
    config = sequence_config(
        width=20, height=20, hidden_channels=8, genotype_channels=6,
        initial_density=0.30, batch_size=2, message_steps=1,
        candidate_probes=12, local_radius=4, max_visible_edges=100,
        cell_architecture=architecture,
    )
    batch = task.batch(2, torch.Generator().manual_seed(15))
    model = CellularSequenceModel(config, layout=task.key, seed=15)

    result = model(batch.tokens, capture_trace=False)
    loss = torch.nn.functional.cross_entropy(
        result.logits[batch.loss_mask], batch.targets[batch.loss_mask]
    )
    loss.backward()

    assert result.logits.shape == (2, task.sequence_length, 10)
    assert model.cell_rule.architecture == architecture
    assert model.substrate.synapse_weight.grad is not None
    assert torch.isfinite(result.logits).all()


def test_legacy_gru_checkpoint_keys_migrate_to_architecture_wrapper() -> None:
    legacy = {
        "cell_rule.weight_ih": torch.ones(2, 2),
        "cell_rule.weight_hh": torch.ones(2, 2),
        "cell_rule.bias_ih": torch.ones(2),
        "cell_rule.bias_hh": torch.ones(2),
        "class_bias": torch.zeros(2),
    }

    migrated = _migrate_model_state(legacy)

    assert "cell_rule.weight_ih" not in migrated
    assert "cell_rule.rule.weight_ih" in migrated
    assert migrated["class_bias"] is legacy["class_bias"]


def test_sequence_model_streams_measured_token_frames() -> None:
    task = resolve_sequence_task("tiny_language")
    batch = task.batch(2, torch.Generator().manual_seed(12))
    model = CellularSequenceModel(small_config(), layout=task.key, seed=12)
    observed: list[tuple[int, str, tuple[int, ...]]] = []

    result = model(
        batch.tokens,
        frame_callback=lambda frame, logits: observed.append(
            (frame.token_position, frame.stage, tuple(logits.shape))
        ),
    )

    assert len(observed) == task.sequence_length
    assert [position for position, _, _ in observed] == list(range(task.sequence_length))
    assert all(stage == "token" for _, stage, _ in observed)
    assert all(shape == (2, 10) for _, _, shape in observed)
    assert len(result.frames) == task.sequence_length


def test_sequence_snapshot_reports_tokens_curriculum_and_real_graph() -> None:
    experiment = SequenceExperiment(
        "associative_recall", small_config(), seed=3, device="cpu"
    )
    snapshot = build_snapshot(experiment)
    assert snapshot["experiment"] == "associative_recall"
    assert snapshot["task"]["kind"] == "sequence"
    assert snapshot["task"]["recallPairs"] == 1
    assert len(snapshot["task"]["tokens"]) == 8
    assert len(snapshot["field"]["indices"]) == len(snapshot["field"]["cells"])
    assert snapshot["metrics"]["edgeCount"] >= len(snapshot["edges"]["source"])
    field_control = next(
        parameter for parameter in snapshot["configuration"]["parameters"]
        if parameter["key"] == "field_size"
    )
    assert field_control["choices"] == [16, 32, 64, 128, 256, 512, 1024]


def test_trace_free_training_updates_metrics_without_replacing_visible_frames() -> None:
    experiment = SequenceExperiment(
        "tiny_language", small_config(), seed=8, device="cpu"
    )
    original_trace = experiment.last_trace
    original_tokens = experiment.last_tokens.clone()
    original_predictions = experiment.last_predictions.clone()

    experiment.train_updates(2)

    assert experiment.training_step == 2
    assert experiment.tick == 2
    assert experiment.last_trace is original_trace
    assert torch.equal(experiment.last_tokens, original_tokens)
    assert torch.equal(experiment.last_predictions, original_predictions)
    assert experiment.seen_examples == experiment.config.batch_size * 2
    experiment.refresh_visual_trace()
    assert experiment.last_trace is not original_trace
    assert len(experiment.last_trace) == experiment.task.sequence_length + 2


def test_visual_training_reports_real_compute_phases_and_finishes_on_structure() -> None:
    experiment = SequenceExperiment(
        "tiny_language", small_config(), seed=9, device="cpu"
    )
    progress: list[tuple[str, int, int]] = []

    experiment.train_visual_update(
        lambda phase, current, total: progress.append((phase, current, total))
    )

    phases = [phase for phase, _, _ in progress]
    assert phases.count("forward") == experiment.task.sequence_length
    backward = [item for item in progress if item[0] == "backward"]
    assert len(backward) == experiment.task.sequence_length + 1
    assert backward[-1] == (
        "backward", experiment.task.sequence_length, experiment.task.sequence_length
    )
    assert "optimizer" in phases
    assert ("credit", 1, 1) in progress
    assert ("lifecycle", 1, 1) in progress
    assert experiment.training_step == 1
    assert experiment.last_frame.stage == "structural"


def test_corpus_prompt_can_generate_one_more_character() -> None:
    vocabulary = ("\n", "a", "b", "c", "�")
    encoded = torch.tensor([1, 2, 3, 0, 1, 2, 3, 0], dtype=torch.long)

    def batch(size: int, generator: torch.Generator) -> SequenceBatch:
        del generator
        rows = encoded[:5].repeat(size, 1)
        return SequenceBatch(rows[:, :-1], rows[:, 1:], torch.ones(size, 4, dtype=torch.bool))

    task = SequenceTask(
        key="tiny_shakespeare", title="Corpus fixture", description="fixture",
        vocabulary=vocabulary, sequence_length=4, generator=batch,
        encode=lambda text: [vocabulary.index(character) if character in vocabulary else 4 for character in text],
        decode=lambda tokens: "".join(vocabulary[token] for token in tokens),
        dataset_name="fixture", dataset_characters=8,
    )
    experiment = SequenceExperiment(task, small_config(), seed=4, device="cpu")
    experiment.set_prompt("ab")
    generated = experiment.generate_token()
    snapshot = build_snapshot(experiment)

    assert generated in vocabulary
    assert len(experiment.generated_text) == 1
    assert snapshot["task"]["interactive"] is True
    assert snapshot["task"]["interactivePrompt"] == "ab"
    assert snapshot["task"]["generatedText"] == generated


def test_token_corpus_uses_distributed_ports_and_incremental_state() -> None:
    text = "".join(
        f"Once upon a time, a small fox found number {index}.\n"
        for index in range(80)
    )
    task = build_token_task(text, context_length=8, vocabulary_size=48)
    config = corpus_config()
    layout = sequence_layout(task.key, len(task.vocabulary))
    model = CellularSequenceModel(
        config, layout=layout, vocab_size=len(task.vocabulary), max_length=8, seed=41
    )
    batch = task.batch(2, torch.Generator().manual_seed(41))

    full = model(batch.tokens, capture_trace=False)
    prefix = model(batch.tokens[:, :3], capture_trace=False)
    suffix = model(
        batch.tokens[:, 3:], capture_trace=False, runtime_state=prefix.runtime_state
    )

    assert model.distributed_io is True
    assert layout.input_count == 64
    assert layout.output_count == 64
    assert task.unigram_baseline_accuracy is not None
    assert task.bigram_baseline_accuracy is not None
    assert 0 <= task.unigram_baseline_accuracy <= 1
    assert task.bigram_baseline_accuracy > task.unigram_baseline_accuracy
    assert full.logits.shape == (2, 8, len(task.vocabulary))
    assert torch.allclose(full.logits[:, 3:], suffix.logits, atol=1e-5, rtol=1e-5)


def test_corpus_stream_windows_are_contiguous_across_training_updates() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    positions = torch.tensor([3, 11])

    first, next_positions = task.stream_batch(positions)
    second, _ = task.stream_batch(next_positions)

    assert torch.equal(first.targets[:, -1], second.tokens[:, 0])
    assert torch.equal(next_positions, positions + task.sequence_length)


def test_continuous_training_carries_detached_neuron_state_between_updates() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    config = replace(
        corpus_config(), batch_size=1, structural_enabled=0, lifecycle_enabled=0
    )
    experiment = SequenceExperiment(
        task, config, seed=44, device="cpu", stream_mode="continuous"
    )
    initial_positions = experiment._training_stream_positions.clone()

    experiment.train_updates(1)
    first_state = experiment._training_runtime_state
    experiment.train_updates(1)

    assert first_state is not None
    assert experiment._training_runtime_state is not None
    assert first_state.hidden.grad_fn is None
    assert experiment._training_runtime_state.position == task.sequence_length * 2
    assert torch.equal(
        experiment._training_stream_positions,
        initial_positions + task.sequence_length * 2,
    )


def test_continuous_training_state_survives_checkpoint_resume(tmp_path: Path) -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    config = replace(
        corpus_config(), batch_size=1, structural_enabled=0, lifecycle_enabled=0
    )
    original = SequenceExperiment(
        task, config, seed=45, device="cpu", stream_mode="continuous",
        state_retention=0.9,
    )
    original.train_updates(1)
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(checkpoint, original, context_length=8, amp_mode="off")

    restored = SequenceExperiment(
        task, config, seed=45, device="cpu", stream_mode="continuous",
        state_retention=0.9,
    )
    restore_checkpoint(restored, load_checkpoint(checkpoint, torch.device("cpu")))

    assert restored.stream_mode == "continuous"
    assert restored.state_retention == pytest.approx(0.9)
    assert torch.equal(
        restored._training_stream_positions, original._training_stream_positions
    )
    assert torch.equal(
        restored._training_runtime_state.hidden,
        original._training_runtime_state.hidden,
    )


def test_cell_death_preserves_surviving_continuous_state() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    model = CellularSequenceModel(
        corpus_config(), layout=sequence_layout(task.key, len(task.vocabulary)),
        vocab_size=len(task.vocabulary), max_length=8, seed=46,
    )
    batch = task.stream_batch(torch.tensor([5]))[0]
    state = model(batch.tokens, capture_trace=False).runtime_state.detached()
    protected = set(model.substrate.input_sites.tolist() + model.substrate.output_sites.tolist())
    victim = next(site for site in state.sites.tolist() if site not in protected)
    survivor = next(site for site in state.sites.tolist() if site not in protected and site != victim)
    old_index = int((state.sites == survivor).nonzero()[0])
    model.substrate.occupied[victim] = False

    reconciled = model.reconcile_runtime_state(state)
    new_index = int((reconciled.sites == survivor).nonzero()[0])

    assert victim not in reconciled.sites.tolist()
    assert torch.equal(reconciled.hidden[:, new_index], state.hidden[:, old_index])


def test_electrical_relaxation_preserves_structure_and_age_without_hard_reset() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    model = CellularSequenceModel(
        corpus_config(), layout=sequence_layout(task.key, len(task.vocabulary)),
        vocab_size=len(task.vocabulary), max_length=8, seed=48,
    )
    batch = task.stream_batch(torch.tensor([5]))[0]
    state = model(batch.tokens, capture_trace=False).runtime_state.detached()

    retained = model.relax_runtime_state(state, 1.0)
    relaxed = model.relax_runtime_state(state, 0.9)

    assert torch.equal(retained.hidden, state.hidden)
    assert torch.equal(relaxed.sites, state.sites)
    assert relaxed.position == state.position
    assert not torch.equal(relaxed.hidden, state.hidden)
    assert bool(torch.isfinite(relaxed.hidden).all())
    with pytest.raises(ValueError, match="retention"):
        model.relax_runtime_state(state, 1.1)


def test_state_ablation_reuses_identical_validation_stream_and_one_rng_advance() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    config = replace(corpus_config(), batch_size=1)
    experiment = SequenceExperiment(
        task, config, seed=47, device="cpu", stream_mode="continuous"
    )
    before = experiment.eval_generator.get_state().clone()

    carried, cold = experiment.evaluate_state_ablation(2)
    paired_after = experiment.eval_generator.get_state().clone()
    experiment.eval_generator.set_state(before)
    single = experiment.evaluate_metrics(2, carry_state=True)

    assert carried["stateCarry"] is True
    assert cold["stateCarry"] is False
    assert carried["positionIndices"] == cold["positionIndices"]
    assert torch.equal(paired_after, experiment.eval_generator.get_state())
    assert carried == single

    diagnostic = _held_out_diagnostics(experiment, 2)
    assert diagnostic["coldStateAccuracy"] >= 0
    assert diagnostic["stateCarryAccuracyDelta"] == pytest.approx(
        diagnostic["accuracy"] - diagnostic["coldStateAccuracy"]
    )


def test_token_corpus_uses_one_64_port_column_per_boundary() -> None:
    config = sequence_config("tiny_stories")
    layout = sequence_layout("tiny_stories", vocabulary_size=2_048)
    substrate = SpatialSubstrate(config, layout=layout, seed=41)

    assert (config.width, config.height) == (68, 68)
    assert (substrate.input_sites % config.width).unique().tolist() == [1]
    assert (substrate.output_sites % config.width).unique().tolist() == [66]
    assert sorted((substrate.input_sites // config.width).tolist()) == list(range(1, 65))
    assert sorted((substrate.output_sites // config.width).tolist()) == list(range(1, 65))


def test_token_routing_control_is_balanced_and_has_no_position_shortcut() -> None:
    task = token_routing_task(mapping_size=8)
    batch = task.batch(8, torch.Generator().manual_seed(3))

    assert task.key == "tiny_stories"
    assert batch.tokens.shape == batch.targets.shape == (8, 1)
    assert batch.tokens[:, 0].tolist() == list(range(8))
    assert batch.targets[:, 0].tolist() == list(range(8, 16))
    assert bool(batch.loss_mask.all())


def test_token_context_control_requires_both_distributed_tokens() -> None:
    task = token_context_task()
    batch = task.batch(8, torch.Generator().manual_seed(5))

    assert task.key == "tiny_stories"
    assert batch.tokens[:4].tolist() == [[0, 2], [0, 3], [1, 2], [1, 3]]
    assert batch.targets[:4, 1].tolist() == [4, 5, 5, 4]
    assert not bool(batch.loss_mask[:, 0].any())
    assert bool(batch.loss_mask[:, 1].all())
    for column in (0, 1):
        selected = batch.targets[batch.tokens[:, 0] == column, 1]
        assert sorted(selected.unique().tolist()) == [4, 5]
    for query in (2, 3):
        selected = batch.targets[batch.tokens[:, 1] == query, 1]
        assert sorted(selected.unique().tolist()) == [4, 5]


def test_token_memory_control_requires_delayed_context() -> None:
    task = token_memory_task()
    batch = task.batch(8, torch.Generator().manual_seed(7))

    assert task.key == "tiny_stories"
    assert batch.tokens[:, 1].unique().tolist() == [2]
    assert batch.targets[:, 1].tolist() == [3, 4, 3, 4, 3, 4, 3, 4]
    assert not bool(batch.loss_mask[:, 0].any())
    assert bool(batch.loss_mask[:, 1].all())


def test_token_stream_requires_persistent_context_at_every_prediction() -> None:
    task = token_stream_task()
    batch = task.batch(8, torch.Generator().manual_seed(11))

    assert task.key == "tiny_stories"
    assert batch.tokens.shape == batch.targets.shape == (8, 5)
    assert batch.tokens[:, 0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert not bool(batch.loss_mask[:, 0].any())
    assert bool(batch.loss_mask[:, 1:].all())
    for position in range(1, task.sequence_length):
        assert batch.targets[:, position].tolist().count(4) == 4
        assert batch.targets[:, position].tolist().count(5) == 4
        for rule in (0, 1):
            selected = batch.targets[batch.tokens[:, 0] == rule, position]
            assert sorted(selected.unique().tolist()) == [4, 5]
        for bit_token in (2, 3):
            selected = batch.targets[batch.tokens[:, position] == bit_token, position]
            assert sorted(selected.unique().tolist()) == [4, 5]


def test_token_pipeline_aligns_targets_two_token_clocks_after_input() -> None:
    task = token_pipeline_task()
    batch = task.batch(8, torch.Generator().manual_seed(13))

    assert task.key == "tiny_stories"
    assert batch.tokens.shape == batch.targets.shape == (8, 7)
    assert not bool(batch.loss_mask[:, :3].any())
    assert bool(batch.loss_mask[:, 3:].all())
    assert bool((batch.tokens[:, 5:] == 4).all())
    for output_offset, position in enumerate(range(3, 7)):
        delayed_input = batch.tokens[:, output_offset + 1] - 2
        expected = 5 + (delayed_input ^ batch.tokens[:, 0])
        assert torch.equal(batch.targets[:, position], expected)
        assert batch.targets[:, position].tolist().count(5) == 4
        assert batch.targets[:, position].tolist().count(6) == 4
    for position in (3, 4):
        for current_bit in (2, 3):
            selected = batch.targets[batch.tokens[:, position] == current_bit, position]
            assert sorted(selected.unique().tolist()) == [5, 6]


def test_token_settling_gives_context_two_clocks_before_aligned_targets() -> None:
    task = token_settling_task()
    batch = task.batch(8, torch.Generator().manual_seed(17))

    assert task.key == "tiny_stories"
    assert batch.tokens.shape == batch.targets.shape == (8, 7)
    assert bool((batch.tokens[:, 1:3] == 2).all())
    assert not bool(batch.loss_mask[:, :3].any())
    assert bool(batch.loss_mask[:, 3:].all())
    for offset, position in enumerate(range(3, 7)):
        current_input = batch.tokens[:, position] - 3
        expected = 5 + (current_input ^ batch.tokens[:, 0])
        assert torch.equal(batch.targets[:, position], expected)
        assert batch.targets[:, position].tolist().count(5) == 4
        assert batch.targets[:, position].tolist().count(6) == 4


def test_settled_pipeline_combines_context_setup_with_two_clock_output() -> None:
    task = token_settled_pipeline_task()
    batch = task.batch(8, torch.Generator().manual_seed(19))

    assert task.key == "tiny_stories"
    assert batch.tokens.shape == batch.targets.shape == (8, 9)
    assert bool((batch.tokens[:, 1:3] == 2).all())
    assert bool((batch.tokens[:, 7:9] == 2).all())
    assert not bool(batch.loss_mask[:, :5].any())
    assert bool(batch.loss_mask[:, 5:].all())
    for offset, position in enumerate(range(5, 9)):
        delayed_input = batch.tokens[:, offset + 3] - 3
        expected = 5 + (delayed_input ^ batch.tokens[:, 0])
        assert torch.equal(batch.targets[:, position], expected)
        assert batch.targets[:, position].tolist().count(5) == 4
        assert batch.targets[:, position].tolist().count(6) == 4
    for position in (5, 6):
        for current_bit in (3, 4):
            selected = batch.targets[batch.tokens[:, position] == current_bit, position]
            assert sorted(selected.unique().tolist()) == [5, 6]


def test_autoregressive_grammar_requires_rule_and_two_symbol_context() -> None:
    task = token_grammar_task()
    batch = task.batch(32, torch.Generator().manual_seed(23))

    assert task.key == "tiny_stories"
    assert batch.tokens.shape == batch.targets.shape == (32, 9)
    assert not bool(batch.loss_mask[:, :2].any())
    assert bool(batch.loss_mask[:, 2:].all())
    assert torch.equal(batch.targets[:, 2:8], batch.tokens[:, 3:9])
    assert batch.tokens[:, 0].tolist().count(0) == 16
    assert batch.tokens[:, 0].tolist().count(1) == 16
    for position in range(2, 9):
        assert sorted(batch.targets[:, position].unique().tolist()) == [2, 3, 4, 5]
        for rule in (0, 1):
            selected = batch.targets[batch.tokens[:, 0] == rule, position]
            assert sorted(selected.unique().tolist()) == [2, 3, 4, 5]
        for token in (2, 3, 4, 5):
            current = batch.targets[batch.tokens[:, position] == token, position]
            previous = batch.targets[batch.tokens[:, position - 1] == token, position]
            assert sorted(current.unique().tolist()) == [2, 3, 4, 5]
            assert sorted(previous.unique().tolist()) == [2, 3, 4, 5]
    rule = batch.tokens[:, 0]
    previous = batch.tokens[:, 7] - 2
    current = batch.tokens[:, 8] - 2
    assert torch.equal(batch.targets[:, 8], 2 + (previous + current + rule) % 4)


def test_zero_broadcast_gain_is_a_hard_workspace_ablation() -> None:
    config = small_config()
    config = replace(
        config, broadcast_gain=0.0, lifecycle_enabled=0,
        structural_warmup_trials=10_000,
    )
    experiment = SequenceExperiment("tiny_language", config, seed=19, device="cpu")

    experiment.train_updates(1)

    assert experiment.model.broadcast_gain.grad is None
    assert experiment.model.broadcast_key.weight.grad is None
    assert experiment.model.broadcast_query.weight.grad is None
    assert experiment.model.broadcast_value.weight.grad is None


def test_greedy_generation_diagnostic_preserves_training_state() -> None:
    text = "Once upon a time there was a little fox. " * 80
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    experiment = SequenceExperiment(task, corpus_config(), seed=42, device="cpu")
    experiment.model.train()
    generator_state = experiment.generator.get_state().clone()
    interactive_state = (
        experiment.interactive_prompt,
        experiment.generated_text,
        experiment.next_token_prediction,
    )

    first, first_ids = experiment.greedy_completion("Once upon", max_tokens=4)
    second, second_ids = experiment.greedy_completion("Once upon", max_tokens=4)

    assert first == second
    assert first_ids == second_ids
    assert len(first_ids) == 4
    assert experiment.model.training is True
    assert torch.equal(experiment.generator.get_state(), generator_state)
    assert (
        experiment.interactive_prompt,
        experiment.generated_text,
        experiment.next_token_prediction,
    ) == interactive_state


def test_headless_scientific_metrics_separate_routing_and_lifecycle() -> None:
    text = "Once upon a time there was a little fox. " * 80
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    experiment = SequenceExperiment(task, corpus_config(), seed=43, device="cpu")
    metrics = _scientific_metrics(experiment)

    assert metrics["edgeCount"] >= metrics["conductingEdgeCount"]
    assert metrics["contextReachableOutputs"] >= metrics["tokenReachableOutputs"]
    assert metrics["reachableOutputs"] >= metrics["contextReachableOutputs"]
    assert metrics["outputCount"] == 64
    assert metrics["cumulativeGrownEdges"] == 0
    assert metrics["cumulativePrunedEdges"] == 0
    assert metrics["electricalStateTokens"] == 0


def test_structural_update_reports_unbounded_exact_prune_count() -> None:
    substrate = SpatialSubstrate(small_config(), layout="tiny_language", seed=44)
    _, _, sources = substrate.edge_list()
    source = int(sources[0])
    substrate.occupied[source] = False
    expected = int(((substrate.dendrite_source >= 0) & ~substrate.active_edge_mask).sum())

    update = substrate.structural_step(apply_lifecycle=False, apply_topology=False)

    assert expected > 0
    assert update.pruned_edges == expected
    assert update.grown_edges == 0


def test_excitotoxicity_stuns_recovers_and_only_repetition_becomes_lethal() -> None:
    config = sequence_config(
        width=20, height=20, hidden_channels=8, genotype_channels=6,
        initial_density=0.30, batch_size=2, message_steps=1,
        candidate_probes=12, local_radius=4, max_visible_edges=100,
        stun_load_threshold=0.01, stun_recovery_probability=1.0,
        stun_min_generations=1, excitotoxic_damage_per_stun=0.05,
        excitotoxic_death_threshold=0.5, max_deaths_per_generation=1,
    )
    substrate = SpatialSubstrate(config, layout="tiny_language", seed=43)
    sites = substrate.living_sites
    count = sites.numel()
    zeros = torch.zeros(count)
    zero_edges = torch.zeros_like(substrate.synapse_weight)
    zero_vectors = torch.zeros(count, config.hidden_channels)

    events = substrate.record_trial(
        sites, zeros, torch.ones(count), zeros, zero_edges, zero_edges,
        zero_vectors, zero_vectors, zeros, 0.0, homeostasis_active=True,
    )
    stunned = substrate.stunned.clone()
    update = substrate.structural_step(apply_lifecycle=True, apply_topology=False)

    assert any(event["type"] == "stunned" for event in events)
    assert bool(stunned.any())
    assert update.deaths == 0
    assert update.recoveries == int(stunned.sum())
    assert not bool(substrate.stunned.any())

    victim = int((substrate.occupied & ~substrate.anchor_mask).nonzero()[0])
    substrate.neuron_age[victim] = config.juvenile_trials
    substrate.energy[victim] = 1.0
    substrate.excitotoxic_damage[victim] = config.excitotoxic_death_threshold
    lethal = substrate.structural_step(apply_lifecycle=True, apply_topology=False)

    assert lethal.deaths == 1
    assert lethal.death_causes["excitotoxicity"] == 1
    assert not bool(substrate.occupied[victim])
