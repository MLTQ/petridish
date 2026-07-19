"""Scientific contracts for token-stream graph experiments."""

from __future__ import annotations

import json
import copy
import math
import sys
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
from petridish.sequence_model import CellularSequenceModel, SequenceRuntimeState
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
    expand_persistent_stream_domains,
    expand_persistent_state_lanes,
    _fresh_config,
    _held_out_diagnostics,
    _migrate_model_state,
    _phase_metric_history,
    _phase_novel_exposure,
    _scientific_metrics,
    _record_process_failure,
    load_checkpoint,
    main as train_shakespeare_main,
    plasticity_phase_config,
    reconcile_plasticity_phase_status,
    restore_checkpoint,
    save_checkpoint,
    structural_checkpoint_due,
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


def test_plasticity_continuation_refuses_to_construct_a_fresh_organism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "petridish.train_shakespeare",
            "--device", "cpu",
            "--checkpoint-dir", str(tmp_path),
            "--resume-plasticity",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        train_shakespeare_main()

    assert (
        "requires resume to be enabled and an existing latest.pt"
        in capsys.readouterr().err
    )


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
    assert task.unigram_baseline_loss is not None
    assert task.bigram_baseline_loss is not None
    assert 0 <= task.unigram_baseline_accuracy <= 1
    assert task.bigram_baseline_accuracy > task.unigram_baseline_accuracy
    assert task.bigram_baseline_loss < task.unigram_baseline_loss
    assert full.logits.shape == (2, 8, len(task.vocabulary))
    assert torch.allclose(full.logits[:, 3:], suffix.logits, atol=1e-5, rtol=1e-5)


def test_byte_token_corpus_has_complete_round_trip_without_unknown_class() -> None:
    text = "Once upon a time, café fox ran.\n" * 20
    task = build_token_task(
        text, context_length=8, vocabulary_size=256, tokenizer_profile="byte"
    )

    encoded = task.encode("A café.\n") if task.encode is not None else []

    assert len(task.vocabulary) == 256
    assert task.tokenizer_profile == "byte"
    assert task.special_token_ids == ()
    assert task.unknown_token_id is None
    assert task.validation_unknown_token_rate == 0.0
    assert task.decode is not None
    assert task.decode(encoded) == "A café.\n"
    assert task.training_stream is not None
    assert int(task.training_stream.min()) >= 0
    assert int(task.training_stream.max()) < 256
    with pytest.raises(ValueError, match="256-token"):
        build_token_task(
            text, context_length=8, vocabulary_size=128,
            tokenizer_profile="byte",
        )


def test_corpus_stream_windows_are_contiguous_across_training_updates() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    positions = torch.tensor([3, 11])

    first, next_positions = task.stream_batch(positions)
    second, _ = task.stream_batch(next_positions)

    assert torch.equal(first.targets[:, -1], second.tokens[:, 0])
    assert torch.equal(next_positions, positions + task.sequence_length)


def test_corpus_stream_windows_wrap_inside_independent_lane_domains() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 100
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    assert task.training_stream is not None
    positions = torch.tensor([124, 252])
    domains = torch.tensor([128, 256])

    batch, next_positions = task.stream_batch(
        positions, stream_lengths=domains
    )

    offsets = torch.arange(8)
    assert torch.equal(
        batch.tokens[0], task.training_stream[(positions[0] + offsets) % 128]
    )
    assert torch.equal(
        batch.tokens[1], task.training_stream[(positions[1] + offsets) % 256]
    )
    assert torch.equal(next_positions, torch.tensor([4, 4]))
    with pytest.raises(ValueError, match="match"):
        task.stream_batch(positions, stream_lengths=torch.tensor([128]))
    with pytest.raises(ValueError, match="exceeds"):
        task.stream_batch(positions, stream_lengths=torch.tensor([128, 99_999]))


def test_repeated_training_shard_preserves_cursor_and_full_validation() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 100
    full = build_token_task(text, context_length=8, vocabulary_size=32)
    shard = build_token_task(
        text, context_length=8, vocabulary_size=32, training_shard_tokens=128
    )
    preserved_cursor = torch.tensor([503])

    batch, next_positions = shard.stream_batch(preserved_cursor)

    assert shard.training_stream_tokens == 128
    assert shard.full_training_stream_tokens == full.training_stream_tokens
    assert shard.training_shard_tokens == 128
    assert torch.equal(shard.evaluation_stream, full.evaluation_stream)
    assert batch.tokens.shape == (1, 8)
    assert next_positions.item() == (503 + 8) % 128
    with pytest.raises(ValueError, match="complete context"):
        build_token_task(
            text, context_length=8, vocabulary_size=32, training_shard_tokens=9
        )


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
    assert set(experiment.last_gradient_norms) == {
        "classBiasGradientNorm", "outputReadoutGradientNorm",
        "tokenEncoderGradientNorm", "cellRuleGradientNorm", "synapseGradientNorm",
        "totalGradientNorm", "gradientClipScale",
    }
    assert all(
        math.isfinite(value) and value >= 0
        for value in experiment.last_gradient_norms.values()
    )
    assert 0 < experiment.last_gradient_norms["gradientClipScale"] <= 1
    assert experiment._training_runtime_state.position == task.sequence_length * 2
    assert torch.equal(
        experiment._training_stream_positions,
        initial_positions + task.sequence_length * 2,
    )


def test_disposable_auxiliary_gradient_is_rejected_before_organism_mutation() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    config = replace(
        corpus_config(), batch_size=1, structural_enabled=0, lifecycle_enabled=0
    )
    experiment = SequenceExperiment(
        task, config, seed=144, device="cpu", stream_mode="continuous",
        random_offset_auxiliary_weight=0.25,
    )
    initial_positions = experiment._training_stream_positions.clone()
    initial_model = {
        name: value.detach().clone()
        for name, value in experiment.model.state_dict().items()
    }
    initial_optimizer = experiment.optimizer.state_dict()

    with pytest.raises(RuntimeError, match="disposable cold-context gradients"):
        experiment.train_updates(1)

    assert experiment.training_step == 0
    assert experiment.tick == 0
    assert experiment._training_runtime_state is None
    assert experiment._training_runtime_bank == [None]
    assert torch.equal(experiment._training_stream_positions, initial_positions)
    assert experiment.optimizer.state_dict() == initial_optimizer
    for name, value in experiment.model.state_dict().items():
        assert torch.equal(value, initial_model[name])


def test_full_corpus_auxiliary_samples_beyond_the_persistent_lane_shard() -> None:
    text = "a" * 256 + "z" * 1_024
    task = build_token_task(
        text, context_length=8, vocabulary_size=256, tokenizer_profile="byte",
        training_shard_tokens=128,
    )
    config = replace(
        corpus_config(), batch_size=1, structural_enabled=0,
        lifecycle_enabled=0,
    )
    shard = SequenceExperiment(
        task, config, seed=145, device="cpu", stream_mode="continuous",
        random_offset_auxiliary_weight=0.25,
        random_offset_auxiliary_scope="active_shard",
    )
    full = SequenceExperiment(
        task, config, seed=145, device="cpu", stream_mode="continuous",
        random_offset_auxiliary_weight=0.25,
        random_offset_auxiliary_scope="full_corpus",
    )

    shard_tokens = torch.cat(
        [shard._random_offset_auxiliary_batch().tokens.flatten() for _ in range(32)]
    )
    full_tokens = torch.cat(
        [full._random_offset_auxiliary_batch().tokens.flatten() for _ in range(32)]
    )

    assert set(shard_tokens.tolist()) == {ord("a")}
    assert ord("z") in set(full_tokens.tolist())
    assert task.training_stream_tokens == 128
    assert task.full_training_stream_tokens > task.training_stream_tokens


def test_auxiliary_weight_and_scope_round_trip_in_checkpoint(
    tmp_path: Path,
) -> None:
    task = build_token_task(
        ("One fox ran. Two birds flew. " * 40), context_length=8,
        vocabulary_size=32, training_shard_tokens=128,
    )
    config = replace(corpus_config(), batch_size=1)
    original = SequenceExperiment(
        task, config, seed=146, device="cpu", stream_mode="continuous",
        random_offset_auxiliary_weight=0.25,
        random_offset_auxiliary_scope="full_corpus",
    )
    checkpoint = tmp_path / "auxiliary.pt"
    save_checkpoint(
        checkpoint, original, context_length=8, amp_mode="off",
        organism_id="organism-auxiliary", phase_index=18,
        phase_name="full corpus auxiliary",
    )
    payload = load_checkpoint(checkpoint, torch.device("cpu"))
    restored = SequenceExperiment(
        task, config, seed=146, device="cpu", stream_mode="continuous",
    )
    restore_checkpoint(restored, payload)

    assert payload["task"]["random_offset_auxiliary_weight"] == pytest.approx(0.25)
    assert payload["task"]["random_offset_auxiliary_scope"] == "full_corpus"
    assert restored.random_offset_auxiliary_weight == pytest.approx(0.25)
    assert restored.random_offset_auxiliary_scope == "full_corpus"


def test_round_robin_state_lanes_add_trajectory_diversity_at_batch_one() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    config = replace(
        corpus_config(), batch_size=1, structural_enabled=0, lifecycle_enabled=0
    )
    experiment = SequenceExperiment(
        task, config, seed=49, device="cpu", stream_mode="continuous",
        state_retention=0.9, state_lanes=2,
    )

    experiment.train_updates(3)
    metrics = _scientific_metrics(experiment)

    assert experiment._training_stream_positions.shape == (2, 1)
    assert [state.position for state in experiment._training_runtime_bank] == [16, 8]
    assert experiment._training_runtime_state is experiment._training_runtime_bank[0]
    assert metrics["stateLanes"] == 2
    assert metrics["minimumElectricalStateTokens"] == 8
    assert metrics["maximumElectricalStateTokens"] == 16
    assert metrics["structureUnlocked"] is False
    assert metrics["structureUnlockReason"] == "disabled by configuration"
    assert metrics["structuralWarmupRemaining"] >= 0
    assert metrics["lifecycleReason"] == "disabled by configuration"


def test_disabling_lifecycle_clears_prior_phase_active_status() -> None:
    experiment = SequenceExperiment(
        resolve_sequence_task("associative_recall"),
        replace(corpus_config(), lifecycle_enabled=0),
        seed=91, device="cpu",
    )
    experiment.lifecycle_active = True

    active = experiment._should_activate_lifecycle(experiment.training_step)

    assert active is False
    assert experiment.lifecycle_active is False
    assert experiment.lifecycle_reason == "disabled by configuration"


@pytest.mark.parametrize("starting_lanes", (1, 2))
def test_state_lane_expansion_preserves_every_existing_trajectory(
    tmp_path: Path, starting_lanes: int
) -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    experiment = SequenceExperiment(
        task, replace(corpus_config(), batch_size=1), seed=58, device="cpu",
        stream_mode="continuous", state_retention=0.9, state_lanes=starting_lanes,
    )
    experiment.train_updates(starting_lanes)
    original_positions = experiment._training_stream_positions.clone()
    original_lengths = experiment._training_stream_lengths.clone()
    original_states = list(experiment._training_runtime_bank)
    original_rng = experiment.generator.get_state().clone()
    original_model = {
        name: value.detach().clone()
        for name, value in experiment.model.state_dict().items()
    }
    original_optimizer = copy.deepcopy(experiment.optimizer.state_dict())

    expand_persistent_state_lanes(experiment, 4)
    metrics = _scientific_metrics(experiment)

    assert experiment.state_lanes == 4
    assert experiment._training_stream_positions.shape == (4, 1)
    expected_positions = (
        original_positions.unsqueeze(0) if starting_lanes == 1 else original_positions
    )
    assert torch.equal(
        experiment._training_stream_positions[:starting_lanes], expected_positions
    )
    old_phases = set(expected_positions.flatten().remainder(8).tolist())
    new_phases = experiment._training_stream_positions[
        starting_lanes:
    ].flatten().remainder(8).tolist()
    assert len(set(new_phases)) == len(new_phases)
    assert old_phases.isdisjoint(new_phases)
    expected_lengths = (
        original_lengths.unsqueeze(0) if starting_lanes == 1 else original_lengths
    )
    assert torch.equal(
        experiment._training_stream_lengths[:starting_lanes], expected_lengths
    )
    for index, original_state in enumerate(original_states):
        assert experiment._training_runtime_bank[index] is original_state
    assert experiment._training_runtime_bank[starting_lanes:] == [None] * (
        4 - starting_lanes
    )
    assert not torch.equal(experiment.generator.get_state(), original_rng)
    for name, value in experiment.model.state_dict().items():
        assert torch.equal(value, original_model[name])
    assert (
        experiment.optimizer.state_dict()["param_groups"]
        == original_optimizer["param_groups"]
    )
    for parameter, state in experiment.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            original_value = original_optimizer["state"][parameter][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, original_value)
            else:
                assert value == original_value
    assert metrics["minimumElectricalStateTokens"] == 0
    assert metrics["maximumElectricalStateTokens"] == 8
    assert metrics["activeStateLanes"] == starting_lanes
    assert metrics["coldStateLanes"] == 4 - starting_lanes
    assert metrics["experienceTrajectoryCount"] == 4
    assert metrics["laneStreamDomains"] == [
        {
            "tokens": task.training_stream_tokens,
            "lanes": 4,
            "firstLane": 0,
            "uniqueCursorPhases": 4,
            "cursorPhaseCoverage": 0.5,
            "minimumCursorPhaseLanes": 0,
            "maximumCursorPhaseLanes": 1,
        }
    ]
    assert metrics["uniqueCursorPhases"] == len(
        set(experiment._training_stream_positions.flatten().remainder(8).tolist())
    )
    assert metrics["cursorPhaseCoverage"] == pytest.approx(
        metrics["uniqueCursorPhases"] / 8
    )
    assert metrics["minimumCursorPhaseLanes"] == 0
    assert metrics["maximumCursorPhaseLanes"] >= 1

    checkpoint = tmp_path / f"lanes-{starting_lanes}.pt"
    save_checkpoint(
        checkpoint, experiment, context_length=8, amp_mode="off",
        organism_id="organism-lanes", phase_index=3, phase_name="lane expansion",
    )
    restored = SequenceExperiment(
        task, replace(corpus_config(), batch_size=1), seed=58, device="cpu",
        stream_mode="continuous", state_retention=0.9, state_lanes=4,
    )
    restore_checkpoint(restored, load_checkpoint(checkpoint, torch.device("cpu")))
    assert restored.state_lanes == 4
    assert torch.equal(
        restored._training_stream_positions[:starting_lanes], expected_positions
    )
    assert torch.equal(restored._training_stream_lengths, experiment._training_stream_lengths)
    for index, original_state in enumerate(original_states):
        assert original_state is not None
        assert restored._training_runtime_bank[index] is not None
        assert torch.equal(
            restored._training_runtime_bank[index].hidden, original_state.hidden
        )
    assert restored._training_runtime_bank[starting_lanes:] == [None] * (
        4 - starting_lanes
    )
    with pytest.raises(ValueError, match="cannot discard"):
        expand_persistent_state_lanes(experiment, 2)


def test_stream_domain_expansion_preserves_organism_state_and_next_cursor() -> None:
    text = "a" * 128 + "z" * 384
    old_task = build_token_task(
        text, context_length=8, vocabulary_size=256,
        tokenizer_profile="byte", training_shard_tokens=128,
    )
    broader_task = build_token_task(
        text, context_length=8, vocabulary_size=256,
        tokenizer_profile="byte", training_shard_tokens=256,
    )
    experiment = SequenceExperiment(
        old_task, replace(corpus_config(), batch_size=1), seed=159,
        device="cpu", stream_mode="continuous", state_lanes=2,
    )
    experiment.train_updates(2)
    positions = experiment._training_stream_positions.clone()
    runtime_bank = list(experiment._training_runtime_bank)
    model = {
        name: value.detach().clone()
        for name, value in experiment.model.state_dict().items()
    }
    optimizer = copy.deepcopy(experiment.optimizer.state_dict())
    generator = experiment.generator.get_state().clone()
    experiment.task = broader_task

    expand_persistent_stream_domains(experiment, 256)

    assert torch.equal(experiment._training_stream_positions, positions)
    assert torch.equal(
        experiment._training_stream_lengths, torch.full_like(positions, 256)
    )
    assert all(
        current is original
        for current, original in zip(experiment._training_runtime_bank, runtime_bank)
    )
    assert torch.equal(experiment.generator.get_state(), generator)
    assert experiment.optimizer.state_dict()["param_groups"] == optimizer["param_groups"]
    for name, value in experiment.model.state_dict().items():
        assert torch.equal(value, model[name])

    boundary = torch.tensor([124])
    old_window, _ = broader_task.stream_batch(
        boundary, stream_lengths=torch.tensor([128])
    )
    expanded_window, _ = broader_task.stream_batch(
        boundary, stream_lengths=torch.tensor([256])
    )
    assert torch.equal(old_window.tokens[:, :4], expanded_window.tokens[:, :4])
    assert old_window.targets[0, 3] != expanded_window.targets[0, 3]

    active_lane = experiment.training_step % experiment.state_lanes
    experiment._training_stream_positions[active_lane, 0] = 130
    experiment.train_updates(1)
    assert experiment.last_novel_stream_token_fraction == pytest.approx(1.0)
    assert experiment.last_novel_stream_rows == 1
    metrics = _scientific_metrics(experiment)
    assert metrics["novelStreamTokenFraction"] == pytest.approx(1.0)
    assert metrics["novelStreamRows"] == 1


def test_large_lane_expansion_fills_every_cursor_phase_without_moving_old_lanes() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 100
    task = build_token_task(text, context_length=64, vocabulary_size=32)
    experiment = SequenceExperiment(
        task, replace(corpus_config(), batch_size=1), seed=62, device="cpu",
        stream_mode="continuous", state_lanes=32,
    )
    preserved = torch.tensor(list(range(26)) + list(range(6))).reshape(32, 1)
    experiment._training_stream_positions = preserved.clone()
    experiment._training_stream_lengths = torch.full_like(
        preserved, task.training_stream_tokens
    )

    expand_persistent_state_lanes(experiment, 96)
    metrics = _scientific_metrics(experiment)

    assert torch.equal(experiment._training_stream_positions[:32], preserved)
    assert experiment._training_runtime_bank[:32] == [None] * 32
    assert experiment._training_runtime_bank[32:] == [None] * 64
    assert metrics["uniqueCursorPhases"] == 64
    assert metrics["minimumCursorPhaseLanes"] == 1
    assert metrics["maximumCursorPhaseLanes"] == 2


def test_lane_expansion_preserves_old_domain_and_assigns_new_domain_only_to_new_lanes(
    tmp_path: Path,
) -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 100
    old_task = build_token_task(
        text, context_length=8, vocabulary_size=32, training_shard_tokens=128
    )
    expanded_task = build_token_task(
        text, context_length=8, vocabulary_size=32, training_shard_tokens=256
    )
    config = replace(corpus_config(), batch_size=1)
    original = SequenceExperiment(
        old_task, config, seed=61, device="cpu", stream_mode="continuous",
        state_lanes=2,
    )
    original.train_updates(2)
    old_positions = original._training_stream_positions.clone()
    old_states = [state.cloned_detached() for state in original._training_runtime_bank]
    checkpoint = tmp_path / "old-domain.pt"
    save_checkpoint(
        checkpoint, original, context_length=8, amp_mode="off",
        organism_id="organism-domains", phase_index=4, phase_name="128 tokens",
    )
    payload = load_checkpoint(checkpoint, torch.device("cpu"))
    del payload["experiment"]["_training_stream_lengths"]

    restored = SequenceExperiment(
        expanded_task, config, seed=61, device="cpu", stream_mode="continuous",
        state_lanes=4,
    )
    restore_checkpoint(restored, payload)
    expand_persistent_state_lanes(restored, 4)
    metrics = _scientific_metrics(restored)

    assert restored.state_lanes == 4
    assert torch.equal(restored._training_stream_positions[:2], old_positions)
    assert torch.equal(
        restored._training_stream_lengths[:, 0],
        torch.tensor([128, 128, 256, 256]),
    )
    assert restored._training_runtime_bank[2:] == [None, None]
    for index, state in enumerate(old_states):
        assert torch.equal(restored._training_runtime_bank[index].hidden, state.hidden)
        assert restored._training_runtime_bank[index].position == state.position
    assert metrics["laneStreamDomains"] == [
        {
            "tokens": 128, "lanes": 2, "firstLane": 0,
            "uniqueCursorPhases": 2, "cursorPhaseCoverage": 0.25,
            "minimumCursorPhaseLanes": 0, "maximumCursorPhaseLanes": 1,
        },
        {
            "tokens": 256, "lanes": 2, "firstLane": 2,
            "uniqueCursorPhases": 2, "cursorPhaseCoverage": 0.25,
            "minimumCursorPhaseLanes": 0, "maximumCursorPhaseLanes": 1,
        },
    ]
    restored.train_updates(4)
    assert [state.position for state in restored._training_runtime_bank] == [16, 16, 8, 8]
    positions_before_audit = restored._training_stream_positions.clone()
    lane_two_state = restored._training_runtime_bank[2]
    assert lane_two_state is not None
    trajectory = restored.evaluate_metrics(
        1,
        evaluation_split="trajectory",
        trajectory_lane=2,
        initial_runtime_state=lane_two_state,
    )
    assert trajectory["trajectoryLane"] == 2
    assert trajectory["trajectoryStreamTokens"] == 256
    assert trajectory["initialStateTokens"] == lane_two_state.position
    assert torch.equal(restored._training_stream_positions, positions_before_audit)

    shrunken_task = build_token_task(
        text, context_length=8, vocabulary_size=32, training_shard_tokens=64
    )
    shrunken = SequenceExperiment(
        shrunken_task, config, seed=61, device="cpu", stream_mode="continuous",
        state_lanes=2,
    )
    with pytest.raises(ValueError, match="cannot shrink"):
        restore_checkpoint(shrunken, payload)


def test_lane_expansion_balances_phases_within_the_destination_domain() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 100
    task = build_token_task(
        text, context_length=8, vocabulary_size=32, training_shard_tokens=256
    )
    experiment = SequenceExperiment(
        task, replace(corpus_config(), batch_size=1), seed=64, device="cpu",
        stream_mode="continuous", state_lanes=4,
    )
    positions = torch.tensor([[0], [1], [2], [3]])
    lengths = torch.tensor([[128], [128], [256], [256]])
    experiment._training_stream_positions = positions.clone()
    experiment._training_stream_lengths = lengths.clone()

    expand_persistent_state_lanes(experiment, 10)
    metrics = _scientific_metrics(experiment)

    assert torch.equal(experiment._training_stream_positions[:4], positions)
    assert torch.equal(experiment._training_stream_lengths[:4], lengths)
    new_domain = experiment._training_stream_positions[
        experiment._training_stream_lengths[:, 0] == 256
    ]
    assert set(new_domain.flatten().remainder(8).tolist()) == set(range(8))
    domains = {domain["tokens"]: domain for domain in metrics["laneStreamDomains"]}
    assert domains[128]["uniqueCursorPhases"] == 2
    assert domains[256]["uniqueCursorPhases"] == 8
    assert domains[256]["minimumCursorPhaseLanes"] == 1
    assert domains[256]["maximumCursorPhaseLanes"] == 1


def test_state_lane_expansion_places_new_cursors_on_checkpoint_device() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    experiment = SequenceExperiment(
        task, replace(corpus_config(), batch_size=1), seed=59, device="cpu",
        stream_mode="continuous", state_lanes=1,
    )
    experiment._training_stream_positions = (
        experiment._training_stream_positions.to(device="meta")
    )

    expand_persistent_state_lanes(experiment, 2)

    assert experiment._training_stream_positions.device.type == "meta"
    assert experiment._training_stream_positions.shape == (2, 1)


def test_continuous_training_state_survives_checkpoint_resume(tmp_path: Path) -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(
        text, context_length=8, vocabulary_size=32, training_shard_tokens=128
    )
    config = replace(
        corpus_config(), batch_size=1, structural_enabled=0, lifecycle_enabled=0
    )
    original = SequenceExperiment(
        task, config, seed=45, device="cpu", stream_mode="continuous",
        state_retention=0.9,
    )
    original.train_updates(1)
    checkpoint = tmp_path / "latest.pt"
    save_checkpoint(
        checkpoint, original, context_length=8, amp_mode="off",
        organism_id="organism-test", phase_index=0, phase_name="warm-up",
    )

    phase_config = plasticity_phase_config(
        config, structure=True, lifecycle=False, lifecycle_profile="off",
        topology_profile="prune_only",
        gradient_clip=5.0,
        max_grown_per_generation=64,
        axon_growth_cost=0.08,
        axon_growth_energy_reserve=0.25,
        new_axon_initial_utility=0,
    )
    restored = SequenceExperiment(
        task, phase_config, seed=45, device="cpu", stream_mode="continuous",
        state_retention=0.9,
        topology_profile="prune_only",
    )
    payload = load_checkpoint(checkpoint, torch.device("cpu"))
    restore_checkpoint(restored, payload)
    restored.topology_profile = "prune_only"
    reconcile_plasticity_phase_status(restored)

    assert payload["lineage"] == {
        "organism_id": "organism-test", "phase_index": 0, "phase_name": "warm-up",
    }
    assert payload["task"]["training_shard_tokens"] == 128
    assert restored.config.structural_enabled == 1
    assert restored.config.gradient_clip == pytest.approx(5.0)
    assert restored.config.max_grown_per_generation == 64
    assert restored.config.axon_growth_cost == pytest.approx(0.08)
    assert restored.config.axon_growth_energy_reserve == pytest.approx(0.25)
    assert restored.config.new_axon_initial_utility == 0
    assert restored.topology_profile == "prune_only"
    assert restored.stream_mode == "continuous"
    assert restored.state_retention == pytest.approx(0.9)
    assert torch.equal(
        restored._training_stream_positions, original._training_stream_positions
    )
    assert torch.equal(
        restored._training_runtime_state.hidden,
        original._training_runtime_state.hidden,
    )
    assert torch.equal(
        restored.model.substrate.occupied, original.model.substrate.occupied
    )
    assert torch.equal(
        restored.model.substrate.dendrite_source,
        original.model.substrate.dendrite_source,
    )
    assert torch.equal(
        restored.model.substrate.synapse_weight,
        original.model.substrate.synapse_weight,
    )
    assert torch.equal(
        restored.model.substrate.edge_age, original.model.substrate.edge_age
    )
    assert torch.equal(
        restored.model.substrate.edge_utility, original.model.substrate.edge_utility
    )
    assert torch.equal(
        restored.model.substrate.genotype, original.model.substrate.genotype
    )
    assert restored.model.substrate.generation == original.model.substrate.generation
    assert restored.best_rolling_accuracy == original.best_rolling_accuracy
    assert (
        restored.last_accuracy_improvement_step
        == original.last_accuracy_improvement_step
    )
    assert restored.cumulative_grown_edges == original.cumulative_grown_edges
    assert restored.cumulative_pruned_edges == original.cumulative_pruned_edges


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


def test_population_turnover_reconciles_every_persistent_state_lane() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    experiment = SequenceExperiment(
        task,
        replace(
            corpus_config(), batch_size=1, structural_enabled=0, lifecycle_enabled=0
        ),
        seed=63,
        device="cpu",
        stream_mode="continuous",
        state_retention=0.9,
        state_lanes=2,
    )
    experiment.train_updates(2)
    substrate = experiment.model.substrate
    original_bank = list(experiment._training_runtime_bank)
    assert all(state is not None for state in original_bank)
    protected = set(substrate.input_sites.tolist() + substrate.output_sites.tolist())
    victim = next(
        site for site in substrate.living_sites.tolist() if site not in protected
    )
    newborn = next(
        site for site in range(experiment.config.site_count)
        if not bool(substrate.occupied[site]) and site not in protected
    )
    survivor = next(
        site for site in substrate.living_sites.tolist()
        if site not in protected and site != victim
    )
    survivor_values = []
    positions = []
    for state in original_bank:
        assert state is not None
        survivor_index = int((state.sites == survivor).nonzero()[0])
        survivor_values.append(state.hidden[:, survivor_index].clone())
        positions.append(state.position)
    previous_current_index = next(
        index for index, state in enumerate(original_bank)
        if state is experiment._training_runtime_state
    )
    substrate.occupied[victim] = False
    substrate.occupied[newborn] = True

    experiment._reconcile_persistent_runtime_bank()

    for index, state in enumerate(experiment._training_runtime_bank):
        assert state is not None
        assert victim not in state.sites.tolist()
        assert newborn in state.sites.tolist()
        survivor_index = int((state.sites == survivor).nonzero()[0])
        assert torch.equal(state.hidden[:, survivor_index], survivor_values[index])
        assert state.position == positions[index]
        assert bool(torch.isfinite(state.hidden).all())
    assert (
        experiment._training_runtime_state
        is experiment._training_runtime_bank[previous_current_index]
    )
    experiment.train_updates(2)


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


def test_runtime_state_counterfactual_clone_copies_every_electrical_channel() -> None:
    state = SequenceRuntimeState(
        sites=torch.tensor([2, 5]),
        hidden=torch.randn(1, 2, 3),
        cell_memory=torch.randn(1, 2, 3),
        workspace=torch.randn(1, 2, 3),
        fast_memory=torch.randn(1, 3, 3),
        binding_memory=torch.randn(1, 2, 3),
        previous_binding_key=torch.randn(1, 3),
        position=123,
    )

    cloned = state.cloned_detached()

    assert cloned.position == state.position
    for original, copied in (
        (state.sites, cloned.sites),
        (state.hidden, cloned.hidden),
        (state.cell_memory, cloned.cell_memory),
        (state.workspace, cloned.workspace),
        (state.fast_memory, cloned.fast_memory),
        (state.binding_memory, cloned.binding_memory),
        (state.previous_binding_key, cloned.previous_binding_key),
    ):
        assert original is not None and copied is not None
        assert torch.equal(original, copied)
        assert original.data_ptr() != copied.data_ptr()


def test_state_ablation_reuses_identical_validation_stream_and_one_rng_advance() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    config = replace(corpus_config(), batch_size=1)
    experiment = SequenceExperiment(
        task, config, seed=47, device="cpu", stream_mode="continuous"
    )
    experiment.train_updates(1)
    assert experiment._training_runtime_state is not None
    checkpoint_state = experiment._training_runtime_state.cloned_detached()
    before = experiment.eval_generator.get_state().clone()

    carried, cold = experiment.evaluate_state_ablation(2)
    paired_after = experiment.eval_generator.get_state().clone()
    experiment.eval_generator.set_state(before)
    single = experiment.evaluate_metrics(
        2,
        carry_state=True,
        initial_runtime_state=experiment._training_runtime_state,
    )

    assert carried["stateCarry"] is True
    assert cold["stateCarry"] is False
    assert carried["initialStateTokens"] == task.sequence_length
    assert cold["initialStateTokens"] == 0
    assert carried["positionIndices"] == cold["positionIndices"]
    assert torch.equal(paired_after, experiment.eval_generator.get_state())
    assert carried == single

    diagnostic = _held_out_diagnostics(experiment, 2)
    assert diagnostic["coldStateAccuracy"] >= 0
    assert diagnostic["stateCarryAccuracyDelta"] == pytest.approx(
        diagnostic["accuracy"] - diagnostic["coldStateAccuracy"]
    )
    assert diagnostic["stateCarryLossDelta"] == pytest.approx(
        diagnostic["coldStateLoss"] - diagnostic["loss"]
    )

    horizon_before = experiment.eval_generator.get_state().clone()
    curve = experiment.evaluate_state_horizons(4, horizons=(1, 2, 4))
    horizon_after = experiment.eval_generator.get_state().clone()
    experiment.eval_generator.set_state(horizon_before)
    experiment.evaluate_metrics(
        4,
        carry_state=True,
        state_horizon_windows=1,
        initial_runtime_state=experiment._training_runtime_state,
    )
    assert [point["windows"] for point in curve] == [1, 2, 4]
    assert [point["tokens"] for point in curve] == [8, 16, 32]
    assert torch.equal(horizon_after, experiment.eval_generator.get_state())
    assert experiment._training_runtime_state.position == checkpoint_state.position
    assert torch.equal(
        experiment._training_runtime_state.hidden, checkpoint_state.hidden
    )
    assert torch.equal(
        experiment._training_runtime_state.workspace, checkpoint_state.workspace
    )


def test_fixed_seed_checkpoint_audit_is_repeatable_and_sampler_read_only() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    experiment = SequenceExperiment(
        task, replace(corpus_config(), batch_size=1), seed=53,
        device="cpu", stream_mode="continuous",
    )
    experiment.train_updates(1)
    before = experiment.eval_generator.get_state().clone()

    first = _held_out_diagnostics(experiment, 3, evaluation_seed=12345)
    middle = experiment.eval_generator.get_state().clone()
    second = _held_out_diagnostics(experiment, 3, evaluation_seed=12345)

    assert first == second
    assert first["evaluationSeed"] == 12345
    assert first["evaluationBatches"] == 3
    assert first["evaluatedTokens"] == 24
    assert first["accuracy"] == first["graphReferenceAccuracy"]
    assert first["loss"] == pytest.approx(first["graphReferenceLoss"])
    assert torch.equal(before, middle)
    assert torch.equal(before, experiment.eval_generator.get_state())

    shard = _held_out_diagnostics(
        experiment, 3, evaluation_seed=12345, evaluation_split="training"
    )
    assert shard["evaluationSplit"] == "training"
    assert shard["evaluationSeed"] == 12345
    assert shard.get("trainingShardTokens") is None
    assert torch.equal(before, experiment.eval_generator.get_state())

    training_rng_before = experiment.generator.get_state().clone()
    random_context = _held_out_diagnostics(
        experiment, 3, evaluation_seed=12345,
        evaluation_split="random_context",
    )
    repeated_random_context = _held_out_diagnostics(
        experiment, 3, evaluation_seed=12345,
        evaluation_split="random_context",
    )
    assert random_context == repeated_random_context
    assert random_context["evaluationSplit"] == "random_context"
    assert random_context["evaluationSeed"] == 12345
    assert random_context["stateCarry"] is False
    assert "coldStateAccuracy" not in random_context
    assert "stateCarryAccuracyDelta" not in random_context
    assert random_context["accuracy"] == random_context["graphReferenceAccuracy"]
    assert random_context["loss"] == pytest.approx(
        random_context["graphReferenceLoss"]
    )
    assert torch.equal(training_rng_before, experiment.generator.get_state())
    assert torch.equal(before, experiment.eval_generator.get_state())

    full_corpus_context = _held_out_diagnostics(
        experiment, 3, evaluation_seed=12345,
        evaluation_split="full_corpus_context",
    )
    assert full_corpus_context["evaluationSplit"] == "full_corpus_context"
    assert full_corpus_context["stateCarry"] is False
    assert full_corpus_context["accuracy"] == pytest.approx(
        full_corpus_context["graphReferenceAccuracy"]
    )
    assert "coldStateAccuracy" not in full_corpus_context
    assert torch.equal(training_rng_before, experiment.generator.get_state())
    assert torch.equal(before, experiment.eval_generator.get_state())

    trajectory = _held_out_diagnostics(
        experiment, 3, evaluation_seed=12345, evaluation_split="trajectory",
        trajectory_lane=0,
    )
    assert trajectory["evaluationSplit"] == "trajectory"
    assert trajectory["trajectoryLane"] == 0
    assert trajectory["trajectoryStreamTokens"] == task.training_stream_tokens
    assert trajectory["initialStateTokens"] == experiment._training_runtime_state.position
    assert torch.equal(before, experiment.eval_generator.get_state())
    with pytest.raises(ValueError, match="requires the trajectory"):
        experiment.evaluate_metrics(
            1, evaluation_split="validation", trajectory_lane=0
        )
    with pytest.raises(ValueError, match="between 0 and 0"):
        experiment.evaluate_metrics(
            1, evaluation_split="trajectory", trajectory_lane=1
        )


def test_phase_metric_history_survives_same_phase_worker_resume(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    records = [
        {
            "type": "train", "phaseIndex": 17, "phaseName": "breadth",
            "accuracy": 0.1, "loss": 4.0,
        },
        {
            "type": "train", "phaseIndex": 18, "phaseName": "auxiliary",
            "accuracy": 0.2, "loss": 3.0,
        },
        {
            "type": "diagnostic", "phaseIndex": 18, "phaseName": "auxiliary",
            "accuracy": 1.0, "loss": 0.0,
        },
        {
            "type": "train", "phaseIndex": 18, "phaseName": "auxiliary",
            "accuracy": 0.4, "loss": 2.0,
        },
        {
            "type": "train", "phaseIndex": 18, "phaseName": "auxiliary",
            "accuracy": 0.6, "loss": 1.0,
        },
    ]
    metrics.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n{partial",
        encoding="utf-8",
    )

    accuracy, loss = _phase_metric_history(
        metrics, phase_index=18, phase_name="auxiliary", maximum=2
    )

    assert list(accuracy) == [0.4, 0.6]
    assert list(loss) == [2.0, 1.0]
    assert accuracy.maxlen == 2
    assert loss.maxlen == 2


def test_phase_novel_exposure_survives_legacy_and_cumulative_metrics(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    records = [
        {
            "type": "train", "phaseIndex": 21, "phaseName": "breadth",
            "novelStreamTokenFraction": 0.25,
        },
        {
            "type": "train", "phaseIndex": 21, "phaseName": "breadth",
            "novelStreamTokenFraction": 0.0,
        },
        {
            "type": "train", "phaseIndex": 22, "phaseName": "control",
            "novelStreamTokenFraction": 1.0,
        },
    ]
    metrics.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n{partial",
        encoding="utf-8",
    )

    assert _phase_novel_exposure(
        metrics, phase_index=21, phase_name="breadth",
        sensory_tokens_per_update=64,
    ) == (16, 128, 1, 2)

    with metrics.open("a", encoding="utf-8") as stream:
        stream.write("\n" + json.dumps({
            "type": "train", "phaseIndex": 21, "phaseName": "breadth",
            "phaseNovelStreamTokens": 28, "phaseSensoryTokens": 192,
            "phaseNovelStreamWindows": 2, "phaseTrainingWindows": 3,
        }) + "\n")
    assert _phase_novel_exposure(
        metrics, phase_index=21, phase_name="breadth",
        sensory_tokens_per_update=64,
    ) == (28, 192, 2, 3)


def test_graph_ablation_is_causal_matched_and_restores_the_organism() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    config = replace(corpus_config(), batch_size=1)
    experiment = SequenceExperiment(
        task, config, seed=51, device="cpu", stream_mode="continuous"
    )
    control = SequenceExperiment(
        task, config, seed=51, device="cpu", stream_mode="continuous"
    )
    experiment.train_updates(1)
    control.train_updates(1)
    assert experiment._training_runtime_state is not None
    checkpoint_state = experiment._training_runtime_state.cloned_detached()
    sources = experiment.model.substrate.dendrite_source.clone()
    weights = experiment.model.substrate.synapse_weight.detach().clone()

    reference, silenced, rotated, reassigned, broadcast_silenced = (
        experiment.evaluate_graph_ablation(2)
    )
    control_reference = control.evaluate_metrics(
        2, initial_runtime_state=control._training_runtime_state
    )

    assert reference == control_reference
    assert torch.equal(
        experiment.eval_generator.get_state(), control.eval_generator.get_state()
    )
    assert torch.equal(experiment.model.substrate.dendrite_source, sources)
    assert torch.equal(experiment.model.substrate.synapse_weight, weights)

    trajectory_reference, trajectory_silenced, *_ = (
        experiment.evaluate_graph_ablation(2, evaluation_split="trajectory")
    )
    assert trajectory_reference["evaluationSplit"] == "trajectory"
    assert trajectory_silenced["trajectoryLane"] == 0
    assert torch.equal(experiment.model.substrate.dendrite_source, sources)
    assert torch.equal(experiment.model.substrate.synapse_weight, weights)
    assert experiment.model.broadcast_gain.item() == pytest.approx(
        control.model.broadcast_gain.item()
    )
    assert experiment._training_runtime_state.position == checkpoint_state.position
    assert torch.equal(
        experiment._training_runtime_state.hidden, checkpoint_state.hidden
    )
    for metrics in (
        reference, silenced, rotated, reassigned, broadcast_silenced
    ):
        assert math.isfinite(metrics["loss"])
        assert 0 <= metrics["accuracy"] <= 1

    assert reassigned["positionIndices"] == reference["positionIndices"]
    assert broadcast_silenced["positionIndices"] == reference["positionIndices"]

    training_reference, training_silenced, *_ = experiment.evaluate_graph_ablation(
        2, evaluation_split="training"
    )
    assert training_reference["evaluationSplit"] == "training"
    assert training_silenced["evaluationSplit"] == "training"
    assert torch.equal(experiment.model.substrate.dendrite_source, sources)
    assert torch.equal(experiment.model.substrate.synapse_weight, weights)

    random_reference, random_silenced, *_ = experiment.evaluate_graph_ablation(
        2, evaluation_split="random_context"
    )
    assert random_reference["evaluationSplit"] == "random_context"
    assert random_reference["stateCarry"] is False
    assert random_silenced["evaluationSplit"] == "random_context"
    assert torch.equal(experiment.model.substrate.dendrite_source, sources)
    assert torch.equal(experiment.model.substrate.synapse_weight, weights)
    assert experiment._training_runtime_state.position == checkpoint_state.position
    assert torch.equal(
        experiment._training_runtime_state.hidden, checkpoint_state.hidden
    )

    full_reference, full_silenced, *_ = experiment.evaluate_graph_ablation(
        2, evaluation_split="full_corpus_context"
    )
    assert full_reference["evaluationSplit"] == "full_corpus_context"
    assert full_reference["stateCarry"] is False
    assert full_silenced["evaluationSplit"] == "full_corpus_context"
    assert torch.equal(experiment.model.substrate.dendrite_source, sources)
    assert torch.equal(experiment.model.substrate.synapse_weight, weights)


def test_zero_broadcast_graph_audit_reports_identical_reference_branch() -> None:
    text = "One fox ran. Two birds flew. Three cats slept. " * 30
    task = build_token_task(text, context_length=8, vocabulary_size=32)
    config = replace(corpus_config(), batch_size=1, broadcast_gain=0.0)
    experiment = SequenceExperiment(
        task, config, seed=54, device="cpu", stream_mode="continuous"
    )
    experiment.train_updates(1)

    reference, _, _, _, broadcast_silenced = experiment.evaluate_graph_ablation(2)
    diagnostic = _held_out_diagnostics(experiment, 2, evaluation_seed=123)

    assert broadcast_silenced == reference
    assert diagnostic["broadcastAblationApplicable"] is False
    assert diagnostic["broadcastSilencedAccuracyDelta"] == 0.0
    assert diagnostic["broadcastSilencedLossDelta"] == 0.0


def test_process_failure_is_bounded_and_persisted_before_first_update(tmp_path: Path) -> None:
    checkpoint = tmp_path / "failed"

    _record_process_failure(
        ["--checkpoint-dir", str(checkpoint)],
        RuntimeError("out of memory\n" + "x" * 2_000),
    )

    record = json.loads((checkpoint / "metrics.jsonl").read_text().strip())
    assert record["type"] == "failure"
    assert record["failureType"] == "RuntimeError"
    assert "\n" not in record["failureMessage"]
    assert len(record["failureMessage"]) == 1_000


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


def test_structural_checkpoint_boundary_precedes_every_possible_mutation() -> None:
    task = build_token_task(
        "Once upon a time there was a small persistent neuron. " * 24,
        context_length=8, vocabulary_size=32,
    )
    lifecycle = SequenceExperiment(
        task,
        replace(
            corpus_config(), structural_enabled=0, lifecycle_enabled=True,
            lifecycle_warmup_trials=4, lifecycle_interval=3,
        ),
        seed=45, device="cpu", topology_profile="fixed",
    )
    lifecycle.training_step = 2
    assert structural_checkpoint_due(lifecycle) is False
    lifecycle.training_step = 3
    assert structural_checkpoint_due(lifecycle) is True
    lifecycle.training_step = 4
    assert structural_checkpoint_due(lifecycle) is False
    lifecycle.training_step = 6
    assert structural_checkpoint_due(lifecycle) is True

    topology = SequenceExperiment(
        task,
        replace(
            corpus_config(), structural_enabled=1, lifecycle_enabled=False,
            structural_warmup_trials=5, structural_interval=2,
        ),
        seed=46, device="cpu", topology_profile="adaptive",
    )
    topology.training_step = 4
    assert structural_checkpoint_due(topology) is True
    topology.topology_profile = "fixed"
    assert structural_checkpoint_due(topology) is False


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
