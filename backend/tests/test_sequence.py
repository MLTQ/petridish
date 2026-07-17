"""Scientific contracts for token-stream graph experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from petridish.benchmark_sequences import _write_result
from petridish.graph_layout import LAYOUTS, sequence_layout
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
from petridish.train_shakespeare import _migrate_model_state


def small_config():
    return sequence_config(
        width=20, height=20, hidden_channels=8, genotype_channels=6,
        initial_density=0.30, batch_size=3, message_steps=1,
        candidate_probes=12, local_radius=4, max_visible_edges=100,
    )


def test_benchmark_artifact_replacement_is_atomic(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"

    _write_result(output, {"status": "running", "completedSteps": 20})
    _write_result(output, {"status": "complete", "completedSteps": 40})

    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "complete"
    assert list(output.parent.glob(".*.tmp")) == []


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


def test_68_field_size_is_available_only_for_tiny_shakespeare() -> None:
    tiny = sequence_config("tiny_shakespeare")
    choices = hyperparameter_payload(
        tiny, include_sequence=True, task_key="tiny_shakespeare"
    )[0]["choices"]
    assert choices == [16, 32, 64, 68, 128, 256, 512, 1024]
    assert configured(tiny, {"field_size": 68}, task_key="tiny_shakespeare").width == 68

    regular = sequence_config("tiny_language")
    assert 68 not in hyperparameter_payload(
        regular, include_sequence=True, task_key="tiny_language"
    )[0]["choices"]
    with pytest.raises(ValueError, match="power of two"):
        configured(regular, {"field_size": 68}, task_key="tiny_language")


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
