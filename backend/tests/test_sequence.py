"""Scientific contracts for token-stream graph experiments."""

from __future__ import annotations

import torch

from petridish.graph_layout import LAYOUTS
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


def small_config():
    return sequence_config(
        width=20, height=20, hidden_channels=8, genotype_channels=6,
        initial_density=0.30, batch_size=3, message_steps=1,
        candidate_probes=12, local_radius=4, max_visible_edges=100,
    )


def test_sequence_layouts_are_directional_port_permutations() -> None:
    recall = LAYOUTS["associative_recall"]
    language = LAYOUTS["tiny_language"]
    assert recall.input_position_order != tuple(range(10))
    assert recall.output_position_order != tuple(range(10))
    assert language.input_side == "right"
    assert language.output_side == "left"
    assert language.flow_direction == -1


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
