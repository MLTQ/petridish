import torch
from torch.utils.data import TensorDataset

from petridish.mnist_experiment import MnistExperiment
from petridish.mnist_model import CellularGraphClassifier, MnistModelConfig
from petridish.protocol import build_snapshot


def tiny_config() -> MnistModelConfig:
    return MnistModelConfig(
        hidden_channels=8,
        route_channels=4,
        edge_slots=2,
        development_steps=1,
        inference_steps=1,
        routing_interval=1,
        batch_size=4,
        evaluation_interval=2,
        evaluation_batches=1,
    )


def synthetic_digits(count: int = 24) -> TensorDataset:
    generator = torch.Generator().manual_seed(44)
    images = torch.rand(count, 1, 28, 28, generator=generator)
    labels = torch.arange(count) % 10
    return TensorDataset(images, labels)


def test_cellular_classifier_builds_empty_then_routed_graph_with_gradients() -> None:
    config = tiny_config()
    model = CellularGraphClassifier(config, seed=3)
    images, labels = synthetic_digits(8).tensors
    result = model(images[:4])
    loss = torch.nn.functional.cross_entropy(result.logits, labels[:4]) + model.regularization(result)
    loss.backward()

    assert result.logits.shape == (4, 10)
    assert result.trajectory_logits.shape == (
        4,
        config.development_steps + config.inference_steps + 1,
        10,
    )
    assert result.state.shape == (4, 256, 8)
    assert result.graph.destination.shape == (4, 256, 2)
    assert len(result.frames) == config.total_steps + 1
    assert (result.frames[0].graph.destination < 0).all()
    assert (result.frames[1].graph.destination >= 0).all()
    assert model.cell_rule.weight_ih.grad is not None
    assert model.patch_encoder[0].weight.grad is not None
    assert model.router.key.weight.grad is not None
    assert torch.isfinite(result.state).all()


def test_mnist_experiment_trains_replays_evaluates_and_serializes() -> None:
    dataset = synthetic_digits()
    experiment = MnistExperiment(
        tiny_config(),
        seed=5,
        device="cpu",
        train_dataset=dataset,
        test_dataset=dataset,
    )
    assert (experiment.last_frame.graph.destination < 0).all()
    experiment.rewire_now()
    experiment.step(1)
    held_out = experiment.evaluate(2)
    snapshot = build_snapshot(experiment)

    assert experiment.training_step == 1
    assert 0 <= held_out <= 1
    assert snapshot["experiment"] == "mnist"
    assert snapshot["task"]["kind"] == "mnist"
    assert snapshot["task"]["phase"] == "sensing"
    assert snapshot["task"]["assemblyStep"] == 1
    assert len(snapshot["task"]["image"]) == 784
    assert len(snapshot["field"]["cells"]) == 256
    assert snapshot["metrics"]["edgeCount"] > 0
    assert len({len(values) for values in snapshot["edges"].values()}) == 1


def test_mnist_lesion_reassembles_without_incident_edges() -> None:
    dataset = synthetic_digits()
    experiment = MnistExperiment(
        tiny_config(),
        seed=8,
        device="cpu",
        train_dataset=dataset,
        test_dataset=dataset,
    )
    experiment.step(1)
    damaged = experiment.lesion(8, 8, 2.5)
    experiment.step(1)
    snapshot = build_snapshot(experiment)

    assert damaged > 0
    assert snapshot["metrics"]["livingCells"] == 256 - damaged
    sources = snapshot["edges"]["source"]
    destinations = snapshot["edges"]["destination"]
    assert all(experiment.model.lesion_mask[index] > 0.5 for index in sources)
    assert all(experiment.model.lesion_mask[index] > 0.5 for index in destinations)
