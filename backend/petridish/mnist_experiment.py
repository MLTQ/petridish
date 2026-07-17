"""Live outer-loop training and micro-step playback for self-assembling MNIST."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset

from .config import resolve_device
from .mnist_data import load_mnist_datasets, make_loader
from .mnist_model import CellularGraphClassifier, MnistForward, MnistFrame, MnistModelConfig


class MnistExperiment:
    """Meta-train one shared cell program while replaying each assembly trace."""

    experiment_name = "mnist"

    def __init__(
        self,
        config: MnistModelConfig | None = None,
        *,
        seed: int = 1,
        device: str = "auto",
        train_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.config = config or MnistModelConfig()
        self.seed = seed
        self.device = resolve_device(device)
        if train_dataset is None or test_dataset is None:
            canonical_train, canonical_test = load_mnist_datasets(data_root)
            train_dataset = train_dataset or canonical_train
            test_dataset = test_dataset or canonical_test
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader = make_loader(
            train_dataset, batch_size=self.config.batch_size, seed=seed,
            shuffle=True, device=self.device,
        )
        self.test_loader = make_loader(
            test_dataset, batch_size=max(32, self.config.batch_size), seed=seed + 1,
            shuffle=False, device=self.device,
        )
        self._train_iterator: Iterator[Any] = iter(self.train_loader)
        self._test_iterator: Iterator[Any] = iter(self.test_loader)
        torch.manual_seed(seed)
        self.model = CellularGraphClassifier(self.config, seed=seed).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.tick = 0
        self.training_step = 0
        self.seen_examples = 0
        self.last_loss = 0.0
        self.last_batch_accuracy = 0.0
        self.test_accuracy: float | None = None
        self.accuracy_history: deque[float] = deque(maxlen=100)
        self.loss_history: deque[float] = deque(maxlen=100)
        self.events: list[dict[str, Any]] = []
        self.last_trace: list[MnistFrame] = []
        self.visual_cursor = 0
        self.last_image = torch.zeros(1, 28, 28)
        self.last_label = 0
        self.last_prediction = 0
        self.last_confidence = 0.1
        self._prime_preview()

    def _next_batch(self, *, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        loader = self.train_loader if training else self.test_loader
        attribute = "_train_iterator" if training else "_test_iterator"
        iterator = getattr(self, attribute)
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            setattr(self, attribute, iterator)
            images, labels = next(iterator)
        return (
            images.to(self.device, non_blocking=self.device.type == "cuda"),
            labels.to(self.device, non_blocking=self.device.type == "cuda"),
        )

    @torch.no_grad()
    def _prime_preview(self) -> None:
        self.model.eval()
        images, labels = self._next_batch(training=True)
        result = self.model(images[:1], capture_trace=True)
        self._remember(result, images[:1], labels[:1])

    def step(self, count: int = 1) -> None:
        """Advance trace playback; train a new episode after reaching readout."""

        for _ in range(count):
            self.tick += 1
            if self.visual_cursor + 1 < len(self.last_trace):
                self.visual_cursor += 1
                self._publish_frame_events()
            else:
                self._train_episode()

    def _train_episode(self) -> None:
        self.model.train()
        images, labels = self._next_batch(training=True)
        self.optimizer.zero_grad(set_to_none=True)
        result = self.model(images, capture_trace=True)
        classification_loss = torch.nn.functional.cross_entropy(result.logits, labels)
        trajectory_labels = labels.unsqueeze(1).expand(-1, result.trajectory_logits.shape[1]).reshape(-1)
        trajectory_loss = torch.nn.functional.cross_entropy(
            result.trajectory_logits.reshape(-1, 10), trajectory_labels
        )
        loss = classification_loss + 0.45 * trajectory_loss + self.model.regularization(result)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        self.optimizer.step()

        predictions = result.logits.argmax(dim=1)
        batch_accuracy = float((predictions == labels).float().mean())
        self.training_step += 1
        self.seen_examples += int(labels.numel())
        self.last_loss = float(classification_loss.detach())
        self.last_batch_accuracy = batch_accuracy
        self.accuracy_history.append(batch_accuracy)
        self.loss_history.append(self.last_loss)
        self._remember(result, images, labels)
        if self.training_step % self.config.evaluation_interval == 0:
            self.evaluate(self.config.evaluation_batches)

    @torch.no_grad()
    def _remember(self, result: MnistForward, images: torch.Tensor, labels: torch.Tensor) -> None:
        probabilities = result.logits[0].softmax(dim=0)
        self.last_trace = result.frames
        self.visual_cursor = 0
        self.events = []
        self.last_image = images[0].detach().cpu()
        self.last_label = int(labels[0])
        self.last_prediction = int(probabilities.argmax())
        self.last_confidence = float(probabilities.max())

    def _publish_frame_events(self) -> None:
        self.events = [dict(event, tick=self.tick) for event in self.last_frame.events]

    def rewire_now(self) -> None:
        """Skip to a newly trained episode and restart playback from an empty graph."""

        self.tick += 1
        self._train_episode()

    @torch.no_grad()
    def evaluate(self, batches: int = 5) -> float:
        """Measure a bounded held-out slice without retaining assembly traces."""

        self.model.eval()
        correct = 0
        total = 0
        for _ in range(max(1, min(50, batches))):
            images, labels = self._next_batch(training=False)
            logits = self.model(images, capture_trace=False).logits
            correct += int((logits.argmax(dim=1) == labels).sum())
            total += int(labels.numel())
        self.test_accuracy = correct / max(1, total)
        return self.test_accuracy

    @torch.no_grad()
    def lesion(self, x: float, y: float, radius: float) -> int:
        """Mask cells, publish severed current axons, and replay the same digit."""

        before = self.model.lesion_mask.clone()
        graph = self.last_frame.graph
        count = self.model.lesion(x, y, radius)
        damaged = (before > 0.5) & (self.model.lesion_mask < 0.5)
        destination = graph.destination.clamp_min(0)
        incident = damaged.unsqueeze(1) | damaged[destination]
        active = (graph.destination >= 0) & (graph.strength > 0.18)
        severed: list[dict[str, Any]] = []
        for source, slot in (incident & active).nonzero(as_tuple=False)[:128].tolist():
            severed.append(
                {
                    "type": "pruned",
                    "source": source,
                    "destination": int(graph.destination[source, slot]),
                    "tick": self.tick,
                }
            )
        image = self.last_image.unsqueeze(0).to(self.device)
        label = torch.tensor([self.last_label], device=self.device)
        self.model.eval()
        result = self.model(image, capture_trace=True)
        self._remember(result, image, label)
        self.events = severed
        return count

    @property
    def last_frame(self) -> MnistFrame:
        return self.last_trace[self.visual_cursor]

    @property
    def last_state(self) -> torch.Tensor:
        return self.last_frame.state

    @property
    def rolling_accuracy(self) -> float:
        return sum(self.accuracy_history) / len(self.accuracy_history) if self.accuracy_history else 0.0

    @property
    def rolling_loss(self) -> float:
        return sum(self.loss_history) / len(self.loss_history) if self.loss_history else self.last_loss

    @property
    def epoch(self) -> float:
        return self.seen_examples / max(1, len(self.train_dataset))
