"""Persistent-lifetime MNIST training with trace playback and structural cycles."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset

from .config import resolve_device
from .graph_layout import GraphLayout, resolve_layout
from .mnist_config import MnistModelConfig
from .mnist_curriculum import CurriculumStage, build_curriculum
from .mnist_data import load_mnist_datasets, make_loader
from .mnist_model import CellularGraphClassifier, MnistForward, MnistFrame


class MnistExperiment:
    """Train one organism whose neurons and connectome persist across digits."""

    experiment_name = "mnist"

    def __init__(
        self,
        config: MnistModelConfig | None = None,
        *,
        layout: str | GraphLayout = "mnist",
        seed: int = 1,
        device: str = "auto",
        train_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
        data_root: Path | None = None,
    ) -> None:
        self.config = config or MnistModelConfig()
        self.layout = resolve_layout(layout)
        self.experiment_name = self.layout.key
        self.seed = seed
        self.device = resolve_device(device)
        if train_dataset is None or test_dataset is None:
            canonical_train, canonical_test = load_mnist_datasets(data_root)
            train_dataset = train_dataset or canonical_train
            test_dataset = test_dataset or canonical_test
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.curriculum: list[CurriculumStage] = build_curriculum(train_dataset, seed=seed)
        self.curriculum_stage_index = 0
        self.curriculum_stage_updates = 0
        self.stage_accuracy_history: deque[float] = deque(
            maxlen=self.config.curriculum_window_trials
        )
        self.train_loader = make_loader(
            self.curriculum[0].dataset,
            batch_size=self.curriculum[0].examples,
            seed=seed,
            shuffle=True,
            device=self.device,
        )
        self.test_loader = make_loader(
            test_dataset,
            batch_size=max(16, self.config.batch_size),
            seed=seed + 1,
            shuffle=False,
            device=self.device,
        )
        self._train_iterator: Iterator[Any] = iter(self.train_loader)
        self._test_iterator: Iterator[Any] = iter(self.test_loader)
        torch.manual_seed(seed)
        self.model = CellularGraphClassifier(
            self.config, layout=self.layout, seed=seed
        ).to(self.device)
        readout_parameters = list(self.model.readout_parameters())
        readout_ids = {id(parameter) for parameter in readout_parameters}
        rule_parameters = [
            parameter for parameter in self.model.shared_parameters()
            if id(parameter) not in readout_ids
        ]
        self.optimizer = torch.optim.Adam(
            [
                {"params": readout_parameters, "lr": self.config.readout_learning_rate},
                {"params": rule_parameters, "lr": self.config.learning_rate},
            ]
        )
        self.synapse_optimizer = torch.optim.Adam(
            [self.model.substrate.synapse_weight], lr=self.config.synapse_learning_rate
        )
        self.tick = 0
        self.training_step = 0
        self.seen_examples = 0
        self.last_loss = 0.0
        self.last_reward = 0.0
        self.last_batch_accuracy = 0.0
        self.test_accuracy: float | None = None
        self.accuracy_history: deque[float] = deque(maxlen=160)
        self.loss_history: deque[float] = deque(maxlen=160)
        self.reward_history: deque[float] = deque(maxlen=160)
        self.events: list[dict[str, Any]] = []
        self.last_trace: list[MnistFrame] = []
        self.visual_cursor = 0
        self.last_image = torch.zeros(1, 28, 28)
        self.last_label = 0
        self.last_prediction = 0
        self.last_confidence = 0.1
        self.last_births = 0
        self.last_deaths = 0
        self.last_death_causes = {"starvation": 0, "overload": 0, "maintenance": 0}
        self.cumulative_births = 0
        self.cumulative_deaths = 0
        self.cumulative_death_causes = {
            "starvation": 0,
            "overload": 0,
            "maintenance": 0,
        }
        self.last_synapse_update_ratio = 0.0
        self.last_mean_attention_entropy = 0.0
        self.best_rolling_accuracy = 0.0
        self.last_accuracy_improvement_step = 0
        self.structure_unlocked = False
        self.structure_unlock_reason = "minimum learning warm-up"
        self.lifecycle_active = False
        self.lifecycle_reason = (
            "waiting for lifecycle warm-up"
            if self.config.lifecycle_enabled
            else "disabled by configuration"
        )
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
        zero = torch.zeros(result.sites.numel(), device=self.device)
        result.frames.append(
            self.model.make_frame(
                result.sites,
                result.final_state[0],
                credit=zero,
                stage="feedback",
                step=self.config.message_steps + 1,
                edge_flow=result.edge_flow,
            )
        )
        result.frames.append(
            self.model.make_frame(
                result.sites,
                result.final_state[0],
                stage="structural",
                step=self.config.message_steps + 2,
            )
        )
        self._remember(result, images[:1], labels[:1])

    def step(self, count: int = 1) -> None:
        """Advance measured trace phases and train once the trace is exhausted."""

        for _ in range(count):
            self.tick += 1
            if self.visual_cursor + 1 < len(self.last_trace):
                self.visual_cursor += 1
                self._publish_frame_events()
            else:
                self._train_trial()

    def _train_trial(self) -> None:
        self.model.train()
        images, labels = self._next_batch(training=True)
        self.optimizer.zero_grad(set_to_none=True)
        self.synapse_optimizer.zero_grad(set_to_none=True)
        result = self.model(images, capture_trace=True)
        classification_loss = torch.nn.functional.cross_entropy(result.logits, labels)
        tail = result.trajectory_logits[:, -min(4, self.config.message_steps) :]
        tail_labels = labels[:, None].expand(-1, tail.shape[1]).reshape(-1)
        trajectory_loss = torch.nn.functional.cross_entropy(tail.reshape(-1, 10), tail_labels)
        loss = (
            classification_loss
            + self.config.trajectory_loss_weight * trajectory_loss
            + self.model.regularization()
        )
        loss.backward()

        neuron_credit = torch.zeros(result.sites.numel(), device=self.device)
        credited_states = 0
        for state in result.retained_states:
            if state.grad is not None:
                neuron_credit += -(state.grad.detach() * state.detach()).mean(dim=(0, 2))
                credited_states += 1
        neuron_credit /= max(1, credited_states)
        edge_gradient = self.model.substrate.synapse_weight.grad
        if edge_gradient is None:
            edge_credit = torch.zeros_like(self.model.substrate.synapse_weight)
        else:
            edge_credit = -edge_gradient.detach() * self.model.substrate.synapse_weight.detach()

        completed_step = self.training_step + 1
        optimization_phase = self._optimization_phase(completed_step)
        if optimization_phase == "readout":
            allowed = {id(parameter) for parameter in self.model.probe_parameters()}
            for parameter in self.model.shared_parameters():
                if id(parameter) not in allowed:
                    parameter.grad = None
        if optimization_phase != "synapse":
            self.model.substrate.synapse_weight.grad = None

        active_edges = self.model.substrate.active_edge_mask
        weight_before = self.model.substrate.synapse_weight.detach().clone()
        shared_with_grad = [
            parameter for parameter in self.model.shared_parameters()
            if parameter.grad is not None
        ]
        if shared_with_grad and optimization_phase != "readout":
            clip_grad_norm_(shared_with_grad, max_norm=self.config.gradient_clip)
        if self.model.substrate.synapse_weight.grad is not None:
            clip_grad_norm_(
                [self.model.substrate.synapse_weight], max_norm=self.config.gradient_clip
            )
        self.optimizer.step()
        if optimization_phase == "synapse":
            self.synapse_optimizer.step()
        with torch.no_grad():
            delta = (self.model.substrate.synapse_weight - weight_before).abs()
            denominator = weight_before.abs().clamp_min(
                self.config.initial_weight_scale * 0.1
            )
            self.last_synapse_update_ratio = (
                float((delta[active_edges] / denominator[active_edges]).mean())
                if active_edges.any() and optimization_phase == "synapse"
                else 0.0
            )

        predictions = result.logits.argmax(dim=1)
        batch_accuracy = float((predictions == labels).float().mean())
        reward = max(
            -1.0,
            min(1.0, (math.log(10) - float(classification_loss.detach())) / math.log(10)),
        )
        self._record_learning_progress(batch_accuracy, completed_step)
        lifecycle_active = self._should_activate_lifecycle(completed_step)
        structure_unlocked = self._should_unlock_structure(completed_step, batch_accuracy)
        self.model.substrate.record_trial(
            result.sites,
            result.stimulation,
            result.load,
            neuron_credit,
            result.edge_flow,
            edge_credit,
            result.advertised_query,
            result.advertised_key,
            result.emission,
            reward,
            homeostasis_active=lifecycle_active,
        )
        feedback = self.model.make_frame(
            result.sites,
            result.final_state[0],
            stimulation=result.stimulation,
            load=result.load,
            credit=neuron_credit,
            stage="feedback",
            step=self.config.message_steps + 1,
            edge_flow=result.edge_flow,
            edge_credit=edge_credit,
        )
        result.frames.append(feedback)

        structural_events: list[dict[str, Any]] = []
        self.last_births = 0
        self.last_deaths = 0
        self.last_death_causes = {"starvation": 0, "overload": 0, "maintenance": 0}
        lifecycle_due = (
            lifecycle_active
            and (completed_step - self.config.lifecycle_warmup_trials)
            % self.config.lifecycle_interval
            == 0
        )
        structure_due = (
            structure_unlocked
            and (completed_step - self.config.structural_warmup_trials)
            % self.config.structural_interval
            == 0
        )
        if lifecycle_due or structure_due:
            update = self.model.substrate.structural_step(
                apply_lifecycle=lifecycle_due,
                apply_topology=structure_due,
            )
            self._reset_mutation_optimizer_state(
                update.changed_edges, update.changed_sites
            )
            structural_events = update.events
            self.last_births = update.births
            self.last_deaths = update.deaths
            self.last_death_causes.update(update.death_causes)
            self.cumulative_births += update.births
            self.cumulative_deaths += update.deaths
            for cause, count in update.death_causes.items():
                self.cumulative_death_causes[cause] += count
        current_sites = self.model.substrate.living_sites
        state_by_site = torch.zeros(
            self.config.site_count, self.config.hidden_channels, device=self.device
        )
        state_by_site[result.sites] = result.final_state[0].detach()
        result.frames.append(
            self.model.make_frame(
                current_sites,
                state_by_site[current_sites],
                stage="structural",
                step=self.config.message_steps + 2,
                events=structural_events,
            )
        )

        self.training_step += 1
        self.seen_examples += int(labels.numel())
        self.last_loss = float(classification_loss.detach())
        self.last_reward = reward
        self.last_batch_accuracy = batch_accuracy
        self.last_mean_attention_entropy = result.mean_attention_entropy
        self.accuracy_history.append(batch_accuracy)
        self.loss_history.append(self.last_loss)
        self.reward_history.append(reward)
        self._remember(result, images, labels)
        self._maybe_advance_curriculum()
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

    def _optimization_phase(self, completed_step: int) -> str:
        """Return the parameter family allowed to learn on this update."""

        if completed_step <= self.config.readout_only_trials:
            return "readout"
        if completed_step <= self.config.synapse_unlock_trials:
            return "rule"
        return "synapse"

    def _record_learning_progress(self, batch_accuracy: float, completed_step: int) -> None:
        self.curriculum_stage_updates += 1
        self.stage_accuracy_history.append(batch_accuracy)
        history = list(self.accuracy_history) + [batch_accuracy]
        rolling = sum(history) / len(history)
        if rolling >= self.best_rolling_accuracy + self.config.structure_plateau_delta:
            self.best_rolling_accuracy = rolling
            self.last_accuracy_improvement_step = completed_step

    def _should_unlock_structure(self, completed_step: int, batch_accuracy: float) -> bool:
        if self.structure_unlocked:
            return True
        if completed_step < self.config.structural_warmup_trials:
            self.structure_unlock_reason = "minimum learning warm-up"
            return False
        recent = list(self.accuracy_history) + [batch_accuracy]
        rolling = sum(recent) / max(1, len(recent))
        if rolling >= self.config.structure_accuracy_threshold:
            self.structure_unlocked = True
            self.structure_unlock_reason = "accuracy competence reached"
        elif completed_step - self.last_accuracy_improvement_step >= self.config.structure_plateau_trials:
            self.structure_unlocked = True
            self.structure_unlock_reason = "learning plateau reached"
        else:
            self.structure_unlock_reason = "waiting for competence or plateau"
        return self.structure_unlocked

    def _should_activate_lifecycle(self, completed_step: int) -> bool:
        if not self.config.lifecycle_enabled:
            self.lifecycle_reason = "disabled by configuration"
            return False
        if completed_step < self.config.lifecycle_warmup_trials:
            self.lifecycle_reason = "waiting for lifecycle warm-up"
            return False
        self.lifecycle_active = True
        self.lifecycle_reason = "energy pressure and turnover active"
        return True

    def _maybe_advance_curriculum(self) -> None:
        stage = self.curriculum[self.curriculum_stage_index]
        if stage.target_accuracy is None:
            return
        if self.curriculum_stage_updates < self.config.curriculum_min_trials:
            return
        accuracy = sum(self.stage_accuracy_history) / max(1, len(self.stage_accuracy_history))
        if accuracy < stage.target_accuracy:
            return
        self.curriculum_stage_index += 1
        self.curriculum_stage_updates = 0
        self.stage_accuracy_history.clear()
        next_stage = self.curriculum[self.curriculum_stage_index]
        self.train_loader = make_loader(
            next_stage.dataset,
            batch_size=(
                next_stage.examples
                if next_stage.examples <= 20
                else self.config.batch_size
            ),
            seed=self.seed + self.curriculum_stage_index,
            shuffle=True,
            device=self.device,
        )
        self._train_iterator = iter(self.train_loader)

    def lifecycle_now(self) -> None:
        """Force one lifecycle plus topology cycle and replay the current digit."""

        update = self.model.substrate.structural_step(
            apply_lifecycle=True, apply_topology=True
        )
        self._reset_mutation_optimizer_state(
            update.changed_edges, update.changed_sites
        )
        self.last_births = update.births
        self.last_deaths = update.deaths
        self.last_death_causes = {
            "starvation": 0,
            "overload": 0,
            "maintenance": 0,
            **update.death_causes,
        }
        self.cumulative_births += update.births
        self.cumulative_deaths += update.deaths
        for cause, count in update.death_causes.items():
            self.cumulative_death_causes[cause] += count
        self._replay_current(events=update.events)

    def rewire_now(self) -> None:
        """Compatibility alias for the former structural-cycle command."""

        self.lifecycle_now()

    @torch.no_grad()
    def _reset_mutation_optimizer_state(
        self, changed_edges: torch.Tensor, changed_sites: torch.Tensor
    ) -> None:
        """Clear Adam moments for dendrite identities and inherited genotypes."""

        parameter = self.model.substrate.synapse_weight
        state = self.synapse_optimizer.state.get(parameter, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(key)
            if isinstance(value, torch.Tensor) and value.shape == changed_edges.shape:
                value.masked_fill_(changed_edges, 0)
        genotype = self.model.substrate.genotype
        genotype_state = self.optimizer.state.get(genotype, {})
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = genotype_state.get(key)
            if isinstance(value, torch.Tensor) and value.shape == genotype.shape:
                value[changed_sites] = 0

    @torch.no_grad()
    def _replay_current(self, *, events: list[dict[str, Any]] | None = None) -> None:
        self.model.eval()
        image = self.last_image.unsqueeze(0).to(self.device)
        label = torch.tensor([self.last_label], device=self.device)
        result = self.model(image, capture_trace=True)
        result.frames.append(
            self.model.make_frame(
                result.sites,
                result.final_state[0],
                stage="feedback",
                step=self.config.message_steps + 1,
                edge_flow=result.edge_flow,
            )
        )
        result.frames.append(
            self.model.make_frame(
                result.sites,
                result.final_state[0],
                stage="structural",
                step=self.config.message_steps + 2,
                events=events,
            )
        )
        self._remember(result, image, label)

    @torch.no_grad()
    def evaluate(self, batches: int = 5) -> float:
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
        count = self.model.lesion(x, y, radius)
        self._replay_current()
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
    def rolling_reward(self) -> float:
        return sum(self.reward_history) / len(self.reward_history) if self.reward_history else 0.0

    @property
    def epoch(self) -> float:
        return self.seen_examples / max(1, len(self.train_dataset))

    @property
    def structural_warmup_remaining(self) -> int:
        return max(0, self.config.structural_warmup_trials - self.training_step)

    @property
    def lifecycle_warmup_remaining(self) -> int:
        if not self.config.lifecycle_enabled:
            return 0
        return max(0, self.config.lifecycle_warmup_trials - self.training_step)

    @property
    def learning_phase(self) -> str:
        if self.structure_unlocked:
            return "structure"
        return self._optimization_phase(self.training_step + 1)

    @property
    def curriculum_stage(self) -> CurriculumStage:
        return self.curriculum[self.curriculum_stage_index]
