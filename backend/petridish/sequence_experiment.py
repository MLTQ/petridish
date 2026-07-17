"""Training lifecycle for persistent token-stream neural organisms."""

from __future__ import annotations

from collections import deque
import math
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .config import resolve_device
from .mnist_config import MnistModelConfig
from .sequence_config import sequence_config
from .sequence_model import CellularSequenceModel, SequenceForward, SequenceFrame
from .sequence_tasks import SequenceBatch, SequenceTask, resolve_sequence_task


class SequenceExperiment:
    """Learn a synthetic sequence distribution in one persistent organism."""

    def __init__(
        self,
        task: str | SequenceTask,
        config: MnistModelConfig | None = None,
        *,
        seed: int = 1,
        device: str = "auto",
    ) -> None:
        self.task = resolve_sequence_task(task)
        self.experiment_name = self.task.key
        self.config = config or sequence_config()
        self.seed = seed
        self.device = resolve_device(device)
        self.generator = torch.Generator().manual_seed(seed)
        self.eval_generator = torch.Generator().manual_seed(seed + 10_000)
        self.model = CellularSequenceModel(
            self.config, layout=self.task.key, vocab_size=len(self.task.vocabulary), seed=seed
        ).to(self.device)
        synapse = self.model.substrate.synapse_weight
        shared = [parameter for parameter in self.model.shared_parameters()]
        readout_ids = {
            id(parameter) for parameter in self.model.output_bank_readout.parameters()
        } | {id(self.model.class_bias)}
        readout = [parameter for parameter in shared if id(parameter) in readout_ids]
        rule = [parameter for parameter in shared if id(parameter) not in readout_ids]
        self.optimizer = torch.optim.Adam(
            [
                {"params": readout, "lr": self.config.readout_learning_rate},
                {"params": rule, "lr": self.config.learning_rate},
                {"params": [synapse], "lr": self.config.synapse_learning_rate},
            ]
        )
        self.tick = 0
        self.training_step = 0
        self.seen_examples = 0
        self.last_loss = math.log(len(self.task.vocabulary))
        self.last_reward = 0.0
        self.last_batch_accuracy = 0.0
        self.test_accuracy: float | None = None
        self.accuracy_history: deque[float] = deque(maxlen=160)
        self.loss_history: deque[float] = deque(maxlen=160)
        self.reward_history: deque[float] = deque(maxlen=160)
        self.last_trace: list[SequenceFrame] = []
        self.visual_cursor = 0
        self.events: list[dict[str, Any]] = []
        self.last_tokens = torch.zeros(self.task.sequence_length, dtype=torch.long)
        self.last_targets = torch.full((self.task.sequence_length,), -100, dtype=torch.long)
        self.last_predictions = torch.zeros(self.task.sequence_length, dtype=torch.long)
        self.last_confidences = torch.full((self.task.sequence_length,), 0.1)
        self.last_births = 0
        self.last_deaths = 0
        self.last_death_causes = {"starvation": 0, "overload": 0, "maintenance": 0}
        self.cumulative_births = 0
        self.cumulative_deaths = 0
        self.cumulative_death_causes = {
            "starvation": 0, "overload": 0, "maintenance": 0
        }
        self.last_synapse_update_ratio = 0.0
        self.last_mean_attention_entropy = 0.0
        self.lifecycle_active = False
        self.lifecycle_reason = (
            "waiting for lifecycle warm-up"
            if self.config.lifecycle_enabled else "disabled by configuration"
        )
        self.structure_unlocked = False
        self.structure_unlock_reason = "minimum learning warm-up"
        self.best_rolling_accuracy = 0.0
        self.last_accuracy_improvement_step = 0
        self._prime_preview()

    def _batch(self, size: int, *, evaluation: bool = False) -> SequenceBatch:
        generator = self.eval_generator if evaluation else self.generator
        batch = self.task.batch(size, generator)
        return SequenceBatch(
            batch.tokens.to(self.device), batch.targets.to(self.device), batch.loss_mask.to(self.device)
        )

    @staticmethod
    def _masked_loss_accuracy(
        logits: torch.Tensor, batch: SequenceBatch
    ) -> tuple[torch.Tensor, float]:
        selected = batch.loss_mask
        loss = F.cross_entropy(logits[selected], batch.targets[selected])
        accuracy = float((logits.argmax(dim=2)[selected] == batch.targets[selected]).float().mean())
        return loss, accuracy

    @torch.no_grad()
    def _prime_preview(self) -> None:
        self.model.eval()
        batch = self._batch(1)
        result = self.model(batch.tokens, capture_trace=True)
        result.frames.extend(self._terminal_frames(result))
        self._remember(result, batch)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.tick += 1
            if self.visual_cursor + 1 < len(self.last_trace):
                self.visual_cursor += 1
                self.events = [dict(event, tick=self.tick) for event in self.last_frame.events]
            else:
                self._train_trial()

    def _train_trial(self) -> None:
        self.model.train()
        batch = self._batch(self.config.batch_size)
        self.optimizer.zero_grad(set_to_none=True)
        result = self.model(batch.tokens, capture_trace=True)
        task_loss, accuracy = self._masked_loss_accuracy(result.logits, batch)
        loss = task_loss + self.model.regularization()
        loss.backward()
        neuron_credit = torch.zeros(result.sites.numel(), device=self.device)
        credited = 0
        for state in result.retained_states:
            if state.grad is not None:
                neuron_credit += -(state.grad.detach() * state.detach()).mean(dim=(0, 2))
                credited += 1
        neuron_credit /= max(1, credited)
        edge_gradient = self.model.substrate.synapse_weight.grad
        edge_credit = (
            torch.zeros_like(self.model.substrate.synapse_weight)
            if edge_gradient is None
            else -edge_gradient.detach() * self.model.substrate.synapse_weight.detach()
        )
        active = self.model.substrate.active_edge_mask
        before = self.model.substrate.synapse_weight.detach().clone()
        clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        with torch.no_grad():
            delta = (self.model.substrate.synapse_weight - before).abs()
            denominator = before.abs().clamp_min(self.config.initial_weight_scale * 0.1)
            self.last_synapse_update_ratio = (
                float((delta[active] / denominator[active]).mean()) if active.any() else 0.0
            )
        completed = self.training_step + 1
        reward = max(-1.0, min(1.0, (math.log(10) - float(task_loss)) / math.log(10)))
        lifecycle_active = self._should_activate_lifecycle(completed)
        structure_unlocked = self._should_unlock_structure(completed, accuracy)
        self.model.substrate.record_trial(
            result.sites, result.stimulation, result.load, neuron_credit,
            result.edge_flow, edge_credit, result.advertised_query,
            result.advertised_key, result.emission, reward,
            homeostasis_active=lifecycle_active,
        )
        result.frames.append(
            self.model.make_frame(
                result.sites, result.final_state[0], stimulation=result.stimulation,
                load=result.load, credit=neuron_credit, edge_flow=result.edge_flow,
                edge_credit=edge_credit, stage="feedback", step=self.task.sequence_length,
            )
        )
        events: list[dict[str, Any]] = []
        self.last_births = 0
        self.last_deaths = 0
        self.last_death_causes = {"starvation": 0, "overload": 0, "maintenance": 0}
        lifecycle_due = lifecycle_active and (
            completed - self.config.lifecycle_warmup_trials
        ) % self.config.lifecycle_interval == 0
        structure_due = structure_unlocked and (
            completed - self.config.structural_warmup_trials
        ) % self.config.structural_interval == 0
        if lifecycle_due or structure_due:
            update = self.model.substrate.structural_step(
                apply_lifecycle=lifecycle_due, apply_topology=structure_due
            )
            self._reset_mutation_optimizer_state(update.changed_edges, update.changed_sites)
            events = update.events
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
                current_sites, state_by_site[current_sites], stage="structural",
                step=self.task.sequence_length + 1, events=events,
            )
        )
        self.training_step = completed
        self.seen_examples += self.config.batch_size
        self.last_loss = float(task_loss.detach())
        self.last_batch_accuracy = accuracy
        self.last_reward = reward
        self.last_mean_attention_entropy = result.mean_attention_entropy
        self.accuracy_history.append(accuracy)
        self.loss_history.append(self.last_loss)
        self.reward_history.append(reward)
        rolling = self.rolling_accuracy
        if rolling >= self.best_rolling_accuracy + self.config.structure_plateau_delta:
            self.best_rolling_accuracy = rolling
            self.last_accuracy_improvement_step = completed
        self._remember(result, batch)
        if completed % self.config.evaluation_interval == 0:
            self.evaluate(self.config.evaluation_batches)

    def _terminal_frames(self, result: SequenceForward) -> list[SequenceFrame]:
        return [
            self.model.make_frame(
                result.sites, result.final_state[0], stage="feedback",
                step=self.task.sequence_length,
            ),
            self.model.make_frame(
                result.sites, result.final_state[0], stage="structural",
                step=self.task.sequence_length + 1,
            ),
        ]

    @torch.no_grad()
    def _remember(self, result: SequenceForward, batch: SequenceBatch) -> None:
        probabilities = result.logits[0].softmax(dim=1)
        self.last_trace = result.frames
        self.visual_cursor = 0
        self.events = []
        self.last_tokens = batch.tokens[0].detach().cpu()
        self.last_targets = batch.targets[0].detach().cpu()
        self.last_predictions = probabilities.argmax(dim=1).detach().cpu()
        self.last_confidences = probabilities.max(dim=1).values.detach().cpu()

    def _should_activate_lifecycle(self, completed: int) -> bool:
        if not self.config.lifecycle_enabled:
            self.lifecycle_reason = "disabled by configuration"
            return False
        if completed < self.config.lifecycle_warmup_trials:
            self.lifecycle_reason = "waiting for lifecycle warm-up"
            return False
        self.lifecycle_active = True
        self.lifecycle_reason = "energy pressure and turnover active"
        return True

    def _should_unlock_structure(self, completed: int, accuracy: float) -> bool:
        if self.structure_unlocked:
            return True
        if completed < self.config.structural_warmup_trials:
            self.structure_unlock_reason = "minimum learning warm-up"
            return False
        recent = list(self.accuracy_history) + [accuracy]
        rolling = sum(recent) / len(recent)
        if rolling >= self.config.structure_accuracy_threshold:
            self.structure_unlocked = True
            self.structure_unlock_reason = "accuracy competence reached"
        elif completed - self.last_accuracy_improvement_step >= self.config.structure_plateau_trials:
            self.structure_unlocked = True
            self.structure_unlock_reason = "learning plateau reached"
        else:
            self.structure_unlock_reason = "waiting for competence or plateau"
        return self.structure_unlocked

    @torch.no_grad()
    def _reset_mutation_optimizer_state(
        self, changed_edges: torch.Tensor, changed_sites: torch.Tensor
    ) -> None:
        for parameter, changed in (
            (self.model.substrate.synapse_weight, changed_edges),
            (self.model.substrate.genotype, changed_sites),
        ):
            state = self.optimizer.state.get(parameter, {})
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = state.get(key)
                if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                    if changed.ndim == value.ndim:
                        value.masked_fill_(changed, 0)
                    else:
                        value[changed] = 0

    @torch.no_grad()
    def evaluate(self, batches: int = 8) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for _ in range(max(1, min(50, batches))):
            batch = self._batch(self.config.batch_size, evaluation=True)
            logits = self.model(batch.tokens, capture_trace=False).logits
            correct += int((logits.argmax(dim=2)[batch.loss_mask] == batch.targets[batch.loss_mask]).sum())
            total += int(batch.loss_mask.sum())
        self.test_accuracy = correct / max(1, total)
        return self.test_accuracy

    @torch.no_grad()
    def _replay_current(self, *, events: list[dict[str, Any]] | None = None) -> None:
        self.model.eval()
        batch = SequenceBatch(
            self.last_tokens.unsqueeze(0).to(self.device),
            self.last_targets.unsqueeze(0).to(self.device),
            (self.last_targets >= 0).unsqueeze(0).to(self.device),
        )
        result = self.model(batch.tokens, capture_trace=True)
        terminal = self._terminal_frames(result)
        terminal[-1].events = list(events or [])
        result.frames.extend(terminal)
        self._remember(result, batch)

    def lifecycle_now(self) -> None:
        update = self.model.substrate.structural_step(
            apply_lifecycle=True, apply_topology=True
        )
        self._reset_mutation_optimizer_state(update.changed_edges, update.changed_sites)
        self.last_births = update.births
        self.last_deaths = update.deaths
        self.last_death_causes = {
            "starvation": 0, "overload": 0, "maintenance": 0, **update.death_causes
        }
        self.cumulative_births += update.births
        self.cumulative_deaths += update.deaths
        for cause, count in update.death_causes.items():
            self.cumulative_death_causes[cause] += count
        self._replay_current(events=update.events)

    def lesion(self, x: float, y: float, radius: float) -> int:
        count = self.model.substrate.lesion(x, y, radius)
        self._replay_current()
        return count

    @property
    def last_frame(self) -> SequenceFrame:
        return self.last_trace[self.visual_cursor]

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
    def lifecycle_warmup_remaining(self) -> int:
        return max(0, self.config.lifecycle_warmup_trials - self.training_step)

    @property
    def structural_warmup_remaining(self) -> int:
        return max(0, self.config.structural_warmup_trials - self.training_step)

    @property
    def learning_phase(self) -> str:
        return "structure" if self.structure_unlocked else "joint gradient learning"


__all__ = ["SequenceExperiment"]
