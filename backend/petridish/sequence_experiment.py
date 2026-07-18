"""Training lifecycle for persistent token-stream neural organisms."""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Callable

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from .config import resolve_device
from .graph_layout import sequence_layout
from .mnist_config import MnistModelConfig
from .sequence_config import sequence_config
from .sequence_model import (
    CellularSequenceModel, SequenceForward, SequenceFrame, SequenceRuntimeState,
)
from .sequence_tasks import (
    STREAM_MODES,
    SequenceBatch,
    SequenceTask,
    associative_recall_batch,
    resolve_sequence_task,
)


class SequenceExperiment:
    """Learn a synthetic sequence distribution in one persistent organism."""

    def __init__(
        self,
        task: str | SequenceTask,
        config: MnistModelConfig | None = None,
        *,
        seed: int = 1,
        device: str = "auto",
        amp_mode: str = "off",
        recall_pair_count: int | None = None,
        recall_pair_max: int = 3,
        stream_mode: str = "windowed",
        state_retention: float = 1.0,
        state_lanes: int = 1,
    ) -> None:
        self.task = resolve_sequence_task(task)
        self.experiment_name = self.task.key
        self.config = config or sequence_config(self.task.key)
        self.seed = seed
        if stream_mode not in STREAM_MODES:
            raise ValueError(f"unknown sequence stream mode: {stream_mode}")
        if stream_mode == "continuous" and self.task.training_stream is None:
            raise ValueError(f"task {self.task.key} has no contiguous training stream")
        if not 0 <= state_retention <= 1:
            raise ValueError("state retention must be between zero and one")
        if not 1 <= state_lanes <= 16:
            raise ValueError("state lanes must be between one and sixteen")
        self.stream_mode = stream_mode
        self.state_retention = state_retention
        self.state_lanes = state_lanes
        self.device = resolve_device(device)
        if amp_mode not in {"off", "bfloat16"}:
            raise ValueError("amp_mode must be 'off' or 'bfloat16'")
        if amp_mode != "off" and self.device.type != "cuda":
            raise ValueError("AMP is supported only on CUDA")
        self.amp_mode = amp_mode
        self.amp_dtype = torch.bfloat16 if amp_mode == "bfloat16" else None
        self.generator = torch.Generator().manual_seed(seed)
        self.eval_generator = torch.Generator().manual_seed(seed + 10_000)
        stream_positions = (
            self.task.initial_stream_positions(
                self.config.batch_size * state_lanes, self.generator
            ).reshape(state_lanes, self.config.batch_size)
            if stream_mode == "continuous" else None
        )
        self._training_stream_positions = (
            stream_positions[0] if stream_positions is not None and state_lanes == 1
            else stream_positions
        )
        self._training_runtime_state: SequenceRuntimeState | None = None
        self._training_runtime_bank: list[SequenceRuntimeState | None] = [
            None for _ in range(state_lanes)
        ]
        if self.task.key == "associative_recall":
            initial_pairs = 1 if recall_pair_count is None else recall_pair_count
            if initial_pairs not in {1, 2, 3} or recall_pair_max not in {1, 2, 3}:
                raise ValueError("recall pair count and maximum must be between one and three")
            if initial_pairs > recall_pair_max:
                raise ValueError("initial recall pairs cannot exceed the curriculum maximum")
            self.recall_pair_count = initial_pairs
            self.recall_pair_max = recall_pair_max
        else:
            self.recall_pair_count = 0
            self.recall_pair_max = 0
        self.stage_accuracy_history: deque[float] = deque(maxlen=24)
        layout = sequence_layout(self.task.key, len(self.task.vocabulary))
        self.model = CellularSequenceModel(
            self.config,
            layout=layout,
            vocab_size=len(self.task.vocabulary),
            max_length=max(12, self.task.sequence_length),
            seed=seed,
        ).to(self.device)
        self.compute_model: Callable[..., SequenceForward] = self.model
        synapse = self.model.substrate.synapse_weight
        shared = [parameter for parameter in self.model.shared_parameters()]
        readout_ids = {
            id(parameter) for parameter in self.model.output_bank_readout.parameters()
        } | {id(self.model.class_bias)}
        if self.model.logit_scale is not None:
            readout_ids.add(id(self.model.logit_scale))
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
        self.interactive_prompt = ""
        self.generated_text = ""
        self.next_token_prediction = ""
        self._interactive_state: SequenceRuntimeState | None = None
        self._interactive_next_logits: torch.Tensor | None = None
        self._interactive_token_ids: list[int] = []
        self.last_births = 0
        self.last_deaths = 0
        self.last_death_causes = {"starvation": 0, "excitotoxicity": 0, "maintenance": 0}
        self.last_stuns = 0
        self.last_recoveries = 0
        self.cumulative_stuns = 0
        self.cumulative_recoveries = 0
        self.cumulative_births = 0
        self.cumulative_deaths = 0
        self.last_grown_edges = 0
        self.last_pruned_edges = 0
        self.cumulative_grown_edges = 0
        self.cumulative_pruned_edges = 0
        self.cumulative_death_causes = {
            "starvation": 0, "excitotoxicity": 0, "maintenance": 0
        }
        self.last_synapse_update_ratio = 0.0
        self.last_mean_attention_entropy = 0.0
        self.lifecycle_active = False
        self.lifecycle_reason = (
            "waiting for lifecycle warm-up"
            if self.config.lifecycle_enabled else "disabled by configuration"
        )
        self.structure_unlocked = False
        self.structure_unlock_reason = (
            "disabled by configuration"
            if not self.config.structural_enabled else "minimum learning warm-up"
        )
        self.best_rolling_accuracy = 0.0
        self.last_accuracy_improvement_step = 0
        self._prime_preview()

    def enable_compile(self, mode: str = "default") -> None:
        """Compile only model forward computation; optimizer and topology stay eager."""

        if mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError(f"unsupported compile mode: {mode}")
        self.compute_model = torch.compile(self.model, mode=mode)

    def _batch(self, size: int, *, evaluation: bool = False) -> SequenceBatch:
        generator = self.eval_generator if evaluation else self.generator
        batch = (
            associative_recall_batch(size, generator, self.recall_pair_count)
            if self.task.key == "associative_recall"
            else self.task.batch(size, generator, evaluation=evaluation)
        )
        return SequenceBatch(
            batch.tokens.to(self.device), batch.targets.to(self.device), batch.loss_mask.to(self.device)
        )

    def _continuous_batch(
        self,
    ) -> tuple[SequenceBatch, torch.Tensor, int, SequenceRuntimeState | None]:
        """Read the next corpus window without advancing state until update success."""

        if self._training_stream_positions is None:
            raise RuntimeError("continuous stream positions were not initialized")
        lane = self.training_step % self.state_lanes
        positions = (
            self._training_stream_positions
            if self.state_lanes == 1 else self._training_stream_positions[lane]
        )
        batch, next_positions = self.task.stream_batch(positions)
        runtime_state = (
            self._training_runtime_state
            if self.state_lanes == 1 else self._training_runtime_bank[lane]
        )
        return SequenceBatch(
            batch.tokens.to(self.device), batch.targets.to(self.device),
            batch.loss_mask.to(self.device),
        ), next_positions, lane, runtime_state

    def _masked_loss_accuracy(
        self, logits: torch.Tensor, batch: SequenceBatch
    ) -> tuple[torch.Tensor, float]:
        selected = batch.loss_mask
        loss = F.cross_entropy(logits[selected].float(), batch.targets[selected])
        metric_mask = selected
        if self.task.key == "tiny_language":
            metric_mask = torch.zeros_like(selected)
            metric_mask[:, 2:4] = True
        accuracy = float(
            (logits.argmax(dim=2)[metric_mask] == batch.targets[metric_mask]).float().mean()
        )
        return loss, accuracy

    @torch.no_grad()
    def _prime_preview(self) -> None:
        self.model.eval()
        batch = self._batch(1)
        if self.task.encode is not None and batch.tokens.shape[1] > 4:
            batch = SequenceBatch(
                batch.tokens[:, -4:], batch.targets[:, -4:], batch.loss_mask[:, -4:]
            )
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

    def _train_trial(
        self,
        *,
        capture_trace: bool = True,
        auto_evaluate: bool = True,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self.model.train()
        if self.stream_mode == "continuous":
            batch, next_stream_positions, state_lane, runtime_state = (
                self._continuous_batch()
            )
        else:
            batch = self._batch(self.config.batch_size)
            next_stream_positions = None
            state_lane = 0
            runtime_state = None
        if progress_callback is not None:
            self._begin_visible_trial(batch)

        def observe_frame(frame: SequenceFrame, logits: torch.Tensor) -> None:
            probabilities = logits[0].softmax(dim=0)
            position = frame.token_position
            self.last_predictions[position] = int(probabilities.argmax().cpu())
            self.last_confidences[position] = float(probabilities.max().cpu())
            self.last_trace = [frame]
            self.visual_cursor = 0
            self.events = [dict(event, tick=self.tick) for event in frame.events]
            progress_callback("forward", position + 1, batch.tokens.shape[1])

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_dtype is not None,
        ):
            result = self.compute_model(
                batch.tokens,
                capture_trace=capture_trace,
                frame_callback=observe_frame if progress_callback is not None else None,
                runtime_state=runtime_state,
            )
        task_loss, accuracy = self._masked_loss_accuracy(result.logits, batch)
        loss = task_loss + self.model.regularization()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite loss before backward")
        gradient_hooks: list[torch.utils.hooks.RemovableHandle] = []
        if progress_callback is not None:
            backward_total = max(1, len(result.retained_states))
            backward_progress = 0
            progress_callback("backward", 0, backward_total)

            def observe_gradient(
                position: int, state: torch.Tensor
            ) -> Callable[[torch.Tensor], torch.Tensor]:
                def hook(gradient: torch.Tensor) -> torch.Tensor:
                    nonlocal backward_progress
                    backward_progress += 1
                    credit = -(gradient.detach() * state.detach()).mean(dim=(0, 2))
                    forward_frame = result.frames[position]
                    frame = self.model.make_frame(
                        result.sites,
                        state[0],
                        stimulation=forward_frame.stimulation,
                        load=forward_frame.load,
                        credit=credit,
                        input_signal=forward_frame.input_signal,
                        edge_flow=forward_frame.graph.flow,
                        stage="feedback",
                        step=batch.tokens.shape[1] + backward_progress - 1,
                        token_position=position,
                    )
                    self.last_trace = [frame]
                    self.visual_cursor = 0
                    progress_callback("backward", backward_progress, backward_total)
                    return gradient

                return hook

            gradient_hooks = [
                state.register_hook(observe_gradient(position, state))
                for position, state in enumerate(result.retained_states)
            ]
        try:
            loss.backward()
        finally:
            for handle in gradient_hooks:
                handle.remove()
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
        if progress_callback is not None:
            progress_callback("optimizer", 0, 1)
        clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        with torch.no_grad():
            delta = (self.model.substrate.synapse_weight - before).abs()
            denominator = before.abs().clamp_min(self.config.initial_weight_scale * 0.1)
            self.last_synapse_update_ratio = (
                float((delta[active] / denominator[active]).mean()) if active.any() else 0.0
            )
        completed = self.training_step + 1
        uniform_loss = math.log(len(self.task.vocabulary))
        reward = max(
            -1.0,
            min(1.0, (uniform_loss - float(task_loss.detach())) / uniform_loss),
        )
        lifecycle_active = self._should_activate_lifecycle(completed)
        structure_unlocked = self._should_unlock_structure(completed, accuracy)
        if progress_callback is not None:
            progress_callback("credit", 0, 1)
        homeostasis_events = self.model.substrate.record_trial(
            result.sites, result.stimulation, result.load, neuron_credit,
            result.edge_flow, edge_credit, result.advertised_query,
            result.advertised_key, result.emission, reward,
            homeostasis_active=lifecycle_active,
        )
        if capture_trace:
            feedback_frame = self.model.make_frame(
                result.sites, result.final_state[0], stimulation=result.stimulation,
                load=result.load, credit=neuron_credit, edge_flow=result.edge_flow,
                edge_credit=edge_credit, stage="feedback", step=self.task.sequence_length,
            )
            result.frames.append(feedback_frame)
            if progress_callback is not None:
                self.last_trace = [feedback_frame]
                self.visual_cursor = 0
                progress_callback("credit", 1, 1)
        events: list[dict[str, Any]] = list(homeostasis_events)
        self.last_stuns = sum(event["type"] == "stunned" for event in events)
        self.last_recoveries = 0
        self.cumulative_stuns += self.last_stuns
        self.last_births = 0
        self.last_deaths = 0
        self.last_grown_edges = 0
        self.last_pruned_edges = 0
        self.last_death_causes = {"starvation": 0, "excitotoxicity": 0, "maintenance": 0}
        lifecycle_due = lifecycle_active and (
            completed - self.config.lifecycle_warmup_trials
        ) % self.config.lifecycle_interval == 0
        structure_due = structure_unlocked and (
            completed - self.config.structural_warmup_trials
        ) % self.config.structural_interval == 0
        if progress_callback is not None:
            progress_callback("lifecycle", 0, 1)
        if lifecycle_due or structure_due:
            update = self.model.substrate.structural_step(
                apply_lifecycle=lifecycle_due, apply_topology=structure_due
            )
            self._reset_mutation_optimizer_state(update.changed_edges, update.changed_sites)
            events.extend(update.events)
            self.last_recoveries = update.recoveries
            self.cumulative_recoveries += update.recoveries
            self.last_births = update.births
            self.last_deaths = update.deaths
            self.last_grown_edges = update.grown_edges
            self.last_pruned_edges = update.pruned_edges
            self.cumulative_grown_edges += update.grown_edges
            self.cumulative_pruned_edges += update.pruned_edges
            self.last_death_causes.update(update.death_causes)
            self.cumulative_births += update.births
            self.cumulative_deaths += update.deaths
            for cause, count in update.death_causes.items():
                self.cumulative_death_causes[cause] += count
        if self.stream_mode == "continuous":
            assert next_stream_positions is not None
            if self.state_lanes == 1:
                self._training_stream_positions = next_stream_positions
            else:
                self._training_stream_positions[state_lane] = next_stream_positions
            carried_state = self.model.relax_runtime_state(
                result.runtime_state, self.state_retention
            )
            self._training_runtime_state = carried_state
            self._training_runtime_bank[state_lane] = carried_state
        if capture_trace:
            current_sites = self.model.substrate.living_sites
            state_by_site = torch.zeros(
                self.config.site_count, self.config.hidden_channels, device=self.device
            )
            state_by_site[result.sites] = result.final_state[0].detach()
            structural_frame = self.model.make_frame(
                current_sites, state_by_site[current_sites], stage="structural",
                step=self.task.sequence_length + 1, events=events,
            )
            result.frames.append(structural_frame)
            if progress_callback is not None:
                self.last_trace = [structural_frame]
                self.visual_cursor = 0
                self.events = [dict(event, tick=self.tick) for event in events]
                progress_callback("lifecycle", 1, 1)
        self.training_step = completed
        self.seen_examples += self.config.batch_size
        self.last_loss = float(task_loss.detach())
        self.last_batch_accuracy = accuracy
        self.last_reward = reward
        self.last_mean_attention_entropy = result.mean_attention_entropy
        self.accuracy_history.append(accuracy)
        self.stage_accuracy_history.append(accuracy)
        self.loss_history.append(self.last_loss)
        self.reward_history.append(reward)
        rolling = self.rolling_accuracy
        if rolling >= self.best_rolling_accuracy + self.config.structure_plateau_delta:
            self.best_rolling_accuracy = rolling
            self.last_accuracy_improvement_step = completed
        if capture_trace:
            self._remember(
                result, batch, replace_trace=True,
                cursor_at_end=progress_callback is not None,
            )
        self._maybe_advance_recall_curriculum()
        if auto_evaluate and completed % self.config.evaluation_interval == 0:
            self.evaluate(
                self.config.evaluation_batches,
                progress_callback=progress_callback,
            )

    def train_visual_update(
        self, progress_callback: Callable[[str, int, int], None]
    ) -> None:
        """Run one optimizer update while exposing measured intermediate states."""

        self.tick += 1
        self._train_trial(progress_callback=progress_callback)

    def train_updates(self, count: int = 1) -> None:
        """Run optimizer updates directly without constructing viewer traces."""

        for _ in range(max(1, count)):
            self.tick += 1
            self._train_trial(capture_trace=False, auto_evaluate=False)

    def _begin_visible_trial(self, batch: SequenceBatch) -> None:
        """Align visible tokens before streamed forward frames arrive."""

        length = batch.tokens.shape[1]
        self.last_tokens = batch.tokens[0].detach().cpu()
        self.last_targets = batch.targets[0].detach().cpu()
        self.last_predictions = torch.full((length,), -1, dtype=torch.long)
        self.last_confidences = torch.zeros(length)
        self.events = []

    @torch.no_grad()
    def refresh_visual_trace(self) -> None:
        """Rebuild one authoritative trace after trace-free training."""

        self._replay_current()

    def _terminal_frames(self, result: SequenceForward) -> list[SequenceFrame]:
        length = result.logits.shape[1]
        return [
            self.model.make_frame(
                result.sites, result.final_state[0], stage="feedback",
                step=length,
            ),
            self.model.make_frame(
                result.sites, result.final_state[0], stage="structural",
                step=length + 1,
            ),
        ]

    @torch.no_grad()
    def _remember(
        self,
        result: SequenceForward,
        batch: SequenceBatch,
        *,
        replace_trace: bool = True,
        cursor_at_end: bool = False,
    ) -> None:
        probabilities = result.logits[0].softmax(dim=1)
        if replace_trace:
            self.last_trace = result.frames
            self.visual_cursor = len(result.frames) - 1 if cursor_at_end else 0
            self.events = []
        self.last_tokens = batch.tokens[0].detach().cpu()
        self.last_targets = batch.targets[0].detach().cpu()
        self.last_predictions = probabilities.argmax(dim=1).detach().cpu()
        self.last_confidences = probabilities.max(dim=1).values.detach().cpu()
        self.next_token_prediction = self.task.vocabulary[int(self.last_predictions[-1])]

    @torch.no_grad()
    def set_prompt(self, text: str) -> None:
        """Install an interactive corpus prompt and replay it without training."""

        if self.task.encode is None or self.task.decode is None:
            raise ValueError("interactive prompting is available only for corpus tasks")
        self.interactive_prompt = text
        self.generated_text = ""
        self._interactive_state = None
        self._interactive_next_logits = None
        self._interactive_token_ids = []
        self._preview_text(text)

    @torch.no_grad()
    def generate_token(self) -> str:
        """Sample one next token, append it, and make the new context visible."""

        if self.task.encode is None or self.task.decode is None:
            raise ValueError("token generation is available only for corpus tasks")
        if self._interactive_next_logits is None or self._interactive_state is None:
            self._preview_text(self.interactive_prompt or "Once upon a time")
        assert self._interactive_next_logits is not None
        assert self._interactive_state is not None
        self.model.eval()
        probabilities = (self._interactive_next_logits / 0.85).softmax(dim=0).cpu()
        token = int(torch.multinomial(probabilities, 1, generator=self.generator))
        generated = self.task.decode([token])
        self.generated_text += generated
        self._interactive_token_ids.append(token)
        next_input = torch.tensor([[token]], device=self.device)
        result = self.model(
            next_input, capture_trace=True, runtime_state=self._interactive_state
        )
        self._interactive_state = result.runtime_state
        self._interactive_next_logits = result.logits[0, -1]
        window = self._interactive_token_ids[-self.task.sequence_length :]
        self.last_tokens = torch.tensor(window, dtype=torch.long)
        self.last_targets = torch.full_like(self.last_tokens, -100)
        self.last_predictions = torch.full_like(self.last_tokens, -1)
        self.last_confidences = torch.zeros(len(window))
        next_probabilities = self._interactive_next_logits.softmax(dim=0)
        self.last_predictions[-1] = int(next_probabilities.argmax().cpu())
        self.last_confidences[-1] = float(next_probabilities.max().cpu())
        for frame in result.frames:
            frame.token_position = len(window) - 1
            frame.step = len(window) - 1
        result.frames.extend(self._terminal_frames(result))
        self.last_trace = result.frames
        self.visual_cursor = max(0, len(result.frames) - 1)
        self.next_token_prediction = self.task.vocabulary[int(self.last_predictions[-1])]
        return generated

    @torch.no_grad()
    def greedy_completion(
        self, prompt: str, *, max_tokens: int = 16
    ) -> tuple[str, list[int]]:
        """Generate a deterministic diagnostic without mutating interactive state."""

        if self.task.encode is None or self.task.decode is None:
            raise ValueError("generation diagnostics are available only for corpus tasks")
        token_ids = self.task.encode(prompt or "\n")[-self.task.sequence_length :]
        tokens = torch.tensor(token_ids, device=self.device).unsqueeze(0)
        was_training = self.model.training
        self.model.eval()
        try:
            result = self.model(tokens, capture_trace=False)
            runtime_state = result.runtime_state
            next_logits = result.logits[0, -1]
            generated_ids: list[int] = []
            for _ in range(max(1, max_tokens)):
                token = int(next_logits.argmax())
                generated_ids.append(token)
                next_input = torch.tensor([[token]], device=self.device)
                result = self.model(
                    next_input, capture_trace=False, runtime_state=runtime_state
                )
                runtime_state = result.runtime_state
                next_logits = result.logits[0, -1]
        finally:
            self.model.train(was_training)
        return self.task.decode(generated_ids), generated_ids

    @torch.no_grad()
    def _preview_text(self, text: str) -> None:
        assert self.task.encode is not None
        token_ids = self.task.encode(text or "\n")[-self.task.sequence_length :]
        tokens = torch.tensor(token_ids, device=self.device).unsqueeze(0)
        targets = torch.full_like(tokens, -100)
        batch = SequenceBatch(tokens, targets, torch.zeros_like(tokens, dtype=torch.bool))
        self.model.eval()
        result = self.model(tokens, capture_trace=True)
        self._interactive_state = result.runtime_state
        self._interactive_next_logits = result.logits[0, -1]
        self._interactive_token_ids = list(token_ids)
        self._remember(result, batch)

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
        if not self.config.structural_enabled:
            self.structure_unlocked = False
            self.structure_unlock_reason = "disabled by configuration"
            return False
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

    def _maybe_advance_recall_curriculum(self) -> None:
        """Increase binding count only after the current memory task is reliable."""

        if (
            self.task.key != "associative_recall"
            or self.recall_pair_count >= self.recall_pair_max
        ):
            return
        if len(self.stage_accuracy_history) < self.stage_accuracy_history.maxlen:
            return
        if sum(self.stage_accuracy_history) / len(self.stage_accuracy_history) < 0.90:
            return
        self.recall_pair_count += 1
        self.stage_accuracy_history.clear()

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
    def evaluate(
        self,
        batches: int = 8,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> float:
        """Evaluate held-out accuracy while preserving the historical API."""

        return self.evaluate_metrics(
            batches, progress_callback=progress_callback
        )["accuracy"]

    @torch.no_grad()
    def evaluate_metrics(
        self,
        batches: int = 8,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
        carry_state: bool | None = None,
        state_horizon_windows: int | None = None,
        initial_runtime_state: SequenceRuntimeState | None = None,
    ) -> dict[str, Any]:
        """Return held-out loss and accuracy without mutating training state."""

        if carry_state is not None and self.stream_mode != "continuous":
            raise ValueError("state-carry evaluation requires a continuous corpus stream")
        if initial_runtime_state is not None and self.stream_mode != "continuous":
            raise ValueError("checkpoint-state evaluation requires continuous mode")
        if initial_runtime_state is not None and carry_state is False:
            raise ValueError("cold-state evaluation cannot receive checkpoint state")
        if state_horizon_windows is not None:
            if self.stream_mode != "continuous":
                raise ValueError("state horizon requires a continuous corpus stream")
            if state_horizon_windows < 1:
                raise ValueError("state horizon must be at least one window")
        state_carry = self.stream_mode == "continuous" and carry_state is not False

        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        loss_items = 0
        position_correct = [0] * self.task.sequence_length
        position_total = [0] * self.task.sequence_length
        slot_correct = [0] * self.recall_pair_count
        slot_total = [0] * self.recall_pair_count
        presented_value_predictions = 0
        distractor_predictions = 0
        absent_value_predictions = 0
        batch_count = max(1, min(50, batches))
        stream_positions = (
            self.task.initial_stream_positions(
                self.config.batch_size, self.eval_generator, evaluation=True
            )
            if self.stream_mode == "continuous" else None
        )
        runtime_state = (
            initial_runtime_state.cloned_detached()
            if initial_runtime_state is not None else None
        )
        initial_state_tokens = runtime_state.position if runtime_state is not None else 0
        for index in range(batch_count):
            if stream_positions is None:
                batch = self._batch(self.config.batch_size, evaluation=True)
            else:
                cpu_batch, stream_positions = self.task.stream_batch(
                    stream_positions, evaluation=True
                )
                batch = SequenceBatch(
                    cpu_batch.tokens.to(self.device),
                    cpu_batch.targets.to(self.device),
                    cpu_batch.loss_mask.to(self.device),
                )
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                evaluation_result = self.compute_model(
                    batch.tokens, capture_trace=False, runtime_state=runtime_state
                )
                logits = evaluation_result.logits
                runtime_state = (
                    self.model.relax_runtime_state(
                        evaluation_result.runtime_state, self.state_retention
                    )
                    if (
                        stream_positions is not None
                        and state_carry
                        and (
                            state_horizon_windows is None
                            or (index + 1) % state_horizon_windows != 0
                        )
                    )
                    else None
                )
            selected_logits = logits[batch.loss_mask].float()
            selected_targets = batch.targets[batch.loss_mask]
            loss_sum += float(
                F.cross_entropy(selected_logits, selected_targets, reduction="sum")
            )
            loss_items += int(selected_targets.numel())
            metric_mask = batch.loss_mask
            if self.task.key == "tiny_language":
                metric_mask = torch.zeros_like(batch.loss_mask)
                metric_mask[:, 2:4] = True
            predictions = logits.argmax(dim=2)
            correct += int((predictions[metric_mask] == batch.targets[metric_mask]).sum())
            total += int(metric_mask.sum())
            for position in range(self.task.sequence_length):
                selected_rows = metric_mask[:, position]
                position_total[position] += int(selected_rows.sum())
                position_correct[position] += int(
                    (
                        predictions[selected_rows, position]
                        == batch.targets[selected_rows, position]
                    ).sum()
                )
            if self.task.key == "associative_recall":
                keys = batch.tokens[:, : self.recall_pair_count * 2 : 2]
                query_slots = (keys == batch.tokens[:, -1].unsqueeze(1)).long().argmax(dim=1)
                predictions = logits[:, -1].argmax(dim=1)
                row_correct = predictions == batch.targets[:, -1]
                values = batch.tokens[:, 1 : self.recall_pair_count * 2 : 2]
                predicts_presented_value = (
                    predictions.unsqueeze(1) == values
                ).any(dim=1)
                presented_value_predictions += int(predicts_presented_value.sum())
                distractor_predictions += int(
                    (predicts_presented_value & ~row_correct).sum()
                )
                absent_value_predictions += int((~predicts_presented_value).sum())
                for slot in range(self.recall_pair_count):
                    selected_rows = query_slots == slot
                    slot_total[slot] += int(selected_rows.sum())
                    slot_correct[slot] += int(row_correct[selected_rows].sum())
            if progress_callback is not None:
                progress_callback("evaluation", index + 1, batch_count)
        self.test_accuracy = correct / max(1, total)
        metrics: dict[str, Any] = {
            "loss": loss_sum / max(1, loss_items),
            "accuracy": self.test_accuracy,
            "streamMode": self.stream_mode,
            "stateRetention": self.state_retention,
            "stateLanes": self.state_lanes,
            "stateCarry": state_carry,
            "initialStateTokens": initial_state_tokens,
            "stateHorizonWindows": state_horizon_windows,
            "positionIndices": [
                position
                for position, count in enumerate(position_total)
                if count > 0
            ],
            "positionAccuracy": [
                position_correct[position] / position_total[position]
                for position in range(self.task.sequence_length)
                if position_total[position] > 0
            ],
        }
        if self.task.key == "associative_recall":
            metrics["slotAccuracy"] = [
                slot_correct[index] / max(1, slot_total[index])
                for index in range(self.recall_pair_count)
            ]
            metrics["presentedValueRate"] = presented_value_predictions / max(1, total)
            metrics["distractorRate"] = distractor_predictions / max(1, total)
            metrics["absentValueRate"] = absent_value_predictions / max(1, total)
        return metrics

    @torch.no_grad()
    def evaluate_state_ablation(
        self, batches: int = 8
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compare checkpoint and cold state on identical validation tokens."""

        if self.stream_mode != "continuous":
            raise ValueError("state ablation requires continuous training mode")
        before = self.eval_generator.get_state().clone()
        carried = self.evaluate_metrics(
            batches,
            carry_state=True,
            initial_runtime_state=self._training_runtime_state,
        )
        after = self.eval_generator.get_state().clone()
        try:
            self.eval_generator.set_state(before)
            cold = self.evaluate_metrics(batches, carry_state=False)
        finally:
            self.eval_generator.set_state(after)
            self.test_accuracy = float(carried["accuracy"])
        return carried, cold

    @torch.no_grad()
    def evaluate_graph_ablation(
        self, batches: int = 8
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Measure causal graph value on identical tokens without altering the organism."""

        substrate = self.model.substrate
        before_rng = self.eval_generator.get_state().clone()
        original_sources = substrate.dendrite_source.clone()
        original_weights = substrate.synapse_weight.detach().clone()
        active = substrate.active_edge_mask.clone()
        checkpoint_state = self._training_runtime_state
        reference = self.evaluate_metrics(
            batches, initial_runtime_state=checkpoint_state
        )
        after_rng = self.eval_generator.get_state().clone()
        try:
            self.eval_generator.set_state(before_rng)
            substrate.synapse_weight[active] = 0
            silenced = self.evaluate_metrics(
                batches, initial_runtime_state=checkpoint_state
            )

            substrate.synapse_weight.copy_(original_weights)
            self.eval_generator.set_state(before_rng)
            active_sources = original_sources[active]
            if active_sources.numel() > 1:
                substrate.dendrite_source[active] = active_sources.roll(1)
            substrate._diagnostic_cache = None
            source_rotated = self.evaluate_metrics(
                batches, initial_runtime_state=checkpoint_state
            )
        finally:
            substrate.dendrite_source.copy_(original_sources)
            substrate.synapse_weight.copy_(original_weights)
            substrate._diagnostic_cache = None
            self.eval_generator.set_state(after_rng)
            self.test_accuracy = float(reference["accuracy"])
        return reference, silenced, source_rotated

    @torch.no_grad()
    def evaluate_state_horizons(
        self,
        batches: int = 16,
        horizons: tuple[int, ...] = (1, 2, 4, 8, 16),
    ) -> list[dict[str, float | int]]:
        """Measure useful electrical-memory lifetime on one identical token stream."""

        if self.stream_mode != "continuous":
            raise ValueError("state horizon requires continuous training mode")
        bounded_batches = max(1, min(50, batches))
        selected = tuple(sorted({min(bounded_batches, max(1, h)) for h in horizons}))
        before = self.eval_generator.get_state().clone()
        after: torch.Tensor | None = None
        results: list[dict[str, float | int]] = []
        try:
            for horizon in selected:
                self.eval_generator.set_state(before)
                metrics = self.evaluate_metrics(
                    bounded_batches,
                    carry_state=True,
                    state_horizon_windows=horizon,
                    initial_runtime_state=self._training_runtime_state,
                )
                if after is None:
                    after = self.eval_generator.get_state().clone()
                results.append(
                    {
                        "windows": horizon,
                        "tokens": horizon * self.task.sequence_length,
                        "loss": float(metrics["loss"]),
                        "accuracy": float(metrics["accuracy"]),
                    }
                )
        finally:
            self.eval_generator.set_state(after if after is not None else before)
            if results:
                self.test_accuracy = float(results[-1]["accuracy"])
        return results

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
        self.last_grown_edges = update.grown_edges
        self.last_pruned_edges = update.pruned_edges
        self.cumulative_grown_edges += update.grown_edges
        self.cumulative_pruned_edges += update.pruned_edges
        self.last_death_causes = {
            "starvation": 0, "excitotoxicity": 0, "maintenance": 0, **update.death_causes
        }
        self.last_recoveries = update.recoveries
        self.cumulative_recoveries += update.recoveries
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
        if not self.config.structural_enabled:
            return "fixed graph learning"
        return "structure" if self.structure_unlocked else "joint gradient learning"


__all__ = ["SequenceExperiment"]
