"""Reproducible short learning sweeps for sequence stepping stones."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from .sequence_config import sequence_config
from .sequence_cells import CELL_ARCHITECTURES
from .sequence_experiment import SequenceExperiment
from .token_compositional_grammar_task import (
    compositional_generation_audit,
    compositional_grammar_provenance,
    token_compositional_grammar_task,
)
from .token_context_task import token_context_task
from .token_grammar_task import token_grammar_task
from .token_memory_task import token_memory_task
from .token_pipeline_task import token_pipeline_task
from .token_routing_task import token_routing_task
from .token_settling_task import token_settling_task
from .token_settled_pipeline_task import token_settled_pipeline_task
from .token_stream_task import token_stream_task


PROFILES: dict[str, dict[str, Any]] = {
    "shallow32": {"message_steps": 2},
    "default32": {},
    "compact24": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0,
    },
    "compact24_no_broadcast": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0, "broadcast_gain": 0.0,
    },
    "compact24_no_global_memory": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0, "broadcast_gain": 0.0, "fast_weight_gain": 0.0,
    },
    "compact24_fast_weights": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0, "fast_weight_gain": 0.5,
    },
    "compact24_binding_owners": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0, "binding_memory_gain": 1.0,
        "binding_memory_temperature": 0.08,
    },
    "compact24_binding_tokens": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0, "binding_memory_gain": 1.0,
        "binding_memory_temperature": 0.08, "binding_token_values": 1,
    },
    "compact24_binding_sparse": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0, "binding_memory_gain": 1.0,
        "binding_memory_temperature": 0.05, "binding_token_values": 1,
        "binding_address_regularization": 0.02,
    },
    "token_route68": {},
    "token_settling68": {},
    "token_settled_pipeline68": {},
    "token_context68": {},
    "token_compositional_grammar68": {},
    "token_grammar68": {},
    "token_memory68": {},
    "token_pipeline68": {},
    "token_stream68": {},
}


TOKEN_CONTROL_PROFILES = {
    "token_routing": "token_route68",
    "token_settling": "token_settling68",
    "token_settled_pipeline": "token_settled_pipeline68",
    "token_context": "token_context68",
    "token_compositional_grammar": "token_compositional_grammar68",
    "token_grammar": "token_grammar68",
    "token_memory": "token_memory68",
    "token_pipeline": "token_pipeline68",
    "token_stream": "token_stream68",
}
POSITION_SIGNALS = ("learned", "none")


def _gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    """Return one finite L2 norm for the most recent optimizer gradient group."""

    squared = sum(
        float(parameter.grad.detach().float().square().sum())
        for parameter in parameters if parameter.grad is not None
    )
    return math.sqrt(squared)


def _scale_learning_rates(config: Any, scale: float) -> Any:
    """Scale every optimizer group together for a matched stability control."""

    if not 0.01 <= scale <= 1.0:
        raise ValueError("learning-rate scale must be between 0.01 and 1.0")
    return replace(
        config,
        learning_rate=config.learning_rate * scale,
        readout_learning_rate=config.readout_learning_rate * scale,
        synapse_learning_rate=config.synapse_learning_rate * scale,
    )


def _override_batch_size(config: Any, batch_size: int | None) -> Any:
    """Apply an explicit memory-bounded batch without changing other controls."""

    if batch_size is None:
        return config
    if not 1 <= batch_size <= 64:
        raise ValueError("batch size must be between 1 and 64")
    return replace(config, batch_size=batch_size)


def _configure_position_signal(
    position_identity: torch.nn.Embedding, signal: str,
) -> None:
    """Apply a declared absolute-clock intervention before the first update."""

    if signal not in POSITION_SIGNALS:
        raise ValueError(f"unknown position signal: {signal}")
    if signal == "none":
        with torch.no_grad():
            position_identity.weight.zero_()
        position_identity.weight.requires_grad_(False)


def _graph_ablation_summary(
    measurements: tuple[dict[str, Any], dict[str, Any], dict[str, Any],
                        dict[str, Any], dict[str, Any]],
) -> dict[str, float]:
    """Compress a matched graph audit into directional causal deltas."""

    reference, silenced, rotated, reassigned, broadcast_silenced = measurements
    summary = {
        "referenceAccuracy": float(reference["accuracy"]),
        "referenceLoss": float(reference["loss"]),
    }
    for prefix, condition in (
        ("silenced", silenced),
        ("sourceRotated", rotated),
        ("weightReassigned", reassigned),
        ("broadcastSilenced", broadcast_silenced),
    ):
        summary[f"{prefix}Accuracy"] = float(condition["accuracy"])
        summary[f"{prefix}Loss"] = float(condition["loss"])
        summary[f"{prefix}AccuracyDelta"] = (
            float(reference["accuracy"]) - float(condition["accuracy"])
        )
        summary[f"{prefix}LossDelta"] = (
            float(condition["loss"]) - float(reference["loss"])
        )
    return summary


def run_benchmark(
    task: str, profile: str, *, steps: int, seed: int, device: str,
    architecture: str = "gru",
    fixed_recall_pairs: int | None = None,
    message_steps: int | None = None,
    broadcast_gain: float | None = None,
    learning_rate_scale: float = 1.0,
    batch_size: int | None = None,
    amp_mode: str = "off",
    position_signal: str = "learned",
    autoregressive_feedback_probability: float = 0.0,
    autoregressive_feedback_warmup: int = 0,
    output_path: Path | None = None,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Train one controlled run and return checkpointed held-out metrics."""

    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    expected_profile = TOKEN_CONTROL_PROFILES.get(task)
    if expected_profile is not None and profile != expected_profile:
        raise ValueError(f"{task} requires the {expected_profile} profile")
    if expected_profile is None and profile in TOKEN_CONTROL_PROFILES.values():
        raise ValueError("token-control profiles require their matching task")
    if message_steps is not None and expected_profile is None:
        raise ValueError("message-step overrides apply only to token controls")
    if broadcast_gain is not None and expected_profile is None:
        raise ValueError("broadcast-gain overrides apply only to token controls")
    if broadcast_gain is not None and broadcast_gain < 0:
        raise ValueError("broadcast gain cannot be negative")
    if not 0 <= autoregressive_feedback_probability <= 1:
        raise ValueError("autoregressive feedback probability must be between zero and one")
    if autoregressive_feedback_warmup < 0:
        raise ValueError("autoregressive feedback warm-up cannot be negative")
    if (
        autoregressive_feedback_probability > 0
        and task != "token_compositional_grammar"
    ):
        raise ValueError("autoregressive feedback is limited to compositional grammar")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    overrides = dict(PROFILES[profile])
    if message_steps is not None:
        overrides["message_steps"] = message_steps
    if broadcast_gain is not None:
        overrides["broadcast_gain"] = broadcast_gain
    config = sequence_config(
        "tiny_stories" if expected_profile is not None else None,
        cell_architecture=architecture,
        lifecycle_enabled=0,
        structural_warmup_trials=max(steps + 1, 10_000),
        **overrides,
    )
    config = _scale_learning_rates(config, learning_rate_scale)
    config = _override_batch_size(config, batch_size)
    if fixed_recall_pairs is not None and task != "associative_recall":
        raise ValueError("fixed recall pairs apply only to associative recall")
    task_definition = (
        token_routing_task() if task == "token_routing"
        else token_context_task() if task == "token_context"
        else token_compositional_grammar_task()
        if task == "token_compositional_grammar"
        else token_grammar_task() if task == "token_grammar"
        else token_memory_task() if task == "token_memory"
        else token_pipeline_task() if task == "token_pipeline"
        else token_settling_task() if task == "token_settling"
        else token_settled_pipeline_task() if task == "token_settled_pipeline"
        else token_stream_task() if task == "token_stream"
        else task
    )
    experiment = SequenceExperiment(
        task_definition, config, seed=seed, device=device,
        amp_mode=amp_mode,
        recall_pair_count=fixed_recall_pairs,
        recall_pair_max=fixed_recall_pairs or 3,
    )
    _configure_position_signal(experiment.model.position_identity, position_signal)
    started = time.perf_counter()
    checkpoints: list[dict[str, Any]] = []
    final_graph_audit: dict[str, float] | None = None
    free_running_audit: dict[str, object] | None = None
    split_provenance = (
        compositional_grammar_provenance()
        if task == "token_compositional_grammar" else None
    )
    feedback_generator = torch.Generator().manual_seed(seed + 20_000)
    parameter_count = sum(parameter.numel() for parameter in experiment.model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in experiment.model.parameters()
        if parameter.requires_grad
    )
    initial_cuda_gib = (
        torch.cuda.memory_allocated(experiment.device) / 2**30
        if experiment.device.type == "cuda" else 0.0
    )
    if experiment.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(experiment.device)

    def result(status: str, completed_steps: int) -> dict[str, Any]:
        diagnostics = experiment.model.substrate.graph_diagnostics()
        return {
            "task": task, "profile": profile, "architecture": architecture,
            "intervention": (
                f"{config.message_steps} ticks · "
                f"broadcast {'on' if config.broadcast_gain > 0 else 'off'} · "
                f"position {position_signal} · feedback≤"
                f"{autoregressive_feedback_probability:g}"
                f"/{autoregressive_feedback_warmup} warm-up · "
                f"lr×{learning_rate_scale:g}"
                if expected_profile is not None else None
            ),
            "messageSteps": config.message_steps,
            "broadcastGain": config.broadcast_gain,
            "learningRateScale": learning_rate_scale,
            "positionSignal": position_signal,
            "autoregressiveFeedbackProbability": (
                autoregressive_feedback_probability
            ),
            "autoregressiveFeedbackWarmup": autoregressive_feedback_warmup,
            "batchSize": config.batch_size,
            "ampMode": experiment.amp_mode,
            "cudaAllocatorConfig": (
                os.environ.get("PYTORCH_ALLOC_CONF")
                or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            ),
            "outputCount": experiment.model.substrate.output_count,
            "sequenceLength": experiment.task.sequence_length,
            "dependencyTokens": (
                experiment.task.sequence_length - 1
                if task in {
                    "token_stream", "token_pipeline", "token_settling",
                    "token_settled_pipeline", "token_grammar",
                    "token_compositional_grammar",
                }
                else 1 if task in {"token_context", "token_memory"} else 0
            ),
            "chanceAccuracy": (
                1 / 8 if task == "token_routing"
                else 0.25 if task in {
                    "token_grammar", "token_compositional_grammar",
                }
                else 0.5
                if task in {
                    "token_context", "token_memory", "token_stream", "token_pipeline",
                    "token_settling",
                    "token_settled_pipeline",
                }
                else None
            ),
            "finalGraphAudit": final_graph_audit,
            "freeRunningAudit": free_running_audit,
            "splitProvenance": split_provenance,
            "recallMode": (
                "direct_mapping" if task == "token_routing" else
                "delayed_copy" if task == "token_memory" else
                "context_xor" if task == "token_context" else
                "context_stream" if task == "token_stream" else
                "context_pipeline" if task == "token_pipeline" else
                "context_settling" if task == "token_settling" else
                "settled_pipeline" if task == "token_settled_pipeline" else
                "autoregressive_grammar" if task == "token_grammar" else
                "held_out_rule_composition"
                if task == "token_compositional_grammar" else
                f"fixed_{fixed_recall_pairs}"
                if fixed_recall_pairs is not None else "adaptive"
            ),
            "seed": seed, "device": str(experiment.device), "steps": steps,
            "deterministic": deterministic,
            "completedSteps": completed_steps, "status": status,
            "seconds": round(time.perf_counter() - started, 2),
            "parameterCount": parameter_count,
            "trainableParameterCount": trainable_parameter_count,
            "cudaAllocatedGiB": round(initial_cuda_gib, 4),
            "peakCudaAllocatedGiB": round(
                torch.cuda.max_memory_allocated(experiment.device) / 2**30, 4
            ) if experiment.device.type == "cuda" else 0.0,
            "bindingDiagnostics": experiment.model.binding_memory_diagnostics(),
            "livingCells": int(experiment.model.substrate.occupied.sum()),
            "edgeCount": int(experiment.model.substrate.active_edge_mask.sum()),
            "minimumOutputHops": diagnostics.minimum_output_hops,
            "temporallyReachableOutputs": diagnostics.temporally_reachable_outputs,
            "contextReachableOutputs": sum(
                hops <= config.message_steps * experiment.task.sequence_length
                for hops in diagnostics.output_hops
            ),
            "checkpoints": list(checkpoints),
        }

    if output_path is not None:
        _write_result(output_path, result("running", 0))
    interval = max(1, min(20, steps))
    completed_steps = 0
    try:
        for update in range(1, steps + 1):
            feedback_probability = autoregressive_feedback_probability
            if autoregressive_feedback_warmup > 0:
                feedback_probability *= min(
                    1.0, update / autoregressive_feedback_warmup
                )
            experiment.train_updates(
                1,
                autoregressive_feedback_probability=feedback_probability,
                feedback_generator=feedback_generator,
            )
            completed_steps = update
            if update % interval == 0 or update == steps:
                held_out = experiment.evaluate_metrics(4)
                model = experiment.model
                checkpoints.append(
                    {
                        "update": update,
                        "trainingAccuracy": round(experiment.rolling_accuracy, 4),
                        "heldOutAccuracy": round(float(held_out["accuracy"]), 4),
                        "heldOutSlotAccuracy": [
                            round(float(accuracy), 4)
                            for accuracy in held_out.get("slotAccuracy", [])
                        ],
                        "heldOutPositionAccuracy": [
                            round(float(accuracy), 4)
                            for accuracy in held_out.get("positionAccuracy", [])
                        ],
                        "heldOutPositionIndices": list(
                            held_out.get("positionIndices", [])
                        ),
                        "heldOutPresentedValueRate": round(
                            float(held_out.get("presentedValueRate", 0.0)), 4
                        ),
                        "heldOutDistractorRate": round(
                            float(held_out.get("distractorRate", 0.0)), 4
                        ),
                        "heldOutAbsentValueRate": round(
                            float(held_out.get("absentValueRate", 0.0)), 4
                        ),
                        "loss": round(experiment.rolling_loss, 5),
                        "recallPairs": experiment.recall_pair_count,
                        "autoregressiveFeedbackProbability": round(
                            feedback_probability, 6
                        ),
                        "autoregressiveFeedbackFraction": round(
                            experiment.last_autoregressive_feedback_fraction, 6
                        ),
                        "gradientNorms": {
                            "tokenIdentity": round(
                                _gradient_norm([model.token_identity.weight]), 7
                            ),
                            "inputProjection": round(_gradient_norm(
                                list(model.input_value.parameters())
                                if model.input_value else []
                            ), 7),
                            "cellRule": round(
                                _gradient_norm(list(model.cell_rule.parameters())), 7
                            ),
                            "synapse": round(
                                _gradient_norm([model.substrate.synapse_weight]), 7
                            ),
                            "broadcast": round(_gradient_norm(
                                list(model.broadcast_key.parameters())
                                + list(model.broadcast_query.parameters())
                                + list(model.broadcast_value.parameters())
                                + [model.broadcast_gain]
                            ), 7),
                            "outputReadout": round(_gradient_norm(
                                list(model.output_bank_readout.parameters())
                            ), 7),
                        },
                    }
                )
                if output_path is not None:
                    _write_result(output_path, result("running", update))
    except Exception as error:
        if experiment.device.type == "cuda":
            torch.cuda.empty_cache()
        failed = result("failed", completed_steps)
        failed["failureType"] = type(error).__name__
        failed["failureMessage"] = str(error).splitlines()[0][:500]
        if output_path is not None:
            _write_result(output_path, failed)
        raise
    try:
        if task == "token_compositional_grammar":
            experiment.model.eval()

            @torch.no_grad()
            def predict_next(tokens: torch.Tensor) -> torch.Tensor:
                with torch.autocast(
                    device_type=experiment.device.type,
                    dtype=experiment.amp_dtype,
                    enabled=experiment.amp_dtype is not None,
                ):
                    generated = experiment.compute_model(
                        tokens.to(experiment.device), capture_trace=False,
                    )
                return generated.logits[:, -1].argmax(dim=1).cpu()

            free_running_audit = compositional_generation_audit(
                predict_next, batch_size=config.batch_size,
            )
        if expected_profile is not None:
            final_graph_audit = _graph_ablation_summary(
                experiment.evaluate_graph_ablation(4)
            )
    except Exception as error:
        if experiment.device.type == "cuda":
            torch.cuda.empty_cache()
        failed = result("failed", completed_steps)
        failed["failureType"] = type(error).__name__
        failed["failureMessage"] = str(error).splitlines()[0][:500]
        if output_path is not None:
            _write_result(output_path, failed)
        raise
    final = result("complete", steps)
    if output_path is not None:
        _write_result(output_path, final)
    return final


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one live benchmark artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=(
            "associative_recall", "tiny_language", "token_routing", "token_memory",
            "token_context", "token_stream", "token_pipeline", "token_settling",
            "token_settled_pipeline", "token_grammar",
            "token_compositional_grammar",
        ),
        required=True,
    )
    parser.add_argument("--profile", choices=tuple(PROFILES), default="default32")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--architecture", choices=CELL_ARCHITECTURES, default="gru")
    parser.add_argument("--fixed-recall-pairs", type=int, choices=(1, 2, 3))
    parser.add_argument("--message-steps", type=int, choices=range(1, 17))
    parser.add_argument("--broadcast-gain", type=float)
    parser.add_argument("--learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--amp", choices=("off", "bfloat16"), default="off")
    parser.add_argument(
        "--position-signal", choices=POSITION_SIGNALS, default="learned"
    )
    parser.add_argument(
        "--autoregressive-feedback-probability", type=float, default=0.0
    )
    parser.add_argument("--autoregressive-feedback-warmup", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    result = run_benchmark(
        args.task, args.profile, steps=max(1, args.steps), seed=args.seed,
        device=args.device, architecture=args.architecture,
        fixed_recall_pairs=args.fixed_recall_pairs,
        message_steps=args.message_steps,
        broadcast_gain=args.broadcast_gain,
        learning_rate_scale=args.learning_rate_scale,
        batch_size=args.batch_size,
        amp_mode=args.amp,
        position_signal=args.position_signal,
        autoregressive_feedback_probability=(
            args.autoregressive_feedback_probability
        ),
        autoregressive_feedback_warmup=args.autoregressive_feedback_warmup,
        output_path=args.output,
        deterministic=args.deterministic,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
