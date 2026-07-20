"""Autoregressive grammar with rule compositions withheld from training."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .sequence_tasks import SequenceBatch, SequenceTask


DIGIT_COUNT = 4
OPERATOR_COUNT = 2
OFFSET_COUNT = 4
OPERATOR_OFFSET = 0
OFFSET_OFFSET = OPERATOR_OFFSET + OPERATOR_COUNT
DIGIT_OFFSET = OFFSET_OFFSET + OFFSET_COUNT
SEQUENCE_LENGTH = 12
CONTINUATION_LENGTH = 9

TRAIN_RULE_PAIRS = ((0, 0), (0, 2), (0, 3), (1, 0), (1, 1), (1, 3))
HELD_OUT_RULE_PAIRS = ((0, 1), (1, 2))

VOCABULARY = (
    "sum", "difference", "offset-0", "offset-1", "offset-2", "offset-3",
    "digit-0", "digit-1", "digit-2", "digit-3",
)


def _states(rule_pairs: tuple[tuple[int, int], ...]) -> torch.Tensor:
    """Enumerate every seed for a declared set of operator/offset pairs."""

    return torch.tensor([
        (operator, offset, first, second)
        for operator, offset in rule_pairs
        for first in range(DIGIT_COUNT)
        for second in range(DIGIT_COUNT)
    ], dtype=torch.long)


def _sample_states(
    states: torch.Tensor, batch_size: int, generator: torch.Generator
) -> torch.Tensor:
    """Draw without replacement inside shuffled split-local cycles."""

    cycles = [
        states[torch.randperm(states.shape[0], generator=generator)]
        for _ in range((batch_size + states.shape[0] - 1) // states.shape[0])
    ]
    return torch.cat(cycles, dim=0)[:batch_size]


def _stream(states: torch.Tensor) -> torch.Tensor:
    """Generate seed digits followed by the selected modular recurrence."""

    operator, offset, first, second = states.unbind(dim=1)
    digits = [first, second]
    sign = torch.where(operator == 0, 1, -1)
    for _ in range(CONTINUATION_LENGTH):
        digits.append((digits[-2] + sign * digits[-1] + offset) % DIGIT_COUNT)
    return torch.stack(digits, dim=1)


def _batch(states: torch.Tensor) -> SequenceBatch:
    """Encode one state set as rule tokens plus aligned next-token targets."""

    stream = _stream(states)
    operator = states[:, 0] + OPERATOR_OFFSET
    offset = states[:, 1] + OFFSET_OFFSET
    tokens = torch.cat((
        operator.unsqueeze(1), offset.unsqueeze(1), stream[:, :10] + DIGIT_OFFSET,
    ), dim=1)
    targets = torch.full_like(tokens, -100)
    targets[:, 3:] = stream[:, 2:11] + DIGIT_OFFSET
    return SequenceBatch(tokens, targets, targets >= 0)


def compositional_grammar_cases(
    *, evaluation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every prompt and continuation in one disjoint split."""

    states = _states(HELD_OUT_RULE_PAIRS if evaluation else TRAIN_RULE_PAIRS)
    batch = _batch(states)
    return batch.tokens[:, :4], batch.targets[:, 3:]


def compositional_grammar_provenance() -> dict[str, object]:
    """Describe the exact state split persisted in benchmark artifacts."""

    def names(pairs: tuple[tuple[int, int], ...]) -> list[str]:
        return [f"{VOCABULARY[operator]}+offset-{offset}" for operator, offset in pairs]

    training = _states(TRAIN_RULE_PAIRS)
    held_out = _states(HELD_OUT_RULE_PAIRS)
    return {
        "kind": "held_out_rule_pair_composition",
        "trainingRulePairs": names(TRAIN_RULE_PAIRS),
        "heldOutRulePairs": names(HELD_OUT_RULE_PAIRS),
        "trainingStateCount": int(training.shape[0]),
        "heldOutStateCount": int(held_out.shape[0]),
        "stateOverlap": 0,
    }


def compositional_generation_audit(
    predict_next: Callable[[torch.Tensor], torch.Tensor], *, batch_size: int,
) -> dict[str, object]:
    """Free-run every withheld composition through a black-box next-token model."""

    prompts, expected = compositional_grammar_cases(evaluation=True)
    generated_batches: list[torch.Tensor] = []
    for start in range(0, prompts.shape[0], batch_size):
        prefix = prompts[start : start + batch_size].clone()
        generated: list[torch.Tensor] = []
        for _ in range(expected.shape[1]):
            next_token = predict_next(prefix).detach().to(device="cpu", dtype=torch.long)
            if next_token.shape != (prefix.shape[0],):
                raise ValueError("next-token predictor returned an invalid batch shape")
            generated.append(next_token)
            prefix = torch.cat((prefix, next_token.unsqueeze(1)), dim=1)
        generated_batches.append(torch.stack(generated, dim=1))
    predicted = torch.cat(generated_batches, dim=0)
    correct = predicted == expected
    valid_digit = (predicted >= DIGIT_OFFSET) & (
        predicted < DIGIT_OFFSET + DIGIT_COUNT
    )
    first = 0
    return {
        "split": "held_out_rule_pairs",
        "cases": int(expected.shape[0]),
        "generatedTokens": int(expected.numel()),
        "tokenAccuracy": float(correct.float().mean()),
        "sequenceAccuracy": float(correct.all(dim=1).float().mean()),
        "invalidTokenRate": float((~valid_digit).float().mean()),
        "positionAccuracy": [float(value) for value in correct.float().mean(dim=0)],
        "positionIndices": list(range(expected.shape[1])),
        "examplePrompt": [VOCABULARY[token] for token in prompts[first].tolist()],
        "exampleExpected": [VOCABULARY[token] for token in expected[first].tolist()],
        "exampleGenerated": [VOCABULARY[token] for token in predicted[first].tolist()],
    }


def token_compositional_grammar_task() -> SequenceTask:
    """Return a grammar whose validation rule pairs never occur in training."""

    training_states = _states(TRAIN_RULE_PAIRS)
    evaluation_states = _states(HELD_OUT_RULE_PAIRS)

    def generate(batch_size: int, generator: torch.Generator) -> SequenceBatch:
        return _batch(_sample_states(training_states, batch_size, generator))

    def evaluate(batch_size: int, generator: torch.Generator) -> SequenceBatch:
        return _batch(_sample_states(evaluation_states, batch_size, generator))

    return SequenceTask(
        key="tiny_stories",
        title="Compositional autoregressive cellular grammar",
        description=(
            "Compose familiar operator and offset tokens in rule pairs absent from "
            "training, then predict and free-run the resulting modular language."
        ),
        vocabulary=VOCABULARY,
        sequence_length=SEQUENCE_LENGTH,
        generator=generate,
        evaluation_generator=evaluate,
        dataset_name="disjoint-rule-pair second-order modular grammar",
        dataset_tokens=(len(TRAIN_RULE_PAIRS) + len(HELD_OUT_RULE_PAIRS)) * 16 * 12,
        tokenizer_name="two factorized rules · four symbols · held-out compositions",
    )


__all__ = [
    "CONTINUATION_LENGTH", "DIGIT_COUNT", "DIGIT_OFFSET", "HELD_OUT_RULE_PAIRS",
    "TRAIN_RULE_PAIRS", "VOCABULARY", "compositional_generation_audit",
    "compositional_grammar_cases", "compositional_grammar_provenance",
    "token_compositional_grammar_task",
]
