# token_compositional_grammar_task.py

## Purpose

Defines the first autoregressive stepping stone whose validation compositions are
provably absent from training, plus a free-running generation audit over the complete
held-out split.

## Components

### `token_compositional_grammar_task`
- **Does**: Trains on six operator/offset pairs and evaluates on two disjoint pairs.
- **Does**: Exposes nine aligned next-token targets from a second-order modular stream.
- **Rationale**: Both operator tokens and every offset token remain familiar; only
  their pairing is novel, separating composition from vocabulary novelty.

### `compositional_grammar_cases`
- **Does**: Enumerates every seed prompt and deterministic continuation in either split.
- **Interacts with**: Unit balance tests and the benchmark generation audit.

### `compositional_grammar_provenance`
- **Does**: Publishes named rule pairs, state counts, and zero overlap for artifacts.

### `compositional_generation_audit`
- **Does**: Repeatedly invokes a black-box next-token predictor on all 32 withheld
  prompts and reports token, exact-sequence, invalid-token, and position accuracy.
- **Does**: Separates the two constant target continuations from the 30 nonconstant
  continuations and reports exact nonconstant sequences explicitly.
- **Rationale**: Teacher-forced accuracy alone can hide cascading generation failure,
  while overall exact-sequence accuracy can reward a collapsed constant-token predictor.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `benchmark_sequences.py` | Task key uses the 68×68 distributed substrate | Task key or vocabulary |
| Tests | Train/evaluation state sets are disjoint and exactly balanced | Split or recurrence |
| Laboratory | Provenance and free-running fields remain JSON-serializable | Field names |

## Notes

- `sum+offset-1` and `difference+offset-2` are the held-out compositions.
- The final target never appears as an input token.
