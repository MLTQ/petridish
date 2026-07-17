# sequence_cells.py

## Purpose

Defines controlled homogeneous recurrent rules for sequence organisms while
preserving private per-neuron state and shared parameters per architecture.

## Components

### `CELL_ARCHITECTURES`
- **Does**: Names architectures with implemented model, trainer, checkpoint, and lab paths.

### `SequenceCellRule`
- **Does**: Presents one update interface for GRU, LSTM, ESN, and temporal-transformer cells.
- **Interacts with**: `CellularSequenceModel` in `sequence_model.py`.

### GRU rule
- **Does**: Preserves the original shared GRUCell baseline exactly.

### LSTM rule
- **Does**: Gives every neuron private hidden and protected cell state; forget bias starts at one.

### ESN rule
- **Does**: Uses a fixed orthogonal radius-0.9 reservoir and leak 0.35 with learned input/output projections.

### Transformer rule
- **Does**: Gives every neuron four private temporal message slots and one-head
  content-addressed recall followed by a residual feed-forward update.
- **Rationale**: Dendritic attention remains inter-neuron communication; these
  slots test transformer-like temporal memory inside each physical neuron.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `sequence_model.py` | Rule returns hidden state plus optional private memory | Forward signature |
| Checkpoints | Architecture determines parameter and memory behavior | Architecture names |
| Experiments | Parameters are shared; state is private to each neuron/example | Per-neuron parameters |

## Notes

- Transformer memory is differentiable and intentionally expensive; initial tests
  should use associative recall or small corpus batches.
- Homogeneous controls precede mixtures so architecture effects are identifiable.
