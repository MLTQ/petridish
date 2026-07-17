# chart.ts

## Purpose

Maintains bounded viewer-local objective and accuracy histories for the selected
experiment and renders both into the compact SVG.

## Components

### `HistoryChart`
- **Does**: Resets on experiment switches, appends snapshots, normalizes MNIST
  loss into a rising objective, trims to 160 frames, and updates lines/summary.
- **Interacts with**: `main.ts` and snapshot metrics.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | Constructor accepts reward line, accuracy line, and SVG | Constructor signature |
| `index.html` | Chart view box is 300×90 | Coordinate system changes |

## Notes

History is intentionally presentation-only. Scientific recording belongs in the
backend so browser refreshes cannot alter experiment data.
