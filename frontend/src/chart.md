# chart.ts

## Purpose

Maintains bounded viewer-local objective and accuracy histories and
renders both into the compact SVG.

## Components

### `HistoryChart`
- **Does**: Normalizes loss into a rising objective, appends accuracy, trims both
  to 160 frames, and updates lines/summary.
- **Does**: Clears history when switching organisms so tasks are never joined.
- **Interacts with**: `main.ts` and snapshot metrics.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | Constructor accepts reward line, accuracy line, and SVG | Constructor signature |
| `index.html` | Chart view box is 300×90 | Coordinate system changes |

## Notes

History is intentionally presentation-only. Scientific recording belongs in the
backend so browser refreshes cannot alter experiment data.
