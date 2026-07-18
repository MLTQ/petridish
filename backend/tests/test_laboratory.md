# test_laboratory.py

## Purpose

Protects the remote laboratory's filesystem, bounded telemetry, geometry, and
control-enable contracts without requiring NVIDIA hardware or launching trainers.

## Components

### Metrics and discovery tests
- **Does**: Verify partial JSONL records are ignored, payload size is bounded, and
  latest training/held-out measurements remain distinct.
- **Does**: Keeps the latest scientific diagnostic separate from optimizer and
  held-out/generation records.

### Safety tests
- **Does**: Reject path traversal and process launch when control is disabled.
- **Does**: Verify a defunct Linux trainer is not reported as a live run merely
  because its PID still exists.

### Geometry test
- **Does**: Keeps 68×68 mandatory for both 66-port and 64-port single-column
  corpus layouts.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `laboratory.py` | Safe run IDs, opt-in launch, bounded metrics | Relaxing validation |
| CI | Tests run without CUDA or `nvidia-smi` | Hardware-dependent setup |
