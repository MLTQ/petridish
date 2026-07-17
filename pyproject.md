# pyproject.toml

## Purpose

Defines the installable Python backend, runtime dependencies, and test settings.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Backend modules | Python 3.11+ with PyTorch, TorchVision, NumPy, and FastAPI | Dependency or Python floor changes |
| Contributors | `uv sync --extra dev` installs tests and runtime | Extra name or package layout |
| Pytest | Backend package importable from `backend/` | Package directory or test path |

## Notes

PyTorch is intentionally not CUDA-pinned. `uv` resolves the platform-appropriate
wheel; accelerator-specific installs can override it without changing source.
TorchVision supplies the canonical MNIST files and parsing without introducing a
second machine-learning runtime.
