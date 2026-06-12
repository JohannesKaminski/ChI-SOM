# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role

Act as a programming companion, not an individual developer. Prefer explaining concepts and analysing problems over proposing concrete code changes. Keep answers digestible by splitting them into meaningful, individual points.

## Commands

```bash
# Install (dev dependencies)
uv sync --group dev

# Install with CUDA support (choose one)
uv sync --group dev --extra cu12
uv sync --group dev --extra cu13

# Type checking
uv run ty check

# Linting and formatting
uv run ruff check src/
uv run ruff format src/

# Tests (only test data exists currently, no unit test files)
uv run pytest
uv run pytest tests/path/to/test.py::test_name  # single test

# Docs (local preview)
uv run mkdocs serve
```

> Only Linux is supported (`uv` environment constraint in `pyproject.toml`).

## Architecture

### Two-phase workflow

The library separates **data preparation** from **training**:

1. `HDF5Creator` (one-time) — converts raw SMILES/INCHI text files into an indexed HDF5 store using multiprocessing fingerprint generation. `rdStyleFactory` or `CSVStyleFactory` supplies the fingerprint generator.
2. `HDF5Dataset` — PyTorch `DataLoader`-compatible interface over that HDF5 store. It can also be bypassed entirely; `Som` accepts plain numpy arrays.

### `Som` and the `Trainer` abstraction

`Som` is the sole user-facing training class. On construction it instantiates one of three `Trainer` subclasses:

| Trainer | When selected |
|---|---|
| `CudaTrainer` | `use_cuda=True` (requires `cu12`/`cu13` extra) |
| `CPUTrainer` | default CPU path |
| `CPUTrainerLocal` | `use_local_neighborhood=True` — restricts updates to a hard sigma-bounded neighbourhood for speed |

The hot path in every trainer is identical: **find the best-matching unit (BMU)** via vector distance minimisation, then **update all neurons** using precomputed neighbourhood coefficients. `Trainer.update_coefficients()` must be called once per epoch (by `Som.train()`) before batches are processed.

### Factory functions as the JIT boundary

All performance-critical functions (distance metrics, neighbourhood kernels, U-matrix calculation) are created by `make_*` factory functions in `_core/cpu/distance.py`, `_core/cpu/kernel.py`, and `_core/cuda/`. These factories return Numba-compiled (`@njit` / `@cuda.jit`) callables and are called **once at `Som.__init__` time**, not per batch. This means:

- The first training call incurs Numba compilation overhead.
- Changing a distance metric or kernel requires constructing a new `Som` instance.

### Codebook

Shape is always `(rows, columns, features)`, C-contiguous, `float32`. On the CPU path it lives in a numpy array; on the CUDA path it lives on the GPU device. `Som.codebook` is a property that transfers it back to CPU on get.

### GUI

`start_chisom_viewer()` is entirely independent of training. It takes a precomputed U-matrix, BMU positions, and the `HDF5Dataset` (for structure metadata). The `_interface/` module is Qt/PySide6-based and requires an X11 display — not available in headless environments.

### Type system

Public types are defined in `_core/types.py` and `io/_types.py` as `NDArray` aliases and `Protocol` classes. `py.typed` is present, so downstream consumers get full type information.
