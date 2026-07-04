# Microcosmos 🦠🧬

[Paper]() | [Blog](https://alife.institute/en/blog/microcosmos-release/) 

Microcosmos (ALIFE Conference 2026) is a JAX-accelerated artificial life simulator for evolving and optimizing filament-based organisms in physically grounded environments. The engine is fully differentiable, supporting both gradient-based optimization and evolutionary search.

## Features

- GPU-native simulation with JAX
- Position-Based Dynamics (PBD) using a Cosserat rod formulation
- Lattice-Boltzmann fluid solver with immersed boundary coupling
- Fully differentiable simulation for gradient-based optimization
- Evolutionary search over morphologies and controllers
- Scales linearly with filament count through local-only interactions

## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate
uv sync --dev
```

### GPU (optional)

JAX runs on CPU by default. For CUDA 13:

```bash
uv pip install "jax[cuda13]"
```

## Quick start

```bash
# Sanity check — runs a basic simulation and exits
uv run python src/microcosmos/main.py

# Run tests
uv run pytest
```

## Example notebook

An interactive Jupyter notebook demonstrating how the filaments can be evolved to locomote can be found in the `examples` directory:

```text
examples/01_es_locomotion.ipynb
```

## Experiments

See [`experiments/README.md`](experiments/README.md) for instructions on reproducing the experiments from the paper, including:

- Filament locomotion
- Differentiable filament folding
- Evolution of swimming and foraging behaviors
- Scaling benchmarks

## Citation

If you use Microcosmos in your research, please cite our paper.

```bibtex
% TODO
```