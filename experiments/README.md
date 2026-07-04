# Experiments

Experiments reproduce paper results. Each is a self-contained Python script driven by [Hydra](https://hydra.cc/) config.

## Running

```bash
cd experiments
uv run python main.py experiment=<EXPERIMENT>
```

Output (GIFs, plots, config snapshots) is written to `experiments/outputs/<date>/<time>/`.

---

## worm_swim_example — quick start

A single sine-wave worm swimming in a fluid. Runs in realtime (no optimisation, no NEAT). Good first experiment to verify your install.

```bash
uv run python main.py experiment=worm_swim_example
```

**Output:** realtime animation window.

---

## swim_sweep — viscosity sweep (figure 2)

Runs multiple swimmer morphologies across a range of fluid viscosities and plots displacement vs. viscosity for each.

```bash
uv run python main.py experiment=swim_sweep
```

Override which swimmers to compare:

```bash
uv run python main.py experiment=swim_sweep experiment.swimmers="[worm, tadpole, jellyfish]"
```

Available swimmers: `worm`, `tadpole`, `ray`, `cilia`, `jellyfish`, `squirt`, `fast_worm`, `turbo_worm`

**Output:** displacement-vs-viscosity plot + optional per-viscosity GIFs.

---

## filament_folding — gradient-based shape formation (figure 3)

Optimises a flexible filament to fold into each MNIST digit shape using gradient descent through the physics simulation.

```bash
# Single digit
uv run python main.py experiment=filament_folding experiment.digit=3

# All 10 digits
for digit in {0..9}; do
  uv run python main.py experiment=filament_folding experiment.digit=$digit
done
```

**Output:** per-digit GIF of the folding trajectory + loss curve.

---

## locomotion_cppn — NEAT evolution of a swimming gait

Evolves a CPPN (compositional pattern-producing network) via NEAT to control bending angles along the chain, maximising locomotion distance.

```bash
uv run python main.py experiment=locomotion_cppn
```

Key config knobs (`experiments/conf/experiment/locomotion_cppn.yaml`):

| key | default | meaning |
|-----|---------|---------|
| `neat.num_generations` | — | how long to evolve |
| `neat.use_hyperneat` | `false` | use HyperNEAT substrate |
| `neat.random_seed` | `42` | set `null` for non-deterministic |

**Output:** per-generation best-individual GIF, fitness history plot, best genome `.npz`.

---

## qd_locomotion_cppn — quality-diversity archive of gaits (figure 4)

MAP-Elites search over CPPN-controlled gaits. Fills a 2D archive keyed by two behavioural descriptors (e.g. speed × body extension) to discover a diverse set of locomotion strategies.

```bash
uv run python main.py experiment=qd_locomotion_cppn
```

Key config knobs (`experiments/conf/experiment/qd_locomotion_cppn.yaml`):

| key | default | meaning |
|-----|---------|---------|
| `experiment.bd1` / `bd2` | — | behavioural descriptors to use |
| `experiment.archive_bins` | `[20, 20]` | archive grid resolution |
| `experiment.auto_calibrate_ranges` | `true` | fit BD ranges from seed population |
| `experiment.num_generations` | — | MAP-Elites iterations |

Available BDs: `speed`, `energy`, `turning`, `spatial_freq`, `body_extension`

**Output:** archive fitness heatmap, BD scatter plot, sample grid of GIFs from across the archive.
