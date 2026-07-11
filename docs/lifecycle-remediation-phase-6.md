# Lifecycle remediation Phase 6 evidence

- Recorded: 2026-07-11 UTC
- Environment: Conda `sakana`, Python 3.13.14, JAX 0.8.1
- GPU: NVIDIA GB10 at `cuda:0`
- Guard: `MemoryHigh=20G`, `MemoryMax=24G`, `MemorySwapMax=2G`

## Fused evaluator-shaped benchmark

```bash
systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana python benchmarks/benchmark_ecosystem.py \
  --steps 1000 --max-creatures 32 --nodes-per-creature 8 \
  --output outputs/ecosystem_fused_benchmark.json
```

Both cases allocate 256 nodes on a 128x128 grid with fluid enabled. The
ecosystem starts with 16/32 live slots (128 active nodes); activity and solver
configuration are recorded in the JSON artifact. Each compiled function uses a
fixed-size `lax.scan` and returns only final state (plus compact metrics for the
ecosystem), never a full trajectory.

| Measurement | Equal-node baseline | Ecosystem |
|---|---:|---:|
| Compile + first execution | 1.444593 s | 5.222989 s |
| Warm median, 5 runs | 0.369411 s | 0.499569 s |
| Warm IQR | 0.001305 s | 0.000626 s |
| Warm MAD | 0.000362 s | 0.000499 s |
| Simulated steps/s | 2707.02 | 2001.73 |

- Fused warm median ratio: `1.3523408` ecosystem/baseline.
- Peak process RSS: `1880.53 MiB`.
- The earlier `1.2000x` Python-loop/no-fluid result remains historical only;
  later optimization decisions use this fused measurement.

## Search-shaped rollout API

`microcosmos.rollout.run_ecosystem_chunk` returns final state plus scalar
metrics for productivity, births/deaths, occupancy extrema, resource/energy
bounds, finiteness, identities, and event consistency. It emits no scan history.

`run_ecosystem_chunk_with_snapshots` is a separate explicit debug path. It uses
nested fixed-size chunks and retains only one renderable snapshot per declared
stride, with `f_grid` removed.

## Guarded long-run correctness

`tests/test_ecosystem_long_rollout.py` passed on CUDA in `9.95s`:

- 1,000 fluid-on steps under the 24 GiB guard;
- ten forced lifespan deaths and same-step births reusing slot 0;
- monotonic child IDs 2 through 11 and final `next_individual_id == 12`;
- event-array/count agreement, unique active identities, and static shapes;
- finite state, positive live energy, and mapped resource bounds;
- inactive slot position isolation;
- final native-renderer visibility equivalence.

## Final Phase 6 regression

- Scoped Ruff on benchmark, rollout, and long-rollout files: passed.
- Complete CPU suite: `259 passed, 1 skipped in 103.92s`.
- The single skip is the intentionally GPU-only long-rollout test; that same
  test passed separately under the guard as recorded above.
