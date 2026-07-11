# Lifecycle remediation Phase 4 evidence

- Recorded: 2026-07-11 UTC
- Environment: Conda `sakana`
- GPU: `cuda:0`, guarded by `MemoryHigh=20G` / `MemoryMax=24G`
- Scientific horizon: 5,000 steps at `dt=0.01`
- Body: one eight-node line, spacing `2.0`, body length `14.0`
- Solver: fluid-on `PBD_SCHEME`

## Resource and regression evidence

- Capacity/regeneration use compact periodic paraboloid maps.
- Stock remains in `Fields.energy`; capacity and regeneration maps are flat
  `EcosystemState` arrays.
- One mouth at topology node 0 withdraws concurrent proportional demand.
- Focused Phase 4 CPU suite: `83 passed`.
- Complete CPU suite: `249 passed in 95.43s`.
- Native-renderer equivalence remained green in the complete suite.

## Guarded positive controls

The reproducible command is:

```bash
systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src:. python -m experiments.ecosystem.run_positive_controls \
  --output-dir outputs/ecosystem_positive_controls
```

Movement control:

- prescribed wave displacement: `1.9097157`;
- zero-action displacement: `0.0`;
- body-length fraction: `0.1364083` (gate `>= 0.1`);
- wave/zero drift criterion: passed;
- all final state and trace values finite.

Eight-world paired mouth uptake:

- sensory uptake: `[3.6031, 1.9300, 4.4076, 4.6560, 3.0544, 5.1003, 3.6467, 0.0]`;
- matched sensor-disabled uptake: `[0.9722, 0.4037, 1.0879, 0.9641, 0.7844, 1.2340, 0.4166, 0.0]`;
- mean improvement: `350.26%`;
- median improvement: `314.63%`;
- paired wins: `7/8`;
- all acquisition traces finite.

The runner writes ignored local artifacts under
`outputs/ecosystem_positive_controls/`: raw compressed traces, JSON summary,
fixed-scale capacity/stock/uptake diagnostics, and an optional native-renderer
traveling-wave video. The inspected video was H.264, 512x512, 100 frames, and
used the ordinary Microcosmos field renderer with fluid arrows.
