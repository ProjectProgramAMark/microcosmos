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
- Complete CPU suite at the Phase 4 checkpoint: `249 passed in 95.43s`.
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

- prescribed wave displacement after the Phase 5 tagged-initialization rerun: `2.8855577`;
- zero-action displacement: `0.0`;
- body-length fraction: `0.2061113` (gate `>= 0.1`);
- wave/zero drift criterion: passed;
- all final state and trace values finite.

Eight-world paired mouth uptake:

- sensory uptake: `[4.1475, 4.6738, 5.0146, 4.4094, 5.0436, 4.9673, 5.1179, 3.4291]`;
- matched sensor-disabled uptake: `[0.8928, 0.7772, 1.5870, 0.1833, 0.9135, 0.1029, 1.5828, 0.6354]`;
- mean improvement: `451.38%`;
- median improvement: `477.30%`;
- paired wins: `8/8`;
- all acquisition traces finite.

The runner writes ignored local artifacts under
`outputs/ecosystem_positive_controls/`: raw compressed traces, JSON summary,
fixed-scale capacity/stock/uptake diagnostics, and an optional native-renderer
traveling-wave video. The inspected video was H.264, 512x512, 100 frames, and
used the ordinary Microcosmos field renderer with fluid arrows.
