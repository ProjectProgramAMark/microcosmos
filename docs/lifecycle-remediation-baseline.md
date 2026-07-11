# Lifecycle remediation baseline

- Recorded: 2026-07-11 UTC
- Upstream commit: `ced07003b57f277034f688ab15fcc0c0834a1d9f`
- Baseline ref: `lifecycle-baseline-20260711`
- Environment: Conda `sakana`
- Python: 3.13.14
- JAX: 0.8.1
- Backend: `gpu`
- Device: `CudaDevice(id=0)`

## Worktree scope before checkpoint

The baseline consists of the complete reviewed fixed-capacity ecosystem spike,
including:

- optional node activity/component metadata and masked mechanics;
- fixed-capacity population, resource, energy, death, birth, mutation, and
  telemetry;
- native Microcosmos rendering with inactive-node filtering;
- periodic body centers, topology-derived spawn separation, occupied-center
  placement, and simultaneous-birth reservation;
- ecosystem experiment/configuration, benchmark, documentation, and focused
  tests;
- Linux AArch64/DGX Spark dependency metadata already present in the user-owned
  worktree.

`git status --short` and `git diff --stat` were recorded in the implementing
thread immediately before the checkpoint. No unrelated cleanup was folded into
the baseline.

## Verification evidence

Scoped Ruff:

```text
All checks passed!
```

Focused CPU lifecycle/mask/ecology regression:

```text
26 passed in 18.53s
```

Complete CPU regression:

```text
182 passed in 79.45s
```

Guarded CUDA/native-artifact evidence is recorded in
[`ecosystem-native-rendering-and-placement-plan.md`](./ecosystem-native-rendering-and-placement-plan.md):

- 13 focused CUDA ecosystem tests passed on `CudaDevice(id=0)`;
- native renderer equivalence and inactive-slot omission passed;
- original and lifecycle live videos were generated and inspected;
- topology-derived placement and simultaneous births were verified;
- the provisional Python-loop warm benchmark reported `1.2000x` overhead at
  64 creatures × 16 nodes × 1,000 no-fluid steps.

The provisional benchmark is not an evaluator-shaped performance gate. The
remediation plan replaces it with separate compile and warm fused-scan
measurements.

## Protected baseline decisions

- Keep `Nodes.active=None` as the legacy fast path.
- Keep `PopulationState.alive` as lifecycle authority and project it to node
  activity at ecosystem boundaries.
- Keep the flat `EcosystemState` and current resource stock in `Fields.energy`.
- Keep the native generic renderer; do not restore ecosystem-specific drawing.
- Keep body-aware periodic placement and best-candidate birth placement.
- Keep current legacy environment/API behavior unless a later reviewed plan
  explicitly changes it.
