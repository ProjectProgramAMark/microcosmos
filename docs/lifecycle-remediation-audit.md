# Lifecycle remediation completion audit

Audited against Appendix A of the lifecycle remediation PRD on 2026-07-11.
“Deferred” means deliberately outside this remediation and names its owner; no
item remains accidentally active or unowned.

| Audit item | Final disposition | Authoritative evidence |
|---|---|---|
| Birth configuration can create energy | Completed | `LifecycleConfig.child_initial_energy`, `reproduction_step`; energy-transfer tests in `tests/test_ecology.py`; commit `b497b64` |
| Actuation depends on numerical timestep | Completed | `actuation_energy_by_slot` multiplies power by `dt`; one-second invariance test; commit `b497b64` |
| Periodic parent mean spawned remotely | Protected | `_slot_centers` uses periodic displacements; periodic-center and birth-distance tests |
| Initial/newborn overlap | Protected | body-radius separation, best-candidate placement, simultaneous reservation tests |
| Steric exclusions use storage order | Completed | padded graph neighborhoods in `solver/masks.py`; line/ring/branch and permutation-force tests; commit `c972663` |
| Ring/branch semantics overstated | Completed/restricted | generic mechanics tests exact graph topology; `EcosystemEnv` and experiment explicitly reject non-line controllers |
| Population/node activity can diverge | Completed | population projected at step entry and after lifecycle; deliberate-inconsistency test |
| Mask logic rederived/scattered | Completed | one `activity_masks` derivation in `simulate.step`, passed through constraints, steric, and fluid |
| Partial masks affect synthetic tangents | Completed | edge-masked tangent accumulation and `1e-7` active-reaction regression |
| `Fields.energy` has resource semantics | Deliberate compatibility | stock remains in `Fields.energy`; flat capacity/regeneration maps and mapped-bound tests; commit `41311d8` |
| Flat ecosystem state duplicates legacy fields | Deliberate domain state | flat `EcosystemState` retained and protected; no nested-state rewrite |
| Keyword-heavy constructor lacks validation | Completed | host validation across lifecycle, controller, resource, placement, numerical, and fixed-trait fields |
| `EcosystemEnv` overrides much of base | Deliberate compatibility | inheritance and existing action spec retained; registry/action compatibility tests |
| External action is additive | Completed/documented | residual is added to autonomous rest/bend action; implicit/zero-explicit equivalence test |
| Birth events are dict-based | Deferred | Master Phase 3 scenario/event typing; fixed-shape dict contract remains tested |
| 61-gene MLP opaque/over-scoped | Completed | normalized 23-locus wave controller and exact slice-coverage tests; commit `f0b47b2` |
| Ecological traits evolve with genome | Completed | fixed uptake/assimilation/metabolism config and all-loci fixed-trait regression |
| Mutation coupled to controller | Completed | controller contains no mutation implementation; private baseline lives in `ecology.py` |
| Final heredity policy hard-wired | Deferred | Master Phase 4 after sustained-turnover calibration; no candidate callable added |
| Mutation/spawn randomness depends on rank | Completed | named `fold_in` streams keyed by timestep/child ID; order/insertion and replay tests; commit `cf34143` |
| Uniform resource/no-fluid makes movement irrelevant | Completed | compact mapped patches, one mouth, fluid scientific config, guarded movement/foraging controls |
| Ecosystem renderer bloated generic renderer | Protected/completed | no specialized renderer; exact native-renderer tests and inspected native video |
| Generic renderer buffers frames | Deferred | generic renderer concern outside lifecycle remediation; search chunks retain no histories |
| Python-loop benchmark not evaluator-shaped | Completed | fused `lax.scan` benchmark with compile/warm/dispersion/memory/config evidence |
| Short tests do not prove long-run invariants | Completed | guarded 1,000-step forced-turnover CUDA regression and compact invariant metrics |

## Final evidence summary

- Baseline ref: `lifecycle-baseline-20260711` at `28c8311`.
- Green phase commits: `b497b64`, `c972663`, `f0b47b2`, `41311d8`, `cf34143`.
- Legacy no-context fluid/no-fluid checkpoint comparison: bit-for-bit identical.
- Guarded topology/mask smoke: 1,000 steps, finite and shape-stable.
- Guarded movement/acquisition controls: both pass; 8/8 paired acquisition wins.
- Guarded long rollout: 1,000 fluid steps and ten same-step reuse events pass.
- Fused benchmark: 1.3523408x warm median ratio at equal 256-node allocation;
  complete compile/warm/device/shape/memory record in Phase 6 evidence.
- Final CPU regression: 259 passed, one intentionally GPU-only test skipped;
  the skipped test passed in its guarded CUDA gate.
