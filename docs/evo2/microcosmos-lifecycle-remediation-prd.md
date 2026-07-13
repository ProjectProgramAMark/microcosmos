[PRD]
# PRD: Microcosmos lifecycle remediation

- Status: implemented and verified
- Revision: 4
- Prepared: 2026-07-11
- Completed: 2026-07-11
- Repository: `microcosmos`
- Relationship: focused corrective plan for Master Phases 0–1 of the
  [Evo²-Ecosystem master PRD](./evo2-ecosystem-master-prd.md)

Completion evidence is recorded in
`../microcosmos/docs/lifecycle-remediation-audit.md` and the phase evidence
notes. All remediation acceptance checkpoints pass; deliberately deferred
master-plan work remains deferred.

> Archive notice: do not execute this PRD as a current task list. It is retained
> to preserve decisions and acceptance evidence. Continue from the
> [Microcosmos active implementation plan](./microcosmos-substrate-plan.md).
> References below to deferred heredity work, fixed Gaussian mutation, or old
> Master phase numbers describe the 2026-07-11 checkpoint and are superseded by
> the direct CPPN/TensorNEAT plans. Likewise, archived same-stack bitwise RNG
> checks are not a claim of bitwise-reproducible full GPU ecology.

## 1. Overview

This remediation corrected the lifecycle, mechanics, controller, resource, randomness,
and benchmark defects in the current fixed-capacity ecosystem without replacing
its verified native rendering, body-aware placement, legacy physics API, flat
state shape, or `Nodes.active=None` compatibility path.

The verified worktree proves that fixed slots, activity-aware physics,
birth/death, same-step slot reuse, native rendering, periodic body centers, and
body-aware placement are viable. This revision narrows remediation to defects
that still affect physical invariance, topology correctness, embodied selection,
paired stochastic comparison, or evaluator-shaped performance evidence.

The completed work supports sustained-turnover calibration and the later
lineage/heredity phases. No Shinka work is authorized by this archived record.

## 2. Problem statement

At the baseline checkpoint, the ecosystem passed its regression suite but was
not yet a valid scientific substrate because:

- Child energy is independently configurable from reproduction cost, so an
  accepted configuration can create organism energy.
- Actuation cost is charged per numerical step while metabolism is scaled by
  physical time, making ecology depend on `dt`.
- Constructor validation covers placement and a few resource bounds but not the
  complete lifecycle/controller/resource domain.
- Steric neighbor subtraction uses storage-index proximity rather than graph
  topology. `component_id` only prevents cross-body contamination.
- Activity masks are rederived in multiple mechanics paths, and synthetic-fluid
  tangents do not consistently ignore inactive incident edges.
- `PopulationState.alive` and `Nodes.active` are both stored without an explicit
  synchronization invariant.
- The default 61-gene MLP includes evolvable acquisition/metabolic traits,
  confounding behavioral heredity with trivial ecological optimization.
- Resources initialize and regenerate uniformly, so movement and sensing have
  little selective value.
- The scientific demo disables fluid, leaving no credible swimming mechanism.
- Mutation and spawn randomness are assigned by ranked array position rather
  than immutable individual identity.
- Mutation implementation lives with controller decoding instead of a trusted
  internal ecology baseline.
- The benchmark times a Python loop of individually jitted steps instead of the
  fused scan used by the intended evaluator.
- Existing short tests do not prove long-horizon fluid, turnover, resource, and
  identity invariants.

Periodic placement and parallel-renderer concerns from the original audit are
already resolved and are protected by this plan rather than scheduled again.

## 3. Outcome hypothesis

If energy/config invariants, topology-aware mechanics, the compact controller,
localized resources, fluid-on controls, identity-keyed randomness, and fused
benchmarking are implemented while preserving the verified compatibility
surface, then Microcosmos will support a reproducible embodied ecosystem without
incurring a second simulator or unnecessary core-state migration.

## 4. Goals

- Make every accepted birth configuration non-energy-creating by construction.
- Make fixed-control energy charge invariant to timestep subdivision within a
  declared tolerance.
- Reject invalid lifecycle, controller, resource, and numerical configuration
  before JIT tracing.
- Make steric exclusions graph-topology-aware and independent of node storage
  order.
- Derive node/edge/bending activity once per physics step while retaining
  `Nodes.active=None` for legacy callers.
- Freeze a normalized 23-locus controller genome with fixed ecological traits.
- Make resource stock spatially selective through capacity/regeneration maps
  while retaining current stock in `Fields.energy`.
- Restrict acquisition to one fixed mouth node per organism so appetite does
  not grow with body-node count.
- Demonstrate fluid-on genotype-dependent movement and resource acquisition.
- Derive mutation and spawn randomness from named streams and immutable IDs.
- Separate current trusted Gaussian mutation from controller decoding without
  exposing the final candidate-facing heredity API.
- Replace the Python-loop benchmark with compile and warm fused-scan reporting.
- Preserve all 182 current CPU tests and completed native rendering/placement
  behavior while adding focused regressions.

## 5. Scope boundaries

### In scope

- Energy transfer, actuation power, and complete host-side config validation.
- Explicit synchronization between `PopulationState.alive` and `Nodes.active`.
- Topology-derived steric exclusions and one shared activity-mask context.
- Optional internal physics context parameters that preserve all existing
  positional callers and the `None` legacy path.
- Compact controller and fixed ecological traits from the substrate plan.
- Per-cell resource capacity/regeneration maps with stock retained in
  `Fields.energy`.
- Mouth-only simultaneous resource withdrawal; head/tail and per-hinge resource
  samples remain sensors rather than additional mouths.
- Fluid-on locomotion and paired acquisition positive controls.
- Named, identity-keyed initialization/mutation/spawn RNG streams.
- A trusted internal fixed-Gaussian mutation helper; no final policy injection.
- Fused benchmarking and long-rollout invariant tests.
- Reconciliation of active planning documents with completed native work.

### Non-goals

- Removing `Nodes.active` or changing legacy `Nodes(active=None)` behavior.
- Replacing the flat `EcosystemState` with nested mechanical/resource wrappers.
- Moving current resource stock out of `Fields.energy`.
- Removing `EcosystemEnv` inheritance or its current action-spec compatibility.
- Creating an ecosystem-specific renderer, palette, overlay, or animation path.
- Redesigning the generic renderer or solving its general frame-buffering policy.
- Freezing the final `OffspringPolicy(parent_stats, population_stats, rng)` API.
- Candidate validation, Shinka markers, evaluator execution, or search.
- Founder lineage, catastrophe scenarios, recovery metrics, or sealed manifests.
- Ecological calibration or a claim of sustained turnover.
- Heterogeneous topology, growth, variable body size, or sexual reproduction.
- Repository-wide cleanup of pre-existing Ruff findings.
- Performance optimization before fused measurements identify a bottleneck.
- New third-party runtime dependencies.

## 6. Research snapshot

Date: 2026-07-11

- Local authority: `microcosmos` commit
  `ced07003b57f277034f688ab15fcc0c0834a1d9f` plus the reviewed user-owned dirty
  worktree.
- Runtime authority: Conda `sakana`, Python 3.13.14, JAX 0.8.1, and the GPU/memory
  rules in [`AGENTS.md`](../AGENTS.md).
- Current verification: scoped Ruff passes; 26 focused
  lifecycle/mask/ecology tests pass; the full CPU suite passes with 182 tests.
- Completed implementation record:
  [native rendering and placement](../microcosmos/docs/ecosystem-native-rendering-and-placement-plan.md)
  records 182 CPU tests, 13 guarded CUDA ecosystem tests, native renderer
  equivalence, body-aware periodic placement, and live artifact checks.
- Direct unresolved probe: parent energy 5, reproduction cost 1, and independent
  child energy 10 produced total organism energy 14.
- Direct unresolved probe: one physical second of equal actuation left energy
  approximately 9 at `dt=0.1` and 8 at `dt=0.05`.
- Source review: steric exclusions still use storage offsets, and mutation keys
  still use `jax.random.split(key, capacity)` by ranked entry.
- Source review: the benchmark still dispatches a Python loop; its previous warm
  ratio is provisional rather than evaluator-shaped evidence.

No external research gap blocks implementation. The unresolved performance ratio
is intentionally measured, not assumed.

## 7. Constraints and protected invariants (DO NOT CHANGE)

- DO NOT alter existing legacy environment names, behavior, or public positional
  call signatures.
- DO NOT remove `Nodes.active`; `active=None` must retain the exact legacy path.
- DO NOT treat `Nodes.active` and `PopulationState.alive` as independent
  authorities. Population is lifecycle authority; the environment projects it
  to node activity before mechanics and after lifecycle events.
- DO NOT restore an ecosystem-only renderer. Lifecycle frames must remain native
  Microcosmos field frames with generic inactive-node filtering.
- DO NOT change the completed body-aware initial placement, periodic centroid,
  spawn separation, occupied-center selection, or simultaneous-birth reservation
  except for identity-keyed RNG inputs.
- DO NOT make available geometric space a condition for reproduction success.
- DO NOT change the authoritative lifecycle order:
  sense -> control -> masked physics -> resource update -> energy/age -> death
  -> birth -> telemetry.
- DO NOT allow newborn participation before the next step or dead-parent
  reproduction in the death step.
- DO NOT use dynamic population shapes or host lifecycle decisions inside JIT.
- DO NOT move ecosystem stock out of `Fields.energy` in this phase. Add explicit
  capacity/regeneration maps beside it and use resource terminology locally.
- DO NOT let strategy loci or any phenotype locus change uptake, assimilation,
  metabolism, topology, or actuator limits.
- DO NOT multiply uptake by body nodes. Topology node 0 is the sole mouth in v1.
- DO NOT expose or freeze the final candidate-facing heredity policy before the
  embodied-foraging and sustained-turnover gates pass.
- DO NOT import experiment code from the core package.
- DO NOT overwrite or reformat unrelated user-owned worktree changes.
- DO NOT increase JAX preallocation or the 24 GiB session guard without approval.

## 8. Assumptions and decisions

- The native-rendering/placement implementation is complete and becomes the
  baseline for all phases.
- `EcosystemEnv` remains an `Environment` subclass for now. Its optional action
  is explicitly documented and tested as an additive residual; autonomous
  scientific rollouts pass `None`.
- `EcosystemState` remains flat. Existing node/edge/field/population access paths
  remain valid.
- `Fields.energy` is explicitly documented as renewable resource stock inside
  `EcosystemEnv` and remains the legacy target field elsewhere.
- Add `resource_capacity_map` and `resource_regeneration_map` to
  `EcosystemState`; do not introduce a nested `ResourceState` now.
- Keep `Nodes.component_id` during topology-mask migration. If it has no consumer
  after graph-neighborhood exclusions land, remove it in the same phase only
  after a repository usage audit and full regression.
- Add a small optional `PhysicsContext`/`StericExclusions` argument only where
  required. Existing calls omit it and retain their current trace/behavior.
- Activity masks are computed once at physics-step entry from `Nodes.active` and
  passed to constraint/fluid internals. They are not stored as a second dynamic
  state object.
- The controller module owns only genome layout, decoding, sensing, and action.
  The current trusted Gaussian implementation moves to a private ecology helper
  until the final heredity module is introduced in Master Phase 4.
- Named RNG tags and child-ID derivation are implemented now because they are
  prerequisites for later paired comparison, but no candidate policy is accepted.
- Default lifecycle rendering keeps `animate_energy=false`. Resource positive
  controls use numerical uptake traces and separate fixed-scale diagnostic plots,
  not a new rendering stack.
- The current `1.5x` or `1.2x` Python-loop ratios are not acceptance gates. The
  fused measurement determines later performance work.

## 9. Execution context (AGENTS.md alignment)

- Run commands from `microcosmos/`.
- Run every Python command with `conda run -n sakana`.
- Run the complete regression suite on CPU.
- Run only focused fluid/benchmark workloads on GPU in a fresh process under the
  20 GiB/24 GiB systemd memory guard.
- Before GPU work, count existing JAX sessions and inspect `free -h`; do not start
  a fourth session or start below 40 GiB available memory.
- Keep the configured `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10` for long focused
  workloads. Use `XLA_PYTHON_CLIENT_PREALLOCATE=false` only for tiny debugging
  smoke tests.
- Use fixed-shape JAX arrays and pure functions inside scans.
- Use existing dataclass/pytree, NumPy, NetworkX, OpenCV, pytest, and Ruff patterns.
- Use `apply_patch` for edits and preserve unrelated dirty-tree changes.
- Checkpoint the reviewed current lifecycle tree before Phase 1, then use one
  independently green commit per phase.

## 10. Quality gates

Required environment check before substantive work:

```bash
conda run -n sakana python -c "import os,sys; assert os.path.basename(sys.prefix) == 'sakana'; print(sys.executable)"
```

Required after every story, scoped to touched files:

```bash
conda run -n sakana ruff check <touched paths>
conda run -n sakana env JAX_PLATFORMS=cpu pytest <focused test paths> -q
git diff --check
```

Required at every phase checkpoint:

```bash
conda run -n sakana env JAX_PLATFORMS=cpu pytest -q
```

Required focused GPU gate after mechanics and embodied-foraging phases:

```bash
free -h
systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana python -c "import jax; assert jax.default_backend() == 'gpu'; print(jax.devices())"
systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana pytest -q tests/test_ecosystem_long_rollout.py
```

Required benchmark gate:

```bash
systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana python benchmarks/benchmark_ecosystem.py \
  --steps 1000 --max-creatures 32 --nodes-per-creature 8
```

Visual/artifact verification:

- rerun native-renderer equivalence after any `Nodes` or activity change;
- inspect one fluid-on traveling-wave video in the ordinary Microcosmos view;
- inspect a fixed-scale resource stock/capacity diagnostic plot and confirm it
  agrees with raw uptake/resource traces;
- confirm no ecosystem palette, edge drawer, telemetry overlay, or specialized
  animation function is introduced.

## 11. Implementation phases (dependency-ordered)

### Phase 0: Checkpoint and reconcile the baseline

Objective: preserve the verified current tree and remove plan ambiguity.

Dependencies: none.

Requirements:

- Record commit, dirty status/stat, Python/JAX/device, scoped Ruff, 26 focused
  tests, and 182 full CPU tests.
- Create a reviewed checkpoint ref without unrelated cleanup.
- Mark native rendering/placement completed and protected in every active plan.
- Remove directives for specialized rendering, removal of `Nodes.active`, nested
  state, stock migration, standalone environment conversion, and early final
  heredity injection.

Acceptance checkpoint:

- [x] One named Git ref restores the exact reviewed baseline.
- [x] Active plans have one consistent disposition for every audit item.
- [x] Native rendering/placement verification remains linked from the plan index.

Rollback point: recorded user-owned dirty tree.

### Phase 1: Fix energy and configuration invariants

Objective: make invalid ecology unrepresentable before scientific changes.

Dependencies: Phase 0.

Requirements:

- Replace independent child energy with `birth_transfer_efficiency` and derive
  child energy from successful parent debit.
- Rename `actuation_cost` to `actuation_power_coefficient` and multiply the
  per-slot squared command by `dt`.
- Add one host-side validation path for timestep, grid, initial energy/resource,
  lifecycle thresholds/ages, resource rates/diffusion, genome/controller bounds,
  placement settings, and fixed ecological traits.
- Preserve zero charge on failed allocation and same-step dead-slot reuse.

Acceptance checkpoint:

- [x] No accepted config can produce child energy greater than parent debit.
- [x] Failed allocation charges exactly zero.
- [x] Equal fixed control over equal physical duration at `dt` and `dt/2` differs
  by at most `1e-5` relative plus `1e-6` absolute.
- [x] Invalid values raise deterministic `ValueError` before JIT tracing.
- [x] Native placement/rendering tests and full CPU suite pass.

Rollback point: Phase 0 checkpoint.

### Phase 2: Make topology and activity mechanics explicit

Objective: remove storage-order assumptions and inconsistent mask derivation
without changing legacy callers.

Dependencies: Phase 1.

Requirements:

- Precompute padded graph-neighborhood exclusion indices/valid masks from the
  configured topology and steric neighbor distance.
- Add an optional immutable physics context carrying topology exclusions.
- Compute node, edge, and bending activity once per physics step from
  `Nodes.active`; pass masks through PBD, Cosserat, steric, and fluid internals.
- Mask synthetic-node tangent accumulation by edge activity.
- At `EcosystemEnv.step` entry, project `population.alive[node_slot]` to
  `nodes.active`; project again after death/birth before returning state.
- Remove `component_id` only if no consumer remains after topology exclusions.

Acceptance checkpoint:

- [x] Existing calls with no context match the checkpoint within absolute
  tolerance `1e-7`.
- [x] Line, ring, and one branched fixture have exact graph-correct exclusion
  neighborhoods, including the ring closing edge.
- [x] A storage permutation preserves exclusion membership exactly and steric
  results within `1e-6` on the pinned CPU fixture.
- [x] Deliberately inconsistent input activity is canonicalized from population
  at the ecosystem boundary.
- [x] An inactive incident edge changes active-node fluid reaction by at most
  `1e-7` in the focused fixture.
- [x] A guarded 1,000-step masked-fluid smoke remains finite and shape-stable.
- [x] Full CPU suite passes.

Rollback point: Phase 1 checkpoint.

### Phase 3: Install the compact fixed-trait controller

Objective: make heritable behavior interpretable without allowing metabolic
shortcuts.

Dependencies: Phase 2.

Requirements:

- Replace the MLP with the exact normalized 23-locus layout from the substrate
  plan: 20 phenotype loci and three strategy loci.
- Implement three traveling waves, local-resource polynomial correction,
  head-tail-gradient correction, bias, and positive gain.
- Keep uptake, assimilation, metabolism, topology, and actuator limits in config.
- Restrict the scientific controller to the declared line topology; reject
  unsupported sensor/topology combinations explicitly.
- Move trusted fixed-Gaussian mutation out of `controller.py` into a private
  ecology baseline helper without adding a candidate callable.

Acceptance checkpoint:

- [x] Genome shape is exactly `float32[23]` and bounded to `[-1, 1]`.
- [x] Strategy loci never change controller action bitwise.
- [x] Every phenotype module changes action in a deterministic fixture.
- [x] Action is finite/bounded for extreme genes and sensors.
- [x] Translation and periodic wrapping preserve sensor semantics.
- [x] Eager, JIT, and vmap action agree for identical inputs.
- [x] No genome locus changes a fixed ecological trait.
- [x] Full CPU suite passes.

Rollback point: Phase 2 checkpoint.

### Phase 4: Add localized resources and fluid-on embodied controls

Objective: make controller differences causally affect movement and acquisition.

Dependencies: Phase 3.

Requirements:

- Add `resource_capacity_map` and `resource_regeneration_map` to the flat
  `EcosystemState`; retain current stock in `state.fields.energy`.
- Add compact periodic patch construction and map-based reset.
- Designate topology node 0 as the sole mouth; compute one demand per alive
  organism at its mouth position. Keep head/tail/per-hinge resource samples for
  control only.
- Update resource order to concurrent mouth withdrawal, optional conservative
  diffusion, mapped regeneration, and bounded clipping.
- Enable fluid in scientific configs; keep no-fluid only for unit/debug use.
- Add deterministic traveling-wave and paired resource-sensing positive controls.
- Preserve the native renderer. Use raw traces and a separate fixed-scale
  diagnostic plot for resource verification.

Acceptance checkpoint:

- [x] Stock remains in `[0, capacity_map]` within `1e-6`.
- [x] Depletion equals fulfilled uptake within `1e-5` relative.
- [x] Two mouths in one cell split insufficient stock proportionally, inactive
      mouths consume zero, and body-node count does not change maximum demand.
- [x] Periodic boundary-crossing patches match translated interior patches.
- [x] Resource outside zero-capacity regions remains zero.
- [x] The frozen traveling wave moves at least `0.1` body length and at least
  `5x` the zero-wave drift in fluid.
- [x] Across eight paired seeds, sensory control improves mean/median uptake by
  at least 20%, with at least six seeds favoring sensory control.
- [x] Native renderer equivalence and full CPU suite pass.
- [x] Focused guarded GPU controls pass within the memory limit.

Rollback point: Phase 3 checkpoint. Stop before calibration if either positive
control fails.

### Phase 5: Make endogenous randomness identity-stable

Objective: preserve paired stochastic inputs when population rank or policy path
changes.

Dependencies: Phase 4.

Requirements:

- Define immutable integer RNG tags for initialization, environment, mutation,
  spawn, and future events.
- Derive mutation keys from world/step/child ID rather than ranked entry.
- Derive spawn angular-phase keys from child ID rather than event-array rank.
- Keep initialization and exogenous environment streams independent from
  endogenous birth paths.
- Document the final candidate-facing heredity seam as deferred to Master Phase 4.

Acceptance checkpoint:

- [x] Inserting/removing/reordering unrelated eligible births does not change
  mutation or spawn draws for the same immutable child ID.
- [x] Changing the private baseline mutation function does not alter initial or
  exogenous resource streams.
- [x] Same-stack seeded event traces reproduce bitwise.
- [x] Full CPU suite passes.

Rollback point: Phase 4 checkpoint.

### Phase 6: Replace benchmark methodology and close remediation

Objective: establish evaluator-shaped performance and long-run correctness
evidence before calibration.

Dependencies: Phases 1–5.

Requirements:

- Measure compile+first execution separately from warm jitted fixed-size
  `lax.scan` chunks.
- Compare equal total-node legacy and ecosystem cases with activity/fluid/config
  recorded.
- Report median and dispersion across repeated warm runs.
- Record peak memory when available without another dependency.
- Add a search-shaped chunk mode that returns only final state and compact
  metrics; full node/fluid histories are disabled unless sparse snapshots are
  explicitly requested.
- Add 1,000-step focused regressions covering finiteness, energy/resource bounds,
  identity monotonicity, event consistency, same-step reuse, mask isolation, and
  native visibility.
- Record evidence against the audit coverage matrix.

Acceptance checkpoint:

- [x] Benchmark reports JAX/device, shapes/config, compile seconds, warm episode
  seconds, simulated steps/second, baseline ratio, median, and dispersion.
- [x] A guarded 1,000-step fluid-on forced-turnover rollout remains finite within
  the 24 GiB guard.
- [x] Full CPU suite, scoped Ruff, focused GPU tests, and plan evidence audit pass.
- [x] Later optimization targets use the fused measurement only.

Rollback point: Phase 5 checkpoint. Do not weaken correctness gates to improve a
ratio.

## 12. User stories

### US-001: Preserve the verified baseline

**Phase:** Phase 0

**Description:** As a maintainer, I want a recoverable verified checkpoint so
that remediation cannot destroy native rendering, placement, or user work.

**Acceptance criteria:**

- [x] A named ref and evidence note reproduce the reviewed baseline.
- [x] Active plans consistently mark completed and deferred work.

### US-002: Conserve birth energy

**Phase:** Phase 1

**Description:** As a scientific reviewer, I want reproduction to transfer energy
so that birth cannot create it.

**Acceptance criteria:**

- [x] Every accepted transfer efficiency produces child energy no greater than
  successful parent debit.
- [x] Failed allocation leaves parent and total organism energy unchanged.

### US-003: Make actuation timestep-consistent

**Phase:** Phase 1

**Description:** As a simulation developer, I want actuation expressed as power
so that integration resolution does not redefine ecology.

**Acceptance criteria:**

- [x] Equal-duration fixed-control charges at `dt` and `dt/2` meet the declared
  tolerance.

### US-004: Validate ecosystem configuration

**Phase:** Phase 1

**Description:** As an experimenter, I want invalid settings rejected before JIT
so that failed runs are deterministic and understandable.

**Acceptance criteria:**

- [x] Boundary tests cover every validated numerical/configuration field.

### US-005: Use topology-aware steric exclusions

**Phase:** Phase 2

**Description:** As a mechanics developer, I want graph-derived exclusions so
that physical behavior does not depend on storage order.

**Acceptance criteria:**

- [x] Line/ring/branch neighborhoods are exact and permutation-invariant.

### US-006: Canonicalize activity masks

**Phase:** Phase 2

**Description:** As a JAX developer, I want one activity derivation per step so
that every solver agrees while legacy callers remain unchanged.

**Acceptance criteria:**

- [x] Ecosystem activity is canonicalized from population and inactive mechanics
  remain isolated in focused CPU/GPU tests.

### US-007: Use the compact fixed-trait controller

**Phase:** Phase 3

**Description:** As an evolution researcher, I want an interpretable phenotype
controller so that selection cannot exploit metabolic shortcuts.

**Acceptance criteria:**

- [x] Exact genome layout, strategy isolation, action bounds, and fixed-trait
  tests pass.

### US-008: Localize renewable resources

**Phase:** Phase 4

**Description:** As an ecology developer, I want spatial stock/capacity maps so
that movement changes acquisition.

**Acceptance criteria:**

- [x] Map bounds, conservation, periodic construction, and zero-capacity tests
  pass.
- [x] Mouth contention, inactive-mouth, and body-length-independent demand
  tests pass.

### US-009: Demonstrate embodied foraging

**Phase:** Phase 4

**Description:** As a scientific reviewer, I want positive controls for swimming
and sensing so that later heredity evaluation has a causal substrate.

**Acceptance criteria:**

- [x] Fluid movement and paired uptake gates pass exactly as specified.

### US-010: Key randomness by identity

**Phase:** Phase 5

**Description:** As an evaluator author, I want named identity-keyed streams so
that unrelated population paths do not reassign a child's randomness.

**Acceptance criteria:**

- [x] Rank/reordering invariance and stream-separation tests pass bitwise.

### US-011: Isolate trusted baseline mutation

**Phase:** Phase 3

**Description:** As a heredity integrator, I want controller decoding free of
mutation code so that the final policy seam can be added later without changing
phenotype semantics.

**Acceptance criteria:**

- [x] Controller code contains no mutation implementation, and seeded Gaussian
  behavior remains covered by a trusted internal test.

### US-012: Measure fused rollout performance

**Phase:** Phase 6

**Description:** As a performance reviewer, I want evaluator-shaped measurements
so that optimization decisions use the actual workload.

**Acceptance criteria:**

- [x] Compile and repeated warm fused-scan results contain every required field.

### US-013: Prove long-run invariants

**Phase:** Phase 6

**Description:** As a project owner, I want guarded long-rollout evidence so that
short passing tests are not mistaken for substrate readiness.

**Acceptance criteria:**

- [x] The forced-turnover rollout passes all finite-state, identity, event,
  resource, energy, mask, and visibility checks.

## 13. Functional requirements

- FR-1: Child initial energy shall be derived from reproduction debit and bounded
  efficiency.
- FR-2: Actuation energy shall be proportional to squared command and `dt`.
- FR-3: Invalid ecosystem configuration shall fail before tracing.
- FR-4: Population alive state shall be projected to node activity at ecosystem
  transition boundaries.
- FR-5: `Nodes.active=None` shall preserve legacy mechanics/rendering behavior.
- FR-6: Steric exclusions shall derive from graph neighborhoods.
- FR-7: Activity masks shall be derived once per physics step and shared by
  mechanics internals.
- FR-8: Synthetic-fluid tangents shall ignore inactive incident edges.
- FR-9: Genome layout shall contain exactly 20 phenotype plus three strategy loci.
- FR-10: Ecological traits shall remain outside the genome.
- FR-11: Ecosystem state shall retain stock in `Fields.energy` plus explicit
  capacity/regeneration maps.
- FR-12: Resource withdrawal shall be concurrent/proportional and mapped
  regeneration bounded, with one mouth demand per alive organism.
- FR-13: Scientific positive controls shall use fluid.
- FR-14: Mutation and spawn keys shall derive from named streams and child ID.
- FR-15: Controller code shall contain no mutation implementation.
- FR-16: Benchmarking shall separate compile from warm fused execution.

## 14. Non-functional requirements

- NFR-1: Identical inputs shall reproduce event traces bitwise on the pinned
  software/hardware stack.
- NFR-2: No non-finite value may enter trusted ecosystem state.
- NFR-3: All rollout shapes shall remain fixed by capacity/topology.
- NFR-4: Legacy no-context physics shall remain within `1e-7` of the checkpoint.
- NFR-5: Native rendering and placement behavior shall remain unchanged.
- NFR-6: No new third-party runtime dependency shall be added.
- NFR-7: Core package code shall not import experiment code.
- NFR-8: Focused GPU validation shall remain within the 24 GiB guard.
- NFR-9: Every phase shall be independently runnable and rollbackable.
- NFR-10: Search-shaped rollout chunks shall not retain full node/fluid history.

## 15. Technical considerations

### Activity and topology context

Keep the existing node contract:

```python
@dataclass
class Nodes:
    position: jax.Array
    velocity: jax.Array
    color_bend: jax.Array
    debug_vector: jax.Array
    active: jax.Array | None = None
    component_id: jax.Array | None = None  # remove only if unused after Phase 2
```

Add the smallest optional context required for topology exclusions:

```python
@dataclass
class PhysicsContext:
    steric_exclusion_indices: jax.Array  # int32[N, K]
    steric_exclusion_valid: jax.Array    # bool[N, K]
```

`physics_step(..., physics_context=None)` must accept all current positional
calls. Node/edge/bending activity derives from `nodes.active` once inside the
step and is passed to constraint/fluid functions. Do not store activity masks in
`EcosystemState`.

Do not use a dense `N x N` exclusion matrix. Build padded `N x K` neighborhoods
where `K` is the maximum graph-neighborhood size for the configured topology and
neighbor distance.

### Resource representation

Retain the flat state and existing field:

```python
@dataclass
class EcosystemState:
    nodes: Nodes
    edges: Edges
    fields: Fields  # fields.energy is current renewable stock here
    population: PopulationState
    time: jax.Array
    base_rest_lengths: jax.Array
    base_bending_rest_angles: jax.Array
    resource_capacity_map: jax.Array
    resource_regeneration_map: jax.Array
```

Use resource terminology in ecosystem code, telemetry, and documentation even
though the compatibility storage member remains `energy`.

### RNG derivation

Use immutable integer tags and repeated `jax.random.fold_in`; never Python
`hash()`:

```text
INITIALIZATION = 1
ENVIRONMENT    = 2
MUTATION       = 3
SPAWN          = 4
FUTURE_EVENT   = 5
```

Endogenous mutation/spawn keys incorporate timestep and child individual ID.
Exogenous environment keys never depend on birth count, parent rank, or policy.

### Deferred heredity boundary

Phase 3 may contain a private trusted function such as:

```python
def _fixed_gaussian_child(key, parent_genome, mutation_config): ...
```

Do not add `offspring_policy`, `ParentStats`, or `PopulationStats` here. The
[heredity and Shinka plan](./heredity-shinka-plan.md) adds the final public seam
only after embodied-foraging and sustained-turnover gates pass.

## 16. Rollout and rollback strategy

- Rollout: checkpoint the current verified tree, then land Phases 1–6 as small
  green commits in order.
- Monitoring: focused/full tests per phase; guarded GPU evidence at mechanics,
  embodied-foraging, and final checkpoints; benchmark artifacts at Phase 6.
- Rollback triggers:
  - native renderer or placement differs unexpectedly;
  - legacy no-context physics differs beyond tolerance;
  - energy/resource invariants fail;
  - active results depend on inactive bodies;
  - positive controls fail after the phase timebox;
  - RNG pairing depends on rank;
  - GPU workload exceeds the guard or becomes non-finite.
- Rollback action: return to the previous phase checkpoint, retain failure
  artifacts, and revise the affected contract before continuing.
- Do not weaken scientific gates or use destructive Git operations against the
  original dirty tree.

## 17. Success metrics

- Zero accepted configurations create organism energy at birth.
- Timestep-scaling error is at most `1e-5` relative plus `1e-6` absolute.
- Exact graph-neighborhood tests pass for line/ring/branch fixtures.
- Legacy no-context physics stays within `1e-7`; storage-permutation steric
  comparison stays within `1e-6` on the pinned CPU fixture.
- Fluid movement and paired acquisition positive-control gates pass.
- Child mutation/spawn randomness is invariant to unrelated rank changes.
- A guarded 1,000-step fluid-on forced-turnover rollout remains finite.
- Native rendering/placement equivalence and all full CPU tests remain green.
- Fused benchmark artifacts contain complete compile/warm/device/shape evidence.
- Every Appendix A item has an explicit completed, active, deferred, or protected
  disposition with no conflicting active instruction.

## 18. Open questions

- What fused ecosystem/baseline overhead is acceptable? Impact: later capacity
  and optimization work. Resolve after Phase 6 measurement; no prior Python-loop
  ratio is a gate.
- Should `component_id` remain as generic mechanical metadata after graph-based
  exclusions land? Impact: one optional pytree leaf. Resolve with the Phase 2
  usage audit; either result preserves `Nodes.active`.

Neither question blocks Phase 1.

## 19. Implementation readiness notes

- Risks: optional physics context touches solver internals; equivalence tests must
  precede topology behavior changes.
- Risks: fluid positive controls can expose parameter/sign errors; failure is a
  stop condition, not permission to return to no-fluid evaluation.
- Dependencies: reviewed checkpoint, Conda `sakana`, existing dependencies only.
- Suggested story order:
  `US-001 -> US-002 -> US-003 -> US-004 -> US-005 -> US-006 -> US-007 ->
  US-011 -> US-008 -> US-009 -> US-010 -> US-012 -> US-013`.
- Definition of done: Phases 0–6 and Appendix A are evidenced; sustained-turnover
  calibration can start without energy, timestep, topology, embodied-selection,
  RNG, or benchmark ambiguity.

## Appendix A: audit disposition matrix

| Audit item | Disposition | Evidence/owner |
|---|---|---|
| Birth configuration can create energy | Completed | Phase 1 tests and commit `b497b64` |
| Actuation depends on numerical timestep | Completed | Equal-duration tests and commit `b497b64` |
| Periodic parent mean spawned remotely | Completed/protected | Native placement implementation and tests |
| Initial/newborn overlap | Completed/protected | Body-aware candidate placement and reservation tests |
| Steric exclusions use storage order | Completed | Phase 2 graph/permutation tests and commit `c972663` |
| Ring/branch topology semantics are overstated | Completed/restricted | Generic mechanics exact; ecosystem controller rejects non-line topology |
| Population/node activity can diverge | Completed | Boundary projection and inconsistency regression |
| Mask logic is rederived/scattered | Completed | One step-level derivation passed through mechanics |
| Partial masks affect synthetic tangents | Completed | Edge-masked tangent and `1e-7` reaction regression |
| `Fields.energy` has resource semantics | Deliberate compatibility choice | Capacity/regeneration maps and Phase 4 bound tests |
| Flat ecosystem state duplicates some legacy fields | Deliberate separate domain state | Protected; no nested-state rewrite |
| Keyword-heavy constructor lacks validation | Completed; API retained | Phase 1 boundary tests |
| `EcosystemEnv` overrides much of its base | Deliberate compatibility choice | Inheritance retained; action residual documented/tested |
| External action is additive | Completed/documented | Residual equivalence test; autonomous science passes `None` |
| Birth events are dict-based | Deferred | Master Phase 3 scenario/event typing |
| 61-gene MLP is opaque/over-scoped | Completed | Exact 23-locus controller tests and commit `f0b47b2` |
| Ecological traits evolve with genome | Completed | Fixed-trait regression |
| Mutation is coupled to controller | Completed | Private ecology baseline; controller has no mutation implementation |
| Final heredity policy is hard-wired | Deferred by dependency | Master Phase 4 heredity plan |
| Mutation/spawn randomness depends on rank | Completed | Identity-key tests and commit `cf34143` |
| Uniform resource/no-fluid makes movement irrelevant | Completed | Localized maps and guarded fluid controls |
| Ecosystem renderer bloated generic renderer | Completed/protected | Native renderer equivalence; specialized path deleted |
| Generic renderer buffers frames | Deferred generic concern | Outside lifecycle remediation |
| Python-loop benchmark is not evaluator-shaped | Completed | Fused benchmark and Phase 6 evidence |
| Short tests do not prove long-run invariants | Completed | Guarded 1,000-step turnover regression |

[/PRD]
