# Microcosmos CPPN-ecosystem implementation plan

This is the implemented substrate record. It started from the completed
lifecycle and installed the final CPPN/heredity architecture directly. It did
not calibrate or ship a temporary fixed-wave ecosystem first.

## 1. Preserve completed substrate work

Do not revisit these verified decisions unless a new test exposes a defect:

- flat fixed-capacity physical arrays;
- `PopulationState.alive` as occupancy authority;
- projected `Nodes.active` masks throughout physics;
- finite localized resources and mouth-only uptake;
- stored energy, basal/actuation cost, death, energy-transfer birth, and slot
  reuse;
- immutable individual IDs and identity-keyed RNG;
- body-aware spawn placement and native rendering;
- compact `run_ecosystem_chunk` and opt-in sparse snapshots.

The completed 23-locus wave controller and its tests remain a focused
physics/sensing regression oracle. The active `EcosystemEnv` does not contain a
controller-mode branch.

Completion evidence is in
[lifecycle-remediation-audit.md](../microcosmos/docs/lifecycle-remediation-audit.md).

## 2. Minimal file changes — complete

Add:

```text
src/microcosmos/cppn.py
src/microcosmos/heredity.py
tests/test_cppn_controller.py
tests/test_heredity.py
experiments/evo2_ecosystem/
```

Modify only where integration requires it:

```text
src/microcosmos/structs/population.py
src/microcosmos/gym/ecosystem.py
src/microcosmos/ecology.py
src/microcosmos/rollout.py
src/microcosmos/positive_controls.py
existing ecosystem config and focused tests
tests/test_positive_controls.py
experiments/ecosystem/run_positive_controls.py
experiments/ecosystem/experiment.py
```

Do not add:

- a generic controller protocol, registry, or enum;
- a second `EcosystemEnv`;
- `src/microcosmos/scenarios.py` or `ecosystem_eval.py`;
- copied TensorNEAT internals;
- archive/checkpoint loaders or runtime visualization;
- a synchronous NEAT population manager.

Before closing the migration, `rg` every call site of `GenomeConfig`,
`mutation_std`, `mutation_probability`, `genome_variance`, and old flat
`population.genome` assumptions. Remove vector mutation fields from the active
`EcosystemEnv` API instead of retaining compatibility branches.

## 3. `cppn.py`

### 3.1 Frozen TensorNEAT definition

Keep one private factory as the single source of truth for the forward
contract. Instantiate one module-level inference genome and, in
`heredity.py`, three compatible module-level mutation genomes for parametric,
structural, and mixed profiles:

```python
DefaultGenome(
    num_inputs=4,
    num_outputs=1,
    max_nodes=15,
    max_conns=30,
    init_hidden_layers=(1,),
    node_gene=DefaultNode(
        aggregation_options=[AGG.sum],
        aggregation_default=AGG.sum,
        activation_options=[ACT.identity, ACT.sin, ACT.tanh, ACT.abs],
        activation_default=ACT.identity,
        # activation ordering and bounds identical across every instance
    ),
    conn_gene=DefaultConn(
        # shape, attribute ordering, and bounds identical across instances
    ),
    output_transform=ACT.identity,
)
```

Use library activations only. Do not call the experiment utility that mutates
TensorNEAT's global activation manager with log/inverse/sawtooth functions.
Feed-forward topology and sum aggregation are immutable.

Mutation rates are Python-static in TensorNEAT. The compatible mutation
instances therefore share gene shapes, activation ordering, bounds, input/output
keys, and forward semantics, but differ in both `DefaultMutation` structural
rates and `DefaultNode`/`DefaultConn` value/activation rates. The structural
profile zeros all existing-gene value/activation rates; the parametric profile
zeros all add/delete rates; mixed keeps both. Raw genes remain interoperable,
and only the inference genome performs runtime forward/transform.

Use `DefaultConn`, not historical-marker connections. With asexual reproduction,
no crossover, and no speciation, connection innovation state cannot influence
heredity or selection. Avoiding it removes two global counters while retaining
TensorNEAT's topology mutation and endpoint-aligned distance diagnostics.

### 3.2 Heritable type

Register one pytree:

```python
@dataclass
class CPPNGenome:
    node_genes: jax.Array        # one: [15, 5], population: [C, 15, 5]
    connection_genes: jax.Array  # one: [30, 3], population: [C, 30, 3]
```

Use names that cannot be confused with physical filament nodes/edges.

An unused row is entirely NaN. Valid and unused rows may be interleaved because
TensorNEAT deletion leaves a NaN hole that a later addition can reuse.
Reset-time inactive slots hold the canonical graph. Dead slots may retain the
previous organism's valid genes/cache until reuse; `alive` alone controls
whether they act.

### 3.3 Canonical founder

Hand-construct a graph whose key layout follows `init_hidden_layers=(1,)`:

```text
inputs 0..3 -> sine hidden node 4 -> linear output node 5
```

Connections:

- coordinate and phase to the sine node, with weights forming a traveling wave;
- sine node to output, setting conservative bend amplitude;
- local resource and head-tail difference to output with small steering
  weights.

Input/output node attributes must be valid even though input activations and
the output activation are not used in the normal forward path. Initialize every
unused row to NaN.

Create the founder panel from a frozen seed and slot-indexed small perturbations
of safe biases, responses, and weights. Do not randomly initialize topology.
The same panel is materialized for every policy and world seed; founder IDs are
the initial slot identities.

### 3.4 Inputs and batched action

At every physics step build:

```text
observations [capacity, hinges_per_body, 4]
  0: normalized hinge coordinate in [-1, 1]
  1: phase = physical_time * 5.0 radians per simulation-time unit
  2: local resource in [0, 1]
  3: head resource - tail resource in [-1, 1]
```

Freeze the phase rate at 5.0, matching the released CPPN experiment's
0.05-radian step at `dt=0.01`, but permit CPPN connection weights to change
effective frequency and spatial phase.

Nested `vmap` applies:

1. one transformed controller to all hinges;
2. all fixed-capacity controllers to their hinge observations.

Contain disconnected/nonviable networks:

```python
raw = jnp.nan_to_num(raw, nan=0.0, posinf=RAW_LIMIT, neginf=-RAW_LIMIT)
action = max_bending_delta * jnp.tanh(raw)
action = action * alive_by_hinge
```

This keeps physics finite. A disconnected output may still produce constant
curvature through its finite output-node bias; that is legal and remains
subject to ecological selection rather than crashing the rollout.

### 3.5 Cached transforms

TensorNEAT `transform` returns topological order and a dense connection-index
lookup in addition to raw genes. Store only the derived arrays:

```text
controller_order             int32 [capacity, 15]
controller_connection_index  int32 [capacity, 15, 15]
```

Run transform:

- once for the fixed-capacity founder panel during environment construction;
- once for each actual newborn after trusted mutation.

Never transform every controller on every physics step. Never expose caches to
the heredity policy. Inference reconstructs TensorNEAT's transformed tuple from
the two cache arrays and raw genes.

### 3.6 Validation

Provide small pure/JAX-compatible validators for:

- exact node/connection shapes and float32 dtype;
- every row either fully populated or fully NaN, regardless of row order;
- populated rows fully finite and unused rows fully NaN;
- no infinity or partial-NaN row;
- input keys 0–3 fixed at rows 0–3 and output key 5 fixed at row 5; existence
  elsewhere is insufficient because TensorNEAT forward uses positional rows;
- unique integer-valued node keys;
- integer-valued aggregation and activation fields;
- sum aggregation and activation index in the frozen palette;
- unique connections whose endpoints exist;
- no feed-forward cycle;
- cache shape, sentinel range, and agreement with a fresh transform;
- bounded finite batched output after containment.

Canonicalize neutral attributes before validation: reset every input node's
unused bias/response/aggregation/activation fields to the frozen identity
values, and reset the ignored output activation. This prevents neutral drift
from consuming mutation and inflating compatibility diagnostics.

Full endpoint, duplicate, cycle, positional-I/O, and cache validation runs only
during environment construction for the deterministic founder panel and at
newborn creation. It is trusted containment, not a replacement for TensorNEAT's
own cycle and capacity checks.

## 4. Population-state migration

Change `PopulationState.genome` from a rank-2 float array to `CPPNGenome` and add:

```python
controller_order: jax.Array
controller_connection_index: jax.Array
founder_lineage_id: jax.Array
intake_ema: jax.Array
population_change_ema: jax.Array
```

Initialization:

- all slots receive canonical CPPN genes and a valid cache;
- living founders receive identity-keyed safe value variation;
- living founders have `founder_lineage_id == individual_id`;
- inactive slots have lineage -1 and zero ecological EMAs;
- population-change EMA starts at zero.

Lifecycle:

- child inherits its parent's founder lineage;
- child intake EMA starts at zero;
- death may leave genes/cache intact because alive masking is authoritative;
- every birth overwrites genes, cache, lineage, age, energy, and identity for
  that slot;
- intake EMA updates from fulfilled uptake before death;
- births see the current step's death-only population-change update, while the
  stored EMA is updated once from net natural deaths/births;
- experiment-side culls apply one explicit stored population-EMA update.

Do not add birth and death EMAs, scenario state, or an ecology-memory object.
Use the frozen coefficients, update order, action-diversity mask, and
initial-founder entropy denominator defined in the
[heredity plan](./heredity-shinka-plan.md#6-visible-statistics); do not create a
second implementation.

## 5. Direct `EcosystemEnv` integration

Replace the active call to the wave `controller_action` with the CPPN batched
action. Do not add a conditional backend.

Reuse existing values:

- `_bending_coordinate` for normalized body location;
- sampled normalized resource at bending nodes;
- head/tail resource summaries;
- `state.time * dt` for physical time;
- `population.alive` for the final mask.

Rename `EcosystemTelemetry.genome_variance` to `action_diversity`. Compute:

```text
mean over hinges of masked variance across living organisms
-----------------------------------------------------------
                  max_bending_delta squared
```

Clip to [0, 1]. This is cheap because bend commands already exist. Variable
topology makes flat gene variance meaningless. Graph complexity and
compatibility distance remain finalist diagnostics, not per-step telemetry.

Add only the other telemetry required by the experiment:

```text
operator_counts          int32 [4]
policy_violation_count   int32 scalar
infrastructure_valid     bool scalar
```

Add `maximum_generation` and `steps_at_capacity` to compact chunk metrics for
turnover calibration. Do not add a general metrics registry.

External actions, if retained for existing generic Gym tests, are added after
the CPPN action exactly as today. They are not exposed during Evo² evaluation.

## 6. Legal NaN padding and rollout validity

The current `rollout._all_finite` rejects every legitimate TensorNEAT genome
because padded rows contain NaN. Replace the single generic check with:

- strict `isfinite` over physics, fields, energy, ecological scalars, IDs, and
  caches;
- a cheap row-completeness/numeric check over padded gene arrays;
- strict finite controller output after containment.

Carry the infrastructure-valid result produced by full reset/birth graph
validation into chunk metrics. Do not repeat endpoint, duplicate, cycle, or
cache recomputation every physics step; that would invalidate the measured
performance model.

Test that:

- all-NaN padding is accepted;
- a partial-NaN active row is rejected;
- NaN/Inf in an active attribute is rejected;
- corruption in physical state is still rejected;
- disconnected but structurally valid CPPNs remain valid and produce finite,
  bounded output. The explicit zero-output control remains exactly zero.

Do not globally `nan_to_num` stored genes; that would turn padding into genes.

## 7. Final heredity seam in reproduction

The exact policy is specified in
[heredity-shinka-plan.md](./heredity-shinka-plan.md). The substrate integration
must be final on first implementation:

1. preserve current parent ranking, free-slot ranking, energy debit, child IDs,
   identity-key RNG, and O(capacity) scatter;
2. derive one opaque `MutationContext` per ranked child;
3. compute normalized parent/population statistics;
4. wrap policy, mutation, validation, and transform in
   `lax.cond(birth_count > 0, ...)`;
5. execute only the `birth_count` actual rows with a bounded scalar
   `lax.fori_loop`; vmapping the four-way switch would execute every mutation
   branch for every ranked row;
6. accept only actual birth rows;
7. clone and count a candidate violation when `policy_valid=false`; if trusted
   mutation yields an invalid graph, fall back for containment but raise a
   distinct infrastructure-failure flag;
8. scatter the two immutable `CPPNGenome` leaves directly;
9. scatter transformed caches and continue through existing spawn placement.

Fallback changes heredity only: the scheduled clone birth still pays the normal
parent cost, receives normal child energy and a new ID, and inherits lineage.

`MutationContext` contains an identity-keyed PRNG key and
`new_node_key = DYNAMIC_NODE_KEY_OFFSET + child_id`, with the offset frozen at
1024. Assert the result is below `2**24`, float32's exact-integer boundary.
TensorNEAT still asserts the connection-key ABI even though `DefaultConn` does
not consume the values. Pass exactly
`jnp.zeros((3,), dtype=jnp.float32)` and test it under eager/JIT/vmap.

The no-birth branch must not run policy, mutation, or transform work.

After the CPPN birth tests pass, delete the unused vector-only
`_fixed_gaussian_child` path and remove its mutation settings from the active
ecosystem configuration. Keep `controller.py` and any wave-specific config only
where the standalone regression fixtures require them; do not preserve a
second heredity implementation.

## 8. Statistics exposed to heredity

Trusted code constructs:

```python
ParentStats(
    energy_fraction=clip(energy / reproduction_threshold, 0, 2),
    intake_ema=clip(intake_ema, 0, 1),
    node_fraction=active_nodes / 15,
    connection_fraction=active_connections / 30,
)

PopulationStats(
    alive_fraction=alive_count / capacity,
    population_change_ema=clip(population_change_ema, -1, 1),
    action_diversity=clip(action_diversity, 0, 1),
    lineage_entropy=normalized_masked_entropy,
)
```

No raw IDs, graph arrays, topology cache, time, event identity, world identity,
future information, hidden seeds, or score enters candidate code.

## 9. CPPN-native ecological calibration — complete

Calibration ran only after Sections 3–8 and their gates passed, using the
canonical founder family and frozen mixed TensorNEAT profile throughout.

Sweep only:

- initial resource stock fraction and regeneration;
- uptake and assimilation;
- basal and actuation cost;
- reproduction threshold/cost and child endowment;
- maturity age and maximum lifespan;
- initial living population from the declared panel; the selected value is 8
  of 32.

Keep physics, body, grid, timestep, CPPN contract, founder panel, mutation
profile, capacity, and resource geometry fixed.

Freeze the first parameter set for which the declared calibration panel shows:

- births and natural deaths in a majority of worlds;
- generation 3 or later in a majority;
- neither near-certain early extinction nor permanent early saturation;
- multiple birth opportunities after burn-in;
- valid finite state and graph caches throughout.

The selected configuration is `v2_c00_balanced`, hash
`ceab22348b95db80ec8141628e9f7740a8753cf0448573d0a09bfe5eb072ee4d`:
initial stock 1.0, regeneration 0.03, uptake 1.0, assimilation 0.9, basal
cost 0.01, actuation coefficient 0.002, threshold/cost 3.0/1.5, child transfer
0.8, maturity 50, lifespan 5000, and 8 founders. The failed v1 panel and v2
amendment are retained. This remains viability calibration, not policy
optimization, and no policy receives different ecology.

## 10. Experiment-side catastrophes

Keep event scheduling outside `EcosystemEnv.step`:

```python
state, pre = run_ecosystem_chunk(...)
state, event_record = apply_event(state, scenario, event_key)
state, post = run_ecosystem_chunk(...)
```

Require event steps to be chunk boundaries.

### Resource relocation

- replace capacity and regeneration maps;
- initialize new stock from the frozen fraction of new capacity;
- preserve organisms, CPPNs, caches, energy, physics, IDs, lineage, and time.

### Random bottleneck

- one identity-keyed uniform per living organism;
- remove the frozen fraction independent of energy, genome, lineage, or slot;
- project `Nodes.active` immediately;
- update population-change EMA from before/after occupancy;
- do not reproduce inside the event transform.

### Capped dominant-lineage cull

- final only;
- target the most abundant living founder lineage with deterministic ties;
- cap removal at the frozen total-population fraction;
- choose capped victims with identity-keyed draws;
- update occupancy and population-change EMA exactly as for bottlenecks.

Null events are exact identity transforms at the matched boundary.

## 11. Compact evaluator

`experiments/evo2_ecosystem` owns ordinary frozen Python scenario dataclasses,
canonical JSON manifests, pure jitted event transforms, host chunk orchestration,
scoring, baseline runners, and dependency-light paired analysis.

The canonical manifest evaluator builds and compiles one environment, then
measures each full manifest exactly three times. It selects the coherent
median-scoring execution with stable tie breaking, retains that execution's
complete episode vectors, and requires integrity in all three measurements.
Callers do not add another repeat loop.

Within one measurement, worlds sharing `pair_id`, world seed, and event boundary
run `_prepare_pre_event` once and fork the resulting immutable state through
`_finish_world`. Development and sealed shock/null pairs therefore have exact
shared prehistory. Each training manifest contains one member per pair, so this
does not claim exact prehistory across separately executed policies or outer
arms.

Search retains:

- chunk boundaries and gross/normalized productivity;
- alive, birth, natural-death, and catastrophe-death counts;
- maximum generation and lineage summaries;
- action diversity and controller node/connection summaries at checkpoints;
- validity and policy-violation counts.

Only finalist mode retains fixed-shape birth/death ID edges and sparse
body/resource snapshots. Never retain dense fluid history during search.

## 12. Tests

### `tests/test_cppn_controller.py`

- canonical topology and all-NaN padding;
- founder/fresh-cache equality;
- eager/JIT/nested-vmap output equality;
- output bounds and inactive zero;
- traveling-wave positive control and zero-output no-drift control;
- sensor-enabled paired acquisition control;
- deterministic valid parametric/structural/mixed mutations;
- required I/O, capacity, endpoint, activation, and acyclicity invariants.

Every regression equality involving CPPN genes first requires identical NaN
masks, then compares populated values (or uses `equal_nan=True`). Ordinary
`array_equal`/`allclose` without NaN handling is invalid for padded genomes.

### Existing core tests to update

- `test_ecology.py`: clone, no-birth skip, child-only pytree/cache scatter,
  candidate-invalid versus infrastructure-invalid fallback, and
  energy/identity/lineage preservation;
- `test_ecosystem.py`: controller shapes, direct compiled path, occupancy mask,
  child cache;
- `test_rng_streams.py`: child-ID mutation invariance under slot/history changes;
- `test_rollout_chunk.py`: legal padding versus true numeric invalidity;
- `test_ecosystem_long_rollout.py`: structural descendants retain valid caches;
- fused benchmark: no-birth and birth-active CPPN runs.

### Experiment tests

- exact relocation/null/bottleneck/cull fixtures;
- identity-key catastrophe invariance;
- exact productivity score;
- manifest validation and hashing;
- exact pre-event state reuse and shock/null fork semantics;
- three-measurement coherent median selection, stable ties, and all-repeat
  integrity failure propagation;
- finalist lineage traversal after slot reuse, when a positive finalist makes
  the common-garden claim applicable.

## 13. Guarded performance gates

Use 32 organisms × 8 body nodes × 64 × 64 fluid grid:

- cached CPPN 100-step warm rollout ≤1.75× the frozen wave regression;
- 32-child warm mutation batch ≤5 ms;
- first mutation compile ≤15 s;
- cached and fresh transform outputs agree to `1e-6`;
- a guarded 1,000-step fluid-on rollout with births stays below 24 GiB and has
  no invalid state.

Profile compilation separately from warm execution. Run a fresh GPU process for
each focused workload.

## 14. Current handoff

The direct CPPN migration, trusted heredity seam, controls, GPU gates,
calibration, event protocol, evaluator, manifests, and baseline runner are
complete. Robust three-measurement evidence now confirms catastrophe disruption
and separation among trusted policies. The stable production search completed;
the clean punctuated search follows after an external proposal-service limit
resets. Do not add another controller, lifecycle, scenario framework, or
fidelity path.

## 15. Stop conditions

Stop and amend the architecture if:

- valid cached CPPNs exceed the performance or memory gate;
- legal NaN padding cannot be contained without weakening physical validity;
- TensorNEAT mutation cannot preserve fixed shapes or feed-forward validity;
- viable founders cannot pass locomotion and sensing controls;
- turnover requires policy-specific ecology, free resources, or altered physics;
- event scheduling requires per-step scenario branches;
- candidate evaluation requires dense trajectory retention.
