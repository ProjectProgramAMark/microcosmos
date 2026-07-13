# CPPN heredity and ShinkaEvolve implementation plan

The trusted heredity seam and bounded Shinka task are implemented as part of
the direct CPPN architecture. They are not adapters around a temporary fixed
controller.

## 1. Scientific surface

Shinka may edit exactly one four-argument function:

```python
def make_offspring(
    parent_genome,      # safe [node_fraction, connection_fraction] summary
    parent_stats,       # safe [energy_fraction, intake_ema] summary
    population_stats,  # four safe population summaries
    rng,                # opaque reserved argument; reading it is rejected
):
    return jnp.array([clone, parametric, structural, mixed], dtype=jnp.float32)
```

The historical first-argument name is retained for the task ABI, but raw genes
never cross the boundary. Trusted adapter code applies the returned scores to
the real parent and opaque mutation context. The function does not author a
CPPN. It may choose which trusted
NEAT-compatible heredity profile to apply as a function of ecological state.
This is the smallest surface that:

- lets Shinka discover stress/diversity/complexity-dependent exploration;
- preserves parametric, activation, and topology evolution;
- blocks the trivial exploit of emitting one hard-coded successful controller;
- reuses TensorNEAT rather than maintaining a parallel graph mutator.

Adaptive-scheduling language is used only if the champion's realized operator
choice varies with ecological state and survives ablation. A constant champion
supports only the narrower selected four-operator scheduler claim. Neither case
is arbitrary mutation-algorithm synthesis.

## 2. Core types

Add registered pytrees in `src/microcosmos/heredity.py`:

```python
@dataclass
class ParentStats:
    energy_fraction: jax.Array
    intake_ema: jax.Array
    node_fraction: jax.Array
    connection_fraction: jax.Array


@dataclass
class PopulationStats:
    alive_fraction: jax.Array
    population_change_ema: jax.Array
    action_diversity: jax.Array
    lineage_entropy: jax.Array


@dataclass
class MutationContext:
    key: jax.Array
    new_node_key: jax.Array


@dataclass
class OffspringResult:
    genome: CPPNGenome
    policy_valid: jax.Array
    operator_index: jax.Array
```

`MutationContext` is intentionally opaque to candidate code. Its explicit name
prevents generated code from mistaking it for a raw JAX key. AST validation
forbids attribute access, indexing, comparison, or arithmetic on it.

The `policy_valid` bit records whether operator scores were finite. Returning
this tiny trusted outcome avoids side effects and makes runtime violations
auditable without exposing graph internals.

## 3. One trusted `mutate_cppn` primitive

```python
def mutate_cppn(
    parent: CPPNGenome,
    operator_scores: jax.Array,  # exact shape [4]
    ctx: MutationContext,
) -> OffspringResult:
    ...
```

Behavior:

1. require static shape `(4,)` and float dtype;
2. set `policy_valid = all(isfinite(operator_scores))`;
3. if invalid, return the unchanged parent immediately, `policy_valid=false`,
   and operator index `CLONE`;
4. otherwise clip scores to the frozen range [-20, 20];
5. choose `argmax(scores)` with deterministic lowest-index tie breaking;
6. use `jax.lax.switch` to execute one trusted profile with `ctx.key`;
7. return raw child genes, validity, and selected profile.

Use deterministic `argmax` rather than sampling the profile. Each mutation
profile is already stochastic. Deterministic scheduling makes the operator
decision exact and simplifies diagnosis; it does not make the complete GPU
ecology bitwise reproducible.

For `policy_valid=false`, the scheduled birth still occurs as an exact clone
with normal parent energy debit, child endowment, new child ID, and inherited
lineage. Trusted reproduction records a candidate violation and ultimately
marks that candidate evaluation `correct=false`. A graph-invalid child follows
the same clone containment so the rollout stays well formed, but it is
classified as a trusted implementation failure, not a candidate violation,
because candidates cannot construct graphs. Accepted mutated children are
transformed once and their selected profiles are recorded.

## 4. Frozen mutation profiles

All profiles use compatible copies of the same 4-input/1-output
`DefaultGenome` definition and the same safe activation ordering, bounds,
capacity, gene shapes, input/output keys, and sum aggregation.

TensorNEAT keeps structural rates on `DefaultMutation` but weight, bias,
response, and activation rates on `DefaultNode`/`DefaultConn`. Create one
canonical inference genome plus three module-level mutation-only genome
instances. Parametric, structural, and mixed instances differ in both operation
and gene mutation configuration as required, while their raw genes remain
interoperable. Do not reconstruct these Python-static objects inside JIT.

Freeze the pinned TensorNEAT defaults explicitly:

| Setting | Parametric | Structural | Mixed |
|---|---:|---:|---:|
| Weight/bias/response mutate rate | 0.20 | 0 | 0.20 |
| Weight/bias/response mutate power | 0.15 | 0 | 0.15 |
| Weight/bias/response replace rate | 0.015 | 0 | 0.015 |
| Activation replacement rate | 0.10 | 0 | 0.10 |
| Add/delete connection probability | 0 / 0 | 0.20 / 0.20 | 0.20 / 0.20 |
| Add/delete node probability | 0 / 0 | 0.10 / 0.10 | 0.10 / 0.10 |

Aggregation replacement is zero because sum is the only aggregation. Weight,
bias, and response bounds remain the pinned defaults of [-5, 5].

After every non-clone profile, trusted code canonicalizes all unused input-node
attributes and the ignored output activation before validation. Compatibility
diagnostics therefore measure functional or structurally relevant variation,
not neutral TensorNEAT fields.

### 4.1 Clone

Return the parent genes unchanged.

### 4.2 Parametric

Use TensorNEAT's standard value mutation for:

- connection weights;
- node biases;
- node responses;
- activation replacement within the frozen palette.

Set add/delete node/connection rates to zero.

### 4.3 Structural

Use TensorNEAT's feed-forward topology mutation:

- add node by splitting a connection;
- delete a non-input/non-output node;
- add a cycle-safe connection;
- delete a connection.

Set weight, bias, response, and activation mutation rates to zero. Freeze
`DYNAMIC_NODE_KEY_OFFSET = 1024` and use
`new_node_key = DYNAMIC_NODE_KEY_OFFSET + child_id`, asserting the result is
below `2**24`. With `DefaultConn`, TensorNEAT ignores the supplied
connection-key values but still asserts their ABI; pass exactly
`jnp.zeros((3,), dtype=jnp.float32)`.

### 4.4 Mixed

Use the standard parametric and structural settings together. This is the
canonical conventional baseline and the seed Shinka policy.

Use TensorNEAT's current default mutation rates/powers unless the Phase A
founder/mutation controls expose a mechanical incompatibility. Any necessary
adjustment is made once during those controls, recorded, and frozen before
ecological calibration. Never tune profiles against ecological outcomes or
separately by treatment.

### 4.5 Why no custom distributions

Do not implement Student-t noise, per-module wave mutation, self-adaptive
strategy loci, arbitrary sparse masks, or a second value-mutation kernel. Those
belong to the retired vector controller and would enlarge the code and
validator without answering the CPPN scheduling question.

## 5. Lifecycle injection

`EcosystemEnv` holds one static `offspring_policy` callable. The default is the
trusted fixed-mixed policy.

Inside `reproduction_step`:

1. retain current parent/free-slot ranking, energy debit, child IDs, and
   identity-keyed randomness;
2. compute one `PopulationStats` and ranked `ParentStats` inside a
   `lax.cond(birth_count > 0, ...)` branch;
3. construct one `MutationContext` per ranked child;
4. use a scalar `lax.fori_loop(0, birth_count, ...)` so only real births and
   only each selected mutation branch execute. Do not `vmap` the four-way
   `lax.switch`: JAX lowers that form to selects and executes every branch;
5. process exactly the rows below `birth_count`;
6. combine `policy_valid` with trusted graph validation;
7. for either fallback, complete the scheduled birth with normal energy,
   identity, and lineage semantics; count candidate-invalid scores and classify
   graph-invalid trusted output separately as infrastructure failure;
8. transform valid child graphs and scatter genes/cache through the existing
   O(capacity) path;
9. emit operator counts and policy violations in compact telemetry.

No-birth and full-capacity steps do not invoke the candidate or TensorNEAT
mutation/transform.

Candidates cannot influence parent selection, slot allocation, child IDs,
energy transfer, body placement, lineage, physics, or scores.

## 6. Visible statistics

Trusted normalization:

```text
energy_fraction       clip(parent energy / reproduction threshold, 0, 2)
intake_ema            [0, 1]
node_fraction         active node rows / 15
connection_fraction   active connection rows / 30
alive_fraction        live slots / capacity
population_change_ema [-1, 1]
action_diversity      normalized bend-command variance [0, 1]
lineage_entropy       masked founder entropy / log(max(initial founders, 2))
```

Freeze `INTAKE_EMA_ALPHA = 0.05` and
`POPULATION_EMA_ALPHA = 0.10`. For one physics step:

1. let `N0` be living count at step start;
2. compute `uptake_fraction_i = clip(actual_uptake_i /
   max(uptake_rate_i * dt, eps), 0, 1)`;
3. update each step-start living organism with
   `intake_ema_i' = (1 - 0.05) * intake_ema_i +
   0.05 * uptake_fraction_i`;
4. after energy/death, let `Nd` be survivor count and form the transient value
   seen by births:
   `population_change_for_birth = clip((1 - 0.10) * stored_ema +
   0.10 * (Nd - N0) / capacity, -1, 1)`;
5. compute action diversity from this step's bend commands using only those
   `Nd` survivors; it is zero for fewer than two survivors and otherwise is
   the mean masked per-hinge population variance divided by
   `max_bending_delta**2`;
6. compute lineage entropy from survivor founder frequencies, divided by
   `log(max(initial_founder_count, 2))`; the denominator never shrinks when
   lineages disappear;
7. after normal births, let `N1` be final count and store
   `population_change_ema' = clip((1 - 0.10) * stored_ema +
   0.10 * (N1 - N0) / capacity, -1, 1)`.

Children start with zero intake EMA. An experiment-side cull with counts
`N_before` and `N_after` applies the same one-time stored update using
`(N_after - N_before) / capacity`. Resource relocation does not directly edit
an EMA; its effect appears through subsequent intake.

This is enough to express conservative mutation in stable conditions, more
structural exploration during decline, complexity-aware mutation, and
diversity restoration.

Do not add a precomputed “stress” scalar. Do not expose age, generation,
timestep, scenario/event identity, raw IDs, future state, raw graphs, hidden
seeds, or evaluation metrics.

## 7. Required baselines

Every baseline calls the same trusted `mutate_cppn` primitive with identical
founders, contexts, graph bounds, lifecycle, and worlds.

### Clone

Always select index 0. This measures adaptation from standing variation and
ecological sorting without new heritable change.

### Fixed parametric

Always select index 1. This measures weight/bias/response and categorical
activation mutation without topology evolution.

### Fixed mixed

Always select index 3. This is the principal conventional TensorNEAT anchor and
the initial Shinka program.

### Hand-written stress scheduler

Freeze one transparent formula:

```python
stress = jnp.clip(
    0.45 * (1.0 - population_stats.alive_fraction)
    + 0.35 * jnp.maximum(0.0, -population_stats.population_change_ema)
    + 0.20 * (1.0 - parent_stats.intake_ema),
    0.0,
    1.0,
)

operator_scores = jnp.array([
    0.25 - stress,                  # clone while very stable
    0.20 - jnp.abs(stress - 0.35),  # parametric at modest stress
    stress - 0.75,                  # structural during severe stress
    0.20 - jnp.abs(stress - 0.65),  # mixed at intermediate/high stress
])
```

Freeze the formula before Shinka. If fixtures show a tie or unreachable branch,
fix it before search and record the final code; do not tune against sealed data.

### Shinka treatments

- stable-world-trained scheduler;
- relocation/bottleneck-trained scheduler.

Structural-only is an ablation/sanity policy, not a required final-table row.

## 8. Shinka task layout

Keep the candidate task small:

```text
ShinkaEvolve/examples/evo2_ecosystem/
  initial.py
  evaluate.py
  run_evo.py
  run_spec.py
  run_spec.json
  run_spec.sha256
  freeze_finalist.py
  README.md
```

Trusted simulator/evaluator code remains in the pinned Microcosmos checkout.
Do not copy physics, mutation profiles, manifests, or scores into the task.

`run_evo.py` is one launcher parameterized only by the trusted regime selector
and an explicit matching resume request. Model, 15-generation budget, archive,
outer seed, evaluator timeout, numerical repeat count, manifest/source hashes,
top-three selection, and pinned headless command live in one canonical immutable
run specification shared by both arms. A fresh launch refuses a nonempty arm
directory. The evaluator resolves the corresponding visible training manifest
itself; it does not accept a path. Use one evaluation worker.

Launch the Shinka run itself inside the standard 20 GiB/24 GiB systemd scope so
its evaluator child inherits the memory guard. Do not launch multiple GPU
workers inside one scope.

`initial.py` has immutable imports/prefix/suffix and one evolve block:

```python
import jax.numpy as jnp

# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome,
    parent_stats,
    population_stats,
    rng,
):
    return jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
# EVOLVE-BLOCK-END
```

The seed is the fixed mixed baseline, so every program lineage starts valid and
the search must improve a conventional scheduler over trusted TensorNEAT
operators rather than repair a broken controller.

## 9. Static validator

The evolve block is a deliberately small expression language implemented with
Python syntax.

Require:

- exactly one marker pair;
- immutable prefix/suffix hashes;
- exactly one `make_offspring` with four positional arguments and no defaults,
  varargs, decorators, annotations requiring new imports, or nested definitions;
- one or more direct return paths whose traced shape is exactly `(4,)`;
- no read or reassignment of the opaque `rng` argument.

Allow:

- local scalar/array assignments;
- arithmetic and comparisons over whitelisted stats when comparisons feed a
  whitelisted JAX operation;
- `jnp.array`, `jnp.clip`, `jnp.where`, `jnp.minimum`, `jnp.maximum`,
  `jnp.abs`, `jnp.asarray`, `jnp.exp`, `jnp.log`, `jnp.sqrt`, `jnp.stack`, and
  `jnp.tanh`.

Reject:

- imports, classes, decorators, nested functions, Python `if`/ternary control
  flow, loops, comprehensions, recursion, exceptions, mutation statements, or
  dynamic execution;
- `jax.random`, raw `jax.lax`, host callbacks, NumPy RNG, filesystem, network,
  subprocess, reflection, or environment access;
- any attribute access on task inputs and any read of `rng`; indexing the three
  safe summary arrays is allowed;
- constructors for `CPPNGenome`, `MutationContext`, or `OffspringResult`;
- private/dunder attributes;
- more than 12,000 source bytes, 60 nonblank source lines, or 1,024 AST nodes.

Keep the allowlist and validator helpers in trusted `evaluate.py` rather than
adding a generic validation package. Cover them with focused tests. This is a
task-specific verifier, not a generic Python sandbox.

## 10. Dynamic preflight

Before an ecological episode:

- read and hash the candidate once, validate that exact AST, and compile/execute
  the validated bytes with empty builtins and only trusted `jnp` injected; do
  not reread the file through `importlib` after validation;
- scrub credential-bearing environment variables before candidate execution;
- call normal and boundary fixtures, including empty/full population and
  zero/high diversity;
- require exact `(4,)` finite numeric score output;
- run eager, `jit`, and `vmap` paths;
- replay identical inputs/context and require identical output;
- enforce compile/runtime limits;
- run the seed through the full training manifest before search.

Hash candidate source, evaluator source, and the trusted simulator tree before
evaluation and verify the same hashes afterward.

Invalid candidates write `correct=false`. Ecological extinction remains
`correct=true` and scores zero after extinction.

## 11. Evaluator contract

`evaluate.py` accepts Shinka's `--program_path`, `--results_dir`, and trusted
`--training_regime`. It validates the candidate, resolves only the corresponding
visible training manifest, calls the same direct episode evaluator as baselines,
and writes:

```json
{
  "combined_score": 0.0,
  "public": {
    "score": 0.0,
    "survival_rate": 0.0,
    "mean_final_alive": 0.0,
    "mean_births": 0.0,
    "clone_fraction": 0.0,
    "parametric_fraction": 0.0,
    "structural_fraction": 0.0,
    "mixed_fraction": 0.0
  },
  "private": {
    "candidate_sha256": "...",
    "manifest_sha256": "...",
    "simulator_config_sha256": "...",
    "evaluator_source_sha256": "...",
    "simulator_source_sha256": "...",
    "integrity_valid": true,
    "policy_violation_count": 0,
    "numerical_repeats": 3,
    "repeat_scores": [0.0, 0.0, 0.0],
    "selected_repeat_index": 0
  }
}
```

and:

```json
{"correct": true, "error": ""}
```

Public feedback may include only whole-training-manifest score, survival,
population/birth summaries, aggregate operator use, and numerical range. It
must not expose seeds, per-world or per-scenario traces, event parameters,
lineage details, development/sealed scores, or hidden manifests. Private
provenance also records backend and pinned software metadata.

Syntax, immutable-code, contract, JIT, timeout, policy-validity, or
infrastructure failures are invalid with distinct private codes.

## 12. Fidelity and compile gate

Every search candidate runs the full ecological horizon on the small frozen
training seed panel. Do not build a shortened-horizon fidelity system that may
select immediate rebound over delayed adaptation.

Reevaluate the top three candidates outside the archive on the larger development
panel. Cache by candidate-source and manifest hash.

Before search:

1. load the seed through the exact dynamic path;
2. run the full training manifest under the 24 GiB guard;
3. record validation, compile, warm evaluation, wall time, and peak memory;
4. require compile plus evaluation below 80% of candidate timeout;
5. repeat to separate compile from warm time.

Ordinary GPU replay is not bitwise reproducible because some GPU
scatter/reductions are nondeterministic. Strict deterministic XLA made even a
two-world panel exceed fifteen minutes and was rejected before search. The
canonical evaluator therefore measures the complete eight-world manifest
exactly three times, selects the coherent median-scoring full execution with
stable tie breaking, and requires integrity in all three. Callers invoke this
evaluator once; they must not wrap it in another repeat loop and accidentally
create nine measurements. The three measurements are nested numerical evidence,
not ecological replicates. No host-policy or shortened-horizon path is added.

Within development and sealed manifests, shock/null members with the same
`pair_id`, seed, and event boundary share one exact pre-event rollout in each
measurement and fork only at the event. Each training manifest contains one
member per pair, so separate stable and punctuated candidate evaluations are
seed-matched rather than exact trajectory forks.

## 13. Matched Shinka searches

Stable and punctuated configs match in:

- initial source, system/task prompt, model, temperature, and token cap;
- islands/archive/parent-selection settings;
- valid-candidate and total model/simulation budget;
- timeout, seed count, horizon, chunking, and score;
- top-three development reevaluation and champion-selection rule.

They differ only in training manifests:

- stable: null events at matched boundaries;
- punctuated: resource relocation or random bottleneck.

Run exactly one 15-generation outer search per treatment under the same
canonical run ID/specification. Reevaluate the top three from each archive on
development and select by
`development_score_desc_then_generation_asc`. Report conclusions about the two
selected schedulers; one stochastic outer run per arm cannot establish general
superiority of the training method. Every candidate starts from the same frozen
eight-founder CPPN panel, so the result is also conditional on that panel and
does not establish founder-distribution transfer.

## 14. Tests

### Trusted heredity

- each profile selected exactly by fixed scores;
- fixed context gives deterministic result;
- clone is byte-equivalent including NaN padding;
- parametric cannot alter topology;
- structural has zero explicit value/activation mutation rates, and attributes
  of genes that survive continuously are unchanged;
- over a frozen key panel, mixed can realize both value/activation and
  structural changes;
- every profile preserves input keys 0–3 at rows 0–3, output key 5 at row 5,
  capacity, endpoints, and acyclicity;
- neutral input attributes and output activation are canonical after mutation;
- invalid scores clone immediately and set `policy_valid=false`;
- no-birth path skips candidate and transform;
- child-ID keying is slot/order invariant;
- identical keys replay exactly; over a frozen key panel each trusted non-clone
  profile can realize variation without requiring every key to change a child;
- exact intake/population EMA equations and update order for no-death, death,
  death-plus-birth, and experiment-side cull fixtures;
- action diversity uses pre-birth survivors, and lineage entropy uses the
  frozen initial-founder denominator;
- candidate-invalid and graph-invalid fallbacks still apply the normal birth
  energy debit, child ID, and inherited lineage; the latter is classified as
  infrastructure failure.

### Validator/evaluator

- canonical seed accepted;
- raw/context access, opaque-argument use, imports, loops, constructors,
  wrong signatures, wrong score shape, NaN/Inf, and forbidden APIs rejected;
- eager/JIT/vmap replay;
- timeout cleanup;
- exact JSON schema and hashes;
- extinction distinguished from infrastructure failure;
- training task cannot resolve sealed manifests.

### Shinka smoke

- a 2–3 generation run stores candidates, scores, and program parent lineage;
- only the evolve block changes;
- invalid programs do not enter the correct archive;
- resume honors total-generation semantics;
- champion export is deterministic.

## 15. Current execution order

Trusted heredity, baseline policies, manifests, the canonical task, validator,
evaluator, launcher, training-only routing, and timeout cleanup are complete.
The remaining order is:

1. keep both hidden manifests mode `000`, regenerate robust conventional-policy
   and disruption evidence, then run the matched stable and punctuated searches
   sequentially, making the authenticated first-arm result tree mode `000`
   before the second proposer starts;
2. after both archives close, unlock development only, export the top three from
   each arm, select deterministically, and hash champions and archive lineage;
3. finalize analysis, preregistration, trusted-source hashes, actual budgets,
   and both freeze records while sealed remains unreadable;
4. unlock sealed once for the fixed noninteractive six-policy driver.

There is no raw-graph candidate stage, custom-mutation stage, or fixed-wave
outer search to remove later.

## 16. Security and stop conditions

AST checks and timeouts do not constitute a hostile-code sandbox. Run evaluator
processes after scrubbing credentials and under OS memory/time guards. Because
the read-only headless proposer can inspect files outside its result directory,
remove host read permission from development and sealed manifests before the
first proposal. Restore development only after both searches exit. Keep sealed
mode `000` through development selection and the complete freeze, then restore
it only for the trusted noninteractive final driver. A stronger security claim
would require a separate no-network/read-only container design.

Do not search if:

- candidate code can inspect genes/context or bypass `mutate_cppn`;
- mutation profiles or immutable task code can change;
- candidate-policy replay is nondeterministic beyond documented GPU ecological
  reduction-order variance;
- invalid candidates can be marked correct;
- stable/punctuated configs differ outside manifests;
- evaluation exceeds the frozen memory/time budget;
- feedback exposes sealed or post-hoc diagnostic information.
