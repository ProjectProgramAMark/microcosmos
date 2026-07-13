[PRD]
# PRD: Evo² r5 World-Feasibility Calibration and RSI Execution

Status: **Final — implementation-ready; scientific execution remains conditional on the frozen gates below**

## 1. Overview

Evo² r4 stopped before ShinkaEvolve because one of its two random initial
worlds placed every founder too far from the finite resource patch to begin a
viable lifecycle. The failure occurred before the heredity scheduler or the
actuation-cost disturbance could be evaluated.

Evo² r5 makes one change to the primary benchmark: it samples random worlds
conditional on a prospectively frozen, genome-independent initial-food-access
rule. It otherwise reuses the r4 physics, ecology, TensorNEAT controllers, six
trusted heredity operators, candidate ABI, disturbance, score, and search
budget. It also prospectively repairs the optional common-garden initialization
so that a later mechanism assay starts in a standardized accessible world; that
repair cannot affect search, finalist selection, or the primary result.

R5 is an experiment revision, not a new Microcosmos simulator direction. Its
qualification, manifests, gates, Shinka bindings, evidence, and analysis remain
under the Evo² experiment tree. Existing Microcosmos environments and their
default behavior must not change.

## 2. Problem Statement

The r4 founder screen evaluated 80 valid CPPN founders:

- seed `4101` produced 3–22 births per candidate;
- seed `4102` produced zero births for every candidate;
- seed `4101` began with a mouth 1.25 cells from the resource center, at 98.9%
  of local resource capacity;
- seed `4102` began with its nearest mouth 10.92 cells from the center of a
  radius-12 paraboloid, at 17.2% capacity, and then lost contact with food;
- a 256-seed geometry audit found that only 47.7% of random resets placed any
  live mouth in the patch's at-least-50%-capacity core;
- a disjoint 16-seed diagnostic found 5–17 births in all eight worlds passing
  that geometric rule, versus 0–1 births in the eight worlds failing it.

R4 therefore validated the implementation and integrity checks through founder
screening, but it did **not** test whether ShinkaEvolve could improve heredity.
The experiment needs ecologically usable initial conditions before any claim
about mutation policy is possible.

## 3. Outcome Hypothesis and Estimand

If worlds are selected before search using only initial resource geometry, then
the founder gate will measure founder viability rather than accidental lack of
food access. If all later gates pass, the resulting benchmark can test whether
a ShinkaEvolve descendant improves shocked productivity relative to the exact
heredity program from which its code lineage began.

The r5 estimand is explicitly conditional:

> Performance in randomly placed Microcosmos worlds conditional on at least
> one initially living mouth sampling a resource-capacity fraction of at least
> 0.50 at reset.

R5 does not claim generalization to arbitrary unqualified initial placements.
The qualifier will report the complete ascending selection trace and number of
seeds inspected before the second passing seed in each range. The previously
reported 256-seed diagnostic, not this stopping-rule-biased trace, remains the
descriptive estimate of qualification frequency.

## 4. Goals

- Select fresh founder, training, development, and sealed world seeds using one
  frozen genome- and rollout-independent rule.
- Produce an authenticated founder bank with 4 training, 4 development, and 8
  sealed CPPN founders without changing the ecology.
- Re-run the r4 disturbance and learnable-operator-opportunity gates under the
  qualified world distribution.
- Run one matched stable-objective and one matched punctuated ShinkaEvolve
  search, each with 50 noninitial generation slots, if and only if all
  prerequisites pass.
- Evaluate the punctuated finalist against the exact initial ancestor on sealed
  shocked worlds using the frozen r4 primary metric.
- Preserve every frozen r1–r4 experiment directory, artifact, and negative
  result unchanged, and preserve all legacy/r4 default behavior.
- Add no experiment-specific behavior to the general Microcosmos API.

## 5. Scope Boundaries

### In Scope

- One new experiment revision: `evo2-r5-world-feasibility-v1`.
- One reset-only, genome-independent food-access qualifier.
- Fresh disjoint world ranges and a fresh, prepartitioned founder panel.
- One small `experiments/evo2_ecosystem/r5/` package that composes r4 helpers.
- Schema-v4 routing in the ShinkaEvolve Evo² example so trusted r5 tools are
  bound explicitly instead of through the hard-coded r4 tool directory.
- One thin r5 workflow that performs preflight, gates, search handoff, finalist
  selection, final evaluation, and publication by calling existing functions.
- Full execution through each gate that becomes eligible.
- Conditional common-garden and mechanism-ablation analysis after a positive
  primary performance result.

### Non-Goals

- Changing fluid physics, constraints, resource dynamics, resource geometry,
  metabolism, reproduction cost, population capacity, body topology,
  controller inputs, or the actuation-cost disturbance.
- Moving founders toward food during the primary experiment, enlarging food,
  increasing starting energy, lowering birth costs, or weakening gates.
- Adding a general `ensure_food_access` option to `EcosystemEnv`.
- Adding self-assembly, sexual reproduction, predators, new catastrophes, more
  sealed worlds, or a new controller representation.
- Adding new trusted heredity operators or expanding candidate authority.
- Creating another copied `heredity_adaptation_v5` implementation tree.
- Adding a generic workflow, artifact, authorization, or provenance framework.
- Treating the single stable-versus-punctuated outer-run contrast as a causal
  estimate of curriculum effects.
- Packaging all Evo² research code as one upstream Microcosmos pull request.

## 6. Research Snapshot

Date: 2026-07-13

- Source: `experiments/evo2_ecosystem/heredity_adaptation_v4/founder_screening.jsonl`.
  Finding: 80/80 candidates retained integrity; 0/80 passed because seed 4102
  produced no births for any genome.
- Source: r4 read-only GPU replay and geometry diagnostics performed on
  2026-07-13. Finding: the at-least-0.50 initial-access rule separated healthy
  and unhealthy diagnostic worlds without using genome or outcome data in the
  rule itself.
- Source: `docs/evo2/evo2-actuation-cost-rsi-r4-preregistration.md`. Finding:
  r4 already freezes the candidate ABI, action registry, disturbance family,
  score, gate thresholds, search budget, finalist rules, and sealed success
  criteria reused here.
- Source: `src/microcosmos/structs/nodes.py` and `src/microcosmos/simulate.py`.
  Finding: the existing ecosystem extension is opt-in; legacy calls retain
  optional activity/context defaults. The 100 tests present on upstream `main`
  pass against the current branch on CPU, although undocumented external PyTree
  serialization remains outside that guarantee.
- Source: repository `AGENTS.md`. Finding: all Python work must use Conda
  environment `sakana`; full suites run on CPU; focused GPU work must run under
  the 24 GiB systemd/cgroup guard.

Assumption: the diagnostic relationship between initial access and lifecycle
viability transfers to the fresh r5 ranges. The prospective founder gate, not
the diagnostic data, is authoritative.

## 7. Constraints and Protected Invariants (DO NOT CHANGE)

- DO NOT CHANGE any r1, r2, r3, or r4 frozen source directory, evidence,
  manifest, founder, result, database, hash, lineage, or stop report.
- DO NOT CHANGE the r4 simulator configuration, resource patch, metabolism,
  reproduction, actuation-cost shock, score, action registry, mutation
  profiles, or statistical thresholds.
- DO NOT CHANGE the candidate contract. It receives four bounded numeric
  arrays plus an opaque unreadable `rng` compatibility argument and returns six
  finite `float32` logits.
- DO NOT expose scenario, partition, event, seed, mutation-key, raw-genome,
  filesystem, network, subprocess, development, or sealed information to
  candidate code.
- DO NOT use births, rewards, survival, controller behavior, founder identity,
  or post-reset outcomes to qualify world seeds.
- DO NOT inspect fresh `[8000, 12000)` seeds until all r5 implementation code,
  tests, protocol text, and critical hashes are committed and pushed.
- DO NOT continue after a failed seed, founder, disturbance, or operator-
  opportunity gate. Preserve the failure as the completed r5 result.
- DO NOT exceed 50 noninitial generation slots per Shinka arm. Invalid,
  duplicate, timed-out, or otherwise failed proposals consume their assigned
  slot; there is no performance-based early stop or free replacement.
- DO NOT add r4/r5 conditionals or operator-credit details to additional
  general-purpose Microcosmos modules.
- DO NOT copy r4 modules merely to change constants. Import seed-independent
  helpers and write only the r5-specific composition code.

The only permitted `src/microcosmos/` change is, if required, a generic
keyword-only PRNG-key input to `initialize_cppn_population`. Its default must
produce byte-identical legacy and r4 populations. It must not mention r5,
screening, partitions, or experiment policy. If the fresh panel can be produced
without this change and without duplicating generator logic, make no `src/`
change at all.

## 8. Frozen Decisions

### 8.1 World-feasibility rule

For a seed and the frozen `SimulatorConfig()`:

1. Call the trusted `EcosystemEnv.reset` placement path.
2. Read the eight initially living mouth positions.
3. Convert each position using the exact rounded periodic cell calculation used
   by `sample_grid_nearest` and `resource_step`.
4. Sample `state.resource_capacity_map` at those cells.
5. Divide by configured peak resource capacity.
6. Define `access_score` as the maximum fraction across living mouths.
7. Accept if and only if `access_score >= 0.50`.

The qualifier accepts no genome and performs no physics step.

CPU is the canonical qualification backend. The selected rounded mouth cells
must match on CPU and GPU, and access scores must agree at `atol=1e-6`; only the
canonical CPU score determines selection if a backend differs within that
tolerance.

### 8.2 Seed ranges and selection

Use the first two passing seeds in ascending order from each half-open range:

```text
founder eligibility: [8000, 9000)
training:             [9000, 10000)
development:          [10000, 11000)
sealed:               [11000, 12000)
```

Evaluate seeds in ascending order on CPU, stop a range immediately after its
second pass, and publish one canonical public `world_qualification.json`
containing the frozen rule, complete inspected prefixes, selection flags,
selected panels, configuration hash, implementation commit, critical qualifier
hash, JAX version, and backend. Seed identities are public because the rule and
ranges make them reproducible. Holdout protection applies to evaluation access
and outcomes, not to pretending deterministic seed IDs are secret.

### 8.3 Founder panel

- Panel seed: `5005`.
- Generate this explicit panel on the canonical CPU backend, then transfer the
  small frozen genome batch to GPU for screening. This prevents backend-specific
  normal-sampler roundoff from naming different founders while leaving the
  original no-argument Microcosmos initializer unchanged.
- Candidate ranges and quotas remain the r4 prepartitioned `24/24/32` and
  `4/4/8` design.
- Candidate generation is controlled through an experiment-local
  `ScreeningSpec` in `experiments/evo2_ecosystem/r5/protocol.py` (or the
  existing experiment-layer equivalent), not an `r5=True` branch or a new core
  screening type.
- Default/explicit r4 panel generation must remain byte-identical.

### 8.4 Worlds and disturbance

- Training event steps: `(3500, 4000)`.
- Development event steps: `(3000, 4500)`.
- Sealed event steps: `(2500, 4500)`.
- Multiplier qualification order: `1.25`, `1.50`, `2.00`, `3.00`.
- Use the first multiplier passing every frozen r4 harm, viability, feedback,
  and observable-stress rule.
- Do not add a new catastrophe family. Sealed transfer means unseen qualified
  founders, worlds, mutation keys, and event timing under the same disturbance.
- Omit optional magnitude-transfer testing in r5. The primary uses only the
  first qualified multiplier.

### 8.5 Search and inference

- One stable-objective arm and one punctuated arm.
- Both execute the same paired sham/shock simulations and compute limits. The
  stable archive sees sham objective/feedback only; shock computation remains
  private and is performed solely for matched compute.
- Each arm contains one exact initial program and 50 noninitial generation
  slots. This is a slot budget, not a promise of 51 valid programs or exactly
  50 model API calls.
- Report actual model calls, retries, valid programs, invalid programs,
  duplicates, and timeouts. Report tokens and cost only when existing Shinka
  logs already expose them; do not modify the general engine solely to add
  accounting telemetry.
- Environmental inference averages paired worlds within founder and bootstraps
  founders. With only two world contexts per founder, claims are limited to the
  qualified r5 distribution.
- The punctuated-versus-stable comparison is secondary and exploratory because
  there is only one stochastic outer search per arm.
- Every Shinka candidate, structured-random policy, human scheduler, and fixed
  operator is evaluated through the existing
  `evaluate_manifest_paired_delta(..., ancestor_policy=exact_initial)` path and
  serialized from its `PairedManifestEvaluation`. Do not route an r5 control
  through legacy `evaluate_manifest` or `run_baselines.py`, because those do not
  implement r4's coherent median paired-delta repeat selection.

### 8.6 Primary result

Finalist roles follow the frozen r4 rule. If development contains an adaptive-
eligible punctuated candidate, the highest-scoring such candidate is the
mechanism-primary finalist and the unrestricted champion is retained as a
secondary performance finalist when distinct. If none is adaptive eligible,
the unrestricted champion is the performance-only primary finalist.

The confirmatory primary effect remains:

```text
selected punctuated primary finalist shocked absolute AUC
- exact r4 initial shocked absolute AUC
```

Primary performance success requires:

```text
sealed clone shock-minus-sham 95% upper bound < 0
primary mean >= +0.02
primary founder-first 95% lower bound > 0
sham absolute-AUC delta 95% lower bound > -0.02
```

Adaptive-mechanism claims additionally require the frozen r4 adaptive-
eligibility, lineage-matched ancestor/descendant, and ablation rules. Those
analyses are conditional on positive primary performance and are not gates for
launching search or computing the primary result.

### 8.7 Baseline search

Freeze both human scheduler sources and the structured-random generator in
Phase 1 before fresh seeds or Shinka results are observed.

The structured-random control uses NumPy seed `50005`, the same candidate
inputs, six trusted logits, training manifests, and 50-evaluation budget as the
punctuated arm. It produces exactly:

- 25 threshold schedulers: sample one of the 29 readable scalar inputs, one
  threshold from `{0.1, 0.2, ..., 0.9}`, and two distinct trusted actions;
  emit `+8` for the action selected below/above the threshold and `-8` for the
  other five actions;
- 25 affine schedulers: flatten the same 29 inputs and emit
  `clip(W @ x + b, -8, 8)`, with `W` sampled from
  `Normal(0, 1/sqrt(29))` and `b` from `Normal(0, 0.25)`.

The generated source for every control is validated by the same AST/runtime
boundary as Shinka candidates. Generate, canonicalize, hash, and freeze all 50
sources before evaluating the first one; the generator receives only the
frozen grammar and seed, never evaluation results. The top five unique valid
structured-random candidates enter development selection beside the Shinka
top-five sets. Its development winner freezes before sealed evaluation. This
control is narrower than Shinka's bounded arithmetic-program space, and
conclusions must say so.

## 9. Minimal Architecture and Upstream Boundary

### Microcosmos

Create one compact package:

```text
experiments/evo2_ecosystem/r5/
├── __init__.py
├── protocol.py          # frozen r5 constants, paths, and artifact schemas
├── qualify_worlds.py    # reset-only qualifier and public audit writer
├── tools.py             # seed-parameterized adapters around r4 calculations
├── baselines.py         # six-action, human, and structured-random controls
├── workflow.py          # thin dependency-ordered orchestration
└── analysis.py          # r5-only summary/common-garden glue
```

Requirements:

- `protocol.py` is the single r5 source of seed ranges, threshold, event steps,
  panel seed, and protocol revision.
- `qualify_worlds.py` calls the existing reset/resource sampling path and has no
  dependency on heredity candidates or rollout evaluation.
- `tools.py` exists because the r4 manifest, disturbance, and opportunity
  entrypoints hard-code r4 world seeds. It supplies exactly three thin
  seed-parameterized adapters:
  `build_and_publish_manifests`, `run_disturbance_qualification`, and
  `run_opportunity_qualification`. They accept authenticated r5 manifests or
  seed panels and reuse r4 seed-independent helpers, thresholds, episode
  evaluation, and cross-fit analysis. They must not mutate r4 modules, patch
  module globals, or copy an entire r4 tool. Minimal orchestration duplication
  is allowed only where the hard-coded r4 entrypoints make direct reuse
  impossible.
- `baselines.py` implements the missing r4 six-action roster, the frozen human
  stress/credit schedulers, and the structured-random generator through the
  same trusted candidate adapter and evaluation path.
- `workflow.py` only orders and authenticates calls to existing founder,
  episode, finalist, and analysis functions plus the r5 tools above. It must
  not duplicate their scientific logic.
- `analysis.py` imports r4 seed-independent interval, lineage, and ablation
  helpers. It replaces only the invalid hard-coded common-garden initialization
  and r5 result assembly.
- Do not create `heredity_adaptation_v5/` copies or add a general orchestration
  framework.

R5 seed qualification, shocks, gates, manifests, evidence, and Shinka bindings
are downstream research code and must not be included in a general upstream
Microcosmos PR.

### ShinkaEvolve

Limit changes to the existing `examples/evo2_ecosystem/` integration:

- `run_spec.py`: add strict schema-v4 validation without changing schemas 1–3.
- `run_evo.py`: resolve r5 trusted tools from schema-v4 bindings and validate
  prerequisite artifacts before either arm starts.
- `freeze_finalist.py`: own development reevaluation, adaptive classification,
  finalist selection, and freezing only; it does not own sealed execution.
- Reuse the example's existing `_load_candidate`, `_smoke_validate_candidate`,
  `_build_policy`, and `reevaluate_candidate` seams for Shinka and structured-
  random sources. If visibility must be cleaned up, extract only one narrow
  example-local candidate helper; do not create a second loader in Microcosmos
  or a new general Shinka abstraction.
- Add focused tests. Do not change ShinkaEvolve's general evolutionary engine.

Schema v4 adds:

```text
protocol_revision
protocol_tools: role -> {repository-relative path, sha256}
repository_commits: {microcosmos, shinkaevolve}
prerequisite_artifacts: role -> {path, sha256, passed}
protocol_document: {repository-relative path, sha256}
```

The exact allowed role-to-callable contract is:

```text
world_qualification       -> validate_world_qualification
manifest_generation       -> build_and_publish_manifests
disturbance_qualification -> run_disturbance_qualification
opportunity_qualification -> run_opportunity_qualification
final_analysis            -> run_final_analysis
```

Each role binds one repository-relative path and SHA-256. Validation requires
exactly these five roles, requires each named callable, confines paths to the
allowlisted Evo² experiment roots, and rejects extra roles, absolute paths,
path escapes, symlinks escaping the root, and hash mismatches.

The exact `prerequisite_artifacts` roles are:

```text
world_qualification
disturbance_qualification
operator_opportunity
```

The founder index and four manifests remain in their existing top-level run-
spec bindings and are not duplicated in `prerequisite_artifacts`. Schema v4
uses one `protocol_document` binding to this PRD instead of separate duplicate
preregistration and implementation-plan bindings. Its source-key set retains
the schema-v3 common runner/evaluator/selector sources but removes the four
hard-coded `r4_*` tool hashes; `protocol_tools` replaces those bindings. Schemas
1–3 retain their exact existing keys and meaning.

The schema-v4 run specification is the sole search-authorization artifact. Do
not add a separate `search_authorization.json`. It may be generated only after
every prerequisite artifact exists, authenticates, and records `passed=true`;
`run_evo.py` independently revalidates those inputs.

### Provenance

Bind:

- clean Git commit IDs for Microcosmos and ShinkaEvolve;
- the canonical schema-v4 run specification;
- critical trusted entrypoint hashes;
- seed qualification, founder index, manifests, gate results, archives,
  finalists, and final results.

Do not build a transitive source-hashing framework or hash every incidental
file separately. Git commits identify the complete source trees.

The run specification binds the pushed Phase-1 source commits. Generated
run-spec and evidence files are write-once outputs outside that source snapshot
during execution and may be committed afterward; this avoids a self-referential
commit hash. Preflight verifies tracked source against the bound commits while
allowing new files only beneath the Microcosmos r5 `artifacts/` root, the exact
stop-report path, the exact Shinka run-profile JSON and `.sha256` sidecar paths,
and the declared Shinka results/frozen artifact roots listed below. Any other
tracked modification or untracked path fails preflight.

Scientific execution uses detached sibling worktrees checked out at those two
pushed commits, rather than cleaning the contributor workspaces that retain
earlier evidence and local tooling. Invoke the controller and every worker with
Python bytecode disabled (`python -B` / `PYTHONDONTWRITEBYTECODE=1`) and an
explicit `PYTHONPATH` rooted in those launch worktrees. This prevents imports
from creating ignored cache files before the clean-tree check and ensures the
detached Microcosmos `src/` tree, not an unrelated editable install, supplies
the runtime package.

### Frozen artifact paths

All schema-v4 path strings are relative to the shared
`origins-of-life/` project root, matching the existing `artifact_path`
resolver, and therefore include the repository prefix:

```text
Microcosmos protocol root:
  microcosmos/experiments/evo2_ecosystem/r5/
Protocol document:
  microcosmos/docs/evo2/evo2-r5-world-feasibility-plan.md
World qualification:
  microcosmos/experiments/evo2_ecosystem/r5/artifacts/world_qualification.json
Founder bank:
  microcosmos/experiments/evo2_ecosystem/r5/artifacts/founders/
Manifests:
  microcosmos/experiments/evo2_ecosystem/r5/artifacts/manifests/
Disturbance evidence:
  microcosmos/experiments/evo2_ecosystem/r5/artifacts/disturbance_qualification.json
Opportunity evidence:
  microcosmos/experiments/evo2_ecosystem/r5/artifacts/operator_opportunity.json
Final evidence:
  microcosmos/experiments/evo2_ecosystem/r5/artifacts/final/
Stop report:
  microcosmos/docs/evo2/evo2-r5-world-feasibility-stop-report.md

Shinka run profile:
  ShinkaEvolve/examples/evo2_ecosystem/run_specs/evo2-r5-world-feasibility.json
Shinka run-profile sidecar:
  ShinkaEvolve/examples/evo2_ecosystem/run_specs/evo2-r5-world-feasibility.sha256
Run ID:
  evo2-r5-world-feasibility-20260713
artifact_roots.results:
  ShinkaEvolve/examples/evo2_ecosystem/results
artifact_roots.frozen:
  ShinkaEvolve/examples/evo2_ecosystem/frozen
Resulting run directory:
  ShinkaEvolve/examples/evo2_ecosystem/results/evo2-r5-world-feasibility-20260713/
Resulting frozen-finalist directory:
  ShinkaEvolve/examples/evo2_ecosystem/frozen/evo2-r5-world-feasibility-20260713/
```

## 10. Execution Context and Quality Gates

Starting points:

```text
Microcosmos: 2427b69
ShinkaEvolve: 1740a7e
branch in both repositories: agent/evo2-r5-world-feasibility
```

Every Python command uses `conda run -n sakana`.

Before scientific execution, both repositories must pass:

```text
git diff --check
conda run -n sakana ruff check <changed Python files>
conda run -n sakana env JAX_PLATFORMS=cpu pytest -q
```

The guarded GPU backend check is:

```text
systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -c "import jax; assert jax.default_backend() == 'gpu'; print(jax.devices())"
```

Focused guarded GPU tests must cover:

- reset/access-score agreement with the production environment;
- CPU/GPU access-score equality within `atol=1e-6` and identical
  pass/fail decisions for boundary fixtures;
- one tiny end-to-end r5 smoke that exercises founder screening, the
  actuation-cost adapter, and the operator-opportunity adapter;
- one schema-v4 r5 smoke evaluation selecting the exact initial ancestor.

Fresh-panel reproducibility, default r4 byte equivalence, schema validation,
and baseline generation are CPU tests.

No fresh r5 seed may be used by a test before the source-freeze commit.

## 11. Implementation Phases

### Phase 1: Implement, test, and freeze all r5 code

Objective: finish the complete protocol and execution path before observing any
fresh r5 world.

Requirements:

- Create the r5 branches and commit this document as the binding
  preregistration; do not create a second duplicative prose preregistration.
- Implement the minimal r5 package and schema-v4 Shinka routing.
- If necessary, add only the generic keyword-only CPPN population PRNG-key
  input described in Section 7.
- Implement and freeze the six fixed controls, human stress scheduler, human
  credit scheduler, structured-random grammar, seed, generator, and common
  garden geometry and evaluator before any Shinka or fresh-seed result is
  visible.
- Implement one thin final workflow and all tests using existing r4 diagnostic
  fixtures, synthetic boundary cases, or explicitly non-r5 smoke seeds.
- Red-team genome/outcome leakage, wrong hashes, path escapes, stale commits,
  overwrite attempts, failed prerequisite artifacts, and accidental fallback
  to r4 hard-coded seeds.
- Run all CPU and focused guarded GPU quality gates.
- Commit and push a clean worktree in both repositories.
- Create detached sibling launch worktrees at the two pushed commits; retain
  the original workspaces and all prior evidence unchanged.

Acceptance checkpoint:

- [ ] Existing r1–r4 artifacts are byte-identical.
- [ ] Legacy Microcosmos tests and schemas 1–3 pass unchanged.
- [ ] The default CPPN population is byte-identical if its API was extended.
- [ ] No fresh seed in `[8000, 12000)` has been reset or inspected.
- [ ] Both source commits are pushed and recorded before Phase 2.

Rollback point: the starting commits above. No scientific evidence exists yet.

### Phase 2: Qualify worlds and publish the founder bank

Objective: establish that the primary benchmark begins with food access and
authenticated viable founders.

Requirements:

- Compute the canonical CPU qualification trace over the four frozen ranges in
  staging without publishing it.
- Replay only the eight selected worlds under the guarded GPU backend, verify
  identical rounded cells and access scores within `atol=1e-6`, and then
  atomically publish one combined write-once public audit. CPU remains
  canonical, but any cell mismatch or score mismatch beyond the frozen
  tolerance fails the seed gate and is preserved in the audit.
- Generate the seed-5005 founder panel through the generic/configured path.
- Screen the prepartitioned `24/24/32` candidate ranges using only the two
  authenticated founder-eligibility worlds.
- Publish exactly `4/4/8` founders only if every unchanged founder gate passes.
- Keep development and sealed founder genomes inaccessible to the search
  process by reusing the existing holdout mode/allowlist mechanism; their
  identities and hashes may remain in the authenticated index. During search,
  evaluator APIs receive only training manifest/founder paths, and a test must
  fail if a development or sealed data path is opened.

Acceptance checkpoint:

- [ ] Four disjoint seed panels contain exactly two passing seeds each.
- [ ] Every selected access score is at least 0.50.
- [ ] Selected CPU/GPU rounded cells match and scores agree within `atol=1e-6`.
- [ ] The public audit reports every inspected seed through the second pass in
  each range, each selected score, and the number of seeds inspected. It does
  not mislabel the stopping-rule-biased prefixes as a qualification-frequency
  estimate or complete score distribution.
- [ ] The founder bank contains exactly 4 training, 4 development, and 8 sealed
  founders with valid hashes and unchanged eligibility rules.
- [ ] Failure writes one immutable stop report and Phase 3 cannot start.

Rollback point: none for published scientific evidence. An authenticated
failed scientific gate is final for this r5 run ID. A child-process
launch/runtime failure that publishes no failed-gate evidence is infrastructure,
not an ecological result; preserve its log and resume from existing write-once
evidence after fixing the infrastructure.

### Phase 3: Qualify the disturbance and operator opportunity

Objective: prove that the frozen shock is meaningful, survivable, observable,
and presents a learnable choice among trusted operators.

Requirements:

- Using training founders and training worlds only, test multipliers in the
  frozen order and select the first passing all r4 criteria.
- After the multiplier is known, generate authenticated paired sham/shock
  training, development, and sealed manifests. Do not accept an arbitrary
  command-line multiplier that is not bound to the qualification artifact.
- Enforce that binding in `r5/workflow.py`; do not modify the frozen r4
  manifest generator or add a general Microcosmos CLI setting.
- Keep development and sealed manifest JSON and founder genomes unreadable
  during outer search using the existing mechanism, but publish their SHA-256
  sidecars as read-only (`0444`) so preflight/finalist code can authenticate
  them without opening holdout contents. Training artifacts remain readable.
- Run the unchanged r4 leave-one-founder-out operator-opportunity gate.
- Generate and freeze one schema-v4 run specification only after both gate
  artifacts pass. Bind both repositories' commits, critical tools, world audit,
  founder index, manifests, and gate evidence.

Acceptance checkpoint:

- [ ] One multiplier passes every harm, viability, resolved-feedback, and
  observable-stress criterion.
- [ ] The operator-opportunity advantage is at least `0.03`, its paired 95%
  lower bound is positive, and its other frozen r4 criteria pass.
- [ ] The run specification authenticates all prerequisites and is frozen
  before either Shinka arm starts.
- [ ] Failure writes one immutable stop report and Phase 4 cannot start.

Rollback point: none for published scientific evidence.

### Phase 4: Run matched bounded-RSI searches

Objective: obtain two authenticated code-evolution lineages under matched
compute.

Requirements:

- Use separate result directories for stable-objective and punctuated arms.
- Freeze both arms' source, model configuration, outer seed policy, run spec,
  proposal budget, manifests, and compute limits before starting the first arm.
- Run the stable-objective arm and punctuated arm sequentially under the GPU
  guard. Reuse the existing sibling-isolation check: after the stable archive
  closes, make it proposer-inaccessible before starting punctuated. Do not add a
  new isolation or authorization mechanism.
- Materialize and authenticate the complete 50-source structured-random roster,
  then run it against the punctuated training objective before any development
  access. Validate and evaluate every source through the same Shinka candidate
  boundary and paired evaluator, then close its authenticated training archive
  alongside the two Shinka archives.
- Assign one immutable generation ID to each of the 50 noninitial slots.
- Persist invalid and failed slots honestly and preserve resumable state using
  the existing authenticated resume semantics.
- Close and authenticate all three training archives before development access.

Acceptance checkpoint:

- [ ] Each arm has one exact ancestor plus 50 terminal noninitial slot records,
  not necessarily 51 valid programs.
- [ ] Actual model calls, retries, validity counts, duplicates, and timeouts are
  reported. Tokens and cost are reported only where existing Shinka logs expose
  them.
- [ ] Both Shinka archive databases plus the structured-random roster/results,
  completion markers, and run specifications authenticate before Phase 5.
- [ ] The structured-random control has 50 terminal training evaluations and at
  least five unique valid candidates.
- [ ] Fewer than five unique valid candidates in either Shinka arm or the
  structured-random control produces an immutable search-failure report; do not
  lower `top_k` or grant replacement evaluations.

Rollback point: completed evidence is immutable; an interrupted arm may resume
only when its source, run-spec, manifest, and database hashes match.

### Phase 5: Development selection and one sealed evaluation

Objective: select finalists without sealed feedback and compute the frozen
primary result once, with any preregistered mechanism assays completed inside
the same sealed workflow.

Requirements:

- Authenticate the two closed Shinka archives and the closed structured-random
  archive.
- Select the top five unique valid training candidates from each Shinka arm and
  from the structured-random control.
- Enter one development-access epoch and evaluate all 15 selected candidates
  through the same development path.
- Freeze the highest-scoring unrestricted finalist per arm and, where one
  exists, the highest-scoring adaptive-eligible finalist per the frozen r4
  rule. Freeze the structured-random development champion and every selected
  Shinka finalist's complete code lineage.
- Only after every finalist, analysis function, and output schema is frozen,
  enter one logical sealed-access epoch owned by the frozen `r5/workflow.py`
  controller. There is no later scientific rerun or reopened sealed selection
  step; an infrastructure restart is allowed only through the authenticated
  missing-work resume rule below.
- Treat that controller as the immutable suite owner, not as one
  long-lived JAX process. Reuse the existing `evo2_sealed.workflow` pattern:
  first write a suite record binding the run spec, sealed manifest, founder
  index, simulator config, finalist/control sources, analysis hashes, and
  expected policy list; then launch each policy or analysis batch in a fresh
  guarded 24-GiB GPU subprocess so compiled executables are released on exit.
  On interruption, validate every write-once completed record against the suite
  record and run only missing work. Do not add a generic workflow framework.
- Through one common sealed evaluation path, compare:
  - the exact initial ancestor;
  - all six fixed trusted operators;
  - the human stress scheduler;
  - the human credit scheduler;
  - the frozen structured-random development champion;
  - stable-objective Shinka finalists;
  - punctuated Shinka finalists.
- Capture shock-time and final populations for the pre-frozen Shinka finalists
  during those sealed executions. Persist only the lineage pairs and genomes
  required by the already frozen common-garden analysis; do not reopen sealed
  manifests later to reconstruct them.
- Compute the primary result before interpreting mechanism evidence. If primary
  performance passes and an adaptive-eligible punctuated mechanism-primary
  finalist exists, complete the preregistered no-credit and dominant-action
  ablations during this same sealed-access epoch. Persist the deterministic
  pass/skip decision before launching conditional workers so a resume cannot
  alter it. Otherwise record the preregistered mechanism-analysis skip and make
  no adaptive-mechanism claim.
- For an eligible mechanism-primary finalist, run the lineage-matched common-
  garden analysis from the captured pairs in one standardized accessible world
  implemented only in `r5/analysis.py`:
  - one organism, zero initial velocity, and the frozen post-shock multiplier;
  - unchanged radius-`r` paraboloid resource state;
  - mouth fixed at `resource_center + (0.5 * r, 0)`, where resource capacity is
    exactly `0.75` of peak;
  - four head-to-tail orientations relative to the local inward resource
    gradient: inward, outward, clockwise tangent, and counterclockwise tangent;
  - identical pose panels for each ancestor and descendant.
- Construct the common-garden states analysis-side by rigidly transforming the
  existing reset state and applying periodic wrapping. Do not change
  `EcosystemEnv.reset` or add a general initial-pose option. Average the four
  paired pose effects within each lineage pair, then aggregate lineage-pair
  effects within founder before the founder bootstrap. Poses are repeated
  conditions, not independent ecological replicates.
- Have `r5/workflow.py` compose existing evaluator, lineage, interval, and
  ablation functions rather than implementing a second sealed-evaluation or
  statistical system.
- Publish raw results, founder-first intervals, summary tables, recovery plots,
  archives, lineages, conditional-analysis status, and an exact accounting of
  compute.

Acceptance checkpoint:

- [ ] Development evaluates the three frozen top-five sets before finalist
  freeze; sealed access follows every finalist and analysis freeze.
- [ ] Exactly one frozen suite record owns the logical sealed-access epoch. Its
  controller uses fresh guarded subprocesses and may resume only missing,
  authenticated work; no result-dependent candidate, metric, pose, or analysis
  definition is added after the epoch begins.
- [ ] The primary punctuated-finalist-versus-ancestor interval and sham-safety
  interval use the frozen r4 analysis.
- [ ] Baselines share identical manifests, founders, numerical-repeat
  selection, and simulator configuration.
- [ ] Stable-versus-punctuated is labeled secondary/exploratory.
- [ ] Negative, null, fixed-operator, adaptive, and mechanism-skipped outcomes
  are reported without changing the benchmark.
- [ ] An adaptive-mechanism claim is made only if lineage-matched descendants
  beat ancestors with a positive founder-first interval and a frozen ablation
  removes at least half the gain or has a positive finalist-minus-ablation
  interval.

Rollback point: none. Sealed results are immutable.

## 12. User Stories

### US-001: Freeze an experiment-local r5 implementation

**Phase:** Phase 1

As an independent Microcosmos contributor, I want r5 isolated from general
simulator behavior so that the research project does not pollute or break the
upstream API.

Acceptance criteria:

- [ ] R5-specific code exists only under the Evo² experiment trees and
  ShinkaEvolve example, except for the optional generic backward-compatible
  PRNG-key argument.
- [ ] No copied v5 tool tree or new core r5 branch exists.

### US-002: Qualify feasible random worlds prospectively

**Phase:** Phase 2

As the experiment operator, I want random worlds conditioned on initial food
access so that the lifecycle can begin without selecting on policy outcomes.

Acceptance criteria:

- [ ] The same config and range produce byte-identical canonical audit output.
- [ ] No genome, rollout, reward, survival, or birth value is available to the
  qualifier.

### US-003: Authorize search only through existing evidence

**Phase:** Phase 3

As a verifier author, I want the run specification to authenticate every
prerequisite so that there is one source of truth rather than another layer of
authorization artifacts.

Acceptance criteria:

- [ ] Schema-v4 validation rejects failed/missing gates and wrong commits,
  paths, or hashes.
- [ ] No separate search-authorization receipt exists.

### US-004: Run matched bounded recursive improvement

**Phase:** Phase 4

As a Sakana interviewer, I want matched stable-objective and punctuated code-
evolution lineages so that the artifact being improved is the heredity
scheduler rather than another organism.

Acceptance criteria:

- [ ] Both arms use the exact same bounded candidate ABI and 50-slot budget.
- [ ] Failed proposals consume budget and all resource usage is reported.

### US-005: Evaluate one frozen primary claim

**Phase:** Phase 5

As a researcher, I want one development selection followed by one sealed
comparison so that final performance cannot influence implementation choices.

Acceptance criteria:

- [ ] The punctuated finalist is compared against its exact ancestor under
  paired sealed shocked worlds.
- [ ] Every control uses the same simulator and evaluation path.

### US-006: Explain a positive result without changing it

**Phase:** Phase 5 conditional tail

As a reviewer, I want lineage and ablation evidence after primary success so
that an adaptive-mechanism claim is supported without turning optional analysis
into another search gate.

Acceptance criteria:

- [ ] Common-garden poses and ablation definitions were frozen before seed
  qualification and finalist evaluation.
- [ ] Mechanism analysis cannot modify finalist selection or the primary score.

## 13. Functional Requirements

- FR-1: Seed qualification is reset-only, deterministic, genome-free, and
  outcome-free.
- FR-2: All fresh ranges, thresholds, panel partitions, event steps, and gate
  rules are frozen before first execution.
- FR-3: The fresh founder panel is controlled through immutable configuration,
  with no version-specific core branch.
- FR-4: Development and sealed evaluation data remain inaccessible during
  search even though reproducible seed identities are public.
- FR-5: Schema-v4 run specs bind repository commits, trusted protocol tools,
  manifests, founders, and passed prerequisite artifacts.
- FR-6: `run_evo.py` fails closed when any prerequisite or binding does not
  authenticate.
- FR-7: Every scientific artifact is write-once; rerunning an identical command
  may verify an artifact but may not silently replace it.
- FR-8: One thin final workflow executes the full baseline and finalist matrix
  without duplicating simulator or statistical logic.

## 14. Non-Functional Requirements

- NFR-1: No new third-party runtime dependency.
- NFR-2: Existing fixed-shape, JAX/GPU-native simulation remains unchanged.
- NFR-3: No breaking public Microcosmos API change is permitted. Existing calls
  and default behavior remain unchanged; frozen r1–r4 outputs remain byte-
  identical wherever the existing protocol asserts byte identity.
- NFR-4: Complete regression runs remain CPU-safe; every focused GPU workload
  remains inside the 24 GiB cgroup limit.
- NFR-5: Long-running search state is resumable through existing authenticated
  mechanisms; no new generic resume framework is introduced.
- NFR-6: The implementation favors configuration and composition over copied
  modules, boolean version flags, and repeated artifact schemas.

## 15. Baseline and Claim Hierarchy

### Confirmatory

- Punctuated finalist versus exact initial ancestor on sealed shocked AUC.
- Sham noninferiority and proof that the shock harms clone performance.

### Secondary

- Every fixed trusted operator.
- Human stress and human credit schedulers.
- Matched-budget structured-random scheduler search.
- Stable-objective finalist versus ancestor.
- Punctuated versus stable-objective finalist.

The last comparison describes two realized search lineages. It does not by
itself establish a causal population-level effect of punctuated outer training.

### Conditional mechanism evidence

- Adaptive-eligibility diagnostics.
- Lineage-matched common-garden ancestor/descendant comparison.
- No-credit and dominant-action ablations.

No diversity metric, action count, lineage count, or mechanistic diagnostic is
added to the search reward.

## 16. Rollout and Rollback Strategy

- Before fresh seeds: ordinary code rollback is allowed to the last clean
  implementation checkpoint.
- After fresh seed qualification: never alter or delete published evidence for
  this run ID. A failed gate produces a committed stop report.
- Before search: the single schema-v4 run specification and both repository
  commits freeze both arms.
- During search: an incomplete arm may resume only when its authenticated state
  matches exactly. A completed arm is immutable.
- During development/sealed evaluation: access each partition once in the
  frozen order. No metric or code changes are permitted after seeing it.
- Scientific failure is a valid terminal result, not a rollback trigger.

## 17. Success Metrics

### Minimum completed r5

- The world qualifier and founder bank publish prospectively, or a frozen gate
  fails and is reported without retuning.
- Every eligible prerequisite runs in order and produces authenticated
  evidence.
- If search becomes eligible, both arms terminate with complete slot records
  and one sealed evaluation.

### Positive performance result

- The frozen primary effect satisfies all four thresholds in Section 8.6.

### Strong adaptive-RSI result

- Positive primary performance;
- adaptive eligibility and context-dependent action probabilities;
- lineage-matched descendants beat ancestors with a positive interval;
- an ablation removes the relevant advantage;
- the discovered code mechanism is understandable and preserved in its Shinka
  program lineage.

### Honest alternative outcomes

- Performance succeeds but adaptiveness fails: report automated fixed-operator
  or fixed-mixture discovery.
- Performance is uncertain or negative: report a bounded-RSI negative result.
- A prerequisite fails: report that the benchmark remained infeasible at that
  gate. Do not weaken the gate within r5.

## 18. Open Questions

There are no blocking design questions. Long Shinka execution still depends on
configured model credentials and sufficient GPU time; missing external access
is an execution dependency, not permission to change the protocol.

## 19. Implementation Readiness Notes

- Primary delivery risk: fresh qualified worlds may still fail the unchanged
  founder gate. That is an informative prospective result.
- Secondary delivery risk: nested outer search is costly. The frozen slot
  budget and existing resume mechanism bound the risk without introducing a
  new multi-fidelity system.
- Upstream boundary: do not propose the current research branch wholesale to
  original Microcosmos, and do not scope an upstream extraction into r5. Any
  later upstream proposal must be a separate review of one generic component;
  all Evo² protocol machinery remains downstream.
- Suggested implementation order: US-001 -> US-002 -> US-003 -> US-004 ->
  US-005 -> US-006.
- Definition of ready: this final PRD is committed; Phase 1 may begin. Fresh r5
  seeds remain forbidden until the Phase 1 source-freeze checkpoint passes.

[/PRD]
