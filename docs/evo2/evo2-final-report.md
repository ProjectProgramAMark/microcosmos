# Evo²-Ecosystem final implementation and experiment report

Date: 2026-07-12  
Run ID: `evo2-production-20260712-r1`  
Run-spec SHA-256: `47ac4c3261a64ec37d04f1303b52846329eeffc6efd92f0bf52276adb4ed304f`

## Outcome

The engineering goal is complete. Microcosmos now supports a finite-resource,
multi-generation ecosystem of embodied CPPN-controlled filaments, and
ShinkaEvolve can search a bounded ecology-conditioned heredity scheduler through
an executable, integrity-checked evaluator.

The confirmatory experiment did **not** support the scientific hypothesis.
Catastrophe-trained Shinka was not better than stable-trained Shinka on the
sealed shocked-world comparison, and neither Shinka champion showed a reliable
advantage over the fixed mixed or handwritten stress-responsive policies.

This is a valid null/negative result, not an incomplete run. All six policies
finished the same 72-world sealed suite, each as three complete GPU
measurements, and the preregistered paired analysis used 36 ecological
shock/null pairs with 10,000 bootstrap replicates.

## What was implemented

### Embodied ecosystem

- Fixed-capacity organism slots with physical masks for living and dead bodies.
- Shared viscous-fluid physics, finite localized resources, mouth-only feeding,
  basal and actuation costs, death, food-earned birth, and same-step slot reuse.
- Energy-conserving asexual inheritance, monotonic organism IDs, parent IDs,
  generations, founder lineages, and compact birth/death telemetry.
- Resource relocation, random bottleneck, and capped dominant-lineage-cull
  events with exact shock/null pre-event forks.

### CPPN heredity

- Direct fixed-shape TensorNEAT CPPNs from the start; no temporary fixed wave
  controller in the scientific workflow.
- Fifteen padded nodes and thirty padded connections per genome, with topology
  changes that do not alter array shapes.
- Four sensor inputs: normalized body coordinate, phase, local resource, and
  head-minus-tail resource signal.
- Cached graph transforms at reset and birth, batched JAX inference, complete
  graph/cache validation, and trusted clone, parametric, structural, and mixed
  TensorNEAT mutation operators.

### Bounded recursive program improvement

- Shinka edits only `make_offspring`, which returns four operator scores.
- Candidate code can react to parent energy/intake, population change,
  behavioral diversity, lineage entropy, and current CPPN complexity.
- Trusted code—not candidate code—samples and applies the selected TensorNEAT
  mutation operator.
- The simulator, resources, lifecycle, mutation kernels, events, manifests,
  score, budgets, and hidden seeds remain immutable.
- A strict AST sandbox, environment scrubbing, process-tree timeout, source
  hashing, and isolated holdout permissions prevent evaluator or simulator
  rewriting.

The accurate project description is:

> CPPN-controlled ecological evolution in which ShinkaEvolve searches for an
> ecology-conditioned scheduler over four trusted TensorNEAT mutation
> operators.

It is not a full synchronous generational NEAT loop, because Microcosmos already
provides continuous ecological selection through resource acquisition, birth,
and death.

## Accelerator and differentiation boundary

CPPN inference, physics, resources, lifecycle state, mutation profiles, and
bounded births are fixed-shape JAX and GPU native. Structural mutation, operator
choice, birth, death, catastrophes, nearest-cell lookup, allocation, clipping,
and alive masks are discrete or nonsmooth. The mechanical and CPPN kernels
remain differentiable between transitions, but the whole ecology is
intentionally **not** claimed to be end-to-end differentiable.

## Search and selection

- Stable arm: 15 attempted generations; 14 complete valid evaluations and one
  safely rejected proposal.
- Punctuated arm: 15 complete valid evaluations.
- Both arms used the same initial program, outer seed, model, candidate budget,
  simulator, score, founders, and triplicate numerical protocol.
- The stable archive's unique top three were reevaluated on development; all
  passed, and generation 3 won with median score `0.0668331`.
- The punctuated archive's top three all passed; generation 8 won with median
  score `0.0682406`.
- Candidate selection occurred before sealed was opened.

The stable winner favors parametric refinement under productive/crowded
conditions and adds structural or mixed exploration under stagnation and low
diversity. The punctuated winner similarly computes productivity, fragility,
convergence, stability, and topology headroom, then shifts probability among
clone, parametric, structural, and mixed mutation.

## Sealed results

Aggregate candidate score and survival are descriptive; the primary inference
uses direct shocked-world AUC contrasts.

| Policy | Aggregate score | Survival | Mean shock-minus-null effect |
|---|---:|---:|---:|
| Fixed parametric | 0.12542 | 0.597 | -0.04324 |
| Clone / no mutation | 0.11456 | 0.625 | -0.08836 |
| Shinka stable | 0.08825 | 0.569 | -0.05520 |
| Shinka punctuated | 0.08657 | 0.569 | -0.05982 |
| Stress responsive | 0.08504 | 0.583 | -0.05638 |
| Fixed mixed | 0.06773 | 0.597 | -0.02743 |

Primary shocked-world AUC contrasts, where positive favors the punctuated
champion:

| Contrast | Mean | 95% bootstrap interval | Interpretation |
|---|---:|---:|---|
| Punctuated − stable Shinka | -0.00399 | [-0.01468, 0.00581] | No advantage |
| Punctuated − fixed mixed | 0.00265 | [-0.02805, 0.03348] | No reliable difference |
| Punctuated − stress responsive | -0.00019 | [-0.01321, 0.01326] | No reliable difference |

The secondary punctuated-minus-stable difference-in-differences was `-0.00462`
with 95% interval `[-0.01233, 0.00112]`. The aggregate punctuated-minus-stable
score difference was `-0.00167`.

Therefore the defensible conclusion is:

> The system successfully performed bounded recursive search over embodied
> heredity schedulers, but this single matched outer-search case study found no
> evidence that punctuated training improved sealed catastrophe recovery. The
> conventional policies remained competitive, and fixed parametric mutation
> had the highest aggregate sealed score.

No ancestor/common-garden or mechanism-ablation claim is reported. Those were
preregistered as conditional finalist analyses after a positive adaptive
result, and the condition was not met.

## Integrity notes

Two fail-closed infrastructure defects were found and documented without
changing any scientific result:

1. SQLite completion hashes were initially captured just before the final WAL
   checkpoint. The launcher now checkpoints and closes before hashing; only the
   two derived completion hashes were corrected before development opened.
2. The first sealed stable simulation completed but its serializer could not
   inspect a deliberately synthetic candidate filename. No policy result was
   atomically published. The provenance-only serializer boundary was fixed and
   tested, old metadata was quarantined, and the unchanged six-policy suite was
   rerun from an empty result directory.

The full audit is in
`microcosmos/benchmarks/evidence/evo2_training_runtime_amendment.md`.

## Verification

- Microcosmos full CPU suite: **334 passed, 1 skipped**.
- ShinkaEvolve full CPU suite: **527 passed**.
- Focused GPU feasibility, positive-control, turnover, baseline, matched-search,
  development, and sealed runs all used fresh 24 GiB guarded processes.
- Every sealed record reports `integrity_valid=true`.

## Canonical artifacts

- Preregistered plan: `plans/experiment-analysis-plan.md`
- Frozen run specification:
  `ShinkaEvolve/examples/evo2_ecosystem/run_spec.json`
- Stable finalist:
  `ShinkaEvolve/examples/evo2_ecosystem/frozen/evo2-production-20260712-r1/stable`
- Punctuated finalist:
  `ShinkaEvolve/examples/evo2_ecosystem/frozen/evo2-production-20260712-r1/punctuated`
- Sealed suite ledger:
  `microcosmos/experiments/evo2_sealed/final_results/suite_record.json`
- Sealed summary:
  `microcosmos/experiments/evo2_sealed/final_results/summary.json`
- Runtime amendments:
  `microcosmos/benchmarks/evidence/evo2_training_runtime_amendment.md`

The sealed summary SHA-256 is
`4b5d4bbc16645968b0bf97986626d10ea4e337fd2cf97f5adc2bd57d0253180c`.
