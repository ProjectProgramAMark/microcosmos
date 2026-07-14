# Evo² positive-result campaign log

## Objective

Produce a substantial, reproducible result in which ShinkaEvolve discovers a
bounded heredity program for the GPU-native CPPN ecosystem that outperforms its
ancestor and strong fixed heredity baselines on fresh ecological worlds.

This is an iterative exploratory campaign. Scientific gates are recorded as
diagnostics, not used to prevent outer search. A result becomes a submission
candidate only after a generated program transfers to fresh worlds and beats
the strongest relevant fixed baseline under repeated evaluation.

## Historical context

- R1 completed an end-to-end Shinka experiment but did not support the primary
  recovery claim.
- R3 completed two 15-evaluation Shinka searches. Shinka generated valid JAX
  schedulers and rediscovered parametric mutation, but the selected punctuated
  policy did not beat fixed parametric mutation on sealed worlds.
- R5 stopped before search when its proposed world did not satisfy feasibility
  prerequisites.
- R6 built and screened 16 fresh CPPN founders. Resource relocation produced
  viable populations, resolved operator feedback, and observable stress, but
  its clone-harm interval crossed zero. The confirmatory workflow stopped
  before Shinka.

## Campaign rules

1. Run Shinka before adding any new scientific gate.
2. Keep each exploratory search small enough to obtain feedback in hours.
3. Change a treatment only in response to measured policy results.
4. Preserve every generated program and evaluation, including failures.
5. Use fresh worlds for confirmation, not for selecting the search objective.
6. Do not claim success unless a Shinka-generated program beats the strongest
   fixed baseline relevant to the discovered mechanism.

## 2026-07-14 — Campaign start

- Active substrate: R6 fixed-shape JAX ecosystem with TensorNEAT CPPN genomes.
- Heredity action set: clone, conservative parametric, standard parametric,
  exploratory parametric, structural, and mixed mutation.
- Available training data: four training founders and two qualified training
  worlds, each with matched resource-refresh control and resource-relocation
  treatment.
- Immediate action: create a direct exploratory Shinka runner that consumes
  the existing manifest and founder bank without blocking on the R6 harm gate.
- First-round target: produce valid six-action descendants and determine
  whether any generated scheduler beats the standard-parametric ancestor and
  fixed-operator baselines on the visible training worlds.

### R7 search specification

- Run ID: `evo2-exploratory-r7-20260714`.
- Outer arm: punctuated (matched refresh controls and resource-relocation
  treatments both contribute to the score).
- Outer budget: eight evaluated programs: the standard-parametric ancestor and
  seven LLM-proposed descendants.
- Model: `headless/codex@gpt-5.5?effort=high` through pinned Headless `0.4.0`.
- Numerical repeats during search: one, to obtain feedback quickly. Any
  promising program must later pass repeated evaluation on fresh worlds.
- Search objective: paired candidate-minus-ancestor ecological performance;
  positive values beat standard parametric mutation under matched seeds.
- Execution: serial evaluation and proposal jobs inside the standard 24 GiB
  GPU memory guard. All programs, metrics, prompts, database state, manifest,
  source hashes, and launch metadata are retained under the Shinka result ID.

### R7 launch incident

The first `evo2-exploratory-r7-20260714` launch stopped after generation zero:
the standalone evaluator process could import the installed `shinka` package
but not the repository's `examples` namespace. The recorded initial evaluation
is therefore invalid and contains no ecosystem result. The process was stopped
before accepting a descendant. The evaluator now explicitly adds the pinned
Shinka checkout to its import path; the corrected search uses a distinct run ID
so the failed launch remains intact rather than being overwritten.

The corrected `evo2-exploratory-r7b-20260714` launch completed the ancestor
simulation, but the inherited confirmatory evaluator converted its ecological
score to `-2.0` because two of sixteen candidate episodes became extinct. The
run measured a 0.875 survival rate, 49.7 mean births, 4.94 mean generation gain,
and valid six-action execution; this was ecological variation, not invalid
code or broken physics. R7b was stopped before evaluating a descendant. For
exploratory search, extinction now remains part of the ecological performance
score and survival telemetry rather than making a program invalid. Physical
integrity failures and heredity-contract violations still invalidate a result.
