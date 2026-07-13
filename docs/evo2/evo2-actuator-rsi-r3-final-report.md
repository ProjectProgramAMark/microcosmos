# Evo² actuator-adaptation r3 — final report

Status: **COMPLETE — negative confirmatory result**  
Run ID: `evo2-actuator-rsi-20260713-r3`  
Completion date: 2026-07-13

## Executive conclusion

The implementation and experimental workflow completed successfully, but the
predeclared recursive-improvement claim is not supported.

ShinkaEvolve produced valid, readable heredity schedulers. The punctuated arm's
selected descendant chose fixed parametric mutation for every sealed birth. It
had a small mean direct injured-world advantage over the initial always-mixed
scheduler, `+0.007489`, but the founder-first 95% interval was
`[-0.109019, +0.148769]`. The interval crosses zero broadly, so there is no
reliable evidence that the descendant improved its ancestor.

The selected punctuated program also did not beat fixed parametric mutation:
the mean contrast was `-0.004901`, with interval
`[-0.019298, +0.006234]`. Its sealed operator trace was exactly 100%
parametric. Thus the outer search autonomously rediscovered a conventional
baseline; it did not produce a distinct adaptive heredity mechanism on the
sealed worlds.

The hidden environmental test also failed its intended ecological role. The
tail-actuator injury improved clone productivity relative to its matched sham:
`+0.066261`, interval `[+0.005288, +0.130422]`. In this panel the supposed
catastrophe was beneficial on average, plausibly because disabled tail
actuation reduced movement cost or altered locomotion favorably. Consequently,
r3 cannot establish recovery from an unseen catastrophe even independently of
the negative policy comparison.

The honest final claim is:

> We implemented a GPU-native, fixed-shape Microcosmos ecosystem with inherited
> TensorNEAT CPPN controllers and a bounded ShinkaEvolve heredity scheduler.
> Shinka generated valid scheduler descendants and rediscovered parametric
> mutation, but the selected punctuated descendant did not reliably improve its
> initial ancestor or conventional baselines on sealed founders. The sealed
> tail-actuator intervention was not harmful to clone controls, exposing a
> benchmark-design failure that must be corrected prospectively.

## 1. What was implemented

### 1.1 GPU-native evolving ecosystem

The ecosystem remains fixed-shape and JAX-native throughout simulation:

- 32 preallocated organism slots;
- eight organisms alive initially;
- eight nodes and six controlled hinges per organism;
- alive masks for birth, death, constraints, steric interaction, fluid
  coupling, sensing, and resource uptake;
- finite resource consumption and regeneration;
- per-organism stored energy, metabolism, birth cost, and death;
- inactive-slot reuse for newborns;
- UID, founder-lineage, parent, generation, birth, and death accounting;
- batched fixed-shape TensorNEAT CPPN genomes and controllers;
- trusted parametric, structural, and mixed mutation kernels;
- one trusted heredity action per birth.

Birth and death are dynamic ecological events, but array shapes do not change.
This preserves JIT compilation and GPU execution. Python is used only for the
outer Shinka orchestration and immutable artifact management, not the inner
physics/lifecycle loop.

### 1.2 CPPN-NEAT remains in the workflow

The final system does not replace CPPN-NEAT with a Fourier or fixed numerical
controller. Every organism inherits a direct fixed-shape TensorNEAT CPPN
genome. Its graph contains node and connection genes, and the trusted heredity
kernels can mutate weights, topology, or both.

ShinkaEvolve operates one level above those genomes. Its bounded function sees
only normalized parent and population summaries and returns four finite scores:

```text
clone, parametric mutation, structural mutation, mixed mutation
```

Trusted JAX code takes deterministic `argmax`, applies exactly one kernel,
validates the resulting CPPN graph, and records operator counts. Shinka cannot
read scenario identity, world seed, event type, injury location, absolute time,
raw simulator state, files, or holdouts.

### 1.3 Founder-readiness correction

The earlier founder prerequisite was invalid because it selected founders from
a pre-event state that did not demonstrate a functioning lifecycle. r2/r3
replaced that with a prospective readiness screen:

- canonical world seeds `5000..5063` were screened in order;
- worlds ran uninjured to step 4,500;
- readiness required at least two living organisms, at least two births,
  generation at least one, productivity at least `0.01`, and all integrity
  checks;
- the first passing seeds and then the first passing founders within frozen
  partitions were selected;
- development and sealed partitions stayed inaccessible during outer search.

Authenticated readiness outputs:

| Artifact | SHA-256 |
|---|---|
| founder index canonical identity | `2fcec499fb1d5ecbddafdbbe446c6dc3e52cd644a64e9452ef9de0c56111593f` |
| readiness seed-panel file | `9456902030d459a05dc2a784050815a76297ac040d46730e01c2ff5e3626e727` |
| readiness protocol | `a6d77754ce1eea6feddcee981de40fb23692e65015feeeab6efc712ac77992ef` |
| complete readiness ledger | `4ddaca9da59edd45e76f8948604ee5a83a10e50b9319bbdba0745c3c58b0c3cf` |

Selected partitions:

```text
training seeds:     5002, 5005, 5006
development seeds:  5008, 5011, 5012, 5013
sealed seeds:       5014, 5018, 5019, 5020, 5021, 5023

training founders:    candidates 0, 1, 2
development founders: candidates 16, 17
sealed founders:      candidates 32, 33, 34
```

This fixed the actual prerequisite problem: all downstream worlds now begin
from authenticated founders capable of living and reproducing before injury.
Keeping each world's initial population clonal remains necessary for the causal
question. It prevents recovery from being attributed to selecting a resistant
genotype that was already present at time zero; post-event genetic novelty must
be generated through heredity.

## 2. Experimental protocol

### 2.1 r2 benchmark qualification

r2 ran the complete frozen actuator-gain calibration on the corrected training
founders. Gain `0.1` harmed clone for head hinges, with injury-minus-sham
interval `[-0.135546, -0.009778]`. All clone injured worlds survived and
continued reproducing, so the training substrate was neither trivial nor dead.

No always-on mutation baseline beat clone reliably. This was a scientifically
valid negative result, not a reason to select a favorable repeat. The r2 ledger
SHA-256 is
`ca192df8a555c72db91bfdf4b2958c49415908e9a6950d594e707113d220ac77`.

r3 then corrected a conceptual mistake in the old gate: requiring an always-on
mutation operator to succeed before searching for an adaptive scheduler assumes
away the intended algorithm-discovery target. r3 therefore used the qualified
gain `0.1` and proceeded prospectively without relabeling r2 as positive.

### 2.2 r3 causal partitions

- Stable training: sham at head hinges.
- Punctuated training: persistent gain `0.1` on head hinges `[0, 1]`.
- Development: matched sham and persistent gain `0.1` on middle hinges
  `[2, 3]`.
- Sealed: matched sham and persistent gain `0.1` on tail hinges `[4, 5]`.
- Event steps cycled through 3,500, 4,000, and 4,500.
- Horizon was 8,000 steps with 500-step chunks.
- Every holdout injury world had an exact founder×seed sham fork.

The schema-v2 run specification SHA-256 is
`f89dcbd370ecf7d637a0b8ac5e9225b4fa8e16adf5ab87f3e99c2c0d6c51dfbd`.
The frozen r3 preregistration SHA-256 is
`5e129682ba59e29cbe7052a069db38b3fc1d85b386e0c9c6cd9c6a71fad04f31`.

### 2.3 Matched Shinka searches

Both arms used the same:

- initial program;
- outer seed 17;
- proposal model and pinned command;
- two islands and archive size 16;
- source/AST/line limits;
- candidate timeout;
- three coherent full-manifest measurements;
- simulator and mutation-action budgets;
- hidden development and sealed artifacts.

Each arm completed generations `0..14`. In Shinka's engine, generation zero is
the initial program, so this equals 15 evaluated generations but 14 LLM-created
proposals. The preregistration phrase "15 proposal generations" was therefore
imprecise. Both arms received the same bound engine budget, invalid proposals
consumed budget, and no result was replaced. This wording discrepancy is
reported as a protocol deviation and not silently reinterpreted as 15 generated
descendants.

The stable archive had one invalid proposal; the punctuated archive had one
invalid proposal. Both otherwise completed their matched ledgers.

## 3. Outer-search and development results

### 3.1 Stable arm

The initial always-mixed scheduler scored `0.151624` in stable training.
Shinka produced several interpretable conditional schedulers:

- generation 7 mixed parametric and structural actions on development
  (`52.4% / 47.6%`), training score `0.199949`, development score `0.196019`;
- generation 13 used `92.2%` parametric and `7.8%` structural actions on
  development, training score `0.218251`, development score `0.194910`.

The authenticated selector instead chose generation 5 because its development
median was highest: `0.240518`, from repeats
`[0.230081, 0.252205, 0.240518]`. Despite state-dependent code, its observed
development behavior was 100% mixed—equivalent to the initial scheduler. Its
repeat range `0.022124` exceeded the `0.015` review threshold; the coherent
median was retained, and no favorable repeat was substituted.

### 3.2 Punctuated arm

The initial always-mixed scheduler scored `0.096504` in punctuated training.
Shinka immediately shifted toward parametric mutation. The top three
development candidates were:

| generation | training score | development median | development behavior |
|---:|---:|---:|---|
| 2 | 0.219980 | 0.168809 | 100% parametric |
| 5 | 0.221656 | 0.166041 | 100% parametric |
| 8 | 0.219490 | 0.162315 | 100% parametric |

Generation 2 won the frozen selector. Its code described a rescue-aware policy,
but none of its structural, mixed, or clone scores ever won in observed
development or sealed births. Functionally, it was fixed parametric mutation.

### 3.3 What Shinka usefully created

There are two limited positives:

1. Shinka generated valid, interpretable, hash-audited JAX programs within the
   strict candidate boundary.
2. Under head injury it autonomously discovered that parametric-only mutation
   was much better than the initial mixed-only scheduler on training.

Those are useful engineering and autonomous-rediscovery results. They are not a
new heredity algorithm, and they are not evidence of recursive improvement on
unseen worlds.

## 4. Sealed results

The sealed suite contained three founders × six seeds × matched sham/injury
forks: 36 worlds per policy. Every policy was measured three times in a fresh
24-GiB-guarded GPU process; the coherent median full-manifest execution was
retained.

### 4.1 Policy outcomes

Operator columns use the order clone / parametric / structural / mixed.

| policy | full-manifest score | survival | sealed operator behavior | injury − sham mean | founder-first 95% interval |
|---|---:|---:|---|---:|---|
| clone | 0.274405 | 36/36 | 100 / 0 / 0 / 0% | +0.066261 | [+0.005288, +0.130422] |
| fixed structural | 0.236846 | 36/36 | 0 / 0 / 100 / 0% | +0.059221 | [+0.022678, +0.095229] |
| fixed mixed | 0.222149 | 36/36 | 0 / 0 / 0 / 100% | +0.018767 | [−0.032687, +0.063844] |
| Shinka stable | 0.221313 | 36/36 | 0 / 0 / 0 / 100% | +0.022967 | [−0.024167, +0.065178] |
| fixed parametric | 0.219939 | 36/36 | 0 / 100 / 0 / 0% | +0.050708 | [+0.014444, +0.093495] |
| initial scheduler | 0.217546 | 33/36 | 0 / 0 / 0 / 100% | +0.030713 | [−0.007724, +0.071428] |
| Shinka punctuated | 0.213634 | 36/36 | 0 / 100 / 0 / 0% | +0.053516 | [+0.013954, +0.097551] |
| stress responsive | 0.206950 | 36/36 | 37.9 / 61.2 / 0 / 1.0% | +0.037901 | [+0.018372, +0.059197] |

Clone had the highest absolute sealed score. The fact that injury-minus-sham is
positive for clone, fixed parametric, fixed structural, the punctuated
descendant, and stress-responsive policy means the tail intervention generally
did not function as damage under the frozen objective.

### 4.2 Predeclared primary and secondary contrasts

All values are direct injured-world normalized post-event productivity
differences. Positive favors the punctuated Shinka descendant.

| contrast | mean | founder-first 95% interval | conclusion |
|---|---:|---|---|
| punctuated − initial | +0.007489 | [−0.109019, +0.148769] | primary claim not supported |
| punctuated − stable | +0.007596 | [−0.094626, +0.113826] | no reliable advantage |
| punctuated − fixed parametric | −0.004901 | [−0.019298, +0.006234] | no advantage; same observed operator |
| punctuated − fixed mixed | +0.008860 | [−0.092066, +0.114086] | no reliable advantage |
| punctuated − fixed structural | −0.026064 | [−0.159776, +0.101573] | no reliable advantage |
| punctuated − clone | −0.067144 | [−0.192915, +0.066000] | no reliable advantage |
| punctuated − stress responsive | +0.014491 | [−0.059368, +0.081690] | no reliable advantage |

The secondary punctuated-minus-stable difference-in-differences mean was
`+0.030548`, interval `[-0.022515, +0.094753]`; it also crosses zero.

## 5. Integrity review

### 5.1 Numerical-repeat ranges

| policy | repeat scores | range | review |
|---|---|---:|---|
| clone | 0.270020, 0.274405, 0.275308 | 0.005288 | pass |
| fixed parametric | 0.216425, 0.220368, 0.219939 | 0.003944 | pass |
| fixed structural | 0.220118, 0.239679, 0.236846 | 0.019562 | review threshold exceeded |
| fixed mixed | 0.222149, 0.224920, 0.214616 | 0.010305 | pass |
| initial scheduler | 0.211022, 0.223448, 0.217546 | 0.012426 | pass |
| Shinka stable | 0.221313, 0.221582, 0.217916 | 0.003665 | pass |
| Shinka punctuated | 0.224121, 0.212620, 0.213634 | 0.011501 | pass |
| stress responsive | 0.202489, 0.211562, 0.206950 | 0.009073 | pass |

Fixed structural exceeded the preregistered `0.015` integrity-review trigger.
Its median repeat (`0.236846`) was retained exactly; the highest repeat was not
selected. Every episode was finite and integrity-valid, all 36 worlds survived,
and every recorded birth used structural mutation. The variation therefore did
not reveal a policy-boundary or accounting failure. It does make claims about
the structural baseline less precise, so that baseline is treated as
descriptive and no decisive conclusion depends on it.

The behaviorally equivalent pairs—initial versus fixed mixed and punctuated
versus fixed parametric—also produced modestly different coherent medians. This
is direct evidence of residual GPU numerical sensitivity. It reinforces the
negative conclusion: small point differences between equivalent operator
schedules are not algorithmic discoveries.

### 5.2 Artifact integrity

The global suite record was committed before any GPU worker. Every result was
published atomically only after all three repeats completed. The summary loader
then revalidated:

- finalist and baseline source hashes;
- run-spec, preregistration, implementation-plan, manifest, founder-index, and
  founder-artifact hashes;
- GPU backend identity;
- simulator source and configuration hashes;
- exact world identities and productivity lengths;
- finite bounded productivity;
- score recomputation from stored trajectories;
- coherent median repeat selection;
- policy and graph integrity flags.

Final artifact hashes:

| artifact | SHA-256 |
|---|---|
| suite record | `8c2118a7a69efdfcebfd50f6c86d3b5969e8b9703f8de744cd06d5b667c368eb` |
| summary | `5722caf6e5b25a365ce1835bbb2c236108b72d9b3b7119a5fe3ab3967a7b68ff` |
| Shinka stable | `824126e7aa4f9d98ca4186fc3c01c37a04975815849ae6fd772b108c9e7e6584` |
| Shinka punctuated | `777fc2a59b34b338eed8a366b9d985e548c02a69d9c60c4d2eabec51ba10c723` |
| initial scheduler | `4a052a218fbf5d208c5d3b4a39d523c4be4a9d8dd59920184d29c386bd5a6a18` |
| clone | `ffd1494480fce42d266fbaf63038aa8895398fcafd4640fa2afaa26d9b1d9eb2` |
| fixed parametric | `49751228a9376cadb95c22b51be07c1e232cd93844ce131f1f676b31e6b3478c` |
| fixed structural | `c5865a7960de3c1896ca6b4f92accb244de4c9035bd51a07eea2f13541eaea39` |
| fixed mixed | `7fcaf525263a57fd4c47b0bbd06213c8414fdc97adcd6584e24b777642fbb738` |
| stress responsive | `1c19fc1f8aada032c671fce77d579f75c432f02b32b2d1a35ed05eaa27959a5f` |

### 5.3 Regression tests

- Microcosmos full CPU suite: **413 passed, 1 skipped**.
- ShinkaEvolve full CPU suite: **538 passed**.
- Focused sealed-workflow suite: **10 passed**.

## 6. Why the result is negative

Three separate facts prevent a positive claim:

1. The primary punctuated-minus-initial confidence interval crosses zero
   broadly.
2. The punctuated descendant is behaviorally identical to fixed parametric
   mutation and does not beat it.
3. The unseen tail intervention did not harm clone, so the sealed panel did not
   instantiate the intended recovery problem.

It would be incorrect to claim that Shinka discovered a robust heredity policy,
that punctuated training improved evolutionary recovery, or that the project
demonstrated successful bounded RSI. It did demonstrate the complete machinery
for making such a test falsifiable.

## 7. Recommended next experiment

Do not retune r3 using its sealed outcomes. Preserve this run as a negative
result. A new run ID and entirely new development/sealed founders and seeds are
required.

A clean r4 should make only these targeted changes:

1. Prospectively screen disturbance *families* on training-only controls and
   require a harmful clone injury-minus-sham interval before outer search.
   Candidate families can include resource relocation, globally increased
   actuator cost, viscosity shift, or distributed injury. Avoid selecting only
   a hinge location that happens to improve locomotion.
2. Require the qualified disturbance to leave reproduction active and permit at
   least one trusted policy to generate genetically distinct descendants, but
   do not require a conventional policy to beat clone.
3. Retain the same bounded four-action Shinka ABI, CPPN genome, GPU-native
   lifecycle, founder readiness gate, matched stable/punctuated arms, and
   founder-first analysis. Those components worked.
4. Add an explicit behavior-novelty success criterion: a candidate must invoke
   at least two heredity actions on development to qualify as an *adaptive
   scheduler*. A single-action descendant can still win performance but must be
   reported as autonomous baseline rediscovery rather than a new policy.
5. Improve numerical reproducibility prospectively. Bind deterministic GPU
   settings if available, or use more numerical repeats only after profiling.
   Continue selecting a coherent median full execution rather than mixing
   episodes from different repeats.
6. Correct the outer-budget wording to distinguish evaluated generations from
   LLM-created proposals.

The minimal technical substrate should not be expanded with self-assembly,
sexual reproduction, predators, arbitrary topology, or simulator rewriting.
The next scientific bottleneck is a causally valid disturbance, not more code.

## 8. Key artifact locations

- Frozen r3 contract: `plans/evo2-actuator-rsi-r3-preregistration.md`
- Final implementation plan: `plans/evo2-rsi-final-implementation-plan.md`
- Founder-readiness builder:
  `microcosmos/experiments/evo2_ecosystem/build_ready_founder_bank.py`
- Authenticated founder bank:
  `microcosmos/experiments/evo2_ecosystem/founders/heredity_adaptation_v2/`
- r3 manifests:
  `microcosmos/experiments/evo2_ecosystem/heredity_adaptation_v3/manifests/`
- r3 sealed ledger/results/summary:
  `microcosmos/experiments/evo2_ecosystem/heredity_adaptation_v3/final_results/`
- Bounded Shinka initial scheduler:
  `ShinkaEvolve/examples/evo2_ecosystem/initial.py`
- Immutable evaluator:
  `ShinkaEvolve/examples/evo2_ecosystem/evaluate.py`
- Frozen finalists and lineage records:
  `ShinkaEvolve/examples/evo2_ecosystem/frozen/evo2-actuator-rsi-20260713-r3/`
- Outer archives and development selections:
  `ShinkaEvolve/examples/evo2_ecosystem/results/evo2-actuator-rsi-20260713-r3/`
- Sealed workflow:
  `microcosmos/experiments/evo2_sealed/workflow.py`

