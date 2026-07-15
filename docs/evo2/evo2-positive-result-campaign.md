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

### R7c completed search

`evo2-exploratory-r7c-20260714` completed eight program evaluations on the
punctuated training manifest in 33.6 minutes. Shinka generated seven
descendants for a recorded API cost of $1.4885. Five descendants passed the
bounded program contract and two failed static candidate validation before an
ecosystem simulation.

The best robust training score was generation 2 at `-0.000105` relative to the
standard-parametric ancestor. Generation 5 scored `-0.001057`, but its mean
candidate-minus-ancestor effects were positive in both refresh controls
(`+0.010131`) and resource-relocation treatments (`+0.011535`). It used standard
parametric mutation for 97.1% of births and made small evidence-gated departures
through the other trusted operators. Generation 4 showed the largest adaptive
behavioral change: nonstandard probability rose from 11.6% before relocation to
24.4% after relocation, and its mean relocation effect was `+0.007660`, although
its control cost made the robust aggregate negative.

Decision: do not declare a result from one-repeat training measurements. Freeze
generations 2, 4, and 5, plus all six fixed operators, and evaluate them with
three numerical repeats on the untouched development founders and worlds.

### R7 fresh development evaluation

All three frozen programs were evaluated on four untouched development
founders and two untouched development world seeds, with matched refresh and
relocation roles and three numerical repeats.

- Generation 2 did not transfer: robust score `-0.030647`; all three repeat
  scores were negative.
- Generation 4 remained slightly negative at `-0.004962`; one of three repeats
  was positive. Its mean refresh and relocation effects were both positive,
  but performance remained founder-sensitive.
- Generation 5 transferred: robust score `+0.007952`, with all three repeat
  scores positive (`+0.002128`, `+0.007952`, `+0.021558`). Mean refresh effect
  was `+0.058433`, mean relocation effect was `+0.003872`, and both candidate
  and standard-parametric ancestor survived all sixteen selected-repeat
  episodes. The program used standard mutation for 95.6% of births and made
  bounded evidence-gated departures among the other operators.

Generation 5 is therefore the first positive fresh-world Shinka result in the
campaign. It is not yet the final claim: screen every fixed trusted operator,
identify the strongest fixed alternative, and compare generation 5 directly
against that alternative under repeated paired evaluation.

The first fixed-baseline screen did not simulate: the generated baseline files
used a Python `del` statement to mark unused arguments, while the frozen bounded
candidate grammar intentionally forbids `Delete` AST nodes. All three attempts
recorded `CandidateValidationError`. No scores were used. The invalid sources
and metrics remain preserved; validator-compliant fixed programs omit the
unnecessary statement and are generated under a distinct baseline run ID.

### R8 decision and launch

The corrected one-repeat baseline screen found fixed clone at `+0.085810`
relative to standard parametric mutation on development worlds. Conservative
parametric (`-0.040914`) and exploratory parametric (`-0.040512`) were both
inferior. Structural and mixed screens were still running when R8 launched,
but the clone advantage was already an order of magnitude larger than the R7
Shinka advantage.

This identifies R7's central error: it optimized against a weaker
standard-parametric ancestor. R8 uses the exact fixed-clone program as both its
initial program and paired ancestor. Its search objective is therefore direct:
discover a context-dependent heredity scheduler that preserves clone's strong
ecological performance while using mutation only when online evidence makes it
beneficial. R8 uses ten total evaluations, one search repeat, the same training
manifest, model, bounded six-operator contract, simulator, and resource budget.

R8 launched as `evo2-exploratory-r8-clone-20260714`. The launch record pins the
clone source hash, training manifest hash, founder index hash, Shinka commit,
Microcosmos commit, model, and outer budget. Two remaining fixed-operator
development screens were already running when the R8 initial evaluation began.
Three-way GPU contention increased the initial evaluation time to 599.83
seconds. The screens were temporarily paused to keep resource contention from
triggering the local evaluator's twelve-minute timeout; they will be resumed
after R8. Future exploratory launches use a thirty-minute runaway limit so host
contention is not misclassified as candidate failure.

The generation-zero clone-vs-identical-clone evaluation scored `-0.005961`.
Because the two paired ecosystem trajectories still consume independent policy
randomness, exact self-comparison is not numerically zero. This run therefore
also measures the scale of a one-repeat search fluctuation. R8 search scores are
useful for proposal selection, but any descendant result must be repeated on
fresh worlds before it supports a claim.

### R8 completed search

R8 completed ten program evaluations in 2,576.55 seconds for a recorded API
cost of $2.5204. Seven programs were contract-valid and three failed bounded
candidate validation before simulation. Shinka used the negative results to
move from broad mutation (generation 1, only 2.6% clone, score `-0.068359`) to a
clone-tethered probe (generation 5, 97.4% clone, score `-0.005144`) and then an
evidence-bootstrap probe (generation 6, 97.3% clone, score `+0.002281`).

Generation 6 improved the relocation subset by `+0.017791` while costing
`-0.006315` in refresh controls. Its non-clone selection probability rose from
2.49% before relocation to 2.73% afterward. Shinka independently regenerated
the exact same source at generation 8. That duplicate scored `-0.001750`, with
refresh delta `+0.012206` and relocation delta `+0.008369`. The differing
aggregate sign for byte-identical code confirms that the one-repeat training
advantage is near the evaluator's stochastic/numerical resolution.

Decision: generation 6 is the R8 transfer candidate because it is the best
archive program and has an interpretable clone-preserving, evidence-gated
mechanism. Evaluate its frozen source with three repeats on untouched
development worlds directly against fixed clone. Do not treat the training
score as a positive result by itself.

The remaining fixed development screens completed after R8: fixed structural
scored `-0.030796` and fixed mixed scored `-0.120095`, both relative to standard
parametric. Together with conservative (`-0.040914`) and exploratory
(`-0.040512`), these measurements confirm that fixed clone (`+0.085810`) is the
strongest trusted fixed policy on this development panel.

The R8 development confirmation was frozen under
`evo2-exploratory-r8-confirmation-20260715`. It copies the exact generation-6
source and exact fixed-clone source, records their hashes and both repository
commits, and uses the untouched R6 development manifest. The candidate is being
evaluated directly against clone with three complete numerical repeats on GPU.

### R8 development result and R9 decision

The R8 generation-6 candidate did not transfer as an overall improvement. Its
three development repeat scores were `+0.012590`, `-0.048194`, and `-0.013583`;
the coherent median was `-0.013583`. The candidate improved relocation episodes
by `+0.013249` on average but cost `-0.006860` in matched refresh controls. It
selected clone for 97.4% of births, yet mean clone probability was 97.47% before
relocation and 97.36% afterward. The policy therefore paid mutation cost in
healthy worlds without sharply increasing exploration under the shock.

This is a measured algorithmic failure rather than a simulator blocker. R9
starts from the exact R8 generation-6 source and remains paired directly against
fixed clone. Its proposer feedback states the observed mechanism: retain
effectively pure cloning while parent energy and recent intake are healthy, and
use only bounded scarcity/decline evidence to unlock sparse conservative or
standard parametric probes under ecological stress. Development is now treated
as iterative campaign feedback; any later claim requires untouched sealed
founders and seeds.

R9 launched as `evo2-exploratory-r9-stress-gated-20260715` with twelve total
evaluations. Generation zero is the exact R8 generation-6 source; every program
is paired against the exact fixed-clone source. The launch record pins both
source hashes, the training manifest and founder-index hashes, both repository
commits, model, prompt guidance, and budget. Search evaluation remains one
repeat for throughput; descendants must pass repeated fresh-world evaluation.

### R9 interface audit and R10 relaunch

R9 was intentionally stopped after generations zero through four were recorded.
The first three descendants exceeded the existing 60-nonblank-line candidate
limit and failed before simulation. The valid generation-4 stress policy chose
clone for every realized birth and scored `-0.014883`. Its mean clone
probability increased from 99.29% before relocation to 99.46% afterward—the
opposite of the intended response.

Inspection found a proposer-interface error. The exploratory task said that
population summaries were available but did not document their positions. The
R8 source inherited by R9 mislabeled `population_stats[1]` (mean energy) as
population growth, `[2]` (population change) as diversity, and `[3]` (birth
rate) as lineage entropy, while ignoring `[4]` (death rate) and `[5]` (mean
intake). Shinka therefore generated coherent code against the wrong semantic
interface. The trusted simulator and array values were correct; the outer task
description was incomplete.

The task now explicitly documents all parent, population, and operator indices,
the opaque RNG, and the 60-line budget. Candidate-validation feedback also
includes the safe validation reason so the proposer can repair oversized or
unsupported code. Twenty-six focused CPU tests pass. These changes were
committed before a new search.

R10 launched as `evo2-exploratory-r10-abi-correct-20260715`, starting from and
comparing directly against exact fixed clone for twelve evaluations. It uses the
same training worlds and score; only the proposer now receives the ABI it was
always meant to program against. Any R10 candidate still requires repeated
fresh-world confirmation.

### R10 completed search and treatment decision

R10 completed all twelve evaluations in 3,306.26 seconds. Shinka generated
eleven descendants for a recorded API cost of $1.3200. Ten of the twelve
generation programs were contract-valid; generations 1 and 7 were rejected
before simulation because the bounded grammar did not admit ordinary JAX array
slices. The rejected sources and exact validation feedback remain preserved.

The corrected ABI solved the interface defect, but it did not produce a useful
heredity result on resource relocation. Every contract-valid generation used
clone for 100% of realized births. Generation 2 received the highest one-repeat
score, `+0.005588` relative to clone, and generation 8 received `+0.002175`, but
both were behaviorally clone policies. Their tiny non-clone probabilities never
selected a mutation. These point scores are therefore measurements of residual
clone-versus-clone execution noise, not autonomous improvement.

The remaining valid descendants scored between `-0.015421` and `-0.007619`,
again while realizing only clone. R10 is consequently complete as a negative
treatment result: Shinka produced readable stress schedulers against the now
correct semantic interface, but relocation did not drive their stress signals
far enough to alter heredity. No R10 program advances to development.

Decision: keep the six-action CPPN heredity system, fixed-clone comparator,
paired evaluator, and exact ABI. Retire resource relocation as the active search
treatment. The next exploratory search will use the already implemented
persistent actuator-injury event, because the earlier prospective training
measurement established that head-hinge gain `0.1` was harmful but survivable
for clone (`95%` interval `[-0.135546, -0.009778]`). This changes no physics,
controller, mutation kernel, score, or candidate privilege. It only substitutes
an existing event that changes the controller problem for one that did not
create useful hereditary pressure. The next proposer prompt will ask for
measurable adaptive exploration rather than effectively pure cloning.
