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

### R11 head-actuator search launch

R11 launched as `evo2-exploratory-r11-head-injury-20260715` for sixteen total
evaluations. It uses the R6 training founders and world seeds, with every
resource-relocation fork transformed canonically into a matched uninjured/null
world and a persistent head-actuator injury world. At step 7,500 the first two
of six hinge gains change from `1.0` to `0.1`; all other simulator, resource,
lifecycle, founder, seed, horizon, and scoring values remain unchanged.

The initial program is fixed standard-parametric mutation so generation zero
immediately measures a real mutation operator. Every generation is paired
directly against exact fixed clone. The proposer receives the corrected ABI and
is explicitly asked to realize mutation, preserve uninjured performance, and
improve injured recovery rather than produce unused non-clone probabilities.
The launch record pins Shinka commit `dd3ea42`, Microcosmos commit `d252f87`,
the candidate and ancestor source hashes, founder-index hash, canonical
manifest hash, model, scenario family, and outer budget.

R11 was stopped after generations zero through two were preserved. Generation
zero established the fixed-standard baseline against clone (`-0.045209`) while
using standard mutation for every birth. Its mean direct candidate-minus-clone
effects were positive in both uninjured (`+0.003251`) and injured (`+0.030100`)
worlds, but the robust paired aggregate was negative because effects were highly
founder-sensitive. Generations 1 and 2 were rejected before simulation because
their otherwise bounded JAX expressions used `jnp.mean` and array `.astype`,
which the candidate grammar did not admit. Both failures returned the same
generic call error.

Those operations are pure, fixed-shape, and do not expand candidate privilege.
The grammar now admits `jnp.mean`, fixed slicing, and `.astype`; a focused test
executes all three through the real load and JAX smoke-validation path. R11 is
retained as an interrupted infrastructure run. A distinct R12 run will repeat
the same scientific experiment and budget after this validator-only repair.

### R12 validator replay and stop

R12 launched as `evo2-exploratory-r12-head-injury-20260715` with the same
sixteen-evaluation scientific configuration as R11 and pinned Shinka commit
`e6e0f93` and Microcosmos commit `1369b73`. Generation zero completed normally:
fixed standard mutation scored `-0.073939` against clone and used the standard
operator for every birth. Generations 1 and 2 were again rejected before
simulation. Their preserved sources revealed two additional pure JAX idioms
missing from the candidate grammar: `jnp.max` and immutable indexed updates
such as `logits.at[0].add(...)`.

R12 was stopped before generation 3 could consume another ecosystem
evaluation. The validator now admits common bounded reductions and JAX's pure
indexed-update interface. A regression test executes both exact idioms through
the real R4 loader and smoke validator, and the preserved R12 generation-1 and
generation-2 sources both pass validator replay. This remains a sandbox repair,
not a change to worlds, score, operators, candidate inputs, or physics. R12 is
retained as an interrupted infrastructure run; the next run repeats the same
experiment from a new immutable result directory.

### R13 ABI-orientation repair

R13 repeated the same sixteen-evaluation head-injury experiment. Generation
zero completed with fixed standard mutation at `-0.106121` relative to clone.
Generation 1 was rejected because it used ordinary local tuple unpacking, which
the sandbox still disallowed. More importantly, both generated descendants
transposed the documented operator-stat table: they used
`operator_stats[:, 0]` instead of `operator_stats[0]`, producing three-element
vectors where the policy requires one value for each of six operators.

The run was stopped while generation 2 was evaluating; its partial artifacts
are retained. The sandbox now permits tuple unpacking only into new local names,
and the smoke validator still rejects incompatible shapes. The proposer prompt
now gives executable indexing examples for success, usage, and evidence and
explicitly forbids the transposed form. Twenty-eight focused CPU tests pass.
These changes clarify and admit the intended fixed ABI without changing the
candidate's information, worlds, score, heredity operators, or simulator.

### R14 completed adaptive-mutation search

R14 launched from the preserved, ABI-correct Shinka policy generated in R12
generation 2 rather than paying for fixed-standard initialization again. It
completed all twenty evaluations in 7,236.91 seconds, generated nineteen
proposals for a recorded API cost of `$2.5029`, and retained every source,
metric, prompt, database record, and launch hash. Sixteen of the twenty
generation programs were valid. Four proposals were rejected before simulation
only because they exceeded the
60-nonblank-line source limit.

The initial adaptive policy used 51.5% conservative and 45.7% standard mutation
and scored `-0.087924` relative to clone. Shinka's generation-3
`regime_blended_repair` policy improved the one-repeat training score to
`-0.044892`; it used 51.4% conservative, 41.8% standard, and 5.5% exploratory
mutation and had positive mean effects in both sham (`+0.020082`) and injury
(`+0.016929`) worlds. Generation 19 was close at `-0.049454` and had the largest
positive injury mean (`+0.047668`) with a small sham cost (`-0.008196`). No R14
training policy beat clone on the robust aggregate.

Generation 3 was frozen and evaluated immediately on the fresh development
founders for three numerical repeats. All repeat scores were negative:
`[-0.092445, -0.105449, -0.117987]`. Its selected robust score was `-0.105449`,
so the training improvement did not transfer. The matched fixed-conservative
baseline scored `-0.248492` with repeats
`[-0.248492, -0.255023, -0.246846]`. Shinka therefore discovered a policy that
substantially reduced the cost of conventional mutation, but it did not beat
clone and is not a positive final result.

Mechanistically, nearly every R14 policy mutated nearly every birth. The best
policy used clone for only 0.5% of realized births; the development loss shows
that this exploration budget remains too expensive across founders. The next
search starts from exact clone and asks Shinka to discover sparse,
stress-triggered mutation: high clone probability while ecology is stable, with
mutation spending tied to decline, death, energy, and operator evidence. The
source limit is raised from 60 to 100 nonblank lines because it rejected four
otherwise bounded proposals and does not protect the simulator; AST capability
restrictions, fixed output shape, JAX smoke validation, and runtime limits
remain unchanged.

### R15 completed sparse-mutation search

R15 launched as `evo2-exploratory-r15-sparse-head-injury-20260715` from exact
fixed clone and compared every program directly with that same frozen clone.
It completed twenty evaluations in 8,425.48 seconds, generated nineteen Shinka
proposals for a recorded API cost of `$2.6885`, and preserved every source,
metric, prompt, launch hash, and SQLite archive record. Seventeen generation
programs were contract-valid; generations 3, 17, and 19 were rejected before
simulation for exceeding the documented 100-nonblank-line limit.

Shinka moved immediately from exact clone toward sparse heredity schedulers.
The valid descendants generally cloned 93--99% of births and spent the
remaining budget on conservative and standard parametric mutation. Generation
12 was the one-repeat training champion at `+0.009561` versus clone, generation
15 scored `+0.001078`, and generation 18 scored `+0.000215`. Generation 18 had
positive mean candidate-minus-clone effects in both sham (`+0.049604`) and
injury (`+0.021906`) training worlds while cloning 96.6% of realized births.

None of the apparent training improvements transferred to the fresh
development founders under three complete numerical repeats:

| Frozen R15 program | Median paired score vs clone | Repeat scores |
|---|---:|---|
| generation 1 | -0.034845 | all three negative |
| generation 9 | -0.017296 | all three negative |
| generation 12 | -0.056643 | `[-0.056643, -0.066690, -0.047388]` |
| generation 15 | -0.048429 | `[-0.048429, -0.041304, -0.077611]` |
| generation 18 | -0.044665 | `[-0.040972, -0.056818, -0.044665]` |

The matched fixed-standard baseline scored `-0.048055` with repeats
`[-0.048055, -0.037258, -0.062119]`; the earlier fixed-conservative baseline
scored `-0.248492`. Shinka therefore learned to suppress most of the large cost
of conventional mutation, but no R15 policy beat no-mutation clone on fresh
founders. R15 is a complete negative result, not a finalist.

The mechanism trace explains why another search on the identical treatment is
not justified. The promising policies became *more* clonal after injury. For
example, generation 18's mean clone probability rose from 96.1% before the
event to 97.9% afterward. The injury at step 7,500 left only 4,500 of the
12,000 simulated steps for new variants to be born, receive ecological credit,
and spread. Sparse mutation produced too few post-event trials, while broad
mutation remained founder-sensitive and costly.

### R16 early-injury treatment

R16 keeps the simulator, CPPN controller, six trusted heredity operators,
founders, world seeds, injury gains, score, and total 12,000-step budget fixed.
It changes only the event time from step 7,500 to step 4,000, expanding the
post-injury adaptation window from 4,500 to 8,000 steps without adding compute.
The exploratory and confirmation manifest builders now accept an explicit
chunk-aligned `--event-step`, and the chosen value is recorded in launch
metadata and canonical manifest bytes. Fixed standard mutation and the R15
generation-18 sparse scheduler were each measured against exact clone with
three numerical repeats on the unchanged R6 training founders under this
revised timing.

Continuous fixed-standard mutation scored `-0.048503` on the robust aggregate.
Its stable sham-world effect was `-0.067226`, while its injury-world effect was
`+0.075270`: mutation is useful after the early injury but its continuous cost
in the stable world is larger. R15 generation 18 nearly closed the aggregate
gap at `-0.003519`, with repeat scores
`[+0.017028, -0.026118, -0.003519]`. It cloned 96.3% of realized births and had
a smaller sham cost (`-0.024845`) while retaining a positive injury effect
(`+0.027912`). These are replicated screen results, not a claimed positive
result.

The screen establishes a concrete R16 optimization target: condition sparse
mutation on observable ecological state and accumulated operator evidence so
that mutation is spent when its post-injury benefit is likely to exceed its
stable-world cost. R16 proper starts from the frozen R15 generation-18 source,
compares every descendant directly with frozen fixed clone, and uses twenty
Shinka evaluations on the same early-injury treatment.

### R16 completed search and valid-depth continuation

R16 proper ran as `evo2-exploratory-r16-early-head-injury-20260715`. It
completed twenty nominal generation slots in 4,592.08 seconds, generated
nineteen proposals for a recorded API cost of `$2.8186`, and preserved the
complete archive. Generation 3 was the one-repeat training champion at
`+0.001406` versus clone and used clone for 99.1% of realized births.
Generation 7 was also slightly positive at `+0.000187` with 99.3% clone use.
The other valid descendants were negative.

Generation 3 was immediately frozen and evaluated on untouched development
founders for three numerical repeats. Its repeat scores were
`[-0.000459, +0.013501, -0.000673]`, giving a robust score of `-0.000459`.
Its mean sham effect was `+0.015844` and mean injury effect was `+0.002540`.
This is much closer to clone than the R14 and R15 development results, but it
does not robustly beat clone and is not a finalist. The distinct generation-7
pressure/evidence mechanism was also frozen for development confirmation.

Only seven unique R16 generations reached ecosystem simulation. Twelve
proposals exceeded an arbitrary 100-nonblank-line limit and one exceeded an
arbitrary 1,024-AST-node limit. These were not capability or runtime
violations: two preserved rejected sources execute successfully through the
real fixed-ABI JAX smoke validator when those size checks are omitted. The two
non-safety complexity checks and the corresponding prompt sentence were
therefore removed. The source-byte bound, AST capability allowlist, immutable
regions, opaque RNG, fixed six-value output, JAX smoke execution, process
timeout, fixed trusted operators, worlds, score, and physics remain unchanged.
The change removes code rather than adding infrastructure; twenty-eight focused
CPU tests pass.

Because thirteen of twenty nominal slots did not perform the intended
experiment, a clean continuation starts from frozen generation 3 and retains
fixed clone as the comparator. Its objective is to improve the worst-founder
and repeat behavior while preserving the candidate's positive sham and injury
means. This continuation supplies the valid search depth that R16's obsolete
complexity checks prevented.

### R16b valid-depth continuation and first sealed result

R16b ran as
`evo2-exploratory-r16b-valid-depth-early-head-injury-20260715` from the frozen
R16 generation-3 scheduler, while retaining exact fixed clone as the paired
comparator. All twenty generation programs passed the bounded capability and
runtime contract and reached ecosystem simulation. The run generated nineteen
Shinka proposals, cost `$3.1051`, and completed in 13,563.72 seconds. The
longer elapsed time reflects concurrent confirmation work, not a changed
simulation budget. Every proposal, prompt, metric, source, archive relation,
and launch hash is retained.

Generation 11 was the only policy to survive untouched development
confirmation. Its three complete repeat scores against clone were
`[+0.005238, +0.014974, +0.014813]`, for a robust score of `+0.014813`; all
three repeats were positive. Mean sham and injury effects were respectively
`+0.017653` and `+0.055724`. The policy cloned 95.41% of realized births, used
conservative parametric mutation for 4.46%, and used standard parametric
mutation for 0.13%. It computed separate stable, renewal, stress, and injury
rescue signals and combined them with per-operator success, usage, and evidence
estimates. R16b therefore produced a replicated development-positive
Shinka-generated heredity policy rather than a one-repeat training artifact.

The frozen generation-11 source was then evaluated once on the sealed founder
partition with three numerical repeats. It did **not** beat clone on the
predeclared robust aggregate: the score was `-0.005367`, with repeat scores
`[-0.012919, -0.003857, -0.005367]`. Mean sham and injury effects remained
positive (`+0.003490` and `+0.007455`), and the mean of the selected repeat's
sixteen paired deltas was `+0.000949`, but one injury-world delta of
`-0.101044` dominated the lower tail. This is a sealed negative result and is
not relabeled as success.

The failure mode has changed. The original continuous-mutation policy paid a
large stable-world cost; generation 11 removed that mean tradeoff but remained
too sensitive to particular founder/world combinations. R17 therefore changes
neither physics, injury, horizon, scoring, trusted operators, nor candidate
inputs. It trains on a fixed cross-panel manifest with eight founder genomes
instead of four while holding the number of worlds constant by using one world
seed per founder. A separately seeded founder bank is frozen before R17 begins
and reserved for fresh development and sealed confirmation. This is the
smallest correction that directly addresses the measured founder-tail failure
without training on the failed sealed outcomes or adding another ecological
mechanism.

### R17 completed cross-founder search and sealed failure

R17b ran as
`evo2-exploratory-r17b-crosspanel-founder-generalization-20260715`. It
completed all twenty valid ecosystem evaluations, generated nineteen Shinka
proposals for a recorded API cost of `$3.2743`, and retained every source,
metric, prompt, patch, archive relation, launch hash, and SQLite record. The
run took 16,730.60 seconds while sharing the GPU with confirmations and R18
startup. Generation 2 was the one-repeat training champion at `+0.014354`
relative to exact clone; later training-positive generations 4, 11, 13, and 18
did not exceed it.

Generation 2 was frozen before confirmation. On the independently generated
R17 development founders, its three repeat scores were
`[+0.007631, +0.005428, +0.004459]`; all three were positive and the robust
score was `+0.005428`. The source therefore met the preregistered development
rule. Its one permitted sealed examination failed: the robust score was
`-0.033145`, all three repeat scores were negative, and several individual
founder/world deltas had large negative tails. Direct sealed comparison with
fixed standard mutation was also negative (`-0.011036`). R17 generation 11 was
later checked on development because it represented a distinct mechanism, but
all three repeats were negative and it was rejected. No other one-repeat R17
score was promoted.

The generation-2 policy used approximately 75--80% clone, 18--24%
conservative parametric mutation, and less than 2% standard mutation depending
on the founder panel. It improved the mean in some worlds but spent too much
mutation on fragile founders. The sealed failure therefore supports a narrower
continuation: learn on the complete now-exposed hard founder panel and require
generalization to a new independently generated founder bank.

### R18 frozen hard-founder continuation

Before R18 search, a new `4/4/8` training/development/sealed founder bank was
generated with panel seed `7007`, screening seeds `13000` and `13001`, and the
unchanged clone-only viability screen. Development and sealed founder bytes
were permission-sealed. The complete exposed R17 sealed panel—not only its
worst cases—became the R18 training panel, using one fixed world seed per
founder and the unchanged sixteen-world budget. The simulator, 12,000-step
horizon, step-4,000 injury, score, fixed-clone comparator, policy ABI, and six
trusted heredity operators remain unchanged.

On this exact hard training panel, fixed standard mutation scored `-0.027646`
relative to clone: its sham effect was `-0.016434`, its injury effect was
`+0.010889`, and survival was 0.5625 versus clone's 0.75. The first R18
evaluation of the inherited R17 generation-2 policy scored `-0.022292`, with
negative sham and injury means. R18 generation 1 reduced that gap to
`-0.002222` by cloning 98.19% of births and using conservative mutation for
1.81%, but it remained negative in both treatment means and was not promoted.
R18 continues for the frozen twenty-evaluation Shinka budget. Any candidate
must beat clone in all three fresh development repeats before the one permitted
sealed evaluation.

### R18 lifecycle interruption and minimal recovery

R18's interactive launcher ended after generation 13 when the Codex turn that
owned its transient process scope exited. Generation 13's ecosystem evaluation
completed and its artifacts are intact, but the outer runner did not ingest the
result or propose generation 14. This was an external process-lifecycle failure,
not a candidate, simulator, GPU, or scientific failure.

Generations 11--13 added no promotable result. Generation 12 produced a
one-repeat robust score of `+0.000050`, but its injury effect was `-0.005083`
and its realized policy was only 98.19% clone plus 1.81% conservative mutation.
It therefore failed the preregistered requirement for a real ecology-conditioned
non-clone mechanism and was not opened on development founders. Generations 11
and 13 scored `-0.000826` and `-0.000020`, respectively.

The smallest recovery is R18b, initialized from the frozen generation-12 source
and run as a persistent guarded user service. It uses the same hard training
manifest, founder index, fixed-clone comparator, physics, score, policy ABI,
trusted operators, and hidden founder partitions. Seven evaluations provide one
explicitly recorded duplicate initialization measurement plus six new Shinka
proposals, restoring the intended search depth without adding resume machinery
or changing the experiment. Together R18 and R18b contain nineteen proposals;
the duplicate initialization is accounted for as recovery overhead rather than
as additional search evidence.

The first persistent-service launch stopped before runner construction because
the noninteractive service environment did not include `npx` on `PATH`. It
created only the immutable launch record and copied training manifest; no
candidate was proposed or evaluated. Those two files were preserved in a
`startup-failed-no-npx` directory. The first corrected command also stopped
before Python startup because its replacement `PATH` omitted the Conda
environment's binary directory. The final service environment includes both
the existing `sakana` interpreter and pinned Node installation. No experimental
code or configuration changed in either correction. R18b then launched under a
persistent user service with 20 GiB soft and 24 GiB hard memory guards; the
existing thirty-minute monitor now follows that run.

### R18b negative completion and R19 replicated search

R18b completed all seven evaluations. No program qualified for development.
Generation 1 was the best new valid proposal at `-0.000030` robust, with
`+0.009637` sham and `-0.006563` injury effects. It realized 98.16% clone and
1.84% conservative mutation, so it was neither positive nor a demonstrated
injury-conditioned mechanism. Later valid generations scored between
`-0.001042` and `-0.008758`. Generation 6 failed bounded execution because its
new expression referenced `fragile_lock` before that value was defined; it did
not reach ecosystem simulation.

The duplicate evaluation of frozen R18 generation 12 scored `-0.001631`, after
the identical source scored `+0.000050` in R18. This sign reversal directly
measures the selection problem: improvements at the scale currently being
searched are smaller than one-repeat numerical variation. Continuing to rank
new programs by a single repeat would spend search budget on noise.

R19 therefore changes one measured property only: each hard-panel training
evaluation uses three numerical repeats and the evaluator's existing coherent
median selection. Physics, founders, worlds, horizon, injury, score, fixed-clone
comparator, policy ABI, trusted operators, model, and sealed partitions remain
unchanged. The initial program is frozen R18b generation 1, the closest valid
new descendant. This costs three times more simulation per proposal but gives
Shinka a replicated fitness signal before promotion; it adds no new ecological
feature, gate, or candidate capability.

R19 launched as `evo2-exploratory-r19-replicated-hard-founder-20260716` for
twenty evaluations under the persistent 20/24 GiB guarded service. Its launch
record pins ShinkaEvolve commit `f23926e`, Microcosmos commit `dc9604e`, the
frozen generation-1 source hash, clone hash, training manifest hash, founder
index hash, model, and three-repeat evaluation budget. The recurring monitor
now follows R19.

### R19 negative completion and R20 compact reset

R19 completed twenty replicated evaluations in 13,060.27 seconds and generated
nineteen Shinka proposals for `$3.6230`. Seventeen programs were valid. No
candidate beat clone. Generation 14 was best at `-0.000067`, with all three
repeat scores negative (`-0.000067`, `-0.000060`, and `-0.000109`). Across the
valid archive, realized behavior repeatedly collapsed to approximately
98--99% clone plus a small, nearly constant conservative-mutation fraction.
Programs that used more standard mutation improved some injury measurements but
paid larger sham or founder-tail costs. The three-repeat evaluator therefore
removed false positives as intended and established that another continuation
from the same large program would be low-value.

R20 keeps the replicated evaluator and every ecological input, operator, world,
founder, physical rule, and score fixed, but resets the editable parent to a
compact stress-gated program. The seed directly maps decline, death, population
loss, energy, intake, parent readiness, and accumulated standard-operator
evidence into a clone-to-standard rescue pulse. This is both the smallest test
of whether the existing observable state can support injury-conditioned
mutation and a simpler search surface for Shinka. It adds no simulator state or
privileged event signal. Shinka must improve the compact program rather than
continue appending decorative gates to the R19 lineage.
