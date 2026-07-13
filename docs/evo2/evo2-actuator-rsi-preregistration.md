# Evo² actuator-adaptation prospective experiment preregistration

Status: **FROZEN — immutable prospective contract**  
Draft date: 2026-07-13  
Freeze date: 2026-07-13  
Planned run ID: `evo2-actuator-rsi-20260713-r1`

This is a prospective follow-up to the completed
`evo2-production-20260712-r1` pilot. It does not amend, replace, or reinterpret
the pilot's sealed result.

## 1. Confirmatory question

> Does the selected final descendant of the punctuated-trained Shinka heredity
> program enable completely fresh clonal CPPN populations to attain higher
> normalized post-injury productivity than the initial Shinka program on unseen
> founders, ecological seeds, and a persistent unseen actuator-injury location,
> under identical simulator and mutation-action budgets?

## 2. Primary contrast

The sole confirmatory primary contrast is:

```text
selected final punctuated Shinka program
minus
initial Shinka program
```

on direct sealed injured-world normalized post-event productivity AUC.

The comparison is paired on exact:

- founder ID and artifact hash;
- ecological world seed;
- pair ID;
- actuator-injury vector;
- event boundary;
- horizon and simulator configuration;
- trusted mutation-action budget.

## 3. Evidence prerequisites

The outer searches will not start unless the benchmark gate establishes that:

1. clone births remain bitwise identical to their parent/founder;
2. actuator injury harms clone relative to its exact sham fork;
3. at least one frozen conventional mutation policy beats clone directly in the
   injured worlds with a founder-aware paired 95% interval above zero;
4. at least one mutation policy survives in at least two-thirds of training
   founder×seed worlds;
5. its median injured world contains at least four post-event births;
6. its median maximum generation is at least two greater than the generation at
   the event boundary;
7. every included world remains integrity-valid.

If the gate does not pass under the predefined grid and fallback, the experiment
stops without Shinka search.

## 4. Immutable biological substrate

Use the completed pilot's frozen `SimulatorConfig` and calibrated ecology:

- 32 organism slots;
- 8 nodes and 6 bending hinges per organism;
- 64×64 world;
- fluid enabled;
- 8 initial organisms;
- the existing resource, metabolism, energy, birth, death, and actuation-cost
  constants;
- the existing direct 4-input/1-output padded TensorNEAT CPPN ABI;
- the existing clone, parametric, structural, and mixed mutation kernels.

Do not tune resource, lifecycle, physics, controller, or mutation constants
during actuator-benchmark calibration.

## 5. Persistent actuator treatment

Actuator efficacy is a world-level vector over the six local bending hinges. It
applies to all current organisms and every child born after the event. Intended
actuator commands retain their original metabolic cost.

### Calibration grid

Test only these injured gains:

```text
0.1
0.2
0.4
```

All healthy hinges remain at `1.0`.

### Spatial partitions

- training injury: local hinges `[0, 1]`;
- development injury: local hinges `[2, 3]`;
- sealed injury: local hinges `[4, 5]`.

The exact manifest stores all six gain values. Candidate code never receives the
injury vector, event kind, event time, or absolute ecological time.

### Event schedule

Use 500-step chunk boundaries and an 8,000-step horizon. Training and final
manifests draw from the frozen event-step set:

```text
3500, 4000, 4500
```

Every injured world has an exact sham twin that forks the same pre-event state
and retains all-one actuator gains.

If the initial calibration schedule produces too few post-event births under
every mutation policy, the only permitted schedule fallback uses event step
`4500` for all calibration worlds. No other timing, resource, or lifecycle
tuning is permitted.

## 6. Founder generation and partitions

Generate a fixed candidate pool from the existing deterministic varied-founder
initializer. Candidate ranges are assigned to partitions before viability
screening. Screen only in an uninjured reference world using clone heredity.

Selection may consider only:

- CPPN graph/cache validity;
- finite state;
- uninjured productivity;
- uninjured births, survival, and generation depth.

No founder may be selected by actuator-injury response.

Freeze at minimum:

```text
3 training founders
2 development founders
3 sealed founders
```

All selected genotype hashes must be distinct. The numeric NPZ artifacts,
canonical index, screening record, selection seeds, and hashes are immutable
after creation.

## 7. Ecological observations

Planned minimum panel:

- training search: 3 founders × 3 ecological seeds = 9 worlds per arm;
- development: 2 founders × 4 ecological seeds × injury/sham = 16 worlds;
- sealed: 3 founders × 6 ecological seeds × injury/sham = 36 worlds.

World seeds are disjoint across partitions. Founder is the generalization
cluster; additional seeds from one founder do not count as independent founder
transfer.

The exact disjoint seed panels are:

```text
uninjured founder screening: 131, 257, 389
actuator calibration:         4101, 4211, 4307
outer training:               1101, 1102, 1103
development:                  2101, 2102, 2103, 2104
sealed final:                 3101, 3102, 3103, 3104, 3105, 3106
```

Founder hashes do not yet exist at preregistration freeze because the founder
screen itself is prospective. The write-once founder index and its canonical
hash become an immutable downstream binding immediately after the predeclared
screen completes; no injury result participates in founder selection.

## 8. Benchmark calibration rule

Evaluate clone, fixed parametric, fixed structural, and fixed mixed mutation on
training founders for each gain in the predefined grid.

Select the strongest injury, meaning the smallest gain, satisfying every gate in
Section 3. If no gain passes, apply the one timing fallback and repeat the same
gain grid once. Record every attempted result.

After selection, run one development confirmation with the middle-body injury.
The benchmark must preserve:

- harmful injury-minus-sham clone effect;
- positive injured-world advantage of the winning conventional mutation policy
  over clone;
- survival/birth/generation floors;
- complete integrity.

No retuning follows development access.

## 9. Candidate heredity ABI

The default candidate returns exactly four finite scores for:

```text
clone, parametric, structural, mixed
```

Trusted code clips the scores, uses deterministic `argmax`, applies one trusted
mutation, validates the graph, and fails closed.

The six-value/two-pass MutationPlan is excluded from the core experiment unless
a separately recorded pre-search gate shows that the four-score interface is
materially underpowered. If enabled, every policy receives the same maximum two
mutation actions per birth and compute-matched conventional controls are added.

## 10. Matched outer searches

Run one stable and one punctuated Shinka search sequentially.

- stable: sham all-one event;
- punctuated: head-side persistent actuator injury.

Freeze identical:

- initial program;
- proposal model/command, temperature, effort, and token budget;
- outer seed;
- 15 proposal generations per arm, including invalid proposals; an invalid
  proposal receives no replacement and consumes its generation budget;
- archive/island settings;
- evaluator timeout and repair limits;
- founders, ecological seeds, event steps, horizon, and score;
- simulator and mutation-action budgets;
- three full numerical measurements per candidate;
- top-three development selection rule.

The first completed sibling archive is unreadable to the second proposer.
Development and sealed artifacts remain unreadable throughout both searches.

## 11. Final policy set

Freeze and evaluate:

1. clone;
2. fixed parametric mutation;
3. fixed structural mutation;
4. fixed mixed mutation;
5. handwritten stress-responsive scheduling;
6. initial Shinka program;
7. selected stable-trained Shinka program;
8. selected punctuated-trained Shinka program.

If two-pass mutation is enabled, add compute-matched double-parametric and one
fixed parametric/structural composition.

## 12. Program-lineage assay

Before sealed access, obtain the selected champion's actual `parent_id` chain
from the authenticated Shinka database. Deduplicate source-identical copies by
source SHA while retaining provenance.

Evaluate:

- root/initial program;
- closest distinct ancestor at one-third depth;
- closest distinct ancestor at two-thirds depth;
- direct parent;
- final program.

If the chain is short, evaluate every distinct ancestor. The initial-versus-final
contrast is confirmatory. Intermediate results are descriptive and need not be
monotonic.

Every program begins with completely fresh sealed clonal ecosystems. No organism
or state from outer search is reused.

## 13. Score and secondary metrics

Primary episode score:

```text
P_chunk = fulfilled mouth uptake / (chunk_steps * dt)
P_ref   = sum(post-event resource regeneration map)
q_chunk = clip(P_chunk / max(P_ref, epsilon), 0, 1)
score   = mean(q_chunk over frozen post-event chunks)
```

Candidate score is the equal-weight mean of founder-level means. Do not normalize
by the candidate's own pre-event performance.

Secondary metrics:

- injury-minus-sham effect;
- final productivity;
- survival/extinction;
- censored recovery time;
- post-event births and generation depth;
- operator use;
- parent–child genome distance;
- node/connection changes;
- invalid/fallback counts.

Diversity is diagnostic, not rewarded.

## 14. Numerical measurements and integrity

Execute every complete policy/manifest evaluation exactly three times. Require
integrity in all three and select one coherent median-scoring complete execution
with stable tie-breaking. Never median episodes separately and never count the
three measurements as ecological replicates.

The predeclared acceptable numerical-dispersion warning threshold is an absolute
range greater than `0.015` in full-manifest candidate score. Exceeding the
threshold does not authorize choosing a favorable repeat; it triggers an
integrity review before candidate selection.

Accepted finalist evaluations require:

- zero candidate-policy violations;
- zero invalid graph/infrastructure events;
- exact operator accounting;
- finite state and score;
- verified candidate, simulator, manifest, founder, run-spec, database, and
  analysis hashes.

## 15. Statistical analysis

For every policy contrast:

1. compute direct paired effects for identical
   `founder_id × world_seed × pair_id` observations;
2. resample founders first;
3. resample paired observations within founder;
4. compute founder means;
5. give founders equal aggregate weight;
6. use 10,000 bootstrap replicates, seed `20260712`, and a percentile 95%
   interval;
7. report mean, median, fraction positive, every founder effect, and aggregate
   interval.

The three GPU measurements are not bootstrap units.

Key secondary contrasts:

- final punctuated minus strongest compute-matched conventional baseline;
- final punctuated minus final stable;
- final punctuated minus stress-responsive;
- conventional mutation minus clone.

One outer-search pair supports a selected-policy case study, not a general claim
that punctuated Shinka training is superior.

## 16. Conditional analyses

Run exact organism ancestor–descendant healthy/injured common gardens only if the
primary program-lineage result is positive. Run one minimal code-mechanism
ablation only after writing its hypothesis. If designed after sealed access, use
a fresh post-final panel and label it exploratory.

## 17. Stop rules

Stop without outer search if the mutation-sensitive benchmark gate fails. Stop
the optional MutationPlan if it requires TensorNEAT internal rewriting,
identity-unsafe structural mutation, frequent fallback, or unequal mutation
action budgets. Stop before sealed if holdout permissions, hashes, or finalist
ancestry cannot be verified.

Do not alter the score, founder set, injury grid, sealed patterns, or primary
contrast in response to candidate results.

## 18. Final interpretation

Primary interval entirely above zero:

> The selected descendant heredity program improved over its initial program
> ancestor for the frozen sealed founder panel under the tested budgets.

Primary interval overlapping zero:

> This bounded ShinkaEvolve search did not reliably improve the heredity program
> over its program ancestor under the tested candidate budget.

Even a positive result does not show that ShinkaEvolve rewrote or improved
itself. The supported framing is bounded recursive program improvement of an
embodied heredity process.

## 19. Frozen execution and stopping thresholds

### Founder screen

- deterministic founder generator seed: `0`;
- candidate ranges assigned before screening:
  - training indices `[0, 16)`, select first 3 passers;
  - development indices `[16, 32)`, select first 2 passers;
  - sealed indices `[32, 48)`, select first 3 passers;
- every candidate is screened on all three uninjured seeds `131/257/389`;
- each seed must finish with at least 2 living organisms, at least 2 births,
  maximum generation at least 1, normalized productivity at least `0.01`, and
  complete graph/cache/numerical/infrastructure integrity;
- failure to fill any partition publishes no founder bank and stops the study.

### Calibration and development

- survival floor: `2/3` of injured founder×seed worlds;
- median post-event birth floor: `4`;
- median post-event generation gain floor: `2`;
- clone harm: upper endpoint of the founder-first 95% interval for
  `injured - sham` must be below `0`;
- mutation benefit: lower endpoint of the founder-first 95% interval for
  `mutation - clone` in injured worlds must be above `0`;
- calibration attempts every gain in `0.1/0.2/0.4` before selection;
- the only fallback is one repetition of the same gain grid with event step
  `4500` in every calibration world, and it is permitted only if every mutation
  policy at every primary-grid gain has median post-event births below `4`;
- development is a single no-retuning confirmation on the frozen middle-hinge
  panel; failure stops the outer search.

### Outer program search

- candidate output width: `4`; the optional six-value/two-pass interface is not
  enabled for this run;
- exactly 15 proposal generations per arm; invalid proposals consume their
  generation and receive no replacement;
- evaluator timeout: `00:10:00` per proposal;
- one evaluation job, one proposal job, and one database worker;
- proposal repair limits: 2 patch attempts, 2 patch resamples, and 1 novelty
  attempt;
- candidate limits: 12,000 source bytes, 1,024 AST nodes, and 60 nonblank lines;
- model/command: `headless/codex@gpt-5.5?effort=high` through
  `npx -y @roberttlange/headless@0.4.0`, temperature `0.0`, reasoning effort
  `high`, token cap `8192`;
- outer seed `17`, two islands, archive size `16`, top-K `3`;
- any candidate timeout, validation failure, policy violation, graph failure,
  or numerical/infrastructure integrity failure fails closed and remains part of
  the fixed proposal budget;
- full-manifest numerical-repeat range above `0.015` triggers a manual integrity
  review and does not permit favorable-repeat selection.

## 20. Frozen source and protocol identities

The following hashes were computed only after the final focused and complete CPU
regressions passed. Any change to one of these sources before its corresponding
production gate requires a new preregistration/run ID or an explicit prospective
amendment; it may not silently reuse this run ID.

| Bound artifact | SHA-256 |
|---|---|
| final implementation plan | `4578758492c6488d5a64b1ec0e630550014545c604360877f5c447a1f5a1e1fd` |
| simulator configuration | `cd3278775210c4859ab2e4f0a8b391d45ec1b6d05a506df17d2ea59df2b9a10b` |
| trusted simulator source aggregate | `ac964cb1c3c1f61bdc15df67227db2ecbdf027c3f63f709860fb1cc011a83059` |
| founder-bank builder | `464dda8d2d89382948cdb0d9d0dcdfc7ecb248b15bd0df44dd84677704468dc5` |
| founder artifact layer | `6dfefa893253f7bac003a56f19b6c1d79431a51408a29f1389683acb8e30a739` |
| actuator calibration | `8ae70b42d4dfef2fb7ae90a2ec09fce531a85df4d05fc4910192040e6db0d6ef` |
| development confirmation | `270d207eba04ca7ff0ea0ba57317979d7ff9c1d350e06e913f3066b469e94b76` |
| hierarchical analysis | `90c0377389cafca6dde51cfc19d77daf7f8b9f48ca994399b8383b30407a1181` |
| prospective manifest generator | `9a12a2f41b77f0e358aedbbeb6034f48f8181eb39605d89f6e8316e91b0ca241` |
| Shinka initial program | `e9831cc257f00c0735808e828b04b22f9260bef0094d3f3c88d7ea8b94de5c8a` |
| Shinka evaluator | `b7c4204fc9a94b5b080570a4e6fecbf6616949e6d86e190ba70822875e238c52` |
| Shinka launcher | `2cd208819b0dd13631faa12532cf8d81ca6cb1711684e10af0c95b678c44c758` |
| finalist selector | `3d5d26864006dcdc88f601a7975fda54ee11f852eb7aaad57730dded9f5db9e0` |
| program-lineage selector | `59556b4ca105d03f802ad5218470d39d0d95a3411bed8bed7a21a32044696751` |
| run-spec module | `d9e5e0a9d94e8cf53dec352f91dd5252afbf88db6b1d7e80d1e928f13dd55b57` |
| Microcosmos dependency lock | `d0500d8ed41fd4757bb14454ac03532f0f409217e68cbfed90c36cc14c8c6ace` |

The final SHA-256 of this preregistration is computed after this freeze and must
be recorded in the calibration and downstream result ledgers and in the
prospective schema-v2 Shinka run specification. The founder-screen ledger is
bound instead by its own source/protocol hash, whose exact builder source hash
is frozen above.
