# Evo²-Ecosystem implementation plans

Revised 2026-07-16 after the R15–R20 replicated hard-founder campaign. The
active design uses CPPN controllers from the beginning; there is no temporary
wave-controller implementation stage.

## Reading order

1. [R15–R20 campaign report](./evo2-r15-r20-campaign-report.md) — latest
   replicated Shinka campaign: a real ecology-conditioned development effect,
   negative matched-rate and mechanism controls, and a negative independent
   sealed result that defines the current generalization boundary.
2. [R6 resource-relocation RSI plan](./evo2-r6-resource-relocation-rsi-plan.md) —
   authoritative prospective next experiment: one preregistered antipodal
   resource relocation paired with a same-location stock refresh, fresh worlds
   and founders, corrected fixed-standard viability, sufficient natural credit
   warmup, and matched bounded ShinkaEvolve searches.
3. [R5 world-feasibility stop report](./evo2-r5-world-feasibility-stop-report.md) —
   terminal negative result: every actuation-cost disturbance failed the harm
   gate and the preregistered pre-event feedback requirement was not met, so no
   R5 Shinka search was authorized.
4. [R5 world-feasibility plan](./evo2-r5-world-feasibility-plan.md) — exact
   prospective protocol whose stop rules produced the terminal R5 result.
5. [R4 credit-adaptive RSI plan](./evo2-r4-credit-adaptive-rsi-plan.md) —
   historical design of the six-action TensorNEAT heredity scheduler,
   recent-offspring operator credit, and direct descendant-versus-ancestor RSI
   test inherited by R5 and R6.
6. [R4 preregistration](./evo2-actuation-cost-rsi-r4-preregistration.md) —
   binding ancestry, constants, qualification gates, partitions, outer budget,
   primary metric, and stop rules fixed before r4 implementation experiments.
7. [R3 final report](./evo2-actuator-rsi-r3-final-report.md) — completed negative
   experiment establishing that the tail injury helped clone controls and that
   the punctuated Shinka finalist rediscovered fixed parametric mutation.
8. [R3 preregistration](./evo2-actuator-rsi-r3-preregistration.md) — exact frozen
   protocol used for the r3 search and sealed examination.
9. [Final RSI implementation plan](./evo2-rsi-final-implementation-plan.md) —
   historical detailed follow-up design: persistent actuator degradation,
   clonal founders, the mutation-sensitive benchmark gate, and direct Shinka
   program-lineage evaluation. It preserves the completed pilot as evidence.
10. [Frozen actuator preregistration](./evo2-actuator-rsi-preregistration.md) —
   exact prospective founders, seeds, injury grid, gates, outer budget,
   statistics, and source identities.
11. [Founder-screen stop report](./evo2-actuator-rsi-screening-stop-report.md) —
   complete guarded GPU screen, the preregistered no-bank stop, and the boundary
   on what the result does and does not show.
12. [Master PRD](./evo2-ecosystem-master-prd.md) — original implemented scope,
   architecture, scientific comparison, phase gates, and claim boundary.
13. [Microcosmos plan](./microcosmos-substrate-plan.md) — direct CPPN state,
   cached GPU inference, trusted TensorNEAT mutation, lifecycle integration,
   events, and tests.
14. [Heredity and Shinka plan](./heredity-shinka-plan.md) — the only evolvable
   function, operator profiles, validator, evaluator contract, and matched
   outer searches.
15. [Experiment plan](./experiment-analysis-plan.md) — pilot preregistered manifests,
   scoring, baselines, paired analysis, conditional common gardens, and final
   artifacts.
16. [Final report](./evo2-final-report.md) — completed pilot implementation, matched
   searches, sealed results, integrity amendments, and the supported claim.

The [lifecycle remediation PRD](./microcosmos-lifecycle-remediation-prd.md) is
a completed historical record. It truthfully documents the fixed 23-locus wave
controller that proved lifecycle and embodied-foraging feasibility. That
controller remains a regression oracle, not the active scientific controller
or an implementation stage.

## Current implementation state

The hard substrate and ecology work is complete:

- fixed-capacity organism slots with activity-aware physics;
- birth, death, same-step slot reuse, energy transfer, and monotonic IDs;
- localized renewable resources and mouth-only consumption;
- fluid-on locomotion and resource-acquisition positive controls;
- identity-keyed mutation and spawn randomness;
- compact JAX `lax.scan` rollouts and sparse finalist snapshots;
- a guarded 1,000-step turnover smoke and fused benchmark.

The direct CPPN migration is also complete:

- padded 15-node/30-connection TensorNEAT genomes live in `PopulationState`;
- controller transforms are cached at reset and birth;
- all organism/hinge inference is fixed-shape JAX and GPU-native;
- food-earned asynchronous births apply trusted clone, parametric, structural,
  or mixed TensorNEAT mutation directly to the inherited CPPN;
- founder lineages, ecological summaries, operator telemetry, and graph/cache
  integrity are tracked in the same lifecycle;
- canonical resource-relocation, bottleneck, and capped lineage-cull events,
  absolute-productivity scoring, and paired manifest validation are implemented.

The final complete CPU regressions are green: Microcosmos has 403 passing tests
and one intentional GPU-only skip; ShinkaEvolve has 538 passing tests.

TensorNEAT is already pinned by Microcosmos. A guarded feasibility probe using
32 padded 15-node/30-connection CPPNs found:

- the integrated cached CPPN ecosystem takes 1.312× the frozen wave-controller
  warm time in a 100-step, fluid-on 64 × 64 world (65.61 ms versus 50.00 ms);
- the final 32-controller structural-mutation kernel takes 1.14 ms warm after
  a 3.25 s compile; the complete sequential validated birth path takes 9.53 ms
  for the same worst-case 32 simultaneous births;
- recomputing graph transforms every physics step roughly doubles controller
  inference cost, so transforms must be cached at reset and birth.

The final fluid-on positive controls also pass: the canonical CPPN moves 75.3%
of a body length, while its two frozen sensor edges improve paired uptake by
21.9% mean and 20.6% median with 7/8 wins. These measurements support direct
CPPN integration without changing the memory model or adding a second simulator.

One-time turnover calibration selected `v2_c00_balanced` with ecology hash
`ceab22348b95db80ec8141628e9f7740a8753cf0448573d0a09bfe5eb072ee4d`.
Four of five calibration worlds reproduced, three reached generation three or
later, every world contained natural deaths, and every integrity gate passed.
The exact selected configuration is shared by demos, baselines, and Shinka.

Training, development, and physically separate sealed manifests are frozen and
seed-disjoint. Ordinary GPU execution is not bitwise stable because some XLA
GPU scatter/reduction operations are nondeterministic. Strict deterministic
XLA made even a two-world panel take more than fifteen minutes, so it was
rejected before any Shinka search. The final protocol retains the original
eight independent full-horizon worlds per training regime—four relocation and
four bottleneck worlds—and measures each complete manifest three times. The
trusted evaluator selects one coherent median-scoring full-manifest execution
with stable tie breaking and requires integrity in all three measurements.
These measurements quantify numerical uncertainty; ecological worlds or
shock/null pairs, not GPU repeats, are the scientific replicates and bootstrap
units.

Development and sealed shock/null members sharing `pair_id`, seed, and event
boundary reuse one exact pre-event trajectory inside each measurement and fork
only at the event. Training manifests contain one member per pair, so separate
stable and punctuated searches are seed-matched but are not exact trajectory
forks. Both hidden files remain mode `000` through both searches. Development
alone is unlocked for top-three selection after both archives close; sealed
remains locked until both champions, trusted sources, analysis, preregistration,
and freeze records are final.

Frozen experiment hashes:

- full simulator config: `cd3278775210c4859ab2e4f0a8b391d45ec1b6d05a506df17d2ea59df2b9a10b`;
- stable training: `716f306f684646cfa2ea62c36c3cef5735e1d682425de3eeee5ca1018ae1b021`;
- punctuated training: `535b759c613bb91252934d3b91645b4017add1379b78b7164e984fc58d3e8fbf`;
- development: `563c614250054089bbdbbdae06f78adae27da1d13b29be923c3d70fa99602ace`;
- sealed final: `ff361145bfa4a06572f15c52310d87c44cdd6178eb38e0b7d294e3ad41d00077`.

## Final architecture decision

Evo² will embed fixed-capacity CPPNs and NEAT-style mutation inside the
continuous ecosystem:

```text
CPPN genome -> body commands -> embodied resource competition
      ^                              |
      |                              v
trusted TensorNEAT mutation <- food-earned asynchronous birth
      ^
      |
Shinka policy schedules six actions:
clone / 3 parametric scales / structural / mixed mutation
```

The ecosystem supplies selection through energy, reproduction, and death.
Therefore full synchronous NEAT selection, generation replacement, speciation,
elitism, and crossover are intentionally absent. Adding them would create a
second selection system and break continuous ecological lineages.

The accurate description is:

> CPPN-controlled ecological evolution in which ShinkaEvolve searches for an
> ecology-conditioned scheduler over six trusted TensorNEAT heredity actions.

It is not “full generational CPPN-NEAT inside Microcosmos.”

## Minimal code boundary

Add only two reusable core modules:

```text
src/microcosmos/cppn.py
src/microcosmos/heredity.py
```

- `cppn.py` owns the frozen TensorNEAT genome definition, viable founder,
  cached transform, batched inference, and graph validation.
- `heredity.py` owns normalized statistics, the historical four-action registry,
  the active six-action registry, trusted TensorNEAT mutation primitives, and
  baseline policies.

Do not add a controller registry, runtime backend switch, copied NEAT
population manager, custom graph-mutation implementation, core scenario engine,
archive loader, or second rollout path.

## Completed experiment outcome

Both 15-generation outer searches, top-three development selections, champion
freezes, and the six-policy 72-world sealed evaluation are complete. The final
paired analysis found no reliable advantage for punctuated Shinka training:

- punctuated minus stable shocked-world AUC: `-0.00399`, 95% interval
  `[-0.01468, 0.00581]`;
- punctuated minus fixed mixed: `0.00265`, interval
  `[-0.02805, 0.03348]`;
- punctuated minus stress responsive: `-0.00019`, interval
  `[-0.01321, 0.01326]`.

The implementation goal succeeded; the research hypothesis did not. Conditional
common-garden and mechanism-ablation analyses were not run because the
preregistered positive-result condition was not met. See the
[final report](./evo2-final-report.md) and the canonical sealed summary at
`microcosmos/experiments/evo2_sealed/final_results/summary.json`.

## Prospective actuator follow-up outcome

The stronger causal follow-up was implemented and frozen under
`evo2-actuator-rsi-20260713-r1`. It adds persistent actuator efficacy, clonal
founder injection, deterministic founder artifacts, founder-aware schema-v2
manifests, hierarchical inference, fail-closed calibration/development gates,
schema-v2 Shinka profiles, program-parent-chain export, and hidden-founder
permission checks.

Its full guarded GPU founder screen then selected zero of 48 prepartitioned
candidates. Every candidate remained numerically and structurally valid, but
uninjured seed `131` caused universal extinction before reproduction. The
required `3/2/3` founder bank was not published, so calibration, outer search,
and sealed access were correctly not attempted. This outcome is documented in
the [stop report](./evo2-actuator-rsi-screening-stop-report.md); continuing with
different founder generation or seed-feasibility rules requires a new
prospective run ID rather than changing the completed screen.

## Accelerator and differentiation boundary

CPPN forward execution, physics, resources, ecological state updates, mutation
profiles, and bounded JAX-native births remain fixed-shape and GPU-native. Graph
topology is represented by NaN-padded arrays, so topology can change without
changing tensor shapes or recompiling for each child.

The system is not end-to-end differentiable. CPPN forward computation and the
Microcosmos mechanical kernel remain differentiable between discrete
transitions, but nearest-cell resource lookup, allocation/clipping, alive masks,
birth, death, catastrophes, and structural mutation make the ecological
objective non-differentiable even at fixed topology. Evo² deliberately uses
ecological and program evolution rather than backpropagation. “GPU-native” is
therefore a relevant implementation property; “fully differentiable ecology”
is not a project requirement or claim.

## Non-negotiable execution rules

- Use `conda run -n sakana` for every Python command.
- Run complete pytest suites on CPU.
- Run focused GPU work only in fresh 20 GiB/24 GiB systemd scopes after the
  required session-count and `free -h` preflight.
- Keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10`.
- Evaluate every policy on the same independent world seeds and report the
  bounded lack of bitwise GPU reproducibility as a limitation.
- Measure every full manifest exactly three times through the canonical
  evaluator; do not wrap that evaluator in a second repeat loop.
- Use one GPU evaluator worker for Shinka.
- Run the two arms sequentially; authenticate and make the completed sibling
  result tree mode `000` before the second read-only proposer starts, then
  restore it only after both archives close.
- Freeze mutation profiles, simulator, manifests, metrics, candidate budget,
  and hidden seeds before final evaluation.
- Shinka may alter only the marked `make_offspring` body. Physics, controller
  representation, mutation implementations, ecology, catastrophes, and scoring
  remain trusted.
- Interpret the two outer searches as a selected-policy case study. With one
  run per arm and one frozen eight-founder CPPN panel, the experiment does not
  establish general superiority of catastrophe training or transfer across
  founder distributions.
