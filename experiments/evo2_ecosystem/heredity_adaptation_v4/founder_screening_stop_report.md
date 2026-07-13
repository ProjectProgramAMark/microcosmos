# Evo² r4 prospective stop report

## Terminal outcome

The preregistered r4 founder gate stopped the experiment before disturbance
qualification or ShinkaEvolve search.

```text
status: insufficient_passing_candidates
bank_published: false
training founders selected: 0 / 4
development founders selected: 0 / 4
sealed founders selected: 0 / 8
```

This is the required terminal outcome under the frozen prospective protocol.
No founder threshold, seed, candidate range, ecology constant, or event boundary
was changed after inspecting the result.

## Frozen provenance

```text
Microcosmos source commit:
96a15b26634c0909313d6cfeadd5ee289a1b766c

ShinkaEvolve source commit:
1740a7ef7969132dbcb4133ff2e8c0d6bdef3d03

founder screening protocol SHA-256:
6e9be44cbf129d5395fc69952e87879c8f44868e9d74a7d92534f2cd51cf5077

founder screening JSONL SHA-256:
f0e13a6d89df473334169340c512b7b6e13f4c24ee46b0030cd5788851a77b58

r4 preregistration SHA-256:
e726865acd7cebc00ac2f302e756ea751844c46b23fea58e0f28ec9a6f1b98fb

r4 implementation plan SHA-256:
7037a9566102e41b316a1ce01b9680602993316d0b90cf6edb7ad3dfb272ae99
```

The append-only source evidence is
`founder_screening.jsonl` in this directory.

## Observed gate results

The fixed prepartitioned pool contained 80 candidates:

```text
training range evaluated: 24
development range evaluated: 24
sealed range evaluated: 32
candidate rollouts with valid graph/cache/physics/integrity: 80 / 80
candidates passing every founder gate: 0 / 80
```

Seed 4101 was biologically active across the complete pool:

```text
births before its 3500-step event boundary:
minimum 3, median 9, maximum 22
```

Seed 4102 was the universal limiting context:

```text
births before its 4000-step event boundary: 0 for all 80 candidates
median normalized pre-event productivity: 0.0016386891
productivity range: [0.0007345937, 0.0027367387]
```

Every candidate therefore failed the preregistered minimum of two births,
generation one, and productivity 0.01 in every screening context. This was not
a numerical or infrastructure failure: all 80 candidate evaluations retained
finite state and passed every integrity invariant.

## Consequences

The following r4 stages were intentionally not executed:

- disturbance multiplier qualification;
- causal operator-opportunity qualification;
- stable-world ShinkaEvolve search;
- punctuated-world ShinkaEvolve search;
- development finalist selection;
- sealed evaluation;
- finalist common-garden and ablation analysis.

Running any of those stages would violate the prospective plan because there is
no authenticated 4/4/8 founder bank.

No claim is made that ShinkaEvolve discovered a useful r4 heredity scheduler.
It was not launched in r4.

## Interpretation and next experiment

The r4 software implementation is operational: the source passed 428 CPU
regression tests (one skipped), focused GPU tests, strict lint checks, and this
full GPU/fluid founder screen. The scientific substrate is not viable under one
of the two frozen healthy-founder contexts.

A follow-up must be a separately preregistered r5 experiment. It may diagnose
why seed 4102 prevents food acquisition or define a seed panel whose healthy
clone worlds are calibrated before founder candidates are selected. It must not
retroactively reinterpret this r4 run or overwrite its evidence.
