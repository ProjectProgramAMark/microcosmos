# Evo² r2 clonal-founder readiness preregistration

Status: **FROZEN — immutable founder-selection contract**  
Freeze date: 2026-07-13  
Run namespace: `evo2-actuator-rsi-20260713-r2`

## Purpose

The completed r1 founder screen incorrectly required random clone-only
populations to survive an entire 8,000-step experiment. Seed `131` left every
candidate alive at step 4,500 but without resource intake or births, and every
candidate later went extinct. That made the adaptation experiment impossible
before actuator injury was applied.

This replacement protocol preserves the r1 ledger and changes only founder and
world readiness. A founder is not required to survive indefinitely under clone
heredity. It must be alive, consuming resources, and reproducing by the latest
possible actuator-event boundary. Post-event survival is tested later by the
actual injury/sham calibration.

## Causal boundary

Every individual world remains genetically clonal at reset. Different worlds
may use different frozen founder genotypes. Seed and founder selection may use
only an uninjured world and clone heredity. No actuator vector, injury response,
mutation policy, Shinka program, development score, or sealed score is available
to this workflow.

## Stage 1: qualify ecological seeds

Evaluate the canonical human-written CPPN under clone heredity in the uninjured
full-fluid simulator through step `4500`.

Candidate seeds are the exact ordered range:

```text
5000, 5001, ..., 5063
```

A seed passes only if, at step 4500, it has:

- complete graph/cache/numerical/infrastructure integrity;
- at least 2 living organisms;
- at least 2 births;
- maximum generation at least 1;
- normalized cumulative productivity at least `0.01`.

Assign the first 13 passing seeds, in candidate order, as:

```text
first 3  -> training
next 4   -> development
next 6   -> sealed
```

An uninjured feasibility run performed before this freeze found 35 passing seeds
and deterministically predicted these assignments:

```text
training:    5002, 5005, 5006
development: 5008, 5011, 5012, 5013
sealed:      5014, 5018, 5019, 5020, 5021, 5023
```

The production builder must recompute the complete 64-seed qualification and
obtain the same first 13 passers. It does not trust this prose as an input.

## Stage 2: qualify clonal founders

Generate the unchanged deterministic 48-genome CPPN candidate panel with
founder generator seed `0`. Assign ranges before screening:

```text
training candidates:    [0, 16), select first 3 passers
development candidates: [16, 32), select first 2 passers
sealed candidates:      [32, 48), select first 3 passers
```

Each candidate is evaluated only on the readiness seeds assigned to its
partition. It must pass every seed in that partition using the same step-4500
criteria from Stage 1.

The different seed counts are intentional: they match the planned downstream
`3 training / 4 development / 6 sealed` ecological panels. Founder is the
transfer cluster; seeds within a founder are repeated ecological observations.

## Publication and stop rules

Publish nothing until all `3/2/3` founder quotas pass. On success, atomically
publish:

- eight deterministic, pickle-free founder NPZ artifacts;
- canonical `index.json` with artifact and genotype hashes;
- canonical `readiness_seed_panel.json` and SHA sidecar;
- the complete append-only screening ledger.

Stop without actuator calibration if:

- fewer than 13 candidate seeds pass;
- any founder partition cannot fill its fixed quota;
- any graph, identity, numerical, event, infrastructure, or operator-accounting
  check fails;
- production is not running on the actual JAX GPU backend with fluid enabled;
- an injected runner or backend override is attempted;
- the preregistration or output path is missing, mutable, or already exists.

Do not expand the seed range, founder pool, thresholds, or candidate ranges
under this run namespace after results are observed.

## Fixed simulator

- 32 organism slots;
- 8 nodes and 6 bending hinges per organism;
- 8 initially living clonal organisms;
- 64×64 fluid world;
- unchanged resource, metabolism, birth, death, CPPN, and TensorNEAT settings;
- readiness horizon `4500`;
- chunk size `500`;
- clone heredity only.

## Frozen source identities

| Artifact | SHA-256 |
|---|---|
| r2 readiness builder | `650c4c34929f09750cf9109870869a11943051823814a62696c47f24c3fc187a` |
| preserved v1 screening primitives | `464dda8d2d89382948cdb0d9d0dcdfc7ecb248b15bd0df44dd84677704468dc5` |
| founder artifact layer | `6dfefa893253f7bac003a56f19b6c1d79431a51408a29f1389683acb8e30a739` |
| focused r2 tests | `b487986c03ff2ec985718164023108e92570792f403aa8a8a5268978a6e88fbf` |
| simulator configuration | `cd3278775210c4859ab2e4f0a8b391d45ec1b6d05a506df17d2ea59df2b9a10b` |
| trusted simulator source aggregate | `0f9e059172e096f96f62033a3f55a7c30c1c6954f1eace9193aefe562b9fa4b2` |

The SHA-256 of this document must be recorded by the production screening
ledger. A successful founder bank does not by itself authorize Shinka or sealed
access; the downstream actuator experiment receives a separate frozen protocol
binding this bank and its readiness seed panel.
