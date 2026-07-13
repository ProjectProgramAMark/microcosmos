# Evo² actuator-adaptation founder-screen stop report

Date: 2026-07-13  
Prospective run ID: `evo2-actuator-rsi-20260713-r1`  
Outcome: **STOPPED AT THE FROZEN FOUNDER-BANK GATE**

## Decision

The prospective actuator-adaptation study did not proceed to injury calibration,
development confirmation, ShinkaEvolve search, or sealed evaluation. The frozen
founder screen evaluated its complete prepartitioned 48-candidate panel and
found no candidate that passed all three uninjured viability seeds. The required
`3 training / 2 development / 3 sealed` bank therefore could not be published.

This is the preregistered stopping outcome. The screen, seeds, viability gates,
candidate ranges, or founder selection rule were not weakened after results were
seen.

## Frozen identities

| Artifact | SHA-256 |
|---|---|
| implementation plan | `4578758492c6488d5a64b1ec0e630550014545c604360877f5c447a1f5a1e1fd` |
| frozen preregistration | `e58f8c4e9f879fe65372edf7f276dc5d79e16e9e14daa838af77238ec9c3c592` |
| founder-screen protocol | `a559ad6cfd6c6742529b1201f0b76352b3867260eac5486106ad6d49c032c624` |
| founder selection rule | `4ce2c73c5ac59e3c201ff07bbd2feb318ebae95c070a264d55f2f3c39b485177` |
| completed screening JSONL bytes | `ced085189b913281c7ffad5e21f8946ebc402644522e6069abcd8fd97cf1c6a0` |

Canonical screening record:

```text
microcosmos/experiments/evo2_ecosystem/founders/
  heredity_adaptation.screening.jsonl
```

The intended bank directory does not exist. No partial founder artifacts were
published.

## Frozen screen

- founder generator seed: `0`;
- uninjured ecological seeds: `131`, `257`, `389`;
- horizon: `8,000` steps;
- chunk size: `500` steps;
- full 32-slot, 8-node, 64×64, fluid-enabled simulator;
- clone heredity only;
- training candidates `[0, 16)` with three required;
- development candidates `[16, 32)` with two required;
- sealed candidates `[32, 48)` with three required;
- every seed required:
  - final living population at least 2;
  - births at least 2;
  - maximum generation at least 1;
  - normalized productivity at least `0.01`;
  - complete graph, cache, finite-state, event, identity, infrastructure, and
    operator-accounting integrity.

The guarded production command observed the real JAX GPU backend and used the
full fluid substrate. Production runner and backend injection were disabled.

## Results

All 48 candidates had valid graphs, valid controller caches, finite rollouts,
and complete infrastructure integrity. None passed the cross-seed viability
gate.

| Seed | Candidates | Passed all per-seed gates | Final-alive gate | Birth gate | Generation gate | Productivity gate | Maximum alive | Maximum births | Maximum generation | Maximum productivity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 131 | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0000 |
| 257 | 48 | 35 | 36 | 36 | 37 | 35 | 17 | 17 | 5 | 0.0656 |
| 389 | 48 | 47 | 47 | 48 | 48 | 48 | 32 | 53 | 7 | 0.3867 |

Selected founder indices were empty in every partition:

```json
{"training":[],"development":[],"sealed":[]}
```

Seed `131` was a universal viability failure: every controller went extinct
without births or resource productivity. This was not a numerical or graph
failure. The same fixed founder panel was usually viable under seeds `257` and
`389`, showing that the failure was a robustness failure across the frozen
ecological initialization panel rather than a broken simulator execution.

## Interpretation

The prospective study asked for founders that were already robust enough to
sustain a clonal population across all three uninjured ecological seeds before
injury adaptation was tested. The deterministic varied-founder generator did
not supply such a founder in the 48 preassigned candidates.

This means the planned causal actuator benchmark was not ready for outer program
search under this run specification. Running Shinka anyway would violate the
central prerequisite that the uninjured clonal substrate is viable and that a
later deficit is caused by the actuator treatment rather than baseline founder
extinction.

The result does **not** show that persistent actuator adaptation is impossible,
that mutation cannot help, or that ShinkaEvolve cannot improve heredity. Those
questions were not reached. It shows that random CPPNs drawn by the current
founder initializer are not an adequate prospective source of robust clonal
founders under the frozen screen.

## Work completed despite the scientific stop

The implementation needed to run a repaired experiment is complete and tested:

- persistent fixed-shape actuator efficacy that applies to existing organisms
  and newborns while retaining intended actuation cost;
- optional clonal CPPN founder injection while preserving pilot varied-founder
  behavior;
- deterministic, pickle-free, hash-verified founder artifacts;
- schema-v2 founder-bound injury manifests with exact sham/injury forks;
- founder-first hierarchical paired statistics;
- fail-closed GPU/full-fluid founder screening and actuator calibration;
- one-shot development confirmation with irrevocably non-scientific test seams;
- schema-v2 Shinka run profiles, authenticated program lineages, and sealed
  founder/manifest permission boundaries;
- full CPU regressions: Microcosmos `403 passed, 1 skipped`; ShinkaEvolve
  `538 passed`;
- focused guarded GPU actuator tests passed before the production screen.

The completed pilot and its negative sealed result remain unchanged.

## Permitted next work

Any continuation must use a new prospective run ID and a newly frozen founder
protocol. Reasonable changes to study before that new freeze include:

1. generate founder candidates with an explicit uninjured locomotion/chemotaxis
   search rather than sampling the varied initialization panel;
2. separate environment-initialization feasibility from founder robustness, so
   no frozen seed is universally lethal before heredity can act;
3. increase the founder candidate pool prospectively after a power/runtime
   estimate;
4. preserve the same no-injury founder-selection boundary and never select
   founders for favorable actuator response.

These are recommendations for a distinct follow-up, not amendments to
`evo2-actuator-rsi-20260713-r1`.
