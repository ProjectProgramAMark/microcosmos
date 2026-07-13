# Evo² RSI prospective implementation audit

Date: 2026-07-13  
Purpose: Phase-0 preservation record for the prospective actuator-adaptation
study. This file does not amend the completed pilot or its interpretation.

## Repository starting state

### Microcosmos

- Starting commit: `fe904c6c5040a3adee8953449c797f2e5fdfceb5`
- `git status --short` was clean at the start of this implementation turn.
- The prospective implementation is being applied as uncommitted changes on top
  of this commit. The starting commit is the rollback boundary.

### ShinkaEvolve

- Starting commit: `7939f6b44046a2b92e4baa6687b52b23e6236898`
- Pre-existing user work was present and must be preserved:
  - modified `shinka/launch/local.py`;
  - untracked `examples/evo2_ecosystem/` task and frozen pilot artifacts;
  - untracked Evo² and launcher tests.
- No cleanup, reset, checkout, or deletion of these paths is authorized.

## Preserved pilot evidence

| Artifact | SHA-256 at audit |
|---|---|
| `plans/evo2-final-report.md` | `bca5ccada51ee5519a038269c9ee52734d8092a6274ab6732b7d191482e3c7f3` |
| `plans/experiment-analysis-plan.md` | `09cdea4de1575317e9430c1c226e7cef6a462088c1e40b2f8e3adf4b1744700a` |
| `microcosmos/experiments/evo2_sealed/final_results/summary.json` | `4b5d4bbc16645968b0bf97986626d10ea4e337fd2cf97f5adc2bd57d0253180c` |
| `ShinkaEvolve/examples/evo2_ecosystem/run_spec.json` | `47ac4c3261a64ec37d04f1303b52846329eeffc6efd92f0bf52276adb4ed304f` |

The existing sidecar file itself hashes to
`15c721f8985ea6e0abc18c60a17a76922611fa72b4fb471475a144cc73183c7f`.
Its contents authenticate the canonical run specification.

The final RSI implementation plan was corrected prospectively before the new
study froze. Its final bound SHA-256 is
`4578758492c6488d5a64b1ec0e630550014545c604360877f5c447a1f5a1e1fd`.
The frozen actuator-study preregistration SHA-256 is
`e58f8c4e9f879fe65372edf7f276dc5d79e16e9e14daa838af77238ec9c3c592`.
Every later prospective manifest, calibration record, and outer run must retain
these identities.

## Baseline verification

- Conda environment: `sakana` is mandatory for every Python command.
- Microcosmos CPU suite: `334 passed, 1 skipped`.
- ShinkaEvolve CPU suite: `527 passed`.
- Both full suites were executed with `JAX_PLATFORMS=cpu`.

Repository-wide Ruff is not a valid starting green gate: Microcosmos has 82 and
ShinkaEvolve has 15 pre-existing violations in unrelated upstream examples,
notebooks, and tests. This prospective work will:

1. run Ruff on every Python file it changes;
2. run the complete CPU pytest suites at phase checkpoints;
3. avoid modifying unrelated files solely to make repository-wide Ruff green.

## Preservation rules

- Do not overwrite pilot manifests, frozen finalists, archive lineage exports,
  sealed results, or final report.
- Use a new run ID, result root, manifest hashes, founder hashes, injury hashes,
  source hashes, and preregistration for the prospective study.
- Maintain backward parsing and canonical-byte behavior for pilot schema-v1
  manifests.
- Any verifier defect that can affect candidate ranking requires a documented
  amendment and restart from the corresponding prospective checkpoint.
- Existing ShinkaEvolve dirty files are user-owned and must be edited only when
  required for this objective.

## Prospective gate outcome

The complete guarded GPU founder screen ran after the prospective protocol was
frozen. It evaluated all 48 prepartitioned candidates with valid numerical and
infrastructure integrity but selected no founder because seed `131` was
universally lethal under the uninjured clonal screen. The bank was not
published, and calibration, development, Shinka search, and sealed evaluation
were not opened.

- screening record SHA-256:
  `ced085189b913281c7ffad5e21f8946ebc402644522e6069abcd8fd97cf1c6a0`;
- stop report SHA-256:
  `6020711a112f6d9aeab51ea63a9fabf529409fa407c4675ee47ace2289659ceb`;
- outcome: `insufficient_passing_candidates`.

This is the frozen stop rule working as intended, not authorization to retune
the existing run.
