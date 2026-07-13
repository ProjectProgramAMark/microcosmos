# Evo² GPU replay audit and final training protocol

Date: 2026-07-12  
Status: amended and frozen before the clean matched production searches

Ordinary GPU execution is not bitwise deterministic: repeated trajectories can
diverge because scatter and reduction order is not fixed. XLA's strict
deterministic mode (`--xla_gpu_exclude_nondeterministic_ops`) was tested, but an
eight-world evaluation exceeded ten minutes, a four-world evaluation exceeded
twelve minutes, and a two-world evaluation exceeded fifteen minutes.

The first proposed fallback—one execution over eight independent worlds—was
then audited before production. Three fresh executions of the exact same
fixed-mixed source, manifest, evaluator hash, and simulator-source hash scored
`0.2205205`, `0.2132338`, and `0.1929945` (range `0.0275260`, sample standard
deviation `0.0142619`), despite all three having 100% survival. That spread can
reorder plausible candidate improvements. Two partial outer-search trees were
therefore stopped and discarded without opening development or sealed data.

The final protocol keeps the original eight independent worlds—four
resource-relocation seeds and four random-bottleneck seeds—but measures each
complete manifest three times after one compilation. All three executions must
pass integrity. The evaluator retains the complete median-scoring execution,
with stable repeat-index tie breaking, and records the three aggregate scores.
It never combines worlds from different executions or treats executions as
ecological replicates. The ecological sample size remains eight worlds.

For paired development and sealed worlds, null and shock branches additionally
fork the exact same pre-event state. This removes avoidable prehistory divergence
and halves paired pre-event computation. GPU nondeterminism remains a reported
limitation rather than a hidden source of estimator variance.

The stable and punctuated manifests are restored exactly:

| Manifest | Final SHA-256 |
|---|---|
| Stable training | `716f306f684646cfa2ea62c36c3cef5735e1d682425de3eeee5ca1018ae1b021` |
| Punctuated training | `535b759c613bb91252934d3b91645b4017add1379b78b7164e984fc58d3e8fbf` |

Horizon, chunking, pair identities, scenario parameters, ecology, development,
and sealed data never changed. The discarded smoke/partial searches were used
only to validate infrastructure and exposed the ranking-noise problem; none of
their candidates can enter the clean run, development selection, or final
evaluation. Development and sealed manifests remained mode `000` throughout.

## Provider interruption during the matched run

The stable production arm completed its full 15-generation budget. The first
punctuated launch then completed only its frozen generation-zero evaluation
before the external headless Codex service reported an account usage limit on
every proposal attempt, with a stated reset time of 17:00 UTC. The runner was
stopped rather than allowing provider failures to consume generation IDs. Its
partial directory was moved out of the production run root, made mode `000`,
and is ineligible for selection. No proposed descendant was produced, and both
hidden manifests remained mode `000`.

After provider reset, the punctuated arm is relaunched from a fresh directory
under the unchanged canonical run-spec hash
`47ac4c3261a64ec37d04f1303b52846329eeffc6efd92f0bf52276adb4ed304f`.
The completed stable result tree remains mode `000` throughout, so the second
proposer cannot inspect its sibling treatment.

## Completion-marker checkpoint correction

Both matched searches completed before development was opened. During the
post-run authentication check, both `*.complete.json` files contained the same
hash even though their SQLite archives differed. File timestamps and the
absence of active writers showed why: the marker was written immediately before
SQLite completed its final WAL checkpoint, so it authenticated the shared
pre-checkpoint database image. The candidate sources, metrics, archive rows,
generation counts, and run specification were unaffected.

Before unlocking development, the settled databases were opened read-only,
confirmed to contain the expected complete archives, and hashed as:

| Arm | Archive rows | Correct rows | Settled database SHA-256 |
|---|---:|---:|---|
| Stable | 16 | 15 | `8effa13ecef569d87a8581dd52b33ffd79e5fe31d6cc827de9f5c9cbea37cae5` |
| Punctuated | 16 | 16 | `20bcb4105cab3f416e1002582ad0d6c409bd82ff32893945aa10f71b33867c0a` |

Only the derived `database_sha256` fields in the two completion records were
corrected. The launcher now performs and closes an explicit
`wal_checkpoint(TRUNCATE)` before publishing future completion hashes, with a
regression test covering the ordering. Neither hidden manifest had been opened,
and no search artifact or scientific result was changed.

The first derived finalist directories then exposed a second fail-closed
contract issue: their ranked-candidate entries contained the required repeat
scores and coherent-median index, but the sealed validator also required the
winner's same three fields at the top level. Sealed was still mode `000`. The
producer was aligned with the already-tested schema, its regression test was
strengthened, and the two invalid derived directories were retained under
`aborted_runs/evo2-production-20260712-r1-freeze-record-contract`. The
development evaluations were reused byte-for-byte to regenerate the freeze
records; no candidate was reevaluated, reordered, or changed. Both regenerated
finalists passed the sealed-readiness validator before the sealed manifest was
opened.

## Post-unlock sealed serializer correction

The first sealed worker completed the stable finalist's simulation but failed
before its atomic result write. The generic baseline serializer attempted
`inspect.getsource()` on the sandboxed candidate function, whose synthetic
compile filename intentionally has no source-loader entry. Consequently the
only sealed artifact was the precommitted suite record: no policy result,
episode metric, aggregate score, or summary was published or inspected.

The fix changes no simulator, policy, manifest, metric, repeat, or analysis
logic. Candidate records now give the generic serializer an inspectable trusted
placeholder and immediately replace its provisional hash with the finalist's
already-frozen source SHA-256, which was the existing intended behavior. A
regression test now uses a synthetic dynamically compiled candidate to exercise
this exact boundary.

Because the trusted sealed-workflow source hash changed after unlock, finalist
selection was not rerun—the selector correctly refuses to operate once sealed
is readable. Instead, the already-selected directories were restored unchanged
and their two provenance records were amended only with the old/new workflow
hashes and the fact that no sealed result had been published. The failed suite
record and pre-fix provenance snapshots remain under
`aborted_runs/evo2-production-20260712-r1-sealed-serializer`. Both amended
finalists passed the sealed validator before the unchanged six-policy suite was
restarted from an empty result directory.
