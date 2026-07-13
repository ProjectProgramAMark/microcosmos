# Evo² actuator-adaptation experiment — r2 prospective contract

Status: **FROZEN — immutable prospective contract**  
Freeze date: 2026-07-13  
Run ID: `evo2-actuator-rsi-20260713-r2`

This is a new prospective run after the r1 founder-screen stop. It preserves the
scientific question and causal clonal design, but fixes the invalid prerequisite:
r1 demanded that a clone-only population remain productive through the entire
8,000-step experiment and used an ecological seed under which heredity never
became active. The r2 prerequisite instead verifies exactly what the downstream
injury experiment needs: an uninjured clonal population must be alive,
resource-productive, and reproducing by the latest possible injury boundary.

No r1 injury, development, Shinka, or sealed result exists. The completed pilot
remains a separate descriptive result and is not reused as r2 evidence.

## 1. Confirmatory question

> Does the selected final descendant of the punctuated-trained Shinka heredity
> program produce higher post-injury productivity than its initial program on
> completely fresh, unseen clonal CPPN ecosystems under a fixed simulator and
> mutation-action budget?

The primary sealed contrast is final punctuated descendant minus initial
program in injured worlds. Pairing is exact on founder, world seed, pre-event
state, injury, event boundary, simulator configuration, and compute budget.

## 2. Why worlds remain clonal

Each world starts with eight organisms carrying the same frozen CPPN genome.
This is required for causal attribution: any useful post-injury genetic novelty
must be produced by the tested heredity operator rather than selected from a
mixture already present at initialization. Different worlds use different
frozen founder genotypes, so transfer is still tested across genotypes.

Clone is a negative control, not the desired solution. Clone offspring must be
bitwise identical, and persistent injury must harm clone relative to its exact
sham fork. A mutation policy must then improve directly over clone.

## 3. Frozen readiness prerequisite

The readiness screen was frozen separately before execution in
`evo2-founder-readiness-r2-preregistration.md`. It used no actuator injury and
published one immutable founder bank plus one immutable world-seed panel.

Every selected founder passed, across every seed assigned to its partition, at
step 4,500 with:

- complete graph, cache, numerical, and infrastructure integrity;
- at least 2 living organisms;
- at least 2 births;
- maximum generation at least 1;
- normalized productivity at least 0.01.

The complete 64-seed and 48-founder screens were retained, including failures.
Selection was the first passing item in frozen order—not the best performer.
No injury result participated in founder or seed selection.

Frozen readiness-qualified seed partitions:

```text
training:    5002, 5005, 5006
development: 5008, 5011, 5012, 5013
sealed:      5014, 5018, 5019, 5020, 5021, 5023
```

Frozen founder candidates:

```text
training:    0, 1, 2
development: 16, 17
sealed:      32, 33, 34
```

All downstream components must load and authenticate
`readiness_seed_panel.json`; they may not reintroduce independent seed lists.

## 4. Immutable substrate

- 32 fixed organism slots, 8 initially alive;
- 8 nodes and 6 bending hinges per organism;
- 64×64 full fluid world;
- 8,000 physics steps in 500-step chunks;
- finite resource, metabolism, movement cost, birth, death, and slot reuse;
- fixed direct TensorNEAT CPPN ABI (`cppn-4x1-15n-30c-v1`);
- persistent world-level actuator efficacy inherited by all later offspring;
- fixed trusted clone, parametric, structural, and mixed mutation kernels.

Resource, lifecycle, physics, controller ABI, mutation kernels, score, capacity,
and runtime budgets may not change during the run.

## 5. Injury design

Test only gains `0.1`, `0.2`, and `0.4`; healthy hinges remain `1.0`.

- calibration/training injury: head hinges `[0, 1]`;
- development injury: middle hinges `[2, 3]`;
- sealed injury: tail hinges `[4, 5]`.

Event steps cycle prospectively through `3500, 4000, 4500`. Each injury world
has an exact sham twin forked from the identical pre-event state. Intended
actuation retains its original metabolic cost. Candidate code never sees event
kind, injury location, event time, scenario identity, world seed, or absolute
time.

## 6. Benchmark gate before Shinka

On all training founders and training seeds, evaluate clone, fixed parametric,
fixed structural, and fixed mixed heredity for every gain. Select the smallest
gain satisfying all of:

1. clone offspring remain bitwise identical;
2. every included execution is integrity-valid;
3. founder-first paired 95% interval for clone `injury - sham` is below zero;
4. one conventional mutation policy has founder-first paired 95% interval for
   injured `mutation - clone` above zero;
5. that policy survives in at least two-thirds of injured worlds;
6. median post-event births are at least 4;
7. median post-event generation gain is at least 2.

All gains are evaluated before selection. If no primary schedule passes and
every mutation policy at every gain has median post-event births below 4, the
only permitted fallback reruns the same gain grid with event step 4,500. If no
attempt passes, stop without Shinka.

After calibration, run exactly one no-retuning development confirmation using
the already selected gain and conventional policy. It must preserve clone harm,
mutation advantage, integrity, survival, birth, and generation gates. Failure
stops the experiment before outer search.

## 7. Schema-v2 manifests

After gain selection, publish exactly once:

- stable training: 3 founders × 3 seeds, sham event;
- punctuated training: the exact same 9 founder×seed states, head injury;
- development: 2 founders × 4 seeds × injury/sham = 16 worlds;
- sealed: 3 founders × 6 seeds × injury/sham = 36 worlds.

Manifest bytes, SHA-256 sidecars, founder hashes, seed panel, simulator hash,
event steps, gain, and injury vectors are immutable. Development and sealed
manifests and founder bodies remain unreadable during both outer searches.

## 8. Candidate heredity ABI and outer searches

Shinka edits only the bounded offspring scheduler. It returns four finite scores
for trusted actions:

```text
clone, parametric, structural, mixed
```

Trusted code clips scores, applies deterministic argmax, executes exactly one
trusted mutation action, validates the resulting TensorNEAT graph, and records
operator use. Candidate code cannot edit physics, resources, birth/death,
mutation implementations, score, manifests, seeds, or tests.

Run matched stable and punctuated searches sequentially with:

- identical initial source;
- identical proposal model and settings;
- outer seed 17;
- 15 proposal generations per arm, with invalid proposals consuming budget;
- two islands, archive size 16, top-K 3;
- 10-minute candidate timeout;
- 12,000 source bytes, 1,024 AST nodes, 60 nonblank lines;
- exactly three coherent full-manifest numerical measurements per candidate;
- identical simulator calls and one trusted mutation action per birth;
- sibling archive hidden from the other proposer.

Model command remains
`headless/codex@gpt-5.5?effort=high` via
`npx -y @roberttlange/headless@0.4.0`, temperature 0, token cap 8192.

## 9. Score and analysis

Primary episode score is mean normalized resource productivity over frozen
post-event chunks:

```text
P_chunk = fulfilled mouth uptake / (chunk_steps * dt)
P_ref   = sum(post-event resource regeneration map)
q_chunk = clip(P_chunk / max(P_ref, epsilon), 0, 1)
score   = mean(q_chunk)
```

Candidate score gives founders equal weight. Diversity is diagnostic and is not
rewarded. Report survival, births, generation depth, operator use, genome
distance, structural changes, fallback counts, and injury-minus-sham effects.

For paired inference, resample founders first, then paired seeds within founder;
use 10,000 bootstrap replicates, seed 20260712, and a percentile 95% interval.
The three numerical measurements are integrity checks, not ecological samples.

Before sealed access, freeze the initial program, selected stable descendant,
selected punctuated descendant, conventional baselines, stress-responsive
baseline, and authenticated program ancestry. Every sealed evaluation begins
from fresh clonal ecosystems. The confirmatory claim is bounded recursive
program improvement of heredity—not unrestricted self-improvement.

## 10. Stop rules

- Stop if calibration or one-shot development confirmation fails.
- Stop if hashes, holdout permissions, clone identity, graph validity, operator
  accounting, or numerical integrity fail.
- Do not tune founders, seeds, gain, event timing, score, injury locations,
  policy ABI, or primary contrast after observing downstream results.
- Do not replace a failed proposal or failed scientific gate.
- Do not open the sealed partition before finalists are frozen.

## 11. Frozen identities

| Artifact | SHA-256 |
|---|---|
| readiness preregistration | `686dd2a044924798d11ca5bc608a842c02172740875b5ac03fd68ee382b9f4f1` |
| readiness builder | `650c4c34929f09750cf9109870869a11943051823814a62696c47f24c3fc187a` |
| readiness protocol | `a6d77754ce1eea6feddcee981de40fb23692e65015feeeab6efc712ac77992ef` |
| complete readiness ledger | `4ddaca9da59edd45e76f8948604ee5a83a10e50b9319bbdba0745c3c58b0c3cf` |
| founder index canonical identity | `2fcec499fb1d5ecbddafdbbe446c6dc3e52cd644a64e9452ef9de0c56111593f` |
| founder index file | `5479e9303322d773789207d825cd0311816303951fd2c8f5e65377b6a34397a8` |
| readiness seed-panel file | `9456902030d459a05dc2a784050815a76297ac040d46730e01c2ff5e3626e727` |
| simulator configuration | `cd3278775210c4859ab2e4f0a8b391d45ec1b6d05a506df17d2ea59df2b9a10b` |
| trusted simulator source aggregate | `ebc69f2a24f6eaafa457982aa4c84e31f58ad7c95a4c6ee252438b3a8d11ea34` |
| readiness seed loader | `5007f0645d2115ea343079c248b387fb3819e4d18d728463859c57a3b134f755` |
| actuator calibration | `42769de6c8a5f7f9767853c4a5df4ecf07f398d07ab5c6e869c5a8f0e1e75c6a` |
| development confirmation | `0640bc9925ad38d6b4791dad33d8c07cf6f4f0801c4bcede8c933cbdba0df0ba` |
| schema-v2 manifest generator | `2ba4f0e16f5eb4379f2ae8dbbe40a6a5899649e7c800078b9a1edc9e2b053b8a` |
| hierarchical analysis | `90c0377389cafca6dde51cfc19d77daf7f8b9f48ca994399b8383b30407a1181` |
| Shinka initial program | `e9831cc257f00c0735808e828b04b22f9260bef0094d3f3c88d7ea8b94de5c8a` |
| Shinka evaluator | `b7c4204fc9a94b5b080570a4e6fecbf6616949e6d86e190ba70822875e238c52` |
| Shinka launcher | `2cd208819b0dd13631faa12532cf8d81ca6cb1711684e10af0c95b678c44c758` |
| finalist freezer | `3d5d26864006dcdc88f601a7975fda54ee11f852eb7aaad57730dded9f5db9e0` |
| program-lineage selector | `59556b4ca105d03f802ad5218470d39d0d95a3411bed8bed7a21a32044696751` |
| run-spec module | `d9e5e0a9d94e8cf53dec352f91dd5252afbf88db6b1d7e80d1e928f13dd55b57` |

The SHA-256 of this file is recorded in the calibration ledger and every later
run binding. Any change to a bound artifact requires a new run ID; it may not
silently reuse r2.
