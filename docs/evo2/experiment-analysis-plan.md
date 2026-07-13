# Experiment and analysis implementation plan

This plan freezes the evidence needed for the scientific claims while keeping
the evaluator and analysis small enough for an interview proof of concept.

## 1. Claims and evidence

| Claim | Required evidence |
|---|---|
| Ecosystem mechanics work | unit/invariant tests and long finite turnover rollouts |
| A scheduler improves recovery | direct paired shocked-world post-event productivity on unseen worlds |
| Catastrophe-trained selection helped this case | selected catastrophe-trained versus selected stable-trained scheduler with matched outer budgets |
| Improvement transfers | resource geometry/timing/abundance absent from training |
| Change is heritable | exact descendant versus shock-time ancestor in the same garden |
| Change is post-shock adaptive | reciprocal pre/post gardens and a positive interaction |
| Code mechanism matters | one-factor ablation reduces the paired advantage |

Random bottlenecks measure demographic robustness. Dominant-lineage culls
measure robustness to phylogenetic concentration. Neither alone demonstrates a
new ecological adaptation; resource changes are the adaptation tests.

## 2. Experimental units

- World seed is the unit for paired policy comparisons.
- Scenario family is a stratum, not an independent replicate.
- Outer Shinka run is the unit for claims about search regimes.
- Shock-time ancestor is the unit for common-garden comparisons; multiple
  descendants of one ancestor are nested replicates.
- Exactly three full-manifest GPU executions are nested numerical measurements,
  not additional ecological replicates. Select one coherent median-scoring
  execution with stable tie breaking and require integrity in all three.

Do not treat timesteps, organisms in one world, or many descendants from one
ancestor as independent samples.

## 3. Partitions and manifests

Canonical JSON manifests and SHA-256 hashes are frozen before search.

### Calibration

Used for ecology selection, positive controls, runtime measurement, and score
sanity checks. Mutation-profile constants are already frozen and are not tuned
to ecological outcomes. Calibration supports no final claim.

### Training

Visible to Shinka. Each matched regime contains eight independent full-horizon
worlds: four relocation worlds and four bottleneck worlds. No horizon, event,
or ecological replicate is shortened.

- catastrophe regime: resource relocation and random bottleneck;
- stable regime: null event at the identical boundary in paired worlds.

The score and post-event window are identical.

### Development

Fresh seeds used for Shinka top-three reevaluation, champion selection, and
preliminary mechanism/ablation design. Conventional profiles and baseline
parameters are already frozen and are not retuned here. Do not send development
results back into the running Shinka archive.

The frozen development panel uses shock/null pairs across the visible scenario
families. Within each numerical measurement, pair members with the same
`pair_id`, seed, and event boundary run one exact pre-event trajectory and fork
only at the event. Its file is mode `000` while either outer search is active,
because the read-only headless proposer otherwise has broader filesystem
visibility than the candidate runtime. Trusted host evaluation restores
development access only after both archives close.

### Sealed final

Generate and hash before search; keep outside the Shinka task/results tree.
Open once after candidate sources, baselines, metrics, plots, exclusions,
analysis code, preregistration, trusted-source hashes, and both champion freeze
records are frozen and verified.

Keep the sealed file mode `000` throughout proposal generation, development
selection, and freeze construction. Restore read access only for one trusted,
noninteractive six-policy sealed driver, then record the permission transition
in its immutable ledger. The driver launches each policy in a fresh guarded
process and may resume only missing policies under the identical ledger; it
does not permit an arbitrary second suite.

Core final families:

1. unseen resource relocation geometry and boundary time;
2. unseen resource abundance or regeneration;
3. capped dominant-founder-lineage cull;
4. a paired null-event episode for each shocked episode.

Use independent fresh ecosystems for disturbance families, not chained shocks.
The frozen sealed panel uses shock/null pairs across every core family, with
interval width reported honestly. Each pair shares one exact pre-event
trajectory inside each numerical measurement and forks only at its declared
event.

Viscosity, actuator injury, and different body length are exploratory stretch
families and must not delay the core resource-transfer evaluation.

### Minimal manifest schema

```json
{
  "schema_version": 1,
  "partition": "training",
  "controller_layout": "cppn-4x1-15n-30c-v1",
  "simulator_config_sha256": "...",
  "horizon": 8000,
  "chunk_steps": 500,
  "worlds": [
    {
      "world_seed": 0,
      "scenario_id": "relocate-train-00",
      "pair_id": "train-relocation-00",
      "scenario_family": "resource_relocation",
      "event_kind": "resource_relocation",
      "event_step": 4000,
      "event_parameters": {
        "center": [16.0, 32.0],
        "radius": 12.0,
        "peak_capacity": 1.0,
        "peak_regeneration": 0.03,
        "stock_fraction": 1.0
      }
    }
  ]
}
```

Validate one event, `event_step % chunk_steps == 0`, horizon bounds, scenario
parameter ranges, and unique IDs before hashing.

## 4. Randomness and pairing

For the same world seed, all policies begin with the same viable CPPN founder
panel, positions, energy, ages, and resource field. Keep named streams for
initialization, environment, catastrophe, spawn, and mutation.

The founder panel is one frozen set of eight CPPNs. All confirmatory conclusions
are conditional on that standing variation; this experiment does not establish
transfer across founder distributions.

Endogenous draws are keyed by immutable individual/child ID, step, world, and
stream tag rather than sequential consumption or slot rank. Environmental and
catastrophe randomness is shared across policies. Mutation is the treatment and
may yield different descendants, but it must depend only on its explicit key.

GPU scatter/reduction order is not bitwise stable on this backend. Treat that
as a documented simulator limitation, not as a second experimental axis. Run
each complete policy/manifest evaluation exactly three times, select the
coherent median-scoring full execution with stable tie breaking, and require all
three to pass integrity. Do not independently median episodes or wrap the
canonical evaluator in another repeat loop. World pairs—not backend reruns—are
the units of paired and bootstrap analysis.

Exact pre-event sharing applies only to multiple members of one `pair_id` inside
the same manifest measurement. Separate policies and the separately executed
stable/punctuated training arms are seed-matched but are not claimed to share an
exact floating-point trajectory. Each training manifest contains only one world
per `pair_id`; development and sealed manifests contain the forked shock/null
pairs.

Test that a shared individual receives the same bottleneck uniform after an
unrelated slot/population-path difference.

## 5. Productivity and search score

The environment reward is fulfilled gross mouth uptake for one step. For a
chunk:

```text
P_chunk = cumulative_reward / (chunk_steps * dt)
P_ref = sum(resource_regeneration_map_after_event)
q_chunk = clip(P_chunk / max(P_ref, epsilon), 0, 1)
episode_score = mean(q_chunk over fixed post-event chunks)
candidate_score = mean(episode_score over manifest worlds)
```

This score is absolute, externally normalized, bounded, and auditable. It is not
divided by the candidate's own pre-shock performance. Extinction produces zero
future productivity naturally.

Do not add population, survival, diversity, recovery-time, mutation-size, or
code-complexity bonuses. If a pathological exploit appears, diagnose the
physical/metric bug and restart from a documented pre-search amendment rather
than patching weights after observing candidates.

For a manifest measurement, compute `candidate_score` from one coherent set of
episode vectors. Across the three full-manifest measurements, select the median
score and retain that measurement's complete episode set for all downstream
analysis. Record all three scores and the selected index.

## 6. Final metrics

### Per-policy primary metric

- normalized post-event productivity AUC: mean `q_chunk` over the frozen window.

### Confirmatory primary contrasts

Use direct paired differences in shocked-world AUC:

1. selected punctuated-trained scheduler minus selected stable-trained scheduler;
2. selected punctuated-trained scheduler minus fixed mixed TensorNEAT mutation.

The hand-written stress scheduler is a supporting comparison. Do not substitute
shock-minus-null difference-in-differences for either primary contrast.

### Secondary

- shock-minus-null productivity deficit:

```text
mean(max(0, q_null - q_shock))
```

- immediate resistance: mean `q_shock - q_null` over the first frozen
  post-event chunks;
- final-window productivity;
- survival at horizon, extinction boundary, minimum population, and time of
  minimum;
- sustained recovery: first checkpoint where smoothed shocked productivity is
  at least 80% of paired null productivity for `K` consecutive checkpoints.

Freeze smoothing, reference floor, and `K` before final access. If the paired
null has negligible productivity, mark recovery time not estimable. Report the
fraction recovered, an empirical recovery-time curve with censor markers, and
individual censored values. Do not implement a custom Kaplan–Meier estimator or
average censored times as if observed.

### Diagnostics, never optimized

- action diversity during ecosystem rollouts;
- active CPPN node/connection counts and activation mix;
- parent–child TensorNEAT compatibility distance;
- clone/parametric/structural/mixed operator frequency;
- realized node/connection additions and deletions inferred from parent–child
  graphs;
- birth, natural death, and catastrophe death counts;
- population occupancy, energy, and intake;
- ancestry depth and lineage abundance;
- compile/runtime and policy violation count.

## 7. Baseline protocol

Final table rows:

1. clone/no mutation;
2. fixed parametric TensorNEAT mutation;
3. fixed mixed TensorNEAT mutation;
4. frozen hand-written stress-responsive operator scheduler;
5. selected stable-trained Shinka scheduler;
6. selected catastrophe-trained Shinka scheduler;

Fairness rules:

- identical CPPN standing variation, initial states, parent selection, graph
  bounds, trusted mutation profiles, and final worlds;
- mutation-profile constants frozen before either treatment;
- same valid-candidate/simulation budget for stable and catastrophe searches;
- no final-world tuning;
- report every tuned constant and evaluation count.

The sealed run contains exactly these six policies. The two confirmatory
comparisons are catastrophe-trained Shinka versus stable-trained Shinka and
versus fixed mixed TensorNEAT mutation. Other rows are supporting comparisons.

## 8. Stable versus catastrophe outer search

Match model, prompt, temperature, token cap, initial source, archive/islands,
valid-candidate budget, timeout, training seeds, horizon, score, and champion
selection. Only the training event manifests differ.

Freeze one immutable run specification for exactly one 15-generation outer
search per regime. Use the same run ID, outer seed, pinned proposal command,
archive settings, one evaluator worker, and top-three development rule. Select
by `development_score_desc_then_generation_asc` and record actual proposal,
validation, and evaluation counts in each freeze record.

Run the arms sequentially to avoid adding GPU contention to the numerical
comparison. After the first arm writes its authenticated completion marker,
make that arm's result directory mode `000` before starting the second; the
read-only proposer must not inspect its sibling treatment. Restore access only
after the second arm has also closed.

Report:

> The selected catastrophe-trained policy outperformed/did not outperform the
> selected stable-trained policy on the sealed panel.

Do not claim the training procedure is generally superior from this one pair of
stochastic outer searches. This is a selected-policy case study.

## 9. Statistical analysis

Keep analysis effect-focused and dependency-light.

For each confirmatory comparison and scenario family:

- compute direct paired shocked-world differences in primary AUC between the two
  policies, then compute deficit, difference-in-differences, final productivity,
  and survival as secondary effects;
- report mean, median, fraction favoring the candidate, and all paired points;
- compute a paired percentile bootstrap 95% interval by resampling world pairs
  within scenario using seed `20260712` and 10,000 deterministic bootstrap
  replicates;
- combine scenario families with predeclared equal weights using stratified
  resampling;
- never resample or otherwise count the three GPU measurements as ecological
  observations.

Do not add p-values or multiple-testing machinery unless an external review
explicitly requires them. The proof of concept is better served by transparent
paired effects and interval width.

Implement bootstrap code in NumPy and test it on tiny exact fixtures; do not add
a statistics dependency solely for this analysis.

## 10. Exact ancestor–descendant common garden (positive finalists only)

Population rebound can reflect sorting of pre-existing resistant lineages. The
assay must match descendants to exact ancestors alive at shock time. Implement
and run this finalist-only path only when sealed relocation results support a
positive heritable-adaptation claim; it does not block Shinka integration or a
valid null-result report.

### Capture

For selected sealed relocation worlds and the catastrophe champion:

- save every living shock-time individual's ID, founder lineage, raw CPPN node
  genes, and raw CPPN connection genes;
- retain subsequent birth edges `(step, child_id, parent_id, founder_id)` using
  the finalist event-trace mode;
- at one fixed post-shock generation/time, select descendants and traverse each
  parent chain to its exact shock-time ancestor.

### Assay

Run ancestor and descendant CPPNs separately in identical standardized 8-node
bodies with the same position, energy, age, resource field, and assay keys.
Recompute each saved graph's cache in trusted assay code, disable birth and
mutation, and keep all ecological traits fixed.

Test two gardens:

- pre-shock resource landscape;
- post-shock resource landscape.

Average multiple descendants within ancestor, then ancestors within world.
Report:

```text
post effect = descendant_post - ancestor_post
pre effect  = descendant_pre  - ancestor_pre
interaction = post effect - pre effect
```

A post effect shows heritable performance change in the new environment. A
positive interaction is stronger evidence that the change is specifically
adaptive to the post-shock environment.

## 11. Mechanism and ablation

Use Shinka's parent lineage and direct parent/child reevaluations to locate the
smallest code transition associated with the champion's gain. Write the
mechanism hypothesis before the final ablation.

Choose the smallest applicable one-factor ablation:

- replace one ecological statistic with its neutral reference;
- replace dynamic operator scores with their stable-world mean;
- prevent selection of structural mutation while retaining all other code;
- force the champion to fixed mixed mutation;
- remove one interaction/threshold term from the learned scheduler.

Verify from realized parent–child differences that the ablation changed only
the intended mechanism. If the ablation was designed after final results, run it
on a fresh post-final panel and label it exploratory.

## 12. Raw results and artifacts

Search episode JSON stores only the compact values needed by the fixed score and
integrity gate:

```text
post-event normalized productivity by chunk
final/minimum alive, birth count, natural-death count, catastrophe-death count
maximum_generation
operator_count_clone, operator_count_parametric
operator_count_structural, operator_count_mixed
policy_violation_count
validity flags
```

It also records candidate/policy identity, world/scenario IDs, event parameters,
all source/config/manifest hashes, device/software data, summary metrics,
censoring flags, the three full-manifest scores, and the selected coherent
measurement index. Episode vectors and diagnostics come from that one selected
measurement; the other two scores quantify numerical dispersion.

Only finalist runs store:

- fixed-shape birth/death ID event arrays;
- sparse body/resource snapshots;
- selected videos.

Never retain dense fluid history during candidate search.

Suggested layout:

```text
outputs/evo2_ecosystem/<freeze_id>/
  freeze_record.json
  candidates/
  raw/<policy>/<scenario>/<seed>.json
  finalist_events/<policy>/<scenario>/<seed>.npz
  aggregates/seed_metrics.csv
  aggregates/comparisons.json
  plots/
  videos/
  report/
```

Required figures/table:

- paired post-event productivity/recovery effects with uncertainty;
- baseline comparison table with intervals and budgets;
- operator-use/mechanism plot for the selected policies;
- champion code lineage and smallest applicable ablation;
- lineage/common-garden result only when the positive finalist supports the
  corresponding claim.

Select representative video seeds by a predeclared median/quantile rule, not by
visual spectacle.

## 13. Freeze procedure

Before opening sealed worlds, write one schema-v2 freeze record for each arm.
Each record must contain:

- the shared canonical run specification and SHA-256, with regime binding;
- Microcosmos/Shinka commits and a complete trusted-source hash map;
- the hash of this preregistration and `preregistration_complete=true`;
- initial/candidate source hashes and selected candidate lineage;
- archive database hash plus a compact parent/source/diff lineage export and
  its hash;
- declared and actual generation, proposal, validation, and evaluation budgets;
- training, development, and sealed manifest hashes and simulator config hash;
- all top-three development results, their three scores, selected measurement,
  and `development_integrity_valid=true` for the champion;
- selection rule `development_score_desc_then_generation_asc` and rationale;
- exact baseline sources/parameters and the six sealed policy identities;
- analysis, scoring, evaluator, sealed-driver, and finalist-freezer hashes;
- bootstrap seed `20260712`, 10,000 replicates, 95% interval, direct primary
  contrasts, secondary estimators, exclusions, and known failures;
- conditional common-garden sampling and predeclared ablation rules.

Finalize and hash both records while sealed remains mode `000`. The sealed
driver consumes the records rather than accepting arbitrary candidate paths,
verifies every trusted hash, and creates one immutable global ledger. It runs
the six policies noninteractively in fresh guarded processes and may resume only
missing entries under the identical ledger. Trusted analysis recomputes AUC
from recorded vectors after checking their length and `[0, 1]` bounds rather
than trusting stored scalar scores.

After sealed access, do not change metrics, exclusions, candidate selection, or
plot rules in response to outcomes. A genuine bug requires a documented
amendment and a fresh sealed partition for any new confirmatory run; do not
silently rerun the original panel. Once this plan's hash is recorded as the
preregistration, do not edit this file.

## 14. Analysis tests

- exact score for constant, step, and extinction traces;
- normalization against post-event regeneration map;
- exact shock-null deficit and resistance fixtures;
- recovery threshold, reference-floor, and censor cases;
- paired bootstrap resamples pairs and stratifies scenarios correctly;
- numerical-repeat selection keeps one coherent manifest execution, uses stable
  median tie breaking, and requires integrity across all three measurements;
- paired shock/null execution reuses exactly one pre-event state and forks only
  at the declared event;
- lineage traversal remains correct after slot reuse;
- descendants aggregate within ancestor before world;
- every plot regenerates from raw fixture files.

## 15. Reporting guardrails

Use “bounded recursive program improvement,” “capped dominant-lineage cull,”
and “demographic recovery” precisely. Claim heritable change only from matched
common gardens and post-shock adaptation only from the reciprocal interaction.
Call the champion ecology-conditioned only if realized operator use varies with
ecological statistics and the corresponding ablation supports that mechanism;
otherwise call it a selected scheduler over the four trusted mutation
operators.
The precise search claim is an ecology-conditioned scheduler over four trusted
TensorNEAT mutation operators, not arbitrary heredity or mutation-algorithm
synthesis and not full generational NEAT. With one outer run per arm, discuss
selected policies rather than general superiority of a training regime. With
one fixed eight-founder panel, do not claim founder-distribution transfer.
Report nulls, extinctions, invalid candidates, timeouts, and wide intervals.
