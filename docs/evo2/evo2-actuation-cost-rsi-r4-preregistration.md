# Evo² actuation-cost RSI r4 preregistration

Status: prospective and binding before r4 implementation experiments

Date: 2026-07-13

## Frozen ancestry

```text
Microcosmos implementation parent:
4396359db627ca8d22131d01446e48ee1f7cd327

Frozen r3 Microcosmos evidence parent:
0e32a0a3dc54c3a39119c2990879085eba3424c4

ShinkaEvolve implementation and r3 parent:
65fe36dd2ec21b379d7cf1a450b16679375804b4

Branch in both repositories:
agent/evo2-r4-credit-scheduler
```

The only Microcosmos change between the r3 evidence parent and the r4 branch
parent is the committed r4 plan and documentation-index update. No r1–r3 code,
founder, manifest, result, frozen finalist, database, hash, or report may be
modified.

## Hypothesis

A ShinkaEvolve descendant trained on matched sham and genuinely harmful
actuation-cost transitions will improve sealed shocked absolute productivity
relative to the exact fixed-standard program from which the Shinka lineage
began, without a material loss in sham productivity.

An adaptive-mechanism claim additionally requires context-dependent trusted
selection probabilities, lineage-matched descendant improvement, and loss of
advantage under mechanism ablation.

## Trusted r4 actions

```text
0 clone
1 parametric-conservative
2 parametric-standard
3 parametric-exploratory
4 structural
5 mixed
```

Parametric profiles:

| Parameter | Conservative | Standard | Exploratory |
|---|---:|---:|---:|
| value mutation rate | 0.10 | 0.20 | 0.40 |
| value mutation power | 0.075 | 0.15 | 0.30 |
| value replacement rate | 0.0075 | 0.015 | 0.030 |
| activation replacement rate | 0.05 | 0.10 | 0.20 |

Structural and mixed structure rates remain connection add/delete `0.20` and
node add/delete `0.10`. Mixed uses standard parametric values.

Candidate logits have shape `(6,)`, finite `float32`, and are clipped to
`[-8, 8]`. A unique saturated vector with one exact `8` and every other value
exactly `-8` selects that action deterministically. Other vectors use trusted
`jax.random.categorical` sampling. Fixed baselines bypass sampling.

## Candidate inputs

```text
parent_genome_summary (2):
  active_node_fraction, active_connection_fraction

parent_stats (3):
  energy_fraction, intake_ema, age_fraction

population_stats (6):
  alive_fraction, mean_energy_fraction, population_change_ema,
  birth_rate_ema, death_rate_ema, mean_intake_ema

operator_stats (3, 6):
  success_ema, usage_ema, evidence_ema
```

The candidate receives no event label, multiplier, time, scenario identity,
partition identity, seed, raw genome, simulator object, filesystem, network,
subprocess, or readable random key.

## Operator credit

```text
credit_time_scale = maturity_age + 2 * chunk_steps
outcome_ema_alpha = 0.10
usage_ema_alpha = 0.05
evidence_ema_alpha = 0.10
success_ema_initial = 0.5
usage_ema_initial = 0.0
evidence_ema_initial = 0.0
```

Founders have birth operator `-1` and never generate credit. A non-founder
resolves once: `exp(-time_to_first_reproduction / credit_time_scale)` at first
reproduction, or zero at death before reproduction. Same-step outcomes are
aggregated by operator and updated once with
`w = 1 - (1 - alpha) ** count`; lifecycle order is resolve outcomes, update
credit, create births, then register new offspring provenance.

## Primary disturbance

At the event boundary, multiply intended-action energy cost by one persistent
state scalar while leaving physical actuation unchanged.

Qualification order:

```text
1.25, 1.50, 2.00, 3.00
```

Select the first multiplier passing every qualification rule. If none passes,
stop r4 before Shinka.

Primary sealed evaluation reuses the qualified training multiplier with unseen
founders, worlds, mutation keys, and event timing. Optional magnitude transfer
is mapped before search:

```text
1.25 -> 1.50
1.50 -> 2.00
2.00 -> 3.00
3.00 -> omitted
```

The mapped value is included only if it independently passes harm and viability
on disjoint training-only qualification-control seeds before Shinka.

## Founder eligibility and partitions

Initial populations are clonal copies of one authenticated CPPN, but no
organism is protected. Founder candidate and seed ranges are prepartitioned
before screening. Within each range choose the first candidates passing the
healthy clone check:

```text
alive at event >= 2
births before event >= 2
maximum generation before event >= 1
mean pre-event productivity >= 0.01
all integrity invariants valid
```

Required partitions:

```text
training founders: 4
development founders: 4
sealed founders: 8
training pairs: 8 (2 contexts per founder)
development pairs: 8
sealed pairs: 16
```

Failure to fill a partition is a stop. Candidates or seeds may not be borrowed
from another partition.

## Disturbance qualification

All qualification uses training-only artifacts.

Harm:

```text
clone_effect = shock absolute post-event AUC - sham absolute post-event AUC
mean clone_effect <= -0.03
founder-first paired-bootstrap 95% upper bound < 0
```

Viability:

```text
clone or standard-parametric survival >= 0.75
median post-shock births >= 4
median post-shock generation gain >= 1
at least one genetically distinct non-clone child
at least one distinct child reaches maturity or first reproduction
all integrity checks valid
```

Resolved feedback:

```text
at least 24 resolved non-founder outcomes before shock
at least 24 after shock
at least two non-clone operators with evidence EMA >= 0.10 in each period
```

Observable stress:

```text
candidate-input standardized mean difference >= 0.5
or operator-credit distribution total variation >= 0.20
```

## Learnable operator-opportunity qualification

For fixed imminent-birth states before and after shock, evaluate every trusted
action with eight preregistered mutation keys per state/action. Average repeated
draws before action comparison.

Run leave-one-founder-out cross-fitting of a small scheduler that enumerates:

```text
one real r4 input feature
one threshold in 0.1, 0.2, ..., 0.9
one trusted action below the threshold
one trusted action above the threshold
```

Qualification requires:

```text
out-of-founder mean advantage over the fold-selected best fixed action >= 0.03
paired 95% lower bound > 0
cross-fitted rules use at least two distinct non-clone actions
each exercised action has >= 8 resolved outcomes before and after shock
```

The clairvoyant per-state oracle is descriptive only and cannot qualify the
benchmark.

## Productivity and score

The only scored productivity is the existing absolute measure:

```text
gross_productivity = cumulative_intake / (chunk_steps * dt)
resource_reference = sum(post_event_resource_regeneration_map)
absolute_productivity = clip(gross_productivity / resource_reference, 0, 1)
absolute_auc = mean(post_event absolute_productivity)
```

Candidate-relative pre/post recovery ratios are diagnostics only.

Punctuated search pair score is the harmonic mean of sham and shock absolute
AUC. Stable archive score uses sham AUC only, although the same number of shock
episodes is executed privately for equal compute. Candidate and exact initial
ancestor are evaluated with matched repeat/world keys and differenced within
repeat before aggregation.

Search aggregate:

```text
0.8 * IQM(pair deltas) + 0.2 * lower-quartile mean(pair deltas)
```

Survival and integrity are hard gates. Diversity is diagnostic, never reward.

## Numerical execution

Attempt OpenXLA deterministic exclusion once. Accept only if the complete
focused episode succeeds within `2x` ordinary runtime. Otherwise:

```text
search/development: 3 complete-manifest executions
sealed finalists: 5 complete-manifest executions
```

Choose the coherent execution whose complete-manifest paired
candidate-minus-ancestor delta is the median. Never choose independent candidate
and ancestor medians. Numerical executions are not ecological replicates.

## Search budget

Each stable and punctuated arm receives:

```text
1 exact initial program
50 LLM-generated descendants
2 islands
archive capacity 32
identical manifests, repeats, simulator calls, and timeouts
```

The initial source is the lifted r3 punctuated finalist: exact fixed standard
parametric action. The matched-budget structured-random control receives 50
candidate evaluations in its narrower frozen threshold/linear grammar.

Use uniform fidelity unless a preregistered 12-policy pilot finds cheap/full
Spearman rank correlation `>= 0.70`.

## Development selection

Always freeze the highest-scoring valid unrestricted champion. If an adaptive
eligible candidate exists, freeze the highest-scoring adaptive candidate as the
mechanism-primary finalist; otherwise the unrestricted champion is the
performance-only primary finalist.

Adaptive eligibility:

```text
at least two non-clone actions each have mean trusted probability >= 0.05
pre/post expected action distributions have total variation >= 0.20
founder/repeat bootstrap excludes the fixed-mixture null
```

Realized action counts alone cannot establish adaptiveness.

## Sealed primary and success rules

Primary effect:

```text
punctuated Shinka shocked absolute AUC
- exact r4 initial shocked absolute AUC
```

Full bounded-RSI success requires:

```text
sealed clone shock-minus-sham 95% upper bound < 0
primary mean >= +0.02
primary founder-first 95% lower bound > 0
sham absolute-AUC delta 95% lower bound > -0.02
adaptive eligibility
context-dependent trusted action probabilities
lineage-matched descendants beat ancestors with positive interval
no-credit or dominant-action ablation removes >= half the gain,
  or finalist-minus-ablation interval is positive
```

If performance passes but adaptiveness does not, report automated fixed-operator
or fixed-mixture discovery, not adaptive RSI. If performance is uncertain or
negative, preserve and report the negative result.

## Access, resumption, and artifact rules

- Development and sealed inputs stay inaccessible through search.
- Both archives close before development access.
- Sources, selection, analysis, manifests, and hashes freeze before sealed
  access.
- A completed run directory is immutable.
- An incomplete run may resume only after run-spec, source, manifest, and
  database hashes authenticate.
- Commit durable sources, logs, SQLite databases, completion records, selected
  candidates, archive lineages, manifests, hashes, raw results, and reports.
- Do not commit caches, WAL/SHM files, render scratch, or duplicate attempt
  workspaces.

Any change to the action set, disturbance family, primary metric, founder
partition, outer budget, or sealed suite requires a new prospective run ID.
