# Evo² R15–R20 campaign report

## Ecology-conditioned heredity was discovered, but did not generalize to the sealed founders

**Status:** exploratory mechanism result; negative independent sealed result  
**Campaign date:** July 15–16, 2026  
**Outer system:** ShinkaEvolve with `headless/codex@gpt-5.5?effort=high`  
**Physical substrate:** GPU-native JAX Microcosmos ecosystem with inherited CPPNs

## Executive summary

This campaign tested whether ShinkaEvolve could discover a heredity scheduler
that uses observable ecological state to decide when a reproducing Microcosmos
organism should be cloned or mutated. The simulator, CPPN representation,
TensorNEAT mutation operators, founders, shock timing, score, and exact-clone
comparator were frozen outside the editable program.

Shinka produced a substantive candidate in R15 generation 18. On four untouched
R18 development founders, it beat exact clone in all three numerical repeats:
`+0.001106`, `+0.020162`, and `+0.009246`. Continuous standard and conservative
mutation were strongly negative. A constant sparse mixture matched to the
candidate's mutation frequency was also negative in every repeat, and removing
the candidate's explicit pressure/rescue terms made every repeat negative.
Thus, on the development panel, the gain required the ecology-conditioned code
mechanism rather than sparse mutation alone.

The one permitted evaluation on eight sealed R18 founders was negative in all
three repeats: `-0.000933`, `-0.001146`, and `-0.000301`. The policy improved
sham worlds but harmed injured worlds. Therefore this campaign does **not**
establish a Shinka-generated heredity policy that generalizes better than clone.
The R18 sealed panel is retired and will never be used for further tuning or
candidate evaluation.

The accurate conclusion is:

> ShinkaEvolve discovered a real ecology-conditioned scheduling mechanism that
> outperformed clone and matched-rate controls on fresh development founders,
> but the effect did not transfer to the independent sealed founder panel.

This is stronger than “Shinka generated code,” but weaker than the intended
submission-quality positive RSI result. It is evidence of autonomous mechanism
discovery and a clear measurement of its generalization boundary.

## 1. What Shinka was allowed to improve

Each organism inherits a padded CPPN controller. At every food-earned birth,
the editable function receives bounded summaries of:

- the parent CPPN's size and the parent's energy and intake;
- population size, energy, intake, births, deaths, and recent growth or decline;
- recent success, usage, and evidence for each trusted heredity operator.

It returns logits over six immutable operators:

1. clone;
2. conservative parametric mutation;
3. standard parametric mutation;
4. exploratory parametric mutation;
5. structural mutation;
6. mixed mutation.

Shinka could not edit physics, resources, injury, founders, mutation kernels,
the scoring function, seeds, runtime, or the exact-clone comparator. This is
bounded recursive program improvement: the evolved program controls the
mechanism that generates future evolutionary descendants.

## 2. The discovered R15 mechanism

R15 generation 18 is a sparse, state-dependent scheduler. It computes:

- `calm` from population occupancy, deaths, decline, and mean energy;
- `pressure` from decline, deaths, low energy, low intake, low births, and low
  population;
- `rescue` from the interaction of low intake with decline, deaths, and low
  energy;
- operator value from observed success, evidence, scarcity, and usage.

These signals modify a clone-dominant prior. On the R18 development panel, the
policy realized 98.157% clone, 1.164% conservative mutation, 0.485% standard
mutation, and 0.194% exploratory mutation. It did not simply turn mutation on
continuously.

## 3. Results

| Candidate | Partition | Robust score | Three repeat scores | Sham | Injury |
|---|---|---:|---|---:|---:|
| R20 generation 16 | hard training | +0.000095 | +0.000095, −0.000227, +0.000118 | +0.007004 | +0.000110 |
| R20 generation 16 | R18 development | −0.004069 | −0.004069, −0.014493, −0.003984 | +0.000646 | −0.006874 |
| fixed standard | R18 development | −0.057539 | −0.057539, −0.051515, −0.078614 | −0.020112 | −0.019942 |
| fixed conservative | R18 development | −0.087536 | −0.087536, −0.093678, −0.075106 | −0.044942 | −0.086182 |
| **R15 generation 18** | **R18 development** | **+0.009246** | **+0.001106, +0.020162, +0.009246** | **+0.015556** | **+0.041902** |
| matched sparse | R18 development | −0.024064 | −0.024064, −0.026399, −0.007038 | −0.025339 | −0.004810 |
| no-crisis ablation | R18 development | −0.019043 | −0.019043, −0.028174, −0.008733 | −0.023406 | −0.005925 |
| **R15 generation 18** | **R18 sealed** | **−0.000933** | **−0.000933, −0.001146, −0.000301** | **+0.008478** | **−0.006053** |

The robust score is the coherent median of three full-manifest GPU executions.
The colored points in the figure below are individual numerical repeats; black
diamonds are coherent medians.

![Replicated candidate effects](../../../ShinkaEvolve/examples/evo2_ecosystem/results/evo2-r15-r20-final-artifacts-20260716/repeat_scores.png)

### 3.1 Development mechanism evidence

Two controls isolate the mechanism:

- **Matched sparse:** a constant distribution matched to R15 generation 18's
  development operator fractions. Its three negative repeats show that mutation
  frequency alone does not explain the development result.
- **No-crisis ablation:** the exact R15 program with only the explicit
  `pressure` and `rescue` logit terms removed. Its three negative repeats show
  that those crisis signals were necessary on the development panel.

This supports a narrow exploratory claim: the code's ecological conditioning
caused its development-panel advantage. It does not rescue the failed sealed
generalization claim.

![Sham and injury effects](../../../ShinkaEvolve/examples/evo2_ecosystem/results/evo2-r15-r20-final-artifacts-20260716/treatment_effects.png)

![Realized operator allocation](../../../ShinkaEvolve/examples/evo2_ecosystem/results/evo2-r15-r20-final-artifacts-20260716/operator_fractions.png)

### 3.2 Independent sealed result

The finalist was frozen before sealed access. Its source SHA-256 is
`fc69ac96baa846b7650e474995566228582545b915ddf8dbd4754d161d573526`.
It was evaluated once on eight sealed founders with the exact frozen manifest
and three numerical repeats. All repeats were negative. The selected repeat
improved sham productivity by `+0.008478` but reduced injury productivity by
`−0.006053`.

The sealed result is not “almost positive.” Its magnitude is small, but its sign
was consistently negative. It falsifies the intended cross-founder improvement
claim for this candidate and protocol.

## 4. What the R20 search established

R20 used a harder exposed founder panel and three repeats per proposal. Its
compact parent directly represented a clone-to-standard rescue pulse. Generation
16 crossed zero on the coherent training median, but one training repeat was
negative and all three development repeats were negative. Later descendants
either collapsed toward clone or paid excessive sham costs for more mutation.

The program lineages show that Shinka explored executable descendants and found
small training improvements, but the available signal was near the GPU numerical
variation scale and did not yield robust founder transfer.

![R15 and R20 program lineages](../../../ShinkaEvolve/examples/evo2_ecosystem/results/evo2-r15-r20-final-artifacts-20260716/program_lineages.png)

## 5. Did ShinkaEvolve create anything useful?

Yes, with an important qualification.

Shinka created a nontrivial scheduler that:

- used ecological decline, deaths, energy, intake, and birth pressure;
- incorporated empirical success and evidence for trusted mutation operators;
- preserved cloning as the dominant stable-world action;
- beat clone on every untouched development repeat;
- beat continuous standard and conservative mutation;
- lost its advantage when its crisis terms were removed;
- beat a constant scheduler with matched average operator use.

That is useful algorithmic discovery rather than a hard-coded swimmer or a
cosmetic code edit. However, usefulness on a development panel is not enough.
Because the policy failed every sealed repeat, the project cannot claim that
Shinka produced a generally better heredity algorithm.

## 6. Why the effect probably failed to generalize

The evidence supports three likely causes, stated as hypotheses rather than
post-hoc facts:

1. **Founder-specific ecological calibration.** The thresholds and gains may
   map useful stress states for the development founders but trigger too much
   conservative/standard mutation for the sealed founders.
2. **Sparse, high-variance causal events.** Only a small fraction of births are
   mutated. A few descendants can materially change an ecosystem, so effect
   variance remains high even with full-manifest numerical replication.
3. **Restricted operator portfolio.** The scheduler can choose when to mutate,
   but cannot improve the trusted mutation operators themselves. If available
   mutations are usually destructive for a founder, scheduling has limited
   leverage.

The sealed policy used substantially more mutation than it did in development:
95.335% clone versus 98.157% clone. That behavioral shift is consistent with
the first hypothesis, but it does not by itself establish the cause.

## 7. Claim boundary

Supported:

- a GPU-native embodied ecosystem with birth, death, resources, CPPN heredity,
  injury, lineages, and bounded program evolution runs end to end;
- Shinka autonomously produced structurally distinct executable heredity code;
- one policy had a replicated positive result on untouched development founders;
- matched-rate and crisis-term controls indicate a real state-conditioned
  mechanism on that development panel;
- independent sealed testing found that the mechanism did not generalize.

Not supported:

- a submission-quality positive independent result;
- general improvement over clone across unseen founders;
- open-ended evolution, self-assembly, abiogenesis, or unrestricted RSI;
- recovery curves for the sealed run. The one-shot evaluator retained aggregate
  paired AUC metrics but not full trajectories, and sealed data cannot be rerun
  merely to improve visualization.

## 8. Reproducibility and artifacts

Frozen machine-readable artifacts are in:

```text
ShinkaEvolve/examples/evo2_ecosystem/results/
  evo2-r15-r20-final-artifacts-20260716/
  evo2-exploratory-r15-gen18-r18-analysis-20260716/
  evo2-exploratory-r15-gen18-r18-sealed-20260716/
  evo2-exploratory-r20-gen16-r18-development-20260716/
  evo2-exploratory-r20-compact-stress-20260716/
```

The final artifact directory contains CSV, JSON, Markdown, PNG, and PDF
versions of the comparisons, plus `sha256_manifest.json`. Regenerate it with:

```bash
cd ShinkaEvolve
conda run -n sakana python -B \
  examples/evo2_ecosystem/summarize_r15_r20_campaign.py \
  --output-dir \
  examples/evo2_ecosystem/results/evo2-r15-r20-final-artifacts-20260716
```

The complete chronological record, including failed launches and minimal
corrections, is in
[the positive-result campaign log](./evo2-positive-result-campaign.md).

## 9. Next scientifically valid experiment

Further search remains scientifically justified, but it cannot use the R18
sealed panel. A new campaign should:

1. generate a larger, more heterogeneous training founder bank;
2. optimize a founder-lower-tail objective rather than a tiny coherent median;
3. permit Shinka to improve bounded mutation scales or module selection, not
   only schedule fixed operators;
4. freeze a completely new development bank and a completely new sealed bank;
5. require a material positive margin on all development repeats before the
   single new sealed evaluation.

That is a new experiment, not a patch to the failed sealed result. The present
campaign remains frozen as an honest negative generalization result.
