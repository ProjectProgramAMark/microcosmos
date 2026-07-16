# Evo² R18 hard-founder generalization continuation

Status: **FROZEN before R18 founder generation, candidate selection, or search**

## Observed reason for continuation

R17 generation 2 beat clone on the independently generated development panel
in all three numerical repeats (`+0.007631`, `+0.005428`, `+0.004459`). Its
single sealed examination failed: the robust score was `-0.033145`, with three
negative repeats (`-0.038801`, `-0.025094`, `-0.033145`). The selected repeat
contained founder/world deltas of `-0.271151`, `-0.176056`, and `-0.096570`.
The measured failure is therefore founder generalization, not invalid code,
numerical instability, or absence of injury benefit in development.

## Frozen continuation

- Preserve the simulator, 12,000-step horizon, injury at step 4,000, fixed
  clone comparator, six trusted heredity operators, policy ABI, and scoring.
- Complete the already-running 20-evaluation R17 search and retain its full
  archive as a negative/diagnostic record.
- Treat the entire now-exposed R17 sealed founder panel as R18 training data;
  do not select only the worst founders. Use one fixed world seed per founder,
  maintaining the existing eight-founder/sixteen-world training size. Use the
  lower of the two already-exposed R17 sealed seeds (`15000`) for every founder.
- Initialize R18 from the strongest R17 source under the unchanged training
  score, with preference to a source that also passed independent development.
- Generate a new independent `4/4/8` founder bank before R18 search using CPPN
  panel seed `7007`, screening seeds `13000` and `13001`, the unchanged
  clone-only viability screen, and disjoint preassigned index ranges.
- Use the new training partition only for a smoke comparison, not R18 search.
  Keep the new development partition unavailable until an R18 candidate is
  frozen for three-repeat confirmation.
- Open the new sealed partition once, for one development-qualified finalist.
- If the sealed finalist fails, record the result as negative. Do not try a
  second finalist on the same sealed panel.

## Selection and stopping rule

A candidate advances only if it beats fixed clone in all three numerical
repeats on the new development founders. A defensible positive result requires
a positive robust aggregate on the new sealed founders, direct comparisons
with conventional mutation and matched-rate controls, and an ablation showing
that ecological/operator-evidence conditioning—not mutation quantity alone—
causes the gain.

No new ecological variable, simulator parameter, mutation primitive, score,
or validity gate is introduced by R18.
