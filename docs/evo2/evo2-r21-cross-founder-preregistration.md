# Evo² R21 cross-founder heredity search

Status: **FROZEN before R21 founder generation or search**  
Date: July 16, 2026

## Observed reason for the new experiment

R15 generation 18 beat exact clone in all three R18 development repeats, and
its advantage disappeared under a matched-rate control and a no-crisis
ablation. It then failed all three independent R18 sealed repeats. R20's harder
training search also failed to transfer. The measured limitation is therefore
cross-founder generalization of ecology-conditioned mutation scheduling.

R18 sealed data is retired. It will not be used to train, rank, prompt, or test
an R21 candidate.

## Frozen question

> Can ShinkaEvolve revise the R15 ecology-conditioned scheduler on a diverse
> sixteen-founder training panel so that it beats exact clone on a completely
> new development and sealed founder bank?

## Frozen substrate and evaluator

R21 changes no physical or ecological mechanism. It retains:

- the current Microcosmos simulator and `SimulatorConfig`;
- inherited padded CPPNs and cached fixed-shape GPU inference;
- the six trusted TensorNEAT heredity operators;
- the bounded `make_offspring` ABI and validator;
- the 12,000-step horizon, 500-step chunks, and step-4,000 head injury;
- exact clone as the paired ancestor;
- the existing paired score: 80% trimmed central mean plus 20% lower-quartile
  mean across founder/world pairs;
- three full-manifest numerical repeats and coherent-median selection;
- no scenario, founder, seed, event-time, future, or hidden-partition input to
  candidate code.

## Founder panels

### Training

R21 training uses every non-sealed founder exposed by the completed R18
campaign, without selecting only favorable or difficult cases:

- all eight founders in R18's inherited training panel;
- all four founders in R18's independent training partition;
- all four now-open R18 development founders.

These sixteen founders are copied into a new immutable training index and each
is paired with world seed `19000`. R18's eight sealed founders are excluded.

### New development and sealed holdouts

Before search, generate a new independent `4/4/8` bank with:

- CPPN panel seed `9007`;
- clone-only screening seeds `18000` and `18001`;
- the unchanged 7,500-step viability screen and disjoint assigned ranges;
- selection rule `ascending-first-qualified-clone-r21-independent-v1`.

New development uses seeds `19001` and `19004`. New sealed uses seeds `20000`
and `20001`. Development and sealed founder bytes remain permission-locked
through outer search. The new sealed partition is opened once for one qualified
finalist and never reused.

## Outer search

- Initialize from frozen R15 generation 18, source SHA-256
  `fc69ac96baa846b7650e474995566228582545b915ddf8dbd4754d161d573526`.
- Run twenty Shinka evaluations under the same model and bounded source
  contract.
- Tell Shinka the measured failure: R15 fired substantially more mutation on
  sealed founders than on development founders. It should seek normalized,
  conservative signals that transfer across founder ecology rather than
  identify any hidden event.
- Preserve every proposal, prompt, failure, metric, source hash, cost, and
  runtime.

## Promotion and stopping rule

1. Outer training never opens new development or sealed founders.
2. A real non-clone, ecology-conditioned training candidate may be frozen for
   development only after a positive replicated training aggregate.
3. A finalist qualifies for sealed only if all three new-development repeat
   scores versus exact clone are positive.
4. Exactly one development-qualified source receives exactly one three-repeat
   new-sealed evaluation.
5. A positive result requires a positive sealed coherent median, plus direct
   clone, continuous standard, continuous conservative, matched-rate, and
   minimal mechanism-ablation comparisons on non-sealed data.
6. If sealed fails, record the result as negative and retire that panel. Do not
   try a second source.

This protocol changes training diversity, not the simulator, candidate inputs,
mutation portfolio, or metric. It is a new experiment with a new untouched
holdout, not a patch to the R18 sealed result.

## Prospective amendment after the unpublished founder-screen attempt

The first founder-generation attempt used the originally registered CPPN panel
seed `8007`. It completed on 2026-07-16 without publishing a bank: all eighty
candidates were finite and integrity-valid, but every candidate produced zero
births under both screening worlds. No development or sealed founder was
selected or opened. The complete append-only record is preserved under
`r21_artifacts/failed_panel_seed_8007/`.

Before generating any replacement candidates, the panel seed is changed once
to `9007`. This is the next campaign-series seed after the successful `6007`
and `7007` panels; it is not chosen from candidate performance. Every other
screening setting remains frozen: seeds `18000` and `18001`, the 7,500-step
horizon, 4/4/8 disjoint candidate ranges, viability gates, simulator, and
ascending-first-qualified selection rule are unchanged. If this replacement
also cannot publish the preregistered bank, R21 founder generation stops for a
new protocol decision rather than weakening the gates post hoc.
