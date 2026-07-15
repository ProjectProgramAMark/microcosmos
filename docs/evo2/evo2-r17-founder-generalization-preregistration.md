# Evo² R17 founder-generalization continuation

Recorded before generating the R17 founder bank or running R17 ShinkaEvolve.

## Why this continuation exists

R16b generation 11 beat clone on untouched development founders in all three
numerical repeats (`+0.005238`, `+0.014974`, `+0.014813`) but missed the robust
sealed aggregate in all three repeats (`-0.012919`, `-0.003857`, `-0.005367`).
Both sealed treatment means were positive; one `-0.101044` injury-world delta
dominated the lower tail. R17 therefore targets founder generalization rather
than changing the ecology or adding a new score.

## Fixed continuation

- Initialize Shinka from frozen R16b generation 11.
- Compare every candidate directly with exact fixed clone.
- Keep the six trusted heredity operators, fixed candidate ABI, 12,000-step
  horizon, step-4,000 head injury, physics, resource economy, and robust score.
- Use twenty Shinka evaluations.
- Train on the already exposed R16 training and development founders, using one
  world seed per founder. This doubles founder diversity from four to eight
  without increasing the sixteen-world evaluation size.
- Generate an independent founder bank with CPPN panel seed `6007`, using the
  unchanged R6 uninjured clone screen, eligibility seeds `12000` and `12001`,
  and the unchanged 4/4/8 training/development/sealed split.
- Generate and hash the independent development and sealed manifests before
  R17 search. Keep their founder artifacts outside candidate evaluation.
- Confirm distinct promising R17 programs on the new development partition
  with three numerical repeats. Open the new sealed partition only if a program
  beats clone in all three development repeats.
- A sealed miss remains negative. Do not select another candidate using sealed
  outcomes; another continuation would require a new independently frozen
  panel.

No new ecological parameter, mutation operator, score term, or simulator gate
is introduced by R17.
