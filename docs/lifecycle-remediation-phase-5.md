# Lifecycle remediation Phase 5 RNG contract

The ecosystem uses immutable integer stream tags and repeated
`jax.random.fold_in`:

| Tag | Value | Scope |
|---|---:|---|
| initialization | 1 | initial placement and genomes |
| environment | 2 | exogenous world draws |
| mutation | 3 | timestep plus child individual ID |
| spawn | 4 | timestep plus child individual ID |
| future event | 5 | reserved exogenous event stream |

Mutation and spawn never derive randomness from ranked event-array position.
For a fixed base key, timestep, and immutable child ID, inserting, removing, or
reordering unrelated births leaves both keys and draws unchanged.

Initialization has independent placement/genome substreams. Exogenous
environment streams are disjoint from endogenous mutation/spawn paths, so a
private heredity implementation cannot shift the world RNG sequence.

The only heredity implementation in this remediation phase remains the private
trusted `_fixed_gaussian_child` development baseline. The final public
candidate-facing offspring policy, statistics types, and verifier boundary are
explicitly deferred to Master Phase 4 after sustained-turnover calibration; no
candidate callable is exposed here.

Verification on the pinned CPU stack:

- identity-key insertion/removal/reordering tests pass;
- mutation and spawn draws match bitwise for a retained child ID;
- initialization/resource state is unchanged when the private baseline is replaced;
- a seeded five-step lifecycle event trace replays bitwise;
- complete CPU suite: `255 passed in 98.94s`.

The guarded fluid positive controls were rerun after tagged initialization. Both
remain finite and pass: movement is `0.2061` body lengths versus zero drift;
paired sensing improves mean/median uptake by `451.38%`/`477.30%` with `8/8`
wins.
