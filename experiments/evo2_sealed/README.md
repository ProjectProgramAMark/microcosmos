# Evo² sealed evaluation

`workflow.py` is the only final-evaluation entry point. It reuses the frozen
Microcosmos evaluator, baselines, Shinka candidate validator, and paired
analysis; it does not define another simulator path.

After both development winners and their parent-chain representatives have been
frozen, restore trusted read access to the bound schema-v2 sealed manifest and
founders, then run the coordinator on CPU:

```bash
conda run -n sakana env JAX_PLATFORMS=cpu \
  python -m experiments.evo2_sealed.workflow run-all \
  --stable-frozen-dir <frozen/stable> \
  --punctuated-frozen-dir <frozen/punctuated>
```

The noninteractive driver validates both `freeze_record.json` files, their
matched run specification, preregistration, candidate sources, and every trusted
source hash. It authenticates the schema-v2 founder index and precommits the
sole global ledger under the bound experiment's `final_results/` directory,
then launches the two champions, the initial scheduler, and all five fixed
baselines sequentially. Every policy gets a fresh
24-GiB-guarded `systemd` GPU process and three full numerical measurements; the
coherent median run is retained. Re-running the command resumes only missing
outputs under the identical ledger. A second finalist set is refused.

After all eight atomic result files exist, summarize on CPU:

```bash
conda run -n sakana env JAX_PLATFORMS=cpu \
  python -m experiments.evo2_sealed.workflow summarize
```

The schema-v2 primary comparison is the founder-first, exact-world-paired
shocked-world AUC difference between the punctuated descendant and the frozen
initial scheduler. The stable descendant and every conventional baseline are
secondary comparators. Shock-minus-sham difference-in-differences is also
reported.

## Completed run

Run `evo2-actuator-rsi-20260713-r3` completed all eight policies. Its canonical
ledger, atomic policy records, and bootstrap summary are in
`experiments/evo2_ecosystem/heredity_adaptation_v3/final_results/`. The
punctuated descendant did not show a reliable advantage over its initial
ancestor or any conventional baseline, and the sealed tail injury did not
behave as a catastrophe. See `plans/evo2-actuator-rsi-r3-final-report.md` for
the complete result and integrity audit.
