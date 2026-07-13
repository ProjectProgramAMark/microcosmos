# Evo² sealed evaluation

`workflow.py` is the only final-evaluation entry point. It reuses the frozen
Microcosmos evaluator, baselines, Shinka candidate validator, and paired
analysis; it does not define another simulator path.

After both development winners have been frozen by `freeze_finalist.py`, restore
trusted read access to `final.json` and run:

```bash
conda run -n sakana python -m experiments.evo2_sealed.workflow run-all \
  --stable-frozen-dir <frozen/stable> \
  --punctuated-frozen-dir <frozen/punctuated>
```

The noninteractive driver validates both `freeze_record.json` files, their
matched run specification, preregistration, candidate sources, and every trusted
source hash. It precommits the sole global ledger at
`experiments/evo2_sealed/final_results/suite_record.json`, then launches the two
champions and four fixed baselines sequentially. Every policy gets a fresh
24-GiB-guarded `systemd` GPU process and three full numerical measurements; the
coherent median run is retained. Re-running the command resumes only missing
outputs under the identical ledger. A second finalist set is refused.

After all six atomic result files exist, summarize on CPU:

```bash
conda run -n sakana env JAX_PLATFORMS=cpu \
  python -m experiments.evo2_sealed.workflow summarize
```

The primary comparisons are seed-paired shocked-world AUC differences for the
punctuated champion versus the stable champion, fixed mixed mutation, and the
handwritten stress-responsive baseline. Positive values favor catastrophe
training. Shock-minus-null difference-in-differences is secondary.

## Completed run

Run `evo2-production-20260712-r1` completed all six policies. The canonical
ledger, atomic policy records, and bootstrap summary are in `final_results/`.
The punctuated champion did not show a reliable sealed advantage over the
stable champion or the fixed mixed and stress-responsive baselines. See
`plans/evo2-final-report.md` for the supported claim and integrity audit.
