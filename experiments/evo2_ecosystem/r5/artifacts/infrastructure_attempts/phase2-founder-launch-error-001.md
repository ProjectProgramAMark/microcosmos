# Evo² r5 world-feasibility stop report

- Protocol: `evo2-r5-world-feasibility-v1`
- Run ID: `evo2-r5-world-feasibility-20260713`
- Failed phase: `phase-2-founder-gate`
- Reason: Command '['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemoryHigh=20G', '-p', 'MemoryMax=24G', '-p', 'MemorySwapMax=2G', 'conda', 'run', '-n', 'sakana', 'env', '-u', 'JAX_PLATFORMS', '-u', 'JAX_PLATFORM_NAME', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=/home/mmmoussa/Programming/sakana/origins-of-life/r5-launch/ShinkaEvolve:/home/mmmoussa/Programming/sakana/origins-of-life/r5-launch/microcosmos/src:/home/mmmoussa/Programming/sakana/origins-of-life/r5-launch/microcosmos', '-u', 'XLA_PYTHON_CLIENT_PREALLOCATE', 'python', '-B', '-m', 'experiments.evo2_ecosystem.r5.workflow', 'founder-gate-worker', '--world-qualification', '/home/mmmoussa/Programming/sakana/origins-of-life/r5-launch/microcosmos/experiments/evo2_ecosystem/r5/artifacts/world_qualification.json', '--bank-directory', '/home/mmmoussa/Programming/sakana/origins-of-life/r5-launch/microcosmos/experiments/evo2_ecosystem/r5/artifacts/founders', '--screening-record', '/home/mmmoussa/Programming/sakana/origins-of-life/r5-launch/microcosmos/experiments/evo2_ecosystem/r5/artifacts/founder_screening.jsonl']' returned non-zero exit status 127.
- Result: execution stopped; no later gate is authorized.

## Existing evidence

- `microcosmos/experiments/evo2_ecosystem/r5/artifacts/world_qualification.json` (`8f281d393db25b5b0f92722af30607d0903e979bdf8753db2f8a7feb6c5d813c`)
- `microcosmos/experiments/evo2_ecosystem/r5/artifacts/founder_screening.jsonl` (`missing`)
