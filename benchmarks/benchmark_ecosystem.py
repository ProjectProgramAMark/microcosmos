"""Evaluator-shaped fused-scan ecosystem benchmark."""

import argparse
import json
from pathlib import Path
import platform
import resource
import statistics
import time

import jax
import numpy as np

from microcosmos.gym import EcosystemEnv, LineTopology, MultiAgentEnv
from microcosmos.rollout import run_ecosystem_chunk
from microcosmos.solver.config import PBD_SCHEME, PBD_SCHEME_NO_FLUID


def _block_until_ready(tree) -> None:
    for value in jax.tree.leaves(tree):
        block = getattr(value, "block_until_ready", None)
        if block is not None:
            block()


def _timing_summary(samples: list[float], steps: int) -> dict:
    median = statistics.median(samples)
    lower, upper = np.percentile(samples, [25.0, 75.0])
    absolute_deviation = [abs(sample - median) for sample in samples]
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "iqr_seconds": float(upper - lower),
        "mad_seconds": statistics.median(absolute_deviation),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
        "simulated_steps_per_second": steps / median,
    }


def _measure(compiled, arguments: tuple, steps: int, warm_runs: int) -> dict:
    start = time.perf_counter()
    first_result = compiled(*arguments)
    _block_until_ready(first_result)
    compile_and_first = time.perf_counter() - start

    samples = []
    for _ in range(warm_runs):
        start = time.perf_counter()
        result = compiled(*arguments)
        _block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return {
        "compile_and_first_execution_seconds": compile_and_first,
        "warm": _timing_summary(samples, steps),
    }


def run_benchmark(args) -> dict:
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.warm_runs < 3:
        raise ValueError("warm_runs must be at least three")
    if args.max_creatures < 1 or args.nodes_per_creature < 2:
        raise ValueError("benchmark dimensions must be positive")

    solver = PBD_SCHEME_NO_FLUID if args.no_fluid else PBD_SCHEME
    grid_shape = (args.grid_size, args.grid_size)
    initial_population = (
        max(1, args.max_creatures // 2)
        if args.initial_population is None
        else args.initial_population
    )
    env = EcosystemEnv(
        topology=LineTopology(num_nodes=args.nodes_per_creature),
        max_creatures=args.max_creatures,
        initial_population=initial_population,
        solver_config=solver,
        grid_shape=grid_shape,
        max_steps=args.steps,
    )
    baseline = MultiAgentEnv(
        creatures=tuple(
            LineTopology(num_nodes=args.nodes_per_creature)
            for _ in range(args.max_creatures)
        ),
        solver_config=solver,
        grid_shape=grid_shape,
        max_steps=args.steps,
    )

    reset_key = jax.random.PRNGKey(args.seed)
    rollout_key = jax.random.fold_in(reset_key, 1)
    keys = jax.random.split(rollout_key, args.steps)
    _, ecosystem_state = env.reset(reset_key)
    _, baseline_state = baseline.reset(reset_key)
    baseline_action = baseline.zero_action()

    def baseline_chunk(state, scan_keys):
        def scan_step(current, key):
            _, next_state, _, _, _ = baseline.step(
                key, current, baseline_action
            )
            return next_state, None

        final_state, _ = jax.lax.scan(scan_step, state, scan_keys)
        return final_state

    ecosystem_compiled = jax.jit(
        lambda state, scan_keys: run_ecosystem_chunk(env, state, scan_keys)
    )
    baseline_compiled = jax.jit(baseline_chunk)

    baseline_result = _measure(
        baseline_compiled,
        (baseline_state, keys),
        args.steps,
        args.warm_runs,
    )
    ecosystem_result = _measure(
        ecosystem_compiled,
        (ecosystem_state, keys),
        args.steps,
        args.warm_runs,
    )
    baseline_median = baseline_result["warm"]["median_seconds"]
    ecosystem_median = ecosystem_result["warm"]["median_seconds"]

    peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return {
        "methodology": "fixed-size fused lax.scan; final state and compact metrics only",
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "steps": args.steps,
        "warm_runs": args.warm_runs,
        "shapes": {
            "grid": list(grid_shape),
            "baseline_nodes": int(baseline_state.nodes.position.shape[0]),
            "ecosystem_nodes": int(ecosystem_state.nodes.position.shape[0]),
            "baseline_active_nodes": int(baseline_state.nodes.position.shape[0]),
            "nodes_per_creature": args.nodes_per_creature,
            "max_creatures": args.max_creatures,
            "initial_population": initial_population,
            "initial_active_nodes": int(
                np.asarray(ecosystem_state.nodes.active).sum()
            ),
        },
        "solver": {
            "name": solver.name,
            "fluid_enabled": solver.enable_fluid,
            "steric_enabled": solver.enable_steric,
            "cycles_per_step": solver.cycles_per_step,
            "ibm_iterations": solver.ibm_iterations,
            "ibm_kernel_size": solver.ibm_kernel_size,
        },
        "ecosystem_config": {
            "dt": env.dt,
            "initial_resource_peak": env.initial_resource,
            "resource_capacity_peak": env.resource_config.capacity,
            "resource_regeneration_peak": env.resource_config.regeneration_rate,
            "resource_diffusion_rate": env.resource_config.diffusion_rate,
            "initial_energy": env.initial_energy,
            "reproduction_threshold": env.lifecycle_config.reproduction_threshold,
            "reproduction_cost": env.lifecycle_config.reproduction_cost,
            "birth_transfer_efficiency": (
                env.lifecycle_config.birth_transfer_efficiency
            ),
            "maturity_age": env.lifecycle_config.maturity_age,
            "maximum_lifespan": env.lifecycle_config.maximum_lifespan,
            "uptake_rate": env.uptake_rate,
            "assimilation_efficiency": env.assimilation_efficiency,
            "basal_metabolism": env.basal_metabolism,
            "max_bending_delta": env.genome_config.max_bending_delta,
            "resource_reference": env.genome_config.resource_reference,
        },
        "retention": {
            "full_node_history": False,
            "full_fluid_history": False,
            "sparse_snapshots": False,
        },
        "baseline": baseline_result,
        "ecosystem": ecosystem_result,
        "ecosystem_to_baseline_warm_median_ratio": (
            ecosystem_median / baseline_median
        ),
        "peak_process_rss_mib": peak_rss_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--max-creatures", type=int, default=32)
    parser.add_argument("--initial-population", type=int)
    parser.add_argument("--nodes-per-creature", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-fluid", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_benchmark(arguments)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
