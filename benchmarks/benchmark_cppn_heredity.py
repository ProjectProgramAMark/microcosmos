"""Guarded warm benchmark for a full fixed-capacity heredity batch."""

import argparse
import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp

from microcosmos.cppn import (
    DYNAMIC_NODE_KEY_OFFSET,
    CPPNGenome,
    canonicalize_neutral_attributes,
    initialize_cppn_population,
    transform_population,
)
from microcosmos.ecology import LifecycleConfig, reproduction_step
from microcosmos.heredity import (
    STRUCTURAL_GENOME,
    STRUCTURAL_STATE,
    PopulationStats,
    fixed_structural_policy,
)
from microcosmos.structs.population import PopulationState


def _population(parent_count: int) -> PopulationState:
    capacity = 2 * parent_count
    alive = jnp.arange(capacity) < parent_count
    genome = initialize_cppn_population(capacity, parent_count)
    order, connection_index = transform_population(genome)
    slots = jnp.arange(capacity, dtype=jnp.int32)
    return PopulationState(
        alive=alive,
        energy=jnp.where(alive, 3.0, 0.0),
        age=jnp.zeros(capacity, dtype=jnp.int32),
        generation=jnp.zeros(capacity, dtype=jnp.int32),
        individual_id=jnp.where(alive, slots, -1),
        parent_id=jnp.full(capacity, -1, dtype=jnp.int32),
        genome=genome,
        controller_order=order,
        controller_connection_index=connection_index,
        founder_lineage_id=jnp.where(alive, slots, -1),
        intake_ema=jnp.zeros(capacity, dtype=jnp.float32),
        population_change_ema=jnp.zeros((), dtype=jnp.float32),
        next_individual_id=jnp.asarray(parent_count, dtype=jnp.int32),
    )


def _measure(compiled, arguments: tuple, warm_runs: int) -> dict:
    start = time.perf_counter()
    first = compiled(*arguments)
    jax.block_until_ready(first)
    compile_seconds = time.perf_counter() - start
    samples = []
    for _ in range(warm_runs):
        start = time.perf_counter()
        result = compiled(*arguments)
        jax.block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return {
        "compile_and_first_execution_seconds": compile_seconds,
        "warm_samples_seconds": samples,
        "warm_median_seconds": statistics.median(samples),
    }, first


def run_benchmark(parent_count: int, warm_runs: int) -> dict:
    if parent_count < 1 or warm_runs < 3:
        raise ValueError("parent_count must be positive and warm_runs at least three")
    population = _population(parent_count)
    lifecycle = LifecycleConfig(
        maturity_age=0,
        reproduction_threshold=2.0,
        reproduction_cost=1.0,
    )
    stats = PopulationStats(
        alive_fraction=jnp.array(0.5, dtype=jnp.float32),
        population_change_ema=jnp.array(0.0, dtype=jnp.float32),
        action_diversity=jnp.array(0.1, dtype=jnp.float32),
        lineage_entropy=jnp.array(1.0, dtype=jnp.float32),
    )
    key = jax.random.PRNGKey(0)
    reproduction = jax.jit(
        lambda random_key, state: reproduction_step(
            random_key,
            state,
            lifecycle,
            fixed_structural_policy,
            stats,
        )
    )
    reproduction_result, first = _measure(reproduction, (key, population), warm_runs)

    parents = CPPNGenome(
        population.genome.node_genes[:parent_count],
        population.genome.connection_genes[:parent_count],
    )
    keys = jax.random.split(key, parent_count)
    node_keys = (DYNAMIC_NODE_KEY_OFFSET + parent_count + jnp.arange(parent_count, dtype=jnp.int32)).astype(jnp.float32)

    def mutate_one(random_key, nodes, connections, node_key):
        child_nodes, child_connections = STRUCTURAL_GENOME.mutation.mutate_structure(
            STRUCTURAL_STATE,
            STRUCTURAL_GENOME,
            random_key,
            nodes,
            connections,
            node_key,
            jnp.zeros((3,), dtype=jnp.float32),
        )
        return canonicalize_neutral_attributes(CPPNGenome(child_nodes, child_connections))

    mutation = jax.jit(jax.vmap(mutate_one))
    mutation_result, _ = _measure(
        mutation,
        (keys, parents.node_genes, parents.connection_genes, node_keys),
        warm_runs,
    )
    reproduction_median = reproduction_result["warm_median_seconds"]
    mutation_median = mutation_result["warm_median_seconds"]
    return {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "parent_count": parent_count,
        "birth_count": int(first[1]["birth_count"]),
        "mutation_kernel": {
            **mutation_result,
            "warm_microseconds_per_child": 1e6 * mutation_median / parent_count,
        },
        "validated_reproduction": {
            **reproduction_result,
            "warm_microseconds_per_birth": 1e6 * reproduction_median / parent_count,
        },
        "infrastructure_valid": bool(first[1]["infrastructure_valid"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-count", type=int, default=32)
    parser.add_argument("--warm-runs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(args.parent_count, args.warm_runs)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
