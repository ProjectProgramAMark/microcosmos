from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from microcosmos.cppn import CPPNGenome
from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.heredity import clone_policy
from microcosmos.rollout import (
    initialize_chunk_metrics,
    run_ecosystem_chunk,
    run_ecosystem_chunk_with_snapshots,
)
from microcosmos.solver.config import PBD_SCHEME_NO_FLUID


def _environment():
    return EcosystemEnv(
        topology=LineTopology(num_nodes=4),
        max_creatures=3,
        initial_population=1,
        grid_shape=(20, 20),
        solver_config=PBD_SCHEME_NO_FLUID,
        initial_energy=5.0,
        maturity_age=0,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        uptake_rate=0.0,
        basal_metabolism=0.0,
        actuation_power_coefficient=0.0,
        offspring_policy=clone_policy,
    )


def test_search_chunk_matches_manual_scan_without_returning_history():
    env = _environment()
    _, initial = env.reset(jax.random.PRNGKey(0))
    keys = jax.random.split(jax.random.PRNGKey(1), 6)
    final, metrics = jax.jit(lambda state, scan_keys: run_ecosystem_chunk(env, state, scan_keys))(initial, keys)

    def manual_step(state, key):
        return env.step(key, state)[1], None

    manual, _ = jax.lax.scan(manual_step, initial, keys)
    for chunk_value, manual_value in zip(jax.tree.leaves(final), jax.tree.leaves(manual), strict=True):
        if jnp.issubdtype(chunk_value.dtype, jnp.inexact):
            assert jnp.allclose(
                chunk_value,
                manual_value,
                atol=1e-4,
                rtol=1e-5,
                equal_nan=True,
            )
        else:
            assert jnp.array_equal(chunk_value, manual_value)
    assert int(metrics.steps) == 6
    assert bool(metrics.finite)
    assert bool(metrics.identity_valid)
    assert bool(metrics.events_valid)
    assert bool(metrics.infrastructure_valid)
    assert int(metrics.policy_violation_count) == 0
    assert int(jnp.sum(metrics.operator_counts)) == int(metrics.birth_count)
    assert metrics.operator_counts.tolist() == [1, 0, 0, 0]
    assert int(metrics.maximum_generation) == 1
    assert int(metrics.steps_at_capacity) >= 0
    assert metrics.minimum_resource >= -1e-6
    assert metrics.maximum_resource_excess <= 1e-6
    for name, value in vars(metrics).items():
        if name == "operator_counts":
            assert value.shape == (4,)
        else:
            assert value.ndim == 0


def test_sparse_snapshot_mode_uses_nested_chunks_and_drops_f_grid():
    env = _environment()
    _, initial = env.reset(jax.random.PRNGKey(2))
    keys = jax.random.split(jax.random.PRNGKey(3), 8)
    final, metrics, snapshots = jax.jit(lambda state, scan_keys: run_ecosystem_chunk_with_snapshots(env, state, scan_keys, snapshot_stride=2))(initial, keys)
    nodes, fields, population, times = snapshots
    assert int(final.time) == 8
    assert int(metrics.steps) == 8
    assert nodes.position.shape[0] == 4
    assert fields.steric.shape[0] == 4
    assert fields.f_grid is None
    assert population.alive.shape[0] == 4
    assert times.tolist() == [2, 4, 6, 8]
    assert int(jnp.sum(metrics.operator_counts)) == int(metrics.birth_count)


def test_chunk_validation_accepts_gene_padding_but_rejects_corruption():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(20))

    # TensorNEAT uses complete all-NaN rows as legal fixed-capacity padding.
    assert jnp.any(jnp.isnan(state.population.genome.node_genes))
    assert jnp.any(jnp.isnan(state.population.genome.connection_genes))
    assert bool(initialize_chunk_metrics(state).finite)

    physically_corrupt = replace(
        state,
        nodes=replace(
            state.nodes,
            position=state.nodes.position.at[0, 0].set(jnp.nan),
        ),
    )
    assert not bool(initialize_chunk_metrics(physically_corrupt).finite)

    mixed_gene_row = CPPNGenome(
        state.population.genome.node_genes.at[0, 0, 1].set(jnp.nan),
        state.population.genome.connection_genes,
    )
    genetically_corrupt = replace(
        state,
        population=replace(state.population, genome=mixed_gene_row),
    )
    assert not bool(initialize_chunk_metrics(genetically_corrupt).finite)


@pytest.mark.parametrize("stride", [0, -1])
def test_sparse_snapshot_mode_rejects_invalid_stride(stride):
    env = _environment()
    _, initial = env.reset(jax.random.PRNGKey(4))
    with pytest.raises(ValueError, match="snapshot_stride"):
        run_ecosystem_chunk_with_snapshots(
            env,
            initial,
            jax.random.split(jax.random.PRNGKey(5), 4),
            snapshot_stride=stride,
        )
