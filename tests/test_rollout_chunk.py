import jax
import jax.numpy as jnp
import pytest

from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.rollout import (
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
        mutation_std=0.0,
    )


def test_search_chunk_matches_manual_scan_without_returning_history():
    env = _environment()
    _, initial = env.reset(jax.random.PRNGKey(0))
    keys = jax.random.split(jax.random.PRNGKey(1), 6)
    final, metrics = jax.jit(
        lambda state, scan_keys: run_ecosystem_chunk(env, state, scan_keys)
    )(initial, keys)

    def manual_step(state, key):
        return env.step(key, state)[1], None

    manual, _ = jax.lax.scan(manual_step, initial, keys)
    for chunk_value, manual_value in zip(
        jax.tree.leaves(final), jax.tree.leaves(manual), strict=True
    ):
        if jnp.issubdtype(chunk_value.dtype, jnp.inexact):
            assert jnp.allclose(chunk_value, manual_value, atol=1e-4, rtol=1e-5)
        else:
            assert jnp.array_equal(chunk_value, manual_value)
    assert int(metrics.steps) == 6
    assert bool(metrics.finite)
    assert bool(metrics.identity_valid)
    assert bool(metrics.events_valid)
    assert metrics.minimum_resource >= -1e-6
    assert metrics.maximum_resource_excess <= 1e-6
    assert all(value.ndim == 0 for value in jax.tree.leaves(metrics))


def test_sparse_snapshot_mode_uses_nested_chunks_and_drops_f_grid():
    env = _environment()
    _, initial = env.reset(jax.random.PRNGKey(2))
    keys = jax.random.split(jax.random.PRNGKey(3), 8)
    final, metrics, snapshots = jax.jit(
        lambda state, scan_keys: run_ecosystem_chunk_with_snapshots(
            env, state, scan_keys, snapshot_stride=2
        )
    )(initial, keys)
    nodes, fields, population, times = snapshots
    assert int(final.time) == 8
    assert int(metrics.steps) == 8
    assert nodes.position.shape[0] == 4
    assert fields.steric.shape[0] == 4
    assert fields.f_grid is None
    assert population.alive.shape[0] == 4
    assert times.tolist() == [2, 4, 6, 8]


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
