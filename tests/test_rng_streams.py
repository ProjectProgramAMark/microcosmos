import jax
import jax.numpy as jnp

from microcosmos.cppn import CPPNGenome, initialize_cppn_population
from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.heredity import (
    ParentStats,
    PopulationStats,
    clone_policy,
    fixed_parametric_policy,
    make_mutation_context,
)
from microcosmos.rng import RNGTag, derive_key, keys_for_identities
from microcosmos.solver.config import PBD_SCHEME_NO_FLUID


def _draws_by_identity(base_key, tag, timestep, identities):
    keys = keys_for_identities(base_key, tag, timestep, identities)
    draws = jax.vmap(jax.random.uniform)(keys)
    return {int(identity): (key, draw) for identity, key, draw in zip(identities, keys, draws, strict=True)}


def _array_equal(left, right):
    if jnp.issubdtype(left.dtype, jnp.inexact):
        return jnp.array_equal(left, right, equal_nan=True)
    return jnp.array_equal(left, right)


def _policy_stats():
    return (
        ParentStats(
            energy_fraction=jnp.array(1.0),
            intake_ema=jnp.array(0.5),
            node_fraction=jnp.array(0.5),
            connection_fraction=jnp.array(0.5),
        ),
        PopulationStats(
            alive_fraction=jnp.array(0.5),
            population_change_ema=jnp.array(0.0),
            action_diversity=jnp.array(0.25),
            lineage_entropy=jnp.array(1.0),
        ),
    )


def _mutated_child(key, child_id):
    population = initialize_cppn_population(1, 1)
    parent = CPPNGenome(population.node_genes[0], population.connection_genes[0])
    context, valid = make_mutation_context(key, child_id)
    assert bool(valid)
    parent_stats, population_stats = _policy_stats()
    return fixed_parametric_policy(parent, parent_stats, population_stats, context).genome


def test_rng_tags_are_frozen_unique_integer_streams():
    assert {int(tag) for tag in RNGTag} == {1, 2, 3, 4, 5}
    base = jax.random.PRNGKey(0)
    keys = jnp.stack([derive_key(base, tag) for tag in RNGTag])
    assert jnp.unique(keys, axis=0).shape[0] == len(RNGTag)


def test_identity_draws_ignore_insertion_removal_and_rank_order():
    base = jax.random.PRNGKey(1)
    first = _draws_by_identity(base, RNGTag.MUTATION, 17, jnp.array([41, 7, 99], dtype=jnp.int32))
    reordered = _draws_by_identity(base, RNGTag.MUTATION, 17, jnp.array([123, 99, 41], dtype=jnp.int32))
    for identity in (41, 99):
        assert jnp.array_equal(first[identity][0], reordered[identity][0])
        assert jnp.array_equal(first[identity][1], reordered[identity][1])


def test_same_child_id_keeps_mutation_and_spawn_draws():
    base = jax.random.PRNGKey(2)
    timestep = 23
    mutation_a = keys_for_identities(base, RNGTag.MUTATION, timestep, jnp.array([8, 55, 13]))[1]
    mutation_b = keys_for_identities(base, RNGTag.MUTATION, timestep, jnp.array([55, 999]))[0]
    child_a = _mutated_child(mutation_a, child_id=55)
    child_b = _mutated_child(mutation_b, child_id=55)
    assert all(
        _array_equal(left, right)
        for left, right in zip(
            jax.tree.leaves(child_a),
            jax.tree.leaves(child_b),
            strict=True,
        )
    )

    spawn_a = keys_for_identities(base, RNGTag.SPAWN, timestep, jnp.array([2, 55, 80]))[1]
    spawn_b = keys_for_identities(base, RNGTag.SPAWN, timestep, jnp.array([55]))[0]
    assert jnp.array_equal(spawn_a, spawn_b)
    assert jnp.array_equal(jax.random.uniform(spawn_a), jax.random.uniform(spawn_b))


def test_initialization_and_environment_keys_are_mutation_independent():
    base = jax.random.PRNGKey(3)
    initialization = derive_key(base, RNGTag.INITIALIZATION)
    environment = derive_key(base, RNGTag.ENVIRONMENT, 12)
    mutation_keys = keys_for_identities(base, RNGTag.MUTATION, 12, jnp.arange(100, dtype=jnp.int32))
    _ = jax.vmap(jax.random.normal)(mutation_keys)
    assert jnp.array_equal(initialization, derive_key(base, RNGTag.INITIALIZATION))
    assert jnp.array_equal(environment, derive_key(base, RNGTag.ENVIRONMENT, 12))


def test_changing_offspring_policy_cannot_shift_world_before_any_birth():
    common = dict(
        topology=LineTopology(num_nodes=4),
        max_creatures=2,
        initial_population=1,
        grid_shape=(20, 20),
        solver_config=PBD_SCHEME_NO_FLUID,
        maturity_age=50,
        maximum_lifespan=100,
        resource_regeneration_rate=0.02,
    )
    clone_env = EcosystemEnv(**common, offspring_policy=clone_policy)
    mutation_env = EcosystemEnv(**common, offspring_policy=fixed_parametric_policy)
    reset_key = jax.random.PRNGKey(30)
    _, initial_before = clone_env.reset(reset_key)
    _, initial_after = mutation_env.reset(reset_key)
    _, step_before, _, _, _ = clone_env.step(jax.random.PRNGKey(31), initial_before)
    _, step_after, _, _, _ = mutation_env.step(jax.random.PRNGKey(31), initial_after)

    for before, after in zip(
        jax.tree.leaves(initial_before),
        jax.tree.leaves(initial_after),
        strict=True,
    ):
        assert _array_equal(before, after)
    assert jnp.array_equal(step_before.fields.energy, step_after.fields.energy)
    assert jnp.array_equal(step_before.resource_capacity_map, step_after.resource_capacity_map)


def test_seeded_lifecycle_event_trace_replays_bitwise():
    env = EcosystemEnv(
        topology=LineTopology(num_nodes=4),
        max_creatures=4,
        initial_population=2,
        grid_shape=(24, 24),
        solver_config=PBD_SCHEME_NO_FLUID,
        initial_energy=5.0,
        maturity_age=0,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        birth_transfer_efficiency=0.5,
        uptake_rate=0.0,
        basal_metabolism=0.0,
        actuation_power_coefficient=0.0,
        offspring_policy=fixed_parametric_policy,
    )
    _, initial = env.reset(jax.random.PRNGKey(4))
    keys = jax.random.split(jax.random.PRNGKey(5), 5)

    def rollout(state):
        def scan_step(carry, key):
            _, next_state, _, _, info = env.step(key, carry)
            events = (
                info["birth_parent_ids"],
                info["birth_child_ids"],
                info["death_ids"],
            )
            return next_state, events

        return jax.lax.scan(scan_step, state, keys)

    final_a, trace_a = jax.jit(rollout)(initial)
    final_b, trace_b = jax.jit(rollout)(initial)
    for value_a, value_b in zip(
        jax.tree.leaves((final_a, trace_a)),
        jax.tree.leaves((final_b, trace_b)),
        strict=True,
    ):
        assert _array_equal(value_a, value_b)
