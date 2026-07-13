from dataclasses import replace

import jax
import jax.numpy as jnp

import pytest

from microcosmos.ecology import (
    OperatorCreditConfig,
    reproduction_step,
    resolve_operator_credit,
)
from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.heredity import (
    R4_EXPLORATORY_PARAMETRIC,
    R4_STANDARD_PARAMETRIC,
    fixed_r4_policy,
    make_r4_offspring_policy,
    make_mutation_context,
    mutate_cppn_r4,
    r4_parent_statistics,
    r4_population_statistics,
)
from microcosmos.cppn import canonical_cppn_genome


def _env(**kwargs):
    defaults = dict(
        topology=LineTopology(num_nodes=4),
        max_creatures=4,
        initial_population=2,
        grid_shape=(32, 32),
        max_steps=20,
        maturity_age=1,
        maximum_lifespan=100,
        reproduction_threshold=2.0,
        reproduction_cost=1.0,
        initial_energy=3.0,
        resource_capacity=0.0,
        initial_resource=0.0,
        resource_regeneration_rate=0.0,
        basal_metabolism=0.0,
        actuation_power_coefficient=0.1,
        heredity_contract="r4",
        credit_chunk_steps=2,
        offspring_policy=fixed_r4_policy(R4_STANDARD_PARAMETRIC),
    )
    defaults.update(kwargs)
    return EcosystemEnv(**defaults)


def test_saturated_r4_logits_select_exactly_and_other_logits_are_probabilistic():
    parent = canonical_cppn_genome()
    context, valid = make_mutation_context(jax.random.PRNGKey(2), jnp.asarray(7))
    assert bool(valid)
    logits = jnp.full(6, -8.0).at[R4_EXPLORATORY_PARAMETRIC].set(8.0)
    result = mutate_cppn_r4(parent, logits, context)
    assert int(result.operator_index) == R4_EXPLORATORY_PARAMETRIC
    assert result.selection_probabilities.tolist() == [0, 0, 0, 1, 0, 0]

    mixed = mutate_cppn_r4(parent, jnp.zeros(6), context)
    assert jnp.allclose(mixed.selection_probabilities, jnp.full(6, 1 / 6))


def test_credit_is_founder_excluding_single_resolution_and_order_independent():
    env = _env()
    _, state = env.reset(jax.random.PRNGKey(0))
    pop = replace(
        state.population,
        birth_operator=jnp.asarray([1, 3, -1, -1], dtype=jnp.int32),
        birth_step=jnp.asarray([2, 4, -1, -1], dtype=jnp.int32),
        has_reproduced=jnp.zeros(4, dtype=jnp.bool_),
    )
    died = jnp.asarray([False, True, True, False])
    slots = jnp.asarray([0, -1, -1, -1], dtype=jnp.int32)
    valid = jnp.asarray([True, False, False, False])
    config = OperatorCreditConfig(tau_steps=5.0)
    updated, successes, failures, _ = resolve_operator_credit(
        pop, died, slots, valid, jnp.asarray(7), config
    )
    assert successes.tolist() == [0, 1, 0, 0, 0, 0]
    assert failures.tolist() == [0, 0, 0, 1, 0, 0]
    assert bool(updated.has_reproduced[0])
    assert updated.operator_evidence_ema[1] > 0
    assert updated.operator_evidence_ema[3] > 0

    again, successes2, failures2, _ = resolve_operator_credit(
        updated, jnp.zeros_like(died), slots, valid, jnp.asarray(8), config
    )
    assert int(jnp.sum(successes2)) == 0
    assert int(jnp.sum(failures2)) == 0
    assert jnp.array_equal(again.operator_success_ema, updated.operator_success_ema)


def test_r4_birth_records_action_probabilities_and_provenance():
    env = _env()
    _, state = env.reset(jax.random.PRNGKey(0))
    _, after, _, _, info = env.step(jax.random.PRNGKey(1), state)
    assert info["operator_counts"].shape == (6,)
    assert int(jnp.sum(info["operator_counts"])) == int(info["birth_count"])
    assert jnp.allclose(jnp.sum(info["operator_probability_sum"]), info["birth_count"])
    child_slots = info["telemetry"].birth_child_slots
    for slot in child_slots[child_slots >= 0].tolist():
        assert int(after.population.birth_operator[slot]) == R4_STANDARD_PARAMETRIC
        assert int(after.population.birth_step[slot]) == 0
        assert not bool(after.population.has_reproduced[slot])


def test_actuation_cost_multiplier_changes_energy_not_physical_actuation():
    env = _env(initial_population=1, initial_energy=10.0, reproduction_threshold=20.0)
    _, state = env.reset(jax.random.PRNGKey(0))
    costly = replace(state, actuation_cost_multiplier=jnp.asarray(3.0))
    key = jax.random.PRNGKey(5)
    _, normal_after, _, _, _ = env.step(key, state)
    _, costly_after, _, _, _ = env.step(key, costly)
    assert jnp.allclose(normal_after.nodes.position, costly_after.nodes.position)
    assert jnp.allclose(normal_after.nodes.velocity, costly_after.nodes.velocity)
    assert costly_after.population.energy[0] <= normal_after.population.energy[0]


def test_r4_public_adapter_exposes_only_bounded_arrays_and_requires_float32():
    env = _env()
    _, state = env.reset(jax.random.PRNGKey(0))
    pop = replace(
        state.population,
        age=jnp.where(state.population.alive, 1, state.population.age),
    )
    parent = canonical_cppn_genome()
    parent_stats = r4_parent_statistics(
        parent,
        pop.energy[0],
        pop.intake_ema[0],
        pop.age[0],
        env.lifecycle_config.reproduction_threshold,
        env.lifecycle_config.maximum_lifespan,
    )
    population_stats = r4_population_statistics(
        pop.alive,
        pop.energy,
        pop.intake_ema,
        pop.population_change_ema,
        env.lifecycle_config.reproduction_threshold,
        pop.birth_rate_ema,
        pop.death_rate_ema,
        pop.operator_success_ema,
        pop.operator_usage_ema,
        pop.operator_evidence_ema,
    )
    context, _ = make_mutation_context(jax.random.PRNGKey(1), jnp.asarray(9))
    observed = []

    def candidate(*arrays):
        observed.append(arrays)
        return jnp.asarray([-8, -8, 8, -8, -8, -8], dtype=jnp.float32)

    result = make_r4_offspring_policy(candidate)(
        parent, parent_stats, population_stats, context
    )
    assert int(result.operator_index) == R4_STANDARD_PARAMETRIC
    assert tuple(value.shape for value in observed[0]) == ((2,), (3,), (6,), (3, 6), ())
    assert all(value.dtype == jnp.float32 for value in observed[0])

    bad = make_r4_offspring_policy(
        lambda *_: jnp.zeros(6, dtype=jnp.int32)
    )
    with pytest.raises(TypeError, match="float32"):
        bad(parent, parent_stats, population_stats, context)


def test_r4_credit_batch_is_invariant_to_slot_permutation():
    env = _env()
    _, state = env.reset(jax.random.PRNGKey(0))
    pop = replace(
        state.population,
        alive=jnp.ones(4, dtype=jnp.bool_),
        birth_operator=jnp.asarray([1, 1, 3, 3], dtype=jnp.int32),
        birth_step=jnp.asarray([2, 4, 2, 4], dtype=jnp.int32),
        has_reproduced=jnp.zeros(4, dtype=jnp.bool_),
    )
    died = jnp.asarray([False, True, False, True])
    reproducing = jnp.asarray([0, 2, -1, -1], dtype=jnp.int32)
    valid = reproducing >= 0
    config = OperatorCreditConfig(tau_steps=5.0)
    expected = resolve_operator_credit(pop, died, reproducing, valid, 7, config)

    permutation = jnp.asarray([2, 0, 3, 1])
    inverse = jnp.argsort(permutation)
    permuted = jax.tree.map(
        lambda value: value[permutation]
        if getattr(value, "ndim", 0) > 0 and value.shape[0] == 4
        else value,
        pop,
    )
    permuted_died = died[permutation]
    permuted_reproducing = jnp.where(valid, inverse[jnp.maximum(reproducing, 0)], -1)
    actual = resolve_operator_credit(
        permuted, permuted_died, permuted_reproducing, valid, 7, config
    )
    assert jnp.allclose(actual[0].operator_success_ema, expected[0].operator_success_ema)
    assert jnp.allclose(actual[0].operator_evidence_ema, expected[0].operator_evidence_ema)
    assert jnp.array_equal(actual[1], expected[1])
    assert jnp.array_equal(actual[2], expected[2])


def test_r4_reproduction_can_force_exactly_one_birth():
    env = _env()
    _, state = env.reset(jax.random.PRNGKey(0))
    pop = replace(
        state.population,
        age=jnp.where(state.population.alive, 1, state.population.age),
    )
    stats = r4_population_statistics(
        pop.alive,
        pop.energy,
        pop.intake_ema,
        pop.population_change_ema,
        env.lifecycle_config.reproduction_threshold,
        pop.birth_rate_ema,
        pop.death_rate_ema,
        pop.operator_success_ema,
        pop.operator_usage_ema,
        pop.operator_evidence_ema,
    )
    updated, events = reproduction_step(
        jax.random.PRNGKey(5),
        pop,
        env.lifecycle_config,
        fixed_r4_policy(R4_STANDARD_PARAMETRIC),
        stats,
        heredity_contract="r4",
        died=jnp.zeros_like(pop.alive),
        credit_config=env.credit_config,
        max_births=1,
    )
    assert int(events["birth_count"]) == 1
    assert int(jnp.sum(updated.alive)) == int(jnp.sum(pop.alive)) + 1


def test_r4_birth_policy_sees_credit_resolved_in_the_same_step():
    env = _env(initial_population=1)
    _, state = env.reset(jax.random.PRNGKey(0))
    pop = replace(
        state.population,
        age=state.population.age.at[0].set(1),
        birth_operator=state.population.birth_operator.at[1].set(3),
        birth_step=state.population.birth_step.at[1].set(0),
    )
    stale_stats = r4_population_statistics(
        pop.alive,
        pop.energy,
        pop.intake_ema,
        pop.population_change_ema,
        env.lifecycle_config.reproduction_threshold,
        pop.birth_rate_ema,
        pop.death_rate_ema,
        pop.operator_success_ema,
        pop.operator_usage_ema,
        pop.operator_evidence_ema,
    )

    def evidence_policy(parent, parent_stats, population_stats, context):
        del parent_stats
        selected = jnp.where(population_stats.operator_evidence_ema[3] > 0, 1, 2)
        logits = jnp.full(6, -8.0, dtype=jnp.float32).at[selected].set(8.0)
        return mutate_cppn_r4(parent, logits, context)

    _, events = reproduction_step(
        jax.random.PRNGKey(8),
        pop,
        env.lifecycle_config,
        evidence_policy,
        stale_stats,
        heredity_contract="r4",
        died=jnp.asarray([False, True, False, False]),
        credit_config=env.credit_config,
        max_births=1,
    )
    assert events["operator_counts"].tolist() == [0, 1, 0, 0, 0, 0]
    assert events["resolved_failure_count"].tolist() == [0, 0, 0, 1, 0, 0]
