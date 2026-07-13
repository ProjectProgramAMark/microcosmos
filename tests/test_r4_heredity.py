from dataclasses import replace

import jax
import jax.numpy as jnp

from microcosmos.ecology import OperatorCreditConfig, resolve_operator_credit
from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.heredity import (
    R4_EXPLORATORY_PARAMETRIC,
    R4_STANDARD_PARAMETRIC,
    fixed_r4_policy,
    make_mutation_context,
    mutate_cppn_r4,
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
