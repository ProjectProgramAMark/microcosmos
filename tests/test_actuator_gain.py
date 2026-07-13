from dataclasses import replace

import jax
import jax.numpy as jnp

import microcosmos.gym.ecosystem as ecosystem_module
from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.heredity import clone_policy
from microcosmos.positive_controls import (
    traveling_wave_genome,
    with_genome,
    zero_action_genome,
)
from microcosmos.rollout import initialize_chunk_metrics
from microcosmos.solver.config import PBD_SCHEME_NO_FLUID


def _environment(**kwargs) -> EcosystemEnv:
    defaults = dict(
        topology=LineTopology(num_nodes=4),
        max_creatures=1,
        initial_population=1,
        grid_shape=(20, 20),
        solver_config=PBD_SCHEME_NO_FLUID,
        initial_resource=0.0,
        resource_regeneration_rate=0.0,
        initial_energy=10.0,
        reproduction_threshold=1_000.0,
        maturity_age=1_000,
        uptake_rate=0.0,
        basal_metabolism=0.0,
        actuation_power_coefficient=1.0,
        offspring_policy=clone_policy,
    )
    defaults.update(kwargs)
    return EcosystemEnv(**defaults)


def test_reset_uses_one_world_gain_per_local_hinge_and_step_carries_it():
    env = _environment(max_creatures=3, initial_population=1)
    _, initial = env.reset(jax.random.PRNGKey(0))

    assert initial.actuator_gain.shape == (2,)
    assert initial.actuator_gain.dtype == jnp.float32
    assert jnp.array_equal(initial.actuator_gain, jnp.ones(2, dtype=jnp.float32))

    injured = replace(initial, actuator_gain=jnp.array([0.25, 0.75], dtype=jnp.float32))
    _, result, _, _, _ = jax.jit(env.step)(jax.random.PRNGKey(1), injured)

    assert result.actuator_gain.shape == initial.actuator_gain.shape
    assert jnp.array_equal(result.actuator_gain, injured.actuator_gain)


def test_gain_scales_total_bending_intent_but_not_its_energy_cost(monkeypatch):
    env = _environment(dt=0.1)
    _, initial = env.reset(jax.random.PRNGKey(2))
    initial = with_genome(initial, zero_action_genome())
    external_bend = jnp.array([0.20, -0.30], dtype=jnp.float32)
    action = {
        "d_rest_length": jnp.zeros(env.num_edges, dtype=jnp.float32),
        "d_bending_angle": external_bend,
    }
    injured = replace(
        initial,
        actuator_gain=jnp.array([0.0, 0.5], dtype=jnp.float32),
    )
    key = jax.random.PRNGKey(3)
    acted_edges = []

    def capture_physics(nodes, edges, fields, *_):
        acted_edges.append(edges)
        return nodes, edges, fields

    monkeypatch.setattr(ecosystem_module, "physics_step", capture_physics)

    _, healthy_result, _, _, _ = env.step(key, initial, action)
    _, injured_result, _, _, _ = env.step(key, injured, action)

    assert jnp.allclose(
        acted_edges[0].bending_rest_angles,
        initial.base_bending_rest_angles + external_bend,
    )
    assert jnp.allclose(
        acted_edges[1].bending_rest_angles,
        initial.base_bending_rest_angles + external_bend * injured.actuator_gain,
    )
    assert jnp.allclose(
        injured_result.population.energy,
        healthy_result.population.energy,
    )
    assert float(healthy_result.population.energy[0]) < float(initial.population.energy[0])


def test_zero_gain_blocks_controller_response_without_reducing_intended_cost(monkeypatch):
    env = _environment(dt=0.1, actuation_power_coefficient=10.0)
    _, initial = env.reset(jax.random.PRNGKey(4))
    initial = replace(
        with_genome(initial, traveling_wave_genome()),
        time=jnp.array(10, dtype=jnp.int32),
    )
    disabled = replace(initial, actuator_gain=jnp.zeros_like(initial.actuator_gain))
    key = jax.random.PRNGKey(5)
    acted_edges = []

    def capture_physics(nodes, edges, fields, *_):
        acted_edges.append(edges)
        return nodes, edges, fields

    monkeypatch.setattr(ecosystem_module, "physics_step", capture_physics)

    _, healthy_result, _, _, _ = env.step(key, initial)
    _, disabled_result, _, _, _ = env.step(key, disabled)

    assert not jnp.allclose(
        acted_edges[0].bending_rest_angles,
        initial.base_bending_rest_angles,
    )
    assert jnp.allclose(
        acted_edges[1].bending_rest_angles,
        initial.base_bending_rest_angles,
    )
    assert jnp.allclose(
        disabled_result.population.energy,
        healthy_result.population.energy,
    )
    assert float(disabled_result.population.energy[0]) < float(initial.population.energy[0])


def test_world_gain_persists_when_an_inactive_slot_is_reused_for_a_newborn():
    env = _environment(
        max_creatures=2,
        initial_population=1,
        initial_energy=5.0,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        birth_transfer_efficiency=0.5,
        maturity_age=0,
        actuation_power_coefficient=0.0,
    )
    _, initial = env.reset(jax.random.PRNGKey(6))
    injured = replace(initial, actuator_gain=jnp.zeros_like(initial.actuator_gain))

    _, after_birth, _, _, info = env.step(jax.random.PRNGKey(7), injured)
    _, after_newborn_acts, _, _, _ = env.step(jax.random.PRNGKey(8), after_birth)

    assert int(info["birth_count"]) == 1
    assert int(jnp.sum(after_birth.population.alive)) == 2
    assert jnp.array_equal(after_birth.actuator_gain, injured.actuator_gain)
    assert jnp.array_equal(after_newborn_acts.actuator_gain, injured.actuator_gain)
    assert jnp.allclose(
        after_newborn_acts.edges.bending_rest_angles,
        after_newborn_acts.base_bending_rest_angles,
    )


def test_chunk_validation_rejects_nonfinite_or_out_of_range_gain():
    env = _environment()
    _, state = env.reset(jax.random.PRNGKey(9))

    assert bool(initialize_chunk_metrics(state).finite)
    for invalid_gain in (
        jnp.array([jnp.nan, 1.0], dtype=jnp.float32),
        jnp.array([-0.1, 1.0], dtype=jnp.float32),
        jnp.array([1.0, 1.1], dtype=jnp.float32),
    ):
        invalid = replace(state, actuator_gain=invalid_gain)
        assert not bool(initialize_chunk_metrics(invalid).finite)
