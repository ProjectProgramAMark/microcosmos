import jax
import jax.numpy as jnp
import pytest

from microcosmos.gym import (
    EnvState,
    LineTopology,
    MultiAgentEnv,
    RingTopology,
    make,
    registered_envs,
)
from microcosmos.solver.config import PBD_SCHEME_NO_FLUID


ENV_NAMES = ["line", "ring", "tadpole", "line-ring", "multi-agent"]


def _small_env(name: str, max_steps: int = 4):
    # Tests skip the LBM fluid pass so they run in seconds, not minutes.
    kwargs = {
        "max_steps": max_steps,
        "grid_shape": (32, 32),
        "solver_config": PBD_SCHEME_NO_FLUID,
    }
    if name == "tadpole":
        kwargs.update(head_nodes=6, tail_nodes=6)
    elif name == "line-ring":
        kwargs.update(line_nodes=6, ring_nodes=6)
    elif name == "multi-agent":
        kwargs.update(creatures=(LineTopology(num_nodes=6), RingTopology(num_nodes=6)))
    else:
        kwargs.update(num_nodes=8)
    return make(name, **kwargs)


def _zero_action(env):
    return {
        "d_rest_length": jnp.zeros(env.num_edges),
        "d_bending_angle": jnp.zeros(env.num_bending_pairs),
    }


def test_registered_envs():
    assert set(registered_envs()) == set(ENV_NAMES + ["line-random-food"])



def test_make_unknown_raises():
    with pytest.raises(ValueError):
        make("not-an-env")


@pytest.mark.parametrize("name", ENV_NAMES)
def test_make_returns_env_with_consistent_action_spec(name):
    env = _small_env(name)
    spec = env.action_spec()
    assert spec["d_rest_length"] == (env.num_edges,)
    assert spec["d_bending_angle"] == (env.num_bending_pairs,)
    assert env.num_edges > 0
    assert env.num_bending_pairs > 0


@pytest.mark.parametrize("name", ENV_NAMES)
def test_reset_shapes(name):
    env = _small_env(name)
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert isinstance(state, EnvState)
    assert obs.ndim == 1
    assert state.nodes.position.shape == (env.num_nodes, 2)
    assert state.nodes.velocity.shape == (env.num_nodes, 2)
    assert int(state.time) == 0


@pytest.mark.parametrize("name", ENV_NAMES)
def test_step_returns_5tuple_with_shapes(name):
    env = _small_env(name)
    obs, state = env.reset(jax.random.PRNGKey(0))
    out = env.step(jax.random.PRNGKey(1), state, _zero_action(env))
    assert len(out) == 5
    obs2, state2, reward, done, info = out
    assert obs2.shape == obs.shape
    assert state2.nodes.position.shape == state.nodes.position.shape
    assert reward.shape == ()
    assert done.shape == ()
    assert done.dtype == jnp.bool_
    assert int(state2.time) == 1


@pytest.mark.parametrize("name", ENV_NAMES)
def test_step_changes_state(name):
    env = _small_env(name)
    _, state = env.reset(jax.random.PRNGKey(0))
    action = {
        "d_rest_length": jnp.full(env.num_edges, 0.05),
        "d_bending_angle": jnp.full(env.num_bending_pairs, 0.05),
    }
    _, state2, _, _, _ = env.step(jax.random.PRNGKey(1), state, action)
    # A non-zero action should move at least one node.
    assert not jnp.allclose(state2.nodes.position, state.nodes.position)


def test_jit_step():
    env = _small_env("line")
    _, state = env.reset(jax.random.PRNGKey(0))
    action = _zero_action(env)
    jit_step = jax.jit(env.step)
    obs1, state1, r1, d1, _ = jit_step(jax.random.PRNGKey(1), state, action)
    obs2, state2, r2, d2, _ = jit_step(jax.random.PRNGKey(2), state1, action)
    assert obs1.shape == obs2.shape
    assert int(state2.time) == 2


def test_vmap_batched():
    env = _small_env("line")
    batch = 4
    keys = jax.random.split(jax.random.PRNGKey(0), batch)

    obs, state = jax.vmap(env.reset)(keys)
    assert obs.shape[0] == batch
    assert state.nodes.position.shape == (batch, env.num_nodes, 2)
    assert state.time.shape == (batch,)

    batched_action = {
        "d_rest_length": jnp.zeros((batch, env.num_edges)),
        "d_bending_angle": jnp.zeros((batch, env.num_bending_pairs)),
    }
    step_keys = jax.random.split(jax.random.PRNGKey(1), batch)
    obs2, state2, reward, done, _ = jax.vmap(env.step)(step_keys, state, batched_action)
    assert obs2.shape[0] == batch
    assert reward.shape == (batch,)
    assert done.shape == (batch,)
    assert state2.nodes.position.shape == (batch, env.num_nodes, 2)


def test_line_random_food_reset_depends_on_key():
    env = make(
        "line-random-food",
        num_nodes=8,
        grid_shape=(32, 32),
        energy_size=4.0,
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    _, state0 = env.reset(jax.random.PRNGKey(0))
    _, state1 = env.reset(jax.random.PRNGKey(1))

    assert not jnp.allclose(state0.energy_center, state1.energy_center)


def test_line_random_food_vmap_reset_uses_different_energy_centers():
    env = make(
        "line-random-food",
        num_nodes=8,
        grid_shape=(32, 32),
        energy_size=4.0,
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    keys = jax.random.split(jax.random.PRNGKey(0), 8)

    _, states = jax.vmap(env.reset)(keys)
    unique_centers = jnp.unique(states.energy_center, axis=0)

    assert states.energy_center.shape == (8, 2)
    assert unique_centers.shape[0] > 1


def test_line_default_has_no_food_signal():
    env = make(
        "line",
        num_nodes=8,
        grid_shape=(32, 32),
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    obs, state = env.reset(jax.random.PRNGKey(0))
    energy_readings = obs[-env.num_nodes:]
    assert jnp.all(energy_readings == 0.0)

    _, _, reward, _, _ = env.step(jax.random.PRNGKey(1), state, env.zero_action())
    assert reward == 0.0


def test_line_bending_stiffness_is_configurable():
    env = make(
        "line",
        num_nodes=8,
        grid_shape=(32, 32),
        bending_stiffness=0.25,
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    _, state = env.reset(jax.random.PRNGKey(0))

    assert jnp.all(state.edges.bending_stiffness == 0.25)


def test_multi_agent_accepts_topology_instances():
    env = MultiAgentEnv(
        creatures=(
            LineTopology(num_nodes=5, spacing=1.5, bending_stiffness=0.25),
            RingTopology(num_nodes=6, spacing=2.5, bending_stiffness=0.75),
        ),
        grid_shape=(32, 32),
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    _, state = env.reset(jax.random.PRNGKey(0))

    assert env.num_nodes == 11
    assert state.nodes.position.shape == (11, 2)
    assert jnp.all(state.edges.rest_lengths[:4] == 1.5)
    assert jnp.all(state.edges.rest_lengths[4:] == 2.5)
    assert jnp.any(state.edges.bending_stiffness == 0.25)
    assert jnp.any(state.edges.bending_stiffness == 0.75)


def test_multi_agent_accepts_more_than_two_creatures():
    env = make(
        "multi-agent",
        creatures=(
            LineTopology(num_nodes=5),
            LineTopology(num_nodes=6),
            RingTopology(num_nodes=7),
            RingTopology(num_nodes=8),
            RingTopology(num_nodes=9),
        ),
        grid_shape=(64, 64),
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    _, state = env.reset(jax.random.PRNGKey(0))

    assert len(env.creatures) == 5
    assert env.num_nodes == 35
    assert state.nodes.position.shape == (35, 2)
    assert len(env.node_slices) == 5


def test_multi_agent_can_randomize_initial_creature_positions():
    env = make(
        "multi-agent",
        creatures=(
            LineTopology(num_nodes=5),
            RingTopology(num_nodes=6),
        ),
        random_initial_positions=True,
        position_margin=1.0,
        grid_shape=(64, 64),
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    _, state0 = env.reset(jax.random.PRNGKey(0))
    _, state1 = env.reset(jax.random.PRNGKey(1))

    assert not jnp.allclose(state0.nodes.position, state1.nodes.position)
    assert jnp.all(state0.nodes.position >= 0.0)
    assert jnp.all(state0.nodes.position <= 64.0)
    assert jnp.all(state1.nodes.position >= 0.0)
    assert jnp.all(state1.nodes.position <= 64.0)


def test_multi_agent_fitness_is_sum_of_individual_fitnesses():
    env = make(
        "multi-agent",
        creatures=(
            LineTopology(num_nodes=5),
            RingTopology(num_nodes=6),
        ),
        energy_amplitude=1.0,
        grid_shape=(64, 64),
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    _, state2, reward, _, _ = env.step(jax.random.PRNGKey(1), state, env.zero_action())

    per_creature = env.fitness_per_creature(state2)
    assert per_creature.shape == (2,)
    assert jnp.all(per_creature >= 0.0)
    assert jnp.isclose(reward, jnp.sum(per_creature))


def test_multi_agent_split_and_merge_action_supports_per_creature_mutation():
    env = make(
        "multi-agent",
        creatures=(
            LineTopology(num_nodes=5),
            RingTopology(num_nodes=6),
        ),
        grid_shape=(64, 64),
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    base_action = env.zero_action()
    per_creature_action = list(env.split_action(base_action))
    per_creature_action[0] = {
        "d_rest_length": jnp.full_like(per_creature_action[0]["d_rest_length"], 0.25),
        "d_bending_angle": jnp.full_like(
            per_creature_action[0]["d_bending_angle"], -0.1
        ),
    }

    merged = env.merge_action(per_creature_action)

    first_edges = env.edge_slices[0]
    second_edges = env.edge_slices[1]
    first_bending = env.bending_slices[0]
    second_bending = env.bending_slices[1]

    assert jnp.all(merged["d_rest_length"][first_edges] == 0.25)
    assert jnp.all(merged["d_rest_length"][second_edges] == 0.0)
    assert jnp.all(merged["d_bending_angle"][first_bending] == -0.1)
    assert jnp.all(merged["d_bending_angle"][second_bending] == 0.0)


def test_line_ring_accepts_topology_instances():
    env = make(
        "line-ring",
        line_topology=LineTopology(num_nodes=5),
        ring_topology=RingTopology(num_nodes=7),
        grid_shape=(32, 32),
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    _, state = env.reset(jax.random.PRNGKey(0))

    assert env.num_nodes == 12
    assert state.nodes.position.shape == (12, 2)


def test_line_random_food_default_has_food_signal():
    env = make(
        "line-random-food",
        num_nodes=8,
        grid_shape=(32, 32),
        energy_size=4.0,
        solver_config=PBD_SCHEME_NO_FLUID,
    )
    obs, _ = env.reset(jax.random.PRNGKey(0))
    energy_readings = obs[-env.num_nodes:]
    assert jnp.any(energy_readings > 0.0)


def test_done_after_max_steps():
    env = _small_env("line", max_steps=2)
    _, state = env.reset(jax.random.PRNGKey(0))
    action = _zero_action(env)
    _, state, _, done1, _ = env.step(jax.random.PRNGKey(1), state, action)
    assert bool(done1) is False
    _, state, _, done2, _ = env.step(jax.random.PRNGKey(2), state, action)
    assert bool(done2) is True


@pytest.mark.parametrize("name", ENV_NAMES)
def test_render_returns_rgb_array(name):
    env = _small_env(name)
    _, state = env.reset(jax.random.PRNGKey(0))
    frame = env.render(state, size=64)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == jnp.uint8


@pytest.mark.parametrize("name", ENV_NAMES)
def test_reward_is_finite_scalar(name):
    env = _small_env(name)
    _, state = env.reset(jax.random.PRNGKey(0))
    action = _zero_action(env)
    for i in range(3):
        _, state, reward, _, _ = env.step(jax.random.PRNGKey(i + 1), state, action)
        assert reward.shape == ()
        assert jnp.isfinite(reward)
