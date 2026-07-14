from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem import episode as episode_module  # noqa: E402
from experiments.evo2_ecosystem.episode import SimulatorConfig  # noqa: E402
from experiments.evo2_ecosystem.episode import simulator_config_sha256  # noqa: E402
from experiments.evo2_ecosystem.protocol import CONTROLLER_LAYOUT  # noqa: E402
from experiments.evo2_ecosystem.protocol import EventKind  # noqa: E402
from experiments.evo2_ecosystem.protocol import ResourceRelocationParameters  # noqa: E402
from experiments.evo2_ecosystem.protocol import ScenarioManifest  # noqa: E402
from experiments.evo2_ecosystem.protocol import WorldScenario  # noqa: E402
from experiments.evo2_ecosystem.r6 import protocol  # noqa: E402
from microcosmos.heredity import clone_policy  # noqa: E402


def _r6_pair(*, pair_id="pair", control=None, shock=None):
    common = dict(
        world_seed=17,
        pair_id=pair_id,
        scenario_family=protocol.SCENARIO_FAMILY,
        event_kind=EventKind.RESOURCE_RELOCATION,
        event_step=protocol.EVENT_STEP,
        founder_id="train-00",
        founder_sha256="a" * 64,
    )
    return (
        WorldScenario(
            **common,
            scenario_id=f"{pair_id}-control",
            event_parameters=control or protocol.CONTROL_PARAMETERS,
        ),
        WorldScenario(
            **common,
            scenario_id=f"{pair_id}-shock",
            event_parameters=shock or protocol.SHOCK_PARAMETERS,
        ),
    )


def _manifest(worlds):
    return ScenarioManifest(
        schema_version=3,
        partition="training",
        controller_layout=CONTROLLER_LAYOUT,
        simulator_config_sha256=simulator_config_sha256(SimulatorConfig()),
        horizon=protocol.HORIZON,
        chunk_steps=protocol.CHUNK_STEPS,
        worlds=tuple(worlds),
    )


def test_r6_protocol_freezes_world_founder_timing_and_resource_contracts():
    assert [(item.partition, item.start, item.stop) for item in protocol.WORLD_SEED_RANGES] == [
        ("founder_eligibility", 12_000, 13_000),
        ("training", 13_000, 14_000),
        ("development", 14_000, 15_000),
        ("sealed", 15_000, 16_000),
    ]
    assert [(item.partition, item.start, item.stop, item.required) for item in protocol.FOUNDER_CANDIDATE_RANGES] == [
        ("training", 0, 24, 4),
        ("development", 24, 48, 4),
        ("sealed", 48, 80, 8),
    ]
    assert (protocol.HORIZON, protocol.CHUNK_STEPS, protocol.EVENT_STEP) == (
        12_000,
        500,
        7_500,
    )
    assert protocol.CONTROL_PARAMETERS.center == (48.0, 32.0)
    assert protocol.SHOCK_PARAMETERS.center == (16.0, 32.0)
    assert replace(protocol.CONTROL_PARAMETERS, center=(16.0, 32.0)) == protocol.SHOCK_PARAMETERS


def test_r6_same_kind_pair_resolves_only_exact_roles_and_parameters():
    manifest = _manifest(_r6_pair())
    assert episode_module._resolve_pair_indices(manifest, [0, 1]) == (0, 1)
    assert episode_module._resolve_pair_indices(manifest, [1, 0]) == (0, 1)

    bad_id = replace(manifest.worlds[0], scenario_id="pair-sham")
    with pytest.raises(ValueError, match="exact -control and -shock"):
        episode_module._resolve_pair_indices(replace(manifest, worlds=(bad_id, manifest.worlds[1])), [0, 1])

    wrong_center = replace(
        protocol.CONTROL_PARAMETERS,
        center=(47.0, 32.0),
    )
    malformed = _manifest(_r6_pair(control=wrong_center))
    with pytest.raises(ValueError, match="frozen resource events"):
        episode_module._resolve_pair_indices(malformed, [0, 1])


def test_nonnull_resource_event_marks_only_living_event_time_ancestors():
    config = SimulatorConfig(
        max_creatures=4,
        initial_population=2,
        nodes_per_creature=3,
        grid_shape=(16, 16),
        fluid_enabled=False,
        resource_patch_center=(8.0, 8.0),
        resource_patch_radius=4.0,
    )
    env = episode_module.build_environment(config, clone_policy, horizon=4)
    root_key = jax.random.PRNGKey(17)
    _, state = env.reset(root_key)
    world = WorldScenario(
        world_seed=17,
        scenario_id="relocation",
        pair_id="relocation-pair",
        scenario_family="test",
        event_kind=EventKind.RESOURCE_RELOCATION,
        event_step=2,
        event_parameters=ResourceRelocationParameters(
            center=(4.0, 4.0),
            radius=4.0,
            peak_capacity=1.0,
            peak_regeneration=0.03,
            stock_fraction=1.0,
        ),
    )

    changed, _ = episode_module._apply_event(env, state, world, root_key)

    expected = jnp.where(state.population.alive, state.population.individual_id, -1)
    assert jnp.array_equal(changed.population.shock_ancestor_id, expected)
    assert jnp.array_equal(changed.population.alive, state.population.alive)
    assert jnp.array_equal(
        changed.population.genome.node_genes,
        state.population.genome.node_genes,
        equal_nan=True,
    )
