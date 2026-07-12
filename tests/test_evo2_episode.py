from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.episode import (  # noqa: E402
    EpisodeResult,
    SimulatorConfig,
    evaluate_manifest,
    run_world_scenario,
    simulator_config_sha256,
)
from experiments.evo2_ecosystem import episode as episode_module  # noqa: E402
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    CONTROLLER_LAYOUT,
    DominantLineageCullParameters,
    EventKind,
    NullEventParameters,
    RandomBottleneckParameters,
    ResourceRelocationParameters,
    ScenarioManifest,
    WorldScenario,
    aggregate_candidate_score,
    post_event_productivity_auc,
)
from microcosmos.cppn import CPPNGenome  # noqa: E402
from microcosmos.heredity import (  # noqa: E402
    CLONE,
    OffspringResult,
    clone_policy,
)


def _config() -> SimulatorConfig:
    return SimulatorConfig(
        max_creatures=4,
        initial_population=4,
        nodes_per_creature=3,
        node_spacing=1.0,
        grid_shape=(16, 16),
        dt=0.01,
        fluid_enabled=False,
        resource_capacity=1.0,
        initial_resource=1.0,
        resource_regeneration_rate=0.05,
        resource_diffusion_rate=0.0,
        resource_patch_center=(8.0, 8.0),
        resource_patch_radius=20.0,
        initial_energy=10.0,
        reproduction_threshold=100.0,
        reproduction_cost=1.0,
        maturity_age=10,
        maximum_lifespan=100,
        uptake_rate=0.2,
        basal_metabolism=0.0,
        actuation_power_coefficient=0.0,
        placement_candidates=4,
        position_margin=0.5,
    )


def _worlds() -> tuple[WorldScenario, ...]:
    common = {
        "world_seed": 17,
        "event_step": 2,
        "scenario_family": "tiny",
    }
    return (
        WorldScenario(
            **common,
            scenario_id="null",
            pair_id="null-pair",
            event_kind=EventKind.NULL,
            event_parameters=NullEventParameters(),
        ),
        WorldScenario(
            **common,
            scenario_id="relocation",
            pair_id="relocation-pair",
            event_kind=EventKind.RESOURCE_RELOCATION,
            event_parameters=ResourceRelocationParameters(
                center=(2.0, 2.0),
                radius=20.0,
                peak_capacity=1.0,
                peak_regeneration=0.05,
                stock_fraction=1.0,
            ),
        ),
        WorldScenario(
            **common,
            scenario_id="bottleneck",
            pair_id="bottleneck-pair",
            event_kind=EventKind.RANDOM_BOTTLENECK,
            event_parameters=RandomBottleneckParameters(removal_fraction=1.0),
        ),
        WorldScenario(
            **common,
            scenario_id="lineage-cull",
            pair_id="lineage-pair",
            event_kind=EventKind.DOMINANT_FOUNDER_LINEAGE_CULL,
            event_parameters=DominantLineageCullParameters(maximum_removal_fraction=0.5),
        ),
    )


def _manifest(config: SimulatorConfig) -> ScenarioManifest:
    return ScenarioManifest(
        schema_version=1,
        partition="development",
        controller_layout=CONTROLLER_LAYOUT,
        simulator_config_sha256=simulator_config_sha256(config),
        horizon=4,
        chunk_steps=2,
        worlds=_worlds(),
    )


def _assert_same_tree(first, second):
    first_leaves = jax.tree.leaves(first)
    second_leaves = jax.tree.leaves(second)
    assert len(first_leaves) == len(second_leaves)
    for left, right in zip(first_leaves, second_leaves, strict=True):
        assert jnp.array_equal(left, right, equal_nan=True)


def test_manifest_evaluation_dispatches_each_event_and_scores_exactly():
    config = _config()
    result = evaluate_manifest(_manifest(config), config, clone_policy)
    episodes = result.episodes

    assert [int(episode.event_record.event_code) for episode in episodes] == [
        0,
        1,
        2,
        3,
    ]
    assert [int(episode.event_record.catastrophe_death_count) for episode in episodes] == [
        0,
        0,
        4,
        1,
    ]
    assert [int(episode.minimum_population) for episode in episodes] == [
        4,
        4,
        0,
        3,
    ]
    assert [bool(episode.survived) for episode in episodes] == [
        True,
        True,
        False,
        True,
    ]
    assert int(episodes[3].event_record.targeted_founder_lineage) == 0
    assert jnp.array_equal(
        episodes[2].post_event_productivity,
        jnp.zeros(1),
    )
    for episode in episodes:
        assert isinstance(episode, EpisodeResult)
        assert episode.post_event_productivity.shape == (1,)
        assert episode.primary_score == post_event_productivity_auc(episode.post_event_productivity)
        assert int(episode.birth_count) == 0
        assert int(episode.natural_death_count) == 0
        assert episode.operator_counts.tolist() == [0, 0, 0, 0]
        assert bool(episode.integrity_valid)
        assert episode.shock_population is None

    scores = jnp.stack([episode.primary_score for episode in episodes])
    assert result.candidate_score == aggregate_candidate_score(scores)
    assert result.candidate_score == jnp.mean(scores)
    assert result.repeat_scores.shape == (3,)
    assert int(result.selected_repeat_index) == 1
    assert bool(result.integrity_valid)
    with pytest.raises(FrozenInstanceError):
        episodes[0].primary_score = jnp.array(0.0)


def test_episode_replay_is_deterministic_and_capture_is_explicit():
    config = _config()
    manifest = _manifest(config)
    first = evaluate_manifest(manifest, config, clone_policy, numerical_repeats=1)
    second = evaluate_manifest(manifest, config, clone_policy, numerical_repeats=1)
    _assert_same_tree(first, second)

    captured = run_world_scenario(
        config,
        clone_policy,
        manifest.worlds[0],
        horizon=manifest.horizon,
        chunk_steps=manifest.chunk_steps,
        capture_finalist=True,
    )
    assert captured.shock_population is not None
    assert captured.shock_population.genome.node_genes.shape[0] == 4
    assert int(jnp.sum(captured.shock_population.alive)) == 4


def test_paired_worlds_fork_one_exact_pre_event_state(monkeypatch):
    config = _config()
    worlds = _worlds()
    null = replace(worlds[0], pair_id="shared-pair", scenario_family="paired")
    shock = replace(worlds[2], pair_id="shared-pair", scenario_family="paired")
    manifest = replace(_manifest(config), worlds=(null, shock))
    calls = 0
    original = episode_module._prepare_pre_event

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(episode_module, "_prepare_pre_event", counted)
    result = evaluate_manifest(
        manifest,
        config,
        clone_policy,
        capture_finalists=True,
        numerical_repeats=1,
    )
    assert calls == 1
    _assert_same_tree(
        result.episodes[0].shock_population,
        result.episodes[1].shock_population,
    )


def test_manifest_rejects_a_different_frozen_simulator_config():
    config = _config()
    with pytest.raises(ValueError, match="simulator config hash"):
        evaluate_manifest(
            _manifest(config),
            replace(config, dt=0.02),
            clone_policy,
        )


def _corrupt_genome_policy(
    parent_genome,
    parent_stats,
    population_stats,
    mutation_context,
):
    del parent_stats, population_stats, mutation_context
    corrupted = CPPNGenome(
        parent_genome.node_genes.at[4, 1].set(jnp.nan),
        parent_genome.connection_genes,
    )
    return OffspringResult(
        genome=corrupted,
        policy_valid=jnp.array(True),
        operator_index=jnp.array(CLONE, dtype=jnp.int32),
    )


def test_corrupted_offspring_is_contained_and_fails_episode_integrity():
    config = replace(
        _config(),
        max_creatures=2,
        initial_population=1,
        initial_energy=5.0,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        maturity_age=0,
    )
    world = _worlds()[0]
    result = run_world_scenario(
        config,
        _corrupt_genome_policy,
        world,
        horizon=4,
        chunk_steps=2,
    )

    assert int(result.birth_count) == 1
    assert bool(result.finite)
    assert not bool(result.infrastructure_valid)
    assert not bool(result.integrity_valid)
    assert int(result.policy_violation_count) == 0
    assert bool(result.operator_accounting_valid)
