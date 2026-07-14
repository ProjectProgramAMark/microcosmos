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
from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: E402
    FounderIndex,
    make_founder_record,
    write_founder_artifact,
    write_founder_index,
)
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    ActuatorInjuryParameters,
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
from microcosmos.cppn import (  # noqa: E402
    CONNECTION_WEIGHT,
    CPPNGenome,
    canonical_cppn_genome,
)
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


def _write_training_founder_bank(tmp_path):
    records = []
    for index, delta in enumerate((0.0, 0.25)):
        founder_id = f"train-{index:02d}"
        genome = canonical_cppn_genome()
        genome = CPPNGenome(
            node_genes=genome.node_genes,
            connection_genes=genome.connection_genes.at[0, CONNECTION_WEIGHT].add(delta),
        )
        artifact = f"{founder_id}.npz"
        digests = write_founder_artifact(tmp_path / artifact, genome)
        records.append(
            make_founder_record(
                founder_id=founder_id,
                partition="training",
                artifact=artifact,
                digests=digests,
                selection_rule="uninjured_viability_only",
                selection_seed=index,
            )
        )
    index_path = tmp_path / "index.json"
    write_founder_index(index_path, FounderIndex(founders=tuple(records)))
    return index_path, tuple(records)


def _schema_v2_manifest(config, records):
    worlds = []
    for founder_index, record in enumerate(records):
        common = {
            "world_seed": 30 + founder_index,
            "pair_id": f"{record.founder_id}-pair",
            "scenario_family": "paired-injury",
            "event_step": 2,
            "founder_id": record.founder_id,
            "founder_sha256": record.artifact_sha256,
        }
        worlds.extend(
            (
                WorldScenario(
                    **common,
                    scenario_id=f"{record.founder_id}-sham",
                    event_kind=EventKind.NULL,
                    event_parameters=NullEventParameters(),
                ),
                WorldScenario(
                    **common,
                    scenario_id=f"{record.founder_id}-injury",
                    event_kind=EventKind.ACTUATOR_INJURY,
                    event_parameters=ActuatorInjuryParameters((0.2,)),
                ),
            )
        )
    return ScenarioManifest(
        schema_version=2,
        partition="training",
        controller_layout=CONTROLLER_LAYOUT,
        simulator_config_sha256=simulator_config_sha256(config),
        horizon=4,
        chunk_steps=2,
        worlds=tuple(worlds),
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


def test_episode_dispatch_installs_actuator_injury_without_lifecycle_changes():
    config = _config()
    env = episode_module.build_environment(config, clone_policy, horizon=4)
    root_key = jax.random.PRNGKey(17)
    _, state = env.reset(root_key)
    gains = (0.2,) * state.actuator_gain.shape[0]
    world = WorldScenario(
        world_seed=17,
        scenario_id="injury",
        pair_id="injury-pair",
        scenario_family="actuator-injury",
        event_kind=EventKind.ACTUATOR_INJURY,
        event_step=2,
        event_parameters=ActuatorInjuryParameters(gains),
        founder_id="train-founder-00",
        founder_sha256="b" * 64,
    )

    injured, record = episode_module._apply_event(env, state, world, root_key)

    assert int(record.event_code) == 4
    assert int(record.alive_before) == config.initial_population
    assert int(record.alive_after) == config.initial_population
    assert int(record.catastrophe_death_count) == 0
    assert jnp.array_equal(injured.actuator_gain, jnp.asarray(gains, dtype=jnp.float32))
    expected_population = replace(
        state.population,
        shock_ancestor_id=jnp.where(
            state.population.alive,
            state.population.individual_id,
            -1,
        ),
    )
    _assert_same_tree(injured.population, expected_population)


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


def test_schema_v2_resolves_founders_once_and_reuses_one_compiled_environment(
    tmp_path,
    monkeypatch,
):
    config = _config()
    index_path, records = _write_training_founder_bank(tmp_path)
    manifest = _schema_v2_manifest(config, records)
    counts = {"index": 0, "artifact": 0, "environment": 0}
    original_index = episode_module.load_founder_index
    original_artifact = episode_module.load_founder_artifact
    original_environment = episode_module.build_environment

    def counted_index(*args, **kwargs):
        counts["index"] += 1
        return original_index(*args, **kwargs)

    def counted_artifact(*args, **kwargs):
        counts["artifact"] += 1
        return original_artifact(*args, **kwargs)

    def counted_environment(*args, **kwargs):
        counts["environment"] += 1
        return original_environment(*args, **kwargs)

    monkeypatch.setattr(episode_module, "load_founder_index", counted_index)
    monkeypatch.setattr(episode_module, "load_founder_artifact", counted_artifact)
    monkeypatch.setattr(episode_module, "build_environment", counted_environment)

    result = evaluate_manifest(
        manifest,
        config,
        clone_policy,
        capture_finalists=True,
        numerical_repeats=1,
        founder_index_path=index_path,
    )

    assert counts == {"index": 1, "artifact": 2, "environment": 1}
    _assert_same_tree(
        result.episodes[0].shock_population,
        result.episodes[1].shock_population,
    )
    _assert_same_tree(
        result.episodes[2].shock_population,
        result.episodes[3].shock_population,
    )
    first = result.episodes[0].shock_population.genome.connection_genes
    second = result.episodes[2].shock_population.genome.connection_genes
    assert jnp.array_equal(
        first,
        first[0][None, ...].repeat(config.max_creatures, axis=0),
        equal_nan=True,
    )
    assert jnp.array_equal(
        second,
        second[0][None, ...].repeat(config.max_creatures, axis=0),
        equal_nan=True,
    )
    assert not jnp.array_equal(first, second, equal_nan=True)


def test_schema_v2_requires_a_trusted_matching_founder_index(tmp_path):
    config = _config()
    index_path, records = _write_training_founder_bank(tmp_path)
    manifest = _schema_v2_manifest(config, records)

    with pytest.raises(ValueError, match="trusted founder_index_path"):
        evaluate_manifest(manifest, config, clone_policy, numerical_repeats=1)

    bad_worlds = tuple(replace(world, founder_sha256="c" * 64) if world.founder_id == records[0].founder_id else world for world in manifest.worlds)
    with pytest.raises(ValueError, match="manifest founder hash mismatch"):
        evaluate_manifest(
            replace(manifest, worlds=bad_worlds),
            config,
            clone_policy,
            numerical_repeats=1,
            founder_index_path=index_path,
        )

    with pytest.raises(ValueError, match="belongs to 'training', not 'development'"):
        evaluate_manifest(
            replace(manifest, partition="development"),
            config,
            clone_policy,
            numerical_repeats=1,
            founder_index_path=index_path,
        )


def test_calibration_uses_training_founders_and_pilot_rejects_an_index(tmp_path):
    config = _config()
    index_path, records = _write_training_founder_bank(tmp_path)
    manifest = _schema_v2_manifest(config, records)

    founders = episode_module._resolve_manifest_founders(
        replace(manifest, partition="calibration"),
        index_path,
    )
    assert set(founders) == {record.founder_id for record in records}

    with pytest.raises(ValueError, match="schema_version 1 must not declare"):
        evaluate_manifest(
            _manifest(config),
            config,
            clone_policy,
            numerical_repeats=1,
            founder_index_path=index_path,
        )


def test_schema_v2_rejects_paired_worlds_with_different_founders(tmp_path):
    config = _config()
    index_path, records = _write_training_founder_bank(tmp_path)
    manifest = _schema_v2_manifest(config, records)
    mismatched = replace(
        manifest.worlds[1],
        founder_id=records[1].founder_id,
        founder_sha256=records[1].artifact_sha256,
    )

    with pytest.raises(ValueError, match="paired worlds must share founder identity"):
        evaluate_manifest(
            replace(manifest, worlds=(manifest.worlds[0], mismatched)),
            config,
            clone_policy,
            numerical_repeats=1,
            founder_index_path=index_path,
        )


def test_standalone_schema_v2_world_resolves_founder_through_index(tmp_path):
    config = _config()
    index_path, records = _write_training_founder_bank(tmp_path)
    world = _schema_v2_manifest(config, records).worlds[0]

    result = run_world_scenario(
        config,
        clone_policy,
        world,
        horizon=4,
        chunk_steps=2,
        capture_finalist=True,
        founder_index_path=index_path,
        founder_partition="training",
    )

    genomes = result.shock_population.genome.connection_genes
    assert jnp.array_equal(
        genomes,
        genomes[0][None, ...].repeat(config.max_creatures, axis=0),
        equal_nan=True,
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
