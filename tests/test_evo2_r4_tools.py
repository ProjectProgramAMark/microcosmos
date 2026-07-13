import stat
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.build_founder_bank import _publish_bank  # noqa: E402
from experiments.evo2_ecosystem.episode import _paired_r4_score  # noqa: E402
from experiments.evo2_ecosystem.heredity_adaptation_v4.analysis import (
    extract_lineage_pairs,
)  # noqa: E402
from experiments.evo2_ecosystem.heredity_adaptation_v4.manifest_generator import (
    publish_manifest_bundle,
)  # noqa: E402
from experiments.evo2_ecosystem.heredity_adaptation_v4.opportunity import (
    MUTATION_REPEATS,
    _continue_target_child,
    _inject_one_birth,
    crossfit_opportunity,
)  # noqa: E402
from experiments.evo2_ecosystem.protocol import EventKind  # noqa: E402
from microcosmos.cppn import (  # noqa: E402
    CPPNGenome,
    initialize_cppn_population,
)
from microcosmos.gym import EcosystemEnv, LineTopology  # noqa: E402
from microcosmos.heredity import fixed_r4_policy  # noqa: E402


def _selected_founders():
    population = initialize_cppn_population(16, 16)
    genomes = [
        CPPNGenome(population.node_genes[index], population.connection_genes[index])
        for index in range(16)
    ]
    return {
        "training": [(index, genomes[index]) for index in range(4)],
        "development": [(index + 4, genomes[index + 4]) for index in range(4)],
        "sealed": [(index + 8, genomes[index + 8]) for index in range(8)],
    }


def test_r4_publication_locks_holdout_founders_and_manifests(tmp_path):
    bank = tmp_path / "founders"
    _publish_bank(
        bank,
        _selected_founders(),
        "test-r4",
        lock_holdouts=True,
    )
    assert stat.S_IMODE((bank / "train-00.npz").stat().st_mode) == 0o444
    assert stat.S_IMODE((bank / "dev-00.npz").stat().st_mode) == 0o000
    assert stat.S_IMODE((bank / "sealed-00.npz").stat().st_mode) == 0o000

    manifests = tmp_path / "manifests"
    publish_manifest_bundle(bank / "index.json", 1.5, manifests)
    assert stat.S_IMODE((manifests / "training_stable.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((manifests / "training_punctuated.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((manifests / "development.json").stat().st_mode) == 0o000
    assert stat.S_IMODE((manifests / "sealed.json").stat().st_mode) == 0o000


def test_opportunity_crossfit_counts_realized_heldout_actions():
    records = []
    for founder_index in range(4):
        for context, feature in (("pre", 0.0), ("post", 1.0)):
            rewards = np.full((6, MUTATION_REPEATS), 0.20)
            rewards[1] = 0.75 if context == "pre" else 0.25
            rewards[2] = 0.25 if context == "pre" else 0.75
            records.append(
                {
                    "founder_id": f"train-{founder_index:02d}",
                    "context": context,
                    "features": [feature] + [0.0] * 28,
                    "action_scores": rewards.tolist(),
                    "resolved": np.ones((6, MUTATION_REPEATS), dtype=int).tolist(),
                }
            )
    result = crossfit_opportunity(records)
    assert result["exercised_nonclone_actions"] == [1, 2]
    assert all(
        rule["action_low"] != rule["action_high"]
        for rule in result["selected_rules"]
    )
    assert result["gates"]["two_nonclone_actions"]


def test_opportunity_injects_one_target_then_uses_normal_continuation():
    env = EcosystemEnv(
        topology=LineTopology(num_nodes=4),
        max_creatures=4,
        initial_population=2,
        grid_shape=(16, 16),
        max_steps=20,
        maturity_age=1,
        reproduction_threshold=2.0,
        reproduction_cost=1.0,
        initial_energy=3.0,
        resource_capacity=0.0,
        initial_resource=0.0,
        resource_regeneration_rate=0.0,
        basal_metabolism=0.0,
        heredity_contract="r4",
        credit_chunk_steps=2,
        offspring_policy=fixed_r4_policy(2),
    )
    root = jax.random.PRNGKey(0)
    _, state = env.reset(root)
    state = replace(
        state,
        population=replace(
            state.population,
            age=jnp.where(state.population.alive, 1, state.population.age),
        ),
    )
    injected, child_id, birth_count = jax.jit(
        lambda current, mutation, spawn: _inject_one_birth(
            env, current, mutation, spawn
        )
    )(state, jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    assert int(birth_count) == 1
    assert int(child_id) == 2
    keys = jax.random.split(jax.random.PRNGKey(3), 2)
    continued, _, success, failure = jax.jit(
        lambda current, scan_keys, target: _continue_target_child(
            env, current, scan_keys, target
        )
    )(injected, keys, child_id)
    assert int(continued.time) == 2
    assert bool(success | failure) or bool(
        jnp.any(
            continued.population.alive
            & (continued.population.individual_id == child_id)
        )
    )


def test_lineage_extraction_excludes_the_shock_ancestor_itself():
    env = EcosystemEnv(
        topology=LineTopology(num_nodes=4),
        max_creatures=4,
        initial_population=1,
        grid_shape=(16, 16),
        max_steps=2,
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    before = state.population
    before = replace(
        before,
        alive=jnp.asarray([True, False, False, False]),
        individual_id=jnp.asarray([5, -1, -1, -1], dtype=jnp.int32),
        next_individual_id=jnp.asarray(10, dtype=jnp.int32),
    )
    after = replace(
        before,
        alive=jnp.asarray([True, True, False, False]),
        individual_id=jnp.asarray([5, 10, -1, -1], dtype=jnp.int32),
        shock_ancestor_id=jnp.asarray([5, 5, -1, -1], dtype=jnp.int32),
        next_individual_id=jnp.asarray(11, dtype=jnp.int32),
    )
    world = SimpleNamespace(event_kind=EventKind.ACTUATION_COST_SHIFT, founder_id="train-00")
    episode = SimpleNamespace(shock_population=before, final_population=after)
    pairs = extract_lineage_pairs(
        SimpleNamespace(worlds=(world,)),
        SimpleNamespace(episodes=(episode,)),
    )
    assert len(pairs) == 1
    assert pairs[0].ancestor_id == 5
    assert pairs[0].descendant_id == 10


def test_paired_r4_score_uses_sham_only_for_stable_and_harmonic_for_punctuated():
    worlds = (
        SimpleNamespace(pair_id="pair", event_kind=EventKind.NULL),
        SimpleNamespace(pair_id="pair", event_kind=EventKind.ACTUATION_COST_SHIFT),
    )
    manifest = SimpleNamespace(worlds=worlds)
    candidate = SimpleNamespace(
        episodes=(SimpleNamespace(primary_score=0.8), SimpleNamespace(primary_score=0.6))
    )
    ancestor = SimpleNamespace(
        episodes=(SimpleNamespace(primary_score=0.5), SimpleNamespace(primary_score=0.3))
    )
    stable, *_ = _paired_r4_score(manifest, candidate, ancestor, "stable")
    punctuated, *_ = _paired_r4_score(manifest, candidate, ancestor, "punctuated")
    expected = 2 * 0.8 * 0.6 / 1.4 - 2 * 0.5 * 0.3 / 0.8
    assert np.isclose(stable, 0.3)
    assert np.isclose(punctuated, expected)
