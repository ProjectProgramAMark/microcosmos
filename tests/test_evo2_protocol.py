"""Exact fixtures for the immutable Evo² experiment protocol."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    CONTROLLER_LAYOUT,
    EventKind,
    ScenarioManifest,
    aggregate_candidate_score,
    apply_dominant_founder_lineage_cull,
    apply_null_event,
    apply_random_bottleneck,
    apply_resource_relocation,
    canonical_manifest_bytes,
    final_window_productivity,
    immediate_resistance,
    manifest_from_dict,
    manifest_from_json_bytes,
    manifest_sha256,
    normalized_chunk_productivity,
    post_event_productivity_auc,
    shock_null_productivity_deficit,
    sustained_recovery,
)
from microcosmos.gym.ecosystem import EcosystemEnv
from microcosmos.heredity import update_population_change_ema


def _manifest_dict(event_kind="resource_relocation", event_parameters=None):
    if event_parameters is None:
        event_parameters = {
            "center": [5.0, 7.0],
            "radius": 3.0,
            "peak_capacity": 1.0,
            "peak_regeneration": 0.02,
            "stock_fraction": 0.25,
        }
    return {
        "schema_version": 1,
        "partition": "training",
        "controller_layout": CONTROLLER_LAYOUT,
        "simulator_config_sha256": "a" * 64,
        "horizon": 100,
        "chunk_steps": 10,
        "worlds": [
            {
                "world_seed": 3,
                "scenario_id": "relocate-03",
                "pair_id": "pair-03",
                "scenario_family": "relocation",
                "event_kind": event_kind,
                "event_step": 50,
                "event_parameters": event_parameters,
            }
        ],
    }


def _state(capacity=6):
    env = EcosystemEnv(
        max_creatures=capacity,
        initial_population=capacity,
        grid_shape=(16, 16),
        max_steps=100,
        resource_patch_radius=3.0,
    )
    _, state = env.reset(jax.random.PRNGKey(11))
    return env, state


def _permute_population(state, permutation):
    capacity = state.population.alive.shape[0]

    def permute(value):
        if hasattr(value, "ndim") and value.ndim > 0 and value.shape[0] == capacity:
            return value[permutation]
        return value

    return replace(state, population=jax.tree.map(permute, state.population))


def _alive_by_identity(state):
    return {
        int(identity): bool(alive)
        for identity, alive in zip(
            state.population.individual_id,
            state.population.alive,
            strict=True,
        )
    }


def test_manifest_round_trip_has_stable_canonical_bytes_and_hash():
    manifest = manifest_from_dict(_manifest_dict())
    assert isinstance(manifest, ScenarioManifest)
    assert manifest.worlds[0].event_kind is EventKind.RESOURCE_RELOCATION

    canonical = canonical_manifest_bytes(manifest)
    reparsed = manifest_from_json_bytes(canonical)
    assert reparsed == manifest
    assert canonical_manifest_bytes(reparsed) == canonical
    assert manifest_sha256(manifest) == hashlib.sha256(canonical).hexdigest()
    assert b" " not in canonical

    differently_formatted = json.dumps(_manifest_dict(), indent=4, sort_keys=False).encode()
    assert canonical_manifest_bytes(manifest_from_json_bytes(differently_formatted)) == canonical


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update({"extra": 1}), "invalid manifest keys"),
        (
            lambda value: value["worlds"][0].update(event_step=55),
            "chunk boundary",
        ),
        (
            lambda value: value["worlds"][0]["event_parameters"].update(peak_regeneration=float("nan")),
            "peak_regeneration",
        ),
        (
            lambda value: value["worlds"][0].update(event_parameters={}),
            "event_parameters keys",
        ),
    ],
)
def test_manifest_validation_rejects_noncanonical_or_unsafe_inputs(mutate, message):
    value = _manifest_dict()
    mutate(value)
    with pytest.raises(ValueError, match=message):
        manifest_from_dict(value)


def test_null_and_resource_relocation_are_exact_boundary_transforms():
    _, state = _state(capacity=4)
    null_state, null_record = apply_null_event(state)
    assert null_state is state
    assert int(null_record.catastrophe_death_count) == 0

    relocate = jax.jit(apply_resource_relocation)
    relocated, record = relocate(
        state,
        jnp.array([2.0, 12.0]),
        jnp.array(4.0),
        jnp.array(2.0),
        jnp.array(0.05),
        jnp.array(0.25),
    )
    assert int(record.catastrophe_death_count) == 0
    assert jnp.allclose(relocated.fields.energy, 0.25 * relocated.resource_capacity_map)
    assert jnp.isclose(jnp.max(relocated.resource_capacity_map), 2.0)
    assert jnp.isclose(jnp.max(relocated.resource_regeneration_map), 0.05)
    assert jax.tree.all(
        jax.tree.map(
            lambda first, second: jnp.array_equal(first, second, equal_nan=True),
            relocated.population,
            state.population,
        )
    )
    assert jnp.array_equal(relocated.nodes.position, state.nodes.position)


def test_random_bottleneck_is_identity_keyed_and_slot_order_invariant():
    env, state = _state()
    permutation = jnp.array([4, 1, 5, 0, 3, 2])
    permuted = _permute_population(state, permutation)
    apply = jax.jit(apply_random_bottleneck)

    first, first_record = apply(
        state,
        jax.random.PRNGKey(7),
        50,
        0.5,
        env.node_slot,
    )
    second, second_record = apply(
        permuted,
        jax.random.PRNGKey(7),
        50,
        0.5,
        env.node_slot,
    )
    assert _alive_by_identity(first) == _alive_by_identity(second)
    assert first_record.catastrophe_death_count == second_record.catastrophe_death_count
    expected_ema = update_population_change_ema(
        state.population.population_change_ema,
        first_record.alive_after - first_record.alive_before,
        state.population.alive.shape[0],
    )
    assert first.population.population_change_ema == expected_ema
    assert jnp.array_equal(first.nodes.active, first.population.alive[env.node_slot])


def test_dominant_lineage_cull_has_deterministic_tie_and_exact_cap():
    env, state = _state()
    population = replace(
        state.population,
        founder_lineage_id=jnp.array([7, 7, 7, 7, 2, 2], dtype=jnp.int32),
    )
    state = replace(state, population=population)
    permutation = jnp.array([5, 2, 0, 4, 3, 1])
    permuted = _permute_population(state, permutation)
    apply = jax.jit(apply_dominant_founder_lineage_cull)

    first, first_record = apply(
        state,
        jax.random.PRNGKey(19),
        50,
        0.5,
        env.node_slot,
    )
    second, second_record = apply(
        permuted,
        jax.random.PRNGKey(19),
        50,
        0.5,
        env.node_slot,
    )
    assert int(first_record.targeted_founder_lineage) == 7
    assert int(first_record.catastrophe_death_count) == 3
    assert _alive_by_identity(first) == _alive_by_identity(second)
    assert first_record.targeted_founder_lineage == second_record.targeted_founder_lineage

    tied = replace(
        state,
        population=replace(
            state.population,
            founder_lineage_id=jnp.array([7, 7, 7, 2, 2, 2], dtype=jnp.int32),
        ),
    )
    _, tied_record = apply(
        tied,
        jax.random.PRNGKey(19),
        50,
        1.0,
        env.node_slot,
    )
    assert int(tied_record.targeted_founder_lineage) == 2


def test_absolute_productivity_and_paired_metrics_have_exact_values():
    regeneration = jnp.array([[1.0, 1.0]])
    rewards = jnp.array([1.0, 2.0, 0.0])
    normalized = normalized_chunk_productivity(
        rewards,
        chunk_steps=2,
        dt=0.5,
        post_event_regeneration_map=regeneration,
    )
    assert jnp.allclose(normalized, jnp.array([0.5, 1.0, 0.0]))
    assert post_event_productivity_auc(normalized) == 0.5
    assert aggregate_candidate_score(jnp.array([0.25, 0.75])) == 0.5

    shocked = jnp.array([0.0, 0.5, 1.0, 0.5])
    paired_null = jnp.ones(4)
    assert shock_null_productivity_deficit(shocked, paired_null) == 0.5
    assert immediate_resistance(shocked, paired_null, checkpoints=2) == -0.75
    assert final_window_productivity(shocked, checkpoints=2) == 0.75


def test_sustained_recovery_distinguishes_observed_censored_and_not_estimable():
    paired_null = jnp.ones(5)
    observed = sustained_recovery(
        jnp.array([0.0, 0.4, 0.8, 0.8, 0.8]),
        paired_null,
        threshold=0.8,
        consecutive_checkpoints=3,
        smoothing_window=1,
        reference_floor=0.1,
    )
    assert bool(observed.estimable)
    assert bool(observed.recovered)
    assert int(observed.checkpoint) == 2

    censored = sustained_recovery(
        jnp.full(5, 0.7),
        paired_null,
        threshold=0.8,
        consecutive_checkpoints=2,
        smoothing_window=1,
        reference_floor=0.1,
    )
    assert bool(censored.estimable)
    assert not bool(censored.recovered)
    assert int(censored.checkpoint) == -1

    not_estimable = sustained_recovery(
        jnp.zeros(5),
        jnp.full(5, 0.01),
        threshold=0.8,
        consecutive_checkpoints=2,
        smoothing_window=1,
        reference_floor=0.1,
    )
    assert not bool(not_estimable.estimable)
    assert not bool(not_estimable.recovered)
    assert int(not_estimable.checkpoint) == -1
