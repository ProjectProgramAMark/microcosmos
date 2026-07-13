from dataclasses import replace
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.analysis import (  # noqa: E402
    FounderWorldObservation,
    hierarchical_paired_effect_summary,
    paired_effect_summary,
    paired_world_metrics,
    stratified_equal_weight_summary,
)
from experiments.evo2_ecosystem.episode import (  # noqa: E402
    EpisodeResult,
    ManifestEvaluation,
)
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    CONTROLLER_LAYOUT,
    EventKind,
    EventRecord,
    NullEventParameters,
    RandomBottleneckParameters,
    ScenarioManifest,
    WorldScenario,
)


def _episode(productivity, *, survived=True):
    return EpisodeResult(
        post_event_productivity=jnp.asarray(productivity, dtype=jnp.float32),
        primary_score=jnp.mean(jnp.asarray(productivity, dtype=jnp.float32)),
        survived=jnp.asarray(survived),
        final_alive=jnp.asarray(3 if survived else 0, dtype=jnp.int32),
        minimum_population=jnp.asarray(1 if survived else 0, dtype=jnp.int32),
        maximum_generation=jnp.asarray(2, dtype=jnp.int32),
        birth_count=jnp.asarray(4, dtype=jnp.int32),
        natural_death_count=jnp.asarray(2, dtype=jnp.int32),
        operator_counts=jnp.asarray([0, 0, 0, 4], dtype=jnp.int32),
        operator_probability_sum=jnp.asarray([0, 0, 0, 4], dtype=jnp.float32),
        pre_selection_probability_sum=jnp.zeros(4),
        post_selection_probability_sum=jnp.asarray([0, 0, 0, 4], dtype=jnp.float32),
        pre_selection_probability_count=jnp.asarray(0),
        post_selection_probability_count=jnp.asarray(4),
        resolved_success_count=jnp.zeros(6, dtype=jnp.int32),
        resolved_failure_count=jnp.zeros(6, dtype=jnp.int32),
        distinct_birth_count=jnp.asarray(0),
        resolved_distinct_success_count=jnp.zeros(6, dtype=jnp.int32),
        operator_success_ema=jnp.full(6, 0.5),
        operator_usage_ema=jnp.zeros(6),
        operator_evidence_ema=jnp.zeros(6),
        pre_event_birth_count=jnp.asarray(2, dtype=jnp.int32),
        post_event_birth_count=jnp.asarray(2, dtype=jnp.int32),
        post_event_distinct_birth_count=jnp.asarray(0),
        pre_event_resolved_count=jnp.zeros(6, dtype=jnp.int32),
        post_event_resolved_count=jnp.zeros(6, dtype=jnp.int32),
        post_event_resolved_distinct_success_count=jnp.zeros(6, dtype=jnp.int32),
        pre_event_productivity=jnp.asarray(0.1),
        finite=jnp.asarray(True),
        identity_valid=jnp.asarray(True),
        events_valid=jnp.asarray(True),
        infrastructure_valid=jnp.asarray(True),
        operator_accounting_valid=jnp.asarray(True),
        policy_violation_count=jnp.asarray(0, dtype=jnp.int32),
        integrity_valid=jnp.asarray(True),
        event_record=EventRecord(
            event_code=jnp.asarray(0, dtype=jnp.int32),
            alive_before=jnp.asarray(4, dtype=jnp.int32),
            alive_after=jnp.asarray(4, dtype=jnp.int32),
            catastrophe_death_count=jnp.asarray(0, dtype=jnp.int32),
            targeted_founder_lineage=jnp.asarray(-1, dtype=jnp.int32),
        ),
        shock_population=None,
        final_population=None,
    )


def _manifest():
    common = dict(
        world_seed=7,
        pair_id="pair-7",
        scenario_family="bottleneck",
        event_step=10,
    )
    return ScenarioManifest(
        schema_version=1,
        partition="sealed_final",
        controller_layout=CONTROLLER_LAYOUT,
        simulator_config_sha256="0" * 64,
        horizon=20,
        chunk_steps=5,
        worlds=(
            WorldScenario(
                scenario_id="null-7",
                event_kind=EventKind.NULL,
                event_parameters=NullEventParameters(),
                **common,
            ),
            WorldScenario(
                scenario_id="shock-7",
                event_kind=EventKind.RANDOM_BOTTLENECK,
                event_parameters=RandomBottleneckParameters(0.5),
                **common,
            ),
        ),
    )


def test_paired_world_metrics_use_unique_seed_matched_null():
    null = _episode([0.5, 0.5, 0.5, 0.5])
    shock = _episode([0.1, 0.3, 0.5, 0.5])
    evaluation = ManifestEvaluation(
        episodes=(null, shock),
        candidate_score=jnp.asarray(0.4),
        integrity_valid=jnp.asarray(True),
        repeat_scores=jnp.asarray([0.4]),
        selected_repeat_index=jnp.asarray(0),
    )
    result = paired_world_metrics(_manifest(), evaluation)[0]
    assert result.primary_effect == pytest.approx(-0.15)
    assert result.shock_null_deficit == pytest.approx(0.15)
    assert result.immediate_resistance == pytest.approx(-0.3)
    assert result.final_productivity_effect == pytest.approx(0.0)
    assert result.recovery_estimable
    assert result.recovered


def test_paired_world_metrics_reject_mismatched_pair_seed():
    manifest = _manifest()
    mismatched = replace(
        manifest,
        worlds=(manifest.worlds[0], replace(manifest.worlds[1], world_seed=8)),
    )
    evaluation = ManifestEvaluation(
        episodes=(_episode([0.5] * 4), _episode([0.5] * 4)),
        candidate_score=jnp.asarray(0.5),
        integrity_valid=jnp.asarray(True),
        repeat_scores=jnp.asarray([0.5]),
        selected_repeat_index=jnp.asarray(0),
    )
    with pytest.raises(ValueError, match="share one world seed"):
        paired_world_metrics(mismatched, evaluation)


def test_paired_bootstrap_resamples_pairs():
    first = paired_effect_summary([1.0, 2.0, 3.0], seed=17, replicates=1_000)
    second = paired_effect_summary([1.0, 2.0, 3.0], seed=17, replicates=1_000)
    assert first == second
    assert first.mean == 2.0
    assert first.median == 2.0
    assert first.fraction_positive == 1.0
    assert first.confidence_interval[0] >= 1.0
    assert first.confidence_interval[1] <= 3.0


def test_stratified_summary_weights_families_equally():
    summary = stratified_equal_weight_summary(
        {"large": np.full(10, 1.0), "small": np.asarray([-1.0])},
        seed=3,
        replicates=100,
    )
    assert summary.count == 11
    assert summary.mean == pytest.approx(0.0)
    assert summary.confidence_interval == pytest.approx((0.0, 0.0))


def _founder_observation(founder_id, world_seed, pair_id, value):
    return FounderWorldObservation(
        founder_id=founder_id,
        world_seed=world_seed,
        pair_id=pair_id,
        value=value,
    )


def test_hierarchical_paired_bootstrap_is_deterministic_and_founder_equal():
    candidate = (
        _founder_observation("founder-a", 1, "injury-1", 2.0),
        _founder_observation("founder-a", 2, "injury-2", 2.0),
        _founder_observation("founder-a", 3, "injury-3", 2.0),
        _founder_observation("founder-b", 1, "injury-1", -2.0),
    )
    comparator = tuple(_founder_observation(item.founder_id, item.world_seed, item.pair_id, 0.0) for item in reversed(candidate))

    first = hierarchical_paired_effect_summary(candidate, comparator, replicates=1_000)
    second = hierarchical_paired_effect_summary(candidate, comparator, replicates=1_000)

    assert first == second
    assert first.count == 4
    assert first.founder_count == 2
    assert first.mean == pytest.approx(0.0)
    assert first.median == pytest.approx(0.0)
    assert first.fraction_positive == pytest.approx(0.5)
    assert first.per_founder_effects[0].founder_id == "founder-a"
    assert first.per_founder_effects[0].count == 3
    assert first.per_founder_effects[0].mean == pytest.approx(2.0)
    assert first.per_founder_effects[1].founder_id == "founder-b"
    assert first.per_founder_effects[1].count == 1
    assert first.per_founder_effects[1].mean == pytest.approx(-2.0)
    assert first.confidence_interval[0] == pytest.approx(-2.0)
    assert first.confidence_interval[1] == pytest.approx(2.0)


def test_hierarchical_paired_bootstrap_rejects_unmatched_or_duplicate_keys():
    candidate = (_founder_observation("founder-a", 1, "injury", 1.0),)
    comparator = (_founder_observation("founder-a", 2, "injury", 0.0),)
    with pytest.raises(ValueError, match="identical pairing keys"):
        hierarchical_paired_effect_summary(candidate, comparator, replicates=10)

    duplicate = (candidate[0], candidate[0])
    with pytest.raises(ValueError, match="duplicate candidate pairing key"):
        hierarchical_paired_effect_summary(duplicate, candidate, replicates=10)


def test_hierarchical_paired_bootstrap_resamples_observations_within_founder():
    candidate = (
        _founder_observation("founder-a", 1, "injury-1", 0.0),
        _founder_observation("founder-a", 2, "injury-2", 2.0),
    )
    comparator = tuple(_founder_observation(item.founder_id, item.world_seed, item.pair_id, 0.0) for item in candidate)

    summary = hierarchical_paired_effect_summary(
        candidate,
        comparator,
        seed=5,
        replicates=1_000,
    )

    assert summary.mean == pytest.approx(1.0)
    assert summary.confidence_interval == pytest.approx((0.0, 2.0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"founder_id": "", "world_seed": 1, "pair_id": "injury", "value": 1.0}, "founder_id"),
        ({"founder_id": "founder", "world_seed": -1, "pair_id": "injury", "value": 1.0}, "world_seed"),
        ({"founder_id": "founder", "world_seed": 1, "pair_id": "", "value": 1.0}, "pair_id"),
        ({"founder_id": "founder", "world_seed": 1, "pair_id": "injury", "value": np.nan}, "value"),
        ({"founder_id": "founder", "world_seed": 1, "pair_id": "injury", "value": True}, "value"),
        ({"founder_id": "founder", "world_seed": 1, "pair_id": "injury", "value": 1.0j}, "value"),
    ],
)
def test_founder_world_observation_rejects_invalid_identity_or_value(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FounderWorldObservation(**kwargs)
