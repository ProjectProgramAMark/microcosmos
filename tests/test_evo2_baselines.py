from pathlib import Path
import sys

import jax.numpy as jnp
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.episode import (  # noqa: E402
    EpisodeResult,
    ManifestEvaluation,
    SimulatorConfig,
)
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    CONTROLLER_LAYOUT,
    EventKind,
    EventRecord,
    NullEventParameters,
    ScenarioManifest,
    WorldScenario,
    manifest_sha256,
)
from experiments.evo2_ecosystem.run_baselines import (  # noqa: E402
    BASELINE_NAMES,
    baseline_policy,
    baseline_result_record,
    write_json_atomic,
)


def _manifest(config_hash):
    return ScenarioManifest(
        schema_version=1,
        partition="training",
        controller_layout=CONTROLLER_LAYOUT,
        simulator_config_sha256=config_hash,
        horizon=20,
        chunk_steps=5,
        worlds=(
            WorldScenario(
                world_seed=3,
                scenario_id="stable-3",
                pair_id="stable-3",
                scenario_family="stable",
                event_kind=EventKind.NULL,
                event_step=10,
                event_parameters=NullEventParameters(),
            ),
        ),
    )


def _episode():
    true = jnp.asarray(True)
    return EpisodeResult(
        post_event_productivity=jnp.asarray([0.25, 0.5]),
        primary_score=jnp.asarray(0.375),
        survived=true,
        final_alive=jnp.asarray(4),
        minimum_population=jnp.asarray(2),
        maximum_generation=jnp.asarray(3),
        birth_count=jnp.asarray(5),
        natural_death_count=jnp.asarray(2),
        operator_counts=jnp.asarray([0, 0, 0, 5]),
        operator_probability_sum=jnp.asarray([0, 0, 0, 5], dtype=jnp.float32),
        pre_selection_probability_sum=jnp.zeros(4),
        post_selection_probability_sum=jnp.asarray([0, 0, 0, 5], dtype=jnp.float32),
        pre_selection_probability_count=jnp.asarray(0),
        post_selection_probability_count=jnp.asarray(5),
        resolved_success_count=jnp.zeros(6, dtype=jnp.int32),
        resolved_failure_count=jnp.zeros(6, dtype=jnp.int32),
        distinct_birth_count=jnp.asarray(0),
        resolved_distinct_success_count=jnp.zeros(6, dtype=jnp.int32),
        operator_success_ema=jnp.full(6, 0.5),
        operator_usage_ema=jnp.zeros(6),
        operator_evidence_ema=jnp.zeros(6),
        pre_event_birth_count=jnp.asarray(2, dtype=jnp.int32),
        post_event_birth_count=jnp.asarray(3, dtype=jnp.int32),
        post_event_distinct_birth_count=jnp.asarray(0),
        pre_event_resolved_count=jnp.zeros(6, dtype=jnp.int32),
        post_event_resolved_count=jnp.zeros(6, dtype=jnp.int32),
        post_event_resolved_distinct_success_count=jnp.zeros(6, dtype=jnp.int32),
        pre_event_productivity=jnp.asarray(0.1),
        finite=true,
        identity_valid=true,
        events_valid=true,
        infrastructure_valid=true,
        operator_accounting_valid=true,
        policy_violation_count=jnp.asarray(0),
        integrity_valid=true,
        event_record=EventRecord(
            event_code=jnp.asarray(0),
            alive_before=jnp.asarray(4),
            alive_after=jnp.asarray(4),
            catastrophe_death_count=jnp.asarray(0),
            targeted_founder_lineage=jnp.asarray(-1),
        ),
        shock_population=None,
        final_population=None,
    )


def test_preregistered_baselines_resolve_and_unknown_rejected():
    for name in BASELINE_NAMES:
        assert callable(baseline_policy(name))
    assert callable(baseline_policy("fixed_structural"))
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_policy("free_energy")


def test_baseline_record_is_compact_and_auditable():
    from experiments.evo2_ecosystem.episode import simulator_config_sha256
    from microcosmos.heredity import fixed_mixed_policy

    config = SimulatorConfig()
    manifest = _manifest(simulator_config_sha256(config))
    evaluation = ManifestEvaluation(
        episodes=(_episode(),),
        candidate_score=jnp.asarray(0.375),
        integrity_valid=jnp.asarray(True),
        repeat_scores=jnp.asarray([0.375]),
        selected_repeat_index=jnp.asarray(0),
    )
    record = baseline_result_record(
        manifest,
        config,
        "fixed_mixed",
        fixed_mixed_policy,
        evaluation,
    )
    assert record["manifest_sha256"] == manifest_sha256(manifest)
    assert record["candidate_score"] == pytest.approx(0.375)
    assert record["numerical_repeats"] == 1
    assert record["repeat_scores"] == [pytest.approx(0.375)]
    assert record["selected_repeat_index"] == 0
    assert record["episodes"][0]["post_event_productivity"] == [0.25, 0.5]
    assert "shock_population" not in record["episodes"][0]


def test_atomic_json_rejects_nonfinite_values(tmp_path):
    output = tmp_path / "result.json"
    with pytest.raises(ValueError):
        write_json_atomic(output, {"bad": float("nan")})
    assert not output.exists()
