from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.episode import SimulatorConfig, build_environment  # noqa: E402
from experiments.evo2_ecosystem.protocol import EventKind  # noqa: E402
from experiments.evo2_ecosystem.r5.analysis import (  # noqa: E402
    COMMON_GARDEN_POSES,
    common_garden_mouth_position,
    prepare_common_garden_state,
    primary_result_summary,
)
from experiments.evo2_ecosystem.r5 import analysis as r5_analysis  # noqa: E402
from microcosmos.heredity import fixed_r4_policy  # noqa: E402
from microcosmos.utils import displacement  # noqa: E402


def _garden_environment():
    config = replace(SimulatorConfig(), max_creatures=1, initial_population=1)
    env = build_environment(
        config,
        fixed_r4_policy(0),
        horizon=2_000,
        heredity_contract="r4",
        credit_chunk_steps=500,
    )
    return config, env


@pytest.mark.parametrize(("name", "direction"), COMMON_GARDEN_POSES)
def test_common_garden_pose_is_rigid_accessible_and_zero_velocity(name, direction):
    del name
    config, env = _garden_environment()
    _, reset = env.reset(jax.random.PRNGKey(73))
    posed = prepare_common_garden_state(
        env,
        reset,
        config,
        multiplier=1.5,
        head_to_tail_direction=direction,
    )

    target = np.asarray(common_garden_mouth_position(config))
    assert np.asarray(posed.nodes.position[0]) == pytest.approx(target)
    assert np.asarray(posed.nodes.velocity) == pytest.approx(0.0)
    assert float(posed.actuation_cost_multiplier) == pytest.approx(1.5)

    reset_offsets = np.asarray(
        displacement(config.grid_shape, reset.nodes.position, reset.nodes.position[0])
    )
    posed_offsets = np.asarray(
        displacement(config.grid_shape, posed.nodes.position, posed.nodes.position[0])
    )
    reset_distances = np.linalg.norm(
        reset_offsets[:, None, :] - reset_offsets[None, :, :], axis=-1
    )
    posed_distances = np.linalg.norm(
        posed_offsets[:, None, :] - posed_offsets[None, :, :], axis=-1
    )
    assert posed_distances == pytest.approx(reset_distances, abs=1e-5)
    axis = posed_offsets[-1] / np.linalg.norm(posed_offsets[-1])
    assert axis == pytest.approx(direction, abs=1e-6)

    x, y = np.floor(target + 0.5).astype(int)
    capacity = float(posed.resource_capacity_map[y, x])
    assert capacity / config.resource_capacity == pytest.approx(0.75)


def _episode(score):
    return SimpleNamespace(primary_score=jnp.asarray(score, dtype=jnp.float32))


def _paired_evaluation(candidate_scores, ancestor_scores):
    return SimpleNamespace(
        integrity_valid=jnp.asarray(True),
        episodes=tuple(_episode(score) for score in candidate_scores),
        ancestor_episodes=tuple(_episode(score) for score in ancestor_scores),
    )


def test_primary_result_uses_frozen_four_gates_and_founder_first_effects():
    worlds = []
    for founder in ("founder-a", "founder-b"):
        pair = f"pair-{founder}"
        worlds.extend(
            (
                SimpleNamespace(
                    founder_id=founder,
                    pair_id=pair,
                    event_kind=EventKind.NULL,
                ),
                SimpleNamespace(
                    founder_id=founder,
                    pair_id=pair,
                    event_kind=EventKind.ACTUATION_COST_SHIFT,
                ),
            )
        )
    manifest = SimpleNamespace(worlds=tuple(worlds))
    primary = _paired_evaluation(
        (0.50, 0.60, 0.50, 0.60),
        (0.50, 0.55, 0.50, 0.55),
    )
    clone = _paired_evaluation(
        (0.60, 0.40, 0.60, 0.40),
        (0.60, 0.40, 0.60, 0.40),
    )

    result = primary_result_summary(manifest, primary, clone)

    assert result["primary_shocked_auc_delta"]["mean"] == pytest.approx(0.05)
    assert result["sham_absolute_auc_delta"]["mean"] == pytest.approx(0.0)
    assert result["clone_shock_minus_sham"]["mean"] == pytest.approx(-0.2)
    assert result["gates"] == {
        "clone_shock_harm": True,
        "minimum_primary_mean": True,
        "positive_primary_interval": True,
        "sham_noninferiority": True,
    }
    assert result["passed"] is True


def test_primary_result_fails_closed_on_integrity_failure():
    manifest = SimpleNamespace(worlds=())
    invalid = SimpleNamespace(integrity_valid=jnp.asarray(False))
    with pytest.raises(ValueError, match="integrity-valid"):
        primary_result_summary(manifest, invalid, invalid)


def test_ablation_summary_fails_closed_on_integrity_failure():
    manifest = SimpleNamespace(worlds=())
    valid = SimpleNamespace(integrity_valid=jnp.asarray(True))
    invalid = SimpleNamespace(integrity_valid=jnp.asarray(False))

    with pytest.raises(ValueError, match="integrity-valid"):
        r5_analysis._ablation_summary(manifest, valid, invalid)


def test_common_garden_score_fails_closed_on_integrity_failure(monkeypatch):
    metrics = SimpleNamespace(
        finite=jnp.asarray(True),
        identity_valid=jnp.asarray(True),
        events_valid=jnp.asarray(True),
        infrastructure_valid=jnp.asarray(False),
        operator_counts=jnp.zeros(6, dtype=jnp.int32),
        birth_count=jnp.asarray(0, dtype=jnp.int32),
        policy_violation_count=jnp.asarray(0, dtype=jnp.int32),
        cumulative_reward=jnp.asarray(1.0),
    )
    state = SimpleNamespace(resource_regeneration_map=jnp.ones((2, 2)))
    env = SimpleNamespace(
        reset=lambda *_args, **_kwargs: (None, state),
    )
    monkeypatch.setattr(
        r5_analysis,
        "prepare_common_garden_state",
        lambda *_args, **_kwargs: state,
    )

    with pytest.raises(ValueError, match="integrity invariant"):
        r5_analysis._common_garden_score(
            SimulatorConfig(),
            env,
            lambda *_args: (state, metrics),
            SimpleNamespace(),
            multiplier=1.5,
            direction=(1.0, 0.0),
        )


@pytest.mark.parametrize(
    "ablations",
    (
        {"no_credit_ablation": object()},
        {
            "no_credit_ablation": object(),
            "dominant_action_ablation": object(),
            "unexpected": object(),
        },
    ),
)
def test_mechanism_analysis_requires_exact_frozen_ablations(monkeypatch, ablations):
    monkeypatch.setattr(
        r5_analysis,
        "primary_result_summary",
        lambda *_args: {
            "passed": True,
            "primary_shocked_auc_delta": {"mean": 0.05},
        },
    )
    monkeypatch.setattr(r5_analysis, "manifest_sha256", lambda _manifest: "a" * 64)
    monkeypatch.setattr(
        r5_analysis, "simulator_config_sha256", lambda _config: "b" * 64
    )

    with pytest.raises(ValueError, match="exactly the two frozen ablations"):
        r5_analysis.run_final_analysis(
            SimpleNamespace(),
            SimpleNamespace(),
            SimpleNamespace(),
            lineage_pairs=(object(),),
            multiplier=1.5,
            ablations=ablations,
            adaptive_eligible=True,
        )
