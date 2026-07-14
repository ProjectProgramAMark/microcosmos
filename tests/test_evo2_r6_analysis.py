from pathlib import Path
from types import SimpleNamespace
import sys

import jax.numpy as jnp
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.r6 import analysis  # noqa: E402
from experiments.evo2_ecosystem.r6 import qualification  # noqa: E402


def _manifest():
    founders = (
        SimpleNamespace(founder_id="founder-a", artifact_sha256="a" * 64),
        SimpleNamespace(founder_id="founder-b", artifact_sha256="b" * 64),
    )
    return qualification._build_manifest("training", founders, (17, 18))


def _episode(score):
    return SimpleNamespace(primary_score=jnp.asarray(score, dtype=jnp.float32))


def _paired(candidate_scores, ancestor_scores):
    return SimpleNamespace(
        integrity_valid=jnp.asarray(True),
        episodes=tuple(_episode(score) for score in candidate_scores),
        ancestor_episodes=tuple(_episode(score) for score in ancestor_scores),
    )


def test_primary_result_uses_exact_r6_control_and_shock_roles():
    manifest = _manifest()
    # Four pairs, each serialized control then relocation.
    primary = _paired(
        (0.50, 0.60) * 4,
        (0.50, 0.55) * 4,
    )
    clone = _paired(
        (0.60, 0.40) * 4,
        (0.60, 0.40) * 4,
    )

    result = analysis.primary_result_summary(manifest, primary, clone)

    assert result["primary_relocation_auc_delta"]["mean"] == pytest.approx(0.05)
    assert result["control_absolute_auc_delta"]["mean"] == pytest.approx(0.0)
    assert result["clone_relocation_minus_control"]["mean"] == pytest.approx(-0.2)
    assert result["passed"] is True


def test_positive_primary_without_lineage_pairs_is_partial_not_failure(monkeypatch):
    monkeypatch.setattr(
        analysis,
        "primary_result_summary",
        lambda *_args: {"passed": True},
    )
    monkeypatch.setattr(analysis, "manifest_sha256", lambda _manifest: "a" * 64)
    monkeypatch.setattr(
        analysis,
        "simulator_config_sha256",
        lambda _config: "b" * 64,
    )

    result = analysis.run_final_analysis(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        lineage_pairs=(),
        adaptive_eligible=True,
    )

    assert result["mechanism"] == {
        "status": "partial_no_lineage_pairs",
        "adaptive_eligible": True,
        "claim_supported": False,
    }
