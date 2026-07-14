from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.r6 import qualification  # noqa: E402
from experiments.evo2_ecosystem.r6 import protocol  # noqa: E402


def _viability(*, distinct_birth, distinct_reproducer):
    return {
        "survival": 1.0,
        "median_post_births": 5.0,
        "median_generation_gain": 2.0,
        "distinct_birth": distinct_birth,
        "distinct_reproducer": distinct_reproducer,
        "integrity": True,
    }


def _disturbance_document():
    result = {
        "manifest_sha256": "a" * 64,
        "clone_effect_mean": -0.10,
        "clone_effect_founder_bootstrap_upper_95": -0.01,
        "clone_viability": _viability(distinct_birth=0, distinct_reproducer=0),
        "standard_viability": _viability(distinct_birth=2, distinct_reproducer=1),
        "pre_resolved_by_operator": [0, 8, 8, 8, 0, 0],
        "post_resolved_by_operator": [0, 8, 8, 8, 0, 0],
        "pre_evidence_mean": [0.0, 0.2, 0.2, 0.2, 0.0, 0.0],
        "final_evidence_mean": [0.0, 0.2, 0.2, 0.2, 0.0, 0.0],
        "observable_feature_max_smd": 0.5,
        "observable_credit_tv": 0.0,
        "policy_integrity": {"clone": True, "standard": True, "exploration": True},
    }
    gates = qualification._expected_disturbance_gates(result)
    result.update(gates=gates, passed=all(gates.values()))
    return {
        "founder_index_sha256": "b" * 64,
        "manifest_sha256": "a" * 64,
        "passed": True,
        "protocol_revision": protocol.PROTOCOL_REVISION,
        "result": result,
        "schema_version": 1,
        "simulator_source_sha256": "c" * 64,
        "status": "qualified",
        "training_world_seeds": [17, 18],
        "world_qualification_sha256": "d" * 64,
    }


def test_disturbance_viability_uses_standard_directly_not_clone_selector():
    value = qualification.disturbance_qualification_from_dict(_disturbance_document())
    assert value["passed"]
    assert not qualification.viability_passes(value["result"]["clone_viability"])
    assert qualification.viability_passes(value["result"]["standard_viability"])

    corrupted = _disturbance_document()
    corrupted["result"]["standard_viability"]["distinct_birth"] = 0
    with pytest.raises(ValueError, match="gates disagree"):
        qualification.disturbance_qualification_from_dict(corrupted)


def _population(*, stressed: bool):
    alive = np.asarray([True, True, False, False])
    if stressed:
        success = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        population_change = -0.5
        energy = np.asarray([1.0, 1.0, 0.0, 0.0])
    else:
        success = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        population_change = 0.5
        energy = np.asarray([3.0, 3.0, 0.0, 0.0])
    return SimpleNamespace(
        alive=alive,
        energy=energy,
        population_change_ema=population_change,
        birth_rate_ema=0.2 if stressed else 0.8,
        death_rate_ema=0.8 if stressed else 0.2,
        intake_ema=np.asarray([0.1, 0.1, 0.0, 0.0]) if stressed else np.asarray([0.8, 0.8, 0.0, 0.0]),
        operator_success_ema=success,
        operator_evidence_ema=np.ones(6),
    )


def test_observability_is_final_snapshot_and_founder_first():
    founders = [SimpleNamespace(founder_id=f"train-{index:02d}", artifact_sha256=f"{index + 1}" * 64) for index in range(4)]
    manifest = qualification._build_manifest("training", founders, (17, 18))
    episodes = []
    for world in manifest.worlds:
        episodes.append(SimpleNamespace(final_population=_population(stressed=world.scenario_id.endswith("-shock"))))
    smd, credit_tv = qualification.matched_observability(
        manifest,
        SimpleNamespace(episodes=tuple(episodes)),
    )
    assert smd >= 0.5
    assert credit_tv == pytest.approx(1.0)


def test_imminent_state_wrapper_rejects_reset_state_for_late_window(monkeypatch):
    called = []

    def fake(*args):
        called.append(args)
        return "imminent", "final"

    monkeypatch.setattr(qualification, "_find_imminent_state", fake)
    state = SimpleNamespace(time=np.asarray(protocol.PRE_OPPORTUNITY_START))
    result = qualification._find_imminent_state_checked(
        None,
        None,
        None,
        state,
        None,
        state_step=protocol.PRE_OPPORTUNITY_START,
        start=protocol.PRE_OPPORTUNITY_START,
        stop=protocol.PRE_OPPORTUNITY_STOP,
    )
    assert result == ("imminent", "final")
    assert called

    reset = SimpleNamespace(time=np.asarray(0))
    with pytest.raises(ValueError, match="does not correspond"):
        qualification._find_imminent_state_checked(
            None,
            None,
            None,
            reset,
            None,
            state_step=protocol.PRE_OPPORTUNITY_START,
            start=protocol.PRE_OPPORTUNITY_START,
            stop=protocol.PRE_OPPORTUNITY_STOP,
        )
