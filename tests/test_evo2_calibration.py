from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

CALIBRATE_PATH = Path(__file__).parents[1] / "experiments" / "evo2_ecosystem" / "calibrate.py"
SPEC = importlib.util.spec_from_file_location("evo2_calibrate", CALIBRATE_PATH)
assert SPEC is not None and SPEC.loader is not None
calibrate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibrate
SPEC.loader.exec_module(calibrate)


def _world(**overrides):
    world = {
        "total_births": 4,
        "total_natural_deaths": 2,
        "assessment_births": 3,
        "maximum_generation": 3,
        "burn_in_final_alive": 5,
        "final_alive": 6,
        "assessment_capacity_fraction": 0.25,
        "finite": True,
        "identity_valid": True,
        "events_valid": True,
        "infrastructure_valid": True,
        "fixed_mixed_accounting_valid": True,
        "policy_violation_count": 0,
    }
    return {**world, **overrides}


def test_declared_manifest_uses_exact_frozen_protocol():
    path = Path(calibrate.__file__).with_name("calibration_manifest.json")
    manifest = json.loads(path.read_text())
    pending = calibrate.pending_manifest()
    assert manifest["protocol"] == pending["protocol"]
    assert manifest["protocol_hash"] == pending["protocol_hash"]
    assert manifest["status"] in {
        "pending_production_run",
        "selected",
        "no_viable_configuration",
    }
    calibrate.validate_candidate_panel(
        calibrate.CALIBRATION_CANDIDATES,
        calibrate.PRODUCTION_SPEC,
    )
    assert [candidate["candidate_id"] for candidate in calibrate.CALIBRATION_CANDIDATES] == [
        "v2_c00_balanced",
        "v2_c01_lower_birth_cost",
        "v2_c02_larger_child_endowment",
        "v2_c03_more_regeneration",
        "v2_c04_higher_density",
    ]


def test_production_refuses_cpu_and_nonfinal_shape():
    with pytest.raises(RuntimeError, match="GPU backend"):
        calibrate.validate_execution_contract(
            calibrate.PRODUCTION_SPEC,
            backend="cpu",
            smoke=False,
        )
    with pytest.raises(RuntimeError, match="requires shape"):
        calibrate.validate_execution_contract(
            replace(calibrate.PRODUCTION_SPEC, capacity=31),
            backend="gpu",
            smoke=False,
        )
    calibrate.validate_execution_contract(
        calibrate.SMOKE_SPEC,
        backend="cpu",
        smoke=True,
    )
    with pytest.raises(RuntimeError, match="fluid physics"):
        calibrate.validate_execution_contract(
            replace(calibrate.PRODUCTION_SPEC, fluid_enabled=False),
            backend="gpu",
            smoke=False,
        )


def test_gate_requires_majority_ecology_and_all_world_integrity():
    worlds = [_world() for _ in range(3)] + [
        _world(
            total_births=0,
            total_natural_deaths=0,
            assessment_births=0,
            maximum_generation=0,
            burn_in_final_alive=0,
            final_alive=0,
            assessment_capacity_fraction=1.0,
        )
        for _ in range(2)
    ]
    gate = calibrate.evaluate_gate(worlds, calibrate.PRODUCTION_SPEC)
    assert gate["passed"]
    assert gate["required_worlds"] == 3
    assert all(count == 3 for count in gate["condition_counts"].values())

    worlds[4] = _world(finite=False)
    gate = calibrate.evaluate_gate(worlds, calibrate.PRODUCTION_SPEC)
    assert not gate["passed"]
    assert not gate["integrity_all_worlds"]


def test_ordered_sweep_records_attempts_and_freezes_first_pass(
    monkeypatch,
    tmp_path,
):
    base = calibrate.SMOKE_CANDIDATES[0]["config"]
    candidates = tuple({"candidate_id": candidate_id, "config": dict(base)} for candidate_id in ("first", "second", "never_attempted"))
    monkeypatch.setattr(calibrate, "SMOKE_CANDIDATES", candidates)
    calls = []

    def runner(candidate, _spec):
        calls.append(candidate["candidate_id"])
        return {
            "status": "completed",
            "worlds": [],
            "gate": {"passed": candidate["candidate_id"] == "second"},
        }

    output = tmp_path / "calibration.json"
    manifest = calibrate.run_calibration(
        smoke=True,
        output_path=output,
        runner=runner,
    )
    assert calls == ["first", "second"]
    assert [attempt["candidate_id"] for attempt in manifest["attempts"]] == ["first", "second"]
    assert manifest["status"] == "selected"
    assert manifest["selected_candidate_id"] == "second"
    assert manifest["selected_config_hash"] == calibrate.canonical_hash(base)
    assert json.loads(output.read_text()) == manifest
