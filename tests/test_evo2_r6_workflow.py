from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.r5 import workflow as shared  # noqa: E402
from experiments.evo2_ecosystem.r6 import workflow  # noqa: E402


def _context(tmp_path: Path) -> shared.BoundedWorkflowContext:
    return shared.BoundedWorkflowContext(
        profile_name="evo2-r6-resource-relocation",
        profile_path=tmp_path / "r6.json",
        profile_hash_path=tmp_path / "r6.sha256",
        workflow_module="experiments.evo2_ecosystem.r6.workflow",
        run_id="evo2-r6-resource-relocation-20260714",
        protocol_revision="evo2-r6-resource-relocation-v1",
        schema_version=5,
        sealed_suite_schema_version=2,
        founder_index_path=tmp_path / "founders/index.json",
        manifest_root=tmp_path / "manifests",
        world_qualification_path=tmp_path / "world.json",
        disturbance_path=tmp_path / "disturbance.json",
        result_root=tmp_path / "results",
        development_complete_path=tmp_path / "results/development.complete.json",
        final_root=tmp_path / "final",
        sealed_suite_path=tmp_path / "final/suite.json",
        analysis_module="experiments.evo2_ecosystem.r6.analysis",
        baselines_module="experiments.evo2_ecosystem.r5.baselines",
    )


def test_explicit_r6_context_cannot_leak_r5_command_globals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    r5_default = shared.search_command("stable")
    monkeypatch.setattr(shared, "PROFILE_NAME", "INVALID-R5-PROFILE")
    monkeypatch.setattr(shared, "PROFILE_PATH", Path("/invalid/r5.json"))
    monkeypatch.setattr(shared, "RUN_ID", "invalid-r5-run")

    search = shared.search_command("stable", context=context)
    development = shared.development_command(
        "punctuated",
        "evaluate",
        context=context,
    )
    random = shared.structured_random_command("training", context=context)
    rendered = " ".join((*search, *development, *random))

    assert "experiments.evo2_ecosystem.r6.workflow" in rendered
    assert "evo2-r6-resource-relocation" in rendered
    assert "INVALID-R5-PROFILE" not in rendered
    assert "invalid-r5-run" not in rendered
    assert "experiments.evo2_ecosystem.r5.workflow" in " ".join(r5_default)
    assert "evo2-r5-world-feasibility" in " ".join(r5_default)


def test_schema_v2_disturbance_record_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qualification = tmp_path / "disturbance.json"
    qualification.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(workflow, "DISTURBANCE_QUALIFICATION_PATH", qualification)
    value = {
        "event_kind": "resource_relocation",
        "control": {
            "center": [48.0, 32.0],
            "radius": 12.0,
            "peak_capacity": 1.0,
            "peak_regeneration": 0.03,
            "stock_fraction": 1.0,
        },
        "shock": {
            "center": [16.0, 32.0],
            "radius": 12.0,
            "peak_capacity": 1.0,
            "peak_regeneration": 0.03,
            "stock_fraction": 1.0,
        },
        "qualification": {
            "path": str(qualification.resolve()),
            "sha256": workflow._sha256_file(qualification),
        },
    }

    workflow._validate_disturbance_record(value)
    changed = copy.deepcopy(value)
    changed["shock"]["center"] = [15.0, 32.0]
    with pytest.raises(ValueError, match="parameters"):
        workflow._validate_disturbance_record(changed)
    changed = copy.deepcopy(value)
    changed["event_kind"] = "actuation_cost_shift"
    with pytest.raises(ValueError, match="parameters"):
        workflow._validate_disturbance_record(changed)
    changed = copy.deepcopy(value)
    changed["qualification"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="qualification"):
        workflow._validate_disturbance_record(changed)


def test_r6_adaptive_finalist_cannot_replace_confirmatory_primary(
    tmp_path: Path,
) -> None:
    roots = {regime: SimpleNamespace(frozen=tmp_path / regime) for regime in ("stable", "punctuated")}
    module = SimpleNamespace(
        paths_for=lambda spec, regime: roots[regime],
        structured_random_paths=lambda spec: SimpleNamespace(frozen=tmp_path / "structured"),
    )
    records = {
        "stable_unrestricted": {"candidate_source_sha256": "1" * 64},
        "punctuated_unrestricted": {"candidate_source_sha256": "2" * 64},
        "punctuated_adaptive": {"candidate_source_sha256": "3" * 64},
    }

    _, finalists, primary, adaptive, eligible = shared._development_finalist_sources(
        module,
        {},
        records,
        schema_version=5,
    )

    assert primary == "punctuated_unrestricted"
    assert adaptive == "punctuated_adaptive"
    assert finalists[adaptive] == tmp_path / "punctuated/adaptive/main.py"
    assert adaptive in eligible
    assert primary not in eligible


def test_r6_positive_primary_gets_common_garden_without_adaptive_redefinition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = {
        "schema_version": 2,
        "mechanism_decision": {"result_name": "decision.json"},
        "primary_policy": "punctuated_unrestricted",
        "adaptive_policy": "punctuated_adaptive",
        "work": [
            {
                "work_id": "punctuated_unrestricted",
                "adaptive_eligible": False,
            }
        ],
    }
    evaluation = SimpleNamespace()

    def load_result(path, label):
        del path
        lineage = (object(),) if label == "punctuated_unrestricted" else ()
        return evaluation, lineage, {"evaluation": {"episodes": [{"operator_counts": [0] * 6}]}}

    monkeypatch.setattr(shared, "_load_sealed_suite", lambda path: (suite, "a" * 64))
    monkeypatch.setattr(shared, "load_sealed_policy_result", load_result)
    monkeypatch.setattr(shared, "_load_bound_sealed_manifest", lambda value: object())
    monkeypatch.setattr(
        shared,
        "_sealed_analysis_module",
        lambda value: SimpleNamespace(primary_result_summary=lambda *args: {"passed": True}),
    )
    monkeypatch.setattr(shared, "_sealed_shock_indices", lambda *args: {0})

    decision = shared.commit_mechanism_decision(tmp_path / "suite.json")

    assert decision["run_common_garden"] is True
    assert decision["run_ablations"] is False
    assert decision["run_mechanism"] is False
    assert decision["status"] == "common_garden_only_not_adaptive_eligible"
