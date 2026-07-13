from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path
import sys

import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.calibrate_actuator import (  # noqa: E402
    POLICIES,
    PRIMARY_SCHEDULE_ID,
    PRODUCTION_SPEC,
    CalibrationWorldResult,
    PolicyCalibrationResult,
    build_calibration_manifest,
)
from experiments.evo2_ecosystem.confirm_actuator_development import (  # noqa: E402
    CONFIRMATION_DECISION_SCHEMA_VERSION,
    CONFIRMATION_RECORD_SCHEMA_VERSION,
    _run_development_confirmation,
    evaluate_development_gate,
    run_development_confirmation,
    validate_development_manifest,
)
from experiments.evo2_ecosystem.episode import (  # noqa: E402
    SimulatorConfig,
    simulator_config_sha256,
    simulator_source_sha256,
)
from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: E402
    FounderIndex,
    FounderRecord,
    founder_index_sha256,
    write_founder_index,
)
from experiments.evo2_ecosystem.heredity_adaptation.manifest_generator import (  # noqa: E402
    _build_manifest_bundle,
)
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    EventKind,
    canonical_manifest_bytes,
    manifest_sha256,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _founder_record(founder_id: str, partition: str, ordinal: int) -> FounderRecord:
    digest = f"{ordinal:064x}"
    return FounderRecord(
        founder_id=founder_id,
        partition=partition,
        artifact=f"{founder_id}.npz",
        artifact_sha256=digest,
        node_genes_sha256=f"{ordinal + 100:064x}",
        connection_genes_sha256=f"{ordinal + 200:064x}",
        controller_layout="cppn-4x1-15n-30c-v1",
        selection_rule="test-selection",
        selection_seed=ordinal,
    )


def _founder_index(tmp_path: Path) -> tuple[Path, FounderIndex]:
    records = (
        *(_founder_record(f"training-{index}", "training", index + 1) for index in range(3)),
        *(_founder_record(f"development-{index}", "development", index + 11) for index in range(2)),
        # These sealed artifacts deliberately do not exist.  The one-shot
        # confirmation must never try to open them.
        *(_founder_record(f"sealed-{index}", "sealed", index + 21) for index in range(3)),
    )
    index = FounderIndex(founders=records)
    path = tmp_path / "founders" / "index.json"
    write_founder_index(path, index)
    return path, index


def _write_manifest(path: Path, manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_manifest_bytes(manifest) + b"\n")
    digest = manifest_sha256(manifest)
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )


def _calibration_inputs(tmp_path: Path):
    founder_index_path, index = _founder_index(tmp_path)
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("Status: **FROZEN**\n", encoding="utf-8")

    training = tuple(record for record in index.founders if record.partition == "training")
    selected_manifest = build_calibration_manifest(
        PRODUCTION_SPEC,
        training,
        gain=0.2,
        schedule_id=PRIMARY_SCHEDULE_ID,
    )
    selected_manifest_path = tmp_path / "selected-calibration.json"
    _write_manifest(selected_manifest_path, selected_manifest)

    policies = {name: {"source_sha256": _sha256(inspect.getsource(POLICIES[name][0]).encode("utf-8"))} for name in ("clone", "fixed_parametric")}
    protocol = {
        "backend": "gpu",
        "founder_index_sha256": founder_index_sha256(index),
        "mode": "production",
        "policies": policies,
        "preregistration_sha256": _sha256(preregistration.read_bytes()),
        "simulator_config_sha256": simulator_config_sha256(SimulatorConfig()),
        "simulator_source_sha256": simulator_source_sha256(),
        "spec": json.loads(
            _canonical(
                {
                    **asdict(PRODUCTION_SPEC),
                    "config": asdict(PRODUCTION_SPEC.config),
                }
            )
        ),
    }
    protocol_sha256 = _sha256(_canonical(protocol))
    attempt = {
        "gain": 0.2,
        "manifest_sha256": manifest_sha256(selected_manifest),
        "schedule_id": PRIMARY_SCHEDULE_ID,
    }
    selected = {
        "gain": 0.2,
        "manifest_path": str(selected_manifest_path),
        "manifest_sha256": manifest_sha256(selected_manifest),
        "schedule_id": PRIMARY_SCHEDULE_ID,
        "winning_policy": "fixed_parametric",
    }
    events = (
        {
            "event": "calibration_started",
            "protocol": protocol,
            "protocol_sha256": protocol_sha256,
        },
        {
            "attempt": attempt,
            "event": "gain_evaluated",
            "gate": {"passed": True, "winning_policy": "fixed_parametric"},
            "protocol_sha256": protocol_sha256,
        },
        {
            "event": "calibration_finished",
            "protocol_sha256": protocol_sha256,
            "selected": selected,
            "selected_manifest_written": True,
            "status": "selected",
        },
    )
    calibration_record = tmp_path / "calibration.jsonl"
    calibration_record.write_bytes(b"".join(_canonical({"schema_version": 1, "sequence": index, **event}) + b"\n" for index, event in enumerate(events)))

    development = _build_manifest_bundle(index, 0.2).development
    development_path = tmp_path / "development.json"
    _write_manifest(development_path, development)
    return {
        "calibration_record_path": calibration_record,
        "development_manifest": development,
        "development_manifest_path": development_path,
        "founder_index": index,
        "founder_index_path": founder_index_path,
        "preregistration_path": preregistration,
        "selected_calibration_manifest_path": selected_manifest_path,
    }


def _policy_result(
    manifest,
    policy_name,
    *,
    mutation_score=0.4,
    births=5,
    generation_gain=2,
    integrity=True,
):
    is_clone = policy_name == "clone"
    worlds = []
    for world in manifest.worlds:
        injury = world.event_kind is EventKind.ACTUATOR_INJURY
        score = (0.2 if is_clone else mutation_score) if injury else 0.5
        operator_counts = [0, 0, 0, 0]
        operator_counts[0 if is_clone else 1] = births
        worlds.append(
            CalibrationWorldResult(
                policy_name=policy_name,
                founder_id=world.founder_id,
                founder_sha256=world.founder_sha256,
                world_seed=world.world_seed,
                pair_id=world.pair_id,
                event_step=world.event_step,
                arm="injury" if injury else "sham",
                primary_score=score,
                survived=True,
                post_event_births=births,
                event_generation=1,
                maximum_generation=1 + generation_gain,
                generation_gain=generation_gain,
                operator_counts=tuple(operator_counts),
                integrity_valid=integrity,
                clone_identity_valid=(True if is_clone else None),
            )
        )
    return PolicyCalibrationResult(
        policy_name=policy_name,
        worlds=tuple(worlds),
        repeat_scores=(0.1, 0.2, 0.3),
        selected_repeat_index=1,
        all_repeats_integrity_valid=integrity,
        clone_identity_all_repeats=(True if is_clone else None),
    )


def test_one_shot_driver_uses_only_clone_and_preselected_winner_and_publishes_once(
    tmp_path,
):
    inputs = _calibration_inputs(tmp_path)
    calls = []

    def runner(manifest, policy_name):
        calls.append(policy_name)
        return _policy_result(manifest, policy_name)

    result_path = tmp_path / "development-confirmation.test.jsonl"
    decision_path = tmp_path / "development-decision.test.json"
    result = _run_development_confirmation(
        **{key: value for key, value in inputs.items() if key.endswith("_path") and key != "development_manifest"},
        result_record_path=result_path,
        decision_path=decision_path,
        policy_runner=runner,
    )

    assert result["passed"] is True
    assert result["status"] == "test_only_passed"
    assert result["scientific_evidence"] is False
    assert result["winning_policy"] == "fixed_parametric"
    assert calls == ["clone", "fixed_parametric"]
    events = [json.loads(line) for line in result_path.read_bytes().splitlines()]
    assert [event["sequence"] for event in events] == [0, 1, 2, 3]
    assert all(_canonical(event) == line for event, line in zip(events, result_path.read_bytes().splitlines(), strict=True))
    assert events[0]["schema_version"] == CONFIRMATION_RECORD_SCHEMA_VERSION
    assert events[-1]["decision"]["passed"] is True
    decision = json.loads(decision_path.read_text())
    assert decision["schema_version"] == CONFIRMATION_DECISION_SCHEMA_VERSION
    assert decision["status"] == "test_only_passed"
    assert decision["evidence_class"] == "test_only_non_scientific"
    assert decision["scientific_evidence"] is False
    for path in (result_path, decision_path):
        digest = _sha256(path.read_bytes())
        assert path.with_suffix(".sha256").read_text() == f"{digest}  {path.name}\n"

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _run_development_confirmation(
            **{key: value for key, value in inputs.items() if key.endswith("_path") and key != "development_manifest"},
            result_record_path=result_path,
            decision_path=tmp_path / "second-decision.test.json",
            policy_runner=runner,
        )


def test_gate_failure_is_published_as_stop_without_retuning(tmp_path):
    inputs = _calibration_inputs(tmp_path)

    result = _run_development_confirmation(
        **{key: value for key, value in inputs.items() if key.endswith("_path") and key != "development_manifest"},
        result_record_path=tmp_path / "failed.test.jsonl",
        decision_path=tmp_path / "failed-decision.test.json",
        policy_runner=lambda manifest, name: _policy_result(
            manifest,
            name,
            mutation_score=0.1,
        ),
    )
    assert result["passed"] is False
    assert result["status"] == "test_only_failed_transfer"
    decision = json.loads((tmp_path / "failed-decision.test.json").read_text())
    assert decision["decision"]["mutation_advantage"]["confidence_interval"][1] < 0


def test_development_validation_rejects_gain_or_spatial_retuning(tmp_path):
    inputs = _calibration_inputs(tmp_path)
    manifest = inputs["development_manifest"]
    validate_development_manifest(manifest, inputs["founder_index"], gain=0.2)
    with pytest.raises(ValueError, match="treatment panel"):
        validate_development_manifest(manifest, inputs["founder_index"], gain=0.4)

    injury_index = next(index for index, world in enumerate(manifest.worlds) if world.event_kind is EventKind.ACTUATOR_INJURY)
    world = manifest.worlds[injury_index]
    changed = list(manifest.worlds)
    changed[injury_index] = replace(world, event_step=world.event_step + 500)
    with pytest.raises(ValueError, match="exact pre-event identity|treatment panel"):
        validate_development_manifest(
            replace(manifest, worlds=tuple(changed)),
            inputs["founder_index"],
            gain=0.2,
        )


def test_production_contract_rejects_cpu_and_injected_results_are_test_only(tmp_path):
    inputs = _calibration_inputs(tmp_path)
    with pytest.raises(RuntimeError, match="requires GPU"):
        run_development_confirmation(
            **{key: value for key, value in inputs.items() if key.endswith("_path") and key != "development_manifest"},
            result_record_path=tmp_path / "result.jsonl",
            decision_path=tmp_path / "decision.json",
        )
    with pytest.raises(RuntimeError, match=".test.jsonl/.test.json"):
        _run_development_confirmation(
            **{key: value for key, value in inputs.items() if key.endswith("_path") and key != "development_manifest"},
            result_record_path=tmp_path / "result.jsonl",
            decision_path=tmp_path / "decision.json",
            policy_runner=lambda manifest, name: _policy_result(manifest, name),
        )


def test_direct_gate_requires_integrity_births_generation_and_clone_identity(tmp_path):
    inputs = _calibration_inputs(tmp_path)
    manifest = inputs["development_manifest"]
    clone = _policy_result(manifest, "clone")
    mutation = _policy_result(manifest, "fixed_parametric")
    assert evaluate_development_gate(clone, mutation).passed
    assert not evaluate_development_gate(
        clone,
        _policy_result(manifest, "fixed_parametric", births=3),
    ).passed
    assert not evaluate_development_gate(
        clone,
        _policy_result(manifest, "fixed_parametric", generation_gain=1),
    ).passed
    assert not evaluate_development_gate(
        clone,
        _policy_result(manifest, "fixed_parametric", integrity=False),
    ).passed
    assert not evaluate_development_gate(
        replace(clone, clone_identity_all_repeats=False),
        mutation,
    ).passed
