from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem import calibrate_actuator as calibration_module  # noqa: E402
from experiments.evo2_ecosystem.calibrate_actuator import (  # noqa: E402
    CALIBRATION_GAINS,
    FALLBACK_SCHEDULE_ID,
    MUTATION_POLICIES,
    POLICY_ORDER,
    PRIMARY_SCHEDULE_ID,
    PRODUCTION_EVENT_STEP_BY_SEED,
    PRODUCTION_SPEC,
    SMOKE_SPEC,
    BenchmarkGateResult,
    CalibrationWorldResult,
    GainAttemptResult,
    PolicyCalibrationResult,
    build_calibration_manifest,
    evaluate_benchmark_gate,
    fallback_eligible,
    run_actuator_calibration,
    training_founder_records,
    validate_execution_contract,
)
from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: E402
    FounderIndex,
    make_founder_record,
    write_founder_artifact,
    write_founder_index,
)
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    ActuatorInjuryParameters,
    EventKind,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)
from microcosmos.cppn import CPPNGenome, initialize_cppn_population  # noqa: E402


def _founder_bank(path: Path) -> tuple[Path, tuple]:
    path.mkdir()
    panel = initialize_cppn_population(capacity=3, initial_population=3)
    records = []
    for index in range(3):
        founder_id = f"train-{index:02d}"
        artifact = f"{founder_id}.npz"
        genome = CPPNGenome(
            panel.node_genes[index],
            panel.connection_genes[index],
        )
        digests = write_founder_artifact(path / artifact, genome)
        records.append(
            make_founder_record(
                founder_id=founder_id,
                partition="training",
                artifact=artifact,
                digests=digests,
                selection_rule="test-selection",
                selection_seed=17,
            )
        )
    index_path = path / "index.json"
    write_founder_index(index_path, FounderIndex(founders=tuple(records)))
    return index_path, tuple(records)


def _world_result(
    world,
    policy_name,
    *,
    clone_identity,
    injury_score,
    sham_score=0.5,
    births=5,
    generation_gain=2,
    survived=True,
    integrity=True,
):
    arm = "injury" if world.event_kind is EventKind.ACTUATOR_INJURY else "sham"
    score = injury_score if arm == "injury" else sham_score
    operator = POLICY_ORDER.index(policy_name)
    operator_counts = [0] * 4
    operator_counts[operator] = births
    return CalibrationWorldResult(
        policy_name=policy_name,
        founder_id=world.founder_id,
        founder_sha256=world.founder_sha256,
        world_seed=world.world_seed,
        pair_id=world.pair_id,
        event_step=world.event_step,
        arm=arm,
        primary_score=score,
        survived=survived,
        post_event_births=births,
        event_generation=1,
        maximum_generation=1 + generation_gain,
        generation_gain=generation_gain,
        operator_counts=tuple(operator_counts),
        integrity_valid=integrity,
        clone_identity_valid=(clone_identity if policy_name == "clone" else None),
    )


def _attempt(
    manifest,
    gain,
    schedule_id,
    *,
    mutation_injury_score=0.4,
    mutation_births=5,
    mutation_generation_gain=2,
    mutation_survived=True,
    integrity=True,
):
    policies = []
    for policy_name in POLICY_ORDER:
        is_clone = policy_name == "clone"
        worlds = tuple(
            _world_result(
                world,
                policy_name,
                clone_identity=True,
                injury_score=(0.2 if is_clone else mutation_injury_score),
                births=(5 if is_clone else mutation_births),
                generation_gain=(2 if is_clone else mutation_generation_gain),
                survived=(True if is_clone else mutation_survived),
                integrity=integrity,
            )
            for world in manifest.worlds
        )
        policies.append(
            PolicyCalibrationResult(
                policy_name=policy_name,
                worlds=worlds,
                repeat_scores=(float(np.mean([w.primary_score for w in worlds])),),
                selected_repeat_index=0,
                all_repeats_integrity_valid=integrity,
                clone_identity_all_repeats=(True if is_clone else None),
            )
        )
    return GainAttemptResult(
        gain=gain,
        schedule_id=schedule_id,
        manifest_sha256=manifest_sha256(manifest),
        policies=tuple(policies),
    )


def test_production_contract_and_every_gain_manifest_are_exact_schema_v2(tmp_path):
    _, records = _founder_bank(tmp_path / "founders")

    assert PRODUCTION_SPEC.horizon == 8_000
    assert PRODUCTION_SPEC.chunk_steps == 500
    assert PRODUCTION_SPEC.gains == (0.1, 0.2, 0.4)
    assert PRODUCTION_SPEC.injured_hinges == (0, 1)
    assert PRODUCTION_SPEC.numerical_repeats == 3
    assert PRODUCTION_SPEC.event_steps == PRODUCTION_EVENT_STEP_BY_SEED

    for gain in CALIBRATION_GAINS:
        manifest = build_calibration_manifest(
            PRODUCTION_SPEC,
            records,
            gain=gain,
            schedule_id=PRIMARY_SCHEDULE_ID,
        )
        assert manifest.schema_version == 2
        assert manifest.partition == "calibration"
        assert manifest.horizon == 8_000
        assert manifest.chunk_steps == 500
        assert len(manifest.worlds) == 3 * 3 * 2
        assert manifest_from_json_bytes(canonical_manifest_bytes(manifest)) == manifest
        for pair_index in range(0, len(manifest.worlds), 2):
            sham, injury = manifest.worlds[pair_index : pair_index + 2]
            assert sham.event_kind is EventKind.NULL
            assert injury.event_kind is EventKind.ACTUATOR_INJURY
            assert sham.pair_id == injury.pair_id
            assert sham.world_seed == injury.world_seed
            assert sham.event_step == injury.event_step
            assert sham.founder_id == injury.founder_id
            assert sham.founder_sha256 == injury.founder_sha256
            assert injury.event_step == PRODUCTION_EVENT_STEP_BY_SEED[injury.world_seed]
            parameters = injury.event_parameters
            assert isinstance(parameters, ActuatorInjuryParameters)
            assert parameters.hinge_gains == (gain, gain, 1.0, 1.0, 1.0, 1.0)


def test_fallback_manifest_has_only_step_4500(tmp_path):
    _, records = _founder_bank(tmp_path / "founders")
    manifest = build_calibration_manifest(
        PRODUCTION_SPEC,
        records,
        gain=0.2,
        schedule_id=FALLBACK_SCHEDULE_ID,
    )
    assert {world.event_step for world in manifest.worlds} == {4_500}


def test_training_founder_resolution_rejects_nonexact_or_nontraining_panel(tmp_path):
    _, records = _founder_bank(tmp_path / "founders")
    assert (
        training_founder_records(
            FounderIndex(founders=records),
            required=3,
        )
        == records
    )
    with pytest.raises(ValueError, match="exactly 2 training founders"):
        training_founder_records(FounderIndex(founders=records), required=2)


def test_gate_requires_clone_harm_positive_mutation_ci_viability_and_integrity(tmp_path):
    _, records = _founder_bank(tmp_path / "founders")
    manifest = build_calibration_manifest(
        PRODUCTION_SPEC,
        records,
        gain=0.2,
        schedule_id=PRIMARY_SCHEDULE_ID,
    )
    passing = evaluate_benchmark_gate(
        _attempt(manifest, 0.2, PRIMARY_SCHEDULE_ID),
        PRODUCTION_SPEC,
    )
    assert isinstance(passing, BenchmarkGateResult)
    assert passing.passed
    assert passing.clone_harm.confidence_interval[1] < 0.0
    assert passing.winning_policy == "fixed_parametric"
    assert set(passing.qualifying_mutation_policies) == set(MUTATION_POLICIES)
    for _, advantage in passing.mutation_advantages:
        assert advantage.founder_count == 3
        assert advantage.count == 9
        assert advantage.confidence_interval[0] > 0.0

    too_few_births = evaluate_benchmark_gate(
        _attempt(
            manifest,
            0.2,
            PRIMARY_SCHEDULE_ID,
            mutation_births=3,
        ),
        PRODUCTION_SPEC,
    )
    assert not too_few_births.passed
    assert not too_few_births.qualifying_mutation_policies

    shallow = evaluate_benchmark_gate(
        _attempt(
            manifest,
            0.2,
            PRIMARY_SCHEDULE_ID,
            mutation_generation_gain=1,
        ),
        PRODUCTION_SPEC,
    )
    assert not shallow.passed

    extinct = evaluate_benchmark_gate(
        _attempt(
            manifest,
            0.2,
            PRIMARY_SCHEDULE_ID,
            mutation_survived=False,
        ),
        PRODUCTION_SPEC,
    )
    assert not extinct.passed

    invalid = evaluate_benchmark_gate(
        _attempt(
            manifest,
            0.2,
            PRIMARY_SCHEDULE_ID,
            integrity=False,
        ),
        PRODUCTION_SPEC,
    )
    assert not invalid.passed
    assert not invalid.all_integrity_valid


def test_fallback_is_permitted_only_when_every_mutation_has_too_few_births(tmp_path):
    _, records = _founder_bank(tmp_path / "founders")
    low_birth_attempts = []
    for gain in CALIBRATION_GAINS:
        manifest = build_calibration_manifest(
            PRODUCTION_SPEC,
            records,
            gain=gain,
            schedule_id=PRIMARY_SCHEDULE_ID,
        )
        low_birth_attempts.append(
            _attempt(
                manifest,
                gain,
                PRIMARY_SCHEDULE_ID,
                mutation_births=3,
            )
        )
    assert fallback_eligible(low_birth_attempts, PRODUCTION_SPEC)

    enough_births = list(low_birth_attempts)
    manifest = build_calibration_manifest(
        PRODUCTION_SPEC,
        records,
        gain=0.4,
        schedule_id=PRIMARY_SCHEDULE_ID,
    )
    enough_births[-1] = _attempt(
        manifest,
        0.4,
        PRIMARY_SCHEDULE_ID,
        mutation_births=4,
        mutation_injury_score=0.1,
    )
    assert not fallback_eligible(enough_births, PRODUCTION_SPEC)


def test_driver_runs_one_fallback_selects_smallest_gain_and_publishes_hash_sidecar(
    tmp_path,
    monkeypatch,
):
    index_path, _ = _founder_bank(tmp_path / "founders")
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("Status: **FROZEN**\n", encoding="utf-8")
    record_path = tmp_path / "calibration.jsonl"
    manifest_path = tmp_path / "selected.json"
    calls = []

    def runner(manifest, gain, schedule_id):
        calls.append((gain, schedule_id))
        return _attempt(
            manifest,
            gain,
            schedule_id,
            mutation_births=(3 if schedule_id == PRIMARY_SCHEDULE_ID else 5),
        )

    monkeypatch.setattr(calibration_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        calibration_module,
        "CalibrationEvaluator",
        lambda _spec, _founder_index_path: runner,
    )

    result = run_actuator_calibration(
        smoke=False,
        founder_index_path=index_path,
        preregistration_path=preregistration,
        calibration_record_path=record_path,
        selected_manifest_path=manifest_path,
    )

    assert result["status"] == "selected"
    assert result["fallback_used"] is True
    assert result["selected"]["gain"] == 0.1
    assert result["selected"]["schedule_id"] == FALLBACK_SCHEDULE_ID
    assert calls == [
        *((gain, PRIMARY_SCHEDULE_ID) for gain in CALIBRATION_GAINS),
        *((gain, FALLBACK_SCHEDULE_ID) for gain in CALIBRATION_GAINS),
    ]

    selected = manifest_from_json_bytes(manifest_path.read_bytes())
    assert {world.event_step for world in selected.worlds} == {4_500}
    assert manifest_path.read_bytes() == canonical_manifest_bytes(selected) + b"\n"
    digest = manifest_sha256(selected)
    assert manifest_path.with_suffix(".sha256").read_text() == (f"{digest}  {manifest_path.name}\n")

    events = [json.loads(line) for line in record_path.read_text().splitlines()]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    assert events[0]["event"] == "calibration_started"
    assert events[-1]["event"] == "calibration_finished"
    assert events[-1]["selected"]["manifest_sha256"] == digest
    for line in record_path.read_bytes().splitlines():
        assert (
            json.dumps(
                json.loads(line),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            == line
        )

    with pytest.raises(FileExistsError, match="refusing to replace"):
        run_actuator_calibration(
            smoke=False,
            founder_index_path=index_path,
            preregistration_path=preregistration,
            calibration_record_path=tmp_path / "second-record.jsonl",
            selected_manifest_path=manifest_path,
        )


def test_driver_evaluates_whole_grid_before_selecting_smallest_passing_gain(
    tmp_path,
    monkeypatch,
):
    index_path, _ = _founder_bank(tmp_path / "founders")
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("Status: **FROZEN**\n", encoding="utf-8")

    def runner(manifest, gain, schedule_id):
        return _attempt(
            manifest,
            gain,
            schedule_id,
            mutation_injury_score=(0.1 if gain == 0.1 else 0.4),
        )

    monkeypatch.setattr(calibration_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        calibration_module,
        "CalibrationEvaluator",
        lambda _spec, _founder_index_path: runner,
    )

    result = run_actuator_calibration(
        smoke=False,
        founder_index_path=index_path,
        preregistration_path=preregistration,
        calibration_record_path=tmp_path / "calibration.jsonl",
        selected_manifest_path=tmp_path / "selected.json",
    )
    assert result["fallback_used"] is False
    assert result["selected"]["gain"] == 0.2


def test_execution_contract_separates_smoke_and_production():
    validate_execution_contract(PRODUCTION_SPEC, backend="gpu")
    validate_execution_contract(SMOKE_SPEC, backend="cpu")
    with pytest.raises(RuntimeError, match="GPU"):
        validate_execution_contract(PRODUCTION_SPEC, backend="cpu")
    with pytest.raises(RuntimeError, match="CPU"):
        validate_execution_contract(SMOKE_SPEC, backend="gpu")
    with pytest.raises(RuntimeError, match="frozen spec"):
        validate_execution_contract(
            replace(PRODUCTION_SPEC, horizon=7_500),
            backend="gpu",
        )


def test_selected_sidecar_authenticates_exact_manifest_bytes(tmp_path, monkeypatch):
    index_path, _ = _founder_bank(tmp_path / "founders")
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("Status: **FROZEN**\n", encoding="utf-8")
    manifest_path = tmp_path / "selected.json"

    runner = lambda manifest, gain, schedule: _attempt(  # noqa: E731
        manifest,
        gain,
        schedule,
    )
    monkeypatch.setattr(calibration_module.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        calibration_module,
        "CalibrationEvaluator",
        lambda _spec, _founder_index_path: runner,
    )

    result = run_actuator_calibration(
        smoke=False,
        founder_index_path=index_path,
        preregistration_path=preregistration,
        calibration_record_path=tmp_path / "calibration.jsonl",
        selected_manifest_path=manifest_path,
    )
    digest = result["selected"]["manifest_sha256"]
    canonical = manifest_path.read_bytes()[:-1]
    assert hashlib.sha256(canonical).hexdigest() == digest
    assert manifest_path.with_suffix(".sha256").read_text().split()[0] == digest


def test_production_driver_rejects_backend_or_runner_injection(tmp_path):
    index_path, _ = _founder_bank(tmp_path / "founders")
    preregistration = tmp_path / "preregistration.md"
    preregistration.write_text("Status: **FROZEN**\n", encoding="utf-8")
    common = {
        "smoke": False,
        "founder_index_path": index_path,
        "preregistration_path": preregistration,
        "calibration_record_path": tmp_path / "calibration.jsonl",
        "selected_manifest_path": tmp_path / "selected.json",
    }
    with pytest.raises(RuntimeError, match="forbids injected"):
        run_actuator_calibration(
            **common,
            attempt_runner=lambda manifest, gain, schedule: _attempt(
                manifest,
                gain,
                schedule,
            ),
        )
    with pytest.raises(RuntimeError, match="backend overrides"):
        run_actuator_calibration(**common, backend="gpu")
