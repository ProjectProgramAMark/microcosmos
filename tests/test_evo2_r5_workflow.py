from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.r5 import workflow  # noqa: E402
from experiments.evo2_ecosystem.r5.qualify_worlds import (
    WorldAccessRecord,
    WorldQualificationTrace,
)  # noqa: E402
from experiments.evo2_ecosystem.r5.protocol import WorldSeedRange  # noqa: E402


def _write(path: Path, payload: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _json(path: Path, value: dict) -> Path:
    return _write(
        path,
        (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode(),
    )


def test_world_qualification_commands_separate_cpu_and_guarded_gpu(
    tmp_path: Path,
) -> None:
    cpu, gpu = workflow.world_qualification_commands(
        tmp_path / "cpu.json",
        tmp_path / "gpu.json",
    )
    assert cpu[:5] == ["conda", "run", "-n", "sakana", "env"]
    assert "JAX_PLATFORMS=cpu" in cpu
    assert cpu[-2:] == ["--output", str((tmp_path / "cpu.json").resolve())]
    assert gpu[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "MemoryMax=24G" in gpu
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in gpu
    assert gpu[gpu.index("--cpu-stage") + 1] == str((tmp_path / "cpu.json").resolve())
    pythonpath = next(value for value in gpu if value.startswith("PYTHONPATH="))
    assert pythonpath == "PYTHONPATH=" + ":".join(
        (
            str(workflow.SHINKA_ROOT),
            str(workflow.MICROCOSMOS_ROOT / "src"),
            str(workflow.MICROCOSMOS_ROOT),
        )
    )


def test_scientific_gate_commands_each_use_a_fresh_guarded_worker(
    tmp_path: Path,
) -> None:
    founder = workflow.founder_gate_command(
        tmp_path / "world.json",
        tmp_path / "founders",
        tmp_path / "screen.jsonl",
    )
    disturbance = workflow.disturbance_gate_command(
        tmp_path / "index.json",
        tmp_path / "world.json",
        tmp_path / "disturbance.json",
    )
    opportunity = workflow.opportunity_gate_command(
        tmp_path / "index.json",
        tmp_path / "training.json",
        tmp_path / "disturbance.json",
        tmp_path / "opportunity.json",
    )
    for command, subcommand in (
        (founder, "founder-gate-worker"),
        (disturbance, "disturbance-gate-worker"),
        (opportunity, "opportunity-gate-worker"),
    ):
        assert command[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
        assert "MemoryMax=24G" in command
        assert subcommand in command
        env_start = command.index("env")
        python_start = command.index("python", env_start)
        subprocess.run(
            [*command[env_start:python_start], "true"],
            check=True,
        )


def test_founder_worker_launch_failure_is_resumable_not_a_scientific_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _write(tmp_path / "world.json")
    monkeypatch.setattr(workflow, "validate_world_qualification", lambda path: {})
    monkeypatch.setattr(
        workflow,
        "record_stop_report",
        lambda *args, **kwargs: pytest.fail("infrastructure failure became a stop"),
    )

    def fail_launch(*args, **kwargs):
        raise subprocess.CalledProcessError(127, args[0])

    with pytest.raises(subprocess.CalledProcessError):
        workflow.run_phase2_founder_gate(
            world_qualification_path=world,
            bank_directory=tmp_path / "founders",
            screening_record_path=tmp_path / "screen.jsonl",
            launcher=fail_launch,
        )


def test_shinka_candidate_shortfall_writes_bound_failure_once(
    tmp_path: Path,
) -> None:
    completion = _json(tmp_path / "stable.complete.json", {"complete": True})
    paths = SimpleNamespace(
        run_root=tmp_path,
        arm_results=tmp_path / "stable",
    )
    audit = {"database_sha256": "b" * 64}
    freezer = SimpleNamespace(
        _audit_arm=lambda spec, regime, value: audit,
        discover_candidates=lambda value: (1, 2, 3, 4),
    )
    spec = {"run_id": workflow.RUN_ID}
    with pytest.raises(workflow.WorkflowStopped, match="stable"):
        workflow._require_shinka_candidate_floor(
            spec,
            "a" * 64,
            "stable",
            paths,
            freezer,
        )
    failure_path = tmp_path / "stable.search_failure.json"
    original = failure_path.read_bytes()
    failure = json.loads(original)
    assert failure["completion_sha256"] == hashlib.sha256(
        completion.read_bytes()
    ).hexdigest()
    assert failure["archive_audit_sha256"] == hashlib.sha256(
        workflow._canonical_json_bytes(audit)
    ).hexdigest()
    assert failure["unique_valid_candidate_count"] == 4
    with pytest.raises(workflow.WorkflowStopped, match="stable"):
        workflow._require_shinka_candidate_floor(
            spec,
            "a" * 64,
            "stable",
            paths,
            freezer,
        )
    assert failure_path.read_bytes() == original


def test_cpu_stage_wire_round_trip_uses_only_reset_records(tmp_path: Path) -> None:
    records = (
        WorldAccessRecord(
            seed=7,
            mouth_cells=((1, 2),),
            mouth_capacity_fractions=(0.2,),
            access_score=0.2,
            passed=False,
        ),
        WorldAccessRecord(
            seed=8,
            mouth_cells=((3, 4),),
            mouth_capacity_fractions=(0.8,),
            access_score=0.8,
            passed=True,
        ),
    )
    trace = WorldQualificationTrace(
        seed_range=WorldSeedRange("synthetic", 7, 9),
        inspected=records,
        selected_seeds=(8,),
    )
    path = tmp_path / "cpu.json"
    workflow._write_once(
        path,
        workflow._canonical_json_bytes(
            {
                "schema_version": 1,
                "backend": "cpu",
                "traces": [
                    {
                        "seed_range": {
                            "partition": "synthetic",
                            "start": 7,
                            "stop": 9,
                        },
                        "inspected": [
                            workflow._record_dict(record) for record in records
                        ],
                        "selected_seeds": [8],
                    }
                ],
            }
        ),
        mode=0o600,
    )
    assert workflow._load_cpu_world_stage(path) == (trace,)


def test_stop_report_is_single_write_once_terminal_record(tmp_path: Path) -> None:
    evidence = _write(tmp_path / "evidence.json", b"evidence")
    report = tmp_path / "stop.md"
    workflow.record_stop_report(
        "phase-x",
        "frozen gate failed",
        evidence=(evidence,),
        path=report,
    )
    original = report.read_bytes()
    workflow.record_stop_report(
        "phase-x",
        "frozen gate failed",
        evidence=(evidence,),
        path=report,
    )
    assert report.read_bytes() == original
    with pytest.raises(FileExistsError):
        workflow.record_stop_report("phase-y", "retuned", path=report)


def test_pre_search_gate_state_is_dependency_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = tmp_path / "world.json"
    monkeypatch.setattr(workflow, "STOP_REPORT_PATH", tmp_path / "stop.md")
    monkeypatch.setattr(workflow, "WORLD_QUALIFICATION_PATH", world)
    monkeypatch.setattr(workflow, "FOUNDER_INDEX_PATH", tmp_path / "founders/index.json")
    assert workflow.pre_search_gate_state() == workflow.GateState(
        (), "world_qualification", False
    )
    _json(world, {"passed": True})
    monkeypatch.setattr(
        workflow,
        "validate_world_qualification",
        lambda path: {"passed": True},
    )
    assert workflow.pre_search_gate_state() == workflow.GateState(
        ("world_qualification",), "founder_bank", False
    )
    _write(tmp_path / "stop.md", b"terminal")
    assert workflow.pre_search_gate_state().stopped is True


def test_profile_handoff_delegates_schema_to_shinka(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    world = _json(tmp_path / "world.json", {"passed": True})
    disturbance = _json(tmp_path / "disturbance.json", {"passed": True})
    opportunity = _json(tmp_path / "opportunity.json", {"passed": True})
    profile = tmp_path / "profile.json"
    monkeypatch.setattr(workflow, "WORLD_QUALIFICATION_PATH", world)
    monkeypatch.setattr(workflow, "DISTURBANCE_PATH", disturbance)
    monkeypatch.setattr(workflow, "OPPORTUNITY_PATH", opportunity)
    monkeypatch.setattr(workflow, "validate_world_qualification", lambda path: {"passed": True})
    seen = {}

    def build(**values):
        seen["builder"] = values
        return {
            "schema_version": 4,
            "protocol_revision": workflow.PROTOCOL_REVISION,
        }

    def publish(spec, *, path):
        seen["spec"] = spec
        _json(path, spec)
        _write(path.with_suffix(".sha256"), b"0" * 64)

    assert workflow.publish_run_profile(
        manifest_sha256={"training_stable": "1" * 64},
        founder_index_sha256="2" * 64,
        simulator_source_sha256="3" * 64,
        simulator_config_sha256="4" * 64,
        repository_commits={
            "microcosmos": "a" * 40,
            "shinkaevolve": "b" * 40,
        },
        profile_path=profile,
        builder=build,
        publisher=publish,
    ) == profile
    assert seen["builder"] == {
        "manifest_sha256": {"training_stable": "1" * 64},
        "founder_index_sha256": "2" * 64,
        "simulator_source_sha256": "3" * 64,
        "simulator_config_sha256": "4" * 64,
        "repository_commits": {
            "microcosmos": "a" * 40,
            "shinkaevolve": "b" * 40,
        },
    }


def test_concrete_search_runs_stable_then_isolates_then_punctuated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []
    run_root = tmp_path / "run"
    stable = SimpleNamespace(
        run_root=run_root,
        arm_results=run_root / "stable",
    )
    punctuated = SimpleNamespace(
        run_root=run_root,
        arm_results=run_root / "punctuated",
    )
    structured = SimpleNamespace(
        control_root=run_root / "structured_random",
        roster=run_root / "structured_random/roster",
        training=run_root / "structured_random/training",
    )

    class RunSpec:
        @staticmethod
        def paths_for(spec, regime):
            return stable if regime == "stable" else punctuated

        @staticmethod
        def structured_random_paths(spec):
            return structured

    class Freezer:
        @staticmethod
        def _audit_arm(spec, regime, paths):
            events.append(("audit", regime))
            return {"regime": regime}

        @staticmethod
        def discover_candidates(path):
            return list(range(5))

    monkeypatch.setattr(
        workflow,
        "_load_r5_run_spec",
        lambda: (RunSpec, Freezer, {"run_id": workflow.RUN_ID}, "a" * 64, b"{}"),
    )
    monkeypatch.setattr(
        workflow,
        "_authenticate_structured_training",
        lambda *args: events.append(("audit", "structured_random")) or {"passed": True},
    )
    monkeypatch.setattr(
        workflow,
        "_authenticate_closed_searches",
        lambda *args: {
            name: {} for name in ("stable", "punctuated", "structured_random")
        },
    )

    def launch(command, *, cwd, check):
        assert check is True
        if "--regime" in command:
            regime = command[command.index("--regime") + 1]
            if regime == "punctuated":
                assert stable.arm_results.stat().st_mode & 0o777 == 0
            paths = stable if regime == "stable" else punctuated
            paths.arm_results.mkdir(parents=True)
            _write(paths.arm_results / "programs.sqlite")
            _json(run_root / f"{regime}.complete.json", {})
            events.append(("launch", regime))
        else:
            structured.roster.mkdir(parents=True)
            structured.training.mkdir(parents=True)
            _json(structured.control_root / "complete.json", {})
            events.append(("launch", "structured_random"))

    workflow.run_search_phase(launcher=launch)
    assert events[:4] == [
        ("launch", "stable"),
        ("audit", "stable"),
        ("launch", "punctuated"),
        ("audit", "punctuated"),
    ]
    assert ("launch", "structured_random") in events


def test_development_phase_opens_once_and_relocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = []
    module = SimpleNamespace()
    freezer = SimpleNamespace()
    spec = {"run_id": workflow.RUN_ID}
    monkeypatch.setattr(
        workflow,
        "_load_r5_run_spec",
        lambda: (module, freezer, spec, "a" * 64, b"{}"),
    )
    monkeypatch.setattr(
        workflow,
        "_authenticate_closed_searches",
        lambda *args: order.append("closed"),
    )
    monkeypatch.setattr(
        workflow,
        "_assert_holdout_locked",
        lambda spec, partition: order.append(f"locked:{partition}"),
    )
    monkeypatch.setattr(
        workflow,
        "_enter_holdout_epoch",
        lambda spec, partition: order.append(f"open:{partition}"),
    )
    monkeypatch.setattr(
        workflow,
        "_lock_holdout",
        lambda spec, partition: order.append(f"close:{partition}"),
    )
    frozen = {
        "stable": tmp_path / "frozen/stable",
        "punctuated": tmp_path / "frozen/punctuated",
        "structured_random": tmp_path / "frozen/structured_random",
    }
    monkeypatch.setattr(
        module,
        "paths_for",
        lambda spec, regime: SimpleNamespace(frozen=frozen[regime]),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "structured_random_paths",
        lambda spec: SimpleNamespace(frozen=frozen["structured_random"]),
        raising=False,
    )
    monkeypatch.setattr(
        workflow,
        "_load_development_records",
        lambda *args: {"complete": {}},
    )
    monkeypatch.setattr(
        workflow,
        "DEVELOPMENT_COMPLETE_PATH",
        tmp_path / "development.complete.json",
    )
    monkeypatch.setattr(
        workflow,
        "_development_completion_record",
        lambda *args: {
            "schema_version": 1,
            "complete": True,
            "run_id": workflow.RUN_ID,
        },
    )
    monkeypatch.setattr(workflow, "_make_tree_read_only", lambda path: None)
    launched = []

    def launch(command, **_kwargs):
        if "--structured-random-phase" in command:
            label = "structured_random"
            phase = command[command.index("--structured-random-phase") + 1]
        else:
            label = command[command.index("--regime") + 1]
            phase = command[command.index("--development-phase") + 1]
        if phase == "evaluate":
            assert not any(path.exists() for path in frozen.values())
        else:
            assert [item[1] for item in launched[:3]] == ["evaluate"] * 3
            frozen[label].mkdir(parents=True, exist_ok=True)
        launched.append((label, phase))

    result = workflow.run_development_phase(
        launcher=launch,
    )
    assert result == {"complete": {}}
    assert order[:3] == [
        "closed",
        "locked:sealed",
        "open:development",
    ]
    assert order[-3:] == [
        "close:development",
        "locked:development",
        "locked:sealed",
    ]
    assert launched == [
        ("stable", "evaluate"),
        ("punctuated", "evaluate"),
        ("structured_random", "evaluate"),
        ("stable", "freeze"),
        ("punctuated", "freeze"),
        ("structured_random", "freeze"),
    ]


def test_sealed_epoch_requires_authenticated_development_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow,
        "_load_r5_run_spec",
        lambda: (SimpleNamespace(), SimpleNamespace(), {}, "a" * 64, b"{}"),
    )
    monkeypatch.setattr(
        workflow,
        "DEVELOPMENT_COMPLETE_PATH",
        tmp_path / "missing-development.complete.json",
    )
    with pytest.raises(workflow.WorkflowStopped, match="completed development"):
        workflow.run_sealed_epoch(suite_path=tmp_path / "suite.json")


def _sealed_suite(tmp_path: Path) -> tuple[dict, Path, workflow.SealedWork]:
    run_spec = _json(
        tmp_path / "profile.json",
        {"schema_version": 4, "baselines": []},
    )
    manifest = _write(tmp_path / "sealed.json", b"sealed")
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    _write(
        manifest.with_suffix(".sha256"),
        f"{manifest_digest}  {manifest.name}\n".encode(),
    )
    founder = _write(tmp_path / "index.json", b"founders")
    analysis = Path(workflow.__file__).with_name("analysis.py")
    source_path = (
        workflow.SHINKA_ROOT / "examples/evo2_ecosystem/initial_r4.py"
    )
    source = hashlib.sha256(source_path.read_bytes()).hexdigest()
    item = workflow.SealedWork(
        work_id="exact_initial",
        kind="policy",
        source_sha256=source,
        result_name="policies/exact_initial.json",
        worker_arguments=("-m", "trusted.worker", "--policy", "exact_initial"),
        policy_kind="exact_initial",
        source_path=workflow._project_relative(source_path),
    )
    suite = workflow.build_sealed_suite_record(
        run_spec_path=run_spec,
        sealed_manifest_path=manifest,
        founder_index_path=founder,
        simulator_config_sha256=workflow.simulator_config_sha256(
            workflow.SimulatorConfig()
        ),
        final_analysis_path=analysis,
        work=(item,),
    )
    return suite, tmp_path / "suite.json", item


def test_sealed_precommit_never_reads_locked_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_sha256_file = workflow._sha256_file

    def guarded_sha256_file(path: Path) -> str:
        if path.name == "sealed.json":
            pytest.fail("precommit read the locked sealed manifest")
        return real_sha256_file(path)

    monkeypatch.setattr(workflow, "_sha256_file", guarded_sha256_file)
    suite, _, _ = _sealed_suite(tmp_path)
    assert suite["sealed_manifest"]["file_sha256"] is None


def test_frozen_finalist_authenticates_archive_lineage(tmp_path: Path) -> None:
    source = _write(tmp_path / "main.py", b"source")
    lineage = _write(tmp_path / "archive_lineage.json", b"lineage")
    record = {
        "run_spec_sha256": "a" * 64,
        "selection_rule": {"top_k": workflow.FINALIST_TOP_K},
        "development_integrity_valid": True,
        "candidate_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "archive_lineage_record": lineage.name,
        "archive_lineage_sha256": hashlib.sha256(lineage.read_bytes()).hexdigest(),
    }
    record_path = _json(tmp_path / "freeze_record.json", record)
    assert workflow._freeze_record(record_path, "a" * 64) == record
    lineage.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="lineage does not authenticate"):
        workflow._freeze_record(record_path, "a" * 64)


def test_sealed_suite_is_write_once_and_worker_is_fresh_guarded(
    tmp_path: Path,
) -> None:
    suite, suite_path, item = _sealed_suite(tmp_path)
    digest = workflow.precommit_sealed_suite(suite, path=suite_path)
    assert digest == hashlib.sha256(suite_path.read_bytes()).hexdigest()
    workflow.precommit_sealed_suite(suite, path=suite_path)
    command = workflow.sealed_worker_command(suite_path, item.work_id)
    assert command[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "MemoryHigh=20G" in command
    assert command[-4:] == ["-m", "trusted.worker", "--policy", "exact_initial"]
    changed = {**suite, "run_id": "different"}
    with pytest.raises(FileExistsError):
        workflow.precommit_sealed_suite(changed, path=suite_path)


def test_sealed_resume_authenticates_complete_and_launches_only_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, suite_path, item = _sealed_suite(tmp_path)
    workflow.precommit_sealed_suite(suite, path=suite_path)
    launched = []

    def launch(command, *, cwd, check):
        assert check is True
        launched.append(command)
        _json(
            suite_path.parent / item.result_name,
            {
                "work_id": item.work_id,
                "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                "source_sha256": item.source_sha256,
                "complete": True,
            },
        )

    outputs = workflow.run_missing_sealed_work(
        suite_path=suite_path,
        launcher=launch,
    )
    assert outputs == (suite_path.parent / item.result_name,)
    assert len(launched) == 1
    workflow.run_missing_sealed_work(
        suite_path=suite_path,
        launcher=lambda *args, **kwargs: pytest.fail("completed work relaunched"),
    )
    tampered = json.loads(outputs[0].read_text())
    tampered["source_sha256"] = "0" * 64
    outputs[0].chmod(0o600)
    outputs[0].write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="does not authenticate"):
        workflow.run_missing_sealed_work(suite_path=suite_path)


def test_mechanism_decision_uses_fixed_clone_and_handles_no_lineage_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = {
        "mechanism_decision": {"result_name": "decision.json"},
        "primary_policy": "punctuated_adaptive",
        "adaptive_policy": "punctuated_adaptive",
        "work": [
            {
                "work_id": "punctuated_adaptive",
                "adaptive_eligible": True,
            }
        ],
    }
    manifest = SimpleNamespace(
        worlds=(SimpleNamespace(event_kind=workflow.EventKind.NULL),),
    )
    evaluation = SimpleNamespace()
    loaded = []

    def load_result(path, label):
        loaded.append(label)
        return evaluation, (), {"evaluation": {"episodes": [{}]}}

    monkeypatch.setattr(
        workflow,
        "_load_sealed_suite",
        lambda path: (suite, "a" * 64),
    )
    monkeypatch.setattr(workflow, "load_sealed_policy_result", load_result)
    monkeypatch.setattr(workflow, "_load_bound_sealed_manifest", lambda value: manifest)
    monkeypatch.setattr(
        workflow,
        "primary_result_summary",
        lambda *args: {"passed": True},
    )
    decision = workflow.commit_mechanism_decision(tmp_path / "suite.json")
    assert loaded == ["punctuated_adaptive", "clone"]
    assert decision["status"] == "not_estimable_no_lineage_pairs"
    assert decision["run_mechanism"] is False


def test_mechanism_decision_requires_observed_shock_operators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = {
        "mechanism_decision": {"result_name": "decision.json"},
        "primary_policy": "punctuated_adaptive",
        "adaptive_policy": "punctuated_adaptive",
        "work": [
            {
                "work_id": "punctuated_adaptive",
                "adaptive_eligible": True,
            }
        ],
    }
    manifest = SimpleNamespace(
        worlds=(
            SimpleNamespace(event_kind=workflow.EventKind.ACTUATION_COST_SHIFT),
        ),
    )
    evaluation = SimpleNamespace()

    def load_result(path, label):
        del path
        lineage = (object(),) if label == "punctuated_adaptive" else ()
        return evaluation, lineage, {
            "evaluation": {"episodes": [{"operator_counts": [0] * 6}]}
        }

    monkeypatch.setattr(
        workflow,
        "_load_sealed_suite",
        lambda path: (suite, "a" * 64),
    )
    monkeypatch.setattr(workflow, "load_sealed_policy_result", load_result)
    monkeypatch.setattr(workflow, "_load_bound_sealed_manifest", lambda value: manifest)
    monkeypatch.setattr(
        workflow,
        "primary_result_summary",
        lambda *args: {"passed": True},
    )

    decision = workflow.commit_mechanism_decision(tmp_path / "suite.json")

    assert decision["status"] == "not_estimable_no_operator_counts"
    assert decision["dominant_operator"] is None
    assert decision["operator_observations_available"] is False
    assert decision["run_mechanism"] is False


def test_adaptive_finalist_becomes_primary_without_duplicate_sealed_work(
    tmp_path: Path,
) -> None:
    roots = {
        regime: SimpleNamespace(frozen=tmp_path / regime)
        for regime in ("stable", "punctuated")
    }
    module = SimpleNamespace(
        paths_for=lambda spec, regime: roots[regime],
        structured_random_paths=lambda spec: SimpleNamespace(
            frozen=tmp_path / "structured"
        ),
    )
    records = {
        "stable_unrestricted": {"candidate_source_sha256": "1" * 64},
        "punctuated_unrestricted": {"candidate_source_sha256": "2" * 64},
        "punctuated_adaptive": {"candidate_source_sha256": "2" * 64},
    }
    _, finalists, primary, adaptive, eligible = workflow._development_finalist_sources(
        module,
        {},
        records,
    )
    assert "punctuated_adaptive" not in finalists
    assert primary == "punctuated_unrestricted"
    assert adaptive == primary
    assert primary in eligible
