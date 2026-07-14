"""Thin dependency controller and sealed-suite contract for Evo² R6."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import jax

from experiments.evo2_ecosystem.episode import SimulatorConfig
from experiments.evo2_ecosystem.episode import simulator_config_sha256
from experiments.evo2_ecosystem.episode import simulator_source_sha256
from experiments.evo2_ecosystem.founder_artifacts import founder_index_sha256
from experiments.evo2_ecosystem.founder_artifacts import load_founder_artifact
from experiments.evo2_ecosystem.founder_artifacts import load_founder_index
from experiments.evo2_ecosystem.protocol import EventKind
from experiments.evo2_ecosystem.r5.baselines import FIXED_OPERATOR_NAMES
from experiments.evo2_ecosystem.r5.baselines import HUMAN_CREDIT_NAME
from experiments.evo2_ecosystem.r5.baselines import HUMAN_STRESS_NAME
from experiments.evo2_ecosystem.r5.baselines import human_candidate_sources
from experiments.evo2_ecosystem.r5.workflow import BoundedWorkflowContext
from experiments.evo2_ecosystem.r5.workflow import PROJECT_ROOT
from experiments.evo2_ecosystem.r5.workflow import SHINKA_ROOT
from experiments.evo2_ecosystem.r5.workflow import SealedWork
from experiments.evo2_ecosystem.r5.workflow import WorkflowStopped
from experiments.evo2_ecosystem.r5.workflow import _canonical_json_bytes
from experiments.evo2_ecosystem.r5.workflow import _fixed_source_sha256
from experiments.evo2_ecosystem.r5.workflow import _guarded_python_command
from experiments.evo2_ecosystem.r5.workflow import _load_json
from experiments.evo2_ecosystem.r5.workflow import _project_relative
from experiments.evo2_ecosystem.r5.workflow import _sha256_bytes
from experiments.evo2_ecosystem.r5.workflow import _sha256_file
from experiments.evo2_ecosystem.r5.workflow import _write_once
from experiments.evo2_ecosystem.r5.workflow import run_development_phase
from experiments.evo2_ecosystem.r5.workflow import run_phase5
from experiments.evo2_ecosystem.r5.workflow import run_sealed_analysis_worker
from experiments.evo2_ecosystem.r5.workflow import run_sealed_epoch
from experiments.evo2_ecosystem.r5.workflow import run_sealed_policy_worker
from experiments.evo2_ecosystem.r5.workflow import run_search_phase

from . import analysis
from .protocol import CONTROL_CENTER
from .protocol import DISTURBANCE_QUALIFICATION_PATH
from .protocol import FINAL_ROOT
from .protocol import FINAL_REPORT_PATH
from .protocol import FOUNDER_INDEX_PATH
from .protocol import FOUNDER_ROOT
from .protocol import FOUNDER_SCREENING_RECORD_PATH
from .protocol import MANIFEST_ROOT
from .protocol import OPPORTUNITY_GATE_RECEIPT_PATH
from .protocol import PEAK_CAPACITY
from .protocol import PEAK_REGENERATION
from .protocol import PRIVATE_OPPORTUNITY_STAGING_PATH
from .protocol import PRIVATE_OPPORTUNITY_PATH
from .protocol import PROFILE_NAME
from .protocol import PROTOCOL_REVISION
from .protocol import PRODUCTION_WORLD_QUALIFICATION_SPEC
from .protocol import RESOURCE_RADIUS
from .protocol import RUN_ID
from .protocol import SHOCK_CENTER
from .protocol import STOCK_FRACTION
from .protocol import STOP_REPORT_PATH
from .protocol import WORLD_QUALIFICATION_PATH
from .protocol import WORLD_ACCESS_SCORE_ATOL
from .qualification import build_and_publish_manifests
from .qualification import run_disturbance_qualification
from .qualification import run_opportunity_qualification
from .qualification import run_qualified_founder_screening
from .qualification import validate_disturbance_qualification
from .qualification import validate_opportunity_gate_receipt
from .qualify_worlds import BackendReplay
from .qualify_worlds import InsufficientQualifiedWorlds
from .qualify_worlds import WorldAccessRecord
from .qualify_worlds import WorldQualificationTrace
from .qualify_worlds import build_world_qualification_document
from .qualify_worlds import compare_backend_replay
from .qualify_worlds import make_world_qualifier
from .qualify_worlds import publish_world_qualification
from .qualify_worlds import qualify_world_ranges
from .qualify_worlds import validate_world_qualification


PROFILE_PATH = SHINKA_ROOT / "examples/evo2_ecosystem/run_specs/evo2-r6-resource-relocation.json"
PROFILE_HASH_PATH = PROFILE_PATH.with_suffix(".sha256")
RESULT_ROOT = SHINKA_ROOT / "examples/evo2_ecosystem/results" / RUN_ID
DEVELOPMENT_COMPLETE_PATH = RESULT_ROOT / "development.complete.json"
SEALED_SUITE_PATH = FINAL_ROOT / "suite_record.json"
WORKFLOW_MODULE = "experiments.evo2_ecosystem.r6.workflow"
ANALYSIS_MODULE = "experiments.evo2_ecosystem.r6.analysis"
BASELINES_MODULE = "experiments.evo2_ecosystem.r5.baselines"
_SHA256_LENGTH = 64


def record_stop_report(
    phase: str,
    reason: str,
    *,
    evidence: Sequence[Path] = (),
) -> Path:
    """Publish the sole terminal R6 stop without authorizing later phases."""
    if not phase.strip() or not reason.strip():
        raise ValueError("R6 stop phase and reason must be non-empty")
    lines = [
        "# Evo² R6 resource-relocation stop report",
        "",
        f"- Protocol: `{PROTOCOL_REVISION}`",
        f"- Run ID: `{RUN_ID}`",
        f"- Failed phase: `{phase}`",
        f"- Reason: {reason.strip()}",
        "- Result: execution stopped; no later gate is authorized.",
    ]
    if evidence:
        lines.extend(["", "## Existing evidence", ""])
        for item in evidence:
            readable = item.is_file() and os.access(item, os.R_OK)
            digest = _sha256_file(item) if readable else "missing-or-locked"
            lines.append(f"- `{item.resolve()}` (`{digest}`)")
    _write_once(STOP_REPORT_PATH, ("\n".join(lines) + "\n").encode("utf-8"))
    return STOP_REPORT_PATH


def _record_dict(record: WorldAccessRecord) -> dict[str, Any]:
    return {
        "seed": record.seed,
        "mouth_cells": [list(cell) for cell in record.mouth_cells],
        "mouth_capacity_fractions": list(record.mouth_capacity_fractions),
        "access_score": record.access_score,
        "passed": record.passed,
    }


def _record_from_dict(value: Mapping[str, Any]) -> WorldAccessRecord:
    return WorldAccessRecord(
        seed=value["seed"],
        mouth_cells=tuple(tuple(cell) for cell in value["mouth_cells"]),
        mouth_capacity_fractions=tuple(value["mouth_capacity_fractions"]),
        access_score=value["access_score"],
        passed=value["passed"],
    )


def _write_cpu_world_stage(path: Path) -> None:
    if jax.default_backend() != "cpu":
        raise RuntimeError("R6 world scan requires CPU")
    traces = qualify_world_ranges()
    value = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "backend": "cpu",
        "traces": [
            {
                "seed_range": asdict(trace.seed_range),
                "inspected": [_record_dict(record) for record in trace.inspected],
                "selected_seeds": list(trace.selected_seeds),
            }
            for trace in traces
        ],
    }
    _write_once(path, _canonical_json_bytes(value), mode=0o600)


def _load_cpu_world_stage(path: Path) -> tuple[WorldQualificationTrace, ...]:
    value = _load_json(path)
    if value.get("protocol_revision") != PROTOCOL_REVISION or value.get("backend") != "cpu":
        raise ValueError("R6 CPU world stage has the wrong identity")
    from experiments.evo2_ecosystem.r5.protocol import WorldSeedRange

    traces = tuple(
        WorldQualificationTrace(
            seed_range=WorldSeedRange(**item["seed_range"]),
            inspected=tuple(_record_from_dict(record) for record in item["inspected"]),
            selected_seeds=tuple(item["selected_seeds"]),
        )
        for item in value["traces"]
    )
    if not traces:
        raise ValueError("R6 CPU world stage is empty")
    return traces


def _write_gpu_world_stage(cpu_stage: Path, path: Path) -> None:
    if jax.default_backend() != "gpu":
        raise RuntimeError("R6 selected-world replay requires GPU")
    traces = _load_cpu_world_stage(cpu_stage)
    qualifier = make_world_qualifier(
        PRODUCTION_WORLD_QUALIFICATION_SPEC.config,
        access_threshold=PRODUCTION_WORLD_QUALIFICATION_SPEC.access_threshold,
    )
    replays = []
    for trace in traces:
        by_seed = {record.seed: record for record in trace.inspected}
        for seed in trace.selected_seeds:
            replays.append(
                compare_backend_replay(
                    trace.seed_range.partition,
                    by_seed[seed],
                    qualifier(seed),
                )
            )
    value = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "backend": "gpu",
        "replays": [asdict(item) for item in replays],
    }
    _write_once(path, _canonical_json_bytes(value), mode=0o600)


def _load_gpu_world_stage(path: Path) -> tuple[BackendReplay, ...]:
    value = _load_json(path)
    if value.get("protocol_revision") != PROTOCOL_REVISION or value.get("backend") != "gpu":
        raise ValueError("R6 GPU world stage has the wrong identity")
    return tuple(BackendReplay(**item) for item in value["replays"])


def world_qualification_commands(
    cpu_stage: Path,
    gpu_stage: Path,
) -> tuple[list[str], list[str]]:
    cpu = _guarded_python_command(
        "-m",
        WORKFLOW_MODULE,
        "world-cpu-stage",
        "--output",
        str(cpu_stage.resolve()),
        cpu=True,
    )
    gpu = _guarded_python_command(
        "-m",
        WORKFLOW_MODULE,
        "world-gpu-stage",
        "--cpu-stage",
        str(cpu_stage.resolve()),
        "--output",
        str(gpu_stage.resolve()),
        on_demand_allocator=True,
    )
    return cpu, gpu


def run_world_qualification(
    *,
    launcher: Any = subprocess.run,
) -> dict[str, object]:
    if WORLD_QUALIFICATION_PATH.exists():
        return validate_world_qualification(WORLD_QUALIFICATION_PATH)
    WORLD_QUALIFICATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".r6-world-qualification.",
            dir=WORLD_QUALIFICATION_PATH.parent,
        )
    )
    root = Path(__file__).resolve().parents[3]
    try:
        cpu_stage = staging / "cpu.json"
        gpu_stage = staging / "gpu.json"
        cpu, gpu = world_qualification_commands(cpu_stage, gpu_stage)
        try:
            launcher(cpu, cwd=root, check=True)
        except subprocess.CalledProcessError as error:
            if error.returncode == 20:
                record_stop_report(
                    "world-qualification",
                    "the frozen seed ranges do not contain the required accessible worlds",
                )
            raise
        launcher(gpu, cwd=root, check=True)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        cpu_traces = _load_cpu_world_stage(cpu_stage)
        gpu_replays = _load_gpu_world_stage(gpu_stage)
        if any(not replay.cells_match or not replay.passed_match or replay.score_absolute_error > WORLD_ACCESS_SCORE_ATOL for replay in gpu_replays):
            record_stop_report(
                "world-qualification",
                "selected-world CPU/GPU replay did not match the frozen tolerance",
            )
            raise WorkflowStopped("R6 selected-world backend replay failed")
        document = build_world_qualification_document(
            cpu_traces,
            gpu_replays,
            implementation_commit=commit,
        )
        publish_world_qualification(document, WORLD_QUALIFICATION_PATH)
        return validate_world_qualification(
            WORLD_QUALIFICATION_PATH,
            expected_implementation_commit=commit,
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _run_spec_module() -> Any:
    root = str(SHINKA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("examples.evo2_ecosystem.run_spec")


def authenticated_context(
    profile_path: Path = PROFILE_PATH,
) -> BoundedWorkflowContext:
    """Construct R6 context only after authenticating the schema-v5 profile."""
    module = _run_spec_module()
    spec, _, _ = module.load_run_spec(profile_path)
    if spec.get("schema_version") != 5 or spec.get("run_id") != RUN_ID or spec.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("R6 workflow loaded the wrong authenticated profile")
    expected = {
        "profile": profile_path.resolve(),
        "founder": FOUNDER_INDEX_PATH.resolve(),
        "world": WORLD_QUALIFICATION_PATH.resolve(),
        "disturbance": DISTURBANCE_QUALIFICATION_PATH.resolve(),
        "opportunity": OPPORTUNITY_GATE_RECEIPT_PATH.resolve(),
    }
    actual = {
        "profile": profile_path.resolve(),
        "founder": module.founder_index_path(spec).resolve(),
        "world": module.prerequisite_path(spec, "world_qualification").resolve(),
        "disturbance": module.prerequisite_path(spec, "disturbance_qualification").resolve(),
        "opportunity": module.prerequisite_path(spec, "operator_opportunity").resolve(),
    }
    if actual != expected:
        raise ValueError("schema-v5 profile does not bind the exact R6 context")
    return BoundedWorkflowContext(
        profile_name=PROFILE_NAME,
        profile_path=profile_path,
        profile_hash_path=profile_path.with_suffix(".sha256"),
        workflow_module=WORKFLOW_MODULE,
        run_id=RUN_ID,
        protocol_revision=PROTOCOL_REVISION,
        schema_version=5,
        sealed_suite_schema_version=2,
        founder_index_path=FOUNDER_INDEX_PATH,
        manifest_root=MANIFEST_ROOT,
        world_qualification_path=WORLD_QUALIFICATION_PATH,
        disturbance_path=DISTURBANCE_QUALIFICATION_PATH,
        result_root=RESULT_ROOT,
        development_complete_path=DEVELOPMENT_COMPLETE_PATH,
        final_root=FINAL_ROOT,
        sealed_suite_path=SEALED_SUITE_PATH,
        analysis_module=ANALYSIS_MODULE,
        baselines_module=BASELINES_MODULE,
    )


def _guarded_worker(command: str, *arguments: str) -> list[str]:
    return _guarded_python_command("-m", WORKFLOW_MODULE, command, *arguments)


def _authenticate_founder_bank() -> None:
    index = load_founder_index(FOUNDER_INDEX_PATH, verify_artifacts=False)
    bank = FOUNDER_INDEX_PATH.parent.resolve()
    for record in index.founders:
        path = (bank / record.artifact).resolve()
        if bank not in path.parents:
            raise ValueError("R6 founder artifact escapes the founder bank")
        if record.partition == "training":
            load_founder_artifact(bank, record, expected_partition="training")
        elif not path.is_file() or os.access(path, os.R_OK):
            raise ValueError("R6 holdout founders must remain unreadable")


def run_pre_search_phases(
    *,
    launcher: Any = subprocess.run,
) -> Path:
    """Run exact R6 gates and publish schema v5, or stop terminally."""
    if STOP_REPORT_PATH.is_file():
        raise WorkflowStopped("R6 already ended at a terminal stop report")
    run_world_qualification(launcher=launcher)
    root = Path(__file__).resolve().parents[3]
    if not FOUNDER_INDEX_PATH.is_file():
        launcher(
            _guarded_worker(
                "founder-gate-worker",
                "--world-qualification",
                str(WORLD_QUALIFICATION_PATH.resolve()),
                "--bank-directory",
                str(FOUNDER_ROOT.resolve()),
                "--screening-record",
                str(FOUNDER_SCREENING_RECORD_PATH.resolve()),
            ),
            cwd=root,
            check=True,
        )
    if not FOUNDER_INDEX_PATH.is_file():
        error = WorkflowStopped("R6 founder screen found too few passing candidates")
        record_stop_report(
            "founder-screening",
            str(error),
            evidence=(WORLD_QUALIFICATION_PATH, FOUNDER_SCREENING_RECORD_PATH),
        )
        raise error
    _authenticate_founder_bank()
    try:
        if not DISTURBANCE_QUALIFICATION_PATH.is_file():
            launcher(
                _guarded_worker(
                    "disturbance-gate-worker",
                    "--founder-index",
                    str(FOUNDER_INDEX_PATH.resolve()),
                    "--world-qualification",
                    str(WORLD_QUALIFICATION_PATH.resolve()),
                    "--output",
                    str(DISTURBANCE_QUALIFICATION_PATH.resolve()),
                ),
                cwd=root,
                check=True,
            )
        disturbance = validate_disturbance_qualification(
            DISTURBANCE_QUALIFICATION_PATH,
            founder_index_path=FOUNDER_INDEX_PATH,
            world_qualification_path=WORLD_QUALIFICATION_PATH,
        )
        if disturbance["passed"] is not True:
            raise WorkflowStopped("resource-relocation disturbance did not qualify")
        roles = (
            "training_stable",
            "training_punctuated",
            "development",
            "sealed",
        )
        manifest_paths = tuple(MANIFEST_ROOT / f"{role}.json" for role in roles)
        if not all(path.is_file() and path.with_suffix(".sha256").is_file() for path in manifest_paths):
            if MANIFEST_ROOT.exists():
                raise RuntimeError("partial R6 manifest bundle cannot be resumed")
            build_and_publish_manifests(
                FOUNDER_INDEX_PATH,
                WORLD_QUALIFICATION_PATH,
                DISTURBANCE_QUALIFICATION_PATH,
                MANIFEST_ROOT,
            )
        training_manifest = MANIFEST_ROOT / "training_punctuated.json"
        if not OPPORTUNITY_GATE_RECEIPT_PATH.is_file():
            launcher(
                _guarded_worker(
                    "opportunity-gate-worker",
                    "--founder-index",
                    str(FOUNDER_INDEX_PATH.resolve()),
                    "--training-manifest",
                    str(training_manifest.resolve()),
                    "--disturbance",
                    str(DISTURBANCE_QUALIFICATION_PATH.resolve()),
                    "--private-output",
                    str(PRIVATE_OPPORTUNITY_STAGING_PATH.resolve()),
                    "--receipt-output",
                    str(OPPORTUNITY_GATE_RECEIPT_PATH.resolve()),
                ),
                cwd=root,
                check=True,
            )
        receipt = validate_opportunity_gate_receipt(
            OPPORTUNITY_GATE_RECEIPT_PATH,
            private_path=PRIVATE_OPPORTUNITY_STAGING_PATH,
            founder_index_path=FOUNDER_INDEX_PATH,
            training_manifest_path=training_manifest,
        )
        if receipt["passed"] is not True:
            raise WorkflowStopped("R6 operator-opportunity gate did not pass")
    except subprocess.CalledProcessError:
        raise
    except BaseException as error:
        record_stop_report(
            "qualification",
            str(error),
            evidence=(
                WORLD_QUALIFICATION_PATH,
                FOUNDER_INDEX_PATH,
                DISTURBANCE_QUALIFICATION_PATH,
                OPPORTUNITY_GATE_RECEIPT_PATH,
            ),
        )
        raise

    module = _run_spec_module()
    if not PROFILE_PATH.is_file():
        manifest_hashes = {role: (MANIFEST_ROOT / f"{role}.sha256").read_text(encoding="ascii").split()[0] for role in roles}
        index = load_founder_index(FOUNDER_INDEX_PATH, verify_artifacts=False)
        repositories = {"microcosmos": root, "shinkaevolve": SHINKA_ROOT}
        commits = {
            name: subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            for name, repository in repositories.items()
        }
        spec = module.build_r6_run_spec(
            manifest_sha256=manifest_hashes,
            founder_index_sha256=founder_index_sha256(index),
            simulator_source_sha256=simulator_source_sha256(),
            simulator_config_sha256=simulator_config_sha256(SimulatorConfig()),
            repository_commits=commits,
        )
        module.publish_r6_run_spec(spec, PROFILE_PATH)
    authenticated_context()
    return PROFILE_PATH


def run_search(
    *,
    launcher: Any = subprocess.run,
    context: BoundedWorkflowContext | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Close all searches, then publish the previously committed private gate."""
    active = authenticated_context() if context is None else context
    try:
        audits = run_search_phase(launcher=launcher, context=active)
    except WorkflowStopped as error:
        record_stop_report("outer-search", str(error))
        raise
    receipt = validate_opportunity_gate_receipt(
        OPPORTUNITY_GATE_RECEIPT_PATH,
        private_path=PRIVATE_OPPORTUNITY_STAGING_PATH,
        founder_index_path=FOUNDER_INDEX_PATH,
        training_manifest_path=MANIFEST_ROOT / "training_punctuated.json",
    )
    private_bytes = PRIVATE_OPPORTUNITY_STAGING_PATH.read_bytes()
    if hashlib.sha256(private_bytes).hexdigest() != receipt["private_artifact_sha256"]:
        raise ValueError("R6 private opportunity bytes differ from their receipt")
    _write_once(PRIVATE_OPPORTUNITY_PATH, private_bytes)
    PRIVATE_OPPORTUNITY_PATH.chmod(0o444)
    return audits


def publish_final_report(record: Mapping[str, Any]) -> Path:
    """Publish a compact immutable report from the authenticated final summary."""
    result = record.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("R6 final report requires an authenticated analysis result")
    primary = result.get("primary")
    mechanism = result.get("mechanism")
    if not isinstance(primary, Mapping) or not isinstance(mechanism, Mapping):
        raise ValueError("R6 final analysis lacks primary or mechanism evidence")
    suite_hash = _sha256_file(SEALED_SUITE_PATH)
    summary_path = FINAL_ROOT / "summary.json"
    lines = [
        "# Evo² R6 resource-relocation final report",
        "",
        f"- Protocol: `{PROTOCOL_REVISION}`",
        f"- Run ID: `{RUN_ID}`",
        f"- Sealed suite SHA-256: `{suite_hash}`",
        f"- Final summary SHA-256: `{_sha256_file(summary_path)}`",
        f"- Confirmatory primary passed: `{bool(primary.get('passed'))}`",
        f"- Mechanism status: `{mechanism.get('status')}`",
        f"- Mechanism claim supported: `{bool(mechanism.get('claim_supported', False))}`",
        "",
        "The machine-readable sealed result, confidence intervals, policy matrix,",
        "lineage evidence, and conditional ablations are frozen under",
        f"`{FINAL_ROOT.resolve()}`.",
        "",
        "No metric, threshold, source, seed panel, or sealed result was changed",
        "after final access.",
    ]
    _write_once(FINAL_REPORT_PATH, ("\n".join(lines) + "\n").encode("utf-8"))
    return FINAL_REPORT_PATH


def _sealed_worker_arguments(
    suite_path: Path,
    work_id: str,
    *,
    analysis_worker: bool = False,
) -> tuple[str, ...]:
    command = "sealed-analysis-worker" if analysis_worker else "sealed-policy-worker"
    arguments = (
        "-m",
        WORKFLOW_MODULE,
        command,
        "--suite",
        str(suite_path.resolve()),
    )
    return arguments if analysis_worker else (*arguments, "--policy", work_id)


def _disturbance_record(path: Path) -> dict[str, Any]:
    value = validate_disturbance_qualification(
        path,
        founder_index_path=FOUNDER_INDEX_PATH,
        world_qualification_path=WORLD_QUALIFICATION_PATH,
    )
    if value["passed"] is not True:
        raise WorkflowStopped("R6 disturbance qualification did not pass")
    return {
        "event_kind": EventKind.RESOURCE_RELOCATION.value,
        "control": {
            "center": list(CONTROL_CENTER),
            "radius": RESOURCE_RADIUS,
            "peak_capacity": PEAK_CAPACITY,
            "peak_regeneration": PEAK_REGENERATION,
            "stock_fraction": STOCK_FRACTION,
        },
        "shock": {
            "center": list(SHOCK_CENTER),
            "radius": RESOURCE_RADIUS,
            "peak_capacity": PEAK_CAPACITY,
            "peak_regeneration": PEAK_REGENERATION,
            "stock_fraction": STOCK_FRACTION,
        },
        "qualification": {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        },
    }


def _validate_disturbance_record(value: Mapping[str, Any]) -> None:
    _require_exact_keys(
        value,
        {"event_kind", "control", "shock", "qualification"},
        "R6 sealed disturbance",
    )
    expected = {
        "control": {
            "center": list(CONTROL_CENTER),
            "radius": RESOURCE_RADIUS,
            "peak_capacity": PEAK_CAPACITY,
            "peak_regeneration": PEAK_REGENERATION,
            "stock_fraction": STOCK_FRACTION,
        },
        "shock": {
            "center": list(SHOCK_CENTER),
            "radius": RESOURCE_RADIUS,
            "peak_capacity": PEAK_CAPACITY,
            "peak_regeneration": PEAK_REGENERATION,
            "stock_fraction": STOCK_FRACTION,
        },
    }
    if value["event_kind"] != EventKind.RESOURCE_RELOCATION.value or value["control"] != expected["control"] or value["shock"] != expected["shock"]:
        raise ValueError("R6 sealed disturbance parameters changed")
    qualification = value["qualification"]
    if (
        not isinstance(qualification, Mapping)
        or set(qualification) != {"path", "sha256"}
        or qualification["path"] != str(DISTURBANCE_QUALIFICATION_PATH.resolve())
        or qualification["sha256"] != _sha256_file(DISTURBANCE_QUALIFICATION_PATH)
    ):
        raise ValueError("R6 sealed disturbance qualification changed")


def build_r6_sealed_suite(
    *,
    context: BoundedWorkflowContext,
    run_spec_path: Path,
    structured_random_source: Path,
    finalist_sources: Mapping[str, Path],
    primary_policy_label: str,
    adaptive_policy_label: str | None,
    adaptive_eligible_labels: Sequence[str] = (),
    suite_path: Path = SEALED_SUITE_PATH,
) -> dict[str, Any]:
    """Build the one schema-v2 R6 sealed ledger before sealed access."""
    if context != authenticated_context(run_spec_path):
        raise ValueError("R6 sealed suite requires the authenticated R6 context")
    module = _run_spec_module()
    spec, spec_hash, _ = module.load_run_spec(run_spec_path)
    initial_path = module.initial_program_path(spec)
    works = [
        SealedWork(
            work_id="exact_initial",
            kind="policy",
            source_sha256=_sha256_file(initial_path),
            result_name="policies/exact_initial.json",
            worker_arguments=_sealed_worker_arguments(suite_path, "exact_initial"),
            policy_kind="exact_initial",
            source_path=_project_relative(initial_path),
        )
    ]
    for label in FIXED_OPERATOR_NAMES:
        works.append(
            SealedWork(
                work_id=label,
                kind="policy",
                source_sha256=_fixed_source_sha256(label),
                result_name=f"policies/{label}.json",
                worker_arguments=_sealed_worker_arguments(suite_path, label),
                policy_kind="fixed",
            )
        )
    humans = {item.candidate_id: item for item in human_candidate_sources()}
    for label in (HUMAN_STRESS_NAME, HUMAN_CREDIT_NAME):
        works.append(
            SealedWork(
                work_id=label,
                kind="policy",
                source_sha256=humans[label].sha256,
                result_name=f"policies/{label}.json",
                worker_arguments=_sealed_worker_arguments(suite_path, label),
                policy_kind="human",
            )
        )
    candidates = {"structured_random": structured_random_source, **finalist_sources}
    eligible = set(adaptive_eligible_labels)
    for label, source in candidates.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        works.append(
            SealedWork(
                work_id=label,
                kind="policy",
                source_sha256=_sha256_file(source),
                result_name=f"policies/{label}.json",
                worker_arguments=_sealed_worker_arguments(suite_path, label),
                policy_kind="candidate",
                source_path=_project_relative(source),
                capture_lineage=label in finalist_sources,
                adaptive_eligible=label in eligible,
            )
        )
    policy_ids = {item.work_id for item in works}
    if primary_policy_label not in policy_ids or primary_policy_label != "punctuated_unrestricted":
        raise ValueError("R6 primary must be the preregistered unrestricted punctuated descendant")
    if adaptive_policy_label is not None and adaptive_policy_label not in policy_ids:
        raise ValueError("R6 adaptive policy is absent from the frozen matrix")
    if adaptive_policy_label is not None and adaptive_policy_label not in eligible:
        raise ValueError("R6 adaptive policy lacks frozen adaptive eligibility")
    analysis_path = Path(analysis.__file__).resolve()
    analysis_hash = _sha256_file(analysis_path)
    primary_hash = next(item.source_sha256 for item in works if item.work_id == primary_policy_label)
    for label in ("no_credit_ablation", "dominant_action_ablation"):
        works.append(
            SealedWork(
                work_id=label,
                kind="conditional_policy",
                source_sha256=_sha256_bytes(f"{primary_hash}:{analysis_hash}:{label}".encode("ascii")),
                result_name=f"mechanism/{label}.json",
                worker_arguments=_sealed_worker_arguments(suite_path, label),
                policy_kind=label,
            )
        )
    works.append(
        SealedWork(
            work_id="final_analysis",
            kind="analysis",
            source_sha256=analysis_hash,
            result_name="summary.json",
            worker_arguments=_sealed_worker_arguments(
                suite_path,
                "final_analysis",
                analysis_worker=True,
            ),
        )
    )
    sealed_manifest = module.manifest_path(spec, "sealed")
    founder_index = load_founder_index(FOUNDER_INDEX_PATH, verify_artifacts=False)
    baseline_path = module.baseline_source_path(spec)
    expected_policies = ("exact_initial", *spec["baselines"], *finalist_sources)
    if set(expected_policies) != {item.work_id for item in works if item.kind == "policy"}:
        raise ValueError("R6 policy matrix differs from the schema-v5 baseline contract")
    return {
        "schema_version": 2,
        "protocol_revision": PROTOCOL_REVISION,
        "run_id": RUN_ID,
        "run_spec": {"path": str(run_spec_path.resolve()), "sha256": spec_hash},
        "sealed_manifest": {
            "path": str(sealed_manifest.resolve()),
            "file_sha256": None,
            "sha256": module.manifest_hash(spec, "sealed"),
        },
        "founder_index": {
            "path": str(FOUNDER_INDEX_PATH.resolve()),
            "file_sha256": _sha256_file(FOUNDER_INDEX_PATH),
            "sha256": founder_index_sha256(founder_index),
        },
        "simulator_config_sha256": simulator_config_sha256(SimulatorConfig()),
        "final_analysis": {"path": str(analysis_path), "sha256": analysis_hash},
        "baseline_source": {
            "path": str(baseline_path.resolve()),
            "sha256": _sha256_file(baseline_path),
        },
        "disturbance": _disturbance_record(DISTURBANCE_QUALIFICATION_PATH),
        "expected_policies": list(expected_policies),
        "primary_policy": primary_policy_label,
        "adaptive_policy": adaptive_policy_label,
        "work": [asdict(item) for item in works],
        "mechanism_decision": {
            "result_name": "mechanism_decision.json",
            "rule": "primary-positive-common-garden;primary-adaptive-only-ablations",
        },
        "result_root": str(FINAL_ROOT.resolve()),
    }


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} has the wrong exact keys")


def load_r6_sealed_suite(path: Path) -> tuple[dict[str, Any], str]:
    """Strictly validate the schema-v2 ledger and every bound R6 source."""
    raw = path.read_bytes()
    suite = _load_json(path)
    if raw != _canonical_json_bytes(suite):
        raise ValueError("R6 sealed suite is not canonical JSON")
    _require_exact_keys(
        suite,
        {
            "adaptive_policy",
            "baseline_source",
            "disturbance",
            "expected_policies",
            "final_analysis",
            "founder_index",
            "mechanism_decision",
            "primary_policy",
            "protocol_revision",
            "result_root",
            "run_id",
            "run_spec",
            "schema_version",
            "sealed_manifest",
            "simulator_config_sha256",
            "work",
        },
        "R6 sealed suite",
    )
    if (
        suite["schema_version"] != 2
        or suite["protocol_revision"] != PROTOCOL_REVISION
        or suite["run_id"] != RUN_ID
        or suite["result_root"] != str(FINAL_ROOT.resolve())
    ):
        raise ValueError("R6 sealed suite has the wrong protocol identity")
    module = _run_spec_module()
    spec_path = Path(suite["run_spec"]["path"])
    spec, spec_hash, _ = module.load_run_spec(spec_path)
    if (
        set(suite["run_spec"]) != {"path", "sha256"}
        or spec_path.resolve() != PROFILE_PATH.resolve()
        or spec_hash != suite["run_spec"]["sha256"]
        or spec.get("schema_version") != 5
        or spec.get("run_id") != RUN_ID
        or spec.get("protocol_revision") != PROTOCOL_REVISION
    ):
        raise ValueError("R6 sealed run specification changed")
    _require_exact_keys(
        suite["sealed_manifest"],
        {"path", "file_sha256", "sha256"},
        "R6 sealed manifest binding",
    )
    _require_exact_keys(
        suite["founder_index"],
        {"path", "file_sha256", "sha256"},
        "R6 sealed founder binding",
    )
    for role in ("final_analysis", "baseline_source"):
        _require_exact_keys(suite[role], {"path", "sha256"}, f"R6 {role} binding")
    _require_exact_keys(
        suite["mechanism_decision"],
        {"result_name", "rule"},
        "R6 mechanism-decision binding",
    )
    if suite["mechanism_decision"] != {
        "result_name": "mechanism_decision.json",
        "rule": "primary-positive-common-garden;primary-adaptive-only-ablations",
    }:
        raise ValueError("R6 mechanism-decision rule changed")
    for role in ("sealed_manifest", "founder_index", "final_analysis", "baseline_source"):
        binding = suite[role]
        path_value = Path(binding["path"])
        if not path_value.is_file():
            raise ValueError(f"R6 sealed {role} path is missing")
        if role == "sealed_manifest":
            if path_value.resolve() != module.manifest_path(spec, "sealed").resolve() or module.manifest_hash(spec, "sealed") != binding["sha256"]:
                raise ValueError("R6 sealed manifest semantic hash changed")
        else:
            expected = binding.get("file_sha256", binding.get("sha256"))
            if _sha256_file(path_value) != expected:
                raise ValueError(f"R6 sealed {role} hash changed")
    tool_paths = module.protocol_tool_paths(spec)
    if (
        Path(suite["final_analysis"]["path"]).resolve() != tool_paths["final_analysis"].resolve()
        or suite["final_analysis"]["sha256"] != spec["protocol_tools"]["final_analysis"]["sha256"]
    ):
        raise ValueError("R6 sealed analysis contract changed")
    baseline_path = module.baseline_source_path(spec)
    if Path(suite["baseline_source"]["path"]).resolve() != baseline_path.resolve() or suite["baseline_source"]["sha256"] != spec["baseline_source"]["sha256"]:
        raise ValueError("R6 sealed baseline contract changed")
    if suite["founder_index"]["path"] != str(FOUNDER_INDEX_PATH.resolve()) or suite["founder_index"]["sha256"] != spec["founder_index"]["sha256"]:
        raise ValueError("R6 sealed founder-index path changed")
    _validate_disturbance_record(suite["disturbance"])
    if suite["simulator_config_sha256"] != simulator_config_sha256(SimulatorConfig()):
        raise ValueError("R6 sealed simulator config changed")
    work = suite["work"]
    if not isinstance(work, list) or not work:
        raise ValueError("R6 sealed work matrix is empty")
    for item in work:
        SealedWork(**item)
        source_path = item.get("source_path")
        if source_path:
            bound = (PROJECT_ROOT / source_path).resolve()
            if not bound.is_file() or _sha256_file(bound) != item["source_sha256"]:
                raise ValueError(f"R6 sealed source changed: {item['work_id']}")
        expected_arguments = _sealed_worker_arguments(
            path,
            item["work_id"],
            analysis_worker=item["kind"] == "analysis",
        )
        if tuple(item["worker_arguments"]) != expected_arguments:
            raise ValueError(f"R6 sealed worker routing changed: {item['work_id']}")
    policy_ids = tuple(item["work_id"] for item in work if item["kind"] == "policy")
    if policy_ids != tuple(suite["expected_policies"]):
        raise ValueError("R6 sealed policy matrix changed")
    if suite["primary_policy"] not in policy_ids or (suite["adaptive_policy"] is not None and suite["adaptive_policy"] not in policy_ids):
        raise ValueError("R6 sealed finalist labels changed")
    if suite["primary_policy"] != "punctuated_unrestricted":
        raise ValueError("R6 sealed primary is not the preregistered descendant")
    by_id = {item["work_id"]: item for item in work}
    adaptive_label = suite["adaptive_policy"]
    if adaptive_label is not None and by_id[adaptive_label]["adaptive_eligible"] is not True:
        raise ValueError("R6 sealed adaptive finalist is not eligible")
    humans = {item.candidate_id: item for item in human_candidate_sources()}
    for label in FIXED_OPERATOR_NAMES:
        if by_id[label]["source_sha256"] != _fixed_source_sha256(label):
            raise ValueError(f"R6 fixed baseline changed: {label}")
    for label in (HUMAN_STRESS_NAME, HUMAN_CREDIT_NAME):
        if by_id[label]["source_sha256"] != humans[label].sha256:
            raise ValueError(f"R6 human baseline changed: {label}")
    primary_hash = by_id[suite["primary_policy"]]["source_sha256"]
    for label in ("no_credit_ablation", "dominant_action_ablation"):
        expected_hash = _sha256_bytes((f"{primary_hash}:{suite['final_analysis']['sha256']}:{label}").encode("ascii"))
        if by_id[label]["source_sha256"] != expected_hash:
            raise ValueError(f"R6 ablation binding changed: {label}")
    if by_id["final_analysis"]["source_sha256"] != suite["final_analysis"]["sha256"]:
        raise ValueError("R6 final-analysis work binding changed")
    return suite, hashlib.sha256(raw).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("world-cpu-stage")
    cpu.add_argument("--output", type=Path, required=True)
    gpu = commands.add_parser("world-gpu-stage")
    gpu.add_argument("--cpu-stage", type=Path, required=True)
    gpu.add_argument("--output", type=Path, required=True)
    founder = commands.add_parser("founder-gate-worker")
    founder.add_argument("--world-qualification", type=Path, required=True)
    founder.add_argument("--bank-directory", type=Path, required=True)
    founder.add_argument("--screening-record", type=Path, required=True)
    disturbance = commands.add_parser("disturbance-gate-worker")
    disturbance.add_argument("--founder-index", type=Path, required=True)
    disturbance.add_argument("--world-qualification", type=Path, required=True)
    disturbance.add_argument("--output", type=Path, required=True)
    opportunity = commands.add_parser("opportunity-gate-worker")
    opportunity.add_argument("--founder-index", type=Path, required=True)
    opportunity.add_argument("--training-manifest", type=Path, required=True)
    opportunity.add_argument("--disturbance", type=Path, required=True)
    opportunity.add_argument("--private-output", type=Path, required=True)
    opportunity.add_argument("--receipt-output", type=Path, required=True)
    policy = commands.add_parser("sealed-policy-worker")
    policy.add_argument("--suite", type=Path, required=True)
    policy.add_argument("--policy", required=True)
    final = commands.add_parser("sealed-analysis-worker")
    final.add_argument("--suite", type=Path, required=True)
    shinka = commands.add_parser("gpu-shinka-worker")
    shinka.add_argument("--module", choices=("run_evo", "freeze_finalist"), required=True)
    shinka.add_argument("worker_arguments", nargs=argparse.REMAINDER)
    commands.add_parser("prepare")
    commands.add_parser("search")
    commands.add_parser("development")
    commands.add_parser("sealed")
    commands.add_parser("phase-5")
    commands.add_parser("run")
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    controller_commands = {
        "prepare",
        "search",
        "development",
        "sealed",
        "phase-5",
        "run",
    }
    if arguments.command in controller_commands:
        if not sys.dont_write_bytecode or os.environ.get("JAX_PLATFORMS") != "cpu":
            raise RuntimeError("the R6 controller requires python -B and JAX_PLATFORMS=cpu")
    elif arguments.command != "world-cpu-stage" and jax.default_backend() != "gpu":
        raise RuntimeError(f"{arguments.command} requires the GPU backend")
    if arguments.command == "world-cpu-stage":
        try:
            _write_cpu_world_stage(arguments.output)
        except InsufficientQualifiedWorlds as error:
            print(str(error), file=sys.stderr)
            raise SystemExit(20) from error
    elif arguments.command == "world-gpu-stage":
        _write_gpu_world_stage(arguments.cpu_stage, arguments.output)
    elif arguments.command == "founder-gate-worker":
        run_qualified_founder_screening(
            arguments.world_qualification,
            bank_directory=arguments.bank_directory,
            screening_record_path=arguments.screening_record,
        )
    elif arguments.command == "disturbance-gate-worker":
        run_disturbance_qualification(
            arguments.founder_index,
            arguments.world_qualification,
            arguments.output,
        )
    elif arguments.command == "opportunity-gate-worker":
        run_opportunity_qualification(
            arguments.founder_index,
            arguments.training_manifest,
            arguments.disturbance,
            private_output_path=arguments.private_output,
            receipt_output_path=arguments.receipt_output,
        )
    elif arguments.command == "sealed-policy-worker":
        print(run_sealed_policy_worker(arguments.suite, arguments.policy))
    elif arguments.command == "sealed-analysis-worker":
        print(run_sealed_analysis_worker(arguments.suite))
    elif arguments.command == "gpu-shinka-worker":
        worker_arguments = list(arguments.worker_arguments)
        if worker_arguments[:1] == ["--"]:
            worker_arguments.pop(0)
        module = importlib.import_module(f"examples.evo2_ecosystem.{arguments.module}")
        sys.argv = [str(Path(module.__file__).resolve()), *worker_arguments]
        module.main()
    elif arguments.command == "prepare":
        print(run_pre_search_phases())
    elif arguments.command == "search":
        context = authenticated_context()
        print(json.dumps(run_search(context=context), sort_keys=True, indent=2))
    elif arguments.command == "development":
        context = authenticated_context()
        print(json.dumps(run_development_phase(context=context), sort_keys=True, indent=2))
    elif arguments.command == "sealed":
        context = authenticated_context()
        result = run_sealed_epoch(context=context)
        publish_final_report(result)
        print(json.dumps(result, sort_keys=True, indent=2))
    elif arguments.command == "phase-5":
        context = authenticated_context()
        result = run_phase5(context=context)
        publish_final_report(result)
        print(json.dumps(result, sort_keys=True, indent=2))
    else:
        run_pre_search_phases()
        context = authenticated_context()
        run_search(context=context)
        result = run_phase5(context=context)
        publish_final_report(result)
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
