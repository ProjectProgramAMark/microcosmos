"""Thin, dependency-ordered controller for the frozen Evo² r5 experiment.

The module owns ordering, artifact authentication, process isolation, and
write-once ledgers.  World qualification, founder screening, disturbance and
opportunity calculations, candidate evaluation, finalist selection, and final
statistics remain in their existing trusted modules.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.cppn import CPPNGenome
from microcosmos.heredity import make_r4_offspring_policy

from ..episode import (
    ManifestEvaluation,
    PairedManifestEvaluation,
    SimulatorConfig,
    _paired_r4_score,
    _resolve_pair_indices,
    evaluate_manifest_paired_delta,
    simulator_config_sha256,
    simulator_source_sha256,
)
from ..founder_artifacts import (
    founder_index_sha256,
    load_founder_artifact,
    load_founder_index,
)
from ..protocol import (
    EventKind,
    ScenarioManifest,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)
from .analysis import (
    LineagePair,
    dominant_action_ablation,
    extract_lineage_pairs,
    no_credit_ablation,
    primary_result_summary,
    run_final_analysis,
)
from .baselines import (
    FIXED_OPERATOR_NAMES,
    HUMAN_CREDIT_NAME,
    HUMAN_STRESS_NAME,
    fixed_operator_policy,
    human_candidate_sources,
)

from .protocol import (
    ARTIFACTS_ROOT,
    FINALIST_TOP_K,
    PROTOCOL_REVISION,
    WORLD_QUALIFICATION_PATH,
)
from .qualify_worlds import (
    BackendReplay,
    WorldAccessRecord,
    WorldQualificationTrace,
    build_world_qualification_document,
    compare_backend_replay,
    make_world_qualifier,
    publish_world_qualification,
    qualify_world_ranges,
    validate_world_qualification,
)


MICROCOSMOS_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = MICROCOSMOS_ROOT.parent
SHINKA_ROOT = PROJECT_ROOT / "ShinkaEvolve"
RUN_ID = "evo2-r5-world-feasibility-20260713"
PROFILE_NAME = "evo2-r5-world-feasibility"
PROFILE_PATH = SHINKA_ROOT / "examples/evo2_ecosystem/run_specs/evo2-r5-world-feasibility.json"
PROFILE_HASH_PATH = PROFILE_PATH.with_suffix(".sha256")
FOUNDER_ROOT = ARTIFACTS_ROOT / "founders"
FOUNDER_INDEX_PATH = FOUNDER_ROOT / "index.json"
FOUNDER_SCREENING_RECORD_PATH = ARTIFACTS_ROOT / "founder_screening.jsonl"
MANIFEST_ROOT = ARTIFACTS_ROOT / "manifests"
DISTURBANCE_PATH = ARTIFACTS_ROOT / "disturbance_qualification.json"
OPPORTUNITY_PATH = ARTIFACTS_ROOT / "operator_opportunity.json"
FINAL_ROOT = ARTIFACTS_ROOT / "final"
STOP_REPORT_PATH = MICROCOSMOS_ROOT / "docs/evo2/evo2-r5-world-feasibility-stop-report.md"
RESULT_ROOT = SHINKA_ROOT / "examples/evo2_ecosystem/results" / RUN_ID
DEVELOPMENT_COMPLETE_PATH = RESULT_ROOT / "development.complete.json"
SEALED_SUITE_PATH = FINAL_ROOT / "suite_record.json"

_SHA256_LENGTH = 64


class WorkflowStopped(RuntimeError):
    """The frozen protocol reached a terminal failed gate."""


@dataclass(frozen=True)
class BoundedWorkflowContext:
    """Authenticated paths and bindings for one bounded Evo² protocol.

    R5 remains the default context.  Later protocols must construct an
    explicit context from their authenticated run specification and protocol
    constants; post-profile orchestration must not consult protocol globals.
    """

    profile_name: str
    profile_path: Path
    profile_hash_path: Path
    workflow_module: str
    run_id: str
    protocol_revision: str
    schema_version: int
    sealed_suite_schema_version: int
    founder_index_path: Path
    manifest_root: Path
    world_qualification_path: Path
    disturbance_path: Path
    result_root: Path
    development_complete_path: Path
    final_root: Path
    sealed_suite_path: Path
    analysis_module: str
    baselines_module: str

    def __post_init__(self) -> None:
        if not self.profile_name or not self.workflow_module:
            raise ValueError("workflow context identity must be non-empty")
        if not self.run_id or not self.protocol_revision:
            raise ValueError("workflow context protocol identity must be non-empty")
        if self.schema_version not in {4, 5}:
            raise ValueError("bounded workflow requires schema 4 or 5")
        if self.sealed_suite_schema_version not in {1, 2}:
            raise ValueError("sealed-suite schema must be 1 or 2")
        if (self.schema_version, self.sealed_suite_schema_version) not in {
            (4, 1),
            (5, 2),
        }:
            raise ValueError("run-spec and sealed-suite schemas disagree")


R5_CONTEXT = BoundedWorkflowContext(
    profile_name=PROFILE_NAME,
    profile_path=PROFILE_PATH,
    profile_hash_path=PROFILE_HASH_PATH,
    workflow_module="experiments.evo2_ecosystem.r5.workflow",
    run_id=RUN_ID,
    protocol_revision=PROTOCOL_REVISION,
    schema_version=4,
    sealed_suite_schema_version=1,
    founder_index_path=FOUNDER_INDEX_PATH,
    manifest_root=MANIFEST_ROOT,
    world_qualification_path=WORLD_QUALIFICATION_PATH,
    disturbance_path=DISTURBANCE_PATH,
    result_root=RESULT_ROOT,
    development_complete_path=DEVELOPMENT_COMPLETE_PATH,
    final_root=FINAL_ROOT,
    sealed_suite_path=SEALED_SUITE_PATH,
    analysis_module="experiments.evo2_ecosystem.r5.analysis",
    baselines_module="experiments.evo2_ecosystem.r5.baselines",
)


@dataclass(frozen=True)
class GateState:
    """Current position in the fixed pre-search dependency chain."""

    completed: tuple[str, ...]
    next_gate: str | None
    stopped: bool


@dataclass(frozen=True)
class QualifiedMultiplier:
    """Multiplier authenticated from the one passing disturbance artifact."""

    value: float
    evidence_path: Path
    evidence_sha256: str


@dataclass(frozen=True)
class SealedWork:
    """One predeclared worker in the logical sealed-access epoch."""

    work_id: str
    kind: str
    source_sha256: str
    result_name: str
    worker_arguments: tuple[str, ...]
    policy_kind: str = ""
    source_path: str = ""
    capture_lineage: bool = False
    adaptive_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.work_id or not self.kind or not self.result_name:
            raise ValueError("sealed work identity must be non-empty")
        if len(self.source_sha256) != _SHA256_LENGTH:
            raise ValueError("sealed work source must have a SHA-256")
        result = Path(self.result_name)
        if result.is_absolute() or ".." in result.parts or result == Path("."):
            raise ValueError("sealed result name must stay below the final root")
        if not self.worker_arguments:
            raise ValueError("sealed work must freeze its worker arguments")
        if self.kind in {"policy", "conditional_policy"} and self.policy_kind not in {
            "exact_initial",
            "fixed",
            "human",
            "candidate",
            "no_credit_ablation",
            "dominant_action_ablation",
        }:
            raise ValueError("sealed policy work has an unknown policy kind")
        if self.source_path:
            source = Path(self.source_path)
            if source.is_absolute() or ".." in source.parts:
                raise ValueError("sealed source path must be project-relative")
        elif self.policy_kind in {"exact_initial", "candidate"}:
            raise ValueError("source-backed sealed policy lacks its frozen path")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_once(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    """Publish bytes once; an exact retry verifies rather than replaces them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace frozen artifact: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace frozen artifact: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain one JSON object: {path}")
    return value


def _passing_artifact(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("passed") is not True:
        raise WorkflowStopped(f"gate did not pass: {path}")
    return value


def pre_search_gate_state() -> GateState:
    """Report the next frozen gate without opening development or sealed data."""
    if STOP_REPORT_PATH.exists():
        return GateState((), None, True)
    completed: list[str] = []
    if not WORLD_QUALIFICATION_PATH.is_file():
        return GateState(tuple(completed), "world_qualification", False)
    validate_world_qualification(WORLD_QUALIFICATION_PATH)
    completed.append("world_qualification")
    if not FOUNDER_INDEX_PATH.is_file():
        return GateState(tuple(completed), "founder_bank", False)
    from ..founder_artifacts import load_founder_index

    load_founder_index(FOUNDER_INDEX_PATH, verify_artifacts=False)
    completed.append("founder_bank")
    if not DISTURBANCE_PATH.is_file():
        return GateState(tuple(completed), "disturbance_qualification", False)
    qualified_multiplier(
        DISTURBANCE_PATH,
        founder_index_path=FOUNDER_INDEX_PATH,
        world_qualification_path=WORLD_QUALIFICATION_PATH,
    )
    completed.append("disturbance_qualification")
    manifests = tuple(
        MANIFEST_ROOT / filename
        for filename in (
            "training_stable.json",
            "training_punctuated.json",
            "development.json",
            "sealed.json",
        )
    )
    if not all(path.is_file() and path.with_suffix(".sha256").is_file() for path in manifests):
        return GateState(tuple(completed), "manifest_generation", False)
    completed.append("manifest_generation")
    if not OPPORTUNITY_PATH.is_file():
        return GateState(tuple(completed), "operator_opportunity", False)
    from .tools import validate_opportunity_qualification

    opportunity = validate_opportunity_qualification(
        OPPORTUNITY_PATH,
        founder_index_path=FOUNDER_INDEX_PATH,
        training_manifest_path=MANIFEST_ROOT / "training_punctuated.json",
    )
    if opportunity.get("passed") is not True:
        raise WorkflowStopped("operator-opportunity gate did not pass")
    completed.append("operator_opportunity")
    if not PROFILE_PATH.is_file() or not PROFILE_HASH_PATH.is_file():
        return GateState(tuple(completed), "run_profile", False)
    completed.append("run_profile")
    return GateState(tuple(completed), None, False)


def record_stop_report(
    phase: str,
    reason: str,
    *,
    evidence: Sequence[Path] = (),
    path: Path = STOP_REPORT_PATH,
) -> Path:
    """Persist the single immutable scientific stop for this r5 run ID."""
    if not phase.strip() or not reason.strip():
        raise ValueError("stop phase and reason must be non-empty")
    lines = [
        "# Evo² r5 world-feasibility stop report",
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
            resolved = item.resolve()
            try:
                label = resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                label = str(resolved)
            digest = _sha256_file(resolved) if resolved.is_file() else "missing"
            lines.append(f"- `{label}` (`{digest}`)")
    _write_once(path, ("\n".join(lines) + "\n").encode("utf-8"))
    return path


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


def _write_cpu_world_stage(path: Path) -> dict[str, Any]:
    """CPU worker: inspect the frozen ranges and stage no rollout outcomes."""
    if jax.default_backend() != "cpu":
        raise RuntimeError("world-range scan must run on the CPU backend")
    traces = qualify_world_ranges()
    value = {
        "schema_version": 1,
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
    return value


def _load_cpu_world_stage(path: Path) -> tuple[WorldQualificationTrace, ...]:
    value = _load_json(path)
    if value.get("schema_version") != 1 or value.get("backend") != "cpu":
        raise ValueError("CPU world stage has the wrong schema")
    from .protocol import WorldSeedRange  # local import keeps the wire form tiny

    traces = []
    for item in value.get("traces", []):
        seed_range = WorldSeedRange(**item["seed_range"])
        traces.append(
            WorldQualificationTrace(
                seed_range=seed_range,
                inspected=tuple(_record_from_dict(record) for record in item["inspected"]),
                selected_seeds=tuple(item["selected_seeds"]),
            )
        )
    if not traces:
        raise ValueError("CPU world stage contains no qualification traces")
    return tuple(traces)


def _write_gpu_world_stage(cpu_stage: Path, output: Path) -> dict[str, Any]:
    """GPU worker: replay only the CPU-selected seeds and stage comparisons."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("selected-world replay must run on the GPU backend")
    traces = _load_cpu_world_stage(cpu_stage)
    from .protocol import PRODUCTION_WORLD_QUALIFICATION_SPEC

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
        "backend": "gpu",
        "replays": [
            {
                "partition": replay.partition,
                "seed": replay.seed,
                "mouth_cells": [list(cell) for cell in replay.mouth_cells],
                "access_score": replay.access_score,
                "cells_match": replay.cells_match,
                "score_absolute_error": replay.score_absolute_error,
                "passed_match": replay.passed_match,
            }
            for replay in replays
        ],
    }
    _write_once(output, _canonical_json_bytes(value), mode=0o600)
    return value


def _load_gpu_world_stage(path: Path) -> tuple[BackendReplay, ...]:
    value = _load_json(path)
    if value.get("schema_version") != 1 or value.get("backend") != "gpu":
        raise ValueError("GPU world stage has the wrong schema")
    return tuple(
        BackendReplay(
            partition=item["partition"],
            seed=item["seed"],
            mouth_cells=tuple(tuple(cell) for cell in item["mouth_cells"]),
            access_score=item["access_score"],
            cells_match=item["cells_match"],
            score_absolute_error=item["score_absolute_error"],
            passed_match=item["passed_match"],
        )
        for item in value.get("replays", [])
    )


def _guarded_python_command(
    *arguments: str,
    cpu: bool = False,
    on_demand_allocator: bool = False,
) -> list[str]:
    python = ["conda", "run", "-n", "sakana"]
    project_pythonpath = os.pathsep.join(
        (
            str(SHINKA_ROOT),
            str(MICROCOSMOS_ROOT / "src"),
            str(MICROCOSMOS_ROOT),
        )
    )
    if cpu:
        python.extend(
            [
                "env",
                "-u",
                "JAX_PLATFORM_NAME",
                "JAX_PLATFORMS=cpu",
                "PYTHONDONTWRITEBYTECODE=1",
                f"PYTHONPATH={project_pythonpath}",
            ]
        )
    else:
        python.extend(
            [
                "env",
                "-u",
                "JAX_PLATFORMS",
                "-u",
                "JAX_PLATFORM_NAME",
            ]
        )
        if on_demand_allocator:
            allocator = ["XLA_PYTHON_CLIENT_PREALLOCATE=false"]
        else:
            python.extend(["-u", "XLA_PYTHON_CLIENT_PREALLOCATE"])
            allocator = []
        python.extend(
            [
                "PYTHONDONTWRITEBYTECODE=1",
                f"PYTHONPATH={project_pythonpath}",
                *allocator,
            ]
        )
    python.extend(["python", "-B", *arguments])
    if cpu:
        return python
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "-p",
        "MemoryHigh=20G",
        "-p",
        "MemoryMax=24G",
        "-p",
        "MemorySwapMax=2G",
        *python,
    ]


def world_qualification_commands(
    cpu_stage: Path,
    gpu_stage: Path,
) -> tuple[list[str], list[str]]:
    """Return the fresh CPU-scan and guarded selected-only GPU commands."""
    module = "experiments.evo2_ecosystem.r5.workflow"
    cpu = _guarded_python_command("-m", module, "world-cpu-stage", "--output", str(cpu_stage.resolve()), cpu=True)
    gpu = _guarded_python_command(
        "-m",
        module,
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
    output_path: Path = WORLD_QUALIFICATION_PATH,
    launcher: Callable[..., Any] = subprocess.run,
) -> dict[str, object]:
    """Run CPU selection and GPU replay in fresh processes, then publish once."""
    if output_path.exists():
        return validate_world_qualification(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".world-qualification.", dir=output_path.parent))
    cpu_stage = staging / "cpu.json"
    gpu_stage = staging / "gpu.json"
    try:
        cpu_command, gpu_command = world_qualification_commands(cpu_stage, gpu_stage)
        launcher(cpu_command, cwd=MICROCOSMOS_ROOT, check=True)
        launcher(gpu_command, cwd=MICROCOSMOS_ROOT, check=True)
        traces = _load_cpu_world_stage(cpu_stage)
        replays = _load_gpu_world_stage(gpu_stage)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=MICROCOSMOS_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        document = build_world_qualification_document(
            traces,
            replays,
            implementation_commit=commit,
        )
        publish_world_qualification(document, output_path)
        return validate_world_qualification(output_path, expected_implementation_commit=commit)
    except BaseException as error:
        record_stop_report("phase-2-world-qualification", str(error))
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def founder_gate_command(
    world_qualification_path: Path,
    bank_directory: Path,
    screening_record_path: Path,
) -> list[str]:
    """Construct the fresh guarded founder-screening worker command."""
    return _guarded_python_command(
        "-m",
        "experiments.evo2_ecosystem.r5.workflow",
        "founder-gate-worker",
        "--world-qualification",
        str(world_qualification_path.resolve()),
        "--bank-directory",
        str(bank_directory.resolve()),
        "--screening-record",
        str(screening_record_path.resolve()),
    )


def _run_founder_gate_worker(
    world_qualification_path: Path,
    bank_directory: Path,
    screening_record_path: Path,
) -> None:
    from .tools import run_qualified_founder_screening

    run_qualified_founder_screening(
        world_qualification_path,
        bank_directory=bank_directory,
        screening_record_path=screening_record_path,
    )


def _authenticate_published_founder_bank(bank_directory: Path) -> None:
    """Verify public founders now and preserve locked holdout boundaries."""
    index_path = bank_directory / "index.json"
    index = load_founder_index(index_path, verify_artifacts=False)
    bank = bank_directory.resolve()
    for record in index.founders:
        candidate = bank / record.artifact
        artifact = candidate.resolve()
        if bank not in artifact.parents or candidate.is_symlink():
            raise ValueError("founder artifact escapes the published bank")
        if record.partition == "training":
            load_founder_artifact(
                bank_directory,
                record,
                expected_partition="training",
            )
        elif not artifact.is_file() or os.access(artifact, os.R_OK):
            raise ValueError("development and sealed founders must remain unreadable")


def run_phase2_founder_gate(
    *,
    world_qualification_path: Path = WORLD_QUALIFICATION_PATH,
    bank_directory: Path = FOUNDER_ROOT,
    screening_record_path: Path = FOUNDER_SCREENING_RECORD_PATH,
    launcher: Callable[..., Any] = subprocess.run,
) -> Path:
    """Authenticate qualified worlds and run founder screening in a fresh GPU worker."""
    validate_world_qualification(world_qualification_path)

    try:
        if not (bank_directory / "index.json").exists():
            launcher(
                founder_gate_command(
                    world_qualification_path,
                    bank_directory,
                    screening_record_path,
                ),
                cwd=MICROCOSMOS_ROOT,
                check=True,
            )
        if not (bank_directory / "index.json").is_file():
            raise WorkflowStopped("founder screen did not publish its index")
        _authenticate_published_founder_bank(bank_directory)
    except subprocess.CalledProcessError:
        # Worker-launch failures are resumable infrastructure errors, not
        # evidence that the frozen scientific gate failed.
        raise
    except BaseException as error:
        record_stop_report(
            "phase-2-founder-gate",
            str(error),
            evidence=(world_qualification_path, screening_record_path),
        )
        raise
    return bank_directory / "index.json"


def qualified_multiplier(
    path: Path = DISTURBANCE_PATH,
    *,
    founder_index_path: Path | None = None,
    world_qualification_path: Path | None = None,
) -> QualifiedMultiplier:
    """Return only the first frozen multiplier authenticated by r5/tools.py."""
    from .tools import validate_disturbance_qualification

    value = validate_disturbance_qualification(
        path,
        founder_index_path=founder_index_path,
        world_qualification_path=world_qualification_path,
    )
    if value.get("passed") is not True:
        raise WorkflowStopped("disturbance qualification did not pass")
    multiplier = value.get("selected_multiplier")
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
        raise ValueError("disturbance evidence has no selected multiplier")
    return QualifiedMultiplier(float(multiplier), path, _sha256_file(path))


def disturbance_gate_command(
    founder_index_path: Path,
    world_qualification_path: Path,
    output_path: Path,
) -> list[str]:
    return _guarded_python_command(
        "-m",
        "experiments.evo2_ecosystem.r5.workflow",
        "disturbance-gate-worker",
        "--founder-index",
        str(founder_index_path.resolve()),
        "--world-qualification",
        str(world_qualification_path.resolve()),
        "--output",
        str(output_path.resolve()),
    )


def opportunity_gate_command(
    founder_index_path: Path,
    training_manifest_path: Path,
    disturbance_path: Path,
    output_path: Path,
) -> list[str]:
    return _guarded_python_command(
        "-m",
        "experiments.evo2_ecosystem.r5.workflow",
        "opportunity-gate-worker",
        "--founder-index",
        str(founder_index_path.resolve()),
        "--training-manifest",
        str(training_manifest_path.resolve()),
        "--disturbance",
        str(disturbance_path.resolve()),
        "--output",
        str(output_path.resolve()),
    )


def run_phase3_gates(
    *,
    founder_index_path: Path = FOUNDER_INDEX_PATH,
    world_qualification_path: Path = WORLD_QUALIFICATION_PATH,
    manifest_directory: Path = MANIFEST_ROOT,
    disturbance_path: Path = DISTURBANCE_PATH,
    opportunity_path: Path = OPPORTUNITY_PATH,
    launcher: Callable[..., Any] = subprocess.run,
) -> QualifiedMultiplier:
    """Run disturbance -> bound manifests -> opportunity, stopping on failure."""
    validate_world_qualification(world_qualification_path)
    if not founder_index_path.is_file():
        raise FileNotFoundError(founder_index_path)
    from .tools import (
        build_and_publish_manifests,
        validate_opportunity_qualification,
    )

    try:
        if not disturbance_path.exists():
            launcher(
                disturbance_gate_command(
                    founder_index_path,
                    world_qualification_path,
                    disturbance_path,
                ),
                cwd=MICROCOSMOS_ROOT,
                check=True,
            )
        selected = qualified_multiplier(
            disturbance_path,
            founder_index_path=founder_index_path,
            world_qualification_path=world_qualification_path,
        )
        manifest_paths = tuple(
            manifest_directory / filename
            for filename in (
                "training_stable.json",
                "training_punctuated.json",
                "development.json",
                "sealed.json",
            )
        )
        if not all(path.is_file() and path.with_suffix(".sha256").is_file() for path in manifest_paths):
            if manifest_directory.exists():
                raise RuntimeError("partial r5 manifest bundle cannot be resumed")
            build_and_publish_manifests(
                founder_index_path,
                world_qualification_path,
                disturbance_path,
                output_directory=manifest_directory,
            )
        training_manifest = manifest_directory / "training_punctuated.json"
        if not opportunity_path.exists():
            launcher(
                opportunity_gate_command(
                    founder_index_path,
                    training_manifest,
                    disturbance_path,
                    opportunity_path,
                ),
                cwd=MICROCOSMOS_ROOT,
                check=True,
            )
        opportunity = validate_opportunity_qualification(
            opportunity_path,
            founder_index_path=founder_index_path,
            training_manifest_path=training_manifest,
        )
        if opportunity.get("passed") is not True:
            raise WorkflowStopped("operator-opportunity gate did not pass")
        return selected
    except subprocess.CalledProcessError:
        # The trusted workers publish ordinary failed gate evidence and exit
        # successfully.  A nonzero child exit is therefore infrastructure.
        raise
    except BaseException as error:
        record_stop_report(
            "phase-3-disturbance-or-opportunity",
            str(error),
            evidence=(world_qualification_path, founder_index_path, disturbance_path),
        )
        raise


def publish_run_profile(
    *,
    manifest_sha256: Mapping[str, str],
    founder_index_sha256: str,
    simulator_source_sha256: str,
    simulator_config_sha256: str,
    repository_commits: Mapping[str, str],
    profile_path: Path = PROFILE_PATH,
    builder: Callable[..., dict[str, Any]] | None = None,
    publisher: Callable[..., Any] | None = None,
) -> Path:
    """Handoff authenticated Phase-3 evidence to Shinka's schema-v4 publisher."""
    validate_world_qualification(WORLD_QUALIFICATION_PATH)
    _passing_artifact(DISTURBANCE_PATH)
    _passing_artifact(OPPORTUNITY_PATH)
    if builder is None or publisher is None:
        root = str(SHINKA_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        module = importlib.import_module("examples.evo2_ecosystem.run_spec")
        builder = module.build_r5_run_spec if builder is None else builder
        publisher = module.publish_r5_run_spec if publisher is None else publisher
    spec = builder(
        manifest_sha256=manifest_sha256,
        founder_index_sha256=founder_index_sha256,
        simulator_source_sha256=simulator_source_sha256,
        simulator_config_sha256=simulator_config_sha256,
        repository_commits=repository_commits,
    )
    if spec.get("schema_version") != 4 or spec.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("Shinka builder did not produce the frozen schema-v4 profile")
    publisher(spec, path=profile_path)
    if not profile_path.is_file() or not profile_path.with_suffix(".sha256").is_file():
        raise RuntimeError("Shinka publisher did not materialize the profile and sidecar")
    return profile_path


def _manifest_hash_from_sidecar(path: Path) -> str:
    line = path.with_suffix(".sha256").read_text(encoding="ascii")
    suffix = f"  {path.name}\n"
    if not line.endswith(suffix):
        raise ValueError(f"manifest sidecar has the wrong filename: {path}")
    digest = line[: -len(suffix)]
    if len(digest) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"manifest sidecar has an invalid SHA-256: {path}")
    return digest


def _repository_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_pre_search_phases(
    *,
    launcher: Callable[..., Any] = subprocess.run,
) -> Path:
    """Run the frozen Phase-2/3 gates and publish the sole search profile."""
    if STOP_REPORT_PATH.exists():
        raise WorkflowStopped(f"r5 is stopped by {STOP_REPORT_PATH}")
    run_world_qualification(launcher=launcher)
    run_phase2_founder_gate(launcher=launcher)
    run_phase3_gates(launcher=launcher)
    if PROFILE_PATH.is_file() or PROFILE_HASH_PATH.is_file():
        if not (PROFILE_PATH.is_file() and PROFILE_HASH_PATH.is_file()):
            raise RuntimeError("partial r5 run-profile publication")
        run_spec_module, _, _ = _shinka_modules()
        run_spec_module.load_run_spec(PROFILE_PATH)
        return PROFILE_PATH

    manifests = {
        role: _manifest_hash_from_sidecar(MANIFEST_ROOT / f"{role}.json")
        for role in (
            "training_stable",
            "training_punctuated",
            "development",
            "sealed",
        )
    }
    index = load_founder_index(FOUNDER_INDEX_PATH, verify_artifacts=False)
    return publish_run_profile(
        manifest_sha256=manifests,
        founder_index_sha256=founder_index_sha256(index),
        simulator_source_sha256=simulator_source_sha256(),
        simulator_config_sha256=simulator_config_sha256(SimulatorConfig()),
        repository_commits={
            "microcosmos": _repository_commit(MICROCOSMOS_ROOT),
            "shinkaevolve": _repository_commit(SHINKA_ROOT),
        },
    )


def search_command(
    regime: str,
    *,
    resume: bool = False,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> list[str]:
    """Construct one guarded matched Shinka arm command."""
    if regime not in {"stable", "punctuated"}:
        raise ValueError("Shinka regime must be stable or punctuated")
    arguments = [
        "--regime",
        regime,
        "--profile",
        context.profile_name,
    ]
    if resume:
        arguments.append("--resume")
    return _guarded_shinka_command("run_evo", *arguments, context=context)


def development_command(
    regime: str,
    phase: str,
    *,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> list[str]:
    """Construct one guarded development evaluation or freeze handoff."""
    if regime not in {"stable", "punctuated"}:
        raise ValueError("development regime must be stable or punctuated")
    if phase not in {"evaluate", "freeze"}:
        raise ValueError("development phase must be evaluate or freeze")
    return _guarded_shinka_command(
        "freeze_finalist",
        "--regime",
        regime,
        "--development-phase",
        phase,
        "--profile",
        context.profile_name,
        context=context,
    )


def structured_random_command(
    phase: str,
    *,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> list[str]:
    """Construct the matched structured-random training or freeze command."""
    if phase not in {"training", "evaluate", "freeze"}:
        raise ValueError("structured-random phase must be training, evaluate, or freeze")
    return _guarded_shinka_command(
        "freeze_finalist",
        "--structured-random-phase",
        phase,
        "--profile",
        context.profile_name,
        context=context,
    )


def _guarded_shinka_command(
    module: str,
    *arguments: str,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> list[str]:
    if module not in {"run_evo", "freeze_finalist"}:
        raise ValueError("unknown guarded Shinka worker module")
    return _guarded_python_command(
        "-m",
        context.workflow_module,
        "gpu-shinka-worker",
        "--module",
        module,
        "--",
        *arguments,
    )


def _load_run_spec(
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> tuple[Any, Any, dict[str, Any], str, bytes]:
    run_spec_module, _, freezer = _shinka_modules()
    spec, spec_hash, spec_raw = run_spec_module.load_run_spec(context.profile_path)
    if spec.get("schema_version") != context.schema_version or spec.get("run_id") != context.run_id:
        raise ValueError("workflow loaded the wrong bounded run specification")
    return run_spec_module, freezer, spec, spec_hash, spec_raw


def _load_r5_run_spec() -> tuple[Any, Any, dict[str, Any], str, bytes]:
    """Compatibility wrapper for historical R5 callers and fixtures."""
    return _load_run_spec(R5_CONTEXT)


def _arm_complete(paths: Any, regime: str) -> bool:
    return (paths.run_root / f"{regime}.complete.json").is_file()


def _make_tree_read_only(root: Path) -> None:
    """Close one generated archive without creating a new lock protocol."""
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"archive directory is missing or unsafe: {root}")
    root.chmod(0o500)
    paths = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        if path.is_symlink():
            raise RuntimeError(f"archive contains a symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _assert_tree_read_only(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"closed archive is missing or unsafe: {root}")
    for path in (root, *root.rglob("*")):
        if path.is_symlink() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"closed archive remains writable: {path}")


def _authenticate_structured_training(
    spec: dict[str, Any],
    spec_hash: str,
    run_spec_module: Any,
    freezer: Any,
) -> dict[str, Any]:
    paths = run_spec_module.structured_random_paths(spec)
    baselines = freezer._load_bound_baselines(spec)
    roster = tuple(baselines.load_structured_random_roster(paths.roster))
    completion = freezer._structured_completion(
        paths,
        roster,
        spec=spec,
        spec_hash=spec_hash,
    )
    marker = paths.control_root / "complete.json"
    expected = freezer._json_bytes(completion)
    if not marker.is_file() or marker.read_bytes() != expected:
        raise RuntimeError("structured-random completion marker does not authenticate")
    if completion.get("passed") is not True:
        raise WorkflowStopped("structured-random search has fewer than five unique valid candidates")
    return completion


def _require_shinka_candidate_floor(
    spec: Mapping[str, Any],
    spec_hash: str,
    regime: str,
    paths: Any,
    freezer: Any,
) -> Mapping[str, Any]:
    """Audit one completed arm and immutably record a terminal shortfall."""
    audit = freezer._audit_arm(spec, regime, paths)
    candidates = (
        freezer.discover_candidates(paths.arm_results, minimum_generation=1)
        if spec.get("schema_version") == 5
        else freezer.discover_candidates(paths.arm_results)
    )
    candidate_count = len(candidates)
    if candidate_count >= FINALIST_TOP_K:
        return audit
    completion_path = paths.run_root / f"{regime}.complete.json"
    failure = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "arm": "shinka",
        "regime": regime,
        "reason": "fewer_than_five_unique_valid_candidates",
        "run_spec_sha256": spec_hash,
        "completion_sha256": _sha256_file(completion_path),
        "archive_audit_sha256": _sha256_bytes(_canonical_json_bytes(audit)),
        "unique_valid_candidate_count": candidate_count,
        "required_unique_valid_candidate_count": FINALIST_TOP_K,
        "passed": False,
    }
    _write_once(
        paths.run_root / f"{regime}.search_failure.json",
        _canonical_json_bytes(failure),
    )
    raise WorkflowStopped(f"{regime} has fewer than five unique valid candidates")


def _authenticate_closed_searches(
    run_spec_module: Any,
    freezer: Any,
    spec: dict[str, Any],
    spec_hash: str,
) -> dict[str, Mapping[str, Any]]:
    audits: dict[str, Mapping[str, Any]] = {}
    for regime in ("stable", "punctuated"):
        paths = run_spec_module.paths_for(spec, regime)
        _assert_tree_read_only(paths.arm_results)
        audits[regime] = _require_shinka_candidate_floor(
            spec,
            spec_hash,
            regime,
            paths,
            freezer,
        )
    structured = run_spec_module.structured_random_paths(spec)
    _assert_tree_read_only(structured.roster)
    _assert_tree_read_only(structured.training)
    audits["structured_random"] = _authenticate_structured_training(
        spec,
        spec_hash,
        run_spec_module,
        freezer,
    )
    return audits


def run_search_phase(
    *,
    launcher: Callable[..., Any] = subprocess.run,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> dict[str, Mapping[str, Any]]:
    """Run and close stable, punctuated, then structured-random training."""
    loaded = _load_r5_run_spec() if context is R5_CONTEXT else _load_run_spec(context)
    run_spec_module, freezer, spec, spec_hash, _ = loaded
    stable = run_spec_module.paths_for(spec, "stable")
    punctuated = run_spec_module.paths_for(spec, "punctuated")

    if stable.arm_results.exists() and (stable.arm_results.stat().st_mode & 0o777) == 0:
        stable.arm_results.chmod(0o500)
    if not _arm_complete(stable, "stable"):
        resume = stable.arm_results.exists() and any(stable.arm_results.iterdir())
        launcher(
            search_command("stable", resume=resume, context=context),
            cwd=SHINKA_ROOT,
            check=True,
        )
    _require_shinka_candidate_floor(spec, spec_hash, "stable", stable, freezer)

    # Reuse the sibling-isolation contract already enforced by run_evo.py.
    stable.arm_results.chmod(0o000)
    try:
        if not _arm_complete(punctuated, "punctuated"):
            resume = punctuated.arm_results.exists() and any(punctuated.arm_results.iterdir())
            launcher(
                search_command("punctuated", resume=resume, context=context),
                cwd=SHINKA_ROOT,
                check=True,
            )
        _require_shinka_candidate_floor(
            spec,
            spec_hash,
            "punctuated",
            punctuated,
            freezer,
        )
    except BaseException:
        # Preserve proposer isolation across an authenticated punctuated resume.
        stable.arm_results.chmod(0o000)
        raise

    stable.arm_results.chmod(0o500)
    for paths in (stable, punctuated):
        freezer._audit_arm(spec, paths.arm_results.name, paths)
        _make_tree_read_only(paths.arm_results)

    structured = run_spec_module.structured_random_paths(spec)
    if not (structured.control_root / "complete.json").is_file():
        launcher(
            structured_random_command("training", context=context),
            cwd=SHINKA_ROOT,
            check=True,
        )
    _authenticate_structured_training(
        spec,
        spec_hash,
        run_spec_module,
        freezer,
    )
    _make_tree_read_only(structured.roster)
    _make_tree_read_only(structured.training)
    (structured.control_root / "complete.json").chmod(0o444)
    return _authenticate_closed_searches(
        run_spec_module,
        freezer,
        spec,
        spec_hash,
    )


def _holdout_paths(
    spec: Mapping[str, Any],
    partition: str,
    *,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> tuple[Path, tuple[Path, ...]]:
    if partition not in {"development", "sealed"}:
        raise ValueError("holdout partition must be development or sealed")
    run_spec_module, _, _ = _shinka_modules()
    manifest_path = run_spec_module.manifest_path(spec, partition)
    index = load_founder_index(context.founder_index_path, verify_artifacts=False)
    bank = context.founder_index_path.parent.resolve()
    founders = []
    for record in index.founders:
        if record.partition != partition:
            continue
        candidate = bank / record.artifact
        artifact = candidate.resolve()
        if bank not in artifact.parents or candidate.is_symlink():
            raise ValueError("holdout founder path escapes the frozen founder bank")
        founders.append(artifact)
    if not founders:
        raise ValueError(f"no {partition} founders are present")
    return manifest_path, tuple(founders)


def _assert_holdout_locked(
    spec: Mapping[str, Any],
    partition: str,
    *,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> None:
    manifest_path, founders = _holdout_paths(spec, partition, context=context)
    for path in (manifest_path, *founders):
        if not path.is_file() or os.access(path, os.R_OK):
            raise RuntimeError(f"{partition} holdout must be present and unreadable: {path}")


def _authenticate_open_holdout(
    spec: Mapping[str, Any],
    partition: str,
    *,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> None:
    """Authenticate one already-open holdout without changing its mode."""
    run_spec_module, _, _ = _shinka_modules()
    manifest_path, founders = _holdout_paths(spec, partition, context=context)
    if not all(path.is_file() and os.access(path, os.R_OK) for path in (manifest_path, *founders)):
        raise RuntimeError(f"{partition} holdout is only partially accessible")
    expected_hash = run_spec_module.manifest_hash(spec, partition)
    sidecar = manifest_path.with_suffix(".sha256")
    if sidecar.read_text(encoding="ascii") != f"{expected_hash}  {manifest_path.name}\n":
        raise ValueError(f"{partition} manifest sidecar does not authenticate")
    raw = manifest_path.read_bytes()
    manifest = manifest_from_json_bytes(raw)
    if raw != canonical_manifest_bytes(manifest) + b"\n" or manifest_sha256(manifest) != expected_hash:
        raise ValueError(f"{partition} manifest does not match its binding")
    from ..founder_artifacts import load_founder_artifact

    index = load_founder_index(context.founder_index_path, verify_artifacts=False)
    for record in index.founders:
        if record.partition == partition:
            load_founder_artifact(
                context.founder_index_path.parent,
                record,
                expected_partition=partition,
            )


def _unlock_holdout(
    spec: Mapping[str, Any],
    partition: str,
    *,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> None:
    """Authenticate public bindings, then open exactly one holdout partition."""
    _assert_holdout_locked(spec, partition, context=context)
    manifest_path, founders = _holdout_paths(spec, partition, context=context)
    manifest_path.chmod(0o400)
    for path in founders:
        path.chmod(0o400)
    try:
        _authenticate_open_holdout(spec, partition, context=context)
    except BaseException:
        _lock_holdout(spec, partition, context=context)
        raise


def _enter_holdout_epoch(
    spec: Mapping[str, Any],
    partition: str,
    *,
    resume_artifact: Path | None = None,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> bool:
    """Open a locked epoch or authenticate an interrupted open epoch."""
    manifest_path, founders = _holdout_paths(spec, partition, context=context)
    access = [os.access(path, os.R_OK) for path in (manifest_path, *founders)]
    if not any(access):
        _unlock_holdout(spec, partition, context=context)
        return False
    if not all(access):
        raise RuntimeError(f"{partition} holdout has mixed access modes")
    if resume_artifact is not None and not resume_artifact.is_file():
        raise RuntimeError(f"open {partition} epoch lacks its authenticated resume artifact")
    _authenticate_open_holdout(spec, partition, context=context)
    return True


def _lock_holdout(
    spec: Mapping[str, Any],
    partition: str,
    *,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> None:
    manifest_path, founders = _holdout_paths(spec, partition, context=context)
    for path in (manifest_path, *founders):
        if path.exists():
            path.chmod(0o000)


def _freeze_record(path: Path, spec_hash: str) -> dict[str, Any]:
    record = _load_json(path)
    selection = record.get("selection_rule", {})
    if (
        record.get("run_spec_sha256") != spec_hash
        or not isinstance(selection, Mapping)
        or selection.get("top_k") != FINALIST_TOP_K
        or record.get("development_integrity_valid") is not True
    ):
        raise ValueError(f"development freeze record does not authenticate: {path}")
    frozen_spec = record.get("run_spec")
    if (
        isinstance(frozen_spec, Mapping)
        and frozen_spec.get("schema_version") == 5
        and (
            not isinstance(record.get("selected_generation"), int) or isinstance(record.get("selected_generation"), bool) or record["selected_generation"] <= 0
        )
    ):
        raise ValueError("schema-v5 finalist must be a noninitial descendant")
    source = path.parent / "main.py"
    if not source.is_file() or _sha256_file(source) != record.get("candidate_source_sha256"):
        raise ValueError(f"frozen finalist source does not authenticate: {path}")
    lineage = path.parent / "archive_lineage.json"
    if record.get("archive_lineage_record") != lineage.name or not lineage.is_file() or _sha256_file(lineage) != record.get("archive_lineage_sha256"):
        raise ValueError(f"frozen finalist lineage does not authenticate: {path}")
    return record


def _load_development_records(
    run_spec_module: Any,
    spec: dict[str, Any],
    spec_hash: str,
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for regime in ("stable", "punctuated"):
        root = run_spec_module.paths_for(spec, regime).frozen
        records[f"{regime}_unrestricted"] = _freeze_record(
            root / "unrestricted/freeze_record.json",
            spec_hash,
        )
        adaptive = root / "adaptive/freeze_record.json"
        if adaptive.is_file():
            record = _freeze_record(adaptive, spec_hash)
            eligibility = record.get("adaptive_eligibility", {})
            if not isinstance(eligibility, Mapping) or eligibility.get("eligible") is not True:
                raise ValueError("adaptive finalist lacks frozen adaptive eligibility")
            records[f"{regime}_adaptive"] = record
    structured = run_spec_module.structured_random_paths(spec)
    records["structured_random"] = _freeze_record(
        structured.frozen / "freeze_record.json",
        spec_hash,
    )
    return records


def _development_completion_record(
    run_spec_module: Any,
    spec: dict[str, Any],
    spec_hash: str,
    records: Mapping[str, Mapping[str, Any]],
    *,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> dict[str, Any]:
    artifacts: dict[str, dict[str, str]] = {}
    for label in sorted(records):
        if label == "structured_random":
            directory = run_spec_module.structured_random_paths(spec).frozen
        else:
            regime, finalist_type = label.split("_", 1)
            directory = run_spec_module.paths_for(spec, regime).frozen / finalist_type
        record_path = directory / "freeze_record.json"
        source_path = directory / "main.py"
        lineage_path = directory / "archive_lineage.json"
        artifacts[label] = {
            "freeze_record_sha256": _sha256_file(record_path),
            "source_sha256": _sha256_file(source_path),
            "archive_lineage_sha256": _sha256_file(lineage_path),
        }
    return {
        "schema_version": 1,
        "complete": True,
        "run_id": context.run_id,
        "run_spec_sha256": spec_hash,
        "finalists": artifacts,
    }


def _load_completed_development(
    run_spec_module: Any,
    spec: dict[str, Any],
    spec_hash: str,
    *,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> dict[str, Mapping[str, Any]]:
    completion_path = DEVELOPMENT_COMPLETE_PATH if context is R5_CONTEXT else context.development_complete_path
    if not completion_path.is_file():
        raise WorkflowStopped("sealed access requires an authenticated completed development epoch")
    records = _load_development_records(run_spec_module, spec, spec_hash)
    expected = _canonical_json_bytes(
        _development_completion_record(
            run_spec_module,
            spec,
            spec_hash,
            records,
            context=context,
        )
    )
    if completion_path.read_bytes() != expected:
        raise ValueError("development completion marker does not authenticate")
    return records


def run_development_phase(
    *,
    launcher: Callable[..., Any] = subprocess.run,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> dict[str, Mapping[str, Any]]:
    """Authenticate three closed searches, open development once, and freeze."""
    loaded = _load_r5_run_spec() if context is R5_CONTEXT else _load_run_spec(context)
    run_spec_module, freezer, spec, spec_hash, _ = loaded
    context_kwargs = {} if context is R5_CONTEXT else {"context": context}
    completion_path = DEVELOPMENT_COMPLETE_PATH if context is R5_CONTEXT else context.development_complete_path
    _authenticate_closed_searches(run_spec_module, freezer, spec, spec_hash)
    if completion_path.is_file():
        _assert_holdout_locked(spec, "development", **context_kwargs)
        _assert_holdout_locked(spec, "sealed", **context_kwargs)
        return _load_completed_development(
            run_spec_module,
            spec,
            spec_hash,
            context=context,
        )
    _assert_holdout_locked(spec, "sealed", **context_kwargs)
    _enter_holdout_epoch(spec, "development", **context_kwargs)
    try:
        # Complete and authenticate all 15 development evaluations before the
        # first finalist can be frozen.
        for regime in ("stable", "punctuated"):
            launcher(
                development_command(regime, "evaluate", context=context),
                cwd=SHINKA_ROOT,
                check=True,
            )
        structured = run_spec_module.structured_random_paths(spec)
        launcher(
            structured_random_command("evaluate", context=context),
            cwd=SHINKA_ROOT,
            check=True,
        )
        for regime in ("stable", "punctuated"):
            launcher(
                development_command(regime, "freeze", context=context),
                cwd=SHINKA_ROOT,
                check=True,
            )
        launcher(
            structured_random_command("freeze", context=context),
            cwd=SHINKA_ROOT,
            check=True,
        )
        records = _load_development_records(
            run_spec_module,
            spec,
            spec_hash,
        )
        for regime in ("stable", "punctuated"):
            _make_tree_read_only(run_spec_module.paths_for(spec, regime).frozen)
        _make_tree_read_only(structured.frozen)
        completion = _development_completion_record(
            run_spec_module,
            spec,
            spec_hash,
            records,
            **context_kwargs,
        )
        _write_once(
            completion_path,
            _canonical_json_bytes(completion),
        )
        return records
    finally:
        _lock_holdout(spec, "development", **context_kwargs)
        _assert_holdout_locked(spec, "development", **context_kwargs)
        _assert_holdout_locked(spec, "sealed", **context_kwargs)


def build_sealed_suite_record(
    *,
    run_spec_path: Path,
    sealed_manifest_path: Path,
    founder_index_path: Path,
    simulator_config_sha256: str,
    final_analysis_path: Path,
    work: Sequence[SealedWork],
    finalist_labels: Sequence[str] = (),
    primary_policy_label: str | None = None,
    adaptive_policy_label: str | None = None,
    disturbance_multiplier: float | None = None,
    sealed_manifest_semantic_sha256: str | None = None,
    founder_index_semantic_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the sole immutable ledger before any sealed worker starts."""
    if not work or len({item.work_id for item in work}) != len(work):
        raise ValueError("sealed work IDs must be nonempty and unique")
    for path in (
        run_spec_path,
        sealed_manifest_path,
        founder_index_path,
        final_analysis_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    run_spec = _load_json(run_spec_path)
    if run_spec.get("schema_version") != 4:
        raise ValueError("sealed suite requires the frozen schema-v4 profile")
    baselines = run_spec.get("baselines")
    if not isinstance(baselines, list) or not all(isinstance(item, str) for item in baselines):
        raise ValueError("schema-v4 profile has an invalid baseline list")
    expected_policies = ("exact_initial", *baselines, *finalist_labels)
    if len(set(expected_policies)) != len(expected_policies):
        raise ValueError("sealed policy labels must be unique")
    actual_policies = tuple(item.work_id for item in work if item.kind == "policy")
    if set(actual_policies) != set(expected_policies):
        raise ValueError("sealed work does not contain the exact frozen policy matrix")
    primary = primary_policy_label or (finalist_labels[0] if finalist_labels else "exact_initial")
    if primary not in actual_policies or (adaptive_policy_label is not None and adaptive_policy_label not in actual_policies):
        raise ValueError("primary/adaptive labels must name frozen sealed policies")
    if disturbance_multiplier is not None and (
        isinstance(disturbance_multiplier, bool)
        or not isinstance(disturbance_multiplier, (int, float))
        or not np.isfinite(disturbance_multiplier)
        or disturbance_multiplier <= 0.0
    ):
        raise ValueError("sealed disturbance multiplier must be finite and positive")
    manifest_file_hash = None
    founder_file_hash = _sha256_file(founder_index_path)
    manifest_semantic_hash = sealed_manifest_semantic_sha256 or _manifest_hash_from_sidecar(sealed_manifest_path)
    founder_semantic_hash = founder_index_semantic_sha256 or founder_file_hash
    if any(len(value) != _SHA256_LENGTH for value in (manifest_semantic_hash, founder_semantic_hash)):
        raise ValueError("sealed semantic bindings must be SHA-256 values")
    return {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "run_id": RUN_ID,
        "run_spec": {
            "path": str(run_spec_path.resolve()),
            "sha256": _sha256_file(run_spec_path),
        },
        "sealed_manifest": {
            "path": str(sealed_manifest_path.resolve()),
            "file_sha256": manifest_file_hash,
            "sha256": manifest_semantic_hash,
        },
        "founder_index": {
            "path": str(founder_index_path.resolve()),
            "file_sha256": founder_file_hash,
            "sha256": founder_semantic_hash,
        },
        "simulator_config_sha256": simulator_config_sha256,
        "final_analysis": {
            "path": str(final_analysis_path.resolve()),
            "sha256": _sha256_file(final_analysis_path),
        },
        "expected_policies": list(expected_policies),
        "primary_policy": primary,
        "adaptive_policy": adaptive_policy_label,
        "disturbance_multiplier": disturbance_multiplier,
        "work": [asdict(item) for item in work],
        "mechanism_decision": {
            "result_name": "mechanism_decision.json",
            "rule": "primary-positive-and-adaptive-eligible",
        },
    }


def _project_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"sealed source escapes the project root: {path}") from error


def _fixed_source_sha256(label: str) -> str:
    payload = Path(__file__).with_name("baselines.py").read_bytes() + b"\0fixed:" + label.encode()
    return _sha256_bytes(payload)


def _sealed_policy_command(suite_path: Path, label: str) -> tuple[str, ...]:
    return (
        "-m",
        "experiments.evo2_ecosystem.r5.workflow",
        "sealed-policy-worker",
        "--suite",
        str(suite_path.resolve()),
        "--policy",
        label,
    )


def build_r5_sealed_suite(
    *,
    run_spec_path: Path = PROFILE_PATH,
    structured_random_source: Path,
    finalist_sources: Mapping[str, Path],
    primary_policy_label: str,
    adaptive_policy_label: str | None,
    adaptive_eligible_labels: Sequence[str] = (),
    suite_path: Path = SEALED_SUITE_PATH,
) -> dict[str, Any]:
    """Build the exact r5 policy matrix and predeclare conditional workers."""
    spec = _load_json(run_spec_path)
    if spec.get("schema_version") != 4:
        raise ValueError("r5 sealed suite requires schema v4")
    initial_binding = spec.get("initial_program", {})
    initial_path = PROJECT_ROOT / str(initial_binding.get("path", ""))
    if not initial_path.is_file() or _sha256_file(initial_path) != initial_binding.get("sha256"):
        raise ValueError("exact initial source does not authenticate")
    works = [
        SealedWork(
            work_id="exact_initial",
            kind="policy",
            source_sha256=initial_binding["sha256"],
            result_name="policies/exact_initial.json",
            worker_arguments=_sealed_policy_command(suite_path, "exact_initial"),
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
                worker_arguments=_sealed_policy_command(suite_path, label),
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
                worker_arguments=_sealed_policy_command(suite_path, label),
                policy_kind="human",
            )
        )
    candidate_sources = {"structured_random": structured_random_source, **finalist_sources}
    eligible = set(adaptive_eligible_labels)
    for label, source in candidate_sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        works.append(
            SealedWork(
                work_id=label,
                kind="policy",
                source_sha256=_sha256_file(source),
                result_name=f"policies/{label}.json",
                worker_arguments=_sealed_policy_command(suite_path, label),
                policy_kind="candidate",
                source_path=_project_relative(source),
                capture_lineage=label in finalist_sources,
                adaptive_eligible=label in eligible,
            )
        )
    analysis_hash = _sha256_file(Path(__file__).with_name("analysis.py"))
    primary_items = [item for item in works if item.work_id == primary_policy_label]
    if len(primary_items) != 1:
        raise ValueError("primary policy must name exactly one frozen policy")
    primary_hash = primary_items[0].source_sha256
    for label, policy_kind in (
        ("no_credit_ablation", "no_credit_ablation"),
        ("dominant_action_ablation", "dominant_action_ablation"),
    ):
        source_hash = _sha256_bytes(f"{primary_hash}:{analysis_hash}:{label}".encode("ascii"))
        works.append(
            SealedWork(
                work_id=label,
                kind="conditional_policy",
                source_sha256=source_hash,
                result_name=f"mechanism/{label}.json",
                worker_arguments=_sealed_policy_command(suite_path, label),
                policy_kind=policy_kind,
            )
        )
    works.append(
        SealedWork(
            work_id="final_analysis",
            kind="analysis",
            source_sha256=analysis_hash,
            result_name="summary.json",
            worker_arguments=(
                "-m",
                "experiments.evo2_ecosystem.r5.workflow",
                "sealed-analysis-worker",
                "--suite",
                str(suite_path.resolve()),
            ),
        )
    )
    disturbance = qualified_multiplier(
        DISTURBANCE_PATH,
        founder_index_path=FOUNDER_INDEX_PATH,
        world_qualification_path=WORLD_QUALIFICATION_PATH,
    )
    sealed_manifest_path = MANIFEST_ROOT / "sealed.json"
    sealed_binding = spec["manifests"]["sealed"]
    if sealed_manifest_path.resolve() != (PROJECT_ROOT / sealed_binding["path"]).resolve():
        raise ValueError("sealed manifest path differs from the run specification")
    if _manifest_hash_from_sidecar(sealed_manifest_path) != sealed_binding["sha256"]:
        raise ValueError("sealed manifest sidecar differs from the run specification")
    founder_index = load_founder_index(FOUNDER_INDEX_PATH, verify_artifacts=False)
    return build_sealed_suite_record(
        run_spec_path=run_spec_path,
        sealed_manifest_path=sealed_manifest_path,
        founder_index_path=FOUNDER_INDEX_PATH,
        simulator_config_sha256=simulator_config_sha256(SimulatorConfig()),
        final_analysis_path=Path(__file__).with_name("analysis.py"),
        work=works,
        finalist_labels=tuple(finalist_sources),
        primary_policy_label=primary_policy_label,
        adaptive_policy_label=adaptive_policy_label,
        disturbance_multiplier=disturbance.value,
        sealed_manifest_semantic_sha256=sealed_binding["sha256"],
        founder_index_semantic_sha256=founder_index_sha256(founder_index),
    )


def _load_sealed_suite_v1(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    suite = _load_json(path)
    if raw != _canonical_json_bytes(suite):
        raise ValueError("sealed suite is not canonical JSON")
    if suite.get("schema_version") != 1 or suite.get("protocol_revision") != PROTOCOL_REVISION or suite.get("run_id") != RUN_ID:
        raise ValueError("sealed suite has the wrong protocol identity")
    for role in ("run_spec", "sealed_manifest", "founder_index", "final_analysis"):
        binding = suite.get(role)
        if not isinstance(binding, Mapping):
            raise ValueError(f"sealed suite lacks {role} binding")
        bound_path = Path(str(binding.get("path", "")))
        if not bound_path.is_file():
            raise ValueError(f"sealed suite {role} binding does not authenticate")
        if role == "sealed_manifest":
            if _manifest_hash_from_sidecar(bound_path) != binding.get("sha256"):
                raise ValueError("sealed manifest sidecar does not authenticate")
            file_hash = binding.get("file_sha256")
            if file_hash is not None and (not os.access(bound_path, os.R_OK) or _sha256_file(bound_path) != file_hash):
                raise ValueError("sealed manifest file binding does not authenticate")
            continue
        file_hash = binding.get("file_sha256", binding.get("sha256"))
        if _sha256_file(bound_path) != file_hash:
            raise ValueError(f"sealed suite {role} binding does not authenticate")
    if suite["simulator_config_sha256"] != simulator_config_sha256(SimulatorConfig()):
        raise ValueError("sealed suite simulator config changed")
    if suite["final_analysis"]["sha256"] != _sha256_file(Path(__file__).with_name("analysis.py")):
        raise ValueError("sealed suite final analysis changed")
    work = suite.get("work")
    if not isinstance(work, list) or not all(isinstance(item, Mapping) for item in work):
        raise ValueError("sealed suite work matrix is invalid")
    if len({item.get("work_id") for item in work}) != len(work):
        raise ValueError("sealed suite work matrix is invalid")
    policy_ids = tuple(item.get("work_id") for item in work if item.get("kind") == "policy")
    if policy_ids != tuple(suite.get("expected_policies", ())):
        raise ValueError("sealed suite policy matrix changed")
    if suite.get("primary_policy") not in policy_ids or (suite.get("adaptive_policy") is not None and suite.get("adaptive_policy") not in policy_ids):
        raise ValueError("sealed suite primary policy is invalid")
    human_sources = {item.candidate_id: item for item in human_candidate_sources()}
    by_id = {item.get("work_id"): item for item in work}
    for item in work:
        if not isinstance(item, Mapping):
            raise ValueError("sealed suite contains a non-object work item")
        SealedWork(**item)
        policy_kind = item.get("policy_kind")
        source_path = item.get("source_path")
        if source_path:
            path_value = (PROJECT_ROOT / source_path).resolve()
            if _project_relative(path_value) != source_path or not path_value.is_file() or _sha256_file(path_value) != item.get("source_sha256"):
                raise ValueError(f"sealed source changed: {item.get('work_id')}")
        elif policy_kind == "fixed" and item.get("source_sha256") != _fixed_source_sha256(str(item.get("work_id"))):
            raise ValueError(f"sealed fixed source changed: {item.get('work_id')}")
        elif policy_kind == "human":
            source = human_sources.get(str(item.get("work_id")))
            if source is None or source.sha256 != item.get("source_sha256"):
                raise ValueError(f"sealed human source changed: {item.get('work_id')}")
        elif policy_kind in {"no_credit_ablation", "dominant_action_ablation"}:
            primary = by_id[suite["primary_policy"]]
            expected_hash = _sha256_bytes((f"{primary['source_sha256']}:{suite['final_analysis']['sha256']}:{item['work_id']}").encode("ascii"))
            if item.get("source_sha256") != expected_hash:
                raise ValueError(f"sealed ablation source changed: {item.get('work_id')}")
    return suite, _sha256_bytes(raw)


def _load_sealed_suite(path: Path) -> tuple[dict[str, Any], str]:
    """Dispatch sealed-suite validation without broadening R5 schema 1."""
    value = _load_json(path)
    version = value.get("schema_version")
    if version == 1:
        return _load_sealed_suite_v1(path)
    if version == 2:
        module = importlib.import_module("experiments.evo2_ecosystem.r6.workflow")
        return module.load_r6_sealed_suite(path)
    raise ValueError("sealed suite has an unsupported schema version")


def _load_bound_sealed_manifest(suite: Mapping[str, Any]) -> ScenarioManifest:
    path = Path(suite["sealed_manifest"]["path"])
    raw = path.read_bytes()
    manifest = manifest_from_json_bytes(raw)
    if raw != canonical_manifest_bytes(manifest) + b"\n":
        raise ValueError("sealed manifest is not canonical JSON")
    if manifest_sha256(manifest) != suite["sealed_manifest"]["sha256"]:
        raise ValueError("sealed manifest hash changed")
    if manifest.partition != "sealed_final":
        raise ValueError("sealed manifest has the wrong partition")
    return manifest


def _sealed_analysis_module(suite: Mapping[str, Any]) -> Any:
    version = suite.get("schema_version")
    if version in {None, 1}:
        return SimpleNamespace(
            extract_lineage_pairs=extract_lineage_pairs,
            primary_result_summary=primary_result_summary,
            run_final_analysis=run_final_analysis,
        )
    if version == 2:
        return importlib.import_module("experiments.evo2_ecosystem.r6.analysis")
    raise ValueError("sealed suite has an unsupported analysis binding")


def _sealed_shock_indices(
    suite: Mapping[str, Any],
    manifest: ScenarioManifest,
) -> frozenset[int]:
    if suite.get("schema_version") in {None, 1}:
        return frozenset(index for index, world in enumerate(manifest.worlds) if world.event_kind is not EventKind.NULL)
    grouped: dict[str, list[int]] = {}
    for index, world in enumerate(manifest.worlds):
        grouped.setdefault(world.pair_id, []).append(index)
    return frozenset(_resolve_pair_indices(manifest, indices)[1] for indices in grouped.values())


def _shinka_modules():
    root = str(SHINKA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return (
        importlib.import_module("examples.evo2_ecosystem.run_spec"),
        importlib.import_module("examples.evo2_ecosystem.evaluate"),
        importlib.import_module("examples.evo2_ecosystem.freeze_finalist"),
    )


def _authenticate_sealed_runtime(
    spec: Mapping[str, Any],
    run_spec_module: Any,
) -> None:
    """Require the sealed worker to execute the source-bound runtime."""
    task = SHINKA_ROOT / "examples/evo2_ecosystem"
    version = run_spec_module.schema_version(spec)
    if version == 4:
        analysis_path = Path(__file__).with_name("analysis.py")
        baseline_path = Path(__file__).with_name("baselines.py")
    elif version == 5:
        analysis_path = run_spec_module.protocol_tool_paths(spec)["final_analysis"]
        baseline_path = run_spec_module.baseline_source_path(spec)
    else:
        raise ValueError("sealed runtime requires schema 4 or 5")
    source_paths = {
        "initial": run_spec_module.initial_program_path(spec),
        "evaluator": task / "evaluate.py",
        "analysis": analysis_path,
        "baseline": baseline_path,
        "dependency_lock": run_spec_module.DEPENDENCY_LOCK_PATH,
        "launcher": task / "run_evo.py",
        "finalist_selector": task / "freeze_finalist.py",
        "lineage_selector": task / "program_lineage.py",
        "run_spec_module": Path(run_spec_module.__file__),
        "adaptive_selector": task / "r4_selection.py",
    }
    expected_sources = spec.get("source_sha256", {})
    for name, path in source_paths.items():
        if not path.is_file() or _sha256_file(path) != expected_sources.get(name):
            raise ValueError(f"sealed runtime source changed: {name}")
    if simulator_source_sha256() != expected_sources.get("simulator"):
        raise ValueError("sealed simulator source differs from the run specification")
    if simulator_config_sha256(SimulatorConfig()) != spec.get("simulator_config_sha256"):
        raise ValueError("sealed simulator configuration differs from the run specification")
    for role, binding in spec.get("protocol_tools", {}).items():
        path = (PROJECT_ROOT / binding["path"]).resolve()
        if not path.is_file() or _sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"sealed protocol tool changed: {role}")
    if version == 5:
        for role, binding in spec.get("protocol_sources", {}).items():
            path = (PROJECT_ROOT / binding["path"]).resolve()
            if not path.is_file() or _sha256_file(path) != binding.get("sha256"):
                raise ValueError(f"sealed R6 protocol source changed: {role}")
        for role, binding in spec.get("prerequisite_artifacts", {}).items():
            path = (PROJECT_ROOT / binding["path"]).resolve()
            if not path.is_file() or _sha256_file(path) != binding.get("sha256"):
                raise ValueError(f"sealed R6 prerequisite changed: {role}")
    launcher = importlib.import_module("examples.evo2_ecosystem.run_evo")
    launcher._verify_repository_state(spec)


def _source_candidate_policy(
    source_path: Path,
    spec: Mapping[str, Any],
    boundary: Any,
):
    initial = PROJECT_ROOT / spec["initial_program"]["path"]
    contract = spec["candidate_contract"]["version"]
    module = boundary._load_candidate(source_path, contract, initial)
    boundary._smoke_validate_candidate(
        module,
        spec["candidate_output_width"],
        contract_version=contract,
        runtime_budget_ms=float(spec["candidate_contract"]["runtime_budget_ms"]),
    )
    return boundary._build_policy(
        module,
        make_r4_offspring_policy,
        contract_version=contract,
    )


def _human_policy(label: str, spec: Mapping[str, Any], boundary: Any):
    sources = {item.candidate_id: item for item in human_candidate_sources()}
    candidate = sources[label]
    with tempfile.TemporaryDirectory(prefix="evo2-r5-human-") as directory:
        path = Path(directory) / "main.py"
        path.write_text(candidate.source, encoding="utf-8")
        return _source_candidate_policy(path, spec, boundary)


def _work_item(suite: Mapping[str, Any], work_id: str) -> dict[str, Any]:
    matches = [item for item in suite["work"] if item.get("work_id") == work_id]
    if len(matches) != 1:
        raise KeyError(f"unknown sealed work ID: {work_id!r}")
    return matches[0]


def _policy_from_item(
    suite: Mapping[str, Any],
    item: Mapping[str, Any],
    spec: Mapping[str, Any],
    boundary: Any,
    *,
    decision: Mapping[str, Any] | None = None,
):
    kind = item["policy_kind"]
    if kind == "fixed":
        return fixed_operator_policy(item["work_id"])
    if kind == "human":
        return _human_policy(item["work_id"], spec, boundary)
    if kind in {"exact_initial", "candidate"}:
        source = (PROJECT_ROOT / item["source_path"]).resolve()
        return _source_candidate_policy(source, spec, boundary)
    primary_item = _work_item(suite, suite["primary_policy"])
    primary = _policy_from_item(suite, primary_item, spec, boundary)
    if kind == "no_credit_ablation":
        return no_credit_ablation(primary)
    if kind == "dominant_action_ablation":
        if decision is None or not isinstance(decision.get("dominant_operator"), int):
            raise ValueError("dominant-action ablation requires the frozen mechanism decision")
        return dominant_action_ablation(decision["dominant_operator"])
    raise ValueError(f"unknown sealed policy kind: {kind}")


def _array_record(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return {
        "dtype": "float32",
        "shape": list(array.shape),
        "base64": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def _array_from_record(value: Mapping[str, Any]) -> jax.Array:
    if value.get("dtype") != "float32" or not isinstance(value.get("shape"), list):
        raise ValueError("encoded genome array has the wrong schema")
    raw = base64.b64decode(value["base64"], validate=True)
    array = np.frombuffer(raw, dtype=np.float32)
    shape = tuple(value["shape"])
    if array.size != int(np.prod(shape, dtype=np.int64)):
        raise ValueError("encoded genome array has the wrong byte length")
    return jnp.asarray(array.reshape(shape))


def _lineage_record(pairs: Sequence[LineagePair]) -> list[dict[str, Any]]:
    return [
        {
            "founder_id": pair.founder_id,
            "ancestor_id": pair.ancestor_id,
            "descendant_id": pair.descendant_id,
            "ancestor_node_genes": _array_record(pair.ancestor.node_genes),
            "ancestor_connection_genes": _array_record(pair.ancestor.connection_genes),
            "descendant_node_genes": _array_record(pair.descendant.node_genes),
            "descendant_connection_genes": _array_record(pair.descendant.connection_genes),
        }
        for pair in pairs
    ]


def _lineage_from_record(records: Sequence[Mapping[str, Any]]) -> tuple[LineagePair, ...]:
    return tuple(
        LineagePair(
            founder_id=str(item["founder_id"]),
            ancestor_id=int(item["ancestor_id"]),
            descendant_id=int(item["descendant_id"]),
            ancestor=CPPNGenome(
                _array_from_record(item["ancestor_node_genes"]),
                _array_from_record(item["ancestor_connection_genes"]),
            ),
            descendant=CPPNGenome(
                _array_from_record(item["descendant_node_genes"]),
                _array_from_record(item["descendant_connection_genes"]),
            ),
        )
        for item in records
    )


def _paired_evaluation_record(
    manifest: ScenarioManifest,
    evaluation: PairedManifestEvaluation,
    episode_record: Callable[[Any, Any], dict[str, Any]],
) -> dict[str, Any]:
    return {
        "candidate_score": float(np.asarray(evaluation.candidate_score)),
        "integrity_valid": bool(np.asarray(evaluation.integrity_valid)),
        "repeat_scores": [float(value) for value in np.asarray(evaluation.repeat_scores)],
        "selected_repeat_index": int(np.asarray(evaluation.selected_repeat_index)),
        "pair_deltas": [float(value) for value in np.asarray(evaluation.pair_deltas)],
        "sham_auc_delta": float(np.asarray(evaluation.sham_auc_delta)),
        "shock_auc_delta": float(np.asarray(evaluation.shock_auc_delta)),
        "episodes": [episode_record(world, episode) for world, episode in zip(manifest.worlds, evaluation.episodes, strict=True)],
        "ancestor_episodes": [episode_record(world, episode) for world, episode in zip(manifest.worlds, evaluation.ancestor_episodes, strict=True)],
        "adaptive_observations": list(evaluation.adaptive_observations),
    }


def _episode_namespaces(
    manifest: ScenarioManifest,
    records: Any,
) -> tuple[SimpleNamespace, ...]:
    if not isinstance(records, list) or len(records) != len(manifest.worlds):
        raise ValueError("sealed episode records do not match the manifest")
    episodes = []
    for world, record in zip(manifest.worlds, records, strict=True):
        expected = {
            "scenario_id": world.scenario_id,
            "pair_id": world.pair_id,
            "scenario_family": world.scenario_family,
            "world_seed": world.world_seed,
            "event_kind": world.event_kind.value,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("sealed episode identity differs from the manifest")
        productivity = np.asarray(record.get("post_event_productivity"), dtype=np.float64)
        score = record.get("primary_score")
        if (
            productivity.ndim != 1
            or productivity.size != (manifest.horizon - world.event_step) // manifest.chunk_steps
            or not np.all(np.isfinite(productivity))
            or np.any((productivity < 0.0) | (productivity > 1.0))
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not np.isclose(score, np.mean(productivity), atol=1e-6, rtol=1e-6)
            or record.get("integrity_valid") is not True
        ):
            raise ValueError("sealed episode measurements are invalid")
        episodes.append(
            SimpleNamespace(
                primary_score=jnp.asarray(score, dtype=jnp.float32),
                operator_counts=jnp.asarray(record.get("operator_counts", [0] * 6)),
            )
        )
    return tuple(episodes)


def _evaluation_from_record(
    manifest: ScenarioManifest,
    value: Mapping[str, Any],
) -> PairedManifestEvaluation:
    episodes = _episode_namespaces(manifest, value.get("episodes"))
    ancestors = _episode_namespaces(manifest, value.get("ancestor_episodes"))
    repeats = np.asarray(value.get("repeat_scores"), dtype=np.float64)
    selected = value.get("selected_repeat_index")
    score = value.get("candidate_score")
    if (
        repeats.shape != (3,)
        or not np.all(np.isfinite(repeats))
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not np.isfinite(score)
        or not isinstance(selected, int)
        or isinstance(selected, bool)
        or not 0 <= selected < 3
        or selected != int(np.argsort(repeats, kind="stable")[1])
        or not np.isclose(score, repeats[selected], atol=1e-6, rtol=1e-6)
        or value.get("integrity_valid") is not True
    ):
        raise ValueError("sealed paired-repeat audit is invalid")
    candidate = ManifestEvaluation(episodes, jnp.asarray(0.0), jnp.asarray(True), jnp.asarray([0.0]), jnp.asarray(0))
    ancestor = ManifestEvaluation(ancestors, jnp.asarray(0.0), jnp.asarray(True), jnp.asarray([0.0]), jnp.asarray(0))
    recomputed, deltas, sham, shock = _paired_r4_score(manifest, candidate, ancestor, "punctuated")
    if (
        not np.isclose(recomputed, score, atol=1e-6, rtol=1e-6)
        or not np.allclose(deltas, value.get("pair_deltas"), atol=1e-6, rtol=1e-6)
        or not np.isclose(sham, value.get("sham_auc_delta"), atol=1e-6, rtol=1e-6)
        or not np.isclose(shock, value.get("shock_auc_delta"), atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("sealed paired score does not recompute")
    grouped: dict[str, list[int]] = {}
    for index, world in enumerate(manifest.worlds):
        grouped.setdefault(world.pair_id, []).append(index)
    control_indices = {_resolve_pair_indices(manifest, indices)[0] for indices in grouped.values()}
    sham_episodes = tuple(episode for index, episode in enumerate(episodes) if index in control_indices)
    return PairedManifestEvaluation(
        episodes=episodes,
        sham_episodes=sham_episodes,
        ancestor_episodes=ancestors,
        candidate_score=jnp.asarray(score, dtype=jnp.float32),
        integrity_valid=jnp.asarray(True),
        repeat_scores=jnp.asarray(repeats, dtype=jnp.float32),
        selected_repeat_index=jnp.asarray(selected, dtype=jnp.int32),
        pair_deltas=jnp.asarray(deltas),
        sham_auc_delta=jnp.asarray(sham, dtype=jnp.float32),
        shock_auc_delta=jnp.asarray(shock, dtype=jnp.float32),
        adaptive_observations=tuple(value.get("adaptive_observations", ())),
    )


def run_sealed_policy_worker(
    suite_path: Path,
    policy_label: str,
    *,
    require_gpu: bool = True,
    evaluator: Callable[..., PairedManifestEvaluation] = evaluate_manifest_paired_delta,
) -> Path:
    """Evaluate one predeclared policy against its exact initial ancestor."""
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("sealed policy evaluation requires the GPU backend")
    suite, suite_hash = _load_sealed_suite(suite_path)
    item = _work_item(suite, policy_label)
    if item["kind"] not in {"policy", "conditional_policy"}:
        raise ValueError("sealed policy worker received non-policy work")
    decision = None
    if item["kind"] == "conditional_policy":
        decision = _load_json(suite_path.parent / suite["mechanism_decision"]["result_name"])
        authorized = decision.get(
            "run_ablations",
            decision.get("run_mechanism"),
        )
        if decision.get("suite_sha256") != suite_hash or authorized is not True:
            raise WorkflowStopped("conditional mechanism worker is not authorized")
    run_spec_module, boundary, freezer = _shinka_modules()
    spec_path = Path(suite["run_spec"]["path"])
    spec, spec_hash, _ = run_spec_module.load_run_spec(spec_path)
    if spec_hash != suite["run_spec"]["sha256"]:
        raise ValueError("sealed suite has the wrong run-spec hash")
    _authenticate_sealed_runtime(spec, run_spec_module)
    manifest = _load_bound_sealed_manifest(suite)
    config = SimulatorConfig()
    founder_path = Path(suite["founder_index"]["path"])
    # Only sealed founders are readable in this epoch.  Each founder actually
    # used by the evaluator is hash-verified by the existing founder loader.
    index = load_founder_index(founder_path, verify_artifacts=False)
    if founder_index_sha256(index) != spec["founder_index"]["sha256"]:
        raise ValueError("sealed founder index differs from the run profile")
    ancestor_item = _work_item(suite, "exact_initial")
    ancestor_policy = _policy_from_item(suite, ancestor_item, spec, boundary)
    policy = _policy_from_item(
        suite,
        item,
        spec,
        boundary,
        decision=decision,
    )
    evaluation = evaluator(
        manifest,
        config,
        policy,
        ancestor_policy,
        regime="punctuated",
        founder_index_path=founder_path,
        numerical_repeats=int(spec["numerical_repeats"]),
        capture_finalists=bool(item["capture_lineage"]),
    )
    if not bool(np.asarray(evaluation.integrity_valid)):
        raise RuntimeError("sealed policy evaluation failed integrity")
    analysis = _sealed_analysis_module(suite)
    lineage_extractor = analysis.extract_lineage_pairs if suite.get("schema_version") == 1 else analysis.extract_relocation_lineage_pairs
    lineages = lineage_extractor(manifest, evaluation) if item["capture_lineage"] else ()
    record = {
        "schema_version": 1,
        "complete": True,
        "work_id": policy_label,
        "suite_sha256": suite_hash,
        "source_sha256": item["source_sha256"],
        "run_spec_sha256": spec_hash,
        "manifest_sha256": manifest_sha256(manifest),
        "founder_index_sha256": founder_index_sha256(index),
        "simulator_config_sha256": simulator_config_sha256(config),
        "simulator_source_sha256": simulator_source_sha256(),
        "backend": jax.default_backend(),
        "numerical_repeats": int(spec["numerical_repeats"]),
        "evaluation": _paired_evaluation_record(
            manifest,
            evaluation,
            freezer._episode_record,
        ),
        "lineage_pairs": _lineage_record(lineages),
    }
    output = suite_path.parent / item["result_name"]
    _write_once(output, _canonical_json_bytes(record))
    return output


def load_sealed_policy_result(
    suite_path: Path,
    policy_label: str,
) -> tuple[PairedManifestEvaluation, tuple[LineagePair, ...], dict[str, Any]]:
    """Strictly authenticate and reconstruct one compact paired result."""
    suite, suite_hash = _load_sealed_suite(suite_path)
    item = _work_item(suite, policy_label)
    path = suite_path.parent / item["result_name"]
    raw = path.read_bytes()
    record = _load_json(path)
    if raw != _canonical_json_bytes(record):
        raise ValueError("sealed policy result is not canonical JSON")
    expected = {
        "schema_version": 1,
        "complete": True,
        "work_id": policy_label,
        "suite_sha256": suite_hash,
        "source_sha256": item["source_sha256"],
        "run_spec_sha256": suite["run_spec"]["sha256"],
        "manifest_sha256": suite["sealed_manifest"]["sha256"],
        "founder_index_sha256": suite["founder_index"]["sha256"],
        "simulator_config_sha256": suite["simulator_config_sha256"],
        "simulator_source_sha256": simulator_source_sha256(),
        "backend": "gpu",
        "numerical_repeats": 3,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"sealed result does not authenticate: {policy_label}")
    manifest = _load_bound_sealed_manifest(suite)
    evaluation = _evaluation_from_record(manifest, record.get("evaluation", {}))
    lineage_records = record.get("lineage_pairs")
    if not isinstance(lineage_records, list):
        raise ValueError("sealed lineage record must be a list")
    pairs = _lineage_from_record(lineage_records)
    if not item["capture_lineage"] and pairs:
        raise ValueError("sealed lineage capture disagrees with the suite")
    return evaluation, pairs, record


def commit_mechanism_decision(suite_path: Path) -> dict[str, Any]:
    """Freeze the primary pass/skip and dominant action before ablation workers."""
    suite, suite_hash = _load_sealed_suite(suite_path)
    decision_path = suite_path.parent / suite["mechanism_decision"]["result_name"]
    primary, lineage_pairs, primary_record = load_sealed_policy_result(
        suite_path,
        suite["primary_policy"],
    )
    clone, _, _ = load_sealed_policy_result(suite_path, "clone")
    manifest = _load_bound_sealed_manifest(suite)
    analysis = _sealed_analysis_module(suite)
    primary_summary = analysis.primary_result_summary(manifest, primary, clone)
    primary_item = _work_item(suite, suite["primary_policy"])
    adaptive = suite.get("adaptive_policy") == suite["primary_policy"] and primary_item.get("adaptive_eligible") is True
    shock_counts = np.zeros(6, dtype=np.int64)
    shock_indices = _sealed_shock_indices(suite, manifest)
    for index, episode in enumerate(
        primary_record["evaluation"]["episodes"],
    ):
        if index in shock_indices:
            counts = np.asarray(episode["operator_counts"], dtype=np.int64)
            if counts.shape != (6,) or np.any(counts < 0):
                raise ValueError("primary operator counts are invalid")
            shock_counts += counts
    operator_observations_available = int(np.sum(shock_counts)) > 0
    dominant = int(np.argmax(shock_counts)) if operator_observations_available else None
    lineage_pairs_available = bool(lineage_pairs)
    mechanism_estimable = lineage_pairs_available and operator_observations_available
    if suite.get("schema_version") == 2:
        # R6 separates performance evidence from mechanism evidence.  A
        # positive preregistered primary receives its common-garden test even
        # when it is not the independently selected adaptive finalist.  The
        # ablations remain conditional on the primary itself being adaptive.
        run_common_garden = bool(primary_summary["passed"] and lineage_pairs_available)
        run_ablations = bool(run_common_garden and adaptive and operator_observations_available)
        run_mechanism = run_ablations
        if not primary_summary["passed"]:
            mechanism_status = "skipped_primary_not_positive"
            reason = "primary-not-positive"
        elif not lineage_pairs_available:
            mechanism_status = "partial_no_lineage_pairs"
            reason = "no-lineage-pairs"
        elif not adaptive:
            mechanism_status = "common_garden_only_not_adaptive_eligible"
            reason = "primary-not-adaptive-eligible"
        elif not operator_observations_available:
            mechanism_status = "common_garden_only_no_operator_counts"
            reason = "no-operator-counts"
        else:
            mechanism_status = "run_common_garden_and_ablations"
            reason = "primary-positive-adaptive-and-estimable"
    else:
        run_mechanism = bool(primary_summary["passed"] and adaptive and mechanism_estimable)
        run_common_garden = run_mechanism
        run_ablations = run_mechanism
        if run_mechanism:
            mechanism_status = "run"
            reason = "primary-positive-and-adaptive-eligible"
        elif primary_summary["passed"] and adaptive and not lineage_pairs_available:
            mechanism_status = "not_estimable_no_lineage_pairs"
            reason = "no-lineage-pairs"
        elif primary_summary["passed"] and adaptive and not operator_observations_available:
            mechanism_status = "not_estimable_no_operator_counts"
            reason = "no-operator-counts"
        elif primary_summary["passed"]:
            mechanism_status = "skipped_not_adaptive_eligible"
            reason = "primary-not-adaptive-eligible"
        else:
            mechanism_status = "skipped_primary_not_positive"
            reason = "primary-not-positive"
    decision = {
        "schema_version": 1,
        "complete": True,
        "suite_sha256": suite_hash,
        "primary_policy": suite["primary_policy"],
        "primary": primary_summary,
        "adaptive_eligible": adaptive,
        "lineage_pairs_available": lineage_pairs_available,
        "operator_observations_available": operator_observations_available,
        "mechanism_estimable": mechanism_estimable,
        "run_common_garden": run_common_garden,
        "run_ablations": run_ablations,
        "run_mechanism": run_mechanism,
        "status": mechanism_status,
        "reason": reason,
        "dominant_operator": dominant,
        "shock_operator_counts": shock_counts.tolist(),
    }
    _write_once(decision_path, _canonical_json_bytes(decision))
    return decision


def run_sealed_analysis_worker(
    suite_path: Path,
    *,
    require_gpu_for_mechanism: bool = True,
) -> Path:
    """Run the already-bound final analysis, including conditional common garden."""
    suite, suite_hash = _load_sealed_suite(suite_path)
    run_spec_module, _, _ = _shinka_modules()
    spec, spec_hash, _ = run_spec_module.load_run_spec(Path(suite["run_spec"]["path"]))
    if spec_hash != suite["run_spec"]["sha256"]:
        raise ValueError("sealed analysis loaded the wrong run specification")
    _authenticate_sealed_runtime(spec, run_spec_module)
    decision = commit_mechanism_decision(suite_path)
    run_common_garden = decision.get(
        "run_common_garden",
        decision["run_mechanism"],
    )
    run_ablations = decision.get("run_ablations", decision["run_mechanism"])
    if run_common_garden and require_gpu_for_mechanism and jax.default_backend() != "gpu":
        raise RuntimeError("common-garden mechanism analysis requires GPU")
    manifest = _load_bound_sealed_manifest(suite)
    primary, lineage_pairs, _ = load_sealed_policy_result(
        suite_path,
        suite["primary_policy"],
    )
    clone, _, _ = load_sealed_policy_result(suite_path, "clone")
    ablations = None
    pairs: tuple[LineagePair, ...] = ()
    multiplier = None
    if run_ablations:
        ablations = {label: load_sealed_policy_result(suite_path, label)[0] for label in ("no_credit_ablation", "dominant_action_ablation")}
    if run_common_garden:
        pairs = lineage_pairs
        if not pairs:
            raise RuntimeError("common-garden analysis requires captured lineage pairs")
        if suite.get("schema_version") == 1:
            multiplier = suite["disturbance_multiplier"]
    analysis = _sealed_analysis_module(suite)
    analysis_kwargs = {
        "lineage_pairs": pairs,
        "ablations": ablations,
        "adaptive_eligible": bool(decision["adaptive_eligible"]),
    }
    if suite.get("schema_version") == 1:
        analysis_kwargs["multiplier"] = multiplier
    result = analysis.run_final_analysis(
        manifest,
        primary,
        clone,
        **analysis_kwargs,
    )
    if suite.get("schema_version") == 1 and decision["status"].startswith("not_estimable_"):
        result["mechanism"] = {
            "status": "not_estimable",
            "reason": decision["reason"].replace("-", "_"),
            "claim_supported": False,
        }
    elif suite.get("schema_version") == 1 and decision["status"] == "skipped_not_adaptive_eligible":
        result["mechanism"] = {
            "status": "skipped_not_adaptive_eligible",
            "claim_supported": False,
        }
    item = _work_item(suite, "final_analysis")
    record = {
        "schema_version": 1,
        "complete": True,
        "work_id": "final_analysis",
        "suite_sha256": suite_hash,
        "source_sha256": item["source_sha256"],
        "mechanism_decision_sha256": _sha256_file(suite_path.parent / suite["mechanism_decision"]["result_name"]),
        "result": result,
    }
    output = suite_path.parent / item["result_name"]
    _write_once(output, _canonical_json_bytes(record))
    return output


def precommit_sealed_suite(
    suite: Mapping[str, Any],
    *,
    path: Path = SEALED_SUITE_PATH,
) -> str:
    """Publish or authenticate the one logical sealed-access suite record."""
    payload = _canonical_json_bytes(suite)
    _write_once(path, payload)
    return _sha256_bytes(payload)


def sealed_worker_command(suite_path: Path, work_id: str) -> list[str]:
    """Construct a fresh 24-GiB worker command; construction is CPU-testable."""
    suite = _load_json(suite_path)
    matches = [item for item in suite.get("work", []) if item.get("work_id") == work_id]
    if len(matches) != 1:
        raise KeyError(f"unknown sealed work ID: {work_id!r}")
    arguments = matches[0].get("worker_arguments")
    if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
        raise ValueError("sealed worker arguments are invalid")
    return _guarded_python_command(*arguments)


def _validate_completed_sealed_work(
    result_path: Path,
    item: Mapping[str, Any],
    suite_sha256: str,
) -> None:
    record = _load_json(result_path)
    expected = {
        "work_id": item["work_id"],
        "suite_sha256": suite_sha256,
        "source_sha256": item["source_sha256"],
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError(f"sealed result does not authenticate: {item['work_id']}")
    if record.get("complete") is not True:
        raise ValueError(f"sealed result is not complete: {item['work_id']}")


def _validate_sealed_analysis_result(suite_path: Path) -> dict[str, Any]:
    suite, suite_hash = _load_sealed_suite(suite_path)
    item = _work_item(suite, "final_analysis")
    result_path = suite_path.parent / item["result_name"]
    raw = result_path.read_bytes()
    record = _load_json(result_path)
    if raw != _canonical_json_bytes(record):
        raise ValueError("sealed final analysis is not canonical JSON")
    decision_path = suite_path.parent / suite["mechanism_decision"]["result_name"]
    expected = {
        "schema_version": 1,
        "complete": True,
        "work_id": "final_analysis",
        "suite_sha256": suite_hash,
        "source_sha256": item["source_sha256"],
        "mechanism_decision_sha256": _sha256_file(decision_path),
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("sealed final analysis does not authenticate")
    result = record.get("result")
    if (
        not isinstance(result, Mapping)
        or result.get("manifest_sha256") != suite["sealed_manifest"]["sha256"]
        or result.get("simulator_config_sha256") != suite["simulator_config_sha256"]
    ):
        raise ValueError("sealed final analysis has the wrong protocol bindings")
    return record


def run_missing_sealed_work(
    *,
    suite_path: Path = SEALED_SUITE_PATH,
    kinds: Sequence[str] = ("policy",),
    launcher: Callable[..., Any] = subprocess.run,
) -> tuple[Path, ...]:
    """Resume only selected predeclared work after core authentication."""
    suite, suite_sha256 = _load_sealed_suite(suite_path)
    selected_kinds = set(kinds)
    if not selected_kinds <= {"policy", "conditional_policy", "analysis"}:
        raise ValueError("sealed work kind is invalid")
    outputs = []
    for item in suite.get("work", []):
        if item["kind"] not in selected_kinds:
            continue
        output = suite_path.parent / item["result_name"]
        if output.exists():
            _validate_completed_sealed_work(output, item, suite_sha256)
            outputs.append(output)
            continue
        launcher(
            sealed_worker_command(suite_path, item["work_id"]),
            cwd=MICROCOSMOS_ROOT,
            check=True,
        )
        if not output.is_file():
            raise RuntimeError(f"sealed worker did not publish {item['work_id']!r}")
        _validate_completed_sealed_work(output, item, suite_sha256)
        outputs.append(output)
    return tuple(outputs)


def _development_finalist_sources(
    run_spec_module: Any,
    spec: dict[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    *,
    schema_version: int = 4,
) -> tuple[Path, dict[str, Path], str, str | None, tuple[str, ...]]:
    structured = run_spec_module.structured_random_paths(spec).frozen / "main.py"
    finalists: dict[str, Path] = {}
    eligible: set[str] = set()
    for regime in ("stable", "punctuated"):
        root = run_spec_module.paths_for(spec, regime).frozen
        unrestricted_label = f"{regime}_unrestricted"
        finalists[unrestricted_label] = root / "unrestricted/main.py"
        adaptive_label = f"{regime}_adaptive"
        if adaptive_label in records:
            unrestricted_hash = records[unrestricted_label]["candidate_source_sha256"]
            adaptive_hash = records[adaptive_label]["candidate_source_sha256"]
            if adaptive_hash == unrestricted_hash:
                eligible.add(unrestricted_label)
            else:
                finalists[adaptive_label] = root / "adaptive/main.py"
                eligible.add(adaptive_label)
    if schema_version == 5:
        # R6 preregisters the highest-ranked noninitial punctuated descendant as
        # the sole confirmatory primary.  The independently selected adaptive
        # mechanism finalist remains descriptive and must never replace it.
        primary = "punctuated_unrestricted"
        adaptive = primary if primary in eligible else "punctuated_adaptive" if "punctuated_adaptive" in finalists else None
    elif "punctuated_adaptive" in finalists:
        primary = "punctuated_adaptive"
        adaptive: str | None = primary
    else:
        primary = "punctuated_unrestricted"
        adaptive = primary if primary in eligible else None
    return structured, finalists, primary, adaptive, tuple(sorted(eligible))


def run_sealed_epoch(
    *,
    suite_path: Path | None = None,
    launcher: Callable[..., Any] = subprocess.run,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> dict[str, Any]:
    """Execute or authentically resume the one logical sealed-access epoch."""
    suite_path = context.sealed_suite_path if suite_path is None else suite_path
    loaded = _load_r5_run_spec() if context is R5_CONTEXT else _load_run_spec(context)
    run_spec_module, _, spec, spec_hash, _ = loaded
    context_kwargs = {} if context is R5_CONTEXT else {"context": context}
    records = _load_completed_development(
        run_spec_module,
        spec,
        spec_hash,
        **context_kwargs,
    )
    structured, finalists, primary, adaptive, eligible = _development_finalist_sources(
        run_spec_module,
        spec,
        records,
        schema_version=context.schema_version,
    )
    _assert_holdout_locked(spec, "development", **context_kwargs)
    if context.sealed_suite_schema_version == 1:
        suite = build_r5_sealed_suite(
            run_spec_path=context.profile_path,
            structured_random_source=structured,
            finalist_sources=finalists,
            primary_policy_label=primary,
            adaptive_policy_label=adaptive,
            adaptive_eligible_labels=eligible,
            suite_path=suite_path,
        )
    else:
        builder = importlib.import_module(context.workflow_module).build_r6_sealed_suite
        suite = builder(
            context=context,
            run_spec_path=context.profile_path,
            structured_random_source=structured,
            finalist_sources=finalists,
            primary_policy_label=primary,
            adaptive_policy_label=adaptive,
            adaptive_eligible_labels=eligible,
            suite_path=suite_path,
        )
    precommit_sealed_suite(suite, path=suite_path)
    loaded, _ = _load_sealed_suite(suite_path)
    if loaded != suite:
        raise ValueError("sealed suite differs from its frozen construction")
    summary_path = suite_path.parent / _work_item(suite, "final_analysis")["result_name"]
    if summary_path.is_file():
        manifest_path, _ = _holdout_paths(spec, "sealed", **context_kwargs)
        if os.access(manifest_path, os.R_OK):
            _authenticate_open_holdout(spec, "sealed", **context_kwargs)
            _lock_holdout(spec, "sealed", **context_kwargs)
        else:
            _assert_holdout_locked(spec, "sealed", **context_kwargs)
        return _validate_sealed_analysis_result(suite_path)
    try:
        _enter_holdout_epoch(
            spec,
            "sealed",
            resume_artifact=suite_path,
            **context_kwargs,
        )

        run_missing_sealed_work(
            suite_path=suite_path,
            kinds=("policy",),
            launcher=launcher,
        )
        for label in suite["expected_policies"]:
            load_sealed_policy_result(suite_path, label)

        decision = commit_mechanism_decision(suite_path)
        if decision.get("run_ablations", decision["run_mechanism"]):
            run_missing_sealed_work(
                suite_path=suite_path,
                kinds=("conditional_policy",),
                launcher=launcher,
            )
            for label in ("no_credit_ablation", "dominant_action_ablation"):
                load_sealed_policy_result(suite_path, label)

        run_missing_sealed_work(
            suite_path=suite_path,
            kinds=("analysis",),
            launcher=launcher,
        )
        return _validate_sealed_analysis_result(suite_path)
    finally:
        _lock_holdout(spec, "sealed", **context_kwargs)
        _assert_holdout_locked(spec, "development", **context_kwargs)
        _assert_holdout_locked(spec, "sealed", **context_kwargs)


def run_phase5(
    *,
    launcher: Callable[..., Any] = subprocess.run,
    context: BoundedWorkflowContext = R5_CONTEXT,
) -> dict[str, Any]:
    """Run the single development epoch followed by the single sealed epoch."""
    run_development_phase(launcher=launcher, context=context)
    return run_sealed_epoch(launcher=launcher, context=context)


def run_all_phases(
    *,
    launcher: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Execute the frozen r5 protocol in its sole dependency order."""
    run_pre_search_phases(launcher=launcher)
    run_search_phase(launcher=launcher)
    return run_phase5(launcher=launcher)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("world-cpu-stage")
    cpu.add_argument("--output", type=Path, required=True)
    gpu = commands.add_parser("world-gpu-stage")
    gpu.add_argument("--cpu-stage", type=Path, required=True)
    gpu.add_argument("--output", type=Path, required=True)
    qualify = commands.add_parser("qualify-worlds")
    qualify.add_argument("--output", type=Path, default=WORLD_QUALIFICATION_PATH)
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
    opportunity.add_argument("--output", type=Path, required=True)
    policy = commands.add_parser("sealed-policy-worker")
    policy.add_argument("--suite", type=Path, required=True)
    policy.add_argument("--policy", required=True)
    analysis = commands.add_parser("sealed-analysis-worker")
    analysis.add_argument("--suite", type=Path, required=True)
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


def _require_cpu_controller(command: str) -> None:
    if command in {"prepare", "search", "development", "sealed", "phase-5", "run"}:
        if not sys.dont_write_bytecode:
            raise RuntimeError("the r5 controller must be launched with python -B")
        platforms = os.environ.get("JAX_PLATFORMS", "")
        if platforms != "cpu":
            raise RuntimeError("the long-lived r5 controller must use JAX_PLATFORMS=cpu; its scientific workers select GPU in fresh guarded processes")


def _require_gpu_worker(command: str) -> None:
    if (
        command
        in {
            "world-gpu-stage",
            "founder-gate-worker",
            "disturbance-gate-worker",
            "opportunity-gate-worker",
            "sealed-policy-worker",
            "sealed-analysis-worker",
            "gpu-shinka-worker",
        }
        and jax.default_backend() != "gpu"
    ):
        raise RuntimeError(f"{command} requires the GPU backend")


def main() -> None:
    arguments = _parse_args()
    _require_cpu_controller(arguments.command)
    _require_gpu_worker(arguments.command)
    if arguments.command == "world-cpu-stage":
        _write_cpu_world_stage(arguments.output)
    elif arguments.command == "world-gpu-stage":
        _write_gpu_world_stage(arguments.cpu_stage, arguments.output)
    elif arguments.command == "qualify-worlds":
        result = run_world_qualification(output_path=arguments.output)
        print(json.dumps(result, sort_keys=True, indent=2))
    elif arguments.command == "founder-gate-worker":
        _run_founder_gate_worker(
            arguments.world_qualification,
            arguments.bank_directory,
            arguments.screening_record,
        )
    elif arguments.command == "disturbance-gate-worker":
        from .tools import run_disturbance_qualification

        run_disturbance_qualification(
            arguments.founder_index,
            arguments.world_qualification,
            output_path=arguments.output,
        )
    elif arguments.command == "opportunity-gate-worker":
        from .tools import run_opportunity_qualification

        run_opportunity_qualification(
            arguments.founder_index,
            arguments.training_manifest,
            disturbance_qualification_path=arguments.disturbance,
            output_path=arguments.output,
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
        print(json.dumps(run_search_phase(), sort_keys=True, indent=2))
    elif arguments.command == "development":
        print(json.dumps(run_development_phase(), sort_keys=True, indent=2))
    elif arguments.command == "sealed":
        print(json.dumps(run_sealed_epoch(), sort_keys=True, indent=2))
    elif arguments.command == "phase-5":
        print(json.dumps(run_phase5(), sort_keys=True, indent=2))
    else:
        print(json.dumps(run_all_phases(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
