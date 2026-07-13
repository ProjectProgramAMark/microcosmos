"""Run and summarize the one-time sealed Evo² evaluation.

This module deliberately reuses the frozen ecosystem evaluator, the Shinka task's
candidate boundary, and the paired-effect analysis.  It contains no simulator or
heredity implementation of its own.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import hmac
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

from microcosmos.heredity import mutate_cppn

from experiments.evo2_ecosystem.analysis import (
    FounderWorldObservation,
    PairedWorldMetrics,
    hierarchical_paired_effect_summary,
    paired_world_metrics,
    stratified_equal_weight_summary,
)
from experiments.evo2_ecosystem.episode import (
    ManifestEvaluation,
    SimulatorConfig,
    evaluate_manifest,
    simulator_config_sha256,
    simulator_source_sha256,
)
from experiments.evo2_ecosystem.frozen_config import NUMERICAL_REPEATS
from experiments.evo2_ecosystem.founder_artifacts import (
    founder_index_sha256,
    load_founder_index,
)
from experiments.evo2_ecosystem.protocol import (
    ScenarioManifest,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)
from experiments.evo2_ecosystem.run_baselines import (
    BASELINE_NAMES,
    baseline_policy,
    baseline_result_record,
    write_json_atomic,
)


MICROCOSMOS_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = MICROCOSMOS_ROOT.parent
SHINKA_ROOT = PROJECT_ROOT / "ShinkaEvolve"
SEALED_MANIFEST_PATH = (
    MICROCOSMOS_ROOT
    / "experiments/evo2_ecosystem/heredity_adaptation_v3/manifests/sealed.json"
)
SEALED_MANIFEST_SHA256 = "29449c7a96497cf52009a865f91c6be649a3027ea699dd401e68bbf0efc362fa"
SEALED_RESULTS_DIR = (
    MICROCOSMOS_ROOT
    / "experiments/evo2_ecosystem/heredity_adaptation_v3/final_results"
)
BOOTSTRAP_SEED = 20_260_712
BOOTSTRAP_REPLICATES = 10_000
SELECTION_TOP_K = 3
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

TRUSTED_SOURCE_PATHS = {
    "sealed_workflow": Path(__file__).resolve(),
    "initial": SHINKA_ROOT / "examples/evo2_ecosystem/initial.py",
    "evaluator": SHINKA_ROOT / "examples/evo2_ecosystem/evaluate.py",
    "launcher": SHINKA_ROOT / "examples/evo2_ecosystem/run_evo.py",
    "finalist_selector": SHINKA_ROOT / "examples/evo2_ecosystem/freeze_finalist.py",
    "run_spec_module": SHINKA_ROOT / "examples/evo2_ecosystem/run_spec.py",
    "analysis": MICROCOSMOS_ROOT / "experiments/evo2_ecosystem/analysis.py",
    "baseline": MICROCOSMOS_ROOT / "experiments/evo2_ecosystem/run_baselines.py",
    "dependency_lock": MICROCOSMOS_ROOT / "uv.lock",
    "heredity": MICROCOSMOS_ROOT / "src/microcosmos/heredity.py",
    "preregistration": PROJECT_ROOT / "plans/experiment-analysis-plan.md",
}

V2_TRUSTED_SOURCE_PATHS = {
    "initial": SHINKA_ROOT / "examples/evo2_ecosystem/initial.py",
    "evaluator": SHINKA_ROOT / "examples/evo2_ecosystem/evaluate.py",
    "launcher": SHINKA_ROOT / "examples/evo2_ecosystem/run_evo.py",
    "finalist_selector": SHINKA_ROOT / "examples/evo2_ecosystem/freeze_finalist.py",
    "lineage_selector": SHINKA_ROOT / "examples/evo2_ecosystem/program_lineage.py",
    "run_spec_module": SHINKA_ROOT / "examples/evo2_ecosystem/run_spec.py",
    "analysis": MICROCOSMOS_ROOT / "experiments/evo2_ecosystem/analysis.py",
    "baseline": MICROCOSMOS_ROOT / "experiments/evo2_ecosystem/run_baselines.py",
    "dependency_lock": MICROCOSMOS_ROOT / "uv.lock",
}

INITIAL_POLICY_LABEL = "initial_scheduler"


@dataclass(frozen=True)
class FrozenFinalist:
    """One validated finalist directory and its immutable selection record."""

    frozen_dir: Path
    regime: str
    source_path: Path
    source_sha256: str
    freeze_record: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_dir, Path) or not isinstance(self.source_path, Path):
            raise ValueError("finalist paths must be Path values")
        if not _SHA256.fullmatch(self.source_sha256):
            raise ValueError("finalist source_sha256 must be a lowercase SHA-256")
        if self.regime not in {"stable", "punctuated"}:
            raise ValueError("finalist regime must be 'stable' or 'punctuated'")

    @property
    def label(self) -> str:
        return f"shinka_{self.regime}"


@dataclass(frozen=True)
class _LoadedCandidate:
    finalist: FrozenFinalist
    module: ModuleType
    policy: Callable[..., Any]


def load_sealed_manifest(path: Path | None = None) -> ScenarioManifest:
    """Load the final manifest only through its hard-coded integrity boundary."""
    path = SEALED_MANIFEST_PATH if path is None else path
    raw = path.read_bytes()
    manifest = manifest_from_json_bytes(raw)
    if raw != canonical_manifest_bytes(manifest) + b"\n":
        raise RuntimeError("sealed manifest is not stored as canonical JSON")
    if manifest_sha256(manifest) != SEALED_MANIFEST_SHA256:
        raise RuntimeError("sealed manifest does not match its frozen SHA-256")
    if manifest.partition != "sealed_final":
        raise RuntimeError("sealed manifest has the wrong partition")
    return manifest


def _run_spec_sha256(value: Any) -> str:
    module = _load_run_spec_module()
    return module.sha256_bytes(module.canonical_json_bytes(value))


def _load_run_spec_module() -> ModuleType:
    root = str(SHINKA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("examples.evo2_ecosystem.run_spec")


def _trusted_source_hashes(spec: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Hash the exact source set named by a v1 or v2 run specification."""
    paths = (
        V2_TRUSTED_SOURCE_PATHS
        if spec is not None and spec.get("schema_version") == 2
        else TRUSTED_SOURCE_PATHS
    )
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"trusted source is missing: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes["simulator"] = simulator_source_sha256()
    if spec is not None and spec.get("schema_version") == 2:
        module = _load_run_spec_module()
        for name in ("preregistration", "implementation_plan"):
            path = module.artifact_path(spec[name])
            if not path.is_file():
                raise RuntimeError(f"trusted source is missing: {name}")
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _validate_bound_file(binding: Mapping[str, Any], label: str) -> Path:
    """Resolve and authenticate one project-relative v2 artifact binding."""
    module = _load_run_spec_module()
    path = module.artifact_path(dict(binding))
    if not path.is_file():
        raise ValueError(f"bound {label} is missing")
    if label == "founder index":
        digest = founder_index_sha256(load_founder_index(path, verify_artifacts=True))
    elif label.endswith("manifest"):
        raw = path.read_bytes()
        manifest = manifest_from_json_bytes(raw)
        if raw != canonical_manifest_bytes(manifest) + b"\n":
            raise ValueError(f"bound {label} is not canonical JSON")
        digest = manifest_sha256(manifest)
    else:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != binding.get("sha256"):
        raise ValueError(f"bound {label} has the wrong SHA-256")
    return path


def _validate_run_spec(spec: Any) -> None:
    if not isinstance(spec, dict):
        raise ValueError("run_spec must be an object")
    _load_run_spec_module()._validate(spec)
    module = _load_run_spec_module()
    if (
        spec.get("top_k") != SELECTION_TOP_K
        or spec.get("numerical_repeats") != NUMERICAL_REPEATS
        or spec.get("simulator_config_sha256") != simulator_config_sha256(SimulatorConfig())
        or module.manifest_hash(spec, "sealed") != SEALED_MANIFEST_SHA256
        or spec.get("bootstrap")
        != {
            "confidence": 0.95,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        }
    ):
        raise ValueError("run_spec does not match the sealed protocol")
    trusted = _trusted_source_hashes(spec)
    expected_sources = {key: trusted[key] for key in spec["source_sha256"]}
    if spec.get("source_sha256") != expected_sources:
        raise ValueError("run_spec trusted sources do not match the codebase")
    if spec.get("schema_version") == 2:
        for role, binding in spec["manifests"].items():
            _validate_bound_file(binding, f"{role} manifest")
        _validate_bound_file(spec["founder_index"], "founder index")
        _validate_bound_file(spec["preregistration"], "preregistration")
        _validate_bound_file(spec["implementation_plan"], "implementation plan")


def _validate_matched_run_counts(value: Any, spec: Mapping[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"stable", "punctuated"}:
        raise ValueError("matched run completion audit is missing")
    for regime, counts in value.items():
        if not isinstance(counts, dict):
            raise ValueError(f"{regime} completion audit is invalid")
        generation_budget = spec["generations"]
        completed = counts.get("completed_evaluation_count")
        full = counts.get("full_evaluation_count")
        invalid = counts.get("invalid_proposal_count")
        correct = counts.get("correct_evaluation_count")
        command_hashes = counts.get("command_record_sha256")
        if (
            counts.get("generation_budget") != generation_budget
            or completed != generation_budget
            or not all(isinstance(item, int) for item in (full, invalid, correct))
            or full + invalid != completed
            or not 0 <= correct <= full
            or counts.get("integrity_failed_evaluation_count") != full - correct
            or not _SHA256.fullmatch(str(counts.get("database_sha256", "")))
            or not isinstance(command_hashes, dict)
            or not command_hashes
            or any(not isinstance(name, str) or not name or not _SHA256.fullmatch(str(digest)) for name, digest in command_hashes.items())
        ):
            raise ValueError(f"{regime} outer search was not completely preregistered")


def _expected_selection_rule() -> dict[str, Any]:
    return {
        "name": "development_score_desc_then_generation_asc",
        "training_rank": "descending combined_score, then generation",
        "top_k": SELECTION_TOP_K,
        "development_rank": "integrity, median repeated score, then generation",
        "numerical_repeats": NUMERICAL_REPEATS,
    }


def _validate_repeat_metadata(value: Mapping[str, Any], expected_score: float) -> None:
    repeats = value.get("repeat_scores")
    selected = value.get("selected_repeat_index")
    if (
        value.get("numerical_repeats") != NUMERICAL_REPEATS
        or not isinstance(repeats, list)
        or len(repeats) != NUMERICAL_REPEATS
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in repeats)
        or not isinstance(selected, int)
        or isinstance(selected, bool)
        or not 0 <= selected < NUMERICAL_REPEATS
    ):
        raise ValueError("development repeat metadata is invalid")
    array = np.asarray(repeats, dtype=np.float64)
    median_index = int(np.argsort(array, kind="stable")[NUMERICAL_REPEATS // 2])
    if selected != median_index or not np.isclose(expected_score, array[selected]):
        raise ValueError("development score is not the coherent median repeat")


def load_frozen_finalist(frozen_dir: Path, regime: str) -> FrozenFinalist:
    """Validate one frozen directory without accepting a loose source path."""
    frozen_dir = frozen_dir.resolve()
    source_path = frozen_dir / "main.py"
    record_path = frozen_dir / "freeze_record.json"
    if not source_path.is_file() or not record_path.is_file():
        raise ValueError(f"{regime} frozen directory is incomplete")
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{regime} freeze_record.json is invalid") from error
    if not isinstance(record, dict) or record.get("schema_version") != 2:
        raise ValueError(f"{regime} freeze record has an unsupported schema")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if record.get("regime") != regime or record.get("candidate_source_sha256") != source_hash:
        raise ValueError(f"{regime} freeze record does not match main.py")
    development_score = record.get("development_score")
    if isinstance(development_score, bool) or not isinstance(development_score, (int, float)) or not math.isfinite(development_score):
        raise ValueError(f"{regime} finalist has an invalid development score")
    run_spec = record.get("run_spec")
    _validate_run_spec(run_spec)
    trusted = record.get("source_sha256")
    if trusted != _trusted_source_hashes(run_spec):
        raise ValueError(f"{regime} trusted source hashes do not match the codebase")
    if record.get("run_spec_sha256") != _run_spec_sha256(run_spec):
        raise ValueError(f"{regime} preregistered run specification hash is invalid")
    preregistration_hash = (
        run_spec["preregistration"]["sha256"]
        if run_spec.get("schema_version") == 2
        else trusted["preregistration"]
    )
    if (
        record.get("selection_rule") != _expected_selection_rule()
        or record.get("bootstrap_config") != run_spec["bootstrap"]
        or record.get("sealed_manifest_sha256") != SEALED_MANIFEST_SHA256
        or record.get("sealed_manifest_locked") is not True
        or record.get("preregistration_complete") is not True
        or record.get("preregistration_sha256") != preregistration_hash
        or record.get("development_integrity_valid") is not True
        or record.get("simulator_config_sha256") != simulator_config_sha256(SimulatorConfig())
    ):
        raise ValueError(f"{regime} freeze record violates the preregistration")
    _validate_matched_run_counts(record.get("matched_run_counts"), run_spec)
    _validate_repeat_metadata(record, float(development_score))
    ranked = record.get("ranked_development_candidates")
    if (
        not isinstance(ranked, list)
        or len(ranked) != SELECTION_TOP_K
        or ranked[0].get("candidate_source_sha256") != source_hash
        or ranked[0].get("development_integrity_valid") is not True
        or not np.isclose(ranked[0].get("development_score"), development_score)
    ):
        raise ValueError(f"{regime} selected finalist is not the ranked valid winner")
    for item in ranked:
        score = item.get("development_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"{regime} ranked finalist score is invalid")
        _validate_repeat_metadata(item, float(score))
    return FrozenFinalist(
        frozen_dir=frozen_dir,
        regime=regime,
        source_path=source_path,
        source_sha256=source_hash,
        freeze_record=record,
    )


def _validate_matched_finalists(
    stable: FrozenFinalist,
    punctuated: FrozenFinalist,
) -> tuple[FrozenFinalist, FrozenFinalist]:
    for field in (
        "run_spec",
        "run_spec_sha256",
        "source_sha256",
        "preregistration_complete",
        "selection_rule",
        "bootstrap_config",
        "matched_run_counts",
        "sealed_manifest_sha256",
    ):
        if stable.freeze_record.get(field) != punctuated.freeze_record.get(field):
            raise ValueError(f"frozen finalists disagree on {field}")
    return stable, punctuated


def load_frozen_finalists(
    stable_dir: Path,
    punctuated_dir: Path,
) -> tuple[FrozenFinalist, FrozenFinalist]:
    """Load the two matched development-selected finalists."""
    return _validate_matched_finalists(
        load_frozen_finalist(stable_dir, "stable"),
        load_frozen_finalist(punctuated_dir, "punctuated"),
    )


def _load_shinka_boundary() -> ModuleType:
    evaluator_path = TRUSTED_SOURCE_PATHS["evaluator"]
    initial_path = evaluator_path.with_name("initial.py")
    if not evaluator_path.is_file() or not initial_path.is_file():
        raise RuntimeError("Shinka Evo² candidate boundary is unavailable")
    root = str(SHINKA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    digest = hashlib.sha256(str(evaluator_path.resolve()).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(
        f"evo2_sealed_candidate_boundary_{digest}",
        evaluator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Shinka Evo² candidate boundary cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_candidate(finalist: FrozenFinalist) -> _LoadedCandidate:
    """Hash-check, sandbox-validate, and compile-smoke-test one finalist."""
    source = finalist.source_path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if not hmac.compare_digest(actual, finalist.source_sha256):
        raise RuntimeError(f"candidate {finalist.label!r} does not match its frozen hash")
    boundary = _load_shinka_boundary()
    module = boundary._load_candidate(finalist.source_path)
    boundary._smoke_validate_candidate(module)
    policy = boundary._build_policy(module, mutate_cppn)
    return _LoadedCandidate(finalist, module, policy)


def validate_frozen_candidate_source(finalist: FrozenFinalist) -> None:
    """Run the fixed AST/import boundary without initializing a JAX backend."""
    _load_shinka_boundary()._load_candidate(finalist.source_path)


def _suite_record(
    finalists: tuple[FrozenFinalist, FrozenFinalist],
    manifest: ScenarioManifest,
    config: SimulatorConfig,
) -> dict[str, Any]:
    stable, punctuated = _validate_matched_finalists(*finalists)
    run_spec = stable.freeze_record["run_spec"]
    if _load_run_spec_module().manifest_hash(run_spec, "sealed") != manifest_sha256(manifest):
        raise ValueError("finalist run specification names a different final manifest")
    if run_spec["simulator_config_sha256"] != simulator_config_sha256(config):
        raise ValueError("finalist run specification names a different simulator")
    return {
        "schema_version": 2,
        "manifest_sha256": manifest_sha256(manifest),
        "simulator_config_sha256": simulator_config_sha256(config),
        "finalists": [
            {
                "label": finalist.label,
                "source_sha256": finalist.source_sha256,
                "freeze_record_sha256": hashlib.sha256((finalist.frozen_dir / "freeze_record.json").read_bytes()).hexdigest(),
                "regime": finalist.regime,
            }
            for finalist in (stable, punctuated)
        ],
        "run_spec": run_spec,
        "run_spec_sha256": stable.freeze_record["run_spec_sha256"],
        "source_sha256": stable.freeze_record["source_sha256"],
        "selection_rule": stable.freeze_record["selection_rule"],
        "matched_run_counts": stable.freeze_record["matched_run_counts"],
        "preregistration_complete": True,
        "preregistration_sha256": (
            run_spec["preregistration"]["sha256"]
            if run_spec.get("schema_version") == 2
            else run_spec["source_sha256"]["preregistration"]
        ),
        "expected_policies": [
            stable.label,
            punctuated.label,
            *(
                [INITIAL_POLICY_LABEL]
                if run_spec.get("schema_version") == 2
                else []
            ),
            *run_spec["baselines"],
        ],
    }


def _commit_or_match_suite_record(
    expected: dict[str, Any],
    *,
    result_dir: Path,
    allow_create: bool,
) -> None:
    suite_path = result_dir / "suite_record.json"
    if not result_dir.exists():
        if not allow_create:
            raise RuntimeError("global sealed suite has not been precommitted")
        result_dir.mkdir(parents=True)
        write_json_atomic(suite_path, expected)
        return
    if not result_dir.is_dir() or not suite_path.is_file():
        raise RuntimeError("global sealed result directory has no suite record")
    actual = json.loads(suite_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise RuntimeError("a different sealed suite was already precommitted")


def _load_suite_context(
    stable_dir: Path,
    punctuated_dir: Path,
    *,
    manifest_loader: Callable[[], ScenarioManifest],
    validate_candidate_code: bool = True,
) -> tuple[
    tuple[FrozenFinalist, FrozenFinalist],
    dict[str, _LoadedCandidate],
    ScenarioManifest,
    SimulatorConfig,
    dict[str, Any],
]:
    finalists = load_frozen_finalists(stable_dir, punctuated_dir)
    # Both bounded programs are validated before opening the final manifest.
    if validate_candidate_code:
        loaded = {finalist.label: load_frozen_candidate(finalist) for finalist in finalists}
    else:
        for finalist in finalists:
            validate_frozen_candidate_source(finalist)
        loaded = {}
    manifest = manifest_loader()
    if manifest.partition != "sealed_final" or manifest_sha256(manifest) != SEALED_MANIFEST_SHA256:
        raise RuntimeError("sealed loader returned an unrecognized final manifest")
    config = SimulatorConfig()
    if manifest.simulator_config_sha256 != simulator_config_sha256(config):
        raise RuntimeError("sealed manifest has the wrong simulator hash")
    suite = _suite_record(finalists, manifest, config)
    return finalists, loaded, manifest, config, suite


def run_sealed_policy(
    stable_dir: Path,
    punctuated_dir: Path,
    policy_label: str,
    *,
    result_dir: Path = SEALED_RESULTS_DIR,
    require_gpu: bool = True,
    manifest_loader: Callable[[], ScenarioManifest] = load_sealed_manifest,
    evaluator: Callable[..., ManifestEvaluation] = evaluate_manifest,
) -> Path:
    """Worker: evaluate one policy under an already-precommitted global suite."""
    if require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("sealed evaluation requires the GPU backend")
    _, loaded, manifest, config, suite = _load_suite_context(
        stable_dir,
        punctuated_dir,
        manifest_loader=manifest_loader,
    )
    if policy_label not in suite["expected_policies"]:
        raise ValueError(f"unknown sealed policy {policy_label!r}")
    _commit_or_match_suite_record(suite, result_dir=result_dir, allow_create=False)
    output = result_dir / f"{policy_label}.json"
    if output.exists():
        raise FileExistsError(f"sealed result for {policy_label!r} already exists")

    loaded_candidate = loaded.get(policy_label)
    initial_module = None
    if policy_label == INITIAL_POLICY_LABEL:
        boundary = _load_shinka_boundary()
        initial_path = TRUSTED_SOURCE_PATHS["initial"]
        expected_hash = suite["run_spec"]["source_sha256"]["initial"]
        if hashlib.sha256(initial_path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError("initial scheduler source no longer matches the run specification")
        initial_module = boundary._load_candidate(initial_path)
        boundary._smoke_validate_candidate(initial_module)
        policy = boundary._build_policy(initial_module, mutate_cppn)
        kind = "reference"
        source_callable = load_frozen_candidate
    elif loaded_candidate is None:
        kind = "baseline"
        policy = baseline_policy(policy_label)
        source_callable = policy
    else:
        kind = "candidate"
        policy = loaded_candidate.policy
        # ``baseline_result_record`` hashes an inspectable callable before this
        # worker replaces that provisional value with the frozen candidate hash.
        # Validated candidate functions are deliberately compiled under a
        # synthetic filename, so their source is not available to ``inspect``.
        source_callable = load_frozen_candidate
    evaluation_kwargs: dict[str, Any] = {"numerical_repeats": NUMERICAL_REPEATS}
    run_spec = suite["run_spec"]
    if run_spec.get("schema_version") == 2:
        evaluation_kwargs["founder_index_path"] = _load_run_spec_module().founder_index_path(run_spec)
    evaluation = evaluator(manifest, config, policy, **evaluation_kwargs)
    record = baseline_result_record(
        manifest,
        config,
        policy_label,
        source_callable,
        evaluation,
    )
    record["policy_kind"] = kind
    if initial_module is not None:
        record["policy_source_sha256"] = suite["run_spec"]["source_sha256"]["initial"]
    elif loaded_candidate is not None:
        frozen = loaded_candidate.finalist
        record["policy_source_sha256"] = frozen.source_sha256
        record["training_regime"] = frozen.regime
    write_json_atomic(output, record)
    return output


def precommit_sealed_suite(
    stable_dir: Path,
    punctuated_dir: Path,
    *,
    result_dir: Path = SEALED_RESULTS_DIR,
    manifest_loader: Callable[[], ScenarioManifest] = load_sealed_manifest,
) -> tuple[dict[str, Any], ScenarioManifest]:
    """Freeze the only global final ledger before launching any GPU child."""
    _, _, manifest, _, suite = _load_suite_context(
        stable_dir,
        punctuated_dir,
        manifest_loader=manifest_loader,
        validate_candidate_code=False,
    )
    _commit_or_match_suite_record(suite, result_dir=result_dir, allow_create=True)
    return suite, manifest


def _validate_completed_result(
    path: Path,
    label: str,
    suite: Mapping[str, Any],
    manifest: ScenarioManifest,
) -> None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing sealed result is invalid: {label}") from error
    if record.get("policy_name") != label:
        raise ValueError(f"existing sealed result has the wrong policy: {label}")
    finalist_hashes = {finalist["label"]: finalist["source_sha256"] for finalist in suite["finalists"]}
    expected_source_hash = finalist_hashes.get(label)
    if label == INITIAL_POLICY_LABEL:
        expected_source_hash = suite["run_spec"]["source_sha256"]["initial"]
    if expected_source_hash is None:
        expected_source_hash = hashlib.sha256(inspect.getsource(baseline_policy(label)).encode("utf-8")).hexdigest()
    if record.get("policy_source_sha256") != expected_source_hash:
        raise ValueError(f"existing sealed policy hash is wrong: {label}")
    _result_evaluation(manifest, record)


def run_all_sealed(
    stable_dir: Path,
    punctuated_dir: Path,
    *,
    result_dir: Path = SEALED_RESULTS_DIR,
    manifest_loader: Callable[[], ScenarioManifest] = load_sealed_manifest,
    launcher: Callable[..., Any] = subprocess.run,
) -> tuple[Path, ...]:
    """Launch every missing final policy in a fresh guarded GPU process."""
    suite, manifest = precommit_sealed_suite(
        stable_dir,
        punctuated_dir,
        result_dir=result_dir,
        manifest_loader=manifest_loader,
    )
    outputs = []
    for policy in suite["expected_policies"]:
        output = result_dir / f"{policy}.json"
        if output.exists():
            _validate_completed_result(output, policy, suite, manifest)
            outputs.append(output)
            continue
        command = [
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
            "env",
            "-u",
            "JAX_PLATFORMS",
            "-u",
            "JAX_PLATFORM_NAME",
            "conda",
            "run",
            "-n",
            "sakana",
            "python",
            "-m",
            "experiments.evo2_sealed.workflow",
            "worker",
            "--stable-frozen-dir",
            str(stable_dir.resolve()),
            "--punctuated-frozen-dir",
            str(punctuated_dir.resolve()),
            "--policy",
            policy,
        ]
        launcher(command, cwd=MICROCOSMOS_ROOT, check=True)
        if not output.is_file():
            raise RuntimeError(f"sealed worker did not publish {policy!r}")
        _validate_completed_result(output, policy, suite, manifest)
        outputs.append(output)
    return tuple(outputs)


def _result_evaluation(
    manifest: ScenarioManifest,
    record: Mapping[str, Any],
) -> ManifestEvaluation:
    """Validate compact result JSON and expose only fields used by analysis."""
    if record.get("schema_version") != 1:
        raise ValueError("result has an unsupported schema_version")
    if record.get("manifest_sha256") != manifest_sha256(manifest):
        raise ValueError("result manifest hash does not match the sealed manifest")
    if record.get("simulator_config_sha256") != manifest.simulator_config_sha256:
        raise ValueError("result simulator hash does not match the sealed manifest")
    if record.get("integrity_valid") is not True:
        raise ValueError("result failed ecosystem integrity checks")
    if record.get("backend") != "gpu":
        raise ValueError("sealed result was not produced on the GPU backend")
    if record.get("simulator_source_sha256") != simulator_source_sha256():
        raise ValueError("sealed result has the wrong simulator source hash")
    repeats = record.get("repeat_scores")
    selected_repeat = record.get("selected_repeat_index")
    if (
        record.get("numerical_repeats") != NUMERICAL_REPEATS
        or not isinstance(repeats, list)
        or len(repeats) != NUMERICAL_REPEATS
        or not all(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) for value in repeats)
        or not isinstance(selected_repeat, int)
        or isinstance(selected_repeat, bool)
        or not 0 <= selected_repeat < NUMERICAL_REPEATS
    ):
        raise ValueError("sealed result has invalid numerical-repeat metadata")
    raw_episodes = record.get("episodes")
    if not isinstance(raw_episodes, list) or len(raw_episodes) != len(manifest.worlds):
        raise ValueError("result episodes do not match the sealed manifest")

    episodes = []
    scores = []
    for world, raw in zip(manifest.worlds, raw_episodes, strict=True):
        if not isinstance(raw, dict):
            raise ValueError("result episode must be an object")
        expected = {
            "scenario_id": world.scenario_id,
            "pair_id": world.pair_id,
            "scenario_family": world.scenario_family,
            "world_seed": world.world_seed,
            "event_kind": world.event_kind.value,
        }
        if any(raw.get(key) != value for key, value in expected.items()):
            raise ValueError("result episode identity does not match the manifest")
        score = raw.get("primary_score")
        raw_productivity = raw.get("post_event_productivity")
        if not isinstance(raw_productivity, list):
            raise ValueError("result productivity must be a JSON array")
        productivity = np.asarray(raw_productivity, dtype=np.float64)
        expected_length = (manifest.horizon - world.event_step) // manifest.chunk_steps
        survived = raw.get("survived")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or productivity.ndim != 1
            or productivity.size != expected_length
            or not np.all(np.isfinite(productivity))
            or np.any(productivity < 0.0)
            or np.any(productivity > 1.0)
            or not isinstance(survived, bool)
            or raw.get("integrity_valid") is not True
        ):
            raise ValueError("result episode contains invalid measurements")
        recomputed_score = float(np.mean(productivity))
        if not np.isclose(score, recomputed_score, atol=1e-6, rtol=1e-6):
            raise ValueError("result primary_score is inconsistent with productivity")
        scores.append(recomputed_score)
        episodes.append(
            SimpleNamespace(
                primary_score=recomputed_score,
                post_event_productivity=productivity,
                survived=survived,
            )
        )
    candidate_score = record.get("candidate_score")
    if (
        isinstance(candidate_score, bool)
        or not isinstance(candidate_score, (int, float))
        or not math.isfinite(candidate_score)
        or not np.isclose(candidate_score, np.mean(scores), atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("result candidate_score is inconsistent with its episodes")
    repeat_array = np.asarray(repeats, dtype=np.float64)
    median_index = int(np.argsort(repeat_array, kind="stable")[NUMERICAL_REPEATS // 2])
    if selected_repeat != median_index or not np.isclose(
        candidate_score,
        repeat_array[selected_repeat],
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("sealed result is not the coherent median repeat")
    return ManifestEvaluation(
        episodes=tuple(episodes),
        candidate_score=candidate_score,
        integrity_valid=True,
        repeat_scores=repeat_array,
        selected_repeat_index=selected_repeat,
    )


def _summary_dict(summary: Any) -> dict[str, Any]:
    value = asdict(summary)
    value["confidence_interval"] = list(value["confidence_interval"])
    return value


def _stratified_summary(
    metrics: Sequence[PairedWorldMetrics],
    values: Sequence[float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    by_family: dict[str, list[float]] = {}
    for metric, value in zip(metrics, values, strict=True):
        by_family.setdefault(metric.scenario_family, []).append(value)
    return _summary_dict(
        stratified_equal_weight_summary(
            by_family,
            seed=seed,
            replicates=replicates,
        )
    )


def _founder_observations(
    manifest: ScenarioManifest,
    evaluation: ManifestEvaluation,
    *,
    shocked: bool,
) -> list[FounderWorldObservation]:
    observations = []
    for world, episode in zip(manifest.worlds, evaluation.episodes, strict=True):
        if (world.event_kind.value != "null") != shocked:
            continue
        observations.append(
            FounderWorldObservation(
                founder_id=world.founder_id or "legacy-founder",
                world_seed=world.world_seed,
                pair_id=world.pair_id,
                value=float(np.asarray(episode.primary_score)),
            )
        )
    return observations


def _paired_metric_observations(
    manifest: ScenarioManifest,
    metrics: Sequence[PairedWorldMetrics],
) -> list[FounderWorldObservation]:
    shocks = {
        world.pair_id: world
        for world in manifest.worlds
        if world.event_kind.value != "null"
    }
    return [
        FounderWorldObservation(
            founder_id=shocks[metric.pair_id].founder_id or "legacy-founder",
            world_seed=shocks[metric.pair_id].world_seed,
            pair_id=metric.pair_id,
            value=metric.primary_effect,
        )
        for metric in metrics
    ]


def _direct_policy_contrast(
    manifest: ScenarioManifest,
    candidate: ManifestEvaluation,
    comparator: ManifestEvaluation,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    if manifest.schema_version == 2:
        return _summary_dict(
            hierarchical_paired_effect_summary(
                _founder_observations(manifest, candidate, shocked=True),
                _founder_observations(manifest, comparator, shocked=True),
                seed=seed,
                replicates=replicates,
            )
        )
    metrics = paired_world_metrics(manifest, candidate)
    candidate_scores = {
        world.pair_id: float(np.asarray(episode.primary_score))
        for world, episode in zip(manifest.worlds, candidate.episodes, strict=True)
        if world.event_kind.value != "null"
    }
    comparator_scores = {
        world.pair_id: float(np.asarray(episode.primary_score))
        for world, episode in zip(manifest.worlds, comparator.episodes, strict=True)
        if world.event_kind.value != "null"
    }
    return _stratified_summary(
        metrics,
        [candidate_scores[item.pair_id] - comparator_scores[item.pair_id] for item in metrics],
        seed=seed,
        replicates=replicates,
    )


def summarize_sealed_records(
    manifest: ScenarioManifest,
    records: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20_260_712,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    """Build the preregistered paired final comparison and baseline table."""
    stable_champion = "shinka_stable"
    punctuated_champion = "shinka_punctuated"
    required_baselines = tuple(
        label
        for label, record in records.items()
        if record.get("policy_kind") == "baseline"
    )
    if stable_champion not in records or punctuated_champion not in records:
        raise ValueError("both champion records are required")
    if records[stable_champion].get("policy_kind") != "candidate":
        raise ValueError("stable champion must be a candidate result")
    if records[punctuated_champion].get("policy_kind") != "candidate":
        raise ValueError("punctuated champion must be a candidate result")
    if records[stable_champion].get("training_regime") != "stable":
        raise ValueError("stable champion record has the wrong training regime")
    if records[punctuated_champion].get("training_regime") != "punctuated":
        raise ValueError("punctuated champion record has the wrong training regime")
    if (
        manifest.schema_version == 2
        and records.get(INITIAL_POLICY_LABEL, {}).get("policy_kind") != "reference"
    ):
        raise ValueError("the frozen initial scheduler result is required")
    missing_baselines = [name for name in required_baselines if name not in records]
    if missing_baselines:
        raise ValueError(f"required baseline results are missing: {missing_baselines}")
    if any(records[name].get("policy_kind") != "baseline" for name in required_baselines):
        raise ValueError("required baseline records have the wrong policy kind")
    for label, record in records.items():
        if record.get("policy_name") != label:
            raise ValueError("result mapping label does not match policy_name")
        if not _SHA256.fullmatch(str(record.get("policy_source_sha256", ""))):
            raise ValueError("result has an invalid policy source hash")
    evaluations = {label: _result_evaluation(manifest, record) for label, record in records.items()}
    paired = {label: paired_world_metrics(manifest, evaluation) for label, evaluation in evaluations.items()}
    stable_by_pair = {item.pair_id: item for item in paired[stable_champion]}
    punctuated_by_pair = {item.pair_id: item for item in paired[punctuated_champion]}
    if stable_by_pair.keys() != punctuated_by_pair.keys():
        raise ValueError("champion pair identities do not match")
    ordered_ids = [item.pair_id for item in paired[stable_champion]]
    difference_metrics = [punctuated_by_pair[pair_id] for pair_id in ordered_ids]
    resilience_differences = [punctuated_by_pair[pair_id].primary_effect - stable_by_pair[pair_id].primary_effect for pair_id in ordered_ids]
    shock_scores = {}
    for label, evaluation in evaluations.items():
        shock_scores[label] = {
            world.pair_id: float(np.asarray(episode.primary_score))
            for world, episode in zip(manifest.worlds, evaluation.episodes, strict=True)
            if world.event_kind.value != "null"
        }
        if set(shock_scores[label]) != set(ordered_ids):
            raise ValueError("result shocked-world identities do not match")

    policy_summaries = {}
    baseline_table = []
    for label, evaluation in evaluations.items():
        metrics = paired[label]
        policy_seed = bootstrap_seed + int(
            hashlib.sha256(label.encode("utf-8")).hexdigest()[:8],
            16,
        )
        if manifest.schema_version == 2:
            effect = _summary_dict(
                hierarchical_paired_effect_summary(
                    _founder_observations(manifest, evaluation, shocked=True),
                    _founder_observations(manifest, evaluation, shocked=False),
                    seed=policy_seed,
                    replicates=bootstrap_replicates,
                )
            )
        else:
            effect = _stratified_summary(
                metrics,
                [item.primary_effect for item in metrics],
                seed=policy_seed,
                replicates=bootstrap_replicates,
            )
        survival = float(np.mean([float(np.asarray(episode.survived)) for episode in evaluation.episodes]))
        policy_summaries[label] = {
            "policy_kind": records[label].get("policy_kind"),
            "policy_source_sha256": records[label]["policy_source_sha256"],
            "candidate_score": float(np.asarray(evaluation.candidate_score)),
            "survival_rate": survival,
            "shock_minus_null_primary_effect": effect,
        }
        if records[label].get("policy_kind") == "baseline":
            baseline_table.append({"policy": label, **policy_summaries[label]})
    baseline_table.sort(key=lambda row: row["policy"])

    if manifest.schema_version == 2:
        comparison = _summary_dict(
            hierarchical_paired_effect_summary(
                _paired_metric_observations(manifest, paired[punctuated_champion]),
                _paired_metric_observations(manifest, paired[stable_champion]),
                seed=bootstrap_seed,
                replicates=bootstrap_replicates,
            )
        )
    else:
        comparison = _stratified_summary(
            difference_metrics,
            resilience_differences,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        )
    direct_contrasts = {}
    comparators = {"punctuated_minus_stable": stable_champion}
    if INITIAL_POLICY_LABEL in records:
        comparators = {
            "punctuated_minus_initial": INITIAL_POLICY_LABEL,
            **comparators,
        }
    comparators.update(
        {f"punctuated_minus_{name}": name for name in required_baselines}
    )
    for contrast_name, comparator in comparators.items():
        differences = [shock_scores[punctuated_champion][pair_id] - shock_scores[comparator][pair_id] for pair_id in ordered_ids]
        contrast = _direct_policy_contrast(
            manifest,
            evaluations[punctuated_champion],
            evaluations[comparator],
            seed=bootstrap_seed + int(hashlib.sha256(comparator.encode("utf-8")).hexdigest()[:8], 16),
            replicates=bootstrap_replicates,
        )
        contrast["paired_effects"] = [
            {
                "pair_id": metric.pair_id,
                "scenario_family": metric.scenario_family,
                "effect": effect,
            }
            for metric, effect in zip(difference_metrics, differences, strict=True)
        ]
        direct_contrasts[contrast_name] = contrast
    return {
        "schema_version": 2 if manifest.schema_version == 2 else 1,
        "manifest_sha256": manifest_sha256(manifest),
        "simulator_source_sha256": simulator_source_sha256(),
        "numerical_repeats": NUMERICAL_REPEATS,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_confidence": 0.95,
        "champions": {
            "stable": stable_champion,
            "punctuated": punctuated_champion,
        },
        "primary_shocked_world_auc_contrasts": direct_contrasts,
        "primary_contrast": direct_contrasts.get("punctuated_minus_initial"),
        "champion_comparison": {
            "candidate_score_difference": (policy_summaries[punctuated_champion]["candidate_score"] - policy_summaries[stable_champion]["candidate_score"]),
            "secondary_punctuated_minus_stable_difference_in_differences": comparison,
        },
        "policies": policy_summaries,
        "baseline_table": baseline_table,
    }


def _validated_suite_record(
    result_dir: Path,
    manifest: ScenarioManifest,
) -> dict[str, Any]:
    path = result_dir / "suite_record.json"
    if not path.is_file():
        raise ValueError("sealed suite record is missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "manifest_sha256",
        "simulator_config_sha256",
        "finalists",
        "run_spec",
        "run_spec_sha256",
        "source_sha256",
        "selection_rule",
        "matched_run_counts",
        "preregistration_complete",
        "preregistration_sha256",
        "expected_policies",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError("sealed suite record has an invalid schema")
    if record["schema_version"] != 2:
        raise ValueError("sealed suite record has an unsupported version")
    if record["manifest_sha256"] != manifest_sha256(manifest):
        raise ValueError("sealed suite record has the wrong manifest hash")
    if record["simulator_config_sha256"] != simulator_config_sha256(SimulatorConfig()):
        raise ValueError("sealed suite record has the wrong simulator hash")
    if record["source_sha256"] != _trusted_source_hashes(record["run_spec"]):
        raise ValueError("sealed suite record has stale trusted source hashes")
    _validate_run_spec(record["run_spec"])
    if record["run_spec_sha256"] != _run_spec_sha256(record["run_spec"]):
        raise ValueError("sealed suite record has the wrong run-spec hash")
    _validate_matched_run_counts(record["matched_run_counts"], record["run_spec"])
    if record["selection_rule"] != _expected_selection_rule():
        raise ValueError("sealed suite record has the wrong selection rule")
    if not _SHA256.fullmatch(str(record["preregistration_sha256"])):
        raise ValueError("sealed suite record has an invalid preregistration hash")
    if record["preregistration_complete"] is not True:
        raise ValueError("sealed suite record lacks completed preregistration")
    expected_preregistration = (
        record["run_spec"]["preregistration"]["sha256"]
        if record["run_spec"].get("schema_version") == 2
        else record["source_sha256"]["preregistration"]
    )
    if record["preregistration_sha256"] != expected_preregistration:
        raise ValueError("sealed suite record has the wrong preregistration hash")
    expected_policies = [
        "shinka_stable",
        "shinka_punctuated",
        *(
            [INITIAL_POLICY_LABEL]
            if record["run_spec"].get("schema_version") == 2
            else []
        ),
        *record["run_spec"]["baselines"],
    ]
    if record["expected_policies"] != expected_policies:
        raise ValueError("sealed suite record has the wrong policy set")
    finalists = record["finalists"]
    if (
        not isinstance(finalists, list)
        or len(finalists) != 2
        or [finalist.get("label") for finalist in finalists] != ["shinka_stable", "shinka_punctuated"]
        or [finalist.get("regime") for finalist in finalists] != ["stable", "punctuated"]
        or any(
            not _SHA256.fullmatch(str(finalist.get("source_sha256", ""))) or not _SHA256.fullmatch(str(finalist.get("freeze_record_sha256", "")))
            for finalist in finalists
        )
    ):
        raise ValueError("sealed suite record has invalid finalists")
    return record


def summarize_result_directory(
    result_dir: Path = SEALED_RESULTS_DIR,
    output: Path | None = None,
) -> dict[str, Any]:
    """Require a complete frozen suite and atomically publish its summary."""
    output = result_dir / "summary.json" if output is None else output
    if output.exists():
        raise FileExistsError("sealed summary output already exists")
    manifest = load_sealed_manifest()
    suite = _validated_suite_record(result_dir, manifest)
    bootstrap = suite["run_spec"]["bootstrap"]
    records: dict[str, Mapping[str, Any]] = {}
    for label in suite["expected_policies"]:
        path = result_dir / f"{label}.json"
        if not path.is_file():
            raise ValueError(f"sealed result is missing: {label}")
        _validate_completed_result(path, label, suite, manifest)
        record = json.loads(path.read_text(encoding="utf-8"))
        records[label] = record
    summary = summarize_sealed_records(
        manifest,
        records,
        bootstrap_seed=bootstrap["seed"],
        bootstrap_replicates=bootstrap["replicates"],
    )
    write_json_atomic(output, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_all = subparsers.add_parser("run-all", help="run or resume the fixed final suite")
    worker = subparsers.add_parser("worker", help="internal one-policy GPU worker")
    for command in (run_all, worker):
        command.add_argument("--stable-frozen-dir", type=Path, required=True)
        command.add_argument("--punctuated-frozen-dir", type=Path, required=True)
    worker.add_argument("--policy", required=True)
    subparsers.add_parser("summarize", help="summarize the complete fixed final suite")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "worker":
        output = run_sealed_policy(
            args.stable_frozen_dir,
            args.punctuated_frozen_dir,
            args.policy,
        )
        print(json.dumps({"output": str(output)}, sort_keys=True))
        return
    if args.command == "run-all":
        outputs = run_all_sealed(
            args.stable_frozen_dir,
            args.punctuated_frozen_dir,
        )
        print(json.dumps({"outputs": [str(path) for path in outputs]}, sort_keys=True))
        return
    summary = summarize_result_directory()
    print(json.dumps(summary["champion_comparison"], sort_keys=True))


if __name__ == "__main__":
    main()
