from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import jax.numpy as jnp
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.episode import (  # noqa: E402
    EpisodeResult,
    ManifestEvaluation,
    SimulatorConfig,
    simulator_config_sha256,
)
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    CONTROLLER_LAYOUT,
    EventKind,
    EventRecord,
    NullEventParameters,
    RandomBottleneckParameters,
    ScenarioManifest,
    WorldScenario,
    canonical_manifest_bytes,
    manifest_sha256,
)
from experiments.evo2_sealed import workflow  # noqa: E402


def _manifest() -> ScenarioManifest:
    common = {
        "world_seed": 7,
        "pair_id": "sealed-pair-7",
        "scenario_family": "bottleneck",
        "event_step": 10,
    }
    return ScenarioManifest(
        schema_version=1,
        partition="sealed_final",
        controller_layout=CONTROLLER_LAYOUT,
        simulator_config_sha256=simulator_config_sha256(SimulatorConfig()),
        horizon=30,
        chunk_steps=5,
        worlds=(
            WorldScenario(
                scenario_id="sealed-null-7",
                event_kind=EventKind.NULL,
                event_parameters=NullEventParameters(),
                **common,
            ),
            WorldScenario(
                scenario_id="sealed-shock-7",
                event_kind=EventKind.RANDOM_BOTTLENECK,
                event_parameters=RandomBottleneckParameters(0.5),
                **common,
            ),
        ),
    )


def _run_spec(manifest: ScenarioManifest) -> dict:
    trusted = workflow._trusted_source_hashes()
    return {
        "schema_version": 1,
        "run_id": "synthetic-matched-run",
        "generations": 3,
        "outer_seed": 17,
        "model": "headless/test-model",
        "headless_command": "npx -y @roberttlange/headless@0.4.0",
        "top_k": workflow.SELECTION_TOP_K,
        "numerical_repeats": workflow.NUMERICAL_REPEATS,
        "baselines": list(workflow.BASELINE_NAMES),
        "manifest_sha256": {
            "training_stable": "1" * 64,
            "training_punctuated": "2" * 64,
            "development": "3" * 64,
            "sealed": manifest_sha256(manifest),
        },
        "simulator_config_sha256": simulator_config_sha256(SimulatorConfig()),
        "source_sha256": {
            key: trusted[key]
            for key in (
                "initial",
                "evaluator",
                "simulator",
                "analysis",
                "baseline",
                "dependency_lock",
                "preregistration",
            )
        },
        "archive": {"size": 1},
        "bootstrap": {
            "confidence": 0.95,
            "replicates": workflow.BOOTSTRAP_REPLICATES,
            "seed": workflow.BOOTSTRAP_SEED,
        },
        "search": {"budget": "matched"},
    }


def _frozen_dir(
    root: Path,
    regime: str,
    source: bytes,
    manifest: ScenarioManifest,
) -> Path:
    frozen = root / regime
    frozen.mkdir(parents=True)
    (frozen / "main.py").write_bytes(source)
    run_spec = _run_spec(manifest)
    trusted_sources = workflow._trusted_source_hashes()
    matched_counts = {
        arm: {
            "generation_budget": 3,
            "completed_evaluation_count": 3,
            "correct_evaluation_count": 2,
            "full_evaluation_count": 2,
            "invalid_proposal_count": 1,
            "integrity_failed_evaluation_count": 0,
            "database_sha256": ("4" if arm == "stable" else "5") * 64,
            "command_record_sha256": {"launch.json": ("6" if arm == "stable" else "7") * 64},
        }
        for arm in ("stable", "punctuated")
    }
    candidate_hash = hashlib.sha256(source).hexdigest()
    record = {
        "schema_version": 2,
        "run_id": run_spec["run_id"],
        "regime": regime,
        "run_spec": run_spec,
        "run_spec_sha256": workflow._run_spec_sha256(run_spec),
        "selection_rule": workflow._expected_selection_rule(),
        "candidate_source_sha256": candidate_hash,
        "development_score": 0.5,
        "development_integrity_valid": True,
        "numerical_repeats": workflow.NUMERICAL_REPEATS,
        "repeat_scores": [0.49, 0.5, 0.51],
        "selected_repeat_index": 1,
        "simulator_config_sha256": simulator_config_sha256(SimulatorConfig()),
        "sealed_manifest_sha256": manifest_sha256(manifest),
        "sealed_manifest_locked": True,
        "source_sha256": trusted_sources,
        "preregistration_complete": True,
        "preregistration_sha256": trusted_sources["preregistration"],
        "matched_run_counts": matched_counts,
        "bootstrap_config": run_spec["bootstrap"],
        "ranked_development_candidates": [
            {
                "candidate_source_sha256": digest,
                "development_score": score,
                "development_integrity_valid": True,
                "numerical_repeats": workflow.NUMERICAL_REPEATS,
                "repeat_scores": [score - 0.01, score, score + 0.01],
                "selected_repeat_index": 1,
            }
            for digest, score in (
                (candidate_hash, 0.5),
                ("8" * 64, 0.4),
                ("9" * 64, 0.3),
            )
        ],
    }
    (frozen / "freeze_record.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return frozen


def _episode(score: float) -> EpisodeResult:
    valid = jnp.asarray(True)
    return EpisodeResult(
        post_event_productivity=jnp.asarray([score] * 4, dtype=jnp.float32),
        primary_score=jnp.asarray(score, dtype=jnp.float32),
        survived=valid,
        final_alive=jnp.asarray(4),
        minimum_population=jnp.asarray(2),
        maximum_generation=jnp.asarray(3),
        birth_count=jnp.asarray(5),
        natural_death_count=jnp.asarray(2),
        operator_counts=jnp.asarray([0, 0, 0, 5]),
        operator_probability_sum=jnp.asarray([0, 0, 0, 5], dtype=jnp.float32),
        pre_selection_probability_sum=jnp.zeros(4),
        post_selection_probability_sum=jnp.asarray([0, 0, 0, 5], dtype=jnp.float32),
        pre_selection_probability_count=jnp.asarray(0),
        post_selection_probability_count=jnp.asarray(5),
        resolved_success_count=jnp.zeros(6, dtype=jnp.int32),
        resolved_failure_count=jnp.zeros(6, dtype=jnp.int32),
        distinct_birth_count=jnp.asarray(0),
        resolved_distinct_success_count=jnp.zeros(6, dtype=jnp.int32),
        operator_success_ema=jnp.full(6, 0.5),
        operator_usage_ema=jnp.zeros(6),
        operator_evidence_ema=jnp.zeros(6),
        pre_event_birth_count=jnp.asarray(2, dtype=jnp.int32),
        post_event_birth_count=jnp.asarray(3, dtype=jnp.int32),
        post_event_distinct_birth_count=jnp.asarray(0),
        pre_event_resolved_count=jnp.zeros(6, dtype=jnp.int32),
        post_event_resolved_count=jnp.zeros(6, dtype=jnp.int32),
        post_event_resolved_distinct_success_count=jnp.zeros(6, dtype=jnp.int32),
        pre_event_productivity=jnp.asarray(0.1),
        finite=valid,
        identity_valid=valid,
        events_valid=valid,
        infrastructure_valid=valid,
        operator_accounting_valid=valid,
        policy_violation_count=jnp.asarray(0),
        integrity_valid=valid,
        event_record=EventRecord(
            event_code=jnp.asarray(0),
            alive_before=jnp.asarray(4),
            alive_after=jnp.asarray(4),
            catastrophe_death_count=jnp.asarray(0),
            targeted_founder_lineage=jnp.asarray(-1),
        ),
        shock_population=None,
        final_population=None,
    )


def _candidate_function(parent_genome, parent_stats, population_stats, rng):
    del parent_genome, parent_stats, population_stats, rng
    return jnp.asarray([0.0, 0.0, 0.0, 1.0])


def _record(
    manifest: ScenarioManifest,
    label: str,
    scores: tuple[float, float],
    *,
    kind: str,
    source_sha256: str | None = None,
) -> dict:
    episodes = []
    for world, score in zip(manifest.worlds, scores, strict=True):
        episodes.append(
            {
                "scenario_id": world.scenario_id,
                "pair_id": world.pair_id,
                "scenario_family": world.scenario_family,
                "world_seed": world.world_seed,
                "event_kind": world.event_kind.value,
                "primary_score": score,
                "post_event_productivity": [score] * 4,
                "survived": True,
                "integrity_valid": True,
            }
        )
    record = {
        "schema_version": 1,
        "policy_name": label,
        "policy_kind": kind,
        "policy_source_sha256": source_sha256 or hashlib.sha256(label.encode("utf-8")).hexdigest(),
        "manifest_sha256": manifest_sha256(manifest),
        "simulator_config_sha256": manifest.simulator_config_sha256,
        "simulator_source_sha256": workflow.simulator_source_sha256(),
        "candidate_score": sum(scores) / len(scores),
        "integrity_valid": True,
        "numerical_repeats": workflow.NUMERICAL_REPEATS,
        "repeat_scores": [
            sum(scores) / len(scores) - 0.01,
            sum(scores) / len(scores),
            sum(scores) / len(scores) + 0.01,
        ],
        "selected_repeat_index": 1,
        "backend": "gpu",
        "episodes": episodes,
    }
    if kind == "candidate":
        record["training_regime"] = "punctuated" if "punctuated" in label else "stable"
    return record


def _patch_final_hash(monkeypatch: pytest.MonkeyPatch, manifest: ScenarioManifest) -> None:
    monkeypatch.setattr(workflow, "SEALED_MANIFEST_SHA256", manifest_sha256(manifest))


def test_sealed_loader_checks_canonical_bytes_and_frozen_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    path = tmp_path / "final.json"
    path.write_bytes(canonical_manifest_bytes(manifest) + b"\n")
    _patch_final_hash(monkeypatch, manifest)
    assert workflow.load_sealed_manifest(path) == manifest
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="canonical JSON"):
        workflow.load_sealed_manifest(path)


def test_frozen_directory_and_fixed_shinka_boundary_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    _patch_final_hash(monkeypatch, manifest)
    initial = workflow.TRUSTED_SOURCE_PATHS["initial"].read_bytes()
    frozen = _frozen_dir(tmp_path, "stable", initial, manifest)
    finalist = workflow.load_frozen_finalist(frozen, "stable")
    loaded = workflow.load_frozen_candidate(finalist)
    assert loaded.finalist == finalist
    assert callable(loaded.policy)

    (frozen / "main.py").write_bytes(initial + b"\n")
    with pytest.raises(ValueError, match="does not match main.py"):
        workflow.load_frozen_finalist(frozen, "stable")


def test_matched_freeze_records_and_trusted_sources_are_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    _patch_final_hash(monkeypatch, manifest)
    stable = _frozen_dir(tmp_path, "stable", b"stable", manifest)
    punctuated = _frozen_dir(tmp_path, "punctuated", b"punctuated", manifest)
    record_path = punctuated / "freeze_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["run_spec"]["outer_seed"] = 18
    record["run_spec_sha256"] = workflow._run_spec_sha256(record["run_spec"])
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree on run_spec"):
        workflow.load_frozen_finalists(stable, punctuated)


def test_worker_requires_global_ledger_and_resumes_only_missing_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    _patch_final_hash(monkeypatch, manifest)
    stable = _frozen_dir(tmp_path, "stable", b"stable", manifest)
    punctuated = _frozen_dir(tmp_path, "punctuated", b"punctuated", manifest)
    result_dir = tmp_path / "results"
    finalists = workflow.load_frozen_finalists(stable, punctuated)
    dynamic_namespace = {}
    exec(  # noqa: S102 - reproduces the validated synthetic candidate module
        compile(
            "def make_offspring(*args, **kwargs):\n    return None\n",
            "<validated-dynamic-candidate>",
            "exec",
        ),
        dynamic_namespace,
    )
    dynamic_candidate = dynamic_namespace["make_offspring"]
    loaded = {
        finalist.label: workflow._LoadedCandidate(
            finalist,
            SimpleNamespace(make_offspring=dynamic_candidate),
            _candidate_function,
        )
        for finalist in finalists
    }
    validation_calls = []

    def load_candidate(finalist):
        validation_calls.append(finalist.label)
        return loaded[finalist.label]

    monkeypatch.setattr(workflow, "load_frozen_candidate", load_candidate)
    monkeypatch.setattr(workflow, "validate_frozen_candidate_source", lambda _: None)

    def evaluate(*_, numerical_repeats):
        assert numerical_repeats == workflow.NUMERICAL_REPEATS
        return ManifestEvaluation(
            episodes=(_episode(0.5), _episode(0.25)),
            candidate_score=jnp.asarray(0.375),
            integrity_valid=jnp.asarray(True),
            repeat_scores=jnp.asarray([0.36, 0.375, 0.39]),
            selected_repeat_index=jnp.asarray(1),
        )

    with pytest.raises(RuntimeError, match="has not been precommitted"):
        workflow.run_sealed_policy(
            stable,
            punctuated,
            "shinka_stable",
            result_dir=result_dir,
            require_gpu=False,
            manifest_loader=lambda: manifest,
            evaluator=evaluate,
        )
    workflow.precommit_sealed_suite(
        stable,
        punctuated,
        result_dir=result_dir,
        manifest_loader=lambda: manifest,
    )
    output = workflow.run_sealed_policy(
        stable,
        punctuated,
        "shinka_stable",
        result_dir=result_dir,
        require_gpu=False,
        manifest_loader=lambda: manifest,
        evaluator=evaluate,
    )
    assert output == result_dir / "shinka_stable.json"
    assert validation_calls == ["shinka_stable", "shinka_punctuated"] * 2
    with pytest.raises(FileExistsError):
        workflow.run_sealed_policy(
            stable,
            punctuated,
            "shinka_stable",
            result_dir=result_dir,
            require_gpu=False,
            manifest_loader=lambda: manifest,
            evaluator=evaluate,
        )


def test_run_all_launches_six_fresh_guarded_children_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    _patch_final_hash(monkeypatch, manifest)
    stable = _frozen_dir(tmp_path, "stable", b"stable", manifest)
    punctuated = _frozen_dir(tmp_path, "punctuated", b"punctuated", manifest)
    result_dir = tmp_path / "results"
    launched = []
    monkeypatch.setattr(workflow, "validate_frozen_candidate_source", lambda _: None)

    def launch(command, *, cwd, check):
        assert cwd == workflow.MICROCOSMOS_ROOT
        assert check is True
        assert command[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
        assert "MemoryMax=24G" in command
        policy = command[command.index("--policy") + 1]
        suite = json.loads((result_dir / "suite_record.json").read_text(encoding="utf-8"))
        finalist_hashes = {finalist["label"]: finalist["source_sha256"] for finalist in suite["finalists"]}
        record = _record(
            manifest,
            policy,
            (0.5, 0.4),
            kind="candidate" if policy.startswith("shinka_") else "baseline",
            source_sha256=finalist_hashes.get(policy)
            or hashlib.sha256(workflow.inspect.getsource(workflow.baseline_policy(policy)).encode("utf-8")).hexdigest(),
        )
        workflow.write_json_atomic(result_dir / f"{policy}.json", record)
        launched.append(command)

    outputs = workflow.run_all_sealed(
        stable,
        punctuated,
        result_dir=result_dir,
        manifest_loader=lambda: manifest,
        launcher=launch,
    )
    assert len(outputs) == len(workflow.BASELINE_NAMES) + 2
    assert len(launched) == len(outputs)
    workflow.run_all_sealed(
        stable,
        punctuated,
        result_dir=result_dir,
        manifest_loader=lambda: manifest,
        launcher=lambda *_args, **_kwargs: pytest.fail("completed policy relaunched"),
    )
    alternate = tmp_path / "alternate"
    alternate_stable = _frozen_dir(alternate, "stable", b"different-stable", manifest)
    alternate_punctuated = _frozen_dir(
        alternate,
        "punctuated",
        b"different-punctuated",
        manifest,
    )
    with pytest.raises(RuntimeError, match="different sealed suite"):
        workflow.run_all_sealed(
            alternate_stable,
            alternate_punctuated,
            result_dir=result_dir,
            manifest_loader=lambda: manifest,
            launcher=lambda *_args, **_kwargs: pytest.fail("second suite launched"),
        )


def test_summary_uses_direct_shocked_auc_contrasts_and_secondary_dod() -> None:
    manifest = _manifest()
    records = {
        "shinka_stable": _record(manifest, "shinka_stable", (0.5, 0.2), kind="candidate"),
        "shinka_punctuated": _record(manifest, "shinka_punctuated", (0.5, 0.4), kind="candidate"),
        "clone": _record(manifest, "clone", (0.5, 0.1), kind="baseline"),
        "fixed_parametric": _record(manifest, "fixed_parametric", (0.5, 0.25), kind="baseline"),
        "fixed_mixed": _record(manifest, "fixed_mixed", (0.5, 0.3), kind="baseline"),
        "stress_responsive": _record(manifest, "stress_responsive", (0.5, 0.35), kind="baseline"),
    }
    summary = workflow.summarize_sealed_records(
        manifest,
        records,
        bootstrap_seed=3,
        bootstrap_replicates=100,
    )
    contrasts = summary["primary_shocked_world_auc_contrasts"]
    assert contrasts["punctuated_minus_stable"]["mean"] == pytest.approx(0.2)
    assert contrasts["punctuated_minus_fixed_mixed"]["mean"] == pytest.approx(0.1)
    assert contrasts["punctuated_minus_stress_responsive"]["mean"] == pytest.approx(0.05)
    secondary = summary["champion_comparison"]["secondary_punctuated_minus_stable_difference_in_differences"]
    assert secondary["mean"] == pytest.approx(0.2)


@pytest.mark.parametrize("tamper", ["score", "length", "range", "repeats"])
def test_result_loader_recomputes_and_bounds_productivity(tamper: str) -> None:
    manifest = _manifest()
    record = _record(manifest, "clone", (0.5, 0.2), kind="baseline")
    if tamper == "score":
        record["episodes"][0]["primary_score"] = 0.6
    elif tamper == "length":
        record["episodes"][0]["post_event_productivity"].pop()
    elif tamper == "range":
        record["episodes"][0]["post_event_productivity"][0] = 1.1
    else:
        record["selected_repeat_index"] = 0
    with pytest.raises(ValueError):
        workflow._result_evaluation(manifest, record)
