"""One-shot prospective development confirmation for actuator adaptation.

This module opens only the frozen development manifest.  It authenticates the
training calibration decision, reruns clone and the *already selected*
conventional mutation policy on exact middle-body injury/sham forks, and
publishes an immutable decision.  It cannot select a gain, policy, event time,
or founder and it has no sealed-manifest input.

Production execution is full-scale, fluid-enabled, and GPU-only.  Tests inject
policy results so the evidence boundary can be exercised on CPU without
running the simulator or weakening the production contract.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np

from microcosmos.rollout import run_ecosystem_chunk

from .analysis import (
    FounderWorldObservation,
    HierarchicalEffectSummary,
    hierarchical_paired_effect_summary,
)
from .calibrate_actuator import (
    CALIBRATION_RECORD_SCHEMA_VERSION,
    MUTATION_POLICIES,
    POLICIES,
    PRODUCTION_SPEC,
    CalibrationEvaluator,
    CalibrationWorldResult,
    PolicyCalibrationResult,
    build_calibration_manifest,
    production_spec_for_seed_panel,
)
from .episode import (
    SimulatorConfig,
    build_environment,
    simulator_config_sha256,
    simulator_source_sha256,
)
from .founder_artifacts import (
    FounderIndex,
    FounderRecord,
    founder_index_sha256,
    load_founder_artifact,
    load_founder_index,
)
from .heredity_adaptation.manifest_generator import (
    DEVELOPMENT_SEEDS,
    EVENT_STEPS,
    HINGE_COUNT,
)
from .protocol import (
    SCHEMA_VERSION,
    ActuatorInjuryParameters,
    EventKind,
    NullEventParameters,
    ScenarioManifest,
    WorldScenario,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)
from .readiness_seed_panel import (
    ReadinessSeedPanel,
    load_readiness_seed_panel,
)


CONFIRMATION_RECORD_SCHEMA_VERSION = 1
CONFIRMATION_DECISION_SCHEMA_VERSION = 1
CONFIRMATION_RULE_ID = "one-shot-middle-injury-transfer-v1"
EXPECTED_DEVELOPMENT_FOUNDERS = 2
EXPECTED_DEVELOPMENT_PAIRS = 8


@dataclass(frozen=True)
class CalibrationSelection:
    """Authenticated immutable choice made before development access."""

    gain: float
    schedule_id: str
    winning_policy: str
    manifest_sha256: str
    protocol_sha256: str
    calibration_record_sha256: str


@dataclass(frozen=True)
class DevelopmentGateDecision:
    """Preregistered qualitative transfer decision."""

    passed: bool
    clone_harm: HierarchicalEffectSummary
    mutation_advantage: HierarchicalEffectSummary
    winning_policy: str
    all_integrity_valid: bool
    clone_identity_valid: bool
    survival_fraction: float
    median_post_event_births: float
    median_generation_gain: float
    viability_passed: bool


PolicyRunner = Callable[[ScenarioManifest, str], PolicyCalibrationResult]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_sha256(value: object) -> str:
    return _sha256(inspect.getsource(value).encode("utf-8"))


def _verify_manifest_file(path: Path) -> ScenarioManifest:
    """Require canonical bytes and the manifest-style canonical hash sidecar."""
    payload = path.read_bytes()
    manifest = manifest_from_json_bytes(payload)
    expected_payload = canonical_manifest_bytes(manifest) + b"\n"
    if payload != expected_payload:
        raise ValueError(f"manifest is not canonically encoded: {path}")
    digest = manifest_sha256(manifest)
    sidecar = path.with_suffix(".sha256")
    expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
    if sidecar.read_bytes() != expected_sidecar:
        raise ValueError(f"manifest hash sidecar is invalid: {sidecar}")
    return manifest


def _parse_calibration_record(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load the append-only record while rejecting noncanonical or reordered lines."""
    payload = path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("calibration record must be nonempty and newline terminated")
    events: list[dict[str, Any]] = []
    for sequence, line in enumerate(payload.splitlines()):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("calibration record contains invalid JSON") from error
        if not isinstance(value, dict) or _canonical_json_bytes(value) != line:
            raise ValueError("calibration record lines must be canonical JSON objects")
        if value.get("schema_version") != CALIBRATION_RECORD_SCHEMA_VERSION:
            raise ValueError("calibration record schema version is not supported")
        if value.get("sequence") != sequence:
            raise ValueError("calibration record sequence is not contiguous")
        events.append(value)
    return events, _sha256(payload)


def authenticate_calibration_selection(
    *,
    calibration_record_path: Path,
    selected_manifest_path: Path,
    founder_index: FounderIndex,
    preregistration_path: Path,
    seed_panel: ReadinessSeedPanel | None = None,
    seed_panel_path: Path | None = None,
) -> CalibrationSelection:
    """Bind the selected gain/policy to the complete prior calibration record."""
    events, record_sha256 = _parse_calibration_record(calibration_record_path)
    if events[0].get("event") != "calibration_started":
        raise ValueError("calibration record must start with calibration_started")
    if events[-1].get("event") != "calibration_finished":
        raise ValueError("calibration record must end with calibration_finished")
    protocol = events[0].get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("calibration start record is missing its protocol")
    protocol_sha256 = _sha256(_canonical_json_bytes(protocol))
    if events[0].get("protocol_sha256") != protocol_sha256 or any(event.get("protocol_sha256") != protocol_sha256 for event in events):
        raise ValueError("calibration protocol identity changed within the record")
    if protocol.get("mode") != "production" or protocol.get("backend") != "gpu":
        raise ValueError("development requires a production GPU calibration")
    if protocol.get("founder_index_sha256") != founder_index_sha256(founder_index):
        raise ValueError("calibration and development founder-index hashes differ")
    preregistration_sha256 = _sha256(preregistration_path.read_bytes())
    if protocol.get("preregistration_sha256") != preregistration_sha256:
        raise ValueError("calibration and development preregistration hashes differ")
    if protocol.get("simulator_config_sha256") != simulator_config_sha256(SimulatorConfig()):
        raise ValueError("calibration simulator configuration is not current")
    expected_production_spec = (
        PRODUCTION_SPEC
        if seed_panel is None
        else production_spec_for_seed_panel(seed_panel)
    )
    if seed_panel is not None:
        if seed_panel_path is None:
            raise ValueError("readiness seed panel is missing its authenticated path")
        recorded_path = protocol.get("readiness_seed_panel_path")
        if (
            not isinstance(recorded_path, str)
            or Path(recorded_path).resolve() != seed_panel_path.resolve()
            or protocol.get("readiness_seed_panel_sha256")
            != seed_panel.artifact_sha256
            or protocol.get("readiness_protocol_sha256")
            != seed_panel.protocol_sha256
        ):
            raise ValueError("calibration readiness seed-panel identity changed")
    expected_spec = json.loads(
        _canonical_json_bytes(
            {
                **asdict(expected_production_spec),
                "config": asdict(expected_production_spec.config),
            }
        )
    )
    if protocol.get("spec") != expected_spec:
        raise ValueError("calibration did not use the frozen production spec")
    if protocol.get("simulator_source_sha256") != simulator_source_sha256():
        raise ValueError("trusted simulator source changed after calibration")

    final = events[-1]
    if final.get("status") != "selected" or final.get("selected_manifest_written") is not True:
        raise ValueError("calibration did not finish with a selected benchmark")
    selected = final.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("calibration finish record is missing its selection")
    recorded_manifest_path = selected.get("manifest_path")
    if not isinstance(recorded_manifest_path, str) or Path(recorded_manifest_path).resolve() != selected_manifest_path.resolve():
        raise ValueError("selected calibration manifest path changed after selection")
    gain = selected.get("gain")
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or float(gain) not in PRODUCTION_SPEC.gains:
        raise ValueError("selected gain is outside the frozen calibration grid")
    winner = selected.get("winning_policy")
    if winner not in MUTATION_POLICIES:
        raise ValueError("selected policy is not a frozen conventional mutation policy")
    schedule_id = selected.get("schedule_id")
    if not isinstance(schedule_id, str) or not schedule_id:
        raise ValueError("selected calibration schedule is invalid")

    manifest = _verify_manifest_file(selected_manifest_path)
    selected_manifest_sha256 = manifest_sha256(manifest)
    if manifest.schema_version != SCHEMA_VERSION or manifest.partition != "calibration" or selected.get("manifest_sha256") != selected_manifest_sha256:
        raise ValueError("selected calibration manifest identity is invalid")
    training_founders = tuple(
        sorted(
            (founder for founder in founder_index.founders if founder.partition == "training"),
            key=lambda founder: founder.founder_id,
        )
    )
    expected_manifest = build_calibration_manifest(
        expected_production_spec,
        training_founders,
        gain=float(gain),
        schedule_id=schedule_id,
    )
    if manifest_sha256(expected_manifest) != selected_manifest_sha256:
        raise ValueError("selected calibration manifest is not the frozen grid result")

    matching_attempts = [
        event
        for event in events
        if event.get("event") == "gain_evaluated"
        and isinstance(event.get("attempt"), dict)
        and event["attempt"].get("gain") == float(gain)
        and event["attempt"].get("schedule_id") == schedule_id
        and event["attempt"].get("manifest_sha256") == selected_manifest_sha256
    ]
    if len(matching_attempts) != 1:
        raise ValueError("calibration selection must name exactly one evaluated attempt")
    gate = matching_attempts[0].get("gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True or gate.get("winning_policy") != winner:
        raise ValueError("selected calibration attempt did not pass with its winner")

    recorded_policies = protocol.get("policies")
    if not isinstance(recorded_policies, dict):
        raise ValueError("calibration protocol is missing policy provenance")
    for name in ("clone", winner):
        record = recorded_policies.get(name)
        if not isinstance(record, dict) or record.get("source_sha256") != _source_sha256(POLICIES[name][0]):
            raise ValueError(f"trusted policy source changed after calibration: {name}")

    return CalibrationSelection(
        gain=float(gain),
        schedule_id=schedule_id,
        winning_policy=winner,
        manifest_sha256=selected_manifest_sha256,
        protocol_sha256=protocol_sha256,
        calibration_record_sha256=record_sha256,
    )


def _development_founders(index: FounderIndex) -> tuple[FounderRecord, ...]:
    records = tuple(
        sorted(
            (founder for founder in index.founders if founder.partition == "development"),
            key=lambda founder: founder.founder_id,
        )
    )
    if len(records) != EXPECTED_DEVELOPMENT_FOUNDERS:
        raise ValueError("development confirmation requires exactly two founders")
    return records


def validate_development_manifest(
    manifest: ScenarioManifest,
    founder_index: FounderIndex,
    *,
    gain: float,
    seed_panel: ReadinessSeedPanel | None = None,
) -> tuple[FounderRecord, ...]:
    """Validate only the development panel; never construct or open sealed data."""
    founders = _development_founders(founder_index)
    if (
        manifest.schema_version != SCHEMA_VERSION
        or manifest.partition != "development"
        or manifest.simulator_config_sha256 != simulator_config_sha256(SimulatorConfig())
        or manifest.horizon != PRODUCTION_SPEC.horizon
        or manifest.chunk_steps != PRODUCTION_SPEC.chunk_steps
    ):
        raise ValueError("development manifest does not use the frozen full protocol")
    if len(manifest.worlds) != EXPECTED_DEVELOPMENT_PAIRS * 2:
        raise ValueError("development manifest must contain exactly sixteen worlds")
    expected_founders = {founder.founder_id: founder.artifact_sha256 for founder in founders}
    pairs: dict[str, list[WorldScenario]] = {}
    for world in manifest.worlds:
        pairs.setdefault(world.pair_id, []).append(world)
    if len(pairs) != EXPECTED_DEVELOPMENT_PAIRS:
        raise ValueError("development manifest must contain eight exact pairs")

    development_seeds = (
        DEVELOPMENT_SEEDS if seed_panel is None else seed_panel.development
    )
    expected_keys = {
        (founder.founder_id, seed)
        for founder in founders
        for seed in development_seeds
    }
    actual_keys: set[tuple[str, int]] = set()
    event_steps = {
        seed: EVENT_STEPS[index % len(EVENT_STEPS)]
        for index, seed in enumerate(development_seeds)
    }
    expected_gains = [1.0] * HINGE_COUNT
    expected_gains[2] = gain
    expected_gains[3] = gain
    expected_injury = ActuatorInjuryParameters(tuple(expected_gains))
    for pair_id, worlds in pairs.items():
        if len(worlds) != 2:
            raise ValueError("every development pair must contain two worlds")
        sham = [world for world in worlds if world.event_kind is EventKind.NULL]
        injury = [world for world in worlds if world.event_kind is EventKind.ACTUATOR_INJURY]
        if len(sham) != 1 or len(injury) != 1:
            raise ValueError("every development pair needs one sham and one injury")
        sham_world, injury_world = sham[0], injury[0]
        common = (
            sham_world.world_seed,
            sham_world.pair_id,
            sham_world.scenario_family,
            sham_world.event_step,
            sham_world.founder_id,
            sham_world.founder_sha256,
        )
        if common != (
            injury_world.world_seed,
            injury_world.pair_id,
            injury_world.scenario_family,
            injury_world.event_step,
            injury_world.founder_id,
            injury_world.founder_sha256,
        ):
            raise ValueError("development forks do not share exact pre-event identity")
        if (
            not isinstance(sham_world.event_parameters, NullEventParameters)
            or injury_world.event_parameters != expected_injury
            or sham_world.scenario_id != f"{pair_id}-sham"
            or injury_world.scenario_id != f"{pair_id}-injured"
            or sham_world.scenario_family != "actuator_injury"
            or sham_world.founder_id not in expected_founders
            or sham_world.founder_sha256 != expected_founders[sham_world.founder_id]
            or sham_world.world_seed not in development_seeds
            or sham_world.event_step != event_steps[sham_world.world_seed]
        ):
            raise ValueError("development pair violates its frozen treatment panel")
        actual_keys.add((sham_world.founder_id, sham_world.world_seed))
    if actual_keys != expected_keys:
        raise ValueError("development manifest does not contain the exact founder×seed panel")
    return founders


class DevelopmentEvaluator(CalibrationEvaluator):
    """Reuse calibration's exact-fork evaluator on development founders only."""

    def __init__(
        self,
        manifest: ScenarioManifest,
        founder_index_path: Path,
        founder_records: Sequence[FounderRecord],
        *,
        gain: float,
        winning_policy: str,
        seed_panel: ReadinessSeedPanel | None = None,
    ):
        event_steps = tuple(sorted({(world.world_seed, world.event_step) for world in manifest.worlds}))
        self.spec = replace(
            PRODUCTION_SPEC,
            world_seeds=(
                DEVELOPMENT_SEEDS if seed_panel is None else seed_panel.development
            ),
            event_step_by_seed=event_steps,
            gains=(gain,),
            injured_hinges=(2, 3),
            required_training_founders=EXPECTED_DEVELOPMENT_FOUNDERS,
        )
        self.founder_index_path = Path(founder_index_path)
        self.records = tuple(founder_records)
        self.founders = {
            record.founder_id: load_founder_artifact(
                self.founder_index_path.parent,
                record,
                expected_partition="development",
            )
            for record in self.records
        }
        self._runtimes: dict[str, tuple[Any, Callable, int]] = {}
        for policy_name in ("clone", winning_policy):
            policy, expected_operator = POLICIES[policy_name]
            env = build_environment(
                self.spec.config,
                policy,
                horizon=self.spec.horizon,
            )
            compiled = jax.jit(
                lambda state, keys, environment=env: run_ecosystem_chunk(
                    environment,
                    state,
                    keys,
                )
            )
            self._runtimes[policy_name] = (env, compiled, expected_operator)

    def __call__(
        self,
        manifest: ScenarioManifest,
        policy_name: str,
    ) -> PolicyCalibrationResult:
        if policy_name not in self._runtimes:
            raise ValueError("development evaluator received an unselected policy")
        return self._evaluate_policy(manifest, policy_name)


def _worlds_by_arm(
    policy: PolicyCalibrationResult,
    arm: str,
) -> tuple[CalibrationWorldResult, ...]:
    return tuple(world for world in policy.worlds if world.arm == arm)


def _observations(
    worlds: Sequence[CalibrationWorldResult],
) -> tuple[FounderWorldObservation, ...]:
    return tuple(
        FounderWorldObservation(
            founder_id=world.founder_id,
            world_seed=world.world_seed,
            pair_id=world.pair_id,
            value=world.primary_score,
        )
        for world in worlds
    )


def evaluate_development_gate(
    clone: PolicyCalibrationResult,
    mutation: PolicyCalibrationResult,
) -> DevelopmentGateDecision:
    """Apply the frozen causal, integrity, and viability transfer rule."""
    if clone.policy_name != "clone" or mutation.policy_name not in MUTATION_POLICIES:
        raise ValueError("gate requires clone and one conventional mutation policy")
    clone_injury = _worlds_by_arm(clone, "injury")
    clone_sham = _worlds_by_arm(clone, "sham")
    mutation_injury = _worlds_by_arm(mutation, "injury")
    if not clone_injury or len(clone_injury) != len(clone_sham):
        raise ValueError("clone result must retain complete injury/sham pairs")
    clone_harm = hierarchical_paired_effect_summary(
        _observations(clone_injury),
        _observations(clone_sham),
        seed=PRODUCTION_SPEC.bootstrap_seed,
        replicates=PRODUCTION_SPEC.bootstrap_replicates,
    )
    mutation_advantage = hierarchical_paired_effect_summary(
        _observations(mutation_injury),
        _observations(clone_injury),
        seed=PRODUCTION_SPEC.bootstrap_seed,
        replicates=PRODUCTION_SPEC.bootstrap_replicates,
    )
    survival_fraction = float(np.mean([world.survived for world in mutation_injury]))
    median_births = float(np.median([world.post_event_births for world in mutation_injury]))
    median_generation_gain = float(np.median([world.generation_gain for world in mutation_injury]))
    viability = bool(
        survival_fraction >= PRODUCTION_SPEC.gates.minimum_survival_fraction
        and median_births >= PRODUCTION_SPEC.gates.minimum_median_post_event_births
        and median_generation_gain >= PRODUCTION_SPEC.gates.minimum_median_generation_gain
    )
    all_integrity = bool(
        clone.all_repeats_integrity_valid and mutation.all_repeats_integrity_valid and all(world.integrity_valid for world in (*clone.worlds, *mutation.worlds))
    )
    clone_identity = clone.clone_identity_all_repeats is True
    passed = bool(
        clone_identity and all_integrity and clone_harm.confidence_interval[1] < 0.0 and mutation_advantage.confidence_interval[0] > 0.0 and viability
    )
    return DevelopmentGateDecision(
        passed=passed,
        clone_harm=clone_harm,
        mutation_advantage=mutation_advantage,
        winning_policy=mutation.policy_name,
        all_integrity_valid=all_integrity,
        clone_identity_valid=clone_identity,
        survival_fraction=survival_fraction,
        median_post_event_births=median_births,
        median_generation_gain=median_generation_gain,
        viability_passed=viability,
    )


def _summary_dict(summary: HierarchicalEffectSummary) -> dict[str, object]:
    return {
        "confidence_interval": list(summary.confidence_interval),
        "count": summary.count,
        "founder_count": summary.founder_count,
        "fraction_positive": summary.fraction_positive,
        "mean": summary.mean,
        "median": summary.median,
        "per_founder_effects": [asdict(effect) for effect in summary.per_founder_effects],
    }


def _decision_dict(decision: DevelopmentGateDecision) -> dict[str, object]:
    return {
        "all_integrity_valid": decision.all_integrity_valid,
        "clone_harm": _summary_dict(decision.clone_harm),
        "clone_identity_valid": decision.clone_identity_valid,
        "median_generation_gain": decision.median_generation_gain,
        "median_post_event_births": decision.median_post_event_births,
        "mutation_advantage": _summary_dict(decision.mutation_advantage),
        "passed": decision.passed,
        "survival_fraction": decision.survival_fraction,
        "viability_passed": decision.viability_passed,
        "winning_policy": decision.winning_policy,
    }


def _policy_dict(policy: PolicyCalibrationResult) -> dict[str, object]:
    return {
        "all_repeats_integrity_valid": policy.all_repeats_integrity_valid,
        "clone_identity_all_repeats": policy.clone_identity_all_repeats,
        "policy_name": policy.policy_name,
        "repeat_scores": list(policy.repeat_scores),
        "selected_repeat_index": policy.selected_repeat_index,
        "worlds": [asdict(world) for world in policy.worlds],
    }


def _validate_policy_result(
    policy: PolicyCalibrationResult,
    manifest: ScenarioManifest,
    *,
    policy_name: str,
) -> None:
    if not isinstance(policy, PolicyCalibrationResult):
        raise TypeError("policy runner must return PolicyCalibrationResult")
    if policy.policy_name != policy_name:
        raise ValueError("policy runner changed policy identity")
    if len(policy.repeat_scores) != PRODUCTION_SPEC.numerical_repeats:
        raise ValueError("development requires exactly three coherent repeats")
    if not 0 <= policy.selected_repeat_index < len(policy.repeat_scores):
        raise ValueError("selected coherent repeat index is invalid")
    repeat_scores = np.asarray(policy.repeat_scores, dtype=np.float64)
    if not np.all(np.isfinite(repeat_scores)):
        raise ValueError("numerical repeat scores must be finite")
    coherent_median = int(np.argsort(repeat_scores, kind="stable")[PRODUCTION_SPEC.numerical_repeats // 2])
    if policy.selected_repeat_index != coherent_median:
        raise ValueError("policy result did not select the coherent median repeat")
    if len(policy.worlds) != len(manifest.worlds):
        raise ValueError("policy result does not contain every development world")
    expected = {
        (
            world.founder_id,
            world.founder_sha256,
            world.world_seed,
            world.pair_id,
            world.event_step,
            "injury" if world.event_kind is EventKind.ACTUATOR_INJURY else "sham",
        )
        for world in manifest.worlds
    }
    actual = {
        (
            world.founder_id,
            world.founder_sha256,
            world.world_seed,
            world.pair_id,
            world.event_step,
            world.arm,
        )
        for world in policy.worlds
    }
    if len(actual) != len(policy.worlds) or actual != expected:
        raise ValueError("policy result changed the frozen development panel")
    expected_operator = POLICIES[policy_name][1]
    for world in policy.worlds:
        if world.policy_name != policy_name:
            raise ValueError("world result changed policy identity")
        if (
            len(world.operator_counts) != len(POLICIES)
            or any(count < 0 for count in world.operator_counts)
            or sum(world.operator_counts) < world.post_event_births
            or any(count != 0 for index, count in enumerate(world.operator_counts) if index != expected_operator)
        ):
            raise ValueError("world result violates trusted operator accounting")


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace immutable result: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _publish_confirmation(
    *,
    record_path: Path,
    decision_path: Path,
    record_events: Sequence[Mapping[str, object]],
    decision: Mapping[str, object],
) -> tuple[str, str]:
    """Publish one canonical append-only event record and its bound decision."""
    paths = (
        record_path,
        record_path.with_suffix(".sha256"),
        decision_path,
        decision_path.with_suffix(".sha256"),
    )
    if len(set(paths)) != len(paths):
        raise ValueError("record and decision artifact paths must be distinct")
    existing = next((path for path in paths if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"refusing to replace immutable result: {existing}")
    record_payload = b"".join(
        _canonical_json_bytes(
            {
                "schema_version": CONFIRMATION_RECORD_SCHEMA_VERSION,
                "sequence": sequence,
                **event,
            }
        )
        + b"\n"
        for sequence, event in enumerate(record_events)
    )
    record_digest = _sha256(record_payload)
    decision_value = {
        "schema_version": CONFIRMATION_DECISION_SCHEMA_VERSION,
        "confirmation_record_sha256": record_digest,
        **decision,
    }
    decision_payload = _canonical_json_bytes(decision_value) + b"\n"
    decision_digest = _sha256(decision_payload)
    written: list[Path] = []
    try:
        for path, payload in (
            (record_path, record_payload),
            (
                record_path.with_suffix(".sha256"),
                f"{record_digest}  {record_path.name}\n".encode("ascii"),
            ),
            (decision_path, decision_payload),
            (
                decision_path.with_suffix(".sha256"),
                f"{decision_digest}  {decision_path.name}\n".encode("ascii"),
            ),
        ):
            _write_once(path, payload)
            written.append(path)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    return record_digest, decision_digest


def _run_development_confirmation(
    *,
    development_manifest_path: Path,
    founder_index_path: Path,
    calibration_record_path: Path,
    selected_calibration_manifest_path: Path,
    preregistration_path: Path,
    result_record_path: Path,
    decision_path: Path,
    policy_runner: PolicyRunner | None = None,
    backend: str | None = None,
    readiness_seed_panel_path: Path | None = None,
) -> dict[str, object]:
    """Run production confirmation or an irrevocably test-only injected seam."""
    test_only = policy_runner is not None
    actual_backend = jax.default_backend()
    if backend is not None:
        raise RuntimeError("backend overrides are forbidden")
    if not test_only and actual_backend != "gpu":
        raise RuntimeError("production development confirmation requires GPU")
    if test_only and (not result_record_path.name.endswith(".test.jsonl") or not decision_path.name.endswith(".test.json")):
        raise RuntimeError("injected development results require .test.jsonl/.test.json artifacts")
    if SimulatorConfig() != PRODUCTION_SPEC.config or not PRODUCTION_SPEC.config.fluid_enabled:
        raise RuntimeError("development confirmation requires the full fluid config")
    preregistration_payload = preregistration_path.read_text(encoding="utf-8")
    if "Status: **FROZEN" not in preregistration_payload:
        raise RuntimeError("development confirmation requires frozen preregistration")

    # Parse only index metadata here.  Exact development artifacts are verified
    # lazily by the evaluator; sealed artifacts and manifests are never opened.
    founder_index = load_founder_index(founder_index_path, verify_artifacts=False)
    seed_panel = (
        None
        if readiness_seed_panel_path is None
        else load_readiness_seed_panel(readiness_seed_panel_path)
    )
    selection = authenticate_calibration_selection(
        calibration_record_path=calibration_record_path,
        selected_manifest_path=selected_calibration_manifest_path,
        founder_index=founder_index,
        preregistration_path=preregistration_path,
        seed_panel=seed_panel,
        seed_panel_path=readiness_seed_panel_path,
    )
    development_manifest = _verify_manifest_file(development_manifest_path)
    founders = validate_development_manifest(
        development_manifest,
        founder_index,
        gain=selection.gain,
        seed_panel=seed_panel,
    )

    runner = policy_runner
    if runner is None:
        runner = DevelopmentEvaluator(
            development_manifest,
            founder_index_path,
            founders,
            gain=selection.gain,
            winning_policy=selection.winning_policy,
            seed_panel=seed_panel,
        )
    clone = runner(development_manifest, "clone")
    mutation = runner(development_manifest, selection.winning_policy)
    _validate_policy_result(clone, development_manifest, policy_name="clone")
    _validate_policy_result(
        mutation,
        development_manifest,
        policy_name=selection.winning_policy,
    )
    gate = evaluate_development_gate(clone, mutation)

    provenance = {
        "backend": actual_backend,
        "evidence_class": ("test_only_non_scientific" if test_only else "production"),
        "bootstrap_replicates": PRODUCTION_SPEC.bootstrap_replicates,
        "bootstrap_seed": PRODUCTION_SPEC.bootstrap_seed,
        "calibration": asdict(selection),
        "confirmation_rule_id": CONFIRMATION_RULE_ID,
        "confirmation_source_sha256": _sha256(Path(__file__).read_bytes()),
        "development_manifest_path": str(development_manifest_path),
        "development_manifest_sha256": manifest_sha256(development_manifest),
        "founder_index_path": str(founder_index_path),
        "founder_index_sha256": founder_index_sha256(founder_index),
        "numerical_repeats": PRODUCTION_SPEC.numerical_repeats,
        "policies": {name: {"source_sha256": _source_sha256(POLICIES[name][0])} for name in ("clone", selection.winning_policy)},
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": _sha256(preregistration_path.read_bytes()),
        "simulator_config": asdict(PRODUCTION_SPEC.config),
        "simulator_config_sha256": simulator_config_sha256(PRODUCTION_SPEC.config),
        "simulator_source_sha256": simulator_source_sha256(),
    }
    if seed_panel is not None:
        provenance["readiness_seed_panel_path"] = str(
            readiness_seed_panel_path
        )
        provenance["readiness_seed_panel_sha256"] = seed_panel.artifact_sha256
        provenance["readiness_protocol_sha256"] = seed_panel.protocol_sha256
    gate_dict = _decision_dict(gate)
    status = ("test_only_passed" if gate.passed else "test_only_failed_transfer") if test_only else ("passed" if gate.passed else "failed_transfer_stop")
    events = (
        {
            "event": "development_confirmation_started",
            "provenance": provenance,
        },
        {"event": "policy_evaluated", "result": _policy_dict(clone)},
        {"event": "policy_evaluated", "result": _policy_dict(mutation)},
        {"decision": gate_dict, "event": "development_confirmation_finished"},
    )
    record_sha256, decision_sha256 = _publish_confirmation(
        record_path=result_record_path,
        decision_path=decision_path,
        record_events=events,
        decision={
            "decision": gate_dict,
            "development_manifest_sha256": manifest_sha256(development_manifest),
            "evidence_class": provenance["evidence_class"],
            "scientific_evidence": not test_only,
            "status": status,
        },
    )
    return {
        "decision_path": str(decision_path),
        "decision_sha256": decision_sha256,
        "passed": gate.passed,
        "result_record_path": str(result_record_path),
        "result_record_sha256": record_sha256,
        "scientific_evidence": not test_only,
        "status": status,
        "winning_policy": selection.winning_policy,
    }


def run_development_confirmation(
    *,
    development_manifest_path: Path,
    founder_index_path: Path,
    calibration_record_path: Path,
    selected_calibration_manifest_path: Path,
    preregistration_path: Path,
    result_record_path: Path,
    decision_path: Path,
    readiness_seed_panel_path: Path | None = None,
) -> dict[str, object]:
    """Run the real GPU confirmation without injectable scientific results."""
    return _run_development_confirmation(
        development_manifest_path=development_manifest_path,
        founder_index_path=founder_index_path,
        calibration_record_path=calibration_record_path,
        selected_calibration_manifest_path=selected_calibration_manifest_path,
        preregistration_path=preregistration_path,
        result_record_path=result_record_path,
        decision_path=decision_path,
        readiness_seed_panel_path=readiness_seed_panel_path,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--founder-index", type=Path, required=True)
    parser.add_argument("--calibration-record", type=Path, required=True)
    parser.add_argument("--selected-calibration-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--result-record", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--readiness-seed-panel", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_development_confirmation(
        development_manifest_path=args.development_manifest,
        founder_index_path=args.founder_index,
        calibration_record_path=args.calibration_record,
        selected_calibration_manifest_path=args.selected_calibration_manifest,
        preregistration_path=args.preregistration,
        result_record_path=args.result_record,
        decision_path=args.decision,
        readiness_seed_panel_path=args.readiness_seed_panel,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
