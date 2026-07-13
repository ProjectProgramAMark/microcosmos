"""Thin, seed-parameterized protocol tools for Evo² r5.

The r4 scientific calculations are reused as-is.  This module supplies only
the experiment-local configuration conversion and orchestration that r5 needs
because the frozen r4 entrypoints embed their own world seeds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Callable

import jax
import numpy as np

from microcosmos.heredity import (
    R4_CLONE,
    R4_NUM_OPERATORS,
    R4_STANDARD_PARAMETRIC,
    fixed_r4_policy,
)
from microcosmos.rollout import run_ecosystem_chunk

from ..build_founder_bank import (
    CandidateRange as BuilderCandidateRange,
    ScreeningGates as BuilderScreeningGates,
    ScreeningSpec as BuilderScreeningSpec,
    run_founder_bank_builder,
)
from ..episode import (
    SimulatorConfig,
    _apply_event,
    _resolve_manifest_founders,
    build_environment,
    evaluate_manifest,
    simulator_config_sha256,
    simulator_source_sha256,
)
from ..founder_artifacts import founder_index_sha256, load_founder_index
from ..heredity_adaptation_v4.manifest_generator import (
    CHUNK_STEPS,
    HORIZON,
    R4ManifestBundle,
    _founders as _r4_founders,
    _paired_worlds as _r4_paired_worlds,
    validate_manifest_bundle as _validate_r4_manifest_bundle,
)
from ..heredity_adaptation_v4.opportunity import (
    ASSESSMENT_STEPS,
    MUTATION_REPEATS,
    _continue_target_child,
    _find_imminent_state,
    _first_imminent_in_chunk,
    _inject_one_birth,
    _score_action,
    _state_features,
    crossfit_opportunity,
)
from ..heredity_adaptation_v4.qualification import (
    _founder_bootstrap_upper,
    _generation_gain,
    _observable_stress,
    _pair_effects,
    _shock_episodes,
    uniform_exploration_policy,
)
from ..protocol import (
    R4_SCHEMA_VERSION,
    ActuationCostShiftParameters,
    EventKind,
    ScenarioManifest,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)
from .protocol import (
    ACTUATION_COST_MULTIPLIERS,
    ARTIFACTS_ROOT,
    DEVELOPMENT_EVENT_STEPS,
    FOUNDER_PANEL_SEED,
    PROTOCOL_REVISION,
    SEALED_EVENT_STEPS,
    TRAINING_EVENT_STEPS,
    ScreeningSpec,
    founder_screening_spec,
)
from .qualify_worlds import validate_world_qualification


DISTURBANCE_QUALIFICATION_SCHEMA_VERSION = 1
OPPORTUNITY_QUALIFICATION_SCHEMA_VERSION = 1
DISTURBANCE_QUALIFICATION_PATH = ARTIFACTS_ROOT / "disturbance_qualification.json"
OPPORTUNITY_QUALIFICATION_PATH = ARTIFACTS_ROOT / "operator_opportunity.json"
MANIFEST_DIRECTORY = ARTIFACTS_ROOT / "manifests"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DISTURBANCE_GATES = frozenset(
    {"harm", "integrity", "observable_stress", "resolved_feedback", "viability"}
)
_EVENT_STEPS = {
    "training": TRAINING_EVENT_STEPS,
    "development": DEVELOPMENT_EVENT_STEPS,
    "sealed": SEALED_EVENT_STEPS,
}
_REQUIRED_FOUNDERS = {"training": 4, "development": 4, "sealed": 8}
_MANIFEST_FILENAMES = (
    "training_stable.json",
    "training_punctuated.json",
    "development.json",
    "sealed.json",
)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"invalid {context} keys; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return value


def _require_finite(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{context} must be a finite number")
    return float(value)


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _publish_canonical_json(
    value: Mapping[str, Any],
    path: str | Path,
    validator: Callable[[Mapping[str, Any]], dict[str, object]],
) -> str:
    """Atomically create one read-only canonical JSON document."""
    destination = Path(path)
    normalized = validator(value)
    payload = _canonical_json_bytes(normalized) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable evidence: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace immutable evidence: {destination}"
            ) from error
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _load_canonical_json(
    path: str | Path,
    validator: Callable[[Mapping[str, Any]], dict[str, object]],
) -> dict[str, object]:
    payload = Path(path).read_bytes()
    encoded = payload[:-1] if payload.endswith(b"\n") else payload
    if b"\n" in encoded or b"\r" in encoded:
        raise ValueError("canonical evidence must contain one JSON line")
    try:
        raw = json.loads(encoded.decode("ascii"), object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("evidence is not strict ASCII JSON") from error
    if not isinstance(raw, Mapping):
        raise ValueError("evidence must be a JSON object")
    normalized = validator(raw)
    if payload != _canonical_json_bytes(normalized) + b"\n":
        raise ValueError("evidence does not use canonical JSON bytes")
    return normalized


def _selected_world_seeds(world_qualification: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    selected = world_qualification["selected_seeds"]
    if not isinstance(selected, Mapping):
        raise ValueError("world qualification selected_seeds must be an object")
    expected = {"founder_eligibility", "training", "development", "sealed"}
    _require_exact_keys(selected, expected, "selected world seeds")
    result = {}
    for partition in ("founder_eligibility", "training", "development", "sealed"):
        values = selected[partition]
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in values)
        ):
            raise ValueError(f"{partition} requires exactly two integer world seeds")
        result[partition] = (values[0], values[1])
    return result


def builder_screening_spec(spec: ScreeningSpec) -> BuilderScreeningSpec:
    """Convert the strict r5 screen into the generic founder-builder contract."""
    if not isinstance(spec, ScreeningSpec):
        raise TypeError("spec must be an r5 ScreeningSpec")
    return BuilderScreeningSpec(
        mode="r4_production",
        config=spec.config,
        seeds=spec.seeds,
        horizon=spec.horizon,
        chunk_steps=spec.chunk_steps,
        candidate_ranges=tuple(
            BuilderCandidateRange(item.partition, item.start, item.stop, item.required)
            for item in spec.candidate_ranges
        ),
        gates=BuilderScreeningGates(**asdict(spec.gates)),
        selection_rule_id=spec.selection_rule_id,
    )


def run_qualified_founder_screening(
    world_qualification_path: str | Path,
    *,
    bank_directory: str | Path,
    screening_record_path: str | Path,
) -> dict[str, object]:
    """Run the generic GPU founder builder on authenticated eligibility seeds."""
    world_qualification = validate_world_qualification(world_qualification_path)
    eligibility_seeds = _selected_world_seeds(world_qualification)["founder_eligibility"]
    spec = builder_screening_spec(founder_screening_spec(eligibility_seeds))
    record_path = Path(screening_record_path)
    try:
        return run_founder_bank_builder(
            smoke=False,
            bank_directory=Path(bank_directory),
            screening_record_path=record_path,
            configured_spec=spec,
            panel_seed=FOUNDER_PANEL_SEED,
        )
    finally:
        if record_path.is_file():
            record_path.chmod(0o444)


def _build_manifest(
    partition: str,
    founders: Sequence[Any],
    seeds: tuple[int, int],
    multiplier: float,
) -> ScenarioManifest:
    worlds = tuple(
        world
        for founder in founders
        for seed, event_step in zip(seeds, _EVENT_STEPS[partition], strict=True)
        for world in _r4_paired_worlds(
            partition,
            founder,
            seed,
            event_step,
            multiplier,
        )
    )
    return ScenarioManifest(
        schema_version=R4_SCHEMA_VERSION,
        partition="sealed_final" if partition == "sealed" else partition,
        controller_layout="cppn-4x1-15n-30c-v1",
        simulator_config_sha256=simulator_config_sha256(SimulatorConfig()),
        horizon=HORIZON,
        chunk_steps=CHUNK_STEPS,
        worlds=worlds,
    )


def build_manifest_bundle(
    founder_index_path: str | Path,
    selected_seeds: Mapping[str, tuple[int, int]],
    multiplier: float,
) -> R4ManifestBundle:
    """Build an r4-compatible bundle from explicit authenticated r5 seeds."""
    multiplier = _require_finite(multiplier, "multiplier")
    if multiplier not in ACTUATION_COST_MULTIPLIERS:
        raise ValueError("multiplier is not in the frozen r5 qualification grid")
    _require_exact_keys(selected_seeds, set(_REQUIRED_FOUNDERS), "manifest seed panel")
    for partition, seeds in selected_seeds.items():
        if (
            not isinstance(seeds, tuple)
            or len(seeds) != 2
            or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        ):
            raise ValueError(f"{partition} manifest requires two integer seeds")
    index = load_founder_index(founder_index_path, verify_artifacts=False)
    founders = _r4_founders(index)
    training = _build_manifest("training", founders["training"], selected_seeds["training"], multiplier)
    bundle = R4ManifestBundle(
        training_stable=training,
        training_punctuated=training,
        development=_build_manifest(
            "development",
            founders["development"],
            selected_seeds["development"],
            multiplier,
        ),
        sealed=_build_manifest(
            "sealed",
            founders["sealed"],
            selected_seeds["sealed"],
            multiplier,
        ),
    )
    validate_manifest_bundle(bundle, selected_seeds, multiplier)
    return bundle


def validate_manifest_bundle(
    bundle: R4ManifestBundle,
    selected_seeds: Mapping[str, tuple[int, int]],
    multiplier: float,
) -> None:
    """Apply r4 invariants plus the explicit r5 seed/timing contract."""
    _validate_r4_manifest_bundle(bundle, multiplier)
    manifests = {
        "training": bundle.training_stable,
        "development": bundle.development,
        "sealed": bundle.sealed,
    }
    for partition, manifest in manifests.items():
        expected_seeds = selected_seeds[partition]
        actual_seeds = tuple(dict.fromkeys(world.world_seed for world in manifest.worlds))
        if actual_seeds != expected_seeds:
            raise ValueError(f"{partition} manifest is bound to the wrong r5 seeds")
        expected_steps = dict(zip(expected_seeds, _EVENT_STEPS[partition], strict=True))
        if any(world.event_step != expected_steps[world.world_seed] for world in manifest.worlds):
            raise ValueError(f"{partition} manifest is bound to the wrong r5 event steps")
        founders = {world.founder_id for world in manifest.worlds}
        if len(founders) != _REQUIRED_FOUNDERS[partition]:
            raise ValueError(f"{partition} manifest has the wrong founder count")


def _publish_manifest_bundle(bundle: R4ManifestBundle, output_directory: str | Path) -> dict[str, str]:
    destination = Path(output_directory)
    staging = destination.with_name(f".{destination.name}.staging")
    if destination.exists() or staging.exists():
        raise FileExistsError("refusing to replace an r5 manifest directory")
    staging.mkdir(parents=True)
    hashes: dict[str, str] = {}
    try:
        for filename, manifest in bundle.named():
            payload = canonical_manifest_bytes(manifest) + b"\n"
            manifest_path = staging / filename
            manifest_path.write_bytes(payload)
            if manifest_from_json_bytes(payload) != manifest:
                raise RuntimeError("manifest changed during canonical round trip")
            digest = manifest_sha256(manifest)
            sidecar = manifest_path.with_suffix(".sha256")
            sidecar.write_text(f"{digest}  {filename}\n", encoding="ascii")
            hashes[filename] = digest
        for filename in _MANIFEST_FILENAMES:
            manifest_mode = 0o444 if filename.startswith("training_") else 0o000
            (staging / filename).chmod(manifest_mode)
            (staging / filename).with_suffix(".sha256").chmod(0o444)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return hashes


def build_and_publish_manifests(
    founder_index_path: str | Path,
    world_qualification_path: str | Path,
    disturbance_qualification_path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
    output_directory: str | Path = MANIFEST_DIRECTORY,
) -> dict[str, str]:
    """Publish manifests using only the first authenticated passing multiplier."""
    world_qualification = validate_world_qualification(world_qualification_path)
    selected = _selected_world_seeds(world_qualification)
    disturbance = validate_disturbance_qualification(
        disturbance_qualification_path,
        founder_index_path=founder_index_path,
        world_qualification_path=world_qualification_path,
    )
    if disturbance["passed"] is not True:
        raise RuntimeError("disturbance qualification did not pass")
    multiplier = disturbance["selected_multiplier"]
    if not isinstance(multiplier, (int, float)) or isinstance(multiplier, bool):
        raise RuntimeError("qualified disturbance has no multiplier")
    bundle = build_manifest_bundle(
        founder_index_path,
        {partition: selected[partition] for partition in _REQUIRED_FOUNDERS},
        float(multiplier),
    )
    return _publish_manifest_bundle(bundle, output_directory)


def _viability(episodes: Sequence[Any]) -> dict[str, object]:
    return {
        "survival": float(np.mean([bool(item.survived) for item in episodes])),
        "median_post_births": float(
            np.median([int(item.post_event_birth_count) for item in episodes])
        ),
        "median_generation_gain": float(
            np.median([_generation_gain(item) for item in episodes])
        ),
        "distinct_birth": int(
            sum(int(item.post_event_distinct_birth_count) for item in episodes)
        ),
        "distinct_reproducer": int(
            sum(
                int(np.sum(np.asarray(item.post_event_resolved_distinct_success_count)))
                for item in episodes
            )
        ),
        "integrity": bool(all(bool(item.integrity_valid) for item in episodes)),
    }


def qualify_disturbance_manifest(
    founder_index_path: str | Path,
    manifest: ScenarioManifest,
) -> dict[str, object]:
    """Apply the frozen r4 disturbance criteria to one explicit r5 manifest."""
    shock_multipliers = {
        world.event_parameters.multiplier
        for world in manifest.worlds
        if world.event_kind is EventKind.ACTUATION_COST_SHIFT
        and isinstance(world.event_parameters, ActuationCostShiftParameters)
    }
    if len(shock_multipliers) != 1:
        raise ValueError("disturbance manifest must use one actuation-cost multiplier")
    multiplier = next(iter(shock_multipliers))
    config = SimulatorConfig()
    common = {
        "capture_finalists": True,
        "numerical_repeats": 3,
        "founder_index_path": founder_index_path,
    }
    clone = evaluate_manifest(
        manifest,
        config,
        fixed_r4_policy(R4_CLONE),
        **common,
    )
    standard = evaluate_manifest(
        manifest,
        config,
        fixed_r4_policy(R4_STANDARD_PARAMETRIC),
        **common,
    )
    exploration = evaluate_manifest(
        manifest,
        config,
        uniform_exploration_policy,
        **common,
    )

    effects = _pair_effects(manifest, clone)
    effect_values = np.asarray([value for _, value in effects])
    harm_mean = float(np.mean(effect_values))
    harm_upper = _founder_bootstrap_upper(effects)
    clone_shock = _shock_episodes(manifest, clone)
    standard_shock = _shock_episodes(manifest, standard)
    exploration_shock = _shock_episodes(manifest, exploration)
    clone_viability = _viability(clone_shock)
    standard_viability = _viability(standard_shock)
    viable_policy = max(
        (clone_viability, standard_viability),
        key=lambda item: (item["survival"], item["median_post_births"]),
    )
    pre_resolved = np.sum(
        np.stack([np.asarray(item.pre_event_resolved_count) for item in exploration_shock]),
        axis=0,
    )
    post_resolved = np.sum(
        np.stack([np.asarray(item.post_event_resolved_count) for item in exploration_shock]),
        axis=0,
    )
    final_evidence = np.mean(
        np.stack([np.asarray(item.operator_evidence_ema) for item in exploration_shock]),
        axis=0,
    )
    exercised = (pre_resolved > 0) & (post_resolved > 0) & (final_evidence >= 0.10)
    smd, credit_tv = _observable_stress(
        exploration_shock,
        config.reproduction_threshold,
    )
    gates = {
        "harm": harm_mean <= -0.03 and harm_upper < 0.0,
        "viability": (
            viable_policy["survival"] >= 0.75
            and viable_policy["median_post_births"] >= 4
            and viable_policy["median_generation_gain"] >= 1
            and viable_policy["distinct_birth"] >= 1
            and viable_policy["distinct_reproducer"] >= 1
            and viable_policy["integrity"]
        ),
        "resolved_feedback": (
            int(np.sum(pre_resolved)) >= 24
            and int(np.sum(post_resolved)) >= 24
            and int(np.sum(exercised[1:])) >= 2
        ),
        "observable_stress": smd >= 0.5 or credit_tv >= 0.20,
        "integrity": bool(
            clone.integrity_valid
            & standard.integrity_valid
            & exploration.integrity_valid
        ),
    }
    return {
        "multiplier": multiplier,
        "manifest_sha256": manifest_sha256(manifest),
        "clone_effect_mean": harm_mean,
        "clone_effect_founder_bootstrap_upper_95": harm_upper,
        "clone_viability": clone_viability,
        "standard_viability": standard_viability,
        "pre_resolved_by_operator": pre_resolved.tolist(),
        "post_resolved_by_operator": post_resolved.tolist(),
        "final_evidence_mean": final_evidence.tolist(),
        "observable_feature_max_smd": smd,
        "observable_credit_tv": credit_tv,
        "policy_integrity": {
            "clone": bool(clone.integrity_valid),
            "exploration": bool(exploration.integrity_valid),
            "standard": bool(standard.integrity_valid),
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


_DISTURBANCE_RESULT_KEYS = {
    "clone_effect_founder_bootstrap_upper_95",
    "clone_effect_mean",
    "clone_viability",
    "final_evidence_mean",
    "gates",
    "manifest_sha256",
    "multiplier",
    "observable_credit_tv",
    "observable_feature_max_smd",
    "passed",
    "policy_integrity",
    "post_resolved_by_operator",
    "pre_resolved_by_operator",
    "standard_viability",
}

_VIABILITY_KEYS = {
    "distinct_birth",
    "distinct_reproducer",
    "integrity",
    "median_generation_gain",
    "median_post_births",
    "survival",
}
_POLICY_INTEGRITY_KEYS = {"clone", "exploration", "standard"}


def _validate_viability(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(value, _VIABILITY_KEYS, context)
    survival = _require_finite(value["survival"], f"{context} survival")
    births = _require_finite(
        value["median_post_births"], f"{context} median post births"
    )
    generation = _require_finite(
        value["median_generation_gain"], f"{context} median generation gain"
    )
    if not 0.0 <= survival <= 1.0 or births < 0.0 or generation < 0.0:
        raise ValueError(f"{context} contains an out-of-range metric")
    counts: dict[str, int] = {}
    for name in ("distinct_birth", "distinct_reproducer"):
        item = value[name]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"{context} {name} must be a nonnegative integer")
        counts[name] = item
    if not isinstance(value["integrity"], bool):
        raise ValueError(f"{context} integrity must be boolean")
    return {
        **counts,
        "integrity": value["integrity"],
        "median_generation_gain": generation,
        "median_post_births": births,
        "survival": survival,
    }


def _validate_operator_vector(
    value: object,
    context: str,
    *,
    integer: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (R4_NUM_OPERATORS,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{context} must contain six finite values")
    if integer and (
        not np.issubdtype(array.dtype, np.integer) or np.any(array < 0)
    ):
        raise ValueError(f"{context} must contain nonnegative integers")
    return array


def _expected_disturbance_gates(value: Mapping[str, Any]) -> dict[str, bool]:
    clone_viability = _validate_viability(value["clone_viability"], "clone viability")
    standard_viability = _validate_viability(
        value["standard_viability"], "standard viability"
    )
    pre_resolved = _validate_operator_vector(
        value["pre_resolved_by_operator"], "pre-event resolved feedback", integer=True
    )
    post_resolved = _validate_operator_vector(
        value["post_resolved_by_operator"],
        "post-event resolved feedback",
        integer=True,
    )
    evidence = _validate_operator_vector(
        value["final_evidence_mean"], "final evidence"
    )
    if np.any((evidence < 0.0) | (evidence > 1.0)):
        raise ValueError("final evidence must lie in [0, 1]")
    integrity = value["policy_integrity"]
    if not isinstance(integrity, Mapping):
        raise ValueError("policy integrity must be an object")
    _require_exact_keys(integrity, _POLICY_INTEGRITY_KEYS, "policy integrity")
    if any(not isinstance(item, bool) for item in integrity.values()):
        raise ValueError("policy integrity values must be booleans")

    harm_mean = _require_finite(value["clone_effect_mean"], "clone effect mean")
    harm_upper = _require_finite(
        value["clone_effect_founder_bootstrap_upper_95"],
        "clone effect upper bound",
    )
    smd = _require_finite(
        value["observable_feature_max_smd"], "observable feature SMD"
    )
    credit_tv = _require_finite(value["observable_credit_tv"], "observable credit TV")
    if smd < 0.0 or credit_tv < 0.0:
        raise ValueError("observable-stress metrics must be nonnegative")

    viable_policy = max(
        (clone_viability, standard_viability),
        key=lambda item: (item["survival"], item["median_post_births"]),
    )
    exercised = (pre_resolved > 0) & (post_resolved > 0) & (evidence >= 0.10)
    return {
        "harm": harm_mean <= -0.03 and harm_upper < 0.0,
        "viability": bool(
            viable_policy["survival"] >= 0.75
            and viable_policy["median_post_births"] >= 4
            and viable_policy["median_generation_gain"] >= 1
            and viable_policy["distinct_birth"] >= 1
            and viable_policy["distinct_reproducer"] >= 1
            and viable_policy["integrity"]
        ),
        "resolved_feedback": bool(
            int(np.sum(pre_resolved)) >= 24
            and int(np.sum(post_resolved)) >= 24
            and int(np.sum(exercised[1:])) >= 2
        ),
        "observable_stress": smd >= 0.5 or credit_tv >= 0.20,
        "integrity": bool(all(integrity.values())),
    }


def _validate_disturbance_result(value: object) -> dict[str, object]:
    """Validate the r5 binding around a result produced by frozen r4 code."""
    if not isinstance(value, Mapping):
        raise ValueError("disturbance result must be an object")
    _require_exact_keys(value, _DISTURBANCE_RESULT_KEYS, "disturbance result")
    multiplier = _require_finite(value["multiplier"], "disturbance multiplier")
    if multiplier not in ACTUATION_COST_MULTIPLIERS:
        raise ValueError("disturbance multiplier is outside the frozen grid")
    _require_sha256(value["manifest_sha256"], "disturbance manifest hash")
    gates = value["gates"]
    if not isinstance(gates, Mapping):
        raise ValueError("disturbance gates must be an object")
    _require_exact_keys(gates, set(_DISTURBANCE_GATES), "disturbance gates")
    if any(not isinstance(item, bool) for item in gates.values()):
        raise ValueError("disturbance gates must be booleans")
    expected_gates = _expected_disturbance_gates(value)
    if dict(gates) != expected_gates:
        raise ValueError("disturbance gates disagree with the stored metrics")
    if (
        not isinstance(value["passed"], bool)
        or value["passed"] != all(expected_gates.values())
    ):
        raise ValueError("disturbance passed flag disagrees with its recomputed gates")
    # Canonical encoding below rejects NaN/Inf anywhere in the trusted r4 result.
    _canonical_json_bytes(value)
    return {
        **dict(value),
        "clone_viability": _validate_viability(
            value["clone_viability"], "clone viability"
        ),
        "gates": expected_gates,
        "multiplier": multiplier,
        "policy_integrity": dict(value["policy_integrity"]),
        "standard_viability": _validate_viability(
            value["standard_viability"], "standard viability"
        ),
    }


def disturbance_qualification_from_dict(
    value: Mapping[str, Any],
) -> dict[str, object]:
    """Strictly validate and normalize one r5 disturbance artifact."""
    if not isinstance(value, Mapping):
        raise ValueError("disturbance qualification must be an object")
    expected = {
        "founder_index_sha256",
        "passed",
        "protocol_revision",
        "results",
        "schema_version",
        "selected_multiplier",
        "simulator_source_sha256",
        "status",
        "training_world_seeds",
        "world_qualification_sha256",
    }
    _require_exact_keys(value, expected, "disturbance qualification")
    if value["schema_version"] != DISTURBANCE_QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported disturbance-qualification schema")
    if value["protocol_revision"] != PROTOCOL_REVISION:
        raise ValueError("wrong r5 protocol revision")
    _require_sha256(value["founder_index_sha256"], "founder index hash")
    _require_sha256(value["simulator_source_sha256"], "simulator source hash")
    _require_sha256(value["world_qualification_sha256"], "world qualification hash")
    seeds = value["training_world_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 2
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
    ):
        raise ValueError("disturbance qualification requires two training seeds")
    if not isinstance(value["results"], list) or not value["results"]:
        raise ValueError("disturbance qualification requires result evidence")
    results = [_validate_disturbance_result(item) for item in value["results"]]
    multipliers = [item["multiplier"] for item in results]
    if multipliers != list(ACTUATION_COST_MULTIPLIERS[: len(results)]):
        raise ValueError("disturbance results must be the ascending frozen grid prefix")
    passing = [item["multiplier"] for item in results if item["passed"]]
    if passing:
        expected_passed = True
        expected_selected: float | None = passing[0]
        expected_status = "qualified"
        if results[-1]["multiplier"] != expected_selected or len(passing) != 1:
            raise ValueError("disturbance search must stop at its first passing multiplier")
    else:
        expected_passed = False
        expected_selected = None
        expected_status = "stop_no_qualified_multiplier"
        if len(results) != len(ACTUATION_COST_MULTIPLIERS):
            raise ValueError("failed disturbance search must exhaust the frozen grid")
    if (
        value["passed"] is not expected_passed
        or value["selected_multiplier"] != expected_selected
        or value["status"] != expected_status
    ):
        raise ValueError("top-level disturbance status disagrees with result gates")
    return {
        **dict(value),
        "results": results,
        "training_world_seeds": list(seeds),
    }


def publish_disturbance_qualification(
    value: Mapping[str, Any],
    path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
) -> str:
    return _publish_canonical_json(
        value,
        path,
        disturbance_qualification_from_dict,
    )


def validate_disturbance_qualification(
    path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
    *,
    founder_index_path: str | Path | None = None,
    world_qualification_path: str | Path | None = None,
) -> dict[str, object]:
    result = _load_canonical_json(path, disturbance_qualification_from_dict)
    index = None
    if founder_index_path is not None:
        index = load_founder_index(founder_index_path, verify_artifacts=False)
        if result["founder_index_sha256"] != founder_index_sha256(index):
            raise ValueError("disturbance evidence is bound to the wrong founder index")
    if world_qualification_path is not None:
        world = validate_world_qualification(world_qualification_path)
        if result["world_qualification_sha256"] != _file_sha256(world_qualification_path):
            raise ValueError("disturbance evidence is bound to the wrong world qualification")
        if result["training_world_seeds"] != list(
            _selected_world_seeds(world)["training"]
        ):
            raise ValueError("disturbance evidence is bound to the wrong training seeds")
        if index is not None:
            founders = _r4_founders(index)["training"]
            seeds = tuple(result["training_world_seeds"])
            for item in result["results"]:
                expected_manifest = _build_manifest(
                    "training",
                    founders,
                    seeds,
                    item["multiplier"],
                )
                if item["manifest_sha256"] != manifest_sha256(expected_manifest):
                    raise ValueError(
                        "disturbance result is bound to the wrong candidate manifest"
                    )
    if result["simulator_source_sha256"] != simulator_source_sha256():
        raise ValueError("disturbance evidence is bound to stale simulator source")
    return result


def run_disturbance_qualification(
    founder_index_path: str | Path,
    world_qualification_path: str | Path,
    output_path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
) -> dict[str, object]:
    """Run and publish the first-passing r5 disturbance gate."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("scientific disturbance qualification requires JAX GPU")
    world = validate_world_qualification(world_qualification_path)
    selected = _selected_world_seeds(world)
    training_seeds = selected["training"]
    index = load_founder_index(founder_index_path, verify_artifacts=False)
    founders = _r4_founders(index)["training"]
    results = []
    selected_multiplier = None
    for multiplier in ACTUATION_COST_MULTIPLIERS:
        manifest = _build_manifest("training", founders, training_seeds, multiplier)
        result = qualify_disturbance_manifest(founder_index_path, manifest)
        results.append(result)
        if result["passed"]:
            selected_multiplier = multiplier
            break
    value = disturbance_qualification_from_dict(
        {
            "founder_index_sha256": founder_index_sha256(index),
            "passed": selected_multiplier is not None,
            "protocol_revision": PROTOCOL_REVISION,
            "results": results,
            "schema_version": DISTURBANCE_QUALIFICATION_SCHEMA_VERSION,
            "selected_multiplier": selected_multiplier,
            "simulator_source_sha256": simulator_source_sha256(),
            "status": (
                "qualified"
                if selected_multiplier is not None
                else "stop_no_qualified_multiplier"
            ),
            "training_world_seeds": list(training_seeds),
            "world_qualification_sha256": _file_sha256(world_qualification_path),
        }
    )
    publish_disturbance_qualification(value, output_path)
    return validate_disturbance_qualification(
        output_path,
        founder_index_path=founder_index_path,
        world_qualification_path=world_qualification_path,
    )


def _load_training_manifest(path: str | Path) -> ScenarioManifest:
    payload = Path(path).read_bytes()
    manifest = manifest_from_json_bytes(payload)
    if payload != canonical_manifest_bytes(manifest) + b"\n":
        raise ValueError("training manifest is not canonically encoded")
    if manifest.partition != "training" or manifest.schema_version != R4_SCHEMA_VERSION:
        raise ValueError("operator opportunity requires an r4 training manifest")
    return manifest


def collect_opportunity_records(
    founder_index_path: str | Path,
    manifest: ScenarioManifest,
) -> list[dict[str, object]]:
    """Collect r4 opportunity observations from an explicit r5 manifest."""
    founder_genomes = _resolve_manifest_founders(manifest, founder_index_path)
    config = SimulatorConfig()
    base_env = build_environment(
        config,
        fixed_r4_policy(2),
        horizon=manifest.horizon,
        heredity_contract="r4",
        credit_chunk_steps=manifest.chunk_steps,
    )
    probe = jax.jit(
        lambda state, keys: _first_imminent_in_chunk(base_env, state, keys)
    )
    base_chunk = jax.jit(
        lambda state, keys: run_ecosystem_chunk(base_env, state, keys)
    )
    action_envs = [
        build_environment(
            config,
            fixed_r4_policy(action),
            horizon=manifest.horizon + ASSESSMENT_STEPS,
            heredity_contract="r4",
            credit_chunk_steps=manifest.chunk_steps,
        )
        for action in range(R4_NUM_OPERATORS)
    ]
    injectors = [
        jax.jit(
            lambda state, mutation_key, spawn_key, env=env: _inject_one_birth(
                env,
                state,
                mutation_key,
                spawn_key,
            )
        )
        for env in action_envs
    ]
    continue_target = jax.jit(
        lambda state, keys, child_id: _continue_target_child(
            base_env,
            state,
            keys,
            child_id,
        )
    )
    records = []
    shock_worlds = [
        world
        for world in manifest.worlds
        if world.event_kind is EventKind.ACTUATION_COST_SHIFT
    ]
    for world in shock_worlds:
        root_key = jax.random.PRNGKey(world.world_seed)
        _, reset_state = base_env.reset(
            root_key,
            founder_genome=founder_genomes[world.founder_id],
        )
        pre_state, event_state = _find_imminent_state(
            base_env,
            probe,
            base_chunk,
            reset_state,
            root_key,
            0,
            world.event_step,
            manifest.chunk_steps,
        )
        event_state, _ = _apply_event(base_env, event_state, world, root_key)
        post_state, _ = _find_imminent_state(
            base_env,
            probe,
            base_chunk,
            event_state,
            root_key,
            world.event_step,
            manifest.horizon,
            manifest.chunk_steps,
        )
        if pre_state is None or post_state is None:
            raise RuntimeError(f"no imminent-birth state for {world.pair_id}")
        for context, state in (("pre", pre_state), ("post", post_state)):
            action_scores = np.zeros(
                (R4_NUM_OPERATORS, MUTATION_REPEATS),
                dtype=np.float64,
            )
            resolved = np.zeros_like(action_scores, dtype=np.int64)
            child_ids = np.zeros_like(action_scores, dtype=np.int64)
            for action, (env, inject) in enumerate(
                zip(action_envs, injectors, strict=True)
            ):
                for repeat_index in range(MUTATION_REPEATS):
                    score, outcome, birth_count, child_id = _score_action(
                        env,
                        inject,
                        continue_target,
                        state,
                        root_key,
                        repeat_index,
                    )
                    if birth_count != 1:
                        raise RuntimeError(
                            "opportunity assay did not inject exactly one birth"
                        )
                    action_scores[action, repeat_index] = score
                    resolved[action, repeat_index] = outcome
                    child_ids[action, repeat_index] = child_id
            records.append(
                {
                    "action_scores": action_scores.tolist(),
                    "context": context,
                    "features": _state_features(base_env, state).tolist(),
                    "founder_id": world.founder_id,
                    "pair_id": world.pair_id,
                    "resolved": resolved.tolist(),
                    "target_child_ids": child_ids.tolist(),
                }
            )
    return records


_OPPORTUNITY_RECORD_KEYS = {
    "action_scores",
    "context",
    "features",
    "founder_id",
    "pair_id",
    "resolved",
    "target_child_ids",
}


def _validate_opportunity_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("opportunity record must be an object")
    _require_exact_keys(value, _OPPORTUNITY_RECORD_KEYS, "opportunity record")
    if value["context"] not in {"pre", "post"}:
        raise ValueError("opportunity context must be pre or post")
    if not isinstance(value["features"], list) or len(value["features"]) != 29:
        raise ValueError("opportunity record requires 29 readable features")
    for name in ("action_scores", "resolved", "target_child_ids"):
        array = np.asarray(value[name])
        if array.shape != (R4_NUM_OPERATORS, MUTATION_REPEATS):
            raise ValueError(f"{name} has the wrong r4 assay shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains nonfinite values")
    if not np.all(np.isin(np.asarray(value["resolved"]), (0, 1))):
        raise ValueError("resolved evidence must be binary")
    _canonical_json_bytes(value)
    return dict(value)


def opportunity_qualification_from_dict(
    value: Mapping[str, Any],
) -> dict[str, object]:
    """Strictly validate and normalize one r5 opportunity artifact."""
    if not isinstance(value, Mapping):
        raise ValueError("opportunity qualification must be an object")
    expected = {
        "founder_index_sha256",
        "manifest_sha256",
        "multiplier",
        "passed",
        "protocol_revision",
        "records",
        "schema_version",
        "summary",
    }
    _require_exact_keys(value, expected, "opportunity qualification")
    if value["schema_version"] != OPPORTUNITY_QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported opportunity-qualification schema")
    if value["protocol_revision"] != PROTOCOL_REVISION:
        raise ValueError("wrong r5 protocol revision")
    _require_sha256(value["founder_index_sha256"], "founder index hash")
    _require_sha256(value["manifest_sha256"], "training manifest hash")
    multiplier = _require_finite(value["multiplier"], "opportunity multiplier")
    if multiplier not in ACTUATION_COST_MULTIPLIERS:
        raise ValueError("opportunity multiplier is outside the frozen grid")
    if not isinstance(value["records"], list) or not value["records"]:
        raise ValueError("opportunity qualification requires records")
    records = [_validate_opportunity_record(item) for item in value["records"]]
    expected_summary = crossfit_opportunity(records)
    if value["summary"] != expected_summary:
        raise ValueError("opportunity summary does not match the frozen r4 cross-fit")
    if not isinstance(value["passed"], bool) or value["passed"] != expected_summary["passed"]:
        raise ValueError("top-level opportunity passed flag disagrees with summary")
    return {
        **dict(value),
        "multiplier": multiplier,
        "records": records,
        "summary": expected_summary,
    }


def publish_opportunity_qualification(
    value: Mapping[str, Any],
    path: str | Path = OPPORTUNITY_QUALIFICATION_PATH,
) -> str:
    return _publish_canonical_json(
        value,
        path,
        opportunity_qualification_from_dict,
    )


def validate_opportunity_qualification(
    path: str | Path = OPPORTUNITY_QUALIFICATION_PATH,
    *,
    founder_index_path: str | Path | None = None,
    training_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    result = _load_canonical_json(path, opportunity_qualification_from_dict)
    if founder_index_path is not None:
        index = load_founder_index(founder_index_path, verify_artifacts=False)
        if result["founder_index_sha256"] != founder_index_sha256(index):
            raise ValueError("opportunity evidence is bound to the wrong founder index")
    if training_manifest_path is not None:
        manifest = _load_training_manifest(training_manifest_path)
        if result["manifest_sha256"] != manifest_sha256(manifest):
            raise ValueError("opportunity evidence is bound to the wrong training manifest")
        expected = [
            (world.founder_id, world.pair_id, context)
            for world in manifest.worlds
            if world.event_kind is EventKind.ACTUATION_COST_SHIFT
            for context in ("pre", "post")
        ]
        actual = [
            (record["founder_id"], record["pair_id"], record["context"])
            for record in result["records"]
        ]
        if actual != expected:
            raise ValueError("opportunity records do not match the training manifest")
        multipliers = {
            world.event_parameters.multiplier
            for world in manifest.worlds
            if world.event_kind is EventKind.ACTUATION_COST_SHIFT
            and isinstance(world.event_parameters, ActuationCostShiftParameters)
        }
        if multipliers != {result["multiplier"]}:
            raise ValueError("opportunity evidence uses the wrong qualified multiplier")
    return result


def run_opportunity_qualification(
    founder_index_path: str | Path,
    training_manifest_path: str | Path,
    disturbance_qualification_path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
    output_path: str | Path = OPPORTUNITY_QUALIFICATION_PATH,
) -> dict[str, object]:
    """Run and publish the unchanged r4 cross-fit gate on an r5 manifest."""
    if jax.default_backend() != "gpu":
        raise RuntimeError("scientific operator-opportunity qualification requires JAX GPU")
    disturbance = validate_disturbance_qualification(
        disturbance_qualification_path,
        founder_index_path=founder_index_path,
    )
    if disturbance["passed"] is not True:
        raise RuntimeError("disturbance qualification did not pass")
    manifest = _load_training_manifest(training_manifest_path)
    multiplier = disturbance["selected_multiplier"]
    shocks = [
        world
        for world in manifest.worlds
        if world.event_kind is EventKind.ACTUATION_COST_SHIFT
    ]
    if not shocks or any(
        not isinstance(world.event_parameters, ActuationCostShiftParameters)
        or world.event_parameters.multiplier != multiplier
        for world in shocks
    ):
        raise ValueError("training manifest does not use the qualified multiplier")
    manifest_seeds = list(dict.fromkeys(world.world_seed for world in manifest.worlds))
    if manifest_seeds != disturbance["training_world_seeds"]:
        raise ValueError("training manifest does not use the qualified training worlds")
    selected_result = next(
        item
        for item in disturbance["results"]
        if item["multiplier"] == multiplier
    )
    if selected_result["manifest_sha256"] != manifest_sha256(manifest):
        raise ValueError("training manifest does not match the qualified disturbance")
    records = collect_opportunity_records(founder_index_path, manifest)
    summary = crossfit_opportunity(records)
    index = load_founder_index(founder_index_path, verify_artifacts=False)
    value = opportunity_qualification_from_dict(
        {
            "founder_index_sha256": founder_index_sha256(index),
            "manifest_sha256": manifest_sha256(manifest),
            "multiplier": multiplier,
            "passed": summary["passed"],
            "protocol_revision": PROTOCOL_REVISION,
            "records": records,
            "schema_version": OPPORTUNITY_QUALIFICATION_SCHEMA_VERSION,
            "summary": summary,
        }
    )
    publish_opportunity_qualification(value, output_path)
    return validate_opportunity_qualification(
        output_path,
        founder_index_path=founder_index_path,
        training_manifest_path=training_manifest_path,
    )


__all__ = [
    "build_and_publish_manifests",
    "run_disturbance_qualification",
    "run_opportunity_qualification",
    "run_qualified_founder_screening",
    "validate_disturbance_qualification",
    "validate_opportunity_qualification",
]
