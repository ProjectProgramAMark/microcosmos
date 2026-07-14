"""Qualification, manifest, and opportunity gates for Evo² R6."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import numpy as np

from microcosmos.heredity import R4_CLONE
from microcosmos.heredity import R4_NUM_OPERATORS
from microcosmos.heredity import R4_STANDARD_PARAMETRIC
from microcosmos.heredity import fixed_r4_policy
from microcosmos.rollout import run_ecosystem_chunk

from experiments.evo2_ecosystem.build_founder_bank import CandidateRange as BuilderCandidateRange
from experiments.evo2_ecosystem.build_founder_bank import ScreeningGates as BuilderScreeningGates
from experiments.evo2_ecosystem.build_founder_bank import ScreeningSpec as BuilderScreeningSpec
from experiments.evo2_ecosystem.build_founder_bank import run_founder_bank_builder
from experiments.evo2_ecosystem.episode import SimulatorConfig
from experiments.evo2_ecosystem.episode import _apply_event
from experiments.evo2_ecosystem.episode import _resolve_manifest_founders
from experiments.evo2_ecosystem.episode import _resolve_pair_indices
from experiments.evo2_ecosystem.episode import _step_keys
from experiments.evo2_ecosystem.episode import build_environment
from experiments.evo2_ecosystem.episode import evaluate_manifest
from experiments.evo2_ecosystem.episode import simulator_config_sha256
from experiments.evo2_ecosystem.episode import simulator_source_sha256
from experiments.evo2_ecosystem.founder_artifacts import founder_index_sha256
from experiments.evo2_ecosystem.founder_artifacts import load_founder_index
from experiments.evo2_ecosystem.heredity_adaptation_v4.manifest_generator import R4ManifestBundle
from experiments.evo2_ecosystem.heredity_adaptation_v4.manifest_generator import _founders
from experiments.evo2_ecosystem.heredity_adaptation_v4.qualification import _generation_gain
from experiments.evo2_ecosystem.heredity_adaptation_v4.qualification import _population_features
from experiments.evo2_ecosystem.heredity_adaptation_v4.qualification import uniform_exploration_policy
from experiments.evo2_ecosystem.heredity_adaptation_v4.opportunity import MUTATION_REPEATS
from experiments.evo2_ecosystem.heredity_adaptation_v4.opportunity import _continue_target_child
from experiments.evo2_ecosystem.heredity_adaptation_v4.opportunity import _find_imminent_state
from experiments.evo2_ecosystem.heredity_adaptation_v4.opportunity import _first_imminent_in_chunk
from experiments.evo2_ecosystem.heredity_adaptation_v4.opportunity import _inject_one_birth
from experiments.evo2_ecosystem.heredity_adaptation_v4.opportunity import _score_action
from experiments.evo2_ecosystem.heredity_adaptation_v4.opportunity import _state_features
from experiments.evo2_ecosystem.heredity_adaptation_v4.opportunity import crossfit_opportunity
from experiments.evo2_ecosystem.protocol import CONTROLLER_LAYOUT
from experiments.evo2_ecosystem.protocol import EventKind
from experiments.evo2_ecosystem.protocol import R4_SCHEMA_VERSION
from experiments.evo2_ecosystem.protocol import ScenarioManifest
from experiments.evo2_ecosystem.protocol import WorldScenario
from experiments.evo2_ecosystem.protocol import canonical_manifest_bytes
from experiments.evo2_ecosystem.protocol import manifest_from_json_bytes
from experiments.evo2_ecosystem.protocol import manifest_sha256
from experiments.evo2_ecosystem.r5.tools import _file_sha256
from experiments.evo2_ecosystem.r5.tools import _load_canonical_json
from experiments.evo2_ecosystem.r5.tools import _publish_canonical_json
from experiments.evo2_ecosystem.r5.tools import _publish_manifest_bundle
from experiments.evo2_ecosystem.r5.tools import _require_exact_keys
from experiments.evo2_ecosystem.r5.tools import _require_finite
from experiments.evo2_ecosystem.r5.tools import _require_sha256

from .protocol import ASSESSMENT_STEPS
from .protocol import BOOTSTRAP_REPLICATES
from .protocol import BOOTSTRAP_SEED
from .protocol import CHUNK_STEPS
from .protocol import CONTROL_PARAMETERS
from .protocol import DISTURBANCE_QUALIFICATION_PATH
from .protocol import EVENT_STEP
from .protocol import FOUNDER_PANEL_SEED
from .protocol import HORIZON
from .protocol import MANIFEST_ROOT
from .protocol import NUMERICAL_REPEATS
from .protocol import OPPORTUNITY_GATE_RECEIPT_PATH
from .protocol import POST_OPPORTUNITY_START
from .protocol import POST_OPPORTUNITY_STOP
from .protocol import PRE_OPPORTUNITY_START
from .protocol import PRE_OPPORTUNITY_STOP
from .protocol import PRIVATE_OPPORTUNITY_STAGING_PATH
from .protocol import PROTOCOL_REVISION
from .protocol import SCENARIO_FAMILY
from .protocol import SHOCK_PARAMETERS
from .protocol import ScreeningSpec
from .protocol import founder_screening_spec
from .protocol import scenario_id
from .qualify_worlds import validate_world_qualification


DISTURBANCE_QUALIFICATION_SCHEMA_VERSION = 1
_REQUIRED_FOUNDERS = {"training": 4, "development": 4, "sealed": 8}
_MANIFEST_FILENAMES = (
    "training_stable.json",
    "training_punctuated.json",
    "development.json",
    "sealed.json",
)
_DISTURBANCE_GATES = frozenset({"harm", "integrity", "observable_stress", "resolved_feedback", "viability"})
_VIABILITY_KEYS = {
    "distinct_birth",
    "distinct_reproducer",
    "integrity",
    "median_generation_gain",
    "median_post_births",
    "survival",
}


def _selected_world_seeds(value: Mapping[str, Any]) -> dict[str, tuple[int, int]]:
    selected = value.get("selected_seeds")
    expected = {"founder_eligibility", "training", "development", "sealed"}
    if not isinstance(selected, Mapping):
        raise ValueError("world qualification selected_seeds must be an object")
    _require_exact_keys(selected, expected, "selected world seeds")
    result = {}
    for partition in ("founder_eligibility", "training", "development", "sealed"):
        seeds = selected[partition]
        if not isinstance(seeds, list) or len(seeds) != 2 or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            raise ValueError(f"{partition} requires exactly two integer seeds")
        result[partition] = (seeds[0], seeds[1])
    return result


def builder_screening_spec(spec: ScreeningSpec) -> BuilderScreeningSpec:
    if not isinstance(spec, ScreeningSpec):
        raise TypeError("spec must be an R6 ScreeningSpec")
    return BuilderScreeningSpec(
        mode="r4_production",
        config=spec.config,
        seeds=spec.seeds,
        horizon=spec.horizon,
        chunk_steps=spec.chunk_steps,
        candidate_ranges=tuple(BuilderCandidateRange(item.partition, item.start, item.stop, item.required) for item in spec.candidate_ranges),
        gates=BuilderScreeningGates(**asdict(spec.gates)),
        selection_rule_id=spec.selection_rule_id,
    )


def run_qualified_founder_screening(
    world_qualification_path: str | Path,
    *,
    bank_directory: str | Path,
    screening_record_path: str | Path,
) -> dict[str, object]:
    world = validate_world_qualification(world_qualification_path)
    seeds = _selected_world_seeds(world)["founder_eligibility"]
    spec = builder_screening_spec(founder_screening_spec(seeds))
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


def _paired_worlds(partition: str, founder, seed: int) -> tuple[WorldScenario, WorldScenario]:
    pair_id = f"{partition}-{founder.founder_id}-{seed}"
    common = dict(
        world_seed=seed,
        pair_id=pair_id,
        scenario_family=SCENARIO_FAMILY,
        event_kind=EventKind.RESOURCE_RELOCATION,
        event_step=EVENT_STEP,
        founder_id=founder.founder_id,
        founder_sha256=founder.artifact_sha256,
    )
    return (
        WorldScenario(
            **common,
            scenario_id=scenario_id(pair_id, "same_location_refresh_control"),
            event_parameters=CONTROL_PARAMETERS,
        ),
        WorldScenario(
            **common,
            scenario_id=scenario_id(pair_id, "antipodal_relocation_shock"),
            event_parameters=SHOCK_PARAMETERS,
        ),
    )


def _build_manifest(
    partition: str,
    founders: Sequence[Any],
    seeds: tuple[int, int],
) -> ScenarioManifest:
    worlds = tuple(world for founder in founders for seed in seeds for world in _paired_worlds(partition, founder, seed))
    return ScenarioManifest(
        schema_version=R4_SCHEMA_VERSION,
        partition="sealed_final" if partition == "sealed" else partition,
        controller_layout=CONTROLLER_LAYOUT,
        simulator_config_sha256=simulator_config_sha256(SimulatorConfig()),
        horizon=HORIZON,
        chunk_steps=CHUNK_STEPS,
        worlds=worlds,
    )


def build_manifest_bundle(
    founder_index_path: str | Path,
    selected_seeds: Mapping[str, tuple[int, int]],
) -> R4ManifestBundle:
    _require_exact_keys(selected_seeds, set(_REQUIRED_FOUNDERS), "manifest seed panel")
    index = load_founder_index(founder_index_path, verify_artifacts=False)
    founders = _founders(index)
    training = _build_manifest("training", founders["training"], selected_seeds["training"])
    bundle = R4ManifestBundle(
        training_stable=training,
        training_punctuated=training,
        development=_build_manifest("development", founders["development"], selected_seeds["development"]),
        sealed=_build_manifest("sealed", founders["sealed"], selected_seeds["sealed"]),
    )
    validate_manifest_bundle(bundle, selected_seeds)
    return bundle


def validate_manifest_bundle(
    bundle: R4ManifestBundle,
    selected_seeds: Mapping[str, tuple[int, int]],
) -> None:
    if bundle.training_stable != bundle.training_punctuated:
        raise ValueError("stable and punctuated R6 training manifests must match")
    manifests = {
        "training": bundle.training_stable,
        "development": bundle.development,
        "sealed": bundle.sealed,
    }
    for partition, manifest in manifests.items():
        if (
            manifest.schema_version != R4_SCHEMA_VERSION
            or manifest.horizon != HORIZON
            or manifest.chunk_steps != CHUNK_STEPS
            or manifest.simulator_config_sha256 != simulator_config_sha256(SimulatorConfig())
        ):
            raise ValueError(f"{partition} manifest has the wrong frozen R6 envelope")
        actual_seeds = tuple(dict.fromkeys(world.world_seed for world in manifest.worlds))
        if actual_seeds != selected_seeds[partition]:
            raise ValueError(f"{partition} manifest uses the wrong R6 seeds")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, world in enumerate(manifest.worlds):
            grouped[world.pair_id].append(index)
        if len(grouped) != _REQUIRED_FOUNDERS[partition] * 2:
            raise ValueError(f"{partition} manifest has the wrong pair count")
        for indices in grouped.values():
            control_index, shock_index = _resolve_pair_indices(manifest, indices)
            control = manifest.worlds[control_index]
            shock = manifest.worlds[shock_index]
            if (
                control.event_step != EVENT_STEP
                or shock.event_step != EVENT_STEP
                or control.event_parameters != CONTROL_PARAMETERS
                or shock.event_parameters != SHOCK_PARAMETERS
                or (control.world_seed, control.founder_id, control.founder_sha256) != (shock.world_seed, shock.founder_id, shock.founder_sha256)
            ):
                raise ValueError("R6 control/shock pair does not match the frozen contract")


def build_and_publish_manifests(
    founder_index_path: str | Path,
    world_qualification_path: str | Path,
    disturbance_qualification_path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
    output_directory: str | Path = MANIFEST_ROOT,
) -> dict[str, str]:
    world = validate_world_qualification(world_qualification_path)
    disturbance = validate_disturbance_qualification(
        disturbance_qualification_path,
        founder_index_path=founder_index_path,
        world_qualification_path=world_qualification_path,
    )
    if disturbance["passed"] is not True:
        raise RuntimeError("R6 disturbance qualification did not pass")
    selected = _selected_world_seeds(world)
    bundle = build_manifest_bundle(
        founder_index_path,
        {partition: selected[partition] for partition in _REQUIRED_FOUNDERS},
    )
    return _publish_manifest_bundle(bundle, output_directory)


def _shock_and_control_episodes(manifest: ScenarioManifest, evaluation):
    controls = []
    shocks = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, world in enumerate(manifest.worlds):
        grouped[world.pair_id].append(index)
    for indices in grouped.values():
        control_index, shock_index = _resolve_pair_indices(manifest, indices)
        controls.append(evaluation.episodes[control_index])
        shocks.append(evaluation.episodes[shock_index])
    return tuple(controls), tuple(shocks)


def _pair_effects(manifest: ScenarioManifest, evaluation) -> list[tuple[str, float]]:
    effects = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, world in enumerate(manifest.worlds):
        grouped[world.pair_id].append(index)
    for indices in grouped.values():
        control_index, shock_index = _resolve_pair_indices(manifest, indices)
        world = manifest.worlds[shock_index]
        effects.append(
            (
                str(world.founder_id),
                float(evaluation.episodes[shock_index].primary_score) - float(evaluation.episodes[control_index].primary_score),
            )
        )
    return effects


def _founder_bootstrap_upper(effects: Sequence[tuple[str, float]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for founder, effect in effects:
        grouped[founder].append(effect)
    founder_means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(
        founder_means,
        size=(BOOTSTRAP_REPLICATES, len(founder_means)),
        replace=True,
    )
    return float(np.quantile(np.mean(samples, axis=1), 0.975))


def viability_metrics(episodes: Sequence[Any]) -> dict[str, object]:
    return {
        "survival": float(np.mean([bool(item.survived) for item in episodes])),
        "median_post_births": float(np.median([int(item.post_event_birth_count) for item in episodes])),
        "median_generation_gain": float(np.median([_generation_gain(item) for item in episodes])),
        "distinct_birth": int(sum(int(item.post_event_distinct_birth_count) for item in episodes)),
        "distinct_reproducer": int(sum(int(np.sum(np.asarray(item.post_event_resolved_distinct_success_count))) for item in episodes)),
        "integrity": bool(all(bool(item.integrity_valid) for item in episodes)),
    }


def viability_passes(value: Mapping[str, Any]) -> bool:
    _require_exact_keys(value, _VIABILITY_KEYS, "fixed-standard viability")
    return bool(
        float(value["survival"]) >= 0.75
        and float(value["median_post_births"]) >= 4
        and float(value["median_generation_gain"]) >= 1
        and int(value["distinct_birth"]) >= 1
        and int(value["distinct_reproducer"]) >= 1
        and value["integrity"] is True
    )


def _credit_probabilities(population) -> np.ndarray:
    credit = np.asarray(population.operator_success_ema, dtype=np.float64) * np.asarray(population.operator_evidence_ema, dtype=np.float64)
    return credit / max(float(np.sum(credit)), 1e-8)


def matched_observability(manifest: ScenarioManifest, evaluation) -> tuple[float, float]:
    """Compute final-step, founder-first control/relocation observability."""
    by_founder: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"control_features": [], "shock_features": [], "control_credit": [], "shock_credit": []}
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, world in enumerate(manifest.worlds):
        grouped[world.pair_id].append(index)
    threshold = SimulatorConfig().reproduction_threshold
    for indices in grouped.values():
        control_index, shock_index = _resolve_pair_indices(manifest, indices)
        founder = str(manifest.worlds[shock_index].founder_id)
        control = evaluation.episodes[control_index].final_population
        shock = evaluation.episodes[shock_index].final_population
        if control is None or shock is None:
            raise RuntimeError("R6 observability requires captured final populations")
        by_founder[founder]["control_features"].append(_population_features(control, threshold))
        by_founder[founder]["shock_features"].append(_population_features(shock, threshold))
        by_founder[founder]["control_credit"].append(_credit_probabilities(control))
        by_founder[founder]["shock_credit"].append(_credit_probabilities(shock))
    control_features = []
    shock_features = []
    credit_tvs = []
    for founder in sorted(by_founder):
        record = by_founder[founder]
        control_features.append(np.mean(record["control_features"], axis=0))
        shock_features.append(np.mean(record["shock_features"], axis=0))
        control_credit = np.mean(record["control_credit"], axis=0)
        shock_credit = np.mean(record["shock_credit"], axis=0)
        credit_tvs.append(0.5 * np.sum(np.abs(shock_credit - control_credit)))
    controls = np.stack(control_features)
    shocks = np.stack(shock_features)
    pooled = np.sqrt((np.var(controls, axis=0, ddof=1) + np.var(shocks, axis=0, ddof=1)) / 2 + 1e-8)
    smd = np.max(np.abs(np.mean(shocks, axis=0) - np.mean(controls, axis=0)) / pooled)
    return float(smd), float(np.mean(credit_tvs))


def _expected_disturbance_gates(value: Mapping[str, Any]) -> dict[str, bool]:
    standard_viability = value["standard_viability"]
    pre_resolved = np.asarray(value["pre_resolved_by_operator"], dtype=np.int64)
    post_resolved = np.asarray(value["post_resolved_by_operator"], dtype=np.int64)
    pre_evidence = np.asarray(value["pre_evidence_mean"], dtype=np.float64)
    final_evidence = np.asarray(value["final_evidence_mean"], dtype=np.float64)
    for name, array in (
        ("pre resolved", pre_resolved),
        ("post resolved", post_resolved),
        ("pre evidence", pre_evidence),
        ("final evidence", final_evidence),
    ):
        if array.shape != (R4_NUM_OPERATORS,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain six finite values")
    exercised = (pre_evidence >= 0.10) & (final_evidence >= 0.10)
    integrity = value["policy_integrity"]
    return {
        "harm": float(value["clone_effect_mean"]) <= -0.03 and float(value["clone_effect_founder_bootstrap_upper_95"]) < 0.0,
        "viability": viability_passes(standard_viability),
        "resolved_feedback": bool(int(np.sum(pre_resolved)) >= 24 and int(np.sum(post_resolved)) >= 24 and int(np.sum(exercised[1:])) >= 2),
        "observable_stress": float(value["observable_feature_max_smd"]) >= 0.50 or float(value["observable_credit_tv"]) >= 0.20,
        "integrity": bool(all(integrity.values())),
    }


def qualify_disturbance_manifest(
    founder_index_path: str | Path,
    manifest: ScenarioManifest,
) -> dict[str, object]:
    config = SimulatorConfig()
    common = dict(
        capture_finalists=True,
        numerical_repeats=NUMERICAL_REPEATS,
        founder_index_path=founder_index_path,
    )
    clone = evaluate_manifest(manifest, config, fixed_r4_policy(R4_CLONE), **common)
    standard = evaluate_manifest(manifest, config, fixed_r4_policy(R4_STANDARD_PARAMETRIC), **common)
    exploration = evaluate_manifest(manifest, config, uniform_exploration_policy, **common)
    effects = _pair_effects(manifest, clone)
    clone_control, clone_shock = _shock_and_control_episodes(manifest, clone)
    del clone_control
    _, standard_shock = _shock_and_control_episodes(manifest, standard)
    _, exploration_shock = _shock_and_control_episodes(manifest, exploration)
    pre_resolved = np.sum(
        np.stack([np.asarray(item.pre_event_resolved_count) for item in exploration_shock]),
        axis=0,
    )
    post_resolved = np.sum(
        np.stack([np.asarray(item.post_event_resolved_count) for item in exploration_shock]),
        axis=0,
    )
    pre_evidence = np.mean(
        np.stack([np.asarray(item.shock_population.operator_evidence_ema) for item in exploration_shock]),
        axis=0,
    )
    final_evidence = np.mean(
        np.stack([np.asarray(item.operator_evidence_ema) for item in exploration_shock]),
        axis=0,
    )
    smd, credit_tv = matched_observability(manifest, exploration)
    value: dict[str, object] = {
        "manifest_sha256": manifest_sha256(manifest),
        "clone_effect_mean": float(np.mean([effect for _, effect in effects])),
        "clone_effect_founder_bootstrap_upper_95": _founder_bootstrap_upper(effects),
        "clone_viability": viability_metrics(clone_shock),
        "standard_viability": viability_metrics(standard_shock),
        "pre_resolved_by_operator": pre_resolved.tolist(),
        "post_resolved_by_operator": post_resolved.tolist(),
        "pre_evidence_mean": pre_evidence.tolist(),
        "final_evidence_mean": final_evidence.tolist(),
        "observable_feature_max_smd": smd,
        "observable_credit_tv": credit_tv,
        "policy_integrity": {
            "clone": bool(clone.integrity_valid),
            "standard": bool(standard.integrity_valid),
            "exploration": bool(exploration.integrity_valid),
        },
    }
    gates = _expected_disturbance_gates(value)
    return {**value, "gates": gates, "passed": all(gates.values())}


def _validate_viability(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_exact_keys(value, _VIABILITY_KEYS, context)
    result = {
        "survival": _require_finite(value["survival"], f"{context} survival"),
        "median_post_births": _require_finite(value["median_post_births"], f"{context} median births"),
        "median_generation_gain": _require_finite(value["median_generation_gain"], f"{context} generation gain"),
        "distinct_birth": value["distinct_birth"],
        "distinct_reproducer": value["distinct_reproducer"],
        "integrity": value["integrity"],
    }
    if any(
        not isinstance(result[name], int) or isinstance(result[name], bool) or result[name] < 0 for name in ("distinct_birth", "distinct_reproducer")
    ) or not isinstance(result["integrity"], bool):
        raise ValueError(f"{context} counts/integrity are invalid")
    return result


def disturbance_qualification_from_dict(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "founder_index_sha256",
        "manifest_sha256",
        "passed",
        "protocol_revision",
        "result",
        "schema_version",
        "simulator_source_sha256",
        "status",
        "training_world_seeds",
        "world_qualification_sha256",
    }
    if not isinstance(value, Mapping):
        raise ValueError("disturbance qualification must be an object")
    _require_exact_keys(value, expected, "disturbance qualification")
    if value["schema_version"] != DISTURBANCE_QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported R6 disturbance schema")
    if value["protocol_revision"] != PROTOCOL_REVISION:
        raise ValueError("wrong R6 protocol revision")
    for name in ("founder_index_sha256", "manifest_sha256", "simulator_source_sha256", "world_qualification_sha256"):
        _require_sha256(value[name], name)
    seeds = value["training_world_seeds"]
    if not isinstance(seeds, list) or len(seeds) != 2 or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("R6 disturbance requires two training seeds")
    result = value["result"]
    result_keys = {
        "clone_effect_founder_bootstrap_upper_95",
        "clone_effect_mean",
        "clone_viability",
        "final_evidence_mean",
        "gates",
        "manifest_sha256",
        "observable_credit_tv",
        "observable_feature_max_smd",
        "passed",
        "policy_integrity",
        "post_resolved_by_operator",
        "pre_evidence_mean",
        "pre_resolved_by_operator",
        "standard_viability",
    }
    if not isinstance(result, Mapping):
        raise ValueError("disturbance result must be an object")
    _require_exact_keys(result, result_keys, "disturbance result")
    normalized_result = dict(result)
    normalized_result["clone_viability"] = _validate_viability(result["clone_viability"], "clone viability")
    normalized_result["standard_viability"] = _validate_viability(result["standard_viability"], "standard viability")
    if result["manifest_sha256"] != value["manifest_sha256"]:
        raise ValueError("disturbance manifest hashes disagree")
    integrity = result["policy_integrity"]
    if (
        not isinstance(integrity, Mapping)
        or set(integrity) != {"clone", "standard", "exploration"}
        or any(not isinstance(item, bool) for item in integrity.values())
    ):
        raise ValueError("invalid policy integrity")
    expected_gates = _expected_disturbance_gates(normalized_result)
    if result["gates"] != expected_gates or result["passed"] != all(expected_gates.values()):
        raise ValueError("disturbance gates disagree with stored metrics")
    expected_passed = bool(result["passed"])
    if value["passed"] is not expected_passed or value["status"] != ("qualified" if expected_passed else "stop_no_qualified_relocation"):
        raise ValueError("top-level disturbance status disagrees with result")
    return {**dict(value), "result": {**normalized_result, "gates": expected_gates}}


def publish_disturbance_qualification(
    value: Mapping[str, Any],
    path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
) -> str:
    return _publish_canonical_json(value, path, disturbance_qualification_from_dict)


def validate_disturbance_qualification(
    path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
    *,
    founder_index_path: str | Path | None = None,
    world_qualification_path: str | Path | None = None,
) -> dict[str, object]:
    result = _load_canonical_json(path, disturbance_qualification_from_dict)
    if founder_index_path is not None:
        index = load_founder_index(founder_index_path, verify_artifacts=False)
        if result["founder_index_sha256"] != founder_index_sha256(index):
            raise ValueError("disturbance is bound to the wrong founder index")
    if world_qualification_path is not None:
        world = validate_world_qualification(world_qualification_path)
        if result["world_qualification_sha256"] != _file_sha256(world_qualification_path):
            raise ValueError("disturbance is bound to the wrong world qualification")
        if result["training_world_seeds"] != list(_selected_world_seeds(world)["training"]):
            raise ValueError("disturbance is bound to the wrong training seeds")
    if result["simulator_source_sha256"] != simulator_source_sha256():
        raise ValueError("disturbance is bound to stale simulator source")
    return result


def run_disturbance_qualification(
    founder_index_path: str | Path,
    world_qualification_path: str | Path,
    output_path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
) -> dict[str, object]:
    if jax.default_backend() != "gpu":
        raise RuntimeError("scientific R6 disturbance qualification requires JAX GPU")
    world = validate_world_qualification(world_qualification_path)
    training_seeds = _selected_world_seeds(world)["training"]
    index = load_founder_index(founder_index_path, verify_artifacts=False)
    founders = _founders(index)["training"]
    manifest = _build_manifest("training", founders, training_seeds)
    result = qualify_disturbance_manifest(founder_index_path, manifest)
    value = disturbance_qualification_from_dict(
        {
            "founder_index_sha256": founder_index_sha256(index),
            "manifest_sha256": manifest_sha256(manifest),
            "passed": result["passed"],
            "protocol_revision": PROTOCOL_REVISION,
            "result": result,
            "schema_version": DISTURBANCE_QUALIFICATION_SCHEMA_VERSION,
            "simulator_source_sha256": simulator_source_sha256(),
            "status": ("qualified" if result["passed"] else "stop_no_qualified_relocation"),
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


def load_training_manifest(path: str | Path) -> ScenarioManifest:
    payload = Path(path).read_bytes()
    manifest = manifest_from_json_bytes(payload)
    if payload != canonical_manifest_bytes(manifest) + b"\n":
        raise ValueError("training manifest is not canonical")
    if manifest.partition != "training" or manifest.schema_version != R4_SCHEMA_VERSION:
        raise ValueError("R6 opportunity requires a schema-v3 training manifest")
    return manifest


def _advance_state(compiled_chunk, state, root_key, start: int, stop: int):
    if start % CHUNK_STEPS or stop % CHUNK_STEPS or stop < start:
        raise ValueError("R6 natural-state advances must use frozen chunk boundaries")
    current = state
    for chunk_start in range(start, stop, CHUNK_STEPS):
        current, _ = compiled_chunk(
            current,
            _step_keys(root_key, chunk_start, CHUNK_STEPS),
        )
    return current


def _find_imminent_state_checked(
    env,
    compiled_probe,
    compiled_chunk,
    state,
    root_key,
    *,
    state_step: int,
    start: int,
    stop: int,
):
    """Call the inherited helper only with a state at its declared start."""
    if state_step != start or int(np.asarray(state.time)) != start:
        raise ValueError("imminent-state input does not correspond to the search start")
    return _find_imminent_state(
        env,
        compiled_probe,
        compiled_chunk,
        state,
        root_key,
        start,
        stop,
        CHUNK_STEPS,
    )


def collect_opportunity_records(
    founder_index_path: str | Path,
    manifest: ScenarioManifest,
) -> list[dict[str, object]]:
    """Collect the private late-pre/settled-post six-action assay records."""
    founder_genomes = _resolve_manifest_founders(manifest, founder_index_path)
    config = SimulatorConfig()
    base_env = build_environment(
        config,
        fixed_r4_policy(R4_STANDARD_PARAMETRIC),
        horizon=manifest.horizon,
        heredity_contract="r4",
        credit_chunk_steps=manifest.chunk_steps,
    )
    probe = jax.jit(lambda state, keys: _first_imminent_in_chunk(base_env, state, keys))
    base_chunk = jax.jit(lambda state, keys: run_ecosystem_chunk(base_env, state, keys))
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
    injectors = [jax.jit(lambda state, mutation_key, spawn_key, env=env: _inject_one_birth(env, state, mutation_key, spawn_key)) for env in action_envs]
    continue_target = jax.jit(lambda state, keys, child_id: _continue_target_child(base_env, state, keys, child_id))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, world in enumerate(manifest.worlds):
        grouped[world.pair_id].append(index)
    shock_worlds = [manifest.worlds[_resolve_pair_indices(manifest, indices)[1]] for indices in grouped.values()]
    records: list[dict[str, object]] = []
    for world in shock_worlds:
        root_key = jax.random.PRNGKey(world.world_seed)
        _, reset_state = base_env.reset(
            root_key,
            founder_genome=founder_genomes[world.founder_id],
        )
        state_6000 = _advance_state(
            base_chunk,
            reset_state,
            root_key,
            0,
            PRE_OPPORTUNITY_START,
        )
        pre_state, _ = _find_imminent_state_checked(
            base_env,
            probe,
            base_chunk,
            state_6000,
            root_key,
            state_step=PRE_OPPORTUNITY_START,
            start=PRE_OPPORTUNITY_START,
            stop=PRE_OPPORTUNITY_STOP,
        )
        event_state = _advance_state(
            base_chunk,
            state_6000,
            root_key,
            PRE_OPPORTUNITY_START,
            EVENT_STEP,
        )
        event_state, _ = _apply_event(base_env, event_state, world, root_key)
        state_8000 = _advance_state(
            base_chunk,
            event_state,
            root_key,
            EVENT_STEP,
            POST_OPPORTUNITY_START,
        )
        post_state, _ = _find_imminent_state_checked(
            base_env,
            probe,
            base_chunk,
            state_8000,
            root_key,
            state_step=POST_OPPORTUNITY_START,
            start=POST_OPPORTUNITY_START,
            stop=POST_OPPORTUNITY_STOP,
        )
        if pre_state is None or post_state is None:
            raise RuntimeError(f"no imminent-birth state in a frozen R6 window: {world.pair_id}")
        for context, state in (("pre", pre_state), ("post", post_state)):
            action_scores = np.zeros((R4_NUM_OPERATORS, MUTATION_REPEATS), dtype=np.float64)
            resolved = np.zeros_like(action_scores, dtype=np.int64)
            child_ids = np.zeros_like(action_scores, dtype=np.int64)
            for action, (env, inject) in enumerate(zip(action_envs, injectors, strict=True)):
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
                        raise RuntimeError("R6 opportunity assay did not inject one birth")
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
                    "state_step": int(np.asarray(state.time)),
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
    "state_step",
    "target_child_ids",
}


def _validate_opportunity_record(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("opportunity record must be an object")
    _require_exact_keys(value, _OPPORTUNITY_RECORD_KEYS, "opportunity record")
    if value["context"] not in {"pre", "post"}:
        raise ValueError("opportunity context must be pre or post")
    expected_window = (PRE_OPPORTUNITY_START, PRE_OPPORTUNITY_STOP) if value["context"] == "pre" else (POST_OPPORTUNITY_START, POST_OPPORTUNITY_STOP)
    if not isinstance(value["state_step"], int) or isinstance(value["state_step"], bool) or not expected_window[0] <= value["state_step"] < expected_window[1]:
        raise ValueError("opportunity state lies outside its frozen window")
    if not isinstance(value["features"], list) or len(value["features"]) != 29:
        raise ValueError("opportunity record requires 29 readable features")
    for name in ("action_scores", "resolved", "target_child_ids"):
        array = np.asarray(value[name])
        if array.shape != (R4_NUM_OPERATORS, MUTATION_REPEATS) or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} has the wrong finite six-by-eight shape")
    if not np.all(np.isin(np.asarray(value["resolved"]), (0, 1))):
        raise ValueError("resolved evidence must be binary")
    return dict(value)


def private_opportunity_from_dict(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "founder_index_sha256",
        "manifest_sha256",
        "protocol_revision",
        "records",
        "schema_version",
        "summary",
    }
    if not isinstance(value, Mapping):
        raise ValueError("private opportunity artifact must be an object")
    _require_exact_keys(value, expected, "private opportunity artifact")
    if value["schema_version"] != 1 or value["protocol_revision"] != PROTOCOL_REVISION:
        raise ValueError("wrong private opportunity protocol")
    _require_sha256(value["founder_index_sha256"], "founder index hash")
    _require_sha256(value["manifest_sha256"], "manifest hash")
    if not isinstance(value["records"], list) or not value["records"]:
        raise ValueError("private opportunity artifact requires records")
    records = [_validate_opportunity_record(record) for record in value["records"]]
    crossfit_records = [{key: item[key] for key in _OPPORTUNITY_RECORD_KEYS if key != "state_step"} for item in records]
    summary = crossfit_opportunity(crossfit_records)
    if value["summary"] != summary:
        raise ValueError("private opportunity summary does not recompute")
    return {**dict(value), "records": records, "summary": summary}


def _public_opportunity_summary(summary: Mapping[str, Any]) -> dict[str, object]:
    exercised = summary["exercised_nonclone_actions"]
    pre = np.asarray(summary["pre_resolved_by_operator"], dtype=np.int64)
    post = np.asarray(summary["post_resolved_by_operator"], dtype=np.int64)
    return {
        "exercised_nonclone_action_count": len(exercised),
        "founder_bootstrap_lower_95": summary["founder_bootstrap_lower_95"],
        "gates": dict(summary["gates"]),
        "mean_advantage": summary["mean_advantage"],
        "minimum_selected_post_resolved": int(min(post[action] for action in exercised)),
        "minimum_selected_pre_resolved": int(min(pre[action] for action in exercised)),
        "passed": summary["passed"],
    }


def opportunity_gate_receipt_from_dict(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "founder_index_sha256",
        "manifest_sha256",
        "passed",
        "private_artifact_sha256",
        "protocol_revision",
        "schema_version",
        "summary",
        "thresholds",
    }
    if not isinstance(value, Mapping):
        raise ValueError("opportunity gate receipt must be an object")
    _require_exact_keys(value, expected, "opportunity gate receipt")
    if value["schema_version"] != 1 or value["protocol_revision"] != PROTOCOL_REVISION:
        raise ValueError("wrong opportunity gate protocol")
    for name in ("founder_index_sha256", "manifest_sha256", "private_artifact_sha256"):
        _require_sha256(value[name], name)
    expected_thresholds = {
        "mean_advantage": 0.03,
        "founder_bootstrap_lower_strictly_greater_than": 0.0,
        "minimum_nonclone_actions": 2,
        "minimum_resolved_draws_each_period": 8,
    }
    if value["thresholds"] != expected_thresholds:
        raise ValueError("opportunity receipt thresholds changed")
    summary = value["summary"]
    expected_summary_keys = {
        "exercised_nonclone_action_count",
        "founder_bootstrap_lower_95",
        "gates",
        "mean_advantage",
        "minimum_selected_post_resolved",
        "minimum_selected_pre_resolved",
        "passed",
    }
    if not isinstance(summary, Mapping):
        raise ValueError("opportunity receipt summary must be an object")
    _require_exact_keys(summary, expected_summary_keys, "opportunity receipt summary")
    gates = summary["gates"]
    if (
        not isinstance(gates, Mapping)
        or set(gates)
        != {
            "mean_advantage",
            "lower_interval",
            "two_nonclone_actions",
            "outcome_evidence",
        }
        or any(not isinstance(item, bool) for item in gates.values())
    ):
        raise ValueError("opportunity receipt gates are invalid")
    if summary["passed"] != all(gates.values()) or value["passed"] != summary["passed"]:
        raise ValueError("opportunity receipt pass flag disagrees with gates")
    return {**dict(value), "summary": {**dict(summary), "gates": dict(gates)}}


def publish_private_opportunity(
    value: Mapping[str, Any],
    path: str | Path = PRIVATE_OPPORTUNITY_STAGING_PATH,
) -> str:
    return _publish_canonical_json(value, path, private_opportunity_from_dict)


def publish_opportunity_gate_receipt(
    value: Mapping[str, Any],
    path: str | Path = OPPORTUNITY_GATE_RECEIPT_PATH,
) -> str:
    return _publish_canonical_json(value, path, opportunity_gate_receipt_from_dict)


def validate_private_opportunity(path: str | Path) -> dict[str, object]:
    return _load_canonical_json(path, private_opportunity_from_dict)


def validate_opportunity_gate_receipt(
    path: str | Path = OPPORTUNITY_GATE_RECEIPT_PATH,
    *,
    private_path: str | Path | None = None,
    founder_index_path: str | Path | None = None,
    training_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    receipt = _load_canonical_json(path, opportunity_gate_receipt_from_dict)
    if private_path is not None:
        private = validate_private_opportunity(private_path)
        if receipt["private_artifact_sha256"] != _file_sha256(private_path):
            raise ValueError("opportunity receipt is bound to the wrong private bytes")
        if (
            receipt["founder_index_sha256"] != private["founder_index_sha256"]
            or receipt["manifest_sha256"] != private["manifest_sha256"]
            or receipt["summary"] != _public_opportunity_summary(private["summary"])
        ):
            raise ValueError("opportunity receipt disagrees with private evidence")
    if founder_index_path is not None:
        index = load_founder_index(founder_index_path, verify_artifacts=False)
        if receipt["founder_index_sha256"] != founder_index_sha256(index):
            raise ValueError("opportunity receipt uses the wrong founder index")
    if training_manifest_path is not None:
        manifest = load_training_manifest(training_manifest_path)
        if receipt["manifest_sha256"] != manifest_sha256(manifest):
            raise ValueError("opportunity receipt uses the wrong training manifest")
    return receipt


def run_opportunity_qualification(
    founder_index_path: str | Path,
    training_manifest_path: str | Path,
    disturbance_qualification_path: str | Path = DISTURBANCE_QUALIFICATION_PATH,
    *,
    private_output_path: str | Path = PRIVATE_OPPORTUNITY_STAGING_PATH,
    receipt_output_path: str | Path = OPPORTUNITY_GATE_RECEIPT_PATH,
) -> dict[str, object]:
    if jax.default_backend() != "gpu":
        raise RuntimeError("scientific R6 opportunity qualification requires JAX GPU")
    disturbance = validate_disturbance_qualification(
        disturbance_qualification_path,
        founder_index_path=founder_index_path,
    )
    if disturbance["passed"] is not True:
        raise RuntimeError("R6 disturbance qualification did not pass")
    manifest = load_training_manifest(training_manifest_path)
    if disturbance["manifest_sha256"] != manifest_sha256(manifest):
        raise ValueError("opportunity manifest differs from qualified relocation")
    records = collect_opportunity_records(founder_index_path, manifest)
    crossfit_records = [{key: item[key] for key in _OPPORTUNITY_RECORD_KEYS if key != "state_step"} for item in records]
    summary = crossfit_opportunity(crossfit_records)
    index = load_founder_index(founder_index_path, verify_artifacts=False)
    private_value = private_opportunity_from_dict(
        {
            "founder_index_sha256": founder_index_sha256(index),
            "manifest_sha256": manifest_sha256(manifest),
            "protocol_revision": PROTOCOL_REVISION,
            "records": records,
            "schema_version": 1,
            "summary": summary,
        }
    )
    publish_private_opportunity(private_value, private_output_path)
    private_hash = _file_sha256(private_output_path)
    receipt_value = opportunity_gate_receipt_from_dict(
        {
            "founder_index_sha256": founder_index_sha256(index),
            "manifest_sha256": manifest_sha256(manifest),
            "passed": summary["passed"],
            "private_artifact_sha256": private_hash,
            "protocol_revision": PROTOCOL_REVISION,
            "schema_version": 1,
            "summary": _public_opportunity_summary(summary),
            "thresholds": {
                "mean_advantage": 0.03,
                "founder_bootstrap_lower_strictly_greater_than": 0.0,
                "minimum_nonclone_actions": 2,
                "minimum_resolved_draws_each_period": 8,
            },
        }
    )
    publish_opportunity_gate_receipt(receipt_value, receipt_output_path)
    return validate_opportunity_gate_receipt(
        receipt_output_path,
        private_path=private_output_path,
        founder_index_path=founder_index_path,
        training_manifest_path=training_manifest_path,
    )


__all__ = [
    "build_and_publish_manifests",
    "build_manifest_bundle",
    "builder_screening_spec",
    "collect_opportunity_records",
    "disturbance_qualification_from_dict",
    "opportunity_gate_receipt_from_dict",
    "private_opportunity_from_dict",
    "load_training_manifest",
    "matched_observability",
    "publish_disturbance_qualification",
    "publish_opportunity_gate_receipt",
    "publish_private_opportunity",
    "qualify_disturbance_manifest",
    "run_disturbance_qualification",
    "run_opportunity_qualification",
    "run_qualified_founder_screening",
    "validate_disturbance_qualification",
    "validate_manifest_bundle",
    "validate_opportunity_gate_receipt",
    "validate_private_opportunity",
    "viability_metrics",
    "viability_passes",
]
