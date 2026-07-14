"""R6 role-aware primary, lineage, common-garden, and ablation analysis."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from experiments.evo2_ecosystem.episode import EpisodeResult
from experiments.evo2_ecosystem.episode import ManifestEvaluation
from experiments.evo2_ecosystem.episode import PairedManifestEvaluation
from experiments.evo2_ecosystem.episode import SimulatorConfig
from experiments.evo2_ecosystem.episode import _resolve_pair_indices
from experiments.evo2_ecosystem.episode import simulator_config_sha256
from experiments.evo2_ecosystem.heredity_adaptation_v4.analysis import LineagePair
from experiments.evo2_ecosystem.heredity_adaptation_v4.analysis import extract_lineage_pairs
from experiments.evo2_ecosystem.heredity_adaptation_v4.analysis import founder_bootstrap_interval
from experiments.evo2_ecosystem.protocol import ScenarioManifest
from experiments.evo2_ecosystem.protocol import manifest_sha256
from experiments.evo2_ecosystem.r5.analysis import common_garden_analysis
from experiments.evo2_ecosystem.r5.analysis import dominant_action_ablation
from experiments.evo2_ecosystem.r5.analysis import no_credit_ablation

from .protocol import BOOTSTRAP_SEED
from .protocol import SHOCK_CENTER


PRIMARY_EFFECT_THRESHOLD = 0.02
SHAM_NONINFERIORITY_MARGIN = -0.02


def _finite(value: object, label: str) -> float:
    result = float(np.asarray(value))
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _role_indices(manifest: ScenarioManifest) -> tuple[tuple[int, ...], tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for index, world in enumerate(manifest.worlds):
        grouped.setdefault(world.pair_id, []).append(index)
    controls = []
    shocks = []
    for pair_id in sorted(grouped):
        control, shock = _resolve_pair_indices(manifest, grouped[pair_id])
        controls.append(control)
        shocks.append(shock)
    return tuple(controls), tuple(shocks)


def _candidate_reference_effects(
    manifest: ScenarioManifest,
    candidate: Sequence[EpisodeResult],
    reference: Sequence[EpisodeResult],
    *,
    role: str,
) -> list[tuple[str, float]]:
    controls, shocks = _role_indices(manifest)
    indices = controls if role == "control" else shocks if role == "shock" else None
    if indices is None:
        raise ValueError("role must be control or shock")
    return [
        (
            str(manifest.worlds[index].founder_id),
            _finite(candidate[index].primary_score, "candidate score") - _finite(reference[index].primary_score, "reference score"),
        )
        for index in indices
    ]


def _shock_minus_control_effects(
    manifest: ScenarioManifest,
    episodes: Sequence[EpisodeResult],
) -> list[tuple[str, float]]:
    controls, shocks = _role_indices(manifest)
    return [
        (
            str(manifest.worlds[shock].founder_id),
            _finite(episodes[shock].primary_score, "shock score") - _finite(episodes[control].primary_score, "control score"),
        )
        for control, shock in zip(controls, shocks, strict=True)
    ]


def primary_result_summary(
    manifest: ScenarioManifest,
    primary: PairedManifestEvaluation,
    clone: PairedManifestEvaluation,
) -> dict[str, object]:
    """Compute the four frozen R6 primary gates using exact scenario roles."""
    if not bool(np.asarray(primary.integrity_valid)) or not bool(np.asarray(clone.integrity_valid)):
        raise ValueError("R6 primary analysis requires integrity-valid evaluations")
    primary_effects = _candidate_reference_effects(
        manifest,
        primary.episodes,
        primary.ancestor_episodes,
        role="shock",
    )
    control_effects = _candidate_reference_effects(
        manifest,
        primary.episodes,
        primary.ancestor_episodes,
        role="control",
    )
    clone_effects = _shock_minus_control_effects(manifest, clone.episodes)
    primary_interval = founder_bootstrap_interval(primary_effects, seed=BOOTSTRAP_SEED)
    control_interval = founder_bootstrap_interval(control_effects, seed=BOOTSTRAP_SEED)
    clone_interval = founder_bootstrap_interval(clone_effects, seed=BOOTSTRAP_SEED)
    gates = {
        "clone_relocation_harm": clone_interval["upper_95"] < 0.0,
        "minimum_primary_mean": primary_interval["mean"] >= PRIMARY_EFFECT_THRESHOLD,
        "positive_primary_interval": primary_interval["lower_95"] > 0.0,
        "control_noninferiority": control_interval["lower_95"] > SHAM_NONINFERIORITY_MARGIN,
    }
    return {
        "primary_relocation_auc_delta": primary_interval,
        "control_absolute_auc_delta": control_interval,
        "clone_relocation_minus_control": clone_interval,
        "thresholds": {
            "primary_mean": PRIMARY_EFFECT_THRESHOLD,
            "control_lower_bound": SHAM_NONINFERIORITY_MARGIN,
            "clone_upper_bound": 0.0,
            "primary_lower_bound": 0.0,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def extract_relocation_lineage_pairs(
    manifest: ScenarioManifest,
    evaluation: PairedManifestEvaluation,
) -> tuple[LineagePair, ...]:
    """Reuse the trusted extractor after selecting only exact relocation worlds."""
    _, shocks = _role_indices(manifest)
    shock_manifest = replace(manifest, worlds=tuple(manifest.worlds[index] for index in shocks))
    shock_evaluation = ManifestEvaluation(
        episodes=tuple(evaluation.episodes[index] for index in shocks),
        candidate_score=evaluation.candidate_score,
        integrity_valid=evaluation.integrity_valid,
        repeat_scores=evaluation.repeat_scores,
        selected_repeat_index=evaluation.selected_repeat_index,
    )
    return extract_lineage_pairs(shock_manifest, shock_evaluation)


def _ablation_summary(
    manifest: ScenarioManifest,
    primary: PairedManifestEvaluation,
    ablation: PairedManifestEvaluation,
) -> dict[str, float]:
    if not bool(np.asarray(primary.integrity_valid)) or not bool(np.asarray(ablation.integrity_valid)):
        raise ValueError("R6 ablation analysis requires integrity-valid evaluations")
    return founder_bootstrap_interval(
        _candidate_reference_effects(
            manifest,
            primary.episodes,
            ablation.episodes,
            role="shock",
        ),
        seed=BOOTSTRAP_SEED,
    )


def _write_json_once(path: Path, value: Mapping[str, object]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to replace R6 final analysis at {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_final_analysis(
    manifest: ScenarioManifest,
    primary: PairedManifestEvaluation,
    clone: PairedManifestEvaluation,
    *,
    output_path: str | Path | None = None,
    lineage_pairs: tuple[LineagePair, ...] = (),
    ablations: Mapping[str, PairedManifestEvaluation] | None = None,
    adaptive_eligible: bool = False,
) -> dict[str, object]:
    """Assemble R6 performance and conditional mechanism evidence."""
    primary_result = primary_result_summary(manifest, primary, clone)
    result: dict[str, object] = {
        "schema_version": 2,
        "manifest_sha256": manifest_sha256(manifest),
        "simulator_config_sha256": simulator_config_sha256(SimulatorConfig()),
        "primary": primary_result,
        "mechanism": {"status": "skipped_primary_not_positive"},
    }
    if primary_result["passed"]:
        if not lineage_pairs:
            mechanism: dict[str, object] = {
                "status": "partial_no_lineage_pairs",
                "adaptive_eligible": adaptive_eligible,
                "claim_supported": False,
            }
        else:
            garden_config = replace(SimulatorConfig(), resource_patch_center=SHOCK_CENTER)
            garden = common_garden_analysis(
                lineage_pairs,
                1.0,
                config=garden_config,
            )
            garden_positive = garden["founder_first_interval"]["lower_95"] > 0.0
            mechanism = {
                "status": "completed_common_garden",
                "adaptive_eligible": adaptive_eligible,
                "common_garden": garden,
                "ablations": {},
                "claim_supported": False,
            }
            if adaptive_eligible:
                required = {"dominant_action_ablation", "no_credit_ablation"}
                if not ablations:
                    mechanism["status"] = "partial_ablations_not_estimable"
                else:
                    if set(ablations) != required:
                        raise ValueError("adaptive R6 analysis requires both frozen ablations")
                    primary_mean = primary_result["primary_relocation_auc_delta"]["mean"]
                    ablation_supported = False
                    for name, evaluation in sorted(ablations.items()):
                        summary = _ablation_summary(manifest, primary, evaluation)
                        removes_half = summary["mean"] >= 0.5 * primary_mean
                        positive_interval = summary["lower_95"] > 0.0
                        mechanism["ablations"][name] = {
                            **summary,
                            "removes_at_least_half_primary_gain": removes_half,
                            "positive_finalist_minus_ablation_interval": positive_interval,
                        }
                        ablation_supported |= removes_half or positive_interval
                    mechanism["status"] = "completed"
                    mechanism["claim_supported"] = garden_positive and ablation_supported
        result["mechanism"] = mechanism
    if output_path is not None:
        _write_json_once(Path(output_path), result)
    return result


__all__ = [
    "LineagePair",
    "dominant_action_ablation",
    "extract_relocation_lineage_pairs",
    "no_credit_ablation",
    "primary_result_summary",
    "run_final_analysis",
]
