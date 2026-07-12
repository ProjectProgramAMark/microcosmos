"""Paired, effect-focused analysis for frozen Evo² evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .episode import ManifestEvaluation
from .protocol import (
    EventKind,
    ScenarioManifest,
    final_window_productivity,
    immediate_resistance,
    shock_null_productivity_deficit,
    sustained_recovery,
)


@dataclass(frozen=True)
class PairedWorldMetrics:
    """One shocked-minus-null comparison; the world pair is the unit."""

    pair_id: str
    scenario_family: str
    primary_effect: float
    shock_null_deficit: float
    immediate_resistance: float
    final_productivity_effect: float
    survival_effect: float
    recovered: bool
    recovery_estimable: bool
    recovery_checkpoint: int


@dataclass(frozen=True)
class EffectSummary:
    """Transparent paired-effect summary with a percentile interval."""

    count: int
    mean: float
    median: float
    fraction_positive: float
    confidence_interval: tuple[float, float]


def paired_world_metrics(
    manifest: ScenarioManifest,
    evaluation: ManifestEvaluation,
    *,
    immediate_checkpoints: int = 2,
    final_checkpoints: int = 2,
    recovery_threshold: float = 0.8,
    recovery_consecutive_checkpoints: int = 2,
    recovery_smoothing_window: int = 2,
    recovery_reference_floor: float = 0.05,
) -> tuple[PairedWorldMetrics, ...]:
    """Compare each non-null world to its unique paired null world."""
    if len(manifest.worlds) != len(evaluation.episodes):
        raise ValueError("manifest and evaluation must have equal world counts")
    by_pair: dict[str, list[tuple[object, object]]] = {}
    for world, episode in zip(manifest.worlds, evaluation.episodes, strict=True):
        by_pair.setdefault(world.pair_id, []).append((world, episode))

    metrics = []
    for pair_id, entries in by_pair.items():
        nulls = [entry for entry in entries if entry[0].event_kind is EventKind.NULL]
        shocks = [entry for entry in entries if entry[0].event_kind is not EventKind.NULL]
        if len(nulls) != 1 or len(shocks) != 1 or len(entries) != 2:
            raise ValueError(f"pair {pair_id!r} must contain exactly one null and one shock")
        null_world, null_episode = nulls[0]
        shock_world, shock_episode = shocks[0]
        if null_world.world_seed != shock_world.world_seed:
            raise ValueError(f"pair {pair_id!r} must share one world seed")

        null_productivity = np.asarray(null_episode.post_event_productivity)
        shock_productivity = np.asarray(shock_episode.post_event_productivity)
        recovery = sustained_recovery(
            shock_productivity,
            null_productivity,
            threshold=recovery_threshold,
            consecutive_checkpoints=recovery_consecutive_checkpoints,
            smoothing_window=recovery_smoothing_window,
            reference_floor=recovery_reference_floor,
        )
        metrics.append(
            PairedWorldMetrics(
                pair_id=pair_id,
                scenario_family=shock_world.scenario_family,
                primary_effect=float(shock_episode.primary_score - null_episode.primary_score),
                shock_null_deficit=float(shock_null_productivity_deficit(shock_productivity, null_productivity)),
                immediate_resistance=float(
                    immediate_resistance(
                        shock_productivity,
                        null_productivity,
                        immediate_checkpoints,
                    )
                ),
                final_productivity_effect=float(
                    final_window_productivity(shock_productivity, final_checkpoints) - final_window_productivity(null_productivity, final_checkpoints)
                ),
                survival_effect=float(bool(shock_episode.survived)) - float(bool(null_episode.survived)),
                recovered=bool(recovery.recovered),
                recovery_estimable=bool(recovery.estimable),
                recovery_checkpoint=int(recovery.checkpoint),
            )
        )
    return tuple(metrics)


def paired_effect_summary(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int,
    replicates: int = 10_000,
    confidence: float = 0.95,
) -> EffectSummary:
    """Summarize paired effects by resampling whole world pairs."""
    array = _effect_array(values)
    _validate_bootstrap(replicates, confidence)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = np.mean(array[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    interval = np.quantile(means, [tail, 1.0 - tail])
    return EffectSummary(
        count=int(array.size),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        fraction_positive=float(np.mean(array > 0.0)),
        confidence_interval=(float(interval[0]), float(interval[1])),
    )


def stratified_equal_weight_summary(
    effects_by_family: Mapping[str, Sequence[float] | np.ndarray],
    *,
    seed: int,
    replicates: int = 10_000,
    confidence: float = 0.95,
) -> EffectSummary:
    """Give each scenario family equal weight while resampling within family."""
    if not effects_by_family:
        raise ValueError("effects_by_family must not be empty")
    arrays = tuple(_effect_array(effects_by_family[name]) for name in sorted(effects_by_family))
    _validate_bootstrap(replicates, confidence)
    rng = np.random.default_rng(seed)
    family_bootstraps = []
    for array in arrays:
        indices = rng.integers(0, array.size, size=(replicates, array.size))
        family_bootstraps.append(np.mean(array[indices], axis=1))
    means = np.mean(np.stack(family_bootstraps, axis=1), axis=1)
    observed_family_means = np.asarray([np.mean(array) for array in arrays])
    observed_values = np.concatenate(arrays)
    tail = (1.0 - confidence) / 2.0
    interval = np.quantile(means, [tail, 1.0 - tail])
    return EffectSummary(
        count=int(observed_values.size),
        mean=float(np.mean(observed_family_means)),
        median=float(np.median(observed_values)),
        fraction_positive=float(np.mean(observed_values > 0.0)),
        confidence_interval=(float(interval[0]), float(interval[1])),
    )


def _effect_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("effects must be a non-empty finite vector")
    return array


def _validate_bootstrap(replicates: int, confidence: float) -> None:
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
