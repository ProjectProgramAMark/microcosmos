"""Frozen constants and typed contracts for Evo² r5.

This module contains experiment policy only.  It intentionally does not alter
the Microcosmos simulator, the r4 protocol, or any existing default.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from experiments.evo2_ecosystem.episode import SimulatorConfig
from experiments.evo2_ecosystem.episode import simulator_config_sha256


PROTOCOL_REVISION = "evo2-r5-world-feasibility-v1"
WORLD_QUALIFICATION_SCHEMA_VERSION = 1
WORLD_ACCESS_RULE_ID = "initial-living-mouth-capacity-v1"
WORLD_ACCESS_THRESHOLD = 0.50
WORLD_PASSES_PER_PARTITION = 2
WORLD_ACCESS_SCORE_ATOL = 1e-6

FOUNDER_PANEL_SEED = 5_005
FOUNDER_SCREENING_HORIZON = 4_000
FOUNDER_SCREENING_CHUNK_STEPS = 500
FOUNDER_SELECTION_RULE_ID = "ascending-first-qualified-clone-r5-v1"

TRAINING_EVENT_STEPS = (3_500, 4_000)
DEVELOPMENT_EVENT_STEPS = (3_000, 4_500)
SEALED_EVENT_STEPS = (2_500, 4_500)
ACTUATION_COST_MULTIPLIERS = (1.25, 1.50, 2.00, 3.00)

SHINKA_NONINITIAL_SLOT_BUDGET = 50
FINALIST_TOP_K = 5
STRUCTURED_RANDOM_SEED = 50_005
STRUCTURED_RANDOM_CANDIDATE_COUNT = 50

R5_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = R5_ROOT / "artifacts"
WORLD_QUALIFICATION_PATH = ARTIFACTS_ROOT / "world_qualification.json"


@dataclass(frozen=True)
class WorldSeedRange:
    """One public half-open seed range with a fixed scientific role."""

    partition: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.partition, str) or not self.partition:
            raise ValueError("partition must be a non-empty string")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (self.start, self.stop)
        ):
            raise TypeError("seed-range bounds must be integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("seed range must be nonempty and nonnegative")


WORLD_SEED_RANGES = (
    WorldSeedRange("founder_eligibility", 8_000, 9_000),
    WorldSeedRange("training", 9_000, 10_000),
    WorldSeedRange("development", 10_000, 11_000),
    WorldSeedRange("sealed", 11_000, 12_000),
)


@dataclass(frozen=True)
class FounderCandidateRange:
    """One preassigned TensorNEAT founder-candidate interval."""

    partition: str
    start: int
    stop: int
    required: int

    def __post_init__(self) -> None:
        if self.partition not in {"training", "development", "sealed"}:
            raise ValueError(f"unknown founder partition: {self.partition!r}")
        values = (self.start, self.stop, self.required)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TypeError("founder-candidate range values must be integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("founder-candidate range must be nonempty and nonnegative")
        if not 0 < self.required <= self.stop - self.start:
            raise ValueError("required founders must fit inside their assigned range")


FOUNDER_CANDIDATE_RANGES = (
    FounderCandidateRange("training", 0, 24, 4),
    FounderCandidateRange("development", 24, 48, 4),
    FounderCandidateRange("sealed", 48, 80, 8),
)


@dataclass(frozen=True)
class FounderScreeningGates:
    """Unchanged r4 clone-viability thresholds."""

    minimum_final_alive: int = 2
    minimum_births: int = 2
    minimum_generation: int = 1
    minimum_productivity: float = 0.01

    def __post_init__(self) -> None:
        integers = (
            self.minimum_final_alive,
            self.minimum_births,
            self.minimum_generation,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in integers
        ):
            raise ValueError("integer founder gates must be nonnegative")
        if (
            not isinstance(self.minimum_productivity, (int, float))
            or isinstance(self.minimum_productivity, bool)
            or not math.isfinite(self.minimum_productivity)
            or self.minimum_productivity < 0.0
        ):
            raise ValueError("minimum_productivity must be nonnegative")


@dataclass(frozen=True)
class ScreeningSpec:
    """Experiment-local founder-screening contract for qualified r5 worlds."""

    seeds: tuple[int, int]
    config: SimulatorConfig = SimulatorConfig()
    panel_seed: int = FOUNDER_PANEL_SEED
    horizon: int = FOUNDER_SCREENING_HORIZON
    chunk_steps: int = FOUNDER_SCREENING_CHUNK_STEPS
    candidate_ranges: tuple[FounderCandidateRange, ...] = FOUNDER_CANDIDATE_RANGES
    gates: FounderScreeningGates = FounderScreeningGates()
    selection_rule_id: str = FOUNDER_SELECTION_RULE_ID

    def __post_init__(self) -> None:
        if (
            not isinstance(self.seeds, tuple)
            or len(self.seeds) != 2
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in self.seeds
            )
            or len(set(self.seeds)) != 2
        ):
            raise ValueError("r5 founder screening requires two unique nonnegative seeds")
        if not isinstance(self.config, SimulatorConfig):
            raise TypeError("config must be a SimulatorConfig")
        if self.config != SimulatorConfig():
            raise ValueError("r5 founder screening requires the unchanged simulator config")
        if self.panel_seed != FOUNDER_PANEL_SEED:
            raise ValueError("r5 founder screening requires the frozen panel seed")
        if self.horizon != FOUNDER_SCREENING_HORIZON:
            raise ValueError("r5 founder screening requires the frozen horizon")
        if self.chunk_steps != FOUNDER_SCREENING_CHUNK_STEPS:
            raise ValueError("r5 founder screening requires the frozen chunk size")
        if self.candidate_ranges != FOUNDER_CANDIDATE_RANGES:
            raise ValueError("r5 founder screening requires the frozen candidate ranges")
        if self.gates != FounderScreeningGates():
            raise ValueError("r5 founder screening requires the unchanged r4 gates")
        if self.selection_rule_id != FOUNDER_SELECTION_RULE_ID:
            raise ValueError("r5 founder screening requires the frozen selection rule")

    @property
    def candidate_count(self) -> int:
        return self.candidate_ranges[-1].stop


@dataclass(frozen=True)
class WorldQualificationSpec:
    """Reset-only world-selection contract."""

    seed_ranges: tuple[WorldSeedRange, ...]
    config: SimulatorConfig = SimulatorConfig()
    access_threshold: float = WORLD_ACCESS_THRESHOLD
    passes_per_partition: int = WORLD_PASSES_PER_PARTITION

    def __post_init__(self) -> None:
        if not isinstance(self.seed_ranges, tuple) or not self.seed_ranges:
            raise ValueError("seed_ranges must be a non-empty tuple")
        if any(not isinstance(item, WorldSeedRange) for item in self.seed_ranges):
            raise TypeError("seed_ranges must contain WorldSeedRange values")
        partitions = tuple(item.partition for item in self.seed_ranges)
        if len(partitions) != len(set(partitions)):
            raise ValueError("seed-range partitions must be unique")
        ordered = sorted(self.seed_ranges, key=lambda item: item.start)
        if any(left.stop > right.start for left, right in zip(ordered, ordered[1:])):
            raise ValueError("seed ranges must be disjoint")
        if not isinstance(self.config, SimulatorConfig):
            raise TypeError("config must be a SimulatorConfig")
        if (
            not isinstance(self.access_threshold, (int, float))
            or isinstance(self.access_threshold, bool)
            or not 0.0 <= self.access_threshold <= 1.0
        ):
            raise ValueError("access_threshold must be within [0, 1]")
        if (
            not isinstance(self.passes_per_partition, int)
            or isinstance(self.passes_per_partition, bool)
            or self.passes_per_partition < 1
        ):
            raise ValueError("passes_per_partition must be a positive integer")


PRODUCTION_WORLD_QUALIFICATION_SPEC = WorldQualificationSpec(
    seed_ranges=WORLD_SEED_RANGES,
)
FROZEN_SIMULATOR_CONFIG_SHA256 = simulator_config_sha256(
    PRODUCTION_WORLD_QUALIFICATION_SPEC.config
)


def world_seed_range(partition: str) -> WorldSeedRange:
    """Return one frozen seed range by role."""
    for seed_range in WORLD_SEED_RANGES:
        if seed_range.partition == partition:
            return seed_range
    raise KeyError(f"unknown r5 world partition: {partition!r}")


def founder_screening_spec(founder_eligibility_seeds: tuple[int, int]) -> ScreeningSpec:
    """Bind the frozen founder screen to the qualified eligibility worlds."""
    eligible = world_seed_range("founder_eligibility")
    if any(not eligible.start <= seed < eligible.stop for seed in founder_eligibility_seeds):
        raise ValueError("founder-screening seeds must come from the eligibility range")
    return ScreeningSpec(seeds=founder_eligibility_seeds)


__all__ = [
    "ACTUATION_COST_MULTIPLIERS",
    "ARTIFACTS_ROOT",
    "DEVELOPMENT_EVENT_STEPS",
    "FINALIST_TOP_K",
    "FOUNDER_CANDIDATE_RANGES",
    "FOUNDER_PANEL_SEED",
    "FOUNDER_SCREENING_CHUNK_STEPS",
    "FOUNDER_SCREENING_HORIZON",
    "FROZEN_SIMULATOR_CONFIG_SHA256",
    "FounderCandidateRange",
    "FounderScreeningGates",
    "PRODUCTION_WORLD_QUALIFICATION_SPEC",
    "PROTOCOL_REVISION",
    "R5_ROOT",
    "SEALED_EVENT_STEPS",
    "SHINKA_NONINITIAL_SLOT_BUDGET",
    "STRUCTURED_RANDOM_CANDIDATE_COUNT",
    "STRUCTURED_RANDOM_SEED",
    "ScreeningSpec",
    "TRAINING_EVENT_STEPS",
    "WORLD_ACCESS_RULE_ID",
    "WORLD_ACCESS_SCORE_ATOL",
    "WORLD_ACCESS_THRESHOLD",
    "WORLD_PASSES_PER_PARTITION",
    "WORLD_QUALIFICATION_PATH",
    "WORLD_QUALIFICATION_SCHEMA_VERSION",
    "WORLD_SEED_RANGES",
    "WorldQualificationSpec",
    "WorldSeedRange",
    "founder_screening_spec",
    "world_seed_range",
]
