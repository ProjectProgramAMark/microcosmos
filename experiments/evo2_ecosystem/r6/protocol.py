"""Frozen constants and typed contracts for Evo² R6.

This module owns experiment policy only.  It does not modify Microcosmos
physics, ecology, controllers, heredity operators, or the R1--R5 protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.evo2_ecosystem.episode import SimulatorConfig
from experiments.evo2_ecosystem.episode import simulator_config_sha256
from experiments.evo2_ecosystem.protocol import ResourceRelocationParameters
from experiments.evo2_ecosystem.r5.protocol import FounderCandidateRange
from experiments.evo2_ecosystem.r5.protocol import FounderScreeningGates
from experiments.evo2_ecosystem.r5.protocol import WorldQualificationSpec
from experiments.evo2_ecosystem.r5.protocol import WorldSeedRange


PROTOCOL_REVISION = "evo2-r6-resource-relocation-v1"
RUN_ID = "evo2-r6-resource-relocation-20260714"
PROFILE_NAME = "evo2-r6-resource-relocation"

WORLD_QUALIFICATION_SCHEMA_VERSION = 1
WORLD_ACCESS_RULE_ID = "initial-living-mouth-capacity-v1"
WORLD_ACCESS_THRESHOLD = 0.50
WORLD_PASSES_PER_PARTITION = 2
WORLD_ACCESS_SCORE_ATOL = 1e-6

FOUNDER_PANEL_SEED = 6_006
FOUNDER_SCREENING_HORIZON = 7_500
FOUNDER_SCREENING_CHUNK_STEPS = 500
FOUNDER_SELECTION_RULE_ID = "ascending-first-qualified-clone-r6-v1"

HORIZON = 12_000
CHUNK_STEPS = 500
EVENT_STEP = 7_500
PRE_OPPORTUNITY_START = 6_000
PRE_OPPORTUNITY_STOP = EVENT_STEP
POST_SETTLING_STOP = 8_000
POST_OPPORTUNITY_START = POST_SETTLING_STOP
POST_OPPORTUNITY_STOP = 10_000
ASSESSMENT_STEPS = 1_000

SCENARIO_FAMILY = "r6-resource-relocation"
CONTROL_ROLE = "same_location_refresh_control"
SHOCK_ROLE = "antipodal_relocation_shock"
CONTROL_SUFFIX = "-control"
SHOCK_SUFFIX = "-shock"
CONTROL_CENTER = (48.0, 32.0)
SHOCK_CENTER = (16.0, 32.0)
RESOURCE_RADIUS = 12.0
PEAK_CAPACITY = 1.0
PEAK_REGENERATION = 0.03
STOCK_FRACTION = 1.0

CONTROL_PARAMETERS = ResourceRelocationParameters(
    center=CONTROL_CENTER,
    radius=RESOURCE_RADIUS,
    peak_capacity=PEAK_CAPACITY,
    peak_regeneration=PEAK_REGENERATION,
    stock_fraction=STOCK_FRACTION,
)
SHOCK_PARAMETERS = ResourceRelocationParameters(
    center=SHOCK_CENTER,
    radius=RESOURCE_RADIUS,
    peak_capacity=PEAK_CAPACITY,
    peak_regeneration=PEAK_REGENERATION,
    stock_fraction=STOCK_FRACTION,
)

SHINKA_NONINITIAL_SLOT_BUDGET = 50
FINALIST_TOP_K = 5
STRUCTURED_RANDOM_SEED = 50_005
STRUCTURED_RANDOM_CANDIDATE_COUNT = 50
NUMERICAL_REPEATS = 3
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_712

R6_ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = R6_ROOT / "artifacts"
WORLD_QUALIFICATION_PATH = ARTIFACTS_ROOT / "world_qualification.json"
FOUNDER_ROOT = ARTIFACTS_ROOT / "founders"
FOUNDER_INDEX_PATH = FOUNDER_ROOT / "index.json"
FOUNDER_SCREENING_RECORD_PATH = ARTIFACTS_ROOT / "founder_screening.jsonl"
DISTURBANCE_QUALIFICATION_PATH = ARTIFACTS_ROOT / "disturbance_qualification.json"
OPPORTUNITY_GATE_RECEIPT_PATH = ARTIFACTS_ROOT / "operator_opportunity_gate.json"
PRIVATE_OPPORTUNITY_PATH = ARTIFACTS_ROOT / "private" / "operator_opportunity.json"
MANIFEST_ROOT = ARTIFACTS_ROOT / "manifests"
FINAL_ROOT = ARTIFACTS_ROOT / "final"

MICROCOSMOS_ROOT = Path(__file__).resolve().parents[3]
PRIVATE_STAGING_ROOT = MICROCOSMOS_ROOT.parent / "r6-private" / RUN_ID
PRIVATE_OPPORTUNITY_STAGING_PATH = PRIVATE_STAGING_ROOT / "operator_opportunity.json"
PRD_PATH = MICROCOSMOS_ROOT / "docs/evo2/evo2-r6-resource-relocation-rsi-plan.md"
STOP_REPORT_PATH = MICROCOSMOS_ROOT / "docs/evo2/evo2-r6-resource-relocation-stop-report.md"
FINAL_REPORT_PATH = MICROCOSMOS_ROOT / "docs/evo2/evo2-r6-resource-relocation-final-report.md"


WORLD_SEED_RANGES = (
    WorldSeedRange("founder_eligibility", 12_000, 13_000),
    WorldSeedRange("training", 13_000, 14_000),
    WorldSeedRange("development", 14_000, 15_000),
    WorldSeedRange("sealed", 15_000, 16_000),
)

FOUNDER_CANDIDATE_RANGES = (
    FounderCandidateRange("training", 0, 24, 4),
    FounderCandidateRange("development", 24, 48, 4),
    FounderCandidateRange("sealed", 48, 80, 8),
)


@dataclass(frozen=True)
class ScreeningSpec:
    """Founder-screening contract bound to two qualified eligibility worlds."""

    seeds: tuple[int, int]
    config: SimulatorConfig = SimulatorConfig()
    panel_seed: int = FOUNDER_PANEL_SEED
    horizon: int = FOUNDER_SCREENING_HORIZON
    chunk_steps: int = FOUNDER_SCREENING_CHUNK_STEPS
    candidate_ranges: tuple[FounderCandidateRange, ...] = FOUNDER_CANDIDATE_RANGES
    gates: FounderScreeningGates = FounderScreeningGates()
    selection_rule_id: str = FOUNDER_SELECTION_RULE_ID

    def __post_init__(self) -> None:
        eligible = world_seed_range("founder_eligibility")
        if (
            not isinstance(self.seeds, tuple)
            or len(self.seeds) != 2
            or len(set(self.seeds)) != 2
            or any(not isinstance(seed, int) or isinstance(seed, bool) or not eligible.start <= seed < eligible.stop for seed in self.seeds)
        ):
            raise ValueError("R6 founder screening requires two eligibility seeds")
        if self.config != SimulatorConfig():
            raise ValueError("R6 founder screening requires the frozen simulator config")
        if self.panel_seed != FOUNDER_PANEL_SEED:
            raise ValueError("R6 founder screening requires panel seed 6006")
        if self.horizon != FOUNDER_SCREENING_HORIZON:
            raise ValueError("R6 founder screening requires horizon 7500")
        if self.chunk_steps != FOUNDER_SCREENING_CHUNK_STEPS:
            raise ValueError("R6 founder screening requires chunk size 500")
        if self.candidate_ranges != FOUNDER_CANDIDATE_RANGES:
            raise ValueError("R6 founder screening requires the frozen candidate ranges")
        if self.gates != FounderScreeningGates():
            raise ValueError("R6 founder screening requires the unchanged viability gates")
        if self.selection_rule_id != FOUNDER_SELECTION_RULE_ID:
            raise ValueError("R6 founder screening requires the frozen selection rule")

    @property
    def candidate_count(self) -> int:
        return self.candidate_ranges[-1].stop


PRODUCTION_WORLD_QUALIFICATION_SPEC = WorldQualificationSpec(
    seed_ranges=WORLD_SEED_RANGES,
    config=SimulatorConfig(),
    access_threshold=WORLD_ACCESS_THRESHOLD,
    passes_per_partition=WORLD_PASSES_PER_PARTITION,
)
FROZEN_SIMULATOR_CONFIG_SHA256 = simulator_config_sha256(PRODUCTION_WORLD_QUALIFICATION_SPEC.config)


def world_seed_range(partition: str) -> WorldSeedRange:
    for value in WORLD_SEED_RANGES:
        if value.partition == partition:
            return value
    raise KeyError(f"unknown R6 world partition: {partition!r}")


def founder_screening_spec(seeds: tuple[int, int]) -> ScreeningSpec:
    return ScreeningSpec(seeds=seeds)


def scenario_id(pair_id: str, role: str) -> str:
    if role == CONTROL_ROLE:
        return f"{pair_id}{CONTROL_SUFFIX}"
    if role == SHOCK_ROLE:
        return f"{pair_id}{SHOCK_SUFFIX}"
    raise ValueError(f"unknown R6 scenario role: {role!r}")


__all__ = [
    "ARTIFACTS_ROOT",
    "ASSESSMENT_STEPS",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CHUNK_STEPS",
    "CONTROL_CENTER",
    "CONTROL_PARAMETERS",
    "CONTROL_ROLE",
    "DISTURBANCE_QUALIFICATION_PATH",
    "EVENT_STEP",
    "FINALIST_TOP_K",
    "FINAL_REPORT_PATH",
    "FINAL_ROOT",
    "FOUNDER_CANDIDATE_RANGES",
    "FOUNDER_INDEX_PATH",
    "FOUNDER_PANEL_SEED",
    "FOUNDER_ROOT",
    "FOUNDER_SCREENING_CHUNK_STEPS",
    "FOUNDER_SCREENING_HORIZON",
    "FOUNDER_SCREENING_RECORD_PATH",
    "FROZEN_SIMULATOR_CONFIG_SHA256",
    "HORIZON",
    "MANIFEST_ROOT",
    "MICROCOSMOS_ROOT",
    "NUMERICAL_REPEATS",
    "OPPORTUNITY_GATE_RECEIPT_PATH",
    "PEAK_CAPACITY",
    "PEAK_REGENERATION",
    "POST_OPPORTUNITY_START",
    "POST_OPPORTUNITY_STOP",
    "POST_SETTLING_STOP",
    "PRE_OPPORTUNITY_START",
    "PRE_OPPORTUNITY_STOP",
    "PRIVATE_OPPORTUNITY_PATH",
    "PRIVATE_OPPORTUNITY_STAGING_PATH",
    "PRIVATE_STAGING_ROOT",
    "PRD_PATH",
    "PRODUCTION_WORLD_QUALIFICATION_SPEC",
    "PROTOCOL_REVISION",
    "RESOURCE_RADIUS",
    "RUN_ID",
    "SCENARIO_FAMILY",
    "SHINKA_NONINITIAL_SLOT_BUDGET",
    "SHOCK_CENTER",
    "SHOCK_PARAMETERS",
    "SHOCK_ROLE",
    "STOCK_FRACTION",
    "STOP_REPORT_PATH",
    "STRUCTURED_RANDOM_CANDIDATE_COUNT",
    "STRUCTURED_RANDOM_SEED",
    "ScreeningSpec",
    "WORLD_ACCESS_RULE_ID",
    "WORLD_ACCESS_SCORE_ATOL",
    "WORLD_ACCESS_THRESHOLD",
    "WORLD_PASSES_PER_PARTITION",
    "WORLD_QUALIFICATION_PATH",
    "WORLD_QUALIFICATION_SCHEMA_VERSION",
    "WORLD_SEED_RANGES",
    "founder_screening_spec",
    "scenario_id",
    "world_seed_range",
]
