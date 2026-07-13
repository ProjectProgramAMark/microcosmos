"""Prospective mutation-sensitive actuator benchmark calibration.

This module is intentionally separate from :mod:`calibrate`, which is the
completed pilot's ecological calibration.  It evaluates the four frozen
TensorNEAT heredity baselines on exact clonal-founder injury/sham forks, applies
the preregistered founder-aware gate, and writes an append-only calibration
ledger plus one canonical selected manifest.

Production execution is GPU/fluid/full-scale only.  ``--smoke`` exercises the
same interfaces on a tiny CPU substrate but is never accepted as scientific
calibration evidence.  Running this file does not generate or select founders;
it consumes the separately frozen founder bank.
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
import jax.numpy as jnp
import numpy as np

from microcosmos.cppn import CPPNGenome
from microcosmos.heredity import (
    CLONE,
    MIXED,
    PARAMETRIC,
    STRUCTURAL,
    OffspringPolicy,
    clone_policy,
    fixed_mixed_policy,
    fixed_parametric_policy,
    fixed_structural_policy,
)
from microcosmos.rollout import combine_chunk_metrics, run_ecosystem_chunk

from .analysis import (
    DEFAULT_HIERARCHICAL_BOOTSTRAP_SEED,
    FounderWorldObservation,
    HierarchicalEffectSummary,
    hierarchical_paired_effect_summary,
)
from .episode import (
    SimulatorConfig,
    _apply_event,
    _prepare_pre_event,
    _run_chunks,
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
from .frozen_config import NUMERICAL_REPEATS
from .protocol import (
    ActuatorInjuryParameters,
    CONTROLLER_LAYOUT,
    EventKind,
    NullEventParameters,
    SCHEMA_VERSION,
    ScenarioManifest,
    WorldScenario,
    canonical_manifest_bytes,
    manifest_sha256,
    normalized_chunk_productivity,
    post_event_productivity_auc,
)
from .readiness_seed_panel import (
    ReadinessSeedPanel,
    load_readiness_seed_panel,
)


CALIBRATION_RECORD_SCHEMA_VERSION = 1
CALIBRATION_RESULT_SCHEMA_VERSION = 1
PRIMARY_SCHEDULE_ID = "mapped-3500-4000-4500-v1"
FALLBACK_SCHEDULE_ID = "all-4500-v1"
SELECTION_RULE_ID = "smallest-passing-gain-v1"

PRODUCTION_HORIZON = 8_000
PRODUCTION_CHUNK_STEPS = 500
PRODUCTION_WORLD_SEEDS = (4_101, 4_211, 4_307)
PRODUCTION_EVENT_STEP_BY_SEED = {
    4_101: 3_500,
    4_211: 4_000,
    4_307: 4_500,
}
CALIBRATION_GAINS = (0.1, 0.2, 0.4)
TRAINING_INJURED_HINGES = (0, 1)

POLICIES: Mapping[str, tuple[OffspringPolicy, int]] = {
    "clone": (clone_policy, CLONE),
    "fixed_parametric": (fixed_parametric_policy, PARAMETRIC),
    "fixed_structural": (fixed_structural_policy, STRUCTURAL),
    "fixed_mixed": (fixed_mixed_policy, MIXED),
}
POLICY_ORDER = tuple(POLICIES)
MUTATION_POLICIES = POLICY_ORDER[1:]


@dataclass(frozen=True)
class CalibrationGates:
    """Frozen causal and viability requirements for one injury gain."""

    minimum_survival_fraction: float = 2.0 / 3.0
    minimum_median_post_event_births: int = 4
    minimum_median_generation_gain: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_survival_fraction, bool)
            or not isinstance(self.minimum_survival_fraction, (int, float))
            or not math.isfinite(self.minimum_survival_fraction)
            or not 0.0 < self.minimum_survival_fraction <= 1.0
        ):
            raise ValueError("minimum_survival_fraction must be within (0, 1]")
        for name in (
            "minimum_median_post_event_births",
            "minimum_median_generation_gain",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class ActuatorCalibrationSpec:
    """Complete static calibration execution contract."""

    mode: str
    config: SimulatorConfig
    horizon: int
    chunk_steps: int
    world_seeds: tuple[int, ...]
    event_step_by_seed: tuple[tuple[int, int], ...]
    gains: tuple[float, ...]
    injured_hinges: tuple[int, ...]
    required_training_founders: int
    numerical_repeats: int
    bootstrap_replicates: int
    bootstrap_seed: int
    gates: CalibrationGates

    def __post_init__(self) -> None:
        if self.mode not in {"production", "smoke"}:
            raise ValueError("mode must be 'production' or 'smoke'")
        if not isinstance(self.config, SimulatorConfig):
            raise TypeError("config must be a SimulatorConfig")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(self.chunk_steps, int) or isinstance(self.chunk_steps, bool) or self.chunk_steps <= 0 or self.horizon % self.chunk_steps:
            raise ValueError("chunk_steps must positively divide horizon")
        if (
            not isinstance(self.world_seeds, tuple)
            or not self.world_seeds
            or len(set(self.world_seeds)) != len(self.world_seeds)
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.world_seeds)
        ):
            raise ValueError("world_seeds must be unique nonnegative integers")
        mapping = dict(self.event_step_by_seed)
        if len(mapping) != len(self.event_step_by_seed) or set(mapping) != set(self.world_seeds):
            raise ValueError("event-step mapping must name every world seed once")
        if any(not isinstance(step, int) or isinstance(step, bool) or not 0 < step < self.horizon or step % self.chunk_steps for step in mapping.values()):
            raise ValueError("event steps must be interior chunk boundaries")
        if (
            not isinstance(self.gains, tuple)
            or not self.gains
            or tuple(sorted(set(self.gains))) != self.gains
            or any(isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or not 0.0 <= gain < 1.0 for gain in self.gains)
        ):
            raise ValueError("gains must be sorted unique finite values in [0, 1)")
        hinge_count = max(0, self.config.nodes_per_creature - 2)
        if (
            not isinstance(self.injured_hinges, tuple)
            or not self.injured_hinges
            or len(set(self.injured_hinges)) != len(self.injured_hinges)
            or any(not isinstance(hinge, int) or isinstance(hinge, bool) or not 0 <= hinge < hinge_count for hinge in self.injured_hinges)
        ):
            raise ValueError("injured_hinges must be distinct valid local hinges")
        if not isinstance(self.required_training_founders, int) or isinstance(self.required_training_founders, bool) or self.required_training_founders < 1:
            raise ValueError("required_training_founders must be positive")
        if (
            not isinstance(self.numerical_repeats, int)
            or isinstance(self.numerical_repeats, bool)
            or self.numerical_repeats < 1
            or self.numerical_repeats % 2 == 0
        ):
            raise ValueError("numerical_repeats must be positive and odd")
        if not isinstance(self.bootstrap_replicates, int) or isinstance(self.bootstrap_replicates, bool) or self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if not isinstance(self.bootstrap_seed, int) or isinstance(self.bootstrap_seed, bool) or self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be nonnegative")
        if not isinstance(self.gates, CalibrationGates):
            raise TypeError("gates must be CalibrationGates")

    @property
    def event_steps(self) -> dict[int, int]:
        return dict(self.event_step_by_seed)


PRODUCTION_SPEC = ActuatorCalibrationSpec(
    mode="production",
    config=SimulatorConfig(),
    horizon=PRODUCTION_HORIZON,
    chunk_steps=PRODUCTION_CHUNK_STEPS,
    world_seeds=PRODUCTION_WORLD_SEEDS,
    event_step_by_seed=tuple(PRODUCTION_EVENT_STEP_BY_SEED.items()),
    gains=CALIBRATION_GAINS,
    injured_hinges=TRAINING_INJURED_HINGES,
    required_training_founders=3,
    numerical_repeats=NUMERICAL_REPEATS,
    bootstrap_replicates=10_000,
    bootstrap_seed=DEFAULT_HIERARCHICAL_BOOTSTRAP_SEED,
    gates=CalibrationGates(),
)


def production_spec_for_seed_panel(
    seed_panel: ReadinessSeedPanel,
) -> ActuatorCalibrationSpec:
    """Bind the frozen calibration design to readiness-qualified training worlds."""
    if not isinstance(seed_panel, ReadinessSeedPanel):
        raise TypeError("seed_panel must be a ReadinessSeedPanel")
    event_steps = (3_500, 4_000, 4_500)
    return replace(
        PRODUCTION_SPEC,
        world_seeds=seed_panel.training,
        event_step_by_seed=tuple(zip(seed_panel.training, event_steps, strict=True)),
    )

SMOKE_SPEC = ActuatorCalibrationSpec(
    mode="smoke",
    config=replace(
        SimulatorConfig(),
        max_creatures=4,
        initial_population=2,
        nodes_per_creature=4,
        grid_shape=(16, 16),
        fluid_enabled=False,
        resource_patch_center=(8.0, 8.0),
        resource_patch_radius=6.0,
        placement_candidates=4,
        position_margin=0.5,
    ),
    horizon=8,
    chunk_steps=2,
    world_seeds=(0, 1, 2),
    event_step_by_seed=((0, 2), (1, 4), (2, 6)),
    gains=CALIBRATION_GAINS,
    injured_hinges=TRAINING_INJURED_HINGES,
    required_training_founders=3,
    numerical_repeats=1,
    bootstrap_replicates=100,
    bootstrap_seed=DEFAULT_HIERARCHICAL_BOOTSTRAP_SEED,
    gates=CalibrationGates(),
)


@dataclass(frozen=True)
class CalibrationWorldResult:
    """One arm of one exact founder×seed state fork."""

    policy_name: str
    founder_id: str
    founder_sha256: str
    world_seed: int
    pair_id: str
    event_step: int
    arm: str
    primary_score: float
    survived: bool
    post_event_births: int
    event_generation: int
    maximum_generation: int
    generation_gain: int
    operator_counts: tuple[int, ...]
    integrity_valid: bool
    clone_identity_valid: bool | None


@dataclass(frozen=True)
class PolicyCalibrationResult:
    """One coherent numerical measurement plus the complete repeat audit."""

    policy_name: str
    worlds: tuple[CalibrationWorldResult, ...]
    repeat_scores: tuple[float, ...]
    selected_repeat_index: int
    all_repeats_integrity_valid: bool
    clone_identity_all_repeats: bool | None

    @property
    def injury_worlds(self) -> tuple[CalibrationWorldResult, ...]:
        return tuple(world for world in self.worlds if world.arm == "injury")

    @property
    def sham_worlds(self) -> tuple[CalibrationWorldResult, ...]:
        return tuple(world for world in self.worlds if world.arm == "sham")


@dataclass(frozen=True)
class GainAttemptResult:
    """All four trusted policies for one exact gain and schedule."""

    gain: float
    schedule_id: str
    manifest_sha256: str
    policies: tuple[PolicyCalibrationResult, ...]

    def policy(self, name: str) -> PolicyCalibrationResult:
        matches = [policy for policy in self.policies if policy.policy_name == name]
        if len(matches) != 1:
            raise ValueError(f"attempt must contain exactly one {name!r} policy")
        return matches[0]


@dataclass(frozen=True)
class BenchmarkGateResult:
    """Founder-aware decision for one gain attempt."""

    passed: bool
    clone_harm: HierarchicalEffectSummary
    mutation_advantages: tuple[tuple[str, HierarchicalEffectSummary], ...]
    qualifying_mutation_policies: tuple[str, ...]
    winning_policy: str | None
    all_integrity_valid: bool
    clone_identity_valid: bool
    policy_viability: tuple[tuple[str, dict[str, float | bool]], ...]


AttemptRunner = Callable[
    [ScenarioManifest, float, str],
    GainAttemptResult,
]


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


def validate_execution_contract(
    spec: ActuatorCalibrationSpec,
    *,
    backend: str,
) -> None:
    """Fail closed across the smoke/production evidence boundary."""
    if not isinstance(spec, ActuatorCalibrationSpec):
        raise TypeError("spec must be an ActuatorCalibrationSpec")
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    if spec.mode == "production":
        expected_seed_agnostic = replace(
            PRODUCTION_SPEC,
            world_seeds=spec.world_seeds,
            event_step_by_seed=spec.event_step_by_seed,
        )
        if spec != expected_seed_agnostic or tuple(spec.event_steps.values()) != (
            3_500,
            4_000,
            4_500,
        ):
            raise RuntimeError("production calibration requires the frozen spec")
        if backend != "gpu":
            raise RuntimeError("production calibration requires the JAX GPU backend")
        if not spec.config.fluid_enabled:
            raise RuntimeError("production calibration requires fluid physics")
        expected = (
            32,
            8,
            (64, 64),
            8_000,
            500,
        )
        actual = (
            spec.config.max_creatures,
            spec.config.nodes_per_creature,
            spec.config.grid_shape,
            spec.horizon,
            spec.chunk_steps,
        )
        if actual != expected:
            raise RuntimeError(f"production calibration requires full scale {expected}, got {actual}")
        return
    if spec != SMOKE_SPEC:
        raise RuntimeError("smoke calibration requires the frozen tiny spec")
    if backend != "cpu":
        raise RuntimeError("smoke calibration requires the JAX CPU backend")
    if spec.config.fluid_enabled:
        raise RuntimeError("smoke calibration must disable fluid physics")


def training_founder_records(
    index: FounderIndex,
    *,
    required: int,
) -> tuple[FounderRecord, ...]:
    """Resolve only the prepartitioned training founders in canonical order."""
    if not isinstance(index, FounderIndex):
        raise TypeError("index must be a FounderIndex")
    records = tuple(
        sorted(
            (record for record in index.founders if record.partition == "training"),
            key=lambda record: record.founder_id,
        )
    )
    if len(records) != required:
        raise ValueError(f"calibration requires exactly {required} training founders, found {len(records)}")
    return records


def _injury_vector(spec: ActuatorCalibrationSpec, gain: float) -> tuple[float, ...]:
    if gain not in spec.gains:
        raise ValueError("gain is not in the frozen calibration grid")
    count = spec.config.nodes_per_creature - 2
    values = [1.0] * count
    for hinge in spec.injured_hinges:
        values[hinge] = float(gain)
    return tuple(values)


def build_calibration_manifest(
    spec: ActuatorCalibrationSpec,
    founders: Sequence[FounderRecord],
    *,
    gain: float,
    schedule_id: str,
) -> ScenarioManifest:
    """Build one strict schema-v2 training-only injury/sham panel."""
    if len(founders) != spec.required_training_founders or any(founder.partition != "training" for founder in founders):
        raise ValueError("manifest builders accept only the frozen training panel")
    if schedule_id not in {PRIMARY_SCHEDULE_ID, FALLBACK_SCHEDULE_ID}:
        raise ValueError("unknown calibration schedule")
    injury = _injury_vector(spec, gain)
    worlds: list[WorldScenario] = []
    for founder in founders:
        for seed in spec.world_seeds:
            event_step = spec.event_steps[seed] if schedule_id == PRIMARY_SCHEDULE_ID else 4_500
            if spec.mode == "smoke" and schedule_id == FALLBACK_SCHEDULE_ID:
                # The scientific fallback is exactly step 4500.  Smoke cannot
                # represent it and therefore never invokes fallback.
                raise ValueError("the production-only fallback has no smoke analogue")
            pair_id = f"cal-{founder.founder_id}-seed-{seed}"
            common = {
                "world_seed": seed,
                "pair_id": pair_id,
                "scenario_family": "actuator_injury",
                "event_step": event_step,
                "founder_id": founder.founder_id,
                "founder_sha256": founder.artifact_sha256,
            }
            worlds.extend(
                (
                    WorldScenario(
                        scenario_id=f"{pair_id}-sham",
                        event_kind=EventKind.NULL,
                        event_parameters=NullEventParameters(),
                        **common,
                    ),
                    WorldScenario(
                        scenario_id=f"{pair_id}-injury-g{int(round(gain * 10)):02d}",
                        event_kind=EventKind.ACTUATOR_INJURY,
                        event_parameters=ActuatorInjuryParameters(injury),
                        **common,
                    ),
                )
            )
    return ScenarioManifest(
        schema_version=SCHEMA_VERSION,
        partition="calibration",
        controller_layout=CONTROLLER_LAYOUT,
        simulator_config_sha256=simulator_config_sha256(spec.config),
        horizon=spec.horizon,
        chunk_steps=spec.chunk_steps,
        worlds=tuple(worlds),
    )


def _bitwise_clonal_population(
    population_genome: CPPNGenome,
    expected: CPPNGenome,
) -> bool:
    """Include legal NaN padding in an exact float32 bit-pattern check."""
    for actual, reference in (
        (population_genome.node_genes, expected.node_genes),
        (population_genome.connection_genes, expected.connection_genes),
    ):
        actual_array = np.ascontiguousarray(jax.device_get(actual))
        expected_array = np.ascontiguousarray(jax.device_get(reference))
        broadcast = np.ascontiguousarray(np.broadcast_to(expected_array, actual_array.shape))
        if not np.array_equal(
            actual_array.view(np.uint32),
            broadcast.view(np.uint32),
        ):
            return False
    return True


class CalibrationEvaluator:
    """Cache one compiled rollout per trusted policy across all gain attempts."""

    def __init__(
        self,
        spec: ActuatorCalibrationSpec,
        founder_index_path: Path,
    ):
        self.spec = spec
        self.founder_index_path = Path(founder_index_path)
        # Partition metadata are authenticated here; only the training artifacts
        # below may be opened during calibration.
        index = load_founder_index(self.founder_index_path, verify_artifacts=False)
        self.records = training_founder_records(
            index,
            required=spec.required_training_founders,
        )
        self.founders = {
            record.founder_id: load_founder_artifact(
                self.founder_index_path.parent,
                record,
                expected_partition="training",
            )
            for record in self.records
        }
        self._runtimes = {}
        for policy_name, (policy, expected_operator) in POLICIES.items():
            env = build_environment(spec.config, policy, horizon=spec.horizon)
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
        gain: float,
        schedule_id: str,
    ) -> GainAttemptResult:
        if manifest_sha256(manifest) != manifest_sha256(
            build_calibration_manifest(
                self.spec,
                self.records,
                gain=gain,
                schedule_id=schedule_id,
            )
        ):
            raise ValueError("attempt manifest does not match the frozen builder")
        policies = tuple(self._evaluate_policy(manifest, policy_name) for policy_name in POLICY_ORDER)
        return GainAttemptResult(
            gain=gain,
            schedule_id=schedule_id,
            manifest_sha256=manifest_sha256(manifest),
            policies=policies,
        )

    def _evaluate_policy(
        self,
        manifest: ScenarioManifest,
        policy_name: str,
    ) -> PolicyCalibrationResult:
        env, compiled, expected_operator = self._runtimes[policy_name]
        repeats: list[tuple[CalibrationWorldResult, ...]] = []
        repeat_scores = []
        for _ in range(self.spec.numerical_repeats):
            worlds = self._evaluate_once(
                manifest,
                policy_name,
                env,
                compiled,
                expected_operator,
            )
            repeats.append(worlds)
            injury_scores = [world.primary_score for world in worlds if world.arm == "injury"]
            repeat_scores.append(float(np.mean(injury_scores)))
        selected = int(np.argsort(np.asarray(repeat_scores), kind="stable")[self.spec.numerical_repeats // 2])
        all_integrity = all(world.integrity_valid for repeat_worlds in repeats for world in repeat_worlds)
        clone_identity = None
        if policy_name == "clone":
            clone_identity = all(world.clone_identity_valid is True for repeat_worlds in repeats for world in repeat_worlds)
        return PolicyCalibrationResult(
            policy_name=policy_name,
            worlds=repeats[selected],
            repeat_scores=tuple(repeat_scores),
            selected_repeat_index=selected,
            all_repeats_integrity_valid=all_integrity,
            clone_identity_all_repeats=clone_identity,
        )

    def _evaluate_once(
        self,
        manifest: ScenarioManifest,
        policy_name: str,
        env: Any,
        compiled: Callable,
        expected_operator: int,
    ) -> tuple[CalibrationWorldResult, ...]:
        grouped: dict[str, list[WorldScenario]] = {}
        for world in manifest.worlds:
            grouped.setdefault(world.pair_id, []).append(world)
        results = []
        for pair_id in sorted(grouped):
            pair = grouped[pair_id]
            if len(pair) != 2:
                raise ValueError("every calibration pair must contain sham and injury")
            reference = pair[0]
            if reference.founder_id is None:
                raise ValueError("schema-v2 calibration requires founder metadata")
            founder = self.founders[reference.founder_id]
            root_key, pre_state, pre_metrics = _prepare_pre_event(
                env,
                compiled,
                reference,
                chunk_steps=manifest.chunk_steps,
                founder_genome=founder,
            )
            live_generation = jnp.where(
                pre_state.population.alive,
                pre_state.population.generation,
                0,
            )
            event_generation = int(np.asarray(jnp.max(live_generation)))
            expected_founder = CPPNGenome(
                pre_state.population.genome.node_genes[0],
                pre_state.population.genome.connection_genes[0],
            )
            for world in pair:
                results.append(
                    self._finish_arm(
                        env,
                        compiled,
                        world,
                        root_key,
                        pre_state,
                        pre_metrics,
                        policy_name=policy_name,
                        expected_operator=expected_operator,
                        expected_founder=expected_founder,
                        event_generation=event_generation,
                    )
                )
        return tuple(results)

    def _finish_arm(
        self,
        env: Any,
        compiled: Callable,
        world: WorldScenario,
        root_key: jax.Array,
        pre_state: Any,
        pre_metrics: Any,
        *,
        policy_name: str,
        expected_operator: int,
        expected_founder: CPPNGenome,
        event_generation: int,
    ) -> CalibrationWorldResult:
        state, _event_record = _apply_event(env, pre_state, world, root_key)
        post_event_regeneration_map = state.resource_regeneration_map
        state, post_metrics, rewards = _run_chunks(
            compiled,
            state,
            root_key,
            first_step=world.event_step,
            stop_step=self.spec.horizon,
            chunk_steps=self.spec.chunk_steps,
            collect_rewards=True,
        )
        combined = combine_chunk_metrics(pre_metrics, post_metrics)
        productivity = jnp.stack(
            [
                normalized_chunk_productivity(
                    reward,
                    self.spec.chunk_steps,
                    env.dt,
                    post_event_regeneration_map,
                )
                for reward in rewards
            ]
        )
        operator_counts = np.asarray(combined.operator_counts, dtype=np.int64)
        exact_operator_accounting = bool(
            operator_counts[expected_operator] == int(np.asarray(combined.birth_count)) and np.sum(operator_counts) == int(np.asarray(combined.birth_count))
        )
        integrity = (
            bool(
                np.asarray(
                    combined.finite & combined.identity_valid & combined.events_valid & combined.infrastructure_valid & (combined.policy_violation_count == 0)
                )
            )
            and exact_operator_accounting
        )
        clone_identity = None
        if policy_name == "clone":
            clone_identity = _bitwise_clonal_population(
                state.population.genome,
                expected_founder,
            )
        maximum_generation = int(np.asarray(post_metrics.maximum_generation))
        if world.founder_id is None or world.founder_sha256 is None:
            raise ValueError("calibration world lost founder identity")
        return CalibrationWorldResult(
            policy_name=policy_name,
            founder_id=world.founder_id,
            founder_sha256=world.founder_sha256,
            world_seed=world.world_seed,
            pair_id=world.pair_id,
            event_step=world.event_step,
            arm=("injury" if world.event_kind is EventKind.ACTUATOR_INJURY else "sham"),
            primary_score=float(np.asarray(post_event_productivity_auc(productivity))),
            survived=bool(np.asarray(jnp.any(state.population.alive))),
            post_event_births=int(np.asarray(post_metrics.birth_count)),
            event_generation=event_generation,
            maximum_generation=maximum_generation,
            generation_gain=max(0, maximum_generation - event_generation),
            operator_counts=tuple(int(value) for value in operator_counts),
            integrity_valid=integrity,
            clone_identity_valid=clone_identity,
        )


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


def evaluate_benchmark_gate(
    attempt: GainAttemptResult,
    spec: ActuatorCalibrationSpec,
) -> BenchmarkGateResult:
    """Apply every preregistered gate without optimizing a scalar score."""
    if attempt.gain not in spec.gains:
        raise ValueError("attempt gain is outside the frozen grid")
    if tuple(policy.policy_name for policy in attempt.policies) != POLICY_ORDER:
        raise ValueError("attempt must contain the four policies in frozen order")
    clone = attempt.policy("clone")
    clone_harm = hierarchical_paired_effect_summary(
        _observations(clone.injury_worlds),
        _observations(clone.sham_worlds),
        seed=spec.bootstrap_seed,
        replicates=spec.bootstrap_replicates,
    )
    all_integrity = all(policy.all_repeats_integrity_valid and all(world.integrity_valid for world in policy.worlds) for policy in attempt.policies)
    clone_identity = clone.clone_identity_all_repeats is True

    mutation_advantages = []
    viability = []
    qualifiers = []
    for policy_name in MUTATION_POLICIES:
        policy = attempt.policy(policy_name)
        advantage = hierarchical_paired_effect_summary(
            _observations(policy.injury_worlds),
            _observations(clone.injury_worlds),
            seed=spec.bootstrap_seed,
            replicates=spec.bootstrap_replicates,
        )
        mutation_advantages.append((policy_name, advantage))
        injury_worlds = policy.injury_worlds
        survival_fraction = float(np.mean([world.survived for world in injury_worlds]))
        median_births = float(np.median([world.post_event_births for world in injury_worlds]))
        median_generation_gain = float(np.median([world.generation_gain for world in injury_worlds]))
        policy_viable = (
            survival_fraction >= spec.gates.minimum_survival_fraction
            and median_births >= spec.gates.minimum_median_post_event_births
            and median_generation_gain >= spec.gates.minimum_median_generation_gain
            and policy.all_repeats_integrity_valid
        )
        viability.append(
            (
                policy_name,
                {
                    "passed": policy_viable,
                    "survival_fraction": survival_fraction,
                    "median_post_event_births": median_births,
                    "median_generation_gain": median_generation_gain,
                },
            )
        )
        if advantage.confidence_interval[0] > 0.0 and policy_viable:
            qualifiers.append(policy_name)

    advantage_by_policy = dict(mutation_advantages)
    winner = (
        max(
            qualifiers,
            key=lambda name: (
                advantage_by_policy[name].mean,
                -MUTATION_POLICIES.index(name),
            ),
        )
        if qualifiers
        else None
    )
    harmful_clone_effect = clone_harm.confidence_interval[1] < 0.0
    passed = bool(clone_identity and all_integrity and harmful_clone_effect and winner is not None)
    return BenchmarkGateResult(
        passed=passed,
        clone_harm=clone_harm,
        mutation_advantages=tuple(mutation_advantages),
        qualifying_mutation_policies=tuple(qualifiers),
        winning_policy=winner,
        all_integrity_valid=all_integrity,
        clone_identity_valid=clone_identity,
        policy_viability=tuple(viability),
    )


def fallback_eligible(
    attempts: Sequence[GainAttemptResult],
    spec: ActuatorCalibrationSpec,
) -> bool:
    """Permit the one later-event fallback only for universal birth scarcity."""
    if not attempts:
        return False
    return all(
        np.median([world.post_event_births for world in attempt.policy(policy_name).injury_worlds]) < spec.gates.minimum_median_post_event_births
        for attempt in attempts
        for policy_name in MUTATION_POLICIES
    )


def _hierarchical_summary_dict(summary: HierarchicalEffectSummary) -> dict[str, object]:
    return {
        "confidence_interval": list(summary.confidence_interval),
        "count": summary.count,
        "founder_count": summary.founder_count,
        "fraction_positive": summary.fraction_positive,
        "mean": summary.mean,
        "median": summary.median,
        "per_founder_effects": [asdict(effect) for effect in summary.per_founder_effects],
    }


def _gate_dict(gate: BenchmarkGateResult) -> dict[str, object]:
    return {
        "all_integrity_valid": gate.all_integrity_valid,
        "clone_harm": _hierarchical_summary_dict(gate.clone_harm),
        "clone_identity_valid": gate.clone_identity_valid,
        "mutation_advantages": {name: _hierarchical_summary_dict(summary) for name, summary in gate.mutation_advantages},
        "passed": gate.passed,
        "policy_viability": dict(gate.policy_viability),
        "qualifying_mutation_policies": list(gate.qualifying_mutation_policies),
        "winning_policy": gate.winning_policy,
    }


def _attempt_dict(attempt: GainAttemptResult) -> dict[str, object]:
    return {
        "gain": attempt.gain,
        "manifest_sha256": attempt.manifest_sha256,
        "policies": [
            {
                "all_repeats_integrity_valid": policy.all_repeats_integrity_valid,
                "clone_identity_all_repeats": policy.clone_identity_all_repeats,
                "policy_name": policy.policy_name,
                "repeat_scores": list(policy.repeat_scores),
                "selected_repeat_index": policy.selected_repeat_index,
                "worlds": [asdict(world) for world in policy.worlds],
            }
            for policy in attempt.policies
        ],
        "schedule_id": attempt.schedule_id,
    }


class _AppendOnlyCalibrationRecord:
    """Exclusive-create canonical JSONL ledger with durable appends."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
                0o644,
            )
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace append-only calibration record: {path}") from error
        self._sequence = 0

    def append(self, event: dict[str, object]) -> None:
        payload = (
            _canonical_json_bytes(
                {
                    "schema_version": CALIBRATION_RECORD_SCHEMA_VERSION,
                    "sequence": self._sequence,
                    **event,
                }
            )
            + b"\n"
        )
        written = os.write(self._descriptor, payload)
        if written != len(payload):
            raise OSError("short write to calibration record")
        os.fsync(self._descriptor)
        self._sequence += 1

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> _AppendOnlyCalibrationRecord:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _write_manifest_once(path: Path, manifest: ScenarioManifest) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_manifest_bytes(manifest) + b"\n"
    digest = manifest_sha256(manifest)
    sidecar = path.with_suffix(".sha256")
    if path.exists() or sidecar.exists():
        existing = path if path.exists() else sidecar
        raise FileExistsError(f"refusing to replace selected calibration manifest artifact: {existing}")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace selected calibration manifest: {path}") from error
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError("short write to selected calibration manifest")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    sidecar_payload = f"{digest}  {path.name}\n".encode("ascii")
    try:
        sidecar_descriptor = os.open(
            sidecar,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        try:
            written = os.write(sidecar_descriptor, sidecar_payload)
            if written != len(sidecar_payload):
                raise OSError("short write to selected manifest hash sidecar")
            os.fsync(sidecar_descriptor)
        finally:
            os.close(sidecar_descriptor)
    except BaseException:
        # Neither artifact is considered published until both exact files exist.
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return digest


def _protocol_record(
    spec: ActuatorCalibrationSpec,
    index: FounderIndex,
    *,
    founder_index_path: Path,
    preregistration_path: Path,
    backend: str,
    seed_panel: ReadinessSeedPanel | None,
    seed_panel_path: Path | None,
) -> dict[str, object]:
    protocol: dict[str, object] = {
        "backend": backend,
        "bootstrap": {
            "replicates": spec.bootstrap_replicates,
            "seed": spec.bootstrap_seed,
            "unit": "founder-first-then-paired-world-within-founder",
        },
        "calibration_source_sha256": _sha256(Path(__file__).read_bytes()),
        "founder_index_path": str(founder_index_path),
        "founder_index_sha256": founder_index_sha256(index),
        "gates": asdict(spec.gates),
        "mode": spec.mode,
        "policies": {
            name: {
                "expected_operator": expected_operator,
                "source_sha256": _source_sha256(policy),
            }
            for name, (policy, expected_operator) in POLICIES.items()
        },
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": _sha256(preregistration_path.read_bytes()),
        "selection_rule_id": SELECTION_RULE_ID,
        "simulator_config": asdict(spec.config),
        "simulator_config_sha256": simulator_config_sha256(spec.config),
        "simulator_source_sha256": simulator_source_sha256(),
        "spec": {
            **asdict(spec),
            "config": asdict(spec.config),
        },
    }
    if seed_panel is not None:
        if seed_panel_path is None:
            raise RuntimeError("authenticated seed panel is missing its path")
        protocol["readiness_seed_panel_path"] = str(seed_panel_path)
        protocol["readiness_seed_panel_sha256"] = seed_panel.artifact_sha256
        protocol["readiness_protocol_sha256"] = seed_panel.protocol_sha256
    return protocol


def _validate_attempt(
    attempt: GainAttemptResult,
    manifest: ScenarioManifest,
    gain: float,
    schedule_id: str,
) -> None:
    if not isinstance(attempt, GainAttemptResult):
        raise TypeError("attempt runner must return GainAttemptResult")
    if attempt.gain != gain or attempt.schedule_id != schedule_id:
        raise ValueError("attempt runner changed gain or schedule identity")
    if attempt.manifest_sha256 != manifest_sha256(manifest):
        raise ValueError("attempt runner changed manifest identity")


def run_actuator_calibration(
    *,
    smoke: bool,
    founder_index_path: Path,
    preregistration_path: Path,
    calibration_record_path: Path,
    selected_manifest_path: Path,
    attempt_runner: AttemptRunner | None = None,
    backend: str | None = None,
    readiness_seed_panel_path: Path | None = None,
) -> dict[str, object]:
    """Run the frozen grid once and publish no mutable scientific state."""
    seed_panel = (
        None
        if readiness_seed_panel_path is None
        else load_readiness_seed_panel(readiness_seed_panel_path)
    )
    if smoke and seed_panel is not None:
        raise RuntimeError("smoke calibration cannot consume scientific seed panels")
    spec = (
        SMOKE_SPEC
        if smoke
        else (
            PRODUCTION_SPEC
            if seed_panel is None
            else production_spec_for_seed_panel(seed_panel)
        )
    )
    actual_backend = jax.default_backend()
    if not smoke and (attempt_runner is not None or backend is not None):
        raise RuntimeError("production actuator calibration forbids injected runners and backend overrides")
    if backend is not None and backend != actual_backend:
        raise RuntimeError("declared smoke backend does not match jax.default_backend()")
    validate_execution_contract(spec, backend=actual_backend)
    if selected_manifest_path.exists() or selected_manifest_path.with_suffix(".sha256").exists():
        raise FileExistsError(f"refusing to replace selected calibration manifest or sidecar: {selected_manifest_path}")
    preregistration_payload = preregistration_path.read_text(encoding="utf-8")
    if not smoke and "Status: **FROZEN" not in preregistration_payload:
        raise RuntimeError("production calibration requires a frozen preregistration")

    # Calibration must not open development or sealed founder bytes.
    index = load_founder_index(founder_index_path, verify_artifacts=False)
    founders = training_founder_records(
        index,
        required=spec.required_training_founders,
    )
    runner = CalibrationEvaluator(spec, founder_index_path) if attempt_runner is None else attempt_runner
    protocol = _protocol_record(
        spec,
        index,
        founder_index_path=founder_index_path,
        preregistration_path=preregistration_path,
        backend=actual_backend,
        seed_panel=seed_panel,
        seed_panel_path=readiness_seed_panel_path,
    )
    protocol_sha256 = _sha256(_canonical_json_bytes(protocol))
    attempted: list[tuple[ScenarioManifest, GainAttemptResult, BenchmarkGateResult]] = []

    with _AppendOnlyCalibrationRecord(calibration_record_path) as record:
        record.append(
            {
                "event": "calibration_started",
                "protocol": protocol,
                "protocol_sha256": protocol_sha256,
            }
        )

        def run_schedule(schedule_id: str) -> None:
            for gain in spec.gains:
                manifest = build_calibration_manifest(
                    spec,
                    founders,
                    gain=gain,
                    schedule_id=schedule_id,
                )
                attempt = runner(manifest, gain, schedule_id)
                _validate_attempt(attempt, manifest, gain, schedule_id)
                gate = evaluate_benchmark_gate(attempt, spec)
                attempted.append((manifest, attempt, gate))
                record.append(
                    {
                        "attempt": _attempt_dict(attempt),
                        "event": "gain_evaluated",
                        "gate": _gate_dict(gate),
                        "protocol_sha256": protocol_sha256,
                    }
                )

        run_schedule(PRIMARY_SCHEDULE_ID)
        passing = [item for item in attempted if item[2].passed]
        fallback_used = False
        eligible = fallback_eligible(
            [item[1] for item in attempted],
            spec,
        )
        record.append(
            {
                "eligible": eligible,
                "event": "fallback_decision",
                "permitted_schedule_id": FALLBACK_SCHEDULE_ID,
                "protocol_sha256": protocol_sha256,
            }
        )
        if not passing and eligible and spec.mode == "production":
            fallback_used = True
            run_schedule(FALLBACK_SCHEDULE_ID)
            passing = [item for item in attempted if item[2].passed]

        selected = (
            min(
                passing,
                key=lambda item: (
                    item[1].gain,
                    0 if item[1].schedule_id == PRIMARY_SCHEDULE_ID else 1,
                ),
            )
            if passing
            else None
        )
        if selected is None:
            status = "smoke_completed_no_scientific_selection" if smoke else "no_predeclared_gain_passed"
            record.append(
                {
                    "event": "calibration_finished",
                    "fallback_used": fallback_used,
                    "protocol_sha256": protocol_sha256,
                    "selected_manifest_written": False,
                    "status": status,
                }
            )
            return {
                "fallback_used": fallback_used,
                "protocol_sha256": protocol_sha256,
                "selected": None,
                "status": status,
            }

        selected_manifest, selected_attempt, selected_gate = selected
        selected_sha256 = _write_manifest_once(
            selected_manifest_path,
            selected_manifest,
        )
        selection = {
            "gain": selected_attempt.gain,
            "manifest_path": str(selected_manifest_path),
            "manifest_sha256": selected_sha256,
            "schedule_id": selected_attempt.schedule_id,
            "winning_policy": selected_gate.winning_policy,
        }
        record.append(
            {
                "event": "calibration_finished",
                "fallback_used": fallback_used,
                "protocol_sha256": protocol_sha256,
                "selected": selection,
                "selected_manifest_written": True,
                "status": "selected",
            }
        )
        return {
            "fallback_used": fallback_used,
            "protocol_sha256": protocol_sha256,
            "selected": selection,
            "status": "selected",
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--founder-index", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--calibration-record", type=Path, required=True)
    parser.add_argument("--selected-manifest", type=Path, required=True)
    parser.add_argument("--readiness-seed-panel", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_actuator_calibration(
        smoke=args.smoke,
        founder_index_path=args.founder_index,
        preregistration_path=args.preregistration,
        calibration_record_path=args.calibration_record,
        selected_manifest_path=args.selected_manifest,
        readiness_seed_panel_path=args.readiness_seed_panel,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
