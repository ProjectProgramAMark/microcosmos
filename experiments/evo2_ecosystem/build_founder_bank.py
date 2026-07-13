"""Build the frozen Evo² heredity-adaptation founder bank.

The builder deliberately screens only uninjured clone-policy viability.  It
assigns candidate-index ranges to partitions before any rollout, records every
evaluated candidate in a canonical append-only JSONL audit, and publishes no
founder artifacts until every partition has enough passing candidates.

Production execution is GPU/fluid/full-scale only.  ``--smoke`` is a tiny CPU
workflow check and must never be used as a scientific founder bank.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.cppn import (
    FOUNDER_PANEL_SEED,
    CPPNGenome,
    cached_genome_valid,
    initialize_cppn_population,
    transform_and_validate_genome,
)
from microcosmos.heredity import clone_policy
from microcosmos.rng import RNGTag, derive_key
from microcosmos.rollout import combine_chunk_metrics, run_ecosystem_chunk

from .episode import SimulatorConfig, build_environment, simulator_config_sha256
from .founder_artifacts import (
    FounderIndex,
    canonical_founder_artifact_bytes,
    founder_index_sha256,
    load_founder_index,
    make_founder_record,
    write_founder_artifact,
    write_founder_index,
)
from .protocol import normalized_chunk_productivity


SCREENING_SCHEMA_VERSION = 1
SELECTION_RULE_ID = "ascending_first_pass_uninjured_v1"
PARTITION_ORDER = ("training", "development", "sealed")
PRODUCTION_SCREENING_SEEDS = (131, 257, 389)
SMOKE_SCREENING_SEEDS = (0,)


@dataclass(frozen=True)
class CandidateRange:
    """One immutable pre-screening candidate assignment."""

    partition: str
    start: int
    stop: int
    required: int

    def __post_init__(self) -> None:
        if self.partition not in PARTITION_ORDER:
            raise ValueError(f"unknown founder partition: {self.partition!r}")
        values = (self.start, self.stop, self.required)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TypeError("candidate range values must be integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("candidate range must be nonempty and nonnegative")
        if not 0 < self.required <= self.stop - self.start:
            raise ValueError("required founders must fit inside the candidate range")


@dataclass(frozen=True)
class ScreeningGates:
    """Frozen uninjured viability thresholds applied independently per seed."""

    minimum_final_alive: int
    minimum_births: int
    minimum_generation: int
    minimum_productivity: float

    def __post_init__(self) -> None:
        integer_values = (
            self.minimum_final_alive,
            self.minimum_births,
            self.minimum_generation,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in integer_values):
            raise ValueError("integer screening gates must be nonnegative")
        if (
            not isinstance(self.minimum_productivity, (int, float))
            or isinstance(self.minimum_productivity, bool)
            or not np.isfinite(self.minimum_productivity)
            or self.minimum_productivity < 0.0
        ):
            raise ValueError("minimum_productivity must be finite and nonnegative")


@dataclass(frozen=True)
class ScreeningSpec:
    """Complete frozen execution and selection contract."""

    mode: str
    config: SimulatorConfig
    seeds: tuple[int, ...]
    horizon: int
    chunk_steps: int
    candidate_ranges: tuple[CandidateRange, ...]
    gates: ScreeningGates
    selection_rule_id: str = SELECTION_RULE_ID

    def __post_init__(self) -> None:
        if self.mode not in {"production", "r4_production", "smoke"}:
            raise ValueError("unknown screening mode")
        if not isinstance(self.config, SimulatorConfig):
            raise TypeError("config must be a SimulatorConfig")
        if (
            not isinstance(self.seeds, tuple)
            or not self.seeds
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
        ):
            raise ValueError("screening seeds must be unique nonnegative integers")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(self.chunk_steps, int) or isinstance(self.chunk_steps, bool) or self.chunk_steps <= 0 or self.horizon % self.chunk_steps:
            raise ValueError("chunk_steps must positively divide horizon")
        if not isinstance(self.candidate_ranges, tuple):
            raise TypeError("candidate_ranges must be a tuple")
        if tuple(item.partition for item in self.candidate_ranges) != PARTITION_ORDER:
            raise ValueError("candidate ranges must use the frozen partition order")
        previous_stop = 0
        for item in self.candidate_ranges:
            if item.start != previous_stop:
                raise ValueError("candidate ranges must be contiguous and disjoint")
            previous_stop = item.stop
        required = tuple(item.required for item in self.candidate_ranges)
        expected_required = (4, 4, 8) if self.mode == "r4_production" else (3, 2, 3)
        if required != expected_required:
            raise ValueError(f"founder bank requires exactly the {expected_required} panel")
        if not isinstance(self.gates, ScreeningGates):
            raise TypeError("gates must be ScreeningGates")

    @property
    def candidate_count(self) -> int:
        return self.candidate_ranges[-1].stop


PRODUCTION_SPEC = ScreeningSpec(
    mode="production",
    config=SimulatorConfig(),
    seeds=PRODUCTION_SCREENING_SEEDS,
    horizon=8_000,
    chunk_steps=500,
    candidate_ranges=(
        CandidateRange("training", 0, 16, 3),
        CandidateRange("development", 16, 32, 2),
        CandidateRange("sealed", 32, 48, 3),
    ),
    gates=ScreeningGates(
        minimum_final_alive=2,
        minimum_births=2,
        minimum_generation=1,
        minimum_productivity=0.01,
    ),
)

R4_PRODUCTION_SPEC = ScreeningSpec(
    mode="r4_production",
    config=SimulatorConfig(),
    seeds=(4_101, 4_102),
    horizon=4_000,
    chunk_steps=500,
    candidate_ranges=(
        CandidateRange("training", 0, 24, 4),
        CandidateRange("development", 24, 48, 4),
        CandidateRange("sealed", 48, 80, 8),
    ),
    gates=ScreeningGates(
        minimum_final_alive=2,
        minimum_births=2,
        minimum_generation=1,
        minimum_productivity=0.01,
    ),
    selection_rule_id="ascending_first_pass_healthy_clone_r4_v1",
)

SMOKE_SPEC = ScreeningSpec(
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
        initial_energy=10.0,
        reproduction_threshold=100.0,
        basal_metabolism=0.0,
        actuation_power_coefficient=0.0,
        placement_candidates=4,
        position_margin=0.5,
    ),
    seeds=SMOKE_SCREENING_SEEDS,
    horizon=4,
    chunk_steps=2,
    candidate_ranges=(
        CandidateRange("training", 0, 4, 3),
        CandidateRange("development", 4, 8, 2),
        CandidateRange("sealed", 8, 12, 3),
    ),
    # Smoke proves code paths, not scientific reproduction or adaptation.
    gates=ScreeningGates(
        minimum_final_alive=1,
        minimum_births=0,
        minimum_generation=0,
        minimum_productivity=0.0,
    ),
)


@dataclass(frozen=True)
class SeedScreenResult:
    """Compact result of one uninjured clone-policy rollout."""

    seed: int
    finite: bool
    integrity_valid: bool
    final_alive: int
    birth_count: int
    maximum_generation: int
    productivity: float


@dataclass(frozen=True)
class CandidateScreenResult:
    """Graph/cache checks plus every fixed-seed viability measurement."""

    graph_valid: bool
    cache_valid: bool
    seed_results: tuple[SeedScreenResult, ...]
    error_type: str | None = None
    error: str | None = None


CandidateRunner = Callable[[CPPNGenome, ScreeningSpec], CandidateScreenResult]


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


class _AppendOnlyScreeningRecord:
    """Create one canonical JSONL audit and append each event durably."""

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
            raise FileExistsError(f"refusing to replace append-only screening record: {path}") from error
        self._sequence = 0

    def append(self, event: dict[str, object]) -> None:
        value = {
            "schema_version": SCREENING_SCHEMA_VERSION,
            "sequence": self._sequence,
            **event,
        }
        payload = _canonical_json_bytes(value) + b"\n"
        written = os.write(self._descriptor, payload)
        if written != len(payload):
            raise OSError("short write to screening record")
        os.fsync(self._descriptor)
        self._sequence += 1

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> _AppendOnlyScreeningRecord:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _selection_rule(spec: ScreeningSpec) -> dict[str, object]:
    return {
        "candidate_ranges": [asdict(item) for item in spec.candidate_ranges],
        "injury_screening": False,
        "partition_order": list(PARTITION_ORDER),
        "rule_id": spec.selection_rule_id,
        "rule_text": ("Within each range fixed before screening, select the first required passing candidates in ascending candidate-index order."),
    }


def selection_rule_sha256(spec: ScreeningSpec) -> str:
    """Hash the exact deterministic partition and selection rule."""
    return _sha256(_canonical_json_bytes(_selection_rule(spec)))


def _protocol(spec: ScreeningSpec) -> dict[str, object]:
    seed_horizons = _screening_horizons(spec)
    return {
        "builder_source_sha256": _sha256(Path(__file__).read_bytes()),
        "candidate_generator": {
            "candidate_count": spec.candidate_count,
            "founder_panel_seed": FOUNDER_PANEL_SEED,
            "function": ("microcosmos.cppn.initialize_cppn_population"),
            "function_source_sha256": _source_sha256(initialize_cppn_population),
            "initial_population": spec.candidate_count,
        },
        "chunk_steps": spec.chunk_steps,
        "clone_policy_source_sha256": _source_sha256(clone_policy),
        "gates": asdict(spec.gates),
        "horizon": spec.horizon,
        "mode": spec.mode,
        "screening_environment": "uninjured_only",
        "screening_seeds": list(spec.seeds),
        "screening_seed_horizons": list(seed_horizons),
        "selection_rule": _selection_rule(spec),
        "selection_rule_sha256": selection_rule_sha256(spec),
        "simulator_config": asdict(spec.config),
        "simulator_config_sha256": simulator_config_sha256(spec.config),
    }


def validate_execution_contract(spec: ScreeningSpec, *, backend: str) -> None:
    """Prevent smoke or modified substrates from publishing production evidence."""
    if not isinstance(spec, ScreeningSpec):
        raise TypeError("spec must be a ScreeningSpec")
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    if spec.mode in {"production", "r4_production"}:
        expected_spec = R4_PRODUCTION_SPEC if spec.mode == "r4_production" else PRODUCTION_SPEC
        if spec != expected_spec:
            raise RuntimeError("production screening requires the frozen full spec")
        if backend != "gpu":
            raise RuntimeError("production screening requires the JAX GPU backend")
        if not spec.config.fluid_enabled:
            raise RuntimeError("production screening requires fluid physics")
        expected_shape = (
            SimulatorConfig().max_creatures,
            SimulatorConfig().nodes_per_creature,
            SimulatorConfig().grid_shape,
        )
        actual_shape = (
            spec.config.max_creatures,
            spec.config.nodes_per_creature,
            spec.config.grid_shape,
        )
        if actual_shape != expected_shape:
            raise RuntimeError(f"production screening requires full shape {expected_shape}")
        return
    if spec != SMOKE_SPEC:
        raise RuntimeError("smoke screening requires the frozen tiny spec")
    if backend != "cpu":
        raise RuntimeError("smoke screening requires the JAX CPU backend")
    if spec.config.fluid_enabled:
        raise RuntimeError("smoke screening must use the no-fluid substrate")


def candidate_pool(spec: ScreeningSpec) -> tuple[CPPNGenome, ...]:
    """Generate the one fixed varied-founder panel named by ``spec``."""
    population = initialize_cppn_population(
        capacity=spec.candidate_count,
        initial_population=spec.candidate_count,
    )
    return tuple(
        CPPNGenome(
            population.node_genes[index],
            population.connection_genes[index],
        )
        for index in range(spec.candidate_count)
    )


def _step_keys(root_key: jax.Array, first_step: int, count: int) -> jax.Array:
    steps = jnp.arange(first_step, first_step + count, dtype=jnp.uint32)
    return jax.vmap(lambda step: derive_key(root_key, RNGTag.ENVIRONMENT, step))(steps)


def _screening_horizons(spec: ScreeningSpec) -> tuple[int, ...]:
    """Return the exact pre-event boundary paired with each screening seed."""
    if spec.mode == "r4_production":
        return (3_500, 4_000)
    return (spec.horizon,) * len(spec.seeds)


def make_screen_runner(spec: ScreeningSpec) -> CandidateRunner:
    """Build one shared compiled uninjured clone-policy screening runner."""
    env = build_environment(spec.config, clone_policy, horizon=spec.horizon)
    compiled_chunk = jax.jit(lambda state, keys: run_ecosystem_chunk(env, state, keys))

    def screen(genome: CPPNGenome, _: ScreeningSpec) -> CandidateScreenResult:
        canonical, order, connection_index, graph_valid = transform_and_validate_genome(genome)
        cache_valid = bool(graph_valid & cached_genome_valid(canonical, order, connection_index))
        if not bool(graph_valid) or not cache_valid:
            return CandidateScreenResult(
                graph_valid=bool(graph_valid),
                cache_valid=cache_valid,
                seed_results=(),
            )

        seed_results = []
        for seed, screen_horizon in zip(
            spec.seeds, _screening_horizons(spec), strict=True
        ):
            root_key = jax.random.PRNGKey(seed)
            _, state = env.reset(root_key, founder_genome=canonical)
            combined = None
            for first_step in range(0, screen_horizon, spec.chunk_steps):
                keys = _step_keys(root_key, first_step, spec.chunk_steps)
                state, metrics = compiled_chunk(state, keys)
                combined = metrics if combined is None else combine_chunk_metrics(combined, metrics)
            if combined is None:  # Defensive; the spec requires a positive horizon.
                raise RuntimeError("screening produced no rollout metrics")
            final_alive = int(np.asarray(jnp.sum(state.population.alive)))
            operator_accounting_valid = bool(np.asarray(jnp.sum(combined.operator_counts) == combined.birth_count))
            integrity_valid = bool(
                np.asarray(
                    combined.finite
                    & combined.identity_valid
                    & combined.events_valid
                    & combined.infrastructure_valid
                    & operator_accounting_valid
                    & (combined.policy_violation_count == 0)
                )
            )
            productivity = float(
                np.asarray(
                    normalized_chunk_productivity(
                        combined.cumulative_reward,
                        screen_horizon,
                        env.dt,
                        state.resource_regeneration_map,
                    )
                )
            )
            seed_results.append(
                SeedScreenResult(
                    seed=seed,
                    finite=bool(np.asarray(combined.finite)),
                    integrity_valid=integrity_valid,
                    final_alive=final_alive,
                    birth_count=int(np.asarray(combined.birth_count)),
                    maximum_generation=int(np.asarray(combined.maximum_generation)),
                    productivity=productivity,
                )
            )
        return CandidateScreenResult(
            graph_valid=True,
            cache_valid=True,
            seed_results=tuple(seed_results),
        )

    return screen


def _gate_result(
    result: CandidateScreenResult,
    spec: ScreeningSpec,
) -> tuple[bool, tuple[str, ...], dict[str, object]]:
    reasons = []
    if result.error_type is not None:
        reasons.append("runner_error")
    if not result.graph_valid:
        reasons.append("invalid_graph")
    if not result.cache_valid:
        reasons.append("invalid_controller_cache")
    if tuple(item.seed for item in result.seed_results) != spec.seeds:
        reasons.append("seed_panel_mismatch")
    if not result.seed_results:
        reasons.append("no_seed_results")

    if result.seed_results:
        minimum_final_alive = min(item.final_alive for item in result.seed_results)
        minimum_births = min(item.birth_count for item in result.seed_results)
        minimum_generation = min(item.maximum_generation for item in result.seed_results)
        minimum_productivity = min(item.productivity for item in result.seed_results)
        all_finite = all(item.finite for item in result.seed_results)
        all_integrity_valid = all(item.integrity_valid for item in result.seed_results)
    else:
        minimum_final_alive = 0
        minimum_births = 0
        minimum_generation = 0
        minimum_productivity = 0.0
        all_finite = False
        all_integrity_valid = False

    if not all_finite:
        reasons.append("nonfinite_rollout")
    if not all_integrity_valid:
        reasons.append("integrity_failure")
    if minimum_final_alive < spec.gates.minimum_final_alive:
        reasons.append("insufficient_survival")
    if minimum_births < spec.gates.minimum_births:
        reasons.append("insufficient_reproduction")
    if minimum_generation < spec.gates.minimum_generation:
        reasons.append("insufficient_generations")
    if minimum_productivity < spec.gates.minimum_productivity:
        reasons.append("insufficient_productivity")

    summary = {
        "all_finite": all_finite,
        "all_integrity_valid": all_integrity_valid,
        "minimum_births": minimum_births,
        "minimum_final_alive": minimum_final_alive,
        "minimum_generation": minimum_generation,
        "minimum_productivity": minimum_productivity,
    }
    return not reasons, tuple(reasons), summary


def _screen_result_dict(result: CandidateScreenResult) -> dict[str, object]:
    return {
        "cache_valid": result.cache_valid,
        "error": result.error,
        "error_type": result.error_type,
        "graph_valid": result.graph_valid,
        "seed_results": [asdict(item) for item in result.seed_results],
    }


def _founder_name(partition: str, selected_index: int) -> str:
    prefix = {
        "training": "train",
        "development": "dev",
        "sealed": "sealed",
    }[partition]
    return f"{prefix}-{selected_index:02d}"


def _publish_bank(
    bank_directory: Path,
    selected: dict[str, list[tuple[int, CPPNGenome]]],
    selection_rule_id: str = SELECTION_RULE_ID,
    *,
    lock_holdouts: bool = False,
) -> tuple[FounderIndex, str]:
    """Publish through the trusted artifact API after all quotas pass."""
    staging = bank_directory.with_name(f".{bank_directory.name}.staging")
    if bank_directory.exists():
        raise FileExistsError(f"refusing to replace immutable founder bank: {bank_directory}")
    if staging.exists():
        raise FileExistsError(f"founder-bank staging path already exists: {staging}")
    staging.mkdir(parents=True)
    records = []
    try:
        for partition in PARTITION_ORDER:
            for selected_index, (_candidate_index, genome) in enumerate(selected[partition]):
                founder_id = _founder_name(partition, selected_index)
                artifact = f"{founder_id}.npz"
                digests = write_founder_artifact(staging / artifact, genome)
                records.append(
                    make_founder_record(
                        founder_id=founder_id,
                        partition=partition,
                        artifact=artifact,
                        digests=digests,
                        selection_rule=selection_rule_id,
                        selection_seed=FOUNDER_PANEL_SEED,
                    )
                )
                # Candidate index is intentionally kept in the append-only
                # screening record; the immutable index stores final identities.
        index = FounderIndex(founders=tuple(records))
        index_sha256 = write_founder_index(staging / "index.json", index)
        if index_sha256 != founder_index_sha256(index):
            raise RuntimeError("founder index hash changed during publication")
        load_founder_index(staging / "index.json", verify_artifacts=True)
        if lock_holdouts:
            for record in records:
                mode = 0o444 if record.partition == "training" else 0o000
                (staging / record.artifact).chmod(mode)
            (staging / "index.json").chmod(0o444)
        bank_directory.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(bank_directory)
        return index, index_sha256
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_founder_bank_builder(
    *,
    smoke: bool,
    r4: bool = False,
    bank_directory: Path,
    screening_record_path: Path,
    runner: CandidateRunner | None = None,
    backend: str | None = None,
) -> dict[str, object]:
    """Screen a prepartitioned pool and atomically publish a complete bank."""
    if smoke and r4:
        raise ValueError("smoke and r4 modes are mutually exclusive")
    spec = SMOKE_SPEC if smoke else (R4_PRODUCTION_SPEC if r4 else PRODUCTION_SPEC)
    actual_backend = jax.default_backend()
    if not smoke and (runner is not None or backend is not None):
        raise RuntimeError("production founder screening forbids injected runners and backend overrides")
    if backend is not None and backend != actual_backend:
        raise RuntimeError("declared smoke backend does not match jax.default_backend()")
    validate_execution_contract(spec, backend=actual_backend)
    if bank_directory.exists():
        raise FileExistsError(f"refusing to replace immutable founder bank: {bank_directory}")
    if screening_record_path.resolve() == bank_directory.resolve():
        raise ValueError("screening record and bank directory must differ")

    protocol = _protocol(spec)
    protocol_sha256 = _sha256(_canonical_json_bytes(protocol))
    pool = candidate_pool(spec)
    screen = make_screen_runner(spec) if runner is None else runner
    selected: dict[str, list[tuple[int, CPPNGenome]]] = {partition: [] for partition in PARTITION_ORDER}

    with _AppendOnlyScreeningRecord(screening_record_path) as record:
        record.append(
            {
                "backend": actual_backend,
                "event": "screening_started",
                "protocol": protocol,
                "protocol_sha256": protocol_sha256,
            }
        )
        for assigned_range in spec.candidate_ranges:
            for candidate_index in range(
                assigned_range.start,
                assigned_range.stop,
            ):
                genome = pool[candidate_index]
                _, digests = canonical_founder_artifact_bytes(genome)
                try:
                    result = screen(genome, spec)
                    if not isinstance(result, CandidateScreenResult):
                        raise TypeError("candidate runner must return CandidateScreenResult")
                except Exception as error:
                    result = CandidateScreenResult(
                        graph_valid=False,
                        cache_valid=False,
                        seed_results=(),
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                passed, reasons, summary = _gate_result(result, spec)
                record.append(
                    {
                        "candidate_artifact_sha256": digests.artifact_sha256,
                        "candidate_connection_genes_sha256": (digests.connection_genes_sha256),
                        "candidate_index": candidate_index,
                        "candidate_node_genes_sha256": (digests.node_genes_sha256),
                        "event": "candidate_screened",
                        "gate": {
                            "passed": passed,
                            "reasons": list(reasons),
                            "summary": summary,
                        },
                        "partition": assigned_range.partition,
                        "protocol_sha256": protocol_sha256,
                        "result": _screen_result_dict(result),
                    }
                )
                if passed:
                    partition_selected = selected[assigned_range.partition]
                    if len(partition_selected) < assigned_range.required:
                        partition_selected.append((candidate_index, genome))
                    if spec.mode == "r4_production" and len(partition_selected) == assigned_range.required:
                        break
            record.append(
                {
                    "event": "partition_screening_completed",
                    "partition": assigned_range.partition,
                    "protocol_sha256": protocol_sha256,
                    "required": assigned_range.required,
                    "selected_candidate_indices": [item[0] for item in selected[assigned_range.partition]],
                }
            )

        complete = all(len(selected[item.partition]) == item.required for item in spec.candidate_ranges)
        selection = {partition: [index for index, _ in selected[partition]] for partition in PARTITION_ORDER}
        record.append(
            {
                "complete": complete,
                "event": "selection_completed",
                "protocol_sha256": protocol_sha256,
                "selected_candidate_indices": selection,
                "selection_rule_sha256": selection_rule_sha256(spec),
            }
        )
        if not complete:
            record.append(
                {
                    "bank_published": False,
                    "event": "screening_finished",
                    "protocol_sha256": protocol_sha256,
                    "status": "insufficient_passing_candidates",
                }
            )
            return {
                "bank_published": False,
                "protocol_sha256": protocol_sha256,
                "screening_record": str(screening_record_path),
                "selected_candidate_indices": selection,
                "status": "insufficient_passing_candidates",
            }

        index, index_sha256 = _publish_bank(
            bank_directory,
            selected,
            spec.selection_rule_id,
            lock_holdouts=spec.mode == "r4_production",
        )
        record.append(
            {
                "bank_directory": str(bank_directory),
                "bank_published": True,
                "event": "screening_finished",
                "founder_artifact_sha256": {item.founder_id: item.artifact_sha256 for item in index.founders},
                "founder_index_sha256": index_sha256,
                "protocol_sha256": protocol_sha256,
                "status": "published",
            }
        )
    return {
        "bank_directory": str(bank_directory),
        "bank_published": True,
        "founder_index_sha256": index_sha256,
        "protocol_sha256": protocol_sha256,
        "screening_record": str(screening_record_path),
        "selected_candidate_indices": selection,
        "status": "published",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the tiny CPU/no-fluid workflow check; never scientific",
    )
    parser.add_argument(
        "--r4",
        action="store_true",
        help="run the frozen 4/4/8 r4 GPU founder screen",
    )
    parser.add_argument(
        "--bank-directory",
        type=Path,
        help=("immutable bank destination; production defaults to the planned heredity_adaptation directory, while smoke requires an explicit path"),
    )
    parser.add_argument(
        "--screening-record",
        type=Path,
        help="append-only JSONL audit path (defaults beside the bank)",
    )
    args = parser.parse_args()

    bank_directory = args.bank_directory
    if bank_directory is None:
        if args.smoke:
            parser.error("--smoke requires an explicit --bank-directory")
        suffix = "heredity_adaptation_v4" if args.r4 else "heredity_adaptation"
        bank_directory = Path(__file__).with_name("founders") / suffix
    screening_record = args.screening_record
    if screening_record is None:
        screening_record = bank_directory.with_suffix(".screening.jsonl")

    result = run_founder_bank_builder(
        smoke=args.smoke,
        r4=args.r4,
        bank_directory=bank_directory,
        screening_record_path=screening_record,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
