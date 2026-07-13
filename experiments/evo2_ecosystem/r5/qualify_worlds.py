"""Reset-only, genome-independent world qualification for Evo² r5."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import jax
import numpy as np

from experiments.evo2_ecosystem.episode import SimulatorConfig
from microcosmos.ecology import sample_grid_nearest
from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.solver.config import PBD_SCHEME, PBD_SCHEME_NO_FLUID
from microcosmos.structs.population import EcosystemState

from .protocol import (
    FROZEN_SIMULATOR_CONFIG_SHA256,
    PRODUCTION_WORLD_QUALIFICATION_SPEC,
    PROTOCOL_REVISION,
    WORLD_ACCESS_RULE_ID,
    WORLD_ACCESS_SCORE_ATOL,
    WORLD_QUALIFICATION_PATH,
    WORLD_QUALIFICATION_SCHEMA_VERSION,
    WorldQualificationSpec,
    WorldSeedRange,
)


_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CELL_ROUNDING = "floor(position + 0.5) modulo grid"
_SELECTION_RULE = "first passing seeds in ascending order; stop after required passes"


@dataclass(frozen=True)
class WorldAccessRecord:
    """Initial food-access measurement for one reset seed."""

    seed: int
    mouth_cells: tuple[tuple[int, int], ...]
    mouth_capacity_fractions: tuple[float, ...]
    access_score: float
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.mouth_cells, tuple) or not self.mouth_cells:
            raise ValueError("mouth_cells must be a nonempty tuple")
        if any(
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in cell)
            for cell in self.mouth_cells
        ):
            raise ValueError("mouth_cells must contain integer coordinate pairs")
        if (
            not isinstance(self.mouth_capacity_fractions, tuple)
            or len(self.mouth_capacity_fractions) != len(self.mouth_cells)
        ):
            raise ValueError("one capacity fraction is required per living mouth")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in self.mouth_capacity_fractions):
            raise ValueError("mouth capacity fractions must be finite and within [0, 1]")
        if not math.isfinite(self.access_score) or not 0.0 <= self.access_score <= 1.0:
            raise ValueError("access_score must be finite and within [0, 1]")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be a bool")


@dataclass(frozen=True)
class WorldQualificationTrace:
    """Complete inspected prefix for one frozen seed range."""

    seed_range: WorldSeedRange
    inspected: tuple[WorldAccessRecord, ...]
    selected_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed_range, WorldSeedRange):
            raise TypeError("seed_range must be a WorldSeedRange")
        if not isinstance(self.inspected, tuple):
            raise TypeError("inspected must be a tuple")
        if any(not isinstance(record, WorldAccessRecord) for record in self.inspected):
            raise TypeError("inspected must contain WorldAccessRecord values")
        if not isinstance(self.selected_seeds, tuple):
            raise TypeError("selected_seeds must be a tuple")


@dataclass(frozen=True)
class BackendReplay:
    """GPU replay of one CPU-selected seed."""

    partition: str
    seed: int
    mouth_cells: tuple[tuple[int, int], ...]
    access_score: float
    cells_match: bool
    score_absolute_error: float
    passed_match: bool


class InsufficientQualifiedWorlds(RuntimeError):
    """A frozen seed range ended before enough worlds passed."""

    def __init__(self, trace: WorldQualificationTrace, required: int):
        self.trace = trace
        self.required = required
        super().__init__(
            f"{trace.seed_range.partition} produced {len(trace.selected_seeds)} "
            f"qualified worlds; {required} required"
        )


WorldQualifier = Callable[[int], WorldAccessRecord]


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"invalid {context} keys; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _qualifier_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _rounded_periodic_cells(
    positions: np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    """Return ``(x, y)`` cells using the production nearest-grid rule."""
    height, width = grid_shape
    x = np.floor(positions[:, 0] + 0.5).astype(np.int64) % width
    y = np.floor(positions[:, 1] + 0.5).astype(np.int64) % height
    return tuple((int(x_value), int(y_value)) for x_value, y_value in zip(x, y, strict=True))


def measure_initial_access(
    state: EcosystemState,
    *,
    seed: int,
    peak_capacity: float,
    access_threshold: float,
) -> WorldAccessRecord:
    """Measure living-mouth food access without executing a physics step."""
    if not isinstance(state, EcosystemState):
        raise TypeError("state must be an EcosystemState")
    if not math.isfinite(peak_capacity) or peak_capacity <= 0.0:
        raise ValueError("peak_capacity must be finite and positive")
    if not math.isfinite(access_threshold) or not 0.0 <= access_threshold <= 1.0:
        raise ValueError("access_threshold must be within [0, 1]")

    alive = np.asarray(state.population.alive, dtype=np.bool_)
    positions = np.asarray(state.nodes.position)
    capacity = alive.shape[0]
    if capacity < 1 or positions.shape[0] % capacity:
        raise ValueError("node positions must divide evenly across population slots")
    nodes_per_creature = positions.shape[0] // capacity
    mouth_positions = positions.reshape(capacity, nodes_per_creature, 2)[alive, 0, :]
    if mouth_positions.shape[0] == 0:
        raise ValueError("world qualification requires at least one living mouth")

    grid_shape = tuple(int(value) for value in state.resource_capacity_map.shape)
    cells = _rounded_periodic_cells(mouth_positions, grid_shape)
    sampled = np.asarray(
        sample_grid_nearest(state.resource_capacity_map, state.nodes.position.reshape(capacity, nodes_per_creature, 2)[alive, 0, :]),
        dtype=np.float64,
    )
    fractions_array = sampled / peak_capacity
    fractions = tuple(float(value) for value in fractions_array)
    access_score = float(np.max(fractions_array))
    return WorldAccessRecord(
        seed=seed,
        mouth_cells=cells,
        mouth_capacity_fractions=fractions,
        access_score=access_score,
        passed=access_score >= access_threshold,
    )


def make_world_qualifier(
    config: SimulatorConfig,
    *,
    access_threshold: float,
) -> WorldQualifier:
    """Build a reusable reset-only qualifier for one immutable config."""
    if not isinstance(config, SimulatorConfig):
        raise TypeError("config must be a SimulatorConfig")
    solver = PBD_SCHEME if config.fluid_enabled else PBD_SCHEME_NO_FLUID
    env = EcosystemEnv(
        topology=LineTopology(
            num_nodes=config.nodes_per_creature,
            spacing=config.node_spacing,
            bending_stiffness=config.bending_stiffness,
        ),
        max_creatures=config.max_creatures,
        initial_population=config.initial_population,
        grid_shape=config.grid_shape,
        dt=config.dt,
        max_steps=1,
        solver_config=solver,
        resource_capacity=config.resource_capacity,
        initial_resource=config.initial_resource,
        resource_regeneration_rate=config.resource_regeneration_rate,
        resource_diffusion_rate=config.resource_diffusion_rate,
        resource_patch_center=config.resource_patch_center,
        resource_patch_radius=config.resource_patch_radius,
        initial_energy=config.initial_energy,
        birth_transfer_efficiency=config.birth_transfer_efficiency,
        reproduction_threshold=config.reproduction_threshold,
        reproduction_cost=config.reproduction_cost,
        maturity_age=config.maturity_age,
        maximum_lifespan=config.maximum_lifespan,
        max_bending_delta=config.max_bending_delta,
        uptake_rate=config.uptake_rate,
        assimilation_efficiency=config.assimilation_efficiency,
        basal_metabolism=config.basal_metabolism,
        actuation_power_coefficient=config.actuation_power_coefficient,
        spawn_separation=config.spawn_separation,
        placement_candidates=config.placement_candidates,
        position_margin=config.position_margin,
    )

    def qualify(seed: int) -> WorldAccessRecord:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _, state = env.reset(jax.random.PRNGKey(seed))
        return measure_initial_access(
            state,
            seed=seed,
            peak_capacity=config.resource_capacity,
            access_threshold=access_threshold,
        )

    return qualify


def scan_seed_range(
    seed_range: WorldSeedRange,
    qualifier: WorldQualifier,
    *,
    passes_required: int,
) -> WorldQualificationTrace:
    """Inspect one ascending prefix and stop immediately after enough passes."""
    if not isinstance(seed_range, WorldSeedRange):
        raise TypeError("seed_range must be a WorldSeedRange")
    if not callable(qualifier):
        raise TypeError("qualifier must be callable")
    if not isinstance(passes_required, int) or isinstance(passes_required, bool) or passes_required < 1:
        raise ValueError("passes_required must be a positive integer")

    inspected = []
    selected = []
    for seed in range(seed_range.start, seed_range.stop):
        record = qualifier(seed)
        if not isinstance(record, WorldAccessRecord) or record.seed != seed:
            raise RuntimeError("qualifier returned a record for the wrong seed")
        inspected.append(record)
        if record.passed:
            selected.append(seed)
            if len(selected) == passes_required:
                return WorldQualificationTrace(
                    seed_range=seed_range,
                    inspected=tuple(inspected),
                    selected_seeds=tuple(selected),
                )

    trace = WorldQualificationTrace(
        seed_range=seed_range,
        inspected=tuple(inspected),
        selected_seeds=tuple(selected),
    )
    raise InsufficientQualifiedWorlds(trace, passes_required)


def qualify_world_ranges(
    spec: WorldQualificationSpec = PRODUCTION_WORLD_QUALIFICATION_SPEC,
    *,
    qualifier: WorldQualifier | None = None,
) -> tuple[WorldQualificationTrace, ...]:
    """Qualify every range without publishing or inspecting any rollout outcome."""
    if not isinstance(spec, WorldQualificationSpec):
        raise TypeError("spec must be a WorldQualificationSpec")
    if qualifier is None:
        if jax.default_backend() != "cpu":
            raise RuntimeError("canonical world qualification requires the JAX CPU backend")
        qualifier = make_world_qualifier(
            spec.config,
            access_threshold=spec.access_threshold,
        )
    return tuple(
        scan_seed_range(
            seed_range,
            qualifier,
            passes_required=spec.passes_per_partition,
        )
        for seed_range in spec.seed_ranges
    )


def compare_backend_replay(
    partition: str,
    cpu_record: WorldAccessRecord,
    gpu_record: WorldAccessRecord,
) -> BackendReplay:
    """Compare one selected GPU reset with its canonical CPU record."""
    if cpu_record.seed != gpu_record.seed:
        raise ValueError("CPU and GPU replay seeds must match")
    error = abs(cpu_record.access_score - gpu_record.access_score)
    return BackendReplay(
        partition=partition,
        seed=cpu_record.seed,
        mouth_cells=gpu_record.mouth_cells,
        access_score=gpu_record.access_score,
        cells_match=gpu_record.mouth_cells == cpu_record.mouth_cells,
        score_absolute_error=error,
        passed_match=gpu_record.passed == cpu_record.passed,
    )


def _access_record_dict(
    record: WorldAccessRecord,
    *,
    selected: bool,
) -> dict[str, object]:
    return {
        "access_score": record.access_score,
        "mouth_capacity_fractions": list(record.mouth_capacity_fractions),
        "mouth_cells": [list(cell) for cell in record.mouth_cells],
        "passed": record.passed,
        "seed": record.seed,
        "selected": selected,
    }


def _trace_dict(trace: WorldQualificationTrace) -> dict[str, object]:
    selected = set(trace.selected_seeds)
    return {
        "inspected": [
            _access_record_dict(record, selected=record.seed in selected)
            for record in trace.inspected
        ],
        "inspected_count": len(trace.inspected),
        "partition": trace.seed_range.partition,
        "selected_seeds": list(trace.selected_seeds),
        "start": trace.seed_range.start,
        "stop": trace.seed_range.stop,
    }


def _replay_dict(replay: BackendReplay) -> dict[str, object]:
    return {
        "access_score": replay.access_score,
        "cells_match": replay.cells_match,
        "mouth_cells": [list(cell) for cell in replay.mouth_cells],
        "partition": replay.partition,
        "passed_match": replay.passed_match,
        "score_absolute_error": replay.score_absolute_error,
        "seed": replay.seed,
    }


def build_world_qualification_document(
    traces: Sequence[WorldQualificationTrace],
    gpu_replays: Sequence[BackendReplay],
    *,
    implementation_commit: str,
) -> dict[str, object]:
    """Build and validate the one canonical public qualification document."""
    if not _GIT_COMMIT.fullmatch(implementation_commit):
        raise ValueError("implementation_commit must be a lowercase 40-character Git hash")
    value: dict[str, object] = {
        "canonical_backend": "cpu",
        "gpu_backend": "gpu",
        "gpu_replays": [_replay_dict(replay) for replay in gpu_replays],
        "implementation_commit": implementation_commit,
        "jax_version": jax.__version__,
        "passed": True,
        "protocol_revision": PROTOCOL_REVISION,
        "qualifier_source_sha256": _qualifier_source_sha256(),
        "rule": {
            "access_rule_id": WORLD_ACCESS_RULE_ID,
            "access_threshold": PRODUCTION_WORLD_QUALIFICATION_SPEC.access_threshold,
            "cell_rounding": _CELL_ROUNDING,
            "performs_rollout": False,
            "required_passes_per_partition": PRODUCTION_WORLD_QUALIFICATION_SPEC.passes_per_partition,
            "selection": _SELECTION_RULE,
            "uses_genome": False,
        },
        "schema_version": WORLD_QUALIFICATION_SCHEMA_VERSION,
        "seed_ranges": [
            {
                "partition": item.partition,
                "start": item.start,
                "stop": item.stop,
            }
            for item in PRODUCTION_WORLD_QUALIFICATION_SPEC.seed_ranges
        ],
        "selected_seeds": {
            trace.seed_range.partition: list(trace.selected_seeds) for trace in traces
        },
        "simulator_config_sha256": FROZEN_SIMULATOR_CONFIG_SHA256,
        "traces": [_trace_dict(trace) for trace in traces],
    }
    return world_qualification_from_dict(value)


def _parse_access_record(
    value: Mapping[str, Any],
    *,
    threshold: float,
    initial_population: int,
    grid_shape: tuple[int, int],
) -> tuple[WorldAccessRecord, bool]:
    _require_exact_keys(
        value,
        {
            "access_score",
            "mouth_capacity_fractions",
            "mouth_cells",
            "passed",
            "seed",
            "selected",
        },
        "world access record",
    )
    cells_value = value["mouth_cells"]
    fractions_value = value["mouth_capacity_fractions"]
    if not isinstance(cells_value, list) or not isinstance(fractions_value, list):
        raise ValueError("mouth cells and fractions must be JSON arrays")
    record = WorldAccessRecord(
        seed=value["seed"],
        mouth_cells=tuple(tuple(cell) for cell in cells_value),
        mouth_capacity_fractions=tuple(fractions_value),
        access_score=value["access_score"],
        passed=value["passed"],
    )
    if len(record.mouth_cells) != initial_population:
        raise ValueError("world record must contain every initially living mouth")
    height, width = grid_shape
    if any(not 0 <= x < width or not 0 <= y < height for x, y in record.mouth_cells):
        raise ValueError("mouth cell lies outside the periodic grid")
    expected_score = max(record.mouth_capacity_fractions)
    if not math.isclose(record.access_score, expected_score, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("access_score does not equal the maximum mouth fraction")
    if record.passed != (record.access_score >= threshold):
        raise ValueError("world pass flag disagrees with the frozen threshold")
    if not isinstance(value["selected"], bool):
        raise ValueError("selected must be a bool")
    return record, value["selected"]


def world_qualification_from_dict(value: Mapping[str, Any]) -> dict[str, object]:
    """Strictly validate and normalize an r5 world-qualification document."""
    if not isinstance(value, Mapping):
        raise ValueError("world qualification must be an object")
    _require_exact_keys(
        value,
        {
            "canonical_backend",
            "gpu_backend",
            "gpu_replays",
            "implementation_commit",
            "jax_version",
            "passed",
            "protocol_revision",
            "qualifier_source_sha256",
            "rule",
            "schema_version",
            "seed_ranges",
            "selected_seeds",
            "simulator_config_sha256",
            "traces",
        },
        "world qualification",
    )
    if value["schema_version"] != WORLD_QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("unsupported world-qualification schema")
    if value["protocol_revision"] != PROTOCOL_REVISION:
        raise ValueError("wrong protocol revision")
    if value["canonical_backend"] != "cpu" or value["gpu_backend"] != "gpu":
        raise ValueError("world qualification requires canonical CPU plus GPU replay")
    if value["passed"] is not True:
        raise ValueError("world qualification did not pass")
    if not _GIT_COMMIT.fullmatch(value["implementation_commit"]):
        raise ValueError("invalid implementation commit")
    if not isinstance(value["jax_version"], str) or not value["jax_version"]:
        raise ValueError("jax_version must be a non-empty string")
    if not _SHA256.fullmatch(value["qualifier_source_sha256"]):
        raise ValueError("invalid qualifier source hash")
    if value["qualifier_source_sha256"] != _qualifier_source_sha256():
        raise ValueError("qualifier source hash does not match the current implementation")
    if value["simulator_config_sha256"] != FROZEN_SIMULATOR_CONFIG_SHA256:
        raise ValueError("simulator config hash does not match the frozen r5 config")

    rule = value["rule"]
    if not isinstance(rule, Mapping):
        raise ValueError("rule must be an object")
    _require_exact_keys(
        rule,
        {
            "access_rule_id",
            "access_threshold",
            "cell_rounding",
            "performs_rollout",
            "required_passes_per_partition",
            "selection",
            "uses_genome",
        },
        "qualification rule",
    )
    expected_rule = {
        "access_rule_id": WORLD_ACCESS_RULE_ID,
        "access_threshold": PRODUCTION_WORLD_QUALIFICATION_SPEC.access_threshold,
        "cell_rounding": _CELL_ROUNDING,
        "performs_rollout": False,
        "required_passes_per_partition": PRODUCTION_WORLD_QUALIFICATION_SPEC.passes_per_partition,
        "selection": _SELECTION_RULE,
        "uses_genome": False,
    }
    if dict(rule) != expected_rule:
        raise ValueError("qualification rule does not match the frozen r5 rule")

    expected_ranges = [
        {"partition": item.partition, "start": item.start, "stop": item.stop}
        for item in PRODUCTION_WORLD_QUALIFICATION_SPEC.seed_ranges
    ]
    if value["seed_ranges"] != expected_ranges:
        raise ValueError("seed ranges do not match the frozen r5 ranges")
    if not isinstance(value["traces"], list) or not isinstance(value["selected_seeds"], Mapping):
        raise ValueError("traces and selected_seeds have invalid types")
    if len(value["traces"]) != len(expected_ranges):
        raise ValueError("one trace is required per frozen seed range")

    cpu_records: dict[tuple[str, int], WorldAccessRecord] = {}
    normalized_traces = []
    expected_selected: dict[str, list[int]] = {}
    config = PRODUCTION_WORLD_QUALIFICATION_SPEC.config
    threshold = PRODUCTION_WORLD_QUALIFICATION_SPEC.access_threshold
    required = PRODUCTION_WORLD_QUALIFICATION_SPEC.passes_per_partition
    for raw_trace, seed_range in zip(
        value["traces"],
        PRODUCTION_WORLD_QUALIFICATION_SPEC.seed_ranges,
        strict=True,
    ):
        if not isinstance(raw_trace, Mapping):
            raise ValueError("trace must be an object")
        _require_exact_keys(
            raw_trace,
            {"inspected", "inspected_count", "partition", "selected_seeds", "start", "stop"},
            "qualification trace",
        )
        if (
            raw_trace["partition"] != seed_range.partition
            or raw_trace["start"] != seed_range.start
            or raw_trace["stop"] != seed_range.stop
        ):
            raise ValueError("qualification trace is bound to the wrong seed range")
        if not isinstance(raw_trace["inspected"], list) or not raw_trace["inspected"]:
            raise ValueError("qualification trace must contain an inspected prefix")
        if raw_trace["inspected_count"] != len(raw_trace["inspected"]):
            raise ValueError("inspected_count does not match the trace")
        parsed = []
        selected = []
        for offset, raw_record in enumerate(raw_trace["inspected"]):
            if not isinstance(raw_record, Mapping):
                raise ValueError("world access record must be an object")
            record, selected_flag = _parse_access_record(
                raw_record,
                threshold=threshold,
                initial_population=config.initial_population,
                grid_shape=config.grid_shape,
            )
            if record.seed != seed_range.start + offset:
                raise ValueError("qualification trace is not a complete ascending prefix")
            if selected_flag != record.passed:
                raise ValueError("the ascending stopping rule must select every passing prefix seed")
            if selected_flag:
                selected.append(record.seed)
            parsed.append(record)
            cpu_records[(seed_range.partition, record.seed)] = record
        if len(selected) != required or not parsed[-1].passed:
            raise ValueError("trace must stop exactly at the required passing seed")
        if raw_trace["selected_seeds"] != selected:
            raise ValueError("trace selected_seeds disagree with selection flags")
        expected_selected[seed_range.partition] = selected
        normalized_traces.append(raw_trace)

    if dict(value["selected_seeds"]) != expected_selected:
        raise ValueError("top-level selected seeds disagree with qualification traces")

    raw_replays = value["gpu_replays"]
    if not isinstance(raw_replays, list):
        raise ValueError("gpu_replays must be a JSON array")
    expected_replay_keys = [
        (partition, seed)
        for partition, seeds in expected_selected.items()
        for seed in seeds
    ]
    if len(raw_replays) != len(expected_replay_keys):
        raise ValueError("every selected seed requires exactly one GPU replay")
    seen_replays = []
    for raw_replay, expected_key in zip(raw_replays, expected_replay_keys, strict=True):
        if not isinstance(raw_replay, Mapping):
            raise ValueError("GPU replay must be an object")
        _require_exact_keys(
            raw_replay,
            {
                "access_score",
                "cells_match",
                "mouth_cells",
                "partition",
                "passed_match",
                "score_absolute_error",
                "seed",
            },
            "GPU replay",
        )
        key = (raw_replay["partition"], raw_replay["seed"])
        if key != expected_key or key in seen_replays:
            raise ValueError("GPU replays must match selected seeds in canonical order")
        seen_replays.append(key)
        cpu_record = cpu_records[key]
        cells = raw_replay["mouth_cells"]
        if not isinstance(cells, list):
            raise ValueError("GPU mouth_cells must be a JSON array")
        if any(
            not isinstance(cell, list)
            or len(cell) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in cell)
            for cell in cells
        ):
            raise ValueError("GPU mouth_cells must contain integer coordinate pairs")
        normalized_cells = tuple(tuple(cell) for cell in cells)
        if raw_replay["cells_match"] is not True or normalized_cells != cpu_record.mouth_cells:
            raise ValueError("GPU rounded mouth cells do not match canonical CPU cells")
        score = raw_replay["access_score"]
        error = raw_replay["score_absolute_error"]
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError("GPU access_score must be finite and within [0, 1]")
        expected_error = abs(float(score) - cpu_record.access_score)
        if (
            not isinstance(error, (int, float))
            or isinstance(error, bool)
            or not math.isclose(float(error), expected_error, rel_tol=0.0, abs_tol=1e-12)
            or float(error) > WORLD_ACCESS_SCORE_ATOL
        ):
            raise ValueError("GPU access score exceeds the frozen CPU agreement tolerance")
        expected_passed_match = (float(score) >= threshold) == cpu_record.passed
        if raw_replay["passed_match"] is not expected_passed_match:
            raise ValueError("GPU passed_match diagnostic is inconsistent")

    return json.loads(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def canonical_world_qualification_bytes(value: Mapping[str, Any]) -> bytes:
    """Return deterministic canonical bytes for a validated document."""
    normalized = world_qualification_from_dict(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def publish_world_qualification(
    value: Mapping[str, Any],
    path: Path = WORLD_QUALIFICATION_PATH,
) -> str:
    """Atomically create the canonical write-once qualification artifact."""
    if not isinstance(path, Path):
        path = Path(path)
    payload = canonical_world_qualification_bytes(value) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace world qualification: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace world qualification: {path}") from error
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def validate_world_qualification(
    path: str | Path = WORLD_QUALIFICATION_PATH,
    *,
    expected_implementation_commit: str | None = None,
) -> dict[str, object]:
    """Authenticate canonical bytes and every frozen r5 qualification rule."""
    path = Path(path)
    payload = path.read_bytes()
    try:
        decoded = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("world qualification is not valid ASCII JSON") from error
    normalized = world_qualification_from_dict(decoded)
    if payload != canonical_world_qualification_bytes(normalized) + b"\n":
        raise ValueError("world qualification does not use canonical JSON bytes")
    if (
        expected_implementation_commit is not None
        and normalized["implementation_commit"] != expected_implementation_commit
    ):
        raise ValueError("world qualification is bound to the wrong implementation commit")
    return normalized


__all__ = [
    "BackendReplay",
    "InsufficientQualifiedWorlds",
    "WorldAccessRecord",
    "WorldQualificationTrace",
    "build_world_qualification_document",
    "canonical_world_qualification_bytes",
    "compare_backend_replay",
    "make_world_qualifier",
    "measure_initial_access",
    "publish_world_qualification",
    "qualify_world_ranges",
    "scan_seed_range",
    "validate_world_qualification",
    "world_qualification_from_dict",
]
