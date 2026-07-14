"""Reset-only, genome-independent world qualification for Evo² R6."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import jax

from experiments.evo2_ecosystem.r5.qualify_worlds import BackendReplay
from experiments.evo2_ecosystem.r5.qualify_worlds import InsufficientQualifiedWorlds
from experiments.evo2_ecosystem.r5.qualify_worlds import WorldAccessRecord
from experiments.evo2_ecosystem.r5.qualify_worlds import WorldQualificationTrace
from experiments.evo2_ecosystem.r5.qualify_worlds import compare_backend_replay
from experiments.evo2_ecosystem.r5.qualify_worlds import make_world_qualifier
from experiments.evo2_ecosystem.r5.qualify_worlds import qualify_world_ranges as _qualify_world_ranges
from experiments.evo2_ecosystem.r5.qualify_worlds import scan_seed_range

from .protocol import FROZEN_SIMULATOR_CONFIG_SHA256
from .protocol import PRODUCTION_WORLD_QUALIFICATION_SPEC
from .protocol import PROTOCOL_REVISION
from .protocol import WORLD_ACCESS_RULE_ID
from .protocol import WORLD_ACCESS_SCORE_ATOL
from .protocol import WORLD_QUALIFICATION_PATH
from .protocol import WORLD_QUALIFICATION_SCHEMA_VERSION


_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CELL_ROUNDING = "floor(position + 0.5) modulo grid"
_SELECTION_RULE = "first passing seeds in ascending order; stop after required passes"


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(f"invalid {context} keys; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def qualify_world_ranges(
    spec=PRODUCTION_WORLD_QUALIFICATION_SPEC,
    *,
    qualifier=None,
) -> tuple[WorldQualificationTrace, ...]:
    """Run the generic reset-only scan under the exact R6 contract."""
    if spec != PRODUCTION_WORLD_QUALIFICATION_SPEC:
        raise ValueError("R6 qualification requires the frozen production spec")
    return _qualify_world_ranges(spec, qualifier=qualifier)


def _record_dict(record: WorldAccessRecord, *, selected: bool) -> dict[str, object]:
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
        "inspected": [_record_dict(record, selected=record.seed in selected) for record in trace.inspected],
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
    if not _GIT_COMMIT.fullmatch(implementation_commit):
        raise ValueError("implementation_commit must be a lowercase Git hash")
    value = {
        "canonical_backend": "cpu",
        "gpu_backend": "gpu",
        "gpu_replays": [_replay_dict(value) for value in gpu_replays],
        "implementation_commit": implementation_commit,
        "jax_version": jax.__version__,
        "passed": True,
        "protocol_revision": PROTOCOL_REVISION,
        "qualifier_source_sha256": _source_sha256(),
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
        "seed_ranges": [{"partition": item.partition, "start": item.start, "stop": item.stop} for item in PRODUCTION_WORLD_QUALIFICATION_SPEC.seed_ranges],
        "selected_seeds": {trace.seed_range.partition: list(trace.selected_seeds) for trace in traces},
        "simulator_config_sha256": FROZEN_SIMULATOR_CONFIG_SHA256,
        "traces": [_trace_dict(trace) for trace in traces],
    }
    return world_qualification_from_dict(value)


def _parse_record(value: Mapping[str, Any]) -> tuple[WorldAccessRecord, bool]:
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
    if not isinstance(value["mouth_cells"], list) or not isinstance(value["mouth_capacity_fractions"], list):
        raise ValueError("mouth cells and fractions must be arrays")
    record = WorldAccessRecord(
        seed=value["seed"],
        mouth_cells=tuple(tuple(cell) for cell in value["mouth_cells"]),
        mouth_capacity_fractions=tuple(value["mouth_capacity_fractions"]),
        access_score=value["access_score"],
        passed=value["passed"],
    )
    config = PRODUCTION_WORLD_QUALIFICATION_SPEC.config
    if len(record.mouth_cells) != config.initial_population:
        raise ValueError("world record must contain every initially living mouth")
    height, width = config.grid_shape
    if any(not 0 <= x < width or not 0 <= y < height for x, y in record.mouth_cells):
        raise ValueError("mouth cell lies outside the periodic grid")
    if not math.isclose(
        record.access_score,
        max(record.mouth_capacity_fractions),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("access score does not match mouth fractions")
    expected_pass = record.access_score >= PRODUCTION_WORLD_QUALIFICATION_SPEC.access_threshold
    if record.passed != expected_pass or not isinstance(value["selected"], bool):
        raise ValueError("world pass/selection flag is invalid")
    return record, value["selected"]


def world_qualification_from_dict(value: Mapping[str, Any]) -> dict[str, object]:
    """Strictly validate and normalize the canonical R6 world artifact."""
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
        raise ValueError("unsupported world qualification schema")
    if value["protocol_revision"] != PROTOCOL_REVISION:
        raise ValueError("wrong R6 protocol revision")
    if value["canonical_backend"] != "cpu" or value["gpu_backend"] != "gpu":
        raise ValueError("R6 requires canonical CPU plus GPU replay")
    if value["passed"] is not True:
        raise ValueError("world qualification did not pass")
    if not _GIT_COMMIT.fullmatch(value["implementation_commit"]):
        raise ValueError("invalid implementation commit")
    if not isinstance(value["jax_version"], str) or not value["jax_version"]:
        raise ValueError("invalid JAX version")
    if not _SHA256.fullmatch(value["qualifier_source_sha256"]) or value["qualifier_source_sha256"] != _source_sha256():
        raise ValueError("qualifier source hash mismatch")
    if value["simulator_config_sha256"] != FROZEN_SIMULATOR_CONFIG_SHA256:
        raise ValueError("simulator config hash mismatch")

    expected_rule = {
        "access_rule_id": WORLD_ACCESS_RULE_ID,
        "access_threshold": PRODUCTION_WORLD_QUALIFICATION_SPEC.access_threshold,
        "cell_rounding": _CELL_ROUNDING,
        "performs_rollout": False,
        "required_passes_per_partition": PRODUCTION_WORLD_QUALIFICATION_SPEC.passes_per_partition,
        "selection": _SELECTION_RULE,
        "uses_genome": False,
    }
    if value["rule"] != expected_rule:
        raise ValueError("qualification rule mismatch")
    expected_ranges = [{"partition": item.partition, "start": item.start, "stop": item.stop} for item in PRODUCTION_WORLD_QUALIFICATION_SPEC.seed_ranges]
    if value["seed_ranges"] != expected_ranges:
        raise ValueError("seed ranges do not match R6")
    if not isinstance(value["traces"], list) or len(value["traces"]) != len(expected_ranges):
        raise ValueError("one trace is required per R6 seed range")

    required = PRODUCTION_WORLD_QUALIFICATION_SPEC.passes_per_partition
    cpu_records: dict[tuple[str, int], WorldAccessRecord] = {}
    expected_selected: dict[str, list[int]] = {}
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
            or not isinstance(raw_trace["inspected"], list)
            or not raw_trace["inspected"]
            or raw_trace["inspected_count"] != len(raw_trace["inspected"])
        ):
            raise ValueError("qualification trace does not match its R6 range")
        selected = []
        for offset, raw_record in enumerate(raw_trace["inspected"]):
            if not isinstance(raw_record, Mapping):
                raise ValueError("world access record must be an object")
            record, selected_flag = _parse_record(raw_record)
            if record.seed != seed_range.start + offset:
                raise ValueError("trace is not a complete ascending prefix")
            if selected_flag != record.passed:
                raise ValueError("ascending rule must select every passing prefix seed")
            if selected_flag:
                selected.append(record.seed)
            cpu_records[(seed_range.partition, record.seed)] = record
        if len(selected) != required or not raw_trace["inspected"][-1]["passed"]:
            raise ValueError("trace must stop exactly at the required passing seed")
        if raw_trace["selected_seeds"] != selected:
            raise ValueError("trace selected seeds mismatch")
        expected_selected[seed_range.partition] = selected
    if value["selected_seeds"] != expected_selected:
        raise ValueError("top-level selected seeds mismatch")

    expected_replays = [(partition, seed) for partition, seeds in expected_selected.items() for seed in seeds]
    if not isinstance(value["gpu_replays"], list) or len(value["gpu_replays"]) != len(expected_replays):
        raise ValueError("every selected seed requires one GPU replay")
    for raw, expected_key in zip(value["gpu_replays"], expected_replays, strict=True):
        if not isinstance(raw, Mapping):
            raise ValueError("GPU replay must be an object")
        _require_exact_keys(
            raw,
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
        key = (raw["partition"], raw["seed"])
        if key != expected_key:
            raise ValueError("GPU replay order/identity mismatch")
        cpu = cpu_records[key]
        cells = tuple(tuple(cell) for cell in raw["mouth_cells"])
        if raw["cells_match"] is not True or cells != cpu.mouth_cells:
            raise ValueError("GPU mouth cells mismatch")
        score = float(raw["access_score"])
        error = abs(score - cpu.access_score)
        if (
            not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            or not math.isclose(float(raw["score_absolute_error"]), error, abs_tol=1e-12)
            or error > WORLD_ACCESS_SCORE_ATOL
            or raw["passed_match"] is not ((score >= PRODUCTION_WORLD_QUALIFICATION_SPEC.access_threshold) == cpu.passed)
        ):
            raise ValueError("GPU replay disagrees with canonical CPU evidence")

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
    path: str | Path = WORLD_QUALIFICATION_PATH,
) -> str:
    path = Path(path)
    payload = canonical_world_qualification_bytes(value) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace world qualification: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def validate_world_qualification(
    path: str | Path = WORLD_QUALIFICATION_PATH,
    *,
    expected_implementation_commit: str | None = None,
) -> dict[str, object]:
    path = Path(path)
    payload = path.read_bytes()
    normalized = world_qualification_from_dict(json.loads(payload.decode("ascii")))
    if payload != canonical_world_qualification_bytes(normalized) + b"\n":
        raise ValueError("world qualification does not use canonical JSON bytes")
    if expected_implementation_commit is not None and normalized["implementation_commit"] != expected_implementation_commit:
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
    "publish_world_qualification",
    "qualify_world_ranges",
    "scan_seed_range",
    "validate_world_qualification",
    "world_qualification_from_dict",
]
