"""Frozen six-action and structured-random controls for Evo² r5.

This module defines experiment artifacts only.  All returned policies still
pass through the trusted r4 six-operator registry and the shared Shinka
candidate boundary; no mutation kernel is implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from microcosmos.heredity import (
    R4_CLONE,
    R4_CONSERVATIVE_PARAMETRIC,
    R4_EXPLORATORY_PARAMETRIC,
    R4_MIXED,
    R4_NUM_OPERATORS,
    R4_STANDARD_PARAMETRIC,
    R4_STRUCTURAL,
    fixed_r4_policy,
)

from .protocol import (
    STRUCTURED_RANDOM_CANDIDATE_COUNT,
    STRUCTURED_RANDOM_SEED,
)


_FIXED_OPERATORS = (
    ("clone", R4_CLONE),
    ("conservative_parametric", R4_CONSERVATIVE_PARAMETRIC),
    ("standard_parametric", R4_STANDARD_PARAMETRIC),
    ("exploratory_parametric", R4_EXPLORATORY_PARAMETRIC),
    ("structural", R4_STRUCTURAL),
    ("mixed", R4_MIXED),
)
FIXED_OPERATOR_NAMES = tuple(name for name, _ in _FIXED_OPERATORS)
HUMAN_STRESS_NAME = "human_stress"
HUMAN_CREDIT_NAME = "human_credit"
THRESHOLDS = tuple(index / 10.0 for index in range(1, 10))

_ARGUMENTS = (
    "parent_genome_summary",
    "parent_stats",
    "population_stats",
    "operator_stats",
    "rng",
)
_FEATURE_EXPRESSIONS = (
    *(f"parent_genome_summary[{index}]" for index in range(2)),
    *(f"parent_stats[{index}]" for index in range(3)),
    *(f"population_stats[{index}]" for index in range(6)),
    *(
        f"operator_stats[{row}, {column}]"
        for row in range(3)
        for column in range(R4_NUM_OPERATORS)
    ),
)
READABLE_SCALAR_COUNT = len(_FEATURE_EXPRESSIONS)


@dataclass(frozen=True)
class CandidateSource:
    """One deterministic bounded-policy source artifact."""

    candidate_id: str
    family: str
    source: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.family:
            raise ValueError("candidate identity and family must be non-empty")
        if hashlib.sha256(self.source.encode("utf-8")).hexdigest() != self.sha256:
            raise ValueError("candidate source SHA-256 does not match")


def _source(function_body: str) -> str:
    arguments = ",\n    ".join(_ARGUMENTS)
    return (
        "import jax.numpy as jnp\n\n\n"
        "# EVOLVE-BLOCK-START\n"
        "def make_offspring(\n"
        f"    {arguments},\n"
        "):\n"
        f"{function_body}"
        "\n\n# EVOLVE-BLOCK-END\n"
    )


def _candidate(candidate_id: str, family: str, function_body: str) -> CandidateSource:
    source = _source(function_body)
    return CandidateSource(
        candidate_id=candidate_id,
        family=family,
        source=source,
        sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def fixed_operator_policy(name: str):
    """Return one trusted deterministic r4 operator by frozen label."""
    operators = dict(_FIXED_OPERATORS)
    if name not in operators:
        raise KeyError(f"unknown fixed r5 operator: {name!r}")
    operator = operators[name]
    return fixed_r4_policy(operator)


def human_candidate_sources() -> tuple[CandidateSource, CandidateSource]:
    """Return the two prospectively frozen human scheduler sources."""
    stress = _candidate(
        HUMAN_STRESS_NAME,
        "human",
        """    stress = jnp.clip(
        0.25 * (1.0 - population_stats[0])
        + 0.20 * (1.0 - population_stats[1])
        + 0.20 * jnp.maximum(0.0, -population_stats[2])
        + 0.15 * population_stats[4]
        + 0.20 * (1.0 - parent_stats[1]),
        0.0,
        1.0,
    )
    return jnp.asarray(
        jnp.stack(
            [
                -4.0 + 1.0 * stress,
                2.0 - 4.0 * stress,
                3.0 - 3.0 * stress,
                -2.0 + 6.0 * stress,
                -3.0 + 4.0 * stress,
                -1.0 + 3.0 * stress,
            ]
        ),
        dtype=jnp.float32,
    )
""",
    )
    credit = _candidate(
        HUMAN_CREDIT_NAME,
        "human",
        """    success = operator_stats[0]
    usage = operator_stats[1]
    evidence = operator_stats[2]
    stress = jnp.clip(
        0.30 * (1.0 - population_stats[0])
        + 0.20 * (1.0 - population_stats[1])
        + 0.25 * jnp.maximum(0.0, -population_stats[2])
        + 0.15 * population_stats[4]
        + 0.10 * (1.0 - population_stats[5]),
        0.0,
        1.0,
    )
    exploitation = 4.0 * evidence * (success - 0.5)
    uncertainty = 1.5 * (1.0 - evidence)
    underused = 0.5 * (1.0 - usage)
    stress_bias = stress * jnp.array(
        [-2.0, -0.5, 0.0, 2.0, 1.0, 1.5],
        dtype=jnp.float32,
    )
    return jnp.asarray(
        jnp.clip(
            exploitation + uncertainty + underused + stress_bias,
            -8.0,
            8.0,
        ),
        dtype=jnp.float32,
    )
""",
    )
    return stress, credit


def _threshold_source(
    candidate_index: int,
    feature_index: int,
    threshold: float,
    low_action: int,
    high_action: int,
) -> CandidateSource:
    below_logits = [
        8.0 if action == low_action else -8.0
        for action in range(R4_NUM_OPERATORS)
    ]
    above_logits = [
        8.0 if action == high_action else -8.0
        for action in range(R4_NUM_OPERATORS)
    ]
    expressions = ",\n                ".join(
        f"jnp.where(below, {below_logits[action]!r}, {above_logits[action]!r})"
        for action in range(R4_NUM_OPERATORS)
    )
    body = f"""    below = {_FEATURE_EXPRESSIONS[feature_index]} < {threshold!r}
    return jnp.asarray(
        jnp.stack(
            [
                {expressions}
            ]
        ),
        dtype=jnp.float32,
    )
"""
    return _candidate(
        f"structured-threshold-{candidate_index:03d}",
        "threshold",
        body,
    )


def _affine_source(
    candidate_index: int,
    weights: np.ndarray,
    bias: np.ndarray,
) -> CandidateSource:
    feature_lines = ",\n                ".join(_FEATURE_EXPRESSIONS)
    weight_rows = ",\n                ".join(
        "[" + ", ".join(repr(float(value)) for value in row) + "]"
        for row in weights
    )
    bias_values = ", ".join(repr(float(value)) for value in bias)
    body = f"""    features = jnp.stack(
        [
                {feature_lines}
        ]
    )
    weights = jnp.array(
        [
                {weight_rows}
        ],
        dtype=jnp.float32,
    )
    bias = jnp.array([{bias_values}], dtype=jnp.float32)
    return jnp.asarray(
        jnp.clip(weights @ features + bias, -8.0, 8.0),
        dtype=jnp.float32,
    )
"""
    return _candidate(
        f"structured-affine-{candidate_index:03d}",
        "affine",
        body,
    )


def structured_random_sources(
    seed: int = STRUCTURED_RANDOM_SEED,
) -> tuple[CandidateSource, ...]:
    """Generate the complete frozen 25-threshold/25-affine roster."""
    if seed != STRUCTURED_RANDOM_SEED:
        raise ValueError("r5 structured-random search requires the frozen seed")
    rng = np.random.default_rng(seed)
    candidates: list[CandidateSource] = []
    for index in range(25):
        feature_index = int(rng.integers(READABLE_SCALAR_COUNT))
        threshold = float(THRESHOLDS[int(rng.integers(len(THRESHOLDS)))])
        low_action, high_action = (
            int(value)
            for value in rng.choice(R4_NUM_OPERATORS, size=2, replace=False)
        )
        candidates.append(
            _threshold_source(
                index,
                feature_index,
                threshold,
                low_action,
                high_action,
            )
        )
    scale = 1.0 / np.sqrt(READABLE_SCALAR_COUNT)
    for index in range(25, STRUCTURED_RANDOM_CANDIDATE_COUNT):
        weights = rng.normal(0.0, scale, size=(R4_NUM_OPERATORS, READABLE_SCALAR_COUNT))
        bias = rng.normal(0.0, 0.25, size=(R4_NUM_OPERATORS,))
        candidates.append(_affine_source(index, weights, bias))
    if len(candidates) != STRUCTURED_RANDOM_CANDIDATE_COUNT:
        raise RuntimeError("structured-random roster has the wrong size")
    return tuple(candidates)


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def publish_structured_random_roster(output_directory: str | Path) -> dict[str, object]:
    """Atomically publish all sources and their index before any evaluation."""
    destination = Path(output_directory)
    staging = destination.with_name(f".{destination.name}.staging")
    if destination.exists():
        candidates = load_structured_random_roster(destination)
        return {
            "candidate_count": len(candidates),
            "index_sha256": hashlib.sha256(
                (destination / "index.json").read_bytes()
            ).hexdigest(),
        }
    if staging.exists():
        raise FileExistsError("structured-random roster staging path already exists")
    candidates = structured_random_sources()
    staging.mkdir(parents=True)
    try:
        records = []
        for candidate in candidates:
            filename = f"{candidate.candidate_id}.py"
            source_path = staging / filename
            source_path.write_text(candidate.source, encoding="utf-8")
            source_path.chmod(0o444)
            records.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "path": filename,
                    "sha256": candidate.sha256,
                }
            )
        index = {
            "candidate_count": STRUCTURED_RANDOM_CANDIDATE_COUNT,
            "grammar": {
                "affine": 25,
                "readable_scalar_count": READABLE_SCALAR_COUNT,
                "threshold": 25,
                "threshold_grid": list(THRESHOLDS),
            },
            "records": records,
            "seed": STRUCTURED_RANDOM_SEED,
        }
        payload = _canonical_json_bytes(index)
        digest = hashlib.sha256(payload).hexdigest()
        index_path = staging / "index.json"
        sidecar_path = staging / "index.sha256"
        index_path.write_bytes(payload)
        sidecar_path.write_text(
            f"{digest}  index.json\n",
            encoding="ascii",
        )
        index_path.chmod(0o444)
        sidecar_path.chmod(0o444)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"candidate_count": len(candidates), "index_sha256": digest}


def load_structured_random_roster(
    directory: str | Path,
) -> tuple[CandidateSource, ...]:
    """Authenticate the complete frozen roster and return its expected sources."""
    root = Path(directory)
    index_path = root / "index.json"
    sidecar_path = root / "index.sha256"
    try:
        payload = index_path.read_bytes()
        index = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("structured-random roster index is invalid") from error
    if not isinstance(index, dict) or payload != _canonical_json_bytes(index):
        raise ValueError("structured-random roster index is not canonical")
    digest = hashlib.sha256(payload).hexdigest()
    if sidecar_path.read_text(encoding="ascii") != f"{digest}  index.json\n":
        raise ValueError("structured-random roster sidecar does not authenticate")

    candidates = structured_random_sources()
    expected_records = [
        {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "path": f"{candidate.candidate_id}.py",
            "sha256": candidate.sha256,
        }
        for candidate in candidates
    ]
    expected_index = {
        "candidate_count": STRUCTURED_RANDOM_CANDIDATE_COUNT,
        "grammar": {
            "affine": 25,
            "readable_scalar_count": READABLE_SCALAR_COUNT,
            "threshold": 25,
            "threshold_grid": list(THRESHOLDS),
        },
        "records": expected_records,
        "seed": STRUCTURED_RANDOM_SEED,
    }
    if index != expected_index:
        raise ValueError("structured-random roster does not match the frozen grammar")
    for candidate, record in zip(candidates, expected_records, strict=True):
        source_path = (root / record["path"]).resolve()
        if source_path.parent != root.resolve() or not source_path.is_file():
            raise ValueError("structured-random source path is invalid")
        source = source_path.read_text(encoding="utf-8")
        if source != candidate.source or hashlib.sha256(source.encode()).hexdigest() != record["sha256"]:
            raise ValueError(f"structured-random source changed: {candidate.candidate_id}")
    for path in (*root.glob("*.py"), index_path, sidecar_path):
        if os.stat(path).st_mode & 0o222:
            raise ValueError(f"structured-random artifact is writable: {path.name}")
    return candidates


__all__ = [
    "CandidateSource",
    "FIXED_OPERATOR_NAMES",
    "HUMAN_CREDIT_NAME",
    "HUMAN_STRESS_NAME",
    "READABLE_SCALAR_COUNT",
    "fixed_operator_policy",
    "human_candidate_sources",
    "load_structured_random_roster",
    "publish_structured_random_roster",
    "structured_random_sources",
]
