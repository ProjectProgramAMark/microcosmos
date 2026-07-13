"""Authenticated world-seed panels emitted by the readiness screen.

The readiness builder is the only component that selects scientific world
seeds.  Downstream calibration, manifest construction, development, and sealed
evaluation consume this immutable artifact instead of carrying independent
hard-coded seed lists that can drift back to worlds where heredity is inert.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


READINESS_SEED_PANEL_SCHEMA_VERSION = 1
READINESS_SEED_SELECTION_RULE_ID = "first_canonical_ready_world_seeds_v1"
PARTITION_SEED_COUNTS = {
    "training": 3,
    "development": 4,
    "sealed": 6,
}


@dataclass(frozen=True)
class ReadinessSeedPanel:
    """Disjoint, readiness-qualified seeds for every experimental partition."""

    training: tuple[int, ...]
    development: tuple[int, ...]
    sealed: tuple[int, ...]
    protocol_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        partitions = {
            "training": self.training,
            "development": self.development,
            "sealed": self.sealed,
        }
        for name, expected_count in PARTITION_SEED_COUNTS.items():
            seeds = partitions[name]
            if (
                not isinstance(seeds, tuple)
                or len(seeds) != expected_count
                or len(set(seeds)) != len(seeds)
                or any(
                    not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                    for seed in seeds
                )
            ):
                raise ValueError(
                    f"readiness seed panel requires {expected_count} unique "
                    f"nonnegative {name} seeds"
                )
        all_seeds = self.training + self.development + self.sealed
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("readiness seed partitions must be disjoint")
        for name in ("protocol_sha256", "artifact_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @property
    def partitions(self) -> dict[str, tuple[int, ...]]:
        return {
            "training": self.training,
            "development": self.development,
            "sealed": self.sealed,
        }


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def load_readiness_seed_panel(path: str | Path) -> ReadinessSeedPanel:
    """Authenticate and parse an immutable readiness-seed artifact."""
    panel_path = Path(path)
    payload = panel_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected_sidecar = f"{digest}  {panel_path.name}\n".encode("ascii")
    sidecar_path = panel_path.with_suffix(".sha256")
    if sidecar_path.read_bytes() != expected_sidecar:
        raise ValueError(f"readiness seed-panel hash sidecar is invalid: {sidecar_path}")
    if not payload.endswith(b"\n"):
        raise ValueError("readiness seed panel must be newline terminated")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("readiness seed panel contains invalid JSON") from error
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value) + b"\n":
        raise ValueError("readiness seed panel must use canonical JSON bytes")
    if set(value) != {
        "partitions",
        "protocol_sha256",
        "schema_version",
        "selection_rule_id",
    }:
        raise ValueError("readiness seed panel has invalid keys")
    if value["schema_version"] != READINESS_SEED_PANEL_SCHEMA_VERSION:
        raise ValueError("readiness seed panel schema version is not supported")
    if value["selection_rule_id"] != READINESS_SEED_SELECTION_RULE_ID:
        raise ValueError("readiness seed panel selection rule is not frozen")
    partitions = value["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != set(PARTITION_SEED_COUNTS):
        raise ValueError("readiness seed panel partitions are invalid")
    return ReadinessSeedPanel(
        training=tuple(partitions["training"]),
        development=tuple(partitions["development"]),
        sealed=tuple(partitions["sealed"]),
        protocol_sha256=value["protocol_sha256"],
        artifact_sha256=digest,
    )
