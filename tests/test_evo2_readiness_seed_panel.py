"""Authenticated readiness seeds must flow into every downstream panel."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from experiments.evo2_ecosystem.calibrate_actuator import (  # noqa: E402
    production_spec_for_seed_panel,
)
from experiments.evo2_ecosystem.readiness_seed_panel import (  # noqa: E402
    load_readiness_seed_panel,
)


def _write_panel(path: Path) -> Path:
    value = {
        "partitions": {
            "development": [20, 21, 22, 23],
            "sealed": [30, 31, 32, 33, 34, 35],
            "training": [10, 11, 12],
        },
        "protocol_sha256": "a" * 64,
        "schema_version": 1,
        "selection_rule_id": "first_canonical_ready_world_seeds_v1",
    }
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    return path


def test_authenticated_panel_drives_calibration_worlds(tmp_path: Path) -> None:
    path = _write_panel(tmp_path / "readiness_seed_panel.json")
    panel = load_readiness_seed_panel(path)
    spec = production_spec_for_seed_panel(panel)

    assert spec.world_seeds == (10, 11, 12)
    assert spec.event_step_by_seed == ((10, 3_500), (11, 4_000), (12, 4_500))
    assert panel.development == (20, 21, 22, 23)
    assert panel.sealed == (30, 31, 32, 33, 34, 35)


def test_seed_panel_rejects_tampering_and_overlap(tmp_path: Path) -> None:
    path = _write_panel(tmp_path / "readiness_seed_panel.json")
    payload = path.read_bytes().replace(b"[30,31,32,33,34,35]", b"[20,31,32,33,34,35]")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="disjoint"):
        load_readiness_seed_panel(path)

    path = _write_panel(tmp_path / "second.json")
    path.write_bytes(path.read_bytes().replace(b"10", b"99", 1))
    with pytest.raises(ValueError, match="sidecar"):
        load_readiness_seed_panel(path)
