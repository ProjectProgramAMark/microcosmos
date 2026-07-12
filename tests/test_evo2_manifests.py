"""Frozen configuration and partition tests for Evo² manifests."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

# ``experiments`` is intentionally not part of the Microcosmos wheel.
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from experiments.evo2_ecosystem.episode import (  # noqa: E402
    SimulatorConfig,
    simulator_config_sha256,
)
from experiments.evo2_ecosystem.frozen_config import (  # noqa: E402
    CALIBRATED_ECOLOGY,
    CALIBRATED_ECOLOGY_SHA256,
)
from experiments.evo2_ecosystem.manifests import (  # noqa: E402
    VISIBLE_MANIFEST_SHA256,
    load_development_manifest,
    load_training_manifest,
)
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    EventKind,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)


SEALED_PATH = REPOSITORY / "experiments" / "evo2_sealed" / "final.json"


def _seeds(manifest):
    return {world.world_seed for world in manifest.worlds}


def _assert_shock_null_pairs(manifest):
    pairs = defaultdict(list)
    for world in manifest.worlds:
        pairs[world.pair_id].append(world)
    for worlds in pairs.values():
        assert len(worlds) == 2
        assert {world.event_kind for world in worlds} != {EventKind.NULL}
        assert sum(world.event_kind is EventKind.NULL for world in worlds) == 1
        assert len({world.world_seed for world in worlds}) == 1
        assert len({world.event_step for world in worlds}) == 1
        assert len({world.scenario_family for world in worlds}) == 1


def test_calibrated_ecology_is_the_exact_selected_default():
    selected = dict(CALIBRATED_ECOLOGY)
    encoded = json.dumps(
        selected,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == CALIBRATED_ECOLOGY_SHA256

    calibration = json.loads((REPOSITORY / "experiments" / "evo2_ecosystem" / "calibration_manifest.json").read_text())
    assert calibration["selected_candidate_id"] == "v2_c00_balanced"
    assert calibration["selected_config"] == selected
    assert calibration["selected_config_hash"] == CALIBRATED_ECOLOGY_SHA256

    config = SimulatorConfig()
    assert config.initial_resource == selected["initial_resource_fraction"]
    for name in selected.keys() - {"initial_resource_fraction"}:
        assert getattr(config, name) == selected[name]

    demo = yaml.safe_load((REPOSITORY / "experiments" / "conf" / "experiment" / "ecosystem.yaml").read_text())["experiment"]
    assert demo["max_creatures"] == config.max_creatures
    assert demo["initial_resource"] == selected["initial_resource_fraction"]
    for name in selected.keys() - {"initial_resource_fraction"}:
        assert demo[name] == selected[name]


def test_visible_manifests_are_canonical_frozen_and_correctly_paired():
    stable = load_training_manifest("stable")
    punctuated = load_training_manifest("punctuated")
    development = load_development_manifest()
    expected_config_hash = simulator_config_sha256(SimulatorConfig())

    for filename, manifest in (
        ("training_stable.json", stable),
        ("training_punctuated.json", punctuated),
        ("development.json", development),
    ):
        path = REPOSITORY / "experiments" / "evo2_ecosystem" / "manifests" / filename
        assert path.read_bytes() == canonical_manifest_bytes(manifest) + b"\n"
        assert manifest_sha256(manifest) == VISIBLE_MANIFEST_SHA256[filename]
        assert manifest.simulator_config_sha256 == expected_config_hash

    stable_pairs = {world.pair_id: world for world in stable.worlds}
    punctuated_pairs = {world.pair_id: world for world in punctuated.worlds}
    assert len(stable.worlds) == len(punctuated.worlds) == 8
    assert (
        _seeds(stable)
        == _seeds(punctuated)
        == {
            1001,
            1002,
            1003,
            1004,
            1101,
            1102,
            1103,
            1104,
        }
    )
    assert stable_pairs.keys() == punctuated_pairs.keys()
    assert all(world.event_kind is EventKind.NULL for world in stable.worlds)
    assert {world.event_kind for world in punctuated.worlds} == {
        EventKind.RESOURCE_RELOCATION,
        EventKind.RANDOM_BOTTLENECK,
    }
    for pair_id, stable_world in stable_pairs.items():
        punctuated_world = punctuated_pairs[pair_id]
        assert stable_world.world_seed == punctuated_world.world_seed
        assert stable_world.event_step == punctuated_world.event_step
        assert stable_world.scenario_family == punctuated_world.scenario_family

    _assert_shock_null_pairs(development)


def test_partition_seeds_are_disjoint_and_sealed_data_has_no_training_route():
    training = load_training_manifest("punctuated")
    development = load_development_manifest()
    sealed_raw = SEALED_PATH.read_bytes()
    sealed = manifest_from_json_bytes(sealed_raw)

    assert _seeds(training).isdisjoint(_seeds(development))
    assert _seeds(training).isdisjoint(_seeds(sealed))
    assert _seeds(development).isdisjoint(_seeds(sealed))
    assert sealed.partition == "sealed_final"
    assert sealed_raw == canonical_manifest_bytes(sealed) + b"\n"
    recorded_hash = SEALED_PATH.with_suffix(".sha256").read_text().split()[0]
    assert manifest_sha256(sealed) == recorded_hash
    _assert_shock_null_pairs(sealed)

    with pytest.raises(ValueError, match="training regime"):
        load_training_manifest("sealed")  # type: ignore[arg-type]
