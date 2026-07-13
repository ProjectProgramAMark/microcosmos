"""Prospective manifest construction for the actuator-adaptation experiment."""

from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: E402
    FounderIndex,
    canonical_founder_index_bytes,
    load_founder_index,
    make_founder_record,
    write_founder_artifact,
    write_founder_index,
)
from experiments.evo2_ecosystem.heredity_adaptation.manifest_generator import (  # noqa: E402
    CHUNK_STEPS,
    DEVELOPMENT_SEEDS,
    EVENT_STEPS,
    HORIZON,
    SEALED_SEEDS,
    TRAINING_SEEDS,
    build_manifest_bundle,
    publish_manifest_bundle,
    validate_manifest_bundle,
)
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    ActuatorInjuryParameters,
    EventKind,
    NullEventParameters,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)
from microcosmos.cppn import (  # noqa: E402
    CONNECTION_WEIGHT,
    CPPNGenome,
    canonical_cppn_genome,
)


def _founder(weight_delta: float) -> CPPNGenome:
    genome = canonical_cppn_genome()
    return CPPNGenome(
        node_genes=genome.node_genes,
        connection_genes=genome.connection_genes.at[0, CONNECTION_WEIGHT].add(
            weight_delta
        ),
    )


def _founder_bank(
    root: Path,
    counts: tuple[int, int, int] = (3, 2, 3),
) -> Path:
    records = []
    offset = 0
    for partition, count, prefix in zip(
        ("training", "development", "sealed"),
        counts,
        ("train", "dev", "sealed"),
        strict=True,
    ):
        for item in range(count):
            founder_id = f"{prefix}-{item:02d}"
            artifact = f"{founder_id}.npz"
            digests = write_founder_artifact(
                root / artifact,
                _founder(0.01 * offset),
            )
            records.append(
                make_founder_record(
                    founder_id=founder_id,
                    partition=partition,
                    artifact=artifact,
                    digests=digests,
                    selection_rule="uninjured_viability_only",
                    selection_seed=100 + offset,
                )
            )
            offset += 1
    index_path = root / "index.json"
    write_founder_index(index_path, FounderIndex(founders=tuple(records)))
    return index_path


def _assert_forks(manifest, expected_vector):
    pairs = defaultdict(list)
    for world in manifest.worlds:
        pairs[world.pair_id].append(world)
    for pair_id, worlds in pairs.items():
        assert len(worlds) == 2
        assert {world.event_kind for world in worlds} == {
            EventKind.NULL,
            EventKind.ACTUATOR_INJURY,
        }
        sham = next(world for world in worlds if world.event_kind is EventKind.NULL)
        injured = next(
            world
            for world in worlds
            if world.event_kind is EventKind.ACTUATOR_INJURY
        )
        assert isinstance(sham.event_parameters, NullEventParameters)
        assert injured.event_parameters == ActuatorInjuryParameters(expected_vector)
        assert sham.scenario_id == f"{pair_id}-sham"
        assert injured.scenario_id == f"{pair_id}-injured"
        assert (
            sham.world_seed,
            sham.pair_id,
            sham.scenario_family,
            sham.event_step,
            sham.founder_id,
            sham.founder_sha256,
        ) == (
            injured.world_seed,
            injured.pair_id,
            injured.scenario_family,
            injured.event_step,
            injured.founder_id,
            injured.founder_sha256,
        )


def _readiness_panel(path: Path) -> Path:
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
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
    return path


def test_bundle_can_use_only_readiness_qualified_worlds(tmp_path):
    index_path = _founder_bank(tmp_path / "founders")
    panel_path = _readiness_panel(tmp_path / "readiness_seed_panel.json")
    bundle = build_manifest_bundle(index_path, 0.2, panel_path)

    assert {world.world_seed for world in bundle.training_stable.worlds} == {
        10,
        11,
        12,
    }
    assert {world.world_seed for world in bundle.development.worlds} == {
        20,
        21,
        22,
        23,
    }
    assert {world.world_seed for world in bundle.sealed.worlds} == {
        30,
        31,
        32,
        33,
        34,
        35,
    }
def test_bundle_has_exact_partitions_seed_panels_injuries_and_forks(tmp_path):
    index_path = _founder_bank(tmp_path / "founders")
    index = load_founder_index(index_path)
    bundle = build_manifest_bundle(index_path, 0.2)
    validate_manifest_bundle(bundle, index, 0.2)

    assert bundle.training_stable.schema_version == 2
    assert bundle.training_punctuated.schema_version == 2
    assert bundle.development.schema_version == 2
    assert bundle.sealed.schema_version == 2
    assert bundle.training_stable.partition == "training"
    assert bundle.training_punctuated.partition == "training"
    assert bundle.development.partition == "development"
    assert bundle.sealed.partition == "sealed_final"
    assert all(
        manifest.horizon == HORIZON and manifest.chunk_steps == CHUNK_STEPS
        for manifest in bundle
    )
    assert len(bundle.training_stable.worlds) == 3 * 3
    assert len(bundle.training_punctuated.worlds) == 3 * 3
    assert len(bundle.development.worlds) == 2 * 4 * 2
    assert len(bundle.sealed.worlds) == 3 * 6 * 2

    for stable, punctuated in zip(
        bundle.training_stable.worlds,
        bundle.training_punctuated.worlds,
        strict=True,
    ):
        assert stable.event_kind is EventKind.NULL
        assert isinstance(stable.event_parameters, NullEventParameters)
        assert punctuated.event_kind is EventKind.ACTUATOR_INJURY
        assert punctuated.event_parameters == ActuatorInjuryParameters(
            (0.2, 0.2, 1.0, 1.0, 1.0, 1.0)
        )
        assert replace(
            punctuated,
            event_kind=stable.event_kind,
            event_parameters=stable.event_parameters,
        ) == stable

    _assert_forks(
        bundle.development,
        (1.0, 1.0, 0.2, 0.2, 1.0, 1.0),
    )
    _assert_forks(
        bundle.sealed,
        (1.0, 1.0, 1.0, 1.0, 0.2, 0.2),
    )

    training_seeds = {world.world_seed for world in bundle.training_stable.worlds}
    development_seeds = {world.world_seed for world in bundle.development.worlds}
    sealed_seeds = {world.world_seed for world in bundle.sealed.worlds}
    assert training_seeds == set(TRAINING_SEEDS)
    assert development_seeds == set(DEVELOPMENT_SEEDS)
    assert sealed_seeds == set(SEALED_SEEDS)
    assert training_seeds.isdisjoint(development_seeds | sealed_seeds)
    assert development_seeds.isdisjoint(sealed_seeds)
    for manifest, seeds in (
        (bundle.training_stable, TRAINING_SEEDS),
        (bundle.development, DEVELOPMENT_SEEDS),
        (bundle.sealed, SEALED_SEEDS),
    ):
        expected_steps = {
            seed: EVENT_STEPS[index % len(EVENT_STEPS)]
            for index, seed in enumerate(seeds)
        }
        assert all(
            world.event_step == expected_steps[world.world_seed]
            for world in manifest.worlds
        )


def test_publish_is_atomic_canonical_hashed_and_write_once(tmp_path):
    index_path = _founder_bank(tmp_path / "founders")
    output = tmp_path / "heredity_adaptation" / "manifests"
    hashes = publish_manifest_bundle(index_path, 0.4, output)

    assert set(hashes) == {
        "training_stable.json",
        "training_punctuated.json",
        "development.json",
        "sealed.json",
    }
    for filename, expected_hash in hashes.items():
        path = output / filename
        manifest = manifest_from_json_bytes(path.read_bytes())
        assert path.read_bytes() == canonical_manifest_bytes(manifest) + b"\n"
        assert manifest_sha256(manifest) == expected_hash
        assert hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest() == expected_hash
        assert path.with_suffix(".sha256").read_text() == (
            f"{expected_hash}  {filename}\n"
        )

    with pytest.raises(FileExistsError, match="refusing to replace"):
        publish_manifest_bundle(index_path, 0.4, output)


def test_generator_requires_exact_verified_founders_and_frozen_gain(tmp_path):
    incomplete_index = _founder_bank(tmp_path / "incomplete", counts=(3, 2, 2))
    with pytest.raises(ValueError, match="exactly 3 training, 2 development"):
        build_manifest_bundle(incomplete_index, 0.2)

    valid_index = _founder_bank(tmp_path / "valid")
    with pytest.raises(ValueError, match="frozen calibration grid"):
        build_manifest_bundle(valid_index, 0.3)
    with pytest.raises(ValueError, match="finite number"):
        build_manifest_bundle(valid_index, True)

    index = load_founder_index(valid_index, verify_artifacts=False)
    artifact = valid_index.parent / index.founders[0].artifact
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_manifest_bundle(valid_index, 0.2)


def test_validator_rejects_training_or_fork_identity_drift(tmp_path):
    index_path = _founder_bank(tmp_path / "founders")
    index = load_founder_index(index_path)
    bundle = build_manifest_bundle(index_path, 0.2)

    punctuated_worlds = list(bundle.training_punctuated.worlds)
    punctuated_worlds[0] = replace(
        punctuated_worlds[0],
        event_step=punctuated_worlds[0].event_step + CHUNK_STEPS,
    )
    broken_training = replace(
        bundle,
        training_punctuated=replace(
            bundle.training_punctuated,
            worlds=tuple(punctuated_worlds),
        ),
    )
    with pytest.raises(ValueError, match="event step|differ only"):
        validate_manifest_bundle(broken_training, index, 0.2)

    development_worlds = list(bundle.development.worlds)
    development_worlds[1] = replace(
        development_worlds[1],
        founder_sha256="0" * 64,
    )
    broken_fork = replace(
        bundle,
        development=replace(
            bundle.development,
            worlds=tuple(development_worlds),
        ),
    )
    with pytest.raises(ValueError, match="identity and artifact hash|share seed"):
        validate_manifest_bundle(broken_fork, index, 0.2)


def test_generator_preserves_the_pilot_manifests():
    legacy = REPOSITORY / "experiments" / "evo2_ecosystem" / "manifests"
    expected = {
        "training_stable.json": "716f306f684646cfa2ea62c36c3cef5735e1d682425de3eeee5ca1018ae1b021",
        "training_punctuated.json": "535b759c613bb91252934d3b91645b4017add1379b78b7164e984fc58d3e8fbf",
        "development.json": "563c614250054089bbdbbdae06f78adae27da1d13b29be923c3d70fa99602ace",
    }
    for filename, expected_hash in expected.items():
        manifest = manifest_from_json_bytes((legacy / filename).read_bytes())
        assert manifest.schema_version == 1
        assert manifest_sha256(manifest) == expected_hash


def test_founder_index_fixture_is_canonical(tmp_path):
    index_path = _founder_bank(tmp_path / "founders")
    index = load_founder_index(index_path)
    assert index_path.read_bytes() == canonical_founder_index_bytes(index) + b"\n"
