"""Freeze the R18 training panel and independent confirmation manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .build_r18_founder_panel import (
    PANEL_SEED,
    SCREENING_HORIZON,
    SCREENING_SEEDS,
)
from .episode import SimulatorConfig, simulator_config_sha256
from .founder_artifacts import (
    FounderIndex,
    FounderRecord,
    founder_index_sha256,
    load_founder_artifact,
    load_founder_index,
    make_founder_record,
    write_founder_artifact,
    write_founder_index,
)
from .protocol import (
    CONTROLLER_LAYOUT,
    R4_SCHEMA_VERSION,
    ActuatorInjuryParameters,
    EventKind,
    NullEventParameters,
    ScenarioManifest,
    WorldScenario,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)


HORIZON = 12_000
CHUNK_STEPS = 500
EVENT_STEP = 4_000
R17_TRAINING_SEED = 15_000
R18_SMOKE_SEEDS = (16_003,)
R18_DEVELOPMENT_SEEDS = (16_001, 16_004)
R18_SEALED_SEEDS = (17_000, 17_001)
INJURY_GAINS = (0.1, 0.1, 1.0, 1.0, 1.0, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_r17_sealed_as_training(source: Path, destination: Path) -> FounderIndex:
    source_index = load_founder_index(source / "index.json", verify_artifacts=True)
    sealed = tuple(
        sorted(
            (record for record in source_index.founders if record.partition == "sealed"),
            key=lambda record: record.founder_id,
        )
    )
    if len(sealed) != 8:
        raise ValueError("R18 training requires all eight R17 sealed founders")

    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for index, source_record in enumerate(sealed):
        founder_id = f"train-{index:02d}"
        artifact = f"{founder_id}.npz"
        genome = load_founder_artifact(
            source,
            source_record,
            expected_partition="sealed",
        )
        digests = write_founder_artifact(destination / artifact, genome)
        records.append(
            make_founder_record(
                founder_id=founder_id,
                partition="training",
                artifact=artifact,
                digests=digests,
                selection_rule="r18-complete-r17-sealed-panel-v1",
                selection_seed=6_007,
            )
        )
    index = FounderIndex(founders=tuple(records))
    write_founder_index(destination / "index.json", index)
    load_founder_index(destination / "index.json", verify_artifacts=True)
    return index


def _paired_worlds(
    manifest_partition: str,
    founder: FounderRecord,
    world_seed: int,
) -> tuple[WorldScenario, WorldScenario]:
    label = "sealed" if manifest_partition == "sealed_final" else manifest_partition
    pair_id = f"{label}-{founder.founder_id}-{world_seed}"
    common = {
        "world_seed": world_seed,
        "pair_id": pair_id,
        "scenario_family": "actuator_injury",
        "event_step": EVENT_STEP,
        "founder_id": founder.founder_id,
        "founder_sha256": founder.artifact_sha256,
    }
    return (
        WorldScenario(
            scenario_id=f"{pair_id}-sham",
            event_kind=EventKind.NULL,
            event_parameters=NullEventParameters(),
            **common,
        ),
        WorldScenario(
            scenario_id=f"{pair_id}-injured",
            event_kind=EventKind.ACTUATOR_INJURY,
            event_parameters=ActuatorInjuryParameters(INJURY_GAINS),
            **common,
        ),
    )


def _manifest(
    partition: str,
    founders: tuple[FounderRecord, ...],
    seeds: tuple[int, ...],
) -> ScenarioManifest:
    worlds = tuple(
        world
        for founder in founders
        for seed in seeds
        for world in _paired_worlds(partition, founder, seed)
    )
    return ScenarioManifest(
        schema_version=R4_SCHEMA_VERSION,
        partition=partition,
        controller_layout=CONTROLLER_LAYOUT,
        simulator_config_sha256=simulator_config_sha256(SimulatorConfig()),
        horizon=HORIZON,
        chunk_steps=CHUNK_STEPS,
        worlds=worlds,
    )


def _publish_manifest(path: Path, manifest: ScenarioManifest) -> dict[str, str]:
    payload = canonical_manifest_bytes(manifest) + b"\n"
    _write_once(path, payload)
    parsed = manifest_from_json_bytes(path.read_bytes())
    if parsed != manifest:
        raise RuntimeError(f"manifest round-trip failed: {path}")
    return {
        "canonical_sha256": manifest_sha256(manifest),
        "file_sha256": _sha256(path),
    }


def main() -> None:
    artifact_root = Path(__file__).with_name("r18_artifacts")
    r17_root = Path(__file__).with_name("r17_artifacts")
    training_directory = artifact_root / "training_founders"
    training_index = _copy_r17_sealed_as_training(
        r17_root / "founders",
        training_directory,
    )
    independent_index = load_founder_index(
        artifact_root / "founders" / "index.json",
        verify_artifacts=True,
    )
    partitions = {
        partition: tuple(
            sorted(
                (
                    record
                    for record in independent_index.founders
                    if record.partition == partition
                ),
                key=lambda record: record.founder_id,
            )
        )
        for partition in ("training", "development", "sealed")
    }
    if {key: len(value) for key, value in partitions.items()} != {
        "training": 4,
        "development": 4,
        "sealed": 8,
    }:
        raise ValueError("R18 independent founder bank must contain exactly 4/4/8")

    manifests = {
        "training_exposed_r17_sealed_early_injury.json": _manifest(
            "training",
            training_index.founders,
            (R17_TRAINING_SEED,),
        ),
        "training_smoke_early_injury.json": _manifest(
            "training",
            partitions["training"],
            R18_SMOKE_SEEDS,
        ),
        "development_early_injury.json": _manifest(
            "development",
            partitions["development"],
            R18_DEVELOPMENT_SEEDS,
        ),
        "sealed_early_injury.json": _manifest(
            "sealed_final",
            partitions["sealed"],
            R18_SEALED_SEEDS,
        ),
    }
    manifest_hashes = {
        name: _publish_manifest(artifact_root / "manifests" / name, manifest)
        for name, manifest in manifests.items()
    }
    freeze = {
        "event_step": EVENT_STEP,
        "horizon": HORIZON,
        "injury_gains": list(INJURY_GAINS),
        "r17_source_founder_index_canonical_sha256": founder_index_sha256(
            load_founder_index(r17_root / "founders" / "index.json")
        ),
        "r18_independent_founder_index_canonical_sha256": founder_index_sha256(
            independent_index
        ),
        "r18_independent_founder_index_file_sha256": _sha256(
            artifact_root / "founders" / "index.json"
        ),
        "r18_founder_panel_seed": PANEL_SEED,
        "r18_founder_screening_horizon": SCREENING_HORIZON,
        "r18_founder_screening_record_sha256": _sha256(
            artifact_root / "founder_screening.jsonl"
        ),
        "r18_founder_screening_seeds": list(SCREENING_SEEDS),
        "r18_training_founder_index_canonical_sha256": founder_index_sha256(
            training_index
        ),
        "r18_training_founder_index_file_sha256": _sha256(
            training_directory / "index.json"
        ),
        "r18_training_seed": R17_TRAINING_SEED,
        "manifests": manifest_hashes,
    }
    _write_once(
        artifact_root / "freeze.json",
        json.dumps(freeze, indent=2, sort_keys=True).encode("ascii") + b"\n",
    )
    for path in (
        artifact_root / "freeze.json",
        artifact_root / "manifests" / "development_early_injury.json",
        artifact_root / "manifests" / "sealed_early_injury.json",
    ):
        path.chmod(0o444)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
