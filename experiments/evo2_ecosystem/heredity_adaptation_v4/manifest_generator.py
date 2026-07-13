"""Build immutable schema-v3 paired actuation-cost manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import shutil

from ..episode import SimulatorConfig, simulator_config_sha256
from ..founder_artifacts import FounderIndex, FounderRecord, load_founder_index
from ..protocol import (
    R4_SCHEMA_VERSION,
    ActuationCostShiftParameters,
    EventKind,
    NullEventParameters,
    ScenarioManifest,
    WorldScenario,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)


HORIZON = 8_000
CHUNK_STEPS = 500
TRAINING_SEEDS = (4_101, 4_102)
DEVELOPMENT_SEEDS = (5_101, 5_102)
SEALED_SEEDS = (6_101, 6_102)
EVENT_STEPS = {
    "training": (3_500, 4_000),
    "development": (3_000, 4_500),
    "sealed": (2_500, 4_500),
}
REQUIRED = {"training": 4, "development": 4, "sealed": 8}


@dataclass(frozen=True)
class R4ManifestBundle:
    training_stable: ScenarioManifest
    training_punctuated: ScenarioManifest
    development: ScenarioManifest
    sealed: ScenarioManifest

    def named(self):
        return (
            ("training_stable.json", self.training_stable),
            ("training_punctuated.json", self.training_punctuated),
            ("development.json", self.development),
            ("sealed.json", self.sealed),
        )


def _founders(index: FounderIndex) -> dict[str, tuple[FounderRecord, ...]]:
    result = {
        partition: tuple(sorted(
            (item for item in index.founders if item.partition == partition),
            key=lambda item: item.founder_id,
        ))
        for partition in REQUIRED
    }
    actual = {key: len(value) for key, value in result.items()}
    if actual != REQUIRED:
        raise ValueError(f"r4 founder index must contain exactly {REQUIRED}; got {actual}")
    return result


def _paired_worlds(
    partition: str,
    founder: FounderRecord,
    seed: int,
    event_step: int,
    multiplier: float,
) -> tuple[WorldScenario, WorldScenario]:
    pair_id = f"{partition}-{founder.founder_id}-{seed}"
    common = dict(
        world_seed=seed,
        pair_id=pair_id,
        scenario_family="actuation_cost",
        event_step=event_step,
        founder_id=founder.founder_id,
        founder_sha256=founder.artifact_sha256,
    )
    return (
        WorldScenario(
            scenario_id=f"{pair_id}-sham",
            event_kind=EventKind.NULL,
            event_parameters=NullEventParameters(),
            **common,
        ),
        WorldScenario(
            scenario_id=f"{pair_id}-shock",
            event_kind=EventKind.ACTUATION_COST_SHIFT,
            event_parameters=ActuationCostShiftParameters(multiplier),
            **common,
        ),
    )


def _manifest(partition: str, founders, seeds, multiplier: float) -> ScenarioManifest:
    manifest_partition = "sealed_final" if partition == "sealed" else partition
    worlds = tuple(
        world
        for founder in founders
        for index, seed in enumerate(seeds)
        for world in _paired_worlds(
            partition, founder, seed, EVENT_STEPS[partition][index], multiplier
        )
    )
    return ScenarioManifest(
        schema_version=R4_SCHEMA_VERSION,
        partition=manifest_partition,
        controller_layout="cppn-4x1-15n-30c-v1",
        simulator_config_sha256=simulator_config_sha256(SimulatorConfig()),
        horizon=HORIZON,
        chunk_steps=CHUNK_STEPS,
        worlds=worlds,
    )


def build_manifest_bundle(founder_index_path: str | Path, multiplier: float) -> R4ManifestBundle:
    index = load_founder_index(founder_index_path, verify_artifacts=True)
    founders = _founders(index)
    training = _manifest("training", founders["training"], TRAINING_SEEDS, multiplier)
    bundle = R4ManifestBundle(
        training_stable=training,
        training_punctuated=training,
        development=_manifest("development", founders["development"], DEVELOPMENT_SEEDS, multiplier),
        sealed=_manifest("sealed", founders["sealed"], SEALED_SEEDS, multiplier),
    )
    validate_manifest_bundle(bundle, multiplier)
    return bundle


def validate_manifest_bundle(bundle: R4ManifestBundle, multiplier: float) -> None:
    partitions = ("training", "development", "sealed")
    expected_pairs = (8, 8, 16)
    if bundle.training_stable != bundle.training_punctuated:
        raise ValueError("stable and punctuated r4 arms must execute the same paired manifest")
    for partition, manifest, pair_count in zip(partitions, (bundle.training_stable, bundle.development, bundle.sealed), expected_pairs, strict=True):
        if manifest.schema_version != R4_SCHEMA_VERSION:
            raise ValueError("r4 manifests require schema version 3")
        pairs: defaultdict[str, list[WorldScenario]] = defaultdict(list)
        for world in manifest.worlds:
            pairs[world.pair_id].append(world)
        if len(pairs) != pair_count:
            raise ValueError(f"{partition} manifest must contain {pair_count} pairs")
        for worlds in pairs.values():
            if len(worlds) != 2 or {world.event_kind for world in worlds} != {EventKind.NULL, EventKind.ACTUATION_COST_SHIFT}:
                raise ValueError("each r4 pair requires exactly one sham and one cost shock")
            shock = next(world for world in worlds if world.event_kind is EventKind.ACTUATION_COST_SHIFT)
            if shock.event_parameters != ActuationCostShiftParameters(multiplier):
                raise ValueError("cost multiplier differs from the qualified value")
            first, second = worlds
            if (first.world_seed, first.event_step, first.founder_id, first.founder_sha256) != (second.world_seed, second.event_step, second.founder_id, second.founder_sha256):
                raise ValueError("paired worlds must share seed, timing, and founder")


def publish_manifest_bundle(
    founder_index_path: str | Path,
    multiplier: float,
    output_directory: str | Path,
) -> dict[str, str]:
    destination = Path(output_directory)
    staging = destination.with_name(f".{destination.name}.staging")
    if destination.exists() or staging.exists():
        raise FileExistsError("refusing to replace an r4 manifest directory")
    bundle = build_manifest_bundle(founder_index_path, multiplier)
    staging.mkdir(parents=True)
    hashes = {}
    try:
        for filename, manifest in bundle.named():
            payload = canonical_manifest_bytes(manifest) + b"\n"
            (staging / filename).write_bytes(payload)
            digest = manifest_sha256(manifest)
            (staging / filename.replace(".json", ".sha256")).write_text(
                f"{digest}  {filename}\n", encoding="ascii"
            )
            if manifest_from_json_bytes(payload) != manifest:
                raise RuntimeError("manifest changed during canonical round trip")
            hashes[filename] = digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--founder-index", type=Path, required=True)
    parser.add_argument("--multiplier", type=float, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    hashes = publish_manifest_bundle(args.founder_index, args.multiplier, args.output_directory)
    print(hashes)


if __name__ == "__main__":
    main()
