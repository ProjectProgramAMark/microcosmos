"""Build the prospective schema-v2 actuator-adaptation manifests.

This module composes the trusted Evo² manifest and founder-artifact APIs.  It
does not select founders, calibrate an injury, or create production files on
import.  A caller must provide a complete, artifact-verified founder index and
one gain from the preregistered calibration grid.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import shutil
from typing import Iterable, Sequence

from ..episode import SimulatorConfig, simulator_config_sha256
from ..founder_artifacts import FounderIndex, FounderRecord, load_founder_index
from ..protocol import (
    SCHEMA_VERSION,
    ActuatorInjuryParameters,
    EventKind,
    NullEventParameters,
    ScenarioManifest,
    WorldScenario,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)
from ..readiness_seed_panel import (
    ReadinessSeedPanel,
    load_readiness_seed_panel,
)


HORIZON = 8_000
CHUNK_STEPS = 500
HINGE_COUNT = 6
EVENT_STEPS = (3_500, 4_000, 4_500)
CALIBRATION_INJURY_GAINS = (0.1, 0.2, 0.4)

TRAINING_SEEDS = (1_101, 1_102, 1_103)
DEVELOPMENT_SEEDS = (2_101, 2_102, 2_103, 2_104)
SEALED_SEEDS = (3_101, 3_102, 3_103, 3_104, 3_105, 3_106)

_REQUIRED_FOUNDER_COUNTS = {
    "training": 3,
    "development": 2,
    "sealed": 3,
}
_SCENARIO_FAMILY = "actuator_injury"
_MANIFEST_FILENAMES = (
    "training_stable.json",
    "training_punctuated.json",
    "development.json",
    "sealed.json",
)
DEFAULT_MANIFEST_DIRECTORY = Path(__file__).with_name("manifests")


@dataclass(frozen=True)
class HeredityAdaptationManifests:
    """The complete prospective manifest bundle, prior to publication."""

    training_stable: ScenarioManifest
    training_punctuated: ScenarioManifest
    development: ScenarioManifest
    sealed: ScenarioManifest

    def named(self) -> tuple[tuple[str, ScenarioManifest], ...]:
        """Return manifests in their canonical publication order."""
        return tuple(zip(_MANIFEST_FILENAMES, self, strict=True))

    def __iter__(self):
        yield self.training_stable
        yield self.training_punctuated
        yield self.development
        yield self.sealed


def _selected_gain(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("selected injury gain must be a finite number")
    gain = float(value)
    if gain not in CALIBRATION_INJURY_GAINS:
        raise ValueError(
            "selected injury gain must come from the frozen calibration grid "
            f"{CALIBRATION_INJURY_GAINS}"
        )
    return gain


def _partitioned_founders(index: FounderIndex) -> dict[str, tuple[FounderRecord, ...]]:
    founders = {
        partition: tuple(
            sorted(
                (
                    founder
                    for founder in index.founders
                    if founder.partition == partition
                ),
                key=lambda founder: founder.founder_id,
            )
        )
        for partition in _REQUIRED_FOUNDER_COUNTS
    }
    actual = {partition: len(records) for partition, records in founders.items()}
    if actual != _REQUIRED_FOUNDER_COUNTS:
        raise ValueError(
            "founder index must contain exactly 3 training, 2 development, "
            f"and 3 sealed founders; got {actual}"
        )
    return founders


def _injury_vector(gain: float, injured_hinges: tuple[int, int]) -> tuple[float, ...]:
    values = [1.0] * HINGE_COUNT
    for hinge in injured_hinges:
        values[hinge] = gain
    return tuple(values)


def _event_step(seed_index: int) -> int:
    return EVENT_STEPS[seed_index % len(EVENT_STEPS)]


def _pair_id(partition: str, founder: FounderRecord, world_seed: int) -> str:
    return f"{partition}-{founder.founder_id}-{world_seed}"


def _training_world(
    founder: FounderRecord,
    world_seed: int,
    event_step: int,
) -> WorldScenario:
    pair_id = _pair_id("training", founder, world_seed)
    return WorldScenario(
        world_seed=world_seed,
        scenario_id=pair_id,
        pair_id=pair_id,
        scenario_family=_SCENARIO_FAMILY,
        event_kind=EventKind.NULL,
        event_step=event_step,
        event_parameters=NullEventParameters(),
        founder_id=founder.founder_id,
        founder_sha256=founder.artifact_sha256,
    )


def _forked_worlds(
    partition: str,
    founder: FounderRecord,
    world_seed: int,
    event_step: int,
    injury: ActuatorInjuryParameters,
) -> tuple[WorldScenario, WorldScenario]:
    pair_id = _pair_id(partition, founder, world_seed)
    common = dict(
        world_seed=world_seed,
        pair_id=pair_id,
        scenario_family=_SCENARIO_FAMILY,
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
            scenario_id=f"{pair_id}-injured",
            event_kind=EventKind.ACTUATOR_INJURY,
            event_parameters=injury,
            **common,
        ),
    )


def _manifest(
    partition: str,
    worlds: Sequence[WorldScenario],
    config_sha256: str,
) -> ScenarioManifest:
    return ScenarioManifest(
        schema_version=SCHEMA_VERSION,
        partition=partition,
        controller_layout="cppn-4x1-15n-30c-v1",
        simulator_config_sha256=config_sha256,
        horizon=HORIZON,
        chunk_steps=CHUNK_STEPS,
        worlds=tuple(worlds),
    )


def _build_manifest_bundle(
    index: FounderIndex,
    selected_injury_gain: float,
    seed_panel: ReadinessSeedPanel | None = None,
) -> HeredityAdaptationManifests:
    gain = _selected_gain(selected_injury_gain)
    founders = _partitioned_founders(index)
    config = SimulatorConfig()
    if config.nodes_per_creature - 2 != HINGE_COUNT:
        raise RuntimeError(
            "the frozen simulator configuration must expose exactly six local hinges"
        )
    config_sha256 = simulator_config_sha256(config)
    training_seeds = TRAINING_SEEDS if seed_panel is None else seed_panel.training
    development_seeds = (
        DEVELOPMENT_SEEDS if seed_panel is None else seed_panel.development
    )
    sealed_seeds = SEALED_SEEDS if seed_panel is None else seed_panel.sealed

    stable_worlds = tuple(
        _training_world(founder, seed, _event_step(seed_index))
        for founder in founders["training"]
        for seed_index, seed in enumerate(training_seeds)
    )
    head_injury = ActuatorInjuryParameters(_injury_vector(gain, (0, 1)))
    punctuated_worlds = tuple(
        replace(
            world,
            event_kind=EventKind.ACTUATOR_INJURY,
            event_parameters=head_injury,
        )
        for world in stable_worlds
    )

    middle_injury = ActuatorInjuryParameters(_injury_vector(gain, (2, 3)))
    development_worlds = tuple(
        world
        for founder in founders["development"]
        for seed_index, seed in enumerate(development_seeds)
        for world in _forked_worlds(
            "development",
            founder,
            seed,
            _event_step(seed_index),
            middle_injury,
        )
    )

    tail_injury = ActuatorInjuryParameters(_injury_vector(gain, (4, 5)))
    sealed_worlds = tuple(
        world
        for founder in founders["sealed"]
        for seed_index, seed in enumerate(sealed_seeds)
        for world in _forked_worlds(
            "sealed",
            founder,
            seed,
            _event_step(seed_index),
            tail_injury,
        )
    )

    bundle = HeredityAdaptationManifests(
        training_stable=_manifest("training", stable_worlds, config_sha256),
        training_punctuated=_manifest(
            "training",
            punctuated_worlds,
            config_sha256,
        ),
        development=_manifest(
            "development",
            development_worlds,
            config_sha256,
        ),
        sealed=_manifest("sealed_final", sealed_worlds, config_sha256),
    )
    validate_manifest_bundle(bundle, index, gain, seed_panel=seed_panel)
    return bundle


def build_manifest_bundle(
    founder_index_path: str | Path,
    selected_injury_gain: float,
    readiness_seed_panel_path: str | Path | None = None,
) -> HeredityAdaptationManifests:
    """Verify the founder bank and construct the complete manifest bundle."""
    index = load_founder_index(founder_index_path, verify_artifacts=True)
    seed_panel = (
        None
        if readiness_seed_panel_path is None
        else load_readiness_seed_panel(readiness_seed_panel_path)
    )
    return _build_manifest_bundle(index, selected_injury_gain, seed_panel)


def _world_common(world: WorldScenario) -> tuple[object, ...]:
    return (
        world.world_seed,
        world.scenario_id,
        world.pair_id,
        world.scenario_family,
        world.event_step,
        world.founder_id,
        world.founder_sha256,
    )


def _fork_common(world: WorldScenario) -> tuple[object, ...]:
    return (
        world.world_seed,
        world.pair_id,
        world.scenario_family,
        world.event_step,
        world.founder_id,
        world.founder_sha256,
    )


def _validate_common_manifest(
    manifest: ScenarioManifest,
    *,
    partition: str,
    founder_records: Sequence[FounderRecord],
    seeds: Sequence[int],
    worlds_per_pair: int,
) -> None:
    config_sha256 = simulator_config_sha256(SimulatorConfig())
    if manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("heredity-adaptation manifests must use schema version 2")
    if manifest.partition != partition:
        raise ValueError(f"manifest partition must be {partition!r}")
    if manifest.simulator_config_sha256 != config_sha256:
        raise ValueError("manifest simulator configuration hash is not frozen")
    if manifest.horizon != HORIZON or manifest.chunk_steps != CHUNK_STEPS:
        raise ValueError("manifest must use the frozen 8000/500 horizon and chunk")

    expected_founders = {
        founder.founder_id: founder.artifact_sha256 for founder in founder_records
    }
    expected_observations = Counter(
        (founder_id, seed)
        for founder_id in expected_founders
        for seed in seeds
        for _ in range(worlds_per_pair)
    )
    actual_observations = Counter(
        (world.founder_id, world.world_seed) for world in manifest.worlds
    )
    if actual_observations != expected_observations:
        raise ValueError("manifest does not contain the exact founder-by-seed panel")
    seed_steps = {
        seed: _event_step(seed_index) for seed_index, seed in enumerate(seeds)
    }
    for world in manifest.worlds:
        if world.founder_sha256 != expected_founders.get(world.founder_id):
            raise ValueError("world founder identity and artifact hash do not match")
        if world.event_step != seed_steps[world.world_seed]:
            raise ValueError("world event step does not match the frozen cyclic schedule")
        if world.scenario_family != _SCENARIO_FAMILY:
            raise ValueError("world scenario family must remain actuator_injury")


def _validate_training_pair(
    stable: ScenarioManifest,
    punctuated: ScenarioManifest,
    injury: ActuatorInjuryParameters,
) -> None:
    if len(stable.worlds) != 9 or len(punctuated.worlds) != 9:
        raise ValueError("each training arm must contain exactly nine worlds")
    for stable_world, punctuated_world in zip(
        stable.worlds,
        punctuated.worlds,
        strict=True,
    ):
        if _world_common(stable_world) != _world_common(punctuated_world):
            raise ValueError("training arms may differ only in event treatment")
        if (
            stable_world.event_kind is not EventKind.NULL
            or not isinstance(stable_world.event_parameters, NullEventParameters)
        ):
            raise ValueError("stable training worlds must use the null sham event")
        if (
            punctuated_world.event_kind is not EventKind.ACTUATOR_INJURY
            or punctuated_world.event_parameters != injury
        ):
            raise ValueError("punctuated training worlds must use the head injury")


def _validate_fork_pairs(
    manifest: ScenarioManifest,
    injury: ActuatorInjuryParameters,
    expected_pairs: int,
) -> None:
    pairs: defaultdict[str, list[WorldScenario]] = defaultdict(list)
    for world in manifest.worlds:
        pairs[world.pair_id].append(world)
    if len(pairs) != expected_pairs:
        raise ValueError("manifest has the wrong number of paired forks")
    for pair_id, worlds in pairs.items():
        if len(worlds) != 2:
            raise ValueError("every fork pair must contain exactly two worlds")
        if {world.event_kind for world in worlds} != {
            EventKind.NULL,
            EventKind.ACTUATOR_INJURY,
        }:
            raise ValueError("every fork pair must contain one null and one injury")
        sham = next(world for world in worlds if world.event_kind is EventKind.NULL)
        injured = next(
            world
            for world in worlds
            if world.event_kind is EventKind.ACTUATOR_INJURY
        )
        if _fork_common(sham) != _fork_common(injured):
            raise ValueError("fork pairs must share seed, founder, hash, and event step")
        if not isinstance(sham.event_parameters, NullEventParameters):
            raise ValueError("fork sham must use empty null-event parameters")
        if injured.event_parameters != injury:
            raise ValueError("fork injury vector does not match its partition")
        if sham.scenario_id != f"{pair_id}-sham":
            raise ValueError("fork sham scenario ID is not canonical")
        if injured.scenario_id != f"{pair_id}-injured":
            raise ValueError("fork injury scenario ID is not canonical")


def validate_manifest_bundle(
    bundle: HeredityAdaptationManifests,
    founder_index: FounderIndex,
    selected_injury_gain: float,
    *,
    seed_panel: ReadinessSeedPanel | None = None,
) -> None:
    """Fail closed unless a bundle exactly matches the prospective protocol."""
    if not isinstance(bundle, HeredityAdaptationManifests):
        raise TypeError("bundle must be a HeredityAdaptationManifests")
    if not isinstance(founder_index, FounderIndex):
        raise TypeError("founder_index must be a FounderIndex")
    gain = _selected_gain(selected_injury_gain)
    founders = _partitioned_founders(founder_index)
    training_seeds = TRAINING_SEEDS if seed_panel is None else seed_panel.training
    development_seeds = (
        DEVELOPMENT_SEEDS if seed_panel is None else seed_panel.development
    )
    sealed_seeds = SEALED_SEEDS if seed_panel is None else seed_panel.sealed

    _validate_common_manifest(
        bundle.training_stable,
        partition="training",
        founder_records=founders["training"],
        seeds=training_seeds,
        worlds_per_pair=1,
    )
    _validate_common_manifest(
        bundle.training_punctuated,
        partition="training",
        founder_records=founders["training"],
        seeds=training_seeds,
        worlds_per_pair=1,
    )
    _validate_common_manifest(
        bundle.development,
        partition="development",
        founder_records=founders["development"],
        seeds=development_seeds,
        worlds_per_pair=2,
    )
    _validate_common_manifest(
        bundle.sealed,
        partition="sealed_final",
        founder_records=founders["sealed"],
        seeds=sealed_seeds,
        worlds_per_pair=2,
    )

    head_injury = ActuatorInjuryParameters(_injury_vector(gain, (0, 1)))
    middle_injury = ActuatorInjuryParameters(_injury_vector(gain, (2, 3)))
    tail_injury = ActuatorInjuryParameters(_injury_vector(gain, (4, 5)))
    _validate_training_pair(
        bundle.training_stable,
        bundle.training_punctuated,
        head_injury,
    )
    _validate_fork_pairs(bundle.development, middle_injury, expected_pairs=8)
    _validate_fork_pairs(bundle.sealed, tail_injury, expected_pairs=18)

    seed_panels = (
        set(training_seeds),
        set(development_seeds),
        set(sealed_seeds),
    )
    if any(
        left & right
        for index, left in enumerate(seed_panels)
        for right in seed_panels[index + 1 :]
    ):
        raise RuntimeError("training, development, and sealed seeds must be disjoint")


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace immutable manifest: {path}") from error


def _verify_published_manifest(path: Path, expected: ScenarioManifest) -> None:
    payload = path.read_bytes()
    if payload != canonical_manifest_bytes(expected) + b"\n":
        raise RuntimeError(f"published manifest is not canonical: {path.name}")
    parsed = manifest_from_json_bytes(payload)
    if parsed != expected:
        raise RuntimeError(f"published manifest changed during round trip: {path.name}")
    digest = manifest_sha256(parsed)
    expected_sidecar = f"{digest}  {path.name}\n".encode("ascii")
    if path.with_suffix(".sha256").read_bytes() != expected_sidecar:
        raise RuntimeError(f"published manifest sidecar is invalid: {path.name}")


def publish_manifest_bundle(
    founder_index_path: str | Path,
    selected_injury_gain: float,
    output_directory: str | Path = DEFAULT_MANIFEST_DIRECTORY,
    readiness_seed_panel_path: str | Path | None = None,
) -> dict[str, str]:
    """Atomically publish a complete, canonical bundle exactly once."""
    destination = Path(output_directory)
    staging = destination.with_name(f".{destination.name}.staging")
    if destination.exists():
        raise FileExistsError(
            f"refusing to replace immutable manifest directory: {destination}"
        )
    if staging.exists():
        raise FileExistsError(f"manifest staging directory already exists: {staging}")

    bundle = build_manifest_bundle(
        founder_index_path,
        selected_injury_gain,
        readiness_seed_panel_path,
    )
    staging.mkdir(parents=True)
    hashes: dict[str, str] = {}
    try:
        for filename, manifest in bundle.named():
            digest = manifest_sha256(manifest)
            _write_once(
                staging / filename,
                canonical_manifest_bytes(manifest) + b"\n",
            )
            _write_once(
                (staging / filename).with_suffix(".sha256"),
                f"{digest}  {filename}\n".encode("ascii"),
            )
            hashes[filename] = digest
        for filename, manifest in bundle.named():
            _verify_published_manifest(staging / filename, manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return hashes


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point used only after founders and calibration are frozen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--founder-index", type=Path, required=True)
    parser.add_argument(
        "--selected-injury-gain",
        type=float,
        choices=CALIBRATION_INJURY_GAINS,
        required=True,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_MANIFEST_DIRECTORY,
    )
    parser.add_argument(
        "--readiness-seed-panel",
        type=Path,
        help="authenticated readiness_seed_panel.json from the founder bank",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    hashes = publish_manifest_bundle(
        args.founder_index,
        args.selected_injury_gain,
        args.output_directory,
        args.readiness_seed_panel,
    )
    print(json.dumps(hashes, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
