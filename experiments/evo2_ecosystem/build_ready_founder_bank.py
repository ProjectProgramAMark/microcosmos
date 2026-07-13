"""Build a clonal founder bank from pre-injury-ready worlds.

The failed v1 founder screen required clone-only survival through the complete
8,000-step experiment.  That was stronger than the causal experiment needs.
This separately versioned builder first qualifies ecological seeds with the
canonical CPPN in an uninjured world, then asks candidate founders only to be
alive, reproductive, and resource-productive at the latest possible injury
boundary (step 4,500).

No actuator injury is constructed or evaluated here.  Production execution is
full-scale, fluid-enabled, GPU-only, and forbids injected runners.  The v1
builder and its completed negative ledger remain unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
from typing import Callable

import jax

from microcosmos.cppn import FOUNDER_PANEL_SEED, CPPNGenome, canonical_cppn_genome

from .build_founder_bank import (
    CandidateRange,
    CandidateRunner,
    CandidateScreenResult,
    ScreeningGates,
    ScreeningSpec,
    SeedScreenResult,
    _gate_result,
    _screen_result_dict,
    candidate_pool,
    make_screen_runner,
)
from .episode import SimulatorConfig, simulator_config_sha256
from .founder_artifacts import (
    FounderIndex,
    founder_index_sha256,
    load_founder_index,
    make_founder_record,
    write_founder_artifact,
    write_founder_index,
)


SCREENING_SCHEMA_VERSION = 2
SEED_PANEL_SCHEMA_VERSION = 1
SELECTION_RULE_ID = "first_preinjury_ready_founders_v2"
SEED_SELECTION_RULE_ID = "first_canonical_ready_world_seeds_v1"
READINESS_STEP = 4_500
CHUNK_STEPS = 500
PARTITION_ORDER = ("training", "development", "sealed")
PARTITION_SEED_COUNTS = {"training": 3, "development": 4, "sealed": 6}
PRODUCTION_SEED_CANDIDATES = tuple(range(5_000, 5_064))
SMOKE_SEED_CANDIDATES = tuple(range(16))


@dataclass(frozen=True)
class ReadyFounderSpec:
    """Frozen seed-qualification and founder-readiness contract."""

    mode: str
    screening: ScreeningSpec
    seed_candidates: tuple[int, ...]
    partition_seed_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if self.mode not in {"production", "smoke"}:
            raise ValueError("mode must be production or smoke")
        if self.screening.mode != self.mode:
            raise ValueError("nested screening mode must match")
        if self.mode == "production" and self.screening.horizon != READINESS_STEP:
            raise ValueError("founders must be screened at step 4500")
        if self.screening.chunk_steps != CHUNK_STEPS and self.mode == "production":
            raise ValueError("production readiness chunks must be 500 steps")
        if (
            not self.seed_candidates
            or len(set(self.seed_candidates)) != len(self.seed_candidates)
            or any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                for seed in self.seed_candidates
            )
        ):
            raise ValueError("seed candidates must be unique nonnegative integers")
        counts = dict(self.partition_seed_counts)
        if tuple(counts) != PARTITION_ORDER or counts != PARTITION_SEED_COUNTS:
            raise ValueError("seed partitions must be exact 3/4/6 panels")
        if sum(counts.values()) > len(self.seed_candidates):
            raise ValueError("seed candidate pool is too small")

    @property
    def required_seed_count(self) -> int:
        return sum(dict(self.partition_seed_counts).values())


_PRODUCTION_SCREEN = ScreeningSpec(
    mode="production",
    config=SimulatorConfig(),
    seeds=(0,),  # Replaced only by the qualified partition panel.
    horizon=READINESS_STEP,
    chunk_steps=CHUNK_STEPS,
    candidate_ranges=(
        CandidateRange("training", 0, 16, 3),
        CandidateRange("development", 16, 32, 2),
        CandidateRange("sealed", 32, 48, 3),
    ),
    gates=ScreeningGates(
        minimum_final_alive=2,
        minimum_births=2,
        minimum_generation=1,
        minimum_productivity=0.01,
    ),
)

PRODUCTION_SPEC = ReadyFounderSpec(
    mode="production",
    screening=_PRODUCTION_SCREEN,
    seed_candidates=PRODUCTION_SEED_CANDIDATES,
    partition_seed_counts=tuple(PARTITION_SEED_COUNTS.items()),
)

_SMOKE_SCREEN = ScreeningSpec(
    mode="smoke",
    config=replace(
        SimulatorConfig(),
        max_creatures=4,
        initial_population=2,
        nodes_per_creature=4,
        grid_shape=(16, 16),
        fluid_enabled=False,
        resource_patch_center=(8.0, 8.0),
        resource_patch_radius=6.0,
        initial_energy=10.0,
        reproduction_threshold=100.0,
        basal_metabolism=0.0,
        actuation_power_coefficient=0.0,
        placement_candidates=4,
        position_margin=0.5,
    ),
    seeds=(0,),
    horizon=4,
    chunk_steps=2,
    candidate_ranges=(
        CandidateRange("training", 0, 4, 3),
        CandidateRange("development", 4, 8, 2),
        CandidateRange("sealed", 8, 12, 3),
    ),
    gates=ScreeningGates(
        minimum_final_alive=1,
        minimum_births=0,
        minimum_generation=0,
        minimum_productivity=0.0,
    ),
)

SMOKE_SPEC = ReadyFounderSpec(
    mode="smoke",
    screening=_SMOKE_SCREEN,
    seed_candidates=SMOKE_SEED_CANDIDATES,
    partition_seed_counts=tuple(PARTITION_SEED_COUNTS.items()),
)


SeedRunner = Callable[[ScreeningSpec], CandidateScreenResult]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_sha256(value: object) -> str:
    return _sha256(inspect.getsource(value).encode("utf-8"))


class _AppendOnlyRecord:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
                0o644,
            )
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to replace readiness record: {path}"
            ) from error
        self._sequence = 0

    def append(self, event: dict[str, object]) -> None:
        payload = _canonical_json_bytes(
            {
                "schema_version": SCREENING_SCHEMA_VERSION,
                "sequence": self._sequence,
                **event,
            }
        ) + b"\n"
        written = os.write(self._descriptor, payload)
        if written != len(payload):
            raise OSError("short write to readiness record")
        os.fsync(self._descriptor)
        self._sequence += 1

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _screening_spec(spec: ReadyFounderSpec, seeds: tuple[int, ...]) -> ScreeningSpec:
    return replace(spec.screening, seeds=seeds)


def _seed_passed(result: SeedScreenResult, gates: ScreeningGates) -> bool:
    return bool(
        result.finite
        and result.integrity_valid
        and result.final_alive >= gates.minimum_final_alive
        and result.birth_count >= gates.minimum_births
        and result.maximum_generation >= gates.minimum_generation
        and result.productivity >= gates.minimum_productivity
    )


def _actual_seed_runner(spec: ReadyFounderSpec) -> SeedRunner:
    candidate_spec = _screening_spec(spec, spec.seed_candidates)
    runner = make_screen_runner(candidate_spec)
    founder = canonical_cppn_genome()
    return lambda _: runner(founder, candidate_spec)


def qualify_seeds(
    spec: ReadyFounderSpec,
    runner: SeedRunner | None = None,
) -> tuple[dict[str, tuple[int, ...]], CandidateScreenResult]:
    """Select the first 13 uninjured canonical-ready seeds."""
    candidate_spec = _screening_spec(spec, spec.seed_candidates)
    result = (runner or _actual_seed_runner(spec))(candidate_spec)
    if not isinstance(result, CandidateScreenResult):
        raise TypeError("seed runner must return CandidateScreenResult")
    if tuple(item.seed for item in result.seed_results) != spec.seed_candidates:
        raise ValueError("seed runner did not evaluate the frozen candidate pool")
    passing = tuple(
        item.seed
        for item in result.seed_results
        if _seed_passed(item, spec.screening.gates)
    )
    if len(passing) < spec.required_seed_count:
        return {partition: () for partition in PARTITION_ORDER}, result
    assignments: dict[str, tuple[int, ...]] = {}
    offset = 0
    for partition, count in spec.partition_seed_counts:
        assignments[partition] = passing[offset : offset + count]
        offset += count
    return assignments, result


def _selection_rule(spec: ReadyFounderSpec) -> dict[str, object]:
    return {
        "founder_rule_id": SELECTION_RULE_ID,
        "founder_rule": (
            "Within ranges assigned before screening, select the first required "
            "founders passing every seed in their partition at step 4500."
        ),
        "candidate_ranges": [asdict(item) for item in spec.screening.candidate_ranges],
        "seed_rule_id": SEED_SELECTION_RULE_ID,
        "seed_rule": (
            "Evaluate the canonical CPPN uninjured and assign the first 3/4/6 "
            "passing seeds to training/development/sealed in that order."
        ),
        "seed_candidates": list(spec.seed_candidates),
        "injury_screening": False,
    }


def _protocol(
    spec: ReadyFounderSpec,
    preregistration_path: Path | None,
) -> dict[str, object]:
    protocol = {
        "builder_source_sha256": _sha256(Path(__file__).read_bytes()),
        "canonical_founder_source_sha256": _source_sha256(canonical_cppn_genome),
        "candidate_generator_source_sha256": _source_sha256(candidate_pool),
        "founder_panel_seed": FOUNDER_PANEL_SEED,
        "mode": spec.mode,
        "readiness_step": spec.screening.horizon,
        "chunk_steps": spec.screening.chunk_steps,
        "gates": asdict(spec.screening.gates),
        "selection_rule": _selection_rule(spec),
        "selection_rule_sha256": _sha256(_canonical_json_bytes(_selection_rule(spec))),
        "simulator_config": asdict(spec.screening.config),
        "simulator_config_sha256": simulator_config_sha256(spec.screening.config),
    }
    if preregistration_path is not None:
        protocol["preregistration_path"] = str(preregistration_path)
        protocol["preregistration_sha256"] = _sha256(
            preregistration_path.read_bytes()
        )
    else:
        protocol["preregistration_path"] = None
        protocol["preregistration_sha256"] = None
    return protocol


def validate_execution_contract(spec: ReadyFounderSpec, *, backend: str) -> None:
    if spec.mode == "production":
        if spec != PRODUCTION_SPEC:
            raise RuntimeError("production readiness screening requires frozen spec")
        if backend != "gpu":
            raise RuntimeError("production readiness screening requires GPU")
        if not spec.screening.config.fluid_enabled:
            raise RuntimeError("production readiness screening requires fluid physics")
    else:
        if spec != SMOKE_SPEC or backend != "cpu":
            raise RuntimeError("smoke readiness screening requires frozen CPU spec")


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to replace immutable artifact: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _publish_bank(
    bank_directory: Path,
    selected: dict[str, list[tuple[int, CPPNGenome]]],
    seed_assignments: dict[str, tuple[int, ...]],
    protocol_sha256: str,
) -> tuple[FounderIndex, str, str]:
    staging = bank_directory.with_name(f".{bank_directory.name}.staging")
    if bank_directory.exists() or staging.exists():
        raise FileExistsError("refusing to replace founder bank or staging directory")
    staging.mkdir(parents=True)
    records = []
    try:
        for partition in PARTITION_ORDER:
            prefix = {"training": "train", "development": "dev", "sealed": "sealed"}[
                partition
            ]
            for selected_index, (_candidate_index, genome) in enumerate(
                selected[partition]
            ):
                founder_id = f"{prefix}-{selected_index:02d}"
                artifact = f"{founder_id}.npz"
                digests = write_founder_artifact(staging / artifact, genome)
                records.append(
                    make_founder_record(
                        founder_id=founder_id,
                        partition=partition,
                        artifact=artifact,
                        digests=digests,
                        selection_rule=SELECTION_RULE_ID,
                        selection_seed=FOUNDER_PANEL_SEED,
                    )
                )
        index = FounderIndex(founders=tuple(records))
        index_sha256 = write_founder_index(staging / "index.json", index)
        seed_panel = {
            "schema_version": SEED_PANEL_SCHEMA_VERSION,
            "protocol_sha256": protocol_sha256,
            "selection_rule_id": SEED_SELECTION_RULE_ID,
            "partitions": {
                partition: list(seed_assignments[partition])
                for partition in PARTITION_ORDER
            },
        }
        seed_payload = _canonical_json_bytes(seed_panel) + b"\n"
        seed_sha256 = _sha256(seed_payload)
        _write_once(staging / "readiness_seed_panel.json", seed_payload)
        _write_once(
            staging / "readiness_seed_panel.sha256",
            f"{seed_sha256}  readiness_seed_panel.json\n".encode("ascii"),
        )
        load_founder_index(staging / "index.json", verify_artifacts=True)
        if founder_index_sha256(index) != index_sha256:
            raise RuntimeError("founder index hash changed during publication")
        bank_directory.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(bank_directory)
        return index, index_sha256, seed_sha256
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_ready_founder_bank_builder(
    *,
    smoke: bool,
    bank_directory: Path,
    screening_record_path: Path,
    preregistration_path: Path | None = None,
    seed_runner: SeedRunner | None = None,
    founder_runner: CandidateRunner | None = None,
) -> dict[str, object]:
    """Qualify worlds, screen founders at step 4500, and publish atomically."""
    spec = SMOKE_SPEC if smoke else PRODUCTION_SPEC
    backend = jax.default_backend()
    if not smoke and (seed_runner is not None or founder_runner is not None):
        raise RuntimeError("production readiness screening forbids injected runners")
    validate_execution_contract(spec, backend=backend)
    if not smoke:
        if preregistration_path is None:
            raise RuntimeError("production readiness screening requires preregistration")
        preregistration = preregistration_path.read_text(encoding="utf-8")
        if "Status: **FROZEN" not in preregistration:
            raise RuntimeError("production readiness preregistration is not frozen")
    if bank_directory.exists():
        raise FileExistsError(f"refusing to replace founder bank: {bank_directory}")

    protocol = _protocol(spec, preregistration_path)
    protocol_sha256 = _sha256(_canonical_json_bytes(protocol))
    selected: dict[str, list[tuple[int, CPPNGenome]]] = {
        partition: [] for partition in PARTITION_ORDER
    }

    with _AppendOnlyRecord(screening_record_path) as record:
        record.append(
            {
                "event": "readiness_screen_started",
                "backend": backend,
                "protocol": protocol,
                "protocol_sha256": protocol_sha256,
            }
        )
        assignments, seed_result = qualify_seeds(spec, seed_runner)
        for result in seed_result.seed_results:
            record.append(
                {
                    "event": "world_seed_qualified",
                    "protocol_sha256": protocol_sha256,
                    "passed": _seed_passed(result, spec.screening.gates),
                    "result": asdict(result),
                }
            )
        seeds_complete = all(assignments.values())
        record.append(
            {
                "event": "world_seed_selection_completed",
                "complete": seeds_complete,
                "assignments": {key: list(value) for key, value in assignments.items()},
                "protocol_sha256": protocol_sha256,
            }
        )
        if not seeds_complete:
            record.append(
                {
                    "event": "readiness_screen_finished",
                    "bank_published": False,
                    "status": "insufficient_ready_world_seeds",
                    "protocol_sha256": protocol_sha256,
                }
            )
            return {
                "bank_published": False,
                "status": "insufficient_ready_world_seeds",
                "protocol_sha256": protocol_sha256,
            }

        pool = candidate_pool(spec.screening)
        for candidate_range in spec.screening.candidate_ranges:
            partition_spec = _screening_spec(
                spec,
                assignments[candidate_range.partition],
            )
            screen = founder_runner or make_screen_runner(partition_spec)
            for candidate_index in range(candidate_range.start, candidate_range.stop):
                result = screen(pool[candidate_index], partition_spec)
                if not isinstance(result, CandidateScreenResult):
                    raise TypeError("founder runner must return CandidateScreenResult")
                passed, reasons, summary = _gate_result(result, partition_spec)
                record.append(
                    {
                        "event": "founder_candidate_screened",
                        "candidate_index": candidate_index,
                        "partition": candidate_range.partition,
                        "protocol_sha256": protocol_sha256,
                        "gate": {
                            "passed": passed,
                            "reasons": list(reasons),
                            "summary": summary,
                        },
                        "result": _screen_result_dict(result),
                    }
                )
                if passed and len(selected[candidate_range.partition]) < candidate_range.required:
                    selected[candidate_range.partition].append(
                        (candidate_index, pool[candidate_index])
                    )
            record.append(
                {
                    "event": "founder_partition_completed",
                    "partition": candidate_range.partition,
                    "required": candidate_range.required,
                    "selected_candidate_indices": [
                        index for index, _ in selected[candidate_range.partition]
                    ],
                    "protocol_sha256": protocol_sha256,
                }
            )

        complete = all(
            len(selected[item.partition]) == item.required
            for item in spec.screening.candidate_ranges
        )
        selection = {
            partition: [index for index, _ in selected[partition]]
            for partition in PARTITION_ORDER
        }
        if not complete:
            record.append(
                {
                    "event": "readiness_screen_finished",
                    "bank_published": False,
                    "selected_candidate_indices": selection,
                    "status": "insufficient_ready_founders",
                    "protocol_sha256": protocol_sha256,
                }
            )
            return {
                "bank_published": False,
                "selected_candidate_indices": selection,
                "status": "insufficient_ready_founders",
                "protocol_sha256": protocol_sha256,
            }

        index, index_sha256, seed_panel_sha256 = _publish_bank(
            bank_directory,
            selected,
            assignments,
            protocol_sha256,
        )
        record.append(
            {
                "event": "readiness_screen_finished",
                "bank_published": True,
                "founder_index_sha256": index_sha256,
                "founder_artifact_sha256": {
                    item.founder_id: item.artifact_sha256 for item in index.founders
                },
                "readiness_seed_panel_sha256": seed_panel_sha256,
                "selected_candidate_indices": selection,
                "status": "published",
                "protocol_sha256": protocol_sha256,
            }
        )
        return {
            "bank_directory": str(bank_directory),
            "bank_published": True,
            "founder_index_sha256": index_sha256,
            "readiness_seed_panel_sha256": seed_panel_sha256,
            "selected_candidate_indices": selection,
            "status": "published",
            "protocol_sha256": protocol_sha256,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bank-directory", type=Path)
    parser.add_argument("--screening-record", type=Path)
    parser.add_argument("--preregistration", type=Path)
    args = parser.parse_args()
    bank = args.bank_directory
    if bank is None:
        if args.smoke:
            parser.error("--smoke requires --bank-directory")
        bank = Path(__file__).with_name("founders") / "heredity_adaptation_v2"
    record = args.screening_record or bank.with_suffix(".screening.jsonl")
    result = run_ready_founder_bank_builder(
        smoke=args.smoke,
        bank_directory=bank,
        screening_record_path=record,
        preregistration_path=args.preregistration,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
