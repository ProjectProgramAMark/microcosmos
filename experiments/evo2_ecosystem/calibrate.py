"""One-time Phase B viability calibration for the final Evo² ecosystem."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp

from microcosmos.gym import EcosystemEnv, LineTopology
from microcosmos.heredity import MIXED, fixed_mixed_policy
from microcosmos.rollout import (
    EcosystemChunkMetrics,
    combine_chunk_metrics,
    run_ecosystem_chunk,
)
from microcosmos.solver.config import PBD_SCHEME, PBD_SCHEME_NO_FLUID

SCHEMA_VERSION = 1
CALIBRATION_PANEL_VERSION = 2
PREVIOUS_FAILED_PROTOCOL_HASH = "67f4f184e500ac0906f342a86d4c960a68fc400d02ac20ae0070e7894f7c4441"
FINAL_CAPACITY = 32
FINAL_NODES_PER_CREATURE = 8
FINAL_GRID_SHAPE = (64, 64)
FINAL_RESOURCE_PATCH_CENTER = (48.0, 32.0)
FINAL_RESOURCE_PATCH_RADIUS = 12.0
PRODUCTION_SEEDS = (11, 23, 37, 53, 71)


@dataclass(frozen=True)
class CalibrationSpec:
    capacity: int
    nodes_per_creature: int
    grid_shape: tuple[int, int]
    seeds: tuple[int, ...]
    chunk_steps: int
    burn_in_chunks: int
    assessment_chunks: int
    fluid_enabled: bool
    saturation_fraction_limit: float = 0.8
    minimum_post_burn_in_births: int = 2
    minimum_final_alive: int = 2

    @property
    def horizon(self) -> int:
        return self.chunk_steps * (self.burn_in_chunks + self.assessment_chunks)


PRODUCTION_SPEC = CalibrationSpec(
    capacity=FINAL_CAPACITY,
    nodes_per_creature=FINAL_NODES_PER_CREATURE,
    grid_shape=FINAL_GRID_SHAPE,
    seeds=PRODUCTION_SEEDS,
    chunk_steps=500,
    burn_in_chunks=4,
    assessment_chunks=12,
    fluid_enabled=True,
)

SMOKE_SPEC = CalibrationSpec(
    capacity=4,
    nodes_per_creature=4,
    grid_shape=(16, 16),
    seeds=(0,),
    chunk_steps=2,
    burn_in_chunks=1,
    assessment_chunks=1,
    fluid_enabled=False,
)


def _production_candidates() -> tuple[dict[str, Any], ...]:
    balanced = {
        "initial_resource_fraction": 1.0,
        "resource_regeneration_rate": 0.03,
        "uptake_rate": 1.0,
        "assimilation_efficiency": 0.9,
        "basal_metabolism": 0.01,
        "actuation_power_coefficient": 0.002,
        "reproduction_threshold": 3.0,
        "reproduction_cost": 1.5,
        "birth_transfer_efficiency": 0.8,
        "maturity_age": 50,
        "maximum_lifespan": 5_000,
        "initial_population": 8,
    }
    lower_birth_cost = {
        **balanced,
        "reproduction_threshold": 2.5,
        "reproduction_cost": 1.25,
    }
    larger_child_endowment = {
        **lower_birth_cost,
        "birth_transfer_efficiency": 1.0,
    }
    more_regeneration = {
        **larger_child_endowment,
        "resource_regeneration_rate": 0.05,
    }
    higher_density = {**more_regeneration, "initial_population": 12}
    configs = (
        ("v2_c00_balanced", balanced),
        ("v2_c01_lower_birth_cost", lower_birth_cost),
        ("v2_c02_larger_child_endowment", larger_child_endowment),
        ("v2_c03_more_regeneration", more_regeneration),
        ("v2_c04_higher_density", higher_density),
    )
    return tuple({"candidate_id": candidate_id, "config": config} for candidate_id, config in configs)


CALIBRATION_CANDIDATES = _production_candidates()
SMOKE_CANDIDATES = (
    {
        "candidate_id": "smoke",
        "config": {
            **CALIBRATION_CANDIDATES[0]["config"],
            "initial_population": 2,
        },
    },
)

REQUIRED_CONFIG_KEYS = frozenset(CALIBRATION_CANDIDATES[0]["config"])


def canonical_json(value: Any) -> str:
    """Return the exact JSON representation used for protocol hashes."""
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def validate_candidate_panel(candidates: Sequence[dict[str, Any]], spec: CalibrationSpec) -> None:
    """Reject accidental search-space drift before a simulation starts."""
    if not candidates:
        raise ValueError("calibration candidate panel must not be empty")
    identifiers = [candidate["candidate_id"] for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("calibration candidate identifiers must be unique")
    for candidate in candidates:
        config = candidate["config"]
        if frozenset(config) != REQUIRED_CONFIG_KEYS:
            raise ValueError("every candidate must declare the complete config")
        for name, value in config.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real scalar")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0 <= config["initial_resource_fraction"] <= 1:
            raise ValueError("initial resource fraction must be within [0, 1]")
        if not 0 < config["initial_population"] <= spec.capacity:
            raise ValueError("initial population must fit the declared capacity")


def validate_execution_contract(
    spec: CalibrationSpec,
    *,
    backend: str,
    smoke: bool,
) -> None:
    """Prevent a production result from using a development-scale substrate."""
    if smoke:
        return
    if not spec.fluid_enabled:
        raise RuntimeError("production calibration requires fluid physics")
    if backend != "gpu":
        raise RuntimeError("production calibration requires the JAX GPU backend")
    actual_shape = (
        spec.capacity,
        spec.nodes_per_creature,
        spec.grid_shape,
    )
    expected_shape = (
        FINAL_CAPACITY,
        FINAL_NODES_PER_CREATURE,
        FINAL_GRID_SHAPE,
    )
    if actual_shape != expected_shape:
        raise RuntimeError(f"production calibration requires shape {expected_shape}, got {actual_shape}")


def _gate_definition(spec: CalibrationSpec) -> dict[str, Any]:
    return {
        "aggregation": "strict-majority per ecological condition; all worlds for integrity",
        "births": ">=1 total birth in a strict majority of worlds",
        "natural_deaths": ">=1 natural death in a strict majority of worlds",
        "generation": ">=3 in a strict majority of worlds",
        "post_burn_in_births": (f">={spec.minimum_post_burn_in_births} assessment births in a strict majority of worlds"),
        "nontrivial_survival": (f">={spec.minimum_final_alive} organisms alive at the end in a strict majority of worlds"),
        "early_extinction": "burn-in ends nonempty in a strict majority of worlds",
        "permanent_saturation": (f"assessment fraction at capacity is below {spec.saturation_fraction_limit} in a strict majority of worlds"),
        "integrity": (
            "finite, identity-valid, event-valid, infrastructure-valid, zero policy violations, and exact fixed-mixed operator accounting in every world"
        ),
    }


def evaluate_gate(worlds: Sequence[dict[str, Any]], spec: CalibrationSpec) -> dict[str, Any]:
    """Apply the preregistered viability gate without an optimization score."""
    if len(worlds) != len(spec.seeds):
        raise ValueError("gate requires one result for every declared seed")
    majority = len(worlds) // 2 + 1

    def count(predicate: Callable[[dict[str, Any]], bool]) -> int:
        return sum(bool(predicate(world)) for world in worlds)

    counts = {
        "births": count(lambda world: world["total_births"] >= 1),
        "natural_deaths": count(lambda world: world["total_natural_deaths"] >= 1),
        "generation": count(lambda world: world["maximum_generation"] >= 3),
        "post_burn_in_births": count(lambda world: world["assessment_births"] >= spec.minimum_post_burn_in_births),
        "nontrivial_survival": count(lambda world: world["final_alive"] >= spec.minimum_final_alive),
        "no_early_extinction": count(lambda world: world["burn_in_final_alive"] > 0),
        "no_permanent_saturation": count(lambda world: world["assessment_capacity_fraction"] < spec.saturation_fraction_limit),
    }
    integrity_fields = (
        "finite",
        "identity_valid",
        "events_valid",
        "infrastructure_valid",
        "fixed_mixed_accounting_valid",
    )
    integrity = all(all(bool(world[field]) for field in integrity_fields) and world["policy_violation_count"] == 0 for world in worlds)
    ecological = all(value >= majority for value in counts.values())
    return {
        "passed": integrity and ecological,
        "required_worlds": majority,
        "condition_counts": counts,
        "integrity_all_worlds": integrity,
    }


def _phase_summary(
    metrics: EcosystemChunkMetrics,
    chunk_end_alive: list[int],
) -> dict[str, Any]:
    return {
        "steps": int(metrics.steps),
        "births": int(metrics.birth_count),
        "natural_deaths": int(metrics.death_count),
        "maximum_generation": int(metrics.maximum_generation),
        "steps_at_capacity": int(metrics.steps_at_capacity),
        "chunk_end_alive": chunk_end_alive,
        "finite": bool(metrics.finite),
        "identity_valid": bool(metrics.identity_valid),
        "events_valid": bool(metrics.events_valid),
        "operator_counts": [int(value) for value in metrics.operator_counts],
        "policy_violation_count": int(metrics.policy_violation_count),
        "infrastructure_valid": bool(metrics.infrastructure_valid),
    }


def _run_phase(
    compiled_chunk: Callable,
    state,
    *,
    root_key: jax.Array,
    first_chunk_index: int,
    chunks: int,
    chunk_steps: int,
):
    combined = None
    alive_trace: list[int] = []
    for offset in range(chunks):
        chunk_index = first_chunk_index + offset
        chunk_key = jax.random.fold_in(root_key, chunk_index)
        keys = jax.random.split(chunk_key, chunk_steps)
        state, metrics = compiled_chunk(state, keys)
        jax.block_until_ready((state, metrics))
        combined = metrics if combined is None else combine_chunk_metrics(combined, metrics)
        alive_trace.append(int(jnp.sum(state.population.alive)))
    if combined is None:
        raise ValueError("a calibration phase must contain at least one chunk")
    return state, combined, alive_trace


def _environment(config: dict[str, Any], spec: CalibrationSpec) -> EcosystemEnv:
    solver = PBD_SCHEME if spec.fluid_enabled else PBD_SCHEME_NO_FLUID
    return EcosystemEnv(
        topology=LineTopology(num_nodes=spec.nodes_per_creature),
        max_creatures=spec.capacity,
        initial_population=int(config["initial_population"]),
        grid_shape=spec.grid_shape,
        dt=0.01,
        max_steps=spec.horizon,
        solver_config=solver,
        resource_capacity=1.0,
        initial_resource=float(config["initial_resource_fraction"]),
        resource_regeneration_rate=float(config["resource_regeneration_rate"]),
        resource_diffusion_rate=0.0,
        resource_patch_center=FINAL_RESOURCE_PATCH_CENTER,
        resource_patch_radius=FINAL_RESOURCE_PATCH_RADIUS,
        initial_energy=2.0,
        birth_transfer_efficiency=float(config["birth_transfer_efficiency"]),
        reproduction_threshold=float(config["reproduction_threshold"]),
        reproduction_cost=float(config["reproduction_cost"]),
        maturity_age=int(config["maturity_age"]),
        maximum_lifespan=int(config["maximum_lifespan"]),
        max_bending_delta=0.35,
        uptake_rate=float(config["uptake_rate"]),
        assimilation_efficiency=float(config["assimilation_efficiency"]),
        basal_metabolism=float(config["basal_metabolism"]),
        actuation_power_coefficient=float(config["actuation_power_coefficient"]),
        placement_candidates=16,
        offspring_policy=fixed_mixed_policy,
    )


def run_candidate(candidate: dict[str, Any], spec: CalibrationSpec) -> dict[str, Any]:
    """Run one candidate on the complete paired world panel."""
    env = _environment(candidate["config"], spec)
    compiled_chunk = jax.jit(lambda state, keys: run_ecosystem_chunk(env, state, keys))
    started = time.perf_counter()
    worlds = []
    for seed in spec.seeds:
        root_key = jax.random.PRNGKey(seed)
        reset_key = jax.random.fold_in(root_key, 0)
        rollout_key = jax.random.fold_in(root_key, 1)
        _, state = env.reset(reset_key)
        state, burn, burn_alive = _run_phase(
            compiled_chunk,
            state,
            root_key=rollout_key,
            first_chunk_index=0,
            chunks=spec.burn_in_chunks,
            chunk_steps=spec.chunk_steps,
        )
        burn_summary = _phase_summary(burn, burn_alive)
        state, assessment, assessment_alive = _run_phase(
            compiled_chunk,
            state,
            root_key=rollout_key,
            first_chunk_index=spec.burn_in_chunks,
            chunks=spec.assessment_chunks,
            chunk_steps=spec.chunk_steps,
        )
        assessment_summary = _phase_summary(assessment, assessment_alive)
        operator_counts = [
            left + right
            for left, right in zip(
                burn_summary["operator_counts"],
                assessment_summary["operator_counts"],
                strict=True,
            )
        ]
        total_births = burn_summary["births"] + assessment_summary["births"]
        worlds.append(
            {
                "seed": seed,
                "burn_in": burn_summary,
                "assessment": assessment_summary,
                "total_births": total_births,
                "total_natural_deaths": (burn_summary["natural_deaths"] + assessment_summary["natural_deaths"]),
                "assessment_births": assessment_summary["births"],
                "maximum_generation": max(
                    burn_summary["maximum_generation"],
                    assessment_summary["maximum_generation"],
                ),
                "burn_in_final_alive": burn_alive[-1],
                "final_alive": assessment_alive[-1],
                "assessment_capacity_fraction": (assessment_summary["steps_at_capacity"] / assessment_summary["steps"]),
                "finite": burn_summary["finite"] and assessment_summary["finite"],
                "identity_valid": burn_summary["identity_valid"] and assessment_summary["identity_valid"],
                "events_valid": burn_summary["events_valid"] and assessment_summary["events_valid"],
                "infrastructure_valid": (burn_summary["infrastructure_valid"] and assessment_summary["infrastructure_valid"]),
                "policy_violation_count": (burn_summary["policy_violation_count"] + assessment_summary["policy_violation_count"]),
                "fixed_mixed_accounting_valid": (operator_counts[MIXED] == total_births and sum(operator_counts) == total_births),
            }
        )
    return {
        "status": "completed",
        "elapsed_seconds": time.perf_counter() - started,
        "worlds": worlds,
        "gate": evaluate_gate(worlds, spec),
    }


def protocol_record(spec: CalibrationSpec, candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "calibration_panel_version": CALIBRATION_PANEL_VERSION,
        "previous_failed_protocol_hash": PREVIOUS_FAILED_PROTOCOL_HASH,
        "mode": "production" if spec.fluid_enabled else "smoke",
        "shape": {
            "capacity": spec.capacity,
            "nodes_per_creature": spec.nodes_per_creature,
            "grid_shape": list(spec.grid_shape),
        },
        "physics": {
            "dt": 0.01,
            "solver": "PBD_SCHEME" if spec.fluid_enabled else "PBD_SCHEME_NO_FLUID",
            "fluid_enabled": spec.fluid_enabled,
            "steric_enabled": False,
        },
        "resource_geometry": {
            "capacity": 1.0,
            "diffusion_rate": 0.0,
            "patch_center": list(FINAL_RESOURCE_PATCH_CENTER),
            "patch_radius": FINAL_RESOURCE_PATCH_RADIUS,
        },
        "fixed_ecology": {
            "initial_energy": 2.0,
            "max_bending_delta": 0.35,
            "placement_candidates": 16,
            "offspring_policy": "fixed_mixed_policy",
        },
        "seeds": list(spec.seeds),
        "chunk_steps": spec.chunk_steps,
        "burn_in_chunks": spec.burn_in_chunks,
        "assessment_chunks": spec.assessment_chunks,
        "horizon": spec.horizon,
        "gate": _gate_definition(spec),
        "candidate_order": [
            {
                "candidate_id": candidate["candidate_id"],
                "config": candidate["config"],
                "config_hash": canonical_hash(candidate["config"]),
            }
            for candidate in candidates
        ],
    }


def pending_manifest() -> dict[str, Any]:
    protocol = protocol_record(PRODUCTION_SPEC, CALIBRATION_CANDIDATES)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "B_viability_calibration",
        "status": "pending_production_run",
        "protocol": protocol,
        "protocol_hash": canonical_hash(protocol),
        "attempts": [],
        "selected_candidate_id": None,
        "selected_config": None,
        "selected_config_hash": None,
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_calibration(
    *,
    smoke: bool,
    output_path: Path | None = None,
    runner: Callable[[dict[str, Any], CalibrationSpec], dict[str, Any]] = run_candidate,
) -> dict[str, Any]:
    """Try candidates in order and stop permanently at the first viable one."""
    spec = SMOKE_SPEC if smoke else PRODUCTION_SPEC
    candidates = SMOKE_CANDIDATES if smoke else CALIBRATION_CANDIDATES
    validate_candidate_panel(candidates, spec)
    validate_execution_contract(
        spec,
        backend=jax.default_backend(),
        smoke=smoke,
    )
    protocol = protocol_record(spec, candidates)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "phase": "B_viability_calibration",
        "status": "running",
        "protocol": protocol,
        "protocol_hash": canonical_hash(protocol),
        "attempts": [],
        "selected_candidate_id": None,
        "selected_config": None,
        "selected_config_hash": None,
    }
    if output_path is not None:
        _write_manifest(output_path, manifest)

    for candidate in candidates:
        config_hash = canonical_hash(candidate["config"])
        try:
            result = runner(candidate, spec)
        except Exception as error:  # Preserve every attempted setting.
            result = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "gate": {"passed": False},
            }
        attempt = {
            "candidate_id": candidate["candidate_id"],
            "config": candidate["config"],
            "config_hash": config_hash,
            **result,
        }
        manifest["attempts"].append(attempt)
        if result["gate"]["passed"]:
            manifest.update(
                status="selected",
                selected_candidate_id=candidate["candidate_id"],
                selected_config=candidate["config"],
                selected_config_hash=config_hash,
            )
        if output_path is not None:
            _write_manifest(output_path, manifest)
        if manifest["status"] == "selected":
            break

    if manifest["status"] != "selected":
        manifest["status"] = "no_viable_configuration"
        if output_path is not None:
            _write_manifest(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use a tiny CPU/no-fluid execution check; never a scientific result",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=("manifest path; production defaults to calibration_manifest.json while smoke writes only when explicitly requested"),
    )
    args = parser.parse_args()
    output = args.output
    if output is None and not args.smoke:
        output = Path(__file__).with_name("calibration_manifest.json")
    manifest = run_calibration(smoke=args.smoke, output_path=output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
