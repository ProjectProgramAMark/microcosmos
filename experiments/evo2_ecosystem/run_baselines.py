"""Run frozen heredity baselines through the canonical episode evaluator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import jax
import numpy as np

from microcosmos.heredity import (
    OffspringPolicy,
    clone_policy,
    fixed_mixed_policy,
    fixed_parametric_policy,
    fixed_structural_policy,
    stress_responsive_policy,
)

from .episode import (
    ManifestEvaluation,
    SimulatorConfig,
    evaluate_manifest,
    simulator_config_sha256,
    simulator_source_sha256,
)
from .protocol import (
    ScenarioManifest,
    manifest_from_json_bytes,
    manifest_sha256,
)


BASELINE_NAMES = (
    "clone",
    "fixed_parametric",
    "fixed_mixed",
    "stress_responsive",
)


def baseline_policy(name: str) -> OffspringPolicy:
    """Resolve only the preregistered trusted policies."""
    policies = {
        "clone": clone_policy,
        "fixed_parametric": fixed_parametric_policy,
        "fixed_structural": fixed_structural_policy,
        "fixed_mixed": fixed_mixed_policy,
        "stress_responsive": stress_responsive_policy,
    }
    try:
        return policies[name]
    except KeyError as error:
        raise ValueError(f"unknown baseline policy {name!r}") from error


def evaluate_baseline(
    manifest: ScenarioManifest,
    policy_name: str,
    *,
    config: SimulatorConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one policy and return its complete auditable raw record."""
    config = SimulatorConfig() if config is None else config
    policy = baseline_policy(policy_name)
    evaluation = evaluate_manifest(manifest, config, policy)
    return baseline_result_record(
        manifest,
        config,
        policy_name,
        policy,
        evaluation,
    )


def baseline_result_record(
    manifest: ScenarioManifest,
    config: SimulatorConfig,
    policy_name: str,
    policy: OffspringPolicy,
    evaluation: ManifestEvaluation,
) -> dict[str, Any]:
    """Serialize device arrays without retaining dense trajectories."""
    if len(manifest.worlds) != len(evaluation.episodes):
        raise ValueError("manifest and evaluation must have equal world counts")
    episodes = []
    for world, episode in zip(manifest.worlds, evaluation.episodes, strict=True):
        event = episode.event_record
        episodes.append(
            {
                "scenario_id": world.scenario_id,
                "pair_id": world.pair_id,
                "scenario_family": world.scenario_family,
                "world_seed": world.world_seed,
                "event_kind": world.event_kind.value,
                "primary_score": _scalar(episode.primary_score),
                "post_event_productivity": _vector(episode.post_event_productivity),
                "survived": bool(episode.survived),
                "final_alive": int(episode.final_alive),
                "minimum_population": int(episode.minimum_population),
                "maximum_generation": int(episode.maximum_generation),
                "birth_count": int(episode.birth_count),
                "natural_death_count": int(episode.natural_death_count),
                "operator_counts": [int(value) for value in np.asarray(episode.operator_counts)],
                "catastrophe_death_count": int(event.catastrophe_death_count),
                "targeted_founder_lineage": int(event.targeted_founder_lineage),
                "policy_violation_count": int(episode.policy_violation_count),
                "integrity_valid": bool(episode.integrity_valid),
            }
        )
    return {
        "schema_version": 1,
        "policy_name": policy_name,
        "policy_source_sha256": hashlib.sha256(inspect.getsource(policy).encode("utf-8")).hexdigest(),
        "manifest_sha256": manifest_sha256(manifest),
        "simulator_config_sha256": simulator_config_sha256(config),
        "simulator_source_sha256": simulator_source_sha256(),
        "simulator_config": asdict(config),
        "candidate_score": _scalar(evaluation.candidate_score),
        "integrity_valid": bool(evaluation.integrity_valid),
        "numerical_repeats": int(np.asarray(evaluation.repeat_scores).size),
        "repeat_scores": _vector(evaluation.repeat_scores),
        "selected_repeat_index": int(evaluation.selected_repeat_index),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "episodes": episodes,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish one result after the full evaluation succeeds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _scalar(value: Any) -> float:
    return float(np.asarray(value))


def _vector(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=BASELINE_NAMES,
        default=list(BASELINE_NAMES),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = manifest_from_json_bytes(args.manifest.read_bytes())
    for policy_name in args.policies:
        result = evaluate_baseline(manifest, policy_name)
        output = args.output_dir / f"{policy_name}.json"
        write_json_atomic(output, result)
        print(
            json.dumps(
                {
                    "policy": policy_name,
                    "score": result["candidate_score"],
                    "integrity_valid": result["integrity_valid"],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
