"""Focused tests for the prospectively frozen Evo² r5 world qualifier."""

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.episode import (  # noqa: E402
    SimulatorConfig,
    build_environment,
    simulator_config_sha256,
)
from experiments.evo2_ecosystem.r5.protocol import (  # noqa: E402
    FOUNDER_CANDIDATE_RANGES,
    FOUNDER_PANEL_SEED,
    FROZEN_SIMULATOR_CONFIG_SHA256,
    PRODUCTION_WORLD_QUALIFICATION_SPEC,
    PROTOCOL_REVISION,
    WORLD_ACCESS_THRESHOLD,
    WORLD_SEED_RANGES,
    ScreeningSpec,
    WorldSeedRange,
    founder_screening_spec,
)
from experiments.evo2_ecosystem.r5.qualify_worlds import (  # noqa: E402
    BackendReplay,
    InsufficientQualifiedWorlds,
    WorldAccessRecord,
    WorldQualificationTrace,
    build_world_qualification_document,
    canonical_world_qualification_bytes,
    compare_backend_replay,
    make_world_qualifier,
    measure_initial_access,
    publish_world_qualification,
    scan_seed_range,
    validate_world_qualification,
    world_qualification_from_dict,
)
from microcosmos.gym import EcosystemEnv, LineTopology  # noqa: E402
from microcosmos.heredity import clone_policy  # noqa: E402


def _record(seed: int, *, score: float, threshold: float = WORLD_ACCESS_THRESHOLD):
    fractions = (score,) + (0.0,) * 7
    cells = tuple((index, index) for index in range(8))
    return WorldAccessRecord(
        seed=seed,
        mouth_cells=cells,
        mouth_capacity_fractions=fractions,
        access_score=score,
        passed=score >= threshold,
    )


def _synthetic_production_evidence():
    traces = []
    replays = []
    for seed_range in WORLD_SEED_RANGES:
        records = (
            _record(seed_range.start, score=0.50),
            _record(seed_range.start + 1, score=0.75),
        )
        trace = WorldQualificationTrace(
            seed_range=seed_range,
            inspected=records,
            selected_seeds=tuple(record.seed for record in records),
        )
        traces.append(trace)
        replays.extend(
            compare_backend_replay(seed_range.partition, record, record)
            for record in records
        )
    return tuple(traces), tuple(replays)


def test_r5_protocol_freezes_only_experiment_local_constants():
    assert PROTOCOL_REVISION == "evo2-r5-world-feasibility-v1"
    assert FOUNDER_PANEL_SEED == 5_005
    assert WORLD_ACCESS_THRESHOLD == 0.50
    assert [(item.start, item.stop) for item in WORLD_SEED_RANGES] == [
        (8_000, 9_000),
        (9_000, 10_000),
        (10_000, 11_000),
        (11_000, 12_000),
    ]
    assert [(item.start, item.stop, item.required) for item in FOUNDER_CANDIDATE_RANGES] == [
        (0, 24, 4),
        (24, 48, 4),
        (48, 80, 8),
    ]
    assert FROZEN_SIMULATOR_CONFIG_SHA256 == simulator_config_sha256(SimulatorConfig())
    assert FROZEN_SIMULATOR_CONFIG_SHA256 == "cd3278775210c4859ab2e4f0a8b391d45ec1b6d05a506df17d2ea59df2b9a10b"


def test_founder_screening_spec_binds_only_eligibility_seeds():
    spec = founder_screening_spec((8_123, 8_456))
    assert isinstance(spec, ScreeningSpec)
    assert spec.panel_seed == 5_005
    assert spec.candidate_count == 80
    with pytest.raises(ValueError, match="eligibility range"):
        founder_screening_spec((8_123, 9_123))


def test_seed_scan_is_an_ascending_prefix_and_stops_at_second_pass():
    seed_range = WorldSeedRange("test", 10, 20)
    called = []

    def qualify(seed):
        called.append(seed)
        return _record(seed, score=0.75 if seed in {11, 13} else 0.25)

    trace = scan_seed_range(seed_range, qualify, passes_required=2)
    assert called == [10, 11, 12, 13]
    assert tuple(record.seed for record in trace.inspected) == (10, 11, 12, 13)
    assert trace.selected_seeds == (11, 13)


def test_seed_scan_preserves_failed_prefix_in_exception():
    seed_range = WorldSeedRange("test", 20, 23)
    with pytest.raises(InsufficientQualifiedWorlds) as captured:
        scan_seed_range(
            seed_range,
            lambda seed: _record(seed, score=0.25),
            passes_required=2,
        )
    assert tuple(record.seed for record in captured.value.trace.inspected) == (20, 21, 22)
    assert captured.value.trace.selected_seeds == ()


def test_measure_initial_access_uses_production_periodic_rounding():
    env = EcosystemEnv(
        topology=LineTopology(num_nodes=2),
        max_creatures=2,
        initial_population=2,
        grid_shape=(4, 4),
        max_steps=1,
        resource_capacity=1.0,
        resource_patch_center=(2.0, 2.0),
        resource_patch_radius=2.0,
    )
    _, state = env.reset(jax.random.PRNGKey(17))
    positions = state.nodes.position.reshape(2, 2, 2)
    positions = positions.at[0, 0].set(jnp.asarray([-0.51, 3.49]))
    positions = positions.at[1, 0].set(jnp.asarray([3.50, -0.50]))
    capacity = jnp.zeros((4, 4), dtype=jnp.float32)
    capacity = capacity.at[3, 3].set(0.75)
    capacity = capacity.at[0, 0].set(0.25)
    state = replace(
        state,
        nodes=replace(state.nodes, position=positions.reshape(-1, 2)),
        resource_capacity_map=capacity,
    )

    record = measure_initial_access(
        state,
        seed=17,
        peak_capacity=1.0,
        access_threshold=0.50,
    )
    assert record.mouth_cells == ((3, 3), (0, 0))
    assert np.allclose(record.mouth_capacity_fractions, (0.75, 0.25))
    assert record.access_score == 0.75
    assert record.passed


def test_reset_only_qualifier_matches_direct_existing_grid_sampling():
    config = replace(
        SimulatorConfig(),
        max_creatures=4,
        initial_population=2,
        nodes_per_creature=4,
        grid_shape=(16, 16),
        fluid_enabled=False,
        resource_patch_center=(8.0, 8.0),
        resource_patch_radius=6.0,
        placement_candidates=4,
        position_margin=0.5,
    )
    qualifier = make_world_qualifier(config, access_threshold=0.50)
    record = qualifier(73)  # Explicit non-r5 diagnostic seed.
    trusted_env = build_environment(config, clone_policy, horizon=1)
    _, trusted_state = trusted_env.reset(jax.random.PRNGKey(73))
    direct = measure_initial_access(
        trusted_state,
        seed=73,
        peak_capacity=config.resource_capacity,
        access_threshold=0.50,
    )
    assert record == direct
    assert len(record.mouth_cells) == config.initial_population
    assert np.isclose(record.access_score, max(record.mouth_capacity_fractions))
    assert "genome" not in inspect.signature(make_world_qualifier).parameters
    assert "genome" not in inspect.signature(measure_initial_access).parameters


def test_backend_comparison_records_cells_score_and_decision():
    cpu = _record(7, score=0.75)
    gpu = WorldAccessRecord(
        seed=7,
        mouth_cells=cpu.mouth_cells,
        mouth_capacity_fractions=cpu.mouth_capacity_fractions,
        access_score=0.75 + 5e-7,
        passed=True,
    )
    replay = compare_backend_replay("training", cpu, gpu)
    assert replay.cells_match
    assert replay.passed_match
    assert replay.score_absolute_error < 1e-6


def test_gpu_threshold_roundoff_does_not_veto_canonical_cpu_selection():
    traces, replays = _synthetic_production_evidence()
    document = build_world_qualification_document(
        traces,
        replays,
        implementation_commit="a" * 40,
    )
    replay = document["gpu_replays"][0]
    replay["access_score"] = WORLD_ACCESS_THRESHOLD - 0.5e-6
    replay["score_absolute_error"] = 0.5e-6
    replay["passed_match"] = False

    validated = world_qualification_from_dict(document)

    assert validated["selected_seeds"] == document["selected_seeds"]
    assert validated["gpu_replays"][0]["passed_match"] is False


def test_qualification_document_is_canonical_strict_and_write_once(tmp_path):
    traces, replays = _synthetic_production_evidence()
    document = build_world_qualification_document(
        traces,
        replays,
        implementation_commit="a" * 40,
    )
    canonical = canonical_world_qualification_bytes(document)
    assert canonical == canonical_world_qualification_bytes(json.loads(canonical))

    path = tmp_path / "world_qualification.json"
    digest = publish_world_qualification(document, path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert validate_world_qualification(
        path,
        expected_implementation_commit="a" * 40,
    ) == document
    with pytest.raises(FileExistsError, match="refusing to replace"):
        publish_world_qualification(document, path)


def test_qualification_validation_fails_closed_on_rule_or_gpu_mismatch():
    traces, replays = _synthetic_production_evidence()
    document = build_world_qualification_document(
        traces,
        replays,
        implementation_commit="b" * 40,
    )

    wrong_rule = json.loads(json.dumps(document))
    wrong_rule["rule"]["access_threshold"] = 0.49
    with pytest.raises(ValueError, match="frozen r5 rule"):
        world_qualification_from_dict(wrong_rule)

    wrong_cells = json.loads(json.dumps(document))
    wrong_cells["gpu_replays"][0]["mouth_cells"][0] = [63, 63]
    with pytest.raises(ValueError, match="do not match"):
        world_qualification_from_dict(wrong_cells)

    wrong_score = json.loads(json.dumps(document))
    wrong_score["gpu_replays"][0]["access_score"] += 2e-6
    wrong_score["gpu_replays"][0]["score_absolute_error"] = 2e-6
    with pytest.raises(ValueError, match="agreement tolerance"):
        world_qualification_from_dict(wrong_score)

    wrong_diagnostic = json.loads(json.dumps(document))
    wrong_diagnostic["gpu_replays"][0]["passed_match"] = False
    with pytest.raises(ValueError, match="diagnostic is inconsistent"):
        world_qualification_from_dict(wrong_diagnostic)


def test_protocol_spec_rejects_overlapping_seed_ranges():
    with pytest.raises(ValueError, match="disjoint"):
        replace(
            PRODUCTION_WORLD_QUALIFICATION_SPEC,
            seed_ranges=(
                WorldSeedRange("a", 0, 10),
                WorldSeedRange("b", 9, 20),
            ),
        )


def test_backend_replay_type_carries_no_hidden_outcomes():
    assert set(BackendReplay.__dataclass_fields__) == {
        "partition",
        "seed",
        "mouth_cells",
        "access_score",
        "cells_match",
        "score_absolute_error",
        "passed_match",
    }
