"""Focused, seed-free tests for the Evo² r5 protocol adapters."""

import hashlib
from pathlib import Path
import stat
import sys

import numpy as np
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.build_founder_bank import (  # noqa: E402
    _publish_bank,
)
from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: E402
    founder_index_sha256,
    load_founder_index,
)
from experiments.evo2_ecosystem.protocol import (  # noqa: E402
    EventKind,
    canonical_manifest_bytes,
    manifest_sha256,
)
from experiments.evo2_ecosystem.r5.protocol import (  # noqa: E402
    ACTUATION_COST_MULTIPLIERS,
    FOUNDER_PANEL_SEED,
    PROTOCOL_REVISION,
    founder_screening_spec,
)
from experiments.evo2_ecosystem.r5 import tools  # noqa: E402
from microcosmos.cppn import CPPNGenome, initialize_cppn_population  # noqa: E402


def _founder_bank(path: Path) -> Path:
    population = initialize_cppn_population(16, 16)
    genomes = [
        CPPNGenome(population.node_genes[index], population.connection_genes[index])
        for index in range(16)
    ]
    selected = {
        "training": [(index, genomes[index]) for index in range(4)],
        "development": [(index + 4, genomes[index + 4]) for index in range(4)],
        "sealed": [(index + 8, genomes[index + 8]) for index in range(8)],
    }
    _publish_bank(
        path,
        selected,
        "test-r5",
        selection_seed=FOUNDER_PANEL_SEED,
    )
    return path / "index.json"


def _seed_panel() -> dict[str, tuple[int, int]]:
    # Explicit non-r5 diagnostic seeds: these tests never reset a scientific world.
    return {
        "training": (101, 102),
        "development": (201, 202),
        "sealed": (301, 302),
    }


def _viability() -> dict[str, object]:
    return {
        "distinct_birth": 2,
        "distinct_reproducer": 1,
        "integrity": True,
        "median_generation_gain": 1.0,
        "median_post_births": 4.0,
        "survival": 1.0,
    }


def _disturbance_result(
    multiplier: float,
    *,
    passed: bool,
    manifest_hash: str = "a" * 64,
) -> dict[str, object]:
    gates = {
        "harm": passed,
        "integrity": passed,
        "observable_stress": passed,
        "resolved_feedback": passed,
        "viability": passed,
    }
    return {
        "clone_effect_founder_bootstrap_upper_95": -0.04 if passed else 0.01,
        "clone_effect_mean": -0.05 if passed else 0.0,
        "clone_viability": _viability() if passed else {
            **_viability(),
            "distinct_birth": 0,
            "distinct_reproducer": 0,
            "integrity": False,
            "median_generation_gain": 0.0,
            "median_post_births": 0.0,
            "survival": 0.0,
        },
        "final_evidence_mean": ([0.2] * 6) if passed else ([0.0] * 6),
        "gates": gates,
        "manifest_sha256": manifest_hash,
        "multiplier": multiplier,
        "observable_credit_tv": 0.25 if passed else 0.0,
        "observable_feature_max_smd": 0.75 if passed else 0.0,
        "passed": passed,
        "policy_integrity": {
            "clone": passed,
            "exploration": passed,
            "standard": passed,
        },
        "post_resolved_by_operator": ([8] * 6) if passed else ([0] * 6),
        "pre_resolved_by_operator": ([8] * 6) if passed else ([0] * 6),
        "standard_viability": _viability() if passed else {
            **_viability(),
            "distinct_birth": 0,
            "distinct_reproducer": 0,
            "integrity": False,
            "median_generation_gain": 0.0,
            "median_post_births": 0.0,
            "survival": 0.0,
        },
    }


def _disturbance_document(
    index_path: Path,
    *,
    multiplier: float = ACTUATION_COST_MULTIPLIERS[0],
    manifest_hash: str = "a" * 64,
    world_hash: str = "b" * 64,
) -> dict[str, object]:
    index = load_founder_index(index_path, verify_artifacts=False)
    selected_index = ACTUATION_COST_MULTIPLIERS.index(multiplier)
    return {
        "founder_index_sha256": founder_index_sha256(index),
        "passed": True,
        "protocol_revision": PROTOCOL_REVISION,
        "results": [
            _disturbance_result(
                candidate_multiplier,
                passed=candidate_multiplier == multiplier,
                manifest_hash=manifest_hash,
            )
            for candidate_multiplier in ACTUATION_COST_MULTIPLIERS[: selected_index + 1]
        ],
        "schema_version": tools.DISTURBANCE_QUALIFICATION_SCHEMA_VERSION,
        "selected_multiplier": multiplier,
        "simulator_source_sha256": tools.simulator_source_sha256(),
        "status": "qualified",
        "training_world_seeds": [101, 102],
        "world_qualification_sha256": world_hash,
    }


def _opportunity_record(founder_id: str, pair_id: str, context: str) -> dict[str, object]:
    return {
        "action_scores": np.full((6, 8), 0.5).tolist(),
        "context": context,
        "features": [0.25] * 29,
        "founder_id": founder_id,
        "pair_id": pair_id,
        "resolved": np.ones((6, 8), dtype=int).tolist(),
        "target_child_ids": np.arange(48, dtype=int).reshape(6, 8).tolist(),
    }


def _opportunity_summary(founders: list[str]) -> dict[str, object]:
    return {
        "exercised_nonclone_actions": [1, 2],
        "founder_bootstrap_lower_95": 0.01,
        "gates": {
            "lower_interval": True,
            "mean_advantage": True,
            "outcome_evidence": True,
            "two_nonclone_actions": True,
        },
        "mean_advantage": 0.05,
        "passed": True,
        "post_resolved_by_operator": [16] * 6,
        "pre_resolved_by_operator": [16] * 6,
        "selected_rules": [
            {
                "action_high": 2,
                "action_low": 1,
                "best_fixed_action": 2,
                "feature": 0,
                "held_founder": founder,
                "threshold": 0.5,
            }
            for founder in founders
        ],
    }


def test_r5_screening_contract_converts_to_generic_builder_without_running_worlds():
    r5 = founder_screening_spec((8_123, 8_456))
    converted = tools.builder_screening_spec(r5)

    assert converted.mode == "r4_production"
    assert converted.seeds == (8_123, 8_456)
    assert converted.horizon == 4_000
    assert converted.chunk_steps == 500
    assert converted.selection_rule_id == r5.selection_rule_id
    assert [(item.start, item.stop, item.required) for item in converted.candidate_ranges] == [
        (0, 24, 4),
        (24, 48, 4),
        (48, 80, 8),
    ]


def test_founder_adapter_passes_only_qualified_seeds_and_fresh_panel_to_builder(
    tmp_path,
    monkeypatch,
):
    captured = {}
    monkeypatch.setattr(
        tools,
        "validate_world_qualification",
        lambda _path: {
            "selected_seeds": {
                "founder_eligibility": [8_123, 8_456],
                "training": [9_123, 9_456],
                "development": [10_123, 10_456],
                "sealed": [11_123, 11_456],
            }
        },
    )

    def fake_builder(**kwargs):
        captured.update(kwargs)
        kwargs["screening_record_path"].write_text(
            '{"status":"not-run"}\n', encoding="utf-8"
        )
        return {"status": "not-run"}

    monkeypatch.setattr(tools, "run_founder_bank_builder", fake_builder)
    result = tools.run_qualified_founder_screening(
        tmp_path / "world.json",
        bank_directory=tmp_path / "founders",
        screening_record_path=tmp_path / "screen.jsonl",
    )

    assert result == {"status": "not-run"}
    assert captured["configured_spec"].seeds == (8_123, 8_456)
    assert captured["panel_seed"] == FOUNDER_PANEL_SEED
    assert captured["smoke"] is False
    assert stat.S_IMODE((tmp_path / "screen.jsonl").stat().st_mode) == 0o444


def test_founder_adapter_locks_terminal_failure_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tools,
        "validate_world_qualification",
        lambda _path: {
            "selected_seeds": {
                "founder_eligibility": [8_123, 8_456],
                "training": [9_123, 9_456],
                "development": [10_123, 10_456],
                "sealed": [11_123, 11_456],
            }
        },
    )

    def fail_builder(**kwargs):
        kwargs["screening_record_path"].write_text(
            '{"status":"failed"}\n', encoding="utf-8"
        )
        raise RuntimeError("terminal founder failure")

    monkeypatch.setattr(tools, "run_founder_bank_builder", fail_builder)
    record = tmp_path / "screen.jsonl"
    with pytest.raises(RuntimeError, match="terminal founder failure"):
        tools.run_qualified_founder_screening(
            tmp_path / "world.json",
            bank_directory=tmp_path / "founders",
            screening_record_path=record,
        )
    assert stat.S_IMODE(record.stat().st_mode) == 0o444


def test_manifest_adapter_uses_supplied_seeds_and_r5_event_steps(tmp_path):
    index_path = _founder_bank(tmp_path / "founders")
    seeds = _seed_panel()
    bundle = tools.build_manifest_bundle(index_path, seeds, 1.5)

    for partition, manifest, event_steps in (
        ("training", bundle.training_stable, (3_500, 4_000)),
        ("development", bundle.development, (3_000, 4_500)),
        ("sealed", bundle.sealed, (2_500, 4_500)),
    ):
        assert tuple(dict.fromkeys(world.world_seed for world in manifest.worlds)) == seeds[partition]
        expected = dict(zip(seeds[partition], event_steps, strict=True))
        assert all(world.event_step == expected[world.world_seed] for world in manifest.worlds)
        assert 4_101 not in {world.world_seed for world in manifest.worlds}
        assert 4_102 not in {world.world_seed for world in manifest.worlds}


def test_manifest_publication_is_canonical_write_once_and_keeps_sidecars_readable(
    tmp_path,
):
    index_path = _founder_bank(tmp_path / "founders")
    bundle = tools.build_manifest_bundle(index_path, _seed_panel(), 1.5)
    destination = tmp_path / "manifests"
    hashes = tools._publish_manifest_bundle(bundle, destination)

    for filename, manifest in bundle.named():
        path = destination / filename
        sidecar = path.with_suffix(".sha256")
        expected_mode = 0o444 if filename.startswith("training_") else 0o000
        assert stat.S_IMODE(path.stat().st_mode) == expected_mode
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o444
        assert sidecar.read_text(encoding="ascii") == (
            f"{hashes[filename]}  {filename}\n"
        )
        if expected_mode == 0o444:
            assert path.read_bytes() == canonical_manifest_bytes(manifest) + b"\n"
            assert hashes[filename] == manifest_sha256(manifest)

    with pytest.raises(FileExistsError, match="refusing to replace"):
        tools._publish_manifest_bundle(bundle, destination)


def test_disturbance_evidence_is_strict_canonical_top_level_and_write_once(tmp_path):
    index_path = _founder_bank(tmp_path / "founders")
    document = _disturbance_document(index_path)
    path = tmp_path / "disturbance.json"
    digest = tools.publish_disturbance_qualification(document, path)

    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tools.validate_disturbance_qualification(path)["passed"] is True
    with pytest.raises(FileExistsError, match="refusing to replace"):
        tools.publish_disturbance_qualification(document, path)

    invalid = dict(document)
    invalid["passed"] = False
    with pytest.raises(ValueError, match="top-level disturbance status"):
        tools.disturbance_qualification_from_dict(invalid)

    forged = _disturbance_document(index_path)
    forged["results"][0]["clone_effect_mean"] = 999.0
    with pytest.raises(ValueError, match="gates disagree with the stored metrics"):
        tools.disturbance_qualification_from_dict(forged)


def test_disturbance_adapter_stops_at_first_pass_and_never_uses_r4_seeds(
    tmp_path,
    monkeypatch,
):
    index_path = _founder_bank(tmp_path / "founders")
    world_path = tmp_path / "synthetic-world.json"
    world_path.write_bytes(b"synthetic fixture")
    selected = {
        "founder_eligibility": [1, 2],
        "training": [101, 102],
        "development": [201, 202],
        "sealed": [301, 302],
    }
    monkeypatch.setattr(tools.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        tools,
        "validate_world_qualification",
        lambda _path: {"selected_seeds": selected},
    )
    observed = []

    def fake_qualify(_index_path, manifest):
        seeds = tuple(dict.fromkeys(world.world_seed for world in manifest.worlds))
        observed.append(seeds)
        multiplier = next(
            world.event_parameters.multiplier
            for world in manifest.worlds
            if world.event_kind is EventKind.ACTUATION_COST_SHIFT
        )
        return _disturbance_result(
            multiplier,
            passed=True,
            manifest_hash=manifest_sha256(manifest),
        )

    monkeypatch.setattr(tools, "qualify_disturbance_manifest", fake_qualify)
    output = tmp_path / "disturbance.json"
    result = tools.run_disturbance_qualification(index_path, world_path, output)

    assert result["passed"] is True
    assert result["selected_multiplier"] == ACTUATION_COST_MULTIPLIERS[0]
    assert observed == [(101, 102)]


def test_opportunity_evidence_is_strict_canonical_top_level_and_write_once(tmp_path):
    index_path = _founder_bank(tmp_path / "founders")
    records = [
        _opportunity_record(f"train-{founder:02d}", f"pair-{founder:02d}", context)
        for founder in range(4)
        for context in ("pre", "post")
    ]
    summary = tools.crossfit_opportunity(records)
    document = {
        "founder_index_sha256": founder_index_sha256(
            load_founder_index(index_path, verify_artifacts=False)
        ),
        "manifest_sha256": "c" * 64,
        "multiplier": 1.5,
        "passed": summary["passed"],
        "protocol_revision": PROTOCOL_REVISION,
        "records": records,
        "schema_version": tools.OPPORTUNITY_QUALIFICATION_SCHEMA_VERSION,
        "summary": summary,
    }
    path = tmp_path / "opportunity.json"
    digest = tools.publish_opportunity_qualification(document, path)

    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tools.validate_opportunity_qualification(path)["passed"] is summary["passed"]
    with pytest.raises(FileExistsError, match="refusing to replace"):
        tools.publish_opportunity_qualification(document, path)

    invalid = dict(document)
    invalid["passed"] = not document["passed"]
    with pytest.raises(ValueError, match="top-level opportunity passed"):
        tools.opportunity_qualification_from_dict(invalid)


def test_opportunity_adapter_uses_explicit_manifest_and_r4_crossfit(
    tmp_path,
    monkeypatch,
):
    index_path = _founder_bank(tmp_path / "founders")
    bundle = tools.build_manifest_bundle(index_path, _seed_panel(), 1.5)
    manifest_path = tmp_path / "training.json"
    manifest_path.write_bytes(canonical_manifest_bytes(bundle.training_punctuated) + b"\n")
    disturbance_path = tmp_path / "disturbance.json"
    tools.publish_disturbance_qualification(
        _disturbance_document(
            index_path,
            multiplier=1.5,
            manifest_hash=manifest_sha256(bundle.training_punctuated),
        ),
        disturbance_path,
    )
    expected = [
        (world.founder_id, world.pair_id, context)
        for world in bundle.training_punctuated.worlds
        if world.event_kind is EventKind.ACTUATION_COST_SHIFT
        for context in ("pre", "post")
    ]
    records = [_opportunity_record(*item) for item in expected]
    founders = sorted({item[0] for item in expected})
    monkeypatch.setattr(tools.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(
        tools,
        "collect_opportunity_records",
        lambda _index, manifest: records,
    )
    monkeypatch.setattr(
        tools,
        "crossfit_opportunity",
        lambda _records: _opportunity_summary(founders),
    )
    output = tmp_path / "opportunity.json"
    result = tools.run_opportunity_qualification(
        index_path,
        manifest_path,
        disturbance_path,
        output,
    )

    assert result["passed"] is True
    assert [
        (record["founder_id"], record["pair_id"], record["context"])
        for record in result["records"]
    ] == expected
    assert {world.world_seed for world in bundle.training_punctuated.worlds} == {101, 102}


def test_failed_disturbance_document_must_exhaust_grid_and_remain_top_level_false(
    tmp_path,
):
    index_path = _founder_bank(tmp_path / "founders")
    index = load_founder_index(index_path, verify_artifacts=False)
    document = {
        "founder_index_sha256": founder_index_sha256(index),
        "passed": False,
        "protocol_revision": PROTOCOL_REVISION,
        "results": [
            _disturbance_result(multiplier, passed=False)
            for multiplier in ACTUATION_COST_MULTIPLIERS
        ],
        "schema_version": tools.DISTURBANCE_QUALIFICATION_SCHEMA_VERSION,
        "selected_multiplier": None,
        "simulator_source_sha256": tools.simulator_source_sha256(),
        "status": "stop_no_qualified_multiplier",
        "training_world_seeds": [101, 102],
        "world_qualification_sha256": "d" * 64,
    }
    normalized = tools.disturbance_qualification_from_dict(document)
    assert normalized["passed"] is False
    assert normalized["selected_multiplier"] is None

    incomplete = dict(document)
    incomplete["results"] = incomplete["results"][:-1]
    with pytest.raises(ValueError, match="must exhaust"):
        tools.disturbance_qualification_from_dict(incomplete)


def test_tools_do_not_embed_the_frozen_r4_world_seeds():
    source = Path(tools.__file__).read_text(encoding="utf-8")
    assert "4_101" not in source
    assert "4_102" not in source
    assert "4101" not in source
    assert "4102" not in source
