from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import jax
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem import build_founder_bank as builder  # noqa: E402
from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: E402
    canonical_founder_artifact_bytes,
    load_founder_index,
)


def _result(spec, *, passed: bool = True) -> builder.CandidateScreenResult:
    gates = spec.gates
    return builder.CandidateScreenResult(
        graph_valid=passed,
        cache_valid=passed,
        seed_results=tuple(
            builder.SeedScreenResult(
                seed=seed,
                finite=passed,
                integrity_valid=passed,
                final_alive=gates.minimum_final_alive if passed else 0,
                birth_count=gates.minimum_births if passed else 0,
                maximum_generation=gates.minimum_generation if passed else 0,
                productivity=gates.minimum_productivity if passed else 0.0,
            )
            for seed in spec.seeds
        ),
    )


def _candidate_indices(spec):
    return {canonical_founder_artifact_bytes(genome)[1].artifact_sha256: index for index, genome in enumerate(builder.candidate_pool(spec))}


def _canonical_lines(path: Path) -> list[dict]:
    payloads = path.read_bytes().splitlines(keepends=True)
    values = [json.loads(payload) for payload in payloads]
    for payload, value in zip(payloads, values, strict=True):
        assert payload == builder._canonical_json_bytes(value) + b"\n"
    assert [value["sequence"] for value in values] == list(range(len(values)))
    return values


def test_specs_prepartition_candidates_and_keep_scientific_contracts_distinct():
    production = builder.PRODUCTION_SPEC
    smoke = builder.SMOKE_SPEC

    assert [asdict(item) for item in production.candidate_ranges] == [
        {"partition": "training", "start": 0, "stop": 16, "required": 3},
        {
            "partition": "development",
            "start": 16,
            "stop": 32,
            "required": 2,
        },
        {"partition": "sealed", "start": 32, "stop": 48, "required": 3},
    ]
    assert tuple(item.required for item in smoke.candidate_ranges) == (3, 2, 3)
    assert production.config.fluid_enabled
    assert not smoke.config.fluid_enabled
    assert production.horizon == 8_000
    assert smoke.horizon == 4
    assert builder._selection_rule(production)["injury_screening"] is False
    assert builder.selection_rule_sha256(production) == builder.selection_rule_sha256(production)


def test_execution_contract_rejects_cpu_production_and_gpu_smoke():
    with pytest.raises(RuntimeError, match="GPU backend"):
        builder.validate_execution_contract(builder.PRODUCTION_SPEC, backend="cpu")
    with pytest.raises(RuntimeError, match="CPU backend"):
        builder.validate_execution_contract(builder.SMOKE_SPEC, backend="gpu")

    builder.validate_execution_contract(builder.PRODUCTION_SPEC, backend="gpu")
    builder.validate_execution_contract(builder.SMOKE_SPEC, backend="cpu")


def test_configured_r4_contract_accepts_caller_frozen_horizon_only():
    configured = replace(
        builder.R4_PRODUCTION_SPEC,
        seeds=(12_000, 12_001),
        horizon=7_500,
        selection_rule_id="prospective-r6-rule",
    )
    builder.validate_configured_r4_execution_contract(configured, backend="gpu")

    with pytest.raises(RuntimeError, match="may change only"):
        builder.validate_configured_r4_execution_contract(
            replace(configured, chunk_steps=250),
            backend="gpu",
        )


def test_production_builder_rejects_backend_or_runner_injection(tmp_path):
    common = {
        "smoke": False,
        "bank_directory": tmp_path / "bank",
        "screening_record_path": tmp_path / "screening.jsonl",
    }
    with pytest.raises(RuntimeError, match="forbids injected"):
        builder.run_founder_bank_builder(
            **common,
            runner=lambda _genome, spec: _result(spec),
        )
    with pytest.raises(RuntimeError, match="backend overrides"):
        builder.run_founder_bank_builder(**common, backend="gpu")


def test_success_screens_every_preassigned_candidate_before_atomic_publication(
    tmp_path,
    monkeypatch,
):
    spec = builder.SMOKE_SPEC
    indices = _candidate_indices(spec)
    screened = []
    artifact_writes = []
    original_write = builder.write_founder_artifact

    def runner(genome, received_spec):
        assert received_spec is spec
        digest = canonical_founder_artifact_bytes(genome)[1].artifact_sha256
        screened.append(indices[digest])
        return _result(spec)

    def write_after_screening(path, genome):
        assert screened == list(range(spec.candidate_count))
        artifact_writes.append(Path(path).name)
        return original_write(path, genome)

    monkeypatch.setattr(builder, "write_founder_artifact", write_after_screening)
    bank = tmp_path / "bank"
    audit = tmp_path / "screening.jsonl"
    result = builder.run_founder_bank_builder(
        smoke=True,
        bank_directory=bank,
        screening_record_path=audit,
        runner=runner,
        backend="cpu",
    )

    assert result["status"] == "published"
    assert screened == list(range(12))
    assert result["selected_candidate_indices"] == {
        "training": [0, 1, 2],
        "development": [4, 5],
        "sealed": [8, 9, 10],
    }
    assert len(artifact_writes) == 8
    index = load_founder_index(bank / "index.json")
    assert [record.founder_id for record in index.founders] == [
        "train-00",
        "train-01",
        "train-02",
        "dev-00",
        "dev-01",
        "sealed-00",
        "sealed-01",
        "sealed-02",
    ]
    assert {record.selection_rule for record in index.founders} == {builder.SELECTION_RULE_ID}

    lines = _canonical_lines(audit)
    screened_lines = [value for value in lines if value["event"] == "candidate_screened"]
    assert len(screened_lines) == spec.candidate_count
    assert [value["candidate_index"] for value in screened_lines] == list(range(spec.candidate_count))
    assert all(value["gate"]["passed"] for value in screened_lines)
    assert lines[0]["protocol"]["screening_environment"] == "uninjured_only"
    assert lines[-1]["bank_published"] is True


def test_insufficient_partition_keeps_bank_absent_and_records_every_outcome(
    tmp_path,
    monkeypatch,
):
    spec = builder.SMOKE_SPEC
    indices = _candidate_indices(spec)
    attempted_writes = []

    def runner(genome, _):
        digest = canonical_founder_artifact_bytes(genome)[1].artifact_sha256
        return _result(spec, passed=indices[digest] < 8)

    def unexpected_write(*args, **kwargs):
        attempted_writes.append((args, kwargs))
        raise AssertionError("artifacts must not be written before every quota passes")

    monkeypatch.setattr(builder, "write_founder_artifact", unexpected_write)
    bank = tmp_path / "bank"
    audit = tmp_path / "screening.jsonl"
    result = builder.run_founder_bank_builder(
        smoke=True,
        bank_directory=bank,
        screening_record_path=audit,
        runner=runner,
        backend="cpu",
    )

    assert result["status"] == "insufficient_passing_candidates"
    assert not result["bank_published"]
    assert not bank.exists()
    assert not attempted_writes
    lines = _canonical_lines(audit)
    screened = [value for value in lines if value["event"] == "candidate_screened"]
    assert len(screened) == spec.candidate_count
    assert all(value["gate"]["passed"] is False for value in screened if value["partition"] == "sealed")
    assert lines[-1] == {
        "bank_published": False,
        "event": "screening_finished",
        "protocol_sha256": result["protocol_sha256"],
        "schema_version": builder.SCREENING_SCHEMA_VERSION,
        "sequence": lines[-1]["sequence"],
        "status": "insufficient_passing_candidates",
    }


def test_outputs_are_write_once(tmp_path):
    bank = tmp_path / "bank"
    audit = tmp_path / "screening.jsonl"

    builder.run_founder_bank_builder(
        smoke=True,
        bank_directory=bank,
        screening_record_path=audit,
        runner=lambda _genome, spec: _result(spec),
        backend="cpu",
    )
    with pytest.raises(FileExistsError, match="immutable founder bank"):
        builder.run_founder_bank_builder(
            smoke=True,
            bank_directory=bank,
            screening_record_path=tmp_path / "second.jsonl",
            runner=lambda _genome, spec: _result(spec),
            backend="cpu",
        )
    with pytest.raises(FileExistsError, match="append-only screening record"):
        builder.run_founder_bank_builder(
            smoke=True,
            bank_directory=tmp_path / "other-bank",
            screening_record_path=audit,
            runner=lambda _genome, spec: _result(spec),
            backend="cpu",
        )


def test_actual_smoke_runner_checks_uninjured_finite_clone_viability():
    if jax.default_backend() != "cpu":
        pytest.skip("the founder-bank smoke runner is explicitly CPU-only")
    spec = builder.SMOKE_SPEC
    result = builder.make_screen_runner(spec)(builder.candidate_pool(spec)[0], spec)
    passed, reasons, summary = builder._gate_result(result, spec)

    assert passed, reasons
    assert result.graph_valid
    assert result.cache_valid
    assert tuple(item.seed for item in result.seed_results) == spec.seeds
    assert summary["all_finite"]
    assert summary["all_integrity_valid"]
