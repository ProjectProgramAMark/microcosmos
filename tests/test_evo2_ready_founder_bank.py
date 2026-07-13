"""Pre-injury-ready founder-bank workflow."""

import json
from pathlib import Path
import sys

import jax
import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from experiments.evo2_ecosystem.build_founder_bank import (  # noqa: E402
    CandidateScreenResult,
    SeedScreenResult,
)
from experiments.evo2_ecosystem import build_ready_founder_bank as ready_module  # noqa: E402
from experiments.evo2_ecosystem.build_ready_founder_bank import (  # noqa: E402
    PARTITION_SEED_COUNTS,
    PRODUCTION_SPEC,
    READINESS_STEP,
    SMOKE_SPEC,
    qualify_seeds,
    run_ready_founder_bank_builder,
)
from experiments.evo2_ecosystem.founder_artifacts import (  # noqa: E402
    load_founder_index,
)


def _seed_result(seed: int, *, passed: bool = True) -> SeedScreenResult:
    return SeedScreenResult(
        seed=seed,
        finite=True,
        integrity_valid=True,
        final_alive=2 if passed else 0,
        birth_count=2 if passed else 0,
        maximum_generation=1 if passed else 0,
        productivity=0.01 if passed else 0.0,
    )


def test_production_contract_is_preinjury_and_partitioned() -> None:
    assert PRODUCTION_SPEC.screening.horizon == READINESS_STEP == 4_500
    assert PRODUCTION_SPEC.screening.chunk_steps == 500
    assert dict(PRODUCTION_SPEC.partition_seed_counts) == PARTITION_SEED_COUNTS
    assert tuple(item.required for item in PRODUCTION_SPEC.screening.candidate_ranges) == (
        3,
        2,
        3,
    )


def test_seed_qualification_uses_first_passing_worlds_in_fixed_order() -> None:
    passing = set(PRODUCTION_SPEC.seed_candidates[1:15])

    def runner(_):
        return CandidateScreenResult(
            graph_valid=True,
            cache_valid=True,
            seed_results=tuple(
                _seed_result(seed, passed=seed in passing)
                for seed in PRODUCTION_SPEC.seed_candidates
            ),
        )

    assignments, _ = qualify_seeds(PRODUCTION_SPEC, runner)
    selected = tuple(
        seed
        for partition in ("training", "development", "sealed")
        for seed in assignments[partition]
    )
    assert selected == PRODUCTION_SPEC.seed_candidates[1:14]
    assert len(assignments["training"]) == 3
    assert len(assignments["development"]) == 4
    assert len(assignments["sealed"]) == 6
    assert len(set(selected)) == 13


def test_seed_qualification_fails_closed_when_too_few_worlds_are_ready() -> None:
    passing = set(PRODUCTION_SPEC.seed_candidates[:12])

    def runner(_):
        return CandidateScreenResult(
            graph_valid=True,
            cache_valid=True,
            seed_results=tuple(
                _seed_result(seed, passed=seed in passing)
                for seed in PRODUCTION_SPEC.seed_candidates
            ),
        )

    assignments, _ = qualify_seeds(PRODUCTION_SPEC, runner)
    assert assignments == {"training": (), "development": (), "sealed": ()}


def test_actual_smoke_publishes_verified_bank_and_seed_panel(tmp_path: Path) -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("the readiness smoke contract is explicitly CPU-only")
    bank = tmp_path / "bank"
    record = tmp_path / "screening.jsonl"

    result = run_ready_founder_bank_builder(
        smoke=True,
        bank_directory=bank,
        screening_record_path=record,
    )

    assert result["status"] == "published"
    assert result["bank_published"] is True
    index = load_founder_index(bank / "index.json", verify_artifacts=True)
    assert [item.partition for item in index.founders].count("training") == 3
    assert [item.partition for item in index.founders].count("development") == 2
    assert [item.partition for item in index.founders].count("sealed") == 3
    seed_panel = json.loads((bank / "readiness_seed_panel.json").read_text())
    assert {key: len(value) for key, value in seed_panel["partitions"].items()} == (
        PARTITION_SEED_COUNTS
    )
    events = [json.loads(line) for line in record.read_text().splitlines()]
    assert events[0]["event"] == "readiness_screen_started"
    assert events[-1]["event"] == "readiness_screen_finished"
    assert events[-1]["status"] == "published"

    with pytest.raises(FileExistsError, match="refusing to replace"):
        run_ready_founder_bank_builder(
            smoke=True,
            bank_directory=bank,
            screening_record_path=tmp_path / "second.jsonl",
        )


def test_production_forbids_injected_runners(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="forbids injected"):
        run_ready_founder_bank_builder(
            smoke=False,
            bank_directory=tmp_path / "bank",
            screening_record_path=tmp_path / "record.jsonl",
            seed_runner=lambda _: CandidateScreenResult(True, True, ()),
        )


def test_production_requires_frozen_preregistration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ready_module.jax, "default_backend", lambda: "gpu")
    with pytest.raises(RuntimeError, match="requires preregistration"):
        run_ready_founder_bank_builder(
            smoke=False,
            bank_directory=tmp_path / "bank",
            screening_record_path=tmp_path / "record.jsonl",
        )
    draft = tmp_path / "draft.md"
    draft.write_text("Status: DRAFT\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not frozen"):
        run_ready_founder_bank_builder(
            smoke=False,
            bank_directory=tmp_path / "bank",
            screening_record_path=tmp_path / "record.jsonl",
            preregistration_path=draft,
        )


def test_seed_runner_must_cover_exact_candidate_pool() -> None:
    with pytest.raises(ValueError, match="frozen candidate pool"):
        qualify_seeds(
            SMOKE_SPEC,
            lambda _: CandidateScreenResult(
                graph_valid=True,
                cache_valid=True,
                seed_results=(_seed_result(0),),
            ),
        )
