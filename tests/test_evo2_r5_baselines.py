import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.r5.baselines import (  # noqa: E402
    FIXED_OPERATOR_NAMES,
    HUMAN_CREDIT_NAME,
    HUMAN_STRESS_NAME,
    READABLE_SCALAR_COUNT,
    CandidateSource,
    fixed_operator_policy,
    human_candidate_sources,
    load_structured_random_roster,
    publish_structured_random_roster,
    structured_random_sources,
)
from microcosmos.cppn import canonical_cppn_genome  # noqa: E402
from microcosmos.heredity import R4_NUM_OPERATORS, make_mutation_context  # noqa: E402


EXPECTED_HUMAN_SOURCE_HASHES = {
    HUMAN_STRESS_NAME: "d25c5f4862f08b050bec2ee95f0870d91e7b6cda6cf06d60b16d276cfd0e6aba",
    HUMAN_CREDIT_NAME: "2e25ee2091a1b9e520896492bf343af40b7d80a946f95662bfa81f4316cd921c",
}
EXPECTED_STRUCTURED_ROSTER_HASH = (
    "fd80d64c6bd90145665640212156240dab9d58962828b77d1e5562946e0929ff"
)
EXPECTED_STRUCTURED_INDEX_HASH = (
    "dd4ba34eae792253aaf25b35014e8f99d67871ae2a77fd4c65cb0f186af5d4bb"
)


def _candidate_function(candidate: CandidateSource):
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - executes only deterministic sources generated in this module
        compile(candidate.source, f"<{candidate.candidate_id}>", "exec"),
        namespace,
        namespace,
    )
    return namespace["make_offspring"]


def _candidate_inputs(value: float):
    return (
        jnp.full((2,), value, dtype=jnp.float32),
        jnp.full((3,), value, dtype=jnp.float32),
        jnp.full((6,), value, dtype=jnp.float32),
        jnp.full((3, 6), value, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )


def _roster_hash(candidates: tuple[CandidateSource, ...]) -> str:
    payload = "\n".join(
        f"{candidate.candidate_id}:{candidate.sha256}" for candidate in candidates
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_fixed_operator_roster_maps_names_to_all_six_trusted_actions() -> None:
    assert FIXED_OPERATOR_NAMES == (
        "clone",
        "conservative_parametric",
        "standard_parametric",
        "exploratory_parametric",
        "structural",
        "mixed",
    )

    parent = canonical_cppn_genome()
    for expected_operator, name in enumerate(FIXED_OPERATOR_NAMES):
        context, valid = make_mutation_context(
            jax.random.PRNGKey(71), jnp.asarray(100 + expected_operator)
        )
        result = fixed_operator_policy(name)(parent, None, None, context)
        assert bool(valid)
        assert bool(result.policy_valid)
        assert int(result.operator_index) == expected_operator
        assert result.selection_probabilities.tolist() == [
            int(index == expected_operator) for index in range(R4_NUM_OPERATORS)
        ]

    with pytest.raises(KeyError, match="unknown fixed r5 operator"):
        fixed_operator_policy("not-an-operator")


def test_human_sources_and_formulas_are_prospectively_frozen() -> None:
    stress, credit = human_candidate_sources()
    assert (stress.candidate_id, credit.candidate_id) == (
        HUMAN_STRESS_NAME,
        HUMAN_CREDIT_NAME,
    )
    assert stress.family == credit.family == "human"
    assert {
        candidate.candidate_id: candidate.sha256 for candidate in (stress, credit)
    } == EXPECTED_HUMAN_SOURCE_HASHES

    stress_function = _candidate_function(stress)
    # Signed population change is neutral, and no births or deaths are occurring.
    stable_inputs = list(_candidate_inputs(1.0))
    stable_inputs[2] = jnp.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    stable = np.asarray(stress_function(*stable_inputs))
    assert stable.dtype == np.float32
    assert stable.tolist() == [-4.0, 2.0, 3.0, -2.0, -3.0, -1.0]

    stressed_inputs = list(_candidate_inputs(0.0))
    stressed_inputs[2] = jnp.asarray([0.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    stressed = np.asarray(stress_function(*stressed_inputs))
    assert np.allclose(
        stressed, [-3.0, -2.0, 0.0, 4.0, 1.0, 2.0], atol=1e-6
    )

    credit_function = _candidate_function(credit)
    population = jnp.asarray([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    success = np.asarray([0.9, 0.1, 0.5, 0.7, 0.2, 0.6], dtype=np.float32)
    usage = np.asarray([0.8, 0.2, 0.4, 0.1, 0.6, 0.3], dtype=np.float32)
    evidence = np.asarray([1.0, 1.0, 0.5, 0.75, 0.25, 0.0], dtype=np.float32)
    operator = jnp.asarray(np.stack([success, usage, evidence]))
    actual = np.asarray(
        credit_function(
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.ones((3,), dtype=jnp.float32),
            population,
            operator,
            jnp.asarray(123.0, dtype=jnp.float32),
        )
    )
    expected = np.clip(
        4.0 * evidence * (success - 0.5)
        + 1.5 * (1.0 - evidence)
        + 0.5 * (1.0 - usage),
        -8.0,
        8.0,
    ).astype(np.float32)
    assert actual.dtype == np.float32
    assert np.allclose(actual, expected)


def test_structured_random_roster_is_deterministic_and_matches_25_25_grammar() -> None:
    first = structured_random_sources()
    second = structured_random_sources()

    assert len(first) == 50
    assert [candidate.source for candidate in first] == [
        candidate.source for candidate in second
    ]
    assert [candidate.sha256 for candidate in first] == [
        candidate.sha256 for candidate in second
    ]
    assert len({candidate.candidate_id for candidate in first}) == 50
    assert len({candidate.sha256 for candidate in first}) == 50
    assert Counter(candidate.family for candidate in first) == {
        "threshold": 25,
        "affine": 25,
    }
    assert READABLE_SCALAR_COUNT == 29
    assert _roster_hash(first) == EXPECTED_STRUCTURED_ROSTER_HASH

    with pytest.raises(ValueError, match="frozen seed"):
        structured_random_sources(seed=50_006)

    zeros = _candidate_inputs(0.0)
    ones = _candidate_inputs(1.0)
    for candidate in first[:25]:
        function = _candidate_function(candidate)
        below = np.asarray(function(*zeros))
        above = np.asarray(function(*ones))
        assert below.shape == above.shape == (R4_NUM_OPERATORS,)
        assert below.dtype == above.dtype == np.float32
        assert sorted(below.tolist()) == [-8.0] * 5 + [8.0]
        assert sorted(above.tolist()) == [-8.0] * 5 + [8.0]
        assert int(np.argmax(below)) != int(np.argmax(above))

    # The largest source remains inside the current Shinka r4 AST/source limits,
    # and a representative from each grammar is eager- and JIT-compatible.
    for candidate in first:
        tree = ast.parse(candidate.source)
        assert len(candidate.source.encode("utf-8")) <= 12_000
        assert sum(bool(line.strip()) for line in candidate.source.splitlines()) <= 60
        assert len(list(ast.walk(tree))) <= 1_024
        assert hashlib.sha256(candidate.source.encode("utf-8")).hexdigest() == candidate.sha256
    for candidate in (first[0], first[25]):
        function = _candidate_function(candidate)
        eager = np.asarray(function(*_candidate_inputs(0.5)))
        compiled = np.asarray(jax.jit(function)(*_candidate_inputs(0.5)))
        assert eager.shape == (R4_NUM_OPERATORS,)
        assert eager.dtype == np.float32
        assert np.array_equal(eager, compiled)
        assert np.all(np.isfinite(eager))
        assert np.all(np.abs(eager) <= 8.0)


def test_structured_random_publication_is_atomic_hashed_and_write_once(tmp_path) -> None:
    destination = tmp_path / "structured-random"
    result = publish_structured_random_roster(destination)
    assert result == {
        "candidate_count": 50,
        "index_sha256": EXPECTED_STRUCTURED_INDEX_HASH,
    }

    index_bytes = (destination / "index.json").read_bytes()
    assert hashlib.sha256(index_bytes).hexdigest() == EXPECTED_STRUCTURED_INDEX_HASH
    assert (destination / "index.sha256").read_text(encoding="ascii") == (
        f"{EXPECTED_STRUCTURED_INDEX_HASH}  index.json\n"
    )
    index = json.loads(index_bytes)
    assert index["seed"] == 50_005
    assert index["candidate_count"] == 50
    assert index["grammar"] == {
        "affine": 25,
        "readable_scalar_count": 29,
        "threshold": 25,
        "threshold_grid": [index / 10 for index in range(1, 10)],
    }
    assert Counter(record["family"] for record in index["records"]) == {
        "threshold": 25,
        "affine": 25,
    }
    assert len(index["records"]) == 50
    for record in index["records"]:
        source_path = destination / record["path"]
        assert source_path.parent == destination
        assert source_path.name == f"{record['candidate_id']}.py"
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == record["sha256"]
        assert source_path.stat().st_mode & 0o222 == 0
    assert (destination / "index.json").stat().st_mode & 0o222 == 0
    assert (destination / "index.sha256").stat().st_mode & 0o222 == 0
    assert len(load_structured_random_roster(destination)) == 50

    assert publish_structured_random_roster(destination) == result

    first_source = destination / index["records"][0]["path"]
    first_source.chmod(0o644)
    first_source.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        load_structured_random_roster(destination)
