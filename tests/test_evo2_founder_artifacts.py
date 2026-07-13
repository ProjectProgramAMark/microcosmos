from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import jax.numpy as jnp
import numpy as np
import pytest

# ``experiments`` is intentionally not part of the Microcosmos wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evo2_ecosystem.founder_artifacts import (
    ARTIFACT_MEMBERS,
    CONNECTION_MEMBER,
    NODE_MEMBER,
    FounderIndex,
    FounderRecord,
    canonical_founder_artifact_bytes,
    canonical_founder_index_bytes,
    founder_index_from_json_bytes,
    founder_index_sha256,
    load_founder,
    load_founder_artifact,
    load_founder_index,
    make_founder_record,
    validate_disjoint_partitions,
    write_founder_artifact,
    write_founder_index,
)  # noqa: E402
from experiments.evo2_ecosystem.protocol import CONTROLLER_LAYOUT  # noqa: E402
from microcosmos.cppn import (
    CONNECTION_WEIGHT,
    CPPNGenome,
    canonical_cppn_genome,
)  # noqa: E402


def _founder(weight_delta: float = 0.0) -> CPPNGenome:
    genome = canonical_cppn_genome()
    return CPPNGenome(
        node_genes=genome.node_genes,
        connection_genes=genome.connection_genes.at[0, CONNECTION_WEIGHT].add(weight_delta),
    )


def _write_record(
    root: Path,
    *,
    founder_id: str,
    partition: str,
    weight_delta: float,
    selection_seed: int,
) -> FounderRecord:
    artifact = f"{founder_id}.npz"
    digests = write_founder_artifact(root / artifact, _founder(weight_delta))
    return make_founder_record(
        founder_id=founder_id,
        partition=partition,
        artifact=artifact,
        digests=digests,
        selection_rule="uninjured_viability_only",
        selection_seed=selection_seed,
    )


def test_artifact_encoding_is_deterministic_numeric_and_write_once(tmp_path):
    first, first_digests = canonical_founder_artifact_bytes(_founder())
    second, second_digests = canonical_founder_artifact_bytes(_founder())
    assert first == second
    assert first_digests == second_digests
    assert hashlib.sha256(first).hexdigest() == first_digests.artifact_sha256

    artifact = tmp_path / "train-00.npz"
    written = write_founder_artifact(artifact, _founder())
    assert written == first_digests
    assert artifact.read_bytes() == first
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_founder_artifact(artifact, _founder(0.1))
    assert artifact.read_bytes() == first

    with zipfile.ZipFile(artifact) as archive:
        assert tuple(archive.namelist()) == ARTIFACT_MEMBERS
        with np.load(artifact, allow_pickle=False) as arrays:
            assert set(arrays.files) == {"node_genes", "connection_genes"}
            assert arrays["node_genes"].shape == (15, 5)
            assert arrays["connection_genes"].shape == (30, 3)
            assert arrays["node_genes"].dtype == np.float32
            assert arrays["connection_genes"].dtype == np.float32
        assert hashlib.sha256(archive.read(NODE_MEMBER)).hexdigest() == written.node_genes_sha256
        assert hashlib.sha256(archive.read(CONNECTION_MEMBER)).hexdigest() == written.connection_genes_sha256


def test_index_and_artifacts_round_trip_canonically_with_strict_partition(tmp_path):
    training = _write_record(
        tmp_path,
        founder_id="train-00",
        partition="training",
        weight_delta=0.0,
        selection_seed=100,
    )
    development = _write_record(
        tmp_path,
        founder_id="dev-00",
        partition="development",
        weight_delta=0.1,
        selection_seed=200,
    )
    sealed = _write_record(
        tmp_path,
        founder_id="sealed-00",
        partition="sealed",
        weight_delta=0.2,
        selection_seed=300,
    )
    index = FounderIndex(founders=(training, development, sealed))
    index_path = tmp_path / "index.json"

    digest = write_founder_index(index_path, index)
    assert digest == founder_index_sha256(index)
    assert index_path.read_bytes() == canonical_founder_index_bytes(index) + b"\n"
    assert load_founder_index(index_path) == index

    loaded = load_founder(
        index_path,
        "dev-00",
        expected_partition="development",
    )
    assert jnp.array_equal(
        loaded.connection_genes,
        _founder(0.1).connection_genes,
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="belongs to"):
        load_founder(index_path, "dev-00", expected_partition="training")
    with pytest.raises(KeyError, match="unknown founder_id"):
        index.record("missing", expected_partition="training")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        write_founder_index(index_path, index)


def test_index_rejects_noncanonical_unknown_duplicate_and_unsafe_metadata(tmp_path):
    record = _write_record(
        tmp_path,
        founder_id="train-00",
        partition="training",
        weight_delta=0.0,
        selection_seed=100,
    )
    index = FounderIndex(founders=(record,))
    value = json.loads(canonical_founder_index_bytes(index))

    pretty = (json.dumps(value, indent=2) + "\n").encode()
    with pytest.raises(ValueError, match="one canonical JSON line|not canonically encoded"):
        founder_index_from_json_bytes(pretty)

    value["unexpected"] = True
    with pytest.raises(ValueError, match="invalid root keys"):
        founder_index_from_json_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )

    duplicate_key = canonical_founder_index_bytes(index).replace(
        b'{"founders":',
        b'{"founders":[],"founders":',
        1,
    )
    with pytest.raises(ValueError, match="invalid founder index JSON"):
        founder_index_from_json_bytes(duplicate_key + b"\n")

    value = json.loads(canonical_founder_index_bytes(index))
    value["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        founder_index_from_json_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        )

    with pytest.raises(ValueError, match="safe relative"):
        replace(record, artifact="../escape.npz")
    with pytest.raises(ValueError, match="partition"):
        replace(record, partition="sealed_final")
    with pytest.raises(ValueError, match="controller_layout"):
        replace(record, controller_layout="other-layout")
    with pytest.raises(ValueError, match="selection_seed"):
        replace(record, selection_seed=True)


def test_disjoint_validation_rejects_reused_identity_path_artifact_or_genotype(tmp_path):
    training = _write_record(
        tmp_path,
        founder_id="train-00",
        partition="training",
        weight_delta=0.0,
        selection_seed=100,
    )
    development = _write_record(
        tmp_path,
        founder_id="dev-00",
        partition="development",
        weight_delta=0.1,
        selection_seed=200,
    )
    mutations = (
        replace(development, founder_id=training.founder_id),
        replace(development, artifact=training.artifact),
        replace(development, artifact_sha256=training.artifact_sha256),
        replace(
            development,
            node_genes_sha256=training.node_genes_sha256,
            connection_genes_sha256=training.connection_genes_sha256,
        ),
    )
    for duplicate in mutations:
        with pytest.raises(ValueError, match="duplicate"):
            validate_disjoint_partitions((training, duplicate))


def test_loader_fails_closed_on_hash_member_layout_and_graph_corruption(tmp_path):
    record = _write_record(
        tmp_path,
        founder_id="train-00",
        partition="training",
        weight_delta=0.0,
        selection_seed=100,
    )

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_founder_artifact(
            tmp_path,
            replace(record, artifact_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="node_genes hash mismatch"):
        load_founder_artifact(
            tmp_path,
            replace(record, node_genes_sha256="0" * 64),
        )
    with pytest.raises(ValueError, match="belongs to"):
        load_founder_artifact(tmp_path, record, expected_partition="development")
    with pytest.raises(ValueError, match="runtime"):
        load_founder_artifact(
            tmp_path,
            record,
            expected_controller_layout="cppn-other",
        )

    invalid = _founder()
    invalid = CPPNGenome(
        invalid.node_genes,
        invalid.connection_genes.at[0, 0].set(999.0),
    )
    with pytest.raises(ValueError, match="valid CPPN"):
        write_founder_artifact(tmp_path / "invalid.npz", invalid)


def test_index_loader_verifies_every_referenced_artifact(tmp_path):
    record = _write_record(
        tmp_path,
        founder_id="train-00",
        partition="training",
        weight_delta=0.0,
        selection_seed=100,
    )
    index_path = tmp_path / "index.json"
    write_founder_index(index_path, FounderIndex(founders=(record,)))
    (tmp_path / record.artifact).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_founder_index(index_path)
    assert load_founder_index(index_path, verify_artifacts=False).founders == (record,)


def test_record_matches_frozen_controller_layout(tmp_path):
    record = _write_record(
        tmp_path,
        founder_id="train-00",
        partition="training",
        weight_delta=0.0,
        selection_seed=100,
    )
    assert record.controller_layout == CONTROLLER_LAYOUT
