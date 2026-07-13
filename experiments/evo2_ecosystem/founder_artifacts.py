"""Trusted, deterministic storage for frozen Evo² CPPN founders.

Founder artifacts are deliberately boring: one uncompressed NPZ containing two
float32 NPY members, accompanied by a canonical JSON index.  Pickle/object
arrays, archive metadata variability, implicit layout conversion, and path
traversal are all rejected.  This module creates and validates artifacts; it
does not generate, screen, or partition the final founder bank.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable
import zipfile

import jax.numpy as jnp
import numpy as np

from microcosmos.cppn import (
    MAX_CONNECTIONS,
    MAX_NODES,
    CPPNGenome,
    transform_and_validate_genome,
)

from .protocol import CONTROLLER_LAYOUT


FOUNDER_INDEX_SCHEMA_VERSION = 1
FOUNDER_PARTITIONS = frozenset({"training", "development", "sealed"})
NODE_MEMBER = "node_genes.npy"
CONNECTION_MEMBER = "connection_genes.npy"
ARTIFACT_MEMBERS = (NODE_MEMBER, CONNECTION_MEMBER)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class FounderArtifactDigests:
    """Hashes of one exact artifact and its two encoded NPY members."""

    artifact_sha256: str
    node_genes_sha256: str
    connection_genes_sha256: str

    def __post_init__(self) -> None:
        _require_sha256("artifact_sha256", self.artifact_sha256)
        _require_sha256("node_genes_sha256", self.node_genes_sha256)
        _require_sha256("connection_genes_sha256", self.connection_genes_sha256)


@dataclass(frozen=True)
class FounderRecord:
    """One immutable founder-bank index entry."""

    founder_id: str
    partition: str
    artifact: str
    artifact_sha256: str
    node_genes_sha256: str
    connection_genes_sha256: str
    controller_layout: str
    selection_rule: str
    selection_seed: int

    def __post_init__(self) -> None:
        _require_identifier("founder_id", self.founder_id)
        _require_partition(self.partition)
        _require_artifact_path(self.artifact)
        _require_sha256("artifact_sha256", self.artifact_sha256)
        _require_sha256("node_genes_sha256", self.node_genes_sha256)
        _require_sha256("connection_genes_sha256", self.connection_genes_sha256)
        if self.controller_layout != CONTROLLER_LAYOUT:
            raise ValueError(f"controller_layout must be {CONTROLLER_LAYOUT!r}")
        _require_identifier("selection_rule", self.selection_rule)
        if not isinstance(self.selection_seed, int) or isinstance(self.selection_seed, bool) or self.selection_seed < 0:
            raise ValueError("selection_seed must be a nonnegative integer")

    @property
    def digests(self) -> FounderArtifactDigests:
        return FounderArtifactDigests(
            artifact_sha256=self.artifact_sha256,
            node_genes_sha256=self.node_genes_sha256,
            connection_genes_sha256=self.connection_genes_sha256,
        )


@dataclass(frozen=True)
class FounderIndex:
    """Canonical founder-bank index with globally disjoint partitions."""

    founders: tuple[FounderRecord, ...]
    schema_version: int = FOUNDER_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != FOUNDER_INDEX_SCHEMA_VERSION
        ):
            raise ValueError(f"schema_version must be {FOUNDER_INDEX_SCHEMA_VERSION}")
        if not isinstance(self.founders, tuple):
            raise TypeError("founders must be a tuple")
        if not self.founders:
            raise ValueError("founders must not be empty")
        if not all(isinstance(founder, FounderRecord) for founder in self.founders):
            raise TypeError("every founders entry must be a FounderRecord")
        validate_disjoint_partitions(self.founders)

    def record(self, founder_id: str, *, expected_partition: str | None = None) -> FounderRecord:
        """Return one exact founder identity and optionally enforce its partition."""
        matches = [founder for founder in self.founders if founder.founder_id == founder_id]
        if len(matches) != 1:
            raise KeyError(f"unknown founder_id: {founder_id!r}")
        record = matches[0]
        if expected_partition is not None:
            _require_partition(expected_partition)
            if record.partition != expected_partition:
                raise ValueError(
                    f"founder {founder_id!r} belongs to {record.partition!r}, "
                    f"not {expected_partition!r}"
                )
        return record


def _require_identifier(name: str, value: object) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be a portable identifier")


def _require_partition(value: object) -> None:
    if value not in FOUNDER_PARTITIONS:
        raise ValueError(f"partition must be one of {sorted(FOUNDER_PARTITIONS)}")


def _require_sha256(name: str, value: object) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_artifact_path(value: object) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("artifact must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.suffix != ".npz" or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("artifact must be a safe relative .npz path")
    if path.as_posix() != value:
        raise ValueError("artifact must use canonical POSIX syntax")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_array(array: Any, *, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    if value.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must have dtype float32")
    return np.ascontiguousarray(value, dtype=np.dtype("<f4"))


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False, version=(2, 0))
    return buffer.getvalue()


def _zip_member(name: str) -> zipfile.ZipInfo:
    member = zipfile.ZipInfo(filename=name, date_time=_FIXED_ZIP_TIMESTAMP)
    member.compress_type = zipfile.ZIP_STORED
    member.create_system = 3
    member.external_attr = 0o100644 << 16
    return member


def _canonicalize_and_validate_genome(genome: CPPNGenome) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(genome, CPPNGenome):
        raise TypeError("genome must be a CPPNGenome")
    nodes = _canonical_array(
        genome.node_genes,
        expected_shape=(MAX_NODES, 5),
        name="node_genes",
    )
    connections = _canonical_array(
        genome.connection_genes,
        expected_shape=(MAX_CONNECTIONS, 3),
        name="connection_genes",
    )
    canonical, _, _, valid = transform_and_validate_genome(
        CPPNGenome(jnp.asarray(nodes), jnp.asarray(connections))
    )
    if not bool(valid):
        raise ValueError("founder is not a valid CPPN genome")
    return (
        _canonical_array(
            canonical.node_genes,
            expected_shape=(MAX_NODES, 5),
            name="node_genes",
        ),
        _canonical_array(
            canonical.connection_genes,
            expected_shape=(MAX_CONNECTIONS, 3),
            name="connection_genes",
        ),
    )


def canonical_founder_artifact_bytes(genome: CPPNGenome) -> tuple[bytes, FounderArtifactDigests]:
    """Encode a valid genome into reproducible, pickle-free NPZ bytes."""
    nodes, connections = _canonicalize_and_validate_genome(genome)
    member_payloads = {
        NODE_MEMBER: _npy_bytes(nodes),
        CONNECTION_MEMBER: _npy_bytes(connections),
    }
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        for name in ARTIFACT_MEMBERS:
            archive.writestr(_zip_member(name), member_payloads[name])
    payload = buffer.getvalue()
    return payload, FounderArtifactDigests(
        artifact_sha256=_sha256(payload),
        node_genes_sha256=_sha256(member_payloads[NODE_MEMBER]),
        connection_genes_sha256=_sha256(member_payloads[CONNECTION_MEMBER]),
    )


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


def write_founder_artifact(path: str | Path, genome: CPPNGenome) -> FounderArtifactDigests:
    """Atomically create one deterministic artifact, refusing all replacement."""
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("founder artifact path must end in .npz")
    payload, digests = canonical_founder_artifact_bytes(genome)
    _write_once(destination, payload)
    return digests


def make_founder_record(
    *,
    founder_id: str,
    partition: str,
    artifact: str,
    digests: FounderArtifactDigests,
    selection_rule: str,
    selection_seed: int,
) -> FounderRecord:
    """Build a validated index record from a completed artifact write."""
    return FounderRecord(
        founder_id=founder_id,
        partition=partition,
        artifact=artifact,
        artifact_sha256=digests.artifact_sha256,
        node_genes_sha256=digests.node_genes_sha256,
        connection_genes_sha256=digests.connection_genes_sha256,
        controller_layout=CONTROLLER_LAYOUT,
        selection_rule=selection_rule,
        selection_seed=selection_seed,
    )


def validate_disjoint_partitions(founders: Iterable[FounderRecord]) -> None:
    """Reject reused IDs, paths, archive bytes, or genotypes across the bank."""
    records = tuple(founders)
    identities: dict[str, str] = {}
    paths: dict[str, str] = {}
    artifacts: dict[str, str] = {}
    genotypes: dict[tuple[str, str], str] = {}
    for record in records:
        if not isinstance(record, FounderRecord):
            raise TypeError("partition entries must be FounderRecord instances")
        checks = (
            (identities, record.founder_id, "founder_id"),
            (paths, record.artifact, "artifact path"),
            (artifacts, record.artifact_sha256, "artifact hash"),
            (
                genotypes,
                (record.node_genes_sha256, record.connection_genes_sha256),
                "genotype hash",
            ),
        )
        for seen, value, label in checks:
            previous = seen.get(value)
            if previous is not None:
                raise ValueError(
                    f"duplicate {label} across founder partitions: "
                    f"{previous!r} and {record.founder_id!r}"
                )
            seen[value] = record.founder_id


def _record_dict(record: FounderRecord) -> dict[str, object]:
    return {
        "artifact": record.artifact,
        "artifact_sha256": record.artifact_sha256,
        "connection_genes_sha256": record.connection_genes_sha256,
        "controller_layout": record.controller_layout,
        "founder_id": record.founder_id,
        "node_genes_sha256": record.node_genes_sha256,
        "partition": record.partition,
        "selection_rule": record.selection_rule,
        "selection_seed": record.selection_seed,
    }


def canonical_founder_index_bytes(index: FounderIndex) -> bytes:
    """Return exact canonical JSON bytes, excluding the storage newline."""
    if not isinstance(index, FounderIndex):
        raise TypeError("index must be a FounderIndex")
    value = {
        "founders": [_record_dict(record) for record in index.founders],
        "schema_version": index.schema_version,
    }
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def founder_index_sha256(index: FounderIndex) -> str:
    return _sha256(canonical_founder_index_bytes(index))


def write_founder_index(path: str | Path, index: FounderIndex) -> str:
    """Write one canonical index exactly once and return its canonical hash."""
    destination = Path(path)
    if destination.suffix != ".json":
        raise ValueError("founder index path must end in .json")
    payload = canonical_founder_index_bytes(index)
    _write_once(destination, payload + b"\n")
    return _sha256(payload)


def _founder_record_from_dict(value: object) -> FounderRecord:
    if not isinstance(value, dict):
        raise ValueError("founder entry must be an object")
    expected = {
        "artifact",
        "artifact_sha256",
        "connection_genes_sha256",
        "controller_layout",
        "founder_id",
        "node_genes_sha256",
        "partition",
        "selection_rule",
        "selection_seed",
    }
    if set(value) != expected:
        raise ValueError("founder entry has invalid keys")
    return FounderRecord(**value)


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def founder_index_from_json_bytes(payload: bytes, *, require_canonical: bool = True) -> FounderIndex:
    """Parse a strict founder index and optionally require its exact encoding."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    encoded = payload[:-1] if payload.endswith(b"\n") else payload
    if b"\n" in encoded or b"\r" in encoded:
        raise ValueError("founder index must be one canonical JSON line")
    try:
        value = json.loads(encoded, object_pairs_hook=_strict_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("invalid founder index JSON") from error
    if not isinstance(value, dict) or set(value) != {"founders", "schema_version"}:
        raise ValueError("founder index has invalid root keys")
    if not isinstance(value["founders"], list):
        raise ValueError("founders must be an array")
    index = FounderIndex(
        schema_version=value["schema_version"],
        founders=tuple(_founder_record_from_dict(entry) for entry in value["founders"]),
    )
    if require_canonical and encoded != canonical_founder_index_bytes(index):
        raise ValueError("founder index is not canonically encoded")
    if require_canonical and payload != encoded + b"\n":
        raise ValueError("canonical founder index file must end with one newline")
    return index


def load_founder_index(path: str | Path, *, verify_artifacts: bool = True) -> FounderIndex:
    """Load a canonical index and, by default, verify every referenced artifact."""
    index_path = Path(path)
    index = founder_index_from_json_bytes(index_path.read_bytes())
    if verify_artifacts:
        for record in index.founders:
            load_founder_artifact(index_path.parent, record)
    return index


def _safe_artifact_location(root: Path, relative: str) -> Path:
    _require_artifact_path(relative)
    base = root.resolve()
    destination = (base / PurePosixPath(relative)).resolve()
    if destination == base or base not in destination.parents:
        raise ValueError("artifact path escapes founder-bank directory")
    return destination


def _read_npy_member(archive: zipfile.ZipFile, member: str) -> tuple[np.ndarray, bytes]:
    payload = archive.read(member)
    try:
        array = np.load(BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise ValueError(f"invalid numeric founder member: {member}") from error
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise ValueError(f"founder member must be a numeric ndarray: {member}")
    return array, payload


def load_founder_artifact(
    bank_directory: str | Path,
    record: FounderRecord,
    *,
    expected_partition: str | None = None,
    expected_controller_layout: str = CONTROLLER_LAYOUT,
) -> CPPNGenome:
    """Load one founder only after strict path, hash, layout, and graph checks."""
    if not isinstance(record, FounderRecord):
        raise TypeError("record must be a FounderRecord")
    if expected_partition is not None:
        _require_partition(expected_partition)
        if record.partition != expected_partition:
            raise ValueError(
                f"founder {record.founder_id!r} belongs to {record.partition!r}, "
                f"not {expected_partition!r}"
            )
    if expected_controller_layout != CONTROLLER_LAYOUT or record.controller_layout != expected_controller_layout:
        raise ValueError("founder controller layout does not match the runtime")

    artifact_path = _safe_artifact_location(Path(bank_directory), record.artifact)
    try:
        payload = artifact_path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to read founder artifact: {record.artifact}") from error
    if _sha256(payload) != record.artifact_sha256:
        raise ValueError(f"artifact hash mismatch for founder {record.founder_id!r}")

    try:
        with zipfile.ZipFile(BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != list(ARTIFACT_MEMBERS):
                raise ValueError("founder artifact must contain exactly the canonical members")
            if any(
                info.compress_type != zipfile.ZIP_STORED
                or info.flag_bits & 0x1
                or info.date_time != _FIXED_ZIP_TIMESTAMP
                for info in infos
            ):
                raise ValueError("founder artifact has noncanonical ZIP metadata")
            nodes, node_payload = _read_npy_member(archive, NODE_MEMBER)
            connections, connection_payload = _read_npy_member(archive, CONNECTION_MEMBER)
    except zipfile.BadZipFile as error:
        raise ValueError("founder artifact is not a valid NPZ") from error

    if _sha256(node_payload) != record.node_genes_sha256:
        raise ValueError(f"node_genes hash mismatch for founder {record.founder_id!r}")
    if _sha256(connection_payload) != record.connection_genes_sha256:
        raise ValueError(f"connection_genes hash mismatch for founder {record.founder_id!r}")

    canonical_payload, digests = canonical_founder_artifact_bytes(
        CPPNGenome(
            _canonical_array(nodes, expected_shape=(MAX_NODES, 5), name="node_genes"),
            _canonical_array(
                connections,
                expected_shape=(MAX_CONNECTIONS, 3),
                name="connection_genes",
            ),
        )
    )
    if canonical_payload != payload or digests != record.digests:
        raise ValueError("founder artifact is not in canonical form")
    return CPPNGenome(jnp.asarray(nodes), jnp.asarray(connections))


def load_founder(
    index_path: str | Path,
    founder_id: str,
    *,
    expected_partition: str,
) -> CPPNGenome:
    """Resolve and load one founder through its trusted canonical index."""
    path = Path(index_path)
    index = load_founder_index(path, verify_artifacts=False)
    record = index.record(founder_id, expected_partition=expected_partition)
    return load_founder_artifact(
        path.parent,
        record,
        expected_partition=expected_partition,
    )
