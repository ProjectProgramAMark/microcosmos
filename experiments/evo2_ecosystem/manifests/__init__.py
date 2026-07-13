"""Integrity-checked access to manifests visible during policy development."""

from pathlib import Path
from types import MappingProxyType
from typing import Literal

from ..protocol import (
    ScenarioManifest,
    canonical_manifest_bytes,
    manifest_from_json_bytes,
    manifest_sha256,
)


VISIBLE_MANIFEST_SHA256 = MappingProxyType(
    {
        "training_stable.json": ("716f306f684646cfa2ea62c36c3cef5735e1d682425de3eeee5ca1018ae1b021"),
        "training_punctuated.json": ("535b759c613bb91252934d3b91645b4017add1379b78b7164e984fc58d3e8fbf"),
        "development.json": ("563c614250054089bbdbbdae06f78adae27da1d13b29be923c3d70fa99602ace"),
    }
)


def _load_visible(filename: str, partition: str) -> ScenarioManifest:
    path = Path(__file__).with_name(filename)
    return load_bound_manifest(
        path,
        expected_sha256=VISIBLE_MANIFEST_SHA256[filename],
        expected_partition=partition,
    )


def load_bound_manifest(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_partition: str,
) -> ScenarioManifest:
    """Load one canonical manifest through an externally frozen binding.

    The caller owns the immutable path/hash selection (for example a run
    specification). This loader owns strict parsing, canonical-byte checking,
    and partition enforcement. It deliberately does not discover profiles or
    accept unverified manifests on the candidate's behalf.
    """
    path = Path(path)
    raw = path.read_bytes()
    manifest = manifest_from_json_bytes(raw)
    if raw != canonical_manifest_bytes(manifest) + b"\n":
        raise RuntimeError(f"{path.name} is not stored as canonical JSON")
    if manifest_sha256(manifest) != expected_sha256:
        raise RuntimeError(f"{path.name} does not match its frozen SHA-256")
    if manifest.partition != expected_partition:
        raise RuntimeError(f"{path.name} has the wrong partition")
    return manifest


def load_training_manifest(
    regime: Literal["stable", "punctuated"],
) -> ScenarioManifest:
    """Load one of the two manifests visible to the matched outer searches."""
    if regime not in ("stable", "punctuated"):
        raise ValueError("training regime must be 'stable' or 'punctuated'")
    return _load_visible(f"training_{regime}.json", "training")


def load_development_manifest() -> ScenarioManifest:
    """Load the held-back manifest used only after an outer search finishes."""
    return _load_visible("development.json", "development")


__all__ = [
    "VISIBLE_MANIFEST_SHA256",
    "load_bound_manifest",
    "load_development_manifest",
    "load_training_manifest",
]
