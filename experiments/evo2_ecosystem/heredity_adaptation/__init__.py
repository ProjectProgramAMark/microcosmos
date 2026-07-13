"""Prospective manifests for the clonal heredity-adaptation experiment.

The package contains construction and validation code only.  Production
manifests are deliberately absent until founder screening and injury-gain
calibration have been frozen.
"""

from .manifest_generator import (
    CALIBRATION_INJURY_GAINS,
    CHUNK_STEPS,
    DEVELOPMENT_SEEDS,
    EVENT_STEPS,
    HINGE_COUNT,
    HORIZON,
    SEALED_SEEDS,
    TRAINING_SEEDS,
    HeredityAdaptationManifests,
    build_manifest_bundle,
    publish_manifest_bundle,
    validate_manifest_bundle,
)

__all__ = [
    "CALIBRATION_INJURY_GAINS",
    "CHUNK_STEPS",
    "DEVELOPMENT_SEEDS",
    "EVENT_STEPS",
    "HINGE_COUNT",
    "HORIZON",
    "SEALED_SEEDS",
    "TRAINING_SEEDS",
    "HeredityAdaptationManifests",
    "build_manifest_bundle",
    "publish_manifest_bundle",
    "validate_manifest_bundle",
]
