"""Compact fixed-layout traveling-wave controller."""

from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp


GENOME_SIZE = 23
PHENOTYPE_LOCI = slice(0, 20)
WAVE_LOCI = slice(0, 12)
LOCAL_RESOURCE_LOCI = slice(12, 15)
RESOURCE_GRADIENT_LOCI = slice(15, 18)
BIAS_LOCUS = slice(18, 19)
GAIN_LOCUS = slice(19, 20)
STRATEGY_LOCI = slice(20, 23)

PHENOTYPE_MODULES = (
    WAVE_LOCI,
    LOCAL_RESOURCE_LOCI,
    RESOURCE_GRADIENT_LOCI,
    BIAS_LOCUS,
    GAIN_LOCUS,
)
NUM_WAVES = 3
WAVE_PARAMETERS = 4

GENE_MIN = -1.0
GENE_MAX = 1.0
AMPLITUDE_RANGE = (-0.5, 0.5)
SPATIAL_FREQUENCY_RANGE = (0.0, 4.0 * math.pi)
TEMPORAL_FREQUENCY_RANGE = (0.0, 4.0)
PHASE_RANGE = (-math.pi, math.pi)
SENSORY_COEFFICIENT_RANGE = (-1.0, 1.0)
BIAS_RANGE = (-1.0, 1.0)
GAIN_RANGE = (0.0, 2.0)
INITIALIZATION_STD = 0.1


@dataclass(frozen=True)
class GenomeConfig:
    """Fixed controller limits plus the trusted development mutation settings."""

    mutation_probability: float = 0.05
    mutation_std: float = 0.05
    max_bending_delta: float = 0.35
    resource_reference: float = 0.5

    def __post_init__(self) -> None:
        for name, value in (
            ("mutation_probability", self.mutation_probability),
            ("mutation_std", self.mutation_std),
            ("max_bending_delta", self.max_bending_delta),
            ("resource_reference", self.resource_reference),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be within [0, 1]")
        if self.mutation_std < 0.0:
            raise ValueError("mutation_std must be non-negative")
        if self.max_bending_delta < 0.0:
            raise ValueError("max_bending_delta must be non-negative")
        if not 0.0 <= self.resource_reference <= 1.0:
            raise ValueError("resource_reference must be within [0, 1]")


def genome_size() -> int:
    """Return the immutable scalar genome length."""
    return GENOME_SIZE


def gene_bounds() -> tuple[jax.Array, jax.Array]:
    """Return the normalized bounds shared by every locus."""
    return (
        jnp.full(GENOME_SIZE, GENE_MIN, dtype=jnp.float32),
        jnp.full(GENOME_SIZE, GENE_MAX, dtype=jnp.float32),
    )


def initialize_genomes(key: jax.Array, count: int) -> jax.Array:
    """Create small random normalized genomes with an explicit float32 shape."""
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("count must be a non-negative integer")
    genomes = jax.random.normal(key, (count, GENOME_SIZE), dtype=jnp.float32)
    return jnp.clip(genomes * INITIALIZATION_STD, GENE_MIN, GENE_MAX)


def _finite_clip(value: jax.Array, lower: float, upper: float) -> jax.Array:
    return jnp.nan_to_num(
        jnp.clip(value, lower, upper), nan=0.0, posinf=upper, neginf=lower
    )


def _affine_decode(gene: jax.Array, lower: float, upper: float) -> jax.Array:
    normalized = _finite_clip(gene, GENE_MIN, GENE_MAX)
    return lower + 0.5 * (normalized - GENE_MIN) * (upper - lower)


def _polynomial(coefficients: jax.Array, coordinate: jax.Array) -> jax.Array:
    return (
        coefficients[:, 0]
        + coefficients[:, 1] * coordinate
        + coefficients[:, 2] * coordinate**2
    )


def controller_action(
    genomes: jax.Array,
    normalized_coordinate: jax.Array,
    bending_slot: jax.Array,
    time: jax.Array,
    local_resource: jax.Array,
    head_resource: jax.Array,
    tail_resource: jax.Array,
    alive: jax.Array,
    config: GenomeConfig = GenomeConfig(),
) -> jax.Array:
    """Decode three waves and two resource corrections for each body hinge."""
    slot_genomes = genomes[bending_slot]
    count = slot_genomes.shape[0]
    coordinate = _finite_clip(normalized_coordinate, -1.0, 1.0)
    slot_alive = alive[bending_slot].astype(slot_genomes.dtype)

    wave_genes = slot_genomes[:, WAVE_LOCI].reshape(
        count, NUM_WAVES, WAVE_PARAMETERS
    )
    amplitude = _affine_decode(wave_genes[..., 0], *AMPLITUDE_RANGE)
    spatial_frequency = _affine_decode(
        wave_genes[..., 1], *SPATIAL_FREQUENCY_RANGE
    )
    temporal_frequency = _affine_decode(
        wave_genes[..., 2], *TEMPORAL_FREQUENCY_RANGE
    )
    phase_offset = _affine_decode(wave_genes[..., 3], *PHASE_RANGE)

    time = jnp.asarray(time, dtype=slot_genomes.dtype)
    slot_time = time if time.ndim == 0 else time[bending_slot]
    slot_time = jnp.nan_to_num(slot_time, nan=0.0, posinf=0.0, neginf=0.0)
    wave = jnp.sum(
        amplitude
        * jnp.sin(
            spatial_frequency * coordinate[:, None]
            - temporal_frequency * slot_time[..., None]
            + phase_offset
        ),
        axis=-1,
    )

    local_coefficients = _affine_decode(
        slot_genomes[:, LOCAL_RESOURCE_LOCI], *SENSORY_COEFFICIENT_RANGE
    )
    gradient_coefficients = _affine_decode(
        slot_genomes[:, RESOURCE_GRADIENT_LOCI],
        *SENSORY_COEFFICIENT_RANGE,
    )
    local_value = _finite_clip(local_resource, 0.0, 1.0) * slot_alive
    head_value = (
        _finite_clip(head_resource[bending_slot], 0.0, 1.0) * slot_alive
    )
    tail_value = (
        _finite_clip(tail_resource[bending_slot], 0.0, 1.0) * slot_alive
    )
    local_correction = (local_value - config.resource_reference) * _polynomial(
        local_coefficients, coordinate
    )
    gradient_correction = (head_value - tail_value) * _polynomial(
        gradient_coefficients, coordinate
    )

    bias = _affine_decode(slot_genomes[:, BIAS_LOCUS.start], *BIAS_RANGE)
    gain = _affine_decode(slot_genomes[:, GAIN_LOCUS.start], *GAIN_RANGE)
    raw = bias + wave + local_correction + gradient_correction
    return (
        config.max_bending_delta * jnp.tanh(gain * raw) * slot_alive
    )
