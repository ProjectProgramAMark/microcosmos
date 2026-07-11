"""Fixed-topology heritable neural controller."""

from dataclasses import dataclass
import functools
import math

import jax
import jax.numpy as jnp


CONTROLLER_INPUTS = 5


@dataclass(frozen=True)
class GenomeConfig:
    hidden_size: int = 8
    neural_bound: float = 2.0
    oscillator_min: float = 0.0
    oscillator_max: float = 4.0
    uptake_min: float = 0.0
    uptake_max: float = 2.0
    assimilation_min: float = 0.0
    assimilation_max: float = 1.0
    metabolism_min: float = 0.0
    metabolism_max: float = 1.0
    mutation_probability: float = 0.05
    mutation_std: float = 0.05
    max_bending_delta: float = 0.35

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_size, int) or isinstance(self.hidden_size, bool):
            raise ValueError("hidden_size must be an integer")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least one")

        finite_fields = {
            "neural_bound": self.neural_bound,
            "oscillator_min": self.oscillator_min,
            "oscillator_max": self.oscillator_max,
            "uptake_min": self.uptake_min,
            "uptake_max": self.uptake_max,
            "assimilation_min": self.assimilation_min,
            "assimilation_max": self.assimilation_max,
            "metabolism_min": self.metabolism_min,
            "metabolism_max": self.metabolism_max,
            "mutation_probability": self.mutation_probability,
            "mutation_std": self.mutation_std,
            "max_bending_delta": self.max_bending_delta,
        }
        for name, value in finite_fields.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.neural_bound <= 0.0:
            raise ValueError("neural_bound must be positive")
        for name, lower, upper in (
            ("oscillator", self.oscillator_min, self.oscillator_max),
            ("uptake", self.uptake_min, self.uptake_max),
            ("assimilation", self.assimilation_min, self.assimilation_max),
            ("metabolism", self.metabolism_min, self.metabolism_max),
        ):
            if lower < 0.0 or upper < lower:
                raise ValueError(
                    f"{name} bounds must be non-negative and ordered"
                )
        if self.assimilation_max > 1.0:
            raise ValueError("assimilation_max must be at most one")
        if not 0.0 <= self.mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be within [0, 1]")
        if self.mutation_std < 0.0:
            raise ValueError("mutation_std must be non-negative")
        if self.max_bending_delta < 0.0:
            raise ValueError("max_bending_delta must be non-negative")


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "oscillator_rate",
        "uptake_rate",
        "assimilation_efficiency",
        "basal_metabolism",
    ],
)
@dataclass
class MetabolicPhenotype:
    oscillator_rate: jax.Array
    uptake_rate: jax.Array
    assimilation_efficiency: jax.Array
    basal_metabolism: jax.Array


def genome_size(config: GenomeConfig = GenomeConfig()) -> int:
    """Number of scalar genes for the configured controller."""
    hidden = config.hidden_size
    return CONTROLLER_INPUTS * hidden + hidden + hidden + 1 + 4


def _metabolic_start(config: GenomeConfig) -> int:
    return genome_size(config) - 4


def gene_bounds(config: GenomeConfig) -> tuple[jax.Array, jax.Array]:
    """Per-gene lower and upper bounds."""
    metabolic_start = _metabolic_start(config)
    lower = jnp.full(genome_size(config), -config.neural_bound)
    upper = jnp.full(genome_size(config), config.neural_bound)
    lower = lower.at[metabolic_start:].set(
        jnp.array(
            [
                config.oscillator_min,
                config.uptake_min,
                config.assimilation_min,
                config.metabolism_min,
            ]
        )
    )
    upper = upper.at[metabolic_start:].set(
        jnp.array(
            [
                config.oscillator_max,
                config.uptake_max,
                config.assimilation_max,
                config.metabolism_max,
            ]
        )
    )
    return lower, upper


def initialize_genomes(
    key: jax.Array,
    count: int,
    config: GenomeConfig = GenomeConfig(),
    *,
    oscillator_rate: float = 1.0,
    uptake_rate: float = 0.5,
    assimilation_efficiency: float = 0.8,
    basal_metabolism: float = 0.05,
) -> jax.Array:
    """Create bounded genomes with small random controller weights."""
    size = genome_size(config)
    genomes = jax.random.normal(key, (count, size)) * 0.1
    phenotype = jnp.array(
        [oscillator_rate, uptake_rate, assimilation_efficiency, basal_metabolism]
    )
    genomes = genomes.at[:, -4:].set(phenotype)
    lower, upper = gene_bounds(config)
    return jnp.clip(genomes, lower, upper)


def decode_metabolism(
    genome: jax.Array, config: GenomeConfig = GenomeConfig()
) -> MetabolicPhenotype:
    """Decode and defensively bound metabolic genes."""
    metabolic = genome[..., -4:]
    return MetabolicPhenotype(
        oscillator_rate=jnp.clip(
            metabolic[..., 0], config.oscillator_min, config.oscillator_max
        ),
        uptake_rate=jnp.clip(
            metabolic[..., 1], config.uptake_min, config.uptake_max
        ),
        assimilation_efficiency=jnp.clip(
            metabolic[..., 2], config.assimilation_min, config.assimilation_max
        ),
        basal_metabolism=jnp.clip(
            metabolic[..., 3], config.metabolism_min, config.metabolism_max
        ),
    )


def mutate_genome(
    key: jax.Array,
    genome: jax.Array,
    config: GenomeConfig = GenomeConfig(),
) -> jax.Array:
    """Apply independent Bernoulli/Gaussian mutation and clip to bounds."""
    if config.mutation_std == 0.0 or config.mutation_probability == 0.0:
        return genome
    key_mask, key_noise = jax.random.split(key)
    mask = jax.random.bernoulli(
        key_mask, config.mutation_probability, genome.shape
    )
    noise = jax.random.normal(key_noise, genome.shape) * config.mutation_std
    lower, upper = gene_bounds(config)
    return jnp.clip(genome + mask * noise, lower, upper)


def controller_action(
    genomes: jax.Array,
    normalized_coordinate: jax.Array,
    bending_slot: jax.Array,
    phase: jax.Array,
    local_resource: jax.Array,
    energy_fraction: jax.Array,
    alive: jax.Array,
    config: GenomeConfig = GenomeConfig(),
) -> jax.Array:
    """Evaluate one shared-shape MLP per bending pair."""
    hidden = config.hidden_size
    slot_genomes = genomes[bending_slot]
    offset = 0
    w1_size = CONTROLLER_INPUTS * hidden
    w1 = slot_genomes[:, offset : offset + w1_size].reshape(
        -1, CONTROLLER_INPUTS, hidden
    )
    offset += w1_size
    b1 = slot_genomes[:, offset : offset + hidden]
    offset += hidden
    w2 = slot_genomes[:, offset : offset + hidden]
    offset += hidden
    b2 = slot_genomes[:, offset]

    slot_phase = phase[bending_slot]
    inputs = jnp.stack(
        [
            normalized_coordinate,
            jnp.sin(slot_phase),
            jnp.cos(slot_phase),
            local_resource,
            energy_fraction[bending_slot],
        ],
        axis=-1,
    )
    hidden_value = jnp.tanh(jnp.einsum("bi,bih->bh", inputs, w1) + b1)
    output = jnp.tanh(jnp.sum(hidden_value * w2, axis=-1) + b2)
    return output * config.max_bending_delta * alive[bending_slot]
