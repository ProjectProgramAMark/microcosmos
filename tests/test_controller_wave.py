import jax
import jax.numpy as jnp

from microcosmos.controller import (
    BIAS_LOCUS,
    GAIN_LOCUS,
    GENOME_SIZE,
    LOCAL_RESOURCE_LOCI,
    PHENOTYPE_MODULES,
    RESOURCE_GRADIENT_LOCI,
    STRATEGY_LOCI,
    WAVE_LOCI,
    GenomeConfig,
    _affine_decode,
    controller_action,
    gene_bounds,
    genome_size,
    initialize_genomes,
)
from microcosmos.ecology import sample_grid_nearest


def _action(
    genome,
    *,
    coordinate=0.0,
    time=0.0,
    local_resource=0.5,
    head_resource=0.5,
    tail_resource=0.5,
    alive=True,
    config=GenomeConfig(),
):
    return controller_action(
        genome[None, :],
        jnp.array([coordinate]),
        jnp.array([0], dtype=jnp.int32),
        jnp.asarray(time),
        jnp.array([local_resource]),
        jnp.array([head_resource]),
        jnp.array([tail_resource]),
        jnp.array([alive]),
        config,
    )[0]


def test_genome_layout_is_exact_normalized_float32_and_gap_free():
    genomes = initialize_genomes(jax.random.PRNGKey(0), 4)
    lower, upper = gene_bounds()
    assert genome_size() == GENOME_SIZE == 23
    assert genomes.shape == (4, 23)
    assert genomes.dtype == jnp.float32
    assert jnp.all(genomes >= lower)
    assert jnp.all(genomes <= upper)

    coverage = jnp.zeros(GENOME_SIZE, dtype=jnp.int32)
    for module in PHENOTYPE_MODULES:
        coverage = coverage.at[module].add(1)
    coverage = coverage.at[STRATEGY_LOCI].add(1)
    assert jnp.array_equal(coverage, jnp.ones(GENOME_SIZE, dtype=jnp.int32))
    assert WAVE_LOCI == slice(0, 12)
    assert LOCAL_RESOURCE_LOCI == slice(12, 15)
    assert RESOURCE_GRADIENT_LOCI == slice(15, 18)
    assert BIAS_LOCUS == slice(18, 19)
    assert GAIN_LOCUS == slice(19, 20)


def test_affine_decoder_maps_normalized_endpoints_and_midpoint_exactly():
    genes = jnp.array([-1.0, 0.0, 1.0])
    for lower, upper in ((-0.5, 0.5), (0.0, 4.0), (-jnp.pi, jnp.pi)):
        decoded = _affine_decode(genes, lower, upper)
        assert jnp.allclose(
            decoded,
            jnp.array([lower, (lower + upper) / 2.0, upper]),
            atol=1e-7,
        )


def test_strategy_loci_do_not_change_action_bitwise():
    genome = initialize_genomes(jax.random.PRNGKey(1), 1)[0]
    low_strategy = genome.at[STRATEGY_LOCI].set(-1.0)
    high_strategy = genome.at[STRATEGY_LOCI].set(1.0)
    common = dict(
        coordinate=0.37,
        time=1.25,
        local_resource=0.8,
        head_resource=0.9,
        tail_resource=0.2,
    )
    assert jnp.array_equal(_action(low_strategy, **common), _action(high_strategy, **common))


def test_each_phenotype_module_changes_action_in_a_deterministic_fixture():
    zero = jnp.zeros(GENOME_SIZE, dtype=jnp.float32)
    for time in (-100.0, 0.0, 0.7, 100.0):
        assert _action(zero, time=time, local_resource=0.5) == 0.0

    wave = zero.at[0].set(1.0).at[3].set(0.5)
    assert _action(wave) != 0.0

    local = zero.at[LOCAL_RESOURCE_LOCI.start].set(1.0)
    assert _action(local, local_resource=1.0) != 0.0

    gradient = zero.at[RESOURCE_GRADIENT_LOCI.start].set(1.0)
    assert _action(gradient, head_resource=1.0, tail_resource=0.0) != 0.0

    biased = zero.at[BIAS_LOCUS.start].set(1.0)
    assert _action(biased) != 0.0

    gain_off = biased.at[GAIN_LOCUS.start].set(-1.0)
    gain_high = biased.at[GAIN_LOCUS.start].set(1.0)
    assert _action(gain_off) == 0.0
    assert _action(gain_high) != _action(gain_off)


def test_action_is_finite_bounded_and_inactive_safe_for_extreme_inputs():
    config = GenomeConfig(max_bending_delta=0.27)
    genome = jnp.linspace(-1.0, 1.0, GENOME_SIZE).at[0].set(jnp.inf)
    action = _action(
        genome,
        coordinate=1e30,
        time=1e30,
        local_resource=jnp.inf,
        head_resource=jnp.inf,
        tail_resource=-jnp.inf,
        config=config,
    )
    assert jnp.isfinite(action)
    assert jnp.abs(action) <= config.max_bending_delta
    assert _action(genome, alive=False, config=config) == 0.0


def test_periodic_translation_preserves_resource_sensor_and_action_semantics():
    field = jnp.arange(48, dtype=jnp.float32).reshape(6, 8) / 47.0
    positions = jnp.array([[1.2, 2.7], [7.8, 5.9], [-0.2, -0.1]])
    translated = positions + jnp.array([8.0, 6.0])
    sensor = sample_grid_nearest(field, positions)
    translated_sensor = sample_grid_nearest(field, translated)
    assert jnp.array_equal(sensor, translated_sensor)

    genome = initialize_genomes(jax.random.PRNGKey(2), 1)[0]
    assert jnp.array_equal(
        _action(genome, local_resource=sensor[0]),
        _action(genome, local_resource=translated_sensor[0]),
    )


def test_head_tail_reversal_negates_gradient_only_correction():
    genome = jnp.zeros(GENOME_SIZE, dtype=jnp.float32).at[
        RESOURCE_GRADIENT_LOCI.start
    ].set(1.0)
    forward = _action(genome, head_resource=1.0, tail_resource=0.0)
    reverse = _action(genome, head_resource=0.0, tail_resource=1.0)
    assert jnp.allclose(forward, -reverse, atol=1e-7)


def test_eager_jit_and_vmap_actions_agree():
    config = GenomeConfig()
    genomes = initialize_genomes(jax.random.PRNGKey(3), 2)
    coordinates = jnp.array([-0.8, 0.0, 0.8])
    local = jnp.array([[0.1, 0.4, 0.9], [0.8, 0.3, 0.2]])
    head = jnp.array([0.9, 0.2])
    tail = jnp.array([0.1, 0.7])
    alive = jnp.array([True, True])

    def one(genome, local_values, head_value, tail_value, is_alive):
        return controller_action(
            genome[None, :],
            coordinates,
            jnp.zeros(3, dtype=jnp.int32),
            jnp.array(0.75),
            local_values,
            head_value[None],
            tail_value[None],
            is_alive[None],
            config,
        )

    eager = jax.vmap(one)(genomes, local, head, tail, alive)
    jitted = jax.jit(jax.vmap(one))(genomes, local, head, tail, alive)
    individual = jnp.stack(
        [one(genomes[i], local[i], head[i], tail[i], alive[i]) for i in range(2)]
    )
    assert jnp.allclose(eager, jitted, atol=1e-7, rtol=1e-6)
    assert jnp.allclose(eager, individual, atol=1e-7, rtol=1e-6)
