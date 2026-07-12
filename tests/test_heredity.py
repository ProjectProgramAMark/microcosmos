import jax
import jax.numpy as jnp
import pytest

from microcosmos.cppn import (
    CONNECTION_INPUT,
    CONNECTION_OUTPUT,
    IDENTITY_ACTIVATION,
    MAX_EXACT_FLOAT32_INTEGER,
    NODE_ACTIVATION,
    NODE_AGGREGATION,
    NODE_BIAS,
    NODE_RESPONSE,
    NUM_INPUTS,
    OUTPUT_ROW,
    SUM_AGGREGATION,
    CPPNGenome,
    canonical_cppn_genome,
    transform_and_validate_genome,
)
from microcosmos.heredity import (
    CLONE,
    MIXED,
    PARAMETRIC,
    STRUCTURAL,
    ParentStats,
    PopulationStats,
    action_diversity,
    fixed_mixed_policy,
    fixed_parametric_policy,
    fixed_structural_policy,
    lineage_entropy,
    make_mutation_context,
    mutate_cppn,
    parent_statistics,
    population_statistics,
    update_intake_ema,
    update_population_change_ema,
)


def _equal_with_nan(left: jax.Array, right: jax.Array) -> bool:
    return bool(jnp.all((jnp.isnan(left) & jnp.isnan(right)) | (left == right)))


def _assert_same_genome(left: CPPNGenome, right: CPPNGenome) -> None:
    assert _equal_with_nan(left.node_genes, right.node_genes)
    assert _equal_with_nan(left.connection_genes, right.connection_genes)


def _stats() -> tuple[ParentStats, PopulationStats]:
    return (
        ParentStats(
            energy_fraction=jnp.array(0.8),
            intake_ema=jnp.array(0.6),
            node_fraction=jnp.array(0.4),
            connection_fraction=jnp.array(0.2),
        ),
        PopulationStats(
            alive_fraction=jnp.array(0.5),
            population_change_ema=jnp.array(-0.1),
            action_diversity=jnp.array(0.3),
            lineage_entropy=jnp.array(0.7),
        ),
    )


@pytest.mark.parametrize(
    ("scores", "expected_operator"),
    [
        ([1.0, 0.0, 0.0, 0.0], CLONE),
        ([0.0, 1.0, 0.0, 0.0], PARAMETRIC),
        ([0.0, 0.0, 1.0, 0.0], STRUCTURAL),
        ([0.0, 0.0, 0.0, 1.0], MIXED),
    ],
)
def test_all_trusted_profiles_return_valid_canonical_genomes(scores, expected_operator):
    parent = canonical_cppn_genome()
    context, context_valid = make_mutation_context(jax.random.PRNGKey(11), jnp.array(7))
    result = mutate_cppn(parent, jnp.asarray(scores, dtype=jnp.float32), context)

    assert context_valid
    assert result.policy_valid
    assert result.operator_index == expected_operator
    assert transform_and_validate_genome(result.genome)[-1]
    assert jnp.all(result.genome.node_genes[:NUM_INPUTS, NODE_BIAS] == 0.0)
    assert jnp.all(result.genome.node_genes[:NUM_INPUTS, NODE_RESPONSE] == 1.0)
    assert jnp.all(result.genome.node_genes[:NUM_INPUTS, NODE_AGGREGATION] == SUM_AGGREGATION)
    assert jnp.all(result.genome.node_genes[:NUM_INPUTS, NODE_ACTIVATION] == IDENTITY_ACTIVATION)
    assert result.genome.node_genes[OUTPUT_ROW, NODE_ACTIVATION] == IDENTITY_ACTIVATION
    if expected_operator == CLONE:
        _assert_same_genome(result.genome, parent)


@pytest.mark.parametrize(
    "policy",
    [fixed_parametric_policy, fixed_structural_policy, fixed_mixed_policy],
)
def test_mutation_replays_deterministically_for_identical_context(policy):
    parent = canonical_cppn_genome()
    parent_stats, population_stats = _stats()
    context, valid = make_mutation_context(jax.random.PRNGKey(23), jnp.array(19))

    first = policy(parent, parent_stats, population_stats, context)
    second = policy(parent, parent_stats, population_stats, context)
    assert valid
    _assert_same_genome(first.genome, second.genome)
    assert first.operator_index == second.operator_index
    assert first.policy_valid == second.policy_valid


def test_parametric_profile_preserves_topology_exactly():
    parent = canonical_cppn_genome()
    context, _ = make_mutation_context(jax.random.PRNGKey(31), jnp.array(4))
    result = mutate_cppn(
        parent,
        jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
        context,
    )

    assert _equal_with_nan(result.genome.node_genes[:, 0], parent.node_genes[:, 0])
    assert _equal_with_nan(
        result.genome.connection_genes[:, [CONNECTION_INPUT, CONNECTION_OUTPUT]],
        parent.connection_genes[:, [CONNECTION_INPUT, CONNECTION_OUTPUT]],
    )


@pytest.mark.parametrize("operator", [STRUCTURAL, MIXED])
def test_structural_profiles_stay_valid_across_a_deterministic_key_panel(operator):
    parent = canonical_cppn_genome()
    scores = jnp.zeros(4, dtype=jnp.float32).at[operator].set(1.0)

    @jax.jit
    def mutate_and_validate(key, child_id):
        context, context_valid = make_mutation_context(key, child_id)
        result = mutate_cppn(parent, scores, context)
        return context_valid, result.policy_valid, result.operator_index, transform_and_validate_genome(result.genome)[-1]

    for child_id in range(12):
        outcome = mutate_and_validate(jax.random.PRNGKey(child_id), jnp.array(child_id, dtype=jnp.int32))
        assert all(bool(value) for value in (outcome[0], outcome[1], outcome[3]))
        assert outcome[2] == operator


def test_nonfinite_scores_are_invalid_and_fall_back_to_exact_clone():
    parent = canonical_cppn_genome()
    context, _ = make_mutation_context(jax.random.PRNGKey(0), jnp.array(0))
    for bad_scores in (
        jnp.array([jnp.nan, 1.0, 0.0, 0.0]),
        jnp.array([0.0, jnp.inf, 1.0, 0.0]),
    ):
        result = mutate_cppn(parent, bad_scores, context)
        assert not result.policy_valid
        assert result.operator_index == CLONE
        _assert_same_genome(result.genome, parent)


def test_operator_score_contract_rejects_wrong_shape_and_dtype():
    parent = canonical_cppn_genome()
    context, _ = make_mutation_context(jax.random.PRNGKey(0), jnp.array(0))
    with pytest.raises(ValueError):
        mutate_cppn(parent, jnp.ones(3, dtype=jnp.float32), context)
    with pytest.raises(TypeError):
        mutate_cppn(parent, jnp.ones(4, dtype=jnp.int32), context)


@pytest.mark.parametrize(
    ("child_id", "expected_key", "expected_valid"),
    [
        (-1, 1023, False),
        (0, 1024, True),
        (MAX_EXACT_FLOAT32_INTEGER - 1024 - 1, MAX_EXACT_FLOAT32_INTEGER - 1, True),
        (MAX_EXACT_FLOAT32_INTEGER - 1024, MAX_EXACT_FLOAT32_INTEGER, False),
    ],
)
def test_mutation_context_enforces_exact_float32_node_key_bounds(child_id, expected_key, expected_valid):
    supplied_key = jax.random.PRNGKey(91)
    context, valid = make_mutation_context(supplied_key, jnp.array(child_id, dtype=jnp.int32))
    assert jnp.array_equal(context.key, supplied_key)
    assert context.new_node_key == jnp.float32(expected_key)
    assert bool(valid) is expected_valid


def test_intake_and_population_change_emas_have_exact_single_step_semantics():
    intake = update_intake_ema(
        intake_ema=jnp.array([0.2, 0.8, 0.4]),
        actual_uptake=jnp.array([0.5, 2.0, 100.0]),
        uptake_rate=jnp.array([1.0, 1.0, 1.0]),
        alive=jnp.array([True, True, False]),
        dt=1.0,
        eps=1e-8,
    )
    assert jnp.allclose(intake, jnp.array([0.215, 0.81, 0.4]), atol=1e-7)

    population = update_population_change_ema(
        previous=jnp.array(0.2),
        population_delta=jnp.array(-4),
        capacity=10,
    )
    assert jnp.allclose(population, 0.14, atol=1e-7)


def test_action_diversity_masks_dead_slots_and_handles_small_populations():
    actions = jnp.array([[-1.0, 1.0], [1.0, -1.0], [100.0, 100.0]])
    assert action_diversity(actions, jnp.array([True, True, False]), 1.0) == 1.0
    assert action_diversity(actions, jnp.array([True, False, False]), 1.0) == 0.0
    assert action_diversity(actions, jnp.array([False, False, False]), 1.0) == 0.0


def test_lineage_entropy_is_normalized_and_ignores_dead_slots():
    founder_ids = jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
    assert jnp.allclose(lineage_entropy(founder_ids, jnp.ones(6, dtype=jnp.bool_), 3), 1.0)
    assert lineage_entropy(founder_ids, jnp.array([True, True, False, False, False, False]), 3) == 0.0
    assert lineage_entropy(founder_ids, jnp.zeros(6, dtype=jnp.bool_), 3) == 0.0
    assert lineage_entropy(founder_ids, jnp.ones(6, dtype=jnp.bool_), 1) == 0.0


def test_parent_and_population_statistics_are_normalized_and_clipped():
    parent = canonical_cppn_genome()
    stats = parent_statistics(
        CPPNGenome(parent.node_genes[None, ...], parent.connection_genes[None, ...]),
        energy=jnp.array([30.0]),
        intake_ema=jnp.array([1.4]),
        reproduction_threshold=10.0,
    )
    assert stats.energy_fraction[0] == 2.0
    assert stats.intake_ema[0] == 1.0
    assert jnp.all((stats.node_fraction >= 0.0) & (stats.node_fraction <= 1.0))
    assert jnp.all((stats.connection_fraction >= 0.0) & (stats.connection_fraction <= 1.0))

    population = population_statistics(
        alive=jnp.array([True, True, False, False]),
        population_change_for_birth=jnp.array(-2.0),
        controller_action_diversity=jnp.array(2.0),
        founder_lineage_id=jnp.array([0, 1, -1, -1], dtype=jnp.int32),
        initial_founder_count=2,
    )
    assert population.alive_fraction == 0.5
    assert population.population_change_ema == -1.0
    assert population.action_diversity == 1.0
    assert jnp.allclose(population.lineage_entropy, 1.0)
