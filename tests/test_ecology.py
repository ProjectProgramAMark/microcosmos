from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from microcosmos.cppn import (
    CPPNGenome,
    cached_genome_valid,
    initialize_cppn_population,
    transform_population,
)
from microcosmos.ecology import (
    LifecycleConfig,
    ResourceConfig,
    actuation_energy_by_slot,
    energy_and_death_step,
    make_periodic_resource_patch,
    reproduction_step,
    resource_step,
)
from microcosmos.heredity import (
    NUM_OPERATORS,
    OffspringResult,
    clone_policy,
    fixed_parametric_policy,
    make_mutation_context,
    population_statistics,
)
from microcosmos.structs.population import PopulationState


def _array_equal_nan(left, right):
    return jnp.array_equal(left, right, equal_nan=True)


def _genome_equal(left: CPPNGenome, right: CPPNGenome):
    return all(_array_equal_nan(left_leaf, right_leaf) for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))


def _population(energy=(5.0, 0.0, 0.0)):
    count = len(energy)
    alive = jnp.arange(count) == 0
    genomes = initialize_cppn_population(count, initial_population=1)
    controller_order, controller_connection_index = transform_population(genomes)
    return PopulationState(
        alive=alive,
        energy=jnp.asarray(energy),
        age=jnp.zeros(count, dtype=jnp.int32),
        generation=jnp.zeros(count, dtype=jnp.int32),
        individual_id=jnp.where(alive, jnp.arange(count), -1),
        parent_id=jnp.full(count, -1, dtype=jnp.int32),
        genome=genomes,
        controller_order=controller_order,
        controller_connection_index=controller_connection_index,
        founder_lineage_id=jnp.where(alive, jnp.arange(count), -1),
        intake_ema=jnp.zeros(count, dtype=jnp.float32),
        population_change_ema=jnp.zeros((), dtype=jnp.float32),
        next_individual_id=jnp.array(1, dtype=jnp.int32),
    )


def _population_stats(population):
    return population_statistics(
        population.alive,
        population.population_change_ema,
        jnp.zeros((), dtype=jnp.float32),
        population.founder_lineage_id,
        initial_founder_count=1,
    )


def test_resource_contention_is_proportional_and_conservative():
    resource = jnp.zeros((4, 4)).at[1, 1].set(1.0)
    updated, uptake = resource_step(
        resource,
        jnp.ones((4, 4)),
        jnp.zeros((4, 4)),
        jnp.array([[1.0, 1.0], [1.1, 1.1]]),
        jnp.array([True, True]),
        jnp.array([1.0, 3.0]),
        1.0,
        ResourceConfig(capacity=1.0, regeneration_rate=0.0),
    )
    assert jnp.allclose(uptake, jnp.array([0.25, 0.75]), rtol=1e-5)
    assert jnp.isclose(jnp.sum(resource) - jnp.sum(updated), jnp.sum(uptake))
    assert jnp.min(updated) >= -1e-6


def test_inactive_nodes_consume_zero_and_regeneration_is_capped():
    updated, uptake = resource_step(
        jnp.full((2, 2), 0.9),
        jnp.ones((2, 2)),
        jnp.ones((2, 2)) * 2.0,
        jnp.array([[0.0, 0.0]]),
        jnp.array([False]),
        jnp.array([10.0]),
        1.0,
        ResourceConfig(capacity=1.0, regeneration_rate=2.0),
    )
    assert uptake[0] == 0.0
    assert jnp.max(updated) <= 1.0 + 1e-6


def test_periodic_resource_patch_translates_across_boundaries():
    boundary_capacity, boundary_regeneration = make_periodic_resource_patch((16, 16), (15.0, 8.0), 4.0, 2.0, 0.25)
    interior_capacity, interior_regeneration = make_periodic_resource_patch((16, 16), (7.0, 8.0), 4.0, 2.0, 0.25)
    assert jnp.array_equal(jnp.roll(boundary_capacity, -8, axis=1), interior_capacity)
    assert jnp.array_equal(jnp.roll(boundary_regeneration, -8, axis=1), interior_regeneration)


def test_resource_step_clips_stock_to_capacity_and_zero_capacity_regions():
    capacity, regeneration = make_periodic_resource_patch((9, 9), (4.0, 4.0), 2.5, 1.0, 0.0)
    updated, uptake = resource_step(
        jnp.full((9, 9), 5.0),
        capacity,
        regeneration,
        jnp.array([[4.0, 4.0]]),
        jnp.array([False]),
        jnp.array([1.0]),
        1.0,
        ResourceConfig(capacity=1.0, regeneration_rate=0.0),
    )
    assert uptake[0] == 0.0
    assert jnp.min(updated) >= -1e-6
    assert jnp.all(updated <= capacity + 1e-6)
    assert jnp.all(updated[capacity == 0.0] == 0.0)


def test_one_mouth_demand_does_not_scale_with_body_node_count():
    resource = jnp.zeros((5, 5)).at[2, 2].set(1.0)
    capacity = jnp.ones((5, 5))
    regeneration = jnp.zeros((5, 5))

    def uptake_for_body_size(node_count):
        body = jnp.tile(jnp.array([[2.0, 2.0]]), (node_count, 1))
        mouth_positions = body.reshape(1, node_count, 2)[:, 0, :]
        return resource_step(
            resource,
            capacity,
            regeneration,
            mouth_positions,
            jnp.array([True]),
            jnp.array([0.4]),
            1.0,
            ResourceConfig(capacity=1.0, regeneration_rate=0.0),
        )[1][0]

    assert uptake_for_body_size(4) == uptake_for_body_size(12) == 0.4


def test_resource_diffusion_is_conservative_before_inactive_map_clipping():
    resource = jnp.arange(25, dtype=jnp.float32).reshape(5, 5) / 100.0
    updated, _ = resource_step(
        resource,
        jnp.ones((5, 5)),
        jnp.zeros((5, 5)),
        jnp.array([[0.0, 0.0]]),
        jnp.array([False]),
        jnp.array([0.0]),
        0.5,
        ResourceConfig(
            capacity=1.0,
            regeneration_rate=0.0,
            diffusion_rate=0.5,
        ),
    )
    assert jnp.isclose(jnp.sum(updated), jnp.sum(resource), atol=1e-6)


def test_zero_regeneration_allows_a_finite_patch_to_fully_deplete():
    capacity, regeneration = make_periodic_resource_patch((5, 5), (2.0, 2.0), 0.75, 1.0, 0.0)
    updated, uptake = resource_step(
        capacity,
        capacity,
        regeneration,
        jnp.array([[2.0, 2.0]]),
        jnp.array([True]),
        jnp.array([2.0]),
        1.0,
        ResourceConfig(capacity=1.0, regeneration_rate=0.0),
    )
    assert jnp.isclose(uptake[0], 1.0)
    assert jnp.sum(updated) == 0.0


def test_death_event_is_emitted_once():
    population = _population(energy=(0.1, 0.0, 0.0))
    population, died, death_ids = energy_and_death_step(population, jnp.zeros(3), jnp.ones(3), jnp.ones(3), jnp.zeros(3), 1.0, 100)
    assert died.tolist() == [True, False, False]
    assert death_ids.tolist() == [0, -1, -1]
    _, died_again, death_ids_again = energy_and_death_step(population, jnp.zeros(3), jnp.ones(3), jnp.ones(3), jnp.zeros(3), 1.0, 100)
    assert not jnp.any(died_again)
    assert jnp.all(death_ids_again == -1)


def test_lifespan_death_occurs_at_exact_configured_age():
    population = replace(_population(), age=jnp.array([4, 0, 0]))
    _, died, _ = energy_and_death_step(population, jnp.zeros(3), jnp.ones(3), jnp.zeros(3), jnp.zeros(3), 1.0, 5)
    assert died.tolist() == [True, False, False]


def test_reproduction_assigns_lineage_and_charges_only_success():
    population = replace(_population(), age=jnp.array([5, 0, 0]))
    lifecycle = LifecycleConfig(
        maturity_age=1,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        birth_transfer_efficiency=0.5,
    )
    child_population, events = reproduction_step(
        jax.random.PRNGKey(1),
        population,
        lifecycle,
        clone_policy,
        _population_stats(population),
    )
    assert int(events["birth_count"]) == 1
    assert child_population.alive.tolist() == [True, True, False]
    assert child_population.energy.tolist() == [3.0, 1.0, 0.0]
    assert int(child_population.parent_id[1]) == 0
    assert int(child_population.individual_id[1]) == 1
    assert int(child_population.generation[1]) == 1
    assert int(child_population.founder_lineage_id[1]) == 0
    child_genome = CPPNGenome(
        child_population.genome.node_genes[1],
        child_population.genome.connection_genes[1],
    )
    parent_genome = CPPNGenome(
        population.genome.node_genes[0],
        population.genome.connection_genes[0],
    )
    assert _genome_equal(child_genome, parent_genome)
    assert bool(
        cached_genome_valid(
            child_genome,
            child_population.controller_order[1],
            child_population.controller_connection_index[1],
        )
    )
    assert events["operator_counts"].tolist() == [1, 0, 0, 0]
    assert int(events["policy_violation_count"]) == 0
    assert bool(events["infrastructure_valid"])

    full = replace(child_population, alive=jnp.ones(3, dtype=jnp.bool_))
    no_birth, full_events = reproduction_step(
        jax.random.PRNGKey(2),
        full,
        lifecycle,
        clone_policy,
        _population_stats(full),
    )
    assert int(full_events["birth_count"]) == 0
    assert jnp.array_equal(no_birth.energy, full.energy)
    assert int(jnp.sum(full_events["operator_counts"])) == 0


def test_birth_is_an_energy_transfer_and_cannot_create_organism_energy():
    population = replace(_population(), age=jnp.array([5, 0, 0]))
    lifecycle = LifecycleConfig(
        maturity_age=1,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        birth_transfer_efficiency=1.0,
    )
    before = jnp.sum(population.energy * population.alive)
    child_population, events = reproduction_step(
        jax.random.PRNGKey(7),
        population,
        lifecycle,
        clone_policy,
        _population_stats(population),
    )
    after = jnp.sum(child_population.energy * child_population.alive)
    assert int(events["birth_count"]) == 1
    assert jnp.isclose(after, before)
    assert float(child_population.energy[1]) == lifecycle.child_initial_energy


def test_actuation_energy_scales_with_physical_time_not_step_count():
    bending_delta = jnp.array([0.25, -0.5, 0.75, 0.5])
    bending_slot = jnp.array([0, 0, 1, 1])

    def charge_over_one_second(dt):
        per_step = actuation_energy_by_slot(
            bending_delta,
            bending_slot,
            num_slots=2,
            power_coefficient=0.3,
            dt=dt,
        )
        return per_step * round(1.0 / dt)

    assert jnp.allclose(
        charge_over_one_second(0.1),
        charge_over_one_second(0.05),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capacity": -1.0}, "capacity"),
        ({"regeneration_rate": -0.1}, "regeneration_rate"),
        ({"diffusion_rate": -0.1}, "diffusion_rate"),
        ({"eps": 0.0}, "eps"),
    ],
)
def test_resource_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ResourceConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maximum_lifespan": 0}, "maximum_lifespan"),
        (
            {"maximum_lifespan": 10, "maturity_age": 10},
            "maturity_age",
        ),
        ({"reproduction_cost": 0.0}, "reproduction_cost"),
        (
            {"reproduction_threshold": 1.0, "reproduction_cost": 2.0},
            "reproduction_threshold",
        ),
        ({"birth_transfer_efficiency": 0.0}, "birth_transfer_efficiency"),
        ({"birth_transfer_efficiency": 1.1}, "birth_transfer_efficiency"),
    ],
)
def test_lifecycle_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LifecycleConfig(**kwargs)


def test_parametric_heredity_is_deterministic_and_produces_valid_cache():
    population = replace(_population(), age=jnp.array([5, 0, 0]))
    lifecycle = LifecycleConfig(
        maturity_age=1,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
    )

    def reproduce_once():
        return reproduction_step(
            jax.random.PRNGKey(4),
            population,
            lifecycle,
            fixed_parametric_policy,
            _population_stats(population),
            timestep=9,
        )

    first, first_events = reproduce_once()
    second, second_events = reproduce_once()
    assert _genome_equal(first.genome, second.genome)
    assert jnp.array_equal(first.controller_order, second.controller_order)
    assert jnp.array_equal(
        first.controller_connection_index,
        second.controller_connection_index,
    )
    assert first_events["operator_counts"].tolist() == [0, 1, 0, 0]
    assert jnp.array_equal(first_events["operator_counts"], second_events["operator_counts"])
    child = CPPNGenome(first.genome.node_genes[1], first.genome.connection_genes[1])
    assert bool(
        cached_genome_valid(
            child,
            first.controller_order[1],
            first.controller_connection_index[1],
        )
    )


def test_invalid_policy_falls_back_to_clone_and_is_counted():
    population = replace(_population(), age=jnp.array([5, 0, 0]))
    lifecycle = LifecycleConfig(
        maturity_age=1,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
    )

    def invalid_policy(parent, parent_stats, population_stats, context):
        del parent_stats, population_stats, context
        return OffspringResult(
            parent,
            jnp.array(False),
            jnp.array(NUM_OPERATORS, dtype=jnp.int32),
        )

    child_population, events = reproduction_step(
        jax.random.PRNGKey(8),
        population,
        lifecycle,
        invalid_policy,
        _population_stats(population),
    )
    assert int(events["birth_count"]) == 1
    assert events["operator_counts"].tolist() == [1, 0, 0, 0]
    assert int(events["policy_violation_count"]) == 1
    child = CPPNGenome(
        child_population.genome.node_genes[1],
        child_population.genome.connection_genes[1],
    )
    parent = CPPNGenome(
        population.genome.node_genes[0],
        population.genome.connection_genes[0],
    )
    assert _genome_equal(child, parent)


def test_mutation_context_is_identity_stable_and_exactly_representable():
    context_a, valid_a = make_mutation_context(jax.random.PRNGKey(2), 41)
    context_b, valid_b = make_mutation_context(jax.random.PRNGKey(2), 41)
    assert bool(valid_a & valid_b)
    assert jnp.array_equal(context_a.key, context_b.key)
    assert context_a.new_node_key == context_b.new_node_key
    assert int(context_a.new_node_key) > 41
