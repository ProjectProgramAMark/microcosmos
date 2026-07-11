from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from microcosmos.controller import GenomeConfig, initialize_genomes
from microcosmos.ecology import (
    LifecycleConfig,
    ResourceConfig,
    _fixed_gaussian_child,
    actuation_energy_by_slot,
    energy_and_death_step,
    reproduction_step,
    resource_step,
)
from microcosmos.structs.population import PopulationState


def _population(energy=(5.0, 0.0, 0.0)):
    count = len(energy)
    alive = jnp.arange(count) == 0
    genomes = initialize_genomes(jax.random.PRNGKey(0), count)
    return PopulationState(
        alive=alive,
        energy=jnp.asarray(energy),
        age=jnp.zeros(count, dtype=jnp.int32),
        generation=jnp.zeros(count, dtype=jnp.int32),
        individual_id=jnp.where(alive, jnp.arange(count), -1),
        parent_id=jnp.full(count, -1, dtype=jnp.int32),
        genome=genomes,
        next_individual_id=jnp.array(1, dtype=jnp.int32),
    )


def test_resource_contention_is_proportional_and_conservative():
    resource = jnp.zeros((4, 4)).at[1, 1].set(1.0)
    updated, uptake = resource_step(
        resource,
        jnp.array([[1.0, 1.0], [1.1, 1.1]]),
        jnp.array([True, True]),
        jnp.array([0, 1]),
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
        jnp.array([[0.0, 0.0]]),
        jnp.array([False]),
        jnp.array([0]),
        jnp.array([10.0]),
        1.0,
        ResourceConfig(capacity=1.0, regeneration_rate=2.0),
    )
    assert uptake[0] == 0.0
    assert jnp.max(updated) <= 1.0 + 1e-6


def test_death_event_is_emitted_once():
    population = _population(energy=(0.1, 0.0, 0.0))
    population, died, death_ids = energy_and_death_step(
        population, jnp.zeros(3), jnp.ones(3), jnp.ones(3), jnp.zeros(3), 1.0, 100
    )
    assert died.tolist() == [True, False, False]
    assert death_ids.tolist() == [0, -1, -1]
    _, died_again, death_ids_again = energy_and_death_step(
        population, jnp.zeros(3), jnp.ones(3), jnp.ones(3), jnp.zeros(3), 1.0, 100
    )
    assert not jnp.any(died_again)
    assert jnp.all(death_ids_again == -1)


def test_lifespan_death_occurs_at_exact_configured_age():
    population = replace(_population(), age=jnp.array([4, 0, 0]))
    _, died, _ = energy_and_death_step(
        population, jnp.zeros(3), jnp.ones(3), jnp.zeros(3), jnp.zeros(3), 1.0, 5
    )
    assert died.tolist() == [True, False, False]


def test_reproduction_assigns_lineage_and_charges_only_success():
    genome_config = GenomeConfig(mutation_std=0.0)
    population = replace(_population(), age=jnp.array([5, 0, 0]))
    lifecycle = LifecycleConfig(
        maturity_age=1,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        birth_transfer_efficiency=0.5,
    )
    child_population, events = reproduction_step(jax.random.PRNGKey(1), population, lifecycle, genome_config)
    assert int(events["birth_count"]) == 1
    assert child_population.alive.tolist() == [True, True, False]
    assert child_population.energy.tolist() == [3.0, 1.0, 0.0]
    assert int(child_population.parent_id[1]) == 0
    assert int(child_population.individual_id[1]) == 1
    assert int(child_population.generation[1]) == 1
    assert jnp.array_equal(child_population.genome[1], population.genome[0])

    full = replace(child_population, alive=jnp.ones(3, dtype=jnp.bool_))
    no_birth, full_events = reproduction_step(jax.random.PRNGKey(2), full, lifecycle, genome_config)
    assert int(full_events["birth_count"]) == 0
    assert jnp.array_equal(no_birth.energy, full.energy)


def test_birth_is_an_energy_transfer_and_cannot_create_organism_energy():
    genome_config = GenomeConfig(mutation_std=0.0)
    population = replace(_population(), age=jnp.array([5, 0, 0]))
    lifecycle = LifecycleConfig(
        maturity_age=1,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        birth_transfer_efficiency=1.0,
    )
    before = jnp.sum(population.energy * population.alive)
    child_population, events = reproduction_step(
        jax.random.PRNGKey(7), population, lifecycle, genome_config
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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mutation_probability": -0.1}, "mutation_probability"),
        ({"mutation_probability": 1.1}, "mutation_probability"),
        ({"mutation_std": -0.1}, "mutation_std"),
        ({"max_bending_delta": -0.1}, "max_bending_delta"),
        ({"resource_reference": -0.1}, "resource_reference"),
        ({"resource_reference": 1.1}, "resource_reference"),
    ],
)
def test_genome_config_rejects_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GenomeConfig(**kwargs)


def test_mutation_is_deterministic_bounded_and_zero_std_is_exact():
    genome = initialize_genomes(jax.random.PRNGKey(0), 1)[0]
    config = GenomeConfig(mutation_probability=1.0, mutation_std=100.0)
    a = _fixed_gaussian_child(jax.random.PRNGKey(4), genome, config)
    b = _fixed_gaussian_child(jax.random.PRNGKey(4), genome, config)
    assert jnp.array_equal(a, b)
    assert jnp.all(jnp.isfinite(a))
    clone = _fixed_gaussian_child(
        jax.random.PRNGKey(5), genome, GenomeConfig(mutation_std=0.0)
    )
    assert jnp.array_equal(clone, genome)
