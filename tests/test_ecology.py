from dataclasses import replace

import jax
import jax.numpy as jnp

from microcosmos.controller import GenomeConfig, controller_action, initialize_genomes, mutate_genome
from microcosmos.ecology import LifecycleConfig, ResourceConfig, energy_and_death_step, reproduction_step, resource_step
from microcosmos.structs.population import PopulationState


def _population(energy=(5.0, 0.0, 0.0)):
    count = len(energy)
    alive = jnp.arange(count) == 0
    genomes = initialize_genomes(jax.random.PRNGKey(0), count, GenomeConfig(hidden_size=2))
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
    genome_config = GenomeConfig(hidden_size=2, mutation_std=0.0)
    population = replace(_population(), age=jnp.array([5, 0, 0]))
    lifecycle = LifecycleConfig(
        maturity_age=1,
        reproduction_threshold=4.0,
        reproduction_cost=2.0,
        offspring_initial_energy=1.0,
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


def test_mutation_is_deterministic_bounded_and_zero_std_is_exact():
    genome = initialize_genomes(jax.random.PRNGKey(0), 1, GenomeConfig(hidden_size=2))[0]
    config = GenomeConfig(hidden_size=2, mutation_probability=1.0, mutation_std=100.0)
    a = mutate_genome(jax.random.PRNGKey(4), genome, config)
    b = mutate_genome(jax.random.PRNGKey(4), genome, config)
    assert jnp.array_equal(a, b)
    assert jnp.all(jnp.isfinite(a))
    clone = mutate_genome(jax.random.PRNGKey(5), genome, GenomeConfig(hidden_size=2, mutation_std=0.0))
    assert jnp.array_equal(clone, genome)


def test_controller_depends_on_genome_and_masks_inactive_slots():
    config = GenomeConfig(hidden_size=2)
    genomes = jnp.zeros_like(initialize_genomes(jax.random.PRNGKey(0), 2, config))
    genomes = genomes.at[0, -5].set(-2.0).at[1, -5].set(2.0)
    common = dict(
        genomes=genomes,
        normalized_coordinate=jnp.array([0.0, 0.0]),
        bending_slot=jnp.array([0, 1]),
        phase=jnp.zeros(2),
        local_resource=jnp.zeros(2),
        energy_fraction=jnp.ones(2),
        config=config,
    )
    action = controller_action(alive=jnp.array([True, True]), **common)
    masked = controller_action(alive=jnp.array([True, False]), **common)
    assert action[0] != action[1]
    assert masked[1] == 0.0
