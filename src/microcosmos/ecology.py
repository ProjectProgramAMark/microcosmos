"""Resource, mortality, mutation, and fixed-shape birth operations."""

from dataclasses import dataclass, replace
import math

import jax
import jax.numpy as jnp

from microcosmos.controller import GenomeConfig, mutate_genome
from microcosmos.structs.population import PopulationState


@dataclass(frozen=True)
class ResourceConfig:
    capacity: float = 1.0
    regeneration_rate: float = 0.01
    diffusion_rate: float = 0.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        for name, value in (
            ("capacity", self.capacity),
            ("regeneration_rate", self.regeneration_rate),
            ("diffusion_rate", self.diffusion_rate),
            ("eps", self.eps),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.capacity < 0.0:
            raise ValueError("capacity must be non-negative")
        if self.regeneration_rate < 0.0:
            raise ValueError("regeneration_rate must be non-negative")
        if self.diffusion_rate < 0.0:
            raise ValueError("diffusion_rate must be non-negative")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")


@dataclass(frozen=True)
class LifecycleConfig:
    maximum_lifespan: int = 10_000
    maturity_age: int = 100
    reproduction_threshold: float = 4.0
    reproduction_cost: float = 2.0
    birth_transfer_efficiency: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.maximum_lifespan, int) or isinstance(
            self.maximum_lifespan, bool
        ):
            raise ValueError("maximum_lifespan must be an integer")
        if not isinstance(self.maturity_age, int) or isinstance(
            self.maturity_age, bool
        ):
            raise ValueError("maturity_age must be an integer")
        if self.maximum_lifespan <= 0:
            raise ValueError("maximum_lifespan must be positive")
        if not 0 <= self.maturity_age < self.maximum_lifespan:
            raise ValueError(
                "maturity_age must be non-negative and less than maximum_lifespan"
            )

        for name, value in (
            ("reproduction_threshold", self.reproduction_threshold),
            ("reproduction_cost", self.reproduction_cost),
            ("birth_transfer_efficiency", self.birth_transfer_efficiency),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.reproduction_cost <= 0.0:
            raise ValueError("reproduction_cost must be positive")
        if self.reproduction_threshold < self.reproduction_cost:
            raise ValueError(
                "reproduction_threshold must be at least reproduction_cost"
            )
        if not 0.0 < self.birth_transfer_efficiency <= 1.0:
            raise ValueError("birth_transfer_efficiency must be within (0, 1]")

    @property
    def child_initial_energy(self) -> float:
        return self.reproduction_cost * self.birth_transfer_efficiency


def sample_grid_nearest(field: jax.Array, positions: jax.Array) -> jax.Array:
    """Periodically sample a scalar grid at the nearest cell."""
    h, w = field.shape
    x = jnp.floor(positions[:, 0] + 0.5).astype(jnp.int32) % w
    y = jnp.floor(positions[:, 1] + 0.5).astype(jnp.int32) % h
    return field[y, x]


def resource_step(
    resource: jax.Array,
    positions: jax.Array,
    node_active: jax.Array,
    node_slot: jax.Array,
    uptake_rate: jax.Array,
    dt: float,
    config: ResourceConfig,
) -> tuple[jax.Array, jax.Array]:
    """Conservatively fulfill concurrent demand, then diffuse and regenerate."""
    h, w = resource.shape
    x = jnp.floor(positions[:, 0] + 0.5).astype(jnp.int32) % w
    y = jnp.floor(positions[:, 1] + 0.5).astype(jnp.int32) % h
    flat_index = y * w + x

    demand = node_active.astype(resource.dtype) * uptake_rate[node_slot] * dt
    demand_by_cell = jnp.zeros(h * w, dtype=resource.dtype).at[flat_index].add(demand)
    resource_flat = resource.reshape(-1)
    fulfillment = jnp.minimum(
        1.0, resource_flat / (demand_by_cell + config.eps)
    )
    node_uptake = demand * fulfillment[flat_index]
    fulfilled_by_cell = jnp.zeros(h * w, dtype=resource.dtype).at[flat_index].add(
        node_uptake
    )
    depleted = jnp.maximum(resource_flat - fulfilled_by_cell, 0.0).reshape(h, w)

    if config.diffusion_rate > 0.0:
        neighbor_mean = 0.25 * (
            jnp.roll(depleted, 1, axis=0)
            + jnp.roll(depleted, -1, axis=0)
            + jnp.roll(depleted, 1, axis=1)
            + jnp.roll(depleted, -1, axis=1)
        )
        diffusion = jnp.clip(config.diffusion_rate * dt, 0.0, 1.0)
        depleted = depleted + diffusion * (neighbor_mean - depleted)

    regenerated = jnp.minimum(
        depleted + config.regeneration_rate * dt, config.capacity
    )
    gross_uptake = jnp.zeros_like(uptake_rate).at[node_slot].add(node_uptake)
    return jnp.maximum(regenerated, 0.0), gross_uptake


def actuation_energy_by_slot(
    bending_delta: jax.Array,
    bending_slot: jax.Array,
    num_slots: int,
    power_coefficient: float,
    dt: float,
) -> jax.Array:
    """Convert squared bending commands into per-slot energy for one step."""
    power = jnp.zeros(num_slots, dtype=bending_delta.dtype).at[bending_slot].add(
        bending_delta**2
    )
    return power * power_coefficient * dt


def energy_and_death_step(
    population: PopulationState,
    gross_uptake: jax.Array,
    assimilation_efficiency: jax.Array,
    basal_metabolism: jax.Array,
    actuation_energy: jax.Array,
    dt: float,
    maximum_lifespan: int,
) -> tuple[PopulationState, jax.Array, jax.Array]:
    """Update energy/age and apply starvation and age-based death once."""
    alive = population.alive
    energy = population.energy + alive * (
        gross_uptake * assimilation_efficiency
        - basal_metabolism * dt
        - actuation_energy
    )
    age = population.age + alive.astype(jnp.int32)
    died = alive & ((energy <= 0.0) | (age >= maximum_lifespan))
    death_ids = jnp.where(died, population.individual_id, -1)
    updated = replace(population, alive=alive & ~died, energy=energy, age=age)
    return updated, died, death_ids


def reproduction_step(
    key: jax.Array,
    population: PopulationState,
    lifecycle: LifecycleConfig,
    genome_config: GenomeConfig,
) -> tuple[PopulationState, dict[str, jax.Array]]:
    """Rank parents and free slots, then create a fixed-shape birth batch."""
    capacity = population.alive.shape[0]
    slots = jnp.arange(capacity, dtype=jnp.int32)
    eligible = (
        population.alive
        & (population.age >= lifecycle.maturity_age)
        & (population.energy >= lifecycle.reproduction_threshold)
        & (population.energy >= lifecycle.reproduction_cost)
    )
    free = ~population.alive

    parent_order = jnp.argsort(
        jnp.where(eligible, -population.energy, jnp.inf), stable=True
    ).astype(jnp.int32)
    child_order = jnp.argsort(jnp.where(free, slots, capacity), stable=True).astype(
        jnp.int32
    )
    birth_count = jnp.minimum(jnp.sum(eligible), jnp.sum(free)).astype(jnp.int32)
    valid = slots < birth_count
    parent_slots = jnp.where(valid, parent_order, -1)
    child_slots = jnp.where(valid, child_order, -1)

    parent_safe = jnp.maximum(parent_slots, 0)
    keys = jax.random.split(key, capacity)
    mutated = jax.vmap(lambda k, g: mutate_genome(k, g, genome_config))(
        keys, population.genome[parent_safe]
    )

    # Invalid ranked entries target one-past-capacity and are dropped. This maps
    # ranked births into destination slots in O(C) storage rather than materializing
    # a C x C assignment matrix.
    child_scatter = jnp.where(valid, child_order, capacity)
    child_present = jnp.zeros(capacity, dtype=jnp.bool_).at[child_scatter].set(
        valid, mode="drop"
    )
    parent_for_child = jnp.zeros(capacity, dtype=jnp.int32).at[child_scatter].set(
        parent_safe, mode="drop"
    )
    child_genome = jnp.zeros_like(population.genome).at[child_scatter].set(
        mutated, mode="drop"
    )
    child_ids_ranked = population.next_individual_id + slots
    child_ids = jnp.zeros(capacity, dtype=jnp.int32).at[child_scatter].set(
        child_ids_ranked, mode="drop"
    )

    parent_scatter = jnp.where(valid, parent_order, capacity)
    parent_cost = jnp.zeros(capacity, dtype=population.energy.dtype).at[
        parent_scatter
    ].add(
        valid.astype(population.energy.dtype) * lifecycle.reproduction_cost,
        mode="drop",
    )
    energy = population.energy - parent_cost
    energy = jnp.where(child_present, lifecycle.child_initial_energy, energy)
    age = jnp.where(child_present, 0, population.age)
    generation = jnp.where(
        child_present, population.generation[parent_for_child] + 1, population.generation
    )
    parent_id = jnp.where(
        child_present, population.individual_id[parent_for_child], population.parent_id
    )
    individual_id = jnp.where(child_present, child_ids, population.individual_id)
    genome = jnp.where(child_present[:, None], child_genome, population.genome)

    updated = replace(
        population,
        alive=population.alive | child_present,
        energy=energy,
        age=age,
        generation=generation,
        parent_id=parent_id,
        individual_id=individual_id,
        genome=genome,
        next_individual_id=population.next_individual_id + birth_count,
    )
    events = {
        "birth_count": birth_count,
        "parent_slots": parent_slots,
        "child_slots": child_slots,
        "parent_ids": jnp.where(valid, population.individual_id[parent_safe], -1),
        "child_ids": jnp.where(valid, child_ids_ranked, -1),
    }
    return updated, events
