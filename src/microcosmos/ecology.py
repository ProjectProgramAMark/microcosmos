"""Resource, mortality, mutation, and fixed-shape birth operations."""

from dataclasses import dataclass, replace
import math

import jax
import jax.numpy as jnp

from microcosmos.cppn import (
    CPPNGenome,
    transform_and_validate_genome,
)
from microcosmos.heredity import (
    CLONE,
    NUM_OPERATORS,
    OffspringPolicy,
    ParentStats,
    PopulationStats,
    make_mutation_context,
    parent_statistics,
)
from microcosmos.rng import RNGTag, derive_key
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
        if not isinstance(self.maximum_lifespan, int) or isinstance(self.maximum_lifespan, bool):
            raise ValueError("maximum_lifespan must be an integer")
        if not isinstance(self.maturity_age, int) or isinstance(self.maturity_age, bool):
            raise ValueError("maturity_age must be an integer")
        if self.maximum_lifespan <= 0:
            raise ValueError("maximum_lifespan must be positive")
        if not 0 <= self.maturity_age < self.maximum_lifespan:
            raise ValueError("maturity_age must be non-negative and less than maximum_lifespan")

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
            raise ValueError("reproduction_threshold must be at least reproduction_cost")
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


def make_periodic_resource_patch(
    grid_shape: tuple[int, int],
    center: tuple[float, float] | jax.Array,
    radius: float,
    peak_capacity: float,
    peak_regeneration: float,
) -> tuple[jax.Array, jax.Array]:
    """Construct compact paraboloid capacity/regeneration maps under PBC."""
    height, width = grid_shape
    center = jnp.asarray(center, dtype=jnp.float32)
    y, x = jnp.meshgrid(
        jnp.arange(height, dtype=jnp.float32),
        jnp.arange(width, dtype=jnp.float32),
        indexing="ij",
    )
    dx = (x - center[0] + width / 2.0) % width - width / 2.0
    dy = (y - center[1] + height / 2.0) % height - height / 2.0
    weight = jnp.maximum(0.0, 1.0 - (dx**2 + dy**2) / radius**2)
    return peak_capacity * weight, peak_regeneration * weight


def resource_step(
    resource: jax.Array,
    capacity_map: jax.Array,
    regeneration_map: jax.Array,
    mouth_positions: jax.Array,
    organism_alive: jax.Array,
    uptake_rate: jax.Array,
    dt: float,
    config: ResourceConfig,
) -> tuple[jax.Array, jax.Array]:
    """Withdraw at mouths, diffuse conservatively, regenerate, and map-clip."""
    h, w = resource.shape
    x = jnp.floor(mouth_positions[:, 0] + 0.5).astype(jnp.int32) % w
    y = jnp.floor(mouth_positions[:, 1] + 0.5).astype(jnp.int32) % h
    flat_index = y * w + x

    demand = organism_alive.astype(resource.dtype) * uptake_rate * dt
    demand_by_cell = jnp.zeros(h * w, dtype=resource.dtype).at[flat_index].add(demand)
    resource_flat = resource.reshape(-1)
    fulfillment = jnp.minimum(1.0, resource_flat / (demand_by_cell + config.eps))
    node_uptake = demand * fulfillment[flat_index]
    fulfilled_by_cell = jnp.zeros(h * w, dtype=resource.dtype).at[flat_index].add(node_uptake)
    depleted = jnp.maximum(resource_flat - fulfilled_by_cell, 0.0).reshape(h, w)

    if config.diffusion_rate > 0.0:
        neighbor_mean = 0.25 * (jnp.roll(depleted, 1, axis=0) + jnp.roll(depleted, -1, axis=0) + jnp.roll(depleted, 1, axis=1) + jnp.roll(depleted, -1, axis=1))
        diffusion = jnp.clip(config.diffusion_rate * dt, 0.0, 1.0)
        depleted = depleted + diffusion * (neighbor_mean - depleted)

    regenerated = depleted + regeneration_map * dt
    updated = jnp.clip(regenerated, 0.0, capacity_map)
    return updated, node_uptake


def actuation_energy_by_slot(
    bending_delta: jax.Array,
    bending_slot: jax.Array,
    num_slots: int,
    power_coefficient: float,
    dt: float,
) -> jax.Array:
    """Convert squared bending commands into per-slot energy for one step."""
    power = jnp.zeros(num_slots, dtype=bending_delta.dtype).at[bending_slot].add(bending_delta**2)
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
    energy = population.energy + alive * (gross_uptake * assimilation_efficiency - basal_metabolism * dt - actuation_energy)
    age = population.age + alive.astype(jnp.int32)
    died = alive & ((energy <= 0.0) | (age >= maximum_lifespan))
    death_ids = jnp.where(died, population.individual_id, -1)
    updated = replace(population, alive=alive & ~died, energy=energy, age=age)
    return updated, died, death_ids


def reproduction_step(
    key: jax.Array,
    population: PopulationState,
    lifecycle: LifecycleConfig,
    offspring_policy: OffspringPolicy,
    population_stats: PopulationStats,
    timestep: int | jax.Array = 0,
) -> tuple[PopulationState, dict[str, jax.Array]]:
    """Rank parents/free slots and apply heredity once per actual birth."""
    capacity = population.alive.shape[0]
    slots = jnp.arange(capacity, dtype=jnp.int32)
    eligible = (
        population.alive
        & (population.age >= lifecycle.maturity_age)
        & (population.energy >= lifecycle.reproduction_threshold)
        & (population.energy >= lifecycle.reproduction_cost)
    )
    free = ~population.alive

    parent_order = jnp.argsort(jnp.where(eligible, -population.energy, jnp.inf), stable=True).astype(jnp.int32)
    child_order = jnp.argsort(jnp.where(free, slots, capacity), stable=True).astype(jnp.int32)
    birth_count = jnp.minimum(jnp.sum(eligible), jnp.sum(free)).astype(jnp.int32)
    valid = slots < birth_count
    parent_slots = jnp.where(valid, parent_order, -1)
    child_slots = jnp.where(valid, child_order, -1)

    parent_safe = jnp.maximum(parent_slots, 0)
    child_ids_ranked = population.next_individual_id + slots

    initial_operator_counts = jnp.zeros(NUM_OPERATORS, dtype=jnp.int32)
    initial_policy_violations = jnp.zeros((), dtype=jnp.int32)
    initial_infrastructure_valid = jnp.ones((), dtype=jnp.bool_)

    def create_births(carry):
        def create_one(i, loop_carry):
            (
                current,
                operator_counts,
                policy_violation_count,
                infrastructure_valid,
            ) = loop_carry
            parent_slot = parent_order[i]
            child_slot = child_order[i]
            child_id = population.next_individual_id + i
            mutation_key = derive_key(
                key,
                RNGTag.MUTATION,
                timestep,
                child_id,
            )
            mutation_context, context_valid = make_mutation_context(
                mutation_key,
                child_id,
            )
            parent_genome = CPPNGenome(
                population.genome.node_genes[parent_slot],
                population.genome.connection_genes[parent_slot],
            )
            stats: ParentStats = parent_statistics(
                parent_genome,
                population.energy[parent_slot],
                population.intake_ema[parent_slot],
                lifecycle.reproduction_threshold,
            )
            proposed = offspring_policy(
                parent_genome,
                stats,
                population_stats,
                mutation_context,
            )
            child_genome, child_order_cache, child_connection_index, graph_valid = transform_and_validate_genome(proposed.genome)

            proposed_operator = jnp.asarray(proposed.operator_index, dtype=jnp.int32)
            operator_in_range = (proposed_operator >= 0) & (proposed_operator < NUM_OPERATORS)
            policy_valid = jnp.asarray(proposed.policy_valid, dtype=jnp.bool_) & operator_in_range
            use_child = policy_valid & context_valid & graph_valid
            actual_operator = jnp.where(
                use_child,
                proposed_operator,
                jnp.asarray(CLONE, dtype=jnp.int32),
            )

            selected_nodes = jnp.where(
                use_child,
                child_genome.node_genes,
                parent_genome.node_genes,
            )
            selected_connections = jnp.where(
                use_child,
                child_genome.connection_genes,
                parent_genome.connection_genes,
            )
            selected_order = jnp.where(
                use_child,
                child_order_cache,
                population.controller_order[parent_slot],
            )
            selected_connection_index = jnp.where(
                use_child,
                child_connection_index,
                population.controller_connection_index[parent_slot],
            )

            current = replace(
                current,
                alive=current.alive.at[child_slot].set(True),
                energy=current.energy.at[parent_slot].add(-lifecycle.reproduction_cost).at[child_slot].set(lifecycle.child_initial_energy),
                age=current.age.at[child_slot].set(0),
                generation=current.generation.at[child_slot].set(population.generation[parent_slot] + 1),
                individual_id=current.individual_id.at[child_slot].set(child_id),
                parent_id=current.parent_id.at[child_slot].set(population.individual_id[parent_slot]),
                genome=CPPNGenome(
                    current.genome.node_genes.at[child_slot].set(selected_nodes),
                    current.genome.connection_genes.at[child_slot].set(selected_connections),
                ),
                controller_order=current.controller_order.at[child_slot].set(selected_order),
                controller_connection_index=(current.controller_connection_index.at[child_slot].set(selected_connection_index)),
                founder_lineage_id=current.founder_lineage_id.at[child_slot].set(population.founder_lineage_id[parent_slot]),
                intake_ema=current.intake_ema.at[child_slot].set(0.0),
            )
            operator_counts = operator_counts.at[actual_operator].add(1)
            policy_violation_count = policy_violation_count + (~policy_valid).astype(jnp.int32)
            infrastructure_valid = infrastructure_valid & context_valid & graph_valid
            return (
                current,
                operator_counts,
                policy_violation_count,
                infrastructure_valid,
            )

        return jax.lax.fori_loop(0, birth_count, create_one, carry)

    initial_carry = (
        population,
        initial_operator_counts,
        initial_policy_violations,
        initial_infrastructure_valid,
    )
    (
        updated,
        operator_counts,
        policy_violation_count,
        infrastructure_valid,
    ) = jax.lax.cond(
        birth_count > 0,
        create_births,
        lambda carry: carry,
        initial_carry,
    )
    updated = replace(
        updated,
        next_individual_id=population.next_individual_id + birth_count,
    )
    events = {
        "birth_count": birth_count,
        "parent_slots": parent_slots,
        "child_slots": child_slots,
        "parent_ids": jnp.where(valid, population.individual_id[parent_safe], -1),
        "child_ids": jnp.where(valid, child_ids_ranked, -1),
        "operator_counts": operator_counts,
        "policy_violation_count": policy_violation_count,
        "infrastructure_valid": infrastructure_valid,
    }
    return updated, events
