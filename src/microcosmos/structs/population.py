"""Fixed-capacity ecosystem state and telemetry."""

from dataclasses import dataclass
import functools

import jax

from microcosmos.structs.edges import Edges
from microcosmos.structs.fields import Fields
from microcosmos.structs.nodes import Nodes


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "alive",
        "energy",
        "age",
        "generation",
        "individual_id",
        "parent_id",
        "genome",
        "next_individual_id",
    ],
)
@dataclass
class PopulationState:
    alive: jax.Array
    energy: jax.Array
    age: jax.Array
    generation: jax.Array
    individual_id: jax.Array
    parent_id: jax.Array
    genome: jax.Array
    next_individual_id: jax.Array


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "nodes",
        "edges",
        "fields",
        "population",
        "time",
        "base_rest_lengths",
        "base_bending_rest_angles",
        "resource_capacity_map",
        "resource_regeneration_map",
    ],
)
@dataclass
class EcosystemState:
    nodes: Nodes
    edges: Edges
    fields: Fields
    population: PopulationState
    time: jax.Array
    base_rest_lengths: jax.Array
    base_bending_rest_angles: jax.Array
    resource_capacity_map: jax.Array
    resource_regeneration_map: jax.Array


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "alive_count",
        "birth_count",
        "death_count",
        "birth_parent_ids",
        "birth_child_ids",
        "death_ids",
        "resource_total",
        "population_energy_total",
        "mean_generation",
        "genome_variance",
        "birth_parent_slots",
        "birth_child_slots",
    ],
)
@dataclass
class EcosystemTelemetry:
    alive_count: jax.Array
    birth_count: jax.Array
    death_count: jax.Array
    birth_parent_ids: jax.Array
    birth_child_ids: jax.Array
    death_ids: jax.Array
    resource_total: jax.Array
    population_energy_total: jax.Array
    mean_generation: jax.Array
    genome_variance: jax.Array
    # Slot arrays are useful for deterministic replay and spawn initialization.
    birth_parent_slots: jax.Array
    birth_child_slots: jax.Array
