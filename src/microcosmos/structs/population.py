"""Fixed-capacity ecosystem state and telemetry."""

from dataclasses import dataclass
import functools

import jax

from microcosmos.cppn import CPPNGenome
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
        "controller_order",
        "controller_connection_index",
        "founder_lineage_id",
        "intake_ema",
        "population_change_ema",
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
    genome: CPPNGenome
    controller_order: jax.Array
    controller_connection_index: jax.Array
    founder_lineage_id: jax.Array
    intake_ema: jax.Array
    population_change_ema: jax.Array
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
        "actuator_gain",
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
    actuator_gain: jax.Array
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
        "action_diversity",
        "operator_counts",
        "policy_violation_count",
        "infrastructure_valid",
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
    action_diversity: jax.Array
    operator_counts: jax.Array
    policy_violation_count: jax.Array
    infrastructure_valid: jax.Array
    # Slot arrays are useful for deterministic replay and spawn initialization.
    birth_parent_slots: jax.Array
    birth_child_slots: jax.Array
