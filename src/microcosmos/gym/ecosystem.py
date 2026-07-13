"""Continuous fixed-capacity ecosystem environment."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.cppn import (
    CPPNGenome,
    DYNAMIC_NODE_KEY_OFFSET,
    MAX_EXACT_FLOAT32_INTEGER,
    controller_action,
    initialize_cppn_population,
    transform_and_validate_population,
)
from microcosmos.ecology import (
    LifecycleConfig,
    OperatorCreditConfig,
    ResourceConfig,
    actuation_energy_by_slot,
    energy_and_death_step,
    make_periodic_resource_patch,
    reproduction_step,
    resource_step,
    sample_grid_nearest,
)
from microcosmos.graph import compute_bending_pairs, make_fields
from microcosmos.heredity import (
    NUM_OPERATORS,
    R4_NUM_OPERATORS,
    OffspringPolicy,
    R4OffspringPolicy,
    R4_MIXED,
    action_diversity,
    fixed_mixed_policy,
    fixed_r4_policy,
    population_statistics,
    r4_population_statistics,
    update_intake_ema,
    update_population_change_ema,
)
from microcosmos.rng import RNGTag, derive_key, keys_for_identities
from microcosmos.simulate import step as physics_step
from microcosmos.solver.config import ConstraintSolverConfig, PBD_SCHEME_NO_FLUID
from microcosmos.solver.masks import build_replicated_physics_context
from microcosmos.structs.edges import Edges
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.population import (
    EcosystemState,
    EcosystemTelemetry,
    PopulationState,
)
from microcosmos.utils import displacement, periodic_boundary

from .base import Environment
from .multi_agent import CreatureTopology, LineTopology


class EcosystemEnv(Environment):
    """A homogeneous ecosystem whose arrays are sized by capacity, not occupancy."""

    def __init__(
        self,
        topology: CreatureTopology | None = None,
        max_creatures: int = 16,
        initial_population: int = 4,
        grid_shape: tuple[int, int] = (128, 128),
        dt: float = 0.01,
        max_steps: int = 1_000,
        solver_config: ConstraintSolverConfig = PBD_SCHEME_NO_FLUID,
        resource_capacity: float = 1.0,
        initial_resource: float = 1.0,
        resource_regeneration_rate: float = 0.01,
        resource_diffusion_rate: float = 0.0,
        resource_patch_center: tuple[float, float] | None = None,
        resource_patch_radius: float | None = None,
        initial_energy: float = 2.0,
        birth_transfer_efficiency: float = 0.5,
        reproduction_threshold: float = 4.0,
        reproduction_cost: float = 2.0,
        maturity_age: int = 100,
        maximum_lifespan: int = 10_000,
        max_bending_delta: float = 0.35,
        uptake_rate: float = 0.5,
        assimilation_efficiency: float = 0.8,
        basal_metabolism: float = 0.05,
        actuation_power_coefficient: float = 0.01,
        spawn_separation: float | None = None,
        placement_candidates: int = 16,
        position_margin: float = 1.0,
        offspring_policy: OffspringPolicy | R4OffspringPolicy | None = None,
        *,
        founder_genome: CPPNGenome | None = None,
        heredity_contract: str = "legacy",
        credit_chunk_steps: int = 100,
    ):
        if topology is None:
            topology = LineTopology(num_nodes=8)
        if (
            not isinstance(grid_shape, tuple)
            or len(grid_shape) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in grid_shape)
        ):
            raise ValueError("grid_shape must contain two positive integers")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool):
            raise ValueError("max_steps must be an integer")
        if max_steps < 1:
            raise ValueError("max_steps must be at least one")
        if not isinstance(max_creatures, int) or isinstance(max_creatures, bool):
            raise ValueError("max_creatures must be an integer")
        if max_creatures < 1:
            raise ValueError("max_creatures must be at least one")
        if not isinstance(initial_population, int) or isinstance(initial_population, bool):
            raise ValueError("initial_population must be an integer")
        if not 0 <= initial_population <= max_creatures:
            raise ValueError("initial_population must be within ecosystem capacity")
        if topology.num_nodes < 2:
            raise ValueError("ecosystem topology requires at least two nodes")
        if not np.isfinite(topology.spacing) or topology.spacing <= 0.0:
            raise ValueError("topology spacing must be finite and positive")
        if not np.isfinite(topology.bending_stiffness) or topology.bending_stiffness < 0.0:
            raise ValueError("topology bending_stiffness must be finite and non-negative")
        for name, value in (
            ("initial_resource", initial_resource),
            ("initial_energy", initial_energy),
            ("actuation_power_coefficient", actuation_power_coefficient),
            ("uptake_rate", uptake_rate),
            ("assimilation_efficiency", assimilation_efficiency),
            ("basal_metabolism", basal_metabolism),
            ("max_bending_delta", max_bending_delta),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= assimilation_efficiency <= 1.0:
            raise ValueError("assimilation_efficiency must be within [0, 1]")
        if heredity_contract not in ("legacy", "r4"):
            raise ValueError("heredity_contract must be 'legacy' or 'r4'")
        if not isinstance(credit_chunk_steps, int) or isinstance(credit_chunk_steps, bool) or credit_chunk_steps < 1:
            raise ValueError("credit_chunk_steps must be a positive integer")
        if offspring_policy is None:
            offspring_policy = fixed_r4_policy(R4_MIXED) if heredity_contract == "r4" else fixed_mixed_policy
        if not callable(offspring_policy):
            raise ValueError("offspring_policy must be callable")
        if not isinstance(topology, LineTopology):
            raise ValueError("ecosystem controller requires LineTopology")
        if initial_resource > resource_capacity:
            raise ValueError("initial_resource must not exceed resource_capacity")
        if resource_diffusion_rate * dt > 1.0:
            raise ValueError("resource_diffusion_rate * dt must be at most one")
        if resource_patch_center is None:
            resource_patch_center = (grid_shape[1] / 2.0, grid_shape[0] / 2.0)
        if not isinstance(resource_patch_center, tuple) or len(resource_patch_center) != 2 or any(not np.isfinite(value) for value in resource_patch_center):
            raise ValueError("resource_patch_center must contain two finite values")
        if resource_patch_radius is None:
            resource_patch_radius = min(grid_shape) / 4.0
        if not np.isfinite(resource_patch_radius) or resource_patch_radius <= 0.0:
            raise ValueError("resource_patch_radius must be finite and positive")
        if not np.isfinite(position_margin) or position_margin < 0.0:
            raise ValueError("position_margin must be finite and non-negative")
        if not isinstance(placement_candidates, int) or isinstance(placement_candidates, bool) or placement_candidates < 1:
            raise ValueError("placement_candidates must be a positive integer")
        if spawn_separation is not None and (not np.isfinite(spawn_separation) or spawn_separation < 0.0):
            raise ValueError("spawn_separation must be finite and non-negative")
        maximum_dynamic_key = DYNAMIC_NODE_KEY_OFFSET + initial_population + max_steps * max_creatures
        if maximum_dynamic_key >= MAX_EXACT_FLOAT32_INTEGER:
            raise ValueError("maximum child identity exceeds exact float32 node-key range")

        super().__init__(
            grid_shape=grid_shape,
            dt=dt,
            max_steps=max_steps,
            solver_config=solver_config,
        )
        self.topology = topology
        self.max_creatures = int(max_creatures)
        self.initial_population = int(initial_population)
        self.initial_resource = float(initial_resource)
        self.resource_patch_center = tuple(float(value) for value in resource_patch_center)
        self.resource_patch_radius = float(resource_patch_radius)
        self.initial_energy = float(initial_energy)
        self.actuation_power_coefficient = float(actuation_power_coefficient)
        self.position_margin = float(position_margin)
        self.placement_candidates = int(placement_candidates)
        self.resource_config = ResourceConfig(
            capacity=resource_capacity,
            regeneration_rate=resource_regeneration_rate,
            diffusion_rate=resource_diffusion_rate,
        )
        self.lifecycle_config = LifecycleConfig(
            maximum_lifespan=maximum_lifespan,
            maturity_age=maturity_age,
            reproduction_threshold=reproduction_threshold,
            reproduction_cost=reproduction_cost,
            birth_transfer_efficiency=birth_transfer_efficiency,
        )
        self.max_bending_delta = float(max_bending_delta)
        self.offspring_policy = offspring_policy
        self.heredity_contract = heredity_contract
        self.num_operators = R4_NUM_OPERATORS if heredity_contract == "r4" else NUM_OPERATORS
        self.credit_config = OperatorCreditConfig(
            tau_steps=float(maturity_age + 2 * credit_chunk_steps),
        )
        self.uptake_rate = float(uptake_rate)
        self.assimilation_efficiency = float(assimilation_efficiency)
        self.basal_metabolism = float(basal_metabolism)
        self._uptake_rate_by_slot = jnp.full(self.max_creatures, self.uptake_rate, dtype=jnp.float32)
        self._assimilation_by_slot = jnp.full(
            self.max_creatures,
            self.assimilation_efficiency,
            dtype=jnp.float32,
        )
        self._metabolism_by_slot = jnp.full(self.max_creatures, self.basal_metabolism, dtype=jnp.float32)

        local_pairs, departure_angles = topology.edges()
        local_positions = topology.local_positions().astype(jnp.float32)
        body_radius = float(jnp.max(jnp.linalg.norm(local_positions, axis=-1)))
        self.body_radius = body_radius
        self.spawn_separation = 2.0 * body_radius + self.position_margin if spawn_separation is None else float(spawn_separation)
        local_bending, local_bending_rest = compute_bending_pairs(local_pairs, departure_angles, topology.num_nodes)
        edges_per_slot = int(local_pairs.shape[0])
        bending_per_slot = int(local_bending.shape[0])

        self._nodes_per_slot = topology.num_nodes
        self._edges_per_slot = edges_per_slot
        self._bending_per_slot = bending_per_slot
        self._num_nodes = topology.num_nodes * self.max_creatures
        self._num_edges = edges_per_slot * self.max_creatures
        self._num_bending_pairs = bending_per_slot * self.max_creatures
        self._local_positions = local_positions
        self._node_slices = tuple(slice(i * topology.num_nodes, (i + 1) * topology.num_nodes) for i in range(self.max_creatures))
        self._edge_slices = tuple(slice(i * edges_per_slot, (i + 1) * edges_per_slot) for i in range(self.max_creatures))
        self._bending_slices = tuple(slice(i * bending_per_slot, (i + 1) * bending_per_slot) for i in range(self.max_creatures))

        node_offsets = jnp.arange(self.max_creatures, dtype=jnp.int32) * topology.num_nodes
        edge_offsets = jnp.arange(self.max_creatures, dtype=jnp.int32) * edges_per_slot
        self._edge_pairs = (local_pairs[None, :, :] + node_offsets[:, None, None]).reshape(-1, 2)
        self._bending_pairs = (local_bending[None, :, :] + edge_offsets[:, None, None]).reshape(-1, 2)
        self._rest_lengths = jnp.tile(
            jnp.full(edges_per_slot, topology.spacing, dtype=jnp.float32),
            self.max_creatures,
        )
        self._bending_rest_angles = jnp.tile(local_bending_rest, self.max_creatures)
        self._bending_stiffness = jnp.full(self._num_bending_pairs, topology.bending_stiffness, dtype=jnp.float32)
        self.node_slot = jnp.repeat(jnp.arange(self.max_creatures, dtype=jnp.int32), topology.num_nodes)
        self.edge_slot = jnp.repeat(jnp.arange(self.max_creatures, dtype=jnp.int32), edges_per_slot)
        self.bending_slot = jnp.repeat(jnp.arange(self.max_creatures, dtype=jnp.int32), bending_per_slot)
        self.physics_context = build_replicated_physics_context(
            local_pairs,
            topology.num_nodes,
            self.max_creatures,
            solver_config.steric_neighbor_skip,
        )
        if bending_per_slot:
            local_coordinate = jnp.linspace(-1.0, 1.0, bending_per_slot)
            self._bending_coordinate = jnp.tile(local_coordinate, self.max_creatures)
            self._bending_node = self._edge_pairs[self._bending_pairs[:, 0], 1]
        else:
            self._bending_coordinate = jnp.zeros(0)
            self._bending_node = jnp.zeros(0, dtype=jnp.int32)

        (
            self._initial_genomes,
            self._initial_controller_order,
            self._initial_controller_connection_index,
        ) = self._controller_population(founder_genome)

    def _controller_population(
        self,
        founder_genome: CPPNGenome | None,
    ) -> tuple[CPPNGenome, jax.Array, jax.Array]:
        initial_genomes = initialize_cppn_population(
            self.max_creatures,
            self.initial_population,
            founder_genome=founder_genome,
        )
        genomes, order, connection_index, founder_valid = transform_and_validate_population(initial_genomes)
        if not bool(np.all(np.asarray(jax.device_get(founder_valid)))):
            raise RuntimeError("canonical founder CPPN panel failed validation")
        return genomes, order, connection_index

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def num_bending_pairs(self) -> int:
        return self._num_bending_pairs

    @property
    def node_slices(self) -> tuple[slice, ...]:
        return self._node_slices

    @property
    def edge_slices(self) -> tuple[slice, ...]:
        return self._edge_slices

    @property
    def bending_slices(self) -> tuple[slice, ...]:
        return self._bending_slices

    def _edge_theta(self, positions: jax.Array) -> jax.Array:
        delta = displacement(
            self.grid_shape,
            positions[self._edge_pairs[:, 1]],
            positions[self._edge_pairs[:, 0]],
        )
        return jnp.arctan2(delta[:, 1], delta[:, 0])

    def _select_best_center(
        self,
        candidates: jax.Array,
        occupied_centers: jax.Array,
        occupied: jax.Array,
    ) -> jax.Array:
        """Choose the candidate farthest from its nearest occupied center."""
        delta = displacement(
            self.grid_shape,
            candidates[:, None, :],
            occupied_centers[None, :, :],
        )
        distance_sq = jnp.sum(delta * delta, axis=-1)
        distance_sq = jnp.where(occupied[None, :], distance_sq, jnp.inf)
        nearest_distance_sq = jnp.min(distance_sq, axis=1)
        return candidates[jnp.argmax(nearest_distance_sq)]

    def _slot_centers(self, positions: jax.Array) -> jax.Array:
        """Compute body centers correctly when a body crosses a periodic edge."""
        slot_positions = positions.reshape(self.max_creatures, self._nodes_per_slot, 2)
        anchors = slot_positions[:, :1, :]
        local = displacement(self.grid_shape, slot_positions, anchors)
        return periodic_boundary(self.grid_shape, anchors[:, 0, :] + jnp.mean(local, axis=1))

    def _centers(self, key: jax.Array) -> jax.Array:
        h, w = self.grid_shape
        extent = jnp.max(jnp.abs(self._local_positions), axis=0)
        margin = jnp.minimum(
            extent + self.position_margin,
            jnp.array([w, h], dtype=jnp.float32) / 2.0,
        )
        span = jnp.maximum(jnp.array([w, h], dtype=jnp.float32) - 2 * margin, 0.0)
        candidates = (
            margin
            + jax.random.uniform(
                key,
                (self.max_creatures, self.placement_candidates, 2),
            )
            * span
        )
        live_slots = jnp.arange(self.max_creatures) < self.initial_population

        def place_one(carry, values):
            centers, occupied = carry
            slot, slot_candidates, is_live = values
            selected = self._select_best_center(slot_candidates, centers, occupied)
            # Inactive slots still receive deterministic legal positions, but do
            # not reserve space or influence later active placement.
            selected = jnp.where(is_live, selected, slot_candidates[0])
            centers = centers.at[slot].set(selected)
            occupied = occupied.at[slot].set(is_live)
            return (centers, occupied), None

        initial = (
            jnp.zeros((self.max_creatures, 2), dtype=jnp.float32),
            jnp.zeros(self.max_creatures, dtype=jnp.bool_),
        )
        (centers, _), _ = jax.lax.scan(
            place_one,
            initial,
            (jnp.arange(self.max_creatures), candidates, live_slots),
        )
        return centers

    def _observation(self, state: EcosystemState) -> jax.Array:
        pop = state.population
        energy_scale = max(self.lifecycle_config.reproduction_threshold, 1e-6)
        age_scale = max(self.lifecycle_config.maximum_lifespan, 1)
        return jnp.stack(
            [
                pop.alive.astype(jnp.float32),
                pop.energy / energy_scale,
                pop.age.astype(jnp.float32) / age_scale,
                pop.generation.astype(jnp.float32),
            ],
            axis=-1,
        ).reshape(-1)

    def reset(
        self,
        key: jax.Array,
        *,
        founder_genome: CPPNGenome | None = None,
    ) -> tuple[jax.Array, EcosystemState]:
        if founder_genome is None:
            initial_genomes = self._initial_genomes
            initial_controller_order = self._initial_controller_order
            initial_controller_connection_index = self._initial_controller_connection_index
        else:
            (
                initial_genomes,
                initial_controller_order,
                initial_controller_connection_index,
            ) = self._controller_population(founder_genome)
        initialization_key = derive_key(key, RNGTag.INITIALIZATION)
        key_centers = jax.random.fold_in(initialization_key, 0)
        centers = self._centers(key_centers)
        positions = (centers[:, None, :] + self._local_positions[None, :, :]).reshape(self.num_nodes, 2)
        positions = periodic_boundary(self.grid_shape, positions)
        slot_ids = jnp.arange(self.max_creatures, dtype=jnp.int32)
        alive = slot_ids < self.initial_population
        node_active = alive[self.node_slot]
        nodes = Nodes(
            position=positions,
            velocity=jnp.zeros_like(positions),
            color_bend=(self.node_slot % 3).astype(jnp.float32),
            debug_vector=jnp.zeros_like(positions),
            active=node_active,
            component_id=self.node_slot,
        )
        edges = Edges(
            pairs=self._edge_pairs,
            theta=self._edge_theta(positions),
            rest_lengths=self._rest_lengths,
            bending_pairs=self._bending_pairs,
            bending_rest_angles=self._bending_rest_angles,
            bending_stiffness=self._bending_stiffness,
        )
        population = PopulationState(
            alive=alive,
            energy=jnp.where(alive, self.initial_energy, 0.0),
            age=jnp.zeros(self.max_creatures, dtype=jnp.int32),
            generation=jnp.zeros(self.max_creatures, dtype=jnp.int32),
            individual_id=jnp.where(alive, slot_ids, -1),
            parent_id=jnp.full(self.max_creatures, -1, dtype=jnp.int32),
            genome=initial_genomes,
            controller_order=initial_controller_order,
            controller_connection_index=initial_controller_connection_index,
            founder_lineage_id=jnp.where(alive, slot_ids, -1),
            intake_ema=jnp.zeros(self.max_creatures, dtype=jnp.float32),
            population_change_ema=jnp.zeros((), dtype=jnp.float32),
            birth_rate_ema=jnp.zeros((), dtype=jnp.float32),
            death_rate_ema=jnp.zeros((), dtype=jnp.float32),
            birth_operator=jnp.full(self.max_creatures, -1, dtype=jnp.int32),
            birth_step=jnp.full(self.max_creatures, -1, dtype=jnp.int32),
            has_reproduced=jnp.zeros(self.max_creatures, dtype=jnp.bool_),
            genome_changed_from_parent=jnp.zeros(self.max_creatures, dtype=jnp.bool_),
            shock_ancestor_id=jnp.full(self.max_creatures, -1, dtype=jnp.int32),
            operator_success_ema=jnp.full(6, 0.5, dtype=jnp.float32),
            operator_usage_ema=jnp.zeros(6, dtype=jnp.float32),
            operator_evidence_ema=jnp.zeros(6, dtype=jnp.float32),
            next_individual_id=jnp.array(self.initial_population, dtype=jnp.int32),
        )
        capacity_map, regeneration_map = make_periodic_resource_patch(
            self.grid_shape,
            self.resource_patch_center,
            self.resource_patch_radius,
            self.resource_config.capacity,
            self.resource_config.regeneration_rate,
        )
        initial_fraction = 0.0 if self.resource_config.capacity == 0.0 else min(self.initial_resource / self.resource_config.capacity, 1.0)
        resource_stock = capacity_map * initial_fraction
        fields = replace(make_fields(self.grid_shape), energy=resource_stock)
        state = EcosystemState(
            nodes=nodes,
            edges=edges,
            fields=fields,
            population=population,
            time=jnp.array(0, dtype=jnp.int32),
            base_rest_lengths=edges.rest_lengths,
            base_bending_rest_angles=edges.bending_rest_angles,
            actuator_gain=jnp.ones(self._bending_per_slot, dtype=jnp.float32),
            actuation_cost_multiplier=jnp.ones((), dtype=jnp.float32),
            resource_capacity_map=capacity_map,
            resource_regeneration_map=regeneration_map,
        )
        return self._observation(state), state

    def _spawn_children(
        self,
        key: jax.Array,
        nodes: Nodes,
        edges: Edges,
        population: PopulationState,
        events: dict[str, jax.Array],
        timestep: int | jax.Array,
    ) -> tuple[Nodes, Edges]:
        valid = events["child_slots"] >= 0
        parent_safe = jnp.maximum(events["parent_slots"], 0)
        child_slots = events["child_slots"]
        child_scatter = jnp.where(valid, child_slots, self.max_creatures)
        child_present = jnp.zeros(self.max_creatures, dtype=jnp.bool_).at[child_scatter].set(valid, mode="drop")

        slot_positions = nodes.position.reshape(self.max_creatures, self._nodes_per_slot, 2)
        slot_centers = self._slot_centers(nodes.position)

        def place_births(_):
            child_ids = jnp.maximum(events["child_ids"], 0)
            phase_keys = keys_for_identities(key, RNGTag.SPAWN, timestep, child_ids)
            phase = jax.vmap(
                lambda phase_key: jax.random.uniform(
                    phase_key,
                    (1,),
                    minval=0.0,
                    maxval=2.0 * jnp.pi,
                )
            )(phase_keys)
            angles = phase + (2.0 * jnp.pi * jnp.arange(self.placement_candidates, dtype=nodes.position.dtype) / self.placement_candidates)[None, :]
            directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
            candidates = periodic_boundary(
                self.grid_shape,
                slot_centers[parent_safe, None, :] + directions * self.spawn_separation,
            )
            # The population already contains newborn slots. Remove them from
            # initial occupancy, then reserve selected centers in event order.
            occupied = population.alive & ~child_present

            def place_birth(carry, values):
                centers, reserved = carry
                event_valid, child_slot, slot_candidates = values
                selected = self._select_best_center(slot_candidates, centers, reserved)
                scatter_slot = jnp.where(event_valid, child_slot, self.max_creatures)
                centers = centers.at[scatter_slot].set(selected, mode="drop")
                reserved = reserved.at[scatter_slot].set(event_valid, mode="drop")
                return (centers, reserved), None

            (selected_centers, _), _ = jax.lax.scan(
                place_birth,
                (slot_centers, occupied),
                (valid, child_slots, candidates),
            )
            return selected_centers

        child_centers = jax.lax.cond(jnp.any(valid), place_births, lambda _: slot_centers, operand=None)
        child_positions_by_slot = periodic_boundary(
            self.grid_shape,
            child_centers[:, None, :] + self._local_positions[None, :, :],
        )
        all_positions = jnp.where(child_present[:, None, None], child_positions_by_slot, slot_positions).reshape(self.num_nodes, 2)

        node_alive = population.alive[self.node_slot]
        child_node = child_present[self.node_slot]
        velocity = jnp.where(node_alive[:, None] & ~child_node[:, None], nodes.velocity, 0.0)
        color = population.generation[self.node_slot] % 3
        nodes = replace(
            nodes,
            position=all_positions,
            velocity=velocity,
            color_bend=color.astype(jnp.float32),
            debug_vector=jnp.zeros_like(nodes.debug_vector),
            active=node_alive,
        )

        child_edge = child_present[self.edge_slot]
        child_theta = self._edge_theta(all_positions)
        edges = replace(
            edges,
            theta=jnp.where(child_edge, child_theta, edges.theta),
            rest_lengths=self._rest_lengths,
            bending_rest_angles=self._bending_rest_angles,
        )
        return nodes, edges

    def step(
        self,
        key: jax.Array,
        state: EcosystemState,
        action: dict[str, jax.Array] | None = None,
    ) -> tuple[jax.Array, EcosystemState, jax.Array, jax.Array, dict]:
        pop = state.population
        start_alive_count = jnp.sum(pop.alive).astype(jnp.int32)
        # Population occupancy is authoritative at every transition boundary.
        nodes = replace(state.nodes, active=pop.alive[self.node_slot])
        node_stock = sample_grid_nearest(state.fields.energy, nodes.position)
        node_capacity = sample_grid_nearest(state.resource_capacity_map, nodes.position)
        normalized_node_resource = jnp.clip(
            node_stock / (node_capacity + self.resource_config.eps),
            0.0,
            1.0,
        ).reshape(self.max_creatures, self._nodes_per_slot)
        local_resource = normalized_node_resource.reshape(-1)[self._bending_node]
        sensor_width = min(2, self._nodes_per_slot)
        head_resource = jnp.mean(normalized_node_resource[:, :sensor_width], axis=1)
        tail_resource = jnp.mean(normalized_node_resource[:, -sensor_width:], axis=1)
        controller_bend_action = controller_action(
            pop.genome,
            pop.controller_order,
            pop.controller_connection_index,
            self._bending_coordinate,
            state.time.astype(jnp.float32) * self.dt,
            local_resource,
            head_resource,
            tail_resource,
            pop.alive,
            self.max_bending_delta,
        )
        intended_bend_action = controller_bend_action
        rest_action = jnp.zeros(self.num_edges)
        if action is not None:
            rest_action = rest_action + action["d_rest_length"]
            intended_bend_action = intended_bend_action + action["d_bending_angle"]

        actuator_gain = jnp.broadcast_to(
            state.actuator_gain[None, :],
            (self.max_creatures, self._bending_per_slot),
        ).reshape(self.num_bending_pairs)
        effective_bend_action = intended_bend_action * actuator_gain

        acted_edges = replace(
            state.edges,
            rest_lengths=state.base_rest_lengths + rest_action,
            bending_rest_angles=state.base_bending_rest_angles + effective_bend_action,
        )
        nodes, edges, fields = physics_step(
            nodes,
            acted_edges,
            state.fields,
            self.dt,
            self.solver_config,
            self.physics_context,
        )

        mouth_positions = nodes.position.reshape(self.max_creatures, self._nodes_per_slot, 2)[:, 0, :]
        resource, gross_uptake = resource_step(
            fields.energy,
            state.resource_capacity_map,
            state.resource_regeneration_map,
            mouth_positions,
            pop.alive,
            self._uptake_rate_by_slot,
            self.dt,
            self.resource_config,
        )
        fields = replace(fields, energy=resource)
        bending_energy = actuation_energy_by_slot(
            intended_bend_action,
            self.bending_slot,
            self.max_creatures,
            self.actuation_power_coefficient,
            self.dt,
        ) * state.actuation_cost_multiplier
        intake_ema = update_intake_ema(
            pop.intake_ema,
            gross_uptake,
            self._uptake_rate_by_slot,
            pop.alive,
            self.dt,
            self.resource_config.eps,
        )
        pop = replace(pop, intake_ema=intake_ema)
        population, died, death_ids = energy_and_death_step(
            pop,
            gross_uptake,
            self._assimilation_by_slot,
            self._metabolism_by_slot,
            bending_energy,
            self.dt,
            self.lifecycle_config.maximum_lifespan,
        )
        post_death_count = jnp.sum(population.alive).astype(jnp.int32)
        transient_population_change = update_population_change_ema(
            pop.population_change_ema,
            post_death_count - start_alive_count,
            self.max_creatures,
        )
        controller_action_diversity = action_diversity(
            controller_bend_action.reshape(self.max_creatures, self._bending_per_slot),
            population.alive,
            self.max_bending_delta,
        )
        death_count = jnp.sum(died).astype(jnp.int32)
        death_rate_for_birth = (
            (1.0 - 0.10) * population.death_rate_ema
            + 0.10 * death_count.astype(jnp.float32) / self.max_creatures
        )
        if self.heredity_contract == "r4":
            heredity_population_stats = r4_population_statistics(
                population.alive,
                population.energy,
                population.intake_ema,
                transient_population_change,
                self.lifecycle_config.reproduction_threshold,
                population.birth_rate_ema,
                death_rate_for_birth,
                population.operator_success_ema,
                population.operator_usage_ema,
                population.operator_evidence_ema,
            )
        else:
            heredity_population_stats = population_statistics(
                population.alive,
                transient_population_change,
                controller_action_diversity,
                population.founder_lineage_id,
                self.initial_population,
            )
        population, birth_events = reproduction_step(
            key,
            population,
            self.lifecycle_config,
            self.offspring_policy,
            heredity_population_stats,
            state.time,
            heredity_contract=self.heredity_contract,
            died=died if self.heredity_contract == "r4" else None,
            credit_config=self.credit_config if self.heredity_contract == "r4" else None,
        )
        alive_count = jnp.sum(population.alive).astype(jnp.int32)
        population = replace(
            population,
            population_change_ema=update_population_change_ema(
                pop.population_change_ema,
                alive_count - start_alive_count,
                self.max_creatures,
            ),
            birth_rate_ema=(
                (1.0 - 0.10) * population.birth_rate_ema
                + 0.10 * birth_events["birth_count"].astype(jnp.float32) / self.max_creatures
            ),
            death_rate_ema=death_rate_for_birth,
        )
        nodes, edges = self._spawn_children(
            key,
            nodes,
            edges,
            population,
            birth_events,
            state.time,
        )

        live_count_f = jnp.maximum(alive_count.astype(jnp.float32), 1.0)
        telemetry = EcosystemTelemetry(
            alive_count=alive_count,
            birth_count=birth_events["birth_count"],
            death_count=death_count,
            birth_parent_ids=birth_events["parent_ids"],
            birth_child_ids=birth_events["child_ids"],
            death_ids=death_ids,
            resource_total=jnp.sum(resource),
            population_energy_total=jnp.sum(population.energy * population.alive),
            mean_generation=jnp.sum(population.generation * population.alive) / live_count_f,
            action_diversity=controller_action_diversity,
            birth_parent_slots=birth_events["parent_slots"],
            birth_child_slots=birth_events["child_slots"],
            operator_counts=birth_events["operator_counts"],
            operator_probability_sum=birth_events["operator_probability_sum"],
            resolved_success_count=birth_events["resolved_success_count"],
            resolved_failure_count=birth_events["resolved_failure_count"],
            distinct_birth_count=birth_events["distinct_birth_count"],
            resolved_distinct_success_count=birth_events["resolved_distinct_success_count"],
            policy_violation_count=birth_events["policy_violation_count"],
            infrastructure_valid=birth_events["infrastructure_valid"],
        )
        new_state = EcosystemState(
            nodes=nodes,
            edges=edges,
            fields=fields,
            population=population,
            time=state.time + 1,
            base_rest_lengths=state.base_rest_lengths,
            base_bending_rest_angles=state.base_bending_rest_angles,
            actuator_gain=state.actuator_gain,
            actuation_cost_multiplier=state.actuation_cost_multiplier,
            resource_capacity_map=state.resource_capacity_map,
            resource_regeneration_map=state.resource_regeneration_map,
        )
        observation = self._observation(new_state)
        reward = jnp.sum(gross_uptake)
        done = new_state.time >= self.max_steps
        info = {
            "steps": new_state.time,
            "telemetry": telemetry,
            "alive_count": telemetry.alive_count,
            "birth_count": telemetry.birth_count,
            "death_count": telemetry.death_count,
            "birth_parent_ids": telemetry.birth_parent_ids,
            "birth_child_ids": telemetry.birth_child_ids,
            "death_ids": telemetry.death_ids,
            "resource_total": telemetry.resource_total,
            "population_energy_total": telemetry.population_energy_total,
            "mean_generation": telemetry.mean_generation,
            "action_diversity": telemetry.action_diversity,
            "operator_counts": telemetry.operator_counts,
            "operator_probability_sum": telemetry.operator_probability_sum,
            "resolved_success_count": telemetry.resolved_success_count,
            "resolved_failure_count": telemetry.resolved_failure_count,
            "distinct_birth_count": telemetry.distinct_birth_count,
            "resolved_distinct_success_count": telemetry.resolved_distinct_success_count,
            "policy_violation_count": telemetry.policy_violation_count,
            "infrastructure_valid": telemetry.infrastructure_valid,
        }
        return observation, new_state, reward, done, info

    def render(self, state: EcosystemState, size: int = 512) -> np.ndarray:
        from microcosmos.rendering import render_fields

        nodes_ts = jax.tree.map(lambda x: x[None], state.nodes)
        fields_ts = replace(
            state.fields,
            steric=state.fields.steric[None],
            fluid_velocity=state.fields.fluid_velocity[None],
            f_grid=None,
            energy=state.fields.energy[None],
        )
        return render_fields(fields_ts, nodes_ts, sz=size, animate_energy=False)[0]
