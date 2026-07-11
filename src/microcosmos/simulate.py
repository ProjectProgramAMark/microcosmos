import functools

import jax
import jax.numpy as jnp

from microcosmos.forces import compute_field_steric_force_corrected
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.utils import periodic_boundary
from dataclasses import replace
from microcosmos.solver.config import ConstraintSolverConfig
from microcosmos.structs.fields import Fields
from microcosmos.solver.fluid import immersed_boundary_interaction
from microcosmos.solver.masks import PhysicsContext, activity_masks

@functools.partial(jax.jit, static_argnames=["dt", "num_steps", "solver_config"])
def simulate(
    nodes: Nodes,
    edges: Edges,
    fields: Fields,
    dt: float,
    num_steps: int,
    solver_config: ConstraintSolverConfig,
    physics_context: PhysicsContext | None = None,
) -> tuple[Nodes, Fields]:
    def scan_step(carry, t: float) -> tuple[tuple[Nodes, Edges, Fields], tuple[Nodes, Fields]]:
        nodes, edges, fields = carry
        nodes, edges, fields = step(
            nodes, edges, fields, dt, solver_config, physics_context
        )

        # OPTIMIZATION: Only save what we render
        # Drop f_grid from history to save VRAM.
        # Downsample fluid_velocity to grid_shape if LBM runs at a different resolution.
        fH, fW = fields.fluid_velocity.shape[1], fields.fluid_velocity.shape[2]
        H, W = fields.grid_shape
        fluid_vel_render = (
            jax.image.resize(fields.fluid_velocity, (2, H, W), method="linear")
            if (fH, fW) != (H, W) else fields.fluid_velocity
        )
        fields_for_render = replace(fields, f_grid=None, fluid_velocity=fluid_vel_render)

        return (nodes, edges, fields), (nodes, fields_for_render)

    (nodes, edges, fields), (nodes_timeseries, fields_timeseries) = jax.lax.scan(scan_step, (nodes, edges, fields), length=num_steps)
    return nodes_timeseries, fields_timeseries

@functools.partial(jax.jit, static_argnames=["dt", "solver_config"])
def step(
    nodes: Nodes,
    edges: Edges,
    fields: Fields,
    dt: float,
    solver_config: ConstraintSolverConfig,
    physics_context: PhysicsContext | None = None,
) -> tuple[Nodes, Edges, Fields]:

    # 1. Reset debug vector
    nodes = replace(nodes, debug_vector=jnp.zeros_like(nodes.position))
    active = (
        None
        if nodes.active is None
        else nodes.active.astype(nodes.position.dtype)[:, None]
    )
    masks = None if nodes.active is None else activity_masks(nodes, edges)

    # 2. Store previous position (UNWRAPPED)
    # We need this to calculate the TRUE distance traveled later.
    prev_position = nodes.position
    damped_velocity = nodes.velocity * solver_config.damping
    if active is not None:
        damped_velocity = damped_velocity * active
    nodes = replace(nodes, velocity=damped_velocity)

    # --- INERTIA STEP ---
    # Apply velocity. Note: We do NOT wrap the position here yet.
    predicted_position = prev_position + nodes.velocity * dt

    # 3. Apply External Forces (Steric)
    # FFT-Gaussian field, curl-free bilinear sampling, analytic n-neighbor subtraction.
    # Set steric_neighbor_skip=0 to disable the subtraction (each node feels own + neighbors' bumps).
    if solver_config.enable_steric:
        fields, f_steric = compute_field_steric_force_corrected(
            replace(nodes, position=predicted_position),
            fields,
            sigma=solver_config.steric_sigma,
            neighbor_skip=solver_config.steric_neighbor_skip,
            scatter_value=solver_config.steric_scatter_value,
            masks=masks,
            physics_context=physics_context,
        )
        predicted_position += f_steric * solver_config.steric_strength * dt
    if active is not None:
        predicted_position = jnp.where(active.astype(jnp.bool_), predicted_position, prev_position)
    # Update nodes with this temporary unwrapped position for the solver
    nodes = replace(nodes, position=predicted_position)

    # 5. Solve Constraints
    # For stable Cosserat: store inertial prediction y in debug_vector before the loop.
    # The constraint function reads it each iteration to compute inertia forces.
    # solver_config is static, so this branch is resolved at JIT trace time.
    if solver_config.solver_type == "cosserat":
        nodes = replace(nodes, debug_vector=nodes.position)

    bending_kwargs = {
        "config": solver_config,
        "grid_shape": fields.grid_shape,
    }
    if masks is not None:
        bending_kwargs["masks"] = masks
    bending_fn = functools.partial(
        solver_config.bending_constraint,
        **bending_kwargs,
    )

    carry = (nodes, edges)
    carry = jax.lax.fori_loop(0, solver_config.cycles_per_step, bending_fn, carry)
    nodes, edges = carry

    # 5. Update Node Velocity (The PBD Magic)
    # NOW we calculate velocity. Since we haven't wrapped yet,
    # (nodes.position - prev_position) is the true physical vector.
    current_velocity = (nodes.position - prev_position) / dt
    if active is not None:
        current_velocity = current_velocity * active
    nodes = replace(nodes, velocity=current_velocity)

    # 6. Fluid Evolution + Immersed Boundary Coupling
    if solver_config.enable_fluid:
        # Combined IBM + LBM with properly integrated Guo forcing
        nodes, fields, f_grid_new = immersed_boundary_interaction(
            nodes, edges, fields, fields.f_grid, dt, solver_config, masks
        )
        fields = replace(fields, f_grid=f_grid_new)

    # 7. FINAL WRAP
    # After all physics (pos -> constraints -> vel) are done,
    # wrap the position back into the box for the next frame.
    final_wrapped_position = periodic_boundary(fields.grid_shape, nodes.position)
    if active is not None:
        final_wrapped_position = jnp.where(
            active.astype(jnp.bool_), final_wrapped_position, prev_position
        )
        nodes = replace(nodes, velocity=nodes.velocity * active)
    nodes = replace(nodes, position=final_wrapped_position)

    return nodes, edges, fields
