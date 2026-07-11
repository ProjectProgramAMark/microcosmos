import jax
import jax.numpy as jnp
import numpy as np
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.fields import Fields
from dataclasses import replace
from microcosmos.utils import EPSIL, displacement
from typing import TYPE_CHECKING
from microcosmos.solver.masks import ActivityMasks, activity_masks

if TYPE_CHECKING:
    from .config import ConstraintSolverConfig
    from microcosmos.structs.edges import Edges

# --- Constants ---
LBM_VELOCITIES = np.array([
    [0, 0],   [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
])

LBM_WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Opposite direction indices for D2Q9 (e_ī = -e_i)
OPP_INDICES = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

# --- Helper Functions (Kernel & Interpolation) ---
def _kernel_1d_hat(r, half_width):
    """Linear (hat) kernel with the given half-width."""
    return jnp.maximum(0.0, 1.0 - jnp.abs(r) / half_width)


def _kernel_1d_peskin(r, h):
    """Peskin 4-point cosine kernel."""
    r_abs = jnp.abs(r) / h
    return jnp.where(
        r_abs <= 2.0,
        0.25 * (1.0 + jnp.cos(jnp.pi * r_abs / 2.0)),
        0.0
    )

def get_weights_and_indices(node_pos_grid, grid_shape, kernel_size):
    """Returns indices and weights for a kernel_size × kernel_size neighborhood.

    Works for any integer kernel_size (odd or even).  Weights are normalized
    so they always sum to 1 regardless of where the node sits on the grid.
    """
    h, w = grid_shape
    x, y = node_pos_grid
    half = (kernel_size - 1) // 2
    half_width = kernel_size / 2.0
    start_x = jnp.floor(x) - half
    start_y = jnp.floor(y) - half

    offsets = jnp.arange(float(kernel_size))
    dy, dx = jnp.meshgrid(offsets, offsets)

    dx_flat = dx.flatten()
    dy_flat = dy.flatten()

    neighbor_x = start_x + dx_flat
    neighbor_y = start_y + dy_flat

    dist_x = neighbor_x - x
    dist_y = neighbor_y - y

    wx = _kernel_1d_peskin(dist_x, half_width)
    wy = _kernel_1d_peskin(dist_y, half_width)
    # wx = _kernel_1d_hat(dist_x, half_width)
    # wy = _kernel_1d_hat(dist_y, half_width)
    weights = wx * wy
    weights = weights / (jnp.sum(weights) + EPSIL)

    # Wrap X by width, Y by height
    idx_x = neighbor_x.astype(jnp.int32) % w
    idx_y = neighbor_y.astype(jnp.int32) % h

    # Multiply Y index by width for correct row-major 1D flattening
    flat_indices = idx_y * w + idx_x

    return flat_indices, weights


def _precompute_ibm_weights(grid_pos, h, w, kernel_size):
    """Precompute kernel indices and weights for all nodes once.

    Returns (all_flat_indices, all_weights) each of shape (N, kernel_size**2).
    """
    return jax.vmap(lambda pos: get_weights_and_indices(pos, (h, w), kernel_size))(grid_pos)


def _interpolate_velocity_at_nodes(all_flat_indices, all_weights, fluid_velocity):
    """Interpolate fluid velocity at Lagrangian node positions using precomputed kernel data."""
    def get_node_velocity(flat_indices, weights):
        vel_flat = fluid_velocity.reshape(2, -1)
        v_neighbors = vel_flat[:, flat_indices]
        return jnp.sum(v_neighbors * weights, axis=1)

    return jax.vmap(get_node_velocity)(all_flat_indices, all_weights)


def _interpolate_scalar_at_nodes(all_flat_indices, all_weights, scalar_field):
    """Interpolate a scalar field at Lagrangian node positions using precomputed kernel data."""
    def get_node_value(flat_indices, weights):
        neighbors = scalar_field.reshape(-1)[flat_indices]
        return jnp.sum(neighbors * weights)

    return jax.vmap(get_node_value)(all_flat_indices, all_weights)


def _spread_force_to_grid(all_flat_indices, all_weights, force, h, w):
    """Spread Lagrangian forces to Eulerian grid using precomputed kernel data."""
    n_stencil = all_flat_indices.shape[1]
    flat_idxs = all_flat_indices.reshape(-1)
    fx_expanded = jnp.repeat(force[:, 0], n_stencil)
    fy_expanded = jnp.repeat(force[:, 1], n_stencil)
    weights_expanded = all_weights.reshape(-1)

    momentum_update_x = jnp.zeros(h * w).at[flat_idxs].add(fx_expanded * weights_expanded)
    momentum_update_y = jnp.zeros(h * w).at[flat_idxs].add(fy_expanded * weights_expanded)

    return jnp.stack([momentum_update_x.reshape(h, w), momentum_update_y.reshape(h, w)], axis=0)


def _spread_weight_to_grid(all_flat_indices, all_weights, h, w):
    """Compute accumulated kernel weight at each grid cell from precomputed kernel data.

    When nodes are densely packed (spacing < kernel width), each grid cell
    receives overlapping contributions from multiple nodes. Dividing spread
    forces by this weight field cancels that overcounting.
    """
    flat_idxs = all_flat_indices.reshape(-1)
    weights_expanded = all_weights.reshape(-1)
    weight_grid = jnp.zeros(h * w).at[flat_idxs].add(weights_expanded)
    return weight_grid.reshape(h, w)


def _compute_macroscopic(f_grid):
    """Compute macroscopic density and velocity from distribution."""
    density = jnp.sum(f_grid, axis=0)
    density = jnp.maximum(density, 0.1)
    u_momentum = jnp.tensordot(LBM_VELOCITIES, f_grid, axes=([0], [0]))
    velocity = u_momentum / density
    return density, velocity


def _compute_equilibrium(density, velocity):
    """Compute equilibrium distribution f_eq(rho, u)."""
    u_dot_v = jnp.einsum('axy, ia -> ixy', velocity, LBM_VELOCITIES)
    u_sq = jnp.sum(velocity**2, axis=0)
    return density * LBM_WEIGHTS[:, None, None] * (
        1.0 + 3.0 * u_dot_v + 4.5 * (u_dot_v**2) - 1.5 * u_sq
    )


# --- Core LBM Functions ---
def _trt_omega_minus(omega_plus, lambda_trt=0.25):
    """Compute antisymmetric relaxation rate ω⁻ from the magic parameter Λ.

    Λ = (1/ω⁺ - 0.5)(1/ω⁻ - 0.5)  →  ω⁻ = 1 / (Λ/(1/ω⁺ - 0.5) + 0.5)
    """
    tau_plus_offset = 1.0 / jnp.maximum(omega_plus, EPSIL) - 0.5
    return 1.0 / (lambda_trt / jnp.maximum(tau_plus_offset, EPSIL) + 0.5)


def _trt_collide(f, feq, omega_plus, omega_minus):
    """TRT collision: relax symmetric and antisymmetric non-equilibrium independently.

    ω⁺ relaxes (f⁺ - feq⁺) — controls viscosity.
    ω⁻ relaxes (f⁻ - feq⁻) — free parameter for stability/accuracy.
    """
    f_opp = f[OPP_INDICES]
    feq_opp = feq[OPP_INDICES]
    return f - omega_plus * (0.5 * (f + f_opp) - 0.5 * (feq + feq_opp)) \
             - omega_minus * (0.5 * (f - f_opp) - 0.5 * (feq - feq_opp))


def _guo_source_raw(u_star, force_field):
    """Raw Guo source term (without relaxation prefactor)."""
    e_dot_u = jnp.einsum('ia, axy -> ixy', LBM_VELOCITIES, u_star)
    e_dot_F = jnp.einsum('ia, axy -> ixy', LBM_VELOCITIES, force_field)
    u_dot_F = jnp.sum(u_star * force_field, axis=0)

    term1 = 3.0 * (e_dot_F - u_dot_F[None, :, :])
    term2 = 9.0 * e_dot_u * e_dot_F

    return LBM_WEIGHTS[:, None, None] * (term1 + term2)


def _trt_guo_source(u_star, force_field, omega_plus, omega_minus):
    """Guo source term with TRT-aware prefactors.

    Symmetric part (stress-related) uses (1 - ω⁺/2),
    antisymmetric part (momentum-related) uses (1 - ω⁻/2).
    Reduces to standard Guo when ω⁺ = ω⁻.
    """
    raw = _guo_source_raw(u_star, force_field)
    raw_opp = raw[OPP_INDICES]
    sym = 0.5 * (raw + raw_opp)
    anti = 0.5 * (raw - raw_opp)
    return (1.0 - 0.5 * omega_plus) * sym + (1.0 - 0.5 * omega_minus) * anti


def immersed_boundary_interaction(
    nodes: Nodes,
    edges: "Edges",
    fields: Fields,
    f_grid: jax.Array,
    dt: float,
    solver_config: "ConstraintSolverConfig",
    masks: ActivityMasks | None = None,
) -> tuple[Nodes, Fields, jax.Array]:
    """Combined IBM + LBM step with properly integrated Guo forcing.

    The Guo source term must be applied during collision — not as a
    separate step before or after — to avoid double-counting the force
    in the equilibrium. This function performs:

    1. Macroscopic quantities from distributions
    2. IBM: multi-direct forcing to compute body force
    3. Force magnitude clamp (prevents NaN from overlapping nodes)
    4. Collision with Guo forcing (equilibrium at u*, source term S_i)
    5. Streaming
    6. Node drag (Newton's 3rd law)
    """
    h, w = f_grid.shape[1], f_grid.shape[2]
    if masks is None and nodes.active is not None:
        masks = activity_masks(nodes, edges)
    node_mask = None if masks is None else masks[0]
    edge_mask = None if masks is None else masks[1]

    scale_x = w / fields.grid_shape[0]
    scale_y = h / fields.grid_shape[1]
    scale = jnp.array([scale_x, scale_y])
    mean_scale = 0.5 * (scale_x + scale_y)

    # Scale velocity clamp by resolution so the *physical* velocity limit
    # stays constant.  At scale=2 the same physical velocity is 2× in lattice
    # units; without scaling, the clamp artificially halves the achievable
    # physical velocity and drains momentum every step.
    max_fluid_vel = solver_config.max_fluid_velocity * mean_scale

    node_vel_lattice = nodes.velocity * scale * dt
    # Stop gradient on grid positions: the Peskin kernel's spatial derivatives
    # create numerical instability when differentiated through lax.scan.
    grid_pos = jax.lax.stop_gradient(nodes.position) * scale

    # Synthetic IBM nodes: place a pair of virtual nodes at ±width along the chain normal.
    # These interact with the fluid instead of the control nodes, giving the filament
    # an effective cross-section and increasing normal drag without modifying the PBD rig.
    # When synthetic_node_width=0, fall back to the original single-node behaviour.
    N = nodes.position.shape[0]
    if solver_config.synthetic_node_width > 0.0:
        # Topology-aware perpendicular: accumulate edge vectors at each node so the
        # chain normal is derived from actual connectivity, not storage order.
        src, tgt = edges.pairs[:, 0], edges.pairs[:, 1]
        edge_vecs = displacement(fields.grid_shape, nodes.position[tgt], nodes.position[src])
        if edge_mask is not None:
            edge_vecs = edge_vecs * edge_mask[:, None]
        tangents_raw = jnp.zeros_like(nodes.position)
        tangents_raw = tangents_raw.at[src].add(edge_vecs)
        tangents_raw = tangents_raw.at[tgt].add(edge_vecs)
        tangent_norms = jnp.sqrt(jnp.sum(tangents_raw**2, axis=-1, keepdims=True) + EPSIL)
        tangents = tangents_raw / tangent_norms  # (N, 2)
        perp = jnp.stack([-tangents[:, 1], tangents[:, 0]], axis=-1)  # (N, 2)

        d = solver_config.synthetic_node_width
        ibm_pos = jax.lax.stop_gradient(jnp.concatenate([
            grid_pos + d * perp * mean_scale,
            grid_pos - d * perp * mean_scale,
        ], axis=0))  # (2N, 2)
        ibm_vel = jnp.tile(node_vel_lattice, (2, 1))  # (2N, 2)
        ibm_active = (
            None
            if node_mask is None
            else jnp.tile(node_mask.astype(node_vel_lattice.dtype), 2)
        )
    else:
        ibm_pos = grid_pos        # (N, 2)
        ibm_vel = node_vel_lattice  # (N, 2)
        ibm_active = (
            None if node_mask is None else node_mask.astype(node_vel_lattice.dtype)
        )

    f = jnp.maximum(f_grid, EPSIL)

    # 1. Macroscopic
    density, velocity = _compute_macroscopic(f)
    vel_mag = jnp.sqrt(jnp.sum(velocity**2, axis=0) + EPSIL)
    vel_clamp = jnp.minimum(1.0, max_fluid_vel / (vel_mag + EPSIL))
    velocity = velocity * vel_clamp[None, :, :]

    # 2. IBM: multi-direct forcing to compute velocity correction
    relaxation = solver_config.ibm_relaxation

    # Precompute kernel indices and weights once for all (synthetic) nodes.
    # These depend only on ibm_pos (which is stop_gradient'd and constant
    # within this function), so reusing them avoids 10+ redundant evaluations.
    all_flat_indices, all_kernel_weights = _precompute_ibm_weights(ibm_pos, h, w, solver_config.ibm_kernel_size)
    if ibm_active is not None:
        all_kernel_weights = all_kernel_weights * ibm_active[:, None]

    # Kernel weight field: how much total kernel coverage each grid cell receives
    # from all Lagrangian nodes. Dense nodes (spacing < kernel width) produce
    # weight >> 1; dividing spread corrections by this cancels the overcounting.
    kernel_weight_grid = _spread_weight_to_grid(all_flat_indices, all_kernel_weights, h, w)
    inv_kernel_weight = 1.0 / jnp.maximum(kernel_weight_grid[None, :, :], 1.0)

    def ibm_iteration(carry, _):
        u_corrected, total_node_clipped = carry
        fluid_vel_at_nodes = _interpolate_velocity_at_nodes(
            all_flat_indices, all_kernel_weights, u_corrected)
        velocity_deficit = ibm_vel - fluid_vel_at_nodes

        # Clamp per-node correction to bound both grid injection and node reaction
        step_correction = velocity_deficit * relaxation
        mag = jnp.sqrt(jnp.sum(step_correction**2, axis=-1, keepdims=True) + EPSIL)
        clamp_factor = jnp.minimum(1.0, max_fluid_vel / (mag + EPSIL))
        step_correction = step_correction * clamp_factor

        delta_u_grid = _spread_force_to_grid(
            all_flat_indices, all_kernel_weights, step_correction, h, w)
        # Normalize by kernel weight to remove overcounting from dense node packing.
        delta_u_grid = delta_u_grid * inv_kernel_weight

        clipped = jnp.sum(mag.squeeze(-1) * (1.0 - clamp_factor.squeeze(-1)))
        return (u_corrected + delta_u_grid, total_node_clipped + clipped), None

    (u_corrected, total_node_clipped), _ = jax.lax.scan(
        ibm_iteration,
        (velocity, jnp.zeros(())),
        xs=None, length=solver_config.ibm_iterations,
    )
    # jax.debug.print(
    #     "[clamp] vel={p:.1f}% ({f:.2f} cells)  node={c:.2f}  total_mom={t:.0f}",
    #     p=100.0 * momentum_removed / (total_momentum + EPSIL),
    #     f=jnp.mean(vel_clamp < 1.0),
    #     t=total_momentum,
    #     c=total_node_clipped,
    # )
    # Post-accumulation velocity clamp: IBM iterations clamp per-step corrections,
    # but accumulated u_corrected can still exceed safe LBM velocity. This is the
    # final safety net before collision.
    u_corr_mag = jnp.sqrt(jnp.sum(u_corrected**2, axis=0) + EPSIL)
    u_corr_clamp = jnp.minimum(1.0, max_fluid_vel / (u_corr_mag + EPSIL))
    u_corrected = u_corrected * u_corr_clamp[None, :, :]

    # Body force: F = Δu * ρ
    force_field = (u_corrected - velocity) * density

    # 3. Force magnitude clamp (prevents NaN from overlapping Lagrangian points).
    # Breaks momentum conservation — disabled by default.
    if solver_config.enable_force_clamp:
        force_mag = jnp.sqrt(jnp.sum(force_field**2, axis=0) + EPSIL)
        fclamp = jnp.minimum(1.0, max_fluid_vel / (force_mag + EPSIL))
        force_field = force_field * fclamp[None, :, :]

    # 4. TRT collision with Guo forcing
    omega_plus = 1.0 / (3.0 * solver_config.viscosity + 0.5)
    omega_plus = jnp.clip(omega_plus, 0.0, 1.9)
    omega_minus = _trt_omega_minus(omega_plus, solver_config.lambda_trt)

    u_star = velocity + 0.5 * force_field / density
    feq = _compute_equilibrium(density, u_star)

    f_post = _trt_collide(f, feq, omega_plus, omega_minus) + _trt_guo_source(u_star, force_field, omega_plus, omega_minus)
    f_post = jnp.maximum(f_post, EPSIL)

    # 5. Streaming
    def stream_step(f_i, vel):
        return jnp.roll(f_i, shift=(vel[0].astype(int), vel[1].astype(int)), axis=(1, 0))

    f_new = jax.vmap(stream_step)(f_post, LBM_VELOCITIES)
    f_new = jnp.maximum(f_new, EPSIL)

    # 6. Node drag (Newton's 3rd law)
    # Interpolate the Eulerian force field back to Lagrangian (synthetic) nodes.
    # This reads what was actually injected into the fluid (after spreading and
    # inv_kernel_weight normalisation), so the reaction is exactly momentum-conserving.
    # For synthetic nodes: average the two perpendicular samples back to the control node.
    ibm_force_lattice = _interpolate_velocity_at_nodes(
        all_flat_indices, all_kernel_weights, force_field)
    if solver_config.synthetic_node_width > 0.0:
        node_force_lattice = 0.5 * (ibm_force_lattice[:N] + ibm_force_lattice[N:])
    else:
        node_force_lattice = ibm_force_lattice

    # Node mass in lattice units (resolution-independent).
    # node_vel_lattice ∝ scale, and the velocity conversion divides by scale·dt,
    # so node_mass must stay constant for the physical drag to be grid-size independent.
    node_mass_lattice = solver_config.node_mass

    # Newton's 3rd law: node loses the momentum given to the fluid.
    # force_field > 0 means fluid gained momentum → node must lose it.
    # The caller does `velocity - node_delta_v`, so no extra negation needed here.
    # Convert to physical: Δv_phys = Δv_lat / (scale · dt)
    node_delta_v = (node_force_lattice / node_mass_lattice) / (scale * dt)
    if node_mask is not None:
        node_delta_v = node_delta_v * node_mask[:, None]

    nodes_new = replace(nodes, velocity=nodes.velocity - node_delta_v)

    fields_new = replace(fields, fluid_velocity=u_star)

    return nodes_new, fields_new, f_new
