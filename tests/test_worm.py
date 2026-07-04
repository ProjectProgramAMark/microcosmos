import jax
import jax.numpy as jnp
from dataclasses import replace
from pathlib import Path
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.structs.fields import Fields
from microcosmos.simulate import step
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.graph import make_edges, compute_bending_pairs
from microcosmos.rendering import animate
from microcosmos.utils import displacement


def test_worm():
    """Minimal test that a sine wave swimmer produces locomotion."""
    # Setup
    num_nodes = 20
    grid_shape = (50, 50)
    num_wavelengths = 2
    rotation_speed = 0.1  # rad per step
    init_line_distance = 2.0
    dt = 0.01
    num_steps = 200
    minimum_displacement = 0.2

    # Create line topology
    edge_pairs, edge_departure_angles = make_edges(num_nodes, graph="line", num_instances=1)
    num_edges = edge_pairs.shape[0]

    # Arrange nodes along x-axis with sine wave y positions
    domain_width = grid_shape[1]
    domain_height = grid_shape[0]

    x_positions = jnp.linspace(domain_width * 0.1, domain_width * 0.9, num_nodes)
    amplitude = domain_height * 0.3
    k = 2 * jnp.pi * num_wavelengths / (x_positions[-1] - x_positions[0])

    # Initial phase = 0
    phase = 0.0
    y_positions = domain_height / 2 + amplitude * jnp.sin(k * x_positions + phase)

    positions = jnp.stack([x_positions, y_positions], axis=1)
    size = tuple(grid_shape)

    # Compute initial per-edge theta from positions
    src = edge_pairs[:, 0]
    tgt = edge_pairs[:, 1]
    deltas = displacement(size, positions[tgt], positions[src])
    initial_edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

    # Compute bending pairs
    bending_pairs, bending_rest_angles_init = compute_bending_pairs(edge_pairs, edge_departure_angles, num_nodes)

    # Compute initial bending rest angles from sine wave
    y_prime = amplitude * k * jnp.cos(k * x_positions + phase)
    target_thetas = jnp.arctan(y_prime)
    # Bending rest angles: difference between consecutive edge thetas
    init_bending_rest_angles = target_thetas[1:-1] - target_thetas[:-2]

    # Create Nodes and Edges
    nodes = Nodes(
        position=positions,
        velocity=jnp.zeros_like(positions),
        color_bend=jnp.arange(num_nodes) % 3,
        debug_vector=jnp.zeros((num_nodes, 2)),
    )

    edges = Edges(
        pairs=edge_pairs,
        theta=initial_edge_theta,
        rest_lengths=jnp.full(num_edges, init_line_distance),
        bending_pairs=bending_pairs,
        bending_rest_angles=init_bending_rest_angles,
        bending_stiffness=jnp.ones(bending_pairs.shape[0]) * 0.5,
    )

    # Initialize fields
    h, w = grid_shape
    weights = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    f_grid = jnp.zeros((9, h, w))
    f_grid = f_grid.at[:].set(weights[:, None, None])

    fields = Fields(
        grid_shape=grid_shape,
        steric=jnp.zeros(grid_shape),
        fluid_velocity=jnp.zeros((2, h, w)),
        f_grid=f_grid,
    )

    # Track initial center of mass
    initial_com_x = jnp.mean(positions[:, 0])

    # Store trajectories for visualization
    nodes_trajectory = []
    fields_trajectory = []

    # Run simulation with phase updates
    current_phase = phase
    for _ in range(num_steps):
        # Update bending rest angles based on current phase
        y_prime = amplitude * k * jnp.cos(k * x_positions + current_phase)
        target_thetas = jnp.arctan(y_prime)
        new_bending_rest_angles = target_thetas[1:-1] - target_thetas[:-2]

        edges = edges.__replace__(bending_rest_angles=new_bending_rest_angles)

        # Step simulation
        nodes, edges, fields = step(nodes, edges, fields, dt, PBD_SCHEME)

        # Store for visualization
        nodes_trajectory.append(nodes)
        fields_trajectory.append(fields)

        # Advance phase
        current_phase += rotation_speed

    # Verify results
    assert isinstance(nodes, Nodes)
    assert nodes.position.shape == positions.shape
    assert jnp.all(jnp.isfinite(nodes.position))
    assert jnp.all(jnp.isfinite(nodes.velocity))

    # Verify locomotion occurred (center of mass moved)
    final_com_x = jnp.mean(nodes.position[:, 0])
    displacement_val = float(jnp.abs(initial_com_x - final_com_x))

    # After 200 steps with phase rotation, there should be some displacement
    assert displacement_val > minimum_displacement, f"Expected some displacement, got {displacement_val}"
    assert displacement_val < domain_width, f"Displacement {displacement_val} seems unreasonably large"

    print(f"Worm test passed: displacement = {displacement_val:.4f}")

    # Create visualization
    output_dir = Path("tests/outputs")
    output_dir.mkdir(exist_ok=True)

    # Stack trajectories into timeseries
    nodes_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *nodes_trajectory)
    fields_stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *fields_trajectory)

    # Save animation
    animate(
        nodes_stacked,
        fields_stacked,
        filename=str(output_dir / "worm.gif"),
        subsample=2,  # Save every 2nd frame for smaller file size
        uniform_color=False,  # Use colors to show the wave pattern
        verbose=False,
        animate_fluid_velocity=False,
    )
    print(f"Visualization saved to {output_dir / 'worm.gif'}")


def _run_worm_at_viscosity(viscosity: float, num_steps: int = 500,
                                   displacement_start_step: int = 50) -> float:
    """Run a worm at a given viscosity and return the cumulative displacement.

    Uses periodic-boundary-aware cumulative displacement tracking (matching
    the approach in swim_base.py) and skips an initial transient period.
    """
    from microcosmos.utils import periodic_boundary

    num_nodes = 20
    grid_shape = (64, 64)
    num_wavelengths = 1
    rotation_speed = 0.05
    init_line_distance = 0.5
    dt = 0.01

    edge_pairs, edge_departure_angles = make_edges(num_nodes, graph="line", num_instances=1)
    num_edges = edge_pairs.shape[0]

    domain_width = grid_shape[1]
    domain_height = grid_shape[0]

    x_positions = jnp.linspace(domain_width * 0.3, domain_width * 0.7, num_nodes)
    amplitude = domain_height * 0.3
    k = 2 * jnp.pi * num_wavelengths / (x_positions[-1] - x_positions[0])

    phase = 0.0
    y_positions = domain_height / 2 + amplitude * jnp.sin(k * x_positions + phase)
    positions = jnp.stack([x_positions, y_positions], axis=1)
    size = tuple(grid_shape)

    src = edge_pairs[:, 0]
    tgt = edge_pairs[:, 1]
    deltas = displacement(size, positions[tgt], positions[src])
    initial_edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

    bending_pairs, _ = compute_bending_pairs(edge_pairs, edge_departure_angles, num_nodes)

    y_prime = amplitude * k * jnp.cos(k * x_positions + phase)
    target_thetas = jnp.arctan(y_prime)
    init_bending_rest_angles = target_thetas[1:-1] - target_thetas[:-2]

    nodes = Nodes(
        position=positions,
        velocity=jnp.zeros_like(positions),
        color_bend=jnp.arange(num_nodes) % 3,
        debug_vector=jnp.zeros((num_nodes, 2)),
    )

    edges = Edges(
        pairs=edge_pairs,
        theta=initial_edge_theta,
        rest_lengths=jnp.full(num_edges, init_line_distance),
        bending_pairs=bending_pairs,
        bending_rest_angles=init_bending_rest_angles,
        bending_stiffness=jnp.ones(bending_pairs.shape[0]),
    )

    h, w = grid_shape
    weights = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
    f_grid = jnp.zeros((9, h, w))
    f_grid = f_grid.at[:].set(weights[:, None, None])

    fields = Fields(
        grid_shape=grid_shape,
        steric=jnp.zeros(grid_shape),
        fluid_velocity=jnp.zeros((2, h, w)),
        f_grid=f_grid,
    )

    solver_config = replace(PBD_SCHEME, viscosity=viscosity)

    # Periodic-boundary-aware cumulative displacement tracking (from swim_base)
    def periodic_com(pos):
        ref = pos[0]
        deltas = displacement(size, pos, ref)
        mean_delta = jnp.mean(deltas, axis=0)
        return periodic_boundary(size, ref + mean_delta)

    prev_com = periodic_com(positions)
    cumulative_disp = jnp.zeros(2)

    current_phase = phase
    for i in range(num_steps):
        y_prime = amplitude * k * jnp.cos(k * x_positions + current_phase)
        target_thetas = jnp.arctan(y_prime)
        new_bending_rest_angles = target_thetas[1:-1] - target_thetas[:-2]
        edges = edges.__replace__(bending_rest_angles=new_bending_rest_angles)

        nodes, edges, fields = step(nodes, edges, fields, dt, solver_config)

        current_com = periodic_com(nodes.position)
        step_delta = displacement(size, current_com, prev_com)
        prev_com = current_com
        if i >= displacement_start_step:
            cumulative_disp = cumulative_disp + step_delta

        current_phase += rotation_speed

    assert jnp.all(jnp.isfinite(nodes.position)), f"NaN positions at viscosity={viscosity}"
    assert jnp.all(jnp.isfinite(nodes.velocity)), f"NaN velocities at viscosity={viscosity}"

    return float(jnp.linalg.norm(cumulative_disp))


# def test_worm_viscosity_ordering():
#     """Higher viscosity should produce less displacement than lower viscosity."""
#     viscosities = [0.001, 0.1, 0.9]

#     displacements = {}
#     for v in viscosities:
#         d = _run_worm_at_viscosity(v)
#         displacements[v] = d
#         print(f"viscosity={v}: displacement={d:.4f}")

#     # Assert monotonically decreasing displacement with increasing viscosity
#     for v_low, v_high in zip(viscosities[:-1], viscosities[1:]):
#         assert displacements[v_low] > displacements[v_high], (
#             f"Expected displacement at viscosity={v_low} ({displacements[v_low]:.4f}) "
#             f"> displacement at viscosity={v_high} ({displacements[v_high]:.4f})"
#         )
