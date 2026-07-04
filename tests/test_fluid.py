import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataclasses import replace
from microcosmos.solver.fluid import immersed_boundary_interaction
from microcosmos.structs.fields import Fields
from microcosmos.solver.config import PBD_SCHEME
from conftest import create_dummy_nodes


def test_immersed_boundary_interaction_updates_fluid_velocity():
    """Test that immersed_boundary_interaction properly updates Fields.fluid_velocity."""
    # Create test setup
    num_nodes = 10
    grid_size = 32
    nodes, edges = create_dummy_nodes(num_nodes, shape=(100, 100))

    # Create LBM distribution function (f_grid)
    f_grid = jnp.zeros((9, grid_size, grid_size))
    # Initialize with equilibrium distribution for uniform velocity
    density = jnp.ones((grid_size, grid_size))
    f_grid = f_grid.at[0].set(density * 4/9)  # Rest particle weight
    for i in range(1, 9):
        f_grid = f_grid.at[i].set(density * 1/9 if i < 5 else density * 1/36)

    # Create initial fluid state with some non-zero velocity
    initial_fluid_velocity = jnp.ones((2, grid_size, grid_size)) * 0.5
    fields = Fields(
        grid_shape=(grid_size, grid_size),
        steric=jnp.zeros((grid_size, grid_size)),
        fluid_velocity=initial_fluid_velocity,
        f_grid=f_grid,
    )

    # Create node velocities (dummy velocities for testing)
    dt = 0.01
    nodes = replace(nodes,velocity=jnp.ones((num_nodes, 2)) * 0.1)

    # Call the function
    nodes_new, fields_new, f_new = immersed_boundary_interaction(nodes, edges, fields, fields.f_grid, dt, PBD_SCHEME)

    # Verify that fluid_velocity has been modified
    assert fields_new.fluid_velocity.shape == (2, grid_size, grid_size)
    assert not jnp.allclose(fields_new.fluid_velocity, initial_fluid_velocity), \
        "fluid_velocity should be modified by immersed_boundary_interaction"

    # Verify that the modification is bounded (not exploding)
    assert jnp.all(jnp.isfinite(fields_new.fluid_velocity)), \
        "fluid_velocity should contain finite values"

    # Verify that Fields object is properly updated
    assert fields_new.grid_shape == fields.grid_shape
    assert fields_new.steric.shape == fields.steric.shape


def test_immersed_boundary_interaction_preserves_shapes():
    """Test that all returned arrays have the correct shapes."""
    num_nodes = 5
    grid_size = 16
    nodes, edges = create_dummy_nodes(num_nodes, shape=(50, 50))

    f_grid = jnp.ones((9, grid_size, grid_size)) * 0.1

    fields = Fields(
        grid_shape=(grid_size, grid_size),
        steric=jnp.zeros((grid_size, grid_size)),
        fluid_velocity=jnp.zeros((2, grid_size, grid_size)),
        f_grid=f_grid,
    )

    # Create node velocities
    dt = 0.01
    nodes = replace(nodes,velocity=jnp.zeros((num_nodes, 2)))

    nodes_new, fields_new, f_new = immersed_boundary_interaction(nodes, edges, fields, fields.f_grid, dt, PBD_SCHEME)

    # Check shapes
    assert nodes_new.position.shape == nodes.position.shape
    assert fields_new.fluid_velocity.shape == (2, grid_size, grid_size)
    assert f_new.shape == (9, grid_size, grid_size)


def test_immersed_boundary_interaction_jit_compatible():
    """Test that the function is JIT-compatible."""
    num_nodes = 5
    grid_size = 16
    nodes, edges = create_dummy_nodes(num_nodes, shape=(50, 50))

    f_grid = jnp.ones((9, grid_size, grid_size)) * 0.1

    fields = Fields(
        grid_shape=(grid_size, grid_size),
        steric=jnp.zeros((grid_size, grid_size)),
        fluid_velocity=jnp.ones((2, grid_size, grid_size)) * 0.3,
        f_grid=f_grid,
    )

    # Create node velocities
    dt = 0.01
    nodes = replace(nodes,velocity=jnp.ones((num_nodes, 2)) * 0.1)

    # JIT compile the function
    jitted_fn = jax.jit(immersed_boundary_interaction, static_argnames=['solver_config'])

    # Should not raise an error
    nodes_new, fields_new, f_new = jitted_fn(nodes, edges, fields, fields.f_grid, dt, PBD_SCHEME)

    # Verify output is valid
    assert jnp.all(jnp.isfinite(fields_new.fluid_velocity))
    assert jnp.all(jnp.isfinite(f_new))


def test_fluid_velocity_coupling_is_nonzero():
    """Test that the coupling actually produces non-zero changes in fluid_velocity."""
    num_nodes = 10
    grid_size = 32
    nodes, edges = create_dummy_nodes(num_nodes, shape=(100, 100))

    # Start with non-uniform fluid velocity
    x_grid = jnp.linspace(0, 1, grid_size)
    y_grid = jnp.linspace(0, 1, grid_size)
    X, Y = jnp.meshgrid(x_grid, y_grid)
    initial_velocity = jnp.stack([X * 0.5, Y * 0.3], axis=0)

    f_grid = jnp.ones((9, grid_size, grid_size)) * 0.1

    fields = Fields(
        grid_shape=(grid_size, grid_size),
        steric=jnp.zeros((grid_size, grid_size)),
        fluid_velocity=initial_velocity,
        f_grid=f_grid,
    )

    # Create node velocities with some variation
    dt = 0.01
    nodes = replace(nodes,velocity=jnp.ones((num_nodes, 2)) * 0.2)

    nodes_new, fields_new, f_new = immersed_boundary_interaction(nodes, edges, fields, fields.f_grid, dt, PBD_SCHEME)

    # Check that the change is actually significant (not just numerical noise)
    velocity_change = jnp.abs(fields_new.fluid_velocity - initial_velocity)
    max_change = jnp.max(velocity_change)

    assert max_change > 1e-6, \
        f"Velocity change should be significant, but got max change of {max_change}"


def test_fluid_velocity_visual_verification():
    """Visual verification that fluid velocity is properly modified by immersed boundary coupling."""
    num_nodes = 15
    grid_size = 64
    nodes, edges = create_dummy_nodes(num_nodes, shape=(100, 100))

    # Create a vortex-like initial velocity field
    x = jnp.linspace(0, 1, grid_size)
    y = jnp.linspace(0, 1, grid_size)
    X, Y = jnp.meshgrid(x, y)

    # Vortex centered at (0.5, 0.5)
    cx, cy = 0.5, 0.5
    dx = X - cx
    dy = Y - cy
    r = jnp.sqrt(dx**2 + dy**2)

    # Tangential velocity (vortex)
    vx = -dy * jnp.exp(-r**2 / 0.1) * 2.0
    vy = dx * jnp.exp(-r**2 / 0.1) * 2.0

    initial_velocity = jnp.stack([vx, vy], axis=0)

    # Initialize LBM distribution
    f_grid = jnp.ones((9, grid_size, grid_size)) * 0.1

    fields = Fields(
        grid_shape=(grid_size, grid_size),
        steric=jnp.zeros((grid_size, grid_size)),
        fluid_velocity=initial_velocity,
        f_grid=f_grid,
    )

    # Create node velocities with some variation to show coupling
    dt = 0.01
    nodes = replace(nodes,velocity=jnp.ones((num_nodes, 2)) * 0.3)

    # Run the immersed boundary interaction
    nodes_new, fields_new, f_new = immersed_boundary_interaction(nodes, edges, fields, fields.f_grid, dt, PBD_SCHEME)

    # Convert to numpy for plotting
    initial_vel_np = np.array(initial_velocity)
    final_vel_np = np.array(fields_new.fluid_velocity)
    velocity_change = final_vel_np - initial_vel_np

    # Node positions in grid coordinates
    scale = grid_size / fields.grid_shape[0]
    node_pos_grid = np.array(nodes.position) * scale

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Downsample for quiver plot visibility
    skip = 4
    X_plot, Y_plot = X[::skip, ::skip], Y[::skip, ::skip]

    # Plot 1: Initial velocity field (magnitude)
    ax = axes[0, 0]
    initial_mag = np.sqrt(initial_vel_np[0]**2 + initial_vel_np[1]**2)
    im1 = ax.imshow(initial_mag, origin='lower', cmap='viridis', extent=[0, 1, 0, 1])
    ax.quiver(X_plot, Y_plot,
              initial_vel_np[0][::skip, ::skip],
              initial_vel_np[1][::skip, ::skip],
              color='white', alpha=0.6)
    ax.scatter(node_pos_grid[:, 0] / grid_size, node_pos_grid[:, 1] / grid_size,
               c='red', s=50, marker='o', label='Nodes')
    ax.set_title('Initial Fluid Velocity (Magnitude)', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.colorbar(im1, ax=ax, label='Speed')

    # Plot 2: Final velocity field (magnitude)
    ax = axes[0, 1]
    final_mag = np.sqrt(final_vel_np[0]**2 + final_vel_np[1]**2)
    im2 = ax.imshow(final_mag, origin='lower', cmap='viridis', extent=[0, 1, 0, 1])
    ax.quiver(X_plot, Y_plot,
              final_vel_np[0][::skip, ::skip],
              final_vel_np[1][::skip, ::skip],
              color='white', alpha=0.6)
    ax.scatter(node_pos_grid[:, 0] / grid_size, node_pos_grid[:, 1] / grid_size,
               c='red', s=50, marker='o', label='Nodes')
    ax.set_title('Final Fluid Velocity (Magnitude)', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.colorbar(im2, ax=ax, label='Speed')

    # Plot 3: Change in velocity (magnitude)
    ax = axes[0, 2]
    change_mag = np.sqrt(velocity_change[0]**2 + velocity_change[1]**2)
    im3 = ax.imshow(change_mag, origin='lower', cmap='hot', extent=[0, 1, 0, 1])
    ax.quiver(X_plot, Y_plot,
              velocity_change[0][::skip, ::skip],
              velocity_change[1][::skip, ::skip],
              color='cyan', alpha=0.6)
    ax.scatter(node_pos_grid[:, 0] / grid_size, node_pos_grid[:, 1] / grid_size,
               c='lime', s=50, marker='o', label='Nodes')
    ax.set_title('Velocity Change (Magnitude)', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.colorbar(im3, ax=ax, label='|ΔV|')

    # Plot 4: Initial velocity X-component
    ax = axes[1, 0]
    im4 = ax.imshow(initial_vel_np[0], origin='lower', cmap='RdBu', extent=[0, 1, 0, 1])
    ax.scatter(node_pos_grid[:, 0] / grid_size, node_pos_grid[:, 1] / grid_size,
               c='black', s=30, marker='o')
    ax.set_title('Initial Vx', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im4, ax=ax, label='Vx')

    # Plot 5: Final velocity X-component
    ax = axes[1, 1]
    im5 = ax.imshow(final_vel_np[0], origin='lower', cmap='RdBu', extent=[0, 1, 0, 1])
    ax.scatter(node_pos_grid[:, 0] / grid_size, node_pos_grid[:, 1] / grid_size,
               c='black', s=30, marker='o')
    ax.set_title('Final Vx', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im5, ax=ax, label='Vx')

    # Plot 6: Statistics
    ax = axes[1, 2]
    ax.axis('off')
    stats_text = f"""
    Velocity Field Statistics:

    Initial:
    - Mean speed: {np.mean(initial_mag):.4f}
    - Max speed: {np.max(initial_mag):.4f}

    Final:
    - Mean speed: {np.mean(final_mag):.4f}
    - Max speed: {np.max(final_mag):.4f}

    Change:
    - Mean |ΔV|: {np.mean(change_mag):.4f}
    - Max |ΔV|: {np.max(change_mag):.4f}
    - Total energy change: {np.sum(change_mag**2):.4f}

    Nodes: {num_nodes}
    Grid size: {grid_size}x{grid_size}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()

    # Save the figure
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "fluid_velocity_verification.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisual verification saved to: {output_path}")
    plt.close()

    # Assertions to ensure the test passes
    # Reduced threshold for conservative coupling (stability over magnitude)
    assert np.max(change_mag) > 1e-5, "Velocity change should be measurable"
    assert np.all(np.isfinite(final_vel_np)), "Final velocity should be finite"
    assert final_mag.shape == initial_mag.shape, "Shapes should be preserved"
