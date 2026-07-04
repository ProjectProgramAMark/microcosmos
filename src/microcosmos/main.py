import argparse
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

from microcosmos.utils import jax_timer, displacement
from microcosmos.structs.fields import Fields
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.simulate import simulate
from microcosmos.rendering import animate
from microcosmos.graph import initialize_positions, make_edges, compute_bending_pairs
from microcosmos.graph_builder import GraphBuilder
from microcosmos.solver.config import PBD_SCHEME

# Enable persistent compilation cache to avoid recompiling CUDA kernels on each run
jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/microcosmos/jax_cache"))


LBM_VELOCITIES = np.array([
    [0, 0],   [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1]
])

LBM_WEIGHTS = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])



def make_graph(num_nodes: int, graph_type: str, num_instances: int = 1) -> tuple[jax.Array, jax.Array]:
    """
    Create graph topology using GraphBuilder.

    Args:
        num_nodes: Total number of nodes
        graph_type: Type of graph ('loop', 'line', or 'tree')
        num_instances: Number of independent instances

    Returns:
        edges, edge_departure_angles
    """
    builder = GraphBuilder()
    nodes_per_instance = num_nodes // num_instances
    remainder = num_nodes % num_instances

    for i in range(num_instances):
        # Distribute remainder among first instances
        instance_size = nodes_per_instance + (1 if i < remainder else 0)

        if graph_type == "loop":
            builder.add_loop(instance_size)
        elif graph_type == "line":
            builder.add_line(instance_size)
        elif graph_type == "tree":
            # Tree: trunk with two branches
            trunk_size = int(instance_size * 0.6)
            remaining = instance_size - trunk_size
            branch1_size = remaining // 2
            branch2_size = remaining - branch1_size

            trunk = builder.add_line(trunk_size)
            branch_point = trunk[trunk_size // 2]
            builder.add_branch(from_node=branch_point, length=branch1_size)
            builder.add_branch(from_node=branch_point, length=branch2_size)
        elif graph_type == "tadpole":
            loop_size = int(instance_size * 0.6)
            tail_size = instance_size - loop_size
            loop = builder.add_loop(loop_size)
            branch_point = loop[loop_size // 2]
            builder.add_branch(from_node=branch_point, length=tail_size)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

    return builder.build()


def compute_initial_edge_theta(positions: jax.Array, edge_pairs: jax.Array, size: tuple) -> jax.Array:
    """Compute initial per-edge theta from edge directions."""
    src = edge_pairs[:, 0]
    tgt = edge_pairs[:, 1]
    deltas = displacement(size, positions[tgt], positions[src])
    edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])  # (E,)
    return edge_theta


def make_initial_bending_rest_angles(num_nodes: int, bending_pairs: jax.Array, edge_pairs: jax.Array) -> jax.Array:
    """Generate initial bending rest angles: sine wave pattern along bending pairs."""
    num_bp = bending_pairs.shape[0]
    # Use the outgoing edge's source node index for sine wave pattern
    e_out = bending_pairs[:, 1]  # (B,)
    src_of_out = edge_pairs[e_out, 0]  # (B,) node indices
    t = src_of_out / max(num_nodes - 1, 1)
    rest_angles = 0.15 * jnp.sin(t * 16 * jnp.pi)  # 8 full waves
    return rest_angles


def main(args: argparse.Namespace | None = None) -> None:
    shape_vals = [int(s) for s in args.shape.split(",")]

    if len(shape_vals) != 2:
        raise ValueError("--shape must contain exactly two comma-separated numbers")

    shape = jnp.array(shape_vals)
    key = jax.random.PRNGKey(args.seed)

    edge_pairs, edge_departure_angles = make_graph(
        args.num_nodes, args.graph_type, args.num_instances
    )
    initial_positions = initialize_positions(args.num_nodes, args.graph_type, shape, key, args.num_instances)
    size = tuple(shape.tolist())

    # Compute initial per-edge theta from positions
    initial_edge_theta = compute_initial_edge_theta(initial_positions, edge_pairs, size)

    # Compute bending pairs from edge topology
    bending_pairs, bending_rest_angles = compute_bending_pairs(edge_pairs, edge_departure_angles, args.num_nodes)

    # Add sine wave pattern on top of structural rest angles (preserves junction angles)
    bending_rest_angles = bending_rest_angles + make_initial_bending_rest_angles(args.num_nodes, bending_pairs, edge_pairs)

    # Edge rest lengths
    edge_rest_lengths = jnp.full(edge_pairs.shape[0], 2.0)

    initial_velocities = jnp.zeros_like(initial_positions)
    nodes = Nodes(
        position=initial_positions,
        velocity=initial_velocities,
        color_bend=jnp.arange(args.num_nodes) % 3,
        debug_vector=jnp.zeros_like(initial_positions),
    )
    sim_edges = Edges(
        pairs=edge_pairs,
        theta=initial_edge_theta,
        rest_lengths=edge_rest_lengths,
        bending_pairs=bending_pairs,
        bending_rest_angles=bending_rest_angles,
        bending_stiffness=jnp.ones(bending_pairs.shape[0]) * 0.5,
    )

    # Initialize fields with proper shapes
    h, w = size

    # Fluid grid shape (may differ from display grid_shape)
    if args.fluid_shape is not None:
        fluid_shape_vals = [int(s) for s in args.fluid_shape.split(",")]
        if len(fluid_shape_vals) != 2:
            raise ValueError("--fluid-shape must contain exactly two comma-separated numbers")
        fh, fw = fluid_shape_vals
    else:
        fh, fw = h, w

    # Initialize LBM distribution at fluid resolution
    initial_fluid_velocity = generate_fluid_default(fh, fw)
    initial_fluid_velocity = 0.1 * jnp.tanh(initial_fluid_velocity / 10.0)

    f_grid = initialize_f_from_velocity(initial_fluid_velocity)

    fields = Fields(
        grid_shape=(h, w),
        steric=jnp.zeros((h, w)),
        fluid_velocity=initial_fluid_velocity,
        f_grid=f_grid,
        energy=None,
    )

    print("Running simulation...")
    with jax_timer("Simulation"):
        nodes_timeseries, fields_timeseries = simulate(
            nodes=nodes,
            edges=sim_edges,
            fields=fields,
            num_steps=args.num_steps,
            dt=args.dt,
            solver_config=PBD_SCHEME,
        )
        jax.block_until_ready(nodes_timeseries)
    print("Simulation complete!")

    now = datetime.now().strftime("%m%d_%H%M%S")

    print("\nStarting visualization...")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    animate(
        nodes_timeseries,
        fields_timeseries,
        filename=f"{output_dir}/{now}_nodes={args.num_nodes}_type={args.graph_type}_instances={args.num_instances}.mp4",
        subsample=args.subsample,
        animate_fluid_velocity=True,
        animate_energy=False,
    )

def generate_fluid_vortex(h, w):
    x = jnp.linspace(0, 1, w)
    y = jnp.linspace(0, 1, h)
    X, Y = jnp.meshgrid(x, y)
    cx, cy = 0.5, 0.5  # Vortex center
    dx = X - cx
    dy = Y - cy
    r = jnp.sqrt(dx**2 + dy**2)
    # Tangential velocity (vortex)
    vortex_strength = 1000.0
    vx = -dy * jnp.exp(-r**2 / 0.05) * vortex_strength
    vy = dx * jnp.exp(-r**2 / 0.05) * vortex_strength
    initial_fluid_velocity = jnp.stack([vx, vy], axis=0)
    return initial_fluid_velocity

def generate_fluid_default(h, w):
    return jnp.zeros((2, h, w))


def initialize_f_from_velocity(velocity_field: jax.Array, density: float = 1.0) -> jax.Array:
    """
    Creates the initial LBM distribution (f_grid) from a velocity field.
    velocity_field: (2, H, W)
    density: scalar (default 1.0)
    """
    u = jnp.moveaxis(velocity_field, 0, -1)

    u_dot_v = jnp.einsum('xyc, ic -> ixy', u, LBM_VELOCITIES)
    u_sq = jnp.sum(u**2, axis=-1)

    feq = density * LBM_WEIGHTS[:, None, None] * (
        1.0 + 3.0*u_dot_v + 4.5*(u_dot_v**2) - 1.5*u_sq[None, :, :]
    )

    return feq

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microcosmos particle simulation")
    parser.add_argument("--seed", type=int, default=1, help="PRNG seed")
    parser.add_argument("--num-nodes", type=int, default=80, help="Number of nodes")
    parser.add_argument(
        "--num_steps", type=int, default=400, help="Number of simulation steps"
    )
    parser.add_argument(
        "--dt", type=float, default=0.01, help="Time step size (integration)"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default="128,128",
        help="Domain shape as 'width,height' (comma separated)",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=2,
        help="Subsample factor for animation frames",
    )
    parser.add_argument(
        "--graph_type",
        choices=['tree', 'line', 'loop', 'tadpole'],
        type=str,
        default="loop",
        help="choose for graph topology",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=4,
        help="number of independent instances of the graph",
    )
    parser.add_argument(
        "--fluid-shape",
        type=str,
        default=None,
        help="LBM fluid grid resolution as 'height,width' (defaults to --shape). "
             "Set higher for finer fluid, lower for faster simulation.",
    )

    main(parser.parse_args())
