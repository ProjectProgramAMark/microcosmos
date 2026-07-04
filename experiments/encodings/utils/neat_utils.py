import jax
import jax.numpy as jnp
import sympy as sp

import tensorneat as neat
from tensorneat.common import ACT, AGG, sympy_tools

import time
from microcosmos.utils import index_from_position, displacement as pbc_displacement


def true_random_seed():
    """Random seed from system time + process index."""
    return int(time.time() * 1000) % (2**32) + jax.process_index()

def print_genome(genome, state, neurons, conns, network_path=None):
    try:
        network = genome.network_dict(state, neurons, conns)
        if network_path is not None:
            genome.visualize(network, save_path=network_path)

        sympy_res = genome.sympy_func(
            state, network, sympy_output_transform=ACT.obtain_sympy(genome.output_transform)
        )
        # print(sympy_tools.to_latex_code(*sympy_res))
        print(sympy_tools.to_python_code(*sympy_res))
    except Exception as e:
        print(f"Error printing genome: {e}")


def gaussian(z):
    return jnp.exp(-z**2)

class SympyGaussian(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.exp(-z**2)


def square(z):
    return jnp.sign(jnp.sin(2 * jnp.pi * z))

class SympySquare(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.sign(sp.sin(2 * sp.pi * z))


def sawtooth(z):
    return 2 / jnp.pi * jnp.arctan(1 / jnp.tan(jnp.pi * z))

class SympySawtooth(sp.Function):
    @classmethod
    def eval(cls, z):
        return 2 / sp.pi * sp.atan(1 / sp.tan(sp.pi * z))


def triangle(z):
    return 2 / jnp.pi * jnp.arcsin(jnp.sin(2 * jnp.pi * z))

class SympyTriangle(sp.Function):
    @classmethod
    def eval(cls, z):
        return 2 / sp.pi * sp.asin(sp.sin(2 * sp.pi * z))


def smooth_square(z):
    return jnp.tanh(5.0 * jnp.sin(2 * z))

class SympySmoothSquare(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.tanh(5.0 * sp.sin(2 * z))


def smooth_sawtooth(z):
    return jnp.arcsin(jnp.tanh(5.0 * jnp.sin(z)) * jnp.cos(z))

class SympySmoothSawtooth(sp.Function):
    @classmethod
    def eval(cls, z):
        return sp.asin(sp.tanh(5.0 * sp.sin(z)) * sp.cos(z))


def new_activations(ACT: neat.common.functions.FunctionManager):
    # See https://www.desmos.com/calculator/iw0g3t7gwf for the shapes of these functions

    ACT.add_func("gaussian", gaussian)
    ACT.add_func("square", square)
    ACT.add_func("sawtooth", sawtooth)
    ACT.add_func("triangle", triangle)
    ACT.add_func("smooth_square", smooth_square)
    ACT.add_func("smooth_sawtooth", smooth_sawtooth)

    ACT.update_sympy("gaussian", SympyGaussian)
    ACT.update_sympy("square", SympySquare)
    ACT.update_sympy("sawtooth", SympySawtooth)
    ACT.update_sympy("triangle", SympyTriangle)
    ACT.update_sympy("smooth_square", SympySmoothSquare)
    ACT.update_sympy("smooth_sawtooth", SympySmoothSawtooth)


def init_cppn_genome(num_inputs=2, num_outputs=1):
    # Add new activation functions e.g. sawtooth in the ACT manager
    new_activations(ACT)

    genome = neat.genome.DefaultGenome(
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        max_nodes=15,  # 10
        max_conns=30,  # 30
        init_hidden_layers=(3,),
        node_gene=neat.genome.DefaultNode(  # neuron_gene
            bias_init_mean=0.0,
            response_init_mean=1.0,
            # aggregation_options=[AGG.sum],
            aggregation_options=[AGG.sum, AGG.product, AGG.max, AGG.min],
            aggregation_default=AGG.sum,
            # activation_options=[ACT.identity, ACT.sin],
            # activation_options=[ACT.identity, ACT.sin, ACT.exp, ACT.sigmoid, ACT.tanh],
            activation_options=[
                ACT.identity, ACT.abs, ACT.exp, ACT.log, ACT.inv, ACT.tanh, ACT.relu, ACT.lelu, ACT.gaussian,  # non-periodic functions
                ACT.sin, ACT.square, ACT.sawtooth, ACT.triangle, ACT.smooth_square, ACT.smooth_sawtooth  # periodic functions
            ],
            activation_default=ACT.identity,
        ),
        conn_gene=neat.genome.DefaultConn(
            weight_init_mean=1.0,
        ),
        output_transform=ACT.identity,
    )
    return genome


def compute_actual_edge_metrics(nodes, edges, shape):
    """Compute actual edge lengths and bending angles from current simulation state.

    Call after a completed simulation step to get the physically realized values,
    as opposed to the commanded rest_lengths / bending_rest_angles.
    """
    src = edges.pairs[:, 0]
    tgt = edges.pairs[:, 1]
    current_vecs = pbc_displacement(shape, nodes.position[tgt], nodes.position[src])
    actual_lengths = jnp.linalg.norm(current_vecs, axis=-1)

    bp_e_in  = edges.bending_pairs[:, 0]
    bp_e_out = edges.bending_pairs[:, 1]
    diff = edges.theta[bp_e_out] - edges.theta[bp_e_in]
    actual_angles = (diff + jnp.pi) % (2 * jnp.pi) - jnp.pi

    return actual_angles, actual_lengths


def cppn_inference(cfg, cppn_inputs, batch_forward, neat_state, neat_params):
    """Compute curvatures and distances using the CPPN."""
    # CPPN forward pass to get curvatures and distances for all nodes in a batch
    outputs = batch_forward(neat_state, neat_params, cppn_inputs)

    if cfg.neat.evolve_distances:
        # shape (num_nodes, 2) with curvature and distance
        curvatures, distances = outputs[..., 0].flatten(), outputs[..., 1].flatten()
    else:
        # shape (num_nodes, 1) with just curvature
        curvatures = outputs.flatten()
        distances = None

    # Clip curvatures and distances to prevent extreme values that could destabilize the simulation
    cmax = cfg.neat.clip.max_curvature
    dmax = cfg.neat.clip.max_line_distance
    dmin = cfg.neat.clip.min_line_distance

    if cfg.neat.clip.method == "clip":
        curvatures = jnp.clip(curvatures, -cmax, cmax)
    elif cfg.neat.clip.method == "tanh":
        curvatures = cmax * jnp.tanh(curvatures / cmax)

    if cfg.neat.evolve_distances:
        if cfg.neat.clip.method == "clip":
            distances = jnp.clip(distances, dmin, dmax)
        elif cfg.neat.clip.method == "tanh":
            distances = dmin + (dmax - dmin) * jnp.tanh((distances - dmin) / (dmax - dmin))

    return curvatures, distances


def compute_fitness(cfg, nodes, fields, fitness, prev_center):
    # Total displacement by considering periodic boundaries
    domain_size = jnp.array([cfg.experiment.grid_shape[1], cfg.experiment.grid_shape[0]])
    recentered_pos = (nodes.position - prev_center + domain_size / 2) % domain_size - domain_size / 2
    movement = jnp.mean(recentered_pos, axis=0)
    prev_center += movement
    displacement = jnp.linalg.norm(movement)

    # Total energy collected by the nodes
    y, x = index_from_position(nodes.position)
    energy_at_nodes = fields.energy.at[y, x].get()
    collected_energy = jnp.mean(energy_at_nodes)

    # Combine weighted components into fitness score
    fitness += cfg.neat.fitness_weight_displacement * displacement
    fitness += cfg.neat.fitness_weight_energy * collected_energy

    return fitness, prev_center


def random_energy_field(cfg, randkey=None):
    """Generate a random energy field."""
    # See https://www.desmos.com/calculator/tm04afzmfh for the shapes of these functions

    h, w = cfg.experiment.grid_shape

    if randkey is None:
        randkey = jax.random.PRNGKey(true_random_seed())

    if cfg.energy.init.type == "none":
        energy_field = jnp.zeros((h, w))

    else:
        randkey, *randkeys = jax.random.split(randkey, 3)
        size = cfg.energy.init.size
        amplitude = cfg.energy.init.amplitude

        cx = jax.random.randint(randkeys[0], (), size, w - size)
        cy = jax.random.randint(randkeys[1], (), size, h - size)
        X, Y = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
        distance_sq = (X - cx)**2 + (Y - cy)**2
        distance = jnp.sqrt(distance_sq)

        if cfg.energy.init.type == "gaussian":
            # Initialize energy field with a Gaussian packet at random position
            energy_field = amplitude * jnp.exp(-distance_sq / (2 * (size**2)))
        elif cfg.energy.init.type == "tent":
            # Initialize energy field with a tent-shaped packet at random position
            energy_field = amplitude * jnp.maximum(1 - distance / (4 * size), 0) ** 4

    return energy_field


def get_cppn_io_num(cfg):
    sense = cfg.energy.sense if cfg.energy.use else None

    if sense == "each":
        num_inputs = 3  # Inputs: node_coord, phase, energy_each
    elif sense == "head":
        num_inputs = 3  # Inputs: node_coord, phase, energy_head
    elif sense == "head-tail":
        num_inputs = 4  # Inputs: node_coord, phase, energy_head, energy_tail
    else:
        num_inputs = 2  # Inputs: node_coord, phase

    if cfg.neat.evolve_distances:
        num_outputs = 2  # Output both curvature and distance
    else:
        num_outputs = 1  # Output only curvature

    return num_inputs, num_outputs


def get_cppn_inputs(cfg, node_coords, phase, nodes, fields):
    phase_arr = jnp.full_like(node_coords, phase)

    if cfg.energy.use:
        if cfg.energy.sense == "each":
            # All nodes sense the energy field and feed into CPPN

            y, x = index_from_position(nodes.position)
            energy_each = fields.energy.at[y, x].get()
            cppn_inputs = jnp.stack([node_coords, phase_arr, energy_each], axis=1)

        elif cfg.energy.sense == "head":
            # Only the first node senses the energy field and feeds into CPPN

            y, x = index_from_position(nodes.position[0:1])  # shape (1,)
            energy_head = fields.energy.at[y, x].get()[0]  # scalar
            energy_head_arr = jnp.full_like(node_coords, energy_head)

            cppn_inputs = jnp.stack([node_coords, phase_arr, energy_head_arr], axis=1)

        elif cfg.energy.sense == "head-tail":
            # The first and last node sense the energy field and feed into CPPN

            y, x = index_from_position(nodes.position[0:1])
            energy_head = fields.energy.at[y, x].get()[0]
            energy_head_arr = jnp.full_like(node_coords, energy_head)

            y, x = index_from_position(nodes.position[-1:])
            energy_tail = fields.energy.at[y, x].get()[0]
            energy_tail_arr = jnp.full_like(node_coords, energy_tail)

            cppn_inputs = jnp.stack([node_coords, phase_arr, energy_head_arr, energy_tail_arr], axis=1)

    else:
        cppn_inputs = jnp.stack([node_coords, phase_arr], axis=1)

    return cppn_inputs
