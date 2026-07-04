import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import replace
from functools import partial
from collections import namedtuple
from microcosmos.rendering import animate, animate_realtime
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.env import MicrocosmosEnv

import tensorneat as neat

from ..base_experiment import Experiment
from .utils.neat_utils import *

EvalCarry = namedtuple("EvalCarry", ["nodes", "edges", "fields", "prev_center", "fitness"])

class LocomotionProblem(neat.problem.BaseProblem):
    """ Problem for the CPPN NEAT algorihtm to solve. """

    jitable = True # necessary

    @property
    def input_shape(self):
        return (self.num_inputs, )

    @property
    def output_shape(self):
        return (self.num_outputs, )

    def __init__(self, experiment):
        super().__init__()

        # Get data from the experiment
        self.experiment = experiment
        self.cfg = experiment.cfg
        self.nodes = experiment.nodes
        self.edges = experiment.edges
        self.fields = experiment.fields
        self.env = experiment.env
        self.genome = experiment.genome
        self.num_inputs = experiment.num_inputs
        self.num_outputs = experiment.num_outputs
        self.node_coords = experiment.node_coords
        self.rotation_speed = experiment.rotation_speed
        self.output_dir = experiment.output_dir

    def evaluate(self, neat_state, randkey, neat_forward, neat_params, verify_mode=False):
        """ Evaluate the fitness of the current genome by running a simulation and measuring fitness. """

        batch_forward = jax.jit(jax.vmap(neat_forward, in_axes=(None, None, 0)))

        rotation_speed = self.rotation_speed
        env = self.env

        prev_center = jnp.mean(self.nodes.position, axis=0)
        fitness = 0.0

        pre_steps = self.cfg.neat.eval_pre_steps
        num_steps = self.cfg.neat.eval_num_steps
        random_energy_steps = self.cfg.neat.random_energy_steps

        num_rand = num_steps // random_energy_steps
        randkeys = jax.random.split(randkey, num_rand)
        def scan_step(carry, current_step, is_record=False):
            nonlocal batch_forward, rotation_speed, randkey

            (nodes, edges, fields, prev_center, fitness) = carry
            current_phase = current_step * rotation_speed

            # Randomize energy field at specified intervals to encourage robustness
            rand_div, rand_mod = jnp.divmod(current_step - pre_steps, random_energy_steps)
            fields = jax.lax.cond(
                self.cfg.energy.use & (rand_mod == 0) & (rand_div >= 0),
                lambda _: fields.__replace__(energy=random_energy_field(self.cfg, randkeys[(current_step - pre_steps) // random_energy_steps])),
                lambda _: fields,
                operand=None
            )

            # Update curvatures and distances based on current phase
            cppn_inputs = get_cppn_inputs(self.cfg, self.node_coords, current_phase, nodes, fields)
            cppn_assets = (batch_forward, neat_state, neat_params)
            cppn_outputs = cppn_inference(self.cfg, cppn_inputs, *cppn_assets)
            (new_curvatures, new_distances) = cppn_outputs

            num_bending = edges.bending_rest_angles.shape[0]
            edges = edges.__replace__(bending_rest_angles=new_curvatures[:num_bending])
            if new_distances is not None:
                num_edges = edges.rest_lengths.shape[0]
                edges = edges.__replace__(rest_lengths=new_distances[:num_edges])

            nodes, edges, fields = env.step(nodes, edges, fields)

            # Compute fitness
            fitness, prev_center = compute_fitness(self.cfg, nodes, fields, fitness, prev_center)

            # Reset fitness after pre-steps to avoid biasing fitness with the initial transient
            fitness = jax.lax.cond(
                current_step == pre_steps,
                lambda _: 0.0,
                lambda _: fitness,
                operand=None
            )

            carry = EvalCarry(nodes, edges, fields, prev_center, fitness)
            if is_record:
                return carry, (nodes, fields)
            else:
                return carry, None

        # Run pre-run steps and evaluation steps
        carry = EvalCarry(self.nodes, self.edges, self.fields, prev_center, fitness)
        step_seq = jnp.arange(pre_steps + num_steps)

        if verify_mode:
            scan_step_with_record = partial(scan_step, is_record=True)
            carry, records = jax.lax.scan(scan_step_with_record, carry, step_seq)
            return records
        else:
            scan_step_without_record = partial(scan_step, is_record=False)
            carry, _ = jax.lax.scan(scan_step_without_record, carry, step_seq)
            return carry.fitness

    def show(self, neat_state, randkey, neat_forward, neat_params, *args, **kwargs):
        """ Show the best genome in action. """

        batch_forward = jax.jit(jax.vmap(neat_forward, in_axes=(None, None, 0)))

        _, neurons, conns, _ = neat_params
        print("Best genome structure expressed in Python code:")
        print_genome(self.genome, neat_state, neurons, conns, network_path=str(self.output_dir / "network.svg"))
        print(f"Network diagram saved to {self.output_dir / 'network.svg'}")

        print("Running batch simulation with best genome...")
        self.run_batch(batch_forward, neat_state, neat_params)

        print("Running simulation with best genome...")
        self.run_realtime(batch_forward, neat_state, neat_params)

    def run_realtime(self, batch_forward, neat_state, neat_params):
        env = self.env
        current_phase = 0.0
        edges = self.edges
        current_step = 0

        prev_center = jnp.mean(self.nodes.position, axis=0)
        fitness = 0.0

        def step_func(nodes, fields, dt, solver_config):
            nonlocal current_phase, edges, current_step

            cppn_inputs = get_cppn_inputs(self.cfg, self.node_coords, current_phase, nodes, fields)
            cppn_assets = (batch_forward, neat_state, neat_params)
            cppn_outputs = cppn_inference(self.cfg, cppn_inputs, *cppn_assets)
            (new_curvatures, new_distances) = cppn_outputs

            num_bending = edges.bending_rest_angles.shape[0]
            edges = edges.__replace__(bending_rest_angles=new_curvatures[:num_bending])
            if new_distances is not None:
                num_edges = edges.rest_lengths.shape[0]
                edges = edges.__replace__(rest_lengths=new_distances[:num_edges])

            nodes, edges, fields = env.step(nodes, edges, fields)
            current_phase += self.rotation_speed
            current_step += 1

            return nodes, fields

        def extra_info_func(nodes, fields):
            nonlocal current_step, prev_center, fitness

            fitness, prev_center = compute_fitness(self.cfg, nodes, fields, fitness, prev_center)

            return [f"Step: {current_step}", f"Fitness: {fitness:.4f}"]

        def random_energy_action(nodes, fields):
            randkey = jax.random.PRNGKey(true_random_seed())
            fields = fields.__replace__(energy=random_energy_field(self.cfg, randkey))
            return nodes, fields

        animate_realtime(
            self.nodes,
            self.fields,
            env.dt,
            env.solver_config,
            step_func,
            extra_info_func,
            extra_key_actions=[([ord('r')], random_energy_action)],
            subtitle=self.cfg.experiment.type,
            animate_energy=self.cfg.energy.use,
        )


    def run_batch(self, batch_forward, neat_state, neat_params):
        """ Run the best genome in batch mode to collect data for animation and fitness plotting.  """

        nodes = self.nodes
        edges = self.edges
        fields = self.fields
        env = self.env

        num_steps = self.cfg.simulation.num_steps

        prev_center = jnp.mean(self.nodes.position, axis=0)
        fitness = 0.0

        def scan_step(carry, i):
            nodes, edges, fields, prev_center, fitness = carry
            current_phase = i * self.rotation_speed

            cppn_inputs = get_cppn_inputs(self.cfg, self.node_coords, current_phase, nodes, fields)
            cppn_assets = (batch_forward, neat_state, neat_params)
            cppn_outputs = cppn_inference(self.cfg, cppn_inputs, *cppn_assets)
            (new_curvatures, new_distances) = cppn_outputs

            num_bending = edges.bending_rest_angles.shape[0]
            edges = edges.__replace__(bending_rest_angles=new_curvatures[:num_bending])
            if new_distances is not None:
                num_edges = edges.rest_lengths.shape[0]
                edges = edges.__replace__(rest_lengths=new_distances[:num_edges])

            nodes, edges, fields = env.step(nodes, edges, fields)

            fitness, prev_center = compute_fitness(self.cfg, nodes, fields, fitness, prev_center)

            carry = (nodes, edges, fields, prev_center, fitness)
            record = (nodes, fields, fitness)
            return carry, record

        carry_init = (nodes, edges, fields, prev_center, fitness)
        _, records = jax.lax.scan(scan_step, carry_init, jnp.arange(num_steps))

        print("Simulation complete. Saving results...")
        self.save_results(records)

    def save_results(self, records):
        """Save animation and fitness plot."""
        nodes_trajectory, fields_trajectory, fitness_history = records
        plt.figure()
        plt.plot(jnp.array(fitness_history))
        plt.xlabel('Simulation Step')
        plt.ylabel('Fitness')
        plt.title('Locomotion CPPN')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.savefig(self.output_dir / 'fitness.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Fitness plot saved to {self.output_dir / 'fitness.png'}")

        animate(
            nodes_trajectory,
            fields_trajectory,
            filename=str(self.output_dir / "locomotion_cppn.gif"),
            subsample=self.cfg.simulation.subsample,
            uniform_color=False,
            verbose=True,
            fps=self.cfg.simulation.fps,
            animate_energy=self.cfg.energy.use,
        )
        print(f"Animation saved to {self.output_dir / 'locomotion_cppn.gif'}")

    def save_verify(self, records):
        """Save the recorded trajectory from verify mode as an animation."""
        nodes_trajectory, fields_trajectory = records

        animate(
            nodes_trajectory,
            fields_trajectory,
            filename=str(self.output_dir / "locomotion_cppn_verify.gif"),
            subsample=self.cfg.simulation.subsample,
            uniform_color=False,
            verbose=True,
            fps=self.cfg.simulation.fps,
            animate_energy=self.cfg.energy.use,
        )
        print(f"Verification animation saved to {self.output_dir / 'locomotion_cppn_verify.gif'}")


class LocomotionCPPNExperiment(Experiment):
    """Evolve a CPPN to control filament locomotion using NEAT."""

    def setup(self):
        self.num_nodes = self.cfg.experiment.num_nodes
        self.num_wavelengths = self.cfg.experiment.get('num_wavelengths', 3)
        self.rotation_speed = self.cfg.experiment.get('rotation_speed', 0.05)
        self.grid_shape = tuple(self.cfg.experiment.grid_shape)
        self.init_line_distance = self.cfg.experiment.get('init_line_distance', 2.0)

        raw_seed = self.cfg.neat.get("random_seed", None)
        if raw_seed is None:
            raw_seed = true_random_seed()
            print(f"[LocomotionCPPN] random_seed is null — using true_random_seed: {raw_seed}")
        self.random_seed = int(raw_seed)

        phys = self.cfg.get("physics", {})
        self.pbd_scheme = replace(
            PBD_SCHEME,
            max_fluid_velocity=float(phys.get("max_fluid_velocity", PBD_SCHEME.max_fluid_velocity)),
            synthetic_node_width=float(phys.get("synthetic_node_width", PBD_SCHEME.synthetic_node_width)),
            ibm_kernel_size=int(phys.get("ibm_kernel_size", PBD_SCHEME.ibm_kernel_size)),
        )

        self.env = MicrocosmosEnv(
            grid_shape=self.grid_shape,
            dt=self.cfg.simulation.dt,
            num_steps=self.cfg.simulation.num_steps,
            solver_config=self.pbd_scheme,
        )

        self.nodes, self.edges = self.env.make_chain(self.num_nodes, spacing=self.init_line_distance)

        # Override to locomotion layout: 10–90% width, centered height
        h, w = self.grid_shape
        x_positions = jnp.linspace(w * 0.1, w * 0.9, self.num_nodes)
        self.nodes = self.nodes.__replace__(
            position=jnp.stack([x_positions, jnp.full(self.num_nodes, h * 0.5)], axis=1)
        )

        k = 2 * jnp.pi * self.num_wavelengths
        self.node_coords = jnp.linspace(0.0, 1.0, self.num_nodes) * k

        # Add energy field on top of env's base fields
        randkey = jax.random.PRNGKey(self.random_seed)
        self.fields = self.env.fields.__replace__(energy=random_energy_field(self.cfg, randkey))

    def loss_fn(self, _params):
        raise NotImplementedError("This experiment does not use optimization")

    def run(self):
        if self.cfg.neat.verify_eval:
            self.run_verify_eval()
        else:
            self.run_cppn()

    def run_verify_eval(self):
        """Run a single evaluation of the current genome with recording for verification purposes."""

        randkey = jax.random.PRNGKey(self.random_seed)

        # Define the genome
        self.num_inputs, self.num_outputs = get_cppn_io_num(self.cfg)
        self.genome = init_cppn_genome(num_inputs=self.num_inputs, num_outputs=self.num_outputs)

        # Get dummy assets for verification
        neat_state = self.genome.setup()
        neat_forward = self.genome.forward
        neurons, conns = self.genome.initialize(neat_state, randkey)
        neat_params = self.genome.transform(neat_state, neurons, conns)

        # Run evaluation with recording and save animation
        problem = LocomotionProblem(self)
        print("Running evaluation in verify mode...")
        records = problem.evaluate(neat_state, randkey, neat_forward, neat_params, verify_mode=True)
        print("Saving animation...")
        problem.save_verify(records)

    def run_cppn(self):
        """Run the NEAT algorithm to evolve a CPPN that controls the movement of the nodes."""

        # Define the genome
        self.num_inputs, self.num_outputs = get_cppn_io_num(self.cfg)
        self.genome = init_cppn_genome(num_inputs=self.num_inputs, num_outputs=self.num_outputs)

        neat_algo = neat.algorithm.NEAT(
            pop_size=self.cfg.neat.pop_size,
            species_size=self.cfg.neat.species_size,
            genome=self.genome,
        )

        if self.cfg.neat.use_hyperneat:
            hyperneat_algo = neat.algorithm.HyperNEAT(
                neat=neat_algo,
                substrate=neat.algorithm.hyperneat.FullSubstrate(
                    input_coors=((-1, -1), (0, -1), (1, -1)),
                    hidden_coors=((-1, 0), (0, 0), (1, 0)),
                    output_coors=((0, 1),),
                ),
            )

        pipeline = neat.pipeline.Pipeline(
            algorithm=hyperneat_algo if self.cfg.neat.use_hyperneat else neat_algo,
            problem=LocomotionProblem(self),
            generation_limit=self.cfg.neat.generation_limit,
            fitness_target=self.cfg.neat.fitness_target,
            seed=self.random_seed,
            is_save=True,
            save_dir=self.output_dir / "neat",
        )

        state_path = self.cfg.neat.load_state

        neat_state = None
        if self.cfg.neat.cont:
            try:
                neat_state = neat.common.State.load(state_path)
                print(f"Loaded NEAT state from {state_path}")
            except FileNotFoundError:
                pass

        if neat_state is None:
            neat_state = pipeline.setup()

        try:
            neat_state, best_genome = pipeline.auto_run(neat_state)
        except KeyboardInterrupt:
            print("\nInterrupted!")
            print()
            if pipeline.best_genome is not None:
                best_genome = jax.device_get(pipeline.best_genome)

        final_state_path = self.output_dir / "neat" / "final_state.pkl"
        neat_state.save(final_state_path)
        print(f"Progress saved to {final_state_path}")
        neat_state.save(state_path)
        print(f"Progress saved to {state_path}")
        print()

        latest_genome_path = max((self.output_dir / "neat" / "genomes").glob("*.npz"), key=lambda p: p.name)
        latest_genome_copy_path = self.output_dir.parent.parent / "latest_genome.npz"
        latest_genome_copy_path.write_bytes(latest_genome_path.read_bytes())
        print(f"Latest best genome copied to {latest_genome_copy_path}")

        import shutil
        hydra_cfg = self.output_dir / ".hydra" / "config.yaml"
        latest_genome_cfg = latest_genome_copy_path.with_name("latest_genome_config.yaml")
        if hydra_cfg.exists():
            shutil.copy(hydra_cfg, latest_genome_cfg)
            print(f"Latest genome config copied to {latest_genome_cfg}")

        pipeline.show(neat_state, best_genome)
