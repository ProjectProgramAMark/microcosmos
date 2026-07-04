import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from microcosmos.simulate import step
from microcosmos.structs.fields import Fields
from microcosmos.rendering import animate, animate_realtime, render_fields
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.graph import make_edges, compute_bending_pairs
from microcosmos.structs.nodes import Nodes
from microcosmos.structs.edges import Edges
from microcosmos.utils import displacement
import tensorneat as neat
from tensorneat.common import ACT, AGG

from ...base_experiment import Experiment
from .neat_utils import *

class NeuralControlExperiment(Experiment):
    """Use a CPPN (Compositional Pattern-Producing Network) to control movements, e.g. sine wave."""

    def init_sine_wave_genome(self, max_curv=1.0, num_inputs=2, num_outputs=1):
        # Create genome as sine wave locomotion generator
        genome = neat.genome.DefaultGenome(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            max_nodes=10,
            max_conns=10,
            node_gene=neat.genome.DefaultNode(  # neuron_gene
                aggregation_options=[AGG.sum],
                activation_options=[ACT.identity, ACT.sin],
            ),
            conn_gene=neat.genome.DefaultConn(),
            output_transform=ACT.identity,
        )
        state = genome.setup()

        # Nodes: [index, bias, response, aggregation, activation]
        neuron_spec = []
        # inputs
        inputs_n = len(neuron_spec)
        for _ in range(num_inputs):
            neuron_spec.append([0.0, 1.0, 0, 0])  # input: node_coord, phase, (energy_1, energy_2)
        # outputs
        outputs_n = len(neuron_spec)
        for _ in range(num_outputs):
            neuron_spec.append([0.0, 1.0, 0, 0])  # output: curvature, (distance)
        # hidden
        hidden_n = len(neuron_spec)
        neuron_spec.append([0.0, 1.0, 0, 1])  # hidden: sine

        neurons = jnp.full((genome.max_nodes, 5), jnp.nan)
        for i, spec in enumerate(neuron_spec):
            neurons = neurons.at[i].set([i] + spec)

        # Connections: [in, out, weight]
        conn_spec = []
        conn_spec.append([inputs_n, hidden_n, 1.0])  # node_coord -> sine
        conn_spec.append([inputs_n+1, hidden_n, 1.0])  # phase -> sine
        conn_spec.append([hidden_n, outputs_n, float(max_curv)])  # sine -> curvature
        if num_outputs == 2:
            conn_spec.append([hidden_n, outputs_n+1, 3.0])  # sine -> distance
        if num_inputs >= 3:
            conn_spec.append([inputs_n+2, outputs_n, 0.1])  # energy_head|_each -> curvature
        if num_inputs >= 4:
            conn_spec.append([inputs_n+3, outputs_n, -0.1])  # energy_tail -> curvature

        conns = jnp.full((genome.max_conns, 3), jnp.nan)
        for i, spec in enumerate(conn_spec):
            conns = conns.at[i].set(spec)

        return genome, state, neurons, conns

    def _apply_run_config(self, npz_path, label=""):
        """Find and merge a run's saved config into self.cfg (experiment/neat/physics/energy).

        Searches in order:
          1. <npz_stem>_config.yaml alongside the file  (latest_archive / latest_genome copies)
          2. <ancestor>/.hydra/config.yaml walking up from the file's directory
             (handles both qd/archive.npz and locomotion_cppn/.../neat/genomes/gen_N.npz)
        """
        import pathlib
        from omegaconf import OmegaConf

        p = pathlib.Path(npz_path)

        # 1. Sidecar next to the file
        sidecar = p.with_name(p.stem + "_config.yaml")
        candidates = [sidecar]

        # 2. Walk up looking for .hydra/config.yaml
        for ancestor in p.parents:
            candidates.append(ancestor / ".hydra" / "config.yaml")

        cfg_path = next((c for c in candidates if c.exists()), None)
        if cfg_path is None:
            tag = f"[{label}] " if label else ""
            print(f"  [neural] {tag}No config sidecar found — using current config as-is.")
            return

        run_cfg = OmegaConf.load(cfg_path)
        keys_to_apply = ["experiment", "neat", "physics", "energy"]
        applied = []
        for key in keys_to_apply:
            if key in run_cfg:
                OmegaConf.update(self.cfg, key, run_cfg[key], merge=True)
                applied.append(key)
        tag = f"[{label}] " if label else ""
        print(f"  [neural] {tag}Applied config from {cfg_path}: {', '.join(applied)}")

    def load_qd_archive_genome(self, num_inputs=2, num_outputs=1):
        """Load a genome from a QD archive by specifying BD1 and BD2 indices."""
        archive_path = str(self.cfg.qd.load_archive)
        if not archive_path.startswith("outputs/qd/"):
            archive_path = "outputs/qd/" + archive_path
        if not archive_path.endswith(".npz"):
            archive_path = archive_path + ".npz"
        print(f"Loading QD archive from {archive_path}...")

        with np.load(archive_path) as data:
            archive_fitness = data["fitness"]
            archive_neurons = data["neurons"]
            archive_conns   = data["conns"]

        # Cache archive for interactive navigation in realtime mode
        self._archive_fitness = archive_fitness
        self._archive_neurons = archive_neurons
        self._archive_conns   = archive_conns
        self._archive_nb1, self._archive_nb2 = archive_fitness.shape

        bd1 = int(self.cfg.qd.bd1)
        bd2 = int(self.cfg.qd.bd2)

        if not np.isfinite(archive_fitness[bd1, bd2]):
            raise ValueError(
                f"Archive cell ({bd1}, {bd2}) is empty. "
                "Try different BD values or check archive coverage."
            )
        print(f"  Cell fitness: {archive_fitness[bd1, bd2]:.4f}")

        genome = init_cppn_genome(num_inputs=num_inputs, num_outputs=num_outputs)
        state  = genome.setup()
        neurons = archive_neurons[bd1, bd2]
        conns   = archive_conns[bd1, bd2]
        return genome, state, neurons, conns

    def load_cppn_genome(self, num_inputs=2, num_outputs=1):
        genome = init_cppn_genome(num_inputs=num_inputs, num_outputs=num_outputs)
        state = genome.setup()

        genome_path = str(self.cfg.neat.load_genome)
        if not genome_path.startswith("outputs/locomotion_cppn/"):
            genome_path = "outputs/locomotion_cppn/" + genome_path
        if not genome_path.endswith(".npz"):
            genome_path = genome_path + ".npz"
        print(f"Loading CPPN genome from {genome_path}...")
        with np.load(genome_path) as data:
            neurons, conns = data["nodes"], data["conns"]

        return genome, state, neurons, conns

    def setup(self):
        """Initialize a line of nodes and the CPPN."""
        # Apply the originating run's config before reading any values from self.cfg,
        # so that num_nodes, grid_shape, physics, neat.clip, etc. all match.
        if self.cfg.qd.get("load_archive") is not None:
            archive_path = str(self.cfg.qd.load_archive)
            if not archive_path.startswith("outputs/qd/"):
                archive_path = "outputs/qd/" + archive_path
            if not archive_path.endswith(".npz"):
                archive_path = archive_path + ".npz"
            self._apply_run_config(archive_path, label="qd")
        elif self.cfg.neat.load_genome is not None:
            genome_path = str(self.cfg.neat.load_genome)
            if not genome_path.startswith("outputs/locomotion_cppn/"):
                genome_path = "outputs/locomotion_cppn/" + genome_path
            if not genome_path.endswith(".npz"):
                genome_path = genome_path + ".npz"
            self._apply_run_config(genome_path, label="cppn")

        self.num_nodes = self.cfg.experiment.num_nodes
        self.num_wavelengths = self.cfg.experiment.get('num_wavelengths', 3)
        self.rotation_speed = self.cfg.experiment.get('rotation_speed', 0.05)
        self.grid_shape = tuple(self.cfg.experiment.grid_shape)
        self.init_line_distance = self.cfg.experiment.get('init_line_distance', 2.0)

        raw_seed = self.cfg.neat.get("random_seed", None)
        if raw_seed is None:
            raw_seed = true_random_seed()
            print(f"[Neural] random_seed is null — using true_random_seed: {raw_seed}")
        self.random_seed = int(raw_seed)

        # Physics overrides (same pattern as qd.py)
        phys = self.cfg.get("physics", {})
        self.pbd_scheme = replace(
            PBD_SCHEME,
            max_fluid_velocity=float(phys.get("max_fluid_velocity", PBD_SCHEME.max_fluid_velocity)),
            synthetic_node_width=float(phys.get("synthetic_node_width", PBD_SCHEME.synthetic_node_width)),
            ibm_kernel_size=int(phys.get("ibm_kernel_size", PBD_SCHEME.ibm_kernel_size)),
        )

        # Create line topology (open chain)
        edge_pairs, edge_departure_angles = make_edges(self.num_nodes, graph="line", num_instances=1)
        num_edges = self.num_nodes - 1

        # Arrange nodes along a straight line for initial position
        domain_width = self.grid_shape[1]
        domain_height = self.grid_shape[0]

        k = 2 * jnp.pi * self.num_wavelengths
        self.node_coords = jnp.linspace(0.0, 1.0, self.num_nodes) * k
        amplitude = domain_height * 0.4

        x_positions = jnp.linspace(domain_width * 0.1, domain_width * 0.9, self.num_nodes)
        y_positions = jnp.linspace(domain_height * 0.5, domain_height * 0.5, self.num_nodes)
        positions = jnp.stack([x_positions, y_positions], axis=1)

        # Initialize CPPN
        dy = amplitude * jnp.cos(k * self.node_coords)
        dx = x_positions[-1] - x_positions[0]
        theta_ref = jnp.arctan(dy / dx)
        curv_ref = jnp.roll(theta_ref, -1) - theta_ref
        curv_ref = curv_ref.at[-1].set(0.0)
        max_curv = jnp.max(jnp.abs(curv_ref))

        num_inputs, num_outputs = get_cppn_io_num(self.cfg)

        if self.cfg.qd.get("load_archive") is not None:
            self.genome, self.neat_state, neurons, conns = self.load_qd_archive_genome(num_inputs=num_inputs, num_outputs=num_outputs)
        elif self.cfg.neat.load_genome is not None:
            self.genome, self.neat_state, neurons, conns = self.load_cppn_genome(num_inputs=num_inputs, num_outputs=num_outputs)
        else:
            self.genome, self.neat_state, neurons, conns = self.init_sine_wave_genome(max_curv=max_curv, num_inputs=num_inputs, num_outputs=num_outputs)

        self.neat_params = self.genome.transform(self.neat_state, neurons, conns)
        print_genome(self.genome, self.neat_state, neurons, conns, network_path="outputs/neural/network.svg")

        self.initial_phase = 0.0
        self.batch_forward = jax.jit(jax.vmap(self.genome.forward, in_axes=(None, None, 0)))

        init_distances = jnp.full(num_edges, self.init_line_distance)

        # Compute initial per-edge theta from positions
        size = tuple(self.grid_shape)
        src = edge_pairs[:, 0]
        tgt = edge_pairs[:, 1]
        deltas = displacement(size, positions[tgt], positions[src])
        edge_theta = jnp.arctan2(deltas[:, 1], deltas[:, 0])

        bending_pairs, bending_rest_angles_init = compute_bending_pairs(edge_pairs, edge_departure_angles, self.num_nodes)

        self.nodes = Nodes(
            position=positions,
            velocity=jnp.zeros_like(positions),
            color_bend=jnp.arange(self.num_nodes) % 3,
            debug_vector=jnp.zeros((self.num_nodes, 2)),
        )

        self.edges = Edges(
            pairs=edge_pairs,
            theta=edge_theta,
            rest_lengths=init_distances,
            bending_pairs=bending_pairs,
            bending_rest_angles=bending_rest_angles_init,
            bending_stiffness=jnp.ones(bending_pairs.shape[0]) * 0.5,
        )

        # Initialize fields
        h, w = self.grid_shape
        weights = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        f_grid = jnp.zeros((9, h, w))
        f_grid = f_grid.at[:].set(weights[:, None, None])

        # Initialize energy field
        randkey = jax.random.PRNGKey(self.random_seed)
        energy_field = random_energy_field(self.cfg, randkey)

        self.fields = Fields(
            grid_shape=self.grid_shape,
            steric=jnp.zeros(self.grid_shape),
            fluid_velocity=jnp.zeros((2, h, w)),
            f_grid=f_grid,
            energy=energy_field,
        )

        self.fitness_history = []

    def loss_fn(self, _params):
        raise NotImplementedError("This experiment does not use optimization")

    def run(self):
        realtime = self.cfg.simulation.realtime
        if realtime:
            self.run_realtime()
        else:
            self.run_batch()

    def run_realtime(self):
        dt = self.cfg.simulation.dt
        solver_config = self.pbd_scheme
        current_phase = self.initial_phase
        edges = self.edges
        current_step = 0

        prev_center = jnp.mean(self.nodes.position, axis=0)
        fitness = 0.0

        # Archive navigation state (populated only when a QD archive is loaded)
        has_archive = hasattr(self, '_archive_fitness')
        current_bd1 = int(self.cfg.qd.bd1) if has_archive else 0
        current_bd2 = int(self.cfg.qd.bd2) if has_archive else 0

        def step_func(nodes, fields, dt, solver_config):
            nonlocal current_phase, edges, current_step

            cppn_inputs = get_cppn_inputs(self.cfg, self.node_coords, current_phase, nodes, fields)
            cppn_assets = (self.batch_forward, self.neat_state, self.neat_params)
            cppn_outputs = cppn_inference(self.cfg, cppn_inputs, *cppn_assets)
            (new_curvatures, new_distances) = cppn_outputs

            num_bending = edges.bending_rest_angles.shape[0]
            num_edges = edges.rest_lengths.shape[0]
            edges = edges.__replace__(bending_rest_angles=new_curvatures[:num_bending])
            if new_distances is not None:
                edges = edges.__replace__(rest_lengths=new_distances[:num_edges])

            nodes, edges, fields = step(nodes, edges, fields, dt, solver_config)
            current_phase += self.rotation_speed
            current_step += 1

            return nodes, fields

        def extra_info_func(nodes, fields):
            nonlocal current_step, prev_center, fitness

            fitness, prev_center = compute_fitness(self.cfg, nodes, fields, fitness, prev_center)

            info = [f"Step: {current_step}", f"Fitness: {fitness:.4f}"]
            if has_archive:
                fit_val = self._archive_fitness[current_bd1, current_bd2]
                info.append(f"BD [{current_bd1}, {current_bd2}]  archive fit: {fit_val:.3f}")
            return info

        def random_energy_action(nodes, fields):
            randkey = jax.random.PRNGKey(true_random_seed())
            fields = fields.__replace__(energy=random_energy_field(self.cfg, randkey))
            return nodes, fields

        extra_key_actions = [([ord('r')], random_energy_action)]

        if has_archive:
            nb1 = self._archive_nb1
            nb2 = self._archive_nb2

            def _switch_cell(bd1, bd2):
                """Load CPPN params from archive cell (bd1, bd2); return (nodes, fields) or None."""
                nonlocal current_bd1, current_bd2, edges, current_phase, current_step
                if not np.isfinite(self._archive_fitness[bd1, bd2]):
                    return None, None
                neurons = jnp.array(self._archive_neurons[bd1, bd2])
                conns   = jnp.array(self._archive_conns[bd1, bd2])
                self.neat_params = self.genome.transform(self.neat_state, neurons, conns)
                current_bd1, current_bd2 = bd1, bd2
                edges = self.edges      # reset physical state
                current_phase = 0.0
                current_step  = 0
                return self.nodes, self.fields

            def action_bd1_dec(nodes, fields):  # left arrow  → BD1 - 1
                n, f = _switch_cell(max(0, current_bd1 - 1), current_bd2)
                return (n, f) if n is not None else (nodes, fields)

            def action_bd1_inc(nodes, fields):  # right arrow → BD1 + 1
                n, f = _switch_cell(min(nb1 - 1, current_bd1 + 1), current_bd2)
                return (n, f) if n is not None else (nodes, fields)

            def action_bd2_inc(nodes, fields):  # up arrow    → BD2 + 1
                n, f = _switch_cell(current_bd1, min(nb2 - 1, current_bd2 + 1))
                return (n, f) if n is not None else (nodes, fields)

            def action_bd2_dec(nodes, fields):  # down arrow  → BD2 - 1
                n, f = _switch_cell(current_bd1, max(0, current_bd2 - 1))
                return (n, f) if n is not None else (nodes, fields)

            def print_current_genome(nodes, fields):
                nonlocal current_bd1, current_bd2
                neurons = jnp.array(self._archive_neurons[current_bd1, current_bd2])
                conns   = jnp.array(self._archive_conns[current_bd1, current_bd2])
                print_genome(self.genome, self.neat_state, neurons, conns)
                return nodes, fields

            def animate_current_genome(nodes, fields):
                self.run_batch(subgraph=f"cell{current_bd1}-{current_bd2}")
                n, f = _switch_cell(current_bd1, current_bd2)
                return (n, f) if n is not None else (nodes, fields)

            def animate_longer_current_genome(nodes, fields):
                self.run_batch(subgraph=f"cell{current_bd1}-{current_bd2}", time_factor=2)
                n, f = _switch_cell(current_bd1, current_bd2)
                return (n, f) if n is not None else (nodes, fields)

            def save_snapshot(nodes, fields):
                nodes_ts = jax.tree.map(lambda x: jnp.expand_dims(x, 0), nodes)
                fields_ts = jax.tree.map(lambda x: jnp.expand_dims(x, 0), fields)
                frame = render_fields(fields_ts, nodes_ts, sz=512, animate_energy=self.cfg.energy.use, animate_arrows=False)[0]
                path = self.output_dir / f"neural_cell{current_bd1}-{current_bd2}_fluid.png"
                from PIL import Image as PILImage
                PILImage.fromarray(frame).save(str(path))
                print(f"\nSnapshot saved to {path}")
                return nodes, fields

            # OpenCV arrow key codes after (waitKey & 0xFF) on Linux
            extra_key_actions += [
                ([81], action_bd1_dec),   # left  → BD1 - 1
                ([83], action_bd1_inc),   # right → BD1 + 1
                ([82], action_bd2_inc),   # up    → BD2 + 1
                ([84], action_bd2_dec),   # down  → BD2 - 1
                ([ord('p')], print_current_genome),
                ([ord('s')], animate_current_genome),
                ([ord('d')], animate_longer_current_genome),
                ([ord('a')], save_snapshot),
            ]

        animate_realtime(
            self.nodes,
            self.fields,
            dt,
            solver_config,
            step_func,
            extra_info_func,
            extra_key_actions=extra_key_actions,
            subtitle=self.cfg.experiment.type,
            animate_energy=self.cfg.energy.use,
        )


    def run_batch(self, subgraph=None, time_factor=1):
        """Run simulation with CPPN-controlled curvature."""
        nodes = self.nodes
        edges = self.edges
        fields = self.fields

        num_steps = self.cfg.simulation.num_steps * time_factor
        subsample = self.cfg.simulation.subsample * time_factor
        dt = self.cfg.simulation.dt
        solver_config = self.pbd_scheme

        prev_center = jnp.mean(self.nodes.position, axis=0)
        fitness = 0.0

        print(f"Running neural control simulation ({num_steps} steps)...")
        print(f"  Wavelengths: {self.num_wavelengths}")
        print(f"  Rotation speed: {self.rotation_speed} rad/step")

        def scan_step(carry, i):
            nodes, edges, fields, prev_center, fitness = carry
            current_phase = i * self.rotation_speed

            cppn_inputs = get_cppn_inputs(self.cfg, self.node_coords, current_phase, nodes, fields)
            cppn_assets = (self.batch_forward, self.neat_state, self.neat_params)
            cppn_outputs = cppn_inference(self.cfg, cppn_inputs, *cppn_assets)
            (new_curvatures, new_distances) = cppn_outputs

            num_bending = edges.bending_rest_angles.shape[0]
            num_edges = edges.rest_lengths.shape[0]
            edges = edges.__replace__(bending_rest_angles=new_curvatures[:num_bending])
            if new_distances is not None:
                edges = edges.__replace__(rest_lengths=new_distances[:num_edges])

            nodes, edges, fields = step(nodes, edges, fields, dt, solver_config)

            fitness, prev_center = compute_fitness(self.cfg, nodes, fields, fitness, prev_center)

            carry = (nodes, edges, fields, prev_center, fitness)
            record = (nodes, fields, fitness)
            return carry, record

        carry_init = (nodes, edges, fields, prev_center, fitness)
        _, records = jax.lax.scan(scan_step, carry_init, jnp.arange(num_steps))

        print("Simulation complete. Saving results...")
        self.save_results(records, subgraph, subsample)

    def save_results(self, records, subgraph=None, subsample=1):
        """Save animation and fitness plot of the neural control experiment."""
        nodes_trajectory, fields_trajectory, fitness_history = records

        if subgraph is None:
            plt.figure()
            plt.plot(jnp.array(fitness_history))
            plt.xlabel('Simulation Step')
            plt.ylabel('Fitness')
            plt.title('Neural Control Locomotion')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.savefig(self.output_dir / 'fitness.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Fitness plot saved to {self.output_dir / 'fitness.png'}")

        filename = "neural.gif" if subgraph is None else f"neural_{subgraph}.gif"
        animate(
            nodes_trajectory,
            fields_trajectory,
            filename=str(self.output_dir / filename),
            subsample=self.cfg.simulation.subsample,
            uniform_color=False,
            verbose=True,
            fps=self.cfg.simulation.fps,
            animate_filament=False,
            animate_fluid_velocity=True,
            animate_energy=self.cfg.energy.use,
            animate_arrows=False,
            fluid_world_size=256,
        )
