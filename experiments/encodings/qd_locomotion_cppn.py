import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

from dataclasses import replace

from PIL import Image as PILImage
from microcosmos.rendering import animate, render_fields
from microcosmos.solver.config import PBD_SCHEME
from microcosmos.env import MicrocosmosEnv

from ..base_experiment import Experiment
from .utils.neat_utils import init_cppn_genome, print_genome, compute_actual_edge_metrics, true_random_seed


class QDLocomotionCPPNExperiment(Experiment):
    """
    Quality Diversity experiment — MAP-Elites with CPPN-controlled filament locomotion.

    Two behavioral descriptors (configurable via qd.bd1 / qd.bd2 in the YAML):
      speed          — average displacement per step (units/step)
      energy         — mean squared curvature change per step (rad²/step)
      turning        — average absolute change in locomotion direction (rad/step)
      spatial_freq   — fraction of joints with a sign reversal in curvature (0–1)
      body_extension — head-to-tail distance / arc length (0–1)

    Fitness: avg_speed / (avg_energy + ε)  [ratio]  or  avg_speed − avg_energy  [difference]
    A fast, efficient creature scores highest; a slow but nearly-passive creature
    can still outscore a fast but wasteful one.

    MAP-Elites fills a 2D BD grid; each cell keeps the highest-fitness genome found.
    Mutation: Gaussian weight/bias perturbation + probabilistic activation-function swap.
    Topology is fixed (weight-only mutation), sufficient for diverse locomotion gaits.
    """

    # ─── Experiment setup ────────────────────────────────────────────────────

    def setup(self):
        self.num_nodes       = self.cfg.experiment.num_nodes
        self.num_wavelengths = self.cfg.experiment.get("num_wavelengths", 1)
        self.rotation_speed  = float(self.cfg.experiment.get("rotation_speed", 0.05))
        self.grid_shape      = tuple(self.cfg.experiment.grid_shape)
        self.dt              = float(self.cfg.simulation.dt)
        self.init_line_dist  = float(self.cfg.experiment.get("init_line_distance", 2.0))

        k = 2 * jnp.pi * self.num_wavelengths
        self.node_coords = jnp.linspace(0.0, 1.0, self.num_nodes) * k

        # Physics overrides (defined before env so we can pass the scheme)
        phys = self.cfg.get("physics", {})
        self.pbd_scheme = replace(
            PBD_SCHEME,
            max_fluid_velocity=float(phys.get("max_fluid_velocity", PBD_SCHEME.max_fluid_velocity)),
            synthetic_node_width=float(phys.get("synthetic_node_width", PBD_SCHEME.synthetic_node_width)),
            ibm_kernel_size=int(phys.get("ibm_kernel_size", PBD_SCHEME.ibm_kernel_size)),
        )

        self.env = MicrocosmosEnv(
            grid_shape=self.grid_shape,
            dt=self.dt,
            num_steps=int(self.cfg.simulation.get("num_steps", 400)),
            solver_config=self.pbd_scheme,
        )

        self.init_nodes, self.init_edges = self.env.make_chain(self.num_nodes, spacing=self.init_line_dist)

        # Override to locomotion layout: 10–90% width, centered height
        domain_h, domain_w = self.grid_shape
        x_pos = jnp.linspace(domain_w * 0.1, domain_w * 0.9, self.num_nodes)
        self.init_nodes = self.init_nodes.__replace__(
            position=jnp.stack([x_pos, jnp.full(self.num_nodes, domain_h * 0.5)], axis=1)
        )

        self.init_fields = self.env.fields.__replace__(energy=jnp.zeros(self.grid_shape))

        # CPPN: (node_coord, phase) → curvature [+ edge length if evolve_distances]
        self.evolve_distances = bool(self.cfg.neat.get("evolve_distances", False))
        self.num_inputs  = 2
        self.num_outputs = 2 if self.evolve_distances else 1
        self.genome      = init_cppn_genome(num_inputs=self.num_inputs, num_outputs=self.num_outputs)
        self.neat_state  = self.genome.setup()
        self.max_curv    = float(self.cfg.neat.clip.get("max_curvature", 1.0))
        self.max_dist    = float(self.cfg.neat.clip.get("max_line_distance", 4.0))
        self.min_dist    = float(self.cfg.neat.clip.get("min_line_distance", 0.1))

        # QD hyper-parameters
        qd = self.cfg.get("qd", {})
        raw_seed = qd.get("random_seed", None)
        if raw_seed is None:
            raw_seed = true_random_seed()
            print(f"[QD] random_seed is null — using true_random_seed: {raw_seed}")
        self.random_seed = int(raw_seed)

        self.num_generations   = int(qd.get("num_generations", 100))
        self.batch_size        = int(qd.get("batch_size", 32))
        self.init_batch_size   = int(qd.get("init_batch_size", 200))
        self.archive_bins      = list(qd.get("archive_bins", [20, 20]))
        self.sample_grid       = list(qd.get("render_sample_grid", [4, 4]))
        self.bd1_type          = str(qd.get("bd1", "speed"))
        self.bd2_type          = str(qd.get("bd2", "energy"))
        self.bd1_range         = list(qd.get("bd1_range", [0.0, 0.5]))
        self.bd2_range         = list(qd.get("bd2_range", [0.0, 0.2]))
        self.auto_calibrate    = bool(qd.get("auto_calibrate_ranges", False))
        self.calibrate_pct     = float(qd.get("calibrate_percentile", 95))
        self.eval_pre_steps    = int(qd.get("eval_pre_steps", 20))
        self.eval_num_steps    = int(qd.get("eval_num_steps", 100))
        self.fitness_type      = str(qd.get("fitness_type", "speed"))  # 'speed', 'speed-cost', or 'speed/cost'
        self.weight_BE         = float(qd.get("weight_BE", 1.0))
        self.weight_BS         = float(qd.get("weight_BS", 5.0))
        self.weight_SE         = float(qd.get("weight_SE", 0.5))
        self.weight_SS         = float(qd.get("weight_SS", 2.5))
        self.latest_archive_path = str(qd.get("latest_archive", self.output_dir / "latest_archive.npz"))

        # Available behavioral descriptors (order defines the index used inside JAX)
        self.BD_NAMES = ["speed", "energy", "turning", "spatial_freq", "body_extension",
                         "actuation_mode", "compliance",
                         "bend_effort", "bend_strain", "stretch_effort", "stretch_strain",
                         "efforts", "strains"]
        self.BD_LABELS = {
            "speed":           "Average speed (units / step)",
            "energy":          "Average energy spent per step (weighted effort + strain)",
            "turning":         "Average turning rate (rad / step)",
            "spatial_freq":    "Spatial curvature frequency (sign reversals / joint)",
            "body_extension":  "Head-to-tail extension (fraction of arc length)",
            "actuation_mode":  "Bending effort fraction — bend/(bend+stretch)",
            "compliance":      "Actuation compliance — effort/(effort+strain)",
            "bend_effort":     "Bend effort — ||previous - current curvatures||²",
            "bend_strain":     "Bend strain — ||expected - actual curvatures||²",
            "stretch_effort":  "Stretch effort — ||previous - current distances||²",
            "stretch_strain":  "Stretch strain — ||expected - actual distances||²",
            "efforts":         "Total actuation effort (weighted bend + stretch effort)",
            "strains":         "Total constraint strain (weighted bend + stretch strain)",
        }
        for bd in (self.bd1_type, self.bd2_type):
            if bd not in self.BD_NAMES:
                raise ValueError(f"Unknown BD type '{bd}'. Choose from: {self.BD_NAMES}")
        self.bd1_idx = self.BD_NAMES.index(self.bd1_type)
        self.bd2_idx = self.BD_NAMES.index(self.bd2_type)

    def loss_fn(self, _params):
        raise NotImplementedError("QD experiment uses MAP-Elites, not gradient optimisation")

    def run(self):
        self.run_qd()

    # ─── JAX-compiled batch evaluation ──────────────────────────────────────

    def _build_eval_fn(self):
        """
        Return a JIT-compiled function:
            (batch_neurons, batch_conns) → (fitness, bd1, bd2)

        All five BDs are computed every evaluation; bd1/bd2 are selected by index.

        Available BDs (bd1 / bd2 in config):
          speed          — average displacement per step (units/step)
          energy         — mean squared curvature change per step (rad²/step)
          turning        — average absolute change in locomotion direction (rad/step)
          spatial_freq   — average spatial frequency of curvature sign reversals (1/joint)
          body_extension — average head-to-tail distance / arc length (dimensionless)

        Fitness = total_displacement / (mean_energy_per_step + epsilon)
                  A fast creature with low energy use scores highest; a slow but extremely
                  efficient creature can still outscore a fast but wasteful one.
        """
        genome           = self.genome
        neat_state       = self.neat_state
        init_nodes       = self.init_nodes
        init_edges       = self.init_edges
        init_fields      = self.init_fields
        node_coords      = self.node_coords
        rotation_speed   = self.rotation_speed
        eval_pre_steps   = self.eval_pre_steps
        max_curv         = self.max_curv
        max_dist         = self.max_dist
        min_dist         = self.min_dist
        evolve_distances = self.evolve_distances
        weight_BE        = self.weight_BE
        weight_BS        = self.weight_BS
        weight_SE        = self.weight_SE
        weight_SS        = self.weight_SS
        fitness_type     = self.fitness_type
        env              = self.env
        bd1_idx          = self.bd1_idx   # Python int — not traced by JAX
        bd2_idx          = self.bd2_idx
        num_bending      = int(self.init_edges.bending_rest_angles.shape[0])
        num_edges        = int(self.init_edges.rest_lengths.shape[0])
        total_steps      = eval_pre_steps + self.eval_num_steps
        domain_size      = jnp.array([self.grid_shape[1], self.grid_shape[0]], dtype=jnp.float32)
        arc_length       = float(jnp.sum(self.init_edges.rest_lengths))  # total chain arc length
        init_dist        = init_edges.rest_lengths[:num_edges]   # constant when not evolving distances

        def eval_single(neat_params):
            """Evaluate one genome. Returns (fitness, bd1, bd2)."""
            node_fwd = jax.vmap(genome.forward, in_axes=(None, None, 0))

            prev_center    = jnp.mean(init_nodes.position, axis=0)
            init_curv      = init_edges.bending_rest_angles
            prev_direction = jnp.float32(0.0)   # angle of previous step's motion (rad)

            # Accumulators: [speed, energy, turning, spatial_freq, body_extension,
            #                actuation_mode, compliance,
            #                bend_effort, bend_strain, stretch_effort, stretch_strain,
            #                efforts, strains, count]
            acc0 = jnp.zeros(14, dtype=jnp.float32)

            def scan_step(carry, current_step):
                nodes, edges, fields, prev_center, prev_curv, prev_dist, prev_dir, acc = carry

                phase_arr   = jnp.full_like(node_coords, current_step * rotation_speed)
                cppn_inputs = jnp.stack([node_coords, phase_arr], axis=1)

                outputs  = node_fwd(neat_state, neat_params, cppn_inputs)
                new_curv = jnp.clip(outputs[:, 0].flatten(), -max_curv, max_curv)
                edges = edges.__replace__(bending_rest_angles=new_curv[:num_bending])
                if evolve_distances:
                    new_dist = jnp.clip(outputs[:, 1].flatten(), min_dist, max_dist)
                    edges = edges.__replace__(rest_lengths=new_dist[:num_edges])
                    cur_dist = new_dist[:num_edges]
                else:
                    cur_dist = init_dist
                nodes, edges, fields = env.step(nodes, edges, fields)
                actual_angles, actual_lengths = compute_actual_edge_metrics(nodes, edges, self.grid_shape)

                # ── speed ──────────────────────────────────────────────────
                center     = jnp.mean(nodes.position, axis=0)
                delta      = (center - prev_center + domain_size / 2) % domain_size - domain_size / 2
                movement   = jnp.linalg.norm(delta)
                new_center = prev_center + delta

                # ── energy model ────────────────────────────────────────────
                # efforts: cost of changing commanded values (actuation work)
                # strains: mismatch between commanded and actual (constraint resistance)
                bend_effort    = jnp.sum((new_curv[:num_bending] - prev_curv) ** 2)
                bend_strain    = jnp.sum((new_curv[:num_bending] - actual_angles) ** 2)
                stretch_effort = jnp.sum((cur_dist - prev_dist) ** 2)
                stretch_strain = jnp.sum((cur_dist - actual_lengths) ** 2)
                efforts        = weight_BE * bend_effort + weight_SE * stretch_effort
                strains        = weight_BS * bend_strain + weight_SS * stretch_strain
                energy_spent   = efforts + strains

                # ── actuation mode: fraction of effort from bending ─────────
                actuation_mode = jnp.where(efforts > 0.0, bend_effort / efforts, jnp.float32(0.5))

                # ── compliance: effort as fraction of total energy ───────────
                compliance = jnp.where(energy_spent > 0.0, efforts / energy_spent, jnp.float32(1.0))

                # ── turning: |Δdirection| per step ─────────────────────────
                direction  = jnp.arctan2(delta[1], delta[0])
                raw_dangle = direction - prev_dir
                dangle     = (raw_dangle + jnp.pi) % (2 * jnp.pi) - jnp.pi
                turning    = jnp.where(movement > 0.01, jnp.abs(dangle), jnp.float32(0.0))
                new_dir    = jnp.where(movement > 0.01, direction, prev_dir)

                # ── spatial frequency: sign reversals along body ────────────
                signs      = jnp.sign(new_curv[:num_bending])
                reversals  = jnp.sum(signs[1:] * signs[:-1] < 0)
                spatial_f  = reversals / jnp.float32(max(num_bending - 1, 1))

                # ── body extension: head-to-tail / arc length ───────────────
                ht_delta  = (nodes.position[-1] - nodes.position[0] + domain_size / 2) % domain_size - domain_size / 2
                extension = jnp.linalg.norm(ht_delta) / jnp.float32(arc_length)

                # ── accumulate after warm-up ────────────────────────────────
                is_eval = current_step >= eval_pre_steps
                update  = jnp.array([movement, energy_spent, turning, spatial_f, extension,
                                     actuation_mode, compliance,
                                     bend_effort, bend_strain, stretch_effort, stretch_strain,
                                     efforts, strains, 1.0])
                acc     = jnp.where(is_eval, acc + update, acc)

                return (nodes, edges, fields, new_center, new_curv[:num_bending],
                        cur_dist, new_dir, acc), None

            carry0 = (
                init_nodes, init_edges, init_fields,
                prev_center, init_curv, init_dist, prev_direction, acc0,
            )
            (_, _, _, _, _, _, _, acc), _ = jax.lax.scan(
                scan_step, carry0, jnp.arange(total_steps)
            )

            n = jnp.maximum(acc[13], 1.0)
            bd_vals = acc[:13] / n   # per-step averages for all 13 BDs

            if fitness_type == "speed":
                fitness = bd_vals[0]
            elif fitness_type == "speed-cost":
                fitness = bd_vals[0] - bd_vals[1]   # avg_speed − avg_energy_spent
            elif fitness_type == "speed/cost":
                fitness = bd_vals[0] / (bd_vals[1] + 1e-3)  # avg_speed / avg_energy_spent

            return fitness, bd_vals[bd1_idx], bd_vals[bd2_idx]

        def batch_eval(batch_neurons, batch_conns):
            pop_params = jax.vmap(lambda n, c: genome.transform(neat_state, n, c))(
                batch_neurons, batch_conns
            )
            fitness, bd1, bd2 = jax.vmap(eval_single)(pop_params)
            return fitness, bd1, bd2

        return jax.jit(batch_eval)

    # ─── Mutation ────────────────────────────────────────────────────────────

    def _mutate_batch(self, batch_neurons, batch_conns, randkey):
        """
        Mutate a batch of genomes using TensorNEAT's built-in DefaultMutation.

        Handles weight/bias perturbation, activation-function swaps, and
        structural changes (add/delete nodes and connections) in one JIT-able
        vmap call — identical to what the NEAT algorithm does internally.

        new_node_keys: each individual gets a unique scalar ID for any node it
        might add, derived from the batch-wide maximum key already in use.
        new_conn_keys: shape (n, 3) innovation markers — zeros because MAP-Elites
        does not use NEAT speciation.
        """
        n    = batch_neurons.shape[0]
        keys = jax.random.split(randkey, n)

        valid         = ~jnp.isnan(batch_neurons[:, :, 0])
        max_key       = jnp.max(batch_neurons[:, :, 0], where=valid, initial=0.0)
        new_node_keys = jnp.arange(n, dtype=jnp.float32) + max_key + 1
        new_conn_keys = jnp.zeros((n, 3), dtype=jnp.float32)

        return jax.vmap(
            self.genome.execute_mutation, in_axes=(None, 0, 0, 0, 0, 0)
        )(self.neat_state, keys, batch_neurons, batch_conns, new_node_keys, new_conn_keys)

    # ─── Archive helpers ──────────────────────────────────────────────────────

    def _bd_to_idx(self, bd1_vals, bd2_vals):
        nb1, nb2 = self.archive_bins
        lo1, hi1 = self.bd1_range
        lo2, hi2 = self.bd2_range
        i1 = np.clip((np.nan_to_num(bd1_vals, nan=lo1) - lo1) / (hi1 - lo1) * nb1, 0, nb1 - 1).astype(int)
        i2 = np.clip((np.nan_to_num(bd2_vals, nan=lo2) - lo2) / (hi2 - lo2) * nb2, 0, nb2 - 1).astype(int)
        return i1, i2

    def _update_archive(self, archive_fitness, archive_neurons, archive_conns,
                        batch_neurons, batch_conns, fitnesses, bd1s, bd2s):
        improved = 0
        i1s, i2s    = self._bd_to_idx(bd1s, bd2s)
        neurons_np  = np.asarray(batch_neurons)
        conns_np    = np.asarray(batch_conns)
        for k in range(len(fitnesses)):
            i1, i2 = int(i1s[k]), int(i2s[k])
            if fitnesses[k] > archive_fitness[i1, i2]:
                archive_fitness[i1, i2]  = fitnesses[k]
                archive_neurons[i1, i2]  = neurons_np[k]
                archive_conns[i1, i2]    = conns_np[k]
                improved += 1
        return improved

    def _sample_from_archive(self, archive_fitness, archive_neurons, archive_conns, n):
        occupied = np.argwhere(np.isfinite(archive_fitness))
        if len(occupied) == 0:
            return None
        sel = occupied[self._np_rng.choice(len(occupied), size=n, replace=True)]
        return (
            jnp.array(archive_neurons[sel[:, 0], sel[:, 1]]),
            jnp.array(archive_conns[sel[:, 0], sel[:, 1]]),
        )

    # ─── Visualisation ───────────────────────────────────────────────────────

    def _plot_archive(self, archive_fitness, generation):
        nb1, nb2    = self.archive_bins
        lo1, hi1    = self.bd1_range
        lo2, hi2    = self.bd2_range
        display     = np.where(np.isfinite(archive_fitness), archive_fitness, np.nan)
        num_filled  = int(np.sum(np.isfinite(archive_fitness)))

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            display.T, origin="lower", aspect="auto",
            extent=[lo1, hi1, lo2, hi2],
            cmap="viridis", vmin=0,
        )
        if self.fitness_type == "speed":
            fitness_type_label = "displacement"
        elif self.fitness_type == "speed-cost":
            fitness_type_label = "displacement - energy"
        elif self.fitness_type == "speed/cost":
            fitness_type_label = "displacement / energy"
        plt.colorbar(im, ax=ax, label=f"Fitness ({fitness_type_label})")
        ax.set_xlabel(f"BD1 ({self.bd1_type}): {self.BD_LABELS[self.bd1_type]}")
        ax.set_ylabel(f"BD2 ({self.bd2_type}): {self.BD_LABELS[self.bd2_type]}")
        ax.set_title(
            f"MAP-Elites Archive  —  Gen {generation}  |  "
            f"Coverage {num_filled} / {nb1 * nb2} cells"
        )
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"archive_gen{generation:04d}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    # ─── BD range helpers ────────────────────────────────────────────────────

    def _report_bd_ranges(self, bd1, bd2, fit):
        """Print observed BD statistics from the seeding population."""
        valid = np.isfinite(bd1) & np.isfinite(bd2) & np.isfinite(fit)
        if not np.any(valid):
            return
        b1, b2 = bd1[valid], bd2[valid]
        pct = self.calibrate_pct
        print(
            f"\nBD statistics from seeding population (n={valid.sum()}):\n"
            f"  {self.bd1_type:16s}  min={b1.min():.4f}  p{100-pct:.0f}={np.percentile(b1, 100-pct):.4f}"
            f"  median={np.median(b1):.4f}  p{pct:.0f}={np.percentile(b1, pct):.4f}  max={b1.max():.4f}\n"
            f"  {self.bd2_type:16s}  min={b2.min():.4f}  p{100-pct:.0f}={np.percentile(b2, 100-pct):.4f}"
            f"  median={np.median(b2):.4f}  p{pct:.0f}={np.percentile(b2, pct):.4f}  max={b2.max():.4f}\n"
            f"  fitness          min={fit[valid].min():.4f}  median={np.median(fit[valid]):.4f}"
            f"  max={fit[valid].max():.4f}\n"
            f"Suggested YAML ranges (p{100-pct:.0f}–p{pct:.0f}):\n"
            f"  bd1_range: [0.0, {np.percentile(b1, pct):.3f}]\n"
            f"  bd2_range: [0.0, {np.percentile(b2, pct):.3f}]"
        )
        if self.auto_calibrate:
            print("  (auto_calibrate_ranges: true — applying these now)")
        else:
            print("  (set auto_calibrate_ranges: true to apply automatically)\n")

    def _calibrate_ranges(self, bd1, bd2):
        """Set BD ranges to [0, p{calibrate_pct}] of the seeding distribution."""
        valid = np.isfinite(bd1) & np.isfinite(bd2)
        pct = self.calibrate_pct
        hi1 = float(np.percentile(bd1[valid], pct)) if valid.any() else self.bd1_range[1]
        hi2 = float(np.percentile(bd2[valid], pct)) if valid.any() else self.bd2_range[1]
        # Clamp to a small positive floor so the range is never degenerate
        hi1 = max(hi1, 1e-6)
        hi2 = max(hi2, 1e-6)
        return [0.0, hi1], [0.0, hi2]

    # ─── Main MAP-Elites loop ─────────────────────────────────────────────────

    def run_qd(self):
        nb1, nb2  = self.archive_bins
        max_nodes = self.genome.max_nodes
        max_conns = self.genome.max_conns

        # Archive: fitness grid + corresponding genome arrays
        archive_fitness  = np.full((nb1, nb2), -np.inf)
        archive_neurons  = np.full((nb1, nb2, max_nodes, 5), np.nan)
        archive_conns    = np.full((nb1, nb2, max_conns, 3), np.nan)

        randkey    = jax.random.PRNGKey(self.random_seed)
        self._np_rng = np.random.default_rng(self.random_seed)
        batch_eval = self._build_eval_fn()  # triggers JIT on first call

        # ── Seed the archive with a random initial population ──────────────
        print(f"Seeding archive with {self.init_batch_size} random genomes...")
        randkey, k = jax.random.split(randkey)
        init_keys  = jax.random.split(k, self.init_batch_size)
        init_n, init_c = jax.vmap(
            lambda key: self.genome.initialize(self.neat_state, key)
        )(init_keys)

        fit, bd1, bd2 = batch_eval(init_n, init_c)
        fit, bd1, bd2 = np.array(fit), np.array(bd1), np.array(bd2)

        # ── Auto-calibrate BD ranges from the seeding distribution ────────
        # Prints suggested ranges even when auto_calibrate is off, so users
        # can copy the values into their YAML for future reproducible runs.
        self._report_bd_ranges(bd1, bd2, fit)
        if self.auto_calibrate:
            self.bd1_range, self.bd2_range = self._calibrate_ranges(bd1, bd2)

        self._update_archive(
            archive_fitness, archive_neurons, archive_conns,
            np.array(init_n), np.array(init_c), fit, bd1, bd2,
        )
        num_filled = np.sum(np.isfinite(archive_fitness))
        print(f"Initial coverage: {num_filled} / {nb1 * nb2} cells")
        self._plot_archive(archive_fitness, 0)

        fitness_hist  = []
        coverage_hist = []

        # ── Main QD loop ───────────────────────────────────────────────────
        for gen in range(1, self.num_generations + 1):
            randkey, ks, km = jax.random.split(randkey, 3)

            sampled = self._sample_from_archive(
                archive_fitness, archive_neurons, archive_conns, self.batch_size
            )
            if sampled is None:
                # Archive still empty — generate a fresh random batch
                keys = jax.random.split(ks, self.batch_size)
                batch_n, batch_c = jax.vmap(
                    lambda key: self.genome.initialize(self.neat_state, key)
                )(keys)
            else:
                batch_n, batch_c = sampled
                batch_n, batch_c = self._mutate_batch(batch_n, batch_c, km)

            fit, bd1, bd2 = batch_eval(batch_n, batch_c)
            fit, bd1, bd2 = np.array(fit), np.array(bd1), np.array(bd2)

            n_imp = self._update_archive(
                archive_fitness, archive_neurons, archive_conns,
                np.array(batch_n), np.array(batch_c), fit, bd1, bd2,
            )

            num_filled = int(np.sum(np.isfinite(archive_fitness)))
            valid      = archive_fitness[np.isfinite(archive_fitness)]
            best       = float(valid.max()) if len(valid) else 0.0

            fitness_hist.append(best)
            coverage_hist.append(num_filled)

            if gen % 10 == 0 or gen == self.num_generations:
                print(
                    f"Gen {gen:4d}  |  Coverage {num_filled:4d}/{nb1 * nb2}"
                    f"  |  Best fitness {best:.3f}  |  Cells improved {n_imp}"
                )

            if gen % 10 == 0 or gen == self.num_generations:
                self._plot_archive(archive_fitness, gen)

        self.save_results(archive_fitness, archive_neurons, archive_conns,
                          fitness_hist, coverage_hist, self.sample_grid)

    # ─── Results ─────────────────────────────────────────────────────────────

    def save_results(self, archive_fitness, archive_neurons, archive_conns,
                     fitness_hist, coverage_hist, sample_grid):
        np.savez(
            self.output_dir / "archive.npz",
            fitness=archive_fitness,
            neurons=archive_neurons,
            conns=archive_conns,
        )
        print(f"Archive saved → {self.output_dir / 'archive.npz'}")

        Path(self.latest_archive_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.latest_archive_path,
            fitness=archive_fitness,
            neurons=archive_neurons,
            conns=archive_conns,
        )
        print(f"Archive saved → {self.latest_archive_path}")

        # Copy the Hydra config next to latest_archive.npz so downstream tools
        # (neural_control, figure_4) can auto-load it without knowing the run timestamp.
        import shutil
        hydra_cfg = self.output_dir / ".hydra" / "config.yaml"
        latest_cfg = self.latest_archive_path.replace(".npz", "_config.yaml")
        if hydra_cfg.exists():
            shutil.copy(hydra_cfg, latest_cfg)
            print(f"Archive config copied → {latest_cfg}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(fitness_hist)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Best fitness")
        ax1.set_title("Best Fitness over Generations")
        ax1.grid(True, alpha=0.3)
        ax2.plot(coverage_hist)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Archive cells filled")
        ax2.set_title("Archive Coverage over Generations")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "progress.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Progress plot saved → {self.output_dir / 'progress.png'}")

        self._render_diverse_animations(archive_fitness, archive_neurons, archive_conns, sample_grid)
        self._render_all_cells(archive_fitness, archive_neurons, archive_conns)

        figure4_cfg = self.cfg.get("figure4", None)
        if figure4_cfg is not None:
            self._render_archive_collage(
                archive_fitness, archive_neurons, archive_conns,
                render_step=int(figure4_cfg.get("render_step", 200)),
                render_every=int(figure4_cfg.get("render_every", 1)),
                sz=int(figure4_cfg.get("sz", 512)),
            )

    def _render_archive_collage(self, archive_fitness, archive_neurons, archive_conns,
                                render_step, render_every=1, sz=512):
        """Render a collage image with one frame per archive cell (figure 4)."""
        if not np.any(np.isfinite(archive_fitness)):
            return

        nb1, nb2 = archive_fitness.shape
        border = 4
        h, w = self.grid_shape
        domain_size = jnp.array([w, h], dtype=jnp.float32)
        domain_center = domain_size / 2.0

        ncols = nb1 // render_every
        nrows = nb2 // render_every
        print(f"Rendering archive collage at step {render_step} ({ncols}×{nrows} cells)...")
        collage = np.zeros((nrows * (sz + border), ncols * (sz + border), 3), dtype=np.uint8)

        batch_forward = jax.jit(jax.vmap(self.genome.forward, in_axes=(None, None, 0)))

        def scan_fn(carry, i):
            nodes, edges, fields, neat_params = carry
            current_phase = i * self.rotation_speed
            phase_arr = jnp.full_like(self.node_coords, current_phase)
            cppn_inputs = jnp.stack([self.node_coords, phase_arr], axis=1)
            outputs = batch_forward(self.neat_state, neat_params, cppn_inputs)
            new_curv = jnp.clip(outputs[:, 0].flatten(), -self.max_curv, self.max_curv)
            edges = edges.__replace__(bending_rest_angles=new_curv[:edges.bending_rest_angles.shape[0]])
            if self.evolve_distances:
                new_dist = jnp.clip(outputs[:, 1].flatten(), self.min_dist, self.max_dist)
                edges = edges.__replace__(rest_lengths=new_dist[:edges.rest_lengths.shape[0]])
            nodes, edges, fields = self.env.step(nodes, edges, fields)
            return (nodes, edges, fields, neat_params), None

        def unwrap_chain(positions):
            def body(prev, curr):
                delta = (curr - prev + domain_size / 2) % domain_size - domain_size / 2
                return prev + delta, prev + delta
            _, rest = jax.lax.scan(body, positions[0], positions[1:])
            return jnp.concatenate([positions[0:1], rest], axis=0)

        for bd1 in range(0, nb1, render_every):
            for bd2 in range(0, nb2, render_every):
                if not np.isfinite(archive_fitness[bd1, bd2]):
                    continue
                print(".", end="", flush=True)

                neurons = jnp.array(archive_neurons[bd1, bd2])
                conns = jnp.array(archive_conns[bd1, bd2])
                neat_params = self.genome.transform(self.neat_state, neurons, conns)

                (final_nodes, _, final_fields, _), _ = jax.lax.scan(
                    scan_fn,
                    (self.init_nodes, self.init_edges, self.init_fields, neat_params),
                    jnp.arange(render_step),
                )

                unwrapped = unwrap_chain(final_nodes.position)
                com = jnp.mean(unwrapped, axis=0)
                centered_pos = (unwrapped - com + domain_center) % domain_size
                centered_nodes = final_nodes.__replace__(position=centered_pos)

                wrapped_com = com % domain_size
                shift_x = int(jnp.round(domain_center[0] - wrapped_com[0]))
                shift_y = int(jnp.round(domain_center[1] - wrapped_com[1]))
                centered_fields = jax.tree.map(
                    lambda f: jnp.roll(jnp.roll(f, shift_y, axis=-2), shift_x, axis=-1),
                    final_fields,
                )

                nodes_ts = jax.tree.map(lambda x: jnp.expand_dims(x, 0), centered_nodes)
                fields_ts = jax.tree.map(lambda x: jnp.expand_dims(x, 0), centered_fields)
                cell_img = render_fields(fields_ts, nodes_ts, sz=sz, animate_energy=False, animate_arrows=False)[0]

                row = nrows - 1 - bd2 // render_every
                col = bd1 // render_every
                collage[row * (sz + border):row * (sz + border) + sz,
                        col * (sz + border):col * (sz + border) + sz] = cell_img

            print("|", flush=True)

        output_path = self.output_dir / 'archive_collage.png'
        PILImage.fromarray(collage).save(str(output_path))
        print(f"Archive collage saved to {output_path}")

    def _render_diverse_animations(self, archive_fitness, archive_neurons, archive_conns, sample_grid):
        """
        Render one GIF per cell of a (sg1 × sg2) sampling grid laid over the archive.
        The archive is divided into equal rectangular regions; within each region the
        occupied cell with the highest fitness is chosen as the representative.
        Skips regions that contain no occupied archive cell.
        """
        if not np.any(np.isfinite(archive_fitness)):
            return

        nb1, nb2   = self.archive_bins
        sg1, sg2   = sample_grid

        num_steps   = int(self.cfg.simulation.get("num_steps", 400))
        subsample   = int(self.cfg.simulation.get("subsample", 4))
        fps         = int(self.cfg.simulation.get("fps", 30))
        num_bending      = int(self.init_edges.bending_rest_angles.shape[0])
        num_edges        = int(self.init_edges.rest_lengths.shape[0])
        rot_speed        = self.rotation_speed
        max_curv         = self.max_curv
        max_dist         = self.max_dist
        min_dist         = self.min_dist
        evolve_distances = self.evolve_distances
        nc               = self.node_coords
        genome           = self.genome
        neat_state       = self.neat_state
        gs               = neat_state  # alias used in transform call below
        env              = self.env

        # Single JIT-compiled render function shared across all elites.
        # params is passed as an argument (not closed over), so the function
        # is traced once and reused for every genome without retracing.

        @jax.jit
        def render_genome(params, init_nodes, init_edges, init_fields):
            def scan_step(carry, step_i):
                nodes, edges, fields = carry
                phase_arr   = jnp.full_like(nc, step_i * rot_speed)
                cppn_inputs = jnp.stack([nc, phase_arr], axis=1)
                outputs     = jax.vmap(genome.forward, in_axes=(None, None, 0))(neat_state, params, cppn_inputs)
                curv        = jnp.clip(outputs[:, 0].flatten(), -max_curv, max_curv)
                edges       = edges.__replace__(bending_rest_angles=curv[:num_bending])
                if evolve_distances:
                    dist  = jnp.clip(outputs[:, 1].flatten(), min_dist, max_dist)
                    edges = edges.__replace__(rest_lengths=dist[:num_edges])
                nodes, edges, fields = env.step(nodes, edges, fields)
                return (nodes, edges, fields), (nodes, fields)

            _, (nodes_traj, fields_traj) = jax.lax.scan(
                scan_step,
                (init_nodes, init_edges, init_fields),
                jnp.arange(num_steps),
            )
            return nodes_traj, fields_traj

        rendered = 0
        original_stdout = sys.stdout
        for ri in range(sg1):
            for rj in range(sg2):
                # Index range of archive cells in this sampling region
                i1_lo = ri       * nb1 // sg1
                i1_hi = (ri + 1) * nb1 // sg1
                i2_lo = rj       * nb2 // sg2
                i2_hi = (rj + 1) * nb2 // sg2

                region = archive_fitness[i1_lo:i1_hi, i2_lo:i2_hi]
                if not np.any(np.isfinite(region)):
                    continue  # empty region — skip

                # Best-fitness cell within this region
                local_idx = np.unravel_index(
                    np.nanargmax(np.where(np.isfinite(region), region, np.nan)),
                    region.shape,
                )
                i1 = i1_lo + local_idx[0]
                i2 = i2_lo + local_idx[1]

                neurons = jnp.array(archive_neurons[i1, i2])
                conns   = jnp.array(archive_conns[i1, i2])
                params  = self.genome.transform(gs, neurons, conns)

                nodes_traj, fields_traj = render_genome(
                    params, self.init_nodes, self.init_edges, self.init_fields
                )

                fit_val  = float(archive_fitness[i1, i2])
                filename = str(
                    self.output_dir / f"diverse_r{ri}-{rj}_fit{fit_val:.2f}_cell{i1}-{i2}.gif"
                )
                animate(
                    nodes_traj, fields_traj,
                    filename=filename,
                    subsample=subsample,
                    uniform_color=False, verbose=False, fps=fps,
                )
                rendered += 1
                print(f"Diverse sample [{ri},{rj}] (cell {i1},{i2}, fit {fit_val:.3f})")
                with open(self.output_dir / 'diverse_genome.txt', 'a') as f:
                    sys.stdout = f
                    print_genome(self.genome, neat_state, neurons, conns)
                    sys.stdout = original_stdout

        print(f"Rendered {rendered} / {sg1 * sg2} diverse animations")

    def _render_all_cells(self, archive_fitness, archive_neurons, archive_conns):
        """Render a GIF for every occupied archive cell into output_dir/cells/."""
        if not np.any(np.isfinite(archive_fitness)):
            return

        cells_dir = self.output_dir / "cells"
        cells_dir.mkdir(exist_ok=True)

        num_steps        = int(self.cfg.simulation.get("num_steps", 400))
        subsample        = int(self.cfg.simulation.get("subsample", 4))
        fps              = int(self.cfg.simulation.get("fps", 30))
        num_bending      = int(self.init_edges.bending_rest_angles.shape[0])
        num_edges        = int(self.init_edges.rest_lengths.shape[0])
        rot_speed        = self.rotation_speed
        max_curv         = self.max_curv
        max_dist         = self.max_dist
        min_dist         = self.min_dist
        evolve_distances = self.evolve_distances
        nc               = self.node_coords
        genome           = self.genome
        neat_state       = self.neat_state
        env              = self.env

        @jax.jit
        def render_genome(params, init_nodes, init_edges, init_fields):
            def scan_step(carry, step_i):
                nodes, edges, fields = carry
                phase_arr   = jnp.full_like(nc, step_i * rot_speed)
                cppn_inputs = jnp.stack([nc, phase_arr], axis=1)
                outputs     = jax.vmap(genome.forward, in_axes=(None, None, 0))(neat_state, params, cppn_inputs)
                curv        = jnp.clip(outputs[:, 0].flatten(), -max_curv, max_curv)
                edges       = edges.__replace__(bending_rest_angles=curv[:num_bending])
                if evolve_distances:
                    dist  = jnp.clip(outputs[:, 1].flatten(), min_dist, max_dist)
                    edges = edges.__replace__(rest_lengths=dist[:num_edges])
                nodes, edges, fields = env.step(nodes, edges, fields)
                return (nodes, edges, fields), (nodes, fields)

            _, (nodes_traj, fields_traj) = jax.lax.scan(
                scan_step,
                (init_nodes, init_edges, init_fields),
                jnp.arange(num_steps),
            )
            return nodes_traj, fields_traj

        occupied = list(zip(*np.where(np.isfinite(archive_fitness))))
        print(f"Rendering all {len(occupied)} occupied cells → {cells_dir}")
        for idx, (i1, i2) in enumerate(occupied):
            neurons = jnp.array(archive_neurons[i1, i2])
            conns   = jnp.array(archive_conns[i1, i2])
            params  = genome.transform(neat_state, neurons, conns)
            nodes_traj, fields_traj = render_genome(
                params, self.init_nodes, self.init_edges, self.init_fields
            )
            fit_val  = float(archive_fitness[i1, i2])
            filename = str(cells_dir / f"cell_{i1:02d}-{i2:02d}_fit{fit_val:.2f}.gif")
            animate(
                nodes_traj, fields_traj,
                filename=filename,
                subsample=subsample,
                uniform_color=False, verbose=False, fps=fps,
            )
            print(f"  [{idx+1}/{len(occupied)}] cell ({i1},{i2}) fit {fit_val:.3f}")

        print(f"Done — {len(occupied)} GIFs in {cells_dir}")

