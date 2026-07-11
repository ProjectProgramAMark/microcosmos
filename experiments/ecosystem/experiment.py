"""Hydra experiment for continuous ecosystem rollouts."""

from dataclasses import replace

import jax
import matplotlib.pyplot as plt
import numpy as np

from experiments.base_experiment import Experiment
from microcosmos.gym import EcosystemEnv, LineTopology, RingTopology
from microcosmos.rendering import animate
from microcosmos.solver.config import PBD_SCHEME, PBD_SCHEME_NO_FLUID


class EcosystemExperiment(Experiment):
    def setup(self):
        cfg = self.cfg.experiment
        topology_cls = RingTopology if str(cfg.get("topology", "line")) == "ring" else LineTopology
        topology = topology_cls(
            num_nodes=int(cfg.get("nodes_per_creature", 8)),
            spacing=float(cfg.get("spacing", 2.0)),
            bending_stiffness=float(cfg.get("bending_stiffness", 0.5)),
        )
        physics = self.cfg.get("physics", {})
        enable_fluid = bool(physics.get("enable_fluid", False))
        base_solver = PBD_SCHEME if enable_fluid else PBD_SCHEME_NO_FLUID
        solver = replace(
            base_solver,
            enable_steric=bool(physics.get("enable_steric", False)),
            steric_strength=float(
                physics.get("steric_strength", base_solver.steric_strength)
            ),
            steric_sigma=float(
                physics.get("steric_sigma", base_solver.steric_sigma)
            ),
            steric_neighbor_skip=int(
                physics.get(
                    "steric_neighbor_skip", base_solver.steric_neighbor_skip
                )
            ),
            steric_scatter_value=float(
                physics.get(
                    "steric_scatter_value", base_solver.steric_scatter_value
                )
            ),
        )
        self.env = EcosystemEnv(
            topology=topology,
            max_creatures=int(cfg.max_creatures),
            initial_population=int(cfg.initial_population),
            grid_shape=tuple(cfg.grid_shape),
            dt=float(self.cfg.simulation.dt),
            max_steps=int(self.cfg.simulation.num_steps),
            solver_config=solver,
            resource_capacity=float(cfg.get("resource_capacity", 1.0)),
            initial_resource=float(cfg.get("initial_resource", 1.0)),
            resource_regeneration_rate=float(cfg.get("resource_regeneration_rate", 0.01)),
            resource_diffusion_rate=float(cfg.get("resource_diffusion_rate", 0.0)),
            initial_energy=float(cfg.get("initial_energy", 2.0)),
            birth_transfer_efficiency=float(
                cfg.get("birth_transfer_efficiency", 0.5)
            ),
            reproduction_threshold=float(cfg.get("reproduction_threshold", 4.0)),
            reproduction_cost=float(cfg.get("reproduction_cost", 2.0)),
            maturity_age=int(cfg.get("maturity_age", 100)),
            maximum_lifespan=int(cfg.get("maximum_lifespan", 10_000)),
            mutation_probability=float(cfg.get("mutation_probability", 0.05)),
            mutation_std=float(cfg.get("mutation_std", 0.05)),
            uptake_rate=float(cfg.get("uptake_rate", 0.5)),
            assimilation_efficiency=float(cfg.get("assimilation_efficiency", 0.8)),
            basal_metabolism=float(cfg.get("basal_metabolism", 0.05)),
            actuation_power_coefficient=float(
                cfg.get("actuation_power_coefficient", 0.01)
            ),
            spawn_separation=cfg.get("spawn_separation"),
            placement_candidates=int(cfg.get("placement_candidates", 16)),
        )
        self.key = jax.random.PRNGKey(int(cfg.get("seed", 0)))

    def loss_fn(self, *_args):
        raise NotImplementedError("ecosystem rollouts are not optimization experiments")

    def run(self):
        reset_key, rollout_key = jax.random.split(self.key)
        _, initial_state = self.env.reset(reset_key)
        keys = jax.random.split(rollout_key, int(self.cfg.simulation.num_steps))

        def scan_step(state, key):
            _, new_state, _, _, info = self.env.step(key, state)
            fields_out = replace(new_state.fields, f_grid=None)
            return new_state, (
                new_state.nodes,
                fields_out,
                info["telemetry"],
            )

        final_state, (nodes_ts, fields_ts, telemetry) = jax.lax.scan(
            scan_step, initial_state, keys
        )
        jax.block_until_ready(final_state.population.energy)
        self._write_metrics(telemetry)
        self._write_lineage(telemetry)
        self._write_plots(telemetry)

        animate(
            nodes_ts,
            fields_ts,
            filename=str(self.output_dir / "lifecycle.mp4"),
            subsample=int(self.cfg.simulation.get("subsample", 1)),
            fps=int(self.cfg.simulation.get("fps", 30)),
            animate_filament=False,
            animate_fluid_velocity=True,
            animate_energy=False,
        )

    def _write_metrics(self, telemetry):
        names = [
            "alive_count",
            "birth_count",
            "death_count",
            "resource_total",
            "population_energy_total",
            "mean_generation",
            "genome_variance",
        ]
        arrays = {name: np.asarray(getattr(telemetry, name)) for name in names}
        np.savez(self.output_dir / "metrics.npz", **arrays)
        matrix = np.column_stack([arrays[name] for name in names])
        np.savetxt(
            self.output_dir / "metrics.csv",
            matrix,
            delimiter=",",
            header=",".join(names),
            comments="",
        )

    def _write_lineage(self, telemetry):
        parent = np.asarray(telemetry.birth_parent_ids)
        child = np.asarray(telemetry.birth_child_ids)
        rows = []
        for step, (parents, children) in enumerate(zip(parent, child, strict=True), start=1):
            rows.extend(
                (step, int(p), int(c))
                for p, c in zip(parents, children, strict=True)
                if c >= 0
            )
        array = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
        np.savetxt(
            self.output_dir / "lineage.csv",
            array,
            fmt="%d",
            delimiter=",",
            header="step,parent_id,child_id",
            comments="",
        )

    def _write_plots(self, telemetry):
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(np.asarray(telemetry.alive_count), label="population")
        axes[0].plot(np.asarray(telemetry.birth_count), label="births", alpha=0.7)
        axes[0].plot(np.asarray(telemetry.death_count), label="deaths", alpha=0.7)
        axes[0].legend()
        axes[0].set_ylabel("organisms")
        axes[1].plot(np.asarray(telemetry.resource_total), color="tab:green")
        axes[1].set_ylabel("resource")
        axes[1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(self.output_dir / "ecosystem_metrics.png", dpi=150)
        plt.close(fig)
