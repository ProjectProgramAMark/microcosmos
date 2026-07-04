from abc import ABC, abstractmethod
from pathlib import Path
from omegaconf import DictConfig

import jax.numpy as jnp
import matplotlib.pyplot as plt
from microcosmos.simulate import simulate
from microcosmos.structs.fields import Fields
from microcosmos.rendering import animate
from microcosmos.solver.config import PBD_SCHEME

class Experiment(ABC):
    def __init__(self, cfg: DictConfig, output_dir: Path):
        self.cfg = cfg
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grid_shape = tuple(cfg.experiment.grid_shape)
    @abstractmethod
    def setup(self):
        """Initialize nodes, edges, optimizers, etc."""
        pass

    @abstractmethod
    def loss_fn(self, *args):
        """Define the loss function"""
        pass

    @abstractmethod
    def run(self):
        """Run the optimization loop"""
        pass


    def save_results(self):
        """Save loss plot and final animation."""

        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('Optimization Step')
        plt.ylabel('Loss')
        plt.title('Locomotion Optimization Progress')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / 'loss.png', dpi=150, bbox_inches='tight')
        plt.close()

        final_edges = self.edges.__replace__(
            bending_rest_angles=self.params['bending_rest_angles'],
            rest_lengths=self.params['rest_lengths']
        )

        # Initialize f_grid with equilibrium weights for LBM
        h, w = self.grid_shape
        weights = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        f_grid = jnp.zeros((9, h, w))
        f_grid = f_grid.at[:].set(weights[:, None, None])

        fields = Fields(
            grid_shape=self.grid_shape,
            steric=jnp.zeros(self.grid_shape),
            fluid_velocity=jnp.zeros((2, h, w)),
            f_grid=f_grid,
        )
        final_timeseries, fields_timeseries = simulate(
            self.nodes,
            final_edges,
            fields,
            self.cfg.simulation.dt,
            self.cfg.simulation.num_steps,
            PBD_SCHEME,
        )
        animate(
            final_timeseries,
            fields_timeseries,
            filename=str(self.output_dir / "final.gif"),
            subsample=1,
            uniform_color=True,
            verbose=False
        )
