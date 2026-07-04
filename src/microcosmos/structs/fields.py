import jax
from dataclasses import dataclass
import functools

@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=["grid_shape"],
    data_fields=["steric", "fluid_velocity", "f_grid", "energy"],  # expandable
)
@dataclass
class Fields:
    grid_shape: tuple[int, int]  # metadata, doesn't change
    steric: jax.Array          # density field for self-avoidance (H, W)
    fluid_velocity: jax.Array  # lattice Boltzmann velocity field (2, H, W)
    f_grid: jax.Array          # LBM distribution function (9, H, W)
    energy: jax.Array = None   # energy field (H, W)

    def __getitem__(self, idx):  # for batching if needed
        return Fields(
            grid_shape=self.grid_shape,
            steric=self.steric[idx],
            fluid_velocity=self.fluid_velocity[idx],
            f_grid=self.f_grid[idx],
            energy=self.energy[idx],
        )
