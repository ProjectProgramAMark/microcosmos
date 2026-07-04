import jax
import jax.numpy as jnp

from .line import LineEnv


class RandomFoodLineEnv(LineEnv):
    """Line environment with the Gaussian food source randomized on every reset."""

    def __init__(self, *args, energy_amplitude: float = 1.0, **kwargs):
        super().__init__(*args, energy_amplitude=energy_amplitude, **kwargs)

    def _sample_energy_center(self, key: jax.Array) -> jax.Array:
        h, w = self.grid_shape
        # Keep the centre far enough from the border that the visible Gaussian is
        # mostly inside the world, while still supporting tiny test grids.
        margin = jnp.minimum(
            jnp.array(2.0 * self.energy_size, dtype=jnp.float32),
            0.45 * jnp.array([w, h], dtype=jnp.float32),
        )
        low = margin
        high = jnp.array([w, h], dtype=jnp.float32) - margin
        return jax.random.uniform(key, (2,), minval=low, maxval=high)
