import functools
from contextlib import contextmanager
import time

import jax
import jax.numpy as jnp


EPSIL = jnp.finfo(jnp.float32).eps
PI = jnp.pi

@functools.partial(jax.jit, static_argnames=["axis", "keepdims"])
def norm(v: jax.Array, axis: int = -1, keepdims=True) -> jax.Array:
    # EPSIL in norm was necessary to avoid nans. might introduce bias?
    return jnp.maximum(jnp.linalg.norm(v + EPSIL, axis=axis, keepdims=keepdims), EPSIL)


@jax.jit
def displacement(size: jax.Array | tuple, x: jax.Array, y: jax.Array) -> jax.Array:
    # minimal-image displacement with periodic boundaries.
    # size is stored as (H, W); positions are (x, y), so flip to (W, H) for correct wrapping.
    size_arr = jnp.flip(jnp.asarray(size))
    return jnp.mod(x - y + size_arr / 2.0, size_arr) - size_arr / 2.0

@jax.jit
def index_from_position(position: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.flip(jnp.floor(position).astype(int), axis=-1).T

@jax.jit
def periodic_boundary(size: jax.Array | tuple, position: jax.Array) -> jax.Array:
    # size is stored as (H, W); positions are (x, y), so flip to (W, H) for correct wrapping.
    size_arr = jnp.flip(jnp.asarray(size))
    return jnp.mod(position, size_arr)

@functools.partial(jax.jit, static_argnames=["sigma"])
def gaussian_filter(data: jax.Array, sigma: float) -> jax.Array:
    rows, cols = data.shape
    
    # Generate Gaussian filter directly in frequency domain
    # This avoids boundary artifacts and is faster
    r = jnp.fft.fftfreq(rows)
    c = jnp.fft.fftfreq(cols)
    r, c = jnp.meshgrid(r, c, indexing='ij')
    
    # Fourier transform of Gaussian(sigma) is Gaussian(1/sigma)
    # factor = exp(-2 * pi^2 * sigma^2 * k^2)
    kernel_f = jnp.exp(-2 * jnp.pi**2 * sigma**2 * (r**2 + c**2))
    
    return jnp.fft.ifft2(jnp.fft.fft2(data) * kernel_f).real



@contextmanager
def jax_timer(name: str):
    start = time.time()
    yield
    # Block happens after yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.3f}s")

