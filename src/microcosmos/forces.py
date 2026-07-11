from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from dataclasses import replace

from microcosmos.utils import (
    displacement,
    gaussian_filter,
    periodic_boundary,
)

if TYPE_CHECKING:
    from microcosmos.structs.fields import Fields
    from microcosmos.structs.nodes import Nodes


def force_interp(position: jax.Array, potential: jax.Array, fd_eps: float = 0.5) -> jax.Array:
    """Curl-free force = −∇V at continuous positions, with PBC wrap.

    Central FD of the bilinearly-sampled potential rather than bilinear sampling
    of the spectral gradient. Because the FD gradient of any scalar field has
    equal mixed partials by construction, the resulting vector field is curl-free
    → no spurious work along closed loops → no energy injection in tight coils.
    """
    y = position[..., 1]
    x = position[..., 0]

    def sample_V(yy, xx):
        return jax.scipy.ndimage.map_coordinates(
            potential, jnp.stack([yy, xx], axis=0), order=1, mode="wrap"
        )

    gx = (sample_V(y, x + fd_eps) - sample_V(y, x - fd_eps)) / (2.0 * fd_eps)
    gy = (sample_V(y + fd_eps, x) - sample_V(y - fd_eps, x)) / (2.0 * fd_eps)
    return -jnp.stack([gx, gy], axis=-1)


def compute_field_steric_force_corrected(
    nodes: Nodes,
    fields: Fields,
    sigma: float,
    neighbor_skip: int,
    scatter_value: float = 1.0,
) -> tuple[Fields, jax.Array]:
    """FFT-Gaussian field with analytic self + n-neighbor contribution subtracted.

    Build smoothed density (each node scattered as a unit-area Gaussian of weight
    `scatter_value`). Bilinearly sample force = −∇V at each node, then subtract the
    analytic contribution from j ∈ {i−n, …, i+n}:

        f_corr_i = f_interp_i − (w/σ²) · Σ_{|i−j|≤n} K_σ(r_ij) · r_ij

    where K_σ(r) = exp(−|r|²/2σ²) / (2π σ²) and r_ij = x_i − x_j under PBC.
    Self term (j=i) has r=0 → contributes nothing; included for index symmetry.
    `neighbor_skip=0` disables the subtraction entirely (each node feels its
    own deposited bump plus its neighbors').
    """
    fields = update_steric_potential(nodes, fields, sigma=sigma, scatter_value=scatter_value)

    pos = periodic_boundary(fields.grid_shape, nodes.position)
    raw = force_interp(pos, fields.steric)  # (N, 2)

    N = pos.shape[0]
    offsets = jnp.arange(-neighbor_skip, neighbor_skip + 1)        # (2n+1,)
    j_idx   = jnp.arange(N)[:, None] + offsets[None, :]            # (N, 2n+1)
    valid = (j_idx >= 0) & (j_idx < N)  # line topology
    j_safe  = jnp.clip(j_idx, 0, N - 1)

    if nodes.active is not None:
        active = nodes.active.astype(jnp.bool_)
        valid = valid & active[:, None] & active[j_safe]
    if nodes.component_id is not None:
        valid = valid & (
            nodes.component_id[:, None] == nodes.component_id[j_safe]
        )
    valid = valid.astype(pos.dtype)

    # Bilinear scatter + spectral gradient + bilinear sample is self-consistent at
    # continuous positions — use exact x_j (no floor) to cancel neighbor contribution.
    r = displacement(fields.grid_shape, pos[:, None, :], pos[j_safe])   # (N, 2n+1, 2)
    d2 = jnp.sum(r * r, axis=-1)
    K = jnp.exp(-d2 / (2.0 * sigma * sigma)) / (2.0 * jnp.pi * sigma * sigma)
    neighbor_contrib = (scatter_value / (sigma * sigma)) * jnp.sum(
        (K * valid)[..., None] * r, axis=1
    )                                                              # (N, 2)

    corrected = raw - neighbor_contrib
    if nodes.active is not None:
        corrected = corrected * nodes.active[:, None]
    return fields, corrected


def update_steric_potential(
    nodes: Nodes,
    fields: Fields,
    sigma: float = 12.5,
    scatter_value: float = 8.0,
) -> Fields:
    """Build a smoothed density field via bilinear (Cloud-In-Cell) deposition +
    Gaussian filter. Bilinear scatter matches the bilinear sampling in force_interp
    so there is no sub-cell discretization bias. Total mass per node = `scatter_value`.
    """
    steric = jnp.zeros_like(fields.steric)
    pos = periodic_boundary(fields.grid_shape, nodes.position)
    H, W = fields.grid_shape

    x = pos[..., 0]
    y = pos[..., 1]
    x0 = jnp.floor(x).astype(jnp.int32) % W
    y0 = jnp.floor(y).astype(jnp.int32) % H
    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H
    dx = x - jnp.floor(x)
    dy = y - jnp.floor(y)

    w00 = (1.0 - dx) * (1.0 - dy) * scatter_value
    w10 = dx * (1.0 - dy) * scatter_value
    w01 = (1.0 - dx) * dy * scatter_value
    w11 = dx * dy * scatter_value
    if nodes.active is not None:
        activity = nodes.active.astype(pos.dtype)
        w00 = w00 * activity
        w10 = w10 * activity
        w01 = w01 * activity
        w11 = w11 * activity

    steric = steric.at[y0, x0].add(w00)
    steric = steric.at[y0, x1].add(w10)
    steric = steric.at[y1, x0].add(w01)
    steric = steric.at[y1, x1].add(w11)

    steric = gaussian_filter(steric, sigma=sigma)
    return replace(fields, steric=steric)
