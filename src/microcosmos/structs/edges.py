from dataclasses import dataclass, field
import functools
import jax
import jax.numpy as jnp

@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=["pairs", "theta", "rest_lengths", "bending_pairs", "bending_rest_angles", "bending_stiffness"],
)

# pairs is the edge list (edge→nodes)
# bending_pairs is the hinge list (hinge→edges)
@dataclass
class Edges:
    pairs: jax.Array              # (E, 2) int32 — (node_src, node_tgt) indices. used in position pass
    theta: jax.Array              # (E,) — per-edge orientation angle
    rest_lengths: jax.Array       # (E,) — target distance (optimizable)
    bending_pairs: jax.Array      # (B, 2) int32 — (edge_in, edge_out) for bending constraints
    bending_rest_angles: jax.Array  # (B,) — target angular difference (optimizable)
    bending_stiffness: jax.Array  # (B,) — per-bending-pair stiffness (optimizable)
