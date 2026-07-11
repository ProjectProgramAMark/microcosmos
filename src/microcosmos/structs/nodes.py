from dataclasses import dataclass
import functools
import jax

@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "position",
        "velocity",
        "color_bend",
        "debug_vector",
        "active",
        "component_id",
    ],
)
@dataclass
class Nodes:
    position: jax.Array  # (N, D)  N = num_nodes, D = 2
    velocity: jax.Array  # (N, D)
    color_bend: jax.Array  # (N,) — for rendering
    debug_vector: jax.Array  # (N, D)
    # Optional fixed-capacity metadata. ``None`` retains the legacy fast path.
    active: jax.Array | None = None  # (N,) bool
    component_id: jax.Array | None = None  # (N,) int32

    def __getitem__(self, idx: int | slice) -> "Nodes":
        return Nodes(
            position=self.position[idx],
            velocity=self.velocity[idx],
            color_bend=self.color_bend[idx],
            debug_vector=self.debug_vector[idx],
            active=None if self.active is None else self.active[idx],
            component_id=None if self.component_id is None else self.component_id[idx],
        )

    def __len__(self) -> int:
        return len(self.position)
