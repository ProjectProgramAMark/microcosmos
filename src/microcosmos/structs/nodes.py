from dataclasses import dataclass
import functools
import jax

@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=["position", "velocity", "color_bend", "debug_vector"],
)
@dataclass
class Nodes:
    position: jax.Array  # (N, D)  N = num_nodes, D = 2
    velocity: jax.Array  # (N, D)
    color_bend: jax.Array  # (N,) — for rendering
    debug_vector: jax.Array  # (N, D)

    def __getitem__(self, idx: int | slice) -> "Nodes":
        return Nodes(
            position=self.position[idx],
            velocity=self.velocity[idx],
            color_bend=self.color_bend[idx],
            debug_vector=self.debug_vector[idx],
        )

    def __len__(self) -> int:
        return len(self.position)
