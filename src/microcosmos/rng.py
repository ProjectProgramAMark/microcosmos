"""Named, identity-stable JAX random streams."""

from enum import IntEnum

import jax
import jax.numpy as jnp


class RNGTag(IntEnum):
    """Frozen top-level stream tags; values are part of replay semantics."""

    INITIALIZATION = 1
    ENVIRONMENT = 2
    MUTATION = 3
    SPAWN = 4
    FUTURE_EVENT = 5


def derive_key(
    base_key: jax.Array,
    tag: RNGTag,
    *components: int | jax.Array,
) -> jax.Array:
    """Derive a key without consuming or sequencing any sibling stream."""
    key = jax.random.fold_in(base_key, int(tag))
    for component in components:
        key = jax.random.fold_in(key, jnp.asarray(component, dtype=jnp.uint32))
    return key


def keys_for_identities(
    base_key: jax.Array,
    tag: RNGTag,
    timestep: int | jax.Array,
    identities: jax.Array,
) -> jax.Array:
    """Derive one draw key per immutable identity, independent of array rank."""
    stream_key = derive_key(base_key, tag, timestep)
    return jax.vmap(
        lambda identity: jax.random.fold_in(
            stream_key, jnp.asarray(identity, dtype=jnp.uint32)
        )
    )(identities)
