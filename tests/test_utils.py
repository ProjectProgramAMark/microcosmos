import jax.numpy as jnp
from microcosmos.utils import (
    displacement,
    gaussian_filter,
    index_from_position,
    norm,
    periodic_boundary,
)



def test_norm():
    v = jnp.array([[3.0, 4.0], [0.0, 0.0]])
    n = norm(v, axis=-1, keepdims=False)
    # b ecause of the EPSIL added for stability, the second value should not be exactly zero
    assert jnp.allclose(n[0], 5.0)
    assert n[1] > 0  # Just verify it's positive and non-zero

    n_keepdims = norm(v, axis=-1, keepdims=True)
    assert n_keepdims.shape == (2, 1)
    assert jnp.allclose(n_keepdims[0], 5.0)
    assert n_keepdims[1] > 0  # Second element is sqrt(2)*EPSIL


def test_displacement():
    size = jnp.array([10.0, 10.0])
    x = jnp.array([1.0, 1.0])
    y = jnp.array([9.0, 9.0])
    
    # Shortest path should be crossing the boundary, so +2.0
    d = displacement(size, x, y)
    assert jnp.allclose(d, jnp.array([2.0, 2.0]))

    # Simple case without boundary crossing
    x2 = jnp.array([5.0, 5.0])
    y2 = jnp.array([4.0, 4.0])
    d2 = displacement(size, x2, y2)
    assert jnp.allclose(d2, jnp.array([1.0, 1.0]))


def test_index_from_position():
    pos = jnp.array([[1.2, 3.8], [5.5, 0.1]])
    # floor -> [1, 3], [5, 0]
    # flip -> [3, 1], [0, 5]
    # T -> [[3, 0], [1, 5]]
    idx_y, idx_x = index_from_position(pos)
    
    assert jnp.allclose(idx_y, jnp.array([3, 0]))
    assert jnp.allclose(idx_x, jnp.array([1, 5]))


def test_periodic_boundary():
    size = jnp.array([10.0, 10.0])
    pos = jnp.array([-1.0, 11.0])
    wrapped = periodic_boundary(size, pos)
    assert jnp.allclose(wrapped, jnp.array([9.0, 1.0]))


def test_gaussian_filter():
    data = jnp.zeros((10, 10))
    data = data.at[5, 5].set(1.0)
    filtered = gaussian_filter(data, sigma=1.0)
    
    # Peak should still be at 5,5
    assert jnp.argmax(filtered) == 55 # flattened index
    # Should be symmetric
    assert jnp.allclose(filtered[5, 4], filtered[5, 6])
    assert jnp.allclose(filtered[4, 5], filtered[6, 5])
