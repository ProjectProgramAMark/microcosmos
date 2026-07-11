import jax.numpy as jnp
import numpy as np
import pytest

from microcosmos.forces import compute_field_steric_force_corrected
from microcosmos.graph import make_fields
from microcosmos.solver.masks import build_replicated_physics_context
from microcosmos.structs.nodes import Nodes


def _neighborhoods(context):
    indices = np.asarray(context.steric_exclusion_indices)
    valid = np.asarray(context.steric_exclusion_valid)
    return [set(row[mask].tolist()) for row, mask in zip(indices, valid, strict=True)]


@pytest.mark.parametrize(
    ("pairs", "num_nodes", "distance", "expected"),
    [
        (
            [[0, 1], [1, 2], [2, 3]],
            4,
            1,
            [{0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3}],
        ),
        (
            [[0, 1], [1, 2], [2, 3], [3, 0]],
            4,
            1,
            [{0, 1, 3}, {0, 1, 2}, {1, 2, 3}, {0, 2, 3}],
        ),
        (
            [[0, 1], [0, 2], [0, 3], [3, 4]],
            5,
            2,
            [
                {0, 1, 2, 3, 4},
                {0, 1, 2, 3},
                {0, 1, 2, 3},
                {0, 1, 2, 3, 4},
                {0, 3, 4},
            ],
        ),
    ],
)
def test_context_uses_exact_graph_distance(pairs, num_nodes, distance, expected):
    context = build_replicated_physics_context(
        pairs, num_nodes, num_slots=1, neighbor_distance=distance
    )
    assert _neighborhoods(context) == expected


def test_context_replicates_neighborhoods_without_cross_slot_links():
    context = build_replicated_physics_context(
        [[0, 1], [1, 2]], 3, num_slots=2, neighbor_distance=1
    )
    assert _neighborhoods(context) == [
        {0, 1},
        {0, 1, 2},
        {1, 2},
        {3, 4},
        {3, 4, 5},
        {4, 5},
    ]


def test_topology_aware_steric_force_is_node_permutation_invariant():
    pairs = np.array([[0, 1], [1, 2], [1, 3], [3, 4]], dtype=np.int32)
    positions = jnp.array(
        [[4.2, 5.1], [5.4, 5.3], [6.1, 4.0], [5.8, 6.4], [7.0, 7.1]],
        dtype=jnp.float32,
    )
    permutation = np.array([3, 0, 4, 1, 2], dtype=np.int32)
    inverse = np.argsort(permutation)
    permuted_pairs = inverse[pairs]

    original_context = build_replicated_physics_context(
        pairs, len(positions), num_slots=1, neighbor_distance=1
    )
    permuted_context = build_replicated_physics_context(
        permuted_pairs, len(positions), num_slots=1, neighbor_distance=1
    )
    original_neighborhoods = _neighborhoods(original_context)
    mapped_neighborhoods = [
        {int(permutation[index]) for index in neighborhood}
        for neighborhood in _neighborhoods(permuted_context)
    ]
    restored_order = [set() for _ in range(len(positions))]
    for new_index, old_index in enumerate(permutation):
        restored_order[int(old_index)] = mapped_neighborhoods[new_index]
    assert restored_order == original_neighborhoods

    def force(node_positions, topology_pairs):
        nodes = Nodes(
            position=node_positions,
            velocity=jnp.zeros_like(node_positions),
            color_bend=jnp.zeros(node_positions.shape[0]),
            debug_vector=jnp.zeros_like(node_positions),
        )
        context = build_replicated_physics_context(
            topology_pairs,
            node_positions.shape[0],
            num_slots=1,
            neighbor_distance=1,
        )
        fields, result = compute_field_steric_force_corrected(
            nodes,
            make_fields((16, 16)),
            sigma=1.5,
            neighbor_skip=1,
            physics_context=context,
        )
        return fields.steric, result

    steric, force_original = force(positions, pairs)
    steric_permuted, force_permuted = force(positions[permutation], permuted_pairs)
    assert jnp.allclose(steric, steric_permuted, atol=1e-6)
    assert jnp.allclose(force_original, force_permuted[inverse], atol=1e-6)


def test_steric_force_keeps_legacy_no_context_path():
    positions = jnp.array([[4.0, 4.0], [5.0, 4.0], [6.0, 4.0]])
    nodes = Nodes(
        position=positions,
        velocity=jnp.zeros_like(positions),
        color_bend=jnp.zeros(3),
        debug_vector=jnp.zeros_like(positions),
        component_id=jnp.zeros(3, dtype=jnp.int32),
    )
    fields = make_fields((12, 12))
    result_omitted = compute_field_steric_force_corrected(
        nodes, fields, sigma=1.5, neighbor_skip=1
    )[1]
    result_explicit = compute_field_steric_force_corrected(
        nodes, fields, sigma=1.5, neighbor_skip=1, physics_context=None
    )[1]
    assert jnp.all(jnp.isfinite(result_omitted))
    assert jnp.allclose(result_omitted, result_explicit, atol=0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_nodes": 0},
        {"num_slots": 0},
        {"neighbor_distance": -1},
    ],
)
def test_context_rejects_invalid_dimensions(kwargs):
    arguments = {
        "local_pairs": np.empty((0, 2), dtype=np.int32),
        "num_nodes": 1,
        "num_slots": 1,
        "neighbor_distance": 0,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        build_replicated_physics_context(**arguments)
