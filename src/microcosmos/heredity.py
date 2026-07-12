"""Trusted CPPN heredity profiles and ecology-facing policy statistics."""

from dataclasses import dataclass
import functools
from typing import Callable

import jax
import jax.numpy as jnp
from tensorneat.common import State

from microcosmos.cppn import (
    DYNAMIC_NODE_KEY_OFFSET,
    MAX_CONNECTIONS,
    MAX_EXACT_FLOAT32_INTEGER,
    MAX_NODES,
    CPPNGenome,
    build_tensorneat_genome,
    canonicalize_neutral_attributes,
)


CLONE = 0
PARAMETRIC = 1
STRUCTURAL = 2
MIXED = 3
NUM_OPERATORS = 4

OPERATOR_SCORE_LIMIT = 20.0
INTAKE_EMA_ALPHA = 0.05
POPULATION_EMA_ALPHA = 0.10


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "energy_fraction",
        "intake_ema",
        "node_fraction",
        "connection_fraction",
    ],
)
@dataclass
class ParentStats:
    energy_fraction: jax.Array
    intake_ema: jax.Array
    node_fraction: jax.Array
    connection_fraction: jax.Array


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "alive_fraction",
        "population_change_ema",
        "action_diversity",
        "lineage_entropy",
    ],
)
@dataclass
class PopulationStats:
    alive_fraction: jax.Array
    population_change_ema: jax.Array
    action_diversity: jax.Array
    lineage_entropy: jax.Array


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=["key", "new_node_key"],
)
@dataclass
class MutationContext:
    key: jax.Array
    new_node_key: jax.Array


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=["genome", "policy_valid", "operator_index"],
)
@dataclass
class OffspringResult:
    genome: CPPNGenome
    policy_valid: jax.Array
    operator_index: jax.Array


OffspringPolicy = Callable[
    [CPPNGenome, ParentStats, PopulationStats, MutationContext],
    OffspringResult,
]


PARAMETRIC_GENOME = build_tensorneat_genome(
    value_mutation=True,
    structural_mutation=False,
)
PARAMETRIC_STATE = PARAMETRIC_GENOME.setup(State())
STRUCTURAL_GENOME = build_tensorneat_genome(
    value_mutation=False,
    structural_mutation=True,
)
STRUCTURAL_STATE = STRUCTURAL_GENOME.setup(State())
MIXED_GENOME = build_tensorneat_genome(
    value_mutation=True,
    structural_mutation=True,
)
MIXED_STATE = MIXED_GENOME.setup(State())


def make_mutation_context(
    key: jax.Array,
    child_id: jax.Array,
) -> tuple[MutationContext, jax.Array]:
    """Construct identity-stable TensorNEAT keys and report exact-key validity."""
    new_node_key = jnp.asarray(child_id, dtype=jnp.int32) + DYNAMIC_NODE_KEY_OFFSET
    valid = (new_node_key >= DYNAMIC_NODE_KEY_OFFSET) & (new_node_key < MAX_EXACT_FLOAT32_INTEGER)
    return (
        MutationContext(
            key=key,
            new_node_key=new_node_key.astype(jnp.float32),
        ),
        valid,
    )


def _parametric_mutation(
    parent: CPPNGenome,
    context: MutationContext,
) -> CPPNGenome:
    nodes, connections = PARAMETRIC_GENOME.mutation.mutate_values(
        PARAMETRIC_STATE,
        PARAMETRIC_GENOME,
        context.key,
        parent.node_genes,
        parent.connection_genes,
    )
    return canonicalize_neutral_attributes(CPPNGenome(nodes, connections))


def _structural_mutation(
    parent: CPPNGenome,
    context: MutationContext,
) -> CPPNGenome:
    nodes, connections = STRUCTURAL_GENOME.mutation.mutate_structure(
        STRUCTURAL_STATE,
        STRUCTURAL_GENOME,
        context.key,
        parent.node_genes,
        parent.connection_genes,
        context.new_node_key,
        jnp.zeros((3,), dtype=jnp.float32),
    )
    return canonicalize_neutral_attributes(CPPNGenome(nodes, connections))


def _mixed_mutation(
    parent: CPPNGenome,
    context: MutationContext,
) -> CPPNGenome:
    nodes, connections = MIXED_GENOME.execute_mutation(
        MIXED_STATE,
        context.key,
        parent.node_genes,
        parent.connection_genes,
        context.new_node_key,
        jnp.zeros((3,), dtype=jnp.float32),
    )
    return canonicalize_neutral_attributes(CPPNGenome(nodes, connections))


def mutate_cppn(
    parent: CPPNGenome,
    operator_scores: jax.Array,
    context: MutationContext,
) -> OffspringResult:
    """Select one trusted TensorNEAT heredity profile."""
    operator_scores = jnp.asarray(operator_scores)
    if operator_scores.shape != (NUM_OPERATORS,):
        raise ValueError("operator_scores must have shape (4,)")
    if not jnp.issubdtype(operator_scores.dtype, jnp.floating):
        raise TypeError("operator_scores must have a floating dtype")

    policy_valid = jnp.all(jnp.isfinite(operator_scores))

    def valid_policy(_):
        scores = jnp.clip(
            operator_scores,
            -OPERATOR_SCORE_LIMIT,
            OPERATOR_SCORE_LIMIT,
        )
        operator = jnp.argmax(scores).astype(jnp.int32)

        def clone(_):
            return parent

        def parametric(_):
            return _parametric_mutation(parent, context)

        def structural(_):
            return _structural_mutation(parent, context)

        def mixed(_):
            return _mixed_mutation(parent, context)

        child = jax.lax.switch(
            operator,
            (clone, parametric, structural, mixed),
            operand=None,
        )
        return OffspringResult(child, jnp.array(True), operator)

    def invalid_policy(_):
        return OffspringResult(
            parent,
            jnp.array(False),
            jnp.array(CLONE, dtype=jnp.int32),
        )

    return jax.lax.cond(
        policy_valid,
        valid_policy,
        invalid_policy,
        operand=None,
    )


def clone_policy(
    parent_genome: CPPNGenome,
    parent_stats: ParentStats,
    population_stats: PopulationStats,
    mutation_context: MutationContext,
) -> OffspringResult:
    """Create an exact clone through the common trusted kernel."""
    del parent_stats, population_stats
    return mutate_cppn(
        parent_genome,
        jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        mutation_context,
    )


def fixed_parametric_policy(
    parent_genome: CPPNGenome,
    parent_stats: ParentStats,
    population_stats: PopulationStats,
    mutation_context: MutationContext,
) -> OffspringResult:
    """Always use parameter-and-activation mutation."""
    del parent_stats, population_stats
    return mutate_cppn(
        parent_genome,
        jnp.array([0.0, 1.0, 0.0, 0.0], dtype=jnp.float32),
        mutation_context,
    )


def fixed_structural_policy(
    parent_genome: CPPNGenome,
    parent_stats: ParentStats,
    population_stats: PopulationStats,
    mutation_context: MutationContext,
) -> OffspringResult:
    """Always use topology-only mutation."""
    del parent_stats, population_stats
    return mutate_cppn(
        parent_genome,
        jnp.array([0.0, 0.0, 1.0, 0.0], dtype=jnp.float32),
        mutation_context,
    )


def fixed_mixed_policy(
    parent_genome: CPPNGenome,
    parent_stats: ParentStats,
    population_stats: PopulationStats,
    mutation_context: MutationContext,
) -> OffspringResult:
    """Always use the conventional mixed TensorNEAT profile."""
    del parent_stats, population_stats
    return mutate_cppn(
        parent_genome,
        jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float32),
        mutation_context,
    )


def stress_responsive_policy(
    parent_genome: CPPNGenome,
    parent_stats: ParentStats,
    population_stats: PopulationStats,
    mutation_context: MutationContext,
) -> OffspringResult:
    """Frozen human baseline that increases exploration under decline."""
    stress = jnp.clip(
        0.45 * (1.0 - population_stats.alive_fraction)
        + 0.35 * jnp.maximum(0.0, -population_stats.population_change_ema)
        + 0.20 * (1.0 - parent_stats.intake_ema),
        0.0,
        1.0,
    )
    scores = jnp.array(
        [
            0.25 - stress,
            0.20 - jnp.abs(stress - 0.35),
            stress - 0.75,
            0.20 - jnp.abs(stress - 0.65),
        ],
        dtype=jnp.float32,
    )
    return mutate_cppn(parent_genome, scores, mutation_context)


def update_intake_ema(
    intake_ema: jax.Array,
    actual_uptake: jax.Array,
    uptake_rate: jax.Array,
    alive: jax.Array,
    dt: float,
    eps: float,
) -> jax.Array:
    """Update fulfilled-uptake EMA for organisms alive at step start."""
    fraction = jnp.clip(
        actual_uptake / jnp.maximum(uptake_rate * dt, eps),
        0.0,
        1.0,
    )
    updated = (1.0 - INTAKE_EMA_ALPHA) * intake_ema + INTAKE_EMA_ALPHA * fraction
    return jnp.where(alive, updated, intake_ema)


def update_population_change_ema(
    previous: jax.Array,
    population_delta: jax.Array,
    capacity: int,
) -> jax.Array:
    """Update the normalized net-population-change EMA once."""
    normalized_delta = population_delta.astype(jnp.float32) / capacity
    return jnp.clip(
        (1.0 - POPULATION_EMA_ALPHA) * previous + POPULATION_EMA_ALPHA * normalized_delta,
        -1.0,
        1.0,
    )


def action_diversity(
    actions: jax.Array,
    alive: jax.Array,
    max_bending_delta: float,
) -> jax.Array:
    """Return survivor-masked per-hinge controller variance in [0, 1]."""
    if actions.shape[1] == 0:
        return jnp.zeros((), dtype=actions.dtype)
    count = jnp.sum(alive).astype(actions.dtype)
    safe_count = jnp.maximum(count, 1.0)
    mean = jnp.sum(actions * alive[:, None], axis=0) / safe_count
    variance = (
        jnp.sum(
            (actions - mean[None, :]) ** 2 * alive[:, None],
            axis=0,
        )
        / safe_count
    )
    normalized = jnp.mean(variance) / jnp.maximum(
        jnp.asarray(max_bending_delta, dtype=actions.dtype) ** 2,
        1e-12,
    )
    return jnp.where(count >= 2.0, jnp.clip(normalized, 0.0, 1.0), 0.0)


def lineage_entropy(
    founder_lineage_id: jax.Array,
    alive: jax.Array,
    initial_founder_count: int,
) -> jax.Array:
    """Return living-founder entropy normalized by the frozen founder count."""
    if initial_founder_count <= 1:
        return jnp.zeros((), dtype=jnp.float32)
    founder_ids = jnp.arange(initial_founder_count, dtype=jnp.int32)
    counts = jnp.sum(
        alive[:, None] & (founder_lineage_id[:, None] == founder_ids[None, :]),
        axis=0,
    ).astype(jnp.float32)
    total = jnp.sum(counts)
    probabilities = counts / jnp.maximum(total, 1.0)
    entropy = -jnp.sum(jnp.where(probabilities > 0.0, probabilities * jnp.log(probabilities), 0.0))
    return jnp.where(
        total > 0.0,
        jnp.clip(entropy / math_log(initial_founder_count), 0.0, 1.0),
        0.0,
    )


def math_log(value: int) -> jax.Array:
    """Return a float32 natural log without introducing host values in JIT."""
    return jnp.log(jnp.asarray(value, dtype=jnp.float32))


def parent_statistics(
    genome: CPPNGenome,
    energy: jax.Array,
    intake_ema: jax.Array,
    reproduction_threshold: float,
) -> ParentStats:
    """Compute normalized per-slot statistics exposed to heredity."""
    nodes = jnp.sum(~jnp.isnan(genome.node_genes[..., 0]), axis=-1)
    connections = jnp.sum(~jnp.isnan(genome.connection_genes[..., 0]), axis=-1)
    return ParentStats(
        energy_fraction=jnp.clip(
            energy / jnp.maximum(reproduction_threshold, 1e-8),
            0.0,
            2.0,
        ),
        intake_ema=jnp.clip(intake_ema, 0.0, 1.0),
        node_fraction=nodes.astype(jnp.float32) / MAX_NODES,
        connection_fraction=(connections.astype(jnp.float32) / MAX_CONNECTIONS),
    )


def population_statistics(
    alive: jax.Array,
    population_change_for_birth: jax.Array,
    controller_action_diversity: jax.Array,
    founder_lineage_id: jax.Array,
    initial_founder_count: int,
) -> PopulationStats:
    """Compute normalized population statistics exposed to heredity."""
    return PopulationStats(
        alive_fraction=jnp.mean(alive.astype(jnp.float32)),
        population_change_ema=jnp.clip(population_change_for_birth, -1.0, 1.0),
        action_diversity=jnp.clip(controller_action_diversity, 0.0, 1.0),
        lineage_entropy=lineage_entropy(
            founder_lineage_id,
            alive,
            initial_founder_count,
        ),
    )
