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
    MutationProfile,
    build_tensorneat_genome,
    canonicalize_neutral_attributes,
)


CLONE = 0
PARAMETRIC = 1
STRUCTURAL = 2
MIXED = 3
NUM_OPERATORS = 4

# Versioned r4 action space.  The legacy four-action ABI above remains the
# default so frozen r1-r3 code can still be replayed without reinterpretation.
R4_CLONE = 0
R4_CONSERVATIVE_PARAMETRIC = 1
R4_STANDARD_PARAMETRIC = 2
R4_EXPLORATORY_PARAMETRIC = 3
R4_STRUCTURAL = 4
R4_MIXED = 5
R4_NUM_OPERATORS = 6
R4_LOGIT_LIMIT = 8.0

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
    data_fields=[
        "energy_fraction",
        "intake_ema",
        "age_fraction",
        "node_fraction",
        "connection_fraction",
    ],
)
@dataclass
class R4ParentStats:
    energy_fraction: jax.Array
    intake_ema: jax.Array
    age_fraction: jax.Array
    node_fraction: jax.Array
    connection_fraction: jax.Array


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "alive_fraction",
        "population_change_ema",
        "mean_energy_fraction",
        "birth_rate_ema",
        "death_rate_ema",
        "mean_intake_ema",
        "operator_success_ema",
        "operator_usage_ema",
        "operator_evidence_ema",
    ],
)
@dataclass
class R4PopulationStats:
    alive_fraction: jax.Array
    population_change_ema: jax.Array
    mean_energy_fraction: jax.Array
    birth_rate_ema: jax.Array
    death_rate_ema: jax.Array
    mean_intake_ema: jax.Array
    operator_success_ema: jax.Array
    operator_usage_ema: jax.Array
    operator_evidence_ema: jax.Array


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


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=[],
    data_fields=[
        "genome",
        "policy_valid",
        "operator_index",
        "selection_probabilities",
    ],
)
@dataclass
class R4OffspringResult:
    genome: CPPNGenome
    policy_valid: jax.Array
    operator_index: jax.Array
    selection_probabilities: jax.Array


OffspringPolicy = Callable[
    [CPPNGenome, ParentStats, PopulationStats, MutationContext],
    OffspringResult,
]

R4OffspringPolicy = Callable[
    [CPPNGenome, R4ParentStats, R4PopulationStats, MutationContext],
    R4OffspringResult,
]

R4LogitPolicy = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    jax.Array,
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

CONSERVATIVE_PROFILE = MutationProfile(0.10, 0.075, 0.0075, 0.05, 0.20, 0.10)
EXPLORATORY_PROFILE = MutationProfile(0.40, 0.30, 0.03, 0.20, 0.20, 0.10)
CONSERVATIVE_GENOME = build_tensorneat_genome(
    value_mutation=True,
    structural_mutation=False,
    mutation_profile=CONSERVATIVE_PROFILE,
)
CONSERVATIVE_STATE = CONSERVATIVE_GENOME.setup(State())
EXPLORATORY_GENOME = build_tensorneat_genome(
    value_mutation=True,
    structural_mutation=False,
    mutation_profile=EXPLORATORY_PROFILE,
)
EXPLORATORY_STATE = EXPLORATORY_GENOME.setup(State())


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


def _profiled_parametric_mutation(
    parent: CPPNGenome,
    context: MutationContext,
    genome,
    state,
) -> CPPNGenome:
    nodes, connections = genome.mutation.mutate_values(
        state,
        genome,
        context.key,
        parent.node_genes,
        parent.connection_genes,
    )
    return canonicalize_neutral_attributes(CPPNGenome(nodes, connections))


def mutate_cppn_r4(
    parent: CPPNGenome,
    operator_logits: jax.Array,
    context: MutationContext,
) -> R4OffspringResult:
    """Select one of six trusted operators under the frozen r4 ABI."""
    operator_logits = jnp.asarray(operator_logits)
    if operator_logits.shape != (R4_NUM_OPERATORS,):
        raise ValueError("operator_logits must have shape (6,)")
    if not jnp.issubdtype(operator_logits.dtype, jnp.floating):
        raise TypeError("operator_logits must have a floating dtype")
    policy_valid = jnp.all(jnp.isfinite(operator_logits))

    def valid_policy(_):
        logits = jnp.clip(operator_logits.astype(jnp.float32), -R4_LOGIT_LIMIT, R4_LOGIT_LIMIT)
        at_upper = logits == R4_LOGIT_LIMIT
        exact = (jnp.sum(at_upper) == 1) & jnp.all(jnp.where(at_upper, True, logits == -R4_LOGIT_LIMIT))
        selection_key, mutation_key = jax.random.split(context.key)
        categorical = jax.random.categorical(selection_key, logits).astype(jnp.int32)
        operator = jnp.where(exact, jnp.argmax(logits).astype(jnp.int32), categorical)
        probabilities = jnp.where(
            exact,
            jax.nn.one_hot(operator, R4_NUM_OPERATORS, dtype=jnp.float32),
            jax.nn.softmax(logits),
        )
        mutation_context = MutationContext(
            key=mutation_key,
            new_node_key=context.new_node_key,
        )

        def clone(_):
            return parent

        def conservative(_):
            return _profiled_parametric_mutation(
                parent, mutation_context, CONSERVATIVE_GENOME, CONSERVATIVE_STATE
            )

        def standard(_):
            return _parametric_mutation(parent, mutation_context)

        def exploratory(_):
            return _profiled_parametric_mutation(
                parent, mutation_context, EXPLORATORY_GENOME, EXPLORATORY_STATE
            )

        def structural(_):
            return _structural_mutation(parent, mutation_context)

        def mixed(_):
            return _mixed_mutation(parent, mutation_context)

        child = jax.lax.switch(
            operator,
            (clone, conservative, standard, exploratory, structural, mixed),
            operand=None,
        )
        return R4OffspringResult(child, jnp.array(True), operator, probabilities)

    def invalid_policy(_):
        return R4OffspringResult(
            parent,
            jnp.array(False),
            jnp.array(R4_CLONE, dtype=jnp.int32),
            jax.nn.one_hot(R4_CLONE, R4_NUM_OPERATORS, dtype=jnp.float32),
        )

    return jax.lax.cond(policy_valid, valid_policy, invalid_policy, operand=None)


def fixed_r4_policy(operator: int) -> R4OffspringPolicy:
    """Return a trusted deterministic r4 policy for one registered action."""
    if not isinstance(operator, int) or isinstance(operator, bool) or not 0 <= operator < R4_NUM_OPERATORS:
        raise ValueError("operator must index the r4 action registry")
    logits = jnp.full(R4_NUM_OPERATORS, -R4_LOGIT_LIMIT, dtype=jnp.float32).at[operator].set(R4_LOGIT_LIMIT)

    def policy(parent_genome, parent_stats, population_stats, mutation_context):
        del parent_stats, population_stats
        return mutate_cppn_r4(parent_genome, logits, mutation_context)

    return policy


def make_r4_offspring_policy(logit_policy: R4LogitPolicy) -> R4OffspringPolicy:
    """Adapt the public five-array ABI to the trusted six-operator registry.

    The candidate receives only bounded ecological summaries and an opaque
    sentinel.  Genome bytes and mutation keys remain inside trusted code.
    """

    def policy(parent_genome, parent_stats, population_stats, mutation_context):
        safe_genome = jnp.clip(
            jnp.stack(
                [parent_stats.node_fraction, parent_stats.connection_fraction]
            ),
            0.0,
            1.0,
        )
        safe_parent = jnp.clip(
            jnp.stack(
                [
                    parent_stats.energy_fraction,
                    parent_stats.intake_ema,
                    parent_stats.age_fraction,
                ]
            ),
            0.0,
            1.0,
        )
        safe_population = jnp.stack(
            [
                jnp.clip(population_stats.alive_fraction, 0.0, 1.0),
                jnp.clip(population_stats.mean_energy_fraction, 0.0, 1.0),
                jnp.clip(population_stats.population_change_ema, -1.0, 1.0),
                jnp.clip(population_stats.birth_rate_ema, 0.0, 1.0),
                jnp.clip(population_stats.death_rate_ema, 0.0, 1.0),
                jnp.clip(population_stats.mean_intake_ema, 0.0, 1.0),
            ]
        )
        safe_operator = jnp.clip(
            jnp.stack(
                [
                    population_stats.operator_success_ema,
                    population_stats.operator_usage_ema,
                    population_stats.operator_evidence_ema,
                ]
            ),
            0.0,
            1.0,
        )
        logits = jnp.asarray(
            logit_policy(
                safe_genome,
                safe_parent,
                safe_population,
                safe_operator,
                jnp.array(0.0, dtype=jnp.float32),
            )
        )
        if logits.shape != (R4_NUM_OPERATORS,):
            raise ValueError("r4 candidate logits must have shape (6,)")
        if logits.dtype != jnp.float32:
            raise TypeError("r4 candidate logits must have dtype float32")
        return mutate_cppn_r4(parent_genome, logits, mutation_context)

    return policy


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


def r4_parent_statistics(
    genome: CPPNGenome,
    energy: jax.Array,
    intake_ema: jax.Array,
    age: jax.Array,
    reproduction_threshold: float,
    maximum_lifespan: int,
) -> R4ParentStats:
    """Compute the five frozen r4 parent/genome summary inputs."""
    nodes = jnp.sum(~jnp.isnan(genome.node_genes[..., 0]), axis=-1)
    connections = jnp.sum(~jnp.isnan(genome.connection_genes[..., 0]), axis=-1)
    return R4ParentStats(
        energy_fraction=jnp.clip(energy / jnp.maximum(reproduction_threshold, 1e-8), 0.0, 2.0),
        intake_ema=jnp.clip(intake_ema, 0.0, 1.0),
        age_fraction=jnp.clip(age.astype(jnp.float32) / max(maximum_lifespan, 1), 0.0, 1.0),
        node_fraction=nodes.astype(jnp.float32) / MAX_NODES,
        connection_fraction=connections.astype(jnp.float32) / MAX_CONNECTIONS,
    )


def r4_population_statistics(
    alive: jax.Array,
    energy: jax.Array,
    intake_ema: jax.Array,
    population_change_for_birth: jax.Array,
    reproduction_threshold: float,
    birth_rate_ema: jax.Array,
    death_rate_ema: jax.Array,
    operator_success_ema: jax.Array,
    operator_usage_ema: jax.Array,
    operator_evidence_ema: jax.Array,
) -> R4PopulationStats:
    """Compute the frozen six population and three-by-six credit inputs."""
    live = alive.astype(jnp.float32)
    count = jnp.maximum(jnp.sum(live), 1.0)
    return R4PopulationStats(
        alive_fraction=jnp.mean(live),
        population_change_ema=jnp.clip(population_change_for_birth, -1.0, 1.0),
        mean_energy_fraction=jnp.clip(
            jnp.sum(energy * live) / count / jnp.maximum(reproduction_threshold, 1e-8),
            0.0,
            2.0,
        ),
        birth_rate_ema=jnp.clip(birth_rate_ema, 0.0, 1.0),
        death_rate_ema=jnp.clip(death_rate_ema, 0.0, 1.0),
        mean_intake_ema=jnp.clip(jnp.sum(intake_ema * live) / count, 0.0, 1.0),
        operator_success_ema=jnp.clip(operator_success_ema, 0.0, 1.0),
        operator_usage_ema=jnp.clip(operator_usage_ema, 0.0, 1.0),
        operator_evidence_ema=jnp.clip(operator_evidence_ema, 0.0, 1.0),
    )
