# Microcosmos Continuous Ecosystem Lifecycle

Technical implementation plan for adding birth, death, reproduction, heredity, and limited resources to Microcosmos.

- Status: lifecycle baseline implemented; controller/resource details superseded
- Prepared: 2026-07-10
- Repository: `microcosmos`
- Delivery model: dependency-ordered phases with a verifiable checkpoint after each phase

> This document records the original fixed-capacity lifecycle implementation.
> The controller, resource, RNG, and benchmark details are superseded by the
> lifecycle remediation PRD in `../../plans/`; the fixed-slot lifecycle and
> compatibility decisions remain the historical baseline.

## 1. Overview

Add a new fixed-capacity `EcosystemEnv` supporting limited renewable resources, internal energy, death, asexual reproduction, heredity, mutation, and lineage tracking.

The design preserves Microcosmos's GPU/JIT architecture by preallocating creature slots and activating or deactivating them with masks. Organisms share one fixed topology; the compact controller genome is inherited and mutated while ecological rates remain fixed configuration.

## 2. Problem statement

Microcosmos currently supports multiple creatures and energy-seeking fitness, but:

- Energy sampling does not consume the resource.
- Creature counts remain fixed during a rollout.
- Evolution occurs outside the simulated world.
- There is no organism energy balance, death, birth, or lineage.
- JAX cannot dynamically resize population arrays inside a compiled simulation.

## 3. Outcome hypothesis

If lifecycle events use fixed-capacity masked slots, Microcosmos can support continuous ecological evolution without abandoning GPU batching, JIT compilation, or existing differentiable physics.

## 4. Goals

- Support resource competition with numerically conservative depletion.
- Support starvation and age-based death.
- Support in-world asexual reproduction.
- Copy and mutate the normalized controller genome at birth.
- Maintain fixed array shapes as population size changes.
- Preserve all existing environment and simulator behavior.
- Produce deterministic lineage and ecosystem metrics.

## 5. Scope boundaries

### In scope

- One homogeneous `CreatureTopology` replicated across `max_creatures`.
- Fixed-capacity population with active and inactive slots.
- Renewable, capacity-limited resource grid.
- Internal energy, age, maturity, reproduction threshold, and lifespan.
- Asexual reproduction into available slots.
- Fixed-size heritable genome.
- Compact fixed-topology traveling-wave controller.
- Fixed configured uptake, assimilation, and metabolism parameters.
- Gaussian mutation with configured bounds.
- Individual IDs, parent IDs, and generations.
- PBD, stable Cosserat, steric, and fluid activity masks.
- Ecosystem rendering, metrics, experiment configuration, and benchmarks.

### Non-goals

- Adding or removing body nodes or edges through mutation.
- Heterogeneous topology within one ecosystem.
- Sexual reproduction or crossover.
- Differentiation through discrete birth and death events.
- Growth, development, predation, carcasses, disease, or terrain.
- NEAT speciation inside the simulated world.
- Claiming ecological balance without empirical parameter calibration.

## 6. Research snapshot

Date: 2026-07-10

- Microcosmos identifies large-scale open-ended evolution as a long-term goal in the [Microcosmos paper](https://arxiv.org/abs/2607.02954).
- JAX transformations require statically shaped arrays, making dynamic population resizing unsuitable for the compiled step. See the [JAX documentation](https://docs.jax.dev/en/latest/quickstart.html).
- [`simulate.py`](../src/microcosmos/simulate.py) compiles fixed-shape nodes and edges through `jax.lax.scan`.
- [`multi_agent.py`](../src/microcosmos/gym/multi_agent.py) constructs creature slices and topology before simulation.
- [`base.py`](../src/microcosmos/gym/base.py) samples a Gaussian energy source without depletion.
- `Fields.energy` already exists and remains available to the renderer when an experiment explicitly enables the energy overlay.
- Local verification uses the shared Python 3.13 Conda environment named `sakana`, including CUDA-enabled JAX on NVIDIA hosts.

## 7. Constraints and protected invariants

Do not change:

- Existing registered environments and their public behavior.
- Existing `simulate(nodes, edges, fields, ...)` callers.
- Existing physics results when activity metadata is absent.
- Existing differentiability tests for continuous physics.
- Periodic boundaries and current fluid solver semantics.
- Static node, edge, bending-pair, genome, and event-array shapes.
- The boundary between the core package and `experiments/`; core code must not import experiment code.

Additional constraints:

- Do not introduce new third-party dependencies.
- Do not add a feature flag; expose the new behavior through a separate environment.
- Do not perform lifecycle decisions on the Python host during a JIT rollout.

## 8. Assumptions and decisions

- Reproduction is asexual.
- Death immediately makes a body physically inert and invisible.
- A dead slot can receive a child at the end of the same step.
- A newborn first senses, acts, and consumes resources on the following step.
- Organisms dying during a step cannot reproduce during that step.
- Reproduction cost is charged only when a free slot is successfully assigned.
- Physics remains differentiable between lifecycle events; lifecycle decisions are discrete.
- Resource depletion uses nearest-cell accounting for exact conservation in v1.
- Resource regeneration is enabled; diffusion is configurable and defaults to zero.
- One parent may produce at most one child per step.

## 9. Planned files

### Core state and physics

- Modify `src/microcosmos/structs/nodes.py`.
- Add `src/microcosmos/solver/masks.py`.
- Modify `src/microcosmos/solver/pbd.py`.
- Modify `src/microcosmos/solver/stable_cosserat.py`.
- Modify `src/microcosmos/solver/fluid.py`.
- Modify `src/microcosmos/forces.py`.
- Add `src/microcosmos/structs/population.py`.
- Add `src/microcosmos/ecology.py`.
- Add `src/microcosmos/controller.py`.

### Environment and rendering

- Add `src/microcosmos/gym/ecosystem.py`.
- Modify `src/microcosmos/gym/registry.py`.
- Modify `src/microcosmos/gym/__init__.py`.
- Modify `src/microcosmos/rendering.py`.

### Experiment and verification

- Add `experiments/ecosystem/experiment.py`.
- Add `experiments/conf/experiment/ecosystem.yaml`.
- Modify `experiments/__init__.py`.
- Add `tests/test_activity_masks.py`.
- Add `tests/test_ecology.py`.
- Add `tests/test_ecosystem.py`.
- Add `benchmarks/benchmark_ecosystem.py`.

## 10. Quality gates

### Environment setup

```bash
conda run -n sakana python -c "import os, sys; assert os.path.basename(sys.prefix) == 'sakana'; print(sys.executable)"
```

### Required after every implementation story

```bash
conda run -n sakana ruff check .
conda run -n sakana env JAX_PLATFORMS=cpu pytest -q
```

### Focused ecosystem verification

```bash
conda run -n sakana env JAX_PLATFORMS=cpu pytest tests/test_activity_masks.py tests/test_ecology.py tests/test_ecosystem.py -q
```

### Smoke experiment

```bash
conda run -n sakana python experiments/main.py experiment=ecosystem \
  simulation.num_steps=200 \
  experiment.max_creatures=8 \
  experiment.initial_population=4 \
  physics.enable_fluid=false
```

### Performance verification

```bash
conda run -n sakana python benchmarks/benchmark_ecosystem.py \
  --steps 1000 \
  --max-creatures 64 \
  --nodes-per-creature 16 \
  --no-fluid
```

### Visual verification

- Dead and inactive slots are not rendered.
- New offspring appear at body-scale separation from their parents.
- Resource depletion and regeneration agree across metrics and plots.
- Population, birth, and death metrics agree with the event trace.
- No node appears at a clipped border because an inactive slot was rendered.
- The lifecycle animation uses the same native field renderer as other Microcosmos experiments.

## 11. Authoritative lifecycle order

```mermaid
flowchart LR
    A["Sense resource and energy"] --> B["Compute controller action"]
    B --> C["Run masked physics"]
    C --> D["Consume and regenerate resource"]
    D --> E["Update organism energy and age"]
    E --> F["Apply death"]
    F --> G["Allocate births and mutate genomes"]
    G --> H["Emit fixed-shape events"]
```

This ordering is part of the data contract. A creature that dies during step `t` cannot reproduce during step `t`. A child born at the end of step `t` first participates during step `t + 1`.

## 12. Implementation phases

### Phase 0: Establish the baseline

Objective: obtain a reproducible pre-change test and performance baseline.

Requirements:

- Install Python 3.13 and sync dependencies.
- Run the complete test suite.
- Record warm JIT step time for equal-node `MultiAgentEnv` cases with and without fluid.
- Verify the shared `sakana` environment and the intended JAX backend before substantial work.

Acceptance checkpoint:

- [ ] Full baseline test result is recorded.
- [ ] CPU or GPU device and timing methodology are recorded.
- [ ] The worktree remains clean.

Rollback: none; this phase changes no product code.

### Phase 1: Add optional activity metadata

Objective: represent inactive nodes without changing legacy callers.

Extend `Nodes` with optional fields:

```python
active: jax.Array | None = None        # (N,) bool
component_id: jax.Array | None = None  # (N,) int32
```

When `active is None`, all physics must follow the existing code path. `component_id` identifies a fixed creature slot, not a biological individual.

Add helpers returning node, edge, and bending activity masks with static shapes.

Acceptance checkpoint:

- [ ] Existing constructors remain valid without new arguments.
- [ ] Unmasked one-step physics matches the baseline within absolute tolerance `1e-7`.
- [ ] Mask arrays survive JIT and PyTree transformations.
- [ ] All existing tests pass.

Rollback: revert the optional fields and helper module.

### Phase 2: Make physics mask-aware

Objective: make inactive slots physically nonexistent.

Required behavior:

- PBD and stable Cosserat corrections ignore inactive edges and bending pairs.
- Inactive node positions remain unchanged and velocities remain zero.
- Steric deposition ignores inactive nodes.
- Steric neighbor subtraction only removes neighbors from the same `component_id`.
- Fluid IBM kernel weights are multiplied by node activity.
- Synthetic IBM nodes inherit their source node's activity.
- Inactive nodes receive no fluid reaction force.

Acceptance checkpoint:

- [ ] Inactive bodies do not move an active body in no-fluid tests.
- [ ] Adding inactive nodes does not change the active body's steric potential.
- [ ] Adding inactive nodes changes the fluid field by at most `1e-7`.
- [ ] Existing differentiability tests pass.

Rollback: revert Phase 2 while retaining the inert optional metadata from Phase 1.

### Phase 3: Add fixed-capacity ecosystem state

Objective: construct a homogeneous padded population.

Add `PopulationState`:

```python
alive: bool[C]
energy: float32[C]
age: int32[C]
generation: int32[C]
individual_id: int32[C]
parent_id: int32[C]
genome: float32[C, G]
next_individual_id: int32
```

Add `EcosystemState` containing:

- `nodes`
- `edges`
- `fields`
- `population`
- `time`
- Base rest lengths and bending angles

`EcosystemEnv` precomputes:

- `node_slot: int32[N]`
- `edge_slot: int32[E]`
- `bending_slot: int32[B]`
- Per-slot node, edge, and bending slices
- Local template positions and geometry

Register the environment as `"ecosystem"`.

Acceptance checkpoint:

- [ ] Reset activates exactly `initial_population` slots.
- [ ] All state shapes depend on capacity, never on live population.
- [ ] Reset is deterministic for the same PRNG key.
- [ ] Legacy environment registry entries are unchanged.

Rollback: remove the environment registration and new population files.

### Phase 4: Implement limited resources

Objective: make resource acquisition competitive and conservative.

Add the following core function:

```python
resource_step(
    resource,
    positions,
    node_active,
    node_slot,
    uptake_rate,
    dt,
    config,
) -> tuple[new_resource, gross_uptake_by_slot]
```

For each node:

1. Compute demand `d_i = active_i * uptake_rate[slot_i] * dt`.
2. Scatter demand into its nearest grid cell.
3. Compute the cell fulfillment ratio `f_c = min(1, resource_c / (demand_c + eps))`.
4. Give node `i` uptake `d_i * f_cell(i)`.
5. Subtract fulfilled demand from the field.
6. Regenerate each cell up to `resource_capacity`.

Acceptance checkpoint:

- [ ] Resources never fall below `-1e-6`.
- [ ] Resources never exceed capacity by more than `1e-6`.
- [ ] Pre-regeneration field loss equals gross uptake within relative error `1e-5`.
- [ ] Two organisms in one cell divide insufficient resource proportionally.
- [ ] Inactive nodes consume exactly zero resource.

Rollback: retain ecosystem state but restore an unchanged resource field.

### Phase 5: Implement energy and death

Objective: connect resource acquisition to survival.

Per step:

```text
new energy =
    old energy
    + gross uptake * assimilation efficiency
    - basal metabolism * dt
    - actuation cost
```

Death condition:

```text
energy <= 0 OR age >= maximum lifespan
```

On death:

- Set `alive=False`.
- Set all slot velocities to zero.
- Exclude the slot from subsequent physics and consumption.
- Preserve lineage fields until the slot is reused.
- Emit its individual ID in the fixed-size death event array.

Acceptance checkpoint:

- [ ] A configured starving organism dies on the expected step.
- [ ] A lifespan-limited organism dies at the exact configured age.
- [ ] Dead bodies remain stationary.
- [ ] Dead bodies consume no resources.
- [ ] One death event is emitted exactly once.

Rollback: disable mortality in `EcosystemEnv` while retaining resource accounting.

### Phase 6: Add the heritable controller and genome

Objective: make inherited parameters affect behavior without encoding metabolic shortcuts.

Use a normalized `float32[23]` genome containing:

- Three traveling-wave modules in loci `0:12`.
- Local-resource and head-tail-gradient polynomial corrections in `12:18`.
- Curvature bias and positive gain in `18:20`.
- Three action-neutral mutation-strategy loci in `20:23`.

Controller inputs per bending pair are normalized body coordinate, physical
time, local normalized resource, and fixed-order head/tail resource samples.
Energy is not a controller input.

Controller output:

- One bounded bending-angle delta.
- Rest-length actuation remains zero in v1.

Trusted development mutation:

- Independent Bernoulli mutation mask per gene.
- Gaussian noise for selected genes.
- Clip every gene to `[-1, 1]`.
- No structural mutation or crossover.

Acceptance checkpoint:

- [ ] The same genome, observation, and key produces the same action.
- [ ] Different genomes can produce different actions.
- [ ] Zero mutation standard deviation produces an exact copy.
- [ ] Mutated genes remain within bounds.
- [ ] Controller actions for inactive slots are exactly zero.

Rollback: use zero action while retaining genome storage.

### Phase 7: Add birth, reproduction, and lineage

Objective: create children during the continuous rollout.

Lifecycle allocation:

1. Apply energy and age updates.
2. Mark deaths.
3. Identify surviving, mature parents above the reproduction threshold.
4. Identify free slots after death.
5. Rank eligible parents by descending energy, with slot index as the tie-breaker.
6. Match eligible parents to free slots in ascending slot order.
7. Deduct reproduction cost only for successful assignments.
8. Initialize each child near its parent's center of mass.
9. Copy and mutate the parent genome.
10. Assign new individual, parent, and generation IDs.

Child initialization:

- `age = 0`
- `energy = offspring_initial_energy`
- `generation = parent.generation + 1`
- `parent_id = parent.individual_id`
- Zero velocity
- Base rest geometry
- Periodically wrapped spawn position
- Active at the end of the step

Acceptance checkpoint:

- [ ] An eligible parent with a free slot creates exactly one child.
- [ ] No free slot causes no birth and no energy charge.
- [ ] A dying parent cannot reproduce.
- [ ] Parent-child IDs and generations are correct.
- [ ] Slot reuse does not reuse an individual ID.
- [ ] A newborn does not act or consume until the following step.

Rollback: retain mortality and heredity but disable reproduction calls from `step`.

### Phase 8: Add telemetry, rendering, and the experiment runner

Objective: make ecosystem behavior inspectable.

Return fixed-shape step telemetry:

```python
alive_count
birth_count
death_count
birth_parent_ids[C]  # -1 padded
birth_child_ids[C]   # -1 padded
death_ids[C]         # -1 padded
resource_total
population_energy_total
mean_generation
genome_variance
```

Extend the native Microcosmos rendering path so it skips inactive nodes. Render
ecosystem trajectories through the same field view as existing experiments;
lifecycle-specific resource, population, event, and lineage information remains
in metrics, plots, and lineage artifacts rather than a separate visual style.

Add Hydra configuration and an experiment that writes:

- Metrics CSV or NPZ.
- Lineage table.
- Population and resource plots.
- Lifecycle animation.

Acceptance checkpoint:

- [ ] Event counts match non-padding entries in event arrays.
- [ ] Rendered node count matches active slots multiplied by nodes per creature.
- [ ] The smoke experiment produces finite metrics and an animation.
- [ ] Generated verification artifacts remain in ignored output directories.

Rollback: remove experiment registration; the core simulation and generic active-node rendering remain usable.

### Phase 9: Stability and performance validation

Objective: demonstrate that lifecycle mechanics preserve Microcosmos's execution model.

Required checks:

- 1,000-step deterministic no-fluid rollout.
- 250-step deterministic fluid rollout.
- Empty, one-creature, partially occupied, and full-capacity states.
- Gradient test through masked continuous physics.
- Warm-JIT benchmark at equal padded node counts.

Acceptance checkpoint:

- [ ] All state and field values remain finite.
- [ ] State shapes are identical across population changes.
- [ ] The same seed produces identical lifecycle events and final state.
- [ ] Warm no-fluid step time is no worse than `1.5x` the equal-node baseline.
- [ ] Existing tests pass unchanged.

Rollback: identify the first failing phase and revert only that phase's checkpoint commit.

## 13. User stories

### US-001: Inactive physics slots

As a simulation developer, I want inactive padded bodies to exert no physical effect so that fixed-capacity arrays behave like a variable population.

Acceptance criteria:

- [ ] Inactive nodes remain fixed within absolute tolerance `1e-7`.
- [ ] Active-body results are unchanged when inactive slots are added.
- [ ] PBD, stable Cosserat, steric, and fluid tests cover the mask.

### US-002: Fixed-capacity population

As a researcher, I want population size to change without changing state shapes so that JIT compilation remains reusable.

Acceptance criteria:

- [ ] States with 0, 1, and `max_creatures` live organisms share identical shapes.
- [ ] The same compiled step accepts each occupancy state.

### US-003: Limited resources

As a researcher, I want organisms to deplete a shared resource so that resource access creates competition.

Acceptance criteria:

- [ ] Simultaneous demand cannot remove more than a cell contains.
- [ ] Resource removal matches reported gross uptake within relative error `1e-5`.
- [ ] Regeneration never exceeds configured capacity.

### US-004: Death

As a researcher, I want organisms to die from starvation or age so that unsuccessful individuals leave the population.

Acceptance criteria:

- [ ] Crossing either death threshold emits one death event.
- [ ] A dead organism has no physical, energetic, or rendering effect.

### US-005: Heritable behavior

As a researcher, I want organisms to carry bounded controller and metabolism genes so that behavior and survival traits can be inherited.

Acceptance criteria:

- [ ] Genome decoding returns finite bounded phenotypes.
- [ ] Mutation is deterministic for a fixed key.
- [ ] Zero mutation returns an exact clone.

### US-006: Reproduction

As a researcher, I want eligible organisms to create mutated children in empty slots so that selection operates continuously in the world.

Acceptance criteria:

- [ ] Successful reproduction applies the configured parent energy cost.
- [ ] Failed allocation applies no cost.
- [ ] Child identity, parent identity, and generation are correct.

### US-007: Lifecycle observability

As a researcher, I want lineage events and ecosystem metrics so that results can be analyzed and reproduced.

Acceptance criteria:

- [ ] Every birth and death is represented in fixed-shape telemetry.
- [ ] Aggregate counts agree with event arrays.
- [ ] Metrics contain no NaN or infinite values.

### US-008: Ecosystem visualization

As a researcher, I want a lifecycle animation so that resource competition and population turnover can be inspected visually.

Acceptance criteria:

- [ ] Inactive slots are never visible.
- [ ] Resource depletion and regeneration are visible.
- [ ] Birth and death counters agree with the recorded trace.

### US-009: Performance preservation

As a simulation developer, I want lifecycle mechanics to preserve static-shape GPU execution so that ecological evolution remains scalable.

Acceptance criteria:

- [ ] Population changes trigger no array-shape changes.
- [ ] Warm no-fluid throughput remains within `1.5x` of the equal-node baseline.
- [ ] Continuous masked physics retains finite gradients.

## 14. Functional requirements

- FR-1: `EcosystemEnv` shall preallocate `max_creatures`.
- FR-2: Every node shall map to exactly one fixed creature slot.
- FR-3: Population occupancy shall be represented by boolean activity arrays.
- FR-4: The resource field shall have finite local capacity.
- FR-5: Concurrent consumption shall not overdraw a cell.
- FR-6: Resources shall regenerate at a configured rate.
- FR-7: Organisms shall maintain internal energy and age.
- FR-8: Starvation and lifespan thresholds shall cause death.
- FR-9: Eligible surviving organisms shall reproduce into free slots.
- FR-10: Reproduction shall consume parent energy.
- FR-11: Offspring shall inherit and mutate a fixed-size genome.
- FR-12: Every organism shall have unique lineage metadata.
- FR-13: All lifecycle events shall have fixed-shape JAX outputs.
- FR-14: Inactive nodes shall be ignored by every physics subsystem.
- FR-15: Existing environments shall remain behaviorally unchanged.

## 15. Non-functional requirements

- NFR-1: All rollout state arrays must remain statically shaped.
- NFR-2: The same configuration and PRNG key must produce identical results.
- NFR-3: Resource and energy values must remain finite and bounded.
- NFR-4: Existing continuous physics must remain differentiable.
- NFR-5: Warm no-fluid overhead must remain within the specified benchmark threshold.
- NFR-6: Biological parameters must be configuration-driven and recorded with experiment output.

## 16. Technical considerations

- Use `Nodes.active=None` as the legacy fast path.
- Derive edge activity from both endpoint nodes.
- Derive bending activity from both participating edges.
- Use `component_id` to prevent steric neighbor subtraction across creature boundaries.
- Multiply IBM weights by activity before computing kernel coverage.
- Use array sorting and scattering for parent selection and slot allocation.
- Do not use dynamic boolean indexing inside JIT.
- Keep lifecycle events outside gradient expectations while preserving gradients through the masked physics step.
- Keep resource accounting in `Fields.energy`.
- Do not import the existing experiment-only CPPN utilities into the package.

## 17. Rollout and rollback strategy

Rollout:

- Deliver each phase as an independently testable commit.
- Register `"ecosystem"` only after fixed-slot reset and step tests pass.
- Add the experiment runner only after resource, death, reproduction, and heredity tests pass.
- Keep existing environments as the control group throughout development.

Rollback triggers:

- Legacy physics changes beyond numerical tolerance.
- Existing differentiability tests fail.
- Inactive nodes affect fluid or steric results.
- Population changes alter array shapes.
- Resource conservation exceeds tolerance.
- Performance exceeds the accepted regression threshold.

Rollback action:

- Revert to the most recent phase checkpoint.
- Do not revert unrelated user changes.
- Do not retain partially wired lifecycle state in registered environments.

## 18. Success metrics

- A forced deterministic scenario produces the expected birth and death counts.
- Resource conservation relative error is at most `1e-5`.
- Resource values remain within bounds to `1e-6`.
- No NaN or infinity occurs in the stability rollouts.
- Identical seeds produce identical lineages.
- The existing test suite remains green.
- Warm no-fluid step time is at most `1.5x` the equal-node baseline.
- Rendered active population matches recorded occupancy exactly.

## 19. Open questions

These do not block the architecture:

- Exact resource, metabolism, mutation, and reproduction parameters require empirical calibration.
- Continuous integration should use a Python version compatible with the package metadata; local verification remains authoritative in `sakana`.
- The initial capacity and population used for the first scientific experiment remain configurable.
- If carcass persistence becomes scientifically important, it should be planned as a later feature because it requires a separate decay and resource model.

## 20. Implementation readiness

Primary risks:

- Incorrect masks inside fluid IBM could create hidden forces.
- Metabolic parameters can produce trivial strategies if energetic tradeoffs are poorly calibrated.
- Fixed capacity introduces ecological pressure when the world is full; this must be reported as an explicit model constraint.
- Generic rendering must honor `Nodes.active`; lifecycle state must not introduce a second rendering stack.

Suggested implementation order:

```text
US-001 -> US-002 -> US-003 -> US-004 -> US-005 -> US-006 -> US-007 -> US-008 -> US-009
```

Requirement coverage:

- Birth: Phase 7
- Death: Phase 5
- Reproduction: Phase 7
- Heredity: Phases 6 and 7
- Limited resources: Phase 4
