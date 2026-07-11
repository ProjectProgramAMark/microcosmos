# Ecosystem Native Rendering and Placement Plan

Technical implementation plan for making lifecycle events appear as ordinary
Microcosmos organisms entering and leaving the existing simulation and rendering
pipeline.

- Status: implemented and verified
- Prepared: 2026-07-11
- Repository: `microcosmos`
- Primary objective: remove the parallel ecosystem visualization while fixing the
  actual initial-placement and newborn-placement overlap at its source

Subsequent planning note: the revised
[`plans/microcosmos-lifecycle-remediation-prd.md`](../../plans/microcosmos-lifecycle-remediation-prd.md)
adopts this implemented work as a protected baseline. It retains native
rendering, body-aware periodic placement, and `Nodes.active=None`. Later
topology remediation may add an optional internal physics context, but existing
callers and the no-context path remain unchanged.

## 1. Decision

Birth, death, reproduction, and heredity remain responsibilities of
`EcosystemEnv`. A birth activates and initializes one preallocated organism slot;
a death deactivates one. The resulting `Nodes`, `Edges`, and `Fields` are rendered
through the same functions used by existing Microcosmos experiments.

There will be no ecosystem-only background, organism drawing routine, generation
palette, newborn outline, or telemetry panel. Lifecycle behavior will be visible
because organisms appear, move under the normal solver, reproduce, and disappear.
Detailed lifecycle data will remain in the existing metrics and lineage artifacts.

The physical overlap issue will not be hidden in rendering. Initial organisms and
newborns will be placed with body-scale separation inside `EcosystemEnv`, and the
ecosystem experiment may use Microcosmos's existing steric solver when ongoing
inter-organism collision avoidance is desired.

## 2. Current and target data flow

### Current flow

```text
EcosystemEnv.step
  -> ordinary Microcosmos physics_step(Nodes, Edges, Fields)
  -> resource / energy / death / reproduction
  -> _spawn_children updates preallocated Nodes and Edges
  -> experiment records Nodes, Fields, generations, telemetry
  -> animate_ecosystem
       -> ecosystem-only teal resource texture
       -> ecosystem-only connected-body drawing
       -> ecosystem-only palette and text overlay
```

The lifecycle state is already integrated into the original engine. Only the last
rendering branch is parallel and unnecessary.

### Target flow

```text
EcosystemEnv.step
  -> ordinary Microcosmos physics_step(Nodes, Edges, Fields)
  -> resource / energy / death / reproduction
  -> body-aware placement activates a preallocated organism slot
  -> experiment records Nodes, Fields, telemetry
  -> ordinary Microcosmos animate/render_fields path
```

This preserves one mechanical activity projection: `PopulationState.alive` is
the ecosystem lifecycle authority, `EcosystemEnv` projects it to `Nodes.active`,
and generic physics/rendering consume `Nodes.active`.

## 3. Protected invariants

- Keep fixed-capacity JAX shapes for nodes, edges, populations, and event arrays.
- Do not dynamically append Python objects during a compiled rollout.
- Do not change `simulate`, `physics_step`, PBD, Cosserat, fluid, or force APIs
  as part of this completed rendering/placement change. A later plan may add an
  optional context while preserving every existing caller and the no-context
  behavior.
- Preserve the legacy behavior of nodes whose `active` field is `None`.
- Preserve all existing non-ecosystem renderer defaults and output appearance.
- Do not make lifecycle state a dependency of generic rendering code.
- Do not make available physical space a new reproduction requirement. Placement
  is best-effort when the world becomes crowded; energy and lineage semantics do
  not change.
- Keep resource, population, and lineage observability in `.csv`, `.npz`, and plot
  outputs even though those data are no longer painted over the simulation.
- Add no dependency and no host-side decision inside `jax.lax.scan`.

## 4. Minimal change surface

### 4.1 `src/microcosmos/rendering.py`

Retain only the lifecycle work that is generic and required by fixed-capacity
Microcosmos states:

- Keep inactive-node filtering in `render` and `render_fields`.
- Keep out-of-bounds scatter with `mode="drop"` so inactive padded nodes cannot
  paint a border pixel.
- Keep `_write_mp4` as the shared encoding helper if it remains used by the normal
  `animate` path.

Remove the parallel ecosystem presentation layer:

- `ECOSYSTEM_PALETTE`
- `_ecosystem_resource_texture`
- `_render_ecosystem_frame`
- `render_ecosystem`
- `render_ecosystem_timeseries`
- `animate_ecosystem`

Do not add generation, population, birth, death, resource-capacity, edge-pair, or
telemetry parameters to `render`, `render_fields`, or `animate`. Doing so would
move experiment semantics into a generic visualization API and recreate the
coupling under a different name.

For the lifecycle experiment, use the existing field view with
`animate_energy=False`. This is the same black/blue fluid visualization used by
ordinary Microcosmos batch experiments. Resource abundance remains available in
the metrics plot. The existing optional green energy overlay remains unchanged for
other callers, but it is not the default lifecycle view.

### 4.2 `src/microcosmos/gym/ecosystem.py`

Keep all lifecycle mechanics here. Make two focused changes.

#### Use the normal single-frame renderer

Replace `EcosystemEnv.render`'s call to `render_ecosystem` with the same
`render_fields` route used by other environments:

1. Add a length-one time dimension to `state.nodes` and the renderable field
   leaves.
2. Pass the real `state.fields.fluid_velocity` and `state.fields.steric`.
3. Pass `state.fields.energy` through structurally, but render with
   `animate_energy=False`.
4. Let the already-generic `Nodes.active` filtering omit dead and unused slots.

No `PopulationState` or telemetry enters the renderer.

#### Correct placement where positions are created

The current default `spawn_radius=1.0` is smaller than the body itself. For the
default eight-node line with spacing `2.0`, the body is approximately 14 world
units end to end, so placing its center one unit from its parent necessarily
superimposes most nodes.

Replace that magic constant with topology-derived placement geometry:

```text
body_radius = max(norm(topology.local_positions(), axis=-1))
automatic_spawn_separation = 2 * body_radius + position_margin
```

Use `spawn_separation: float | None = None` in the new environment API. `None`
selects the topology-derived value; an explicit value remains available for
research experiments that intentionally want closer or farther offspring. Validate
that an explicit value is finite and non-negative.

Add one private, JAX-compatible candidate selector in this file rather than a new
module:

```python
_select_best_center(candidates, occupied_centers, occupied_mask)
```

Its behavior is:

1. Compute periodic displacement from every candidate to every occupied center.
2. Mask unused slots out of the distance reduction.
3. Score each candidate by its minimum squared center distance.
4. Select the candidate with the greatest minimum distance.
5. Use deterministic array order to break equal scores.

Reuse this helper in both placement sites:

- `_centers`: generate a fixed-size pool of uniformly sampled legal centers and
  greedily select well-separated centers for the initial live population.
- `_spawn_children`: generate a fixed number of angular candidates at
  `spawn_separation` around each parent, with a random phase derived from the step
  key. Process the fixed-size birth-event array with `jax.lax.scan`, reserving each
  accepted child's center before choosing the next one. This prevents simultaneous
  births from selecting the same location.

Candidate count is a Python/static environment setting, not a traced dynamic
value. Use a small fixed default (for example 16); confirm the final value with the
focused benchmark. Inactive capacity slots must not influence scoring.

This is a best-candidate policy, not a birth veto. If the world is physically too
crowded to maintain ideal clearance, the child is still born at the least crowded
candidate, preserving existing reproduction and energy accounting.

Do not rotate body topology as part of this change. Rotation would affect rest
geometry initialization and adds an independent behavior change.

### 4.3 `experiments/ecosystem/experiment.py`

- Import the ordinary `animate` function instead of `animate_ecosystem`.
- Stop returning and retaining `population.generation` from the rollout scan;
  generation remains in `PopulationState`, telemetry, metrics, and lineage data.
- Pass only `nodes_ts` and `fields_ts` into `animate`.
- Use the normal field view with `animate_energy=False` and the existing
  subsampling and FPS settings.
- Keep all `.csv`, `.npz`, lineage, and plot generation unchanged.

The ordinary animation function currently names the field-view artifact with its
standard `_fluid.mp4` suffix. Prefer that existing convention in this scoped
change rather than adding an ecosystem-specific filename override. If animation
output naming is redesigned later, it should be done once for all callers of
`animate`.

### 4.4 `experiments/conf/experiment/ecosystem.yaml`

Expose the existing solver option rather than implementing collision logic in the
environment or renderer:

```yaml
physics:
  enable_fluid: false
  enable_steric: true
```

In `EcosystemExperiment.setup`, create the experiment-local solver with
`dataclasses.replace(base_solver, enable_steric=...)`. Do not modify
`PBD_SCHEME` or `PBD_SCHEME_NO_FLUID` globally.

The roles remain distinct:

- placement prevents a newborn from being initialized inside its parent or an
  obviously occupied location;
- steric physics discourages organisms from passing through one another later;
- rendering only reports the resulting state.

If focused validation shows that the current steric defaults destabilize the
lifecycle demo, tune strength, sigma, or neighbor-skip only in the ecosystem Hydra
configuration. Do not alter global presets.

### 4.5 `tests/test_ecosystem.py`

Remove tests that assert the deleted ecosystem color palette, resource texture, or
telemetry panel.

Add tests at the behavior boundaries:

1. **Native renderer equivalence**: `env.render(state)` is byte-for-byte equal to
   calling `render_fields` directly on the same length-one state with the same
   arguments.
2. **Inactive-slot omission**: inactive nodes placed at a conspicuous coordinate do
   not alter the native frame.
3. **Automatic spawn separation**: for a one-parent birth in an otherwise empty
   world, periodic parent-child center distance equals the topology-derived spawn
   separation within tolerance and is greater than the body diameter.
4. **Occupied-center avoidance**: when one angular candidate is occupied and a
   clearer candidate exists, the clearer candidate is selected.
5. **Simultaneous-birth reservation**: two births in one step receive distinct
   centers and preserve correct parent/child IDs and generations.
6. **Initial placement**: a seeded reset is deterministic and the chosen live
   centers maximize separation from the supplied fixed candidate pools; inactive
   slots do not affect the result.
7. **Periodic placement**: scoring uses toroidal distance near opposite world
   boundaries.
8. **JIT/static shape regression**: reset and step still compile; empty, partial,
   and full populations retain identical shapes.
9. **Crowded-world behavior**: a birth still occurs when ideal separation is
   impossible, and all state values remain finite.

Avoid screenshot assertions against a hand-authored ecosystem style. The renderer
equivalence test is stronger: any ecosystem frame must be an ordinary Microcosmos
frame for the same physical state.

### 4.6 Documentation

Update `docs/ecosystem-lifecycle-implementation-plan.md` where it currently calls
for ecosystem-specific rendering. State instead that generic rendering must honor
`Nodes.active`, and that lifecycle metrics and lineage are separate artifacts.
Remove the outdated statement that a dedicated active-node rendering path is
required.

## 5. Implementation order

1. Add the placement geometry and candidate-selection tests; verify they fail for
   the current one-unit spawn and unconstrained initialization.
2. Implement topology-derived spawn separation and the shared private center
   selector in `EcosystemEnv`.
3. Route `EcosystemEnv.render` through `render_fields` and add exact equivalence
   coverage.
4. Change the ecosystem experiment to the ordinary `animate` API and remove
   generation timeseries retention.
5. Delete the ecosystem-only renderer and its style-specific tests.
6. Wire `physics.enable_steric` into the experiment-local solver configuration.
7. Correct the rendering sections of the original lifecycle plan.
8. Run focused CPU tests, the complete CPU regression suite, a guarded focused GPU
   smoke test, and both live experiments described below.

This order keeps every intermediate state testable and deletes the parallel
renderer only after all callers have moved.

## 6. Validation matrix

All Python commands must use the `sakana` Conda environment.

### Static and unit validation

```bash
conda run -n sakana python -m compileall src experiments/ecosystem
conda run -n sakana env JAX_PLATFORMS=cpu pytest -q tests/test_ecosystem.py
conda run -n sakana env JAX_PLATFORMS=cpu pytest -q
```

Required results:

- no references to `animate_ecosystem`, `render_ecosystem`,
  `ECOSYSTEM_PALETTE`, or `_ecosystem_resource_texture`;
- all legacy renderer and environment tests pass unchanged;
- ecosystem render output exactly matches the generic renderer;
- all placement and lifecycle states are finite and shape-stable.

### Focused CUDA validation

First assert the CUDA backend, then use a fresh guarded process:

```bash
conda run -n sakana python -c \
  "import jax; assert jax.default_backend() == 'gpu'; print(jax.devices())"

systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana env XLA_PYTHON_CLIENT_PREALLOCATE=false \
  pytest -q tests/test_ecosystem.py
```

Required results:

- JAX reports `CudaDevice(id=0)`;
- the focused ecosystem suite passes without CPU fallback;
- placement selection and lifecycle steps compile under JAX on CUDA.

### Live original-experiment regression

Run one unchanged original experiment, such as the worm swimming example, using
its existing Hydra entry point. Confirm that its output frame style, dimensions,
duration, and node/fluid behavior remain unchanged from the pre-refactor artifact.
This validates that generic renderer defaults were not changed to accommodate the
ecosystem.

### Live lifecycle experiment

Run a short, deliberately active lifecycle configuration that guarantees at least
one birth and one death while keeping the normal field view:

- small fixed capacity;
- short maturity age;
- reproduction threshold reachable within the rollout;
- finite maximum lifespan or metabolism sufficient to produce a death;
- steric enabled in this experiment only;
- `animate_energy=false`.

Validate both state and artifact behavior:

- births equal newly activated organism slots;
- deaths equal newly deactivated organism slots;
- every newborn first appears at its initialized physical position in the normal
  Microcosmos view;
- parent-child and simultaneous-newborn center distances are recorded and checked
  numerically, not judged only from pixels;
- no inactive slot is visible;
- the video does not use the deleted teal/green ecosystem background, palette,
  edge drawing, or telemetry panel;
- `metrics.csv`, `metrics.npz`, `lineage.csv`, the metrics plot, and the animation
  are all produced and mutually consistent;
- `ffprobe` reports a decodable H.264/YUV420p video with the expected frame count,
  dimensions, and FPS.

Extract a small contact sheet from the beginning, around the first birth, around
the first death, and at the final frame. Compare it beside the original-experiment
contact sheet to confirm both use the same renderer rather than merely similar
colors.

## 7. Downstream impact analysis

### Rendering callers

No non-ecosystem caller imports the functions being removed. Generic renderer
signatures and defaults stay unchanged, so `base_experiment`, `main`, swim sweep,
filament folding, encoding experiments, and realtime rendering remain on their
existing paths.

### Physics callers

There is no solver change. Experiment-local `dataclasses.replace` produces a new
immutable solver config and cannot mutate global presets used elsewhere.

### Lifecycle semantics

Population allocation, reproduction eligibility, energy costs, mutation, IDs,
generation, and event telemetry remain unchanged. Only the physical coordinates
assigned to initial and newborn bodies change.

### Performance

Rendering becomes smaller by removing the Python/OpenCV ecosystem frame loop and
its duplicate drawing logic. Placement adds bounded work only on reset and birth:
`O(C * K * C)` for capacity `C` and static candidate count `K`. It does not alter
the dominant per-step physics cost when no births occur, apart from the already
present fixed-size birth branch. Confirm this with a warm-JIT benchmark before and
after; reduce `K` if placement time is material without sacrificing the placement
tests.

### Research outputs

Generation colors and text overlays disappear from the animation by design. The
underlying information is not lost: generation remains in population state and
aggregate metrics, while individual ancestry remains in `lineage.csv`.

## 8. Acceptance criteria

The work is complete only when all of the following hold:

- One rendering implementation produces both ordinary and lifecycle videos.
- A lifecycle video is visually the ordinary Microcosmos simulator with organisms
  appearing and disappearing; it has no ecosystem-only presentation layer.
- The default newborn center offset is derived from topology size, not a one-unit
  magic constant.
- Initial and simultaneous birth placement account for occupied periodic space.
- Optional ongoing collision avoidance uses the existing steric solver through an
  experiment-local configuration.
- Existing renderer APIs and non-ecosystem experiment outputs are unchanged.
- Lifecycle metrics, lineage, and plots remain complete and consistent.
- Focused CPU tests, the complete CPU suite, focused guarded CUDA tests, one live
  original experiment, and one live lifecycle experiment all pass.

## 9. Explicit non-goals

- Redesigning the generic Microcosmos animation API or its `_fluid.mp4` naming.
- Redesigning the existing optional green energy overlay for other experiments.
- Adding organism edge/segment drawing to the normal renderer.
- Adding rotation, growth, variable topology, or collision shapes.
- Guaranteeing non-overlap at impossible packing densities.
- Changing reproduction success based on geometric space.
- Tuning ecological parameters to claim long-term equilibrium.

## 10. Verification record

Completed on 2026-07-11 in the `sakana` Conda environment:

- Complete CPU regression: 182 tests passed.
- Guarded CUDA ecosystem regression: 13 tests passed on `CudaDevice(id=0)`.
- Required warm-JIT benchmark at 64 creatures, 16 nodes per creature, and
  1,000 no-fluid steps: `1.2000x` ecosystem/baseline overhead.
- Unchanged original live experiment:
  `outputs/worm_swim_example/2026-07-11/12-38-59/worm_swim_fluid.mp4`.
- Final native-rendered fluid lifecycle experiment:
  `outputs/ecosystem/2026-07-11/12-45-43/lifecycle_fluid.mp4`.
- Final lifecycle trace: two births, four deaths, population range zero through
  four, finite metrics, and two matching lineage rows.
- Final video: H.264, YUV420p, 512 by 512, 20 FPS, 60 frames. Contact-sheet
  inspection confirmed the normal Microcosmos field view, separated newborn
  bodies, visible disappearance at death, and no ecosystem-only background,
  palette, edge drawing, or telemetry panel.
