# Evo²-Ecosystem

## Recursive Discovery of Heredity Policies for Embodied Artificial Life

**Sakana AI RSI Lab pre-interview project report**  
**Date:** July 12, 2026  
**Experiment run:** `evo2-production-20260712-r1`

> Can an AI-discovered heredity policy help a population of embodied artificial organisms recover from environmental catastrophes better than conventional evolutionary operators?

---

## Executive summary

I built **Evo²-Ecosystem**, a bounded recursive program-improvement experiment in which an outer AI system improves the mechanism by which an inner artificial-life population evolves.

The project extends [Microcosmos](https://arxiv.org/abs/2607.02954)—a JAX-based simulator of elastic filament organisms in viscous fluid—with the minimum ecology necessary for multi-generation evolution:

- multiple organisms in a shared physical world;
- finite, depletable resources;
- stored energy and metabolic costs;
- birth and death;
- inherited CPPN controllers;
- asexual mutation and structural variation;
- parent-child and founder-lineage tracking;
- resource relocations, population bottlenecks, and lineage extinctions.

I then integrated [ShinkaEvolve](https://sakana.ai/shinka-evolve/) as the outer discovery engine. ShinkaEvolve did not directly design organisms and could not edit the simulator. It could edit only a small function that receives bounded ecological summaries and scores four trusted TensorNEAT heredity operators: clone, parametric mutation, structural mutation, and mixed mutation. The selected operator determines how the parent's CPPN is transformed into the child's CPPN.

This creates two nested improvement processes:

```text
ShinkaEvolve improves the heredity policy
                    ↓
The heredity policy generates CPPN descendants
                    ↓
Embodied descendants compete, reproduce, and die
                    ↓
Ecological performance verifies the heredity policy
                    ↓
ShinkaEvolve proposes a better policy
```

The main experiment compared two matched outer searches:

1. **Stable meta-training:** Shinka evolved heredity policies in worlds without ecological shocks.
2. **Punctuated meta-training:** Shinka evolved policies in worlds with resource relocation and random population bottlenecks.

The hypothesis was that punctuated training would cause the outer system to discover a more adaptive balance between conservative and exploratory mutation. Before final testing, I froze one champion from each arm using a separate development set. I then evaluated the champions and four conventional baselines on a sealed suite of 36 unseen shock/null pairs.

The engineering result was successful: the full embodied ecosystem, bounded Shinka task, GPU evaluation pipeline, holdout protocol, and confirmatory experiment all ran to completion. The scientific hypothesis was **not supported**. The punctuated-trained policy did not reliably outperform the stable-trained policy, fixed mixed mutation, or a handwritten stress-responsive policy. Fixed parametric mutation achieved the highest aggregate sealed score.

That negative result is informative. It shows that a plausible ecology-conditioned scheduler can be autonomously discovered and executed, but also that short-horizon operator scheduling over this restricted four-operator portfolio did not beat a strong simple baseline. It further exposed a meaningful distinction between **absolute ecological productivity** and **relative shock resilience**: the policy with the best aggregate productivity was not the policy with the smallest shock-induced deficit.

The most defensible conclusion is therefore:

> Evo²-Ecosystem demonstrates a complete, physically grounded benchmark for bounded recursive improvement of an evolutionary process. In this preregistered case study, however, catastrophe-trained program evolution did not produce better unseen catastrophe recovery than stable training or conventional heredity policies.

---

## 1. Why I built this project

### 1.1 The underlying question

Most evolutionary artificial-life experiments search for better organisms while leaving the rules of heredity fixed. A human chooses how parents mutate, how large mutations are, when topology changes are allowed, and how exploration responds to ecological stress. The organism population evolves, but the mechanism producing future generations does not.

Evo² moves the autonomous search up one level:

> Instead of asking AI to find one successful organism, can AI improve the algorithm that generates future generations of organisms?

This is why the project is called **Evo²**. The inner evolutionary process changes organisms. The outer program-evolution process changes part of the inner evolutionary process.

The project is also motivated by a basic biological observation: evolution does not occur under one stationary objective. Environments relocate resources, populations pass through bottlenecks, dominant lineages disappear, and formerly useful strategies become obsolete. A heredity mechanism that works well in a stable world may not be the mechanism that best supports recovery after abrupt change.

### 1.2 Alignment with the Sakana exercise

The exercise asks for an innovative proof of concept building on Sakana's work in AI-driven discovery and recursive self-improvement. It particularly highlights Physical AI and sample-efficient improvement that compounds through ideas rather than hyperscale compute. Evo² was designed around those priorities.

**AI-driven discovery.** ShinkaEvolve proposes executable heredity algorithms, runs them against an objective verifier, preserves useful code lineages, and uses the measured results to generate descendants. The LLM is not merely analyzing completed data or writing a report; it participates in the algorithm-discovery loop.

**Recursive self-improvement.** The edited artifact is itself part of an improvement process. A better heredity policy should create better future evolutionary search. This is bounded recursive program improvement—not unrestricted self-modification—because the editable function and all external budgets are deliberately constrained.

**Physical AI.** The agents are elastic bodies acting through a simulated viscous fluid. Their controllers must produce coordinated motion, locate resources, pay movement costs, and survive embodied competition. Microcosmos is a hand-written differentiable physical simulator rather than a learned world model, so I use the more precise term **physically grounded simulation substrate**.

**Sample efficiency through ideas.** The search target is a compact program that schedules four trusted mutation operators. Shinka cannot buy performance by increasing population capacity, simulator horizon, resource supply, or GPU budget. The two 15-generation outer searches together cost approximately **$3.99 in LLM API calls** and about **1.98 hours of outer-run wall time**. The core bet is that an algorithmic rule can transfer across many births and worlds.

**Meaningful use of Sakana's systems.** This is not an out-of-the-box ShinkaEvolve demo. The substantive work was building a new embodied evolutionary substrate, defining a secure evolvable boundary, constructing an executable ecological verifier, preregistering the comparison, and running matched stable and punctuated program-evolution experiments.

### 1.3 Relationship to Sakana's research

The project combines ideas from several strands of Sakana's work while assigning each a specific role:

| Sakana work | Idea carried into Evo² |
|---|---|
| [The AI Scientist](https://sakana.ai/ai-scientist/) | Automated experiment execution, explicit hypotheses, reproducible artifacts, and the importance of executable evidence rather than prose-only evaluation |
| [ShinkaEvolve](https://sakana.ai/shinka-evolve/) | LLM-guided evolution of bounded executable programs with an objective verifier and archive-based search |
| [Darwin Gödel Machine](https://sakana.ai/dgm/) | Treat code descendants and retained lineages as scientific objects; improve an improvement mechanism rather than only a final artifact |
| [Digital Red Queen](https://sakana.ai/drq/) | Study algorithm discovery under changing selection pressure rather than one stationary objective |
| Microcosmos | GPU-native embodied artificial life in differentiable viscous-fluid physics |

The resulting combination is distinct: the outer program search improves heredity rules, the inner population evolves through embodied ecological selection, and environmental shocks are a controlled experimental treatment.

---

## 2. Research question, hypotheses, and claims

### 2.1 Primary research question

> Does a heredity policy discovered under punctuated ecological conditions produce better post-catastrophe productivity in unseen worlds than a policy discovered under stable conditions or conventional mutation operators?

### 2.2 Primary hypothesis

Punctuated meta-training should favor policies that react to population decline, low behavioral diversity, or lineage concentration by temporarily increasing structural exploration. Stable meta-training should favor conservative parametric refinement. If that division is useful, the punctuated-trained policy should recover more effectively after unseen ecological disruptions.

### 2.3 What would constitute recursive improvement

A single strong descendant would not be sufficient. The candidate program must improve the process by which many fresh descendants are generated. Every candidate is therefore tested from fixed founder populations across multiple worlds, under the same simulator and ecological budget.

The recursive claim is intentionally narrow:

- the **inner improvement process** is ecological evolution of CPPN-controlled organisms;
- the **outer improvement process** is ShinkaEvolve's evolution of the heredity scheduler;
- the simulator, verifier, mutation kernels, manifests, and budgets are immutable.

I describe this as **bounded recursive program improvement of an embodied evolutionary process**. I do not claim unrestricted RSI, open-ended evolution, self-assembly, or abiogenesis.

### 2.4 Preregistered evidentiary standard

Before the matched searches, I froze an analysis plan specifying:

- training, development, and sealed partitions;
- the ecological unit of replication;
- the primary metric and contrasts;
- the baseline policies;
- the numerical-repeat protocol;
- finalist selection;
- conditional ancestor-descendant and mechanism-ablation analyses;
- the conditions under which each claim could be made.

The preregistration SHA-256 remained unchanged throughout the experiment:

```text
09cdea4de1575317e9430c1c226e7cef6a462088c1e40b2f8e3adf4b1744700a
```

---

## 3. System design

### 3.1 What is an organism?

Each organism is an eight-node elastic filament controlled by a feed-forward compositional pattern-producing network (CPPN). A CPPN is a compact neural network that is repeatedly queried along the body. Its inputs are:

1. normalized position along the filament;
2. time phase;
3. local resource concentration;
4. the resource difference between the head and tail.

Its bounded output sets the desired local bending angle. Because the same network is evaluated at every body location and time, one compact inherited genome can generate coordinated traveling waves and sensory corrections across the whole body.

This keeps CPPN-NEAT within the scientific workflow from the beginning. The project does not replace the released indirect encoding with a temporary fixed wave controller.

### 3.2 What evolves inside the ecosystem?

Each organism slot contains a padded TensorNEAT genome with capacity for:

- 15 nodes;
- 30 connections;
- the activation functions identity, sine, hyperbolic tangent, and absolute value.

Padding allows connection weights, enabled edges, and network topology to vary without changing tensor shapes. This matters because JAX compilation and efficient GPU batching depend on static array dimensions.

The project uses TensorNEAT mutation machinery but not a separate synchronous textbook NEAT generation loop. Instead, the ecosystem itself supplies selection:

- organisms that collect enough resource earn reproduction;
- successful CPPNs therefore produce more descendants;
- descendants receive parametric and/or structural variation;
- organisms that exhaust their energy die;
- the ecological population changes asynchronously through births and deaths.

This is best described as **CPPN-controlled ecological evolution using trusted TensorNEAT heredity operators**.

### 3.3 Fixed-capacity organism slots

Dynamic population size is awkward inside a compiled JAX simulation because appending an organism would change array shapes and trigger recompilation. I instead preallocate a maximum capacity:

```text
32 organism slots
 8 nodes per organism
 8 living founders at initialization
```

The memory for all slots exists from the start, but only slots with `alive=True` represent living organisms. Death clears a slot's active mask. Birth initializes a currently inactive slot with a child state and CPPN.

This provides genuine dynamic birth and death while preserving fixed GPU shapes.

### 3.4 Making dead organisms physically absent

An inactive body must not remain as an invisible obstacle. The alive mask is propagated through every relevant physical subsystem:

- elastic constraints;
- bending constraints;
- steric-field deposition;
- steric-force sampling;
- immersed-boundary fluid coupling;
- velocity updates;
- controller outputs;
- resource intake.

Regression tests verify that changing the position of a dead slot does not alter a living organism's trajectory and that an all-active masked simulation matches the unmasked reference within tolerance.

### 3.5 Resources and metabolism

Organisms share a finite scalar resource field with localized source geometry. Resource regenerates at a fixed rate, but existing stock can be depleted. The first node acts as a mouth.

When mouths compete for the same grid cell, the simulator first aggregates total demand and then allocates the available resource proportionally. It never allows the organisms to consume more than the cell contains.

Stored energy changes according to the conceptual balance

```text
new energy
= fulfilled resource uptake
- basal living cost
- actuation cost
- reproduction cost, when applicable
```

An organism dies when its energy is exhausted. A parent may reproduce only after crossing an immutable energy threshold and only if a slot is available. Birth is energy-conserving: the parent pays at least the child's endowment. A full population does not charge a parent for an impossible birth.

### 3.6 Birth, heredity, and lineages

Reproduction is deliberately asexual for this proof of concept. At birth:

1. an eligible parent is selected by the fixed lifecycle code;
2. the heredity policy receives bounded statistics;
3. the policy scores the four trusted operators;
4. trusted code samples an operator;
5. the operator transforms the parent's TensorNEAT genome;
6. the child CPPN is validated and its graph transform cached;
7. the child body is placed near the parent with zero initial velocity;
8. the lifecycle assigns a unique ID, parent ID, generation, and founder lineage.

The event log records births, deaths, and catastrophe removals. This supports phylogenetic analysis without putting variable-length Python event structures inside the JAX state.

### 3.7 Environmental disturbances

The ecosystem supports three core disturbances:

**Resource relocation.** The source moves to a different region. This changes the behavioral and evolutionary problem and is the clearest adaptation treatment.

**Random bottleneck.** A fixed fraction of organisms is removed independently of performance. This primarily tests demographic recovery.

**Dominant-lineage cull.** The most abundant founder lineage is identified and removed up to a fixed cap. This tests whether a population that appears successful is excessively dependent on one phylogenetic family.

Every shocked development or final world has a paired null world. The pair runs one exact shared trajectory until the event boundary and forks only at the event. This isolates the causal effect of the disturbance from pre-event simulator divergence.

---

## 4. The evolvable heredity boundary

### 4.1 Why edit a scheduler rather than the simulator?

Allowing an LLM to rewrite arbitrary physics, resource rules, or scoring code would make a high score uninterpretable. It could eliminate death, create free energy, expose hidden seeds, or weaken the test. Conversely, letting Shinka change only scalar hyperparameters would reduce the exercise to ordinary tuning.

I chose a middle boundary: Shinka writes a small program that can express ecological strategies, while trusted code owns all state mutation.

The editable API is conceptually:

```python
def make_offspring(
    parent_genome,
    parent_stats,
    population_stats,
    rng,
):
    """Return scores for clone, parametric, structural, and mixed mutation."""
```

The name reflects its role in reproduction, but the function does not receive or return raw genomes. It returns four bounded scores.

### 4.2 Information visible to the policy

The policy may observe:

- fraction of available CPPN nodes in use;
- fraction of available CPPN connections in use;
- normalized parent energy;
- recent parent resource intake;
- fraction of organism slots alive;
- recent population change;
- behavioral/action diversity;
- founder-lineage entropy.

These summaries are sufficient to express hypotheses such as:

- exploit a productive parent with small weight changes;
- try structural changes when the network has unused capacity;
- preserve a genome during demographic fragility;
- increase exploration when lineages or actions converge.

The policy cannot observe:

- raw CPPN weights or graph tensors;
- simulator arrays;
- world or scenario identity;
- whether the current world belongs to training or holdout;
- the event schedule;
- hidden seeds or manifests;
- files, environment secrets, network services, or subprocesses.

### 4.3 Trusted heredity operators

The four fixed operators are:

| Operator | Effect |
|---|---|
| Clone | Preserve the parent CPPN unchanged |
| Parametric | Mutate numerical network parameters while retaining topology |
| Structural | Change enabled connections or network structure using TensorNEAT machinery |
| Mixed | Apply both parametric and structural mutation |

Shinka can invent the **policy that chooses among these operators**, but it cannot redefine the mutation kernels after observing the tests. This cleanly separates algorithm discovery from low-level genome integrity.

### 4.4 Candidate validation and sandboxing

Candidate code passes through a strict AST allowlist and executes with:

- restricted imports;
- scrubbed environment variables;
- no filesystem, network, reflection, or subprocess access;
- fixed execution timeouts with process-tree termination;
- shape, type, finiteness, and bound checks;
- deterministic inputs and explicit random keys;
- source hashing and provenance records.

Invalid candidates fail closed. One stable-arm proposal was rejected safely; it did not consume a valid scientific result.

---

## 5. GPU-native and differentiability design

Microcosmos is designed for accelerator-native artificial life, and the lifecycle extension preserves that style where it is useful.

The following remain fixed-shape JAX operations and execute on the GPU:

- filament and fluid physics;
- batched CPPN inference;
- resource regeneration and allocation;
- organism energy and lifecycle state;
- alive masks;
- cached padded CPPN graphs;
- trusted mutation profiles;
- bounded birth updates.

The entire ecosystem is **not end-to-end differentiable**, and I do not claim that it is. Birth, death, topology mutation, catastrophe removal, nearest-cell resource lookup, masks, clipping, and discrete operator selection introduce nonsmooth transitions.

This is an intentional boundary rather than an implementation failure. The continuous mechanics and controller kernels remain differentiable between ecological transitions, while evolutionary search—not backpropagation through ancestry—is the optimization mechanism studied here.

The static-slot design also avoids unnecessary recompilation. Candidate policy changes do not rewrite the physics kernel or change simulation shapes.

---

## 6. Experimental design

### 6.1 Matched outer searches

I ran two ShinkaEvolve arms:

| Property | Stable arm | Punctuated arm |
|---|---|---|
| Initial program | Identical | Identical |
| Outer seed | Identical | Identical |
| Proposal model and settings | Identical | Identical |
| Generation budget | 15 | 15 |
| Simulator and score | Identical | Identical |
| Founder panel | Identical | Identical |
| Training treatment | Null events | Resource relocation and bottlenecks |

The stable arm stored 15 programs: the initial program and 14 valid descendants. One additional proposal was safely rejected. The punctuated arm stored the initial program and 14 valid descendants.

The stable search used approximately **$2.44** in LLM API calls and 4,015 seconds of wall time. The punctuated search used approximately **$1.56** and 3,128 seconds. Candidate evaluation, rather than proposal generation, dominated runtime.

### 6.2 Training, development, and sealed partitions

**Training** was visible to Shinka. It contained matched worlds under the stable or punctuated treatment.

**Development** remained unreadable until both outer searches closed. For each archive, the top three unique candidates were reevaluated on fresh development worlds. Development results were not fed back into the archive.

**Sealed final** was generated and hashed before search, stored outside the Shinka result tree, and kept unreadable during proposal generation and finalist selection. It was opened once for the frozen six-policy comparison.

To prevent cross-arm leakage, the completed stable archive was made unreadable while the punctuated search ran.

### 6.3 Frozen run identity

The shared run specification binds the model, seeds, budgets, manifests, trusted source files, dependency lock, baseline set, and analysis-plan hash.

```text
Run-spec SHA-256:
47ac4c3261a64ec37d04f1303b52846329eeffc6efd92f0bf52276adb4ed304f
```

Candidate sources, evaluator sources, simulator sources, manifests, and completion records were hashed. Finalists were frozen before sealed evaluation.

### 6.4 Numerical nondeterminism

Ordinary GPU scatter and reduction order on this backend was not bitwise deterministic. Treating backend reruns as additional ecological samples would have been statistically incorrect.

I therefore preregistered the following protocol:

1. execute each complete candidate-manifest evaluation three times;
2. require all three measurements to pass integrity checks;
3. select the coherent full measurement with median aggregate score using stable tie-breaking;
4. retain that measurement's complete vector of world outcomes;
5. treat world pairs—not backend repeats—as the statistical units.

I did not independently median individual episodes, because doing so would synthesize a result that never occurred in one coherent simulator execution.

### 6.5 Baselines

The sealed comparison included six frozen policies:

1. **Clone:** no mutation.
2. **Fixed parametric:** always mutate CPPN parameters.
3. **Fixed mixed:** always apply mixed parametric and structural mutation.
4. **Stress responsive:** a handwritten policy that increases exploration under ecological stress.
5. **Shinka stable:** champion selected from stable meta-training.
6. **Shinka punctuated:** champion selected from punctuated meta-training.

The comparison tests more than whether Shinka beats a deliberately weak Gaussian baseline. Fixed parametric and fixed mixed mutation are direct uses of the trusted TensorNEAT operators, while the handwritten scheduler embodies the main human hypothesis available to Shinka.

### 6.6 Primary outcome

The environment reward is fulfilled gross mouth uptake. For each post-event chunk:

```text
productivity = cumulative fulfilled intake / simulated time
normalized productivity = clip(productivity / post-event regeneration capacity, 0, 1)
```

The per-world primary outcome is the mean normalized productivity over the frozen post-event window—equivalently, a normalized post-event productivity AUC.

The primary confirmatory contrasts use direct paired differences in shocked worlds:

1. punctuated Shinka minus stable Shinka;
2. punctuated Shinka minus fixed mixed mutation.

The stress-responsive comparison is supporting evidence.

The shock-minus-null difference is a secondary measure of disturbance effect. It is not substituted for absolute shocked-world performance.

### 6.7 Statistical unit and uncertainty

The sealed panel contained **36 ecological shock/null pairs**, spanning:

- unseen resource geometry;
- unseen resource abundance;
- dominant-founder-lineage culling.

The same 72-world suite was used for every policy. Confidence intervals were computed with 10,000 paired bootstrap replicates using seed `20260712`.

Timesteps, organisms within a world, and numerical GPU repetitions were not treated as independent samples.

---

## 7. What ShinkaEvolve discovered

### 7.1 Stable-trained champion

The stable archive's top three candidates were reevaluated on development. The selected champion came from outer generation 3 with development median score `0.0668331`.

Its policy derives concepts including:

- parent productivity;
- network compactness and unused topology capacity;
- population stagnation;
- behavioral and lineage diversity gaps;
- population crowding;
- a gated “rescue pressure.”

It generally favors parametric refinement for productive parents in stable or crowded populations. Structural and mixed exploration gain weight when the population is stagnant, diversity is low, and the parent's network has structural headroom.

### 7.2 Punctuated-trained champion

The selected punctuated champion came from generation 8 with development median score `0.0682406`.

It derives:

- CPPN complexity and structural headroom;
- parent productivity;
- demographic fragility;
- behavioral/lineage convergence;
- population stability;
- innovation pressure;
- a recovery signal.

It attempts to balance preservation, parametric refinement, structural innovation, and mixed mutation according to those ecological conditions.

### 7.3 Why the code is scientifically useful even before a win

Both policies are understandable executable hypotheses. Shinka did not merely tune one mutation rate. It composed ecological state into conditional operator-selection logic that resembles a resource-aware exploration/exploitation strategy.

That makes the outcome mechanistically inspectable. The sealed test can ask not just whether an opaque model scored well, but whether this class of ecology-conditioned operator scheduler actually transfers.

---

## 8. Results

### 8.1 Development selection

The top three unique candidates from each archive passed trusted development reevaluation. Selection occurred before the sealed manifest was made readable.

| Search regime | Selected outer generation | Development median score |
|---|---:|---:|
| Stable | 3 | 0.0668331 |
| Punctuated | 8 | 0.0682406 |

The slight development advantage for the punctuated champion did not carry into sealed testing.

### 8.2 Sealed aggregate outcomes

| Policy | Aggregate score | Survival rate | Mean shock-minus-null effect |
|---|---:|---:|---:|
| Fixed parametric | **0.12542** | 0.597 | -0.04324 |
| Clone / no mutation | 0.11456 | **0.625** | -0.08836 |
| Shinka stable | 0.08825 | 0.569 | -0.05520 |
| Shinka punctuated | 0.08657 | 0.569 | -0.05982 |
| Stress responsive | 0.08504 | 0.583 | -0.05638 |
| Fixed mixed | 0.06773 | 0.597 | **-0.02743** |

These aggregate scores are descriptive. The preregistered inference uses paired shocked-world AUC contrasts.

### 8.3 Primary shocked-world contrasts

Positive values favor the punctuated-trained champion.

| Contrast | Mean paired difference | 95% bootstrap interval | Fraction of pairs positive |
|---|---:|---:|---:|
| Punctuated − stable Shinka | -0.00399 | [-0.01468, 0.00581] | 0.083 |
| Punctuated − fixed mixed | 0.00265 | [-0.02805, 0.03348] | 0.278 |
| Punctuated − stress responsive | -0.00019 | [-0.01321, 0.01326] | 0.194 |

None of the intervals excludes zero, and the punctuated champion had no reliable advantage.

The secondary punctuated-minus-stable difference-in-differences was `-0.00462`, with 95% interval `[-0.01233, 0.00112]`. The difference in aggregate candidate score was `-0.00167`.

### 8.4 Result of the hypothesis test

The confirmatory result is negative:

> In this matched outer-search case study, catastrophe-trained Shinka did not improve sealed catastrophe recovery relative to stable-trained Shinka or the conventional comparison policies.

The conventional fixed parametric operator achieved the highest aggregate score. The Shinka policies remained broadly competitive with the handwritten stress-responsive scheduler, but they did not exceed the simpler baseline reliably.

### 8.5 What the baseline pattern reveals

The baseline table contains several useful lessons.

**Absolute performance and resilience are different.** Fixed parametric mutation achieved the highest aggregate productivity, while fixed mixed mutation had the least-negative average shock-minus-null effect. A population can be relatively insensitive to a shock because its null performance is also modest. Conversely, a highly productive population has more performance to lose.

**High clone survival is not evidence of adaptation.** Clone achieved the highest survival rate but the largest shock deficit. This likely reflects persistence of competent standing founder variation rather than effective post-shock hereditary adaptation.

**More structural exploration was not automatically better.** Fixed mixed mutation was the most shock-insensitive by the secondary measure but had the lowest aggregate score. Frequent topology changes may preserve exploratory capacity while damaging short-horizon productivity.

**A scheduler needs a sufficiently informative horizon and action space.** The Shinka policies could choose among four coarse operators, but could not alter mutation intensity, target modules, or selection dynamics. That boundary may have been too narrow for ecological context to provide a consistent advantage over fixed parametric mutation.

### 8.6 Analyses not performed

The preregistration made ancestor-descendant common-garden tests and mechanism ablations conditional on a positive finalist result. That condition was not met, so I did not run them or reinterpret exploratory analyses as confirmatory evidence.

This is an important part of the result. The project stopped where its frozen evidentiary logic said it should stop.

---

## 9. Interpretation

### 9.1 What succeeded

The project succeeded as a systems and research-engineering proof of concept:

- Microcosmos now supports finite-resource, multi-generation embodied ecology.
- CPPN-controlled organisms inherit real parametric and structural TensorNEAT variation.
- Death and birth work within static JAX shapes without leaving ghost physics.
- Environmental shocks have exact paired null counterfactuals.
- ShinkaEvolve can safely modify an algorithmic heredity seam.
- Stable and punctuated outer searches ran under a matched, hashed specification.
- Development selection and sealed evaluation were isolated.
- All six policies completed the full confirmatory suite.
- The analysis returned an honest, interpretable negative result.

### 9.2 What did not succeed

The intended scientific advantage did not appear. Punctuated meta-training did not produce a scheduler with stronger unseen post-catastrophe productivity than stable meta-training or conventional policies.

This does not establish that ShinkaEvolve cannot discover useful evolutionary algorithms. The experiment is one bounded case study with:

- one matched outer search per regime;
- one frozen panel of eight founder CPPNs;
- a 15-generation outer budget;
- four trusted operator choices;
- a finite ecological horizon;
- a specific score emphasizing absolute post-event resource productivity.

The conclusion is conditional on those design choices.

### 9.3 Why the null result matters

It would have been easy to build a visually compelling ecosystem and report the best Shinka score from training. The holdout design reveals why that is not enough. A small development advantage did not survive sealed testing, and simple mutation remained difficult to beat.

This illustrates a central RSI engineering problem: autonomous improvement is only as credible as the verifier, partitioning, and comparison protocol around it. Open-ended proposal generation does not remove the need for conservative scientific inference.

The result also suggests a substantive hypothesis for future work: ecological context may matter less for choosing a coarse mutation **category** than for controlling mutation **magnitude, modularity, and temporal persistence**.

---

## 10. Verifier integrity and audit trail

### 10.1 Immutable components

Shinka could not alter:

- Microcosmos physics;
- resource generation or uptake accounting;
- birth costs and death conditions;
- population capacity;
- catastrophe implementation or schedule;
- trusted TensorNEAT mutation kernels;
- training or holdout manifests;
- metric and aggregation code;
- runtime, GPU-memory, or simulation budgets;
- hidden seeds.

This prevents apparent improvement through free reproduction, disabled mortality, extra resources, longer runs, or weakened evaluation.

### 10.2 Fail-closed infrastructure amendments

Two infrastructure defects were found during the locked workflow. Neither changed a scientific score.

1. **SQLite completion hash timing.** Completion hashes were initially captured just before the final write-ahead-log checkpoint. The launcher was corrected to checkpoint and close before hashing. Only the derived completion hashes were corrected, and this occurred before development was opened.

2. **Sealed provenance serialization.** The first sealed stable simulation finished, but provenance serialization could not call `inspect.getsource` on a dynamically loaded frozen candidate. No atomic policy result was published. I fixed the provenance-only serializer to use the already frozen candidate source hash, quarantined the failed metadata, and reran the unchanged six-policy suite from an empty result directory.

These failures demonstrate the value of atomic publication and fail-closed execution: a partially completed run could not silently enter the final analysis.

### 10.3 Test and integrity status

- Microcosmos full CPU regression suite: **334 passed, 1 skipped**.
- ShinkaEvolve full CPU suite: **527 passed**.
- Every sealed record reports `integrity_valid=true`.
- The final summary was recomputed exactly from six atomic policy records.
- Guarded GPU jobs ran in fresh processes with 24 GiB cgroup limits.

The sealed summary SHA-256 is:

```text
4b5d4bbc16645968b0bf97986626d10ea4e337fd2cf97f5adc2bd57d0253180c
```

---

## 11. Engineering decisions that kept the project minimal

Several tempting features were deliberately excluded.

**No true self-assembly.** Children receive a predefined filament body. Implementing free material, dynamic bonding, organism identity, and growth would have replaced the RSI experiment with a much larger simulator project.

**No arbitrary body topology.** The body remains a fixed filament. Structural evolution occurs in the CPPN controller, not the physical graph.

**No sexual reproduction.** Asexual descent keeps attribution and phylogeny clear and avoids mate-selection confounds.

**No unrestricted NEAT rewrite.** The trusted mutation kernels remain fixed. The outer scientific variable is the ecology-conditioned scheduler.

**No adversarial environment generator.** Stable versus punctuated manifests provide a controlled first experiment. Coevolving environments would add another moving population and a large evaluator-gaming surface.

**No diversity bonus in the primary score.** Diversity is exposed as a diagnostic and policy input, but the verifier rewards resource productivity. This avoids paying the outer system to maintain useless diversity merely for a metric.

**No Python-side organism loop in the physics hot path.** Fixed slots, masks, batched CPPN evaluation, and padded graphs preserve the GPU-native execution model.

These decisions concentrate complexity at one scientific seam: how ecological state should influence heritable variation.

---

## 12. Limitations

### 12.1 One outer run per regime

The experiment compares the selected outcome of one stable and one punctuated Shinka archive. The 36 ecological pairs quantify uncertainty in final world performance; they do not quantify variation across independent Shinka searches. A general claim about the training regime would require multiple independent outer runs.

### 12.2 One founder panel

All confirmatory worlds begin from one frozen panel of eight viable CPPNs. Results are conditional on this standing variation. Different founder distributions could change which heredity operator is useful.

### 12.3 Restricted heredity portfolio

Shinka schedules only clone, parametric, structural, and mixed mutation. It cannot choose mutation magnitude, target controller modules, preserve subgraphs, change parent selection, or use recombination. This keeps attribution clean but limits the space of discoverable algorithms.

### 12.4 Short adaptive horizon

Structural innovations may need several descendant generations before becoming productive. A short post-event window can favor conservative weight refinement even when more disruptive variation would win over a longer evolutionary horizon.

### 12.5 Simplified energetic model

Resource intake, basal cost, and squared actuation cost provide ecological selection, but they are abstractions rather than a physically closed thermodynamic model. Claims concern embodied artificial evolution, not biological metabolism.

### 12.6 No end-to-end differentiability claim

The simulator retains differentiable continuous kernels, but the ecological process includes discrete transitions. Future work could exploit gradients within lifetimes or between births, but this experiment evaluates evolutionary program improvement.

---

## 13. What I would do next

The next experiment should not simply enlarge the same search. It should use the negative result to revise the scientific target while retaining the verifier discipline.

### 13.1 Replicate the outer search

Run multiple stable and punctuated Shinka archives with independent outer seeds. Treat outer run—not candidate generation—as the unit for claims about meta-training regime.

### 13.2 Vary founder populations

Evaluate frozen policies across multiple independently generated founder panels. This would test whether the scheduler transfers beyond one reservoir of standing behavioral variation.

### 13.3 Give heredity a finer action space

Allow the bounded policy to control:

- parametric mutation scale;
- probability and number of structural changes;
- separate rates for sensory and locomotor subgraphs;
- temporary exploration pulses that persist for several births;
- module-preserving crossover after asexual results are understood.

The trusted implementation should still apply and validate every mutation.

### 13.4 Lengthen the evolutionary horizon

Use longer post-shock windows and measure whether early productivity sacrifices enable later recovery. A multi-timescale analysis could separate immediate resistance, demographic recovery, and genuine descendant adaptation.

### 13.5 Preregister a resilience-specific objective

The current score intentionally rewards absolute post-event productivity. A future experiment could preregister a robust aggregate that jointly rewards absolute performance and paired shock resilience. This should be a new experiment, not a retroactive change to the present metric.

### 13.6 Run common-garden assays after a positive result

If a future policy produces a reliable advantage:

1. save pre-shock ancestors;
2. trace the ancestors of successful post-shock descendants;
3. test both in identical pre- and post-shock gardens;
4. look for a descendant-by-environment interaction;
5. ablate the discovered code mechanism.

That would distinguish heritable adaptation from survival of a pre-existing resistant lineage.

### 13.7 Add coevolving stress tests only after replication

An embodied Red Queen extension could search for physically valid environments that discriminate the current policy archive. Anchor tasks and difficulty constraints would be essential to prevent the environment generator from proposing impossible worlds.

---

## 14. Conclusion

Evo²-Ecosystem was built to test a specific form of recursive improvement: an AI system improving the heredity mechanism through which an embodied population generates future adaptive descendants.

The project contributes:

1. a finite-resource, multi-generation ecological extension to Microcosmos;
2. fixed-shape GPU-native CPPN inheritance with TensorNEAT parametric and structural mutation;
3. environmental shocks with exact paired null counterfactuals;
4. a small, secure ShinkaEvolve task over an algorithmically meaningful heredity boundary;
5. a preregistered stable-versus-punctuated program-evolution experiment;
6. a sealed six-policy evaluation with an auditable negative result.

The experiment did not show that punctuated meta-training improved catastrophe recovery. Fixed parametric mutation remained the strongest aggregate baseline, and the two Shinka champions were statistically indistinguishable on the primary comparisons.

The result nevertheless demonstrates the full loop the project set out to build:

```text
AI proposes an executable improvement rule
                    ↓
The rule changes how future organisms inherit variation
                    ↓
Embodied ecology supplies objective consequences
                    ↓
Those consequences guide further program evolution
```

The lesson is not that algorithm discovery failed. It is that a credible self-improvement result requires more than a plausible discovered program or a strong training score. It requires an immutable verifier, matched budgets, hidden worlds, competitive baselines, and a willingness to accept the answer those controls produce.

That combination—creative algorithm generation constrained by rigorous executable evidence—is the aspect of recursive self-improvement I most wanted this project to investigate.

---

## Appendix A: Reproducibility and artifact map

### A.1 Core reports and specifications

| Artifact | Repository path |
|---|---|
| Preregistered analysis plan | `plans/experiment-analysis-plan.md` |
| Master technical design | `plans/evo2-ecosystem-master-prd.md` |
| Code walkthrough | `plans/evo2-ecosystem-code-walkthrough.md` |
| Frozen shared run specification | `ShinkaEvolve/examples/evo2_ecosystem/run_spec.json` |
| Shinka task documentation | `ShinkaEvolve/examples/evo2_ecosystem/README.md` |
| Runtime amendment audit | `microcosmos/benchmarks/evidence/evo2_training_runtime_amendment.md` |

### A.2 Core implementation

| Component | Repository path |
|---|---|
| Ecosystem environment and lifecycle | `microcosmos/src/microcosmos/gym/ecosystem.py` |
| CPPN representation and inference | `microcosmos/src/microcosmos/cppn.py` |
| Trusted heredity operators | `microcosmos/src/microcosmos/heredity.py` |
| Ecosystem episode runner | `microcosmos/experiments/evo2_ecosystem/episode.py` |
| Frozen ecology configuration | `microcosmos/experiments/evo2_ecosystem/frozen_config.py` |
| Protocol and candidate sandbox | `microcosmos/experiments/evo2_ecosystem/protocol.py` |
| Analysis implementation | `microcosmos/experiments/evo2_ecosystem/analysis.py` |
| Shinka candidate evaluator | `ShinkaEvolve/examples/evo2_ecosystem/evaluate.py` |
| Matched outer-run launcher | `ShinkaEvolve/examples/evo2_ecosystem/run_evo.py` |

### A.3 Frozen finalists and final evidence

| Artifact | Repository path |
|---|---|
| Stable Shinka champion | `ShinkaEvolve/examples/evo2_ecosystem/frozen/evo2-production-20260712-r1/stable/main.py` |
| Punctuated Shinka champion | `ShinkaEvolve/examples/evo2_ecosystem/frozen/evo2-production-20260712-r1/punctuated/main.py` |
| Sealed suite ledger | `microcosmos/experiments/evo2_sealed/final_results/suite_record.json` |
| Atomic policy records | `microcosmos/experiments/evo2_sealed/final_results/*.json` |
| Canonical sealed summary | `microcosmos/experiments/evo2_sealed/final_results/summary.json` |

### A.4 Test coverage

Key focused tests include:

- `microcosmos/tests/test_ecosystem.py`
- `microcosmos/tests/test_ecosystem_long_rollout.py`
- `microcosmos/tests/test_cppn_controller.py`
- `microcosmos/tests/test_heredity.py`
- `microcosmos/tests/test_evo2_episode.py`
- `microcosmos/tests/test_evo2_protocol.py`
- `microcosmos/tests/test_evo2_analysis.py`
- `microcosmos/tests/test_evo2_sealed.py`

All Python commands were run from the shared `sakana` Conda environment. Full regressions used CPU JAX; focused simulations and experiments used fresh, memory-guarded GPU processes.

---

## Appendix B: Concise technical pseudocode

### B.1 Lifecycle step

```python
for each fixed-shape simulation step:
    observations = sense(body_state, resource_field)
    bending = batched_cppn_inference(genomes, observations)
    body_state, fluid = advance_masked_physics(bending, alive)

    uptake, resource_field = allocate_finite_resource(mouths, alive)
    energy += uptake - basal_cost - actuation_cost(bending)
    alive &= energy > 0

at each bounded birth boundary:
    parents = eligible_living_organisms(energy, age)
    slots = inactive_slots(alive)
    for each bounded parent-slot pair:
        operator_scores = candidate_policy(safe_parent_stats, safe_population_stats)
        operator = trusted_sample(operator_scores, keyed_rng)
        child_cppn = trusted_tensorneat_mutation(parent_cppn, operator)
        initialize_child(slot, child_cppn, energy_conserving_endowment)
        record_lineage_event(parent_id, child_id)
```

### B.2 Shock/null pairing

```python
pre_event_state = run_to_event_boundary(initial_world)

shock_state = exact_copy(pre_event_state)
null_state = exact_copy(pre_event_state)

apply_declared_event(shock_state)
apply_null_event(null_state)

shock_result = run_post_event(shock_state)
null_result = run_post_event(null_state)
```

---

## References

1. Chris Lu et al., [*Towards End-to-End Automation of AI Research*](https://www.nature.com/articles/s41586-026-10265-5), *Nature* 651, 914–919 (2026).
2. Sakana AI, [*The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*](https://sakana.ai/ai-scientist/).
3. Sakana AI, [*ShinkaEvolve: Evolving New Algorithms with LLMs, Orders of Magnitude More Efficiently*](https://sakana.ai/shinka-evolve/).
4. Sakana AI, [*The Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*](https://sakana.ai/dgm/).
5. Sakana AI, [*Digital Red Queen: Adversarial Program Evolution in Core War with LLMs*](https://sakana.ai/drq/).
6. Mark Tensen et al., [*Microcosmos: Reimagining Artificial Life for the GPU Era*](https://arxiv.org/abs/2607.02954), 2026.
