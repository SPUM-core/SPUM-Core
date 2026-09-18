# OpenSPUM Development Plan

> **Methodological Benchmark**: Code is the theory itself — not "tools for SPUM", but "L0-L4 implementations continuing SPUM theory".
> **Architecture Benchmark**: [L0L1L2_Architecture.md](../L0L1L2_Architecture.md), layered as L0 kernel / L1 intra-frame projection / L2 embedded projection / L3 inter-frame projection / L4 evolution observation.
> **This document location**: `openSPUM/docs/figures/` — in the same directory as visualization outputs, for developers to cross-reference while viewing diagrams and constraints.

---

## 1. Current Status: Implemented L0-L4

| Layer | File | Implemented Content |
|------|------|---------------------|
| L0 | [l0_core.py](../../src/l0/l0_core.py) | Current implementation of the five-step frame rules: propose → resolve → commit → prune, double buffering, deterministic negotiation style |
| L1 | [l1_projection.py](../../src/l1/l1_projection.py) | id + rotation system → network / local frame / σ / angle defect / crystallites |
| L2 | [l2_projection.py](../../src/l2/l2_projection.py) | Embedded projection (coordinates) |
| L3 | [l3_projection.py](../../src/l3/l3_projection.py) | Inter-frame projection |
| L4 | [l4_evolution.py](../../src/l4/l4_evolution.py) | Net annihilation / concentrated phenomena / Imperfection Theorem (cross-frame read-only observation) |

**v1 Archive Reference** (not used directly, only for derivation reference):
- [Phase_3/icosahedron_derivation.py](../../_archive_v1/Phase_3/icosahedron_derivation.py) — Regular icosahedron geometry → shell capacity 2n² derivation
- [Phase_4/simulate_hydrogen.py](../../_archive_v1/Phase_4/simulate_hydrogen.py) — Hydrogen atom simulation prototype

**SPUM-Graph Theory Verification Tools** (located at `src/spum_graph/`):
- [graph.py](../../../src/spum_graph/graph.py) — FrameGraph, Python implementation of 5 axioms
- [dangling.py](../../../src/spum_graph/dangling.py) — DanglingDetector, dangling-end detection + two-tier closure criterion
- [handshaking.py](../../../src/spum_graph/handshaking.py) — HandshakingVerifier, handshake lemma + Euler characteristic + Σ(6−deg)

---

## 2. Six-Stage Roadmap

### Stage Zero: Compliance Packaging of the Existing Kernel

**Objective**: Wrap the L0 frame evolution kernel into a stable public interface without rewriting it.

**Key Tasks**:
1. Expose L0's `propose → resolve → commit → prune` four-step process as a stable API
2. Standardize L1 `project()` output: `edge_list` / `local_frame` / `sigma_of` / `angle_defect` / `crystallites`
3. Standardize L4 `observe(core, nframes)` output as CSV/JSON sequences

**Acceptance Criteria**:
- Each frame of L0 passes `handshaking.py` handshake lemma + Σ(6−deg)=12 validation
- Run out repeatable frame sequences from seeds (icosa / patch / tetra)
- **No "run first, optimize later"** — violation is considered a bug

### Stage One: Discrete Snapshot Visualization

**Objective**: Make the L0 frame sequence visible.

**Tech Stack** (mandatory choices):
- **Disable** D3 force-directed layout (see taboo 1)
- **Disable** smooth tweening (see taboo 2)
- **Use** topological direct layout: node coordinates calculated directly from `deg(v)` / `betweenness` / `σ`
- **Use** stop-motion discrete snapshots: one static image per frame, with explicit frame numbering

**Key Modules**:
1. Topological Layout Engine (minimum tension embedding based on degree + betweenness)
2. Frame Snapshot Renderer (SVG / Canvas)
3. Frame Sequence Player (switch by frame number, no interpolation)
4. Topological Metrics Panel (δ / Δμ / d_topo / σ)

**Milestone**: Run any seed for 100 frames, outputting independently checkable snapshots per frame.

### Stage Two: Construction of the 12-Crystallite Closed Loop

**Objective**: Observe the emergence of the 12-crystallite closed loop within the L0 kernel — SPUM's first "atom".

**Key Corrections**:
- Not starting from "four-ball closed loop" (see taboo 6)
- The endpoint is the 12-crystallite closed loop (regular icosahedron), enforced by Euler identity
- No "recursive subdivision" — use frame-by-frame generation (see taboo 7)

**Key Modules**:
1. Seed Injector (tetra / patch / icosa)
2. Crystallite Identifier (L1 already has `crystallites`)
3. Closed Loop Validator (virtual face ratio ≤ 1/3)
4. Refutation Device (L0's `wallk` capacity parameterized comparison)

**Milestone**: Run from tetra seed to produce a closed loop of exactly 12 crystallites, not 11 or 13.

### Stage Three: Observation of σ Gradient and Net Annihilation

**Objective**: Read SPUM-specific physical readings at L4.

**Key Modules**:
1. σ Field Extractor (L1 `sigma_of` already implemented)
2. Net Annihilation Locator (L4 `annihilation_profile` already implemented)
3. Radial Reader (σ(r) curve)
4. Power Fitter (σ(r) = σ₀ + km/r² → a = −2kc²m/r³)

**Milestone**: Fit out the SPUM-predicted **r⁻³ power law**, **not** r⁻². This is the quantitative divergence point from Newtonian gravity.

### Stage Four: Experimental Scenarios — Seeking Divergence Rather Than Approaching Classical

**Objective**: Use specific scenarios to verify where SPUM systematically diverges from classical predictions. This is the fulfillment of M5 (open ontological commitment).

**Revised Acceptance Criteria**:
- **Disable** "<5% error approximation to classical formulas" as acceptance (see taboo 5)
- **Use** "whether SPUM's specific divergence emerges in simulations" as acceptance

**Priority Scenarios**:

| Scenario | Classical Prediction | SPUM Prediction | Acceptance Criterion |
|--------|----------------------|------------------|----------------------|
| Gravitational Power Law | a ∝ r⁻² | a ∝ r⁻³ | Run out r⁻³, not approaching r⁻² |
| Redshift Accumulation | z ∝ distance (linear Hubble) | z ∝ Σ(σ−σ₀)/σ₀ | Run out aggregate reading form |
| Weak Field Degradation | — | r⁻³ approximates r⁻² numerically in weak field | Validate degradation consistency |

**Disabled Scenarios** (conceptually inconsistent):
- Spring oscillator T=2π√(m/k): uses π, which is a cognitive projection not an ontological constant (see taboo 8)
- Archimedes' buoyancy F=ρgV: contains g (external force), violating "no force-driven" (see taboo 12)

### Stage Five: Open Source Collaboration — Relation-Centric Metrics

**Objective**: Lower the participation threshold, but **do not use entity-based metrics** as milestones.

**Key Corrections**:
- **Disable** "100 Star / 5 contributors" as milestones (see taboo 9)
- **Use** relation-centric metrics: whether external PRs generate topological paths not covered by L0

**Key Actions**:
1. Document public interfaces of L0-L4 (Sphinx / MkDocs)
2. Write a minimal reproducible example from seed to 12 crystallites
3. Contribution guidelines: all PRs pass `handshaking.py` + `dangling.py` two-tier closure criteria
4. CI: automatically run L0 frame sequences for each PR, validate topological invariants

**Milestone** (relation-centric):
- At least 1 external PR runs out a topological path not pre-set by L0
- CI topological validation pass rate 100%

---

## 3. Completion Criteria (Alternative to Timeline)

> **Remove "1-2 months / 2-3 months" etc.** — start/end thinking is an anti-pattern (see taboo 4).
> Use completion criteria instead.

| Phase | Completion Criteria (Non-temporal) |
|------|---------------------|
| Zeroth Phase | Every frame in the L0 frame sequence passes the Σ(6−deg)=12 check |
| First Phase | Any seed can generate 100 discrete snapshots that are independently verifiable |
| Second Phase | The 12-crystallite closed loop emerges from a tetra seed, with exact count = 12 |
| Third Phase | The power law fit of σ(r) is −3, distinguishable from Newtonian r⁻² |
| Fourth Phase | At least one SPUM-specific deviation reading emerges in the simulation |
| Fifth Phase | At least 1 external PR runs a path not pre-set in L0 |

---

## 4. Key Principles

1. **Code is theory**: Based on L0-L4 extensions, no rebuilding. What's added isn't a "tool," but a continuation of the theory.
2. **Topological self-consistency first**: The rule set is initially validated through the handshaking lemma + Euler identity.
3. **Discrete snapshots first**: Use stop-motion for visualization, disable smooth tweening.
4. **Direct topological layout**: Disable force-directed layouts; node coordinates are directly calculated from deg/betweenness/σ.
5. **Seeking divergence**: Experimental validation seeks SPUM-specific deviations, not approaching classical formulas.
6. **Creation accumulation**: Disable "recursive subdivision"; structures are generated by accumulating V⁺ frame by frame.
7. **Relation-centric metrics**: Open-source milestones don't use Star counts, but whether new topological paths are generated.

---

## 5. Common Taboos List (Development Self-Check Sheet)

> Each entry corresponds to a SPUM anti-pattern. **Check this list before submitting every PR.**
> Format: ❌ Wrong practice → ✅ Correct practice → Self-check method.

---

### Taboo 1: Using force-directed layout to render the network

**Corresponding anti-pattern**: Pattern 2 (Residual force-driven thinking)

❌ **Wrong practice**
"Use D3.js force-directed layout, nodes automatically arrange, connection relationships are clear at a glance."

Why it's wrong: D3 force-directed layout literally simulates physical forces (Coulomb repulsion + spring attraction). SPUM explicitly states that "motion is not driven, but coordinate updates produced by deletion and creation events." Using force-directed layout to render the ontology is inserting old paradigm visualization back in.

✅ **Correct practice**
Node coordinates are directly calculated from topological invariants:
- Primary axis: `deg(v)` sorted
- Secondary axis: `betweenness(v)` sorted
- Third dimension: `σ(v)` level
- Inter-frame displacement: Directly mapped from V⁺/V⁻ events, without introducing force as a mediator

**Self-check**: grep code for `force` / `repulsion` / `attraction` / `charge` / `gravity` (in layout context). If found, it's not passing.

---

### Taboo 2: Using smooth animation to show frame evolution

**Corresponding anti-pattern**: Pattern 3 (Residual continuum thinking)

❌ **Wrong practice**
"Use tweening to interpolate between frames, making the network evolve smoothly."

Why it's wrong: Three continuum traps—"smooth" implies continuous change; "interpolation" implies intermediate states between frames; "evolution" in user perception is replaced with "flow." SPUM explicitly states that there are no intermediate states between frames—nodes are either connected or not.

✅ **Correct practice**
Use stop-motion discrete snapshots:
- One static image per frame
- Frame numbers explicitly labeled (Frame #42)
- No transition animation when switching
- Users can move forward/backward/jump in frame numbers

**Self-check**: Check player code for `requestAnimationFrame` used for interpolation. If yes, it's not passing.

---

### Taboo 3: "Run first, optimize later" development pace

**Corresponding anti-pattern**: Violating methodology M4 (Topological constraints as rule selectors)

❌ **Wrong practice**
"First define a random set of rules to get it running, then adjust them to be SPUM-consistent later."

Why it's wrong: M4 principle—topological invariants (Σ(6−deg)=12, minimum degree ≥ 2, open-boundary ratio ≤ 1/3) are hard constraints. The only self-consistent rule set is one that satisfies all these constraints simultaneously. You cannot "run with an incorrect rule set"—what you run is not SPUM, but something else.

✅ **Correct practice**
The rule set must **from the start** pass `handshaking.py` + `dangling.py` two-tier closure criterion. Any modification must immediately run validation; violations are considered bugs, not TODOs.

**Self-check**: CI must have 100% topological validation passing, no skips allowed.

---

### Taboo 4: Using a timeline as milestones

**Corresponding anti-pattern**: Pattern 6 (Start/end thinking)

❌ **Wrong practice**
"Phase 1 will be completed in 1–2 months, Phase 2 in 2–3 months, Phase 5 in 4–6 months."

Why it's wrong: Implies a "start" and an "end." SPUM explicitly states that change is continuous, without beginning or end—the current state always evolves from the previous one. Binding development progress to a timeline projects continuum time onto project management.

✅ **Correct practice**
Use **completion criteria** instead of timelines:
- "Each frame in L0 passes Σ(6−deg)=12 validation" (not "completed within two months")
- "12 crystallite closure emerges from tetra seed, count exactly = 12" (not "completed in three months")

**Self-check**: If milestone descriptions include "within months," "within weeks," or "by end of 2026," it's not passing.

---

### Taboo 5: Using classical formula error as acceptance criteria

**Corresponding anti-pattern**: Violating methodology M5 (Ontological commitment is open)

❌ **Wrong practice**
"Simulation results have an error < 5% compared to Newtonian theory, and Archimedes' buoyancy experiment has an error < 5%."

Why it's wrong: SPUM explicitly predicts a different power law from Newtonian gravity (a = −2kc²m/r³ vs −GM/r²). Expecting SPUM to produce results with <5% error compared to classical formulas is treating SPUM as a "numerical approximation of classical formulas." M5 principle: "What you run is what it is, no parameter adjustment allowed for fitting."

✅ **Correct practice**
Change acceptance criteria to "Does the specific deviation predicted by SPUM emerge in simulation?"
- Gravitational power law: Run out r⁻³, **not** approaching r⁻²
- Redshift form: Run out z ∝ Σ(σ−σ₀)/σ₀
- Weak-field degeneracy: Verify that r⁻³ numerically approaches r⁻² in weak-field approximation (consistency, not approximation)

**Self-check**: If acceptance documents include "error <," "approximate," or "consistent with XX formula," it's not passing.

---

### Taboo 6: Treating "four-ball closure" as the core structure of SPUM

**Corresponding anti-pattern**: Conceptual hierarchy confusion

❌ **Wrong practice**
"The four-ball closure constructor is the starting point of SPUM's geometric construction."

Why it's wrong: SPUM's core axiom explicitly states that the **first "atom" is the 12-crystallite closure (icosahedron)**, not four balls. Four balls are only an **intermediate derivation step** in [Phase_3/icosahedron_derivation.py](../../_archive_v1/Phase_3/icosahedron_derivation.py), not the end structure. Treating "four-ball" as SPUM's core is equating scaffolding with the building itself.

✅ **Correct practice**
- Starting point: Minimal seed (tetra / patch / icosa)
- End point: 12-crystallite closure (icosahedron shell)
- Intermediate steps (four-ball, etc.) are only for derivation reference, not visualization endpoints

**Self-check**: If roadmap or documentation lists "four-ball closure" as a core module, it's not passing.

---

### Taboo 7: Using "Recursive Subdivision" to Describe Structural Growth

**Corresponding Anti-pattern**: Pattern 1 (Residual Container Thinking)

❌ **Incorrect Approach**
"Recursively subdivide the shell layer to simulate the self-similar structure of spatial particles."

Why it's wrong: Implies there is a pre-existing container being split. SPUM's first axiom — there is no background container; bubbles themselves are space. Structures are generated by the **accumulation of creation events**, not from existing structures being "subdivided".

✅ **Correct Approach**
Use "Frame-by-Frame Creation Accumulation":
- Structure = trace of V⁺ events accumulated over a sequence of frames
- There is no "mother body" being subdivided
- Self-similarity arises from the topological constraints of the rule set, not geometric subdivision

**Self-check**: Are there instances of "subdivision", "subdivide", or "recursively split" in code or documentation? If so, it fails.

---

### Taboo 8: Using Formulas with π as a Benchmark

**Corresponding Anti-pattern**: Pattern 5 (Mathematical Physicalization)

❌ **Incorrect Approach**
"Use T=2π√(m/k) as the acceptance criterion for the spring oscillator experiment."

Why it's wrong: SPUM explicitly states that π is a cognitive projection, not a universal constant. The universe's ontology only records integers like 12 and topological constraints. Using formulas with π as a benchmark for SPUM simulations equates to using the cognitive projection layer to validate the ontic layer, which is a category error.

✅ **Correct Approach**
- Prohibit formulas containing π from being used as SPUM acceptance benchmarks
- Instead, use SPUM's own predicted reading forms (e.g., r⁻³ power law, σ aggregate readings)
- If a classical formula must be compared, first strip out π (use proportion / power law / discrete counting)

**Self-check**: Does the acceptance benchmark formula contain π? If yes, it fails.

---

### Taboo 9: Using Entity-based Metrics as Open Source Milestones

**Corresponding Anti-pattern**: Pattern 7 (Entity Priority Over Relations)

❌ **Incorrect Approach**
"GitHub repository gains at least 100 stars and has at least 5 external contributors who submitted PRs."

Why it's wrong: Treating "star count" or "contributor count" as success criteria. SPUM's relational primacy — the health of an open source project should be measured by whether "new structures are generated from relations," not by "whether the number of entities is sufficient."

✅ **Correct Approach**
Use relation-based metrics:
- "At least 1 external PR produces a topological path not pre-defined in the L0 core"
- "CI topological verification pass rate is 100%"
- "Community-contributed experimental scenarios produce SPUM-specific deviation readings"

**Self-check**: Are milestones using "X stars", "X contributors", or "X downloads"? If yes, it fails.

---

### Taboo 10: Rebuilding from Scratch Without L0-L4

**Corresponding Anti-pattern**: Methodological Misalignment (Code as Theory)

❌ **Incorrect Approach**
"Create Node class, Edge class, Network class, and build SPUM data structures from scratch."

Why it's wrong: SPUM methodology explicitly states that **code is the theory itself**. L0-L4 are already the current implementation of the theory; rebuilding from scratch equates to rewriting the theory and is highly prone to introducing pollution from old paradigms (e.g., Node class implies entity priority over relations).

✅ **Correct Approach**
- Extend based on [l0_core.py](../../src/l0/l0_core.py)'s `FrameGraph`
- Use the 5 axiom native implementation in [src/spum_graph/graph.py](../../../src/spum_graph/graph.py)
- Add new modules as inter-layer bridges for L0-L4, not starting from scratch

**Self-check**: Does the new code import `l0_core`, `FrameGraph`, or `HandshakingVerifier`? If not, it fails.

---

### Taboo 11: Using NetworkX/Neo4j as the Underlying Graph Library

**Corresponding Anti-pattern**: Ontological Pollution from Classical Graph Theory

❌ **Incorrect Approach**
"Use NetworkX as the graph theory foundation library and Neo4j as the storage backend."

Why it's wrong: SPUM-Graph Theory v2.0 differs from classical graph theory by "5 axioms + 12 systemic distinctions" — isolated nodes do not exist / dangling ends cannot be eliminated / graph states are frame snapshots / perfection is unattainable / local closure is sufficient. NetworkX contains assumptions from classical graph theory (isolated nodes can exist, graphs are global objects), directly using it would bring in old paradigms.

✅ **Correct Approach**
- Use [src/spum_graph/graph.py](../../../src/spum_graph/graph.py)'s `FrameGraph` (native implementation of the 5 axioms)
- Store using a sequence of frame snapshots (one per frame), not "global graph databases"
- If NetworkX must be used as an algorithm library, convert it to `FrameGraph` only at the boundary, not as the ontic layer

**Self-check**: Does the ontic layer code directly use `networkx.Graph`? If yes, it fails.

---

### Taboo 12: Using "Force" to Explain Any Change

**Corresponding Anti-pattern**: Pattern 2 (Residual Force-driven Thinking)

❌ **Incorrect Approach**
"Gravity is a fundamental force; dangling edges are logically deleted to maintain topological consistency."

Why it's wrong: Introducing "force" as a driving cause. SPUM explicitly states that "force and energy" are one of the seven cognitive prisons. Motion is not driven by forces, but is coordinate updates produced by deletion and creation events. The deletion of dangling edges is a logical enforcement of self-consistency, not a physical process.

✅ **Correct Approach**
- Prohibit using terms like "force", "drive", "attract", or "repel" to describe SPUM internal mechanisms
- Instead, use "event", "update", "delete", "creation", and "annihilation"
- Gravity = macroscopic reading of net annihilation effect (not a force)

**Self-check**: Does the documentation use "force", "drive", "attract", or "repel" to describe SPUM internal mechanisms? If yes, it fails.
(Note: This is related to Forbidden Practice 1's force-directed layout but focuses on conceptual expression, while Forbidden Practice 1 targets visualization implementation)

---

## 6. PR Pre-Submission Self-Check Protocol

Each PR must pass the following self-check, otherwise it will not be merged:

```
□ Taboo 1: Visualization does not use force-directed layout
□ Taboo 2: Frame evolution uses discrete snapshots, not tweening
□ Taboo 3: Rule set passes dual-layer verification via handshaking.py + dangling.py
□ Taboo 4: Milestones use completion criteria, not timelines
□ Taboo 5: Experimental acceptance seeks SPUM-specific deviations, not approaching classical formulas
□ Taboo 6: 12-crystallite closure is the endpoint, four-ball is just an intermediate step
□ Taboo 7: Use "creation accumulation", not "recursive subdivision"
□ Taboo 8: Acceptance criteria do not include π
□ Taboo 9: Open-source milestones do not use Star counts or other materialized metrics
□ Taboo 10: Extend based on L0-L4, not rebuild from scratch
□ Taboo 11: Ontic layer does not use NetworkX/Neo4j
□ Taboo 12: Do not use "force" to explain SPUM internal mechanisms
```

CI automatically executes code-level checks for taboos 1/2/3/8; the rest are reviewed manually against this table.
