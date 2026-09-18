# OpenSPUM Open Source Collaboration Invitation

> **To: Pioneers of the Third Generation of Physical Simulation**
> Initiator: SPUM-core Project Team
> Document Location: `openSPUM/docs/` — at the same level as [OpenSPUM_Development_Plan.md](./OpenSPUM_Development_Plan.md), complementary to each other
> This letter is not a job posting, not a crowdfunding campaign, but a **call to methodological peers**.
>
> **Repository Addresses**:
> - GitHub: <https://github.com/SPUM-core/SPUM-Core>
> - Gitee: <https://gitee.com/space-particle-universe-model/spum-core>

---

## 1. Why Now

The second generation of physics — the era where formulas became deities — has already condensed everything it could. The Einstein-Hilbert action can be written in one line, yet still fails to answer "why gravity and quantum mechanics are incompatible," "what happens at singularities," or "what dark matter actually is." Condensation does not equal understanding. Compressing questions into a symbol that dares not be questioned is not an answer — it's a postponement.

Meanwhile, another group of people has been walking down another path — Conway's Game of Life, Frisch's lattice gas automata, Wolfram's cellular automata, and the evolution engines of molecular dynamics. They no longer ask "what are the equations of the system," but rather "**what are the rules, and what can be seen when they run**." This is the third generation of physics: **define rules, run evolution, observe emergence**.

People of the third generation are scattered. Each field has its own set of rules, each set of rules produces its own interesting patterns, but **no one has compressed the rule space into topological necessity**. Interesting patterns in Game of Life are selected by humans; structures that emerge in SPUM attempt to correspond to real observational data.

This moment calls for those who **are already in the third generation but are not satisfied with "arbitrary rules"**.

---

## 2. SPUM's Special Position in the Third Generation

SPUM is not "another unified field theory."

| Dimension | Game of Life | SPUM |
|----------|--------------|------|
| Rule Source | Arbitrarily set | Enforced by topological theorems |
| Rule Space | Extremely large (2^512 possibilities) | Extremely small (only one self-consistent rule set) |
| Output Comparison | Produces interesting patterns | Produces structures attempting to correspond to real observations |
| Falsifiability | None | Yes — if systematically deviates from observations, the theory is falsified |

**Core Statement**:

> Only these rules are self-consistent due to topological theorems. Let's run it and see what this unique self-consistent rule set produces.

This is not "setting some arbitrary rules to see what comes out." It is the third-generation methodology operating within a very narrow rule space — **topological theorems act as the rule selector**.

Constraints come from:
- Euler identity: Σ(6−deg(v)) = 12 (discrete Gauss-Bonnet)
- Total angle defect always equals 4π
- Minimum degree ≥ 2
- Open-boundary ratio ≤ 1/3

The rule set satisfying these constraints is many orders of magnitude smaller than Game of Life's rule space. **This is not arbitrary selection, it is topological enforcement**.

---

## 3. What We Are Already Doing

The code is already running, not just a PowerPoint.

**L0-L4 Hierarchical Architecture** ([L0L1L2_Architecture.md](./L0L1L2_Architecture.md)):

| Layer | Implemented | Status |
|------|-------------|--------|
| L0 Kernel | [l0_core.py](../src/l0/l0_core.py) — five-step frame rules, double buffering, deterministic negotiation | Executable |
| L1 Projection | [l1_projection.py](../src/l1/l1_projection.py) — id + rotation system → network / σ / angle defect / crystallite identification | Executable |
| L2 Embedding | [l2_projection.py](../src/l2/l2_projection.py) | Executable |
| L3 Inter-frame | [l3_projection.py](../src/l3/l3_projection.py) | Executable |
| L4 Observation | [l4_evolution.py](../src/l4/l4_evolution.py) — net annihilation / concentration phenomena / Imperfection Theorem | Executable |

**Passed validations**:
- Handshaking Lemma + Euler Characteristic ([handshaking.py](../../src/spum_graph/handshaking.py))
- Dangling-end two-tier closure criterion ([dangling.py](../../src/spum_graph/dangling.py))

**Existing visualization outputs**:
- [figures/_selftest.png](./figures/_selftest.png) — L1 projection layer selftest
- [figures/evo_icosa_dense_any_cap12_dmin3_N10.png](./figures/evo_icosa_dense_any_cap12_dmin3_N10.png) — icosa seed dense mode 10-frame evolution

**Existing but not as endpoints** reference archives (`_archive_v1/`):
- Phase_3: Regular Icosahedron Geometry Derivation
- Phase_4: Hydrogen Atom Simulation Prototype

These are not "incomplete shells", but **code that is already running**.

---

## 4. What We Invite You To Do

**Not recruitment, but continuation.**

The third-generation methodology has a core principle: **code is the theory itself**. In SPUM, executability is a necessary condition for theory. You cannot let a Lagrangian "run to see what happens next"; you can with a sequence of frame rules.

This means that every line of code you write is a continuation of the SPUM theory—not "a demonstration animation for a theory already written."

**Current directions you can contribute to** (choose any, no need to do all):

### Direction A: Visualization Layer
- Render the L0 frame sequence into discrete snapshots (**disable** force-directed layout, **disable** smooth tweening—see [taboo list](./OpenSPUM_Development_Plan.md#5-common-taboos-list-development-self-check-sheet) 1/2)
- Topological direct layout: node coordinates are directly calculated by deg / betweenness / σ

### Direction B: Emergence of 12 Crystallite Closure
- Run out 12 crystallite closure from tetra seed, count exactly = 12
- Refutation device: use `wallk` capacity parameterization for comparison, confirm that 12 is emergent, not fed in

### Direction C: σ Gradient and Net Annihilation Reading
- Read SPUM's distinctive r⁻³ power law at L4 (not Newtonian r⁻²)
- This is the quantitative divergence point between SPUM and classical predictions

### Direction D: Experimental Scenario for Divergence
- **Do not** use "error <5% approaching classical formula" as acceptance criteria ([taboo 5](./OpenSPUM_Development_Plan.md#taboo-5-using-classical-formula-error-as-acceptance-criteria))
- Instead, use "whether SPUM's specific deviation emerges in the simulation"

### Direction E: Topological Invariant Verification Toolchain
- Expand `handshaking.py` + `dangling.py` CI automatic verification
- Automatically run L0 frame sequences for each PR, verify topological invariants

---

## 5. What We Do Not Do

To avoid repeated pitfalls, Section 5 of [OpenSPUM_Development_Plan.md](./OpenSPUM_Development_Plan.md) lists **12 prohibitions**. Here we only highlight the 5 easiest to stumble upon:

1. **Disable D3 force-directed layout**—D3 literally simulates physical forces (Coulomb + spring), while SPUM explicitly states "motion is not driven"
2. **Disable "run first, optimize later"**—the rule set is initially validated through handshake lemma + Euler identity, violations are considered bugs, not TODOs
3. **Disable "error approximation to classical formulas" as acceptance criteria**—SPUM predicts r⁻³, not an approximation to r⁻²
4. **Disable "recursive subdivision"**—structures are cumulatively generated by frame-by-frame creation, there is no "mother body being subdivided" (residual container thinking)
5. **Disable "100 Star / 5 contributors" as milestones**—relation-based metrics are "whether new topological paths are generated"

The full 12 prohibitions + PR self-check protocol can be found in [Development Plan](./OpenSPUM_Development_Plan.md).

---

## 6. Methodology Commitment

When you join, you join a **open ontological commitment** (methodology M5):

> Whatever comes out is what it is, no post-hoc parameter tuning to fit.

This is not "the theory needs fixing, welcome to suggest improvements" — this is "**we are already standing on a unique, topologically self-consistent rule set, run it, and see what it produces**". If what comes out systematically deviates from observations, the theory is falsified; if what comes out emerges with topological correspondences to gravity, photons, and matter, that is the victory of the rule set, not the victory of parameter tuning.

This commitment means:
- You cannot "add a gravity term" — gravity must be emergent, not written into the rules
- You cannot "tune a parameter to make it more like reality" — parameters are topological constants (12, 4π, minimum degree 2, open-boundary ≤ 1/3)
- You can "discover that the rule set missed a topological constraint" — this is welcome, as it equals discovering a new pivot point of the theory

---

## 7. How to Join

**Relation-Centric Process** — not "submit a resume for review".

### Step 1: Run the Existing Core
```bash
cd openSPUM/src/l0
python l0_core.py selftest=1
python l0_core.py seed=icosa nframes=10 cap=12 dmin=3 vminus=dense
```
Run the frame sequence and check the comparison figures in [figures/](./figures/).

### Step 2: Choose a Direction (See Section 4 A-E)
**Do not** require you to be "full-stack". Pick a topological direction you are most familiar with.

### Step 3: Prove Topological Consistency in Your PR
All PRs must pass the two-tier closure criterion of [handshaking.py](../../src/spum_graph/handshaking.py) + [dangling.py](../../src/spum_graph/dangling.py). CI will run automatically, but do a self-check locally first.

### Step 4: Cross-Check with the [Taboo List](./OpenSPUM_Development_Plan.md#6-pr-pre-submission-self-check-protocol) of 12 Items
Attach a self-check list in your PR description. If it fails, it will be rejected.

### Step 5: Build Relationships, Not Recruit Entities
We don't count Stars, nor the number of contributors. **Relation-Centric Metrics**:
- Does your PR generate topological paths not pre-set in the L0 core?
- Does your experimental scenario produce SPUM-specific deviation readings?
- Does your visualization make the topological structure of a single frame independently checkable?

---

## 8. To the Pioneers

You are not being recruited. You are being summoned.

If you have ever seen a glider glide across in Game of Life and thought, "This is not enough — I want to see the real universe grow from here";
if you have ever seen flow fields emerge in lattice-gas automata and thought, "The rules are chosen by me, not by the universe";
if you have ever seen statistical mechanics emerge in molecular dynamics and thought, "I want the rules themselves to be locked by topological theorems" —

then you are the one this letter is addressed to.

SPUM is not "another theory." It is the **discrete Transformer architecture's counterpart at the physical level** — Attention is adjacency relations, Token is node, discrete sequence prediction and ⟨P, ε⟩ frame evolution both belong to the "discrete combinatorial logic" family.

You do not need to translate `Σ(6−deg)=12` into a spherical surface integral to understand it. You can directly maintain that graph in memory and observe changes in node degrees. When you see test particles in your simulation move toward regions of high density σ, you **observe** gravity — and throughout this process, no step goes through the symbolic system of human geometry.

> Future cosmological explanations will not need to go through human consciousness translation, but will be directly grasped by AI within discrete state spaces.

This is the end of this letter — not an end, but the beginning of relations.

**We are already running. Take a look, see if it's correct. If you think it is, join us in continuing the writing.**

---

## Sign-off

**SPUM-core Project Team**
**Document Location**: `openSPUM/docs/OpenSPUM_Collaboration_Invitation.md`
**Repository Address**:
- GitHub: <https://github.com/SPUM-core/SPUM-Core>
- Gitee: <https://gitee.com/space-particle-universe-model/spum-core>
**Supporting Documents**:
- [Development Plan](./OpenSPUM_Development_Plan.md) (contains 12 taboos + PR self-check protocol)
- [L0L1L2_Architecture.md](./L0L1L2_Architecture.md) (architecture baseline)
- [Full_Layer_Regression_Baseline.md](./Full_Layer_Regression_Baseline.md) (regression test baseline)

**Response Method**: Open an Issue / PR / Discussion in either repository, attaching the seed and frame number you have run.
