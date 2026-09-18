# OpenSPUM — L0/L1/L2 Architecture Design

> Status: **v2.0** (2026-09-13)
> **L0 paradigm upgrade**: L0 is redefined as a **purely combinatorial 2D closed simplicial complex** — the core rule is "each edge belongs to exactly two faces", geometry is always the product of the projection layer (L1/L2). No floating points, coordinates, distances, or angles appear in L0.
> The v1.x "geometry-prioritized L0" (spherical angular coordinates `nb_dir` + K1–K8 kernel) has been downgraded to **Appendix A**, preserved as historical implementation and empirical records, no longer serving as a benchmark.
> Reference implementation: `src/l0/combinatorial_proto.py` (pure combinatorial prototype of L0, current, the only active file in `src/l0/`); old geometric L0 and spatial_proto have been archived to `_archive_v1/l0_legacy/` (see Appendix A).
> This document is the sole benchmark for the new implementation. When code conflicts with this document, this document takes precedence.

---

## 0. Archiving and Downgrading Records

### 0.1 Archived: v1 Simulator (`_archive_v1/`)

The fundamental issue of v1 architecture (`_archive_v1/Phase_0` + `universe/`) is not a constant error, but **treating the L1 projection as the L0 ontology**:

| # | v1 Practice | Why It Violates Ontology |
|---|-------------|--------------------------|
| 1 | `CRYSTALLITE_DEGREE_THRESHOLD = 42` global degree hard cap | Exclusivity is local: one edge occupies one direction, rejecting other edges from occupying the same direction — no need for a god's-eye counter |
| 2 | `step1_create` O(N³) global triple-gap scan | Gaps are local objects. The gap of particle i is determined by the distribution of neighbors on i, unrelated to distant particles |
| 3 | `run_full_frame` globally sequential execution of step1→5 | Global synchronous frames are narrative of the projection layer. Each particle in L0 only sees its neighbors; frames are synchronization protocols of L1 |
| 4 | 12 crystallite closure is achieved by aggregating channels (ForceDirectedRelaxation) "arranged" into a regular icosahedron | Using force-directed relaxation to arrange 12 balls into a regular icosahedron = external intervention, not emergence |
| 5 | `MAX_CONTACTS = 12` annotated as "emergent result" but used as constraint in code | Contradictory. 12 must emerge from topology, and must not appear in any if statement |

Archiving verdict: **v1 is an L1 simulator** — it simulates the "appearance" of the relational network, but does not implement the relational constraints themselves. All moved to `_archive_v1/`, no patches will be applied.

### 0.2 Downgraded: v1.x Geometric L0 (→ Appendix A)

Design before v2.0 (original §1–§7) avoided the global degree hard cap of v1, instead using **spherical angular coordinates** to express exclusivity: node states include neighbor directions `nb_dir[N×S] = (θ, φ)`, and exclusivity criterion is "angular distance ≥ α_j + α_k". This path advanced one step, but **still kept geometry in L0** — because (θ, φ) and angular radius α are geometric quantities.

This conflicts with the first principle of SPUM: "geometry is the product of relational networks in cognitive projection." Purely combinatorial L0 fully implements this principle:

| Item | v1.x Geometric L0 (Appendix A) | v2.0 Purely Combinatorial L0 (Main Body of This Document) |
|-----|-------------------------------|------------------------------------------------------------|
| L0 State | Adjacency slots + spherical angular coordinates `nb_dir` | Rotation system (each node's cyclic neighbor table) + face set |
| Exclusivity Expression | Angular distance ≥ α_j+α_k (angular inequality) | Each edge belongs to exactly two faces (combinatorial rule) |
| "Gap" | Angular area on the sphere not covered by the dome | Edge with k≥4 cycles (hole); true triangular face (face that can be coned) |
| Cross Criterion | No overlap within angular tolerance | Whether `χ = V−E+F` is conserved, whether `link` is a cycle |
| Radius r=f(deg) | Directly needed in L0 (affects α) | Not needed in L0; radius is a projection quantity |
| Frame-end Repair | K8a/K8b duality accounting + row-level lock | No repair needed: local deterministic sorting + bilateral commit, naturally order-agnostic |

Downgrading verdict: The geometric L0's K1–K8 and angular coordinate exclusivity theorem **are preserved in Appendix A as historical implementation and empirical evidence** (it indeed has empirical conclusions on exclusivity zero violation and determinism), but **no longer serve as the L0 benchmark**. Its open issue "radius law r=f(deg) not derived" (Appendix A §A.12.2) is **automatically resolved** in purely combinatorial L0 — because L0 no longer needs radius.

**Archiving Disposition (executed on 2026-09-13)**: Line B (geometric L0) and Line C (spatial_proto) are moved entirely to `_archive_v1/l0_legacy/` (preserved), `src/l0/` retains only the single active file `combinatorial_proto.py`. A full snapshot was saved separately to `src/l0_backup_20260913/` (safe copy, not involved in building) — this copy was cleaned and deleted on 2026-09-14, `_archive_v1/l0_legacy/` is archived for retention. List of archived files: `host.py`, `frame_kernels.cu`, `particle_state.cuh`, `smoke_test.py`, `test_l0.py`, `debug_frame1.py`, `debug_k2.py`, `diag_crystallite.py`, `diag_exclusivity.py`, `diag_manifold.py`, `spatial_proto.py`.

---

# Part I · L0: Purely Combinatorial 2D Closed Simplicial Complex

## 1. General Principles

1. **Edge-first**: Nodes are induced by relations, not presupposed. There is no node list — the node set is the induced result of the edge set.
2. **No randomness**: The output of a frame is a deterministic function of the initial state of the frame. The same initial state of a frame → the same final state of the frame.
3. **Intra-frame parallelism**: All negotiations occur simultaneously, with a unified submission at the end of the frame.
4. **Hard capacity constraints**: A2 exclusive occupation → each node has an upper limit on its degree.
5. **Each edge belongs to exactly two faces**: This is the core combinatorial rule of L0, and geometry is its projection.
6. **GPU execution, CPU parsing**: GPU performs parallel deterministic negotiations, while CPU handles projections, audits, and visualizations.

## 2. Ontology Structure

### 2.1 Three-Level Objects

L0 is not a regular graph, but a **two-dimensional closed simplicial complex**. The three basic objects are:

| Object | Definition | Constraints |
|--------|------------|-------------|
| Node | Endpoint of a relation confirmation | `deg ≥ 1`, `deg ≤ cap` |
| Edge | Tangential relationship between two nodes (exclusive confirmation event) | Belongs to exactly two faces |
| Face | Three mutually connected nodes | Exactly three edges |

The dependency direction is: **nodes are induced by edges, edges are constrained by faces, and faces are constrained by closure**.

### 2.2 Core Invariants

- **Edge-Face Rule**: Each edge belongs to exactly two faces.
- **Capacity Constraint**: Each node `deg ≤ cap`.
- **Closure Constraint**: No boundary edges, no boundary vertices.
- **Existence Criterion**: `deg = 0` means non-existence.

> The edge-face rule serves two purposes simultaneously, thus replacing the two independent constraints in v1.x: one is **non-crossing/inaccessibility** (two edges will not cross each other — crossing is combinatorially equivalent to χ drop or a bowtie), and the other is **global closure** (no exposed boundary edges).

### 2.3 State Representation

**Abstract layer**, the state of a frame:

- Edge set `E_t`
- Face set `F_t`
- Node degree `deg_t(v)`
- Node capacity `cap(v)`
- Local direction field (VSPT direction, optional)

**Implementation layer**: Use **rotation system** to carry all combinatorial information — each node stores a **rotation-order neighbor list** `rot[v] = [w_0, w_1, ...]` (counter-clockwise order), purely integer, no geometry, no floating point:

```python
rot[v]           # v's rotation-order neighbor list (neighbors around v in order)
succ(v, a)       # successor of a in v's rotation order
pred(v, a)       # predecessor of a in v's rotation order
E = Σ_v len(rot[v]) / 2      # number of edges
```

No node list is stored: the keys of `rot` are the active node set, `deg(v) = len(rot[v])`, and nodes with `deg = 0` do not exist (i.e., they don't exist).

**Faces are derived from the rotation system** (dart cycle): `dart = (v, w)`, `dart_next(v,w) = (w, succ(w, v))`. Starting from any dart and repeatedly taking `dart_next` until returning to itself gives a dart cycle = one face. Thus, "each edge belongs to exactly two faces" is implemented as: **each undirected edge {v,w} is used by exactly two dart cycles** (`(v,w)` and `(w,v)` each in one cycle).

### 2.4 Three χ-Preserving Combinatorial Operations + One Deletion Operation

All evolution of L0 is reduced to four purely combinatorial operations (all are integer-index insertions/deletions, geometry-independent). They share an invariant: **`χ = V − E + F` remains unchanged** (closed sphere `χ = 2`, `genus = 0`).

| Operation | Trigger Condition | ΔV | ΔE | ΔF | Δχ | Degree Born | Semantics |
|-----------|-------------------|----|----|----|----|-------------|-----------|
| `cone_tri_face(i,j,k)` (Type A: Face Coning) | (i,j,k) is a **true triangle face** | +1 | +3 | +2 | 0 | 3 | Cone a new node on an empty face, closed triangulation growth |
| `cone_hole(vs)` (Type B: Hole Coning) | `vs` is a k≥4 hole ring | +1 | +k | +(k−1) | 0 | k | Cone at the hole, closing the opening |
| `cone_edge(i,j)` (Edge Coning) | {i,j} is in some face, both ends `deg ≥ 2` | +1 | +2 | +1 | 0 | 2 | Boundary growth, the edge of a hole is replaced by a polyline |
| `remove_vertex(v)` (V⁻) | Active or pruning | −1 | −d | −(d−1) | 0 | — | Delete node and its d edges, merging its d faces into one d-gon |

> **Geometric meaning is only a post-hoc annotation**: A-type born deg-3 nodes correspond to "three spheres mutually tangent at the hole"; B-type deg-k corresponds to "k-gon opening closure"; edge coning's deg-2 node is a boundary node. The operations themselves only look at the rotation order, not angles.

**Minimum sufficient criterion for coning (core criterion of this L0)**:

```
Triangle (i,j,k) can be safely coned  ⟺  succ_j(i)==k  and  succ_k(j)==i  and  succ_i(k)==j
```

The three local successor checks all hold ⟺ the face containing `dart(i→j)` is exactly `(i→j)→(j→k)→(k→i)`. **Each node only checks its own rotation order**, purely local integer comparison.

If the criterion holds only partially ⟹ this 3-clique is a **separating triangle** (both sides are already occupied). Coning on a separating triangle = inserting a node into an already occupied position = **creating crossing** (χ drop / bowtie appearance). This was exactly the flaw of v1.x engine K2 (see §4.4 and Appendix A §A.5).

## 3. Negotiation Engine

### 3.1 Where Candidates Come From

L0 has no coordinates, so candidates can only come from **topological neighborhoods**. Deterministic expansion:

- **Shared Neighbor Pairs**: Two nodes have a common neighbor (a 3-clique) → candidate. In pure combinatorial implementation, this is a **true triangular face**.
- **Second-order Neighbors**: Neighbors of neighbors, excluding existing edges.
- **Gap Candidates**: Ends of gaps in locally closed subgraphs (holes with k ≥ 4).
- **Direction Candidates**: Nodes in the neighborhood pointed to by VSPT direction.
- **σ Gradient Candidates**: Nodes along the local σ gradient direction.

Each node expands its own candidate set in parallel. The size of the candidate set is determined by the local structure (`deg` is constrained by capacity, usually ≤ 12), and does not explode.

### 3.2 Candidate Types

Not all candidates are "add an edge." Candidates are divided into three types, **proposed and competed simultaneously**:

| Type | Action | Condition |
|------|--------|-----------|
| Creation | Add an edge | Both ends have remaining capacity, and do not violate the edge-face rule (cone on true face/hole) |
| Annihilation | Remove an edge | The edge is in a net annihilation zone, or it is a dangling edge |
| Reconnection | Remove one and add one | Maintain total edge count conservation |

### 3.3 Priority

Each candidate needs a deterministic priority `p(e)`. Requirements:

- **Symmetric**: `p(u,v) = p(v,u)`
- **Local**: Depends only on the local states of both ends
- **Deterministic**: Same local state → same priority
- **Comparable**: Any two candidates competing for the same node are comparable

Priority sources (examples, implementation can choose any combination):

- Sum of remaining capacities at both ends (`(cap(u)−deg(u)) + (cap(v)−deg(v))`)
- Deterministic combination of local σ
- VSPT direction compatibility
- Number of shared neighbors
- Gap size (hole ring length)

**Tie-breaking rules must be deterministic**: local topological hash, lexicographical order, or "neither accepts."

### 3.4 Conflict Resolution (Local Completion)

At each node locally:

1. Collect all candidates pointing to itself;
2. Deterministically sort by priority;
3. Take the first `cap(v) − deg(v)` ones;
4. Reject or delay the rest.

Key: **Decisions are completed locally, independent of global order.**

### 3.5 Bilateral Commit

A candidate is valid **if and only if both ends accept**:

- Intersection operation, deterministic, parallel, order-independent.
- No atomic competition, no dependency on arrival order.

> This is a key simplification in v2.0 compared to v1.x: there is no need for row-level spin locks (Appendix A §A.6) and K8a/K8b dual repair at frame end — because "who writes first" is structurally eliminated: each node first calculates its own acceptance set locally, and bilateral commit only performs the set intersection.

## 4. Frame Execution

### 4.1 Five-step Frame Loop

```
Creation → Connection → Volume Change → Judgment → Deletion
```

In L0, "Volume Change" only updates degrees, not involving geometry.

### 4.2 Frame Barrier and Double Buffering

- Within a frame: All candidate generation, priority calculation, local dissolution, bilateral commit **occur simultaneously**.
- At frame end: **Global barrier**, unified write to the next frame state.
- **Double buffering**: Read `E_t`, write `E_{t+1}`.

### 4.3 Dangling Pruning

At frame end:

1. Each edge in parallel checks if both endpoints have degree 1;
2. Mark all dangling edges;
3. Parallel deletion;
4. Parallel update of degrees;
5. Synchronize deletion of nodes with zero degree.

**Only one deletion per frame, no cascading.** New dangling edges caused by deletion are input for the next frame (implementation of the Imperfection Theorem).

> Deletion threshold: The axiom lower bound is `deg < 2`; the 3D triangulation closure lower bound is 3 (vertex degree of a tetrahedron). The default `dmin` in pure combinatorial prototype is 3, can be set to 2 to observe survival of deg-2 boundary nodes.

### 4.4 Edge-Face Consistency Check

At frame end, check simultaneously:

- Each edge belongs to exactly two faces;
- Each face has exactly three edges;
- Any boundary edges or boundary vertices.

Edges/faces that violate are marked as **"To be repaired"**, prioritized for processing in the next frame.

Four criteria at implementation layer (all local integer checks):

| Criterion | Content | Violation Meaning |
|----------|---------|-------------------|
| C2 | Each node's `link` (induced subgraph of neighbors) is a path or cycle | Otherwise non-manifold (crossing/figure-eight) |
| C3 | No figure-eight: Any two faces share at most one edge | Overlap/crossing |
| χ conservation | `χ = V − E + F` (closed component should be 2) | Drop indicates handle (crossing) |
| Capacity bound | `E ≤ 3V − 6` | Necessary condition for planar graph, exceeding implies non-planar |

`audit()` also classifies each node's `link` into four categories for diagnosis:

- `cycle` = clean internal node (induced subgraph is exactly a cycle)
- `path` = boundary node (induced subgraph is an unclosed path)
- `chord` = `link` contains a chord ⟹ separation triangle exists at this node (face already occupied by both sides)
- `disc` = `link` induced subgraph is disconnected ⟹ true non-manifold (figure-eight/crossing)

## 5. GPU Mapping

### 5.1 Data Layout (Edge-Centric)

On the GPU, it is **edge-centric**:

- **Edge array**: Each edge stores its two endpoints.
- **Endpoint array**: Each node stores the offset and length of the incoming edge list.
- **Face array**: Each face stores three edges.
- **Degree array**: The current degree of each node.
- **Capacity array**: The capacity of each node.

Nodes are not stored separately but induced by the edge array. When the rotation system is implemented as "per-node rotation table," a CSR-style layout is used: `rot_ptr[v]` (offset) + `rot_idx[]` (rotation neighbors), which is isomorphic to the endpoint array.

### 5.2 Parallel Primitives

| Step | GPU Primitive |
|------|---------------|
| Candidate generation | Parallel traversal of edges/nodes, expanding neighborhoods |
| Priority calculation | One thread per candidate |
| Local sorting | One thread block per node |
| Bilateral commit | One thread per candidate, checking the acceptance sets on both ends |
| Dangling marking | One thread per edge |
| Parallel deletion | Parallel compression |
| Degree update | Atomic add/subtract, only write no decision |

### 5.3 Atomic Operation Principle

> **Decisions are locally deterministic, and atomic operations only handle writing.**

- Incorrect usage: Using atomic CAS to compete for capacity, whoever gets it first is valid (= letting hardware order into L0).
- Correct usage: After a node locally decides which edges to accept, atomic writes are performed.

### 5.4 Double Buffering

- Read buffer: `E_t`, `deg_t`, `F_t`.
- Write buffer: `E_{t+1}`, `deg_{t+1}`, `F_{t+1}`.
- No read/write to the same state within a frame.

## 6. Audit and Projection

### 6.1 L0 Audit

Runs automatically after each frame (**read-only, not involved in rules**):

- Handshake Lemma: `Σ deg = 2|E|`.
- Edge-Face Rule: Each edge belongs to exactly two faces.
- Face-Edge Rule: Each face has exactly three edges.
- Capacity Constraint: All `deg ≤ cap`.
- Existence Criterion: No nodes with `deg = 0`.

### 6.2 Count-Chain Audit

The following numbers are **counted from the graph under explicitly declared presets** — **not "without presets"**:

- **Presets** (L0 axiom / operational assumption, **not counted**) : `χ=2` (closed), **no holes** (true triangulation), edge-face rule, uniform saturation (all `deg=C`).
- **Readings** (counting results from the preset objects):

| Number | Meaning |
|--------|---------|
| 12 | Node count for closed saturation solution (`Σ(6−deg)=12` and all `deg=5`) — regular icosahedron |
| 30 | Edge count |
| 20 | Face count |
| 42 | Exclusive truncation |
| 50 | Combinatorial capacity (`E + F`) |
| 62 | Total positions (`V + E + F = 12 + 30 + 20`) |

- **Out-of-bound (projection)**: kissing number 12 / `60°` / `16π` — obtained only with projection operators outside L0, this layer **has no operators**. ⚠️ Do not mix these with the above readings.

> Implementation reference: `combinatorial_proto.audit` (original counting) + **`src/l0/l0_count.py`'s counting projection operator `count(G, obj_class; presets)`** — presets are explicitly passed as parameters, returns readings; **if presets do not hold ⇒ `value=None` and `kind="undefined"` (not 0)**, and reports which preset is blocking. `python l0_count.py selftest=1`.
> ⚠️ **`χ=2` alone does not imply "no boundary"**: the outer boundary of an open patch is a 12-gon **face**, with `χ=2` (`V=19, E=42, F=25`). C1's 12 requires `closed_chi2 ∧ no holes ∧ uniform saturation` **all three to be true** — this condition is given by the reverse control test in `l0_count`. 
> Layer ruling (2026-09-14): **12 = count reading, not an L0 ontic constant**; layer trinity specification (preset/reading/out-of-bound) and strong divisibility theorem `(6−C) | 12` see `数学/spum-数论.md`.
> Seed selftest: tetrahedron `(V,E,F,χ)=(4,6,4,2)`; regular icosahedron `(12,30,20,2)` with all `deg=5`, `S=12`.

### 6.3 L1 Projection (optional, not involved in L0 rules)

**Generate network from id + adjacency without global embedding solution** (branch N4 excludes "using global embedding as a projection method", see §7; the local criterion for this ruling was revised by branch N7 — evidence of non-embeddability is rigid counting + solver unreliability + paradigm choice, **not** `A_v ≤ 2π`):

- **Network skeleton**: nodes = id, edges = "who connects to whom". The rotation system directly provides it, **no coordinates needed** — as long as we know who connects to whom, the projection layer can form the network.
- **Local direction field**: reconstructed from the rotation system. At each point, a radius `r(deg) = (κ/2)·√(deg/π)` (A2 area budget lower bound, §A.13.1), its `deg` adjacent directions are **laid out in ring order** on the sphere at that point (each point has its own local frame, no need to align across points). The layout method uses **ring-sphere closed latitude rings** `ring_sphere` (`z₀ = −1/√n`, `φᵢ = 2πi/n` closed by index, naturally matching the cyclic ring order of `rot[v]`) — adopted in branch N6; `fibonacci_sphere` (spiral not closed at ends) is retained for comparison.
- **Local quantities**: `σ_v = 2/deg_v`; `∇σ_v = mean_{w~v}(σ_w) − σ_v`; geometric angle defect `2π − A_v`, where `A_v` = sum of internal angles of triangles around v (internal angles of triangles are derived from cosine law using the `deg` of three points, purely local `O(F)`). ⚠️ `A_v > 2π` is **negative curvature (saddle point) reading**, **not** an embeddability criterion (revised in branch N7).
- **Identify 12 crystallite loops, 42 clusters** (pure combinatorial criteria, see judgment ③ in §7.3).
- **L2 global coordinates** (for visualization) are just **projection artifacts**: products of alignment propagation + position constraint projection, explicitly not claiming to be "real geometry" (see §6.4).

> **Implementation**: `src/l1/l1_projection.py` (`project()` / `angle_defect()` / `local_frame()` / `grad_sigma()` / `crystallites()`; `python l1_projection.py selftest=1` all green). Landing readings see §7 "First item landing in phase four".

> Difference from v1.x: In v1.x, L1 converts the existing `nb_dir` angular coordinates of L0 into `pos` via BFS (Appendix A §A.8), and v1.x's `solve_embedding` also globally solves for tangent embedding and hits a rigid wall (§A.13.3); in v2.0, L1 directly generates the network and local quantities from **pure combinatorial structure** (id + rotation system) — geometry does not exist in L0 at all, and in L1 it only exists as **local frames**.

### 6.4 L2 Global Projection (Artifact Layer, Never Writes Back to L0)

**Concatenate local quantities from L1 along the rotation system into a globally visible artifact and read out global quantities**. L2 only reads `net`, does not participate in L0 rules; coordinates are **projection artifacts**, explicitly not claiming to be "real geometry" (branch N5 has already excluded solving for global rigid tangential embedding, see §7 branch N5/N7 for reasons).

- **① Global Coordinates**: `aligned_tree` —— perform **parallel transport integration** along BFS spanning tree (`g = M_i·dir_i(w)` ⇒ `p_w = p_i + (r_i + r_w)·g` ⇒ `M_w = R(dir_w(i) → −g)`): tree edges precisely satisfy the tangential target length, `O(E)`, no iteration, deterministic; `relax` then performs **position constraint projection** (Gauss-Seidel: each edge moves half of the distance to meet the constraint + non-edge repulsion + centroid zeroing per round) to compact the artifact.
- **② Global σ Field**: `σ_global = V/E`; local `σ_v = 2/deg_v`, `∇σ_v`; **gravity proxy = local σ extrema** `anchors` (`∇σ_v < 0` sink / `> 0` source; both empty in isometric graphs).
- **③ Redshift**: `redshift_proxy` —— accumulate relative changes in σ along BFS shortest paths `z_v += (σ_w − σ_i)/σ_i` (**proxy reading**, not a physical prediction; `z ≡ 0` in isometric graphs).
- **④ Visualization Export**: `export_obj` (Wavefront `v`/`l`), `export_json` (nodes with `pos`/`r`/`σ`/`∇σ`/`defect`/`redshift` + edges + anchors).
- **⑤ Radial Power Spectrum (Gravity Power Test, Item ③ of Phase 6)**: `radial_bins` (bin σ radially from centroid or specified center, `r` takes the mean actual distance within bins) → `gravity_spectrum` (**four-way power fitting** `σ(r)` / `ΔN(r)=∫_r^∞[σ−σ₀]dr'` / `∇σ(r)` / `∇ΔN(r)`, log-log least squares + **tolerance band ±0.35** to decide `newton`/`trap`/`neither`), pure numerical components `power_fit` (only accepts `r>0 ∧ y>0`) / `radial_derivative` (central difference) / `hop_integral` (hop count integral). **Reading convention**: `slope∇ΔN = a power` (−2 ⇒ newton = authoritative convention `引力.md` Lemma 3 + Corollary 3a; −3 ⇒ trap = deprecated path, only for comparison). **Degradation Guard**: if the σ profile range is `< 1e-9·max(1,|σ₀|)` (isometric graph) ⇒ all four ways are `slope=None`, `label="undefined"` — otherwise 1e-16 floating point noise is **amplified into pseudo-power** by **non-uniform bin spacing** (measured ∇σ path once showed pseudo slope −0.75). **Falsifiability is proven by two independent controls**: `synthetic_calibration` (inject `σ∝r⁻²` / `σ∝r⁻³`, four ways must restore) + `null_control` (isometric graph icosa t=0 must have no signal in all paths). ⚠️ This item only **reads truthfully on the adopted trajectory**, explicitly **does not extrapolate physical conclusions** (L2 is a projection artifact, not real geometry).
- **⑥ Finite Window Operators (Realize branch N1-C, Item ④ of Phase 6)**: `window_ball` (hop count ball window, BFS from `center` to `radius`) / `window_prefix` (size window, take first `V_window` in BFS order) + subgraph construction `_subnet` (`rot` **preserves order truncation**, no reordering, no prune). **Read-only**: the window is a subgraph view of L0 graph, truncation does not alter `net`. **Readings and Verdicts**: `window_readouts` (`σ_window = V/E`, `deg_mean = 2E/V`, `n_boundary`/`frac_boundary`, `n_comp`, `anchor_density`, radial spectrum) + `window_stability` (scan across `sizes`, verdicts **only consider the final relative residual** of `σ_window` `rel_last = |σ_last−σ_prev|/σ_last ≤ tol_rel`, `slope_σ`/`anchor_density` **only serve as side evidence**). ⚠️ Two empirical calibrations: (a) `slope_σ` is **multiple `None`** on window (degradation guard), `anchor_density` is **1.0** on isometric ball window (isometric subgraph grows anchors on boundary — `anchors` only recognize local extrema) ⇒ both cannot enter verdict, otherwise verdict **always "undefined"**; (b) when `V_window ≥ V`, `window_prefix` truncates to full graph ⇒ multiple readings are same ⇒ **pseudo 0 residual**, so deduplicate by actual `v_window` and count `n_truncated`. Also equipped with `radial_window` (automatically selects fitting window based on `r>0 ∧ bin sample count ≥ min_bin_pts`) — **the decision for item ③'s legacy (i) is given here**: `fit_slice` only retains for `synthetic_calibration` (synthetic calibrationconvention), adopted trajectory always uses `radial_window`.
- **Distorted Readings (Not Errors)**: `holonomy` (non-tree edge closure difference = measure of curvature that cannot be accommodated by a plane), edge residuals, log of non-edge overlaps. ⚠️ All three are only **upper bounds** — PBD only drops to the nearest local minimum (branch N7).

> **Implementation**: `src/l2/l2_projection.py` (entry `global_projection()`; `python l2_projection.py selftest=1` all green, fixtures A–H). Grounded readings and branches N5/N6/N7 see §7 "Phase 4 Quadratic Grounding"; ⑤ radial power spectrum empirical readings and verdicts see §7 "Item ③ of Phase 6 Grounding" (CLI `spectrum=1`); ⑥ finite window stability scan see §7 "Item ④ of Phase 6 Grounding" (CLI `window=1`).

> Direction field continues using §6.3's `ring_sphere` (rotation system closed latitude rings, branch N6 adopted): L2's parallel transport follows the same rotation system, so the "artifact" is entirely determined by id + rotation system. Global coordinates also do not alter L0: L0 has no geometry, L2 is only its **read-only projection**.

### 6.5 L3 Frame Sequence Projection (Cross-Frame Layer, Never Writes Back to L0)

> **This section is newly established**: The original document **has no** §6.5, and the full text has **no definition of "L3"**. The establishment of this regulation is based on §7 bifurcation N8 (record → experiment → exclusion).

**L1/L2 projection is "within a single frame", while L3 projection is "between frames"** —— concatenate the L0 frame sequence into a time-axis network, reading out the registered **tension field metric layer** (`图论/spum-图论指标.md` GT-005~008) and **two-tier closure criterion**. **The cross-frame anchor is id** (id persistence = temporal extension of existence criterion), **while the intra-frame structure is still given by rotation system**. L3 only reads frame snapshots, does not participate in L0 rules; and **does not depend on L2 coordinates** —— all four metrics are coordinate-independent ⇒ decoupled from L2 artifacts (can run without global solving).

- **① Inter-frame network skeleton**: `frame_trace` (align adjacent frames by id → alive / born / dead / persistent) + `attr_series` (time series of persistent ids, `deg / r(deg) / σ_v`, with `r`, `σ` taken from §6.3).
- **② Tension field metric layer** (GT-005~008, coordinate-independent, only consumes id + rotation system):
  - `delta(net,k)`: δ_k = `|{v : deg(v) < k}| / |V|` (registered definition `k=2`; also given `k=dmin`, `k=κ` for generalized reading)
  - `anchors_of` + `anchor_drift`: **anchor = local extremum of σ** (same convention as §6.4 `anchors`), Δμ = centroid of degree, aligned by **id**, across frames (one-dimensionalconvention of GT-006), with `churn` = Jaccard distance of anchor set
  - `dtopo`: `d_topo = |E_τ Δ E_{τ+1}|/max(|E_τ|,|E_{τ+1}|)`, `d_norm`, `d_w = w⁺|V⁺| + w⁻|V⁻|` (symmetric-difference convention of edge sets, default 1:2)
  - `sigma_global`: σ = `V/E` = `2/⟨deg⟩`
- **③ Two-tier closure criterion**: `closure_series` —— `δ<θ_δ ∧ Δμ<θ_μ` and must be **continuous k frames** (default θ_δ=0.03, θ_μ=2.0, k=5, N013/N016 suggested values).
- **④ Export**: `export_csv` / `export_json` (per-frame metric sequence + closure judgment + id-aligned trajectory).

> **Implementation**: `src/l3/l3_projection.py` (entry point `global_evolution(core, nframes)`; `python l3_projection.py selftest=1` all green). Field readings are in §7 "Phase Four, Item Five Implementation".

> ⚠️ **δ arm convention (corrected on 2026-09-13 based on empirical data; original judgment that "both ends degenerate" was half wrong)**: Registered definition δ_2 under L0 adoption dynamics is **always 0** (frame-end pruning deletes `deg<dmin`, `dmin≥3` ⇒ must be `deg≥2`) ⇒ δ_2 has no information about closure. Generalizing to δ_κ (`k=κ=6`) shows **empirical value of `[0.51, 1.00]`, not "always ≈ 1"** —— `Σ(6−deg)=12` only guarantees that the **sum is positive** (`⟨deg⟩ = 6 − 12/V < 6`), **not that almost all nodes have `deg<6`**; empirical sparsest frame (t=3) has only 51.5% of nodes as positive angle-deficit ends. Also, **δ_{dmin} (`deg<dmin`) is not always 0** —— deletion deletes only one layer, and the **neighbors** that are reduced to `deg<dmin` are not cleared in the same frame (only appears in large V⁻ frames; empirical data shows only 1 case in 10 frames at t=9 = 1/558). ⇒ Rejudgment see §6.5 judgment 3 (original "open item" has been **closed** in this round).

### 6.6 L4 Evolution Observation Layer (Cross-frame Event Partitioning, Never Writes Back to L0)

> **This section is newly established**: §7 Build Roadmap "Phase Five: Evolution Observation" previously had only four observation items, no implementation specification. The specification and readings are based on §7 "Implementation of Phase Five".

**L1/L2 projection "within a frame", L3 projection "between frames", L4 observation "which region the event falls into".** L4 only reads the frame sequence from L0, **does not participate in L0 rules, does not write back any state**; all groups of readings are coordinate-independent (only takes id + rotation system, **not dependent on L2 coordinates**). The partitioning is **purely combinatorial** degree tripartition:

```
over : deg > κ (overdense / negative curvature end)    flat : deg == κ    under : deg < κ (sparse / positive curvature end)
zone : all over → "core"; contains ≥1 over → "sh" (boundary of the core); others (including empty list) → "bg"
```

- **① Partitioned event account** `event_account` / `annihilation_profile`: V⁺ and V⁻ are **partitioned by event sites** into core/sh/bg, providing net ΔE per partition and net annihilation rate `rho = (ΣV⁻ − ΣV⁺)/(ΣV⁻ + ΣV⁺)` (`>0` net annihilation / `<0` net creation). Site definition: creation site = three corners of a new cell; edge-breaking site = both ends of the broken edge (both ends must survive at frame end); labels are always taken from **the initial frame snapshot**. Also provide two separate readings: `n_follow` (edges that disappear with deletion points) and `viol_sat` (number of creation sites where **any corner's initial degree ≥ cap**, **expected to be 0** — the gate reading for creation suppression).
- **② Concentration phenomena** `concentration_series`: `max_deg` / `n_sat` / `n_over` / `n_flat` / `n_under` / `gini` / `σ` variance.
- **③ Imperfection Theorem** `imperfection_series` (**edge-arm dual convention**, item ④ of Phase Five implementation): `n_under_end` (remaining dangling ends at frame end), **residual arm density** `delta_dmin = n_under_end / V`, **generalized arm density** `delta_kappa = |{deg<κ}|/V` (same convention as L3 / GT-005), `n_pruned`, `n_broken`, L0-A9 (same-frame edge-breaking + deletion points). **The central arm Δμ is not in this layer** (see L3 `anchor_drift`) — this layer only provides data, threshold judgment is handed over to L3 `closure_series`.
- **④ Entry and export** `observe(core, nframes)` + `export_csv` / `export_json`.
- **⑤ Structural closure audit** `closure_of` / `closure_series` (**implementation of item ③ of Phase Five "internal vs external of closed subgraph"**): provides per frame `n_comp` (number of connected components) / `comp_sizes` / `chi` (χ = V_c − E_c + F_c for each component, faces are attributed to the component of the first vertex) / `boundary_edges` (edges included in < 2 faces; **closed surface → 0**) / `sum6deg` (Σ(6−deg)) / `holes` (d = 3V − 6 − E, V=0 → 0) / `crystallites` (number of vertices in 5-regular components, `find_crystallites` convention).

⚠️ **Judgment for item ③ of Phase Five (probe evidence + measurement, icosa and patch subtypes)**: adopt the trajectory where **each frame is a single connected closed surface** — `n_comp ≡ 1`, `boundary_edges ≡ 0`, `χ ≡ 2`. ⇒ **"Internal vs external of closed subgraph" degenerates at component level**: no distinction between internal and external at component level (not "not measured this time", but structurally impossible). Its non-trivial content is ①'s `core`/`bg` partition — **overdense core `core` = local closure internal, background `bg` = external**; thus item ③ is **merged** into ①, no separate "internal/external" criteria are established. Freezable identity: on a closed connected (χ=2) frame, `Σ(6−deg) = 12 + 2·d`, **this formula only holds for χ=2** (general formula is still handshake lemma `Σ(6−deg) = 6V − 2E`; empty graphs and multi-component graphs are not applicable). Measured: icosa t=10 `Σ(6−deg)=446 = 12+2×217`; χ=2 identity holds frame by frame for 11 frames.

⚠️ **Two conventions (established in this round of measurement, to prevent two traps of "treating identities as measurements")**:

1. **V⁺ must be partitioned by "sites", not by "edge endpoints".** One conization is always "**growing a new cell on three corners of a single face**" ⇒ when partitioning by edge endpoints, each new edge will have one end that is a new point not present at the initial frame (must fall into `under`) ⇒ `V⁺core ≡ 0` will become **a forced illusion** (measured: partitioning by edge endpoints gives 0 for all 10 frames; partitioning by sites shows `96/8/160/788` at t≥7 on cap=64 trajectory).
2. **The distribution of V⁻ is rule-enforced, the distribution of V⁺ is empirical reading.** The criterion `vminus="dense"` is "**both ends are over**" ⇒ by construction `V⁻sh ≡ V⁻bg ≡ 0`, `V⁻core` is all broken edges ⇒ `rho_bg = −1.000` is **an identity**, not a discovery. What can be tested is **the partition distribution of V⁺** (conization falls into "all overdense corners face", "face with overdense corner", or "background face") and the core net sign determined by it: `netcore = V⁺core − V⁻core`. Hence, **core net annihilation is not an identity** — cap=12 gives `rho_core = +1.0000` (pure annihilation), cap=64 is weakened to `+0.9284`.

> **Implementation**: `src/l4/l4_evolution.py` (`observe()` / `event_account()` / `annihilation_profile()` / `concentration_series()` / `imperfection_series()` / `closure_of()` / `closure_series()` / `zone()` / `partition()`; `python l4_evolution.py selftest=1` all green, fixtures A–G). Readings implemented are seen in §7 "Implementation of Phase Five First Item" / "Implementation of Phase Five Item ③" / "Implementation of Phase Five Item ④".

> ⚠️ **Judgment for item ④ of Phase Five (continuous verification of the Imperfection Theorem, closed on 2026-09-13)**:
> - **Both conventions of edge arms degenerate (judgment is constant)**: `delta_dmin` **always meets the standard** (11/11 frames `< θ_δ`; 10 frames all 0, only t=9 = 1/558 ≈ 0.001792) — **masked by same-frame creation**, correct statement is "**large V⁻ frame must appear**" rather than "every frame must appear"; `delta_kappa` **always fails to meet the standard** (measured `[0.5102, 1.0000]` ≫ θ_δ=0.03) ⇒ the AND criterion containing δ arms on this path is **always false and trivial**.
> - **Substantial verification falls on the central arm** (L3 `Δμ`): `7/11` frame meets the standard, **longest continuous 4 frames (t=4–7) < k=5**, and t=3/8/10 single-frame exceeds ⇒ **the Imperfection Theorem is independently carried by the central arm and non-trivial** (not relying on "δ always fails to meet" to spin).
> - **Cross-validation**: L4 `n_under_end` and L3 `delta_dmin × V` **are consistent frame by frame** (only non-zero frame is t=9, value same as `1/558`) — two layers independently implemented, same trajectory, readings match.
> - **Over-assertion in L3 has been fixed**: original comment "frame-end pruning ensures `deg ≥ dmin`" is invalid (only one layer deleted, neighbors being trimmed are not cleared in the same frame); L3 selftest has narrowed this assertion to `nframes=4`, and added 10-frame counterexamples (t=9) and frozen assertion that "AND criterion never met in 10 frames".
> - **Candidate conventions** (if future restoration of δ arm resolution is needed): σ field variance, `|Δδ_κ|` frame-to-frame change rate, or `churn` as second arm; **this round does not arbitrarily alter registered semantics**.

---

## 7. Build Roadmap

| Phase | Content | Verification |
|------|------|------|
| One: L0 Kernel | Simplicial complex data structure; edge-face rule checks; capacity constraints; deterministic candidate generation; deterministic priority; local dissolution + bilateral commit | 12, 30, 20 naturally emerge from closed saturated structures |
| Two: Frame Loop | Five-step frame; double buffering; dangling pruning; edge-face consistency check | Dangling-end regeneration, Imperfection Theorem holds |
| Three: GPU Mapping | Edge-centered data layout; parallel candidate generation; parallel priority; parallel local sorting; parallel bilateral commit; parallel pruning | GPU output matches CPU sequential version, order-independent |
| Four: Projection and Audit | L1 local projection (§6.3); L2 global artifact (§6.4); L3 frame sequence projection (§6.5); count-chain audit; visualization | 42 cluster handshake self-consistent, 12 loops stable |
| Five: Evolution Observation | Net annihilation effect; concentration phenomena; internal and external of closed subgraphs; continuous verification of Imperfection Theorem | Align with empirical conclusions in Appendix A |
| Six: Product Layer | Reconnection into L0 (including platform secondary potential); identification and regeneration of stable structures (12-sphere independent components / 42 clusters / gap vector); σ field coordinate + gravity power test; finite window operator + automatic locality assertion | 12-sphere emerges as independent component; gravity power distinguishable r⁻²/r⁻³ (resolvability proven by synthetic calibration); window readings stable against window size |

> **Phase Status (2026-09-13)**: **Phases One to Four Completed** (Phase Three = three increments of GPU decision/commit/write operations + adoption dynamics GPU-ized; Phase Four = L1/L2/L3 three projections implemented); **Phase Five Completed** — ① Net annihilation effect ✅, ② Concentration phenomena ✅, ③ Internal and external of closed subgraphs ✅ (degradation at component level ⇒ merged into ①), **④ Continuous verification of Imperfection Theorem ✅ (constant edge-arm two-convention verdicts ⇒ substantive verification on central arm Δμ, open item closed)**; **Phase Six In Progress** (① Reconnection into L0 + platform secondary potential ✅ see next anchor point; ② Identification and regeneration of stable structures ✅ see next item anchor point; ③ σ field coordinate + gravity power test ✅ see next item anchor point; **④ Finite window operator + automatic locality assertion ✅ see next item anchor point**).

> **Phase Six Goal (2026-09-13 Frozen) // Product Layer: From "able to evolve" to "produce identifiable stable structures and test their physical predictions"**:
> The first five phases delivered the L0 evolution engine + four layers of observation (L1/L2/L3/L4), i.e., both "how a graph evolves" and "how to read it" are established; but the engine **has not yet produced any identifiable stable structures**, and its physical predictions have not been tested on coordinates. Four items (**①②③④ completed**):
> 1. **Reconnection into L0 + platform secondary potential** (Mechanism) — Connect §7.3's `flip_edge` (2-2 Pachner move) from independent probe to L0's **optional `recon` axis** (parallel with creation/breaking edges, `ΔV=ΔE=ΔF=Δχ=0`); provide a **deterministic tie-breaker secondary order** on the Ψ platform, making "Φ not decrease" both reach the Ψ lower bound **and terminate**. Connect to §7.3 remaining bifurcation ①, unactivated items in §7, and legacy in §7.4.
> 2. **Identification and regeneration of stable structures** (Product) — According to §7.3 verdict, establish "closed 12-sphere as **independent connected component**" as a determinable target and provide regeneration path; supplement `crystallites` in adoption trajectory `t≥1` with empty gap; implement "gap" as computable quantity (hole / k≥4 loop / gap direction `n̂` ← GT-032). Connect to spum-core §Seven (photon = moving gap / matter = gap interlock / rest mass = locked kinetic energy), §7 verification list, N8-B.
> 3. **σ field coordinate + gravity power test** (Physical prediction) — L2 coordinates are ready (edge residual mean 0.134, max 0.386), perform σ radial binning on it, **deliver four power readings**: `σ(r)`, `ΔN(r)=∫_r^∞[σ(r')−σ₀]dr'`, `∇σ(r)`, `∇ΔN(r)`. **Goal = `a = c²∇ΔN ∝ r⁻²` (Newton; authoritative convention `引力.md` Lemma 3 + Corollary 3a)**; `∇σ ∝ r⁻³` only as **deprecated trap path reference reading** (`spum-evolution.md` §4.2's `a = −2kc²m/r³` is this path, already judged obsolete by `引力.md` §2.9). Equal-degree graph (`σ≡0.4`, no anchor point, redshift `≡0`) as **null-hypothesis control**; **resolvability** proven by synthetic calibration (inject `σ∝r⁻²` / `σ∝r⁻³` and fitting must restore). Upgrade §6.4's "proxy reading" to falsifiable reading. ⚠️ **Two power readings for σ coexist, no resolution** (`σ∝r⁻²` vs `σ∝r⁻¹`, both return `a∝r⁻²`) — divergence point see next item anchor point.
> 4. **Finite Window Operator + Local Automatic Assertion** (audit completion) — materialize N1-C ("finite window selected by projection layer") as explicit window operator (`V ≤ V_window` subgraph / local spherical window), and prove that the readings within the window are stable with respect to window size; L0-A10 upgrades from "code review + pointer analysis" to **automatic assertion** (kernel access hop distance ≤ second-order neighbor).
> **Verification criterion**: 12-sphere emerges as an independent component; **gravitational power reading is falsifiable** (synthetic calibration restores r⁻²/r⁻³; null hypothesis of equi-degree graph has no signal; no pseudo-power appears on the adoption trajectory); window readings are stable with respect to window size.
> **No new concepts introduced**: four items respectively address the documented discrepancies — ① §7.3 verdict / legacy bifurcation; ② §7.4 verdict 3 + §7.3 verdict + §7 phase four verification list; ③ §6.4 ②③; ④ N1-C / L0-A10.

**Current progress anchor**:

- Phase one/two prototype (`src/l0/combinatorial_proto.py`): rotation system + four χ-preserving operations + `tri_is_face` criterion + `unguarded` comparison mode (reproduce v1.x K2 cross-defect) + seed selftest (tetra/icosa/patch).
- Phase one/two kernel (`src/l0/l0_core.py`): generalize the prototype's "full coning" to **deterministic negotiation with capacity constraints** — candidate generation (true face / hole + dangling) → deterministic priority → per-node `cap−deg` budget local dissolution → endpoint acceptance set intersection commit → double buffering frame → dangling pruning; includes §4.4 four-criteria audit (link four-classification / bowtie / χ / planar boundary) and §6.1 audit. Verified: seed count, frame invariants, determinism and order independence, capacity budget takes effect, V⁻ piercing → B-type hole coning.
- **Drive switch and three probes** (2026-09-13): `drive = slack | saturate` (creation priority direction) × `pairing = none | conserved` (frame net ΔE = 0); `probe=1` outputs three probes (degree distribution maxdeg/gini, closed shell Σ(6−deg)/5-regular 12-point count, state key short cycles). First round of 8 configurations readings are frozen in §7.1.
- **Inactive items**: L0-A9 (dangling-end regeneration) is unreachable under the "pure coning" candidate set — coning creation and dangling pruning are exact inverses (`dmin ≤ 3` deletion of branches never triggers; `dmin ≥ 4` creation and deletion cancel frame by frame, system returns to seed). Probe verification: under `pairing=conserved`, V− = 0 (balance guard ensures no prune trigger, see §7.1). Activating it requires **asymmetric V⁻ annihilation rules** (corresponds to geometric L0's "closed subgraph creation suppression"). **→ §7.4 has activated with `vminus="dense"` (both CPU and GPU sides, see §7 fourth increment).** `cone_edge` and "reconnection" candidates are not yet integrated (reconnection see §7.3).
- **Bare dynamics (`bare=1`, 2026-09-13)**: remove `cap`/priority/`drive`/`pairing`, only keep "coning all true faces + frame-end pruning", long-range readings frozen in §7.2. Verdict: under zero prohibitions, maxdeg doubles every frame, V triples every frame (unbounded exponent, `α → ln2/ln3`) — **saturation does not emerge; upper bound must be given by prohibitions**; and when `dmin ≤ 3`, pruning never triggers, when `dmin ≥ 4`, coning and pruning are exact inverses, **two channels cannot be active simultaneously**.
- **Reconnection dynamics (`recon=1`, 2026-09-13)**: implement `RotNet.flip_edge` (edge flip / 2-2 Pachner move, ΔV=ΔE=ΔF=Δχ=0); fix V run potential function Φ probe (`mode = none | phi1 | phi2`), readings frozen in §7.3. Three verdicts: ① no direction selected ⇒ empirical **period 2** (self-inverse ⇒ must be periodic, verifying "potential function is the only structural solution"); ② strict Φ₁ (ΔΨ < 0) **cannot reach V=12 icosahedron** (stalls at Ψ=16 pseudo-local minimum), but **non-strict Φ₁ (ΔΨ ≤ 0) can** — hence correct form is "Φ does not decrease" rather than "Φ increases"; ③ the 12-shell icosahedron in **connected** triangulations with V > 12 is **structurally impossible** (5-regular induced subgraph ⇒ no edges to outside ⇒ disconnected).
- **GPU Decision Layer (Phase III First Increment, 2026-09-13)**: Added `src/l0/l0_gpu.py`, moving the **decision segment** within a frame (`propose` candidate generation + `priority` priority key) to GPU, while the submission segment remains on CPU. Data layout is flattened into CSR (`ids` / `rot_ptr` / `rot_idx` / `deg`) as per §5.1, with all indices being **position indices**; dart uses flat index encoding, `rev` (reverse dart) is derived from `(owner,target)` monotonic key packing + argsort + searchsorted, and `dart_next` is obtained by modulo within the ring. Face enumeration uses **pointer jumps** in O(log) rounds to find the minimum dart within the ring. **Key equivalence basis**: The iteration order of CPU's `propose()` does not affect the result (`resolve`/`commit` both sort by key, and keys are unique on triangulation), so GPU only needs to produce the "candidate set + correct key". **Verification**: 7 structural invariants (`rev[rev[d]]=d`, `face_id` remains constant along `dn`, Σ ring length = 2E, no bowties, χ conservation) × 2 seeds all passed; bitwise consistency with CPU was achieved for 24 groups (2 seeds × 2 `drive` × 3 `cap`) × 4 frames; end-to-end `frame_gpu` (GPU decision + CPU submission) vs pure CPU 12 groups × 4 frames, frame-by-frame `state_hash` consistent, final state matches the frozen reading in §7.1 (icosa cap=12 slack/none → V=40, saturate → V=36). **Performance**: When `2E=109140` (V=18192), decision segment CPU 265 ms → GPU 46 ms (5.9×); when `2E=7956` (V=1328), 13.6 ms → 15.4 ms (1.13×, transfer and kernel launch overhead dominates) — GPU benefits scale with size. **Unmoved/Leftover**: At that time, `resolve` (local dissolution), `commit` (intersection submission), and `prune` (dangling pruning) were still on CPU — already taken over by the second increment (see next item).
- **GPU Submission Segment (Phase III Second Increment, 2026-09-13)**: `l0_gpu.py` added 7 vectorized operators to move the **judgment** of the submission segment entirely to GPU — `rank_creation` (`key = (type, ±slack, sorted(corners))` lexicographical order rank, `lexsort` row order is key1←key2←key3 and **last row is primary key**, CuPy `lexsort` only accepts 2-D arrays), `incidence` (candidate-corner CSR transposed to "corner-candidate" long table), `resolve_accept` (§3.4 local dissolution: pack `node*Σcand + rank` sorted by corner and truncate to `cap−deg` per group), `intersection` (§3.5 intersection criterion: candidate survives ⟺ each corner accepts it, use `bincount(weights=acc)` count == `cand_len`), `survivors` (sorted by rank ascending), `slots_of` (ragged range slot expansion), `degree_next` (§4.3 pruning marker `deg_{t+1} = deg_t + cones − breaks`, then mark `deg < dmin`), all under the unified entry `gpu_commit_stage`. `frame_gpu` is rewritten as "GPU judgment + host-side write operations", returning the same shape as `L0Core.frame()` (**count**, not list). **Equivalence basis (empirically established)**: `commit` within `cone_tri_face`/`cone_hole` **never returns None** — candidates come from the frame's initial snapshot, and cone only modifies the internal orientation of its own face ("broken" requires two different faces to share the same dart, impossible); 24 configurations × 6 frames count, failure count always zero, so no need for re-verification per item. **Two things still on CPU** (both inherently serial): rotation system write operations (`cone_*` / `break_edge` / `remove_vertex`, and `nid` allocation order must match key ascending submission order) and §3.2 annihilation budget greedy scan (`_plan_breaks` and conserved under per-item budget truncation, default `pairing=none` zero cost). **Verification**: Submission segment (`resolve` + intersection) bitwise consistent with CPU 16 groups (2 seeds × 2 `drive` × 3 `cap` × `punc=0/1`) × 4 frames, **submission order also consistent**; end-to-end 24 groups × 4 frames frame-by-frame `state_hash` consistent, final state matches frozen reading in §7.1 (icosa cap=12 slack/none → V=40, saturate → V=36). CLI added `commit=1`.
- **GPU Write Operations (Phase III Third Increment, 2026-09-13)**: `l0_gpu.py` consolidated the last remaining CPU rotation system write operations (`cone_tri_face` / `cone_hole` / `break_edge` / `remove_vertex`) into a **single CSR rebuild** — `rebuild_rings` (S1 survival mask / S2 old dart retention mask / S3 insertion records / S4 row length / S5 segment start / S6 inline slots / S7 write old rows / S8 write new rows / S9 return `nid2 = nid0 + m`) + `ring_pos_of_dart` (ordered pair → flat dart) + entry `gpu_write_stage` (judgment → rebuild → concatenate `ids2`) + frame end `_to_rotnet` (CSR → `RotNet` materialization). **Key simplification**: All write operations have only three forms (ring insertion / ring deletion / new row creation and discard); `cone_tri_face(i,j,k)` and `cone_hole(vs)` insertions use the same rule (candidate corners preserve ring order `c[0..k-1]`, for each `t` in `owner=c[t]` ring order, insert new cell right after anchor point `c[(t-1) mod k]`; new row `=[c0]+reversed(c[1:])`) — **anchor points are not part of the judgment** (CPU is "insert first then delete node", anchor point deleted and new cell falls exactly on original anchor point position); each dart inserted in this frame is **unique pairwise** (each dart belongs to exactly one face, candidate = face) ⇒ no conflict, fully parallelizable. Thus, the retention criterion for each old dart reduces to pure per-dart predicate `keep = (p survives) ∧ (m survives) ∧ (dart not broken)`, and emission criterion for insertion is `emit = (cr_len ≥ dmin) ∧ (owner survives)` — **anchor point survival does not participate in the judgment** (CPU is "insert first then delete node", anchor point deleted and new cell falls exactly on original anchor point position); output row order = surviving old nodes (ascending position) ‖ surviving new cells (creation order), `opos`/`newrank` given by prefix sums; new ids are always consumed (even if new cells are later deleted, `nid2 = nid0 + m`), so id allocation is naturally consistent with CPU's key ascending submission. **Verification**: 5 standard fixtures (A face coning / B face coning + edge break / C new cell must be deleted / D old node deleted / E empty input boundary, all expected values manually derived) all passed; write operation segment vs CPU `_step` bitwise consistent for 24 groups (2 seeds × 2 `drive` × 6 configurations, including `dmin=4`, `conserved`, piercing) × 4 frames (including `nid` and born/dead/broken/spent counts); end-to-end `frame_gpu` (**GPU judgment + GPU rebuild**) vs pure CPU 24 groups × 4 frames frame-by-frame `state_hash` consistent. **Two things still on CPU** (both inherently serial or interface overhead, not parallelizable decisions): ① §3.2 annihilation budget greedy scan (`_plan_breaks` and conserved under per-item budget truncation; zero cost when default `pairing=none`); ② frame end CSR → `RotNet` materialization (`_to_rotnet`, for `l0_core` read-only query/audit). L0 full frame is now end-to-end closed-loop on GPU (judgment + write), host side only has frame scheduling. CLI added `write=1`.
- **Bifurcation N2 (2026-09-13, record → experiment → exclude) // Phase three, fourth increment: porting the "adoption dynamics" to the GPU**:
  **Cause**: The `vminus="dense" + vplus="any"` adopted in §7.4 **landed only in the CPU kernel** — the whole of `l0_gpu.py` contains no `vminus` / `vplus` / `dense` strings. ⇒ **Phase three (GPU mapping) and phases one/two (kernel rules) are out of sync**: the end-to-end GPU run is still the §7.1 baseline dynamics.
  **Branches**: ① **N2-A** port the adoption dynamics to the GPU; ② **N2-B** leave the GPU running only the baseline dynamics and go straight to phase four (projection).
  **Disposition**: **N2-B excluded** — the §8 decision table has already fixed "GPU = execution model", so the kernel counts as successfully built only if it runs on the GPU; moreover, keeping N2-B would leave §7.4's adoption readings forever on a CPU-only convention, and the phase four projection would have no scalable host. **N2-A adopted**, construction this round (three pure increments, baseline path untouched): ① `candidates_and_keys` gains a `len(vs) ≤ cap` guard (**the GPU counterpart of X1** — the GPU likewise had no capacity check for new points) and a `δ > 0` filter (`vplus="gap"`); ② `gpu_write_stage`'s annihilation budget becomes an explicit `budget` parameter, and supports `budget=None ⇒ break all plans` (dense/none semantics); ③ `frame_gpu` dispatches the plan by `core.vminus` (dense → `_plan_dense_breaks`, factored out as `_annihilation_plan` so the verification path can share it).

  **Experimental readings (fourth increment, frozen 2026-09-13):** all readings **bit-for-bit identical** to the CPU convention (`state_hash` + event counts + `nid`).

  | Test Item | Configuration | Reading | Verdict |
  |--------|------|------|------|
  | Decision stage | icosa cap=12 dmin=3 dense/any × 5 frames | candidates 20/60/38/128/160 all identical | ✅ |
  | Write stage | icosa cap=12 dense/any × 5 frames | V trajectory 32/36/66/82/98; t=1 `broken=30`, t=4 `broken=24` | ✅ `budget=None ⇒ break all plans` confirmed |
  | End-to-end | 2 seeds × 2 `drive` × 5 configurations (dense/any, dense/conserved, dense/gap, cap=64, dmin=4) | frame-by-frame `state_hash` identical | ✅ |
  | X1 guard | icosa cap=12 dense/any 10 frames | `maxdeg ≤ cap` (CPU∧GPU) always true; **V=960** (= §7.4 t=10 reading) | ✅ |
  | X1 guard | icosa cap=64 dense/any 5 frames | `maxdeg ≤ cap` always true; **V=1550** (= N1 sweep cap=64 V@5) | ✅ |
  | Control frozen values | dense/gap → V=62; dense/conserved → V=12 | consistent with §7.4 suppression frozen values | ✅ |
  | Baseline regression | `dangling` 24 groups × 4 frames | readings bit-for-bit unchanged from §7.1 frozen values (icosa cap=12 slack → V=40, saturate → V=36) | ✅ |

  **Verdict:**
  1. **N2 closed — phase three resynchronized with phases one/two.** What the GPU now runs end-to-end is exactly the `dense/any` dynamics adopted in §7.4, and it is bit-for-bit identical to the CPU baseline (including event counts and `nid` allocation order). Henceforth §8's "GPU = execution model" holds for the **adoption dynamics**, not merely for the baseline.
  2. **X1 is likewise a pre-existing defect on the GPU side** (not introduced by this increment): the GPU previously had no capacity check for new points either. The baseline dynamics cannot build a hole with k > cap within 4 frames, so it never surfaced; the adoption dynamics hit it at frame 10 (isomorphic to the path by which the CPU discovered X1). **"cap is a load-bearing structure" holds on the CPU and GPU sides simultaneously.**
  3. **Cost**: the inherent serial item newly entering the CPU hot path is `_plan_dense_breaks` (in dense mode it is non-empty by default, unlike `_plan_breaks`, which only runs under conserved). This is the **only** CPU bottleneck for the GPU adoption dynamics, and a natural candidate for subsequent increments.
- **Bifurcation N3 (2026-09-13, record → exclude) // When does the "kernel" count as successfully built**:
  **Cause**: Phases one~three closed out with the fourth increment. Two "to be continued" items still hang at the tail of §7.1 — do they count as blocking "kernel successfully built"?
  **Item-by-item screening (judging whether each is load-bearing)**:
  1. "The edge-break **selection rule** is a placeholder (the two endpoint degrees sum maximal)" — the placeholder is `_plan_breaks`, which **only enters the path under `pairing="conserved"`**; the adoption dynamics is `pairing="none"` + `dense`, going through `_plan_dense_breaks` (whose criterion is already the SPUM semantic "both endpoints have negative curvature = over-dense"). ⇒ **Not load-bearing for the adopted configuration, excluded.**
  2. "The A criterion (creation suppression in closed shells) is not wired in" — its implementation is exactly `vplus="gap"`, and §7.4 verdict 3 has already measured it: suppressing creation ⇒ the engine loop is cut ⇒ freeze at t≤3. ⇒ **Already excluded by experiment**, no longer listed as a to-do.
  **Verdict**: **Phases one~three complete, no load-bearing leftovers. "Kernel" = phases one~three ⇒ successfully built; next path = phase four (projection and audit).**
  **Bifurcations inherent to the next path (left for the next round to walk by the same rule)**: The first item of phase four, "L1 sphere projection", runs straight into the existing conclusion of §A.13 — global embedding must diverge rigidly once E hits 3N−6 (§A.13.3). The forks are exactly those of §A.13.5: **direction A: local angular capacity projection** (depending only on `deg` + ring order, no global solve) vs a global embedding solve. Purely combinatorial L0 naturally stands on the former (§A.13.5 endnote), but the concrete form of the projection is not yet fixed.
- **Bifurcation N4 (2026-09-13, record → experiment → exclude) // Which route the first item of phase four, "L1 sphere projection", takes**:
  **Cause**: Taking up the fork left by N3. L1 is either a **global embedding solve** or a **local angular capacity projection**.
  **Branches**: ① **N4-A global embedding** — solve `p_v ∈ R³` so that each edge's two endpoint balls are tangent (`|p_u−p_w| = r_u+r_w`) and non-edges do not overlap (`|p_u−p_w| ≥ r_u+r_w`); ② **N4-B local angular capacity projection** — each node **independently**: `deg → radius r(deg)`, `ring order → spherical direction`; zero solve, zero iteration, zero global coordinates.
  **Evidence script**: `%TEMP%\l0_proj_n4.py` (self-contained, `sys.path` already points to `src/l0`).

  **Experimental readings (seed=icosa cap=12 dmin=3, adoption dynamics dense/any, t=0…4):**
  | Test Item | Reading | Verdict |
  |----------|---------|---------|
  | Rigid Counting `d = 3V−6−E` | t=0/1/3/4 **d=0** (pure triangulation ⇒ `E = 3V−6` exact); t=2 d=30 (30 broken edges) | **"E collides with 3N−6" is an identity for pure combinatorial L0, not an accident** ⇒ global embedding **always zero relaxation** |
  | N4-A Analytical Comparison (t=0, no solution) | Regular icosahedron scaling makes "edge = 2·r(5)": edge residual **8.9e-16**, non-edge nearest distance 12.24754, non-edge gap **+4.67814 > 0** | ✅ Constraint set **not inherently unsolvable** (equal degree ⇒ equal radius ⇒ equal edge) — excludes "no solution is a definition problem" |
  | N4-A Global Relaxation Residual | eq phase rel 0.0091→0.0605; comb phase rel 0.0812→0.2052; overlap max 3.49→5.56 | ⚠️ **Solver unreliable** (t=0 known to have exact solution yet residual remains) ⇒ only weak evidence |
  | **N4-A Local Complement Criterion** `A_v ≤ 2π` | t=0: 0.8333 / out-of-bound 0; **t=1: max 1.3820 / out-of-bound 12**; t=3: 1.5117 / **32**; t=4: 1.5962 / **32** | ❌ **Locally infeasible** (deterministic) |
  | Out-of-bound Point Instance | t=1: point v=0 `deg=10`, neighbors degs `[3,3,3,3,3,10,10,10,10,10]` ⇒ `A_v/2π=1.3820` | ❌ Heterogeneous neighborhood (hub surrounded by small balls) directly out-of-bound |
  | N4-B Well-definedness / Cost | 5 frames 228 points **always well-defined**; cost `O(Σdeg)=O(2E)`; pure Python **0.002 s**; A2 utilization ≡ 1 | ✅ |

  **Verdict:**
  1. **N4-A Excluded.** The deterministic criterion is a **pure local necessary condition**: treating each edge as "two balls tangent", then triangle `(i,j,k)` has three side lengths `r_i+r_j, r_j+r_k, r_k+r_i`, and the three internal angles are uniquely determined by **cosine law** (only dependent on three points `deg`); thus, the total angle at v from surrounding triangles `A_v` must be `≤ 2π`, otherwise the neighboring balls around v **must overlap**. This criterion `O(F)` is computable, **no global coordinates needed**. t=0 satisfies (0.8333, consistent with analytical embedding), **t≥1 immediately violates** (out-of-bound points 12→32, max reaches 1.5962). ⇒ Global tangent embedding on the adopted trajectory is **locally impossible** — same source as §A.13.3's "global rigidity wall", but here **earlier and cheaper** to judge (previously only indirectly sensed by diverging iteration residuals).

     > **⚠️ Revision (2026-09-13 branch N7): The "necessary condition" part of this judgment has been disproven.** `A_v > 2π` **does not imply** "neighboring balls must overlap / local embedding impossible" — t=1 exists **exact tangent embedding** (edge residual `3.55e-15`, non-edge minimum gap `+3.85`), and the `A_v` calculated from this embedding position is **bit-for-bit identical** to L1 reading (`1.3820`). The correct semantics of `A_v` is "the total length of the **closed spherical polygon** around v in ring order", and `δ_v = 2π − A_v` is the intrinsic curvature sign (`δ_v < 0` = **saddle point, fully embeddable in R³**). Hence, `viol` has been **downgraded to a curvature reading**; N4-A's exclusion now relies on three weak evidences: "rigid count + solver unreliable + paradigm choice", and "global / incremental embedding solvability" is reclassified as **open item** (see §7 branch N7). This original judgment is retained for record-keeping purposes.

  2. **Byproduct (new reading):** Out-of-bound points **include `deg=6`** (combinatorial angle deficit `δ=6−6=0`, should be "flat") ⇒ **Geometric out-of-bound occurs earlier than combinatorial angle deficit** — heterogeneous radii push the internal angles away from `π/3`. This explains why §A.13's "degree heterogeneity" is a fatal variable for global embedding: it's not that the spherical surface can't hold neighbors (§A.13.2 already proves capacity up to 193), but rather **the angle is blown up**.
  3. **N4-B Adopted.** Local angular capacity projection is always well-defined in any frame, `O(2E)` cost, no iteration, no failure points; and consistent with user paradigm — **as long as "who connects to whom" (id + ring order) is known**, the projection layer can form a network. Global coordinates are downgraded to L2 artifacts (branch N5 adopts "alignment propagation + position constraint projection", see §6.4), clearly labeled as **projection artifacts**, not claimed as "real geometry".
  4. **Scope:** L0 only records relations; L1 = generates network and local quantities directly from **id + adjacency/ring order** (`r(deg)`, `σ_v`, `∇σ_v`, geometric angle deficit `2π−A_v`); **"global embedding solution" does not enter the projection pipeline**. §6.3 has been rewritten accordingly. (Note: the local criterion in judgment 1 was disproven by branch N7, so "whether global embedding is solvable" remains as an **open item**, but this does not affect the above scope — the projection layer only needs id + ring order.)

  > **This local model collaborative verification (`qwen2.5-coder:14b`):** Judged **correct** — V1 (closed triangulation `E=3V−6`), V2 (A2 radius law and `deg_sat=16π`), V3 (gap formula `2r(5)(φ−1)`), V4 (local definition always well-defined, equal degree implies `∇σ≡0`), V6 (`A_v ≤ 2π` is a necessary condition and pure local quantity). Judged **insufficient** — V5: only "number of constraints = number of degrees of freedom" cannot determine general solvability, **linear independence of constraints is also needed** — this prompt led to judgment 1's local criterion. **Two numerical errors** (V3 gap gives 4.61803, correct 4.67812; V7 gives r(3)=1.549, correct 2.93162) ⇒ local model conclusions can be used for **thought verification**, but all numerical values must be verified by code. **(Note: the judged "correct" V6 — "`A_v ≤ 2π` is a necessary condition" — was empirically disproven on 2026-09-13 branch N7, see §7 branch N7 judgment 4.)**

  Reproduce: `python %TEMP%\l0_proj_n4.py`

- **Phase Four First Implementation (2026-09-13) // L1 local projection implemented as per §6.3**: Added `src/l1/l1_projection.py` (pure read-only projection, **not involved in L0 rules**, frame evolution delegated to `l0_core`).
  **§6.3 Four Points Corresponding:** ① Network skeleton `edge_list` (deduplicated undirected edges, directly taken from rotation system ⇒ only needs id + ring order, no coordinates) ② Local direction field `ring_sphere` (adopted in branch N6; `fibonacci_sphere` left as a control) + `radius_of` + `local_frame` (radius `r(deg)=(κ/2)√(deg/π)`, `deg` directions **laid out on the sphere at this point itself** according to `rot[v]` ring order, each point has its own frame) ③ Local quantities `sigma_of` (`σ_v=2/deg`) / `grad_sigma` (`mean_{w~v}σ_w−σ_v`) / `angle_defect` (geometric angle deficit `2π−A_v`, three side lengths taken as `r_i+r_j, r_j+r_k, r_k+r_i`, internal angles determined by cosine law ⇒ pure local `O(F)`; `viol` = `A_v>2π` saddle point reading) ④ `crystallites` (delegated to `find_crystallites`). Entry `project(net)` produces all readings at once.
  **Verification:** `selftest=1` all green — tetra (`A_v=π`, `defect=π`, `Σdef=4π`) and icosa (`A_v=5π/3`, `defect=π/3`, `Σdef=4π`, `∇σ≡0`, skeleton 30 edges, `r(5)=3.7846988`) — and gives curvature readings for the adopted trajectory t=0 saddle point 0 → t=1 saddle point 12 (`max A_v/2π=1.3820`, bit-for-bit consistent with §7 branch N4 reading table; the semantics of this quantity **has been revised by branch N7**).
  **Frame-by-frame projection readings** (`seed=icosa nframes=6 cap=12 dmin=3 dense`, the table below `—` = this frame has holes ⇒ angles are only contributed by triangular faces, aggregated angle deficit readings are incomplete, curvature reading `viol` is unaffected):

  | t | V | E | σ=V/E | r_max | r/R_crit | A_v/2π max | Out-of-bound | Σdef/2π | Holes | maxdeg |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 12 | 30 | 0.4000 | 3.78470 | 0.31539 | 0.8333 | 0 | 2.0000 | 0 | 5 |
  | 1 | 32 | 90 | 0.3556 | 5.35237 | 0.44603 | 1.3820 | 12 | 2.0000 | 0 | 10 |
  | 2 | 36 | 72 | 0.5000 | 4.47812 | 0.37318 | — | 0 | — | 30 | 7 |
  | 3 | 66 | 192 | 0.3438 | 5.86323 | 0.48860 | 1.5117 | 32 | 2.0000 | 0 | 12 |
  | 4 | 82 | 240 | 0.3417 | 5.86323 | 0.48860 | 1.5962 | 32 | 2.0000 | 0 | 12 |
  | 5 | 98 | 264 | 0.3712 | 5.61362 | 0.46780 | — | 22 | — | 24 | 11 |
  | 6 | 154 | 452 | 0.3407 | 5.86323 | 0.48860 | — | 68 | — | 4 | 12 |

  Reading highlights: ① Pure triangulation frame `Σdef/2π ≡ 2` (i.e., `4π`, Gauss-Bonnet holds automatically, consistent with §7 bifurcation N4 note); ② `r_max/R_crit` is always `< 0.5` ⇒ under radius law, no point touches the critical radius `R_crit=2κ=12` (saturation `deg_sat=16π≈50.27`); ③ `maxdeg` is always `≤ cap=12` (X1 guard is also visible on the projection side); ④ Out-of-bound points (`viol`) start appearing from t=1 (12→32→32→22→68), but **non-monotonic**—t=2 is an exception with 0, because edge removal dominates in this frame (`E` drops from 90 to 72, 30 holes, triangular faces drop sharply). L1 is only a **read-only record** of this geometric fact and does not modify L0 based on it (L0 has no geometry).
  Reproduce: `…\python3.13.2\python.exe l1_projection.py selftest=1`; `… l1_projection.py seed=icosa nframes=6 cap=12 dmin=3 vminus=dense`

- **Bifurcation N5 (2026-09-13, record → experiment → exclude // adopt P2+P3) // How L2 global coordinates (projection artifact) are generated**:
  **Cause**: §6.4 original text only mentions "visualization" in three characters, the generation method is not determined. **Branches**: ① **N5-A Global rigid tangent embedding** (solve `p_v` to make all edges tangent simultaneously)—excluded by N4; ② **N5-B Naive propagation** (BFS + local frames at each point summed independently); ③ **N5-C Aligned propagation** (parallel transport integration along the spanning tree); ④ **N5-D Position constraint projection refinement** (PBD / Gauss-Seidel).
  **Evidence script**: `%TEMP%\l2_proj_n5.py` (three-way comparison P1/P2/P3, including N6 direction field comparison section).

  | Test item | icosa t=0 | icosa t=1 | icosa t=2 | patch t=1 | Verdict |
  |----------|-----------|-----------|-----------|-----------|---------|
  | P1 Naive propagation non-tree edge conflicts (count / RMS / max) | 49 / 15.40 / 31.85 | 149 / 17.22 / 36.93 | 109 / 14.92 / 33.05 | 209 / 17.79 / 41.92 | ❌ Conflict magnitude ≥ 4×`r(5)`, excluded |
  | P2 Aligned propagation tree edge residual RMS | 4.0e-16 | 3.4e-16 | 6.3e-16 | 9.7e-16 | ✅ Tree edges **constructed exactly** |
  | P2 Non-tree edge holonomy (count / RMS / max) | 19 / 18.24 / 26.16 | 59 / 21.86 / 47.88 | 37 / 22.32 / 34.68 | 83 / 23.95 / 51.90 | Reading: curvature cannot be accommodated in a plane |
  | P3 Slack edge residual RMS (P2 initial / random initial) | 0.382 / 0.350 | 0.338 / 0.578 | 0.309 / 0.430 | 0.901 / 0.764 | ✅ Converges to **non-zero plateau**; both initial values are of the same magnitude |
  | Holonomy (viol points vs non-viol points) | — | 22.36 / 20.87 (1.07×) | — | 24.52 / 22.55 (1.09×) | Reading: holonomy is **unrelated** to local angle out-of-bound |

  **Verdict**:
  1. **N5-A not adopted**: Global embedding solving does not enter L2 (reasons see below N7—the evidence for excluding it is rigid count + solver unreliability + paradigm choice, **not** local angle criteria).
  2. **N5-B (naive propagation) excluded**: Non-tree edge conflicts 49–209 locations, RMS 13–18, same magnitude or larger than scale `r(5)=3.78` ⇒ independent frames at each point cannot be directly summed.
  3. **N5-C (aligned propagation) adopted**: Tree edge residuals `~1e-16` (construction guaranteed), `O(E)`, no iteration, deterministic; non-tree edges `holonomy` (14–24) **not error but curvature reading**.
  4. **N5-D (position constraint projection refinement) adopted as "compaction" step**: Residuals drop to `0.13–0.90` non-zero plateau, and random initial values and P2 initial values are of the same magnitude ⇒ plateau is not initial value sickness. ⚠️ But N7 correction: plateau is only a **local minimum**, so residuals are an **upper bound**, cannot be used as proof of inembeddability.
  5. **Scope**: L2 coordinates = `aligned_tree` + `relax` **artifact**, clearly not claiming to be "real geometry"; distortion amounts are reported truthfully as readings.
- **Bifurcation N6 (2026-09-13, record → experiment → exclude) // Which local spherical tiling method for L1 direction field**:
  **Cause**: §6.3 writes "`deg` adjacent directions are laid on the point's own sphere in **rotation order**", but the specific tiling method is not determined. **Branches**: ① `fibonacci_sphere` (spherical spiral, locally approximately uniform but **not closed at ends**) ② `ring_sphere` (**a single closed latitude line**: `z₀ = −1/√n`, `φᵢ = 2πi/n`, closed by index ⇒ naturally matches the **cyclic rotation order** of `rot[v]`).
  **Reading (same script N6 section, holonomy RMS)**: icosa t=0 `18.24 → 11.97` (1.52×), t=1 `21.86 → 8.82` (2.48×), t=2 `22.32 → 10.44` (2.14×), patch t=1 `23.95 → 13.17` (1.82×) — **latitude rings are all lower**; and icosa t=0 latitude ring tree edge lengths `min = median = max = 7.5694 = 2r(5)` (exactly equilateral), fibonacci side is `7.5694 / 12.6564 / 21.3108` (**severely unequal**).
  **Verdict**: adopt `ring_sphere` (`local_frame` default `mode="ring"`), `fibonacci_sphere` retained as a control. Basis: rotation order is purely combinatorial data, closed latitude rings make "direction field" and "rotation order" **isomorphic**—direction index is rotation order position, no end discontinuity.
- **Phase four-term implementation (2026-09-13) // L2 global projection as per §6.4**: Add `src/l2/l2_projection.py` (**read-only projection, never write back to L0**). §6.4 four items are implemented item by item: ① `aligned_tree` (BFS parallel movement integration: `g = M_i·dir_i(w)` ⇒ `p_w = p_i + (r_i+r_w)·g` ⇒ `M_w = R(dir_w(i) → −g)`) + `relax` (PBD: edges move half each + non-edge repulsion + centroid zeroing) ② `sigma_field` (`σ_global = V/E`, `σ_min/median/max`, `σ_v`, `∇σ_v`) + `anchors` (**gravity proxy = local σ extremum**: `∇σ_v < 0` convergence / `> 0` source) ③ `redshift_proxy` (along BFS shortest path accumulate relative σ change) ④ `export_obj` / `export_json`. Entry point `global_projection()`.
  **Verification**: `selftest=1` all green — rotation tool (`rot_from_uv`, including 180° reverse); icosa t=0 tree edges 11 **all** = `2r(5)=7.569398`, holonomy 19 mean `11.9701`; isometric graph `σ≡0.4`, **no anchors**, redshift `≡0`; 60 rounds and 150 rounds converge to **same platform** (residual readings consistent to `1e-6`); determinism (same input twice point-by-point identical); OBJ/JSON export structure self-consistent (12 `v` + 30 `l`).
  **Frame-by-frame readings** (`seed=icosa nframes=5 cap=12 dmin=3 dense`, `iters=60`):

  | t | V | E | σ=V/E | σmax | Anchors± | z_max | Tree residual | Edge residual (mean/max) | Holonomy (mean/max) | Overlap |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0 | 12 | 30 | 0.4000 | 0.4000 | 0 | 0.0000 | 8.88e-16 | 0.1339 / 0.3864 | 11.9701 / 26.1846 | 6 |
  | 1 | 32 | 90 | 0.3556 | 0.6667 | 32 | 2.3333 | 5.33e-15 | 0.0000 / 0.0000 | 8.8194 / 34.8658 | 0 |
  | 2 | 36 | 72 | 0.5000 | 0.6667 | 36 | 2.8571 | 1.78e-15 | 0.6398 / 2.7186 | 10.4416 / 33.9909 | 49 |
  | 3 | 66 | 192 | 0.3438 | 0.5000 | 66 | 3.8333 | 3.55e-15 | 0.5503 / 2.8123 | 15.6942 / 72.5876 | 39 |
  | 4 | 82 | 240 | 0.3417 | 0.6667 | 80 | 4.8333 | 5.33e-15 | 0.5307 / 2.4970 | 16.5335 / 79.5305 | 55 |

  Reading highlights: ① **Tree edge residual always `~1e-15`** (ensured by alignment propagation construction; refined coordinates measured separately, `tree_res_max` no longer precise — hence `global_projection` returns `distortion` (refined) and `distortion_raw` (original) separately); ② **Isometric graph (t=0) has no anchors, redshift ≡ 0**, heterogeneity appears later with non-empty anchors (from t=1 `32 → 80`); ③ `z_max` monotonically increases (`0 → 4.83`), while `σ_global` fluctuates between `0.34–0.50`; ④ Edge residual/overlap **non-monotonic** (t=1 exactly `0` — this frame has an exact solution, see N7); ⑤ `holonomy` is curvature reading, **not** error.
  Reproduce: `…\python3.13.2\python.exe l2_projection.py selftest=1`; `… l2_projection.py seed=icosa nframes=4 cap=12 dmin=3 iters=60`
- **Bifurcation N7 (2026-09-13, record → experiment → falsification) // Is `A_v ≤ 2π` a necessary condition for three-dimensional tangent embedding**:
  **Cause**: L2 `_demo` exposes contradiction — t=0 (regular icosahedron, isometric ⇒ analytically must be exactly embeddable) PBD only to residual `0.1339`, 6 overlaps; t=1 (§7 bifurcation N4 ruling 1 ruling 12 out-of-bound points) instead to `5e-16`, zero overlaps. **If ruling 1 holds, t=1 cannot have exact embedding.**
  **Evidence script**: `%TEMP%\l2_diag_n7.py`.

  | Test item | Reading | Verdict |
  |----------|--------|---------|
  | t=1 exact embedding (root=0) | Edge residual max `3.55e-15`; non-edge minimum gap `+3.8468` (no penetration) | ✅ **Exact tangent embedding exists** |
  | Same graph with `root=16` | Edge residual max `1.242`; non-edge minimum gap `−1.2206` | ⚠️ PBD **initialization dependent** (only to nearest local minimum) |
  | t=1 out-of-bound points `A_v/2π` (12 `deg=10` points) | L1 reading `1.3820` = **position back-calculation** `1.3820` | ✅ This embedding indeed achieves `A_v > 2π` |
  | Analytical witness: zigzag 6-closed direction ring (±30° latitude alternately) | Each turn angle `82.82°` ⇒ total length `496.92° = 1.3803×2π > 2π` | ✅ Closed spherical polygon total length can be > 2π (almost coincides with t=1's `1.3820`) |
  | t=2 cold start / hot start (using t=1 solution as initial value) | Residual `1.747`, gap `−3.069` / residual `5.3e-9`, gap `−0.0000` | ⚠️ Frame-by-frame **incremental embedding** exists (hot start reaches tangent limit) |

  **Verdict (N4 ruling 1's "necessary condition" part is falsified):**
  1. **`A_v > 2π` does not imply "embedding impossible".** `A_v` correct semantics is "**total length of closed spherical polygon formed by v's rotation-system neighbor directions**", `δ_v = 2π − A_v` is the **intrinsic curvature sign** at v (`δ_v > 0` convex fold, `δ_v < 0` **saddle point**). **Saddle-point vertices can be fully embedded in R³** — t=1's 12 `A_v/2π = 1.3820` points simultaneously have exact tangent embedding, and position back-calculation `A_v` matches L1 reading bit-for-bit. `2π` boundary only applies to "**convex polyhedron / planar unfoldable**".
  2. **N4 ruling 1's handling (do not delete history, change wording)**: `viol` is downgraded from "**global embedding locally impossible criterion**" to **curvature reading** (`l1_projection.angle_defect` docstring and §6.3 have been revised). **N4-A (global embedding solving) thus loses its decisive criterion**, now only three weak evidences remain: ① rigidity count `d = 3V−6−E ≡ 0` (pure triangulation ⇒ zero relaxation; but N4 footnote already points out "equal number of edges" is insufficient to conclude no solution); ② solver unreliable (PBD initialization dependent + v1.x §A.13.3 divergence); ③ paradigm choice (L0 has no geometry, projection layer only needs id + rotation system). **Per maze rules: N4-A not falsified ⇒ path-based derivation** — L2 takes workpiece path (already implemented), "**global / incremental embedding solvability**" is recorded as **open item** (N7 hot start reading is positive sign), to be verified independently later.
  3. **Byproduct**: `holonomy` and `viol` **unrelated** (viol points / non-viol points `1.07×–1.09×`) ⇒ "**curvature cannot be planar accommodated**" and "**local angle overstepping**" are two different things; PBD residual is only an **upper bound**, not evidence of inembeddability.
  4. **This local model collaborative verification (`qwen2.5-coder:14b`)**: judges that "Proposition 2 (`A_v > 2π` ⇒ inevitable overlap) is correct" — **directly conflicts with the reading (a)**, and it itself in Q2 says "`A_v/2π = 1.3820 > 1` does not imply that R³ tangential embedding is impossible" (self-contradictory); Q3 (relaxation residual cannot determine non-existence of exact embedding) judgment aligns with empirical results; judgments on Proposition 1 and 3 are correct. ⇒ **This local model is unreliable in this round, and all decisions must be based on code empirical results** (same failure mode as N4 in §7).
  Reproduce: `python %TEMP%\l2_diag_n7.py`
- **Bifurcation N8 (2026-09-13, record → experiment → exclusion // adopt N8-A, leave N8-B open) // §6.5 missing: what should L3 be projected onto**:
  **Cause**: The user requested "implement L3 global projection according to §6.5" using the same phrasing as "§6.4 → L2". **Verification**: The document has no §6.5, and the full text has no mention of "L3"; under `openSPUM/src/`, there is only `l0/l1/l2` — that is, **a reduction missing** (not an implementation missing).
  **Branch** (both are natural readings of "L3"): ① **N8-A frame sequence layer (cross-frame / time axis)**; ② **N8-B multi-scale layer (structural axis: node → 12 crystallites → 42 clusters → macroscopic shell aggregation)**.
  **Evidence** (not based on preference, but on the registered semantics of existing materials):

  | Evidence | Points To |
  |---------|-----------|
  | `图论/spum-图论指标.md` GT-005~008 (δ / Δμ / d_topo / σ) **are all cross-frame quantities**, and it notes "record the tension field state of the current frame, and provide numerical boundaries for the next frame..." | N8-A |
  | Two-tier closure criterion `δ<θ_δ ∧ Δμ<θ_μ for k frames` (N013/N016 revised) — **cannot be defined without frame sequence** | N8-A |
  | §7 Build Roadmap "Phase Five: Evolution Observation" — the **continuation** verification of net annihilation effect, concentration phenomena, and Imperfection Theorem | N8-A |
  | §7.4 has adopted asymmetric dynamics (`vminus=dense` activates V⁻) — its entire meaning lies in **cross-frame readings** | N8-A |
  | N015/N016 (anchor drift rate, complete closure verification) in the new pathway **still lack corresponding code** | N8-A |
  | `spum-structure.md` "Fractal-constrained growth / hierarchy `d_max` / 12 crystallites" | N8-B (has materials, but its **combination criteria have been covered by §6.3 `crystallites`**) |

  **Verdict**: Adopt N8-A (frame sequence layer; 5 pieces of evidence, all based on registered semantics); **N8-B is not excluded, left as an open item** — multi-scale aggregation is currently handled by §6.3's `crystallites` (pure combinatorial identification of 12/42), and an independent "aggregation projection layer" will be initiated when specific reading requirements arise.
- **Implementation of Phase Four Item Five (2026-09-13) // L3 frame sequence projection implemented according to §6.5**: Add `src/l3/l3_projection.py` (**read-only cross-frame projection**; does not participate in L0 rules, **independent of L2 coordinates**). §6.5 four items correspond one by one: ① `frame_trace` / `attr_series` ② `delta` / `anchors_of` + `anchor_drift` / `dtopo` / `sigma_global` ③ `closure_series` ④ `export_csv` / `export_json`. Entry `global_evolution(core, nframes)`.
  **Verification**: `selftest=1` all green — id alignment, δ generalized reading (δ_2=0 / δ_5=0 / δ_6=1.0), anchorconvention (equal-degree graph has no anchors), d_topo three conventions (construct single-edge deletion: `d_topo=1/30`, `d_norm=1/59`, `d_w=w⁻`), closure criterion synthesis test cases (δ all 0 while Δμ=9 ⇒ not closed), frame-by-frame self-consistency (`d_topo/d_norm/d_w` uniquely determined by `n_plus/n_minus`), attribute series `σ=2/deg`, determinism (two runs are frame-by-frame consistent), CSV/JSON round-trip.
  **Frame-by-frame readings** (`seed=icosa nframes=8 cap=12 dmin=3 dense/any`):

  | t | V | E | σ=V/E | ⟨deg⟩ | maxdeg | δ_κ | Δμ | churn | d_topo | d_w | V⁺/V⁻ | Events (b/d/br) |
  |---|---|---|-------|--------|--------|------|-----|-------|--------|-----|--------|------------------|
  | 0 | 12 | 30 | 0.4000 | 5.000 | 5 | 1.0000 | — | — | — | — | — | — |
  | 1 | 32 | 90 | 0.3556 | 5.625 | 10 | 0.6250 | 0.000 | 1.000 | 0.667 | 60.0 | 60/0 | 20/0/0 |
  | 2 | 36 | 72 | 0.5000 | 4.000 | 7 | 0.8889 | 1.750 | 0.111 | 0.467 | 72.0 | 12/30 | 4/0/30 |
  | 3 | 66 | 192 | 0.3438 | 5.818 | 12 | 0.5152 | 3.333 | 0.455 | 0.625 | 120.0 | 120/0 | 30/0/0 |
  | 4 | 82 | 240 | 0.3417 | 5.854 | 12 | 0.6098 | 0.750 | 0.220 | 0.200 | 48.0 | 48/0 | 16/0/0 |
  | 5 | 98 | 264 | 0.3712 | 5.388 | 11 | 0.5102 | 0.800 | 0.184 | 0.273 | 96.0 | 48/24 | 16/0/24 |
  | 6 | 154 | 452 | 0.3407 | 5.870 | 12 | 0.5325 | 1.957 | 0.403 | 0.416 | 188.0 | 188/0 | 56/0/0 |
  | 7 | 226 | 568 | 0.3979 | 5.027 | 10 | 0.5575 | 1.622 | 0.345 | 0.556 | 416.0 | 216/100 | 72/0/100 |
  | 8 | 432 | 1243 | 0.3475 | 5.755 | 12 | 0.5810 | 2.814 | 0.477 | 0.619 | 816.0 | 722/47 | 206/0/47 |

  **Reading highlights and verdicts**:
  1. **id is a reliable cross-frame anchor**: from t=0→t=1, the 12 old ids **all persist**, 20 new ones added, none disappeared (V 12→32→36, consistent with frozen readings in §7.4 bit-for-bit); id is not reused (`nid` is monotonic), so `born/dead/persistent` is a **pure set reading**, and the judgment does not depend on `nid` (`nid` is only used for verification).
  2. **Closure criterion under adopted dynamics never holds**: δ_κ ∈ [0.51, 1.00] throughout the process ≫ θ_δ=0.03 ⇒ `closed=False`, continuous segments are empty. This aligns with "complete topological closure is unreachable in standard evolution" (N016 / dual-layer criterion of Imperfection Theorem) — **mutually verified under the sameconvention** — L3 first allows this assertion to be **reproducible** in the new pathway (previously only `src/spum_graph/dangling.py` in v1 pathway).
  3. **δ arm two-convention are both "constant verdicts" (original judgment "both arms degenerate" was half wrong, corrected on 2026-09-13; open item closed)**:
     · `δ_dmin` (residual arm) **always meets the standard** — 11/11 frames `< θ_δ` (10 frames all 0, only t=9 = 1/558 ≈ 0.001792 > 0, longest continuous 11) ⇒ it is not "present every frame", but "**present in large V⁻ frames**";
· `δ_κ` (generalized arm) **never meets the threshold** — measured `[0.5102, 1.0000]` (minimum at frame t=5), with a minimum distance of 0.4802 from θ_δ, and 0/11 frames meet the criterion. **It is not "approximately 1"** — `Σ(6−deg)=12` only ensures the sum is positive, not that almost all nodes have `deg<6`.
 ⇒ The two criteria have **no discriminative power** for closure (constant judgment), making the AND criterion involving δ arm **always false and trivial**. **The actual validation falls on the central arm**: `Δμ` alone `7/11` meets the frame criterion, but **the longest consecutive frames meeting it is only 4 (t=4–7) < k=5**, and at t=3/8/10, single frames exceed ⇒ **the Imperfection Theorem is independently carried by the central arm and is non-trivial** (not relying on "δ never meets the threshold" to spin idle).
 **Candidate conventions (if δ arm discriminative power is to be restored in future)**: variance of σ field, frame-to-frame change rate of δ `|Δδ_κ|`, or use `churn` as the second arm; **this round does not arbitrarily alter registration semantics**.
 4. **Δμ and churn must be viewed together**: at t=0→1, anchor set goes from ∅ to full creation ⇒ Δμ=0.000 but **churn=1.000** (here `Δμ=0` is a **degenerate illusion**, noted in the code docstring); for t≥2, Δμ ∈ [0.75, 3.33], many frames > θ_μ=2.0, but **never 5 consecutive frames meet the criterion**.
 5. **d_topo and L0 event counting cross-check**: `V⁻ = |E_prev − E_cur|` and event `broken` **match frame-by-frame** (t=2: 30/30, t=5: 24/24, t=7: 100/100, t=8: 47/47; in this trajectory `dead ≡ 0`, i.e., no whole-point deletion — if `dead>0`, `V⁻` should also include edges connected to deleted points); `V⁺` is usually `= 3×born` (conical triangulation = adding 1 point and 3 edges: t=1 `20→60`, t=4/5 `16→48`), and for conical k-gon holes it's `k×` (t=3 `30→120`).
 Reproduce: `…\python3.13.2\python.exe l3_projection.py selftest=1`; `… l3_projection.py seed=icosa nframes=8 cap=12 dmin=3 vminus=dense vplus=any`

- **Phase V first implementation (2026-09-13) // L4 evolution observation layer implemented as per §6.6, net annihilation effect confirmed**: added `src/l4/l4_evolution.py` (**read-only cross-frame observation**, not involved in L0 rules, no state write-back, independent of L2 coordinates). §6.6 four sets of readings correspond one by one: ① partition event account `event_account` / `annihilation_profile` ② concentrated phenomena `concentration_series` ③ Imperfection Theorem `imperfection_series` ④ entry `observe` + `export_csv` / `export_json`.

  **This round of collaborative derivation (`qwen2.5-coder:14b`, specification `l4_spec1.txt` ← draft `l4_out1.py` / 4845→9570 characters)**: review judgment **skeleton correct** (function names in 3.1–3.6, return key names, S1–S7 steps all correct), **5 hard bugs + 1 gap** — ① `snap` / `edge_set` called but never defined (spec §2 requires copy, draft missed) ② `l1_projection` not imported yet used directly as `l1_projection.sigma_of` ③ **`zone([])` returns `"core"`** (`all()` is always True for empty sequence; spec fixture B requires `"bg"`) ④ `concentration_series`'s `V==0` branch missing `n_over/n_flat/n_under/n_sat` ⇒ empty graph `NameError` ⑤ `export_csv` writes `None` as `"None"` instead of empty string. → follow "keep skeleton, rewrite body"; fixtures A–F all passed.

  **Measured readings (seed=icosa dmin=3 dense/any, `cap=12`):**

  | t | V | E | maxd | nSat | nOver | V⁺core | V⁺sh | V⁺bg | V⁻core | V⁻sh | V⁻bg | netcore | n_follow | viol_sat |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 1 | 32 | 90 | 10 | 0 | 12 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 |
  | 2 | 36 | 72 | 7 | 0 | 4 | 0 | 4 | 0 | 30 | 0 | 0 | −30 | 0 | 0 |
  | 3 | 66 | 192 | 12 | 4 | 12 | 0 | 17 | 13 | 0 | 0 | 0 | 0 | 0 | 0 |
  | 4 | 82 | 240 | 12 | 12 | 20 | 0 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
  | 5 | 98 | 264 | 11 | 0 | 20 | 0 | 16 | 0 | 24 | 0 | 0 | −24 | 0 | 0 |
  | 6 | 154 | 452 | 12 | 16 | 52 | 0 | 40 | 16 | 0 | 0 | 0 | 0 | 0 | 0 |
  | 7 | 226 | 568 | 10 | 0 | 59 | 0 | 72 | 0 | 100 | 0 | 0 | −100 | 0 | 0 |
  | 8 | 432 | 1243 | 12 | 19 | 136 | 0 | 182 | 24 | 47 | 0 | 0 | −47 | 0 | 0 |
  | 9 | 558 | 1390 | 11 | 0 | 136 | 0 | 131 | 0 | 216 | 0 | 0 | −216 | 48 | 0 |
  | 10 | 960 | 2657 | 12 | 25 | 283 | 0 | 345 | 57 | 115 | 0 | 0 | −115 | 0 | 0 |

  `V` trajectory matches frozen readings in §7.4 bit-for-bit; profile: `rho_core = +1.0000` (net annihilation), `rho_sh = −1.0000`, `rho_bg = −1.0000` (net creation); `dE_core = −532`, `dE_sh = +823`, `dE_bg = +130`; O(2E) pure Python, 10 frames in seconds.

  **Verdict:**
  1. **"Net annihilation effect" confirmed in over-dense core regions**: `V⁺core ≡ 0` while `V⁻core > 0` (non-zero in 6/10 frames, total 532) ⇒ `rho_core = +1.0000`. This is the **executable reading** of §4 "gravity = net annihilation effect" on pure combinatorial L0 (**qualitative description only existed in v1 link and §7.4 before**).
  2. **`viol_sat ≡ 0` (10 frames × 2 seeds)**: generation sites never contain saturated angles ⇒ the gate for generation suppression is `slack`/`resolve` budget (**cap load**), same source as §7.4 X1. **The theorem's structural origin in this link is explicit guarding, not emergence** — record truthfully, no overstatement.
  3. **Two approaches (to prevent "identity as measurement")**: ① **V⁺ must be partitioned by site** — partitioning by edge endpoints would make `V⁺core ≡ 0` a **forced illusion** (each new edge has one endpoint that did not exist at the start of the frame); after partitioning by site, `V⁺core` is non-zero in cap=64 trajectory t≥7 (`96/8/160/788`). ② **V⁻'s partition distribution is rule-forced** (`dense` criterion both ends over ⇒ `V⁻sh ≡ V⁻bg ≡ 0`) ⇒ `rho_bg = −1.000` is a **identity**; what can be tested is the partition distribution of V⁺ and core net sign, and **core net annihilation is not identity** — cap=64 weakened to `rho_core = +0.9284` (`(28350−1052)/29402`, `dE_core = −27298`).
  4. **Imperfect reading degradation (open item)**: At frame end, `deg < dmin` occurred in only 1 of 10 frames, at t=9 (the creation events in the same frame rescued points whose edges were severed), `dead > 0` also occurred only at t=9 (5 instances, with `n_follow = 48`). This is of the same family as the δ-arm degradation in §6.5 — "edge imperfection" is **heavily masked** by **same-frame creation** under adoption dynamics.

  > **Local model collaborative verification (`qwen2.5-coder:14b`, 5 propositions)**: Judged "correct" 4/5 — P1 (partitioning by edge endpoints ⇒ illusion), P2 (`dense` ⇒ `V⁻sh ≡ V⁻bg ≡ 0`), P5 (`ΔE = Σ_{new}deg − (n_minus+n_follow)` is a **tautology**) are consistent with measurements; however, **P1's judgment is correct but the reasoning is wrong** (it claims "the angle of a triangular face must be ≤ 6" — in reality, `V⁺sh > 0` indicates creation on over-dense angles, the correct reasoning is "new edges must have one endpoint as a new node not present at frame start"). **P3 / P4 arithmetic errors are entirely wrong** (P3 gives `ΣV⁻ = 562`, correct value is `532`; P4 gives `ΣV⁺ = 1112 / ΣV⁻ = 29220 / ρ = 0.961`, correct value is `1052 / 28350 / +0.9284`) ⇒ same failure mode as in §7 branch N4/N7: **the local model only performs conceptual checks, all numerical values must be verified by code**.
  Reproduce: `…\python3.13.2\python.exe l4_evolution.py selftest=1`; `… l4_evolution.py seed=icosa nframes=10 cap=12 dmin=3 vminus=dense vplus=any`

- **Phase V Item ③ Implementation (2026-09-13) // "Closure subgraph internal vs external" is deemed degenerate at the component level, merged into "structural closure audit"**:
  **Evidence first (probe before coding)**: Probe script counts `n_comp` / boundary edges / χ / Σ(6−deg) / 5-regular components for adoption trajectories frame by frame — both icosa and patch subgraphs, **frame by frame**, yield `n_comp=1`, `boundary_edges=0`, `χ=2` (single connected closed surface) ⇒ "closure subgraph internal vs external" **degenerates at the component level**, no distinction exists at the component level. Hence, Item ③ **does not establish a separate "internal/external" criterion**, its non-trivial content is Item ①'s `core`/`bg` partition (over-dense core = local closure internal, background = external).

  **This round of construction (pure increment)**: `src/l4/l4_evolution.py` adds §6.6 ⑤ `closure_of(net)` / `closure_series(nets)` (pure combinatorial read-only, coordinate-independent), incorporated into `observe()` output key `closure` and JSON export; fixtures A–G all passed; L0/L1/L2/L3 selftest regression all green.

  **Collaborative derivation (`qwen2.5-coder:14b`, specification `l4_spec2.txt` ← draft `l4_out2.py`, 3274 characters / eval_count=1170)**: Review judges **skeleton and body mostly correct** (returned key names R7, S1–S6 steps, empty graph/isolated point boundaries all correct), only 2 fixes needed — ① `from collections import ...` written inside the function (should be moved to the module top) ② docstring mistakenly describes "χ ascending" as an independent sort (in reality it's a **compound key** of `(vertex count descending, χ ascending)`). → Implemented as "keep skeleton, rewrite body".

  **Measured readings (icosa dmin=3 dense/any cap=12; patch same conclusion)**:

  | Frame | n_comp | boundary_edges | χ | Σ(6−deg) | holes |
  |---|---|---|---|---|---|
  | t=0 (icosa seed) | 1 | 0 | 2 | 12 | 0 |
  | t=2 | 1 | 0 | 2 | 72 | 30 |
  | t=10 | 1 | 0 | 2 | 446 | 217 |

  **Verdict**:
  1. **Item ③ degenerates at the component level** (structurally impossible, not "not measured this round") — always a single closed surface across the network ⇒ no distinction of internal/external at the component level.
  2. **Closure identity `Σ(6−deg) = 12 + 2·d` (`d = 3V−6−E`) holds frame by frame** (11 frames × 2 seeds), and **this formula only holds for χ=2** — general form remains handshake lemma `Σ(6−deg) = 6V−2E`; empty graphs (protocol takes `d=0`) and graphs with isolated points/multiple components **are not applicable** (fixture G already fixes these boundaries as assertions: empty graph → all zeros; icosa + isolated point → `n_comp=2`, `χ=[2,1]`, `Σ(6−deg)=18`, `d=3`).
  3. **5-regular 12-node component (`find_crystallites` convention) only exists at t=0 seed** (`crystallites=[12]`), adoption trajectory t≥1 all `[]` — "12 crystallite closure" under adoption dynamics **does not naturally regenerate** (same family as in §6.3 "t=0 saddle points many, t=1 onwards disappear"). **→ Phase VI Item ② (2026-09-13) has been closed**: this is not a recognition failure, but **structural necessity** (L0-A11 component count does not increase (default `dmin=3`) ⇒ under single-connected seed, shell is coned and irreversible, `cap>5` leads to 12→32 one-way); recognition scope simultaneously tightened to **component level** `V=12 ∧ all deg==5` (old `len≥12` scope in V=14 would list `reg=False` false positives — 12 deg-5 nodes connected as a block but only 24 internal edges). See next item Item ② anchor and code §7.5.
  4. **Imperfect reading degradation (open item) remains unchanged**: this round did not modify L0, `n_under_end` occurred in only 1 of 10 frames at t=9.

  > **Local model collaborative verification (`qwen2.5-coder:14b`, 4 propositions)**: Judged "correct" 3/4 — P1 (`Σ(6−deg)=12+2d` derivation: `6V−Σdeg = 6V−2E = 12+2d`), P2 (this formula **only holds for χ=2**; empty graph with `d=0` gives `0 ≠ 12`), P3 (icosa + isolated point: `n_comp=2`, `χ=[2,1]`, `Σ(6−deg)=18`, `d=3`) consistent with measurements. **P4 two errors**: it calculates `boundary_edges` as `2E−2F=68` (**no such formula**; correct value is 0 — each edge in a closed surface belongs to exactly two faces), and thus concludes "discrepancy between measurement and calculation ⇒ cannot infer closed surface" — incorrect reasoning; its (d) judgment (under `n_comp≡1`, internal/external distinction not achievable) is correct, **but the reasoning is also wrong**. ⇒ Same failure mode as in §7 branch N4/N7 and this layer's first item: **the local model only performs conceptual checks, all numerical/formulae must be verified by code**.
  Reproduce: `…\python3.13.2\python.exe l4_evolution.py selftest=1` (fixtures A–G); `… l4_evolution.py seed=icosa nframes=10 cap=12 dmin=3 vminus=dense vplus=any`

- **Phase V Item ④ Implementation (2026-09-13) // Continuous verification of the Imperfection Theorem: edge-arm two-convention judgments are constant, actual verification falls on the central arm**:
  **Evidence (probe before coding, cross-layer comparison)**: On the same 10-frame adoption trajectory, parallel L3 (`δ_dmin` / `δ_κ` / `Δμ` / closure criterion) and L4 (`n_under_end`) — 

  | Arm / Criterion | Measured Range | Frames Meeting Threshold (threshold) | Longest Continuous (k=5) | Judgment |
  |---|---|---|---|---|
  | Edge-arm · Residual `δ_dmin` (deg<3) | All 0, only `0.001792`@t=9 | 11/11 (θ_δ=0.03) | 11 | **Always meets** (trivial) |
  | Edge-arm · Generalized `δ_κ` (deg<6) | `[0.5102, 1.0000]` (min@t=5) | 0/11 | 0 | **Always fails** (trivial) |
  | Central arm `Δμ` | `[0.000, 3.333]` (exceeds at t=3/8/10) | 7/11 (θ_μ=2.0) | **4 (t=4–7)** | **Non-trivial** |
  | AND (`δ_κ ∧ Δμ`) | — | 0/11 | 0 | **Always false** |

  **Verdict**:
  1. **Both ends of the edge arm are "constant verdicts"**: `δ_dmin` is always met (masked by **same-frame creation** ⇒ the correct expression is "**large V⁻ frames must appear**" rather than "appear in every frame"); `δ_κ` is never met (**not "≈ 1"** — `Σ(6−deg)=12` only ensures that this sum is positive, not that **almost all** nodes have `deg<6`; measured in the sparsest frame t=3, only 51.5%). ⇒ The AND criterion containing δ arms is **always false and trivial** on this link.
  2. **The actual verification falls on the central arm**: `Δμ` alone has the longest continuous sequence of only 4 frames `< k=5` ⇒ **the Imperfection Theorem is independently carried by the central arm and non-trivial** (not relying on "δ always not met" to spin idle).
  3. **Cross-validation**: L4 `n_under_end` and L3 `δ_dmin × V` are **frame-by-frame consistent** (only non-zero frame is t=9, with the same value `1/558`) — two layers independently implemented, matching the same trajectory.
  4. **Correcting the overstatement in L3**: The original comment "frame-end pruning ensures `deg ≥ dmin`" is invalid (only one layer is deleted, and the neighbors being trimmed are not cleared in the same frame); the assertion has been narrowed down, and a new 10-frame counterexample (t=9) and the frozen assertion that "the AND criterion never met in 10 frames" have been added.
  5. **The original "δ arms both degenerate = open item" has been closed** (§6.5 verdict 3 has been rewritten); if future recovery of δ arm resolution is needed, candidates: σ field variance / `|Δδ_κ|` frame-to-frame change rate / `churn`.

  > **This local model collaborative calculation (`qwen2.5-coder:14b`, 3 propositions)**: the verdict level 3/3 matches the measured "consistency" (nucleus verification of `closed=False`, residual arm degeneration, `δ_κ` minimum frame), **but the mechanism explanation and numerical values are wrong in multiple places**: P1 says `Δμ=2.814 > θ_μ` is "met `Δμ<θ_μ`" (reversed comparison direction, verdict correct by luck); P2's ①② mechanisms **are both wrong** (claiming "deletion only reduces current layer node count and does not affect other layers", "deleted nodes may have had degree less than dmin" — the real mechanism is **frame-end `dead=0` ⇒ no neighbor degree reduction ⇒ no residue; large V⁻ frames delete points, and their neighbors each drop 1 degree, but the deletion set is fixed by snapshot, only one layer is deleted ⇒ neighbors remain at frame end**); P3(b) fabricates over-the-limit frames `t=3,4,6,7,9,10` (correct `t=3,8,10`), P3(d) directly answers "cannot" **wrong** (correct answer "**can** determine it is always false, because `δ_κ` is always `> θ_δ`"). ⇒ Same failure mode as N4/N7 and the first two items in this layer: **the local model only performs verdict-level verification; mechanisms and numerical values must be verified by code**.
  Reproduce: `…\python3.13.2\python.exe l4_evolution.py selftest=1`; `… l3_projection.py selftest=1` (includes frozen assertion in item ④); `… l3_projection.py seed=icosa nframes=10 cap=12 dmin=3 vminus=dense vplus=any`

- **Phase Six Item ① Implementation (2026-09-13) // Reconnection to L0 + Platform Secondary Potential: `phi1n` + `recon_settle` Dual Shutdown**:
  **Construction (`src/l0/l0_core.py` pure increment, no GPU / no `combinatorial_proto.py`)**: ① `flip_step` adds `mode="phi1n"` (predicate `ΔΨ ≤ 0`, includes neutral flip); ② adds `recon_settle(net, mode="phi1n", perturb=0, cap=4096)` — dual shutdown criteria (fixed point `done==0` ∨ **state key cycle hit**) + returns the **configuration with minimum Ψ on the ring/path** (in case of ties, the earliest occurrence is taken ⇒ independent of scanning order); ③ adds read-only helper `_net_key` / `_psi`; ④ CLI `mode` whitelist += `phi1n`, adds `settle=1` axis; ⑤ `selftest` item 10 fixed fixture A–E.

  **Collaborative derivation (`qwen2.5-coder:14b`, specification `l0_recon_spec1.txt` ← draft `l0_recon_out1.py`, 2462 characters / eval_count=914)**: Review verdict **skeleton correct** (S1–S4 complete, cycle/optimal recording logic consistent with specification), 2 places to fix — ① **hard bug**: `recon_settle` references `key` / `psi` but the specification only requires output of two functions ⇒ two helpers undefined (same failure mode #1 as `no_alive`); ② **redundant**: `sorted(_edge_set(...))` secondary sort (`_edge_set` already `sorted`). → Implementation adds helpers, removes redundancy.
  **Implementation caught by code verification** (both sides missed): specification original state key `tuple((v, tuple(n.rot[v])) for v in sorted(n.rot.items()))` **is wrong** — `sorted(dict.items())` iterates `(v, list)`, `n.rot[v]` is actually `n.rot[(v, list)]` ⇒ `TypeError: unhashable type: 'list'`; correct writing is `for v in sorted(n.rot)`. **Specification itself has a defect, exposed by runtime fixture** (same discipline as §7 branch N4/N7: any discrepancy with code must be resolved by code).

  **Measured readings (`phi1n` vs strict `phi1`, `perturb=3`, 9 Vs)**:

  | V | Ψ(phi1n) | Ψ(phi1) | phi1n shutdown method | cycle_len |
  |---|---|---|---|---|
  | 12 | 12 | 16 | halting (lower bound already taken, Φ₂ also has no handle) | — |
  | 13 | 14 | 14 | cycle | 2 |
  | 14 | 12 | 16 | halting | — |
  | 15 | 12 | 16 | halting | — |
  | 16 | 12 | 12 | cycle (t=5↔t=7) | 2 |
  | 20 | 12 | 14 | cycle | 3 |
  | 28 | 12 | 22 | cycle | 2 |
  | 42 | 12 | 16 | cycle (t=54↔t=56) | 2 |
  | 60 | 18 | 50 | cycle | 6 |

  **Verdict**:
  1. **`phi1n` is not worse than strict `phi1` on all 9 Vs, with 7 strictly better** (12/14/15/20/28/42/60) ⇒ §7.3 verdict ② "The correct form is 'Φ does not decrease' rather than 'Φ increases'" is **confirmed again**, and a distinction is given: strict `phi1` is stuck in pseudo-local minima on most Vs (V=60 stops at Ψ=50 while `phi1n` reaches 18).
  2. **§7.3 remaining branch ① (shutdown rule) closed: dual criteria**. "Non-halting" **is not unstructured drift, but a limit cycle** (cycle length 2/3/6; pre-period 5/9/54…, V=42 enters the cycle at t=54). "Potential reaching lower bound" alone is insufficient to halt — must be "**fixed point ∨ state key cycle hit**"; returning the optimal configuration is because the limit cycle **has no last frame**.
  3. **C-S lower bound `Ψ ≥ ⌈144/V⌉` always holds but is not tight** (V=13 measured 14 > 12; V=60 measured 18 > 3) ⇒ it can only be used as an **absolute check**, **not as a target value**; the actual platform value must be measured per V. (This secondary verification once mistakenly took this lower bound as a target assertion, has been changed to "relative to strict `phi1` not worse + exists strictly better" as the criterion.)
  4. **Invariant**: for all Vs matrix × all modes, `V/E/F/χ` and `Σ(6−deg)=12` are frame-invariant (edge flips are involutions, self-inverse); `recon_settle` does not modify the input net; predicate inclusion relation `phi1 ⊂ phi1n` is empirically `done(6) ≤ done(13)`.
  5. **Legacy (open item, not claiming completion)**: This round implements `recon` as an **independent callable operator + CLI axis**, with its axis semantics (count-preserving / stoppable / deterministic / invariants) fully defined; however, it has **not yet been integrated into `L0Core.frame()` main loop** — the question of "when to insert reconnection frames within a frame" and "how to queue with creation/deletion of edges" remains undefined. The original text for phase six item ① — "integrate as an optional axis of L0 (in parallel with creation/deletion of edges)" — leaves the **parallel arrangement** part pending for later.

  **Verification**: `l0_core.py selftest=1` passes all tests (10 items, including new fixtures A–E: tetra `phi1n` `done=0`/Ψ=36, icosa Ψ=12, icosa `phi2` `done=10`/Ψ=76, V=16 stop Ψ=12/cycle_len=2, V=42 stop Ψ=12); CLI regression — `recon=1 mode=phi1` reading remains unchanged, `recon=1 mode=phi1n` (without `settle`) reproduces periodic readings, `recon=1 V=16 mode=phi1n settle=1 perturb=3` → `cycle(5,7)/best_Ψ=12`, `V=42` → `cycle(54,56)/Ψ=12`, illegal `mode` exits with error 1.
  Reproduce: `…\python3.13.2\python.exe l0_core.py selftest=1`; `… l0_core.py recon=1 V=16 mode=phi1n settle=1 perturb=3`

- **Phase six item ② implementation (2026-09-13) // stable structure identification and regeneration: component-level 12-shell + gap list + component count conservation (L0-A11)**:
  **Construction (`src/l0/l0_core.py` pure incremental, no GPU / no `combinatorial_proto.py`)**: Add §7.5 subsection — ① `components(net)` (full graph connected components, deterministic); ② `shells(net)` (**component-level** 12-shell: a component with exactly `V=12` and all internal `deg==5`); ③ `gap_report(net)` (gap list: k + ordered boundary ring + k distribution); ④ `_union_nets(nets)` (multi-component construction: each network id is sequentially shifted, no interference); ⑤ `product_probe` / `_print_product` (frame-by-frame reading + component count conservation judgment); ⑥ CLI adds `product=1` axis and `extra=` (connected independently with `seed` to form multi-component seeds); ⑦ `selftest` item 11 (5 fixed fixtures).

  **Criterion tightening (core of this item)**: The old `find_crystallites` criterion finds components on **deg==5 induced subgraphs**, with only `len ≥ 12` as the criterion, **no requirement for regularity** ⇒ it is a **false positive generator** (R6 measured V=14 with exactly 12 deg-5 nodes forming **a single block**, old criterion lists `n=12` but `reg=False` — internally only 24 edges rather than 30, some edges point to deg-6). The strict criterion is only in the form of **component-level**: component has no edges with the outside ⇒ 12-node induced subgraph must be 5-regular; the only 5-regular planar triangulation with 12 vertices = regular icosahedron ⇒ **this component must be an independent connected component**.
  **Collaborative derivation** (`qwen2.5-coder:14b`, specification `l0_product_spec1.txt` ← draft `l0_product_out1.py`, 2763 characters / eval_count=1016): Review judge **skeleton correct** — 4 functions complete, S1–S3 no missing steps, `Counter`/`RotNet` both in scope (no failure mode #1 missing definitions), empty input `max(..., default=0)` guards all, no CuPy details risk. Only rewrite according to this document's style (remove semicolons, write in one line, add docstring basis), no hard bugs / no skeleton gaps.

  **Measured readings (evidence + control group, 2026-09-13)**:

  | # | Configuration | Reading | Judgment basis |
  |---|---|---|---|
  | R1 | `seed=icosa cap=64` 1 frame | t=0 `n_comp=1 n_shells=1`; t=1 `V 12→32`, `n_shells=0` | **Single connected seed shell is coned and irreversible** — independent shell cannot emerge from connected seed |
  | R2 | `seed=icosa extra=tetra cap=64` 2 frames | t=0 `n_shells=1` (id 0..11); t=1 `n_shells=0` (12→32); `n_comp` always 2 | Same as above: only when `cap−deg` budget ≡ 0 can shell be preserved |
  | R3 | `seed=icosa extra=tetra cap=5` 21 frames | `n_comp` always 2 (min/max=2/2); shell `[0..11]` **always present**; tetra arm 4→6 then freezes | **Regeneration path (constructive, only reachable)**: multi-component seed + `cap=κ−1=5` saturation ⇒ shell inert |
  | R4 | `seed=tetra cap=5` 5 frames | `V 4→6` freezes, `n_shells=0` | **Spontaneously impossible**: cannot self-organize into 12-shell |
  | R5 | `seed=patch cap=5` 5 frames | `V 19→19` freezes, `n_holes=1 k=12` | Same as above; and demonstrates gap = k≥4 face ring |
  | R6 | `recon_settle(V, phi1n, perturb=3)`, V∈{12,14,20,42} | `Ψ=12` always "exactly 12 deg-5 + rest deg-6"; deg-5 induced block with V: `[12/30 reg=T]`(V=12) → `[12/24 reg=F]`(V=14) → `[7/9, 5/7 reg=F]`(V=20) → `[2×4, 1×4 reg=F]`(V=42) | **Ideal defect spectrum ≠ closed 12-shell** ⇒ explain §7.3 case ③ this column V>12 all "no" |

  **Verdict**:
  1. **L0-A11 (component count not increasing) established as theorem — applicable domain limited to default `dmin=3` (already tested tightened)**: creation (face coning / hole coning) only adds edges and nodes within this component; edge deletion only merges two faces, does not split components; pruning deletes nodes at most eliminating entire component. ⇒ component count **not increasing frame by frame** (impossible to increase). Measured: R3/R4/R5 each 21/5/5 frames + 240 configuration matrix (`dmin=3`, see verification ②) always holds; `selftest` item 11 asserts `icosa⊔tetra cap=5` 21 frames not increasing. **Convergence/self-correction**: initial claim "always constant under closed triangulation" **too strong**, pressure test refuted and registered **3 counterexamples** (all CLI reproducible, see §9 L0-A11 line): ① `dmin=2 ∧ pairing=conserved` (low threshold open pruning channel + forced edge deletion can cut bridge edges/cut points) — `seed=icosa cap=12 dmin=2 pairing=conserved nframes=4` n_comp `1→2` (t=4, V 12→23), `seed=icosa extra=tetra cap=6 dmin=2 pairing=conserved nframes=3` `2→3` (t=2); ② open patch + large threshold pruning — `seed=patch cap=6 dmin=5 nframes=3` (V `19→7→2→0`, n_comp `1→1→2`); ③ open patch + `dmin=2 ∧ conserved` — `seed=patch cap=8 dmin=2 pairing=conserved nframes=4` (n_comp `1→2`, t=3). ⇒ Theorem only holds on **default `dmin=3` adoption trajectory**; `dmin=2` / open seeds + large `dmin` are registered out-of-bound areas.
  2. **"12 balls emerge as independent components" is structurally impossible under a simply connected seed**: independent shells require no edges to the outside, but under the default `dmin=3` the number of components does not increase (L0-A11) ⇒ once a shell is capped (`cap>5`), it permanently disappears (R1/R2). **§7.3 verdict ③ upgrades from "impossible in V>12 connected subdivisions" to "impossible in any simply connected evolution path"** (limited to `dmin=3` adoption trajectory; even in the `dmin=2` out-of-bound area — where component count can increase — empirical new components are never 5-regular blocks, `n_shells` is always 0, with no single instance of 0→1). ⇒ 12 shells can only be **initial conditions** (consistent with spum-core §7 "12 crystallites closed loop = the universe's first atom, a **starting point not an endpoint**").
  3. **Regeneration path is unique and constructive**: `_union_nets` creates multi-component seeds + `cap=κ−1=5` (saturation ⇒ shell `cap−deg` budget always 0 ⇒ inert). R3 empirical shells `[0..11]` remain in place for 21 frames, `n_comp` is always 2. **This is the positive explanation that `crystallites` are empty in the adoption trajectory `t≥1`**: under the adoption trajectory (simply connected icosa, `cap=12/64`), shells are **necessarily capped**, not a failure to identify but **they indeed do not exist**.
  4. **Gap is already a computable quantity**: `gap_report` provides k + ordered boundary rings + distribution, consistent with `stats()` frame by frame (R5 demonstrates `k=12`). **Gap direction `n̂` (GT-032/G4) is not in this round** — it's a geometric quantity requiring L1/L2 coordinates (L0 has no coordinates); this round only delivers the computable part, `n̂` belongs to item ③ (σ field coordinateization).
  5. **Legacy (open item, not pretending completion)**: This round delivers **read-only criteria + constructive helper + CLI + fixture**, without changing any evolution rules (no changes to `frame()` / `propose` / `commit` / `flip_step`). "**Regeneration**" in this round is **constructive** (given multi-component initial conditions, shells remain), **not** "spontaneously generating independent shells from a simply connected state" — the latter is excluded by L0-A11 (default `dmin=3`).
  
  **Verification (three-tiered progression, all passed)**: ① Fixture — `l0_core.py selftest=1` all green (11 items, 24 assertions; added item 11 with 5 fixtures: icosa `n_shells=1 V/E=12/30`, tetra `n_shells=0`, `icosa⊔tetra V=16 n_comp=2 n_shells=1 sizes=[12,4]`, punctured icosa → `n_holes=1 k=5 dist={5:1}`, `icosa⊔tetra cap=5` 21 frames `n_comp 2→2 frame by frame not increasing`); ② Properties — independent property scripts **240 configurations** (5 seed groups × `cap∈{5,12,64}` × `dmin` default 3 × `pairing∈{none,conserved}` × `vminus∈{dangling,dense}` × `vplus∈{any,gap}` × puncture∈{0,1}, each 3 frames) **0 failures** (⇒ applicable domain is default `dmin=3`; `dmin=2` / open seeds + large `dmin` out-of-bound area see verdict 1 counterexample): P1 component count not increasing, P2 shell criteria self-consistent (V=12 all deg5 ⇔ induced 5-regular/30 edges/**no edges to the outside**), P3 `gap_report` and `stats()` hole diameter consistent frame by frame, P4 component structure consistency (union=ids / pairwise disjoint / no edges between components / internally connected / ascending), P5 three criteria read-only, P6 `_union_nets` translation and determinism + empty graph boundary; ③ End-to-end — existing CLI 8 commands (default / `probe=1` / `bare=1` / `dyn=1` / `recon=1` / `recon settle=1` / new `product=1` ×2) all `exit=0` and readings unchanged, L1~L4 `selftest=1` all `exit=0`.
  > Secondary verification once reported 2 FAILs in P2, investigation found that the **test script itself** missed `//2` (mistakenly took internal degree and 60 as edge count 30), not a product defect; corrected to zero. Same discipline as §7 item ① "bugs in the specification are exposed by running fixtures": **anything inconsistent with code is taken as per code**.
  Reproduce: `…\python3.13.2\python.exe l0_core.py selftest=1`; `… l0_core.py seed=icosa extra=tetra cap=5 nframes=21 product=1` (regeneration: shell remains in place throughout); `… l0_core.py seed=icosa cap=64 nframes=1 product=1` (control: shell capped 12→32); `… l0_core.py seed=patch cap=5 nframes=5 product=1` (gap `k=12`)

- **Phase six item ③ implementation (2026-09-13) // σ field coordinateization + gravity power verification: four power spectra + null hypothesis control + synthetic calibration discriminability**:
  **Construction (`src/l2/l2_projection.py` pure increment, only reads `net`, does not touch L0/L1)**: Added section ⑤ in §6.4 — ① `power_fit(r,y,min_pts=3)` (log-log least squares; only accepts `r>0 ∧ y>0`, insufficient/`sxx=0` → `slope=None`) ② `radial_derivative(r,y)` (central difference, endpoint one-sided, `den=0 → 0.0`, `m<2 → empty`) ③ `hop_integral(r,σ,σ₀)` (hop integral `ΔN(r_k)=Σ_{j>k}(σ_j−σ₀)·w_j`, right endpoint bin width) ④ `radial_bins(net,pos,center=None,nbins=8)` (centroid default; `r_max−r_min<1e-12` → single bin degenerate; only retain non-empty bins) ⑤ `gravity_spectrum(...)` (**degenerate guard** + four fits (derivative path takes `abs` amplitude) + tolerance band judgment) ⑥ `null_control()` (zero hypothesis control with equal-degree icosa t=0) ⑦ `synthetic_calibration(rs,fit_slice=(2,6))` (injected calibration, only use subinterval fit to suppress right endpoint discretization edge effects) ⑧ `_spectrum_demo()` + CLI `spectrum=1` axis ⑨ `verify_fixtures()` (fixtures A–F, attached to `selftest`).

  **Collaborative derivation (`qwen2.5-coder:14b`, specification `l2_sigma_spec1.txt` ← draft `l2_sigma_out1.py`, 11028 characters / eval_count=3982)**: Review verdict **skeleton correct** (9 functions complete, S steps no missing, key names copied, empty input guards complete), need to fix 3 places — ① **hard bug**: draft wrote `sinks, sources = anchors(net)`, but `anchors` returns **dict** not a tuple (same failure mode #1 in family); ② **specification (original plan) defect**: derivative path `∇σ`/`∇ΔN` under `σ∝r⁻²` is **always negative**, filtered out by `power_fit`'s `y>0` condition **entirely** ⇒ fitting must take **amplitude** `abs()` (power only reads shape); ③ **semantic defect**: original verdict `abs(a_exp+2)<abs(a_exp+3)` would also call −0.65 / +1.51 as `newton` ⇒ change to **tolerance band ±0.35** + new `neither` verdict. → Implementation revised according to three corrections. ⚠️ **Implementation again caught 1 specification error by code review**: fixture D original assumption "equal-degree graph centroid ⇒ single bin" **does not hold** (centroid ≠ center of regular icosahedron ⇒ 7 bins, `r=[5.0691,7.2704,8.5031,9.6317,11.5472,12.0069,16.2685]`, `n=[3,1,1,2,3,1,1]`), fixture assertion rewritten accordingly (same discipline as §7 items ①②: anything inconsistent with code is taken as per code).
  **Measured readings (evidence as of 2026-09-13):**

  **(a) Resolvability calibration (synthetic injection, four paths must restore injection power)**:

  | Injection | slopeσ (expected) | slopeΔN (expected) | slope∇σ (expected) | slope∇ΔN (expected) |
  |---|---|---|---|---|
  | `σ∝r⁻²` | **−2.000** (−2) | −1.121 (−1) | **−3.000** (−3) | **−2.000** (−2) |
  | `σ∝r⁻³` | **−3.000** (−3) | −2.029 (−2) | **−4.000** (−4) | **−3.000** (−3) |

  ⇒ Three paths **exactly restored**; only the `ΔN` path had discretization deviation (maximum 0.121 < fixture tolerance 0.25) ⇒ **r⁻² / r⁻³ resolvable** (judgment with ±0.35 can correctly classify).

  **(b) Null-hypothesis control (equal-degree graph icosa t=0):** `σ range = 0.00e+00, anchor = 0, z_max = 0.00e+00 ⇒ no signal on all paths`. ⚠️ **Without degeneracy guard, this graph once gave false slope `slope∇σ=−0.75`** (pure 1e-16 floating-point noise was amplified by **non-uniform bin spacing**) — this is the core evidence for this item's **falsifiability**.

  **(c) Adoption trajectory frame-by-frame spectrum (`seed=icosa cap=12 dmin=3 nbins=8`, relaxed 60 rounds):**

  | t | V | E | bins | slopeσ | slopeΔN | slope∇σ | slope∇ΔN | verdict |
  |---|---|---|---|---|---|---|---|---|
  | 0 | 12 | 30 | 3 | `--` | `--` | `--` | `--` | undefined |
  | 1 | 32 | 90 | 2 | `--` | `--` | `--` | `--` | undefined |
  | 2 | 36 | 72 | 8 | 0.1281 | −1.1360 | −1.5831 | −0.6455 | neither |
  | 3 | 66 | 192 | 8 | −0.1906 | −1.4204 | 0.6787 | −0.7781 | neither |
  | 4 | 82 | 240 | 8 | 0.0202 | −0.8939 | 2.7294 | 0.0726 | neither |
  | 5 | 98 | 264 | 8 | 0.2063 | −0.2095 | 0.9429 | 1.5090 | neither |
  | 6 | 154 | 452 | 8 | 0.1535 | −0.4123 | 2.0172 | 0.9511 | neither |
  | 7 | 226 | 568 | 8 | 0.0830 | −0.5007 | 1.3679 | 0.6581 | neither |
  | 8 | 432 | 1243 | 8 | 0.0876 | −1.0507 | 0.7491 | 0.0796 | neither |

  **Verdict:**
  1. **Four power-law readings are falsifiable** — the criterion does not rely on "expected values coincidentally appearing," but on **two independent controls**: (a) synthetic calibration three paths exactly restore r⁻²/r⁻³ (**resolvability is established**); (b) equal-degree graph null hypothesis has no signal on all paths (false signals eliminated by degeneracy guard). ⇒ Only truly close to −2 is called `newton`, close to −3 is called `trap`, otherwise `neither`.
  2. **Adoption trajectory `t=2..8` all `neither`** (`slope∇ΔN` fluctuates between −0.65…+1.51, no path nears −2 / −3) ⇒ **truthfully recorded: no gravitational power-law signal was read on this coordinate workpiece in this round**. This is **not** a refutation of gravity (L2 is a **projection workpiece**: edge residual mean 0.134 / max 0.386, not real geometry), nor a reading defect ((a)(b) have proven the readings are valid) — it only indicates that **the current adoption trajectory + coordinate system** has not yet grown a clean radial power-law σ profile (at t≤3, V is small, samples in bins are sparse; after t≥4, gradient paths are dominated by workpiece distortion). ⇒ **This prediction remains an open item for this platform, and this round does not conclude with data**.
  3. **`t=0/1`'s `undefined` has two different sources** (already fixed in CLI footnote): ① t=0 is an **equal-degree graph** ⇒ triggers degeneracy guard (σ profile has no structure); ② t=1 non-empty bins = 2 < `min_pts=3` ⇒ **insufficient samples** (its σ range 0.2/0.6667 **does not trigger** the guard). ⇒ **"Insufficient samples" and "no structure" must be read separately**, not both as "no signal".
  4. **σ's power-law convention has two interpretations, no judgment** (user judgment to deliver four readings per authoritative convention) — see below "σ convention divergence registration".
  5. **Legacy (open item, not falsely claimed as completed):** (i) `fit_slice` is only used for synthetic calibration, **fit window selection on adoption trajectory is undetermined** — must be resolved together with item ④ "finite window operator" ⇒ **already resolved in item ④** (`fit_slice` remains only in `synthetic_calibration`; adoption trajectory uses `radial_window` automatic convention, see below anchor judgment 2); (ii) `nbins`'s effect on readings is not scanned (fixed at 8 this round); (iii) notch direction `n̂` (GT-032/G4) **still pending implementation** — it's a geometric quantity, needs local face normal average on this coordinate system; item ② judgment 4 has assigned `n̂` here, **not fulfilled in this round, truthfully recorded**.

  > **σ convention divergence registration (2026-09-13, no judgment):** The same observable `a` has two σ power-law interpretations, **both return to `a∝r⁻²`**, but intermediate quantity power laws differ —
  >
  > | Source | σ(r) convention | Observable expression |
  > |---|---|---|
  > | `docs/Topological_Emergence/SPUM_Topological_Emergence.md` proposition 3 + corollary 3 | `σ ∝ r⁻¹` | `a = c²∇σ` |
  > | `物理学/基本相互作用/引力.md` lemma 3 + corollary 3a (**authoritative**) | `σ ∝ r⁻²` | `a = c²∇(ΔN)`, `ΔN ∝ Ṁ/r` |
  > | `.trae/rules/spum-evolution.md` §4.2 | — | `a = −2kc²m/r³` (**already judged as trap path**, see `引力.md` §2.9) |
  >
  > ⇒ This item **follows the authoritative convention** and uses `slope∇ΔN` as the main criterion (−2 ⇒ newton), `slope∇σ` (−3 is trap) as **reference reading**; **does not judge the −1/−2 divergence in σ itself** (it involves σ field source term `σ ∝ Ṁ/r²` normalization, beyond this layer, must be judged separately in the physics layer). Three σ paths readings this round **are all truthfully listed** (the two columns `slopeσ` / `slope∇σ` in the above table are for this purpose).

  **Verification (three-tiered progression, all passed):** ① Fixture — `l2_projection.py selftest=1` all green (including new fixtures A–F added: A `power_fit` exact slope −2/−3; B `hop_integral` manual integer solution `[2,1,0]`; C `radial_derivative` manual difference `[−3,−1.25,−0.375]`; D null-hypothesis control degeneracy `degenerate=True / slope=None / label=undefined`; E synthetic calibration four paths deviation < 0.25 and judgment with classifiable; F icosa t=1 structural assertion `Σn=V`, four keys complete, label legal); ② Properties — synthetic calibration (table above (a)) three paths exact restoration + null-hypothesis all paths no signal + existing exports (OBJ/JSON) assertions passed within `selftest`; ③ End-to-end — **existing CLI regression bit-for-bit identical** (`selftest=1` readings match §7 "Phase four-term landing" frozen values exactly: t=0 aligned propagation 11 tree edges exact = 7.569398, holonomy 19 mean 11.9701, refined 60 rounds edge residuals mean 0.1339 / max 0.3864, equal-degree graph σ≡0.4 / no anchor / redshift≡0) ⇒ **new capabilities have not broken existing readings**.
Reproduce: `…\python3.13.2\python.exe l2_projection.py selftest=1`; `… l2_projection.py spectrum=1 seed=icosa nframes=8 nbins=8`; `… l2_projection.py seed=icosa nframes=4` (existing regression)

- **Phase 6 Item ④ Implementation (2026-09-14) // Finite Window Operator + Local Automatic Assertion: Fulfill N1-C + L0-A10 Upgrade to Automatic Assertion**:

  **Block A (L2 finite window operator; `src/l2/l2_projection.py` §6.4 ⑥ pure increment, read-only `net`)**: `window_ball` (hop-count ball window) / `window_prefix` (scale window) / `_subnet` (order-preserving truncation) / `window_readouts` / `window_stability` / `radial_window`. Collaborative derivation (`qwen2.5-coder:14b`, specification `l2_window_spec1.txt` ← draft `l2_window_out1.py`, 6215 characters / eval_count=1735): audit judgment **skeleton correct** (5 functions complete, S steps no missing, empty input guards complete), need to fix **1 hard bug**—draft `window_stability` references undefined `gap_sigma_last` (assigned only in else branch, same-family failure mode #1). Upon implementation, **another 2 criterion design flaws** were identified by testing (see below), which is the core evidence for this item.

  **Two criterion design flaws (exposed by testing, fixed)**:
  1. **Judgment always `undefined`**: The initial version required `σ_window` + `anchor_density` + `slope_σ` three readings to **simultaneously** meet criteria. Testing showed: `slope_σ` was **multi-level `--`** on the window (degenerate guard — t=0 full window, t=1 `V=8/32` all absent); `anchor_density` was **always 1.0** on the ball window (equal-degree subgraph grew anchors due to **boundary** — `anchors` only recognized local extrema) ⇒ three readings meeting criteria **impossible**. **Fix**: judgment **only checks the last-level relative residual** `rel_last = |σ_last−σ_prev|/σ_last ≤ tol_rel(0.05)` of `σ_window`; `slope_σ` / `anchor_density` / `frac_boundary` are downgraded to **supporting evidence** (CLI footnote fixed).
  2. **False stable**: When `V_window ≥ V`, `window_prefix` truncates to full graph ⇒ multi-level readings **completely identical** ⇒ false zero residual. Testing showed t=0, `sizes=(8,16,24,32)` were misjudged as `stable` (`rel_last=0.0`). **Fix**: deduplicate based on **actual** `v_window` (`sub.V() == rows[-1]["v_window"] → skip`), add `truncated` field and `n_truncated` count, and supplement fixture H lock.

  **Test readings (`seed=icosa cap=12 dmin=3 dense/any sizes=8,16,24,32 nbins=8 relaxed 60 rounds`)**:

  | t | V | E | Window σ sequence (order-preserving deduplicated) | rel_last | Trend | Judgment |
  |---|---|---|---|---|---|---|
  | 0 | 12 | 30 | 0.5333 / 0.4000 | 0.3333 | converging | unstable |
  | 1 | 32 | 90 | 0.5000 / 0.4103 / 0.3750 / 0.3556 | 0.0547 | converging | unstable |
  | 2 | 36 | 72 | 0.8889 / 0.6154 / 0.6154 / 0.5424 | 0.1346 | non-monotone | unstable |
  | 3 | 66 | 192 | 0.6154 / 0.4571 / 0.4068 / 0.4051 | **0.0042** | converging | **stable** |

  **Verdict**:
  1. **Window stability is "conditional" and accurately recorded**—when `V_window ≪ V` (t=3: max window 32 vs `V=66`, `rel_last=0.42%`) it holds; **when V and window are comparable, boundary effects dominate** (t=0/1/2 all `unstable`, `frac_boundary` as high as 0.75–1.0; t=2 even non-monotonic). ⇒ N1-C "finite window selected by projection layer" **has been realized as explicit operator**, but **cannot unconditionally claim "window readings stable"**—judgment depends on the premise that "window significantly smaller than full graph," which is now explicitly stated in docstring and CLI footnote.
  2. **Legacy item (i) ruling**: `fit_slice=(2,6)` **reserved only for `synthetic_calibration`** (synthetic injection calibration needs fixed sub-interval to suppress discretization edge effects at right endpoint); **fitting window on trajectory** now uses newly added `radial_window` (`r>0 ∧ box sample count ≥ min_bin_pts`, default 3) **automatic aperture**—it removes boxes with insufficient samples from readings, consistent with ruling 3 of item ③ "insufficient samples and no structure must be read separately."
  3. **`anchor_density` is not usable on window** (equal-degree subgraph grows anchors due to boundary ⇒ ball window r=1 at 1.0, full window at 0.0)—this test **corrects the reading in §6.4 ②**: `anchors` are **global** quantities; when truncated to subgraphs, they are pseudo-"anchors" due to boundary; window readings must not mix global proxy quantities.

  **Block B (L0-A10 upgraded to automatic assertion; new `src/l0/l0_locality.py`, non-reused item so directly implemented)**: Replace "code review + pointer analysis" with **readings computed by the program**—for each index artifact in `l0_gpu.py`, measure the **graph distance** from "index base entity → accessed entity" and compare it with the registered boundary. **Hop-count aperture**: `hop(A) = max_i dist(base(i), ent(A[i]))`, where `dist(x,S) = max_{s∈S} dist(x,s)` (**take the farthest node touched by the entity** = strongest local statement; if it holds, "nearest node" aperture must also hold), cross-component recorded as `inf`. Measured per **single access** (not cumulative traversal): each **step** of pointer jump still needs to be within boundary.

  | Artifact | Base | Registered Boundary | Max 480 Test | Notes |
  |---|---|---|---|---|
  | `rot_idx` (dart→target node) | owner | 1 | **1** | CSR neighbor table, target is first-order neighbor |
  | `owner_of_dart` (dart→owner) | owner | 0 | **0** | Self-referential (searchsorted locates owner segment) |
  | `reverse_dart` (dart→rev endpoint) | owner | 1 | **1** | `rev[d]=(v,u)`, farthest endpoint v shares edge with owner |
  | `dart_next_index` (dart→next endpoint) | owner | **2** | **2** | ← **"second-order neighbor" source**; triangular face takes 1, non-triangular face up to 2 |
  | `face_vertices` (single step in face ring) | previous vertex | 2 | 1 | Adjacent position on ring shares **coplanar edge** ⇒ strict boundary = 1, registered 2 as conservative upper bound |
  | `face_rep` (ring minimal dart reduction) | face first vertex | **exempt** | 1 | Intermediate steps (pointer jump / `unique` / `argsort`) are **global communication** ⇒ hop count can be arbitrarily large |

  **Verdict:**
  1. **Automatic assertion holds:** `seed ∈ {icosa, tetra, patch} × cap ∈ {12,16} × dmin ∈ {2,3,4,5} × vminus ∈ {dangling, dense} × vplus ∈ {any, gap} × t ∈ 0..4` (**480 audits**) **zero failures**, and the measured maximum of each non-exempt artifact **exactly matches its registered bound** (1/0/1/2 —— the bound is not empty talk). This is the implementation of the "domain-pressure test" discipline (refer to §9 assertion table L0-A11 for the lesson on narrowing).
  2. **Bounds are actually reached:** The second-order bound of `dart_next` is only achieved when **faces are non-triangular** (holes / long faces) — measured at t=2 (V=36 E=72) reaching `2`, and t=1/3 as `1` ⇒ "≤ second-order neighbor" is a **tight bound**, not empty talk (hence, cannot be written as "always 2").
  3. **Exemptions are truthfully registered, not falsely passed:** The measured value of `face_rep` on these small graphs ≤ 1 is due to **small faces** (cycle length 3), **not** indicating the artifact's local-global nature — its global nature lies in its **intermediate steps**; this audit only covers final artifacts. Hence, it is explicitly marked as `exempt`, and not included in the verdict.
  4. **Falsifiability (criterion is not always true):** `reverse_control()` constructs **cross-component** artifacts (two non-overlapping icosa, taking nodes from another component) ⇒ distance is unreachable ⇒ measured `inf`, and the assertion is **rejected**. If it were to pass, it would indicate that the measurement lacks discriminative power.
  5. **Collaborative derivation cross-check (`qwen2.5-coder:14b`, `l0_locality_ask1.py`, 208 tok):** Let the local model **derive independently based solely on artifact definitions** each hop bound, resulting in `rot_idx=1 / owner=0 / reverse_dart=1 / dart_next=1 / face_vertices=inf / face_rep=log n`. Three correct — including **independently giving `reverse_dart=1`**, which supports the judgment above to raise the registered bound from 0 to 1; two errors occur in **the same two traps**: ① `dart_next=1` only holds when **all faces are triangular**, missing non-triangular faces (holes/long faces) that can make w and u be 2 apart (this audit's 480 scans measured 2); ② `face_vertices=inf` mistakenly assumes "expanding the entire ring" as a **single traversal**, whereas this audit defines **single access** (adjacent positions on co-planar edges ⇒ 1). ⇒ The value of the local model lies in **independent rechecking of criteria**, not replacing measurement: two misjudgments precisely confirm two criteria choices in this module ("taking the farthest reachable node" and "counting by single access").

  **Verification (three-tiered progression, all passed):** ① Fixture — `l2_projection.py selftest=1` all green (new window fixtures A–H: A spherical window `r=0/1/2 → (V,E)=(1,0)/(6,10)/(11,25)`; B prefix window `V=6` and spherical window r=1 **same set**, `V_window ≤ 0 / ≥ V` boundary; C full window reading `(12,30)`, σ=0.4, `deg_mean`=5, anchor 0; D spherical window r=1 → σ_window=0.6, `n_boundary`=5, `frac_boundary`=5/6, **`anchor_density`=1.0**; E `radial_window` only retains `r>0 ∧ n≥min_bin_pts`; F `window_stability(sizes=(6,12))` → `unstable` / `rel_last=0.5`; G t=1 residual decreases (`converging`) but final level still exceeds tolerance; H **truncated deduplication** → 2 rows, `n_truncated=1`, `rel_last=1/3`); ② Properties — `l0_locality.py selftest=1` all green (① single-component 6 artifacts all passed ② reverse control rejected ③ `dart_next` frame-by-frame `[1,2,1]`, bound 2 reached, `reverse_dart`/`face_vertices` always 1) + 480 pressure scans zero failures; ③ End-to-end — windows/audits are **pure incremental**, L2 existing readings (global coordinates/holonomy/spectrum) **bit-for-bit identical**, L0 not rewritten.
  Reproduce: `…\python3.13.2\python.exe l2_projection.py selftest=1`; `… l2_projection.py window=1 seed=icosa nframes=3 sizes=8,16,24,32`; under `src/l0` `… l0_locality.py selftest=1` and `… l0_locality.py seed=icosa nframes=3`

- **L0 Basic Concept Formalization (2026-09-15) // Same-scale equivalence = rooted r-neighborhood subgraph isomorphism (computable in L0-B9):** Added `src/l0/l0_equivalence.py`, converting `L0_Closure_and_Isolation_chi2.md` §5 A3a / §6 L0-B9 from concept to computable quantity. ① `rooted_wl_certificate` — pure Python rooted 1-WL certificate (initial color = (in-sphere degree, in-sphere triangle count); root node degree takes original graph degree to carry r=0 information; compressed labels are assigned by tuple dictionary order, independent of node ID order; final signature retains both WL label and initial color to prevent single-point collapse). ② `vf2_equivalent` / `_vf2_labels` — networkx exact isomorphism (union-find integrated class), node matching includes `is_root` + in-sphere degree + root original graph degree, as authoritative cross-validation. ③ `equivalence_classes(net, r)` returns VF2 authoritative partition + certificate partition + `consistent` (reliability: VF2 same ⇒ certificate same) + `collisions` (1-WL collisions, not bugs). ④ `stable_r(net)` — minimal r where partitioning stops refining = graph's intrinsic resolution, answering "who sets r".

  **Measured (9 graphs × r=0..3, `selftest=1` all green):** ① Certificate reliability zero broken (no sound_breaks), prism⊔K₃,₃ r=2/3 has 36 1-WL collisions (expected, refined by VF2). ② A3a standard demo: prism⊔K₃,₃ (12 nodes all deg=3) r=0 → 1 class, r=1 → 2 classes (prism nodes in sphere contain triangles / K₃,₃ nodes in sphere bipartite no triangles). ③ Regular icosahedron → any r always 1 class (not useful for demonstration). ④ Double cone bi_5 r=0 splits into 2 classes (poles deg=5 / equator deg=4). ⑤ stable_r: icosa=0 (distance regular), prism⊔K₃,₃=1, 3×3 grid=0. ⑥ Reverse control: prism and K₃,₃ root's r=1 certificate differs (certificate not always true).

  **A3 convention synchronization (`l0_kissing.py`):** Pressure test 7 cases (tetra/octa/icosa/bi_5/bi_12/torus3x3/patch_R2) confirm `is_uniform(net) ⟺ equivalence_classes(net,0)["n_classes"]==1` — `l0_kissing`'s A3 (same shell ⇒ same degree) is exactly r=0 equivalence (necessary but not sufficient). RULES and docstring have been annotated and cross-referenced `l0_equivalence.py`.

  **Documentation registration:** `L0_Closure_and_Isolation_chi2.md` §6 L0-B9 line added with computable quantity source; §8 item 2 (who sets r) answered (`stable_r`). Candidate assertions pending ruling will be incorporated into this document §9 (numbering continues as L0-B*).

Reproduce: under `src/l0`, `… l0_equivalence.py selftest=1`; `… l0_equivalence.py seed=prism_k33 r=1`

- **R1 Probe: Can K₆ constrain degree? → No (2026-09-15) // L0-B14 reinforcement impossible proof**: Add `src/l0/probe_r1.py`, the system checks three paths. (a) **Clustering structure**: 9 graphs (tetra/octa/icosa/bi_4..bi_20) have a maximum clique of K₄—K₅/K₆ are excluded by Kuratowski planarity, R1 (forbidding K₆) is always true in triangulation and adds no constraint. (b) **Euler + uniformity**: d-regular spherical triangulation `d·V=6V−12` ⇒ `V=12/(6-d)` ⇒ `(6-d)|12`, only solvable for `d∈{3,4,5}` (V=4,6,12); `d=6` ⇒ V=∞ (unsolvable); `d≥7` ⇒ V negative (absurd). This path is **purely combinatorial** (no π/angles/radii), but A3 (uniformity) is a heavy assumption, not derived from Euler. (c) **Local rule search**: Compare deg≤5 (114 vertices) vs deg≥6 (14 vertices) on four invariants—κ=6−deg, number of triangles inside the sphere, link type, |N₂|. κ and triangle count are just definitions of deg; link type is always cycle (two overlapping sets); |N₂| overlaps in both sets. **The double cone apex (deg=6) and icosahedron vertex (deg=5) have isomorphic 1-neighborhood structures** (both cycle links), locally indistinguishable.

  **Verdict**: R1 has no teeth. **Reinforced the impossibility judgment of `l0_count.no_intrinsic_capacity`**—not only witness families exist, but local rule methods systematically fail to distinguish. Only path: A3 (uniformity, global assumption) + Euler identity ⇒ `d∈{3,4,5}`; without A3, Euler only constrains the mean `⟨deg⟩<6`, not individuals. §8 item 3 already answers: if a single-point upper bound is needed, it can only be via geometric exclusion (A2/κ²) or elevate A3 to an axiom.

  **Documentation registration**: `L0_Closure_and_Isolation_chi2.md` §6 adds L0-B14 (candidate); §8 item 3 updated; §7 adds probe index `probe_r1.py`; one-sentence summary added for R1 conclusion.

  Reproduce: under `src/l0` `… probe_r1.py`

- **Card One Closure: Projection Entry Verdict (2026-09-15)**: Officially downgrade `l0_project.py` to **geometric control device**, establishing the purely combinatorial projection pipeline as the sole entry.

  **Downgrade Verdict**: `l0_project.py`'s `solve()` / `minimal_dim()` use scipy L-BFGS-B for force-directed layout (coordinates + distances + energy minimization), which presumes a coordinate space, conflicting with SPUM axiom "relations are the only primitive, space is the projection of relations." A formal downgrade mark has been added to the file header, explicitly stating "not to be used as an entry in the projection pipeline"; `solve()` / `readout()` are retained for geometric control (pure combinatorial readings and Euclidean embeddings can be compared side by side).

  **Import isolation**: Line 49 of `l0_topology.py` `from l0_project import build_regular, build_rand23` has been cut—two edge table generators have been copied into `l0_topology.py` itself, no longer touching the force-directed layout. `l0_project.py`'s `solve()` / `minimal_dim()` / `readout()` remain untouched.

  **Purely combinatorial projection pipeline (only entry)**:
  ```
  combinatorial_proto.py (RotNet / rot rotation system)
      │  Face is a first-class citizen: tri_is_face / cone_tri_face / flip_edge
      │  Triangular closed loops automatically generate faces (2-simplices); face = rigidity, rigidity = shape
      ▼
  l0_topology.py (purely combinatorial projection entry)
      │  combinatorics(faces=...)  Faces are declared externally (polyadic mode), not inferred from 3-cliques
      │  growth_dim / spectral     Dimension = spectral reading (d_g / λ1 / d_s), not an input parameter
      ▼
  l1_projection.py (N4-B local angular capacity projection)
      │  ring_sphere / sigma_of / angle_defect
      │  No global embedding, no iteration, no coordinates throughout
      ▼
  l0_equivalence.py (r-neighborhood equivalence)
         equivalence_classes / stable_r
  ```

  **Verification (all passed)**: ① `l0_topology.py selftest=1` all green (three graphs closed spherical triangulation χ=2, no coordinate readings); ② `l0_topology.py demo=polyadic` all green (K4 skeleton fixed, different face declarations → χ from −2 to +2, polyadic mode); ③ `l0_project.py selftest=1` still runs (geometric control function not broken); ④ `l0_equivalence.py selftest=1` all green; ⑤ `l1_projection.py` import chain `→ combinatorial_proto` bypasses `l0_project` (KAPPA=6 / R_CRIT=12 / ring_sphere normal).

  Reproduce: `… l0_topology.py selftest=1`; `… l0_topology.py demo=polyadic`; `… l0_project.py selftest=1`

- **Card Two Formal Verdict: Purely combinatorial layer cannot endogenously generate a single-point degree upper bound (2026-09-15)**: The user-selected path "purely combinatorial endogenous, trying to overturn the impossibility judgment" has been successfully traversed and **confirmed impossible**.

  **Code-level marking**: ① `l0_count.py` BOUNDARY entry "relation capacity (degree upper bound)" adds full text of L0-B14 reinforcement proof (r-neighborhood local invariants systematically fail to distinguish deg≤5 from deg≥6); ② `l0_count.py` `no_intrinsic_capacity` docstring expands from three to four paragraphs, with (d) paragraph registering the reinforcement proof (double cone apex and icosahedron vertex have isomorphic 1-neighborhood structures, locally indistinguishable); ③ `l0_kissing.py` A3 docstring marks "heavy assumption" status + cross-reference to L0-B14 verdict.

  **Four-part proof** (`no_intrinsic_capacity` (a)–(d)): (a) Witness family double cone bi_n for any n≥4 legally contains deg=n; (b) Pressure test n=4..64 none blocked; (c) Σ(6−deg)=6χ is a global identity without single-point upper bound; (d) L0-B14 reinforcement—four r-neighborhood local invariants (κ, triangle count, link type, |N₂|) systematically fail to distinguish deg≤5 from deg≥6.

  **R1 (forbidding K₆) has no teeth**: K₅/K₆ are excluded by Kuratowski planarity, maximum clique is always K₄, R1 is always true and adds no constraint. **Only combinatorial path** (non-local rule): A3 (uniformity, global assumption) + Euler identity `d·V=6V−12` ⇒ `(6−d)|12` ⇒ `d∈{3,4,5}`. Cost: A3 must be elevated from assumption to axiom.

  **Verification**: `l0_count.py selftest=1` all green (BOUNDARY prints with L0-B14 reinforcement text); `l0_kissing.py selftest=1` all green.

  Reproduce: `… l0_count.py selftest=1`; `… l0_kissing.py selftest=1`; `… probe_r1.py`

### 7.1 Drive Criterion Probe Readings (Frozen on 2026-09-13)

Three probes = degree distribution (maxdeg / gini) | closure shell (reading of Σ(6−deg) + count of 5-regular 12-node induced subgraphs) | periodicity (short cycles in state keys).
Command form: `python l0_core.py seed=… cap=… dmin=… nframes=… drive=… pairing=… probe=1`

| # | Configuration | V trajectory | Readings |
|---|--------------|-------------|----------|
| A | icosa cap=64 slack none | 12→32→92→272→680→1328→3180 | S constant 12, shell12 constant 0; gini 0→0.29→0.39→0.42→0.41→0.40 (**stagnation**); maxdeg→64 (hit cap). No cycles. Stagnation reason: 12 seed cells filled by cap (deg=64 are exactly id 0..11). |
| B | icosa cap=12 slack none | 12→32→36→38→40→42→44 | Frozen at V=44; period=1, pre-period=6 ([LOOP] from t=7). |
| C | icosa cap=12 saturate none | 12→32→36 | Frozen at V=36; period=1, pre-period=2 ([LOOP] from t=3). |
| D | icosa cap=12 saturate conserved | 12→15→16→17 | Anchor state V=17 E=30 S=42 d=15; period=1, pre-period=4. |
| E | icosa cap=12 slack conserved | 12→15→16→17→18 | Anchor state V=18 E=30 S=48 d=18; period=1, pre-period=5. |
| F | tetra cap=12 saturate conserved | 4→4 | N=0 (each node deg=3=dmin, no removable edges) ⇒ fully frozen. |
| G | icosa dmin=4 | — | Fixed point, period=1, pre-period=0 (same as "unactivated item": coning and pruning are exact inverses). |
| H | icosa cap=64 saturate conserved | Frame-by-frame identical to D | cap becomes ineffective under balancing. |

Three hard readings:

1. **`pairing=conserved` net ΔE = 0 is precisely satisfied frame by frame** (D/E/H each frame `netΔE = 0`), and creation is indeed truncated by budget: icosa first frame creates 20 → 3 (ΔE 60 → 9), compared to `none` first frame creates 20 / removes 0 edges.
2. **No configuration shows "monotonic concentration"**: gini in D/E **decreases in reverse** (0 → 0.1311 → 0.1167 → 0.1059, E continues to 0.0667), maxdeg 5→6→5→4; A's gini stagnates (seed cells filled by cap). **shell12 is 0 for all t ≥ 1** — the only 5-regular 12-node induced subgraph is the seed itself at t=0.
3. **`pairing=conserved` structurally cannot produce a closed triangulation shell**: `break_edge` removes one edge ⇒ two triangular faces merge into a quadrilateral ⇒ each frame's true face candidates are annihilated by themselves. Hence N self-limited decreasing (11→5→3→2), creation decreasing (3→1→1→0), t=4 falls into fixed point; meanwhile S monotonically increases (12→30→36→42), d monotonically increases (0→9→12→15), i.e., **moving further away from Σ(6−deg)=12**.

**`_plan_breaks` guard (finalized on 2026-09-13)**: Edge removal planning requires that **both ends after removal still ≥ dmin** (guard `vdeg > dmin`). Otherwise N is pushed to "all removable edges" = O(|E|) — on icosa, N = 23/30, annihilation end crushes creation end, prune is first triggered by balancing and chain clears the network (12→17→11→4→1→0, χ temporarily = 4). With guard: icosa first frame N 23→11 (each node at most loses 2 degrees, upper bound 12×2/2 = 12), cumulative V− = 0, χ always 2, `cap_ok` always true. Hence N is a **structural upper limit**, without exogenous proportion parameters. Corresponding selftest item 8 assertion: first frame edge removal count == creation ΔE, creation count < `none`, after three frames V > 0 (prevent balancing self-clearing), determinism.

> To be continued: Edge **selection rule** is currently placeholder ("maximum sum of degrees at both ends" priority, tie-breaking by lex order), not yet changed to SPUM semantics "target redundant / least saturated edges"; A criterion (closure shell creation suppression) not yet integrated. Both are pending rulings after §7.1 readings freeze.

### 7.2 Bare Dynamics Reading (Frozen on 2026-09-13)

**Experimental Design (Zero Prohibited Items):** Remove `cap`, priority, `drive`, and `pairing` entirely, leaving only two rules — 
① At frame start, **cone all true faces** (the sole V⁺ channel); ② At frame end, **prune deg < dmin** (the sole V⁻ channel, one deletion per frame). Not going through `L0Core`, so unrelated to §3 negotiation engine. Command: `python l0_core.py seed=icosa nframes=9 bare=1`
(Implementation in `src/l0/l0_core.py` §7.2 block `bare_step` / `bare_probe` / `_print_bare`; readings are obtained via line-by-line JS mirror of RotNet+bare logic, no Python on this machine.)

`seed=icosa, dmin=3` long-range (t = 0…9):

| t | V | E | F | maxdeg | maxdeg× | α | Curvature Dust dust | Hub Debt debt | S | Frozen Core n/m/inner_deg | Configuration Types | Max Class Count |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 12 | 30 | 20 | 5 | — | — | 12 | 0 | 12 | 12/30/[5] | 1 | 12 |
| 1 | 32 | 90 | 60 | 10 | 2.000 | 0.7067 | 60 | 48 | 12 | 12/30/[5] | 2 | 20 |
| 2 | 92 | 270 | 180 | 20 | 2.000 | 0.6806 | 180 | 168 | 12 | 12/30/[5] | 3 | 60 |
| 3 | 272 | 810 | 540 | 40 | 2.000 | 0.6663 | 540 | 528 | 12 | 12/30/[5] | 5 | 120 |
| 4 | 812 | 2430 | 1620 | 80 | 2.000 | 0.6579 | 1620 | 1608 | 12 | 12/30/[5] | 9 | 240 |
| 5 | 2432 | 7290 | 4860 | 160 | 2.000 | 0.6525 | 4860 | 4848 | 12 | 12/30/[5] | 10 | 720 |
| 6 | 7292 | 21870 | 14580 | 320 | 2.000 | 0.6488 | 14580 | 14568 | 12 | 12/30/[5] | 11 | 2160 |
| 7 | 21872 | 65610 | 43740 | 640 | 2.000 | 0.6462 | 43740 | 43728 | 12 | 12/30/[5] | 12 | 6480 |
| 8 | 65612 | 196830 | 131220 | 1280 | 2.000 | 0.6443 | 131220 | 131208 | 12 | 12/30/[5] | 13 | 19440 |
| 9 | 196832 | 590490 | 393660 | 2560 | 2.000 | 0.6428 | 393660 | 393648 | 12 | 12/30/[5] | 14 | 58320 |

Degree histogram (bucket lower limit, `seed=icosa`): Each frame overall **×3**, no redistribution at all.

| t | 3-3 | 6-9 | 10-19 | 20-39 | 40+ | <3 |
|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 20 | 0 | 12 | 0 | 0 | 0 |
| 2 | 60 | 20 | 0 | 12 | 0 | 0 |
| 3 | 180 | 60 | 20 | 0 | 12 | 0 |
| 9 | 131220 | 43740 | 14580 | 4860 | 2432 | 0 |

`seed=tetra, dmin=3` isomorphic: maxdeg 3→6→12→…→1536 (×2), V 4→8→20→56→…→39368, S always 12, frozen core always 4/6/[3]. `dmin=4` and `dmin=5` are both **identity mappings**: each frame `coned=20, pruned=20`, frame-by-frame state identical to seed (V/E/F/maxdeg always 12/30/20/5).

**Three Verdicts:**

1. **Maxdeg growth curve: purely exponential (exactly ×2 per frame), not self-saturated ⇒ upper bound is not emergent.** This hypothesis is falsified. It can be directly read from operational algebra: coning a true triangular face adds +1 degree to each of the three endpoints, and node degrees only increase, so `deg_{t+1} = 2·deg_t`; thus `maxdeg_t = 5·2^t`, `V_t = 10·3^t + 2`, `α = ln(maxdeg/5)/ln(V/12) → ln2/ln3 = 0.63093` (measured 0.6428 and monotonically approaching). **`cap` is a load-bearing structure, not a memory parameter** — removing it eliminates saturation, selection, and "traversal." The capacity constraints in §3 must remain as theoretical objects.
2. **Degree histogram: curvature is not "moved" anywhere, it's replicated proportionally per frame.** `S ≡ 12` always holds (maintaining closed triangulation throughout), `dust − debt ≡ 12`; dust and debt each grow ×3 (dust 12→60→…→393660; debt 0→48→…→393648), the two halves nearly exactly offset. All positive curvature (dust) is concentrated in the **newest shell** (131220 nodes with deg-3, each contributing 3, exactly the full dust amount); all negative curvature (debt) is concentrated in the **oldest hubs** (2432 nodes with deg≥40: 12 original nodes with deg=2560, others graded at 1280/640/320/160/96/48). Histogram overall ×3 per frame (131220/43740 = 14580/4860 = 3). **This is the signature of self-similar replication (F→3F pure combinatorial inflation), not the signature of "traversal"** — bare dynamics do not move curvature.
3. **Repeated local configurations: only "frozen core" appears, no new emergent crystallites.** The 12-point seed maintains internal degree always 5 and internal edges always 30 (`core_degs=[5]`; tetra always 4/6/[3]) — that 5-regular icosahedral cluster is **permanently frozen**, coning only adds shells outside it, never altering its internal connections. Configuration types monotonically +1 (1→2→3→5→9→10→11→12→13→14, t≥5 each frame adds exactly 1 type), **not converging to a finite motif set**, so there is no "repeated few local configurations." Starting from t≥3, the maximum class is always `10-19x1+40x1+6-9x1`, count 60·3^(t−3), which is a **self-similar scaling signature**, not a stable crystallite.

**Structural conclusions derived (more important than readings themselves):**

- **The two channels cannot be active simultaneously.** `dmin=3` → pruning never triggers (pruned always 0), only V⁺ → pure inflation; `dmin≥4` → coning and pruning are **exact inverses** (coned=pruned=20, returning to seed frame-by-frame) → identity mapping. **There is no parameter region where both channels are active without canceling each other.** This is the structural root of §7.1 "L0-A9 not activated."
- With only the pair of inverse operations — coning true faces + pruning — L0 **cannot have any non-trivial net evolution**. To leave the self-similar inflation trajectory, a **operation must be introduced that breaks the coning/pruning inverse symmetry** — candidates: asymmetric V⁻ (`break_edge`/reconnection, making ΣΔE ≠ 0); or let creation and annihilation not share the same criterion (i.e., the pending A criterion in §7.1: creation suppression within closed shells). **(§7.4 has verified: asymmetric V⁻ can leave this trajectory; creation suppression leads to freezing instead.)**

### 7.3 Reconnection Dynamics Reading: Potential Function Φ Probe (Frozen on 2026-09-13)

**Why the action set can only be "edge flips".** §3.2 defines reconnection as "delete one and add one, keeping total edge count conserved". Under the condition of **fixed V, preserving χ, and maintaining triangulation at all times**, the actions that satisfy this definition are **uniquely edge flips (2-2 Pachner moves)**: for an edge {i,j}, delete it and add the opposite edge {k,l} (where k = `succ_j(i)`, l = `succ_i(j)`). Implementation is in `combinatorial_proto.py`'s `flip_delta` / `flip_edge`; it is **an involution** (flipping twice restores bit-for-bit), and ΔV = ΔE = ΔF = Δχ = 0.

- Potential: Ψ = Σ(6−deg)², flip gradient `ΔΨ = 2(d_k+d_l−d_i−d_j) + 4`. (Note: `Σ(6−deg) ≡ 12` is an identity, not a constraint; Ψ ≥ ⌈144/V⌉ is the Cauchy–Schwarz lower bound.)
- Flipping **will not** reduce any vertex to deg < 3: when deg(i) = 3, its link is a triangle ⇒ three neighbors are pairwise adjacent ⇒ k,l must be adjacent ⇒ automatically rejected. Hence no additional guards are needed, and no dangling ends are produced.
- Fixed V ⇒ E = 3V−6, F = 2V−4 locked throughout ⇒ **finite state space, Φ has upper/lower bounds**. Boundedness is provided by "upper layer given V", not hidden in L0.
- Command: `python l0_core.py recon=1 V=13 mode=phi1 nframes=40 perturb=3`
  (`mode = none | phi1 | phi2`; `perturb` = number of Φ₂ steps taken before starting, to push the start point away from Φ₁ optimal; use `seed` when `V=0`.)
- Readings are obtained via **line-by-line JS mirror** through RotNet + §7.3 logic (no Python on host). Invariant assertions `χ = 2 ∧ E = 3V−6 ∧ F = 2V−4 ∧ dust − debt = 12` are carried throughout, **none violated** ⇒ all readings are on legal triangulations.

#### Case ①: `mode=none` (no direction chosen, all flips) ⇒ measured period 2

`V=12`, Ψ sequence: 12 → 52 → 40 → 52 → 30 → 52 → 28 → 34 → 34 → 28 → 38 → 28 → 38 → 28 → 38 → 28. State keys at t=10 and t=12 repeat ⇒ **period = 2** (Ψ alternates between 28 / 38 from t≈9).

> **Verdict**: Action set is self-inverse (involution) ⇒ reachable relation symmetric ⇒ orbits on finite state space must be periodic. **"No direction chosen means no arrows" is confirmed by measurement**, consistent with the argument that "irreversibility ⇔ arrow".

#### Case ②: `mode=phi1` strict descent (only take ΔΨ < 0) ⇒ expected most uniform triangulation

Start always takes 3 steps of Φ₂ (`perturb=3`) to push away from optimal, then runs Φ₁:

| V | Start Ψ | Final Ψ | Converged Frame | Icosahedral 12-shell | Final Histogram |
|---|---|---|---|---|---|
| 12 | 100 | **16** | 4 | No | `[4:2 5:8 6:2]` |
| 13 | 126 | 14 | 4 | No | `[4:1 5:10 6:2]` |
| 14 | 156 | 16 | 3 | No | `[4:2 5:8 6:4]` |
| 15 | 190 | 16 | 5 | No | `[4:2 5:8 6:5]` |
| 16 | 228 | **12** | 4 | No | `[5:12 6:4]` |
| 17 | 270 | 14 | 4 | No | `[4:1 5:10 6:6]` |
| 18 | 316 | **12** | 4 | No | `[5:12 6:6]` |
| 20 | 420 | 14 | 4 | No | `[4:1 5:10 6:9]` |
| 24 | 676 | 18 | 4 | No | `[4:2 5:9 6:12 7:1]` |
| 28 | 996 | 22 | 4 | No | `[4:1 5:14 6:9 7:4]` |
| 34 | 1596 | 24 | 7 | No | `[4:2 5:12 6:16 7:4]` |
| 42 | 2620 | 16 | 7 | No | `[5:14 6:26 7:2]` |

**V=12 attractor depends on start point** (same V, different perturb): `perturb=0` → Ψ=12 (= regular icosahedron `[5:12]`); `perturb=1` → Ψ=20 `[4:4 5:4 6:4]`; `perturb≥2` → Ψ=16 `[4:2 5:8 6:2]`.

> **Verdict (prediction falsified)**: **Strict Φ₁ does not return to regular icosahedron for V=12**. Ψ=16 state is a true local minimum: all 30 edges can be flipped, among which `ΔΨ<0` has **0**, and the smallest `ΔΨ = 0`. Contrast: regular icosahedron itself has minimal `ΔΨ = 4 > 0` (it is Φ₁'s global minimum, Ψ=12 achieves lower bound). **A single pass of deterministic descent in Φ₁ gets stuck in a pseudo-local minimum**—and the sticking point fluctuates between V=12…24, not reaching the Ψ=12 level.

#### Case ③: `mode=phi1n` non-strict monotonic (ΔΨ ≤ 0, neutral steps allowed) ⇒ correction

| V | Start Ψ | Final Ψ | All zero flips? | Icosahedral 12-shell | Final Histogram | Theoretical Optimal |
|---|---|---|---|---|---|---|
| 12 | 100 | **12** | Yes (stopped at frame 5) | **Yes** | `[5:12]` | 12 |
| 13 | 126 | 14 | No (platform wandering) | No | `[4:1 5:10 6:2]` | **14** |
| 14 | 156 | **12** | Yes (stopped at frame 5) | No (V>12) | `[5:12 6:2]` | 12 |
| 15 | 146 | **12** | Yes (stopped at frame 6) | No | `[5:12 6:3]` | 12 |
| 16 | 208 | **12** | No | No | `[5:12 6:4]` | 12 |
| 20 | 394 | **12** | No | No | `[5:12 6:8]` | 12 |
| 28 | 954 | **12** | No | No | `[5:12 6:16]` | 12 |
| 42 | 2550 | **12** | No | No | `[5:12 6:30]` | 12 |
| 64 | 6642 | 20 | No | No | `[5:16 6:44 7:4]` | 12 |

> **Verdict (key correction to the burdened proposition)**: Relaxing "only take steps that increase Φ" to "**only take steps that do not decrease Φ**" allows **V=12 to reach regular icosahedron**—the user's prediction is **restored under this form**. The reason is that on V=12, the Ψ=16 sticking point has `ΔΨ = 0` **neutral edges** (evidence: minimal `ΔΨ = 0`), strict descent rejects them, non-strict monotonic takes them, landing at Ψ=12.
> But **non-strict monotonic does not guarantee termination**: for V ≥ 16, the optimal Ψ=12 is reached, yet it still **wanders on the platform where Ψ remains unchanged** (table's "All zero flips?" is no). Hence a single scalar Φ is insufficient—**either accept pseudo-local minimum (strict), or accept non-termination (non-strict)**. To achieve both requires **two-tier potential**: primary potential strictly increasing to ensure termination, and on the platform where Ψ is equal, use secondary order (deterministic tie-breaker) to decide direction. Selection of secondary order is a legacy bifurcation (see below).

#### Case ④: `mode=phi2` (only ΔΨ > 0) ⇒ curvature concentration state

| V | 12 | 13 | 14 | 16 | 20 | 28 | 42 | 64 |
|---|---|---|---|---|---|---|---|---|
| Final state Ψ | 100 | 126 | 156 | 228 | 420 | 996 | 2620 | 6756 |
| maxdeg | 11 | 12 | 13 | 15 | 19 | 27 | 41 | 63 |
| Histogram | `[3:2 4:8 11:2]` | `[3:2 4:9 12:2]` | `[3:2 4:10 13:2]` | `[3:2 4:12 15:2]` | `[3:2 4:16 19:2]` | `[3:2 4:24 27:2]` | `[3:2 4:38 41:2]` | `[3:2 4:60 63:2]` |

> **Verdict**: Curvature concentration state confirmed. However, the attractor is a **universal double-hub mode** — **exactly two hubs with deg = V−1 + two deg-3 + the rest deg-4**, `maxdeg = V−1` **strictly linear with V**. **Not "a single saturated crystallite"** (a single saturated crystallite should be a single hub).

#### Precise characterization of global optimum: ideal defect spectrum + exception at V=13

- **Global optimum of Φ₁ = ideal defect spectrum**: twelve deg-5 vertices + the rest deg-6 ⇒ Ψ = 12 reaches the lower bound (V ≠ 13).
- **Annealing search verification** (for each V, starting from random perturbations, 80 restarts, minimizing Ψ):

| V | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|
| Minimum Ψ achieved | 12 | **14** | 12 | 12 | 12 | 12 | 12 | 12 |
| All degrees ∈ {5,6}? | Yes | **No** | Yes | Yes | Yes | Yes | Yes | Yes |
| Hits / 80 | 80 | **0** | 80 | 79 | 80 | 79 | 79 | 74 |

> **Verdict (parity argument in §6 does not hold)**: **V = 13 is the only exception among V ≥ 12** — it cannot achieve "all degrees ∈ {5,6}". For odd numbers like V = 15, 17, etc., **it can** (`[5:12 6:3]`, `[5:12 6:5]`), thus disproving the claim that "when V is odd, degrees outside {5,6} must appear".
> Correct statement: Triangulation with all degrees ∈ {5,6} ⟺ its dual is **a fullerene with 2V−4 vertices**. Fullerenes exist for all even n ≥ 20, **except n = 22** (complete enumeration of fullerenes has been published, not re-verified in this round) ⇒ only V = 13 (dual 22 = C₂₂) is impossible. This also explains why the optimal for V=13 is exactly Ψ=14 `[4:1 5:10 6:2]` (the next best integer solution).

#### Judgment on the icosahedral 12-shell (responding to "whether the 12 lowest-degree vertices form a clique")

Induced subgraph of the "12 lowest-degree vertices" in the final state:

| V | 12 | 16 | 20 | 42 |
|---|---|---|---|---|
| Degree sequence of the 12 lowest-degree vertices | 4,4,5⁸,6,6 | 5¹² | 4,5¹⁰,6 | 5¹² |
| Number of induced subgraph vertices / internal edges | 12 / 30 | 12 / 20 | 8 (largest component) / 12 | 5 (largest component) / 6 |
| Internal degree range | 4–6 | 3–4 | 1–4 | 1–3 |
| 5-regular (i.e., icosahedral 12-shell) | No | No | No | No |

> **Verdict (structural impossibility of the judgment condition)**: Icosahedral 12-shell = **a 12-vertex 5-regular induced subgraph** (each vertex has 5 neighbors entirely within the 12 vertices, with 30 internal edges). However, if each vertex's 5 neighbors are all within the 12-vertex set, then these 12 vertices would have **no edges connecting to the outside ⇒ the entire complex is disconnected**. Therefore, **in connected triangulations with V > 12, the icosahedral 12-shell structure cannot appear as a "defect cluster"** (this also explains why internal edges are always < 30 in practice).
> Corollary: **The dodecahedral/icosahedral 12-shell can only exist as an independent connected component (a standalone closed 12-sphere)** — this is fully consistent with SPUM §7 "12 crystallites form a closed loop... the structure inherently distinguishes inside and outside": it is originally a closed, independent structure, not a defect cluster within a large network.
> **Verdict restated**: Φ₁ passes to L1 as the **ideal defect spectrum `[5:12, 6:(V−12)]`** (V ≠ 13); the 12-shell is its only realization when V=12 and connectivity is zero. **Crystallites are not "local maxima of Φ"** (the minimum of Φ₁ is degenerate: `[5:12 6:(V−12)]` has infinitely many realizations), but rather **the unique realization of the degree signature "12 vertices with deg-5" at V=12**.

**Remaining bifurcation**: ① The secondary potential/tie-breaking rule on the platform is undetermined (deciding whether non-strict monotonicity can both reach the optimum and terminate); ② At V=64, a single scan cannot reach Ψ=12 (`[5:16 6:44 7:4]`), requiring better descent/annealing scheduling or proving that the energy barrier of Φ₁ grows with V; ③ Annealing is heuristic and cannot replace existence proofs; ④ Potentials other than Φ₁/Φ₂ have not been scanned ("which Φ to choose" remains an irreducible combinatorial input — though fewer parameters than before, purely combinatorial, no metric).

### 7.4 Asymmetric Dynamics Reading (Frozen on 2026-09-13, with Adoption Ruling)

**Motivation**: The structural conclusion in §7.2 requires introducing an operation that **breaks the inverse symmetry of cone pruning and trimming**. This implementation introduces two switches (`src/l0/l0_core.py`):

- `vminus="dangling" | "dense"` — asymmetric V⁻. `"dense"` = cut "over-dense edges", with the criterion being **both endpoints have negative curvature** (`deg(u) > κ` and `deg(v) > κ`, κ=6), prioritizing by the sum of degrees, guarding at the **axiomatic lower bound 2** (not dmin). The criterion acts on **different regions** than the creation criterion (positive angle defect) ⇒ breaks inverse symmetry; and cutting edges can push endpoints to deg 2 < dmin ⇒ trigger prune ⇒ regenerate dangling ends (L0-A9).
- `vplus="any" | "gap"` — creation suppression. `"gap"` = only retain positive angle defect candidates (`δ = Σ_{corners}(κ − deg) > 0`).

Command: `python l0_core.py seed=icosa nframes=10 cap=12 dmin=3 dyn=1 [vminus=… vplus=…]`

**Four sets of comparisons (seed=icosa cap=12 dmin=3, 10 frames) — locating the frozen culprit:**

| Configuration | Final State | V/Vprev Value Set | maxdeg | L0-A9 (Same-frame Edge Cut + Node Deletion) |
|--------------|-------------|-------------------|--------|--------------------------------------------|
| Baseline `dangling`/`any` (cap=12) | **Frozen saturation V=44** (unchanged for t≥6) | Early growth then → `1.000` | 12 (cap bound) | 0/10 |
| **`dense`/`any` (adoption)** | V: 12→…→**960**, continuous growth | `{1.125…2.667}` **non-constant** | ≤12 (cap bound) | **1/10** (node deletion at t=8, 5 nodes) |
| `dangling`/`gap` | t=1 frozen **V=32** | `{1.0, 2.667}` | 10 | 0/10 |
| `dense`/`gap` | t=3 frozen **V=62** | `{1.0, 1.938, 2.667}` | 10 | 0/10 |

> ⚠️ **Correction (2026-09-13, see anomaly X1)**: The previous version of this section recorded the baseline as "V/E ×3 per frame" — that was the reading **without cap (cap=64)** (§7.1 A / §7.2). **Under cap=12, the baseline is frozen saturation V=44, not ×3**; cap is a load-bearing structure. Also, the previous version recorded `dense`'s maxdeg as 5→48, which was a **false impression** caused by newly created points in `cone_hole` not being checked against cap (overloaded points), X1 fixed this and now maxdeg is always ≤ cap.

**Adopted configuration frame-by-frame readings (`dyn=1` default setting, after X1 fix):**

| t | V | E | F | χ | maxdeg | σ=V/E | V/Vprev | V⁺ creation | V⁻ edge cut | V⁻ node deletion | netΔE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 12 | 30 | 20 | 2 | 5 | 0.4000 | — | 20 | 0 | 0 | 60 |
| 1 | 32 | 90 | 60 | 2 | 10 | 0.3556 | 2.667 | 4 | 30 | 0 | −18 |
| 2 | 36 | 72 | 38 | 2 | 7 | 0.5000 | 1.125 | 30 | 0 | 4 | 120 |
| 3 | 66 | 192 | 128 | 2 | 12 | 0.3438 | 1.833 | 16 | 0 | 0 | 48 |
| 4 | 82 | 240 | 160 | 2 | 12 | 0.3417 | 1.242 | 16 | 24 | 0 | 24 |
| 5 | 98 | 264 | 168 | 2 | 11 | 0.3712 | 1.195 | 56 | 0 | 0 | 188 |
| 6 | 154 | 452 | 300 | 2 | 12 | 0.3407 | 1.571 | 72 | 100 | 0 | 116 |
| 7 | 226 | 568 | 344 | 2 | 10 | 0.3979 | 1.468 | 206 | 47 | 0 | 675 |
| 8 | 432 | 1243 | 813 | 2 | 12 | 0.3475 | 1.912 | 131 | 254 | 5 | 147 |
| 9 | 558 | 1390 | 834 | 2 | 11 | 0.4014 | 1.292 | 402 | 115 | 0 | 1267 |
| 10 | 960 | 2657 | 1699 | 2 | 12 | 0.3613 | 1.720 | — | — | — | — |

The degree histogram (bucket lower bounds) migrates with V, no longer ×3 overall: t=3 `[4-5:30, 6-9:20, 10-19:12]`; t=8 `[3-3:150, 4-5:101, 6-9:98, 10-19:83]`; t=10 `[3-3:252, 4-5:321, 6-9:269, 10-19:118]` — **positive curvature (deg 3) and negative curvature (deg≥10 hubs) coexist in the same frame with comparable magnitudes**, curvature is being **transported** rather than replicated proportionally.

> The `[WARN: invariant!]` at t=2/5/7/8/9/10 is an inevitable product of edge cutting: cutting edges produces boundaries ⇒ `edge_face_bad > 0`, `_invariants_ok` (closure triangulation criterion) must be false, **not a state damage** (χ remains 2 across all frames).

**Verdict:**

1. **Departure from self-similar orbit is a fact.** The adopted configuration `V/Vprev` is non-constant (1.125…2.667), σ fluctuates between 0.34–0.50, and the histogram migrates across all buckets (curvature is being **transported**) — three signatures simultaneously exclude "self-similar inflation." (`maxdeg` is not a signature: it's always capped at ≤12, see anomaly X1.)
2. **Engine = annihilation lowering degree → regeneration of positive angle defect → creation refill → new hub → re-overdensity.** This is the true non-inverse loop: not "creation being canceled by annihilation," but **annihilation itself creates the raw material for the next creation**.
3. **Freezing is caused by `vplus="gap"` rather than `vminus`.** `dangling`/`gap` freezes at t=1 (creation shuts off in saturation zone); `dense`/`gap` freezes at t=3 with V=62 (once creation stops, annihilation quickly finds no over-dense edges ⇒ both sides shut down simultaneously). **Creation suppression cuts off the above loop.**
4. **L0-A9 is activated:** under `dense`/`any`, at t=8 there's a same-frame "edge cut and node deletion" (`V⁻ edge cut=254, V⁻ node deletion=5`), selftest item 7 asserts this (baseline `dangling` has cumulative V⁻=0 across four frames in sleep; adopted `dense` has cumulative V⁻=5 across ten frames activated).

**Costs and leftovers:**

- **V unbounded growth:** pure combinatorial L0 scale-free hierarchy ⇒ σ fluctuates near 1/3 but V keeps growing (this is the "continuous evolution" cost, not self-similar: `V/Vprev` non-constant, histogram bucket migration). **Handling see below "Bifurcation N1" — autonomously excluded "L0 self-limiting," adopted "finite window projection layer."**
- **Reconnection (fixed V, preserving χ) is not in this reading:** `recon=1` route see §7.3 (edge flip involution).
- Selftest coverage: item 7 (L0-A9 baseline sleep/adopted activation), item 9 (adopted configuration off-track + determinism); default `dangling`/`any` readings in §7.1/§7.2 remain unchanged line-by-line.

**Bifurcation N1 derivation path (2026-09-13, autonomously excluded, maze trace left):**

> Cause: leftover item "V unbounded" — should L0 self-limit? According to the rule "encounter direction choice → record → experiment → exclude; if cannot exclude, bifurcate and derive," walk each branch. Evidence script: `%TEMP%\l0_sweep_n1.py`.

| Path | Proposition | Disposition | Evidence |
|------|-------------|-------------|----------|
| **N1-A** | L0 creation/annihilation dynamics contain a **bounded non-trivial** steady state | **Excluded** | Scan `cap∈{12,64} × dmin∈{2,3,4,5} × κ∈{5,6,7}` (24 sets × 15 frames) + `drive∈{slack,saturate} × pairing∈{none,conserved} × dmin∈{3,4}` (8 sets × 12 frames) |
| **N1-B** | Create a self-limiting mechanism **within L0** | **Excluded** (same source as N1-A) | "Self-limiting" only has two types: constrain creation (⇒ `conserved`/`gap` family) or constrain degree (⇒ cap). The former freezes, the latter cannot pin V, so no external constant can give a non-trivial upper bound for V |
| **N1-C** | V unbounded is a **structural property** of pure combinatorial L0; finite window selected by projection layer (L1/L2) | **Survived, adopted** | Consistent with "geometry/graphs are projections of underlying interactions"; §7.3 independently supports — reconnection dynamics "boundedness provided by **upper layer given V**, not hidden in L0" |

**Key readings from N1-A scan (seed=icosa, dense/any):**
```
dmin≤3  Any cap/κ/drive → V exponential growth: V@15/V@10 ∈ [10.5,12.0] (cap=12) / [2.4,2.5] (cap=64)
dmin≥4  Any cap/κ/drive → V always 12 (born=dead=frame candidates=300, 15 frames identity mapping) — trivial, not evolving
pairing=conserved        → seed frozen (born=0, V always 12)
vplus=gap                → t≤3 frozen (V=32 / V=62)
drive: slack 12 frames vs 16.83  vs  saturate 9.69 — only changes growth rate, not boundedness
32 configurations cap_ok all True (X1 fixed no over-cap points)
```

> **Verdict: "Bounded" in L0 ⟺ "Trivial" (frozen / identity). No bounded non-trivial steady state exists.**
> Mechanism: `Σ(6−deg)≡12` is a conserved quantity, not providing an upper bound for V; cap only pins local degrees, cannot pin V; the only knobs that can suppress creation (`conserved` / `vplus=gap`) both **simultaneously shut down annihilation** (once creation stops, over-dense edges quickly deplete ⇒ both sides shut down) ⇒ degenerate to trivial.
> **Hence adopt N1-C:** L0 only responsible for relational evolution; "observed finite universe" selected by projection layer (L1/L2) when window/granularity is chosen. V unbounded **is not** self-similar inflation (`V/Vprev` non-constant, histogram migration), but **fractal-constrained growth** — locally everywhere finite (`deg ≤ cap`), globally scale-free.

Reproduce: `python %TEMP%\l0_sweep_n1.py` (self-contained scan; `sys.path` already points to `src/l0`).

### 7.5 σ Field Gradient Emergence Reading (Frozen on 2026-09-15, Direction 4)

**Motivation**: SPUM theory predicts "gravity = net-annihilation-driven spatial flow; ∇σ gradient field is its reading" (`spum-evolution.md` §4.2). If σ non-uniformity can emerge spontaneously from uniform seed + frame evolution (without any external driver), then the topological origin of gravity is experimentally confirmed. Evidence script: `src/l0/probe_sigma_emergence.py`.

Command: `python -u probe_sigma_emergence.py seed=icosa nframes=10 cap=12 dmin=3 vminus=… vplus=any`

**σ Field Definition** (`l1_projection.py`): σ_v = 2/deg_v (local relational density, localized from global σ = |P|/|ε| and ⟨deg⟩ = 2/σ); ∇σ_v = mean_{w∈rot[v]}(σ_w) − σ_v (purely local gradient).

**Forward and Reverse Control Selftest** (`selftest=1`, all four passed):

| # | Item | Expected | Measured |
|---|------|----------|----------|
| ① | Uniform seed icosa t=0 grad_mean | Exactly = 0 (reverse control) | `0.00e+00` ✓ |
| ② | Non-uniform graph `seed_bipyramid(5)` t=0 grad_mean | > 0 (forward control) | `6.43e-02` ✓ (polar σ=0.4000 / equator σ=0.5000) |
| ③ | 5-frame sprint no crash | len(stats)=6 | `[12,32,36,66,82,98]` ✓ |
| ④ | σ monotonicity | σ(3) > σ(5) > σ(6) | `0.6667 > 0.4000 > 0.3333` ✓ |

**Two Dynamics Comparison (seed=icosa cap=12 dmin=3, 10 frames)**:

| Configuration | t=10 V | t=10 σ_ratio | t=10 \|∇σ\|_mean | t=10 \|∇σ\|_max | V Growth Pattern |
|--------------|--------|--------------|------------------|------------------|------------------|
| **`dense`/`any` (adopted)** | 960 | 4.00 | **0.211** | 0.449 | **Exponential unsaturated** (12→960, V/Vprev∈[1.125, 2.667]) |
| `dangling`/`any` (baseline) | 44 | 4.00 | 0.322 | 0.500 | **Saturated freeze at t=6** (12→44, V constant for t≥6) |

> Both configurations start with `grad_mean=0, σ_ratio=1.0` at t=0 (reverse control effective); both immediately emerge `grad_mean=0.379` (peak) at t=1.

**Frame-by-frame Reading (`dense`/`any`, adopted configuration)**:

| t | V | E | σ_glob | σ_min | σ_max | σ_ratio | \|∇σ\|_mean | \|∇σ\|_max |
|---|---|---|--------|-------|-------|---------|-------------|-----------|
| 0 | 12 | 30 | 0.4000 | 0.4000 | 0.4000 | 1.00 | **0.000000** | 0.000000 |
| 1 | 32 | 90 | 0.3556 | 0.2000 | 0.6667 | 3.33 | 0.379167 | 0.466667 |
| 2 | 36 | 72 | 0.5000 | 0.2857 | 0.6667 | 2.33 | 0.268254 | 0.342857 |
| 3 | 66 | 192 | 0.3438 | 0.1667 | 0.5000 | 3.00 | 0.164310 | 0.263889 |
| 4 | 82 | 240 | 0.3417 | 0.1667 | 0.6667 | 4.00 | 0.197561 | 0.394444 |
| 5 | 98 | 264 | 0.3712 | 0.1818 | 0.6667 | 3.67 | 0.157554 | 0.293651 |
| 6 | 154 | 452 | 0.3407 | 0.1667 | 0.6667 | 4.00 | 0.203994 | 0.416667 |
| 7 | 226 | 568 | 0.3979 | 0.2000 | 0.6667 | 3.33 | 0.224218 | 0.393651 |
| 8 | 432 | 1243 | 0.3475 | 0.1667 | 0.6667 | 4.00 | 0.236059 | 0.460317 |
| 9 | 558 | 1390 | 0.4014 | 0.1818 | 1.0000 | 5.50 | 0.210693 | 0.690476 |
| 10 | 960 | 2657 | 0.3613 | 0.1667 | 0.6667 | 4.00 | 0.210902 | 0.448653 |

**Verdict**:

1. **Spontaneous emergence of σ field gradient has been experimentally confirmed**. At t=0, uniform seed grad_mean=0 (reverse control); from t=1 onward, grad_mean=0.379 (peak) immediately emerges and remains above 0 throughout (fluctuates between 0.16-0.24 at t=2..10). Criterion `|∇σ|_mean significantly increased (>10× t=0) = True` and `σ_ratio increase > 0.01 = True` both satisfied ⇒ ★ `emerged = True`.
2. **Both dynamics produce σ gradient, but with different saturation behaviors**: `dense` mode V exponentially increases without saturation (dynamic non-equilibrium, corresponding to the Imperfection Theorem's "edge imperfection"), while `dangling` mode reaches static saturation at t=6 (V=44, gradient stabilizes at 0.322). Both end with same σ_ratio (4.0) — "σ non-uniformity" is an intrinsic product of frame evolution, independent of dynamics details; however, whether it can sustain non-equilibrium depends on vminus.
3. **Gradient emergence does not depend on σ_global drift**: σ_global fluctuates between 0.34-0.50 (around 1/3), with no monotonic drift; gradient emergence is a product of **local structure**, not global density change.
4. **Structural verification**: χ annotated as "?" (sigma_stats skipped for performance, χ not calculated); §7.4 independently confirmed χ=2 throughout.

**Connection to SPUM theory**:
- `spum-evolution.md` §4.2 predicts "net-annihilation of matter particles causes elevated σ around them, forming a gradient field decreasing outward" — this experiment confirms that σ gradient **emerges spontaneously from frame evolution without pre-assumed matter particles**, serving as the **minimal experimental evidence** for the topological origin of gravity.
- The dual-layer structure of the Imperfection Theorem (`spum-core.md` §2.3) is observable here: `dense` mode continuously produces new dangling ends (edge imperfection) ⇒ V continuously increases; `dangling` mode rapidly eliminates dangling ends ⇒ V saturates. Both maintain σ non-uniformity (central imperfection).
- σ_global fluctuates around 1/3 corresponds to `spum-core.md` §4 "high σ regions have low degree and higher temperature" — evolution naturally pushes the network toward σ ≈ 1/3 (deg ≈ 6, flat threshold), but local deviations from uniformity always occur.

**Cost and Legacy**:
- 30-frame experiments become significantly time-consuming after V reaches thousands; this freeze is at 10-frame reading (V=960). Longer evolution is an open item, not included in the verdict.
- σ field gradient reading is a **local quantity** (O(deg) per node), no global statistical analysis (e.g., ∇σ direction field spatial correlation, power-law spectrum fitting) has been done. The latter belongs to L2 projection layer (gravity power test, `l2_projection.py` `gravity_spectrum`), outside this anchor point's scope.
- This experiment **does not** claim to observe "gravity" — only confirms spontaneous emergence of σ gradient. Judgment on gravity power (r⁻² vs r⁻³) is L2 responsibility, must be done in projection coordinates.

Reproduce:
```
python -u probe_sigma_emergence.py selftest=1
python -u probe_sigma_emergence.py seed=icosa nframes=10 cap=12 dmin=3 vminus=dense vplus=any
python -u probe_sigma_emergence.py seed=icosa nframes=10 cap=12 dmin=3 vminus=dangling vplus=any
```

**Visualization Products (2026-09-15 addition)**: `src/l0/visualize_evolution.py` — generic L0 evolution multi-panel rendering (3×3 grid: V/E time series, σ field, ★ σ gradient emergence, degree+χ, σ ratio, degree distribution heatmap, t=0/N/2/N network snapshots colored by σ). Depends on matplotlib 3.11.2 (Agg) + networkx 3.6.1; CJK font priority: Microsoft YaHei.

Direction 4 verification figure: `docs/figures/evo_icosa_dense_any_cap12_dmin3_N10.png` (dense/any 10 frames, 456 KB).
```
python -u visualize_evolution.py seed=icosa nframes=10 cap=12 dmin=3 vminus=dense vplus=any dpi=120
```

- **Three scenarios minimal teaching demo (2026-09-16) // External "relation rewrite system" scheme reconciliation + `R` parameter CLI exposure**: New `src/l0/demo_minimal.py` (read-only teaching probe, no change to any evolution rules), reconciles three "minimal experiments" from external proposal to current engine; simultaneously exposes `combinatorial_proto.py` CLI with `R=` (only `seed=patch` uses it, default R=2 behavior bit-for-bit identical). Three scenarios 26 assertions all green:

  | Scenario | Engine Correspondence | Frozen Reading | Teaching Verdict |
  |----------|------------------------|----------------|------------------|
  | A Seven-node flower ring | `seed_patch(R=1)` | V=7 E=12 F=7 (6 triangular faces + 1 hexagonal hole face), χ=2, hist `{3:6, 6:1}`, Σ(6−deg)=18, path boundary points=6, opening 6/7 | **χ=2 does not imply closure** — this is a disk not a sphere; full deg6 finite closed triangulation doesn't exist (Σ=0≠12), C=6 structure necessarily has boundary |
  | B Regular Icosahedron | `seed_icosa` | 12/30/20, all deg5, Σ=12, hole=0, crystallite12=1 | 12 points are mutually shells, **no center**; each point's number of triangular faces = degree, "face saturation" is degree capacity, no second counter needed |
  | C Puncture → Regeneration | `puncture` + `mode=hole` | Delete deg5 node → 11/25/16, 5-gon hole, opening 5/16=0.3125 ≤ 1/3, Σ=16 → f1 new point **id=12** (not old id 0) fills hole back to 12/30/20, crystallite12=1 → f2 V⁺=V⁻=0 | Closure shell's only way to create an opening is V⁻ node deletion; filling the hole is **new entity occupying old combination slot** (id anchors frame identity); saturation = **locked shutdown**, not node deletion |

  **Freeze one arithmetic counterexample (external proposal experiment B's "1 center + 12 peripheral radiating")**: V=13, E=30+12=42; if still closed triangulation, 3F=2E forces F=28 ⇒ **χ=13−42+28=−1 (odd number, closed surface χ=2−2g must be even, impossible)**; and 30 shell edges each already belong to 2 shell faces, any (hub,i,j) face would make it belong to a third face, directly violating edge-face rule. ⇒ "12 crystallite closure" has 12 shells themselves, no 13th center; projection radius r(deg)∝√deg under deg12 hypothetical center would be even larger. This counterexample is direct inference from L0-A1 (edge-face rule)/L0-A7 (count chain), registered as demo fixture reading, **no new §9 assertion** (not a new dynamics claim).

  **Regression**: `combinatorial_proto.py selftest=1` all green (tetra/icosa/patch R=2 default value bit-for-bit identical); `seed=patch R=1 nframes=0` CLI reading matches scenario A.

Reproduce:
```
python -u demo_minimal.py selftest=1
python -u combinatorial_proto.py seed=patch R=1 nframes=0
```

## 8. Key Design Decisions

| Decision | Choice | Reason |
|--------|-------|-------|
| Node prior? | No, edge-induced | A0 relation defines existence |
| Random? | No | One frame output is a deterministic function of the initial state of the frame |
| Central? | Edge-centric | Negotiation is an edge event |
| Conflict resolution | Local deterministic ordering | Order-independent |
| Atomic operation | Write-only, no decision | Avoid hardware order entering L0 |
| Geometry | Projection layer | Rules are combinatorial; geometry is the result |
| Edge-face rule | Exactly two faces | A2 exclusivity + global closure |
| Frame | Global barrier | Parallel within frame, synchronized between frames |
| GPU | Execution model | Parallel structure isomorphic to L0 |
| CPU | Parse projection | Sequentiality is observer cognition |

## 9. L0 Falsifiable Assertions

| # | Assertion | Criterion |
|---|----------|-----------|
| L0-A1 | Edge-face rule | In any frame, each undirected edge is used by exactly two dart cycles |
| L0-A2 | χ conservation | After each A/B/edge-cone operation and each `remove_vertex`, `V−E+F` remains unchanged |
| L0-A3 | Closure | Closed components have `χ=2` (`genus=0`); `link` has no `disc` type |
| L0-A4 | Handshaking lemma | `Σ deg = 2|E|` |
| L0-A5 | Capacity constraint | All `deg ≤ cap`; planar boundary `E ≤ 3V−6` |
| L0-A6 | Validity of non-crossing criterion | Coning only occurs where `tri_is_face` is true ⟹ `χ` does not decrease, no separated triangles are coned |
| L0-A7 | Count-chain emergence | Numbers 12/30/20/42/50/62 are counted from the graph; code has no such constants as criteria |
| L0-A8 | Order independence | Candidates/requests can be generated in any concurrent order, and after bilateral commit + dictionary order deduplication, the final state at frame end is unique |
| L0-A9 | Imperfection theorem | After dangling pruning, the next frame must regenerate dangling edges (**§7.4 inspection**: baseline `dangling` sleep — four-frame cumulative V⁻=0; adoption `dense` activate — ten-frame cumulative V⁻=5, t=8 same-frame edge cut + point deletion; selftest item 7 assertion) |
| L0-A10 | Locality | All kernel accesses have hop distance ≤ second-order neighbors (**automatic assertion**, upgraded on 2026-09-14 —— `src/l0/l0_locality.py`: measure hop distance per artifact and compare with registration boundary + cross-component reverse control. **Hop distance definition**: `hop(A) = max_i dist(base(i), ent(A[i]))`, `dist(x,S) = max_{s∈S} dist(x,s)` (take the **farthest** node touched by an entity = strongest locality statement), cross-component recorded as `inf`). **Per-artifact boundary**: `rot_idx ≤ 1`, `owner_of_dart = 0`, `reverse_dart ≤ 1`, **`dart_next_index ≤ 2`** (← second-order source; triangular face takes 1, **non-triangular face reaches up to 2** —— boundary is actually touched), `face_vertices` single step `≤ 2` (strictly 1, registered 2 as a conservative upper bound); **Exemption**: `face_rep` (intermediate steps "pointer jumps / `unique` / `argsort`" are global communication ⇒ hop distance can be arbitrarily large, report actual values without entering judgment — actual small values are due to **small faces**, not locality). **Empirical domain**: `seed∈{icosa,tetra,patch} × cap∈{12,16} × dmin∈{2,3,4,5} × vminus∈{dangling,dense} × vplus∈{any,gap} × t∈0..4` **480 audits with 0 failures**, and maximum values of non-exempt artifacts **exactly match their registration boundaries** (1/0/1/2). ⚠️ Do not write as "always 2" — second-order boundary is only reached when non-triangular faces (holes/long faces) are present (§7 phase six item ④ inspection; `selftest=1` item ③ assertion frame-by-frame `[1,2,1]`) |
| L0-A11 | Component count does not increase (**limited to default `dmin=3`**) | Creation (face coning/hole coning) only adds edges and points within the same component; edge cutting merges two faces; pruning deletes points up to removing an entire component ⇒ connected component count **does not increase frame by frame**. **Empirical domain tightened**: true only when dangling pruning channel does not act on low-degree cut points — holds under `dmin=3` (triangulation default lower bound). (**§7 phase six item ② inspection**: `selftest` item 11 assertion; property matrix 240 configurations (`cap∈{5,12,64}` and `dmin=3`) 0 failures; R3/R4/R5 each 21/5/5 frames always hold). **Registered counterexamples (three types, all reproducible via CLI)**: ① `dmin=2 ∧ pairing=conserved` — `seed=icosa cap=12 dmin=2 pairing=conserved nframes=4 product=1` n_comp `1→2`, `seed=icosa extra=tetra cap=6 dmin=2 pairing=conserved nframes=3 product=1` `2→3` (low threshold open pruning channel + forced edge cutting can cut bridge edges/cut points); ② open patch + large threshold removal — `seed=patch cap=6 dmin=5 nframes=3 product=1` (V `19→7→2→0`, n_comp `1→1→2`, fragmented surface); ③ open patch + `dmin=2 ∧ conserved` — `seed=patch cap=8 dmin=2 pairing=conserved` (n_comp `1→2`). ⇒ **Do not make the strong assertion that it is "always constant under closed triangulation"**; only used as a theorem on the adopted trajectory with default `dmin=3` |
| L0-A12 | Spontaneous emergence of σ field gradient | Uniform seed (icosa, all deg=5) at t=0 has \|∇σ\|_mean **exactly 0**; under any combination of `vminus∈{dangling,dense} × vplus=any`, for t≥1, \|∇σ\|_mean **strictly > 0** (spontaneously emerges without external drive). (**§7.5 inspection**: `selftest` items ①② forward and reverse control; `dense`/`any` 10 frames grad_mean t=1=0.379, t=10=0.211; `dangling`/`any` 10 frames grad_mean t=1=0.379, t=10=0.322). ⚠️ **Applicable domain**: limited to `seed=icosa cap=12 dmin=3 vplus=any`; `vplus=gap` freezes at t≤3, although grad_mean>0 it is a "frozen state reading" not a "emergent state reading", not within this assertion's scope. **This assertion only confirms the emergence of σ gradient, not the power of gravity** (r⁻²/r⁻³ judgment is L2 responsibility). **Open item**: long-range evolution over 30+ frames untested (V thousands after significant time consumption); seed diversity (tetra/octa/patch) not systematically scanned |

> Appendix A's V2-U1–U7 are assertions for the geometric L0, which have been downgraded along with this implementation (see §A.11).

---

# Appendix A · Geometric Pre-L0 Implementation (v1.x, replaced by pure combinatorial L0)

> This appendix preserves the complete design and empirical records of v1.x geometric L0 as historical evidence and reference.
> **It is no longer the L0 baseline**: its orientation of "L0 retains spherical angular coordinates" has been replaced by the first part.
> However, some conclusions (exclusive zero violation, deterministic deduplication, sparse reordering feasibility) are still valuable for understanding system evolution, hence fully preserved.
> Implementation (archived): `_archive_v1/l0_legacy/host.py` + `frame_kernels.cu` + `particle_state.cuh` (testing/debugging: `smoke_test.py`, `test_l0.py`, `debug_*.py`, `diag_*.py`).

## A.1 Ontology Mapping: CUDA Thread = Local View of a Particle

SPUM's first axiom: relations are the only primitive, particles are traces of solidified relations. Mapped to computational architecture:

| SPUM Ontic | Computational Implementation |
|-----------|---------|
| Particle (κ-particle) | A local state managed by a CUDA thread |
| Edge (exclusive confirmation event) | Mutual reference entries in adjacent slots (slot of i stores j, row of j stores i) |
| Space (trace of solidified relations) | Particle's pos/radius — projection of connection topology, not a container |
| Tangency (geometric expression of an edge) | Tangent angular coordinates (θ, φ): direction of neighbors on the particle's sphere |
| Creation V⁺ | Gap occupied by a new particle: three edges established simultaneously |
| Annihilation V⁻ | Both slots of the edge are cleared simultaneously |
| Dangling (deg < 3) | Locally determinable by the particle: count its own slots |
| Discrete frame τ | L1 synchronization point: boundary of a kernel sequence round |

**Core constraint: No kernel is allowed to perform global traversal.** Each thread only reads:

- Its own state (pos, radius, degree, active)
- Its neighbors' states (via adjacent slot indices)
- Second-degree neighbors (neighbors of neighbors, for tangency discovery)

The O(N²) global distance matrix is only allowed to exist in **L1 initialization/diagnostic paths**, never inside the frame loop.

## A.2 Data Layout: SoA + Fixed-Slot Adjacency List

GPU-friendly Structure-of-Arrays. All data resides in device memory, and all information for particle i is located by index i.

```
pos[N]         float3   position (L1 projection coordinates, reconstructed from tangent table)
radius[N]      float    radius (determined by degree)
degree[N]      int      current connection count = redundant cache of nb_count
active[N]      bool     activity (false = potential pool)

nb_idx[N×S]    int      adjacency slots: row i stores neighbor indices for i, -1 = empty slot
nb_dir[N×S]    float2   direction of neighbors on i's sphere (θ, φ)
nb_count[N]    int      number of occupied slots
```

**Slot count S = 64 is the upper bound for memory allocation, not a logical degree limit.**

This distinction is crucial:

- v1: `if degree >= 42: reject connection` —— 42 is a hard-coded magic number (incorrect)
- v1.x: the logical upper limit emerges from the solid angle geometry (see §A.3); S=64 is only a memory budget. If a particle can geometrically accommodate 65 neighbors (extreme size disparity), that means S is too small, just increase S — **the constraint lies in geometry, not in constants**.

Why 64: kissing number of equal-sized spheres is 12; with size disparity, a 40° free angle can accommodate a sphere with r≈1.85, and the theoretical upper bound on degree is determined by the smallest particle size. 64 covers all physically reasonable scenarios and is a power of two (alignment).

## A.3 Exclusivity = Non-penetration in Spherical Angular Coordinates

This is the core theorem of v1.x, unifying two independent constraints into one.

### A.3.1 Angular Radius

Particle j (radius r_j) is tangent to particle i (radius r_i). j occupies a dome region on i's surface, with its **angular radius**:

```
α_j = asin( r_j / (r_i + r_j) )
```

From the center of i, the solid angle Ω_j obscured by j is 2π(1 − cos α_j). For equal tangent spheres (r_j = r_i): α = 30°, Ω ≈ 0.842 sr.

### A.3.2 Exclusivity Theorem

> **j and k do not penetrate each other (non-penetration) ⟺ on i's surface, angular distance(j,k) ≥ α_j + α_k**

Proof: The center-to-center distance between j and k is ≥ r_j + r_k, directly giving the lower bound of angular distance by the cosine law (triangle i−j−k). Conversely, when the angular distance ≥ α_j + α_k, the two domes do not intersect, allowing coexistence of the spheres.

**Conclusion: Non-penetration does not require 3D distance calculation.** Each particle performs exclusivity checks within its own spherical angular coordinate system using pure angular comparison. This is an O(deg²) local operation and only involves angular domain calculations — no need to access other particles' pos.

### A.3.3 Connection Acceptance Criterion (Alternative to Degree Hard Cap)

Particle i considers accepting a new neighbor k (candidate direction d_k, radius r_k):

```
For each existing neighbor j occupying a slot on i:
    If angular_distance(d_k, dir_j) < α_j + α_k − ε:  Reject
All pass: Accept, write into empty slot
```

**Emergence of Degree Upper Bound:** When the spherical surface is covered by neighbors' domes to the point where all remaining gaps are smaller than the minimum allowable angular radius, any candidate will be rejected — saturation occurs naturally. There is no `if degree >= constant`.

### A.3.4 Quantitative Validation: Emergence Path of 12

Taking neighbor layout as spherical triangulation (each added neighbor splits one gap), the number of gaps (faces) when there are n neighbors: `F = 2n − 4` (Euler formula).

> Provisional assumption: The r≡1 in this section is a **placeholder** — the specific radius law r=f(deg) has not yet been derived, only that "the larger the degree, the larger the radius" (monotonically increasing). See §A.12.2.

The maximum radius of a new sphere that can fit into a gap is calculated using Cartesian exact solution — the new sphere is tangent to the center i (radius R) and three equal-sized neighbors (radius r_n), with face center angular distance θ:

```
r_max = R·(R+r_n)·(1 − cosθ) / ( (R+r_n)·cosθ − R + r_n )
For equal spheres R = r_n = 1:  r_max = 2(1 − cosθ) / (2cosθ)
```

| n (degree) | Configuration | Number of gaps 2n−4 | Face center angular distance θ | Maximum allowable new sphere radius r_max | Judgment (r_min = 0.415) |
|---|---|---|---|---|---|
| 4 | Tetrahedron | 4 | 70.53° | 2.00 | All can grow |
| 6 | Octahedron | 8 | 54.74° | 0.73 | All can grow |
| 12 | Icosahedron | 20 | 37.38° | 0.258 | **r_max < r_min → All cannot accommodate → Saturation** |

**12 is not a preset parameter:** It is the intersection of two constraints: "equal-sized spheres + minimum creation volume." r_min is determined by the gap volume threshold in the creation rule — as long as r_min > 0.258, 12 degrees naturally become the saturation point. The code does not use any `12` as a degree criterion.

> Note: Gaps in the icosahedron can still accommodate small spheres with r ≤ 0.258, so strict saturation requires an r_min constraint. r_min = 0.415 corresponds to v1's MIN_GAP_V ≈ 0.3 sphere — semantic meaning: creation has a minimum volume cost, and gaps smaller than this volume are not activated.

## A.4 Port Model and Creation

### A.4.1 Port = Gap

A particle's "port" is not a pre-assigned slot number, but rather **the gap area on the sphere that is not covered by neighboring domes**. Each gap is defined by its three boundary neighbors (a face of a spherical triangulation).

> **Every time a new neighbor is added, the number of gaps increases by 2 (not 3)**: the new neighbor falls within a gap and establishes adjacent arcs with the three boundary neighbors. V+1, E+3, F += 2 by Euler's formula.
> The total free area monotonically decreases, and usable ports (gaps that can accommodate ≥ r_min new spheres) eventually reach zero — this is saturation.

### A.4.2 Two Types of Gaps

**Type A (edge gaps, main growth channels)**: A pair of neighbors (j,k) of particle i are tangent (angular distance ≈ α_j+α_k) → three spheres i,j,k are mutually tangent → there exists one dual hole position on each side of the great circle of jk (Descartes' theorem). A new sphere tangent to i,j,k **gives birth to deg=3**. Detection = enumerating neighbor pairs, O(deg²).

- Analytical solution for the direction of the hole: v_new = x·v_j + y·v_k + z·(v_j×v_k)/|v_j×v_k|, z = ±√(1−|xv_j+yv_k|²) (± indicates the two dual sides).
- Equal-sized sphere chain (r_new = r_i): the deflection angle Δ = 54.74° (dual position of a regular tetrahedron).
- Exclusivity check naturally resolves whether "the hole is already occupied": the side with an existing fourth sphere is rejected, and the empty side is accepted.

**Type B (face holes)**: A triplet of neighbors (j,k,l) of particle i are pairwise tangent → central hole. However, from §A.3.4: tetrahedral hole r_max=0.225, icosahedral hole 0.258, both < r_min=0.415 → **Type B never activates in equal-sized sphere configurations**.

> Design lesson: a pure Type B implementation would lead to zero growth of deg=3 clusters (deg=3 has only C(3,3)=1 triplet face, the central hole, which is always infeasible) — all surface growth comes from Type A.

**Essential difference from v1**: v1 scans all active particle triplets (O(N³)); v1.x checks only a particle's own neighbor pairs (O(deg²)).

### A.4.3 Creation is Connection

When a new particle is born from a Type A gap, it is **naturally tangent to the three spheres i, j, k**. The creation event comes with 3 edges — no need for post-processing to check tangency.

> Deduplication: the same gap is reported by i, j, k on both sides. L1 groups them by sorted triplets, then clusters within each group by pos (two dual gap positions), and creates one per cluster.

Secondary neighbor tangency as a supplementary V⁺ source: i iterates through neighbors of its neighbors, finds unconnected but distance ≤ (r_i+r_k)(1+tol) and passes exclusivity check → sends connection request.

## A.5 Frame Protocol: kernel Sequence and L1 Synchronization Points

A frame = the following kernel sequence. Each kernel is fully parallel within itself, and kernels are synchronized at L1 synchronization points (device-wide barrier).

```
K1 update_volume      per particle: degree → radius (local formula, no communication)
K2 detect_gaps        per particle: enumerate neighbor pairs (j,k) of type A gaps → generation request queue (atomicAdd)
--- L1 synchronization: deduplicate generation requests (sort triplet groups + group-wise pos clustering, ± dual hole positions form clusters) ---
K3 spawn              per request: write new particle pos/radius, establish 3 edges (mutual slot writes, exclusive final review)
K4 connect_existing   per particle: check tangent of second-degree neighbors + exclusive preliminary screening → connection request
--- L1 synchronization: deduplicate connection requests ((i<j) binary key) ---
K5 commit_edges       per request: mutual exclusive final review → mutual slot writes
K5b impenetrability   per particle: check neighbor pair angular distance < α_j+α_k → slightly adjust nb_dir along great circle arc (spherical sliding)
K6 mark_dangling      per particle: nb_count < 3 → mark
K7 purge              per marked particle: clear all slots; per unmarked particle: clear slots pointing to deleted neighbors
K8a check_mutuality   per particle: read-only check mutuality of each slot (j is inactive / j's row does not contain i) → retain mask
K8b compact_mask      per particle: compact own row according to mask, set all trailing slots to EMPTY
```

**Frame Semantic Constraints**:

- Within a frame, K6/K7 are executed only once (no cascade dissolution). New dangling edges produced by deletion become input for K6 in the next frame — implementation of the Imperfection Theorem.
- The sliding in K5b only modifies the angular coordinate of nb_dir, not pos. Global reconstruction of pos (tangent → Cartesian) is handled by the L1 projection layer (BFS reconstruction, see §A.8).
- **K8 is duality accounting, not frame logic**: concurrent edge writes within a frame may leave two types of residues — (a) slots pointing to inactive particles; (b) one-way edges. No cross-row cleanup is done within the frame; K8 fixes them uniformly (K8a generates mask by reading entire row, K8b only writes own row).
- **In comparison with pure combinatorial L0**: K2's "j,k mutual neighbors are gaps" criterion does not include direction, which may also treat **separated triangles** as gap cones → cross defects. Pure combinatorial L0 uses `tri_is_face` in three subsequent checks to correct this (see §2.4).

## A.6 Edge Consistency: Bilateral Confirmation of V⁺

Edges are bilateral, and unilateral writes are the in-memory equivalent of dangling edges. Rules:

1. **Creation edge (K3)**: Three edges are written by the request processing thread into both parties' slots at once. The new particle's slot is brand new; the slots of the three existing particles must atomically occupy their slots (`atomicCAS`). If either party's slot is full/exclusively conflicted → the entire creation request is rolled back (the new particle becomes inactive). Rollback does not clear the other party's written slots — residual entries are uniformly cleaned by K8 at frame end.
2. **Connection edge (K5)**: The request carries the (i,j) sort key, and after deduplication in L1, a single thread writes both parties. Before writing, each side rechecks for exclusivity. If the second side's write fails, no intra-frame cleanup is done — residual half-edges are repaired by K8.
3. **Deletion edge (K7)**: All edges of the deleted particle are cleared by their neighbors from their slots pointing to it.
4. **Repair edge (K8)**: Unified accounting at frame end. All "entry removals" only occur in K7 and K8b, both of which are single-threaded exclusive operations on their own lines, with no cross-line write contention.

Exclusivity checks are executed at two moments: when the request is issued (pre-screening) + when writing (final review). The final review is authoritative.

> Race condition handling (v1.x finalized): When K3 is multi-threaded, thread A's exclusivity final review of g may read the intermediate state of g's adjacency line, occasionally causing rollback or half-edge writes. All such residues are deterministically repaired by K8a/K8b at frame end. If experiments show that the exclusivity violation rate is unacceptable, it will be upgraded to slot locks (locks are added in ascending index order to avoid deadlocks).

## A.7 Dangling and Deletion

- **Verdict**: nb_count of the particle itself in K6. < 3 → dangling. Purely local, no global communication required.
  > Minimum degree in triangulation = 3 (tetrahedron vertex). Its relation to the axiom "minimum degree ≥ 2": 2 is the graph-theoretic self-consistent lower bound, 3 is the 3D triangulation closure lower bound. Implementation uses 3.
- **Deletion**: In K7, particles marked as active=false have their neighbors clear the corresponding slots. **Only one layer is deleted** — new dangling caused by neighbors' deg dropping to 2 is detected by next frame's K6.
- **Potential pool**: Slots with active=false enter the free list, and K3 prioritizes reusing them when allocating.

## A.8 L1 Projection Layer: Coordinate Reconstruction and Global Quantities

L0 only has angular coordinates (nb_dir). Cartesian coordinates pos are projections reconstructed by L1:

1. Starting from any active particle, BFS: known (i_pos, j in i spherical direction dir_ij, r_i + r_j) → j_pos = i_pos + (r_i + r_j)·cart(dir_ij)
2. Multi-path inconsistency → take least squares harmony (this itself is the source of σ field inhomogeneity, not error)
3. Trigger timing: at the end of each frame (if L2 needs observation) or on demand

L1 is also responsible for: deduplication of creation/connection requests, slot allocation, compaction, snapshot output. **L1 makes no physical decisions; it only arbitrates and keeps records.**

## A.9 L2 Emergence Observation: Read-Only Projection

L2 (CPU or asynchronous GPU stream) reads from snapshots, **never writes back**:

- Crystallite detection: deg ≥ threshold → statistics (the threshold itself is an observational definition, not a dynamical constraint)
- 12-crystallite closed loop: connected component of crystallite-induced subgraph, judged as n=12 / E=30 / all deg5 / Σ(6−deg)=12
- Global trajectory of Σ(6−deg), σ=V/E, dangling rate, open-boundary ratio
- Confirmation of regular icosahedron geometry (tangent error)

Honest standard for U4 assertion (emergence of 12-crystallite closed loop): **the closed loop must be naturally evolved by the frame sequence, no external force orientation / relaxation placement is allowed**.

## A.10 correspondence table with v1

| v1 (archived) | v1.x geometric-style L0 | changes |
|---|---|---|
| `step1_create` O(N³) global triple scan | K2 each particle scans its own 2n−4 gaps | O(N³)→O(Σdeg²), global→local |
| `step2_connect` + `max_deg=42` hard cap | K4 exclusive angular distance check | magic number→geometric emergence |
| `step1b_gap_fill` center-mediated two-layer scan | none (K2 already covers) | deleted |
| `enforce_tangency` BFS global reconstruction | L1 projection (at frame end/on demand) | demoted to projection layer tool |
| `step3b_enforce_impenetrability` | K5b spherical sliding (pure angular coordinates) | retained, replaced with angular distance criterion |
| `step4_dangling` / `step5_purge` | K6 / K7 | equivalent, clarified "only delete one layer" |
| `CRYSTALLITE_DEGREE_THRESHOLD=42` | no such constant | saturation determined by r_min + solid angle emergence |
| `MAX_CONTACTS=12` | no such constant | 12 emerges from §A.3.4 chain |

## A.11 Falsifiable assertions (geometric L0, downgraded with implementation)

| # | Assertion | Criterion |
|---|----------|----------|
| V2-U1 | Imperfection Theorem | After each frame K7 execution, the next frame K6 must mark new dangling ends (dangling regeneration rate > 0) |
| V2-U2 | Topological identity | For any frame, Σ(6−deg) = 6V − 2E (bookkeeping identity) |
| V2-U3 | Saturation emergence | Let r_min=0.415, the degree distribution upper bound of equal-sized sphere configuration naturally stops at 12, no constant 12 in code |
| V2-U4 | 12 crystallite closed loop | Pure frame sequence evolution emerges subgraph with component level n=12/E=30/deg5/Σ=12 (no external force) |
| V2-U5 | Exclusivity zero violation | For any frame and any particle, all neighbor diagonal distances ≥ α_j+α_k−ε (must use device-side nb_dir to determine) |
| V2-U6 | Locality | All kernel's maximum memory access jump ≤ second-degree neighbors |
| V2-U7 | Determinism | Same configuration same trajectory (atomicAdd order non-deterministic → deduplicate and sort to ensure determinism) |

## A.12 Implementation Change Log (v1.2, 2026-09-11)

The first round of smoke tests exposed four implementation-level defects (§A.12.1, all fixed); additionally, an issue previously marked as "ontic conflict" was re-evaluated and ruled as **radius law un-derived + neighbor ball reordering missing** (§A.12.2–§A.12.3). After four fixes, V2-U2/U3/U5/U7 all passed.

### A.12.1 Fixed

| # | Defect | Root Cause | Fix |
|---|------|------|------|
| 1 | 16 cases of adjacency mutual reference violation | K3 rollback / K5 failed cleanup called `nb_slot_remove` to do swap-compress, which read `nb_count` lagging behind the concurrent `nb_slot_claim`'s `atomicAdd` | Prohibit cross-row clearing within a frame; add K8a/K8b for unified fix at frame end. Remove `nb_slot_remove` primitive |
| 2 | Final review and slot claiming non-atomic | Two concurrent requests could simultaneously pass exclusive final review and then each claim a slot, with overlapping directions | Add row-level spin lock `p.lock[N]` (`atomicCAS`+`__threadfence`). "Final review + slot claiming" in K3/K5 are completed within the lock |
| 3 | New particle row direction disorder (neighbors collapsed to 3°–11°) | K3 used three parties to directly move their local negative vectors into new particle rows — different coordinate systems' orientations cannot be mixed | New particle rows now use **cosine theorem**: four sphere centers form a rigid tetrahedron, and ni calculates three adjacent angles in its own frame. Purely local and precise |
| 4 | K5b sliding pushed neighbors closer together | When axis `n = v_a×v_b`, turning v_a +δ and v_b −δ would make them move closer to each other | Change to v_a −δ and v_b +δ (pushing apart). Previously, frame 1 didn't trigger because the angle was exactly what was needed, but in frame 2 after radius increased, it triggered, causing a violation burst |

There were also host-side code fixes: CuPy used system default encoding (GBK on Windows) to write temporary source files, and non-GBK characters (`³`, `⁺`, `−`) in source code comments would throw `UnicodeEncodeError`. `host.py::_gbk_safe()` did a downgrade purification before loading (only affecting comments).

Post-fix testing (regular tetrahedron seed, 8 frames, capacity=4096):

```
Frame  V   dev violation  pos violation  overlapping pairs  worst ratio  gap original/deduplicated  connections original/deduplicated
1     8      0           0             0                1.000        12/4                      0/0
2    12      0           0             0                0.982        36/24                     2/1
3    14      0           0             0                0.986        44/39                     0/0
4    15      0           0             0                0.980        48/44                     0/0
5    16      0           5             1                0.980        50/46                     0/0
6    16      0           5             2                0.980        55/51                     0/0
```

- `dev violation` uses device-side `nb_dir` judgment (L0 authoritative state): all 0 ✓
- Adjacency mutual reference violations: 0, deterministic preservation (two runs of V trajectory consistent) ✓
- V2-U2 / U3 / U5 / U7 all passed ✓
- `pos violations / overlapping pairs` comes from L1 projection drift (BFS reconstruction), not L0 violation — L0 is angular coordinate invariant, pos is only for L2 observation

> **Clarification on criterion attribution**: V2-U5 must use device-side `nb_dir` judgment. The initial test used `pos` projection judgment, which equates to measuring projection drift instead of axioms — corrected, and projection drift was listed as INFO separately.

### A.12.2 Radius Semantics Ruling: r = f(deg) Has Not Been Derived, L0 Temporarily Uses Unit Sphere

**User Correction (2026-09-11)**: Degree and radius are **not** directly functionally related. The only known constraint is directional — **higher degree → larger radius** (monotonically increasing); the specific form of r = f(deg) **has not been derived**.

**Revoked an old attribution**: The initial version of this section attributed "V stuck at 16 from frame 6" to "`V = 1+deg` conflicting with rigid close packing ontology". This attribution was overturned by §A.12.1#3 — the real reason V is stuck is **K3 using local coordinate system orientation to move in new particle rows** (mutual pointing deviation non-zero), after fixing this, V trajectory is normal.

**Layered Rulings**:

| Layer | Role of Radius | Value | Basis |
|-----|----------------|-------|-------|
| L0 Geometry (Exclusive/Tangent) | `degree_radius()` single-point entry | Currently r≡1.0 (neutral placeholder) | r=f(deg) not derived |
| L2 Cognitive Projection (σ/temperature/volume) | "Volume grows with degree" expression | To be defined | Projection quantity, not L0 geometry |

The value of `degree_radius()` is a **freely designed lever**, replacing this function allows experimenting with different laws without changing any kernel.

**Measured (8 frames of regular tetrahedron seed, L0 `nb_dir` criterion)**:

| Radius Law | Deviation Violations at Frame End | Worst Ratio |
|-----------|----------------------------------|-------------|
| Equal size r≡1.0 | **0** | 1.000 (zero violations across all frames) |
| r = 1 + 0.01·deg | 36 | 0.956 |
| r = cbrt(0.75(1+deg)/π) | 313 | 0.801 |

**Obstacle Description (Implementation Layer, Not Ontological Conclusion)**: Once an edge is established, the `nb_dir` at both ends is **frozen** local angular coordinates. Any non-constant radius law will change the angular radius α, while frozen angular positions do not follow → exclusivity is destroyed. To allow r to vary with deg and have zero violations, a "neighborhood sphere realignment" mechanism must be paired (see §A.12.3).

> **Pure combinatorial L0 handling of this section**: L0 no longer needs radius; this open question automatically resolves at the L0 level; radius returns as a purely projected quantity (regular icosahedron projects to equal-radius spheres, etc.).

### A.12.3 Neighborhood Sphere Realignment: Implemented (Pure Angular Domain Consistency Solving)

**Implemented**: Pure angular domain neighborhood sphere realignment `host.angular_realign()` (`run_frame(angular_iters=N)`, default 0 = disabled).

- Representation: Each edge e=(i,j) maintains a single global frame spherical direction u_e; i row writes `cart2sph(u_e)`, j row writes `cart2sph(−u_e)` ⟹ mutual pointing `u_ij=−u_ji` is established by construction.
- Solving: Each constraint violation (e,f) is an angular deficit on the sphere; for u_e, u_f, correct each by deficit/2 along great circles, accumulate and normalize, iterate until convergence. **Only modifies nb_dir, not pos**.
- Key Fix: The actual direction of edge e at vertex v is s·U[e] (s=+1 if v is ei, −1 if it's ej). Early versions ignored the sign → solved the wrong constraint system.
- Isolated Verification: Manually perturb nb_dir to create 72 violations → 120 iterations later = 0, mutual pointing deviation restored to 0.03°.

**Measured (14 frames of regular tetrahedron seed, `angular_iters=120`)**:

| Radius Law | Deviation Violations at Frame End | Worst Ratio | Mutual Pointing Deviation | V |
|-----------|----------------------------------|-------------|---------------------------|---|
| r≡1 (unit sphere) | 0 (all frames) | 1.000 | 0.0° | 128 (frame 8) |
| r = 1 + 0.01·deg | 0 (all frames) | 1.000→0.991 (equal after frame 12) | 0.0° | 1059 (frame 14) |

Iteration residuals (max angular deficit) stabilize at ≈0.010 rad. ⇒ Neighborhood sphere realignment connects "r varying with deg" and "zero violations of exclusivity". However, the residual remains stable at ~0.99 (≈1%), not infinitely safe.

**Concept Clarification (User 2026-09-11)**: Realignment is **not** "higher degree pushes others away". That's a residue of background/force-driven thinking — SPUM has no background container, the relative angular positions of neighbors are determined by the consistency of the relationship itself. This step merely **re-solves a set of self-consistent angular coordinates** in projection coordinates; iteration is a numerical solving method, not ontological dynamics.

**Deprecated**: Scheme C — L1 frame-end global realignment (`host.global_realign`). Measured **non-convergent**: 5000 iterations and 300 iterations yield the same result, the embedding geometry is infeasible under over-constrained contact graphs. `run_frame(realign_iters=0)` defaults to disabled.

**Radius Law Lever**: `FrameScheduler(radius_slope=s)` → `degree_radius()` returns `1 + s·deg`; `None` → unit sphere placeholder.

## A.13 spatial_proto minimum 3D creation prototype experiment record (2026-09-12)

> In parallel with §A.5's host.py (K1–K8 distributed implementation), `_archive_v1/l0_legacy/spatial_proto.py` is a smaller experimental prototype:
> CuPy dynamic backend (`cp|np`, falls back to NumPy without GPU), only answering one question — **can non-uniform 3D creation approach the relational capacity of 42/50**.
> It does not perform dangling deletion (V⁻), only creation V⁺ + global rearrangement at frame end. This round's conclusion is fixed, and the code remains unchanged.

### A.13.1 Radius law candidate: A2 area budget derivation

§A.12.2 once ruled that "r = f(deg) has not been derived, L0 temporarily uses unit sphere". spatial_proto proposes a **monotonically increasing lower bound candidate** derived from axioms:

- A2 exclusive occupation: each contact point (relation) occupies a unique area κ² on the sphere
- T8 critical diameter: D_max = 4κ

Combined → spherical πD² ≥ deg·κ² → lower bound `r(deg) = (κ/2)·√(deg/π)`; where lower bound and upper bound D ≤ 4κ coincide
→ `deg = π(4κ)²/κ² = 16π ≈ 50.27`.

`radius_of()` / `saturated()` are implemented accordingly, with **no 12 / 42 / 50 / 16π** written in the code — thresholds are given by the saturation criterion during runtime.

### A.13.2 Spherical shadow exclusive capacity: 50 is the radius limit, not the spherical capacity limit

`shadow_capacity` (Fibonacci spherical spiral sampling + greedy exclusive angular distance) measured spherical shadow capacity:

| Configuration | Spherical Shadow Capacity |
|---|---|
| Homogeneous (equal-sized balls, degree 1) | 14 ≈ kissing number 12 |
| Heterogeneous R_crit + r(3) | 78 |
| Heterogeneous R_crit + r(1) | 193 |

→ The sphere can accommodate far more than 50 neighbors; **50.27 is the radius limit where `r(deg)` hits R_CRIT = 2κ**, not the spherical capacity ceiling.

### A.13.3 Core conclusion: 12→16→21 capacity lineage + homogeneous dynamics + global rigid wall

Three facts proven to be mutually orthogonal:

1. **Homogeneous dynamics pin max_deg at ~16**: The exact spherical shadow capacity of equal-sized balls returns to the kissing number 12, "triangular creation" is naturally homogeneous, so max_deg being capped at 16 is a **dynamics** issue, not a geometric ceiling.
2. **50 is the radius limit** (§A.13.1), the true upper bound of spherical capacity depends on degree heterogeneity (§A.13.2 reaches 193).
3. **Global embedding rigidity divergence**: The residual in `solve_embedding` becomes inevitable once the number of contact edges E hits the 3D contact graph rigidity threshold 3N − 6 (earlier at V=56 / E=162, delayed to V=125 with increased iterations); even after removing global non-edge exclusions, divergence still occurs, proving that the source of divergence is **edge tangency relaxation itself**, not non-edge exclusions.

### A.13.4 User insight: mutual confirmation, not linear progression

> "This is caused by the program's linear creation, you can increase the number of iterations, taking the evolution of the initial node area as the standard. The universe does not need linear logic, just mutual confirmation."

The root cause is each frame only one pass of linear creation + limited relaxation, with a lack of repeated mutual recognition between nodes. After increasing iterations to `solv_iters=1500, step=0.1`: max_deg 16→21, maximum degree ball neighbor median 10→6, initial area max_deg→18.

### A.13.5 Remaining bifurcation (this round selected "keep status quo record")

After increasing iterations, V=125 (frame 7) hits E = 3N − 6, embedding residual rises from ~1e-4 to 2.06e-1, final state V=500 / E=1494 / degree distribution up to 21 / radius basically pinned at lower bound → **max_deg=21 brings global embedding position noise, not pure clean signal**.

- Direction A: Replace the global non-edge exclusion in `solve_embedding` with a projection that only depends on local angular capacity from `create_frame`, then cleanly measure "can iteration → max_deg reach 50" — decoupling "local angular capacity determining max_deg" from "divergent global embedding".
- Direction B (selected this time): Keep the status quo and record the above conclusion.

> **Subsequent closure (2026-09-13, pure combinatorial L0 side)**: Direction A has been completed and adopted by **bifurcation N4** — but not going for "iteration to max_deg=50", instead **completely canceling global embedding**: L1 is only generated from id + rotation system and local quantities (see §6.3). N4 simultaneously provides an `O(F)` **purely local** criterion `A_v ≤ 2π`, moving the "global rigidity wall" to a local judgment (reading see §7 bifurcation N4).

> **Pure combinatorial L0's handling of §A.13**: The "global embedding rigidity divergence" in §A.13 is a **projection layer** issue (embedding combinatorial structure into Cartesian coordinates), unrelated to L0 combinatorial rules. Pure combinatorial L0 naturally avoids this problem — L0 never performs global embedding. Direction A in §A.13.5 (replacing global embedding with local angular capacity) and pure combinatorial L0's "geometry full descent" are two paths of the same direction.

---

## One-Sentence Summary

> **A two-dimensional closed simplicial complex centered on edges, performing deterministic parallel negotiation on the GPU, with intra-frame concurrency, inter-frame barriers, no randomness, no global order, hard capacity constraints, and exactly two faces per edge.**

Nodes are induced by relations, geometry is projected by rules, and audit counts 12/30/20/42/50/62 from the graph.

Below, any layer can be expanded:

- **Priority design in the negotiation layer** (§3.3–§3.5);
- **Edge-centered data layout and compression primitives on the GPU** (§5);
- **Parallel algorithm for edge-face consistency checks** (§4.4);
- **Projection interface from L0 to L1** (§6.3).
