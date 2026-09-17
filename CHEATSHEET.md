# SPUM Cheat Sheet

> Core formulas, key numbers, shortest paths. Used to quickly anchor the AI's reasoning coordinate system.
> For complete derivations see `knowledge.md` / `SPUM2610.md`.

---

## Core Formulas

| Formula | Meaning | Layer |
|---------|---------|-------|
| ⟨P, ε⟩ | relational network as the cosmic ontology = (node set, edge set) | L0 axiom |
| σ = \|P\| / \|ε\| | spatial density = number of nodes / number of edges | L0 |
| ⟨deg⟩ = 2 / σ | average degree = 2 / spatial density | L0 |
| δ = \|D_τ\| / \|V_τ\| | dangling-end density = number of dangling nodes / total number of nodes | L0.5 |
| Σ(6−deg(v)) = 12 | Euler constraint of a closed subgraph = topological constant | L0 |
| μ = M − N + C | cycle rank = edges − nodes + connected components | L1.5 Wood form |
| dv/dt ≤ const | upper bound on relational change within a single frame = 4κ | L0 evolution |
| c = 4κ / τ | speed of light = 4 × capacity limit / frame duration | L0 emergence |
| V⁺ / V⁻ | creation (adding an edge) / annihilation (deleting an edge), the dual operations | L0 evolution |
| ΔS = S_τ − S_{τ−1} | inter-frame change of the Five-Form vector S | L1.5 Five Forms |

## Key Numbers

| Number | Source | Meaning |
|--------|--------|---------|
| **12** | Σ(6−deg) = 12 | topological constant, the demotion-of-π substitute. There are exactly 12 level-0 nodes |
| **4π** | discrete Gauss-Bonnet | total angle defect = 4π, the invariant total of a closed surface |
| **22** | nodes.txt | number of L0 core nodes (N001–N022) |
| **33** | edges.txt | number of core derivation edges |
| **15+** | edges.txt | number of module bridging edges |
| **207** | tests/ | number of unit tests (root `tests/`, `python -m unittest discover -s tests`; the 224 tests of OpenSPUM v1 were archived to `_archive_v1/` together with the implementation) |
| **1/3** | open-boundary ratio | upper bound on the open-boundary ratio derived from Gauss-Bonnet with boundary |
| **D_max = 4κ** | evolution constraint | maximum relational change capacity of a single node within a single frame |

## Layer Diagram

```
L0     ⟨P, ε⟩ ontic substrate (22 nodes, 33 edges)
L0.5   SPUM Graph Theory (5 axioms, GT-001~020)
L1     application modules (Mathematics / Sociology / Economics / Linguistics / Physics)
L1.5   Five-Form diagnostic semantics layer (Water Wood Earth Metal Fire)
L2     Confucian-Buddhist-Daoist Philosophy SPUM reconstruction (14 classics, 57 nodes)
L3     code (src/ + OpenSPUM + experiments/)
```

## Shortest Reasoning Path

```
any concept → backtrack to R1-R9 root nodes → backtrack to N001-N022 → confirm existence
```

## Forbidden Vocabulary

| Forbidden term | Replacement |
|----------------|-------------|
| wave-particle duality | light = eternal particle, propagation = determined by the σ terrain |
| flow of time / spacetime | discrete-frame count |
| container / background space | the bubbles themselves are space |
| force (as a driving cause) | creation/annihilation events + σ gradient |
| probability / superposition state | cognitive projection of incomplete information |
| Big Bang / starting point | change of the relational network has no beginning and no end |
