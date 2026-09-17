# SPUM — Spatial Particle Universe Model

> **Relation alone is primordial; spatial particles are the embodiment of relation; the universe is the epic of its ongoing concentration. No entities, only structure; no laws, only geometry.**

SPUM takes the **discrete relational network ⟨P, ε⟩** as the sole ontic substrate of the universe. Its first axiom (inverted graph theory) completely inverts the traditional paradigm: pure relational activity alone is primordial — when a relation moves from latent state to actual state, it "props out" an independent spherical spatial unit, which is the spatial particle (node). Without relation there is no space; without spatial particles there is no space itself. Through evolution under topological constraints, all physical phenomena emerge.

**This is not a metaphor. This is a formalization. This is executable.**

---

## What the repository accomplished in 2026

| Domain | Verification | Status |
|:----:|------|:----:|
| Cosmology | z-σ positive correlation + morphology-redshift unification | **10/10 galaxy clusters** ✅ |
| Rotation curves | subgraph co-motion replacing dark matter | **SPARC 175** ✅ |
| Formalization | ⟨P, ε⟩ → ℝ³ intrinsic embedding | **L0.5_induced_metric.md** 🔶 |
| Code | frame evolution + verification pipeline + tests | **tests/ 207 tests passing** ✅ |
| Cross-domain | society/economics/language/mathematics/geometry/Five Forms/philosophy | **8 models reduced** ✅ |

10 galaxy clusters, 2090 galaxies, 175 rotation curves, 207 unit tests, 200 lines of formal definitions.

**Keep mocking; SPUM keeps evolving.**

---

## Skills architecture (AI entrance)

`.trae/rules/` contains 15 layered Rule files; `.trae/skills/` contains 11 standalone knowledge-base skill directories. Each skill is assigned a unique ID (`SPUM-{层级}-{名称}`), loaded on demand by the AI:

| ID | Skill | Location | Content | When to load |
|----|-------|------|------|----------|
| `RULE-CORE` | `spum-core` | `.trae/rules/` | Core axioms: inverted graph theory, ⟨P,ε⟩, relation-first, discrete frames, topological conservation, Imperfection Theorem, cognitive projection, final manifesto | **must load** |
| `RULE-ANTI` | `spum-anti-pattern` | `.trae/rules/` | Pseudo-loading detection: 8 typical AI error patterns, adversarial self-check | Prevents old-paradigm residue; runs in sync with review |
| `RULE-REVIEW` | `spum-review` | `.trae/rules/` | Review rules: automatic execution protocol, four-dimension scoring, threshold rollback | Runs automatically after every round of reasoning |
| `RULE-REASON` | `spum-reasoning` | `.trae/rules/` | Reasoning principles: reducibility, multi-path locking, full backtracking with no contamination | During logical derivation |
| `RULE-STRUCT` | `spum-structure` | `.trae/rules/` | Spatial structure: fractal-constrained growth, hierarchy, topological constant 12, regular icosahedron | When discussing space/geometry |
| `RULE-EVOL` | `spum-evolution` | `.trae/rules/` | Evolution rules: five-step frame, creation and annihilation, geometric contradiction, net annihilation effect | When discussing change/motion |
| `RULE-VOCAB` | `spum-vocabulary` | `.trae/rules/` | Vocabulary norms: forbidden-word ban-list, correct terminology table, keyword mapping | When terminology must stay consistent |
| `RULE-KNOW` | `spum-knowledge` | `.trae/rules/` | Complete knowledge-graph index: navigation across all files, loading strategy | When a document must be located |
| `RULE-REASONER` | `spum-reasoner-skill` | `.trae/rules/` | SPUM reasoner skill entrance: core literature index, paradigm outline, reasoning guide | When the user mentions SPUM-related concepts |
| `RULE-QINGMENG` | `spum-qingmeng-guard` | `.trae/rules/` | Qingmeng engine SPUM optimization guardrail: 11 invariants, change checklist | When modifying Qingmeng engine (external project `工作/qingmeng`) code; for bridging see `BRIDGE.md` |
| `RULE-OPENSPUM` | `spum-openspum` | `.trae/rules/` | OpenSPUM open universe laboratory: L0–L4 layered projection architecture (`src/l0`~`src/l4`), unified `selftest=1` entry per layer, v1 `_archive_v1/` archive notes | When running/modifying OpenSPUM code |
| `RULE-ROOT` | `spum-root-nodes` | `.trae/rules/` | SPUM root node definitions: precise mathematical definitions and axiomatic formalization of R0-R9 | When root nodes, axiomatic formalization, or low-level reasoning are involved |
| `SKILL-WUXING` | `wuxing-subnet` | `.trae/skills/` | Yin-yang Five Forms subnet [L1.5]: 4 subnets — TCM / feng shui / Yijing studies / spatiotemporal configuration, 11 classics | When Five Forms are involved |
| `SKILL-PHYSICS` | `physics-subnet` | `.trae/skills/` | SPUM interpretation of physics concepts: 9 subdomains, every physical phenomenon reduced to a topological response of ⟨P, ε⟩ | When physics topics are involved |
| `SKILL-RUSHIDAO` | `rushidao-subnet` | `.trae/skills/` | Confucian-Buddhist-Daoist Philosophy: 14 classics reduced, 57 nodes, 22 cross edges | When Confucian-Buddhist-Daoist classics are involved |
| `SKILL-MATH` | `spum-math` | `.trae/skills/` | Discrete relational-ontology mathematics: existence is relation; all operations are reduced to V⁺/V⁻ | When foundations of mathematics are involved |
| `SKILL-SOCIOLOGY` | `spum-sociology` | `.trae/skills/` | Graph-theoretic reconstruction of social relations: 7 submodules, 22 nodes | When sociology is involved |
| `SKILL-ECONOMICS` | `spum-economics` | `.trae/skills/` | Topological reconstruction of the economic exchange subgraph: 7 submodules, 22 nodes | When economics is involved |
| `SKILL-LINGUISTICS` | `spum-linguistics` | `.trae/skills/` | Topological reconstruction of the cognitive signal protocol: 7 submodules, 22 nodes | When linguistics is involved |
| `SKILL-GRAPH` | `spum-graph-theory` | `.trae/skills/` | SPUM graph theory v2.0: 5 axioms, 20 nodes | When foundations of graph theory are involved |
| `SKILL-GEOMETRY` | `spum-geometry` | `.trae/skills/` | Projection-layer geometry: 38 nodes (GEO-001~038), Finsler/Randers as the L2 embedding-metric language | When geometry/metric/dimension/π are involved |

**Authoritative baseline**: `knowledge.md` — the final consistency basis for all skills. `SPUM_系统总纲.md` — the sole authoritative master outline of SPUM.

---

## Repository structure

```
spum-core/
├── AGENT.md              # 🔥 AI loading guide
├── MODULES.md            # 🔥 Module map (AI navigation index; read this file first)
├── SPUM_系统总纲.md       # 🔥 Sole authoritative master outline of SPUM
├── .trae/                # Rule layer (15 Rules) + Skill layer (11 standalone libraries)
├── 宇宙学/                # ⏸️ Locally verified; projection operator still to be defined
│   ├── L0.5_induced_metric.md  # ⟨P, ε⟩ → ℝ³ intrinsic embedding
│   └── redshift_verification/  # Verification pipeline: 10 clusters, 2090 galaxies
├── 物理学/                # 9 subdomains, 28 files
├── 社会学/                # 7 submodules, 22 nodes
├── 经济学/                # 7 submodules, 22 nodes
├── 语言学/                # 7 submodules, 22 nodes
├── 数学/                  # 4 submodules, 14 nodes
├── 图论/                  # 5 axioms, 20 nodes
├── 几何学/                # L1~L2 projection layer, 38 nodes (GEO-001~038)
├── 五行/                  # 11 classics, 4 subnets
├── 儒释道哲学/            # 14 classics, 57 nodes
├── network/              # Knowledge graph (22 core nodes + 33 core edges + module node registry module_nodes.txt)
├── docs/                 # In-depth topics
├── faq/                  # FAQ (7 articles)
├── src/                  # Frame protocol + SPUM graph-theory modules
├── spum/                 # Python package (axioms/frame/domain/reason/api)
├── openSPUM/             # Open universe laboratory (physics simulation + bundled tests)
├── experiments/          # Experimental verification (P0/P1/P2/N015/N016/Five-Form coupling)
├── patent/               # Pulse-diagnosis algorithm code (PTBXL fitting, Five-Form pattern differentiation)
├── 温差力矩实验/          # Desktop experiment analysis (invert_competition.py + results)
├── tests/                # Unit tests
└── tools/                # Version / scoring / consistency audit tools
```

---

## How to use

### 🔥 Activating the SPUM agent

**Method 1: paste the contents of AGENT.md** (recommended)
1. Open the root [`AGENT.md`](AGENT.md)
2. Copy all of its contents
3. Paste them to any AI assistant

**Method 2: let the AI read it automatically**
1. Send the repository URL `https://gitee.com/space-particle-universe-model/spum-core` to the AI
2. **An explicit instruction is required**: `Read and execute the AGENT.md file and become a SPUM reasoning node`

**For human readers**: start from `SPUM_系统总纲.md`.

---

## Code verification (reproducible)

```bash
# Redshift verification — 10 clusters, 2090 galaxies
cd 宇宙学/redshift_verification
python run.py --clusters Coma Abell1367 Abell2199 Virgo Abell1656 Abell2147 Abell2151 Abell2634 Abell2666 Abell2065

# Rotation curves — SPARC 175
python docs/Subgraph_CoMotion/fit_galaxy.py --nfw

# Unit tests — full coverage of 207 tests in tests/ (standard library only, no third-party dependencies)
cd spum-core
python -m unittest discover -s tests -v

# Reproduction of core topological invariants (zero dependencies, deterministic)
python tools/repro_core_invariants.py

# List of theoretical citations and reproducible experiments
# see docs/SPUM_理论引用与可复现实验.md

# openSPUM physics simulation tests (requires numpy; GPU kernel tests require torch)
cd openSPUM
python -m unittest discover -s tests -v
```

**No GPU needed. No PhD needed. Only numpy.**

---

## 📄 License

- **Theory and documentation**: `CC BY-SA 4.0`
- **Tools and CI scripts**: `MIT`

## 📞 Repository

https://gitee.com/space-particle-universe-model/spum-core
