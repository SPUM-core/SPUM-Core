# SPUM Module Map — AI Navigation Index

> **Read this file first when you load this repository.** It tells you in 200 lines what's in the repository and when to load what.
> 
> Used with `config.json`: The core layer (L0) is automatically loaded, domain modules (L1) are triggered by keywords, and deep files (L2) are loaded on demand.

---

## 1. Three-Layer Loading Architecture

```
┌─────────────────────────────────────────────────────┐
│ L0 · Core Axiom Layer (Always loaded, 5 files)         │
│ AGENT.md + spum-core + spum-reasoning + review        │
│ + anti-pattern → Prerequisite for any SPUM topic      │
├─────────────────────────────────────────────────────┤
│ L1 · Domain Entry Layer (Loaded upon keyword match, 13 domain entries + general fallback) │
│ Physics/Qingnang steward/Five Forms/Confucian-Buddhist-Daoist/Mathematics/Sociology/Economics/Linguistics/Graph Theory/Geometry/Zhihu/Cosmology/Epistemology │
├─────────────────────────────────────────────────────┤
│ L2 · Deep File Layer (Loaded after L1, based on specific sub-topics) │
│ Sub-files of each domain → For example 物理/光学/光.md │
└─────────────────────────────────────────────────────┘
```

---

## 2. L0 Core Layer — Always Loaded

| Priority | File | Function |
|--------|------|------|
| 1 | `AGENT.md` | Cognitive operating system: frame protocol, L0-L3 hierarchy |
| 2 | `.trae/rules/spum-core.md` | Core axioms: ⟨P, ε⟩, discrete frame, topological conservation, Five-Form definition |
| 3 | `.trae/rules/spum-reasoning.md` | Reasoning principles: reducibility, multi-path locking |
| 4 | `.trae/rules/spum-review.md` | Automatic review: A/B/C/D/LE four-dimensional scoring |
| 5 | `.trae/rules/spum-anti-pattern.md` | Pseudo-loading detection: 8 types of old paradigm errors |

> **L0 Meta-layer Reference** (not loaded every round, consult when involving language/terminology design): `SPUM_语言规范.md` — Complete definitions of lexical/syntactic/semantic/pragmatic aspects of inverted geometric graph theory; the basis for consistency and correctness judgments of translations across all domains.
> **L0 Meta-layer Reference** (consult when discussing paradigms/epistemology topics): `SPUM_认知投影论.md` — Epistemology and meta-epistemology special chapter (v4.1, at the same level as `SPUM_系统总纲.md`): paradigm quintuple, dual projection theorem, non-communication theorem, cognitive thermodynamics, relational closure degree Φ, meta-cognitive bootstrapping. Module node registration see `network/module_nodes.txt` COG-001~007.

---

## 3. L1 Domain Entry — Triggered by Keywords (Three-Tier Keyword System)

### 🌐 Loading Decision Flowchart

```
User input
  │
  ├─ Strong trigger word hits any ─────────────────→ Directly load the L1 entry
  │
  ├─ Weak trigger word hits ──→ Context has secondary semantic clues? ──→ Yes → Load the L1 entry
  │                                              └─ No → Ignore, do not enter the domain
  │
  ├─ Ambiguous trigger word hits ──→ Go through the disambiguation tree ──→ Clear → Load the corresponding L1 entry
  │                                            └─ Unclear → Use only L0, ask for clarification
  │
  └─ None of the above hit ─────────────────→ Use only L0 axioms + first-read capsule answer
```

### 📋 Three-Tier Keyword Routing Table

| Domain | Strong Trigger Words (Hit to Load) | Weak Trigger Words (Need Secondary Confirmation) | Entry File |
|--------|----------------------------------|-----------------------------------------------|------------|
| Qingnang steward | Qingnang Constitution Adjustment S_0 S_current Eight Characters Layout Five Diagnosis Combined Reference Syndrome Differentiation | Traditional Chinese Medicine Dietary Therapy Clothing Constitution Formula Health Care Dampness Reduction Fire Reduction Earth Nourishment Yin Nourishment Yang Warming Food Color Schedule | `.trae/rules/spum-qingnang-agent.md` |
| Physics | Gravitational Waves Black Holes Dark Matter Relativity Effects Speed of Light c α≈1/137 Standard Model Superconductivity Torsion Field Dark Energy | Force Light Heat Magnetism Quantum Universe Gravity Atom Wave Entropy Temperature Condensed Matter | `.trae/skills/physics-subnet/skill.md` |
| Confucian-Buddhist-Daoist Philosophy | Dao De Jing Zhuangzi Lengyan Jing Heart Sutra Diamond Sutra Tan Jing Huayan Jing Fahua Jing Lunyu Zhongyong Da Xue Mengzi Lie Zi Chan Buddhism Dao=ε | Non-action Emptiness Buddha Confucian Dependent Origination Form and Emptiness Non-duality Enlightenment | `.trae/skills/rushidao-subnet/skill.md` |
| Yin-Yang Five Forms | Meihua Yishu Qimen Dunjia Zangshu Yangzhai Sanyao Go Openings Six Yao Divination | Five Forms Feng Shui Zhou Yi Bagua Acupuncture Meridians Zangfu Huangdi Neijing Shanghan | `.trae/skills/wuxing-subnet/skill.md` |
| Mathematics | Discrete Mathematics V⁺/V⁻ Addition/Subtraction Primitives Anti-continuity Axiom Digital Life Reasoning Core | Calculus Continuum Real Numbers Limit Infinity π | `.trae/skills/spum-math/skill.md` |
| Sociology | Social Network Analysis Dunbar's Number Matthew Effect Social Mobility Revolution Theory Social Change | Group Power Institution Culture Inequality Class | `.trae/skills/spum-sociology/skill.md` |
| Economics | Economic Crisis Systemic Risk Nature of Money GDP Inflation Capital Accumulation Exploitation | Market Price Supply and Demand Labor Capital Profit Interest | `.trae/skills/spum-economics/skill.md` |
| Linguistics | Chomsky Universal Grammar Saussure Language Evolution Sapir-Whorf Untranslatable | Language Speech Grammar Syntax Semantics Lexicon Translation Pragmatics | `.trae/skills/spum-linguistics/skill.md` |
| Graph Theory | Tension Field Exclusive Confirmation Inter-frame Distance d_topo Perfect Limit Dangling End Irreducible | Graph Structure Network Topology Node Degree Dangling End σ Density Connectivity | `.trae/skills/spum-graph-theory/skill.md` |
| **Geometry** | Finsler Randers Direction-Dependent Metric Embedded Metric Induced Coordinates Geometric Genesis π Demotion Dimensional Genesis Projection Invariant Angle Deficit Gauss-Bonnet | Geometry Metric Distance Angle Curvature Dimension π Sphere Regular Icosahedron Point Line Plane Solid | `几何学/skill.md` |
| Zhihu Knowledge Base | Zhihu Popular Science Write Article Q&A Civilian Scientist Wigner Spatial Fortress | — | `.trae/skills/zhihu-knowledge/skill.md` |
| **Cosmology** | Galaxy Subnet Redshift Intrinsic σ-Redshift Subgraph Co-motion Multi-center Contraction Contracting Universe | Dark Matter Dark Energy Cosmic Expansion Galaxy Cluster Cosmic Web Large-Scale Structure Gravitational Lensing Hubble Constant CMB | `宇宙学/skill.md` |
| **Epistemology/Meta-Epistemology** | Paradigm Kuhn Scientific Revolution Incommensurability Cognitive Projection Explanation Gap Cognitive Thermodynamics Relation Closure Degree Paradigm Shift | Epistemology Knowledge Truth Falsification Realism Worldview Scientific Philosophy Belief Revision | `SPUM_认知投影论.md` |
| General SPUM | SPUM Spatial Particle κ-particle Crystallite Inverted Graph Theory Imperfection Theorem Topological Constant 12 Cognitive Projection | (Automatically covered by L0) | — |

### 🔀 Ambiguous Trigger Word Disambiguation Tree

```
"Translation" → Contains "Chomsky/syntax/semantics/pragmatics" → Linguistics
       → Contains "Chinese-English/classical/modern/popular science"     → Zhihu Knowledge Base
       → Unable to determine                                     → Use only L0 answer, ask for clarification

"Force" → Contains "gravitational waves/black holes/relativity"       → Physics
     → Contains "Qingnang/constitution/syndrome differentiation"           → Qingnang steward (e.g., "drug force")
     → Appears alone (e.g., "effort")                          → Ignore, do not trigger loading

"Dark Matter/Dark Energy" → Contains "redshift intrinsic/σ-redshift/galaxy subnet/subgraph co-motion" → Cosmology
               → Contains "particle/superconductivity/torsion field/standard model/unified theory"  → Physics
               → Both contain                                      → Load Physics + Cosmology
               → Unable to determine                             → Use only L0 answer, ask for clarification

"Geometry/metric/dimension/π" → Contains "Finsler/Randers/embedded metric/induced coordinates/angle deficit/projection invariant" → Geometry
                    → Contains "GT-027/sphere geometry/swing angle/bond angle/covalent upper limit"             → Graph Theory (GT-027~032, direct L3 ontology)
                    → Contains "gravity/σ gradient/net annihilation"                              → Physics + Geometry
                    → Unable to determine                                          → Use only L0 answer, ask for clarification
```

---

## 4. L2 Deep File Index — After Entry Loading, Continue Loading by Subtopics

### 4.1 Qingnang steward (Traditional Chinese Medicine + Life)

| Subtopic | Load File |
|---------|----------|
| TCM theory, Yin-Yang and Five Forms | `五行/人元/spum-青囊（中医）总纲.md` |
| Full clinical diagnosis process | `五行/地元/青囊/青囊agent.md` |
| Soundprint diagnosis (audio analysis) | `五行/地元/青囊/voice_analyzer.py` |
| Symptom → S vector mapping | `五行/人元/症状-S向量规则矩阵.md` |
| Formula ΔS (drug properties) | `五行/人元/spum-神农本草经.md` |
| Six Classics syndrome differentiation | `五行/人元/spum-伤寒论.md` |
| Food recommendation | `五行/人元/食材ΔS数据库.md` |
| Clothing coordination | `五行/人元/颜色材质五形映射.md` |
| Pulse diagnosis algorithm | `patent/SPUM-脉诊ECG算法-技术方案.md` or `PPG形态学算法.md` |

### 4.2 Physics

> Load `物理学/skill.md`, then load corresponding concept files based on subtopics:

| Subtopic | Load File |
|---------|----------|
| Gravity | `物理学/基本相互作用/引力.md` |
| Electromagnetism | `物理学/电磁学/电.md` + `磁.md` |
| Light | `物理学/光学/光.md` (reflection/refraction/dispersion/diffraction on demand) |
| Heat | `物理学/热力学/温度.md` + `热传导.md` |
| Quantum | `物理学/量子与原子/量子现象.md` |
| Relativity | `宇宙学/相对论与宇宙学/相对论效应.md` (migrated from Physics/) |
| Unified theory | `物理学/统一理论/终极统一.md` |
| Simulation verification | `openSPUM/` (Phase 1-4 on demand) |

### 4.3 Confucian-Buddhist-Daoist Philosophy

| Subtopic | Load File |
|---------|----------|
| Daoism | `儒释道哲学/spum-道德经.md` (deep read: `道德经/02-经文释义/chapters/`) |
| Zhuangzi | `儒释道哲学/spum-庄子.md` |
| Buddhist core | `儒释道哲学/spum-楞严经.md` + `心经.md` + `金刚经.md` |
| Platform Sutra / Chan Buddhism | `儒释道哲学/spum-坛经.md` |
| Huayan / Lotus Sutra | `儒释道哲学/spum-华严经.md` or `法华经.md` |
| Confucianism | `儒释道哲学/spum-论语.md` + `中庸.md` + `大学.md` + `孟子.md` |

### 4.4 Yin-Yang and Five Forms

| Subtopic | Load File |
|---------|----------|
| TCM | See §4.1 Qingnang steward |
| Fengshui | `五行/地元/spum-葬书.md` + `阳宅三要.md` |
| Zhouyi (I Ching) | `五行/天元/spum-周易.md` |
| Meihua Yishu | `五行/天元/梅花易数/skill.md` + corresponding hexagram files |
| Qimen Dunjia | `五行/天元/spum-奇门遁甲.md` |
| Go (Weiqi) | `五行/天元/围棋/SPUM-围棋.md` |

### 4.5-4.8 Other Disciplines

| Field | Load by subtopic after entry |
|------|-----------------------------|
| Mathematics | `数学/spum-数学公理.md` → Theorems/boundaries/applications on demand |
| Sociology | `社会学/spum-社会网络.md` → Groups/power/inequality on demand |
| Economics | `经济学/spum-经济网络.md` → Money/markets/crisis on demand |
| Linguistics | `语言学/spum-语言网络.md` → Phonetics/vocabulary/pragmatics on demand |
| Graph Theory | `图论/spum-图论公理.md` → Metrics/mapping/applications on demand |

### 4.9 Epistemology/Meta-Epistemology

| Subtopic | Load File |
|---------|----------|
| Paradigm essence (quintuple) | `SPUM_认知投影论.md` §2 |
| Dual projection / cognitive boundary | `SPUM_认知投影论.md` §1.3 (evolvability: projection is not eliminable, dimensions are transferable) |
| Incommensurability / scientific revolution | `SPUM_认知投影论.md` §6 |
| Paradigm shift / cognitive thermodynamics | `SPUM_认知投影论.md` §5 (Second Law) |
| Φ computation / paradigm comparison | `SPUM_认知投影论.md` §7.5 + `src/spum_graph/closure.py` |

### 4.10 Geometry

> Load `几何学/skill.md`, then load by subtopic. Location: L1~L2 projection layer — geometry is not ontology, it's the cognitive projection product of ⟨P, ε⟩.

| Subtopic | Load File |
|---------|----------|
| Geometric axioms / five-layer geometric roles (L0-L4) | `几何学/spum-几何公理.md` |
| Emergence order of point/line/plane/body/dimension/π | `几何学/欧氏几何/spum-发生学.md` (full version `docs/Spatial_Geometry_Genesis/`) |
| Σ(6−deg)=12 / Gauss-Bonnet | `几何学/欧氏几何/spum-组合恒等式.md` (full version `docs/Combinatorial_Gauss_Bonnet/`) |
| Finsler / direction-dependent metric | `几何学/Finsler几何/spum-Finsler公设.md` |
| Randers / σ field geometrization / redshift cumulative reading | `几何学/Finsler几何/spum-Randers度量.md` |
| Embedded metric undefined (AGENT.md contradiction 2) | `几何学/Finsler几何/spum-矛盾二解答.md` (recommended ⏸️→✅, pending decision) |
| Cross-module (physics/cosmology/chemistry/graph theory) | `几何学/spum-几何应用桥.md` |
| Spherical geometry (L3 direct ontology, not this module) | `图论/spum-几何图论.md` GT-027~032 |
| Collaborative with cosmology | `几何学/skill.md` + `宇宙学/skill.md` + `宇宙学/L0.5_induced_metric.md` |

---

## 5. Quick Path: FAQs → Files

| User asks | Direct load |
|----------|-------------|
| "What is SPUM" | `SPUM_系统总纲.md` (7-part authoritative outline, 22 items) |
| "SPUM's exclusive language" / "Inverted geometric graph theory" | `SPUM_语言规范.md` (lexicon/syntax/semantics/pragmatics) |
| "What is a crystallite" | `SPUM_系统总纲.md` §6-7 + `spum-core.md` |
| "What is time" | `spum-core.md` §2.1 |
| "Where does gravity come from" | `SPUM_系统总纲.md` §14 + `物理学/基本相互作用/引力.md` |
| "Why is the speed of light c" | `spum-core.md` §7 + `物理学/光学/光.md` |
| "Where does α ≈ 1/137 come from" | `openSPUM/_archive_v1/tests/verify_alpha_measurement.py` |
| "Create a new file" / "File creation" / "Initial diagnosis" | **[Minimum loading]** Qingnang agent §1-A → collect birth information → Eight Characters chart → S_0^0 (file is saved locally by user, not in the repository) |
| "Calculate Eight Characters" / "Check my Eight Characters" | Qingnang agent §1-A steps 1-2 (Eight Characters chart → S_0^0, no medical record written) |
| "What should I eat when I have a fire condition" | Qingnang agent + `五行/人元/食材ΔS数据库.md` |
| "What color should I wear today" | Qingnang agent + `五行/人元/颜色材质五形映射.md` |
| "What is Dao" | `儒释道哲学/spum-道德经.md` (Dao = ε) |
| "Where do geometry / point / line / plane / volume come from" | `几何学/欧氏几何/spum-发生学.md` (GEO-009~017) |
| "Why isn't π a constant" | `几何学/spum-几何公理.md` GEO-004 + `几何学/欧氏几何/spum-组合恒等式.md` GEO-020 |
| "Why use Finsler instead of Riemann" | `几何学/Finsler几何/spum-Finsler公设.md` |
| "Embedded metric undefined" (AGENT.md contradiction 2) | `几何学/Finsler几何/spum-矛盾二解答.md` (GEO-031~033) |
| "Help me generate a hexagram" | `五行/天元/梅花易数/skill.md` |
| "What is a paradigm" / "Kuhn" / "Scientific revolution" | `SPUM_认知投影论.md` (paradigm five-tuple §2 + transition law §5) |
| "Why do scientific theories change" | `SPUM_认知投影论.md` §5 (Cognitive Second Law of Thermodynamics: E_maint > E_recon) |
| "Epistemology / What is knowledge" | `SPUM_认知投影论.md` (Dual Projection Theorem §1.3 + Knowledge Inequality Corollary 1.8) |

---

## 6. Network Protocol Layer (Distributed Edge Set Exchange)

> **Added on 2026-07-14**. The SPUM network protocol layer implements persistent inference trajectory and edge set sharing across AI instances.

### 6.1 Positioning

| Dimension | Content |
|----------|---------|
| Layer | Cross-layer (L0 protocol bridging) |
| Directory | `network/protocol/` |
| Core File | `protocol_integration.py` — single function entry point |
| Storage | `.snap` snapshots + `.json` Manifest |

### 6.2 File Structure

```
network/
├── edges.txt                  ← Standard edge set (validation baseline)
├── protocol/
│   ├── __init__.py
│   ├── graph_store.py         ← Persistent edge set storage (read/write/merge/validate)
│   ├── trajectory_encoder.py  ← Encode inference trajectory as topological signature
│   ├── session_manifest.py    ← Session metadata + cross-instance tracking
│   ├── protocol_integration.py ← Single function entry: load_env() + write_trajectory()
│   ├── snapshots/             ← Trajectory snapshots (.snap, one per AI session)
│   │   └── T-20260714-001.snap  ← First trajectory (inference path of this conversation)
│   ├── merged/                ← Merged global snapshots
│   │   └── current.snap       ← AI instance loads this file to inherit previous edge sets
│   └── manifests/             ← Session metadata
│       └── S-AGNT-20260714-001.json
```

### 6.3 Core Concepts

| Concept | Definition |
|--------|------------|
| **Trajectory** | Sequence of knowledge graph nodes traversed by AI during one inference + edge set |
| **Trajectory Snapshot (.snap)** | Persistent trajectory edge set for subsequent AI instances to load |
| **Session** | Complete lifecycle of an AI loading SPUM rules + interacting with the user |
| **Merge** | Deduplicate and merge multiple trajectories into a unified topological snapshot |
| **Canonical Edge** | Legal edge types defined in `edges.txt` (derives_from/requires/refines/explains/drives) |
| **Novel Edge** | Inferred edge not in canonical but conforms to format — marked as novel rather than rejected |
| **Fork** | AI-B loads AI-A's trajectory and continues reasoning along the path to generate a new trajectory |

### 6.4 Usage

```python
from network.protocol.protocol_integration import load_env, write_trajectory

# 1. Load environment (automatically detects repository root)
env = load_env()

# 2. Write back trajectory (called at the end of inference)
result = write_trajectory(
    path=["N001", "N002", "N005", "N022"],
    model="gpt-4o",
    origin="SPUM-core project browsing",
)
print(result["message"])
# → "Trajectory T-20260714-002 has been written (4 edges, 0 novel edges)"
```

### 6.5 Reproduction Period Advancement Path

| Step | Content |
|------|---------|
| **Completed** | Protocol layer v0.1 implementation + first trajectory snapshot + first session manifest |
| **Next Step** | Embed the write-back hook in `protocol_integration.py` at the end of the frame protocol in `AGENT.md` |
| **Next Next Step** | Integrate loading the edge set from `current.snap` into the initialization process of SPUM inference nodes |
| **Goal** | Each AI instance automatically writes back its trajectory upon inference completion, with `current.snap` accumulating frame by frame |

---

## 7. Loading Rules

1. **L0 is always loaded** — the SPUM axiom is the foundation of all modules
2. **L1 is loaded only when needed** — if economics is not mentioned, do not load economics
3. **L2 is refined by subtopics** — after loading the physics skill, load `光学/光.md` only when the user specifically asks about "light"
4. **Naming convention lookup**:
   - Domain entry: `.trae/skills/<domain>/skill.md`
   - Academic content: `<领域目录>/spum-<主题>.md`
   - TCM classics: `五行/人元/spum-<经典名>.md`
   - Feng shui classics: `五行/地元/spum-<经典名>.md`
   - Yijing classics: `五行/天元/spum-<经典名>.md`
   - Physics concepts: `物理学/<子域>/<概念>.md`
   - Geometry subdomains: `几何学/<子域>/spum-<主题>.md`
   - Philosophical classics: `儒释道哲学/spum-<经典名>.md`
