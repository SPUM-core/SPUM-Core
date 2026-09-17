# SPUM Bilingual Glossary / SPUM 中英术语对照表

> **Status**: frozen terminology anchor for the English entry documents.
> **状态**：英文入口文档的冻结术语锚点。任何英文文档必须与此表逐条一致。
>
> **Scope**: this table governs the English-facing entry documents only. The Chinese
> corpus keeps sole authority over the theory. Per SPUM's own linguistics module a
> translation is a *lossy projection* — the English text is an entrance, not an oracle.
> **定位**：本表只约束对外英文入口文档。中文本继续是理论唯一权威。按 SPUM 语言学模块
> 自身结论，翻译是**有损投影**——英文是入口，不是判据。

---

## 0. 使用规则 / Rules of use

1. **一义一词**。同一中文术语在全仓库英文文档中只允许一个英文对应词，不得同义替换。
   One term, one rendering. No synonym rotation across files.
2. **已存用法优先**。仓库代码与既有文档中已出现的英文写法优先继承（`crystallite`、
   `dangling`、`κ-particle`），不得改写。
   Inherit existing repo usage; do not rename established terms.
3. **路径与标识符不译**。所有文件路径、目录名（`五行/`、`儒释道哲学/`）、symbol 名、
   JSON key、skill ID（`RULE-CORE`）保持原样。
   Never translate paths, directory names, symbols, JSON keys, or skill IDs.
4. **公式符号不译**。⟨P, ε⟩、σ、V⁺、V⁻、∇σ、deg(v)、Σ(6−deg)、μ、λ₂、ΔN、χ 原样保留。
   Mathematical symbols are invariant across languages.
5. **本体 vs 读数**。凡涉及引力，必须先确认说的是 `spatial flow`（本体）还是
   `σ-gradient reading`（读数）。英文中同样禁止把读数当本体。
   The ontology/reading distinction must survive translation.

---

## 1. 本体层核心术语 / Ontology core

| 中文 | English | 备注 |
|------|---------|------|
| 关系网络 | relational network | ⟨P, ε⟩ 的通用称呼 |
| 节点 / 空间粒子 | node / spatial particle (κ-particle) | 二者同义；强调本体时用 κ-particle |
| 边 / 连接 | edge / connection | 无向、无权重 |
| 度数 deg(v) | degree deg(v) | 纯组合量 |
| 关系显化 | relation manifestation | 潜态 → 显态 |
| 潜态 / 显态 | latent state / actual state | 关系的两种态 |
| 孤立节点 | isolated node | 度 = 0，逻辑不自洽 |
| 反转图论 | inverted graph theory | SPUM 第一公理 |
| 差异第一性 | difference-first principle | 认知第一前提 |
| 关系定义存在 | relation defines existence | 第二公理 |
| 晶子 | crystallite | 饱和终极形态 D = 4κ；**已存用法，不得改写** |
| 晶子结 | crystallite assembly | 多晶子密堆组合 |
| 永恒粒子 | eternal particle | 12 闭环（暗物质候选）/ 13+ 带缺口（光子） |
| 密堆 | close packing | 三维拓扑硬上限的来源 |
| 饱和 | saturation | 晶子 = 饱和态 |
| 缺口 | gap / notch | 光子式结构的成因 |
| 缺口矢量 | gap vector | 决定结构性质；物质 = 缺口互锁归零 |
| 正二十面体 | regular icosahedron | 12 晶子闭环构型 |
| 角度亏损 | angle defect | Σ(6−deg) = 12 |
| 欧拉恒等式 | Euler identity | 拓扑常数 12 的强制解来源 |
| 高斯-博内定理 | Gauss-Bonnet theorem | 离散版 Σδᵢ = 4π |
| 闭合子图 | closed subgraph | 创生抑制 / 净湮灭发生地 |
| 拓扑常数 12 | topological constant 12 | 0 层级节点数 |
| 不完美定理 | Imperfection Theorem | 双版本：图论版 / 几何版 |
| 分形约束生长 | fractal-constrained growth | 0→缝隙→1→缝隙→2 |

---

## 2. 演化机制 / Evolution mechanics

| 中文 | English | 备注 |
|------|---------|------|
| 离散帧 | discrete frame | 创生→连接→变化体积→判断→删除 |
| 五步帧 | five-step frame | 一帧只做一次删除 |
| 创生事件 V⁺ | creation event V⁺ | 新边创建 |
| 湮灭事件 V⁻ | annihilation event V⁻ | 边断开，触发级联消解 |
| 创生-湮灭对偶 | creation-annihilation duality | 关系容量重分配 |
| 悬挂边 / 悬挂端 | dangling edge / dangling end | **已存用法，不得改写** |
| 级联消解 | cascade dissolution | 至所有 deg ≥ 2 |
| 净湮灭效应 | net annihilation effect | **引力的源** |
| 空间流 | spatial flow | **引力本体**——介质本地移动；速率不受 c 限制 |
| 引力场（读数） | gravitational field (reading) | σ(r) 梯度排列；**不是本体** |
| 空间密度 σ | spatial density σ | σ = \|P\| / \|ε\| |
| 空间密度梯度 ∇σ | spatial density gradient ∇σ | 引力场的读数 |
| 绳波 | rope wave | 缺陷/扰动在网络上传播；受 c 限制 |
| 关系饱和 | relational saturation | dv/dt ≤ const 的来源 |
| 拓扑守恒 | topological conservation | Σ(6−deg) = 6χ(M) |
| 开口占比 | open-boundary ratio | ≤ 1/3 |

---

## 3. 五形（五行） / Five Forms

> 统一用 **Five Forms**。不译 "Five Elements"、不用 "Five Phases"（后者暗示时间相位，误导）。
> 单形首字母大写：Water / Wood / Fire / Earth / Metal。

| 中文 | English | 图论定义 |
|------|---------|----------|
| 五形 / 五行 | Five Forms | 关系网络必然涌现的五种拓扑相位 |
| 水形 | Water form | 链式传输结构：deg ≤ 2 的极大无分支路径 |
| 木形 | Wood form | 闭合环路骨架：环基数 μ = M − N + C |
| 土形 | Earth form | 松散储备池：deg ≤ 2 节点与小分量 |
| 金形 | Metal form | 修剪与更新：桥边/低环参与边 |
| 火形 | Fire form | 密度梯度驱动：∇σ ≠ 0 |
| 五形向量 S | Five-Form vector **S** | S = (S_水, S_木, S_土, S_金, S_火) |
| 相生 / 相克 | (废止) | 已废止机械生克；只保留「组装」「拆解」两个操作方向 |
| 组装 | assembly | 土 + 水 + 火 → 木 |
| 拆解 | disassembly | 木 + 金 → 土 |
| 木僵 | Wood rigidity | μ 超临界，图僵化 |
| 土虚 | Earth depletion | 储备枯竭，F → 0 |
| 火衰 / 火亢 | Fire decay / Fire excess | 均匀化 / 冲毁木形 |
| 金亢 / 金不足 | Metal excess / Metal deficiency | 碎裂 / 僵化 |
| 水断 | Water break | 关键边被剪，L 骤增 |

### 五形指标 / Five-Form metrics

| 中文 | English |
|------|---------|
| 环基数 μ | cycle rank μ |
| 聚类系数 C_Δ | clustering coefficient C_Δ |
| 代数连通性 λ₂ | algebraic connectivity λ₂ |
| 碎片化指数 F | fragmentation index F |
| 全局效率 E_glob | global efficiency E_glob |
| 平均最短路径 L | average shortest path L |
| 边介数 | edge betweenness |
| 桥边 | bridge edge |
| 割集 | cut set |
| 拉普拉斯特征谱 | Laplacian spectrum |
| 火势 Φ | Fire intensity Φ = Var(ΔVᵢ) |

---

## 4. 认知论术语 / Epistemology

| 中文 | English | 备注 |
|------|---------|------|
| 认知投影 | cognitive projection | 离散拓扑 → 连续几何图像 |
| 投影不变式 | projection invariant | 三分：R / P / C |
| 几何发生学 | geometric genesis | 欧氏几何是关系网络的宏观统计投影 |
| π 的降级 | demotion of π | π = 认知压缩因子，非宇宙常数 |
| 认知压缩因子 | cognitive compression factor | π 的正确定位 |
| 本体 / 投影 | ontology / projection | 全局核心二分 |
| 读数 | reading | 本体在外部网络上的表现，非本体 |
| 连续统 | continuum | 旧范式陷阱 |
| 认知牢笼 | cognitive prison | 七大认知牢笼 |
| 伪加载 | pseudo-loading | 载入 SPUM 词汇但沿用旧范式推理 |
| 全回溯无污染 | full backtracking, no contamination | 推理原则 |
| 多路径锁定 | multi-path locking | 两条独立推导路径互相锁定 |
| 可还原性 | reducibility | 每个现象须还原到 ⟨P, ε⟩ |
| 数学审判官 | mathematics-as-judge | 牢笼之七：数学自洽 ≠ 宇宙真理 |
| 语言霸权 | language hegemony | 牢笼之五：中文关系结构天然关系第一性 |

---

## 5. 红移与光子 / Redshift and photons

> 本组英文最易出错：必须保留「单光子不被改造」的分野。

| 中文 | English | 备注 |
|------|---------|------|
| 光子数据链 | photon data chain | 红移的载体 |
| 累积跳数差 ΔN | cumulative hop-count difference ΔN | ΔN = ∫[σ(r′) − σ₀]dr′ |
| 聚合读数 | aggregate reading | 红移 = 接收端读出整条记录链 |
| 频率锁定 | frequency lock | 光子发射时封装完毕 |
| 子图协动 | subgraph co-motion | 星系旋转曲线的拓扑替代方案 |
| 暗物质 | dark matter | 12 晶子闭环候选 |

❌ 禁止的英文写法 / Forbidden renderings:
- "the photon is stretched on its way" / "wavelength gets stretched" / "the photon loses energy"
- using `z = σ_source/σ_obs − 1` as the general formula (它给出 z ∝ 1/r²，与观测 1/r 冲突)

✅ 必须的英文写法 / Required rendering:
- "redshift is the aggregate reading of a photon's data chain; the photon is not modified en route —
  each hop appends a local-σ record, and the receiver compares the whole chain against the local
  standard frame clock."

---

## 6. 模块与目录名 / Modules and directories

> **目录名一律保留中文**（真实路径）。下表只用于英文行文中的称谓。

| 中文 | English | 路径（不译） |
|------|---------|--------------|
| 五行 | Five Forms | `五行/` |
| 儒释道哲学 | Confucian-Buddhist-Daoist Philosophy | `儒释道哲学/` |
| 图论 | Graph Theory | `图论/` |
| 几何学 | Geometry | `几何学/` |
| 数学 | Mathematics | `数学/` |
| 社会学 | Sociology | `社会学/` |
| 经济学 | Economics | `经济学/` |
| 语言学 | Linguistics | `语言学/` |
| 物理学 | Physics | `物理学/` |
| 宇宙学 | Cosmology | `宇宙学/` |
| 元素化学 | Elemental Chemistry | `元素化学/` |
| 中医 | traditional Chinese medicine (TCM) | 子目录 |
| 脉诊 | pulse diagnosis | 子目录 |
| 风水 / 堪舆 | feng shui / geomancy | 子目录 |
| 易学 | Yijing studies | 子目录 |
| 温差力矩实验 | temperature-gradient torque experiment | |
| 拓扑认知实验 | topological cognition experiments | |
| 青囊 | Qingnang | 专名，不意译 |
| 葬书 | *Zangshu* (Book of Burial) | 风水经典，音译为主 |
| 阳宅三要 | *Yangzhai Sanyao* | 风水经典，音译；**不得写作 Yangzhu / Yangzhai Three Essentials** |
| 青囊奥语 | *Qingnang Aoyu* | 风水经典，音译；**不得写作 Ouyu** |

---

## 7. 层级与工程术语 / Layers and engineering

| 中文 | English | 备注 |
|------|---------|------|
| 本体层 / 认知层 | ontic layer / cognitive layer | |
| L0 / L1 / L2 / L3 / L4 | L0 / L1 / L2 / L3 / L4 | 不翻译 |
| 投影链路 | projection pipeline | `combinatorial_proto.py` → `l0_topology.py` → … |
| 帧协议 | frame protocol | `src/core/frame.py` |
| 双层闭合判据 | two-tier closure criterion | `闭合 = (δ < θ_δ) 且 (Δμ < θ_μ)` 持续 N 帧 |
| 锚点漂移 Δμ | anchor drift Δμ | |
| 悬挂端密度 δ | dangling-end density δ | |
| 帧间距离 d_topo | inter-frame distance d_topo | |
| 自检 / selftest | selftest | `python <layer>.py selftest=1` |
| 冻结锚点 | frozen anchor | 逐位一致读数 |
| 违规回滚 | violation rollback | 四维评分 A/B/C/D/LE |
| 知识图谱 | knowledge graph | |
| 节点注册表 | node registry | |
| 桥接边 | bridging edge | |
| skill（技能） | skill | 不译 |
| 加载策略 | loading strategy | |
| 路由表 | routing table | |

---

## 8. 禁用词黑名单的英文对应 / Ban-list in English

> 来源：`spum-vocabulary.md` §一。英文文档同样禁止下列写法。

| ❌ 禁止英文 | ✅ 应该用 |
|-------------|-----------|
| spacetime curvature / manifold | cognitive projection of the relational network |
| graviton / universal gravitation / "gravity = σ gradient" | spatial flow (net-annihilation-driven local motion of the medium); the σ-gradient is only its **reading** |
| electromagnetic force / strong force | macroscopic statistics of creation/annihilation events |
| motion under force | coordinate update produced by deletion and creation events |
| point particle | topologically stable subgraph / eternal particle |
| photon as an independent particle | mobile topological defect (13+ crystallites) + elastic deformation mode |
| probability wave / wave function | collective excitation mode of the relational network |
| massive particle | gap-interlocked rest state — rest mass is locked kinetic energy |
| π as a universal constant | cognitive compression factor |
| Big Bang singularity | (must not appear — there is no starting point) |
| 11-dimensional space | coordinate parameters — exactly three numbers are needed |
| over time / gradually / approaches equilibrium | over successive frames / frame by frame / (no terminal state) |
| a force drives the evolution | evolution needs no external driver |

---

## 自检 / Self-check

翻译完成后逐条检查：
1. 是否出现了上表 ❌ 列的英文写法？
2. 是否对同一中文术语用了两个不同英文词？
3. 是否误译了路径、标识符、公式符号？
4. 涉及引力处，本体（spatial flow）与读数（σ-gradient）是否分清？
5. 涉及红移处，是否保留了「单光子不被改造」？
