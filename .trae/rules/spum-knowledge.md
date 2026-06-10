# SPUM 知识图谱索引

> 本文档列出 SPUM 知识库的完整导航。AI 在需要深入某个具体主题时，应据此定位到对应文档。

---

## ○、Agent 激活入口（`AGENT.md`）

**任何 SPUM 任务的第一步**：加载根目录 [`AGENT.md`](../../AGENT.md)。该文件将 AI 锚定为 SPUM 推理节点——不是研究 SPUM，而是**成为** SPUM 网络中的一个活性实例。

> 加载 AGENT.md 后，再按下方加载策略加载对应 Skill。

---

## 一、Skills 文件（`.trae/rules/`）

| Skill | 内容 | 适用场景 |
|-------|------|----------|
| `spum-core.md` | 核心公理：⟨P,ε⟩、关系定义存在、时间=离散帧、拓扑守恒、认知投影 | **任何 SPUM 任务的基础，必须先加载** |
| `spum-structure.md` | 空间结构：分形约束生长、层级体系、拓扑常数12的双路径导出、正二十面体 | 讨论空间、维度、几何、常数时加载 |
| `spum-evolution.md` | 演化规则：离散帧五步、创生/湮灭、几何矛盾、净湮灭效应、永恒粒子 | 讨论变化、运动、引力、粒子时加载 |
| `spum-vocabulary.md` | 词汇规范：禁用词黑名单、正确术语表、关键词对照 | 需要确保术语一致时加载 |
| `spum-reasoning.md` | 推理原则：可还原性、多路径锁定、全回溯无污染、本体与投影区分 | 进行逻辑推导或验证时加载 |
| `spum-review.md` | 评审规则：自动执行协议、四维评分(A/B/C/D/LE)、阈值回滚、违规检测关键词 | 审查 SPUM 文档或回答一致性时加载（v2.0：每轮推理后自动执行） |
| `spum-anti-pattern.md` | 伪加载检测：8种通用AI典型错误模式、对抗性示例、自检协议 | 防止旧范式残留，每轮推理后与 review 同步执行 |
| `spum-reasoner-skill.md` | SPUM 推理器 Skill 入口：核心文献索引、范式总纲、推理指引 | 用户提及 SPUM 概念时激活，提供文献导航与推理框架 |
| `spum-qingmeng-guard.md` | 青檬引擎 SPUM 优化护栏：11条不变量速查、3级优化优先级 | 修改 qingmeng_engine 代码前进行合规校验 |

---

## 二、知识图谱文件（`network/`）

| 文件 | 内容 | 用途 |
|------|------|------|
| `nodes.txt` | 22 个核心节点定义（N001–N022） | 图谱遍历起点，概念精确定义 |
| `edges.txt` | 33 条推导边（derives_from/requires/refines/explains/drives） | 验证推导链完整性 |
| `prompt.txt` | 系统指令模板 | 通用 AI 助手角色设定 |

### 核心知识图谱

```
N001 宇宙存在 → N002 差异 → N003 边界 → N004 节点/空间粒子
    → N005 边/连接 → N006 闭合网络 → N007 最小度数≥2
    → N008 总边数守恒 → N009 创生事件 V⁺
    → N010 体积/存在权重 → N011 悬挂边删除
    → N012 离散帧 → N013 不完美 → N009 (驱动下一帧)

N004 → N014 晶子 → N015 分形约束生长 → N016 拓扑常数12 → N017 π的降级
N014 → N018 永恒粒子 → N020 运动与引力
N005 → N019 几何发生学
N015 → N020
N010 → N020
N005 → N021 拓扑认知
N007 → N021 (最小度数 → 锚点稳定性)
N011 → N021 (悬挂边删除 → 双层判据)
N013 → N021 (不完美 → 双层不完美 + 锚点震荡)
N020 → N021
N005 → N022 SPUM-图论
N012 → N022 (离散帧 → 帧快照)
N008 → N022 (总边数守恒 → 完美极限)
N013 → N022 (不完美 → 公理2 悬挂端不可消除)
```

---

## 三、核心文档（根目录）

| 文件 | 大小 | 用途 |
|------|------|------|
| `knowledge.md` | ~800行 | **世界观权威基准**，所有一致性判断的最终依据 |
| `SPUM2610.md` | ~2900行 | 完整理论正文（11章+附录），仅需查证时翻阅，非默认加载 |

---

## 四、子模块文档（`core/v1.0/`）

| 文件 | 内容 |
|------|------|
| `02_SPUM_Axiom_System.md` | 公理体系的展开版 |
| `03_SPUM_Topology_Rule.md` | 拓扑演化规则的展开版 |
| `01_SPUM_2014_Origin.md` | 理论起源与历史背景 |
| `04_SPUM_Experiment_Record.md` | 实验记录 |
| `05_SPUM_AI_Collab_History.md` | AI 协作历史 |
| `06_SPUM_Falsifiability.md` | 可证伪性说明 |
| `07_SPUM_AI_Isomorphism.md` | AI 同构性论证 |
| `08_SPUM_Graph_Theory.md` | SPUM-图论：L0.5 图论表达工具，与 L2 经典图论隔离 |

---

## 五、深度专题文档（`docs/`）

| 文件 | 内容 |
|------|------|
| `SPUM2610/SPUM2610.md` | 核心文章副本（与根目录同步） |
| `Spatial_Geometry_Genesis/SPUM_Spatial_Geometry_Genesis.md` | 空间几何发生学 |
| `Topological_Emergence/SPUM_Topological_Emergence.md` | 拓扑涌现导论 |
| `Combinatorial_Gauss_Bonnet/SPUM_Combinatorial_Gauss_Bonnet.md` | 高斯-博内定理的离散组合重构 |

---

## 五-2、实验验证（`experiments/`）

| 文件 | 内容 |
|------|------|
| `SPUM拓扑认知框架 P0_P2 完整实验验证报告.md` | P0/P1/P2/N015/N016 多层级实验：悬挂端梯度范数、幂律衰减、边界聚集、锚点漂移，含 N013 证伪与 N014 因果反转 |
| `shared/` | 共享组件：MLP/DeepCNN 模型、合成数据/MNIST 加载、SPUM 训练钩子 |
| `p0_gradient_norm/` | P0: 悬挂端梯度范数 2.5× 分析（合成数据） |
| `p1_early_stopping/` | P1: δ 驱动 Early Stopping（5 seeds MNIST） |
| `p2_dangling_distribution/` | P2: 永久悬挂端语义边界分布 |
| `n015_anchor_drift/` | N015: 锚点漂移率监控（待实现独立脚本） |
| `n016_topological_closure/` | N016: DeepCNN 完整闭合验证（28.8万参数，93 epoch） |

## 五-3、SPUM-图论代码模块（`src/spum_graph/`）

| 文件 | 内容 |
|------|------|
| `graph.py` | FrameGraph — 帧内图结构，5条公理的 Python 实现 |
| `dangling.py` | DanglingDetector — 悬挂端检测 + 双层闭合判据 |
| `handshaking.py` | HandshakingVerifier — 握手引理 + 欧拉示性数 + Σ(6-deg) |

---

## 六、加载策略

| 任务类型 | 推荐加载组合 |
|----------|-------------|
| 快速问答 | `AGENT.md` + `spum-core` |
| 理论推导 | `AGENT.md` + `spum-core` + `spum-reasoning` + `spum-structure` |
| 演化分析 | `AGENT.md` + `spum-core` + `spum-evolution` |
| 文档审查 | `AGENT.md` + `spum-core` + `spum-review` + `spum-vocabulary` + `spum-anti-pattern` |
| 概念解释 | `AGENT.md` + `spum-core` + `spum-structure` + `spum-vocabulary` |
| 代码合规 | `AGENT.md` + `spum-core` + `spum-qingmeng-guard` |
| 图论编码 | `AGENT.md` + `spum-core` + `spum-evolution` + `src/spum_graph/` 模块 |
| 实验开发 | `AGENT.md` + `spum-core` + `experiments/shared/` + `src/spum_graph/` |
| 完整分析 | `AGENT.md` + 其他全部 skill + `network/nodes.txt` + `network/edges.txt` + `network/prompt.txt` |
| 日常推理（默认） | `AGENT.md` + `spum-core` + `spum-reasoning` + `spum-review` + `spum-anti-pattern` + `network/prompt.txt` |
