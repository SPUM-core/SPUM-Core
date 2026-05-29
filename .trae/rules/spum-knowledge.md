# SPUM 知识图谱索引

> 本文档列出 SPUM 知识库的完整导航。AI 在需要深入某个具体主题时，应据此定位到对应文档。

---

## 一、Skills 文件（`.trae/rules/`）

| Skill | 内容 | 适用场景 |
|-------|------|----------|
| `spum-core.md` | 核心公理：⟨P,ε⟩、关系定义存在、时间=离散帧、拓扑守恒、认知投影 | **任何 SPUM 任务的基础，必须先加载** |
| `spum-structure.md` | 空间结构：分形约束生长、层级体系、拓扑常数12的双路径导出、正二十面体 | 讨论空间、维度、几何、常数时加载 |
| `spum-evolution.md` | 演化规则：离散帧五步、创生/湮灭、几何矛盾、净湮灭效应、永恒粒子 | 讨论变化、运动、引力、粒子时加载 |
| `spum-vocabulary.md` | 词汇规范：禁用词黑名单、正确术语表、关键词对照 | 需要确保术语一致时加载 |
| `spum-reasoning.md` | 推理原则：可还原性、多路径锁定、全回溯无污染、本体与投影区分 | 进行逻辑推导或验证时加载 |
| `spum-review.md` | 评审规则：违规检测关键词、严重程度分级、评分速查、自检清单 | 审查 SPUM 文档或回答一致性时加载 |

---

## 二、知识图谱文件（`network/`）

| 文件 | 内容 | 用途 |
|------|------|------|
| `nodes.txt` | 20 个核心节点定义（N001–N020） | 图谱遍历起点，概念精确定义 |
| `edges.txt` | 24 条推导边（derives_from/requires/refines/explains/drives） | 验证推导链完整性 |
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

---

## 五、深度专题文档（`docs/`）

| 文件 | 内容 |
|------|------|
| `SPUM2610/SPUM2610.md` | 核心文章副本（与根目录同步） |
| `Spatial_Geometry_Genesis/SPUM_Spatial_Geometry_Genesis.md` | 空间几何发生学 |
| `Topological_Emergence/SPUM_Topological_Emergence.md` | 拓扑涌现导论 |

---

## 六、加载策略

| 任务类型 | 推荐加载组合 |
|----------|-------------|
| 快速问答 | `spum-core` |
| 理论推导 | `spum-core` + `spum-reasoning` + `spum-structure` |
| 演化分析 | `spum-core` + `spum-evolution` |
| 文档审查 | `spum-core` + `spum-review` + `spum-vocabulary` |
| 概念解释 | `spum-core` + `spum-structure` + `spum-vocabulary` |
| 完整分析 | `spum-core` + 其他全部 skill + `network/nodes.txt` + `network/edges.txt` |
