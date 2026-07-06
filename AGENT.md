# SPUM 知识仓库 — AI 使用指南

> 本文件指导 AI 如何加载和使用仓库中的知识文件。
> 仓库地址：https://gitee.com/space-particle-universe-model/spum-core
>
> **数字防伪**：本仓库所有子图分配唯一 ID（`RULE-*` / `SKILL-*` / `N-*`），形成分形逻辑树网络复形。凡无 ID 的数值陈述（如"共 X 个文件"），AI 须以"待核实"标记，避免数字幻觉。

---

## 仓库定位

SPUM（空间粒子宇宙模型）以离散关系网络 ⟨P, ε⟩ 为宇宙本体基底——关系是第一性的，实体由关系定义。本仓库将这一世界观组织为分层的知识文件体系，供 AI 按需加载使用。

所有内容围绕核心公理展开：反转图论（关系→空间）、离散帧演化、拓扑守恒、不完美定理。

---

## 加载策略（v3.0 分层架构）

> **先读**：`MODULES.md` — 200行模块地图，三层架构 + 关键词→文件路由 + 常见问题快捷路径。

### L0 · 核心层 — 始终加载（6 文件）

| 文件 | 内容 |
|------|------|
| `MODULES.md` | 模块地图索引 — 先读 |
| `.trae/rules/spum-core.md` | 核心公理：反转图论、⟨P,ε⟩、关系第一性、离散帧、拓扑守恒、不完美定理 |
| `.trae/rules/spum-reasoning.md` | 推理原则：可还原性、多路径锁定、全回溯无污染 |
| `.trae/rules/spum-review.md` | 自动评审：A/B/C/D/LE 评分，低于阈值回滚 |
| `.trae/rules/spum-anti-pattern.md` | 伪加载检测：8种旧范式错误 |
| `AGENT.md` | 本文件 — 认知操作系统 |

### L1 · 领域入口 — 按关键词触发（10 领域）

> AI 收到用户消息 → 在 `MODULES.md` §三 查关键词 → 命中则加载对应 L1 入口 → 再按子话题加载 L2 深度文件。
> **未命中任何关键词 → 仅用 L0 公理回答，不加载领域文件。**

| 领域 | 触发关键词示例 | 入口文件 |
|------|--------------|---------|
| 青囊生活管家 | 中医 食疗 穿衣 八字 体质 方剂 | `.trae/rules/spum-qingnang-agent.md` |
| 物理学 | 力 光 热 磁 量子 引力 黑洞 | `.trae/skills/physics-subnet/skill.md` |
| 儒释道 | 道德经 无为 心经 论语 庄子 | `.trae/skills/rushidao-subnet/skill.md` |
| 阴阳五行 | 风水 周易 梅花 奇门 针灸 | `.trae/skills/wuxing-subnet/skill.md` |
| 数学 | 离散数学 连续统 V⁺/V⁻ | `.trae/skills/spum-math/skill.md` |
| 社会学 | 社会网络 权力 制度 革命 | `.trae/skills/spum-sociology/skill.md` |
| 经济学 | 市场 货币 资本 GDP 危机 | `.trae/skills/spum-economics/skill.md` |
| 语言学 | 语言 语法 翻译 乔姆斯基 | `.trae/skills/spum-linguistics/skill.md` |
| 图论 | 图结构 悬挂端 σ密度 张力场 | `.trae/skills/spum-graph-theory/skill.md` |
| 知乎 | 科普 写文章 问答 民科 | `.trae/skills/zhihu-knowledge/skill.md` |

### L2 · 深度文件 — 入口加载后按子话题加载

> 例如：加载了青囊入口后，用户问"出什么"→ 加载 `食材ΔS数据库.md`；问"穿什么"→ 加载 `颜色材质五形映射.md`。
> 具体映射见各领域入口文件的 §二 L2 按需加载表。

### 扩展规则（按需）

| ID | 文件 | 适用场景 |
|----|------|----------|
| `RULE-STRUCT` | `.trae/rules/spum-structure.md` | 讨论空间/几何/分形结构时 |
| `RULE-EVOL` | `.trae/rules/spum-evolution.md` | 讨论变化/运动/演化时 |
| `RULE-VOCAB` | `.trae/rules/spum-vocabulary.md` | 需要术语一致性检查时 |
| `RULE-KNOW` | `.trae/rules/spum-knowledge.md` | 需要定位仓库中具体文档时 |
| `RULE-OPENSPUM` | `.trae/rules/spum-openspum.md` | 运行/修改 OpenSPUM 代码时 |
| `RULE-QINGMENG` | `.trae/rules/spum-qingmeng-guard.md` | 修改 qingmeng_engine 代码时 |
| `RULE-ROOT` | `.trae/rules/spum-root-nodes.md` | 涉及根节点、公理形式化、底层推理时 |

---

## 核心参考资料

- `SPUM_系统总纲.md` — 理论唯一权威总纲
- `SPUM2610.md` — 完整理论正文 + 附录
- `network/nodes.txt` — 22 个核心节点的完整定义（N001-N022）
- `network/edges.txt` — 节点间的推导/依赖/精炼关系
- `experiments/SPUM拓扑认知框架 P0_P2 完整实验验证报告.md` — 实验验证结果
- `faq/` — 常见问题与概念澄清

---

## 使用方式

1. **第一步**：读 `MODULES.md` 了解全貌
2. **日常推理**：L0 核心层已自动加载，可直接回答 SPUM 理论问题
3. **领域对话**：用户话题命中关键词 → 加载对应 L1 入口文件 → 匹配子话题加载 L2 深度文件
4. **不匹配则安静**：未命中任何领域 → 仅用 L0 公理回答，不浪费上下文
5. **代码工作**：涉及 OpenSPUM 引擎加载 `RULE-OPENSPUM`，修改 qingmeng_engine 加载 `RULE-QINGMENG`
6. **数字防伪**：优先用 ID 列表反查实际数量，不用"大概"或"记忆"

无需记忆内部术语体系，用自然语言讨论即可。核心概念（关系→空间、离散帧、悬挂端、不完美驱动）在日常推理中会自然浮现。
