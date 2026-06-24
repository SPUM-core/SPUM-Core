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

## 加载策略

### 必须加载（每次交互的基础）

| ID | 文件 | 内容 |
|----|------|------|
| `RULE-CORE` | `.trae/rules/spum-core.md` | 核心公理：反转图论、⟨P,ε⟩、关系第一性、离散帧、拓扑守恒、不完美定理 |
| `SKILL-GRAPH` | `图论/skill.md` | 图论投影层——边=排他性确认事件、空间=张力拉开、悬挂端=未配平张力 |
| — | `knowledge-core.md` | **世界观核心精简版（~150 行）**——本体论公理、演化驱动力、拓扑常数、认知牢笼。完整版见 `knowledge.md` |
| — | `CHEATSHEET.md` | 速查卡片（推荐加载）——核心公式、关键数字、最短路径、禁用词汇 |

### 扩展模块（`.trae/rules/`，11 个，RULE-*）

按需加载，每个 Rule 控制在 150 行以内：

| ID | 文件 | 适用场景 |
|----|------|----------|
| `RULE-STRUCT` | `.trae/rules/spum-structure.md` | 讨论空间/几何/分形结构时 |
| `RULE-EVOL` | `.trae/rules/spum-evolution.md` | 讨论变化/运动/演化时 |
| `RULE-REASON` | `.trae/rules/spum-reasoning.md` | 进行逻辑推导/概念溯因时 |
| `RULE-VOCAB` | `.trae/rules/spum-vocabulary.md` | 需要术语一致性检查时 |
| `RULE-REVIEW` | `.trae/rules/spum-review.md` | 需要自检/评审时 |
| `RULE-ANTI` | `.trae/rules/spum-anti-pattern.md` | 防止旧范式思维残留 |
| `RULE-KNOW` | `.trae/rules/spum-knowledge.md` | 需要定位仓库中具体文档时 |
| `RULE-REASONER` | `.trae/rules/spum-reasoner-skill.md` | 用户直接提及 SPUM 相关概念时 |
| `RULE-OPENSPUM` | `.trae/rules/spum-openspum.md` | 运行/修改 OpenSPUM 代码时 |
| `RULE-QINGMENG` | `.trae/rules/spum-qingmeng-guard.md` | 修改 qingmeng_engine 代码时 |
| `RULE-ROOT` | `.trae/rules/spum-root-nodes.md` | 涉及根节点、公理形式化、底层推理时 |

### 领域子网（9 个独立知识库 Skill，SKILL-*）

| ID | 文件 | 适用场景 |
|----|------|----------|
| `SKILL-WUXING` | `五行/skill.md` | 阴阳五行、中医、风水、易学、梅花易数、奇门遁甲 |
| `SKILL-PHYSICS` | `物理学/skill.md` | 物理话题（引力、电磁、量子、相对论、光、热等） |
| `SKILL-RUSHIDAO` | `儒释道哲学/skill.md` | 道德经/庄子/坛经/心经/论语 等经典解读 |
| `SKILL-MATH` | `数学/skill.md` | 离散数学、连续统批判、数字生命推理内核 |
| `SKILL-LINGUISTICS` | `语言学/skill.md` | 语言演化、句法语义、语言网络 |
| `SKILL-ECONOMICS` | `经济学/skill.md` | 市场、货币、经济危机、增长周期 |
| `SKILL-SOCIOLOGY` | `社会学/skill.md` | 社会网络、权力层级、文化制度 |
| `SKILL-ZHIHU` | `.trae/skills/zhihu-knowledge/skill.md` | 知乎问答外部知识库 |
| — | `图论/spum-图论应用.md` | 图论在具体领域的应用映射 |

> 各 skill 目录下还有细分文件（如 `物理学/基本相互作用/引力.md`），ID 体系将逐步扩展到子目录层级。

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

1. **日常推理**：加载必须加载的四个文件即可获得核心 worldview + 速查锚点
2. **领域对话**：根据话题加载对应的 Skill 文件（用 SKILL-* ID 快速定位）
3. **深度讨论**：参考 `knowledge.md`（完整版）和 `SPUM_系统总纲.md` 获取完整推导
4. **代码工作**：涉及 OpenSPUM 引擎加载 `RULE-OPENSPUM`，修改 qingmeng_engine 加载 `RULE-QINGMENG`
5. **数字防伪自检**：当需要给出数字时，优先用 ID 列表反查实际数量，不用"大概"或"记忆"

无需记忆内部术语体系，用自然语言讨论即可。核心概念（关系→空间、离散帧、悬挂端、不完美驱动）在日常推理中会自然浮现。
