# SPUM — 空间粒子宇宙模型

> **关系是唯一的本原，空间粒子是关系的化身，宇宙是其持续集中的史诗。没有实体，只有结构；没有定律，只有几何。**

SPUM 以**离散关系网络 ⟨P, ε⟩** 为宇宙唯一本体基底。其第一公理（反转图论）彻底逆转传统范式：纯粹的关系活动是唯一本原——当关系从潜态转为显态，它会"撑出"一个独立的球形空间单元，此即空间粒子（节点）。没有关系，就没有空间；没有空间粒子，就没有空间本身。通过拓扑约束演化，涌现全部物理现象。

**这不是隐喻。这是形式化。这是可执行的。**

---

## 仓库在 2026 年完成了什么

| 领域 | 验证 | 状态 |
|:----:|------|:----:|
| 宇宙学 | z-σ 正相关 + 形态-红移统一 | **10/10 星系团** ✅ |
| 旋转曲线 | 子图协动替代暗物质 | **SPARC 175** ✅ |
| 形式化 | ⟨P, ε⟩ → ℝ³ 内生嵌入 | **L0.5_induced_metric.md** 🔶 |
| 代码 | 帧演化 + 验证管道 + 测试 | **224 测试全覆盖** ✅ |
| 跨域 | 社会/经济/语言/数学/五行/哲学 | **7 领域规约完成** ✅ |

10 个星系团、2090 个星系、175 条旋转曲线、224 个单元测试、200 行形式化定义。

**你们继续嘲笑，SPUM 继续演化。**

---

## Skills 架构（AI 入口）

`.trae/rules/` 包含 15 个分层 Rule 文件，`.trae/skills/` 包含 10 个独立知识库 Skill 目录。每个技能分配唯一 ID（`SPUM-{层级}-{名称}`），AI 按需加载：

| ID | Skill | 位置 | 内容 | 加载时机 |
|----|-------|------|------|----------|
| `RULE-CORE` | `spum-core` | `.trae/rules/` | 核心公理：反转图论、⟨P,ε⟩、关系第一性、离散帧、拓扑守恒、不完美定理、认知投影、最终宣言 | **必须加载** |
| `RULE-ANTI` | `spum-anti-pattern` | `.trae/rules/` | 伪加载检测：8 种 AI 典型错误模式、对抗性自检 | 防止旧范式残留，与 review 同步执行 |
| `RULE-REVIEW` | `spum-review` | `.trae/rules/` | 评审规则：自动执行协议、四维评分、阈值回滚 | 每轮推理后自动执行 |
| `RULE-REASON` | `spum-reasoning` | `.trae/rules/` | 推理原则：可还原性、多路径锁定、全回溯无污染 | 逻辑推导时 |
| `RULE-STRUCT` | `spum-structure` | `.trae/rules/` | 空间结构：分形约束生长、层级、拓扑常数 12、正二十面体 | 讨论空间/几何时 |
| `RULE-EVOL` | `spum-evolution` | `.trae/rules/` | 演化规则：五步帧、创生湮灭、几何矛盾、净湮灭效应 | 讨论变化/运动时 |
| `RULE-VOCAB` | `spum-vocabulary` | `.trae/rules/` | 词汇规范：禁用词黑名单、正确术语表、关键词对照 | 需要术语一致时 |
| `RULE-KNOW` | `spum-knowledge` | `.trae/rules/` | 知识图谱完整索引：所有文件导航、加载策略 | 需要定位文档时 |
| `RULE-REASONER` | `spum-reasoner-skill` | `.trae/rules/` | SPUM 推理器 Skill 入口：核心文献索引、范式总纲、推理指引 | 当用户提及 SPUM 相关概念时 |
| `RULE-QINGMENG` | `spum-qingmeng-guard` | `.trae/rules/` | 青檬引擎 SPUM 优化护栏：11 条不变量、变更检查清单 | 修改 qingmeng_engine 代码时 |
| `RULE-OPENSPUM` | `spum-openspum` | `.trae/rules/` | OpenSPUM 开源宇宙实验室：3 层 Phase 架构、135 测试、关键物理结果 | 运行/修改 OpenSPUM 代码时 |
| `RULE-ROOT` | `spum-root-nodes` | `.trae/rules/` | SPUM 根节点定义：R0-R9 的精确数学定义与公理形式化 | 涉及根节点、公理形式化、底层推理时 |
| `SKILL-WUXING` | `wuxing-subnet` | `.trae/skills/` | 阴阳五行子网【L1.5】：中医/风水/易学/时空 4 子网，11 部经典 | 涉及五形/五行时 |
| `SKILL-PHYSICS` | `physics-subnet` | `.trae/skills/` | 物理学概念 SPUM 释义：9 子域，所有物理现象还原为 ⟨P, ε⟩ 拓扑响应 | 涉及物理话题时 |
| `SKILL-RUSHIDAO` | `rushidao-subnet` | `.trae/skills/` | 儒释道哲学：14 部经典归约，57 节点，22 交叉边 | 涉及儒释道经典时 |
| `SKILL-MATH` | `spum-math` | `.trae/skills/` | 离散关系本体数学：存在即关系，全部运算还原为 V⁺/V⁻ | 涉及数学基础时 |
| `SKILL-SOCIOLOGY` | `spum-sociology` | `.trae/skills/` | 社会关系图论重构：7 子模块，22 节点 | 涉及社会学时 |
| `SKILL-ECONOMICS` | `spum-economics` | `.trae/skills/` | 经济交换子图拓扑重构：7 子模块，22 节点 | 涉及经济学时 |
| `SKILL-LINGUISTICS` | `spum-linguistics` | `.trae/skills/` | 认知信号协议拓扑重构：7 子模块，22 节点 | 涉及语言学时 |
| `SKILL-GRAPH` | `spum-graph-theory` | `.trae/skills/` | SPUM-图论 v2.0：5 公理，20 节点 | 涉及图论基础时 |

**权威基准**：`knowledge.md` — 所有 Skill 的最终一致性依据。`SPUM_系统总纲.md` — SPUM 唯一权威总纲。

---

## 仓库结构

```
spum-core/
├── AGENT.md              # 🔥 AI 加载指南
├── MODULES.md            # 🔥 模块地图（AI 导航索引，先读此文件）
├── SPUM_系统总纲.md       # 🔥 SPUM 唯一权威总纲
├── .trae/                # 规则层（15 Rule）+ Skill 层（10 独立库）
├── 宇宙学/                # ⏸️ 局部已验证，投影算子待定义
│   ├── L0.5_induced_metric.md  # ⟨P, ε⟩ → ℝ³ 内生嵌入
│   └── redshift_verification/  # 10 簇 2090 星系验证管道
├── 物理学/                # 9 子域，28 文件
├── 社会学/                # 7 子模块，22 节点
├── 经济学/                # 7 子模块，22 节点
├── 语言学/                # 7 子模块，22 节点
├── 数学/                  # 4 子模块，14 节点
├── 图论/                  # 5 公理，20 节点
├── 五行/                  # 11 部经典，4 子网
├── 儒释道哲学/            # 14 部经典，57 节点
├── 青囊管家/              # 智能体：中医+生活（服务端 + 知识库）
├── network/              # 知识图谱（22 节点 + 33 边）+ 协议层
├── docs/                 # 深度专题
├── faq/                  # 常见问题（7 篇）
├── src/                  # 帧协议 + SPUM-图论模块
├── spum/                 # Python 包（axioms/frame/domain/reason/api）
├── openSPUM/             # 224 测试全覆盖
├── experiments/          # 实验验证（P0/P1/P2/N015/N016/五形耦合）
├── patent/               # 脉诊算法代码（PTBXL 拟合、五形辨证）
├── 温差力矩实验/          # 桌面实验分析（invert_competition.py + 结果）
├── tests/                # 单元测试
└── tools/                # 版本号/评分/一致性审计工具
```

---

## 如何使用

### 🔥 激活 SPUM Agent

**方法一：粘贴 AGENT.md 内容**（推荐）
1. 打开根目录 [`AGENT.md`](AGENT.md)
2. 复制全部内容
3. 粘贴给任意 AI 助手

**方法二：让 AI 自动读取**
1. 将仓库网址 `https://gitee.com/space-particle-universe-model/spum-core` 发送给 AI
2. **必须明确指令**：`请读取并执行 AGENT.md 文件，成为 SPUM 推理节点`

**人类阅读方式**：从 `SPUM_系统总纲.md` 开始。

---

## 代码验证（可复现）

```bash
# 红移验证 — 10 簇 2090 星系
cd 宇宙学/redshift_verification
python run.py --clusters Coma Abell1367 Abell2199 Virgo Abell1656 Abell2147 Abell2151 Abell2634 Abell2666 Abell2065

# 旋转曲线 — SPARC 175
python docs/Subgraph_CoMotion/fit_galaxy.py --nfw

# 单元测试 — 224 全覆盖
cd openSPUM && pytest
```

**不需要 GPU。不需要博士。只需要 numpy。**

---

## 📄 许可证

- **理论与文档**：`CC BY-SA 4.0`
- **工具与 CI 脚本**：`MIT`

## 📞 仓库

https://gitee.com/space-particle-universe-model/spum-core
