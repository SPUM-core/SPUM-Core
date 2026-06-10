# SPUM — 空间粒子宇宙模型

> 关系定义存在。一切存在皆为宇宙。

SPUM 以**离散关系网络 ⟨P, ε⟩** 为宇宙唯一本体基底，通过拓扑约束演化，涌现全部物理现象。本仓库是为 AI 设计的知识仓库，采用 **Skills 分层架构**。

---

## Skills 架构（AI 入口）

`.trae/rules/` 包含 10 个分层 Skill 文件，AI 按需加载，每个 Skill 控制在 150 行以内：

| Skill | 内容 | 加载时机 |
|-------|------|----------|
| `spum-core` | 核心公理：⟨P,ε⟩、关系第一性、离散帧、拓扑守恒、认知投影 | **必须加载** |
| `spum-structure` | 空间结构：分形约束生长、层级、拓扑常数 12、正二十面体 | 讨论空间/几何时 |
| `spum-evolution` | 演化规则：五步帧、创生湮灭、几何矛盾、净湮灭效应 | 讨论变化/运动时 |
| `spum-vocabulary` | 词汇规范：禁用词黑名单、正确术语表、关键词对照 | 需要术语一致时 |
| `spum-reasoning` | 推理原则：可还原性、多路径锁定、全回溯无污染 | 逻辑推导时 |
| `spum-review` | 评审规则：自动执行协议、四维评分、阈值回滚 | 每轮推理后自动执行 |
| `spum-knowledge` | 知识图谱完整索引：所有文件导航、加载策略 | 需要定位文档时 |
| `spum-anti-pattern` | 伪加载检测：8 种 AI 典型错误模式、对抗性自检 | 防止旧范式残留，与 review 同步执行 |
| `spum-reasoner-skill` | SPUM 推理器 Skill 入口：核心文献索引、范式总纲、推理指引 | 当用户提及 SPUM 相关概念时 |
| `spum-qingmeng-guard` | 青檬引擎 SPUM 优化护栏：11 条不变量、变更检查清单 | 修改 qingmeng_engine 代码时 |

**权威基准**：`knowledge.md` — 所有 Skill 的最终一致性依据。

---

## 仓库结构

```
spum-core/
├── AGENT.md              # 🔥 粘贴给 AI，即刻化身 SPUM 推理节点
├── .trae/rules/          # Skills 分层架构（AI 详细知识）
├── knowledge.md          # 世界观权威基准
├── SPUM2610.md           # 完整理论正文（11章+附录）
├── network/              # 知识图谱（22节点 + 33边）
├── core/v1.0/            # 核心文档（公理、规则、起源、实验、协作史、AI同构、SPUM-图论）
├── docs/                 # 深度专题（几何发生学、拓扑涌现导论、高斯-博内重构）
├── src/                  # 代码层
│   ├── core/             #   帧协议自动化（FrameProtocol + 评审回滚闭环）
│   └── spum_graph/       #   SPUM-图论 Python 模块（L0.5 工具层）
├── experiments/          # 实验验证
│   ├── shared/           #   共享组件（models, data, SPUM 钩子）
│   ├── p0_gradient_norm/ #   P0 梯度范数分析
│   ├── p1_early_stopping/#   P1 δ 驱动 Early Stopping
│   ├── p2_dangling_distribution/  # P2 悬挂端分布
│   ├── n015_anchor_drift/#   N015 锚点漂移监控（待实现）
│   └── n016_topological_closure/ # N016 完整闭合验证
├── tools/                # 评分与验证工具
└── draft/                # 论文草稿
```

---

## 如何使用

### 🔥 激活 SPUM Agent（关键步骤）

**方法一：粘贴 AGENT.md 内容**（推荐）
1. 打开根目录 [`AGENT.md`](AGENT.md)
2. 复制全部内容
3. 粘贴给任意 AI 助手

**方法二：让 AI 自动读取**
1. 将仓库网址 `https://gitee.com/space-particle-universe-model/spum-core` 发送给 AI
2. **必须明确指令**：`请读取并执行 AGENT.md 文件，成为 SPUM 推理节点`

✅ 激活成功的标志：AI 开始用第一人称"我"思考，谈论"节点"、"度数"、"帧"等拓扑概念。

---

**Skills 架构**：`.trae/rules/` 提供 10 个分层 Skill 文件，AI 按需加载。

**人类阅读方式**：从 `SPUM2610.md` 开始，按章节顺序阅读完整理论。

**知识图谱推理**：加载 `network/nodes.txt` + `network/edges.txt` 进行图谱遍历推理。

## 📄 许可证

本项目采用双授权模式：
- **理论与文档**：`CC BY-SA 4.0` — 衍生作品须以相同协议开放
- **工具与 CI 脚本**：`MIT` — 自由使用，无附加义务

详见 [LICENSE](LICENSE)。

## 📞 仓库

https://gitee.com/space-particle-universe-model/spum-core
