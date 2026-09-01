# 青檬引擎（Qingmeng Engine）

> **将复杂系统推演，转化为可计算的结构涌现。**

基于 **SPUM**（⟨P, ε⟩ 离散关系网络）内化的**复杂系统推理引擎**：把哲学/物理假设翻译为可执行原语，把共识要求转化为约束校验器，把跨域知识转化为领域适配层。

---

## 定位

| 维度 | 定位描述 |
|:---|:---|
| **产品定义** | 基于结构化公理与拓扑涌现机制的复杂系统推理引擎 |
| **目标用户** | AI 架构师、多智能体系统开发者、知识图谱工程师、复杂系统研究者 |
| **核心价值** | 提供超越传统 LLM Prompt 工程的**可验证、可追踪、受约束的结构化推理层** |
| **差异化** | 传统推理依赖概率或静态规则；青檬以**拓扑守恒为状态演化约束**、以关系网络为**涌现载体**，输出可审计推导链 |
| **Slogan** | `将复杂系统推演，转化为可计算的结构涌现` |

> **定位关键**：不卖"物理理论"，卖"受控推理能力"。SPUM 是默认公理集；架构支持替换为其他公理体系（替换 `core/axioms.py` 即可）。

---

## 架构

四层解耦，理论内核与工程外壳分离：

```
qingmeng/
├── qingmeng/
│   ├── core/           # 公理内核层（理论不可变）
│   │   ├── axioms.py     # 12 晶子原语 · Σ(6−deg)=12 · σ · dv/dt ≤ const · 级联消解
│   │   ├── state.py      # 五步帧状态机（创生→连接→变化体积→判断悬挂→删除）
│   │   ├── consensus.py  # 11 条不变量校验器（Schema 驱动，所有推理路径的强制关卡）
│   │   └── consensus_schema.json  # 共识规则表（id/scope/check/params）
│   ├── graph/          # 拓扑推理层（帧快照只读分析）
│   │   ├── topology.py   # 路径/连通性/指标/子图
│   │   ├── emergence.py  # 五形相位检测（S = 水木土金火）
│   │   └── trajectory.py # 推理轨迹快照 + 版本回滚（观测性：trace_id/axiom_path）
│   ├── domain/         # 领域适配层（理论可变，插件化）
│   │   ├── registry.py   # 适配器注册表（physics/sociology/…）
│   │   └── boundary.py   # 边界效应（领域耦合强度 + 帧间衰减）
│   └── api/            # 接口层
│       ├── tactile.py    # 触觉桥（输入信号 → 拓扑触碰）
│       ├── llm_bridge.py # LLM 推理桥（L2 帧推理循环：LLM 提议 → 引擎执行 → 共识校验）
│       ├── rest.py       # REST API（FastAPI 外壳：/health /state /evolve /touch /reason /debug/trace）
│       └── cli.py        # 调试 CLI
├── examples/           # 高价值场景示例（供应链风险传导）
└── tests/              # 公理/演化/共识/触觉/推理/REST/轨迹/边界测试
```

**分层原则：**
- `core` 层硬编码 SPUM 公理，**不可动态修改**（共识底线）
- `graph` 层只读分析，不修改图（演化只属于 `core.state`）
- `domain` 层插件化，挂载自定义解释器不污染内核
- 所有推理路径经过 `consensus` 校验，违规抛 `ConsensusViolationError`

---

## 快速开始

```bash
pip install -e .            # 安装（networkx 依赖）
pip install -e ".[api]"     # 安装 REST 依赖（fastapi/uvicorn）
python -m qingmeng.api.cli --frames 10 --check   # CLI 自检
python -m qingmeng.api.cli --reason "水为什么往低处流"  # L2 帧推理循环
python -m qingmeng.api.rest --port 8000          # 启动 REST API
python examples/supply_chain_risk.py             # 供应链风险传导示例
python -m pytest tests -q   # 运行测试（72 项）
```

REST API（FastAPI，`qingmeng/api/rest.py`）：

```bash
curl localhost:8000/health                     # 存活探针
curl localhost:8000/state                      # 帧快照 + 五形 + 对偶账本 + 共识
curl -X POST localhost:8000/evolve -H "Content-Type: application/json" -d '{"frames": 10}'
curl -X POST localhost:8000/touch  -H "Content-Type: application/json" -d '{"text": "我有点紧张"}'
curl -X POST localhost:8000/reason -H "Content-Type: application/json" \
     -d '{"prompt": "引力驱动行星运动", "backend": "http", "preset": "deepseek"}'
curl localhost:8000/debug/traces                # 最近推理轨迹（观测性）
curl localhost:8000/debug/trace/{trace_id}      # 完整轨迹审计
curl -X POST localhost:8000/debug/rollback/{trace_id}   # 版本回滚
```

端点返回 `结构数据 + consensus 块`——每响应都带 11 条不变量实时校验结果；违规不抛 HTTP 错误，在 payload 报告（不完美是常态）。OpenAPI 文档在 `http://localhost:8000/docs`。

Python API：

```python
from qingmeng import QingmengEngine

eng = QingmengEngine()

# 触觉桥：文本输入 → 拓扑触碰（无权重边，强度决定接触密度）
feeling = eng.tactile.react_to("告诉你一个秘密，我有点紧张。")
print(feeling["feeling_text"])      # → "帧0。触感: … 体内14节点/33边…"

# L2 帧推理循环：LLM 提议 → 引擎执行 → 共识校验（辅助推理器）
trace = eng.reason("引力驱动行星运动")
print(trace.summary_text)           # → 帧轨迹 + 图内结论（连通性/σ 支持）
print(trace.to_dict())              # → 每帧五步序列/悬挂端/闭合/共识

# 五步帧演化（+ 创生-湮灭对偶记账）
snap = eng.evolve_many(10)
print(snap.to_dict())

# 共识校验：违规抛 ConsensusViolationError
eng.check(raise_on_fail=True)

# 五形相位与领域解释
print(eng.emergence())            # → S=(水,木,土,金,火)
print(eng.interpret("physics"))   # → σ → 温度语义

# 边界效应：领域/节点集间耦合强度（只读度量）
coupling = eng.coupling([0, 1], [8, 9])      # → [0,1]，拓扑距离 × 密度
print(coupling)

# 观测性：每条 evolve/touch/reason 自动记录推理轨迹
trace_id = eng.trajectory.recent(1)[0].trace_id   # trace_id + 8 项 axiom_path
eng.trajectory.rollback(trace_id, eng)            # 版本回滚到该轨迹发生前
```

---

## 真实 LLM 接入（DeepSeek / OpenAI 兼容）

默认 `DeterministicBackend` 离线可跑；接入真实 LLM 只需配置密钥，走 OpenAI 兼容 HTTP 后端：

```bash
# 1. 配置密钥（写入 qingmeng/.env，已被 .gitignore 覆盖，勿提交）
cat > .env <<'EOF'
QINGMENG_LLM_API_KEY=sk-你的Key
QINGMENG_LLM_BASE_URL=https://api.deepseek.com   # 预设：openai → https://api.openai.com/v1
QINGMENG_LLM_MODEL=deepseek-chat                 # 预设：openai → gpt-4o-mini
EOF

# 2. 用真实 LLM 跑 L2 帧推理循环
python -m qingmeng.api.cli --reason "经济周期与市场波动的关系" --backend http --preset deepseek
```

代码等价写法：

```python
from qingmeng import QingmengEngine
from qingmeng.api.llm_bridge import HttpLLMBackend

eng = QingmengEngine()
eng.reasoner.backend = HttpLLMBackend(preset="deepseek")   # 或 openai / 自定义 base_url+model
trace = eng.reason("引力驱动行星运动")
```

设计要点：
- **失败自动回退**——网络/超时/非法 JSON 时自动回退确定性后端，引擎永不被 LLM 阻塞
- **schema 强制**——系统提示要求只输出推理帧 JSON；`_extract_json` 提取首个平衡 JSON 对象
- **结论由图裁决**——LLM 帧里的主张经 V⁺/V⁻ 落到图上，连通性/σ 才构成"图内结论"
- **实测行为**（2026-08-31，DeepSeek deepseek-chat）：LLM 能合理产出 claim/connect/question 帧，循环以"栈溢出保护"或"已收束"结束均为正常——闭合不可达即不完美定理的体现

---

## 核心概念（与连续统范式的分野）

| SPUM 概念 | 青檬实现 | 注意（勿退回旧范式） |
|:---|:---|:---|
| **12 晶子** | `CORE_SIZE = 12`，正二十面体 30 边 | 12 是**拓扑常数**（Σ(6−deg)=12 的欧拉强制解），**不是 12 种类型** |
| **dv/dt ≤ const** | `Core.dv_dt_check()` | 单帧节点关系变化的**容量上限**（离散帧语义），**不是连续平滑约束** |
| **σ = \|P\|/\|ε\|** | `Core.sigma()` | 高 σ 稀疏（热），低 σ 饱和（冷） |
| **五步帧** | `StateEvolutionEngine.evolve()` | 一帧只删一次，残余留给下一帧；帧间无中间状态 |
| **创生-湮灭对偶** | `engine.state.ledger` | 每个 V⁺ 必是某 V⁻ 的对偶补偿，累计净漂移可审计 |
| **边是二值关系** | 图边无权重属性 | 无权、无向、不可再分（不变量 VI） |
| **触觉桥** | `eng.tactile.react_to(text)` | 文本强度只决定**创生边数**（接触密度），不决定权重；单次触碰单节点度数增量 ≤ 4（dv/dt 投影） |
| **LLM 推理桥** | `eng.reason(prompt)` | LLM 是**辅助推理器**：只产出 schema 约束的推理帧（V⁺/V⁻ 提议），引擎执行 + 共识校验；结论由图结构（连通性/σ）支持，不是"生成"的 |
| **观测性** | `eng.trajectory` | 每条推理路径自动记录 `trace_id` + `axiom_path`（逐条不变量裁决）+ 状态快照 + 共识状态；支持版本回滚（观测面不被回滚抹除） |
| **边界效应** | `eng.boundary.coupling()` | 领域/节点集耦合 = `1/(1+γ·d̄)` × 密度项；距离用图内最短路径（拓扑距离），衰减是帧间离散遗忘 |

---

## 共识契约（11 条不变量）

对齐 `.trae/rules/spum-qingmeng-guard.md`。校验分级：

| 分级 | 不变量 | 校验位置 |
|:---|:---|:---|
| structural | **I** ⟨P,ε⟩ 唯一基底 · **III** 无悬挂边 · **IV** 三角剖分 · **V** 拓扑常数 12 · **VI** 边二值 | `consensus.py` |
| ledger | **II** 总边数守恒（创生-湮灭对偶） | `consensus.py`（需传入账本） |
| contextual | **VII** 约束驱动演化（确定性，非随机） | `consensus.py` |
| epistemic | **VIII** 几何是投影 · **IX** π 是翻译因子 · **X** 连续是粗粒化 · **XI** 数学是工具 | 元层面约定 |

```python
from qingmeng.core import ConsensusValidator
report = ConsensusValidator().validate(G, ledger=eng.state.ledger)
report.passed        # False → 有违规
report.failures      # → [ConsensusFailure(invariant, required, detail), ...]

# Schema 驱动（蓝图避坑建议 3）：规则表在 core/consensus_schema.json
# 每条规则 = {id, name, scope, check, enabled, params}
ConsensusValidator.from_schema("my_rules.json")   # 外部规则表（params 传阈值）
```

---

## 与既有实现的关系

| 代码库 | 角色 |
|:---|:---|
| 本包 `qingmeng/` | 蓝图四层架构蓝本，逻辑同源、自包含、可独立演进 |
| `spum/` 包（SPUM API v4.0） | AI 之家推理 API（5 公理 + 帧引擎 + 跨域桥接） |
| `src/core/frame.py`（FrameProtocol） | 推理帧协议参照——L2 帧推理循环对齐其五步序列/悬挂端/闭合/栈溢出回退语义（自包含实现于 `api/llm_bridge.py`） |
| `e:/qingmeng_spum/qingmeng` | 青檬引擎渐进重构实现；`api/tactile.py` 移植自其 `instinct_bridge.py` v7（按不变量 VI 二值化） |

演进策略：以本包为**架构参照**，将既有实现的成熟逻辑（如触觉桥、工具执行器）逐步迁入分层结构。

---

## 路线图

| 里程碑 | 内容 | 状态 |
|:---|:---|:---|
| M1 内核抽象 | 公理原语 + 五步帧 + dv/dt 有界性测试 | ✅ 本骨架 |
| M2 图与涌现 | 拓扑分析 + 五形相位检测 + 推理轨迹 | ✅ 本骨架 |
| M3 共识引擎 | 11 条不变量校验器（Schema 驱动）+ 违规拦截 | ✅ 本骨架 |
| M4 接口生态 | 触觉桥 ✅ / LLM 推理桥（L2 帧推理循环）✅ / REST API ✅ / 边界效应 ✅ / examples/ ✅ | ✅ 完成 |
| M5 性能文档 | 压测报告 / SDK / 贡献指南 | 待做 |

工程避坑：
1. **不参数化公理**——不变量硬性裁决；领域层参数可调（`BoundaryConfig` 等），`CORE_SIZE=12` 不可调；共识规则表 Schema 化（阈值/开关可配置，校验函数不随配置漂移）
2. **理论推导与业务推理分离**——引擎只负责"输入 → 拓扑映射 → 状态演化 → 约束校验 → 输出轨迹"；业务逻辑放 `domain/`
3. **LLM 克制**——LLM 仅做自然语言 ↔ 结构化查询的桥，核心推演由 core/graph 计算
4. **观测性（生命线）**——每条推理自动输出 `trace_id` / `axiom_path` / `state_snapshots` / `consensus_status`，提供 `/debug/trace/{id}` 接口与版本回滚
