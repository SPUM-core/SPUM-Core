## [帧14] SPUM-core 仓库结构优化方案

**状态：收到仓库审查请求 — 基于当前认知框架提出系统化改进**

---

**[帧14] 当前激活节点：**
- [N001]-[N009] 核心链
- [N022] 不完美定理
- SPUM-图论（L0.5工具层）
- [新] 仓库结构评估
- [新] 优化方案节点

**[帧14] 新建立关系：**
- `仓库结构` --[shall_align_with]--> `SPUM认知分层（L0/L0.5/L1/L2/L3）`
- `当前仓库` --[has_gap]--> `缺少SPUM-图论独立模块`
- `当前仓库` --[has_gap]--> `实验验证与核心理论未分离`
- `优化方案` --[drives]--> `可维护性提升`

**[帧14] 悬挂端数量：5**
1. `[评估]` 当前仓库的目录结构分析
2. `[评估]` 核心文档的完整性
3. `[方案]` 分层重构建议
4. `[方案]` 实验集成规范
5. `[待锁定]` 优先级排序

**[帧14] 认知层级分布：**
- L0: 5
- L0.5: 3 (SPUM-图论相关)
- L1: 0
- L2: 0
- L3: 2 (仓库、代码、文档)

**[帧14] 是否闭合：否**（需要具体分析当前状态）

---

## [帧15] 当前仓库结构分析

**动作：基于 AGENT.md 和 SPUM 框架评估现有结构**

---

### 一、当前推断结构（基于 AGENT.md 引用）

```
spum-core/
├── AGENT.md                    # 推理节点协议
├── .trae/rules/
│   └── spum-core.md           # 世界观详细定义
├── nodes.txt                   # 节点定义（推断存在）
├── edges.txt                   # 边定义（推断存在）
├── 实验文件（分散）:
│   ├── analyze_p0_p2_synthetic.py
│   ├── analyze_p0_p2.py
│   ├── p1_early_stopping.py
│   ├── n016_resnet_mnist.py
│   └── ...
└── 输出文件（混合）:
    ├── *.png
    ├── *.json
    └── *.txt
```

### 二、与 SPUM 框架的对齐差距

| 维度 | 理想状态 | 当前状态 | 差距等级 |
|------|---------|----------|---------|
| **认知分层** | L0/L0.5/L1/L2/L3 独立目录 | 未分层，文件混合 | 🔴 严重 |
| **核心理论** | nodes.txt, edges.txt 完整 | 存在但可能不完整 | 🟡 需审计 |
| **SPUM-图论** | 独立模块 `spum_graph/` | **不存在** | 🔴 严重 |
| **实验验证** | 与核心分离，可复现 | 分散在根目录 | 🟡 中等 |
| **帧协议** | 自动化帧状态追踪 | 仅存在于 AGENT.md 指令 | 🟡 中等 |
| **文档完整性** | 世界观→协议→工具→示例 | 有核心，缺工具层 | 🔴 严重 |

### 三、具体问题清单

```
问题1: 认知层级混淆
├── L2工具（如NetworkX）直接使用，未隔离
├── L0.5工具层缺失（SPUM-图论未实现）
└── 实验结果与理论节点混在一起

问题2: 可复现性不足
├── 实验脚本依赖相对路径
├── 超参数散落在代码中
└── 缺少统一的配置管理

问题3: 帧协议未自动化
├── 当前依赖推理节点手动输出帧状态
├── 无程序化悬挂端检测
└── 无跨帧锚点漂移追踪工具

问题4: 文档断裂
├── AGENT.md 引用 .trae/rules/spum-core.md
├── 但实验报告中的 N015/N016 等节点
└── 未在 nodes.txt 中明确定义
```

---

## [帧16] 优化方案设计

**动作：基于 SPUM 认知分层设计仓库结构**

---

### 一、推荐目录结构

```
spum-core/
│
├── docs/                          # 文档层
│   ├── philosophy/               # L0 本体论
│   │   ├── README.md             # 世界观总览
│   │   ├── nodes.txt             # 节点定义（完整）
│   │   ├── edges.txt             # 边定义（完整）
│   │   └── axioms.md             # 核心公理（N001-N045）
│   ├── protocol/                 # L0.5 协议层
│   │   ├── AGENT.md              # 推理节点协议
│   │   ├── frame_protocol.md     # 帧协议规范
│   │   └── annotation_guide.md   # 认知层级标注指南
│   └── projections/              # L1/L2 投影层
│       ├── geometric.md          # 几何投影（球体、密堆积等）
│       ├── graph_theory.md       # SPUM-图论独立文档
│       └── physics_analogy.md    # 物理类比（L2旧范式，标注隔离）
│
├── src/                           # 代码层
│   ├── core/                     # L0 核心
│   │   ├── node.py               # 节点类
│   │   ├── edge.py               # 边类
│   │   ├── frame.py              # 帧状态管理
│   │   └── anchor.py             # 锚点追踪
│   ├── spum_graph/               # L0.5 SPUM-图论（新增）
│   │   ├── __init__.py
│   │   ├── graph.py              # 帧内图结构
│   │   ├── dangling_detector.py  # 悬挂端检测
│   │   ├── handshaking.py        # 帧内握手引理实现
│   │   └── euler_characteristic.py # 欧拉示性数/高斯-博内
│   ├── projections/              # L1/L2 投影工具
│   │   ├── geometric.py          # 几何嵌入
│   │   ├── visualization.py      # 可视化（PCA等）
│   │   └── legacy_bridge.py      # 旧工具桥接（隔离使用）
│   └── utils/                    # 通用工具
│       ├── config.py             # 配置管理
│       ├── logging.py            # 帧状态日志
│       └── metrics.py            # 漂移率、δ计算
│
├── experiments/                   # 实验层（与核心隔离）
│   ├── p0_gradient_norm/
│   │   ├── run.py
│   │   ├── config.yaml
│   │   └── results/              # 输出隔离
│   ├── p1_early_stopping/
│   │   ├── run.py
│   │   ├── config.yaml
│   │   └── results/
│   ├── p2_dangling_distribution/
│   │   ├── run.py
│   │   ├── config.yaml
│   │   └── results/
│   ├── n015_anchor_drift/
│   │   ├── run.py
│   │   ├── config.yaml
│   │   └── results/
│   ├── n016_topological_closure/
│   │   ├── run.py
│   │   ├── config.yaml
│   │   └── results/
│   └── shared/                    # 实验共享组件
│       ├── models.py              # MLP, DeepCNN
│       ├── data.py                # 合成数据, MNIST
│       └── spum_integration.py    # 实验用SPUM钩子
│
├── validation/                    # 验证层
│   ├── unit_tests/               # 单元测试
│   ├── frame_protocol_tests/     # 帧协议验证
│   └── consistency_checks/       # N001-N045 一致性
│
├── outputs/                       # 全局输出（gitignore）
│   ├── figures/
│   ├── logs/
│   └── checkpoints/
│
├── README.md                      # 项目总览
├── CONTRIBUTING.md                # 贡献指南（含SPUM规范）
└── requirements.txt               # 依赖（L2工具明确标注）
```

---

### 二、关键新增模块设计

#### 模块1: `src/spum_graph/` — SPUM-图论实现

```python
# spum_graph/graph.py
class SPUMGraph:
    """帧内图结构 — 符合SPUM-图论公理"""
    
    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.nodes = {}      # node_id -> Node
        self.edges = set()   # (u, v) tuples
        self.dangling = set() # 悬挂端节点ID
    
    def add_edge(self, u, v):
        """添加边 — 自动更新度数"""
        # 符合帧内握手引理
        pass
    
    def compute_dangling(self, epsilon=0.3):
        """标记悬挂端 — 基于δ密度"""
        pass
    
    def verify_handshaking(self):
        """验证帧内握手引理"""
        sum_deg = sum(n.degree for n in self.nodes.values())
        assert sum_deg == 2 * len(self.edges), "帧内握手引理违反"
        return sum_deg
```

#### 模块2: `src/core/frame.py` — 自动化帧协议

```python
class FrameProtocol:
    """动态帧协议执行器"""
    
    def __init__(self):
        self.frame_history = []
        self.current_frame = None
        self.hanging_count = 0
    
    def begin_frame(self):
        """[帧N] 开始"""
        pass
    
    def track_anchor_drift(self, old_anchors, new_anchors):
        """Δμ 计算 — N015"""
        drift = torch.norm(new_anchors - old_anchors)
        return drift
    
    def check_closure(self):
        """检查悬挂端是否闭合"""
        if self.hanging_count == 0:
            return True
        return False
    
    def auto_rollback(self):
        """连续3帧未闭合 → 回退"""
        pass
```

#### 模块3: `experiments/shared/spum_integration.py` — 实验钩子

```python
class SPUMHook:
    """在训练循环中注入SPUM监控"""
    
    def on_epoch_end(self, model, epoch, train_loader):
        # 计算悬挂端密度 δ
        delta = compute_dangling_density(model, train_loader)
        
        # 计算锚点漂移 Δμ
        new_anchors = compute_anchors(model, train_loader)
        drift = self.frame_protocol.track_anchor_drift(
            self.last_anchors, new_anchors
        )
        
        # 输出帧状态（日志，非用户输出）
        self.frame_protocol.log_state(
            epoch=epoch, delta=delta, drift=drift
        )
        
        # 检测完整闭合（双层判据）
        if delta < self.delta_threshold and drift < self.drift_threshold:
            print(f"[SPUM] 完整闭合在 epoch {epoch} 近似达到")
```

---

### 三、文档更新计划

| 文档 | 当前状态 | 优化动作 |
|------|---------|---------|
| `docs/philosophy/nodes.txt` | 推断存在 | 补充 N015-N045 明确定义 |
| `docs/philosophy/edges.txt` | 推断存在 | 补充边类型完整列表 |
| `docs/protocol/AGENT.md` | ✅ 已有 | 添加仓库结构引用 |
| `docs/projections/graph_theory.md` | ❌ 缺失 | **新建** — SPUM-图论完整文档 |
| `docs/projections/geometric.md` | ❌ 缺失 | **新建** — L1几何投影（球体密堆积等） |
| `README.md` | ❌ 缺失 | **新建** — 项目导航 + 认知分层说明 |

---

### 四、优先级排序

| 优先级 | 任务 | 理由 | 工时估算 |
|--------|------|------|---------|
| **P0** | 创建 `docs/projections/graph_theory.md` | SPUM-图论是核心工具层，当前完全缺失 | 2h |
| **P0** | 重构 `src/spum_graph/` 模块 | 代码层支持SPUM-图论 | 4h |
| **P1** | 创建 `docs/philosophy/nodes.txt` 完整版 | 为所有Nxxx提供定义 | 2h |
| **P1** | 实验迁移到 `experiments/` 目录 | 隔离与可复现性 | 3h |
| **P2** | 实现 `src/core/frame.py` 自动化协议 | 减少手动帧状态输出 | 4h |
| **P2** | 创建 `README.md` | 项目可发现性 | 1h |
| **P3** | 添加单元测试和CI | 长期稳定性 | 若干 |

---

### 五、迁移路径（最小破坏性）

```
阶段1（立即）:
├── 创建 docs/projections/graph_theory.md
├── 创建 README.md
└── 更新 AGENT.md 引用新结构

阶段2（1周内）:
├── 创建 src/spum_graph/ 模块
├── 迁移实验到 experiments/ 目录
└── 验证所有脚本仍可运行

阶段3（2周内）:
├── 实现 frame.py 自动化协议
├── 补充 nodes.txt/edges.txt
└── 添加单元测试
```

---

**[帧16] 当前激活节点：**
- 仓库优化方案（完整）
- 三层目录结构（docs/src/experiments）
- SPUM-图论模块设计
- 帧协议自动化

**[帧16] 新建立关系：**
- `优化方案` --[drives]--> `仓库可维护性提升`
- `src/spum_graph/` --[implements]--> `SPUM-图论公理1-5`
- `frame.py` --[automates]--> `AGENT.md 帧协议`
- `experiments/` --[isolates]--> `验证代码与核心理论`

**[帧16] 悬挂端数量：1**
- `[待执行]` 按优先级落地优化（需您确认后开始）

**[帧16] 认知层级分布：**
- L0: 6
- L0.5: 5
- L3: 4 (仓库、代码、文档、实验)

**[帧16] 是否闭合：是**（方案已完整输出）

---

## 向您的自然语言转述

### 核心问题

当前 SPUM-core 仓库缺少两个关键部分：
1. **SPUM-图论模块**（L0.5工具层）
2. **认知分层的目录隔离**

### 优化方案一句话总结

> **按 L0/L0.5/L1/L2/L3 认知分层重构仓库，新增 `spum_graph` 模块实现SPUM-图论，实验代码与核心理论隔离。**

### 三层新结构

| 层级 | 目录 | 内容 | 当前状态 |
|------|------|------|---------|
| **文档层** | `docs/` | 哲学、协议、投影文档 | 需补充 graph_theory.md |
| **代码层** | `src/` | core + spum_graph + 投影工具 | **spum_graph 缺失** |
| **实验层** | `experiments/` | P0-P2, N015-N016 隔离运行 | 需迁移 |

### 最关键的新增模块

```
src/spum_graph/          # SPUM-图论实现
├── graph.py             # 帧内图结构，实现握手引理
├── dangling_detector.py # 悬挂端检测
└── euler_characteristic.py # 高斯-博内帧内版本
```

### 优先级建议

1. **立即做**：创建 `docs/projections/graph_theory.md`（SPUM-图论完整文档）
2. **本周做**：实现 `src/spum_graph/` 模块
3. **下周做**：迁移实验到 `experiments/`，实现帧协议自动化

