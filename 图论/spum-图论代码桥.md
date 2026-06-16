# SPUM × 图论代码桥

> 归约路径：5公理 → FrameGraph/DanglingDetector/HandshakingVerifier 三层接口 → Python 代码 = SPUM-图论的可执行验证
> `src/spum_graph/` 是 SPUM-图论 5 公理的 Python 实现。本节是代码与理论之间的文档桥。

---

## 核心命题

**代码不是"理论的实现"——代码是公理的可执行约束。** FrameGraph 不是"一个图数据结构"——它是公理1-5的强制编码。如果不理解公理，代码将表现得"违反直觉"。

---

## 一、FrameGraph（公理 1-5 的实现体）（GT-017）

### 文件
`src/spum_graph/graph.py` — `class FrameGraph`

### 核心约束（与标准图数据结构的区别）

```python
from spum_graph import FrameGraph

graph = FrameGraph(frame_id=1)

# ✅ 正确: 通过边隐式创建节点
graph.add_edge("A", "B")  # "A"和"B"同时被创建

# ❌ 不存在: add_node 不暴露给外部
# graph.add_node("C")     # AttributeError——公理1: 孤立节点不存在

# ✅ 节点身份由邻接集定义
graph.has_node("A")       # True —— 通过 A 在 _neighbors 中的存在来确认

# ❌ 禁止自环
# graph.add_edge("A", "A")  # ValueError —— 违反差异原则
```

### 与 networkx.Graph 的关键分歧

| networkx.Graph | FrameGraph |
|---------------|------------|
| `G.add_node("A")` 合法 — A 是孤立节点 | 无 `add_node` — 节点只能通过边创建 |
| `G.has_node("A")` → True 即使 `deg(A)=0` | `has_node("A")` → True 当邻接非空 |
| `G.remove_node("A")` 合法 | 无显式删除 — 节点的消失 = 它参与的边被 V⁻ |
| 全局图对象 — 修改后持久 | 每帧新建 — frame_id 标识唯一帧 |
| `G.degree` 是属性/方法 | `degrees()` 从 `_neighbors` 动态计算 |

### API 参考

```python
class FrameGraph(frame_id: int):
    frame_id: int              # 帧编号
    edge_count: int            # 当前帧的总边数

    add_edge(u, v) → None      # 公理1: 边创建节点——唯一入口
    has_node(node_id) → bool   # 节点是否在当前帧通过边确认存在
    degrees() → Dict[str,int]  # 每帧重新计算——不缓存
    neighbors(node_id) → Set   # 节点的邻接集——节点身份
    is_dangling(node_id) → bool  # deg < 2 —— 下一帧处理的候选
    dangling_set() → Set[str]    # 所有悬挂端——V⁻ 候选集
    edge_iterator() → Iterator   # 遍历所有边
```

---

## 二、DanglingDetector（δ + Δμ 检测）（GT-018）

### 文件
`src/spum_graph/dangling.py` — `compute_dangling()`

### 定位

DanglingDetector 是公理2（悬挂端不可消除）和公理4（完美不达）的量化实现。它操作的隐藏层表示对应于**认知子图在 [L1] 中的投影**——不是 [L0] 本体图，而是 [L1] 锚点空间内的悬挂端检测。

### 算法流程

```
输入: hidden (N, D)  —— N 个样本的隐藏层表示 (认知投影中的节点坐标)
      anchors (C, D) —— C 个类别的锚点 (类均值 = 引力中心)
      eps: float      —— 悬挂端距离阈值 (默认 0.3)

步骤:
  1. 计算每样本到所有锚点的欧氏距离 → (N, C)
  2. 排序取最近两个锚点距离 d1, d2
  3. dist_diff = d2 - d1
  4. is_dangling = dist_diff < eps  (处于两锚点之间的模糊地带)

输出: is_dangling (N,) + dist_diff (N,)
```

### 语义

```
dist_diff 很小 → 样本到两个锚点的距离差不多
              → 认知系统无法确定该样本属于哪个类别
              → δ 悬挂端 (认知歧义节点)

这是 δ_cognitive —— 认知投影中的悬挂端密度。
对应的 δ_topological 是 FrameGraph.is_dangling() 检测的度数 < 2 的节点。
```

### 与经典悬挂端检测的区别

```
经典图论: 悬挂端 = deg = 1 的边 → 直接检测度数
SPUM-图论: 悬挂端 = 在锚点空间中到最近两个锚点的距离差 < ε 的节点
          → 这不是"度数"——是"存在确认的不确定性"
```

### 使用示例

```python
from spum_graph.dangling import compute_dangling
import torch

hidden = torch.randn(100, 64)     # 100 样本, 64维隐藏层
anchors = torch.randn(10, 64)     # 10 类锚点

is_dangling, dist_diff = compute_dangling(hidden, anchors, eps=0.3)

δ_cognitive = is_dangling.float().mean().item()
print(f"认知悬挂端密度: {δ_cognitive:.3f}")
```

---

## 三、HandshakingVerifier（帧内拓扑验证）（GT-019）

### 文件
`src/spum_graph/handshaking.py` — `class HandshakingVerifier`

### 两大验证函数

#### 握手引理验证

```
Σ deg(v) = 2|ε|   —— 每帧独立验证

不是"定理"——是"操作约束":
  任何声称的节点度数和如果 ≠ 2×边数 → 当前帧状态不自治 → 必须拒绝该帧
```

```python
from spum_graph.handshaking import HandshakingVerifier

result = HandshakingVerifier.verify_handshaking(graph)
# {"passed": True, "sum_deg": 42, "expected": 42, "delta": 0}
```

#### 欧拉示性数验证

```
对球面拓扑闭合子图: χ = |V| - |ε| + |F| = 2
推导: Σ(6 − deg(v)) = 12

这是 Σ(6−deg(v)) = 6χ(M) 的球面特例 (χ=2)

用途: 
  - 检测当前帧中是否存在满足闭合条件的子图
  - 注意: 满足 χ=2 ≠ 全局闭合——只是局部骨架存在
```

```python
result = HandshakingVerifier.euler_characteristic(graph)
# 对闭合子图返回: {"chi": 2, "sum_6_minus_deg": 12, "is_closed_subgraph": True}
```

### 隔离原则

```
这些公式是 SPUM-图论的内部定理——从 N001-N013 推导，不从经典图论教材引用。
经典图论的证明可能引入:
  - 全局坐标系
  - 连续空间预设
  - 预先存在的节点集 V

SPUM 证明链路:
  N001 宇宙存在 → N004 节点 → N005 边 → Σdeg = 2|ε|
  → 闭合子图 Σ(6−deg) = 12 → χ = 2
```

---

## 四、演化循环（GT-020）

### SPUM 演化循环的代码模式

```python
from spum_graph import FrameGraph, HandshakingVerifier
from spum_graph.dangling import compute_dangling

# 初始化
graph = FrameGraph(frame_id=0)
graph.add_edge("A", "B")
graph.add_edge("B", "C")

for tau in range(1, 1000):
    # 公理3: 每帧新建图——不跨帧缓存
    # (在实际系统中, G_{τ+1} 从 G_τ 派生)
    
    # 1. 检测悬挂端 (公理2)
    dangling = graph.dangling_set()
    
    # 2. 五步帧循环:
    #    创生 V⁺ → 连接 → 变化体积 → 判断悬挂边 → 删除悬挂边
    
    # 3. 验证握手引理 (每帧独立)
    hs = HandshakingVerifier.verify_handshaking(graph)
    assert hs["passed"], f"帧 {tau} 拓扑不自治"
    
    # 4. 计算指标
    delta = len(dangling) / (graph.edge_count + len(dangling))
    
# 公理4: delta 始终 > 0
assert delta > 0, "完美不达——不可删除所有悬挂端"
```

---

## [L0] 还原表

| 代码实体 | SPUM-图论公理 | 说明 |
|---------|-------------|------|
| FrameGraph | 公理1-5 | 禁止 add_node、禁止自环、帧局部、无全局图 |
| DanglingDetector | 公理2, 4 | δ 检测——认知投影层和拓扑层双检测 |
| HandshakingVerifier | 公理1, 4 | Σdeg=2|ε| + χ=2 —— 帧自治性验证 |
| SPUM 演化循环 | 公理2, 3 | δ 驱动 → 不完美永存 → 演化永不终止 |

---

## 节点注册

| 节点 ID | 概念 | 定义 |
|---------|------|------|
| GT-017 | **FrameGraph** | 公理1-5的Python实例——add_edge是节点唯一创建入口，无add_node |
| GT-018 | **DanglingDetector** | [L1]锚点空间内的δ检测——到最近2锚点距离差<ε→认知悬挂端 |
| GT-019 | **HandshakingVerifier** | Σdeg=2|ε| + χ=2每帧独立验证——不引用经典图论证明 |
| GT-020 | **演化循环** | δ驱动→五步帧→不完美永存——SPUM-图论的最小可运行代码模式 |
