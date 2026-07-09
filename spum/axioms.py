"""
SPUM 公理引擎 — 5 条核心公理 + 派生常数 + 基本操作
=====================================================

所有 SPUM 推理的底层原语。每条公理以 callable 方法提供，
可直接被 AI 调用。不从任何旧文件导入——这是独立定义层。

公理体系:
    Axiom 1: 关系第一性 — 连接定义存在，无孤立节点
    Axiom 2: 悬挂端不可消除 — 对任意有限帧，悬挂端集合基数 > 0
    Axiom 3: 离散帧快照 — 图状态是帧序列的快照，无全局图
    Axiom 4: 拓扑守恒 — Σ(6-deg(v)) = 12, 角度亏损恒为 4π
    Axiom 5: 不完美 — 双层判据 (δ < θ_δ) ∧ (Δμ < θ_μ)

用法:
    from spum.axioms import Axioms
    ax = Axioms()
    ax.axiom1.add_edge("A", "B")
    ax.axiom4.angle_deficit_sum()  # → {"sum_6_minus_deg": ..., "degrees": {...}}
    ax.sigma()  # → |P|/|ε|
"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, Iterator, List, Any
from dataclasses import dataclass, field
import math


# ══════════════════════════════════════════════════════════════════════
# 派生常数
# ══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Constants:
    """SPUM 宇宙的派生常数。全部从公理推导，非预设。"""
    # 最小差异尺度 — 关系网络自校准形成的基本长度单位
    KAPPA: float = 1.0
    # 离散帧时间单位
    TAU: int = 1
    # 拓扑常数 12 — Σ(6-deg(v)) = 12 的强制解
    TOPOLOGICAL_12: int = 12
    # 晶子度数阈值 — 节点饱和度
    CRYSTALLITE_DEGREE_THRESHOLD: int = 50
    # 晶子等效直径临界值 D_max = 4κ
    CRYSTALLITE_DIAMETER_MAX: float = 4.0
    # 晶子几何容量上限 N_max = 16π ≈ 50.3
    CRYSTALLITE_CAPACITY: float = 16.0 * math.pi
    # 三维密堆最大接触数（接吻数）— 涌现结果
    MAX_CONTACTS: int = 12
    # 悬挂端密度默认阈值
    DANGLING_EPS: float = 0.3
    # 锚点漂移默认阈值
    DRIFT_THRESHOLD: float = 0.01
    # 光速 c = 4κ/τ
    C: float = 4.0
    # 单帧变化硬上限 dv/dt ≤ const
    DV_DT_MAX: float = 4.0 * 4.0  # D_max = 4κ


# ══════════════════════════════════════════════════════════════════════
# Axiom 1: 关系第一性
# ══════════════════════════════════════════════════════════════════════

class Axiom1:
    """公理1: 关系第一性 — 连接定义存在。

    核心命题:
        - 纯粹的关系活动是唯一本原
        - 节点由边隐式定义，无独立"创建节点"操作
        - 孤立节点无法被确认存在
        - 不是体积大的粒子承载更多连接，而是能维持更多
          稳定连接的节点在认知投影中必然呈现为更大体积

    SPUM 反转图论:
        旧范式: 节点先天存在，边是后天的连接
        SPUM: 关系从潜态→显态 → 撑出空间粒子（节点）
    """

    def add_edge(self, u: str, v: str, neighbors: Dict[str, Set[str]],
                 edge_count: List[int]) -> None:
        """添加无向边。节点由边隐式定义。

        Args:
            u: 节点 ID
            v: 节点 ID
            neighbors: 邻接表（副作用更新）
            edge_count: [count] 边数（副作用更新）
        """
        if u == v:
            raise ValueError(f"SPUM 不允许自环: {u}→{u}")
        neighbors[u].add(v)
        neighbors[v].add(u)
        edge_count[0] += 1

    def remove_edge(self, u: str, v: str, neighbors: Dict[str, Set[str]],
                    edge_count: List[int]) -> None:
        """删除无向边。删除后若节点无连接则自动消亡。

        Args:
            u: 节点 ID
            v: 节点 ID
            neighbors: 邻接表（副作用更新）
            edge_count: [count] 边数（副作用更新）
        """
        neighbors[u].discard(v)
        neighbors[v].discard(u)
        edge_count[0] -= 1
        # 清理孤立节点
        for n in (u, v):
            if n in neighbors and not neighbors[n]:
                del neighbors[n]

    def has_node(self, node_id: str, neighbors: Dict[str, Set[str]]) -> bool:
        """节点是否存在取决于它是否有连接。"""
        return node_id in neighbors

    def degree(self, node_id: str, neighbors: Dict[str, Set[str]]) -> int:
        """返回节点度数。未连接的节点视为不存在（返回 0）。"""
        return len(neighbors.get(node_id, set()))

    def nodes(self, neighbors: Dict[str, Set[str]]) -> Iterator[str]:
        """返回所有有连接的节点。"""
        return iter(neighbors)

    def describe(self) -> str:
        return (
            "公理1: 关系第一性 — 连接定义存在。"
            "关系从潜态→显态撑出空间粒子(节点)。"
            "无独立'创建节点'操作，节点由边隐式定义。"
        )


# ══════════════════════════════════════════════════════════════════════
# Axiom 2: 悬挂端不可消除
# ══════════════════════════════════════════════════════════════════════

class Axiom2:
    """公理2: 悬挂端不可消除（有限帧内）。

    核心命题:
        - 度数 < 2 的节点 = 悬挂端
        - 对任意有限帧，悬挂端集合基数 > 0
        - 删除悬挂端必然产生新的悬挂端（被删节点的邻居度数可能降至 1）
        - 悬挂端是正常状态，不是错误

    认知投影（L1）中的悬挂端定义:
        样本到最近两个锚点的距离差 < ε
        → 样本处于多锚点的模糊地带，语义位置不确定
    """

    def is_dangling(self, node_id: str, neighbors: Dict[str, Set[str]]) -> bool:
        """度数 < 2 的节点 = 悬挂端。"""
        return node_id in neighbors and len(neighbors[node_id]) < 2

    def dangling_nodes(self, neighbors: Dict[str, Set[str]]) -> Set[str]:
        """返回所有悬挂端节点。"""
        return {n for n, adj in neighbors.items() if len(adj) < 2}

    def dangling_density(self, neighbors: Dict[str, Set[str]]) -> float:
        """悬挂端密度 δ = |悬挂端| / |总节点|。

        Returns:
            float: 0.0 ~ 1.0。δ > 0 对任意有限帧。
        """
        if not neighbors:
            return 0.0
        return len(self.dangling_nodes(neighbors)) / len(neighbors)

    def compute_dangling_cognitive(
        self,
        hidden: 'torch.Tensor',
        anchors: 'torch.Tensor',
        eps: float = 0.3
    ) -> Tuple:
        """认知投影中的悬挂端计算（L1 层）。

        使用 PyTorch 计算样本到锚点的距离差。
        需要 torch 可用。

        Args:
            hidden: (N, D) 隐藏层表示
            anchors: (C, D) 各类锚点
            eps: 悬挂端距离差阈值

        Returns:
            (is_dangling_tensor, dist_diff_tensor)
        """
        import torch
        dists = torch.cdist(hidden.unsqueeze(1), anchors.unsqueeze(0)).squeeze(1)
        sorted_dists, _ = dists.sort(dim=1)
        dist_diff = sorted_dists[:, 1] - sorted_dists[:, 0]
        is_dangling = dist_diff < eps
        return is_dangling, dist_diff

    def describe(self) -> str:
        return (
            "公理2: 悬挂端不可消除(有限帧内)。"
            "度数<2=悬挂端，删除悬挂端必生新悬挂端。"
            "悬挂端是正常状态，不是错误。"
        )


# ══════════════════════════════════════════════════════════════════════
# Axiom 3: 离散帧快照
# ══════════════════════════════════════════════════════════════════════

class Axiom3:
    """公理3: 离散帧快照 — 图状态是帧序列的快照。

    核心命题:
        - 没有"时间本身"，只有"网络演化到第几步"
        - 帧与帧之间没有中间状态
        - 节点要么连接，要么不连接
        - 每帧独立，不跨帧缓存任何状态
        - dt = τ（离散帧），不是无穷小

    五步帧序列:
        1. 创生 (V⁺): 新边被创建
        2. 连接: 节点连接关系更新，度数重算
        3. 变化体积: 节点度数变化导致等效体积变化，σ 局部波动
        4. 判断: 检查悬挂端
        5. 删除 (V⁻): 删除所有悬挂边

    不变量:
        - 一帧内没有循环，没有级联消解
        - 一帧只做一次删除
        - 删除后新产生的悬挂边是下一帧处理的起点
    """

    # 五步名称
    STEP_NAMES = ["创生(V⁺)", "连接", "变化体积", "判断悬挂", "删除(V⁻)"]

    @dataclass
    class FrameSnapshot:
        """单帧状态快照。

        符合 Axiom 3: 每帧独立，不跨帧缓存状态。
        """
        frame_id: int
        node_count: int
        edge_count: int
        dangling_count: int
        dangling_density: float
        sigma: float
        step_1_done: bool = False
        step_2_done: bool = False
        step_3_done: bool = False
        step_4_done: bool = False
        step_5_done: bool = False
        metadata: Dict[str, Any] = field(default_factory=dict)

        @property
        def all_steps_done(self) -> bool:
            return all([self.step_1_done, self.step_2_done,
                        self.step_3_done, self.step_4_done, self.step_5_done])

        def to_dict(self) -> dict:
            return {
                "frame_id": self.frame_id,
                "node_count": self.node_count,
                "edge_count": self.edge_count,
                "dangling_count": self.dangling_count,
                "dangling_density": self.dangling_density,
                "sigma": self.sigma,
                "steps": {
                    "1_创生V+": self.step_1_done,
                    "2_连接": self.step_2_done,
                    "3_变化体积": self.step_3_done,
                    "4_判断悬挂": self.step_4_done,
                    "5_删除V-": self.step_5_done,
                },
                "all_steps_done": self.all_steps_done,
            }

    def snapshot(self, frame_id: int, neighbors: Dict[str, Set[str]],
                 edge_count: int) -> FrameSnapshot:
        """从当前图状态生成帧快照。"""
        n_nodes = len(neighbors)
        n_dangling = sum(1 for n in neighbors if len(neighbors[n]) < 2)
        return self.FrameSnapshot(
            frame_id=frame_id,
            node_count=n_nodes,
            edge_count=edge_count,
            dangling_count=n_dangling,
            dangling_density=n_dangling / max(n_nodes, 1),
            sigma=n_nodes / max(edge_count, 1),
        )

    def describe(self) -> str:
        return (
            "公理3: 离散帧快照 — 图状态是帧序列的快照。"
            "没有'时间本身'，只有'网络演化到第几步'。"
            "五步序列: 创生→连接→变化体积→判断悬挂→删除。"
            "一帧内无循环无级联。dt=τ不是无穷小。"
        )


# ══════════════════════════════════════════════════════════════════════
# Axiom 4: 拓扑守恒
# ══════════════════════════════════════════════════════════════════════

class Axiom4:
    """公理4: 拓扑守恒。

    核心命题（全部局部可检验）:
        1. 握手引理: Σ deg(v) = 2|ε|
        2. 欧拉示性数: χ = |V| - |E| + |F|, 对闭合球面 χ=2
        3. 离散高斯-博内: Σ(6-deg(v)) = 12 (球面特例)
           → Σδᵢ = 4π (角度亏损总和恒为 4π)
        4. 通用形式: Σ(6-deg(v)) = 6χ(M)
        5. 开口占比 ≤ 1/3
        6. 总边数 |ε| 全局严格守恒（理论外推）
    """

    def handshaking(self, neighbors: Dict[str, Set[str]],
                    edge_count: int) -> dict:
        """握手引理验证: Σ deg(v) = 2|ε|。

        Returns:
            {"passed": bool, "sum_deg": int, "expected": int, "delta": int}
        """
        sum_deg = sum(len(adj) for adj in neighbors.values())
        expected = 2 * edge_count
        return {
            "passed": sum_deg == expected,
            "sum_deg": sum_deg,
            "expected": expected,
            "delta": sum_deg - expected,
        }

    def euler_characteristic(self, vertex_count: int, edge_count: int,
                             face_count: Optional[int] = None) -> dict:
        """欧拉示性数: χ = |V| - |E| + |F|。

        Args:
            vertex_count: 节点数 |V|
            edge_count: 边数 |E|
            face_count: 面数 |F|，若不提供则返回 V-E

        Returns:
            {"v": int, "e": int, "f": int|None, "chi": int|None}
        """
        chi = None if face_count is None else vertex_count - edge_count + face_count
        return {"v": vertex_count, "e": edge_count, "f": face_count, "chi": chi}

    def angle_deficit_sum(self, neighbors: Dict[str, Set[str]]) -> dict:
        """计算 Σ(6-deg(v)) — 角度亏损总和。

        对闭合球面子图（χ=2），此值必须 = 12。
        这是 L0 拓扑常数 12 的帧内验证公式。

        Returns:
            {"sum_6_minus_deg": int, "degree_distribution": dict}
        """
        degrees = {n: len(adj) for n, adj in neighbors.items()}
        total = sum(6 - d for d in degrees.values())
        # 度数分布统计
        dist = defaultdict(int)
        for d in degrees.values():
            dist[d] += 1
        return {
            "sum_6_minus_deg": total,
            "constant_12_satisfied": total == self.topological_constant(),
            "degree_distribution": dict(dist),
            "degrees": degrees,
        }

    def topological_constant(self) -> int:
        """返回拓扑常数 12。

        两条独立推导路径互相锁定:
        1. 几何路径: D_max = 4κ → S = πD² → N_max = 16π → k₀ ≤ 12
        2. 组合路径: Σ(6-deg(v)) = 12 → 每个 0 层级节点贡献 1
           → 必须恰好 12 个节点
        π 在组合路径中完全不出现——12 是纯拓扑的强制解。
        """
        return 12

    def opening_ratio(self, dangling_count: int,
                      total_nodes: int) -> float:
        """开口占比 = |悬挂端| / |总节点|。

        定理: 开口占比 ≤ 1/3（由带边界离散高斯-博内定理导出）。
        """
        if total_nodes == 0:
            return 0.0
        return dangling_count / total_nodes

    def opening_ratio_satisfied(self, ratio: float) -> bool:
        """验证开口占比是否满足 ≤ 1/3。"""
        return ratio <= 1.0 / 3.0

    def describe(self) -> str:
        return (
            "公理4: 拓扑守恒 — Σ(6-deg(v))=12(球面), "
            "χ=V-E+F(欧拉), Σdeg=2|ε|(握手引理), "
            "开口占比≤1/3。全部局部可检验。"
            "总边数|ε|全局严格守恒。"
        )


# ══════════════════════════════════════════════════════════════════════
# Axiom 5: 不完美
# ══════════════════════════════════════════════════════════════════════

class Axiom5:
    """公理5: 不完美 — 双层判据。

    核心命题:
        - 一帧结束时网络永远无法达到"所有节点度数≥2"的完美状态
        - 删除悬挂边必然产生新的悬挂边

    双层不完美（N015/N016 实验修正）:
        1. 边缘不完美: 每帧必残留悬挂端
        2. 中心不完美: 锚点（引力中心/类均值）持续震荡，永不"锁定"

    完整拓扑闭合判据:
        闭合 = (δ < θ_δ) ∧ (Δμ < θ_μ) 持续 N 帧
        单靠 δ 下降不能判定闭合（N013 证伪）。
    """

    def __init__(self, delta_threshold: float = 0.3,
                 drift_threshold: float = 0.01,
                 closure_patience: int = 3):
        self.delta_threshold = delta_threshold
        self.drift_threshold = drift_threshold
        self.closure_patience = closure_patience
        self._history: List[dict] = []
        self._last_anchors = None
        self._closure_counter = 0

    def compute_anchors(self, hidden, labels, num_classes: int):
        """计算各类别的锚点（类均值隐藏表示）。

        Args:
            hidden: (N, D) 隐藏层输出 (torch.Tensor)
            labels: (N,) 类别标签 (torch.Tensor)
            num_classes: 类别数

        Returns:
            anchors: (C, D) 各类锚点
        """
        import torch
        anchors = torch.zeros(num_classes, hidden.shape[1], device=hidden.device)
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                anchors[c] = hidden[mask].mean(dim=0)
        return anchors

    def compute_delta(self, hidden, anchors,
                      eps: Optional[float] = None) -> float:
        """计算 δ 密度（悬挂端比例）。

        Args:
            hidden: (N, D) 隐藏层表示
            anchors: (C, D) 锚点
            eps: 悬挂端阈值，默认 self.delta_threshold

        Returns:
            float: δ 密度
        """
        import torch
        eps = eps or self.delta_threshold
        dists = torch.cdist(hidden.unsqueeze(1), anchors.unsqueeze(0)).squeeze(1)
        sorted_dists, _ = dists.sort(dim=1)
        dist_diff = sorted_dists[:, 1] - sorted_dists[:, 0]
        is_dangling = dist_diff < eps
        return is_dangling.float().mean().item()

    def compute_drift(self, anchors) -> Optional[float]:
        """计算锚点漂移 Δμ。

        Args:
            anchors: 当前帧的锚点

        Returns:
            float: 锚点漂移距离，或 None（首帧无上次锚点）
        """
        if self._last_anchors is None:
            self._last_anchors = anchors.clone()
            return None
        import torch
        drift = torch.norm(anchors - self._last_anchors).item()
        self._last_anchors = anchors.clone()
        return drift

    def update(self, hidden, labels, num_classes: int,
               epoch: int) -> dict:
        """执行一帧的不完美检测。

        Args:
            hidden: (N, D) 隐藏层输出
            labels: (N,) 类别标签
            num_classes: 类别数
            epoch: 帧号/轮次

        Returns:
            dict: {
                "epoch", "delta", "drift",
                "is_delta_stable", "is_drift_stable", "is_closed",
                "closure_frames"
            }
        """
        anchors = self.compute_anchors(hidden, labels, num_classes)
        delta = self.compute_delta(hidden, anchors)
        drift = self.compute_drift(anchors)

        is_delta_stable = delta < self.delta_threshold
        is_drift_stable = drift is not None and drift < self.drift_threshold
        is_closed = is_delta_stable and is_drift_stable

        if is_closed:
            self._closure_counter += 1
        else:
            self._closure_counter = 0

        result = {
            "epoch": epoch,
            "delta": delta,
            "drift": drift,
            "delta_stable": is_delta_stable,
            "drift_stable": is_drift_stable,
            "is_closed": is_closed,
            "closure_frames": self._closure_counter,
        }
        self._history.append(result)
        return result

    def is_fully_closed(self) -> bool:
        """检查是否达到完整闭合（双层判据 + 持续 patience 帧）。"""
        return self._closure_counter >= self.closure_patience

    def check_edge_imperfection(self, neighbors: Dict[str, Set[str]],
                                edge_count: int) -> dict:
        """检查边缘不完美（每帧悬挂端残留）。

        Returns:
            {"dangling_count": int, "dangling_density": float,
             "dangling_nodes": list, "imperfection_proven": bool}
        """
        d_nodes = {n for n, adj in neighbors.items() if len(adj) < 2}
        n_total = len(neighbors)
        density = len(d_nodes) / max(n_total, 1)
        # 核心: 删除悬挂边后检查是否产生新悬挂端
        new_dangling = set()
        for node in d_nodes:
            for nb in list(neighbors.get(node, set())):
                if nb in neighbors and len(neighbors[nb]) <= 2:
                    new_dangling.add(nb)
        return {
            "dangling_count": len(d_nodes),
            "dangling_density": density,
            "dangling_nodes": sorted(d_nodes),
            "new_dangling_after_removal": sorted(new_dangling),
            "imperfection_proven": len(d_nodes) > 0 or len(new_dangling) > 0,
        }

    def history(self) -> List[dict]:
        """返回所有帧状态记录。"""
        return self._history

    def summary(self) -> dict:
        """返回汇总统计。"""
        if not self._history:
            return {}
        deltas = [h["delta"] for h in self._history]
        drifts = [h["drift"] for h in self._history if h["drift"] is not None]
        closed = sum(1 for h in self._history if h["is_closed"])
        return {
            "total_frames": len(self._history),
            "min_delta": min(deltas),
            "final_delta": deltas[-1],
            "min_drift": min(drifts) if drifts else None,
            "final_drift": drifts[-1] if drifts else None,
            "closed_frames": closed,
            "fully_closed": self.is_fully_closed(),
        }

    def describe(self) -> str:
        return (
            "公理5: 不完美 — 双层判据。"
            "边缘不完美: 每帧必残留悬挂端。"
            "中心不完美: 锚点持续震荡永不锁定。"
            "完整闭合=(δ<θ_δ)∧(Δμ<θ_μ)持续N帧。"
            "单靠δ下降不能判定闭合(N013证伪)。"
        )


# ══════════════════════════════════════════════════════════════════════
# 派生操作: σ 密度与 dv/dt
# ══════════════════════════════════════════════════════════════════════

class DerivedOperations:
    """从公理派生的基本操作。

    σ 密度:
        σ = |P|/|ε|
        高 σ = 度数低、连接稀疏 → 温度更高
        低 σ = 度数高、连接饱和 → 温度更低

    dv/dt ≤ const:
        单帧跃迁中单个节点关系变化的上限
        dt = τ, const = D_max = 4κ
    """

    def sigma(self, node_count: int, edge_count: int) -> float:
        """空间密度 σ = |P|/|ε|。

        Returns:
            float: σ。高 σ = 稀疏热区，低 σ = 饱和冷区。
        """
        if edge_count == 0:
            return float('inf')
        return node_count / edge_count

    def sigma_from_neighbors(self, neighbors: Dict[str, Set[str]],
                             edge_count: int) -> float:
        """从邻接表计算 σ。"""
        return self.sigma(len(neighbors), edge_count)

    def avg_degree(self, node_count: int, edge_count: int) -> float:
        """平均度数 ⟨deg⟩ = 2/σ = 2|ε|/|P|。"""
        if node_count == 0:
            return 0.0
        return 2.0 * edge_count / node_count

    def dv_dt_check(self, degree_change: int, frame_count: int = 1) -> dict:
        """检查 dv/dt ≤ const 约束。

        Args:
            degree_change: 度数变化量
            frame_count: 帧数，默认 1

        Returns:
            {"dv": ..., "dt": ..., "const": ..., "satisfied": bool}
        """
        dt = frame_count
        const = Constants.DV_DT_MAX
        return {
            "dv": degree_change,
            "dt": dt,
            "const": const,
            "ratio": degree_change / max(dt, 1),
            "satisfied": degree_change / max(dt, 1) <= const,
        }

    def temperature_indicator(self, sigma: float) -> str:
        """σ 的温度指示。

        高 σ（稀疏）→ 温度更高
        低 σ（饱和）→ 温度更低
        """
        if sigma > 1.0:
            return "高温(稀疏)"
        elif sigma > 0.5:
            return "中温"
        else:
            return "低温(饱和)"


# ══════════════════════════════════════════════════════════════════════
# 公理聚合
# ══════════════════════════════════════════════════════════════════════

class Axioms:
    """SPUM 全部 5 条核心公理 + 派生操作的统一入口。

    用法:
        ax = Axioms()
        ax.axiom1.add_edge("A", "B", neighbors, edge_count)
        ax.axiom4.angle_deficit_sum(neighbors)
        ax.sigma(node_count, edge_count)
    """

    def __init__(self):
        self.axiom1 = Axiom1()
        self.axiom2 = Axiom2()
        self.axiom3 = Axiom3()
        self.axiom4 = Axiom4()
        self.axiom5 = Axiom5()
        self.derived = DerivedOperations()
        self.constants = Constants()

    # ── 便捷方法 ─────────────────────────────────────────────────

    def sigma(self, node_count: int, edge_count: int) -> float:
        return self.derived.sigma(node_count, edge_count)

    def sigma_from_neighbors(self, neighbors: Dict[str, Set[str]],
                             edge_count: int) -> float:
        return self.derived.sigma_from_neighbors(neighbors, edge_count)

    def avg_degree(self, node_count: int, edge_count: int) -> float:
        return self.derived.avg_degree(node_count, edge_count)

    def handshaking(self, neighbors: Dict[str, Set[str]],
                    edge_count: int) -> dict:
        return self.axiom4.handshaking(neighbors, edge_count)

    def angle_deficit(self, neighbors: Dict[str, Set[str]]) -> dict:
        return self.axiom4.angle_deficit_sum(neighbors)

    # ── 公理描述 ─────────────────────────────────────────────────

    def describe_all(self) -> str:
        return "\n".join([
            self.axiom1.describe(),
            self.axiom2.describe(),
            self.axiom3.describe(),
            self.axiom4.describe(),
            self.axiom5.describe(),
            f"常数: κ={self.constants.KAPPA}, "
            f"τ={self.constants.TAU}, "
            f"c={self.constants.C}, "
            f"12={self.constants.TOPOLOGICAL_12}",
        ])

    def summary(self) -> str:
        return (
            f"SPUM 公理系统 | 5 条公理 | "
            f"σ=|P|/|ε| | "
            f"Σ(6-deg)=12 | "
            f"dv/dt≤{Constants.DV_DT_MAX} | "
            f"δ<{Constants.DANGLING_EPS}∧Δμ<{Constants.DRIFT_THRESHOLD}"
        )
