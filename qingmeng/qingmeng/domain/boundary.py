"""
青檬引擎 · 边界效应层 — 领域间耦合强度与衰减
=============================================

蓝图核心功能表"边界效应"：`coupling(域A, 域B) = f(distance, topology)`。

SPUM 校准：
  - 耦合是"度量函数"，不是独立实体（不变量 I）——不注册任何新对象，
    只从 ⟨P, ε⟩ 计算
  - 距离用图内最短路径（拓扑距离），不用欧氏坐标（几何是投影，VIII）
  - 密度差异用局部 σ ≈ 2/⟨deg⟩（认知投影量，标注，不变量 X）
  - 衰减是帧计数的离散遗忘（dv/dt 投影），不是连续时间常数

用法:
    from qingmeng.domain import InteractionBoundary
    b = InteractionBoundary()
    c = b.coupling(G, nodes_a=[0,1], nodes_b=[8,9])   # → [0,1]
    b.decay(0.8, frames=3)                            # → 帧间衰减强度
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx


@dataclass(frozen=True)
class BoundaryConfig:
    """耦合计算参数（领域层可调——公理不可变，参数只限本层）。"""
    # 距离项权重：耦合 ∝ 1/(1 + gamma·d)
    distance_gamma: float = 1.0
    # 密度项权重：σ 越接近耦合越高（[0,1] 内线性混合）
    sigma_weight: float = 0.5
    # 帧间衰减率：强度 × (1 − decay_rate)^frames
    decay_rate: float = 0.05


class InteractionBoundary:
    """两域/两节点集之间的耦合强度与传播衰减。

    核心公式（本实现）：
        coupling(A, B) = dist_term(A,B) × (1 − σ_weight + σ_weight·σ_term)
        dist_term      = 1 / (1 + γ·d̄)          d̄ = 平均最短路径
        σ_term         = 1 − |σ_A − σ_B|         σ ≈ 2/⟨deg⟩（认知投影）
    """

    def __init__(self, config: Optional[BoundaryConfig] = None) -> None:
        self.config = config or BoundaryConfig()

    # ── 耦合强度 ─────────────────────────────────────────────────

    def coupling(
        self,
        G: nx.Graph,
        nodes_a: Iterable[int],
        nodes_b: Iterable[int],
    ) -> float:
        """计算节点集 A 与 B 的耦合强度 ∈ [0,1]。

        0 = 完全无耦合（无路径连通）；1 = 直接相邻且密度完全一致。
        """
        la = list(nodes_a)
        lb = list(nodes_b)
        if not la or not lb:
            return 0.0
        if any(G.degree(n) == 0 for n in la) or any(G.degree(n) == 0 for n in lb):
            return 0.0

        # 拓扑距离项：两域间最短路径均值（不连通对不计入，取可连通的保守均值）
        dists: List[int] = []
        for u in la:
            lengths = nx.single_source_shortest_path_length(G, u)
            for v in lb:
                if v in lengths and v != u:
                    dists.append(lengths[v])
        if not dists:
            return 0.0
        avg_d = sum(dists) / len(dists)
        dist_term = 1.0 / (1.0 + self.config.distance_gamma * avg_d)

        # 密度项：局部 σ ≈ 2/⟨deg⟩（认知投影，不变量 X 标注）
        deg_a = sum(G.degree(n) for n in la) / len(la)
        deg_b = sum(G.degree(n) for n in lb) / len(lb)
        sigma_a = 2.0 / max(deg_a, 1e-6)   # 投影: σ ≈ 2/⟨deg⟩
        sigma_b = 2.0 / max(deg_b, 1e-6)
        sigma_term = 1.0 - min(1.0, abs(sigma_a - sigma_b))

        w = self.config.sigma_weight
        return round(dist_term * ((1.0 - w) + w * sigma_term), 4)

    # ── 衰减 ─────────────────────────────────────────────────────

    def decay(self, intensity: float, frames: int) -> float:
        """帧间离散遗忘：强度 × (1 − decay_rate)^frames。"""
        if frames <= 0:
            return round(intensity, 4)
        return round(intensity * (1.0 - self.config.decay_rate) ** frames, 4)

    # ── 传播强度 ─────────────────────────────────────────────────

    def propagate(self, G: nx.Graph, source: int, target: int) -> Optional[float]:
        """沿拓扑路径的传播强度：1 / (1 + γ·d(source, target))。

        无路径 → None（不可达，耦合为 0）。
        """
        try:
            d = nx.shortest_path_length(G, source, target)
        except nx.NetworkXNoPath:
            return None
        return round(1.0 / (1.0 + self.config.distance_gamma * d), 4)

    # ── 描述 ─────────────────────────────────────────────────────

    def describe(self) -> str:
        return (
            f"边界效应 | coupling = 1/(1+γ·d̄) × (1−σw+σw·σ_term) | "
            f"γ={self.config.distance_gamma}, σw={self.config.sigma_weight}, "
            f"decay={self.config.decay_rate}/帧"
        )
