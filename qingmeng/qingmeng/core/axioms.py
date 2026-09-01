"""
青檬引擎 · 公理内核层 — 不可变的 SPUM 原语
============================================

本模块是理论内核：把 SPUM 的核心公理翻译为可执行原语。
约定（spum-qingmeng-guard.md 11 条不变量）：

  I    ⟨P, ε⟩ 唯一基底          — 不引入独立实体
  V    拓扑常数 12              — CORE_SIZE = 12（正二十面体顶点）
  VI   边是二值关系              — 无权、无向、不可再分
  IV   三角剖分                 — 正二十面体 20 个面全为三角形

关键语义（与连续统范式的分野）：
  - 12 是拓扑常数（Σ(6−deg)=12 的欧拉强制解），不是 12 种类型
  - dv/dt ≤ const 是单帧跃迁中节点关系变化的容量上限，不是连续平滑约束
  - σ = |P|/|ε| 是空间密度，高 σ 稀疏（热），低 σ 饱和（冷）

用法:
    from qingmeng.core import Core
    core = Core()
    G = core.build_icosahedron()     # 12 晶子 · 30 边 · 无孤立节点
    core.angle_deficit(G)            # → {'sum_6_minus_deg': 12, 'satisfied': True}
    core.sigma(G)                    # → 0.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx


# ════════════════════════════════════════════════════════════════════
# 拓扑常数 12 的物理承载：正二十面体（30 边，12 顶点各度 5）
# Σ(6−deg) = 12×(6−5) = 12 —— 离散高斯-博内的强制解
# ════════════════════════════════════════════════════════════════════

ICOSAHEDRON_EDGES: List[Tuple[int, int]] = [
    (0, 1),  (0, 2),  (0, 3),  (0, 4),  (0, 5),          # 上锥顶点 0
    (1, 2),  (2, 3),  (3, 4),  (4, 5),  (5, 1),          # 上五边形环
    (6, 7),  (7, 8),  (8, 9),  (9, 10), (10, 6),         # 下五边形环
    (11, 6), (11, 7), (11, 8), (11, 9), (11, 10),        # 下锥顶点 11
    (1, 6),  (1, 10), (2, 6),  (2, 7),  (3, 7),          # 连接带
    (3, 8),  (4, 8),  (4, 9),  (5, 9),  (5, 10),
]

CORE_SIZE: int = 12


@dataclass(frozen=True)
class Constants:
    """派生常数——全部从公理推导，非预设。"""

    KAPPA: float = 1.0                 # 最小差异尺度（基本长度单位）
    TAU: int = 1                       # 离散帧时间单位
    TOPOLOGICAL_12: int = 12           # Σ(6−deg(v)) = 12 的强制解
    MAX_CONTACTS: int = 12             # 三维密堆最大接触数（涌现结果）
    C: float = 4.0                     # c = 4κ/τ：关系流传播最大速率
    # dv/dt ≤ const：单帧跃迁中单个节点关系变化的容量上限。
    # const = D_max = 4κ（晶子直径上限）—— 关系饱和的必然投影，不是限速规则。
    DV_DT_MAX: float = 4.0
    # 悬挂端密度阈值 δ 与锚点漂移阈值 Δμ（双层闭合判据，N015/N016 实验基准）
    DELTA_THRESHOLD: float = 0.3
    DRIFT_THRESHOLD: float = 0.01


class Core:
    """SPUM 公理原语 — 唯一的 ⟨P, ε⟩ 操作入口（不变量 I）。

    只提供关系操作与只读校验；不引入节点属性（不变量 VI：边无权重、
    节点无预设属性）。演化策略由 core.state 层负责。
    """

    def __init__(self) -> None:
        self.constants = Constants()
        self._graph: Optional[nx.Graph] = None

    # ── 图构建 ────────────────────────────────────────────────────

    def build_icosahedron(self) -> nx.Graph:
        """12 晶子正二十面体骨架。

        满足：CORE_SIZE=12（V）、三角剖分（IV）、无孤立节点（III 的闭合态）、
        Σ(6−deg)=12（V）。边无权重——二值关系（VI）。
        """
        G = nx.Graph()
        G.add_nodes_from(range(CORE_SIZE))
        G.add_edges_from(ICOSAHEDRON_EDGES)
        self._graph = G
        return G

    @property
    def graph(self) -> nx.Graph:
        if self._graph is None:
            self._graph = self.build_icosahedron()
        return self._graph

    # ── 只读拓扑校验 ──────────────────────────────────────────────

    def angle_deficit(self, G: Optional[nx.Graph] = None) -> Dict:
        """Σ(6−deg(v)) 角度亏损总和。

        对闭合球面子图（χ=2），此值必须 = 12（拓扑常数 12 的帧内验证）。
        """
        G = G or self.graph
        sum_loss = sum(6 - d for _, d in G.degree())
        return {
            "sum_6_minus_deg": sum_loss,
            "expected_if_closed": 12,
            "satisfied": sum_loss == self.constants.TOPOLOGICAL_12,
            "degrees": dict(G.degree()),
        }

    def sigma(self, G: Optional[nx.Graph] = None) -> float:
        """空间密度 σ = |P| / |ε|。高 σ 稀疏（热区），低 σ 饱和（冷区）。"""
        G = G or self.graph
        e = G.number_of_edges()
        return float("inf") if e == 0 else G.number_of_nodes() / e

    def avg_degree(self, G: Optional[nx.Graph] = None) -> float:
        """平均度数 ⟨deg⟩ = 2/σ = 2|ε|/|P|。"""
        G = G or self.graph
        n = G.number_of_nodes()
        return 0.0 if n == 0 else 2.0 * G.number_of_edges() / n

    def is_dangling(self, node: int, G: Optional[nx.Graph] = None) -> bool:
        """度数 < 2 即悬挂端（不变量 III）。"""
        G = G or self.graph
        return G.degree(node) < 2

    def dangling_nodes(self, G: Optional[nx.Graph] = None) -> List[int]:
        """返回全部悬挂端节点。"""
        G = G or self.graph
        return sorted(n for n in G.nodes() if G.degree(n) < 2)

    def dangling_density(self, G: Optional[nx.Graph] = None) -> float:
        """悬挂端密度 δ = |悬挂端| / |总节点|。"""
        G = G or self.graph
        n = G.number_of_nodes()
        return 0.0 if n == 0 else len(self.dangling_nodes(G)) / n

    # ── dv/dt ≤ const 容量上限（离散帧语义） ────────────────────────

    def dv_dt_check(self, degree_change: int, frames: int = 1) -> Dict:
        """检查 dv/dt ≤ const。

        dt = τ × frames（离散帧步数，不是无穷小）。
        const = D_max = 4κ：单帧内单个节点关系变化的上限。
        """
        const = self.constants.DV_DT_MAX
        ratio = degree_change / max(frames, 1)
        return {
            "dv": degree_change,
            "dt": frames,
            "const": const,
            "ratio": ratio,
            "satisfied": ratio <= const,
        }

    # ── 级联消解（不变量 III 的执行原语） ───────────────────────────

    def cascade_remove(self, G: nx.Graph, node: int) -> List[int]:
        """删除一个悬挂端节点及其边，直到无新悬挂端产生（一次完整消解）。

        注意：这是"不变量 III 级联消解"的最小原语，供 state 层调用。
        五步帧的"一帧只删一次"约束由 state 层保证，本原语不强制。
        """
        removed: List[int] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n not in G:
                continue
            neighbors = list(G.neighbors(n))
            G.remove_node(n)
            removed.append(n)
            # 邻居度数可能降至 < 2 → 加入消解队列
            for nb in neighbors:
                if nb in G and G.degree(nb) < 2:
                    stack.append(nb)
        return removed

    def describe(self) -> str:
        return (
            f"公理内核 | 12 晶子正二十面体（Σ(6−deg)=12）| "
            f"σ=|P|/|ε| | dv/dt≤{self.constants.DV_DT_MAX}（离散帧容量上限）| "
            f"边为二值关系（无权无向）"
        )
