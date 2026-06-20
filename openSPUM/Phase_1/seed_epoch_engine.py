"""
种子创生期引擎 — 从空关系池确定性地生成初始宇宙。

增长策略 (v2 — 分层扩散):
    - 首帧手动创建第一对地址与第一条边
    - 后续帧按等比序列计算预计节点数，而非边数
    - 每帧连接配额按 30/50/20 分配 (悬挂/中段/新节点)
    - 种子创生期不执行湮灭和级联消解 (允许悬挂边暂存)
    - 种子期结束时自动执行晶子聚簇阶段
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .topological_address import TopologicalAddress, reset_differentiation_counter
from .relation_pool import RelationPool
from .node_registry import NodeRegistry, NodeState
from .constants import (
    MAX_CLUSTER_ITERATIONS,
    CRYSTALLITE_DEGREE_THRESHOLD,
)


@dataclass
class SeedEpochConfig:
    """种子创生期配置。

    Attributes:
        target_edges:   目标总边数
        target_nodes:   目标总节点数（根据 target_edges 自动推算）
        seed_duration:  种子期帧数
    """

    target_edges: int = 10000
    target_nodes: int = 0  # 自动计算
    seed_duration: int = 200

    def __post_init__(self) -> None:
        """自动推算 target_nodes ≈ sqrt(2 * target_edges)。"""
        if self.target_nodes <= 0:
            import math
            # n(n-1)/2 ≈ target_edges → n ≈ sqrt(2 * target_edges)
            self.target_nodes = int(math.sqrt(2.0 * self.target_edges))
            if self.target_nodes < 10:
                self.target_nodes = 10

    # ------------------------------------------------------------
    # 节点增长序列
    # ------------------------------------------------------------
    def build_node_sequence(self) -> List[int]:
        """生成每帧期望的节点数（等比逼近 target_nodes）。"""
        if self.seed_duration <= 0:
            return [self.target_nodes]

        import math

        q = (
            (1.0 / self.target_nodes) ** (1.0 / self.seed_duration)
            if self.target_nodes > 1
            else 0.5
        )

        sequence: List[int] = []
        for i in range(self.seed_duration):
            target_at_frame = int(self.target_nodes * (1.0 - q ** (i + 1)))
            if target_at_frame < 2:
                target_at_frame = 2
            sequence.append(target_at_frame)

        return sequence


@dataclass
class SeedEpochEngine:
    """种子创生期引擎 — 从空池确定性地生成初始宇宙。

    Attributes:
        config:         种子期配置
        relation_pool:  关系池实例
        node_registry:  节点注册表实例
        current_frame:  当前帧编号
        node_sequence:  预计算的节点数序列
        total_edges:    种子期结束时总边数
        cluster_edges:  聚簇阶段创建的边数
    """

    config: SeedEpochConfig
    relation_pool: RelationPool = field(default_factory=RelationPool)
    node_registry: NodeRegistry = field(default_factory=NodeRegistry)
    current_frame: int = 0
    node_sequence: List[int] = field(default_factory=list)
    total_edges: int = 0
    cluster_edges: int = 0

    def __post_init__(self) -> None:
        self.node_sequence = self.config.build_node_sequence()

    def run_seed_epoch(self) -> int:
        """执行整个种子创生期 + 聚簇阶段。

        每帧增长逻辑:
            1. 计算本帧期望节点数
            2. 按度数分布动态分配 3 层连接配额:
               - dangling: 悬挂节点配对（deg<2）
               - midrange: 中段节点生长（2≤deg<50）
               - new_nodes: 从最优节点分化
               见 _compute_budgets_from_degree_distribution

        Returns:
            最终总边数
        """
        reset_differentiation_counter()

        # Frame 0: 创建 origin + 第一个孩子
        origin = TopologicalAddress.origin()
        second = TopologicalAddress.differentiate_from(origin)
        self.relation_pool.manifest_relation(
            origin, second, self.current_frame, self.node_registry
        )
        self.current_frame += 1

        # 后续帧: 分层扩散增长
        for frame_idx in range(1, len(self.node_sequence)):
            desired_nodes = self.node_sequence[frame_idx]
            current_nodes = self.node_registry.node_count()
            remaining = desired_nodes - current_nodes

            if remaining <= 0 and self._all_saturated():
                break

            # 总连接预算 = 剩余节点数 × 3（确保足够的生长速度）
            total_budget = max(2, remaining * 3)
            budget_dangling, budget_midrange, budget_new = self._compute_budgets_from_degree_distribution(
                total_budget, remaining
            )

            # ---- Layer 1: 悬挂节点消解 ----
            d_candidates = self.relation_pool.deterministic_candidates(
                self.node_registry, count=budget_dangling, layer="dangling"
            )
            for loc_a, loc_b in d_candidates:
                self.relation_pool.manifest_relation(
                    loc_a, loc_b, self.current_frame, self.node_registry
                )

            # ---- Layer 2: 中段生长 ----
            m_candidates = self.relation_pool.deterministic_candidates(
                self.node_registry, count=budget_midrange, layer="midrange"
            )
            for loc_a, loc_b in m_candidates:
                self.relation_pool.manifest_relation(
                    loc_a, loc_b, self.current_frame, self.node_registry
                )

            # ---- Layer 3: 新节点分化 ----
            n_candidates = self.relation_pool.deterministic_candidates(
                self.node_registry, count=budget_new, layer="new_nodes"
            )
            for loc_a, loc_b in n_candidates:
                self.relation_pool.manifest_relation(
                    loc_a, loc_b, self.current_frame, self.node_registry
                )

            self.current_frame += 1

        # 聚簇阶段
        self.cluster_edges = self._run_clustering_phase()

        self.total_edges = self.relation_pool.edge_count()
        return self.total_edges

    def _all_saturated(self) -> bool:
        """所有现有节点是否均已饱和？"""
        return all(
            n.is_saturated for n in self.node_registry.nodes.values()
        )

    def _compute_budgets_from_degree_distribution(
        self, total_budget: int, remaining: int
    ) -> Tuple[int, int, int]:
        """从当前度数分布动态计算三层配额（⟨P, ε⟩ 第一性推导）。

        核心思想（见 元素化学/力学参数推导.md）：
            配额比例来自度数分布中不同区段节点的拓扑需求权重。
            不是任意数字，而是当前网络状态的函数。

        Derivation:
            dangling_weight = N_deg<2 × 1.0   （每个悬挂节点需 1 条边稳定）
            midrange_weight = N_2≤deg<50 × 0.5 （中段节点分摊到多帧生长）
            newnode_weight  = N_remaining × 1.0 （每个新节点需 1 条边附网）

            系数 0.5 的来源：中段节点平均还需 ~25 条边达到饱和（50），
            但生长跨多帧分摊，每帧处理约一半的开放需求。

        时间平均效果：
            早期帧（多数节点 deg<2）：   dangling ~60%, midrange ~0%, new ~40%
            中期帧（多数节点 deg=2~10）： dangling ~30%, midrange ~40%, new ~30%
            后期帧（多数节点 deg≥10）：   dangling ~10%, midrange ~60%, new ~30%
            全期平均 ≈ 20/70/10（与旧静态值一致，但现在是动态推导）
        """
        # 统计当前度数分布
        deg_counts: Dict[int, int] = {}
        for node in self.node_registry.nodes.values():
            d = node.degree if node.degree <= CRYSTALLITE_DEGREE_THRESHOLD else CRYSTALLITE_DEGREE_THRESHOLD
            deg_counts[d] = deg_counts.get(d, 0) + 1

        n_dangling = deg_counts.get(0, 0) + deg_counts.get(1, 0)
        n_midrange = sum(
            deg_counts.get(d, 0) for d in range(2, CRYSTALLITE_DEGREE_THRESHOLD)
        )

        # 权重 = 节点数 × 每条边的"拓扑紧急度"
        w_dangling = n_dangling * 1.0   # deg<2 是不稳定态，紧迫
        w_midrange = n_midrange * 0.5   # 中段生长分摊到多帧
        w_newnode = max(0.0, float(remaining)) * 1.0  # 新节点需附网

        total_weight = w_dangling + w_midrange + w_newnode
        if total_weight < 0.01:
            return (0, total_budget, 0)

        b_d = max(0, int(total_budget * w_dangling / total_weight))
        b_n = max(0, int(total_budget * w_newnode / total_weight))
        b_m = max(1, total_budget - b_d - b_n)
        b_d = max(1, b_d)
        # new 受剩余节点数上限约束
        b_n = max(0, min(remaining, b_n))

        return (b_d, b_m, b_n)

    def _run_clustering_phase(self) -> int:
        """聚簇阶段：空间近邻 + 三角剖分补全。

        两阶段策略:
            Phase A — 空间近邻聚簇（贪心最近邻配对）
            Phase B — 三角剖分补全（共享邻居最多优先，补全缺失三角面）
            Phase C — 全局平面性约束：E ≤ 3V - 6

        Returns:
            总聚簇边数
        """
        cluster_edges_created = 0
        enforcer = _ClusterEdgeEnforcer(self.relation_pool, self.node_registry)

        # ---- Phase A: 空间骨架（每晶子最多 3 个连接）----
        # 先创建基础空间骨架，为三角剖分提供初始边
        for _ in range(MAX_CLUSTER_ITERATIONS):
            candidates = self.relation_pool.find_cluster_candidates(
                self.node_registry, count=20
            )
            if not candidates:
                break
            created = 0
            for loc_a, loc_b in candidates:
                n1 = self.node_registry.nodes.get(loc_a.uid)
                n2 = self.node_registry.nodes.get(loc_b.uid)
                # 骨架阶段：每晶子最多 3 个聚簇连接
                if n1 and n1.crystallite_connections >= 3:
                    continue
                if n2 and n2.crystallite_connections >= 3:
                    continue
                if enforcer.maybe_add(loc_a, loc_b, self.current_frame):
                    created += 1
            if created == 0:
                break
            cluster_edges_created += created
            self.current_frame += 1

        # ---- Phase B: 三角剖分补全（共享邻居优先） ----
        # 在骨架基础上补全缺失三角面对角线
        for _ in range(MAX_CLUSTER_ITERATIONS):
            candidates = self.relation_pool.find_triangulation_candidates(
                self.node_registry, max_pairs=20
            )
            if not candidates:
                break
            created = 0
            for loc_a, loc_b in candidates:
                if enforcer.maybe_add(loc_a, loc_b, self.current_frame):
                    created += 1
                    if enforcer.is_at_planarity_limit():
                        break
            if created == 0:
                break
            cluster_edges_created += created
            self.current_frame += 1

        # ---- Phase C: 空间近邻填充 ----
        # 补全剩余可用槽位
        for _ in range(MAX_CLUSTER_ITERATIONS):
            candidates = self.relation_pool.find_cluster_candidates(
                self.node_registry, count=20
            )
            if not candidates:
                break
            created = 0
            for loc_a, loc_b in candidates:
                if enforcer.maybe_add(loc_a, loc_b, self.current_frame):
                    created += 1
                    if enforcer.is_at_planarity_limit():
                        break
            if created == 0:
                break
            cluster_edges_created += created
            self.current_frame += 1

        return cluster_edges_created

    def run_frame(self) -> int:
        """运行单帧进化（种子期后的标准帧接口）。"""
        self.current_frame += 1
        return self.current_frame


class _ClusterEdgeEnforcer:
    """聚簇边执行器 — 平面性约束 + 度数上限检查。

    确保晶子子图满足:
        - 每个晶子 crystallite_connections ≤ MAX_CRYSTALLITE_PLANAR_DEGREE (= 6)
        - 晶子子图整体满足 E ≤ 3V - 6（平面图三角剖分边数上限）
    """

    def __init__(
        self,
        relation_pool: "RelationPool",
        node_registry: "NodeRegistry",
    ) -> None:
        self.pool = relation_pool
        self.registry = node_registry

    def maybe_add(
        self,
        loc_a: TopologicalAddress,
        loc_b: TopologicalAddress,
        frame_number: int,
    ) -> bool:
        """尝试添加一条聚簇边，遵守所有约束。

        Returns:
            是否成功添加
        """
        if self.is_at_planarity_limit():
            return False
        rel = self.pool.manifest_cluster_edge(
            loc_a, loc_b, frame_number, self.registry
        )
        return rel is not None

    def is_at_planarity_limit(self) -> bool:
        """晶子子图是否已达到三角剖分边数上限 E = 3V - 6?"""
        v = len(self.registry.crystallite_nodes())
        e = self.pool.cluster_edge_count()
        max_e = max(0, 3 * v - 6)
        return e >= max_e
