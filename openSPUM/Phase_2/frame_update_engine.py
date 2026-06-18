"""
帧演化引擎 — 网络的持续演化与级联消解

核心流程（每帧）:
    1. 生长阶段: 可选的新边创建（延续种子期增长）
    2. 级联消解: 检测悬挂节点 → 湮灭其边 → 补偿创建 → 递归检测
    3. 不变量校验: 度数守恒 (Σdeg = 2E), Σ(6−deg) 记录
    4. 帧日志: 记录本帧所有事件

设计原则:
    - 湮灭-创生对偶: 湮灭自动触发补偿创建
    - 级联有限: 最大迭代次数保护，悬挂节点 < 2 即终止
    - 聚簇边不受级联影响（保持晶子子图结构完整性）
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from Phase_1.topological_address import TopologicalAddress
from Phase_1.relation_pool import RelationPool
from Phase_1.node_registry import NodeRegistry, NodeState
from Phase_1.constants import CRYSTALLITE_DEGREE_THRESHOLD


@dataclass
class FrameUpdateConfig:
    """帧演化配置。

    Attributes:
        growth_per_frame:      每帧生长阶段创建的新边数（0 = 仅级联）
        max_cascade_iterations: 级联消解最大迭代次数
        enable_cascade:         是否启用级联消解
        directional_cascade:    是否启用定向传播级联（α 测量模式）
        cascade_source_uid:     定向传播的扰动源 UID
        use_natural_cap:        是否用局部拓扑亏损决定补偿边数（默认 False）
    """

    growth_per_frame: int = 0
    max_cascade_iterations: int = 100
    enable_cascade: bool = True
    directional_cascade: bool = False
    cascade_source_uid: str = ""
    use_natural_cap: bool = False


@dataclass
class FrameLog:
    """单帧运行的完整日志。

    Attributes:
        frame_number:      帧编号
        edges_created:     本帧创建的边数（生长 + 补偿）
        edges_annihilated: 本帧湮灭的边数
        net_edge_change:   edges_created - edges_annihilated
        cascade_iters:     级联迭代次数
        dangling_before:   级联前悬挂节点数
        dangling_after:    级联后悬挂节点数
        node_count:        总节点数
        edge_count:        常规边数
        cluster_edge_count: 聚簇边数
        degree_sum:        总度数之和
        spum_invariant:    Σ(6−deg) = 6V - 2E
        crystallite_count: 晶子节点数
        max_degree:        最大度数
        stable:            本帧是否稳定收敛（无残留悬挂）
    """

    frame_number: int = 0
    edges_created: int = 0
    edges_annihilated: int = 0
    net_edge_change: int = 0
    cascade_iters: int = 0
    dangling_before: int = 0
    dangling_after: int = 0
    node_count: int = 0
    edge_count: int = 0
    cluster_edge_count: int = 0
    degree_sum: int = 0
    spum_invariant: int = 0
    crystallite_count: int = 0
    max_degree: int = 0
    stable: bool = True

    # 级联边级追踪（可选，用于粒子追踪和位移分析）
    annihilated_keys: List[Tuple[str, str]] = field(default_factory=list)
    compensated_keys: List[Tuple[str, str]] = field(default_factory=list)

    def brief(self) -> str:
        """返回一行摘要。"""
        return (
            f"帧{self.frame_number:>4}: "
            f"边{self.net_edge_change:+d} "
            f"(创建{self.edges_created}/湮灭{self.edges_annihilated}) "
            f"| 悬挂{self.dangling_before}->{self.dangling_after} "
            f"| 级联{self.cascade_iters}次 "
            f"| 总边={self.edge_count}(+{self.cluster_edge_count}簇) "
            f"| V={self.node_count} "
            f"| Sigma(6-deg)={self.spum_invariant:+d} "
            f"| 晶子={self.crystallite_count} "
            f"{'[OK]' if self.stable else '[!]'}"
        )


@dataclass
class FrameUpdateEngine:
    """帧演化引擎 — 网络持续演化与级联消解。

    使用方式:
        engine = FrameUpdateEngine(relation_pool, node_registry, config)
        log = engine.run_frame()
        print(log.brief())

    与 SeedEpochEngine 的关系:
        - SeedEpochEngine 负责初始创生
        - FrameUpdateEngine 负责后续持续演化
        - 共享同一组 relation_pool + node_registry
    """

    relation_pool: RelationPool
    node_registry: NodeRegistry
    config: FrameUpdateConfig = field(default_factory=FrameUpdateConfig)
    current_frame: int = 0
    logs: List[FrameLog] = field(default_factory=list)

    # ------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------
    def run_frame(self) -> FrameLog:
        """运行一帧演化。

        Returns:
            FrameLog 包含本帧所有事件统计
        """
        log = FrameLog(frame_number=self.current_frame)
        edges_before = self.relation_pool.edge_count()
        cluster_before = self.relation_pool.cluster_edge_count()
        total_before = edges_before + cluster_before

        # ---- Phase 1: 生长（可选） ----
        if self.config.growth_per_frame > 0:
            candidates = self.relation_pool.deterministic_candidates(
                self.node_registry,
                count=self.config.growth_per_frame,
                layer="all",
            )
            for loc_a, loc_b in candidates:
                self.relation_pool.manifest_relation(
                    loc_a, loc_b, self.current_frame, self.node_registry
                )

        # ---- Phase 2: 级联消解 ----
        log.dangling_before = len(self.node_registry.dangling_nodes())
        if self.config.enable_cascade:
            if self.config.directional_cascade and self.config.cascade_source_uid:
                self._resolve_directed_cascade(log)
            else:
                self._resolve_cascade(log)

        edges_after = self.relation_pool.edge_count()
        cluster_after = self.relation_pool.cluster_edge_count()
        total_after = edges_after + cluster_after

        net_change = total_after - total_before
        log.edges_annihilated = max(0, log.edges_annihilated)
        log.edges_created = max(0, net_change + log.edges_annihilated)
        log.net_edge_change = net_change

        log.node_count = self.node_registry.node_count()
        log.edge_count = edges_after
        log.cluster_edge_count = cluster_after
        log.degree_sum = sum(
            n.degree for n in self.node_registry.nodes.values()
        )
        log.spum_invariant = 6 * log.node_count - 2 * (edges_after + cluster_after)
        log.crystallite_count = len(self.node_registry.crystallite_nodes())
        log.max_degree = max(
            (n.degree for n in self.node_registry.nodes.values()),
            default=0,
        )
        log.dangling_after = len(self.node_registry.dangling_nodes())
        log.stable = log.dangling_after == 0

        self.logs.append(log)
        self.current_frame += 1
        return log

    # ------------------------------------------------------------
    # 自然 cap: 从局部拓扑涌现补偿边数
    # ------------------------------------------------------------
    def _compute_natural_cap(
        self,
        batch_keys: Set[Tuple[str, str]],
    ) -> int:
        """从湮灭边的局部邻域拓扑计算自然补偿边数。

        原理 (SPUM 拓扑亏损):
            Σ(6−deg(v)) = 12 是闭合三角剖分的不变量。
            悬挂对湮灭: 局部 Σ(6−deg) 增加 2 (两个 degree=1→0)。
            每条补偿边连接两个可用节点, 各增 1 度,
            可修复 2 单位亏损。

            因此, natural_cap 取决于局部邻域中
            还有多少 degree<2 的可用节点可接收补偿边。

        计算:
            1. 收集湮灭边端点的 1-hop 邻域
            2. 排除刚湮灭的节点自身
            3. 计数 degree<2 且未饱和的节点
            4. natural_cap = count // 2 (每条边需两个端点)

        Returns:
            自然补偿边数 (≥ 0, 上限 len(batch_keys))
        """
        # === Phase 1: 收集 1-hop 邻域 ===
        affected: Set[str] = set()
        for key in batch_keys:
            affected.add(key[0])
            affected.add(key[1])

        # 1-hop 邻居 (从 manifest 中收集)
        neighborhood: Set[str] = set()
        for (ua, ub), _ in self.relation_pool.manifest.items():
            if ua in affected:
                neighborhood.add(ub)
            elif ub in affected:
                neighborhood.add(ua)

        # 排除受影响节点自身
        available = neighborhood - affected

        # === Phase 2: 计数可用度 < 2 节点 ===
        eligible_count = 0
        for uid in available:
            node = self.node_registry.nodes.get(uid)
            if node and node.degree < 2 and not node.is_saturated:
                eligible_count += 1

        natural_cap = eligible_count // 2
        return max(0, min(natural_cap, len(batch_keys)))

    # ------------------------------------------------------------
    # 级联消解
    # ------------------------------------------------------------
    def _resolve_cascade(self, log: FrameLog) -> None:
        """级联消解: 检测悬挂 → 批量湮灭 → 单次补偿 → 递归检测。

        策略:
            - 每轮迭代先批量湮灭所有悬挂边
            - 湮灭后单次补偿（避免逐边补偿的无限循环）
            - 最多 max_cascade_iterations 轮

        终止条件:
            - 无悬挂节点残留
            - 无待湮灭边（悬挂但与簇中其他节点无可用边）
            - 达到最大迭代次数
        """
        total_annihilated = 0
        all_annihilated_keys: List[Tuple[str, str]] = []
        all_compensated_keys: List[Tuple[str, str]] = []

        for iteration in range(self.config.max_cascade_iterations):
            dangling_nodes = self.node_registry.dangling_nodes()
            if not dangling_nodes:
                log.cascade_iters = iteration
                log.edges_annihilated = total_annihilated
                log.annihilated_keys = all_annihilated_keys
                log.compensated_keys = all_compensated_keys
                return

            # 收集本批待湮灭边：仅当两端都是悬挂节点
            # (悬挂节点连接到稳定节点是合法虚面，不应消解)
            dangling_uids = {n.address.uid for n in dangling_nodes}
            batch_keys: List[Tuple[str, str]] = []
            for node in dangling_nodes:
                for key, rel in list(self.relation_pool.manifest.items()):
                    if not (rel.loc1.uid == node.address.uid
                            or rel.loc2.uid == node.address.uid):
                        continue
                    if self.relation_pool.is_cluster_edge(key):
                        continue
                    other_uid = (rel.loc2.uid if rel.loc1.uid == node.address.uid
                                 else rel.loc1.uid)
                    if other_uid in dangling_uids:
                        batch_keys.append(key)

            if not batch_keys:
                log.cascade_iters = iteration + 1
                log.edges_annihilated = total_annihilated
                log.annihilated_keys = all_annihilated_keys
                log.compensated_keys = all_compensated_keys
                return

            # 批量湮灭（直接操作 manifest，不触发逐边补偿）
            batch_uids_to_annihilate = set(batch_keys)
            for key in batch_uids_to_annihilate:
                rel = self.relation_pool.manifest.pop(key, None)
                if rel is None:
                    continue
                all_annihilated_keys.append(key)
                is_cluster = key in self.relation_pool._cluster_edge_keys
                if is_cluster:
                    self.relation_pool._cluster_edge_keys.discard(key)
                    self.node_registry.nodes[rel.loc1.uid].crystallite_connections -= 1
                    self.node_registry.nodes[rel.loc2.uid].crystallite_connections -= 1
                else:
                    self.node_registry.update_degree(rel.loc1.uid, -1)
                    self.node_registry.update_degree(rel.loc2.uid, -1)
                total_annihilated += 1

            # 单次补偿：为本批湮灭的边创建替代连接
            # 【涌现 cap】边数由局部拓扑亏损决定，非硬编码
            # 排除刚湮灭的键，避免立即重建同一条边（振荡）
            if self.config.use_natural_cap:
                comp_count = self._compute_natural_cap(batch_uids_to_annihilate)
            else:
                comp_count = min(len(batch_uids_to_annihilate), 3)  # 旧行为
            candidates = self.relation_pool.deterministic_candidates(
                self.node_registry, count=comp_count, layer="all",
                exclude_keys=batch_uids_to_annihilate,
            )
            for loc_a, loc_b in candidates:
                self.relation_pool.manifest_relation(
                    loc_a, loc_b, self.current_frame, self.node_registry
                )
                all_compensated_keys.append((loc_a.uid, loc_b.uid))

        log.cascade_iters = self.config.max_cascade_iterations
        log.edges_annihilated = total_annihilated
        log.annihilated_keys = all_annihilated_keys
        log.compensated_keys = all_compensated_keys

    # ------------------------------------------------------------
    # 定向级联消解（α 传播测量用）
    # ------------------------------------------------------------
    def _resolve_directed_cascade(self, log: FrameLog) -> None:
        """定向级联消解: 湮灭 + 前方补偿 → 形成定向传播链。

        与 _resolve_cascade 的关键区别:
            - 补偿使用 find_directional_candidates (优先扰动源前方区域)
            - 每轮迭代的补偿上限提升到 5（增强传播性）
            - 记录每轮迭代的传播距离

        这模拟了光子的"吞噬-重排"循环:
            - 湮灭 = 光子被吸收
            - 定向补偿 = 光子在前进方向重新发射
        """
        total_annihilated = 0
        all_annihilated_keys: List[Tuple[str, str]] = []
        all_compensated_keys: List[Tuple[str, str]] = []
        source_uid = self.config.cascade_source_uid

        for iteration in range(self.config.max_cascade_iterations):
            dangling_nodes = self.node_registry.dangling_nodes()
            if not dangling_nodes:
                log.cascade_iters = iteration
                log.edges_annihilated = total_annihilated
                log.annihilated_keys = all_annihilated_keys
                log.compensated_keys = all_compensated_keys
                return

            # 收集本批待湮灭边
            dangling_uids = {n.address.uid for n in dangling_nodes}
            batch_keys: List[Tuple[str, str]] = []
            for node in dangling_nodes:
                for key, rel in list(self.relation_pool.manifest.items()):
                    if not (rel.loc1.uid == node.address.uid
                            or rel.loc2.uid == node.address.uid):
                        continue
                    if self.relation_pool.is_cluster_edge(key):
                        continue
                    other_uid = (rel.loc2.uid if rel.loc1.uid == node.address.uid
                                 else rel.loc1.uid)
                    if other_uid in dangling_uids:
                        batch_keys.append(key)

            if not batch_keys:
                log.cascade_iters = iteration + 1
                log.edges_annihilated = total_annihilated
                log.annihilated_keys = all_annihilated_keys
                log.compensated_keys = all_compensated_keys
                return

            # 批量湮灭
            batch_set = set(batch_keys)
            for key in batch_set:
                rel = self.relation_pool.manifest.pop(key, None)
                if rel is None:
                    continue
                all_annihilated_keys.append(key)
                is_cluster = key in self.relation_pool._cluster_edge_keys
                if is_cluster:
                    self.relation_pool._cluster_edge_keys.discard(key)
                    self.node_registry.nodes[rel.loc1.uid].crystallite_connections -= 1
                    self.node_registry.nodes[rel.loc2.uid].crystallite_connections -= 1
                else:
                    self.node_registry.update_degree(rel.loc1.uid, -1)
                    self.node_registry.update_degree(rel.loc2.uid, -1)
                total_annihilated += 1

            # 定向补偿（上限由局部拓扑决定，排除刚湮灭的键）
            if self.config.use_natural_cap:
                comp_count = self._compute_natural_cap(batch_set)
            else:
                comp_count = min(len(batch_set), 5)  # 旧行为
            candidates = self.relation_pool.find_directional_candidates(
                self.node_registry,
                source_uid=source_uid,
                count=comp_count,
                forward_hemisphere_only=True,
                exclude_keys=batch_set,
            )
            for loc_a, loc_b in candidates:
                self.relation_pool.manifest_relation(
                    loc_a, loc_b, self.current_frame, self.node_registry
                )
                all_compensated_keys.append((loc_a.uid, loc_b.uid))

        log.cascade_iters = self.config.max_cascade_iterations
        log.edges_annihilated = total_annihilated
        log.annihilated_keys = all_annihilated_keys
        log.compensated_keys = all_compensated_keys

    # ------------------------------------------------------------
    # 批量运行
    # ------------------------------------------------------------
    def run_frames(self, n: int) -> List[FrameLog]:
        """运行 N 帧演化。

        Args:
            n: 帧数

        Returns:
            FrameLog 列表
        """
        return [self.run_frame() for _ in range(n)]

    # ------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------
    def summary(self) -> str:
        """返回当前网络状态的完整摘要。"""
        total_edges = self.relation_pool.edge_count()
        cluster_edges = self.relation_pool.cluster_edge_count()
        nodes = self.node_registry.nodes
        deg_sum = sum(n.degree for n in nodes.values())
        v = len(nodes)
        e = total_edges + cluster_edges

        return (
            f"帧演化引擎状态\n"
            f"{'=' * 40}\n"
            f"  当前帧:      {self.current_frame}\n"
            f"  总节点:      {v}\n"
            f"  常规边:      {total_edges}\n"
            f"  聚簇边:      {cluster_edges}\n"
            f"  总度数:      {deg_sum} (常规边应={2 * total_edges}, "
            f"簇连接不计度)\n"
            f"  Σ(6−deg):   {6 * v - 2 * e}\n"
            f"  晶子数:      {len(self.node_registry.crystallite_nodes())}\n"
            f"  悬挂节点:    {len(self.node_registry.dangling_nodes())}\n"
            f"  已运行帧:    {len(self.logs)}\n"
            f"{'=' * 40}"
        )
