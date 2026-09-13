"""
节点注册表 — 空间粒子的状态管理

遵循 SPUM 关系第一性原则:
    - 节点不由自身定义，而是由关系显化被动创建
    - NodeRegistry 不主动创建节点，只响应 manifest_relation 的调用
    - 度数 (degree) 是节点唯一的核心状态属性

NodeState 属性:
    - address:         拓扑地址（不可变标识）
    - degree:          当前度数（连接数）
    - spatial_position: 三维空间坐标 (x, y, z)
    - created_frame:   创建时的帧编号

空间几何:
    - radius 基于度数增长，对应粒子影响力域
    - 新节点通过 _pending_positions 获得位置（由分化目标决定）
    - manifest_relation 中验证球体相切性
    - 接吻数 12 由此自然涌现（三维球体堆叠约束）

设计原则:
    - 12（接吻数/拓扑常数）是涌现结果，非预设上限
    - 度数上限由 CRYSTALLITE_DEGREE_THRESHOLD (50) 定义
    - virtual_faces = max(0, CRYSTALLITE_DEGREE_THRESHOLD - degree)
    - 饱和 (is_saturated) = 度数 ≥ 50，即成为晶子后无法接受新连接
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .topological_address import TopologicalAddress
from .constants import CRYSTALLITE_DEGREE_THRESHOLD, MAX_CRYSTALLITE_CLUSTER_SIZE, MAX_CRYSTALLITE_PLANAR_DEGREE


@dataclass
class NodeState:
    """单个空间粒子的完整状态。

    Attributes:
        address:               拓扑地址（身份标识）
        degree:                当前度数（所有连接的总和）
        spatial_position:      三维空间坐标 (x, y, z)
        crystallite_connections: 到其他晶子的连接数（聚簇用）
        created_frame:         节点的创建帧编号

    Properties:
        virtual_faces: 剩余可连接虚面数（基于 degree 计算，容量上限 = 50）
        is_dangling:   是否为悬挂节点（degree < 2）
        is_saturated:  是否已饱和（degree ≥ 50，已达晶子容量上限）
        is_crystallite: 是否达到晶子阈值（degree ≥ 50）
        can_accept_crystallite_edge: 是否可接受晶子-晶子连接
        radius:        物理半径（基于 degree 增长）
    """

    address: TopologicalAddress
    degree: int = 0
    spatial_position: Optional[Tuple[float, float, float]] = None
    crystallite_connections: int = 0
    created_frame: int = -1

    # ------------------------------------------------------------
    # 计算属性
    # ------------------------------------------------------------
    @property
    def virtual_faces(self) -> int:
        """剩余可连接虚面数。

        上限来自晶子稳定解 (CRYSTALLITE_DEGREE_THRESHOLD = 42, T5)。
        度数增长 → 虚面减少；度数降低（湮灭）→ 虚面恢复。
        下限为 0（饱和态）。

        注意: 接吻数 12 是三维密堆的涌现结果，在此不作为度数上限使用。
        """
        return max(0, CRYSTALLITE_DEGREE_THRESHOLD - self.degree)

    @property
    def radius(self) -> float:
        """物理半径（基于度数增长）。

        公式: r = 0.5 × (1 + degree / CRYSTALLITE_DEGREE_THRESHOLD)
        - degree=0:    r = 0.5  (基础尺度 κ/2)
        - degree=12:   r ≈ 0.62 (接吻数涌现区)
        - degree=50:   r = 1.0  (晶子饱和)

        半径增长意味着粒子影响力域随连接数增加而扩大。
        球体相切条件 (distance = r₁ + r₂) 自然约束接吻数 ≤ 12。
        """
        return 0.5 * (1.0 + self.degree / CRYSTALLITE_DEGREE_THRESHOLD)

    @property
    def is_dangling(self) -> bool:
        """是否为悬挂节点（度数 < 2）。"""
        return self.degree < 2

    @property
    def is_saturated(self) -> bool:
        """是否已饱和（无剩余虚面，无法接受新连接）。

        饱和 = 度数 ≥ CRYSTALLITE_DEGREE_THRESHOLD (50)。
        """
        return self.virtual_faces <= 0

    @property
    def is_crystallite(self) -> bool:
        """是否已达到晶子阈值（度数 ≥ 50）。"""
        return self.degree >= CRYSTALLITE_DEGREE_THRESHOLD

    @property
    def can_accept_crystallite_edge(self) -> bool:
        """是否可接受晶子-晶子连接。

        条件: 自身已是晶子，且聚簇连接数未达上限 (12)。
        """
        return (self.is_crystallite
                and self.crystallite_connections < MAX_CRYSTALLITE_CLUSTER_SIZE)

    @property
    def can_accept_planar_edge(self) -> bool:
        """是否可接受晶子-晶子连接（平面三角剖分约束）。

        使用 min(MAX_CRYSTALLITE_CLUSTER_SIZE, MAX_CRYSTALLITE_PLANAR_DEGREE)
        作为实际限制，保证晶子子图保持平面性。
        """
        limit = min(MAX_CRYSTALLITE_CLUSTER_SIZE, MAX_CRYSTALLITE_PLANAR_DEGREE)
        return self.is_crystallite and self.crystallite_connections < limit


@dataclass
class NodeRegistry:
    """节点注册表 — 管理所有空间粒子的状态。

    被动响应式设计: 节点由 RelationPool.manifest_relation 触发创建，
    Registry 不主动生产节点。
    """

    nodes: Dict[str, NodeState] = field(default_factory=dict)

    # ------------------------------------------------------------
    # 节点生命周期
    # ------------------------------------------------------------
    def create_node(
        self,
        loc: TopologicalAddress,
        frame_number: int,
        position: Optional[Tuple[float, float, float]] = None,
    ) -> NodeState:
        """创建一个新节点。

        如果节点已存在，直接返回现有节点（幂等操作）。
        若指定 position，原点 (origin) 固定为 (0,0,0)。

        Args:
            loc:          节点的拓扑地址
            frame_number: 创建时的帧编号
            position:     三维空间坐标（可选，默认 None）

        Returns:
            新创建（或已存在）的 NodeState
        """
        if loc.uid in self.nodes:
            return self.nodes[loc.uid]

        # 原点固定位置 (0,0,0)
        if loc.uid == "origin" and position is None:
            position = (0.0, 0.0, 0.0)

        state = NodeState(
            address=loc,
            degree=0,
            spatial_position=position,
            created_frame=frame_number,
        )
        self.nodes[loc.uid] = state
        return state

    # ------------------------------------------------------------
    # 度数更新
    # ------------------------------------------------------------
    def update_degree(self, uid: str, delta: int) -> None:
        """更新指定节点的度数。

        virtual_faces 随之自动变化（基于 degree 计算），无需手动维护。

        Args:
            uid:   节点 UID
            delta: 度数变化量（+1 为增加边，-1 为减少边）

        Raises:
            KeyError: 节点不存在
        """
        if uid not in self.nodes:
            raise KeyError(f"节点 {uid} 不存在，无法更新度数")
        self.nodes[uid].degree += delta
        # 度数不应为负
        if self.nodes[uid].degree < 0:
            self.nodes[uid].degree = 0

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------
    def node_count(self) -> int:
        """返回节点总数。"""
        return len(self.nodes)

    def dangling_nodes(self) -> list[NodeState]:
        """返回所有悬挂节点（degree < 2）。"""
        return [n for n in self.nodes.values() if n.is_dangling]

    def saturated_nodes(self) -> list[NodeState]:
        """返回所有已饱和节点（virtual_faces == 0）。"""
        return [n for n in self.nodes.values() if n.is_saturated]

    def crystallite_nodes(self) -> list[NodeState]:
        """返回所有晶子节点（degree >= 42, T5 稳定解）。"""
        return [n for n in self.nodes.values() if n.is_crystallite]
