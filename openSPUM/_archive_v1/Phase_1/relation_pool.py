"""
关系池 — SPUM 确定性引擎的最高优先级存储。

绝对约束:
    - 无随机: 所有候选选择基于拓扑排序 (虚面数/度数/步数/UID)
    - 总边数守恒: 每次湮灭立即补偿创建
    - 关系先于节点: manifest_relation 自动调用 NodeRegistry.create_node
    - 空间几何约束: 节点度数受晶子容量 (50) 限制，饱和后拒绝新连接
    - 球体相切: 每新增关系均需通过三维几何相容性检查

设计原则:
    - 12（接吻数/拓扑常数 Σ(6−deg)=12）是涌现结果，非预设参数
    - 度数上限来自 CRYSTALLITE_DEGREE_THRESHOLD (50)，对应 N_max ≈ 50.3
    - 新节点位置由分化时相对目标的相切位置确定
    - 几何检查仅应用于两端都有空间位置的节点（向后兼容）
    - degree 上限硬约束为 50，晶子聚簇连接不增加 degree
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from math import sqrt
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from uuid import uuid4
import hashlib

from .topological_address import TopologicalAddress

if TYPE_CHECKING:
    from .node_registry import NodeRegistry, NodeState

from .constants import (
    GEOMETRIC_TOLERANCE,
    CRYSTALLITE_DEGREE_THRESHOLD,
    MAX_CRYSTALLITE_CLUSTER_SIZE,
    MAX_CRYSTALLITE_PLANAR_DEGREE,
    DANGLING_QUOTA,
    MIDRANGE_QUOTA,
    NEWNODE_QUOTA,
)


# ============================================================
# 枚举与数据类 (不变)
# ============================================================
class RelationState(Enum):
    LATENT = "latent"
    MANIFEST = "manifest"


class ContactType(Enum):
    REAL_REAL = "real_real"
    REAL_VIRTUAL = "real_virtual"
    VIRTUAL_VIRTUAL = "virtual_virtual"
    UNDEFINED = "undefined"


@dataclass
class Relation:
    """一条关系边（显化或潜在）。"""

    id: str
    loc1: TopologicalAddress
    loc2: TopologicalAddress
    state: RelationState = RelationState.LATENT
    contact_type: ContactType = ContactType.UNDEFINED
    weight: float = 0.0
    created_frame: int = -1

    def manifest(self, frame_number: int,
                 contact_type: ContactType = ContactType.REAL_REAL) -> None:
        self.state = RelationState.MANIFEST
        self.created_frame = frame_number
        self.contact_type = contact_type

    def annihilate(self) -> None:
        self.state = RelationState.LATENT
        self.created_frame = -1
        self.contact_type = ContactType.UNDEFINED

    @property
    def is_manifest(self) -> bool:
        return self.state == RelationState.MANIFEST

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Relation):
            return False
        return self.id == other.id


# ============================================================
# 空间几何工具
# ============================================================

def _uid_to_direction(uid: str) -> Tuple[float, float, float]:
    """从 UID 确定性地映射到单位球面上的方向向量。"""
    import math as _m
    h = hashlib.sha256(uid.encode()).digest()
    theta = (int.from_bytes(h[:4], "big") / 2 ** 32) * 6.283185307179586
    phi = (int.from_bytes(h[4:8], "big") / 2 ** 32) * 3.141592653589793
    sin_phi = _m.sin(phi)
    return (
        sin_phi * _m.cos(theta),
        sin_phi * _m.sin(theta),
        _m.cos(phi),
    )


def _distance(pos1: Tuple[float, float, float],
              pos2: Tuple[float, float, float]) -> float:
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    dz = pos1[2] - pos2[2]
    return sqrt(dx * dx + dy * dy + dz * dz)


def _are_tangent(node1: "NodeState", node2: "NodeState") -> bool:
    """检查两个节点在三维空间中是否近似相切。

    仅当两端都有空间位置时执行检查。
    无位置信息的节点视为"几何透明"——总是返回 True。
    """
    p1, p2 = node1.spatial_position, node2.spatial_position
    if p1 is None or p2 is None:
        return True
    dist = _distance(p1, p2)
    expected = node1.radius + node2.radius
    if expected < 1e-10:
        return dist < 1e-10
    return abs(dist - expected) / expected < GEOMETRIC_TOLERANCE


def _compute_propagation_direction(
    source_node: "NodeState",
    node_registry: "NodeRegistry",
    relation_pool: "RelationPool",
) -> Tuple[float, float, float]:
    """确定从 source_node 出发的传播方向向量。

    策略:
        1. 如果 source 有晶体邻居 → 指向最高连接数的晶体邻居
        2. 如果 source 有高度数 (>2) 邻居 → 指向最高度数邻居
        3. 否则 → 使用 source 的位置向量本身（从原点出发方向）
        4. 回退 → (0, 0, 1)

    Returns:
        归一化方向向量 (x, y, z)
    """
    sp = source_node.spatial_position
    if sp is None:
        return (0.0, 0.0, 1.0)

    # 找有效邻居
    neighbors: List[Tuple[float, str]] = []  # (degree, uid)
    for (ua, ub) in relation_pool.manifest:
        if ua == source_node.address.uid:
            nb = node_registry.nodes.get(ub)
            if nb and nb.spatial_position:
                neighbors.append((nb.degree, ub, nb.spatial_position))
        elif ub == source_node.address.uid:
            nb = node_registry.nodes.get(ua)
            if nb and nb.spatial_position:
                neighbors.append((nb.degree, ua, nb.spatial_position))

    if not neighbors:
        # 无邻居 → 从原点指向 source 的方向
        mag = sqrt(sp[0]**2 + sp[1]**2 + sp[2]**2)
        if mag < 1e-10:
            return (0.0, 0.0, 1.0)
        return (sp[0]/mag, sp[1]/mag, sp[2]/mag)

    # 选最高度数邻居指向方向
    neighbors.sort(key=lambda x: -x[0])
    _, _, nb_pos = neighbors[0]
    dx = nb_pos[0] - sp[0]
    dy = nb_pos[1] - sp[1]
    dz = nb_pos[2] - sp[2]
    mag = sqrt(dx*dx + dy*dy + dz*dz)
    if mag < 1e-10:
        return (0.0, 0.0, 1.0)
    return (dx/mag, dy/mag, dz/mag)


# ============================================================
# 确定性关系池
# ============================================================
@dataclass
class RelationPool:
    """确定性关系池 — SPUM 引擎的最高优先级存储。

    核心设计:
        - manifest: Dict[(uid1, uid2), Relation] — 显化边存储
        - 边键始终按 uid 排序（小 → 大），保证唯一性
        - 候选生成由虚面约束 + 拓扑排序键决定，无随机
        - 饱和节点（degree ≥ 50）不再接受新连接
        - 新节点在分化时自动获得三维空间位置（相切于目标）
        - manifest_relation 中验证球体相切性
        - 晶子聚簇连接 (manifest_cluster_edge) 不增加 degree

    拓扑常数 12 是涌现结果:
        - 不在此处硬编码为度数上限
        - 三维球体堆叠自然限制接吻数 ≤ 12
    """

    manifest: Dict[Tuple[str, str], Relation] = field(default_factory=dict)

    # 聚类边键集 — 不参与度数守恒计算
    _cluster_edge_keys: set = field(default_factory=set)

    # 从常规边升级的聚簇边数量 — 用于度数守恒校验
    _upgraded_edge_count: int = 0

    # 待定空间位置：为新节点预留的相切位置
    _pending_positions: Dict[str, Tuple[float, float, float]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------
    # 分层扩散候选生成
    # ------------------------------------------------------------
    def deterministic_candidates(
        self,
        node_registry: NodeRegistry,
        count: int,
        layer: str = "all",
        exclude_keys: Optional[Set[Tuple[str, str]]] = None,
    ) -> List[Tuple[TopologicalAddress, TopologicalAddress]]:
        """
        按确定性优先级返回 count 对候选地址。

        支持分层扩散模式:
            - layer="all" (默认): 原有行为，适合湮灭补偿
            - layer="dangling":  只配对 degree<2 的悬挂节点
            - layer="midrange":  只配对 degree 2-49 的节点
            - layer="new_nodes": 从最高虚面子分化新节点

        Args:
            node_registry: 节点注册表
            count:         需要的候选对数
            layer:         分层模式
            exclude_keys:  排除的键集（刚湮灭的边不重建）

        Returns:
            最多 count 对 (地址, 地址) 候选
        """
        if count <= 0:
            return []

        nodes = list(node_registry.nodes.values())
        origin_addr = TopologicalAddress.origin()
        candidates: List[Tuple[TopologicalAddress, TopologicalAddress]] = []
        used_keys: set = set()
        if exclude_keys:
            used_keys.update(exclude_keys)

        # ---- Layer: new_nodes - 创建新节点 ----
        if layer == "new_nodes":
            return self._candidates_new_nodes(node_registry, count)

        # ---- Layer: dangling - 悬挂节点配给最佳未饱和节点 ----
        if layer == "dangling":
            dangling = sorted(
                [n for n in nodes if n.degree < 2 and not n.is_saturated],
                key=lambda n: (n.degree, n.address.uid),  # degree=0 优先
            )
            if not dangling:
                return []
            partners = sorted(
                [n for n in nodes if not n.is_saturated],
                key=lambda n: (-n.degree, -n.virtual_faces, n.address.uid),
            )
            for d_node in dangling:
                if len(candidates) >= count:
                    break
                for p_node in partners:
                    if d_node.address.uid == p_node.address.uid:
                        continue
                    key = self._make_key(d_node.address, p_node.address)
                    if key not in self.manifest and key not in used_keys:
                        candidates.append((d_node.address, p_node.address))
                        used_keys.add(key)
                        break
            return candidates[:count]

        # ---- Layer: midrange - 中段节点"高低配对" ----
        if layer == "midrange":
            # 高段候选: degree 2-49 按高度数优先（即将成为晶子的节点）
            high_candidates = sorted(
                [n for n in nodes
                 if 2 <= n.degree < CRYSTALLITE_DEGREE_THRESHOLD
                 and not n.is_saturated],
                key=lambda n: (-n.degree, -n.virtual_faces, n.address.uid),
            )
            # 低段候选: 所有未饱和节点按低度数优先
            low_candidates = sorted(
                [n for n in nodes if not n.is_saturated],
                key=lambda n: (n.degree, n.address.uid),
            )
            if not high_candidates or not low_candidates:
                return []

            used_uids = set()
            for h_node in high_candidates:
                if len(candidates) >= count:
                    break
                if h_node.address.uid in used_uids:
                    continue
                # 找最低度数的可用伙伴（且不是自己）
                for l_node in low_candidates:
                    if l_node.address.uid in used_uids:
                        continue
                    if l_node.address.uid == h_node.address.uid:
                        continue
                    key = self._make_key(h_node.address, l_node.address)
                    if key in self.manifest or key in used_keys:
                        continue
                    candidates.append((h_node.address, l_node.address))
                    used_keys.add(key)
                    used_uids.add(h_node.address.uid)
                    used_uids.add(l_node.address.uid)
                    break

            # 不足时: 晶子配给中段节点（单向拉高）
            while len(candidates) < count:
                crystallites_with_planar = sorted(
                    [n for n in nodes if n.can_accept_planar_edge],
                    key=lambda n: (-n.virtual_faces, n.address.uid),
                )
                remaining_mid = [
                    n for n in high_candidates
                    if n.address.uid not in used_uids
                ]
                if not crystallites_with_planar or not remaining_mid:
                    break
                c_node = crystallites_with_planar[0]
                m_node = remaining_mid[0]
                key = self._make_key(c_node.address, m_node.address)
                if key not in self.manifest and key not in used_keys:
                    candidates.append((c_node.address, m_node.address))
                    used_keys.add(key)
                    used_uids.add(c_node.address.uid)
                    used_uids.add(m_node.address.uid)
                else:
                    break

            return candidates[:count]

        # ---- Layer: all - 所有未饱和节点配对（湮灭补偿用） ----
        #  排序策略: degree<2 节点优先 (补偿专用优先级)
        #  → 补偿边更可能连接低度数节点 → 产生新悬挂 → 跨帧传播
        #  在 degree<2 内部: degree 降序 → deg-1 (网络) 先于 deg-0 (气泡)
        if layer == "all":
            eligible = sorted(
                [n for n in nodes if not n.is_saturated],
                key=lambda n: (
                    0 if n.degree < 2 else 1,   # [alpha] 悬挂优先
                    -n.degree if n.degree < 2 else 0,  # 悬挂内: deg-1 先于 deg-0
                    -n.virtual_faces if n.degree >= 2 else 0,  # 稳定内: 高虚面优先
                    n.address.differentiation_step,
                    n.address.uid,
                ),
            )
            for i, node_a in enumerate(eligible):
                if len(candidates) >= count:
                    break
                for node_b in eligible[i + 1:]:
                    if len(candidates) >= count:
                        break
                    key = self._make_key(node_a.address, node_b.address)
                    if key not in self.manifest and key not in used_keys:
                        candidates.append((node_a.address, node_b.address))
                        used_keys.add(key)

        # Step 2 (仅 "all" 模式): 不足时创建新节点
        if layer == "all":
            while len(candidates) < count:
                new_pairs = self._candidates_new_nodes(
                    node_registry, count - len(candidates)
                )
                if not new_pairs:
                    break
                candidates.extend(new_pairs)

        return candidates[:count]

    def _candidates_new_nodes(
        self,
        node_registry: NodeRegistry,
        count: int,
    ) -> List[Tuple[TopologicalAddress, TopologicalAddress]]:
        """从最高虚面节点分化新节点。

        Returns:
            最多 count 对 (target, new_node) 候选
        """
        origin_addr = TopologicalAddress.origin()
        candidates = []
        target_nodes = [
            n for n in node_registry.nodes.values()
            if not n.is_saturated
        ]
        if not target_nodes:
            return []

        target_nodes.sort(
            key=lambda n: (n.virtual_faces, n.degree),
            reverse=True,
        )

        for target in target_nodes:
            if len(candidates) >= count:
                break
            new_loc = TopologicalAddress.differentiate_from(origin_addr)

            if target.spatial_position is not None:
                direction = _uid_to_direction(new_loc.uid)
                new_radius = 0.5
                dist = target.radius + new_radius
                pos = (
                    target.spatial_position[0] + dist * direction[0],
                    target.spatial_position[1] + dist * direction[1],
                    target.spatial_position[2] + dist * direction[2],
                )
                self._pending_positions[new_loc.uid] = pos

            candidates.append((target.address, new_loc))

        return candidates[:count]

    # ------------------------------------------------------------
    # 显关系操作
    # ------------------------------------------------------------
    def manifest_relation(
        self,
        loc1: TopologicalAddress,
        loc2: TopologicalAddress,
        frame_number: int,
        node_registry: NodeRegistry,
        contact_type: Optional[ContactType] = None,
    ) -> Optional[Relation]:
        """关系显化 (确定性实现 + 空间几何约束)。

        流程:
            1. 自环检查 → 拒绝
            2. 重复检查 → 返回现有
            3. 确保节点存在 (若有预留位置则使用)
            4. 饱和检查 (degree ≥ 50 → 拒绝)
            5. 空间几何检查 (球体相切)
            6. 自动决定接触类型
            7. 创建并存储 Relation
        """
        if loc1.uid == loc2.uid:
            return None

        key = self._make_key(loc1, loc2)

        if key in self.manifest:
            return self.manifest[key]

        for loc in (loc1, loc2):
            if loc.uid not in node_registry.nodes:
                position = self._pending_positions.pop(loc.uid, None)
                node_registry.create_node(loc, frame_number, position=position)

        node1 = node_registry.nodes[loc1.uid]
        node2 = node_registry.nodes[loc2.uid]
        if node1.is_saturated or node2.is_saturated:
            self._pending_positions.pop(loc1.uid, None)
            self._pending_positions.pop(loc2.uid, None)
            return None

        if not _are_tangent(node1, node2):
            return None

        if contact_type is None:
            v1 = node1.virtual_faces
            v2 = node2.virtual_faces
            if v1 == 0 and v2 == 0:
                contact_type = ContactType.REAL_REAL
            elif v1 == 0 or v2 == 0:
                contact_type = ContactType.REAL_VIRTUAL
            else:
                contact_type = ContactType.VIRTUAL_VIRTUAL

        rel = Relation(id=uuid4().hex, loc1=loc1, loc2=loc2)
        rel.manifest(frame_number, contact_type)
        self.manifest[key] = rel

        node_registry.update_degree(loc1.uid, +1)
        node_registry.update_degree(loc2.uid, +1)

        return rel

    def annihilate_relation(
        self,
        edge_key: Tuple[str, str],
        node_registry: NodeRegistry,
        frame_number: int,
    ) -> Optional[Relation]:
        """湮灭一条边，并立即尝试补偿创建一条新边。"""
        if edge_key not in self.manifest:
            reverse_key = (edge_key[1], edge_key[0])
            if reverse_key not in self.manifest:
                return None
            edge_key = reverse_key

        rel = self.manifest.pop(edge_key)
        is_cluster = edge_key in self._cluster_edge_keys
        if is_cluster:
            self._cluster_edge_keys.discard(edge_key)
            # 聚簇边（包括从常规边升级的）：减少 crystallite_connections
            if rel.loc1.uid in node_registry.nodes:
                node_registry.nodes[rel.loc1.uid].crystallite_connections -= 1
            if rel.loc2.uid in node_registry.nodes:
                node_registry.nodes[rel.loc2.uid].crystallite_connections -= 1
            # 升级边原本也有 degree 贡献
            if self._upgraded_edge_count > 0:
                self._upgraded_edge_count -= 1
            node_registry.update_degree(rel.loc1.uid, -1)
            node_registry.update_degree(rel.loc2.uid, -1)
        else:
            node_registry.update_degree(rel.loc1.uid, -1)
            node_registry.update_degree(rel.loc2.uid, -1)

        rel.annihilate()

        candidates = self.deterministic_candidates(
            node_registry, count=1, layer="all"
        )
        if candidates:
            loc_a, loc_b = candidates[0]
            self.manifest_relation(loc_a, loc_b, frame_number, node_registry)

        return rel

    # ------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------
    def has_edge(self, loc1: TopologicalAddress, loc2: TopologicalAddress) -> bool:
        key = self._make_key(loc1, loc2)
        return key in self.manifest

    def get_edge(self, loc1: TopologicalAddress,
                 loc2: TopologicalAddress) -> Optional[Relation]:
        key = self._make_key(loc1, loc2)
        if key in self.manifest:
            return self.manifest[key]
        reverse_key = (key[1], key[0])
        return self.manifest.get(reverse_key)

    def edge_count(self) -> int:
        return len(self.manifest)

    # ------------------------------------------------------------
    # 定向传播候选生成（用于 α 测量中的定向级联）
    # ------------------------------------------------------------
    def find_directional_candidates(
        self,
        node_registry: NodeRegistry,
        source_uid: str,
        count: int = 3,
        forward_hemisphere_only: bool = True,
        exclude_keys: Optional[Set[Tuple[str, str]]] = None,
    ) -> List[Tuple[TopologicalAddress, TopologicalAddress]]:
        """定向传播补偿候选 — 优先选择扰动源前方区域的节点。

        机制:
            - 从 source_uid 确定"传播方向"（指向其最高度数邻居的方向）
            - 过滤 eligible 节点，偏好在源的前方半球面的节点
            - 在优先区域内按拓扑优先级排序（虚面数/度数/UID）

        Args:
            node_registry:        节点注册表
            source_uid:           扰动源 UID
            count:                需要的候选对数
            forward_hemisphere_only: 是否仅选前方半球
            exclude_keys:         排除的键集（刚湮灭的边不重建）

        Returns:
            最多 count 对 (地址, 地址)
        """
        source_node = node_registry.nodes.get(source_uid)
        if source_node is None:
            return self.deterministic_candidates(
                node_registry, count, layer="all",
                exclude_keys=exclude_keys,
            )

        # 确定传播方向向量（无位置时回退到均匀选择）
        if source_node.spatial_position is None:
            return self.deterministic_candidates(
                node_registry, count, layer="all",
                exclude_keys=exclude_keys,
            )
        dir_vec = _compute_propagation_direction(
            source_node, node_registry, self
        )

        # 所有未饱和节点按方向评分排序
        eligible = [
            n for n in node_registry.nodes.values()
            if not n.is_saturated
            and n.spatial_position is not None
            and n.address.uid != source_uid
        ]

        if not eligible:
            return self.deterministic_candidates(
                node_registry, count, layer="all",
                exclude_keys=exclude_keys,
            )

        # 对每个节点计算方向评分 & 拓扑评分
        def _score(node: "NodeState") -> Tuple[float, int, int, str]:
            # 方向评分: 在向前方向上的投影
            dx = node.spatial_position[0] - source_node.spatial_position[0]
            dy = node.spatial_position[1] - source_node.spatial_position[1]
            dz = node.spatial_position[2] - source_node.spatial_position[2]
            mag = sqrt(dx*dx + dy*dy + dz*dz)
            if mag < 1e-10:
                fwd_score = 0.0
            else:
                fwd_score = (dx*dir_vec[0] + dy*dir_vec[1] + dz*dir_vec[2]) / mag

            if forward_hemisphere_only and fwd_score <= 0:
                fwd_score = -999.0  # 排除后方

            # 拓扑评分: degree<2 优先 (补偿专用优先级)
            # 在 degree<2 内部: degree 降序 → deg-1 (网络) 先于 deg-0 (气泡)
            return (0 if node.degree < 2 else 1,
                    -node.degree if node.degree < 2 else 0,
                    -fwd_score,
                    node.address.uid)

        eligible.sort(key=_score)

        # 配对：最高分节点之间互连
        candidates: List[Tuple[TopologicalAddress, TopologicalAddress]] = []
        used_keys: set = set()
        if exclude_keys:
            used_keys.update(exclude_keys)

        for i, n1 in enumerate(eligible):
            if len(candidates) >= count:
                break
            for n2 in eligible[i + 1:]:
                if len(candidates) >= count:
                    break
                key = self._make_key(n1.address, n2.address)
                if key not in self.manifest and key not in used_keys:
                    candidates.append((n1.address, n2.address))
                    used_keys.add(key)

        return candidates[:count]

    # ------------------------------------------------------------
    # 晶子聚簇连接
    # ------------------------------------------------------------
    def find_cluster_candidates(
        self,
        node_registry: NodeRegistry,
        count: int = 10,
    ) -> List[Tuple[TopologicalAddress, TopologicalAddress]]:
        """查找可连接的晶子对，用于聚簇三角剖分。

        选取策略:
            - 两端均为晶子 (degree ≥ 50)
            - 两端均有空余晶子连接槽位 (< 12)
            - 空间位置相切
            - 尚无显化边
            按空间距离升序排序（最近优先）。
        """
        crystallites = [
            n for n in node_registry.nodes.values()
            if n.can_accept_planar_edge
        ]
        if len(crystallites) < 2:
            return []

        pairs = []
        for i, n1 in enumerate(crystallites):
            for n2 in crystallites[i + 1:]:
                key = self._make_key(n1.address, n2.address)
                # 仅检查聚簇边存在性（已有聚簇边才跳过）
                if key in self._cluster_edge_keys:
                    continue
                if _are_tangent(n1, n2):
                    if n1.spatial_position and n2.spatial_position:
                        dist = _distance(n1.spatial_position, n2.spatial_position)
                    else:
                        dist = float("inf")
                    pairs.append((dist, n1.address, n2.address))

        pairs.sort(key=lambda x: x[0])
        return [(a, b) for _, a, b in pairs[:count]]

    def find_triangulation_candidates(
        self,
        node_registry: NodeRegistry,
        max_pairs: int = 10,
    ) -> List[Tuple[TopologicalAddress, TopologicalAddress]]:
        """查找三角剖分补全候选晶子对。

        选取策略:
            - 两端均为晶子且有平面槽位（crystallite_connections < 6）
            - 尚无显化边
            - 按共享邻居数降序排列（补全缺失三角面的对角线）
            - 空间相切检查

        这确保了聚簇逐渐向完整三角剖分趋近。
        """
        from collections import defaultdict

        eligible = [
            n for n in node_registry.nodes.values()
            if n.is_crystallite
            and n.crystallite_connections < MAX_CRYSTALLITE_PLANAR_DEGREE
        ]
        if len(eligible) < 2:
            return []

        # 构建聚簇边邻接表
        adj: Dict[str, set] = defaultdict(set)
        for key in self._cluster_edge_keys:
            adj[key[0]].add(key[1])
            adj[key[1]].add(key[0])

        # 对每对未连接的晶子计算共享邻居数
        scored: List[Tuple[int, TopologicalAddress, TopologicalAddress]] = []
        for i, n1 in enumerate(eligible):
            n1_neighbors = adj.get(n1.address.uid, set())
            for n2 in eligible[i + 1:]:
                key = self._make_key(n1.address, n2.address)
                # 仅检查聚簇边存在性（常规边已占满所有晶子对）
                if key in self._cluster_edge_keys:
                    continue
                n2_neighbors = adj.get(n2.address.uid, set())
                shared = len(n1_neighbors & n2_neighbors)
                if shared > 0 and _are_tangent(n1, n2):
                    scored.append((-shared, n1.address, n2.address))

        scored.sort(key=lambda x: x[0])
        return [(a, b) for _, a, b in scored[:max_pairs]]

    def manifest_cluster_edge(
        self,
        loc1: TopologicalAddress,
        loc2: TopologicalAddress,
        frame_number: int,
        node_registry: NodeRegistry,
    ) -> Optional[Relation]:
        """创建晶子-晶子聚簇边。

        与 manifest_relation 的关键区别:
            - 不增加 degree（保持 degree ≤ 50 硬约束）
            - 只增加 crystallite_connections（上限 12）
            - 跳过 virtual_faces 饱和检查（已是晶子）
        """
        if loc1.uid == loc2.uid:
            return None

        key = self._make_key(loc1, loc2)

        # 已有聚簇边 → 直接返回
        if key in self._cluster_edge_keys:
            return self.manifest.get(key)

        # 已有常规边 → 升级为聚簇边追踪
        # （保持常规边的 degree，仅增加 crystallite_connections）
        if key in self.manifest:
            existing = self.manifest[key]
            self._cluster_edge_keys.add(key)
            self._upgraded_edge_count += 1
            for uid in (loc1.uid, loc2.uid):
                if uid in node_registry.nodes:
                    node_registry.nodes[uid].crystallite_connections += 1
            return existing

        for loc in (loc1, loc2):
            if loc.uid not in node_registry.nodes:
                position = self._pending_positions.pop(loc.uid, None)
                node_registry.create_node(loc, frame_number, position=position)

        node1 = node_registry.nodes[loc1.uid]
        node2 = node_registry.nodes[loc2.uid]

        if not node1.can_accept_planar_edge:
            return None
        if not node2.can_accept_planar_edge:
            return None
        if not _are_tangent(node1, node2):
            return None

        rel = Relation(id=uuid4().hex, loc1=loc1, loc2=loc2)
        rel.manifest(frame_number, ContactType.REAL_REAL)
        self.manifest[key] = rel
        self._cluster_edge_keys.add(key)

        # 关键：不增加 degree！只追踪聚簇连接数
        node1.crystallite_connections += 1
        node2.crystallite_connections += 1

        return rel

    def edge_count(self) -> int:
        """返回常规显化边总数（不含聚簇边）。"""
        return len(self.manifest) - len(self._cluster_edge_keys)

    def cluster_edge_count(self) -> int:
        """返回聚簇边总数。"""
        return len(self._cluster_edge_keys)

    def upstreamed_edge_count(self) -> int:
        """返回从常规边升级为聚簇边的数量。"""
        return self._upgraded_edge_count

    def is_cluster_edge(self, key: Tuple[str, str]) -> bool:
        """检查指定边键是否为聚簇边。"""
        return key in self._cluster_edge_keys or (key[1], key[0]) in self._cluster_edge_keys

    @staticmethod
    def _make_key(loc1: TopologicalAddress,
                  loc2: TopologicalAddress) -> Tuple[str, str]:
        if loc1.uid < loc2.uid:
            return (loc1.uid, loc2.uid)
        return (loc2.uid, loc1.uid)
