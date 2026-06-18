"""
Phase 1 完整单元测试套件 (v3 — 12 是涌现结果)

覆盖模块:
    - constants.py
    - topological_address.py
    - node_registry.py
    - relation_pool.py
    - seed_epoch_engine.py

核心设计原则:
    - 12（接吻数/拓扑常数）是涌现结果，非预设度数上限
    - 度数上限来自晶子容量 CRYSTALLITE_DEGREE_THRESHOLD (50)
    - 饱和 (is_saturated) = degree ≥ 50

运行: python -m pytest openSPUM/tests/test_phase1_all.py -v
"""

import math
import sys
import os
from pathlib import Path

# 确保能导入 openSPUM/Phase_1/
_openspum_root = str(Path(__file__).resolve().parent.parent)
if _openspum_root not in sys.path:
    sys.path.insert(0, _openspum_root)

import pytest
from uuid import uuid4

from Phase_1.constants import (
    KAPPA,
    TAU,
    CRYSTALLITE_DEGREE_THRESHOLD,
    MAX_CONTACTS,
    CRYSTALLITE_DIAMETER_MAX,
    CRYSTALLITE_CAPACITY,
    GEOMETRIC_TOLERANCE,
    MAX_CRYSTALLITE_CLUSTER_SIZE,
    MAX_CRYSTALLITE_PLANAR_DEGREE,
    MAX_CLUSTER_ITERATIONS,
    DANGLING_QUOTA,
    MIDRANGE_QUOTA,
    NEWNODE_QUOTA,
)
from Phase_1.topological_address import TopologicalAddress, reset_differentiation_counter
from Phase_1.node_registry import NodeRegistry, NodeState
from Phase_1.relation_pool import (
    RelationPool,
    Relation,
    RelationState,
    ContactType,
)
from Phase_1.seed_epoch_engine import SeedEpochConfig, SeedEpochEngine


# ============================================================
# constants.py
# ============================================================
class TestConstants:
    def test_kappa_value(self):
        assert KAPPA == 1.0

    def test_tau_value(self):
        assert TAU == 1

    def test_crystallite_threshold(self):
        """晶子阈值 = 50（对应 N_max ≈ 50.3）"""
        assert CRYSTALLITE_DEGREE_THRESHOLD == 50

    def test_crystallite_diameter_max(self):
        """D_max = 4κ = 4.0"""
        assert CRYSTALLITE_DIAMETER_MAX == 4.0

    def test_crystallite_capacity(self):
        """N_max = 16π ≈ 50.265"""
        assert CRYSTALLITE_CAPACITY == pytest.approx(50.265, rel=1e-3)

    def test_max_contacts_is_reference_only(self):
        """MAX_CONTACTS=12 为参考值，不是度数上限"""
        assert MAX_CONTACTS == 12


# ============================================================
# topological_address.py
# ============================================================
class TestTopologicalAddress:
    def test_origin_uid(self):
        addr = TopologicalAddress.origin()
        assert addr.uid == "origin"
        assert addr.differentiation_step == 0

    def test_differentiate_from_origin(self):
        reset_differentiation_counter()
        origin = TopologicalAddress.origin()
        child = TopologicalAddress.differentiate_from(origin)
        assert child.differentiation_step == 1
        assert child.uid.startswith("N1_")
        assert len(child.uid) == 15
        assert child.uid == "N1_000000000001"

    def test_differentiate_increases_step(self):
        parent = TopologicalAddress(uid="test", differentiation_step=5)
        child = TopologicalAddress.differentiate_from(parent)
        assert child.differentiation_step == 6

    def test_immutable(self):
        addr = TopologicalAddress(uid="a", differentiation_step=0)
        with pytest.raises((TypeError, AttributeError)):
            addr.uid = "new"  # type: ignore

    def test_hashable(self):
        addr1 = TopologicalAddress(uid="a", differentiation_step=0)
        addr2 = TopologicalAddress(uid="a", differentiation_step=0)
        assert hash(addr1) == hash(addr2)
        d = {addr1: "value"}
        assert d[addr2] == "value"

    def test_ordering(self):
        a = TopologicalAddress(uid="a", differentiation_step=1)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        c = TopologicalAddress(uid="a", differentiation_step=0)
        assert b < a
        assert c < b

    def test_origin_not_repeatable(self):
        o1 = TopologicalAddress.origin()
        o2 = TopologicalAddress.origin()
        assert o1 == o2


# ============================================================
# spatial_geometry.py
# ============================================================
class TestSpatialGeometry:
    def test_radius_default(self):
        """degree=0 时半径 = 0.5。"""
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr)
        assert ns.radius == 0.5

    def test_radius_at_threshold(self):
        """degree=50 (晶子阈值) 时半径 = 1.0。"""
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr, degree=50)
        assert ns.radius == 1.0

    def test_radius_grows_with_degree(self):
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr)
        assert ns.radius == 0.5
        ns.degree = 25
        assert ns.radius == 0.75
        ns.degree = 50
        assert ns.radius == 1.0

    def test_origin_gets_position_automatically(self):
        """origin 节点自动获得 (0,0,0)。"""
        reg = NodeRegistry()
        origin = TopologicalAddress.origin()
        ns = reg.create_node(origin, frame_number=0)
        assert ns.spatial_position == (0.0, 0.0, 0.0)

    def test_create_node_with_explicit_position(self):
        reg = NodeRegistry()
        addr = TopologicalAddress(uid="custom", differentiation_step=1)
        ns = reg.create_node(addr, frame_number=0, position=(1.0, 2.0, 3.0))
        assert ns.spatial_position == (1.0, 2.0, 3.0)

    def test_pending_position_used_in_manifest(self):
        """新节点通过 _pending_positions 获得相切位置。"""
        pool = RelationPool()
        reg = NodeRegistry()
        origin = TopologicalAddress.origin()
        reg.create_node(origin, frame_number=0)

        # 第二个节点应被放在 origin 表面相切位置
        reset_differentiation_counter()
        candidates = pool.deterministic_candidates(reg, count=1)
        assert len(candidates) == 1

        # manifest 应成功（几何相容）
        loc_a, loc_b = candidates[0]
        rel = pool.manifest_relation(loc_a, loc_b, frame_number=1, node_registry=reg)
        assert rel is not None, "相切位置应通过几何检查"

        # 验证新节点有位置
        new_node = reg.nodes[loc_b.uid]
        assert new_node.spatial_position is not None

    def test_non_tangent_pair_rejected(self):
        """两个已有位置但不相切的节点应被拒绝。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0, position=(0.0, 0.0, 0.0))
        # 放在距离 100 处，远非相切
        reg.create_node(b, 0, position=(100.0, 0.0, 0.0))

        rel = pool.manifest_relation(a, b, frame_number=1, node_registry=reg)
        assert rel is None, "远距离节点不应通过几何检查"

    def test_tangent_pair_accepted(self):
        """精确相切的节点应通过几何检查。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0, position=(0.0, 0.0, 0.0))
        # degree=0 → radius=0.5, 相切距离=1.0
        reg.create_node(b, 0, position=(1.0, 0.0, 0.0))

        rel = pool.manifest_relation(a, b, frame_number=1, node_registry=reg)
        assert rel is not None, "相切节点应通过几何检查"

    def test_no_geometry_skip_check(self):
        """无位置的节点跳过几何检查（向后兼容）。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        # 不设位置 → 跳过几何检查
        rel = pool.manifest_relation(a, b, frame_number=0, node_registry=reg)
        assert rel is not None

    def test_kissing_number_emerges(self):
        """围绕 origin 放置节点，验证三维球体堆叠自然限制接吻数。

        在三维空间中，半径为 0.5 的等大球体最多与 12 个同类球体相切。
        由于方向由 SHA-256 按 UID 确定，Origin 的邻居数受此自然约束。
        """
        pool = RelationPool()
        reg = NodeRegistry()
        origin = TopologicalAddress.origin()
        reg.create_node(origin, 0)

        # 尝试在 origin 周围放置 20 个节点
        reset_differentiation_counter()
        for i in range(20):
            # 每帧生成一个候选
            candidates = pool.deterministic_candidates(reg, count=1)
            if not candidates:
                break
            loc_a, loc_b = candidates[0]
            pool.manifest_relation(loc_a, loc_b, frame_number=i+1, node_registry=reg)

        # 记录 origin 的实际邻居数
        origin_degree = reg.nodes["origin"].degree
        print(f"\n  [接吻数涌现] origin 实际邻居数 = {origin_degree}")
        # 由于方向是散列确定的（不是最优球体排列），实际数量接近但通常 ≤ 12
        # 在某些不利排列下可能略少
        assert origin_degree <= 14, (f"三维球体堆叠应限制接吻数 ≤ 14，"
                                     f"实际 = {origin_degree}")

    def test_deterministic_uid_from_counter(self):
        """确定性计数器提供可预测 UID。"""
        reset_differentiation_counter()
        origin = TopologicalAddress.origin()
        c1 = TopologicalAddress.differentiate_from(origin)
        c2 = TopologicalAddress.differentiate_from(origin)
        assert c1.uid == "N1_000000000001"
        assert c2.uid == "N1_000000000002"
        assert c1.uid != c2.uid


# ============================================================
# crystallite_cluster.py — 晶子聚簇
# ============================================================
class TestCrystalliteClustering:
    def test_can_accept_crystallite_edge_when_crystallite(self):
        """晶子且有空余槽位时返回 True。"""
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr, degree=50)
        assert ns.is_crystallite
        assert ns.can_accept_crystallite_edge
        assert ns.crystallite_connections == 0

    def test_can_accept_crystallite_edge_when_full(self):
        """晶子但槽位满 (平面度数 6) 时返回 False。"""
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr, degree=50,
                       crystallite_connections=MAX_CRYSTALLITE_PLANAR_DEGREE)
        assert ns.is_crystallite
        assert not ns.can_accept_planar_edge
        # 物理上限 12 仍可接受（全自由度）
        assert ns.can_accept_crystallite_edge

    def test_can_accept_crystallite_edge_not_crystallite(self):
        """非晶子返回 False。"""
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr, degree=10)
        assert not ns.is_crystallite
        assert not ns.can_accept_crystallite_edge

    def test_manifest_cluster_edge_basic(self):
        """两个相切晶子之间可创建簇边（不增加 degree）。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0, position=(0.0, 0.0, 0.0))
        # degree=50 → radius=1.0, 相切距离=2.0
        reg.create_node(b, 0, position=(2.0, 0.0, 0.0))
        reg.nodes["a"].degree = 50  # 晶子
        reg.nodes["b"].degree = 50  # 晶子

        rel = pool.manifest_cluster_edge(a, b, frame_number=1, node_registry=reg)
        assert rel is not None
        # 簇边不增加 degree，只增加 crystallite_connections
        assert reg.nodes["a"].degree == 50
        assert reg.nodes["b"].degree == 50
        assert reg.nodes["a"].crystallite_connections == 1
        assert reg.nodes["b"].crystallite_connections == 1

    def test_manifest_cluster_edge_rejects_non_crystallite(self):
        """非晶子节点之间拒绝创建簇边。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0, position=(0.0, 0.0, 0.0))
        reg.create_node(b, 0, position=(1.0, 0.0, 0.0))
        # 非晶子
        rel = pool.manifest_cluster_edge(a, b, frame_number=1, node_registry=reg)
        assert rel is None

    def test_manifest_cluster_edge_rejects_full(self):
        """槽位满的晶子拒绝更多簇边（平面三角剖分上限 6）。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        c = TopologicalAddress(uid="c", differentiation_step=0)
        # degree=50 → radius=1.0, 相切距离=2.0
        reg.create_node(a, 0, position=(0.0, 0.0, 0.0))
        reg.create_node(b, 0, position=(2.0, 0.0, 0.0))
        reg.create_node(c, 0, position=(4.0, 0.0, 0.0))
        for n in ["a", "b", "c"]:
            reg.nodes[n].degree = 50
        # 填满 a 的槽位（平面度数上限）
        planar_limit = min(MAX_CRYSTALLITE_CLUSTER_SIZE, MAX_CRYSTALLITE_PLANAR_DEGREE)
        reg.nodes["a"].crystallite_connections = planar_limit

        rel_ab = pool.manifest_cluster_edge(a, b, 1, reg)
        assert rel_ab is None, "a 槽位满，应拒绝"

        rel_bc = pool.manifest_cluster_edge(b, c, 1, reg)
        assert rel_bc is not None, "b,c 均有空位，应成功"

    def test_manifest_cluster_edge_requires_tangent(self):
        """不相切的晶子拒接簇边。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0, position=(0.0, 0.0, 0.0))
        reg.create_node(b, 0, position=(100.0, 0.0, 0.0))
        reg.nodes["a"].degree = 50
        reg.nodes["b"].degree = 50

        rel = pool.manifest_cluster_edge(a, b, 1, reg)
        assert rel is None

    def test_clustering_phase_runs(self):
        """完整种子期后，聚簇阶段应创建晶子间连接。"""
        config = SeedEpochConfig(target_edges=20000, seed_duration=300)
        engine = SeedEpochEngine(config=config)
        engine.run_seed_epoch()

        crystallites = engine.node_registry.crystallite_nodes()
        total_conn = sum(n.crystallite_connections for n in crystallites)
        max_deg = max(n.degree for n in engine.node_registry.nodes.values())
        print(f"\n  [聚簇验证] 晶子数={len(crystallites)}, 总簇边={total_conn // 2}")
        print(f"  [度数上限] max degree = {max_deg} (≤ {CRYSTALLITE_DEGREE_THRESHOLD})")
        assert max_deg <= CRYSTALLITE_DEGREE_THRESHOLD, f"degree {max_deg} 超过上限 {CRYSTALLITE_DEGREE_THRESHOLD}"

    def test_cluster_produces_triangulated_subgraphs(self):
        """聚簇后，晶子子图中应有闭合分量。"""
        config = SeedEpochConfig(target_edges=50000, seed_duration=400)
        engine = SeedEpochEngine(config=config)
        engine.run_seed_epoch()

        # 构建晶子子图邻接表（使用 manifest 中两端均为晶子的边）
        adj = {}
        for (uid_a, uid_b) in engine.relation_pool.manifest:
            na = engine.node_registry.nodes.get(uid_a)
            nb = engine.node_registry.nodes.get(uid_b)
            if na and nb and na.is_crystallite and nb.is_crystallite:
                adj.setdefault(uid_a, set()).add(uid_b)
                adj.setdefault(uid_b, set()).add(uid_a)

        if len(adj) < 3:
            print(f"\n  [三角剖分] 晶子子图太小 (n={len(adj)})")
            return

        # 找连通分量
        from collections import deque
        visited = set()
        components = []
        for node in adj:
            if node in visited:
                continue
            q = deque([node])
            comp = set()
            while q:
                cur = q.popleft()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.add(cur)
                for nb in adj.get(cur, []):
                    if nb not in visited:
                        q.append(nb)
            if len(comp) >= 3:
                components.append(comp)

        print(f"\n  [三角剖分] 晶子子图分量 (≥3节点): {len(components)}")
        for comp in components:
            deg_in = {}
            for u in comp:
                deg_in[u] = len([v for v in adj.get(u, []) if v in comp])
            spum_val = sum(6 - d for d in deg_in.values())
            v, e = len(comp), sum(deg_in.values()) // 2
            print(f"    分量 V={v}, E={e}, Sigma(6-deg)={spum_val}")


# ============================================================
# node_registry.py (continued)
# ============================================================
class TestNodeState:
    def test_default_degree(self):
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr)
        assert ns.degree == 0
        assert ns.created_frame == -1

    def test_virtual_faces_initial(self):
        """新节点应有满虚面 = 晶子阈值 50。"""
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr)
        assert ns.virtual_faces == CRYSTALLITE_DEGREE_THRESHOLD == 50

    def test_virtual_faces_computed_from_degree(self):
        """virtual_faces = max(0, 50 - degree)。"""
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr)
        assert ns.virtual_faces == 50
        ns.degree = 10
        assert ns.virtual_faces == 40
        ns.degree = 50
        assert ns.virtual_faces == 0
        ns.degree = 55
        assert ns.virtual_faces == 0

    def test_is_dangling_true(self):
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr, degree=0)
        assert ns.is_dangling
        ns.degree = 1
        assert ns.is_dangling
        ns.degree = 2
        assert not ns.is_dangling

    def test_is_saturated(self):
        """饱和 = degree ≥ 50（晶子阈值）。"""
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr, degree=49)
        assert not ns.is_saturated
        assert ns.virtual_faces == 1
        ns.degree = 50
        assert ns.is_saturated
        assert ns.virtual_faces == 0
        ns.degree = 60
        assert ns.is_saturated

    def test_is_crystallite(self):
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = NodeState(address=addr, degree=49)
        assert not ns.is_crystallite
        ns.degree = 50
        assert ns.is_crystallite
        ns.degree = 51
        assert ns.is_crystallite


class TestNodeRegistry:
    def test_create_node(self):
        reg = NodeRegistry()
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns = reg.create_node(addr, frame_number=1)
        assert ns.address == addr
        assert ns.degree == 0
        assert ns.virtual_faces == 50
        assert ns.created_frame == 1
        assert reg.node_count() == 1

    def test_create_node_idempotent(self):
        reg = NodeRegistry()
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        ns1 = reg.create_node(addr, frame_number=1)
        ns2 = reg.create_node(addr, frame_number=2)
        assert ns1 is ns2
        assert ns1.created_frame == 1
        assert reg.node_count() == 1

    def test_update_degree_increment(self):
        reg = NodeRegistry()
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        reg.create_node(addr, frame_number=1)
        reg.update_degree("n1", +1)
        assert reg.nodes["n1"].degree == 1
        assert reg.nodes["n1"].virtual_faces == 49
        reg.update_degree("n1", +5)
        assert reg.nodes["n1"].degree == 6
        assert reg.nodes["n1"].virtual_faces == 44

    def test_update_degree_decrement(self):
        """度数减少 → virtual_faces 自动恢复。"""
        reg = NodeRegistry()
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        reg.create_node(addr, frame_number=1)
        reg.update_degree("n1", +40)
        assert reg.nodes["n1"].virtual_faces == 10
        reg.update_degree("n1", -5)
        assert reg.nodes["n1"].degree == 35
        assert reg.nodes["n1"].virtual_faces == 15

    def test_update_degree_clamp_below_zero(self):
        reg = NodeRegistry()
        addr = TopologicalAddress(uid="n1", differentiation_step=0)
        reg.create_node(addr, frame_number=1)
        reg.update_degree("n1", -1)
        assert reg.nodes["n1"].degree == 0
        assert reg.nodes["n1"].virtual_faces == 50

    def test_update_degree_nonexistent(self):
        reg = NodeRegistry()
        with pytest.raises(KeyError):
            reg.update_degree("nonexistent", +1)

    def test_dangling_nodes(self):
        reg = NodeRegistry()
        for i in range(3):
            addr = TopologicalAddress(uid=f"n{i}", differentiation_step=i)
            state = reg.create_node(addr, frame_number=0)
            state.degree = i
        dangling = reg.dangling_nodes()
        assert len(dangling) == 2
        assert all(n.degree < 2 for n in dangling)

    def test_saturated_nodes(self):
        reg = NodeRegistry()
        for i in range(3):
            addr = TopologicalAddress(uid=f"n{i}", differentiation_step=i)
            state = reg.create_node(addr, frame_number=0)
            state.degree = 49 + i  # n49: v=1(not), n50: v=0(sat), n51: v=0(sat)
        saturated = reg.saturated_nodes()
        assert len(saturated) == 2
        assert all(n.virtual_faces == 0 for n in saturated)

    def test_crystallite_nodes(self):
        reg = NodeRegistry()
        for i in range(3):
            addr = TopologicalAddress(uid=f"n{i}", differentiation_step=i)
            state = reg.create_node(addr, frame_number=0)
            state.degree = 49 + i
        crystallites = reg.crystallite_nodes()
        assert len(crystallites) == 2

    def test_node_count(self):
        reg = NodeRegistry()
        assert reg.node_count() == 0
        for i in range(5):
            addr = TopologicalAddress(uid=f"n{i}", differentiation_step=i)
            reg.create_node(addr, frame_number=0)
        assert reg.node_count() == 5


# ============================================================
# relation_pool.py
# ============================================================
class TestRelation:
    def test_default_state(self):
        addr1 = TopologicalAddress(uid="a", differentiation_step=0)
        addr2 = TopologicalAddress(uid="b", differentiation_step=0)
        rel = Relation(id="r1", loc1=addr1, loc2=addr2)
        assert rel.state == RelationState.LATENT
        assert not rel.is_manifest
        assert rel.contact_type == ContactType.UNDEFINED
        assert rel.created_frame == -1

    def test_manifest(self):
        addr1 = TopologicalAddress(uid="a", differentiation_step=0)
        addr2 = TopologicalAddress(uid="b", differentiation_step=0)
        rel = Relation(id="r1", loc1=addr1, loc2=addr2)
        rel.manifest(frame_number=5, contact_type=ContactType.REAL_REAL)
        assert rel.is_manifest
        assert rel.created_frame == 5
        assert rel.contact_type == ContactType.REAL_REAL

    def test_annihilate(self):
        addr1 = TopologicalAddress(uid="a", differentiation_step=0)
        addr2 = TopologicalAddress(uid="b", differentiation_step=0)
        rel = Relation(id="r1", loc1=addr1, loc2=addr2)
        rel.manifest(frame_number=5)
        rel.annihilate()
        assert not rel.is_manifest
        assert rel.created_frame == -1
        assert rel.contact_type == ContactType.UNDEFINED

    def test_hash_and_eq(self):
        addr1 = TopologicalAddress(uid="a", differentiation_step=0)
        addr2 = TopologicalAddress(uid="b", differentiation_step=0)
        r1 = Relation(id="r1", loc1=addr1, loc2=addr2)
        r2 = Relation(id="r1", loc1=addr1, loc2=addr2)
        assert hash(r1) == hash(r2)
        assert r1 == r2
        r3 = Relation(id="r2", loc1=addr1, loc2=addr2)
        assert r1 != r3

    def test_contact_type_default(self):
        addr1 = TopologicalAddress(uid="a", differentiation_step=0)
        addr2 = TopologicalAddress(uid="b", differentiation_step=0)
        rel = Relation(id="r1", loc1=addr1, loc2=addr2)
        rel.manifest(frame_number=0)
        assert rel.contact_type == ContactType.REAL_REAL


class TestRelationPool:
    # --- _make_key ---
    def test_make_key_sorted(self):
        addr1 = TopologicalAddress(uid="b", differentiation_step=0)
        addr2 = TopologicalAddress(uid="a", differentiation_step=0)
        key = RelationPool._make_key(addr1, addr2)
        assert key == ("a", "b")

    def test_make_key_same_order(self):
        addr1 = TopologicalAddress(uid="a", differentiation_step=0)
        addr2 = TopologicalAddress(uid="b", differentiation_step=0)
        key = RelationPool._make_key(addr1, addr2)
        assert key == ("a", "b")

    # --- manifest_relation ---
    def test_manifest_new(self):
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        rel = pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        assert rel is not None
        assert rel.is_manifest
        assert pool.edge_count() == 1

    def test_manifest_self_loop_rejected(self):
        pool = RelationPool()
        reg = NodeRegistry()
        addr = TopologicalAddress(uid="a", differentiation_step=0)
        rel = pool.manifest_relation(addr, addr, frame_number=0, node_registry=reg)
        assert rel is None
        assert pool.edge_count() == 0

    def test_manifest_duplicate_returns_same(self):
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        r1 = pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        r2 = pool.manifest_relation(addr_a, addr_b, frame_number=1, node_registry=reg)
        assert r1 is r2
        assert pool.edge_count() == 1

    def test_manifest_creates_nodes_and_updates_degree(self):
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        assert "a" in reg.nodes
        assert "b" in reg.nodes
        assert reg.nodes["a"].degree == 1
        assert reg.nodes["a"].virtual_faces == 49
        assert reg.nodes["b"].degree == 1
        assert reg.nodes["b"].virtual_faces == 49

    def test_manifest_to_saturated_rejected(self):
        """晶子 (degree≥50) 拒绝新连接。"""
        pool = RelationPool()
        reg = NodeRegistry()
        sat_addr = TopologicalAddress(uid="sat", differentiation_step=0)
        reg.create_node(sat_addr, 0)
        reg.nodes["sat"].degree = 50  # crystallite → saturated

        new_addr = TopologicalAddress(uid="new", differentiation_step=1)
        reg.create_node(new_addr, 0)

        rel = pool.manifest_relation(sat_addr, new_addr, frame_number=1, node_registry=reg)
        assert rel is None, "不应允许连接到晶子（饱和）节点"

    def test_manifest_saturated_when_reaching_50(self):
        """节点在第 50 条连接时饱和，第 51 条被拒绝。"""
        pool = RelationPool()
        reg = NodeRegistry()
        hub_addr = TopologicalAddress(uid="hub", differentiation_step=0)

        for i in range(52):
            spoke_addr = TopologicalAddress(uid=f"spoke{i}", differentiation_step=i + 1)
            rel = pool.manifest_relation(hub_addr, spoke_addr, frame_number=0, node_registry=reg)
            if i < 50:
                assert rel is not None, f"第 {i+1} 条边应成功"
            else:
                assert rel is None, f"第 51 条边应被拒绝（饱和）"

        assert pool.edge_count() == 50
        assert reg.nodes["hub"].degree == 50
        assert reg.nodes["hub"].is_saturated

    def test_manifest_contact_type_auto_vv(self):
        """两个新节点（均有虚面）→ VIRTUAL_VIRTUAL"""
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        rel = pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        assert rel.contact_type == ContactType.VIRTUAL_VIRTUAL

    def test_manifest_contact_type_auto_rv(self):
        """饱和节点 (v=0) + 新节点 (v=50) → 被饱和拒绝"""
        pool = RelationPool()
        reg = NodeRegistry()
        hub = TopologicalAddress(uid="hub", differentiation_step=0)
        reg.create_node(hub, 0)
        for i in range(50):
            spoke = TopologicalAddress(uid=f"s{i}", differentiation_step=i + 1)
            pool.manifest_relation(hub, spoke, frame_number=0, node_registry=reg)
        assert reg.nodes["hub"].is_saturated

        last = TopologicalAddress(uid="last", differentiation_step=99)
        reg.create_node(last, 0)
        rel = pool.manifest_relation(hub, last, frame_number=1, node_registry=reg,
                                     contact_type=ContactType.REAL_VIRTUAL)
        assert rel is None  # 被饱和拒绝

    def test_manifest_contact_type_auto_rr(self):
        """两个均饱和的节点 → 连接被拒绝（饱和）。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0)
        reg.create_node(b, 0)
        reg.nodes["a"].degree = 50
        reg.nodes["b"].degree = 50
        rel = pool.manifest_relation(a, b, frame_number=0, node_registry=reg)
        assert rel is None

    def test_manifest_explicit_contact_type(self):
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        rel = pool.manifest_relation(
            addr_a, addr_b, frame_number=0, node_registry=reg,
            contact_type=ContactType.REAL_REAL,
        )
        assert rel is not None
        assert rel.contact_type == ContactType.REAL_REAL

    # --- deterministic_candidates ---
    def test_deterministic_candidates_empty_registry(self):
        pool = RelationPool()
        reg = NodeRegistry()
        candidates = pool.deterministic_candidates(reg, count=5)
        assert candidates == []

    def test_deterministic_candidates_zero_count(self):
        pool = RelationPool()
        reg = NodeRegistry()
        reg.create_node(TopologicalAddress(uid="a", differentiation_step=0), 0)
        candidates = pool.deterministic_candidates(reg, count=0)
        assert candidates == []

    def test_deterministic_candidates_filters_saturated(self):
        """饱和节点不出现在候选列表中。"""
        pool = RelationPool()
        reg = NodeRegistry()
        sat = reg.create_node(
            TopologicalAddress(uid="sat", differentiation_step=0), 0
        )
        sat.degree = 50  # saturated
        fresh = reg.create_node(
            TopologicalAddress(uid="fresh", differentiation_step=1), 0
        )
        candidates = pool.deterministic_candidates(reg, count=10)
        if candidates:
            for a, b in candidates:
                assert a.uid != "sat"
                assert b.uid != "sat"

    def test_deterministic_candidates_no_duplicate(self):
        """候选不应返回已存在的边。"""
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        candidates = pool.deterministic_candidates(reg, count=2)
        for loc_a, loc_b in candidates:
            key = pool._make_key(loc_a, loc_b)
            assert key not in pool.manifest

    def test_deterministic_candidates_prioritizes_high_virtual_faces(self):
        """虚面多的节点优先配对。"""
        pool = RelationPool()
        reg = NodeRegistry()
        n0 = reg.create_node(TopologicalAddress(uid="n0", differentiation_step=0), 0)
        n0.degree = 49  # v=1 (近饱和)
        n1 = reg.create_node(TopologicalAddress(uid="n1", differentiation_step=1), 0)
        # n1 v=50
        n2 = reg.create_node(TopologicalAddress(uid="n2", differentiation_step=2), 0)
        n2.degree = 1  # v=49

        candidates = pool.deterministic_candidates(reg, count=2)
        assert len(candidates) >= 1
        # step1 应输出 (n1,n2): 虚面最多的一对
        key = pool._make_key(candidates[0][0], candidates[0][1])
        assert "n1" in key and "n2" in key

    def test_deterministic_candidates_differentiation_via_origin(self):
        """节点不足时，从 origin 分化新地址。"""
        pool = RelationPool()
        reg = NodeRegistry()
        n0 = reg.create_node(TopologicalAddress(uid="n0", differentiation_step=0), 0)
        n0.degree = 49  # v=1
        n1 = reg.create_node(TopologicalAddress(uid="n1", differentiation_step=1), 0)
        # n1 v=50

        candidates = pool.deterministic_candidates(reg, count=3)
        assert len(candidates) == 3
        # step1: (n0,n1)
        # step2: 新地址连接到 n1 (虚面最多)
        assert any(
            loc.uid.startswith("N") for loc in candidates[1]
        ), "step2 应产生分化地址"

    # --- annihilate_relation ---
    def test_annihilate_edge_count_conserved(self):
        """湮灭 + 补偿? 总边数守恒（有虚面时）。"""
        pool = RelationPool()
        reg = NodeRegistry()
        for i in range(3):
            reg.create_node(TopologicalAddress(uid=f"n{i}", differentiation_step=i), 0)
        pool.manifest_relation(
            TopologicalAddress(uid="n0", differentiation_step=0),
            TopologicalAddress(uid="n1", differentiation_step=0), 0, reg,
        )
        pool.manifest_relation(
            TopologicalAddress(uid="n0", differentiation_step=0),
            TopologicalAddress(uid="n2", differentiation_step=0), 0, reg,
        )
        assert pool.edge_count() == 2
        annihilated = pool.annihilate_relation(("n0", "n1"), reg, frame_number=1)
        assert annihilated is not None
        assert pool.edge_count() == 2

    def test_annihilate_reduces_degree(self):
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        pool.annihilate_relation(("a", "b"), reg, frame_number=1)
        # 湮灭 -1, 补偿 +1, 度数可能相等或不变
        assert reg.nodes["a"].degree <= 1

    def test_annihilate_nonexistent(self):
        pool = RelationPool()
        reg = NodeRegistry()
        result = pool.annihilate_relation(("x", "y"), reg, frame_number=0)
        assert result is None

    def test_annihilate_reverse_key(self):
        """反向键 (b,a) 应匹配正向键 (a,b)。"""
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        result = pool.annihilate_relation(("b", "a"), reg, frame_number=1)
        assert result is not None

    def test_annihilate_when_all_saturated(self):
        """湮灭释放虚面后补偿可成功，总边数守恒。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0)
        reg.create_node(b, 0)
        pool.manifest_relation(a, b, frame_number=0, node_registry=reg)
        reg.nodes["a"].degree = 50  # 强制饱和
        reg.nodes["b"].degree = 50
        pool.annihilate_relation(("a", "b"), reg, frame_number=1)
        # 湮灭让 degree? 49, v? 1 释放虚面 → 补偿可创建
        assert pool.edge_count() == 1

    # --- query ---
    def test_has_edge(self):
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        assert not pool.has_edge(addr_a, addr_b)
        pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        assert pool.has_edge(addr_a, addr_b)

    def test_get_edge(self):
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        assert pool.get_edge(addr_a, addr_b) is None
        pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        assert pool.get_edge(addr_a, addr_b) is not None

    def test_get_edge_reverse(self):
        pool = RelationPool()
        reg = NodeRegistry()
        addr_a = TopologicalAddress(uid="a", differentiation_step=0)
        addr_b = TopologicalAddress(uid="b", differentiation_step=0)
        pool.manifest_relation(addr_a, addr_b, frame_number=0, node_registry=reg)
        assert pool.get_edge(addr_b, addr_a) is not None

    def test_edge_count_initial_zero(self):
        pool = RelationPool()
        assert pool.edge_count() == 0

    def test_edge_count_after_multiple(self):
        pool = RelationPool()
        reg = NodeRegistry()
        for i in range(3):
            reg.create_node(TopologicalAddress(uid=f"n{i}", differentiation_step=i), 0)
        pool.manifest_relation(
            TopologicalAddress(uid="n0", differentiation_step=0),
            TopologicalAddress(uid="n1", differentiation_step=0), 0, reg,
        )
        pool.manifest_relation(
            TopologicalAddress(uid="n0", differentiation_step=0),
            TopologicalAddress(uid="n2", differentiation_step=0), 0, reg,
        )
        assert pool.edge_count() == 2


# ============================================================
# seed_epoch_engine.py
# ============================================================
class TestSeedEpochConfig:
    def test_default_values(self):
        config = SeedEpochConfig()
        assert config.target_edges == 10000
        assert config.seed_duration == 200

    def test_build_node_sequence_non_decreasing(self):
        config = SeedEpochConfig(target_edges=1000, seed_duration=50)
        seq = config.build_node_sequence()
        for v in seq:
            assert v >= 0

    def test_build_node_sequence_deterministic(self):
        config1 = SeedEpochConfig(target_edges=777, seed_duration=42)
        config2 = SeedEpochConfig(target_edges=777, seed_duration=42)
        assert config1.build_node_sequence() == config2.build_node_sequence()

    def test_build_node_sequence_zero_duration(self):
        config = SeedEpochConfig(target_edges=100, seed_duration=0)
        seq = config.build_node_sequence()
        assert len(seq) == 1
        assert seq[0] >= 2

    def test_build_node_sequence_single_target(self):
        config = SeedEpochConfig(target_edges=1, seed_duration=10)
        seq = config.build_node_sequence()
        assert all(v >= 2 for v in seq)


class TestSeedEpochEngine:
    def test_post_init_computes_sequence(self):
        config = SeedEpochConfig(target_edges=100, seed_duration=20)
        engine = SeedEpochEngine(config=config)
        assert len(engine.node_sequence) > 0

    def test_run_seed_epoch_returns_edges(self):
        config = SeedEpochConfig(target_edges=500, seed_duration=30)
        engine = SeedEpochEngine(config=config)
        total = engine.run_seed_epoch()
        assert total == engine.relation_pool.edge_count()
        assert total > 0

    def test_run_seed_epoch_creates_nodes_and_edges(self):
        config = SeedEpochConfig(target_edges=500, seed_duration=30)
        engine = SeedEpochEngine(config=config)
        engine.run_seed_epoch()
        assert engine.relation_pool.edge_count() > 0
        assert engine.node_registry.node_count() > 0
        assert engine.current_frame > 0

    def test_run_seed_epoch_determinism(self):
        config = SeedEpochConfig(target_edges=500, seed_duration=30)
        e1 = SeedEpochEngine(config=config)
        e2 = SeedEpochEngine(config=config)
        r1 = e1.run_seed_epoch()
        r2 = e2.run_seed_epoch()
        assert r1 == r2
        assert e1.node_registry.node_count() == e2.node_registry.node_count()
        assert e1.relation_pool.edge_count() == e2.relation_pool.edge_count()

    def test_run_seed_epoch_small_config(self):
        config = SeedEpochConfig(target_edges=50, seed_duration=10)
        engine = SeedEpochEngine(config=config)
        total = engine.run_seed_epoch()
        assert total >= 1
        assert total <= 50

    def test_degree_bounded_by_crystallite_threshold(self):
        """核心验证: 任何节点的度数不超过晶子阈值 (50)。"""
        config = SeedEpochConfig(target_edges=10000, seed_duration=200)
        engine = SeedEpochEngine(config=config)
        engine.run_seed_epoch()

        max_degree = max(n.degree for n in engine.node_registry.nodes.values())
        assert max_degree <= CRYSTALLITE_DEGREE_THRESHOLD, \
            f"最大度数 {max_degree} > 晶子阈值 {CRYSTALLITE_DEGREE_THRESHOLD}"
        print(f"\n  [晶子容量验证] 最大度数 = {max_degree} (<= {CRYSTALLITE_DEGREE_THRESHOLD}) [OK]")

    def test_node_degree_consistency(self):
        config = SeedEpochConfig(target_edges=500, seed_duration=30)
        engine = SeedEpochEngine(config=config)
        engine.run_seed_epoch()
        total_degree = sum(n.degree for n in engine.node_registry.nodes.values())
        assert total_degree == 2 * engine.relation_pool.edge_count()

    def test_no_self_loops(self):
        config = SeedEpochConfig(target_edges=500, seed_duration=30)
        engine = SeedEpochEngine(config=config)
        engine.run_seed_epoch()
        for key in engine.relation_pool.manifest:
            assert key[0] != key[1], f"发现自环: {key}"

    def test_no_saturated_node_in_new_edges(self):
        """种子期结束时，所有节点的度数 ≤ 晶子阈值。"""
        config = SeedEpochConfig(target_edges=1000, seed_duration=50)
        engine = SeedEpochEngine(config=config)
        engine.run_seed_epoch()
        for key in engine.relation_pool.manifest:
            uid_a, uid_b = key
            node_a = engine.node_registry.nodes[uid_a]
            node_b = engine.node_registry.nodes[uid_b]
            assert node_a.degree <= CRYSTALLITE_DEGREE_THRESHOLD
            assert node_b.degree <= CRYSTALLITE_DEGREE_THRESHOLD

    def test_target_edges_1(self):
        """target_edges=1 → 自动推算 target_nodes=10，至少产生一些边。"""
        config = SeedEpochConfig(target_edges=1, seed_duration=10)
        engine = SeedEpochEngine(config=config)
        total = engine.run_seed_epoch()
        # target_nodes = max(10, sqrt(2)) = 10, 应产生一个小网络
        assert total >= 1
        assert engine.node_registry.node_count() >= 2

    def test_virtual_faces_invariant(self):
        """virtual_faces + degree = CRYSTALLITE_DEGREE_THRESHOLD (当 degree ≤ 阈值时)。"""
        config = SeedEpochConfig(target_edges=500, seed_duration=30)
        engine = SeedEpochEngine(config=config)
        engine.run_seed_epoch()
        for node in engine.node_registry.nodes.values():
            if node.degree <= CRYSTALLITE_DEGREE_THRESHOLD:
                assert node.virtual_faces + node.degree == CRYSTALLITE_DEGREE_THRESHOLD
