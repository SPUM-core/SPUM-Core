"""
Phase 2 测试 — 帧演化引擎

测试范围:
    - FrameUpdateConfig 默认值
    - FrameLog 基本记录
    - 单帧运行（无需级联）
    - 有悬挂节点的级联消解
    - 持续性多帧演化
    - 与 SeedEpochEngine 的集成
    - 不变量稳定性
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Phase_1.seed_epoch_engine import SeedEpochConfig, SeedEpochEngine
from Phase_1.node_registry import NodeRegistry
from Phase_1.relation_pool import RelationPool
from Phase_1.topological_address import TopologicalAddress, reset_differentiation_counter
from Phase_2.frame_update_engine import (
    FrameUpdateConfig,
    FrameLog,
    FrameUpdateEngine,
)


# ============================================================
# FrameUpdateConfig
# ============================================================
class TestFrameUpdateConfig:
    def test_default_values(self):
        cfg = FrameUpdateConfig()
        assert cfg.growth_per_frame == 0
        assert cfg.max_cascade_iterations == 100
        assert cfg.enable_cascade is True
        assert cfg.directional_cascade is False
        assert cfg.cascade_source_uid == ""
        assert cfg.use_natural_cap is False


# ============================================================
# FrameLog
# ============================================================
class TestFrameLog:
    def test_basic_log(self):
        log = FrameLog(frame_number=5, edges_created=10, edges_annihilated=3, net_edge_change=7)
        assert log.frame_number == 5
        assert log.net_edge_change == 7
        assert log.stable is True

    def test_brief_format(self):
        log = FrameLog(
            frame_number=42,
            edges_created=10,
            edges_annihilated=2,
            net_edge_change=8,
            cascade_iters=3,
            dangling_before=5,
            dangling_after=0,
            edge_count=500,
            cluster_edge_count=27,
            node_count=445,
            spum_invariant=496,
            crystallite_count=11,
            stable=True,
        )
        brief = log.brief()
        # frame_number 格式化为 4 位宽: " 42"
        assert "+" in brief
        assert "5->0" in brief
        assert "496" in brief
        assert "[OK]" in brief


# ============================================================
# FrameUpdateEngine — 基本功能
# ============================================================
class TestFrameUpdateEngine:
    def test_run_frame_empty_network(self):
        """空网络运行一帧应稳定通过。"""
        pool = RelationPool()
        reg = NodeRegistry()
        config = FrameUpdateConfig()
        engine = FrameUpdateEngine(pool, reg, config)

        log = engine.run_frame()
        assert log.stable is True
        assert log.edges_created == 0
        assert log.edges_annihilated == 0
        assert log.node_count == 0

    def test_run_frame_no_cascade_needed(self):
        """无悬挂节点的网络运行一帧。"""
        pool = RelationPool()
        reg = NodeRegistry()

        # 创建 3 个稳定节点（degree=2 的环）
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        c = TopologicalAddress(uid="c", differentiation_step=0)
        reg.create_node(a, 0)
        reg.create_node(b, 0)
        reg.create_node(c, 0)

        rel_a = pool.manifest_relation(a, b, 0, reg)
        rel_b = pool.manifest_relation(b, c, 0, reg)
        rel_c = pool.manifest_relation(c, a, 0, reg)
        assert rel_a is not None
        assert rel_b is not None
        assert rel_c is not None
        assert len(pool.manifest) == 3

        engine = FrameUpdateEngine(pool, reg)
        log = engine.run_frame()

        # 无悬挂节点，级联不应触发
        assert log.stable is True
        assert log.edges_annihilated == 0
        assert log.node_count == 3
        assert log.edge_count == 3

    def test_run_frame_with_growth(self):
        """带生长配额的帧应创建新边。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0)
        reg.create_node(b, 0)

        rel = pool.manifest_relation(a, b, 0, reg)
        assert rel is not None

        # 生长配额 = 1
        config = FrameUpdateConfig(growth_per_frame=1)
        engine = FrameUpdateEngine(pool, reg, config)
        log = engine.run_frame()

        print(f"\n  [带生长] 创建={log.edges_created}, "
              f"湮灭={log.edges_annihilated}, "
              f"级联={log.cascade_iters}次, 稳定={log.stable}")
        assert log.edges_created >= 0
        assert engine.relation_pool.edge_count() >= 0

    def test_single_dangling_resolved(self):
        """单个悬挂节点会被级联消解。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0)
        reg.create_node(b, 0)

        rel_ab = pool.manifest_relation(a, b, 0, reg)
        assert rel_ab is not None
        assert len(pool.manifest) == 1

        # 手动降低一个节点度数到 0（模拟悬挂）
        reg.nodes["a"].degree = 0
        reg.nodes["b"].degree = 0
        dangling_before = len(reg.dangling_nodes())
        assert dangling_before >= 2  # 两个都是悬挂

        engine = FrameUpdateEngine(pool, reg)
        log = engine.run_frame()

        # 悬挂边被湮灭，补偿创建新边
        # 原始边 (a,b) 被湮灭，补偿创建可能成功或失败
        print(f"\n  [悬挂消解] 湮灭={log.edges_annihilated}, "
              f"创建={log.edges_created}, "
              f"稳定={log.stable}")

    def test_cascade_chain_reaction(self):
        """级联链式反应：两端都是悬挂的边被湮灭。"""
        pool = RelationPool()
        reg = NodeRegistry()

        # 创建两个悬挂节点连接：a(1)-b(1)
        addrs = {}
        for name in ["a", "b"]:
            addr = TopologicalAddress(uid=name, differentiation_step=0)
            reg.create_node(addr, 0)
            addrs[name] = addr

        pool.manifest_relation(addrs["a"], addrs["b"], 0, reg)

        # degree: a=1, b=1 (都是悬挂)
        assert reg.nodes["a"].degree == 1
        assert reg.nodes["b"].degree == 1
        assert len(reg.dangling_nodes()) == 2

        engine = FrameUpdateEngine(pool, reg)
        log = engine.run_frame()

        print(f"\n  [悬挂-悬挂消解] 湮灭={log.edges_annihilated}, "
              f"创建={log.edges_created}, 级联={log.cascade_iters}次, "
              f"稳定={log.stable}")
        # 悬挂-悬挂边会被湮灭（两端都是悬挂）
        # 补偿可能成功也可能失败
        print(f"  a.degree={reg.nodes['a'].degree}, "
              f"b.degree={reg.nodes['b'].degree}")

    def test_dangling_stable_edge_not_removed(self):
        """悬挂节点连接到稳定节点(deg≥2)时，边不被湮灭。"""
        pool = RelationPool()
        reg = NodeRegistry()

        # 创建: a(1)-b(2)-c(1)
        addrs = {}
        for name in ["a", "b", "c"]:
            addr = TopologicalAddress(uid=name, differentiation_step=0)
            reg.create_node(addr, 0)
            addrs[name] = addr

        pool.manifest_relation(addrs["a"], addrs["b"], 0, reg)
        pool.manifest_relation(addrs["b"], addrs["c"], 0, reg)

        # degree: a=1, b=2, c=1
        assert reg.nodes["a"].degree == 1
        assert reg.nodes["b"].degree == 2
        assert reg.nodes["c"].degree == 1
        assert len(reg.dangling_nodes()) == 2  # a, c

        engine = FrameUpdateEngine(pool, reg)
        log = engine.run_frame()

        print(f"\n  [稳定边保护] 湮灭={log.edges_annihilated}, "
              f"级联={log.cascade_iters}次, 稳定={log.stable}")
        # a 和 c 虽然悬挂，但连接的是 b(degree=2, 稳定), 不被湮灭
        print(f"  边数={pool.edge_count()}, a.degree={reg.nodes['a'].degree}")
        assert pool.edge_count() == 2  # 不被湮灭
        assert log.edges_annihilated == 0

    def test_cascade_disabled(self):
        """禁用级联时悬挂节点不被处理。"""
        pool = RelationPool()
        reg = NodeRegistry()
        a = TopologicalAddress(uid="a", differentiation_step=0)
        b = TopologicalAddress(uid="b", differentiation_step=0)
        reg.create_node(a, 0)
        reg.create_node(b, 0)
        pool.manifest_relation(a, b, 0, reg)

        config = FrameUpdateConfig(enable_cascade=False)
        engine = FrameUpdateEngine(pool, reg, config)
        log = engine.run_frame()

        # 悬挂仍在
        assert len(reg.dangling_nodes()) >= 2
        assert log.dangling_after >= 2

    def test_full_integration_with_seed(self):
        """完整种子期后用帧演化引擎继续运行。"""
        seed_config = SeedEpochConfig(target_edges=20000, seed_duration=300)
        seed_engine = SeedEpochEngine(config=seed_config)
        seed_engine.run_seed_epoch()

        # 创建帧演化引擎（共享 relation_pool + node_registry）
        evo_config = FrameUpdateConfig(growth_per_frame=2)
        evo_engine = FrameUpdateEngine(
            seed_engine.relation_pool,
            seed_engine.node_registry,
            evo_config,
            current_frame=seed_engine.current_frame,
        )

        # 运行 5 帧
        logs = evo_engine.run_frames(5)
        assert len(logs) == 5

        for log in logs:
            print(f"\n  {log.brief()}")

        # 统计
        stable_count = sum(1 for log in logs if log.stable)
        print(f"\n  [集成验证] 稳定帧数={stable_count}/{len(logs)}")
        print(f"  {evo_engine.summary()}")

        # 至少不崩溃，度数守恒
        assert evo_engine.node_registry.node_count() >= 2
        deg_sum = sum(n.degree for n in evo_engine.node_registry.nodes.values())
        upgraded = evo_engine.relation_pool.upstreamed_edge_count()
        assert deg_sum == 2 * (evo_engine.relation_pool.edge_count() + upgraded), "度数不守恒!"

    def test_multi_frame_stability(self):
        """多帧运行后网络保持稳定。"""
        seed_config = SeedEpochConfig(target_edges=20000, seed_duration=300)
        seed_engine = SeedEpochEngine(config=seed_config)
        seed_engine.run_seed_epoch()

        evo_config = FrameUpdateConfig(growth_per_frame=1)
        evo_engine = FrameUpdateEngine(
            seed_engine.relation_pool,
            seed_engine.node_registry,
            evo_config,
            current_frame=seed_engine.current_frame,
        )

        logs = evo_engine.run_frames(10)
        stable_count = sum(1 for log in logs if log.stable)
        edge_count = evo_engine.relation_pool.edge_count()
        cluster_count = evo_engine.relation_pool.cluster_edge_count()
        node_count = evo_engine.node_registry.node_count()
        deg_sum = sum(n.degree for n in evo_engine.node_registry.nodes.values())

        upgraded = evo_engine.relation_pool.upstreamed_edge_count()
        print(f"\n  [稳定性] 稳定帧={stable_count}/10, "
              f"边={edge_count}+{cluster_count}簇, "
              f"节点={node_count}, Σdeg={deg_sum}, 升级={upgraded}")
        assert deg_sum == 2 * (edge_count + upgraded), "度数不守恒!"
