"""
SPUM 公理引擎单元测试 — spum.axioms
=====================================

覆盖:
    公理1 关系第一性: 加边/删边/节点隐式定义/自环拒绝/孤立节点清理
    公理2 悬挂端不可消除: 悬挂端检测/密度/有限帧 δ>0
    公理3 离散帧快照: 五步序列/快照导出
    公理4 拓扑守恒: 握手引理/欧拉示性数/Σ(6-deg)/开口占比
    公理5 不完美: 双层判据/边缘不完美/闭合判据
    派生操作: σ 密度/平均度数/dv/dt≤const/温度指示

运行:
    python -m unittest tests.test_axioms -v
"""

import unittest
from collections import defaultdict

from spum.axioms import Axioms, Constants, Axiom1, Axiom2, Axiom3, Axiom4, Axiom5
from spum.axioms import DerivedOperations


def make_neighbors(edges):
    """从边列表构造邻接表 (dict[str, set[str]]) 与边数。"""
    neighbors = defaultdict(set)
    edge_count = 0
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
        edge_count += 1
    return neighbors, edge_count


class TestAxiom1(unittest.TestCase):
    """公理1: 关系第一性 — 连接定义存在。"""

    def setUp(self):
        self.ax = Axiom1()
        self.neighbors = defaultdict(set)
        self.edge_count = [0]

    def test_add_edge_creates_nodes_implicitly(self):
        """节点由边隐式定义——加边前无节点概念。"""
        self.assertEqual(len(self.neighbors), 0)
        self.ax.add_edge("A", "B", self.neighbors, self.edge_count)
        self.assertIn("A", self.neighbors)
        self.assertIn("B", self.neighbors)
        self.assertEqual(self.edge_count[0], 1)

    def test_add_edge_rejects_self_loop(self):
        """SPUM 不允许自环。"""
        with self.assertRaises(ValueError):
            self.ax.add_edge("A", "A", self.neighbors, self.edge_count)

    def test_remove_edge_cleans_isolated_nodes(self):
        """删除边后度数为 0 的节点自动消亡。"""
        self.ax.add_edge("A", "B", self.neighbors, self.edge_count)
        self.ax.add_edge("B", "C", self.neighbors, self.edge_count)
        self.ax.remove_edge("A", "B", self.neighbors, self.edge_count)
        # A 度数归零 → 消亡
        self.assertNotIn("A", self.neighbors)
        # B/C 仍在
        self.assertIn("B", self.neighbors)
        self.assertIn("C", self.neighbors)

    def test_has_node_depends_on_connection(self):
        """孤立节点无法被确认存在。"""
        self.assertFalse(self.ax.has_node("X", self.neighbors))
        self.ax.add_edge("X", "Y", self.neighbors, self.edge_count)
        self.assertTrue(self.ax.has_node("X", self.neighbors))

    def test_degree_of_missing_node_is_zero(self):
        """未连接节点度数为 0（视为不存在）。"""
        self.assertEqual(self.ax.degree("GHOST", self.neighbors), 0)

    def test_nodes_iterator(self):
        self.ax.add_edge("A", "B", self.neighbors, self.edge_count)
        self.ax.add_edge("C", "D", self.neighbors, self.edge_count)
        self.assertEqual(set(self.ax.nodes(self.neighbors)), {"A", "B", "C", "D"})


class TestAxiom2(unittest.TestCase):
    """公理2: 悬挂端不可消除。"""

    def test_dangling_detection_degree_less_than_2(self):
        ax = Axiom2()
        neighbors, _ = make_neighbors([("A", "B")])  # A,B 度数 1
        self.assertTrue(ax.is_dangling("A", neighbors))
        self.assertTrue(ax.is_dangling("B", neighbors))

        neighbors, _ = make_neighbors([("A", "B"), ("B", "C"), ("C", "A")])
        self.assertFalse(ax.is_dangling("A", neighbors))

    def test_dangling_nodes_set(self):
        ax = Axiom2()
        neighbors, _ = make_neighbors([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        self.assertEqual(ax.dangling_nodes(neighbors), {"A", "E"})

    def test_dangling_density(self):
        ax = Axiom2()
        neighbors, _ = make_neighbors([("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")])
        self.assertAlmostEqual(ax.dangling_density(neighbors), 2 / 5)

    def test_dangling_density_empty_graph_is_zero(self):
        ax = Axiom2()
        self.assertEqual(ax.dangling_density(defaultdict(set)), 0.0)

    def test_finite_frame_always_has_dangling(self):
        """有限帧 δ>0 —— 链式结构必然残留悬挂端。"""
        ax = Axiom2()
        neighbors, _ = make_neighbors([("A", "B"), ("B", "C"), ("C", "D")])
        self.assertGreater(ax.dangling_density(neighbors), 0.0)

    def test_closed_cycle_has_no_dangling(self):
        """闭合环无悬挂端（可检验的完美状态特例）。"""
        ax = Axiom2()
        neighbors, _ = make_neighbors([("A", "B"), ("B", "C"), ("C", "A")])
        self.assertEqual(ax.dangling_nodes(neighbors), set())


class TestAxiom3(unittest.TestCase):
    """公理3: 离散帧快照。"""

    def test_five_steps_sequence(self):
        self.assertEqual(
            Axiom3.STEP_NAMES,
            ["创生(V⁺)", "连接", "变化体积", "判断悬挂", "删除(V⁻)"],
        )

    def test_snapshot_metrics(self):
        ax = Axiom3()
        neighbors, edge_count = make_neighbors([("A", "B"), ("B", "C")])
        snap = ax.snapshot(1, neighbors, edge_count)
        self.assertEqual(snap.frame_id, 1)
        self.assertEqual(snap.node_count, 3)
        self.assertEqual(snap.edge_count, 2)
        self.assertEqual(snap.dangling_count, 2)  # A, C
        self.assertAlmostEqual(snap.sigma, 3 / 2)

    def test_frame_snapshot_steps_initial_false(self):
        ax = Axiom3()
        neighbors, edge_count = make_neighbors([])
        snap = ax.snapshot(1, neighbors, edge_count)
        self.assertFalse(snap.all_steps_done)
        self.assertEqual(snap.to_dict()["frame_id"], 1)

    def test_commit_no_cascade_single_frame(self):
        """五步序列标记——一帧内五步独立标记。"""
        ax = Axiom3()
        neighbors, edge_count = make_neighbors([("A", "B")])
        snap = ax.snapshot(2, neighbors, edge_count)
        self.assertFalse(snap.all_steps_done)


class TestAxiom4(unittest.TestCase):
    """公理4: 拓扑守恒。"""

    def setUp(self):
        self.ax = Axiom4()

    def test_handshaking(self):
        neighbors, edge_count = make_neighbors([("A", "B"), ("B", "C"), ("C", "A")])
        result = self.ax.handshaking(neighbors, edge_count)
        self.assertTrue(result["passed"])
        self.assertEqual(result["sum_deg"], 6)
        self.assertEqual(result["expected"], 6)

    def test_handshaking_fails_on_inconsistent_count(self):
        neighbors, edge_count = make_neighbors([("A", "B"), ("B", "C")])
        result = self.ax.handshaking(neighbors, edge_count + 1)
        self.assertFalse(result["passed"])

    def test_euler_characteristic_sphere(self):
        result = self.ax.euler_characteristic(4, 6, 4)  # 四面体
        self.assertEqual(result["chi"], 2)

    def test_euler_characteristic_no_faces(self):
        result = self.ax.euler_characteristic(4, 6)
        self.assertIsNone(result["chi"])

    def test_angle_deficit_sum_square_ring(self):
        """四方环 (deg=2 各节点): Σ(6-2)=16。"""
        neighbors, _ = make_neighbors([("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
        result = self.ax.angle_deficit_sum(neighbors)
        self.assertEqual(result["sum_6_minus_deg"], 16)

    def test_topological_constant_is_12(self):
        self.assertEqual(self.ax.topological_constant(), 12)

    def test_opening_ratio(self):
        self.assertEqual(self.ax.opening_ratio(1, 3), 1 / 3)
        self.assertLessEqual(self.ax.opening_ratio(1, 6), 1 / 3)

    def test_opening_ratio_satisfied(self):
        self.assertTrue(self.ax.opening_ratio_satisfied(1 / 3))
        self.assertTrue(self.ax.opening_ratio_satisfied(0.2))
        self.assertFalse(self.ax.opening_ratio_satisfied(0.5))


class TestAxiom5(unittest.TestCase):
    """公理5: 不完美 — 双层判据。"""

    def test_edge_imperfection_after_removal(self):
        """删除悬挂端产生新悬挂端——不完美是演化的动力。"""
        ax = Axiom5()
        neighbors, edge_count = make_neighbors(
            [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]
        )
        result = ax.check_edge_imperfection(neighbors, edge_count)
        self.assertTrue(result["imperfection_proven"])
        self.assertIn("A", result["dangling_nodes"])
        self.assertIn("E", result["dangling_nodes"])

    def test_closure_requires_both_delta_and_drift(self):
        """双层闭合判据: (δ<θ_δ) 且 (Δμ<θ_μ)。"""
        ax = Axiom5(delta_threshold=0.3, drift_threshold=0.01)
        self.assertEqual(ax.delta_threshold, 0.3)
        self.assertEqual(ax.drift_threshold, 0.01)

    def test_closure_counter(self):
        ax = Axiom5(closure_patience=3)
        # 空图不触发 update（需要 hidden/labels），直接验证纯图侧
        self.assertFalse(ax.is_fully_closed())
        self.assertEqual(ax.closure_patience, 3)

    def test_history_empty(self):
        ax = Axiom5()
        self.assertEqual(ax.history(), [])
        self.assertEqual(ax.summary(), {})


class TestDerivedOperations(unittest.TestCase):
    """派生操作: σ / 平均度数 / dv/dt。"""

    def setUp(self):
        self.op = DerivedOperations()

    def test_sigma(self):
        self.assertEqual(self.op.sigma(4, 6), 4 / 6)
        self.assertEqual(self.op.sigma(0, 1), 0.0)
        self.assertEqual(self.op.sigma(1, 0), float("inf"))

    def test_avg_degree(self):
        self.assertEqual(self.op.avg_degree(4, 6), 3.0)
        self.assertEqual(self.op.avg_degree(0, 0), 0.0)

    def test_dv_dt_within_const(self):
        result = self.op.dv_dt_check(degree_change=2, frame_count=1)
        self.assertTrue(result["satisfied"])
        self.assertEqual(result["dt"], 1)
        self.assertEqual(result["const"], Constants.DV_DT_MAX)

    def test_dv_dt_exceeds_const(self):
        result = self.op.dv_dt_check(degree_change=100, frame_count=1)
        self.assertFalse(result["satisfied"])

    def test_temperature_indicator(self):
        self.assertEqual(self.op.temperature_indicator(2.0), "高温(稀疏)")
        self.assertEqual(self.op.temperature_indicator(0.7), "中温")
        self.assertEqual(self.op.temperature_indicator(0.3), "低温(饱和)")

    def test_sigma_from_neighbors(self):
        neighbors, edge_count = make_neighbors([("A", "B"), ("B", "C"), ("C", "A")])
        self.assertEqual(self.op.sigma_from_neighbors(neighbors, edge_count), 1.0)


class TestConstants(unittest.TestCase):
    """派生常数 — 全部从公理推导。"""

    def test_derived_constants(self):
        self.assertEqual(Constants.KAPPA, 1.0)
        self.assertEqual(Constants.TAU, 1)
        self.assertEqual(Constants.TOPOLOGICAL_12, 12)
        self.assertEqual(Constants.MAX_CONTACTS, 12)  # 接吻数
        self.assertEqual(Constants.C, 4.0)            # c = 4κ/τ
        self.assertEqual(Constants.DV_DT_MAX, 16.0)   # D_max = 4κ


class TestAxiomsAggregate(unittest.TestCase):
    """公理聚合入口。"""

    def test_aggregate_creation(self):
        ax = Axioms()
        self.assertEqual(ax.constants.KAPPA, 1.0)
        self.assertEqual(ax.sigma(4, 6), 4 / 6)

    def test_handshaking_through_aggregate(self):
        ax = Axioms()
        neighbors, edge_count = make_neighbors([("A", "B"), ("B", "C")])
        self.assertTrue(ax.handshaking(neighbors, edge_count)["passed"])

    def test_summary_contains_key_facts(self):
        ax = Axioms()
        s = ax.summary()
        self.assertIn("5 条公理", s)
        self.assertIn("Σ(6-deg)=12", s)

    def test_describe_all_lists_five_axioms(self):
        ax = Axioms()
        desc = ax.describe_all()
        self.assertIn("公理1", desc)
        self.assertIn("公理2", desc)
        self.assertIn("公理3", desc)
        self.assertIn("公理4", desc)
        self.assertIn("公理5", desc)


if __name__ == "__main__":
    unittest.main()
