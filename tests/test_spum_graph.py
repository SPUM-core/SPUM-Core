"""
SPUM-图论模块测试 — src/spum_graph
====================================

覆盖:
    FrameGraph — 帧内图结构 (5 条公理)
    HandshakingVerifier — 握手引理 / 欧拉示性数 / Σ(6-deg)

注意: dangling 模块依赖 torch——本测试仅覆盖无 torch 依赖的纯图论部分。

运行:
    python -m unittest tests.test_spum_graph -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spum_graph.graph import FrameGraph
from spum_graph.handshaking import HandshakingVerifier, euler_characteristic


class TestFrameGraph(unittest.TestCase):
    """帧内图结构。"""

    def test_add_edge_implicit_nodes(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        self.assertTrue(g.has_node("A"))
        self.assertTrue(g.has_node("B"))
        self.assertEqual(g.node_count, 2)
        self.assertEqual(g.edge_count, 1)

    def test_self_loop_rejected(self):
        g = FrameGraph(frame_id=1)
        with self.assertRaises(ValueError):
            g.add_edge("A", "A")

    def test_degrees(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        self.assertEqual(g.degrees(), {"A": 1, "B": 2, "C": 1})

    def test_degree_missing_node_zero(self):
        g = FrameGraph(frame_id=1)
        self.assertEqual(g.degree("GHOST"), 0)

    def test_dangling_detection(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        self.assertTrue(g.is_dangling("A"))
        self.assertTrue(g.is_dangling("B"))
        g.add_edge("B", "C")
        self.assertFalse(g.is_dangling("B"))

    def test_dangling_nodes_set(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        self.assertEqual(g.dangling_nodes(), {"A", "D"})

    def test_dangling_density(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        self.assertAlmostEqual(g.dangling_density(), 2 / 4)

    def test_dangling_density_empty(self):
        g = FrameGraph(frame_id=1)
        self.assertEqual(g.dangling_density(), 0.0)

    def test_closed_cycle_no_dangling(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        self.assertEqual(g.dangling_nodes(), set())

    def test_neighbors(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        self.assertEqual(g.neighbors("A"), {"B", "C"})
        self.assertEqual(g.neighbors("GHOST"), set())

    def test_nodes_iterator(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        self.assertEqual(set(g.nodes()), {"A", "B"})

    def test_snapshot(self):
        g = FrameGraph(frame_id=7)
        g.add_edge("A", "B")
        snap = g.snapshot()
        self.assertEqual(snap["frame_id"], 7)
        self.assertEqual(snap["node_count"], 2)
        self.assertEqual(snap["edge_count"], 1)
        self.assertEqual(snap["dangling_count"], 2)

    def test_repr(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        self.assertIn("FrameGraph", repr(g))
        self.assertIn("frame=1", repr(g))


class TestHandshakingVerifier(unittest.TestCase):
    """握手引理 / 欧拉示性数 / Σ(6-deg)。"""

    def test_handshaking_passes(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        result = HandshakingVerifier.verify_handshaking(g)
        self.assertTrue(result["passed"])
        self.assertEqual(result["sum_deg"], 6)
        self.assertEqual(result["expected"], 6)

    def test_euler_characteristic_tetrahedron(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        g.add_edge("A", "D")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        result = HandshakingVerifier.euler_characteristic(g, faces=4)
        self.assertEqual(result["chi"], 2)

    def test_euler_characteristic_no_faces(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        result = HandshakingVerifier.euler_characteristic(g)
        self.assertIsNone(result["chi"])

    def test_angle_deficit_square(self):
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "D")
        g.add_edge("D", "A")
        result = HandshakingVerifier.angle_deficit_sum(g)
        self.assertEqual(result["sum_6_minus_deg"], 16)

    def test_angle_deficit_degree3_tetrahedron(self):
        """四面体 (deg=3 各节点): Σ(6-3)=12 —— 拓扑常数 12 的帧内验证。"""
        g = FrameGraph(frame_id=1)
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("C", "A")
        g.add_edge("A", "D")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        result = HandshakingVerifier.angle_deficit_sum(g)
        self.assertEqual(result["sum_6_minus_deg"], 12)

    def test_module_level_euler_function(self):
        self.assertEqual(euler_characteristic(4, 6, 4), 2)
        self.assertIsNone(euler_characteristic(4, 6))


class TestPackageImportWithoutTorch(unittest.TestCase):
    """包导入 — torch 未安装时仍可加载。"""

    def test_package_imports(self):
        import spum_graph
        self.assertTrue(hasattr(spum_graph, "FrameGraph"))
        self.assertTrue(hasattr(spum_graph, "HandshakingVerifier"))


if __name__ == "__main__":
    unittest.main()
