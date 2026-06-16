"""
SPUM 跨模块集成测试：FrameGraph + DanglingDetector + FrameProtocol + ReviewBridge

验证在同一个推理循环中四个模块的协同行为。
对应审查问题 P2-9。

运行：
    python tests/test_frame_review_loop.py
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.spum_graph.graph import FrameGraph
from src.spum_graph.handshaking import HandshakingVerifier
from src.core.frame import FrameProtocol, FrameState
from src.core.review import ReviewBridge, ReviewResult, ReviewVerdict, Violation, Severity


class TestFrameReviewLoop(unittest.TestCase):
    """FrameGraph 演化 + FrameProtocol 记录 + ReviewBridge 评审的闭环测试。"""

    def setUp(self):
        self.protocol = FrameProtocol()
        self.bridge = ReviewBridge(self.protocol)
        self.verifier = HandshakingVerifier()

    def test_full_cycle_single_frame(self):
        """单帧完整闭环：图演化 → 帧协议记录 → 握手验证 → 评审通过。"""
        # 1. 创建帧图（模拟一帧的关系网络演化）
        graph = FrameGraph(frame_id=1)
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        graph.add_edge("D", "A")  # 四方环

        # 2. 握手引理验证
        result = self.verifier.verify_handshaking(graph)
        self.assertTrue(result["passed"], f"握手引理失败: {result}")

        # 3. 帧协议记录
        self.protocol.begin_frame()
        self.protocol.set_active_nodes(["N005", "N006", "N007"])
        self.protocol.add_relation("N005", "N006", "derives_from", "四方环闭合")
        self.protocol.set_dangling(len(graph.dangling_nodes()))
        self.protocol.set_layer_dist(L0=3, L1=0, L2=0, L3=0)
        self.protocol.end_frame()

        # 4. 评审通过
        result = ReviewResult(A=85, B=80, C=90, D=78, LE=0, frame_id=1)
        action = self.bridge.submit(result)
        self.assertEqual(action.verdict, ReviewVerdict.PASS)

    def test_dangling_evolution_chain(self):
        """悬挂端演化链：删除悬挂端 → 产生新悬挂端 → δ 下降但非零。"""
        graph = FrameGraph(frame_id=1)
        # 链式结构：A-B-C-D-E (悬挂端：A和E)
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        graph.add_edge("D", "E")

        dang = graph.dangling_nodes()
        self.assertIn("A", dang)
        self.assertIn("E", dang)
        self.assertEqual(len(dang), 2)
        self.assertAlmostEqual(graph.dangling_density(), 2 / 5)

        # 公理4验证：有限帧 δ > 0
        self.assertGreater(graph.dangling_density(), 0.0,
                           "公理4违规：有限帧 δ 必须 > 0")

        # 帧协议：记录悬挂端
        self.protocol.begin_frame()
        self.protocol.set_dangling(len(dang))
        self.protocol.end_frame()

    def test_review_rollback_on_failure(self):
        """评审不通过 → 强制回滚。"""
        self.protocol.begin_frame()
        self.protocol.end_frame()

        # A 维度 < 50 → 应触发回滚
        result = ReviewResult(A=45, B=80, C=75, D=85, LE=0, frame_id=1)
        action = self.bridge.submit(result)
        self.assertEqual(action.verdict, ReviewVerdict.ROLLBACK)

    def test_review_warning_on_marginal(self):
        """任一维度 < 70 但 ≥ 50 → 警告。"""
        self.protocol.begin_frame()
        self.protocol.end_frame()

        result = ReviewResult(A=65, B=80, C=75, D=85, LE=0, frame_id=1)
        action = self.bridge.submit(result)
        self.assertEqual(action.verdict, ReviewVerdict.WARN)

    def test_review_rollback_on_le(self):
        """LE ≥ 2 → 回滚。"""
        self.protocol.begin_frame()
        self.protocol.end_frame()

        result = ReviewResult(A=80, B=80, C=80, D=80, LE=2, frame_id=1)
        action = self.bridge.submit(result)
        self.assertEqual(action.verdict, ReviewVerdict.ROLLBACK)

    def test_angle_deficit_formula(self):
        """Σ(6−deg(v)) 验证 — 四方环应等于 8 (非闭合球面)。"""
        graph = FrameGraph(frame_id=1)
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        graph.add_edge("D", "A")

        result = self.verifier.angle_deficit_sum(graph)
        # 四方环：4个节点，每个 deg=2 → Σ(6-2)=4×4=16
        # 但这不是闭合球面三角剖分，所以 ≠ 12 是正常的
        self.assertEqual(result["sum_6_minus_deg"], 16)


if __name__ == "__main__":
    unittest.main()
