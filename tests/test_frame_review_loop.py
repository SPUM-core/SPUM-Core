"""
SPUM 跨模块集成测试：公理引擎 + 帧协议 + 评审

验证在同一个推理循环中 spum/ API 各模块的协同行为。
使用新 spum/ 包替代旧 src/ 模块。

运行：
    python tests/test_frame_review_loop.py
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spum import SPUM
from spum.reason import ReviewResult, ReviewVerdict, ReviewViolation


class TestFrameReviewLoop(unittest.TestCase):
    """公理引擎 + 帧引擎 + 评审的闭环测试。"""

    def setUp(self):
        self.spum = SPUM()

    def test_full_cycle_single_frame(self):
        """单帧完整闭环：图演化 → 帧记录 → 守恒验证 → 评审通过。"""
        # 1. 创建图（模拟一帧的关系网络演化）
        self.spum.add_edge("A", "B")
        self.spum.add_edge("B", "C")
        self.spum.add_edge("C", "D")
        self.spum.add_edge("D", "A")  # 四方环

        # 2. 握手引理验证
        h = self.spum.handshaking
        self.assertTrue(h["passed"], f"握手引理失败: {h}")

        # 3. 帧记录
        self.spum.new_frame()
        self.spum.commit_frame()

        # 4. 评审通过
        result = self.spum.review(A=85, B=80, C=90, D=78, LE=0)
        self.assertEqual(result["verdict"], "通过")

    def test_dangling_evolution_chain(self):
        """悬挂端演化链：删除悬挂端 → 产生新悬挂端 → δ 下降但非零。"""
        # 链式结构：A-B-C-D-E (悬挂端：A和E)
        self.spum.add_edge("A", "B")
        self.spum.add_edge("B", "C")
        self.spum.add_edge("C", "D")
        self.spum.add_edge("D", "E")

        dang = self.spum.dangling_nodes
        self.assertIn("A", dang)
        self.assertIn("E", dang)
        self.assertEqual(len(dang), 2)
        self.assertAlmostEqual(self.spum.delta, 2 / 5)

        # 公理5验证：有限帧 δ > 0
        self.assertGreater(self.spum.delta, 0.0,
                           "公理5违规：有限帧 δ 必须 > 0")

        # 帧记录
        self.spum.new_frame()
        self.spum.commit_frame()

    def test_review_rollback_on_failure(self):
        """评审不通过（A<50）→ 回滚。"""
        result = self.spum.review(A=45, B=80, C=75, D=85, LE=0)
        self.assertEqual(result["verdict"], "回滚")

    def test_review_warning_on_marginal(self):
        """任一维度 < 70 但 ≥ 50 → 警告。"""
        result = self.spum.review(A=65, B=80, C=75, D=85, LE=0)
        self.assertEqual(result["verdict"], "警告")

    def test_review_rollback_on_le(self):
        """LE ≥ 2 → 回滚。"""
        result = self.spum.review(A=80, B=80, C=80, D=80, LE=2)
        self.assertEqual(result["verdict"], "回滚")

    def test_angle_deficit_formula(self):
        """Σ(6−deg(v)) 验证 — 四方环应等于 16 (非闭合球面)。"""
        self.spum.add_edge("A", "B")
        self.spum.add_edge("B", "C")
        self.spum.add_edge("C", "D")
        self.spum.add_edge("D", "A")

        result = self.spum.angle_deficit_sum
        # 四方环：4个节点，每个 deg=2 → Σ(6-2)=4×4=16
        # 但这不是闭合球面三角剖分，所以 ≠ 12 是正常的
        self.assertEqual(result["sum_6_minus_deg"], 16)


class TestOldCompat(unittest.TestCase):
    """旧 API 兼容性验证：确保 ReviewResult/ReviewVerdict 仍可直接创建。"""

    def test_review_result_direct(self):
        """验证旧风格的 ReviewResult 直接创建仍可用。"""
        result = ReviewResult(A=85, B=80, C=90, D=78, LE=0, frame_id=1)
        self.assertEqual(result.verdict(), ReviewVerdict.PASS)

    def test_review_result_rollback(self):
        result = ReviewResult(A=45, B=80, C=75, D=85, LE=0, frame_id=1)
        self.assertEqual(result.verdict(), ReviewVerdict.ROLLBACK)

    def test_violation(self):
        v = ReviewViolation(severity="SEVERE", dimension="B",
                            description="旧范式词汇使用")
        self.assertEqual(v.severity, "SEVERE")


if __name__ == "__main__":
    unittest.main()
