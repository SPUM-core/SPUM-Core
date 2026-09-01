"""
SPUM 评审回滚闭环测试 — src/core/review.py
============================================

覆盖:
    ReviewResult 四维判定 (PASS/WARN/ROLLBACK)
    ReviewBridge → FrameProtocol 回滚闭环
    回调机制
    评审历史

运行:
    python -m unittest tests.test_review_bridge -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.frame import FrameProtocol
from core.review import (
    ReviewResult, ReviewBridge, ReviewVerdict, ReviewAction,
    Violation, Severity,
)


class TestReviewResult(unittest.TestCase):
    """四维评审判定。"""

    def test_pass(self):
        r = ReviewResult(A=85, B=80, C=90, D=78, LE=0)
        self.assertEqual(r.verdict(), ReviewVerdict.PASS)

    def test_warn_on_marginal(self):
        r = ReviewResult(A=65, B=80, C=90, D=78, LE=0)
        self.assertEqual(r.verdict(), ReviewVerdict.WARN)
        self.assertIn("A", r.warning_dimensions())

    def test_rollback_below_50(self):
        r = ReviewResult(A=45, B=80, C=90, D=78, LE=0)
        self.assertEqual(r.verdict(), ReviewVerdict.ROLLBACK)
        self.assertIn("A", r.failing_dimensions())

    def test_rollback_le2(self):
        r = ReviewResult(A=85, B=80, C=90, D=78, LE=2)
        self.assertEqual(r.verdict(), ReviewVerdict.ROLLBACK)

    def test_severity_deduction(self):
        self.assertEqual(Severity.FATAL.deduction, 40)
        self.assertEqual(Severity.SEVERE.deduction, 20)
        self.assertEqual(Severity.WARNING.deduction, 10)
        self.assertEqual(Severity.REMINDER.deduction, 5)

    def test_violation_record(self):
        v = Violation(Severity.SEVERE, "B", "旧范式词汇")
        self.assertEqual(v.dimension, "B")
        self.assertEqual(v.severity, Severity.SEVERE)

    def test_report_format(self):
        r = ReviewResult(A=85, B=80, C=90, D=78, LE=0, frame_id=2)
        report = r.format_report()
        self.assertIn("REVIEW REPORT 帧2", report)
        self.assertIn("判定：通过", report)

    def test_is_rollback_helper(self):
        r = ReviewResult(A=40, B=80, C=90, D=78, LE=0)
        self.assertTrue(r.is_rollback())


class TestReviewBridge(unittest.TestCase):
    """评审 → 回滚闭环。"""

    def setUp(self):
        self.protocol = FrameProtocol()
        self.bridge = ReviewBridge(self.protocol)

    def _make_closed_frame(self):
        self.protocol.begin_frame()
        self.protocol.set_dangling(0)
        self.protocol.end_frame()

    def test_pass_no_rollback(self):
        self._make_closed_frame()
        action = self.bridge.submit(ReviewResult(A=85, B=80, C=90, D=78, LE=0))
        self.assertEqual(action.verdict, ReviewVerdict.PASS)
        self.assertIsNone(action.rollback_frame)
        self.assertEqual(self.protocol.frame_id, 1)

    def test_warn_message(self):
        self._make_closed_frame()
        action = self.bridge.submit(ReviewResult(A=65, B=80, C=90, D=78, LE=0))
        self.assertEqual(action.verdict, ReviewVerdict.WARN)
        self.assertIn("警告", action.message)

    def test_rollback_triggers_protocol_rollback(self):
        # 帧1: 闭合
        self._make_closed_frame()
        # 帧2: 未闭合
        self.protocol.begin_frame()
        self.protocol.set_dangling(2)
        self.protocol.end_frame()

        action = self.bridge.submit(ReviewResult(A=45, B=80, C=90, D=78, LE=0))
        self.assertEqual(action.verdict, ReviewVerdict.ROLLBACK)
        self.assertEqual(action.rollback_frame, 1)
        # 协议已回退到帧1
        self.assertEqual(self.protocol.frame_id, 1)
        self.assertEqual(len(self.protocol.history), 1)

    def test_rollback_without_rollback_point(self):
        action = self.bridge.submit(ReviewResult(A=45, B=80, C=90, D=78, LE=0))
        self.assertEqual(action.verdict, ReviewVerdict.ROLLBACK)
        self.assertIsNone(action.rollback_frame)
        self.assertIn("重置", action.message)

    def test_submit_with_callbacks(self):
        self._make_closed_frame()
        self.protocol.begin_frame()
        self.protocol.set_dangling(1)
        self.protocol.end_frame()

        rolled_back = []
        action = self.bridge.submit_with_callbacks(
            ReviewResult(A=40, B=80, C=90, D=78, LE=0),
            on_rollback=lambda a: rolled_back.append(a),
        )
        self.assertEqual(len(rolled_back), 1)
        self.assertEqual(action.verdict, ReviewVerdict.ROLLBACK)

    def test_review_history(self):
        self._make_closed_frame()
        self.bridge.submit(ReviewResult(A=85, B=80, C=90, D=78, LE=0))
        self.bridge.submit(ReviewResult(A=85, B=80, C=90, D=78, LE=1))
        self.assertEqual(len(self.bridge.review_history), 2)
        self.assertIsNotNone(self.bridge.last_result())
        self.assertIsNotNone(self.bridge.last_action())

    def test_summary_counts(self):
        self._make_closed_frame()
        self.bridge.submit(ReviewResult(A=85, B=80, C=90, D=78, LE=0))
        self.bridge.submit(ReviewResult(A=65, B=80, C=90, D=78, LE=0))
        s = self.bridge.summary()
        self.assertIn("2 次", s)

    def test_auto_frame_association(self):
        """frame_id=0 时自动关联当前帧号。"""
        self._make_closed_frame()
        result = ReviewResult(A=85, B=80, C=90, D=78, LE=0)
        action = self.bridge.submit(result)
        self.assertEqual(result.frame_id, 1)
        self.assertIsNotNone(action)


if __name__ == "__main__":
    unittest.main()
