"""
SPUM 帧协议自动化测试 — src/core/frame.py
===========================================

覆盖:
    FrameProtocol 帧生命周期 (begin/end)
    关系合法性校验 (VALID_EDGE_TYPES)
    悬挂端追踪与闭合判定
    栈溢出检测
    历史回退
    五步序列
    FrameLogger [帧N] 格式输出

运行:
    python -m unittest tests.test_frame_protocol -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.frame import (
    FrameProtocol, FrameState, FrameLogger, VALID_EDGE_TYPES,
    STACK_OVERFLOW_THRESHOLD,
)


class TestFrameProtocolLifecycle(unittest.TestCase):
    """帧生命周期。"""

    def setUp(self):
        self.protocol = FrameProtocol()

    def test_begin_frame_increments_id(self):
        f1 = self.protocol.begin_frame()
        self.assertEqual(f1.frame_id, 1)
        f2 = self.protocol.begin_frame()
        self.assertEqual(f2.frame_id, 2)

    def test_end_frame_without_begin_raises(self):
        with self.assertRaises(RuntimeError):
            self.protocol.end_frame()

    def test_end_frame_auto_closes(self):
        self.protocol.begin_frame()
        self.protocol.set_dangling(0)
        f = self.protocol.end_frame()
        self.assertTrue(f.is_closed)

    def test_end_frame_open_dangling(self):
        self.protocol.begin_frame()
        self.protocol.set_dangling(3)
        f = self.protocol.end_frame()
        self.assertFalse(f.is_closed)

    def test_set_active_nodes_marks_unknown(self):
        self.protocol.begin_frame()
        self.protocol.set_active_nodes(["N001", "N005", "NEWX"])
        f = self.protocol.end_frame()
        self.assertIn("N001", f.active_nodes)
        self.assertTrue(any("[新节点]" in n for n in f.notes))


class TestRelations(unittest.TestCase):
    """关系添加与校验。"""

    def setUp(self):
        self.protocol = FrameProtocol()

    def test_valid_edge_types(self):
        for et in ["derives_from", "requires", "refines", "explains", "drives"]:
            self.assertIn(et, VALID_EDGE_TYPES)

    def test_add_relation(self):
        self.protocol.begin_frame()
        self.protocol.add_relation("N005", "N006", "derives_from", "闭合推导")
        f = self.protocol.end_frame()
        self.assertEqual(len(f.relations), 1)
        self.assertEqual(f.relations[0][2], "derives_from")

    def test_add_relation_invalid_type(self):
        self.protocol.begin_frame()
        with self.assertRaises(ValueError):
            self.protocol.add_relation("N005", "N006", "magic_links")

    def test_add_relation_without_frame(self):
        with self.assertRaises(RuntimeError):
            self.protocol.add_relation("N005", "N006", "derives_from")


class TestDanglingAndClosure(unittest.TestCase):
    """悬挂端与闭合。"""

    def setUp(self):
        self.protocol = FrameProtocol()

    def test_closure_streak_reset(self):
        self.protocol.begin_frame()
        self.protocol.set_dangling(0)
        self.protocol.end_frame()
        self.assertEqual(self.protocol.unclosed_streak, 0)

    def test_unclosed_streak_increments(self):
        for _ in range(2):
            self.protocol.begin_frame()
            self.protocol.set_dangling(1)
            self.protocol.end_frame()
        self.assertEqual(self.protocol.unclosed_streak, 2)

    def test_stack_overflow_threshold(self):
        self.assertEqual(STACK_OVERFLOW_THRESHOLD, 3)
        for _ in range(3):
            self.protocol.begin_frame()
            self.protocol.set_dangling(1)
            self.protocol.end_frame()
        self.assertTrue(self.protocol.is_stack_overflow())

    def test_rollback_to_closed_frame(self):
        # 帧1: 闭合
        self.protocol.begin_frame()
        self.protocol.set_dangling(0)
        self.protocol.end_frame()
        # 帧2: 未闭合
        self.protocol.begin_frame()
        self.protocol.set_dangling(2)
        self.protocol.end_frame()
        # 回退到帧1
        rolled = self.protocol.rollback()
        self.assertIsNotNone(rolled)
        self.assertEqual(rolled.frame_id, 1)
        self.assertEqual(self.protocol.frame_id, 1)
        self.assertEqual(len(self.protocol.history), 1)

    def test_rollback_without_rollback_point(self):
        self.protocol.begin_frame()
        self.protocol.set_dangling(1)
        self.protocol.end_frame()
        self.assertIsNone(self.protocol.rollback())

    def test_get_rollback_frame(self):
        self.protocol.begin_frame()
        self.protocol.set_dangling(0)
        self.protocol.end_frame()
        self.assertEqual(self.protocol.get_rollback_frame(), 1)


class TestFiveSteps(unittest.TestCase):
    """五步序列。"""

    def test_step_marking(self):
        protocol = FrameProtocol()
        protocol.begin_frame()
        for i in range(1, 6):
            protocol.step(i)
        f = protocol.end_frame()
        self.assertTrue(f.all_steps_done())

    def test_step_invalid(self):
        protocol = FrameProtocol()
        protocol.begin_frame()
        with self.assertRaises(ValueError):
            protocol.step(7)

    def test_step_without_frame(self):
        protocol = FrameProtocol()
        with self.assertRaises(RuntimeError):
            protocol.step(1)

    def test_layer_dist(self):
        protocol = FrameProtocol()
        protocol.begin_frame()
        protocol.set_layer_dist(L0=3, L1=2, L2=1)
        f = protocol.end_frame()
        self.assertEqual(f.layer_dist["L0"], 3)
        self.assertEqual(f.layer_dist["L1"], 2)
        self.assertEqual(f.layer_dist["L2"], 1)


class TestFrameStateValidation(unittest.TestCase):
    """帧状态一致性校验。"""

    def test_validate_consistent(self):
        state = FrameState(frame_id=1, dangling_count=2, is_closed=False)
        self.assertEqual(state.validate(), [])

    def test_validate_inconsistent_closed(self):
        state = FrameState(frame_id=1, dangling_count=2, is_closed=True)
        self.assertGreaterEqual(len(state.validate()), 1)

    def test_validate_dangling_items_mismatch(self):
        state = FrameState(frame_id=1, dangling_count=2,
                           dangling_items=["只有一项"])
        self.assertGreaterEqual(len(state.validate()), 1)

    def test_all_steps_done(self):
        state = FrameState(frame_id=1)
        self.assertFalse(state.all_steps_done())
        for attr in ["step_1_done", "step_2_done", "step_3_done",
                     "step_4_done", "step_5_done"]:
            setattr(state, attr, True)
        self.assertTrue(state.all_steps_done())


class TestFrameLogger(unittest.TestCase):
    """帧日志格式。"""

    def test_format_last_no_frames(self):
        protocol = FrameProtocol()
        self.assertEqual(protocol.logger.format_last(), "[无活跃帧]")

    def test_format_last_full(self):
        protocol = FrameProtocol()
        protocol.begin_frame(notes=["测试帧"])
        protocol.set_active_nodes(["N001", "N005"])
        protocol.set_dangling(1, ["假设未验证"])
        protocol.end_frame()
        text = protocol.logger.format_last()
        self.assertIn("[帧1]", text)
        self.assertIn("N001", text)
        self.assertIn("悬挂端数量：1", text)

    def test_format_summary(self):
        protocol = FrameProtocol()
        protocol.begin_frame()
        protocol.set_dangling(0)
        protocol.end_frame()
        text = protocol.logger.format_summary()
        self.assertIn("帧序列: 1 帧", text)
        self.assertIn("闭合: 1/1", text)

    def test_to_dict(self):
        protocol = FrameProtocol()
        protocol.begin_frame()
        protocol.set_active_nodes(["N001"])
        protocol.set_dangling(0)
        protocol.end_frame()
        d = protocol.to_dict()
        self.assertEqual(d["total_frames"], 1)
        self.assertEqual(d["closed_frames"], 1)


if __name__ == "__main__":
    unittest.main()
