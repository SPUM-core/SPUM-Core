"""
SPUM 帧引擎单元测试 — spum.frame
=================================

覆盖:
    演化帧/推理帧生命周期
    V⁺/V⁻ 操作与事件记录
    五步序列标记
    悬挂端追踪与闭合判定
    栈溢出检测与历史回退

运行:
    python -m unittest tests.test_frame_engine -v
"""

import unittest

from spum.frame import FrameEngine, FrameState, FrameType
from spum.axioms import Axiom3


class TestFrameEngineLifecycle(unittest.TestCase):
    """帧生命周期。"""

    def setUp(self):
        self.engine = FrameEngine()

    def test_new_evolution_frame(self):
        f = self.engine.new_evolution_frame()
        self.assertEqual(f.frame_type, FrameType.EVOLUTION)
        self.assertEqual(self.engine.frame_id, 1)

    def test_new_reasoning_frame(self):
        f = self.engine.new_reasoning_frame(notes=["推导引力本质"])
        self.assertEqual(f.frame_type, FrameType.REASONING)
        self.assertIn("推导引力本质", f.notes)

    def test_commit_without_frame_raises(self):
        with self.assertRaises(RuntimeError):
            self.engine.commit()

    def test_commit_computes_metrics(self):
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        self.engine.v_plus("B", "C")
        frame = self.engine.commit()
        self.assertEqual(len(frame.neighbors), 3)
        self.assertEqual(frame.edge_count, 2)
        self.assertEqual(frame.dangling_count, 2)  # A, C
        self.assertFalse(frame.is_closed)

    def test_closed_frame_cycle(self):
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        self.engine.v_plus("B", "C")
        self.engine.v_plus("C", "A")
        frame = self.engine.commit()
        self.assertTrue(frame.is_closed)
        self.assertEqual(self.engine.unclosed_streak, 0)


class TestVPulsVMinus(unittest.TestCase):
    """V⁺/V⁻ 操作。"""

    def setUp(self):
        self.engine = FrameEngine()

    def test_v_plus_auto_creates_frame(self):
        """无活跃帧时 V⁺ 自动开新演化帧。"""
        result = self.engine.v_plus("A", "B")
        self.assertEqual(result["event"], "V⁺")
        self.assertEqual(result["new_nodes"], ["A", "B"])

    def test_v_minus_without_frame_raises(self):
        with self.assertRaises(RuntimeError):
            self.engine.v_minus("A", "B")

    def test_v_minus_removes_node_when_degree_zero(self):
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        result = self.engine.v_minus("A", "B")
        self.assertEqual(result["event"], "V⁻")
        self.assertIn("A", result["removed_nodes"])
        self.assertIn("B", result["removed_nodes"])

    def test_v_plus_events_recorded(self):
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        self.engine.v_plus("C", "D")
        frame = self.engine.commit()
        self.assertEqual(len(frame.v_plus_events), 2)

    def test_v_minus_events_recorded(self):
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        self.engine.v_plus("B", "C")
        self.engine.v_minus("B", "C")
        frame = self.engine.commit()
        self.assertEqual(len(frame.v_minus_events), 1)


class TestFiveSteps(unittest.TestCase):
    """五步序列标记（推理帧）。"""

    def setUp(self):
        self.engine = FrameEngine()

    def test_step_marking(self):
        self.engine.new_reasoning_frame()
        for i in range(1, 6):
            self.engine.step(i)
        frame = self.engine.commit()
        self.assertTrue(frame.all_steps_done)

    def test_step_invalid_number(self):
        self.engine.new_reasoning_frame()
        with self.assertRaises(ValueError):
            self.engine.step(6)

    def test_step_without_frame_raises(self):
        with self.assertRaises(RuntimeError):
            self.engine.step(1)

    def test_step_notes(self):
        self.engine.new_reasoning_frame()
        self.engine.step(1, "建立新关系")
        frame = self.engine.commit()
        self.assertTrue(any("[步骤1]" in n for n in frame.notes))

    def test_add_relation_reasoning_only(self):
        self.engine.new_evolution_frame()
        with self.assertRaises(RuntimeError):
            self.engine.add_relation("N001", "N002", "derives_from")

    def test_add_relation_records(self):
        self.engine.new_reasoning_frame()
        self.engine.add_relation("N005", "N006", "derives_from", "闭合推导")
        frame = self.engine.commit()
        self.assertIn("N005--[derives_from]-->N006", " ".join(frame.notes))

    def test_set_dangling(self):
        self.engine.new_reasoning_frame()
        self.engine.set_dangling(2, ["假设A未验证", "概念B未追溯"])
        # set_dangling 直接作用于活跃帧
        active = self.engine.last_frame()
        self.assertEqual(active.dangling_count, 2)
        self.assertEqual(len(active.dangling_nodes), 2)
        # commit 会基于邻居图重新计算悬挂端（推理帧无关系 → 悬挂 0）
        frame = self.engine.commit()
        self.assertEqual(frame.dangling_count, 0)
        self.assertEqual(frame.dangling_nodes, [])


class TestStackOverflowAndRollback(unittest.TestCase):
    """栈溢出检测与历史回退。"""

    def setUp(self):
        self.engine = FrameEngine()

    def _open_chain(self):
        """连续未闭合帧。"""
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        self.engine.commit()

    def test_stack_overflow_after_three_unclosed(self):
        for _ in range(3):
            self._open_chain()
        self.assertTrue(self.engine.is_stack_overflow())

    def test_no_overflow_when_closed(self):
        for _ in range(3):
            self.engine.new_evolution_frame()
            self.engine.v_plus("A", "B")
            self.engine.v_plus("B", "C")
            self.engine.v_plus("C", "A")
            self.engine.commit()
        self.assertFalse(self.engine.is_stack_overflow())

    def test_rollback_to_closed_frame(self):
        # 帧1: 闭合 → 回退点
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        self.engine.v_plus("B", "C")
        self.engine.v_plus("C", "A")
        self.engine.commit()
        # 帧2: 未闭合
        self._open_chain()
        # 回退到帧1
        rolled = self.engine.rollback()
        self.assertIsNotNone(rolled)
        self.assertEqual(rolled.frame_id, 1)
        self.assertEqual(self.engine.frame_id, 1)

    def test_rollback_without_rollback_point(self):
        engine = FrameEngine()
        self._open_chain()  # 无闭合帧 → 无回退点
        self.assertIsNone(engine.rollback())

    def test_get_rollback_frame(self):
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        self.engine.v_plus("B", "C")
        self.engine.v_plus("C", "A")
        self.engine.commit()
        self.assertEqual(self.engine.get_rollback_frame(), 1)


class TestFrameStateAndSummary(unittest.TestCase):
    """帧状态导出与摘要。"""

    def test_frame_state_to_dict(self):
        self.engine = FrameEngine()
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        frame = self.engine.commit()
        d = frame.to_dict()
        self.assertEqual(d["frame_id"], 1)
        self.assertEqual(d["edges"], 1)
        self.assertEqual(d["sigma"], 2 / 1)
        self.assertIn("v_plus", d)

    def test_summary_counts(self):
        self.engine = FrameEngine()
        self.engine.new_evolution_frame()
        self.engine.v_plus("A", "B")
        self.engine.v_plus("B", "C")
        self.engine.v_plus("C", "A")
        self.engine.commit()
        self.engine.new_reasoning_frame()
        self.engine.commit()
        s = self.engine.summary()
        self.assertIn("2 帧", s)
        self.assertIn("演化1/推理1", s)

    def test_last_frame(self):
        engine = FrameEngine()
        self.assertIsNone(engine.last_frame())
        engine.new_evolution_frame()
        engine.v_plus("A", "B")
        f = engine.last_frame()
        self.assertEqual(f.frame_id, 1)

    def test_history_immutable_snapshot(self):
        engine = FrameEngine()
        engine.new_evolution_frame()
        engine.v_plus("A", "B")
        engine.commit()
        history = engine.history
        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0], FrameState)

    def test_total_edges(self):
        engine = FrameEngine()
        self.assertEqual(engine.total_edges, 0)
        engine.new_evolution_frame()
        engine.v_plus("A", "B")
        engine.v_plus("B", "C")
        self.assertEqual(engine.total_edges, 2)

    def test_format_frame(self):
        engine = FrameEngine()
        self.assertEqual(engine.format_frame(), "[无帧]")
        engine.new_evolution_frame()
        engine.v_plus("A", "B")
        text = engine.format_frame()
        self.assertIn("[帧1]", text)
        self.assertIn("演化帧", text)


if __name__ == "__main__":
    unittest.main()
