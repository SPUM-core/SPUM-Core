"""
SPUM 推理引擎单元测试 — spum.reason
=====================================

覆盖:
    推导链 derive (BFS)
    多路径 all_paths
    多路径锁定 multi_path_validate
    四维评审 review / auto_review_from_text
    知识图谱查询 query_node / query_edge
    前向/后向推理
    推导完整性验证

运行:
    python -m unittest tests.test_reasoning_engine -v
"""

import unittest

from spum.reason import (
    ReasoningEngine, DerivationChain, DerivationStep,
    ReviewResult, ReviewVerdict, ReviewViolation,
    MultiPathResult, CORE_NODES, CORE_EDGES,
)


class TestDerivationChain(unittest.TestCase):
    """推导链。"""

    def setUp(self):
        self.engine = ReasoningEngine()

    def test_derive_direct_edge(self):
        chain = self.engine.derive("N001", "N002")
        self.assertTrue(chain.is_complete)
        self.assertGreaterEqual(len(chain.steps), 1)
        self.assertEqual(chain.source, "N001")
        self.assertEqual(chain.target, "N002")

    def test_derive_multi_hop(self):
        chain = self.engine.derive("N001", "N004")
        self.assertTrue(chain.is_complete)
        # N001 → N002 → N003 → N004
        self.assertGreaterEqual(len(chain.steps), 3)

    def test_derive_unknown_source(self):
        chain = self.engine.derive("N999", "N004")
        self.assertFalse(chain.is_complete)
        self.assertEqual(chain.steps, [])

    def test_derive_unknown_target(self):
        chain = self.engine.derive("N001", "N999")
        self.assertFalse(chain.is_complete)

    def test_derive_path_str_format(self):
        chain = self.engine.derive("N001", "N002")
        self.assertIn("N001", chain.path_str)
        self.assertIn("N002", chain.path_str)

    def test_chain_to_dict(self):
        chain = self.engine.derive("N001", "N002")
        d = chain.to_dict()
        self.assertEqual(d["source"], "N001")
        self.assertEqual(d["target"], "N002")
        self.assertTrue(d["is_complete"])
        self.assertIn("path", d)

    def test_all_paths(self):
        paths = self.engine.all_paths("N001", "N013", max_paths=3)
        self.assertGreaterEqual(len(paths), 1)
        for p in paths:
            self.assertTrue(p.is_complete)
        # 按长度排序
        lengths = [len(p.steps) for p in paths]
        self.assertEqual(lengths, sorted(lengths))


class TestMultiPathValidation(unittest.TestCase):
    """多路径锁定。"""

    def test_locked_when_all_confirmed(self):
        result = self.engine = ReasoningEngine().multi_path_validate(
            conclusion="旋转曲线无需暗物质",
            paths={"子图协动": True, "σ梯度": True, "边缘衰减": True},
        )
        self.assertTrue(result.locked)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.independent_paths, 3)
        self.assertEqual(result.confirmed_paths, 3)

    def test_not_locked_with_failure(self):
        engine = ReasoningEngine()
        result = engine.multi_path_validate(
            conclusion="X",
            paths={"路径A": True, "路径B": False},
        )
        self.assertFalse(result.locked)
        self.assertEqual(result.confidence, 0.5)

    def test_single_path_not_enough(self):
        engine = ReasoningEngine()
        result = engine.multi_path_validate("X", {"单一路径": True})
        self.assertFalse(result.locked)


class TestReview(unittest.TestCase):
    """四维评审。"""

    def setUp(self):
        self.engine = ReasoningEngine()

    def test_review_pass(self):
        result = self.engine.review(A=85, B=80, C=90, D=78, LE=0)
        self.assertEqual(result.verdict(), ReviewVerdict.PASS)

    def test_review_warn(self):
        result = self.engine.review(A=65, B=80, C=90, D=78, LE=0)
        self.assertEqual(result.verdict(), ReviewVerdict.WARN)

    def test_review_rollback_low_score(self):
        result = self.engine.review(A=45, B=80, C=90, D=78, LE=0)
        self.assertEqual(result.verdict(), ReviewVerdict.ROLLBACK)

    def test_review_rollback_le(self):
        result = self.engine.review(A=80, B=80, C=90, D=78, LE=2)
        self.assertEqual(result.verdict(), ReviewVerdict.ROLLBACK)

    def test_review_report_format(self):
        result = self.engine.review(A=85, B=80, C=90, D=78, LE=0, frame_id=3)
        report = result.format_report()
        self.assertIn("REVIEW REPORT 帧3", report)
        self.assertIn("A 架构纯净度", report)
        self.assertIn("通过", report)

    def test_review_failing_dimensions(self):
        result = self.engine.review(A=45, B=80, C=90, D=78, LE=2)
        self.assertIn("A", result.failing_dimensions())
        self.assertIn("LE", result.failing_dimensions())

    def test_auto_review_spum_clean_text(self):
        text = ("σ 梯度归约到 ⟨P, ε⟩：V⁺/V⁻ 是本体层（L0）的重分配操作。"
                "δ 与 Δμ 构成双层判据，对应认知投影（L1）的锚点漂移，"
                "帧演化逐帧推进。")
        result = self.engine.auto_review_from_text(text)
        self.assertEqual(result.verdict(), ReviewVerdict.PASS)
        self.assertEqual(result.LE, 0)

    def test_auto_review_old_paradigm_text(self):
        text = "引力是一种力，能量在弯曲时空中传播，最终趋于热寂。"
        result = self.engine.auto_review_from_text(text)
        self.assertGreater(result.LE, 0)
        self.assertLess(result.B, 80)

    def test_auto_review_with_violations(self):
        text = "啥也没有"
        result = self.engine.auto_review_from_text(text)
        self.assertGreaterEqual(len(result.violations), 1)


class TestKnowledgeGraph(unittest.TestCase):
    """知识图谱查询。"""

    def setUp(self):
        self.engine = ReasoningEngine()

    def test_core_nodes_count(self):
        self.assertEqual(len(CORE_NODES), 22)
        self.assertIn("N001", CORE_NODES)
        self.assertIn("N022", CORE_NODES)

    def test_core_edges_count(self):
        self.assertGreaterEqual(len(CORE_EDGES), 28)

    def test_query_node(self):
        node = self.engine.query_node("N005")
        self.assertIsNotNone(node)
        self.assertEqual(node["id"], "N005")
        self.assertEqual(node["name"], "边/连接")
        self.assertGreater(node["derivation_count"], 0)

    def test_query_unknown_node(self):
        self.assertIsNone(self.engine.query_node("N999"))

    def test_query_edge(self):
        edge = self.engine.query_edge("N001", "N002")
        self.assertIsNotNone(edge)
        self.assertEqual(edge["edge_type"], "requires")

    def test_query_edge_reverse_direction(self):
        """反向查询（N002 → N001）通过 inverse_of 边可达。"""
        edge = self.engine.query_edge("N002", "N001")
        self.assertIsNotNone(edge)
        self.assertEqual(edge["edge_type"], "inverse_of")

    def test_node_list(self):
        nodes = self.engine.node_list()
        self.assertEqual(len(nodes), 22)


class TestChaining(unittest.TestCase):
    """前向/后向推理。"""

    def setUp(self):
        self.engine = ReasoningEngine()

    def test_forward_chain(self):
        result = self.engine.forward_chain(["N001", "N002"])
        self.assertIn("derived", result)
        self.assertGreaterEqual(result["derived_count"], 2)

    def test_forward_chain_ignores_unknown(self):
        result = self.engine.forward_chain(["NOT_A_NODE"])
        self.assertEqual(result["derived"], [])

    def test_backward_chain(self):
        result = self.engine.backward_chain("N013")
        self.assertIn("required_premises", result)
        self.assertGreaterEqual(result["premise_count"], 0)

    def test_backward_chain_unknown_goal(self):
        result = self.engine.backward_chain("不存在概念XYZ")
        self.assertEqual(result["required_premises"], [])


class TestCompleteness(unittest.TestCase):
    """推导完整性验证。"""

    def test_all_nodes_reachable_from_n001(self):
        engine = ReasoningEngine()
        result = engine.verify_derivation_completeness()
        self.assertEqual(result["total_nodes"], 22)
        self.assertTrue(result["is_fully_connected"],
                        f"存在不可达节点: {result['unreachable_from_N001']}")

    def test_summary(self):
        engine = ReasoningEngine()
        s = engine.summary()
        self.assertIn("22 节点", s)


class TestReviewDataClasses(unittest.TestCase):
    """评审数据结构。"""

    def test_violation_creation(self):
        v = ReviewViolation(severity="SEVERE", dimension="B",
                            description="旧范式词汇使用")
        self.assertEqual(v.severity, "SEVERE")
        self.assertEqual(v.dimension, "B")

    def test_derivation_step(self):
        step = DerivationStep("N001", "N002", "requires", "宇宙存在需要差异")
        self.assertEqual(step.source, "N001")
        self.assertEqual(step.target, "N002")

    def test_multi_path_result_properties(self):
        result = MultiPathResult("C", {"A": True, "B": False})
        self.assertEqual(result.independent_paths, 2)
        self.assertEqual(result.confirmed_paths, 1)


if __name__ == "__main__":
    unittest.main()
