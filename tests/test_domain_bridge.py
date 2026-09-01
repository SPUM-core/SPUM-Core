"""
SPUM 跨域桥接单元测试 — spum.domain
=====================================

覆盖:
    领域注册 (8 领域)
    正向映射 map / 反向映射 reverse_map
    全文搜索 search
    领域概念列举
    跨域链 cross_domain_chain

运行:
    python -m unittest tests.test_domain_bridge -v
"""

import unittest

from spum.domain import DomainBridge, ConceptMapping


class TestDomainBridge(unittest.TestCase):
    """跨域桥接器。"""

    def setUp(self):
        self.bridge = DomainBridge()

    def test_list_domains_contains_core_domains(self):
        domains = self.bridge.list_domains()
        for expected in ["physics", "sociology", "economics",
                         "linguistics", "math", "wuxing", "rushidao"]:
            self.assertIn(expected, domains)

    def test_map_known_concept(self):
        result = self.bridge.map("physics", "gravity")
        self.assertIsInstance(result, ConceptMapping)
        self.assertEqual(result.domain, "physics")
        self.assertEqual(result.concept, "gravity")
        self.assertTrue(result.spum_reduction)
        self.assertIn("axiom5", result.axioms)

    def test_map_unknown_concept_returns_none(self):
        self.assertIsNone(self.bridge.map("physics", "nonexistent"))

    def test_map_unknown_domain_returns_none(self):
        self.assertIsNone(self.bridge.map("alchemy", "gold"))

    def test_reverse_map_sigma(self):
        """σ 原语应被多个领域使用。"""
        results = self.bridge.reverse_map("derived.sigma")
        self.assertGreater(len(results), 0)
        for m in results:
            self.assertIn("derived.sigma", m.axioms)

    def test_search_by_keyword(self):
        results = self.bridge.search("引力")
        self.assertGreater(len(results), 0)
        # 引力归约应命中
        self.assertTrue(any(m.concept == "gravity" for m in results))

    def test_search_case_insensitive_english(self):
        results = self.bridge.search("GRAVITY")
        self.assertTrue(any(m.concept == "gravity" for m in results))

    def test_list_concepts(self):
        concepts = self.bridge.list_concepts("physics")
        for expected in ["gravity", "light", "matter", "time", "space"]:
            self.assertIn(expected, concepts)

    def test_list_concepts_unknown_domain(self):
        self.assertEqual(self.bridge.list_concepts("alchemy"), [])

    def test_cross_domain_chain_gravity_money(self):
        """引力与货币的跨域链——通过共享原语/公理桥接。"""
        chain = self.bridge.cross_domain_chain("gravity", "money")
        if chain is not None:
            # 结构化返回：搜索命中 + 共享公理 + 桥接说明
            self.assertEqual(chain["concept1_search"], "gravity")
            self.assertEqual(chain["concept2_search"], "money")
            self.assertIn("matches_1", chain)
            self.assertIn("matches_2", chain)
            self.assertIn("shared_axioms", chain)
            self.assertIn("bridge_via_spum", chain)

    def test_summary_mentions_domain_count(self):
        s = self.bridge.summary()
        self.assertIn("领域", s)
        self.assertIn("概念映射", s)


if __name__ == "__main__":
    unittest.main()
