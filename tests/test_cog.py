"""
COG 模块测试 — 认知投影论（SPUM_认知投影论.md v4.1）
=====================================================

覆盖 COG-001~007 节点的代码对应物（src/spum_graph/closure.py）：

    COG-001~004, COG-007    — 理论命题（范式五元组/双重投影/不可通约/认知热力学/元认知自举），
                              注册完整性由 TestCogRegistry 验证（module_nodes.txt + edges.txt）
    COG-005 关系闭环度 Φ     — TestCogPhiIntegration：COG 边集自洽性（DAG → C=S=1）
                              及三参量代理性质（S ≤ C 恒成立，评审点 1.2）
    COG-006 代表性边集自举   — TestBootstrapSelection：ε*=argmax Φ(ε) 的正确性、
                              固定点语义（§7.5.1）、跨进程确定性、边界退化、自环集成

运行:
    python -m pytest tests/test_cog.py -v
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spum_graph.closure import (
    ClosureScore,
    PhiReport,
    bootstrap_selection,
    find_directed_cycle,
    load_knowledge_edges,
    steadiness,
    self_consistency,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
NODES_PATH = os.path.join(ROOT, "network", "module_nodes.txt")
EDGES_PATH = os.path.join(ROOT, "network", "edges.txt")

COG_IDS = {f"COG-{i:03d}" for i in range(1, 8)}
COG_BRIDGES = {
    ("N021", "COG-001"),
    ("N019", "COG-002"),
    ("N002", "COG-003"),
    ("N012", "COG-004"),
    ("N022", "COG-005"),
    ("COG-005", "COG-006"),
    ("COG-001", "COG-007"),
}

# 自举测试图：环形 + 叶边（覆盖 {A,B,C}，最优 ε = 4 条边，Φ = 0.75·0.25 = 0.1875）
RING_CAND = {("A", "B"), ("B", "C"), ("C", "A"), ("C", "D")}
RING_COVER = {"A", "B", "C"}
# 链图：A 只能度 1 → missing
CHAIN_CAND = {("A", "B"), ("B", "C"), ("C", "D")}
CHAIN_COVER = {"A", "B", "C"}
# 自环图：A→A 自环 + A→B（min_degree=2 时 B 必缺失；验证 Tarjan/MFAS 的自环集成）
SELF_LOOP_CAND = {("A", "A"), ("A", "B")}
SELF_LOOP_COVER = {"A", "B"}


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def _run_python(code: str, hash_seeds=(0, 1, 7)) -> str:
    """在不同 PYTHONHASHSEED 下运行代码，断言跨进程结果一致，返回输出。

    验证确定性链（find_directed_cycle 排序遍历 + greedy_mfas min 字典序 +
    Tarjan SCC）不依赖字符串哈希序（评审点 2.2-1）。
    """
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    outs = set()
    for seed in hash_seeds:
        env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=src)
        r = subprocess.run(
            [sys.executable, "-c", code],
            env=env, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"跨进程结果不一致：{outs}"
    return outs.pop()


class TestCogRegistry:
    """COG 模块注册完整性（module_nodes.txt + edges.txt 双向一致）。"""

    def test_module_nodes_has_all_cog(self):
        lines = _read_lines(NODES_PATH)
        body = [ln for ln in lines if ln and not ln.startswith("#")]
        registered = {ln.split("|")[0].strip() for ln in body}
        assert COG_IDS <= registered

    def test_edges_has_all_cog_bridges(self):
        arcs = load_knowledge_edges(EDGES_PATH)
        cog_arcs = {(u, v) for u, v in arcs
                    if u.startswith("COG") or v.startswith("COG")}
        assert COG_BRIDGES <= cog_arcs

    def test_cog_nodes_consistent_between_files(self):
        """edges.txt 中的 COG 节点 ⊆ module_nodes.txt 注册的 COG 节点。"""
        arcs = load_knowledge_edges(EDGES_PATH)
        edge_cog = {n for u, v in arcs if u.startswith("COG") or v.startswith("COG")
                    for n in (u, v) if n.startswith("COG")}
        lines = _read_lines(NODES_PATH)
        body = [ln for ln in lines if ln and not ln.startswith("#")]
        node_cog = {ln.split("|")[0].strip() for ln in body
                    if ln.split("|")[0].strip().startswith("COG")}
        assert edge_cog <= node_cog


class TestCogPhiIntegration:
    """COG-005：认知投影论自身的 Φ——元层边集自洽（无矛盾环）。"""

    @pytest.fixture
    def cog_arcs(self):
        arcs = load_knowledge_edges(EDGES_PATH)
        return {(u, v) for u, v in arcs
                if u.startswith("COG") or v.startswith("COG")}

    def test_cog_subgraph_is_acyclic(self, cog_arcs):
        assert find_directed_cycle(cog_arcs) is None

    def test_cog_self_consistency_is_one(self, cog_arcs):
        assert self_consistency(cog_arcs) == 1.0

    def test_cog_steadiness_is_one(self, cog_arcs):
        # 无环边 → 任意单边扰动不破坏矛盾环结构（无环可破坏）
        assert steadiness(cog_arcs) == 1.0

    def test_cog_closure_score(self, cog_arcs):
        s = ClosureScore(arcs=cog_arcs, patch_edges={("_patch_", "")})
        assert s.report.C == 1.0
        assert s.report.S == 1.0
        assert s.report.R == 1
        assert s.report.phi == pytest.approx(1.0)
        assert s.report.level() == "成熟体系"

    def test_cog_related_edges_acyclic(self):
        """认知投影论 + 桥接依赖（N→COG 桥接边 + COG 内部边）整体无环。

        桥接边从 N 单向指向 COG、COG 内部仅 COG-005→COG-006 与
        COG-001→COG-007，无 COG→N 回边 → 元层对底层依赖无循环。
        """
        arcs = load_knowledge_edges(EDGES_PATH)
        related = {(u, v) for u, v in arcs
                   if u.startswith("COG") or v.startswith("COG")}
        assert COG_BRIDGES == related          # COG 相关边恰为 7 条桥接
        assert find_directed_cycle(related) is None
        s = ClosureScore(arcs=related, patch_edges={("_patch_", "")})
        assert s.report.phi == pytest.approx(1.0)
        assert s.report.level() == "成熟体系"

    def test_s_le_c_property(self):
        """评审点 1.2 固化：静态代理下 S ≤ C 恒成立（MFAS ⊆ 环边集）。

        这是代理口径的**行为契约**——测试记录 C、S 并非独立维度，
        Φ = (C·S)/R 的分子对环结构信号两次计数。
        """
        arcs_list = [
            RING_CAND, CHAIN_CAND, SELF_LOOP_CAND,
            {(u, v) for u, v in load_knowledge_edges(EDGES_PATH)
             if u.startswith("COG") or v.startswith("COG")},
        ]
        for arcs in arcs_list:
            C = self_consistency(arcs)
            S = steadiness(arcs)
            assert S <= C + 1e-12, f"S={S} > C={C} 违反代理性质"


class TestBootstrapSelection:
    """COG-006：代表性边集自举 ε* = argmax Φ(ε) s.t. 覆盖约束。"""

    def test_exhaustive_finds_ring_optimum(self):
        ex = bootstrap_selection(RING_CAND, RING_COVER, exhaustive=True)
        assert ex.phi == pytest.approx(0.1875)
        assert len(ex.best_arcs) == 4
        assert ex.missing == set()

    def test_greedy_reaches_exhaustive_optimum(self):
        ex = bootstrap_selection(RING_CAND, RING_COVER, exhaustive=True)
        gr = bootstrap_selection(RING_CAND, RING_COVER, exhaustive=False)
        assert abs(gr.phi - ex.phi) < 1e-9

    def test_greedy_meets_coverage(self):
        gr = bootstrap_selection(RING_CAND, RING_COVER)
        assert RING_COVER <= gr.coverage
        assert gr.missing == set()

    def test_missing_when_min_degree_unreachable(self):
        # A 只有一条候选边 → 无法达 min_degree=2 → 报告缺失
        m = bootstrap_selection(CHAIN_CAND, CHAIN_COVER)
        assert m.missing == {"A"}
        assert m.phi > 0.0  # 剩余骨架仍是合法边集，Φ 恒正（不完美定理）

    def test_phi_positive_always(self):
        for candidates, cover in ((RING_CAND, RING_COVER), (CHAIN_CAND, CHAIN_COVER)):
            for exhaustive in (True, False):
                r = bootstrap_selection(candidates, cover, exhaustive=exhaustive)
                assert r.phi > 0.0
                assert r.R >= 1

    def test_min_degree_one_relaxes_constraint(self):
        # min_degree=1：链图中 A 只出现一次即达标 → missing 空
        m = bootstrap_selection(CHAIN_CAND, CHAIN_COVER, min_degree=1)
        assert m.missing == set()

    def test_exhaustive_evaluation_count(self):
        # 4 条候选 → 穷举至多 2^4 次评估（覆盖约束提前过滤）
        ex = bootstrap_selection(RING_CAND, RING_COVER, exhaustive=True)
        assert ex.evaluated <= 1 << len(RING_CAND)
        assert ex.exhaustive is True

    def test_deterministic(self):
        a = bootstrap_selection(RING_CAND, RING_COVER)
        b = bootstrap_selection(RING_CAND, RING_COVER)
        assert a.best_arcs == b.best_arcs
        assert a.phi == b.phi

    def test_format_report_fields(self):
        r = bootstrap_selection(RING_CAND, RING_COVER)
        text = r.format_report()
        assert "BOOTSTRAP REPORT" in text
        assert "COG-006" in text
        assert "0.1875" in text

    def test_fixed_point_self_bootstrap(self):
        """§7.5.1 固定点语义：以 ε* 为候选再自举，Φ 与 ε* 不变。

        ε* 是候选域内 Φ 的 argmax——再次自举（候选 ⊆ 原候选）不会
        得到更高 Φ，验证"用 Φ 选 ε 不构成循环论证"是可计算事实。
        """
        r1 = bootstrap_selection(RING_CAND, RING_COVER, exhaustive=True)
        r2 = bootstrap_selection(r1.best_arcs, RING_COVER, exhaustive=True)
        assert r2.phi == pytest.approx(r1.phi)
        assert r2.best_arcs == r1.best_arcs

    def test_cross_process_deterministic(self):
        """确定性链（排序遍历 + min 字典序 + Tarjan）不依赖哈希序。"""
        out = _run_python(
            "from spum_graph.closure import bootstrap_selection; "
            "r = bootstrap_selection({('A','B'),('B','C'),('C','A'),('C','D')}, {'A','B','C'}); "
            "print(sorted(r.best_arcs), f'{r.phi:.6f}')"
        )
        assert out == "[('A', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'D')] 0.187500"

    def test_empty_candidates_degenerate(self):
        """空候选 + 空覆盖：退化默认（Φ=0，无缺失，R 由占位补丁保证）。"""
        for exhaustive in (True, False):
            r = bootstrap_selection(set(), set(), exhaustive=exhaustive)
            assert r.phi == 0.0
            assert r.best_arcs == set()
            assert r.missing == set()
            assert r.R >= 1

    def test_cover_unreachable_all_missing(self):
        """覆盖节点不在任何候选：全报缺失，但仍返回非空尽力集。"""
        for exhaustive in (True, False):
            r = bootstrap_selection({("X", "Y")}, {"A", "B"},
                                    min_degree=1, exhaustive=exhaustive)
            assert r.missing == {"A", "B"}
            assert r.best_arcs == {("X", "Y")}
            assert r.phi > 0.0

    def test_single_edge_degree_one(self):
        """单边 DAG + min_degree=1：满覆盖 → C=S=1 → Φ=1.0。"""
        r = bootstrap_selection({("A", "B")}, {"A", "B"},
                                min_degree=1, exhaustive=True)
        assert r.missing == set()
        assert r.phi == pytest.approx(1.0)

    def test_self_loop_tarjan_integration(self):
        """自环集成：A→A 自环参与 MFAS 与环边判定（Tarjan 正确处理）。

        min_degree=2 时 B 只能度 1 → 必缺失；最优集 {A→A, A→B}：
        MFAS={(A,A)} → C=0.5；环边={A→A} → S=0.5；Φ=0.5·0.5/1=0.25。
        """
        for exhaustive in (True, False):
            r = bootstrap_selection(SELF_LOOP_CAND, SELF_LOOP_COVER,
                                    exhaustive=exhaustive)
            assert r.missing == {"B"}
            assert r.C == pytest.approx(0.5)
            assert r.S == pytest.approx(0.5)
            assert r.phi == pytest.approx(0.25)

    def test_explicit_patches_count(self):
        """显式补丁边集 → R 按其计数（占位补丁仅在缺省时注入）。"""
        patches = {("P", "Q"), ("R", "S")}
        r = bootstrap_selection(RING_CAND, RING_COVER, patch_edges=patches,
                                exhaustive=True)
        assert r.R == 2
        assert r.phi == pytest.approx(0.75 * 0.25 / 2)   # 0.09375


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
