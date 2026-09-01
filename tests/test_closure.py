"""
Φ 可计算形式测试 — src/spum_graph/closure.py
=============================================

覆盖（对应 closure.py 的逻辑树拆解）:
    find_directed_cycle   — 有向环检测
    greedy_mfas           — MFAS 贪婪近似（大小 + "删除后无环"有效集性质 + 跨进程确定性）
    cyclic_edges          — 矛盾环边集（Tarjan SCC 线性实现 + 与逐边 BFS 参照等价）
    self_consistency / steadiness / redundancy — 三参量（含 S 代理口径声明）
    PhiReport             — Φ 聚合 / 四层级评级 / 报告格式 / R≥1 不完美定理 / 参量范围校验
    ClosureScore          — 自动模式 / 直接参量模式 / §7.5 复现 / 空图空洞自洽防护
    forecast_transition   — R(t)/R_crit 跃迁预报占位接口（F.4 路线图）
    parse_arc_lines       — 异常格式警告
    run_demo              — 六体系演示输出 + 知识图谱端到端自动段（模式 A）

运行:
    python -m pytest tests/test_closure.py -v
"""

import os
import random
import subprocess
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from spum_graph.closure import (
    ClosureScore,
    PhiReport,
    PhiConfig,
    cyclic_edges,
    estimate_mfas,
    find_directed_cycle,
    forecast_transition,
    greedy_mfas,
    load_knowledge_edges,
    parse_arc_lines,
    redundancy,
    run_demo,
    self_consistency,
    steadiness,
)


# ── 测试数据 ──────────────────────────────────────────────────────────

SINGLE_CYCLE = {("A", "B"), ("B", "C"), ("C", "A")}               # 单矛盾环
CYCLE_PLUS_LEAF = SINGLE_CYCLE | {("C", "D")}                     # 环 + 非环边
TWIN_SHARED = {("A", "B"), ("B", "A"), ("B", "C"), ("C", "A")}    # 两共享边环
TWIN_DISJOINT = {("A", "B"), ("B", "A"), ("C", "D"), ("D", "C")}  # 两不相交环
DAG = {("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")}            # 无环


def _is_acyclic(arcs) -> bool:
    return find_directed_cycle(arcs) is None


class TestFindDirectedCycle:
    def test_single_cycle_found(self):
        cycle = find_directed_cycle(SINGLE_CYCLE)
        assert cycle is not None
        assert len(cycle) >= 3
        assert cycle[0] == cycle[-1]  # 环首尾闭合

    def test_dag_returns_none(self):
        assert find_directed_cycle(DAG) is None

    def test_empty_returns_none(self):
        assert find_directed_cycle(set()) is None

    def test_twin_shared_found(self):
        assert find_directed_cycle(TWIN_SHARED) is not None

    def test_deterministic(self):
        assert find_directed_cycle(SINGLE_CYCLE) == find_directed_cycle(SINGLE_CYCLE)


class TestGreedyMfas:
    def test_single_cycle_size_one(self):
        assert len(greedy_mfas(SINGLE_CYCLE)) == 1

    def test_mfas_removal_is_acyclic(self):
        """通用性质：删除 MFAS 后矛盾环全部消失（有效反馈弧集）。"""
        for arcs in (SINGLE_CYCLE, CYCLE_PLUS_LEAF, TWIN_SHARED, TWIN_DISJOINT):
            fas = greedy_mfas(arcs)
            assert _is_acyclic(arcs - fas)

    def test_dag_mfas_empty(self):
        assert greedy_mfas(DAG) == set()

    def test_twin_disjoint_size_two(self):
        assert len(greedy_mfas(TWIN_DISJOINT)) == 2

    def test_twin_shared_size_one(self):
        # 共享边 A→B 被删一次即可断开两个环
        assert len(greedy_mfas(TWIN_SHARED)) == 1

    def test_deterministic(self):
        assert greedy_mfas(SINGLE_CYCLE) == greedy_mfas(SINGLE_CYCLE)


class TestCyclicEdges:
    def test_single_cycle_edges(self):
        assert cyclic_edges(SINGLE_CYCLE) == SINGLE_CYCLE

    def test_cycle_plus_leaf(self):
        # (C,D) 不参与任何环 → 非环边
        assert cyclic_edges(CYCLE_PLUS_LEAF) == SINGLE_CYCLE

    def test_twin_shared_all_edges(self):
        # A→B 同时属于两环；四条边全在环上
        assert cyclic_edges(TWIN_SHARED) == TWIN_SHARED

    def test_dag_none(self):
        assert cyclic_edges(DAG) == set()


class TestSelfConsistency:
    def test_single_cycle(self):
        assert self_consistency(SINGLE_CYCLE) == pytest.approx(1 - 1 / 3)

    def test_cycle_plus_leaf(self):
        assert self_consistency(CYCLE_PLUS_LEAF) == pytest.approx(0.75)

    def test_dag_is_one(self):
        assert self_consistency(DAG) == 1.0

    def test_explicit_mfas_respected(self):
        # 显式提供 MFAS 时按其计算，不重新推导
        mfas = {("B", "C")}
        assert self_consistency(SINGLE_CYCLE, mfas=mfas) == pytest.approx(1 - 1 / 3)
        assert self_consistency(CYCLE_PLUS_LEAF, mfas=mfas) == pytest.approx(0.75)


class TestSteadiness:
    def test_cycle_plus_leaf(self):
        # 4 条边中 1 条非环边 → S = 1/4
        assert steadiness(CYCLE_PLUS_LEAF) == pytest.approx(0.25)

    def test_twin_shared_zero(self):
        # 全环边 → 任意单边扰动都破坏矛盾环结构 → S = 0
        assert steadiness(TWIN_SHARED) == 0.0

    def test_dag_is_one(self):
        assert steadiness(DAG) == 1.0

    def test_empty_is_one(self):
        assert steadiness(set()) == 1.0


class TestRedundancy:
    def test_count(self):
        assert redundancy({("P", "Q"), ("R", "S")}) == 2

    def test_empty_rejected(self):
        # 不完美定理（N013）禁止 R=0
        with pytest.raises(ValueError):
            redundancy(set())


class TestPhiReport:
    def test_phi_value(self):
        r = PhiReport(C=0.85, S=0.6, R=12)
        assert r.phi == pytest.approx(0.0425)

    def test_phi_rejects_r_lt_1(self):
        with pytest.raises(ValueError):
            PhiReport(C=1.0, S=1.0, R=0).phi

    def test_levels_demo_systems(self):
        """§7.5 六体系的四层级评级（默认阈值）。"""
        cases = [
            (0.850, 0.6, 12, "半闭环叙事"),   # 托勒密
            (0.914, 0.5, 4, "准自洽理论"),    # 哥白尼
            (0.980, 0.9, 3, "准自洽理论"),    # 牛顿
            (0.982, 0.95, 1, "成熟体系"),     # 广义相对论
            (0.867, 0.4, 8, "半闭环叙事"),    # 古典体液说
            (0.967, 0.9, 2, "准自洽理论"),    # 现代病原学
        ]
        for C, S, R, expected in cases:
            assert PhiReport(C=C, S=S, R=R).level() == expected

    def test_open_narrative_very_low_phi(self):
        # R≥20 且 Φ<0.02 → 开放雏形
        r = PhiReport(C=0.5, S=0.3, R=30)
        assert r.level() == "开放雏形"

    def test_custom_config(self):
        cfg = PhiConfig(PHI_MATURE=0.1)
        assert PhiReport(C=0.967, S=0.9, R=2, config=cfg).level() == "成熟体系"

    def test_format_report_fields(self):
        text = PhiReport(C=0.982, S=0.95, R=1).format_report()
        assert "CLOSURE REPORT" in text
        assert "成熟体系" in text
        assert "0.9329" in text


class TestClosureScore:
    def test_auto_mode_full_pipeline(self):
        s = ClosureScore(arcs=CYCLE_PLUS_LEAF, patch_edges={("E", "X")})
        assert s.report.C == pytest.approx(0.75)
        assert s.report.S == pytest.approx(0.25)
        assert s.report.R == 1
        assert s.report.phi == pytest.approx(0.1875)

    def test_auto_mode_missing_arcs_raises(self):
        with pytest.raises(ValueError):
            ClosureScore()

    def test_auto_mode_rejects_empty_patches(self):
        with pytest.raises(ValueError):
            ClosureScore(arcs=SINGLE_CYCLE, patch_edges=set())

    def test_direct_parameter_mode(self):
        s = ClosureScore(C=0.85, S=0.6, R=12, epsilon_size=40, mfas_size=6)
        assert s.report.phi == pytest.approx(0.0425)
        assert s.report.level() == "半闭环叙事"

    def test_explicit_mfas_respected(self):
        mfas = {("B", "C")}
        s = ClosureScore(arcs=CYCLE_PLUS_LEAF, mfas=mfas, patch_edges={("E", "X")})
        assert s.report.mfas_size == 1
        assert s.report.C == pytest.approx(0.75)

    def test_twin_shared_open_narrative(self):
        # 全环边 → S=0 → Φ=0 → 开放雏形
        s = ClosureScore(arcs=TWIN_SHARED, patch_edges={("P", "Q")})
        assert s.report.S == 0.0
        assert s.report.phi == 0.0
        assert s.report.level() == "开放雏形"


class TestEstimateMfas:
    """7.1 MFAS 误差自检：随机重启采样分布。"""

    def test_best_le_deterministic(self):
        # 采样最优不劣于确定性贪婪（对全部测试图）
        for arcs in (SINGLE_CYCLE, CYCLE_PLUS_LEAF, TWIN_SHARED, TWIN_DISJOINT, DAG):
            est = estimate_mfas(arcs, restarts=64, seed=1)
            assert est.best_size <= est.deterministic_size

    def test_distribution_stats(self):
        # twin_shared 对删边选择敏感：worst > best，std > 0
        est = estimate_mfas(TWIN_SHARED, restarts=64, seed=1)
        assert est.best_size == 1
        assert est.worst_size == 2
        assert est.avg_size > 1.0
        assert est.std > 0.0

    def test_restarts_count(self):
        est = estimate_mfas(SINGLE_CYCLE, restarts=16, seed=0)
        assert len(est.sizes) == 16
        assert est.restarts == 16

    def test_reproducible_with_seed(self):
        a = estimate_mfas(TWIN_DISJOINT, restarts=16, seed=7)
        b = estimate_mfas(TWIN_DISJOINT, restarts=16, seed=7)
        assert a.sizes == b.sizes
        assert a.best == b.best

    def test_dag_zero_everywhere(self):
        est = estimate_mfas(DAG, restarts=16, seed=0)
        assert est.best_size == est.avg_size == est.worst_size == 0
        assert est.std == 0.0

    def test_format_report_fields(self):
        est = estimate_mfas(TWIN_SHARED, restarts=8, seed=1)
        text = est.format_report()
        assert "MFAS ESTIMATE" in text
        assert "最优" in text and "平均" in text and "最差" in text and "标准差" in text


class TestKnowledgeGraph:
    """7.2 知识图谱自动抽取（network/nodes.txt + edges.txt → Arc 集）。"""

    def test_parse_arc_lines_skips_comments_and_blank(self):
        lines = [
            "# 注释行",
            "",
            "A | B | derives_from | 描述",
            "C | D",
            "   ",
        ]
        assert parse_arc_lines(lines) == {("A", "B"), ("C", "D")}

    def test_load_knowledge_edges_real(self):
        edges_path = os.path.join(
            os.path.dirname(__file__), "..", "network", "edges.txt"
        )
        arcs = load_knowledge_edges(edges_path)
        assert len(arcs) >= 33                    # 核心推导链 ≥ 33 条
        n_edges = {
            (u, v) for u, v in arcs
            if u.startswith("N") and v.startswith("N")
        }
        assert len(n_edges) >= 33                 # N 节点间边完整
        assert ("N001", "N002") in n_edges        # 核心推导链首条

    def test_closure_score_from_knowledge_graph(self):
        edges_path = os.path.join(
            os.path.dirname(__file__), "..", "network", "edges.txt"
        )
        arcs = load_knowledge_edges(edges_path)
        s = ClosureScore(arcs=arcs, patch_edges={"（临时补丁）"})
        assert s.report.R == 1
        assert s.report.C > 0.0          # 图谱含矛盾环 → C < 1
        assert 0.0 < s.report.phi       # Φ 恒正（不完美定理）
        assert s.report.format_report().startswith("CLOSURE REPORT")


class TestRunDemo:
    def test_contains_all_systems(self):
        out = run_demo()
        for name in (
            "托勒密地心说",
            "哥白尼日心说",
            "牛顿力学",
            "广义相对论",
            "古典体液说",
            "现代病原微生物学",
        ):
            assert name in out

    def test_contains_phi_values(self):
        out = run_demo()
        assert "0.0425" in out   # 托勒密
        assert "0.9329" in out   # 相对论
        assert "0.435" in out    # 病原学（0.4351…）
        assert "成熟体系" in out


class TestCrossProcessDeterminism:
    """评审点 2.2-1：跨进程确定性——不同 PYTHONHASHSEED 下结果一致。

    旧实现 victim 选择依赖无序 set 遍历（字符串哈希序），
    跨进程不可复现；新实现取字典序 min，与哈希序无关。
    """

    # target=B 有两条候选入边 (A,B)/(C,B)：旧实现可能随哈希序选中 (C,B)
    HASH_SENSITIVE = {("A", "B"), ("B", "C"), ("C", "A"), ("C", "B")}

    def _run_subprocess(self, code: str) -> str:
        src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        outs = set()
        for seed in (0, 1, 7):
            env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONPATH=src)
            r = subprocess.run(
                [sys.executable, "-c", code],
                env=env, capture_output=True, text=True,
            )
            assert r.returncode == 0, r.stderr
            outs.add(r.stdout.strip())
        assert len(outs) == 1, f"跨进程结果不一致：{outs}"
        return outs.pop()

    def test_greedy_mfas_cross_process(self):
        out = self._run_subprocess(
            "from spum_graph.closure import greedy_mfas; "
            "print(sorted(greedy_mfas({('A','B'),('B','C'),('C','A'),('C','B')})))"
        )
        # 环内入度最大顶点 B，候选入边 min → (A,B)；第二环候选 min → (B,C)
        assert out == "[('A', 'B'), ('B', 'C')]"

    def test_cyclic_edges_cross_process(self):
        out = self._run_subprocess(
            "from spum_graph.closure import cyclic_edges; "
            "print(sorted(cyclic_edges({('A','B'),('B','C'),('C','A'),('C','B')})))"
        )
        assert out == "[('A', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'B')]"


class TestCyclicEdgesTarjan:
    """评审点 2.2-3：Tarjan SCC 线性化——与逐边 BFS 参照实现等价。"""

    @staticmethod
    def _bfs_reference(arcs):
        """逐边 BFS 参照：e 在环上 ⟺ 自环（u==v）或删 e 后从 v 可达 u。

        自环 u→u 本身即一个环，须直接计入（旧实现漏判此处）。
        """
        result = set()
        for e in sorted(arcs):
            u, v = e
            if u == v:
                result.add(e)
                continue
            rest = arcs - {e}
            adj = {}
            for a, b in rest:
                adj.setdefault(a, set()).add(b)
            seen, frontier = set(), [v]
            while frontier:
                x = frontier.pop()
                for y in adj.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        frontier.append(y)
            if u in seen:
                result.add(e)
        return result

    def test_self_loop_is_cyclic(self):
        assert cyclic_edges({("A", "A")}) == {("A", "A")}

    def test_matches_bfs_reference_on_random_graphs(self):
        rng = random.Random(2026)
        nodes = [f"n{i}" for i in range(6)]
        for _ in range(40):
            arcs = set()
            for u in nodes:
                for v in nodes:
                    if rng.random() < 0.35:
                        arcs.add((u, v))
            assert cyclic_edges(arcs) == self._bfs_reference(arcs)

    def test_deterministic(self):
        assert cyclic_edges(TWIN_SHARED) == cyclic_edges(TWIN_SHARED)


class TestParameterValidation:
    """评审点 2.2-4/5：参量范围校验 + 空图空洞自洽防护。"""

    def test_c_out_of_range_rejected(self):
        for bad in (1.5, -0.1, 1.0000001):
            with pytest.raises(ValueError):
                PhiReport(C=bad, S=0.5, R=1)

    def test_s_out_of_range_rejected(self):
        for bad in (1.5, -0.1):
            with pytest.raises(ValueError):
                PhiReport(C=0.5, S=bad, R=1)

    def test_boundary_zero_accepted(self):
        # 全环图 S=0、自环图 C=0 是合法边界（闭区间 [0,1]）
        r = PhiReport(C=0.0, S=0.0, R=1)
        assert r.phi == 0.0

    def test_direct_mode_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            ClosureScore(C=5, S=0.6, R=12)

    def test_auto_mode_rejects_empty_arcs(self):
        # 空洞自洽防护（§7.5.1 退化防护）：空弧集 C=S=1 无代表力
        with pytest.raises(ValueError):
            ClosureScore(arcs=set(), patch_edges={("E", "X")})


class TestForecastTransition:
    """评审点 2.2-7：R(t)/R_crit 跃迁预报占位接口（F.4 路线图）。"""

    def test_short_history_no_forecast(self):
        assert forecast_transition([1]) is False

    def test_below_default_r_crit(self):
        assert forecast_transition([10, 12, 14]) is False   # 外推 16 ≤ 20

    def test_above_default_r_crit(self):
        assert forecast_transition([18, 19, 20]) is True    # 外推 21 > 20

    def test_custom_r_crit(self):
        assert forecast_transition([5, 6], r_crit=5) is True   # 外推 7 > 5

    def test_flat_series_no_crossing(self):
        assert forecast_transition([19, 19, 19]) is False   # 外推 19 ≤ 20


class TestParseWarnings:
    """评审点 2.2-8：异常格式警告而非静默跳过。"""

    def test_malformed_lines_warn(self):
        with pytest.warns(UserWarning):
            arcs = parse_arc_lines(["A | B", "bad", "C | | desc", " | X"])
        assert arcs == {("A", "B")}

    def test_wellformed_no_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 任何 warning 即失败
            parse_arc_lines(["# 注释", "", "A | B | derives_from | 描述", "C | D"])


class TestRunDemoAuto:
    """评审点 2.2-6：端到端自动抽取 → 自动计算（模式 A 段）。"""

    EDGES = os.path.join(os.path.dirname(__file__), "..", "network", "edges.txt")

    def test_auto_section_present(self):
        out = run_demo(self.EDGES)
        assert "知识图谱实况" in out
        assert "模式 A" in out and "模式 B" in out
        assert "COG 认知投影论" in out

    def test_auto_cog_subgraph_is_mature(self):
        # COG 子图（两端均为 COG 前缀）为 DAG → C=S=1 → Φ=1.0 成熟体系
        out = run_demo(self.EDGES)
        cog_line = next(ln for ln in out.splitlines() if ln.startswith("  COG"))
        assert "Φ=1.0000" in cog_line

    def test_auto_section_absent_by_default(self):
        out = run_demo()
        assert "知识图谱实况" not in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
