"""共识校验器测试 — 11 条不变量拦截违规推理"""

import pytest
import networkx as nx

from qingmeng.core import ConsensusValidator, ConsensusViolationError


def _base_graph() -> nx.Graph:
    """合格基线：12 晶子正二十面体（共识通过）。"""
    from qingmeng.core import Core
    return Core().build_icosahedron()


def test_icosahedron_passes():
    """纯闭合骨架通过全部可检验不变量。"""
    G = _base_graph()
    ledger = {"total_created": 0, "total_annihilated": 0}
    report = ConsensusValidator().validate(G, ledger=ledger)
    assert report.passed is True
    assert len(report.failures) == 0


def test_self_loop_violates_I():
    """自环 = 独立实体 → 违反不变量 I。"""
    G = _base_graph()
    G.add_edge(0, 0)  # 自环
    report = ConsensusValidator().validate(G)
    assert report.passed is False
    assert any(f.invariant == "I" for f in report.failures)


def test_isolated_node_violates_III():
    """孤立节点（deg=0）→ 违反不变量 III（度 0 即孤立，逻辑不自洽）。"""
    G = _base_graph()
    G.add_node(99)  # 孤立节点
    report = ConsensusValidator().validate(G)
    assert report.passed is False
    assert any(f.invariant == "III" for f in report.failures)


def test_weighted_edge_violates_VI():
    """权重边 → 违反不变量 VI（边是二值关系）。"""
    G = _base_graph()
    G.add_edge(12, 13, weight=0.5)
    report = ConsensusValidator().validate(G)
    assert report.passed is False
    assert any(f.invariant == "VI" for f in report.failures)


def test_missing_core_violates_V():
    """核心晶子缺失 → 违反不变量 V（CORE_SIZE = 12）。"""
    G = _base_graph()
    G.remove_node(0)
    report = ConsensusValidator().validate(G)
    assert report.passed is False
    assert any(f.invariant == "V" for f in report.failures)


def test_duality_imbalance_violates_II():
    """净漂移单向累积超量 → 违反不变量 II（创生-湮灭对偶失衡）。"""
    G = _base_graph()
    bad_ledger = {"total_created": 1000, "total_annihilated": 0}
    report = ConsensusValidator().validate(G, ledger=bad_ledger)
    assert report.passed is False
    assert any(f.invariant == "II" for f in report.failures)


def test_raise_on_fail_throws():
    """raise_on_fail=True 时违规抛 ConsensusViolationError 并携带明细。"""
    G = _base_graph()
    G.add_node(99)  # 孤立节点
    with pytest.raises(ConsensusViolationError) as exc:
        ConsensusValidator().validate(G, raise_on_fail=True)
    assert any(f.invariant == "III" for f in exc.value.report.failures)


def test_epistemic_default_pass():
    """认知标注（VIII~XI）为元层面约定，默认通过。"""
    G = _base_graph()
    report = ConsensusValidator().validate(G)
    assert any(c["invariant"] == "VIII-XI" and c["passed"] for c in report.checks)
