"""
边界效应测试 — InteractionBoundary 耦合强度/衰减/传播。
"""

import networkx as nx
import pytest

from qingmeng import QingmengEngine
from qingmeng.core.axioms import Core, CORE_SIZE
from qingmeng.domain import InteractionBoundary, BoundaryConfig


def _chain_graph():
    """1-2-3-4-5 简单链 + 悬挂分支。"""
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (3, 6)])
    return G


def test_coupling_adjacent_higher_than_far():
    """相邻层耦合高于远距离层（距离项单调）。"""
    b = InteractionBoundary()
    G = _chain_graph()
    c_close = b.coupling(G, [1], [2])
    c_far = b.coupling(G, [1], [5])
    assert c_close > c_far
    assert 0.0 <= c_close <= 1.0


def test_coupling_same_density_bonus():
    """密度相近 → 耦合更高。"""
    b = InteractionBoundary()
    G = _chain_graph()
    # 节点 3 度 3（高密），节点 6 度 1（低密）——与 1/2 相比密度差更大
    c_12 = b.coupling(G, [1], [2])
    c_36 = b.coupling(G, [3], [6])
    # 距离相同（均为直接相邻），密度差越大耦合越低
    assert c_12 >= c_36


def test_coupling_disconnected_is_zero():
    """不连通节点集 → 耦合 0。"""
    b = InteractionBoundary()
    G = _chain_graph()
    G.add_node(99)
    assert b.coupling(G, [1], [99]) == 0.0


def test_coupling_empty_set_is_zero():
    """空节点集 → 耦合 0（不崩溃）。"""
    b = InteractionBoundary()
    assert b.coupling(_chain_graph(), [], [1]) == 0.0


def test_decay_monotonic_and_bounded():
    """衰减随帧数递减且不下溢。"""
    b = InteractionBoundary()
    s0 = 0.9
    s1 = b.decay(s0, 1)
    s10 = b.decay(s0, 10)
    assert s1 < s0 and s10 < s1
    assert s10 > 0.0
    assert b.decay(s0, 0) == pytest.approx(s0)


def test_propagate_distance_decay():
    """传播强度随拓扑距离衰减。"""
    b = InteractionBoundary()
    G = _chain_graph()
    d1 = b.propagate(G, 1, 2)      # 直接相邻
    d2 = b.propagate(G, 1, 4)      # 距离 3
    assert d1 > d2
    assert d1 == pytest.approx(1 / 2)      # 1/(1+1)
    assert d2 == pytest.approx(1 / 4)      # 1/(1+3)


def test_propagate_unreachable_none():
    """不可达 → None（非 0——区分"无路径"与"零强度"）。"""
    b = InteractionBoundary()
    G = _chain_graph()
    G.add_node(99)
    assert b.propagate(G, 1, 99) is None


def test_config_tunable():
    """领域层参数可调（γ 越大，远距离衰减越快）。"""
    G = _chain_graph()
    b_weak = InteractionBoundary(BoundaryConfig(distance_gamma=0.1))
    b_strong = InteractionBoundary(BoundaryConfig(distance_gamma=5.0))
    c_weak = b_weak.coupling(G, [1], [5])
    c_strong = b_strong.coupling(G, [1], [5])
    assert c_weak > c_strong


def test_engine_coupling_mounted():
    """引擎已挂载边界效应（正二十面体上直接相邻耦合最高）。"""
    eng = QingmengEngine()
    G = eng.core.graph
    # 找正二十面体上任意一条边（直接相邻）
    u, v = next(iter(G.edges()))
    c_adj = eng.coupling([u], [v])
    # 找距离最远的一对（直径）——取非相邻对
    far = next(((a, b) for a in range(CORE_SIZE) for b in range(CORE_SIZE)
                if a < b and not G.has_edge(a, b)), None)
    if far is not None:
        a, b = far
        assert c_adj > eng.coupling([a], [b])
