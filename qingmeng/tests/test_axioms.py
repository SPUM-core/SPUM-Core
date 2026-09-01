"""公理内核测试 — 正二十面体 · 拓扑常数 12 · σ · dv/dt · 级联消解"""

import networkx as nx

from qingmeng.core import Core, CORE_SIZE, ICOSAHEDRON_EDGES


def test_icosahedron_skeleton():
    """12 晶子 · 30 边 · 每顶点度 5 · 无孤立节点（不变量 I/III/V/VI）。"""
    core = Core()
    G = core.build_icosahedron()
    assert G.number_of_nodes() == CORE_SIZE == 12
    assert G.number_of_edges() == len(ICOSAHEDRON_EDGES) == 30
    degs = set(dict(G.degree()).values())
    assert degs == {5}                       # 正二十面体顶点各度 5
    assert not G.is_directed()
    assert all("weight" not in d for _, _, d in G.edges(data=True))  # 二值关系
    assert all(G.degree(n) >= 2 for n in G.nodes())                  # 无悬挂端


def test_topological_constant_12():
    """Σ(6−deg) = 12 —— 欧拉恒等式的强制解（不变量 V）。"""
    core = Core()
    G = core.build_icosahedron()
    report = core.angle_deficit(G)
    assert report["sum_6_minus_deg"] == 12
    assert report["satisfied"] is True


def test_sigma_and_avg_degree():
    """σ = |P|/|ε|；⟨deg⟩ = 2/σ = 2|ε|/|P|。"""
    core = Core()
    G = core.build_icosahedron()
    assert core.sigma(G) == 12 / 30          # 0.4（稀疏 → 高温区）
    assert core.avg_degree(G) == 2 * 30 / 12 == 5.0


def test_dv_dt_capacity_limit():
    """dv/dt ≤ const 是单帧容量上限（离散帧语义，不是连续平滑）。"""
    core = Core()
    # 容量上限内：单帧变化 4 ≤ 4κ ✓
    ok = core.dv_dt_check(degree_change=4, frames=1)
    assert ok["satisfied"] is True
    # 超上限：单帧变化 5 > 4 ✗
    over = core.dv_dt_check(degree_change=5, frames=1)
    assert over["satisfied"] is False
    # 跨帧分摊：5 变化 / 2 帧 ≤ 4 ✓（变化是离散的，按帧数分摊）
    spread = core.dv_dt_check(degree_change=5, frames=2)
    assert spread["satisfied"] is True


def test_cascade_remove():
    """级联消解：删除悬挂节点后，邻居度数可能降至 <2 并连锁消解。"""
    core = Core()
    G = core.build_icosahedron()
    # 在 12 上挂一条链：12-13-14-15
    G.add_node(12)
    G.add_node(13)
    G.add_node(14)
    G.add_node(15)
    G.add_edges_from([(12, 13), (13, 14), (14, 15)])
    # 删除 15 → 14 度 1 → 删除 → 13 度 1 → 删除 → 12 度 1 → 删除
    removed = core.cascade_remove(G, 15)
    assert sorted(removed) == [12, 13, 14, 15]
    # 骨架不受影响
    assert set(range(12)) <= set(G.nodes())
    assert core.angle_deficit(G)["sum_6_minus_deg"] == 12
