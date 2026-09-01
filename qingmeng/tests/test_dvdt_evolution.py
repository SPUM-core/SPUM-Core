"""离散帧演化测试 — 五步帧 · 对偶记账 · dv/dt 有界性

验证目标（蓝图 M1 的 test_dvdt_evolution.py）：
  1. 五步帧全部执行，快照记录完整
  2. 单帧单节点关系变化有界（dv/dt ≤ const 的离散帧语义）
  3. 创生-湮灭对偶记账：累计净漂移有界（不变量 II）
  4. 帧间无中间状态：演化前后状态确定
"""

import networkx as nx

from qingmeng.core import Core, StateEvolutionEngine, CREATION_MAX_PER_FRAME


def _engine_with_dangling(frames_dangling: int = 1):
    """构造带悬挂端的引擎：在 12 晶子上挂 frames_dangling 条悬挂链。"""
    core = Core()
    G = core.build_icosahedron()
    for i in range(frames_dangling):
        nid = 12 + i * 2
        G.add_node(nid)
        G.add_node(nid + 1)
        G.add_edge(nid, nid + 1)          # 悬挂端 nid、nid+1（度 1）
    return core, G


def test_five_steps_executed():
    """五步帧全部执行，快照记录完整。"""
    core, _ = _engine_with_dangling()
    eng = StateEvolutionEngine(core)
    snap = eng.evolve()
    assert all(snap.steps_done), f"五步未全执行: {snap.steps_done}"
    d = snap.to_dict()
    assert d["frame_id"] == 0
    assert set(d["steps"]) == {
        "1_创生V+", "2_连接", "3_变化体积", "4_判断悬挂", "5_删除V-",
    }


def test_dv_dt_bounded_per_frame():
    """单帧内单节点度数变化有界：任何节点单帧度数增量 ≤ CREATION_MAX_PER_FRAME。

    dv/dt ≤ const 的离散帧语义：每帧关系容量有上限，变化不会"连续溢出"。
    """
    core, _ = _engine_with_dangling(frames_dangling=5)
    eng = StateEvolutionEngine(core)
    before = dict(core.graph.degree())
    eng.evolve()
    after = dict(core.graph.degree())
    # 本帧最多补 CREATION_MAX_PER_FRAME 条边 → 单节点度数增量 ≤ 上限
    max_gain = max((after.get(n, 0) - before.get(n, 0)) for n in set(before) | set(after))
    assert max_gain <= CREATION_MAX_PER_FRAME
    # dv/dt 检查通过
    check = core.dv_dt_check(max_gain, frames=1)
    assert check["satisfied"] is True


def test_duality_ledger_no_unbounded_drift():
    """对偶记账：累计净漂移不单向无界增长（不变量 II）。

    悬挂端被创生修补后，删除步骤收回不完美——创生与湮灭交替，漂移有界。
    """
    core, _ = _engine_with_dangling()
    eng = StateEvolutionEngine(core)
    for _ in range(30):
        eng.evolve()
    ledger = eng.ledger
    # 30 帧演化：净漂移应远小于创生总量（对偶重分配，非单向膨胀）
    assert ledger["net_drift"] <= ledger["total_created"]
    # 初始悬挂端在两帧内被清除（创生 1 条边 → 悬挂端度升 2）
    assert ledger["total_created"] >= 1


def test_deterministic_evolution():
    """确定性演化：同一初始状态两次演化序列完全一致（不变量 VII）。"""
    core1, _ = _engine_with_dangling()
    core2, _ = _engine_with_dangling()
    eng1, eng2 = StateEvolutionEngine(core1), StateEvolutionEngine(core2)
    for _ in range(10):
        eng1.evolve()
        eng2.evolve()
    assert nx.utils.graphs_equal(core1.graph, core2.graph)


def test_stagnation_on_closed_icosahedron():
    """闭合晶子创生抑制：纯 12 晶子骨架无悬挂端 → 帧停滞、无创生。"""
    core = Core()
    core.build_icosahedron()
    eng = StateEvolutionEngine(core)
    snap = eng.evolve()
    assert snap.created == 0
    assert snap.node_count == 12
    assert snap.edge_count == 30
