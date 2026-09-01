"""触觉桥测试 — 输入信号 → 拓扑触碰（无权重 · dv/dt 约束 · 共识通过）"""

import networkx as nx

from qingmeng import QingmengEngine
from qingmeng.api.tactile import TactileBridge, MAX_EDGES_PER_NODE_PER_TOUCH


def _eng():
    return QingmengEngine()


def test_react_to_creates_binary_edges():
    """触碰创建边，全部无权重（不变量 VI：边是二值关系）。"""
    eng = _eng()
    result = eng.tactile.react_to("你好，我担心明天的事情。")
    assert result["intensity"] >= 0.2
    assert result["new_edges"] >= 0
    assert "feeling_text" in result
    # 所有边无 weight 属性
    for _, _, d in eng.core.graph.edges(data=True):
        assert "weight" not in d, f"发现权重边: {d}"


def test_user_node_anchored_and_connected():
    """用户节点通过连接确认存在（不变量 I），无孤立节点注入。"""
    eng = _eng()
    eng.tactile.react_to("你好啊青檬")
    uid = eng.tactile.user_node_id
    assert uid is not None
    assert uid in eng.core.graph
    assert eng.core.graph.degree(uid) >= 1          # 至少一条连接
    # 无孤立节点
    isolated = [n for n in eng.core.graph.nodes()
                if eng.core.graph.degree(n) == 0]
    assert isolated == []


def test_dv_dt_bound_per_touch():
    """单次触碰中单一节点度数增量 ≤ 4（dv/dt ≤ const 的输入注入投影）。"""
    eng = _eng()
    # 长文本 + 强标点 → 高强度触碰
    text = "我真的很担心明天的事情，希望一切顺利！" * 3
    eng.tactile.react_to(text)
    uid = eng.tactile.user_node_id
    # 用户节点初始 1 条边（锚定），触碰后 ≤ 1 + 4
    assert eng.core.graph.degree(uid) <= 1 + MAX_EDGES_PER_NODE_PER_TOUCH


def test_consensus_passes_after_touch():
    """触碰后共识校验通过（无孤立节点 · 无权边 · 12 晶子完整）。"""
    eng = _eng()
    eng.tactile.react_to("今天天气很好，我们一起散步吧。")
    report = eng.check()
    assert report.passed, f"触碰后共识失败: {[f.detail for f in report.failures]}"


def test_intensity_scales_touch_density():
    """强度与接触密度正相关：更强输入产生更多新边。"""
    eng = _eng()
    weak = eng.tactile.react_to("嗯")
    weak_edges = weak["new_edges"]
    strong = eng.tactile.react_to("我真的很担心明天的事情，希望一切顺利！" * 4)
    strong_edges = strong["new_edges"]
    assert strong_edges >= weak_edges


def test_touch_then_evolve_handles_dangling():
    """触碰产生的悬挂端由后续演化帧修补（五步帧闭环）。

    give_idea 意图：内容节点仅挂 1 条边到用户 → 必然残留悬挂端（不完美）。
    """
    eng = _eng()
    eng.tactile.react_to("告诉你一个秘密")   # intent=give_idea
    dangling_before = len(eng.core.dangling_nodes(eng.core.graph))
    assert dangling_before > 0, "触碰后应残留悬挂端（不完美）"
    # 演化若干帧后悬挂端密度下降（被修补或删除）
    for _ in range(20):
        eng.evolve()
    dangling_after = len(eng.core.dangling_nodes(eng.core.graph))
    assert dangling_after < dangling_before


def test_coactivation_memory_grows():
    """共激活结构记忆随触碰增长，且不受边权重影响。"""
    eng = _eng()
    assert len(eng.tactile.coactivation_history) == 0
    eng.tactile.react_to("苹果 香蕉 葡萄")
    eng.tactile.react_to("苹果 香蕉 葡萄")
    assert len(eng.tactile.coactivation_history) > 0
    top = eng.tactile.get_top_coactivations(top_k=5)
    assert len(top) > 0
