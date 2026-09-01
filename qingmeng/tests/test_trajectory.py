"""
观测性 · 推理轨迹测试 — TrajectoryStore 记录/查询/版本回滚。
"""

from qingmeng import QingmengEngine


def _fresh_engine():
    return QingmengEngine()


def test_evolve_records_trace():
    """evolve 自动记录轨迹（trace_id + 8 项 axiom_path）。"""
    eng = _fresh_engine()
    eng.evolve()
    assert len(eng.trajectory) >= 1
    rec = eng.trajectory.recent(1)[0]
    assert rec.kind == "evolve"
    assert len(rec.axiom_path) == 8
    assert all(p.endswith(":pass") for p in rec.axiom_path)
    assert rec.consensus["passed"] is True
    assert rec.prev_state is not None


def test_touch_records_trace():
    """touch 自动记录轨迹。"""
    eng = _fresh_engine()
    eng.touch("你好，我想知道更多关于引力的事情")
    rec = eng.trajectory.recent(1)[0]
    assert rec.kind == "touch"
    assert "引力" in rec.input


def test_reason_records_trace():
    """reason 自动记录轨迹（确定性后端，离线）。"""
    eng = _fresh_engine()
    eng.reason("引力驱动行星运动", max_frames=3)
    rec = eng.trajectory.recent(1)[0]
    assert rec.kind == "reason"


def test_rollback_restores_state():
    """版本回滚：恢复到轨迹发生前的图/帧/账本。"""
    eng = _fresh_engine()
    before_nodes = set(eng.graph.nodes())
    before_edges = set(eng.graph.edges())
    before_frame = eng.state.frame

    eng.touch("你好，让我们靠近一些")
    after_touch = len(eng.trajectory)
    assert eng.state.frame == before_frame      # touch 不改帧
    assert set(eng.graph.edges()) != before_edges

    trace_id = eng.trajectory.recent(1)[0].trace_id
    rec = eng.trajectory.rollback(trace_id, eng)

    assert rec is not None
    assert set(eng.graph.nodes()) == before_nodes
    assert set(eng.graph.edges()) == before_edges
    assert eng.state.frame == before_frame
    assert eng.state.ledger == {"total_created": 0, "total_annihilated": 0,
                                "net_drift": 0}


def test_rollback_preserves_trace():
    """回滚不抹除观测面——轨迹记录保留。"""
    eng = _fresh_engine()
    eng.evolve()
    trace_id = eng.trajectory.recent(1)[0].trace_id
    eng.trajectory.rollback(trace_id, eng)
    assert eng.trajectory.get(trace_id) is not None


def test_rollback_unknown_id():
    """回滚不存在的轨迹 → None（不报错）。"""
    eng = _fresh_engine()
    assert eng.trajectory.rollback("nope", eng) is None


def test_recent_limit():
    """recent 按序返回且受上限约束。"""
    eng = _fresh_engine()
    for _ in range(5):
        eng.evolve()
    recs = eng.trajectory.recent(3)
    assert len(recs) == 3
    assert recs[-1].seq == max(r.seq for r in recs)     # 最新在最末


def test_max_records_ring_buffer():
    """环缓冲：超出上限丢弃最旧（观测面防爆）。"""
    from qingmeng.graph import TrajectoryStore
    store = TrajectoryStore(max_records=3)
    for i in range(5):
        store.record("evolve", f"frame {i}", _fresh_engine())
    assert len(store) == 3
    assert store.recent(1)[0].input == "frame 4"
