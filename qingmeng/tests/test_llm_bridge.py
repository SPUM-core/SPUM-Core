"""LLM 推理桥测试 — L2 帧推理循环（密闭 · 无网络 · 确定性后端）"""

import networkx as nx

from qingmeng import QingmengEngine
from qingmeng.api.llm_bridge import (
    LLMBackend,
    LLMBridge,
    DeterministicBackend,
    HttpLLMBackend,
    L2FrameProtocol,
    MAX_LLM_EDGES_PER_FRAME,
    STACK_OVERFLOW_THRESHOLD,
)


def _eng():
    return QingmengEngine()


# ════════════════════════════════════════════════════════════════════
# L2 帧推理循环
# ════════════════════════════════════════════════════════════════════

def test_reason_produces_closed_trace():
    """默认后端跑通 L2 循环：帧序列 + conclude 收束。"""
    eng = _eng()
    trace = eng.reason("水为什么往低处流")
    assert trace.total_frames >= 3
    assert trace.concluded, "确定性后端应在第 3 帧 conclude"
    assert trace.frames[-1].frame_type == "conclude"
    # 每帧五步序列完整
    for f in trace.frames:
        assert f.all_steps_done()


def test_claim_creates_binary_edges():
    """V⁺ 创建二值边（不变量 VI：无 weight 属性）。"""
    eng = _eng()
    eng.reason("引力导致星体运动")
    for _, _, d in eng.core.graph.edges(data=True):
        assert "weight" not in d, f"发现权重边: {d}"
    # 概念节点已实体化（创建即连接，无孤立注入）
    isolated = [n for n in eng.core.graph.nodes()
                if eng.core.graph.degree(n) == 0]
    assert isolated == []


def test_consensus_passes_after_reason():
    """推理后共识校验通过（12 晶子完整 · 无权边 · 无孤立节点）。"""
    eng = _eng()
    eng.reason("经济周期驱动市场波动")
    report = eng.check()
    assert report.passed, f"推理后共识失败: {[f.detail for f in report.failures]}"


def test_dv_dt_limit_per_frame():
    """单帧 LLM 边操作 ≤ MAX_LLM_EDGES_PER_FRAME（dv/dt ≤ const 投影）。"""
    eng = _eng()
    trace = eng.reason("经济周期驱动市场波动，货币政策需要调节")
    for f in trace.frames:
        assert f.edges_added + f.edges_removed <= MAX_LLM_EDGES_PER_FRAME


def test_concept_registry_reuses_nodes():
    """同一概念跨调用复用同一节点（hash 稳定注册表）。"""
    eng = _eng()
    eng.reason("引力驱动行星运动")
    nid_first = eng.reasoner.concept_registry["引力"]
    eng.reason("引力的本质是什么")
    nid_second = eng.reasoner.concept_registry["引力"]
    assert nid_first == nid_second
    assert nid_first in eng.core.graph


def test_reason_then_evolve_handles_dangling():
    """推理产生的悬挂端由后续演化帧修补（五步帧闭环 + 不完美）。"""
    eng = _eng()
    eng.reason("需求依赖供给")
    dangling_before = len(eng.core.dangling_nodes(eng.core.graph))
    for _ in range(20):
        eng.evolve()
    dangling_after = len(eng.core.dangling_nodes(eng.core.graph))
    assert dangling_after <= dangling_before


# ════════════════════════════════════════════════════════════════════
# 栈溢出回退
# ════════════════════════════════════════════════════════════════════

class _CloseThenDangleBackend(LLMBackend):
    """帧 1 闭合（无操作），此后永远不闭合 → 触发栈溢出回退。"""

    def generate(self, context):
        progress = context.get("trace_frame_count", 0)
        if progress == 0:
            return {
                "frame_type": "conclude", "domain": "general",
                "claim": "开局闭合", "source": "", "target": "",
                "relation": "", "operations": [], "confidence": 1.0,
                "concluded": False,
            }
        return {
            "frame_type": "claim", "domain": "general",
            "claim": "永未决主张", "source": "甲", "target": "乙",
            "relation": "drives", "operations": ["V+"],
            "confidence": 0.5, "concluded": False,
        }


def test_stack_overflow_triggers_rollback():
    """连续 N 帧未闭合 → 栈溢出 → 回退到最近闭合帧（截断轨迹，不撤销图）。"""
    eng = _eng()
    bridge = LLMBridge(eng, backend=_CloseThenDangleBackend())
    trace = bridge.reason("测试回退", max_frames=10)
    assert trace.stack_overflow
    assert trace.rolled_back
    # 轨迹被截断到最近闭合帧（帧 1）
    assert len(trace.frames) == 1
    assert trace.frames[0].is_closed


def test_protocol_streak_matches_threshold():
    """未闭合连续计数与阈值一致（协议层行为）。"""
    proto = L2FrameProtocol()

    class _F:
        def __init__(self, fid, closed):
            self.frame_id = fid
            self.is_closed = closed

    proto.add(_F(1, True))    # 闭合 → streak=0
    proto.add(_F(2, False))   # streak=1
    assert not proto.is_stack_overflow()
    proto.add(_F(3, False))   # streak=2
    proto.add(_F(4, False))   # streak=3 → 溢出
    assert proto.is_stack_overflow()


# ════════════════════════════════════════════════════════════════════
# V⁻ 与孤立保护
# ════════════════════════════════════════════════════════════════════

class _VMinusBackend(LLMBackend):
    """先建边，随后尝试删除会孤立端点的边（应被拒绝）。"""

    def __init__(self):
        self.stage = 0

    def generate(self, context):
        progress = context.get("trace_frame_count", 0)
        if progress == 0:
            return {
                "frame_type": "claim", "domain": "general",
                "claim": "建边", "source": "甲", "target": "乙",
                "relation": "drives", "operations": ["V+"],
                "confidence": 0.5, "concluded": False,
            }
        if progress == 1:
            return {
                "frame_type": "claim", "domain": "general",
                "claim": "断边", "source": "乙", "target": "#",
                "relation": "", "operations": ["V-"],
                "confidence": 0.5, "concluded": False,
            }
        return {
            "frame_type": "conclude", "domain": "general",
            "claim": "收束", "source": "", "target": "",
            "relation": "", "operations": [], "confidence": 0.5,
            "concluded": True,
        }


def test_v_minus_respects_isolation_guard():
    """V⁻ 不得使端点孤立：边不存在/会孤立 → 拒绝并记录，图仍通过共识。"""
    eng = _eng()
    bridge = LLMBridge(eng, backend=_VMinusBackend())
    trace = bridge.reason("测试删边", max_frames=5)
    # 第一次 V⁻（乙→# 不存在）被拒绝；图无孤立节点
    assert all(f.edges_removed == 0 for f in trace.frames)
    report = eng.check()
    assert report.passed


# ════════════════════════════════════════════════════════════════════
# 后端可插拔性
# ════════════════════════════════════════════════════════════════════

class _FakeBackend(LLMBackend):
    """可注入的测试后端——验证桥接层把后端输出翻译为图操作。"""

    def generate(self, context):
        return {
            "frame_type": "claim", "domain": "sociology",
            "claim": "群体依赖制度", "source": "群体", "target": "制度",
            "relation": "requires", "operations": ["V+"],
            "confidence": 0.6, "concluded": True,
        }


def test_custom_backend_injectable():
    """自定义后端可注入；V⁺ 执行、关系保留、conclude 收束。"""
    eng = _eng()
    bridge = LLMBridge(eng, backend=_FakeBackend())
    trace = bridge.reason("测试注入")
    assert len(trace.frames) == 1
    f = trace.frames[0]
    assert f.relation == "requires"
    assert f.consensus_passed
    assert f.concluded
    # 群体—制度 已在图中连通
    a = bridge.concept_registry["群体"]
    b = bridge.concept_registry["制度"]
    assert eng.core.graph.has_edge(a, b)


def test_normalize_spec_clamps_fields():
    """非法帧类型/关系被规整；置信度钳制到 [0.2, 1.0]。"""
    eng = _eng()
    bridge = LLMBridge(eng, backend=DeterministicBackend())
    spec = bridge._normalize_spec({
        "frame_type": "幻觉类型", "relation": "不存在的边",
        "operations": ["V+", "delete_all"], "confidence": 99.0,
        "source": "甲", "target": "乙",
    })
    assert spec["frame_type"] == "claim"
    assert spec["relation"] == ""
    assert spec["operations"] == ["V+"]
    assert spec["confidence"] == 1.0


def test_http_backend_extract_json_and_fallback():
    """HTTP 后端：合法 JSON 提取；无 Key/非法响应回退确定性后端。"""
    # 1. 平衡 JSON 提取
    content = '好的，这是推理帧：{"frame_type":"claim","source":"甲","target":"乙","operations":["V+"],"concluded":false} 完毕'
    spec = HttpLLMBackend._extract_json(content)
    assert spec is not None
    assert spec["source"] == "甲"
    # 2. 无 Key → 回退
    eng = _eng()
    http = HttpLLMBackend(api_key="", fallback=DeterministicBackend())
    spec2 = http.generate({"prompt": "水为什么往低处流", "trace_frame_count": 0,
                           "domain": "general", "evidence": {"connected": set(),
                                                             "present": set()}})
    assert "frame_type" in spec2  # 确定性后端产物
    # 3. 非法 JSON → 回退
    assert HttpLLMBackend._extract_json("没有花括号") is None
    assert HttpLLMBackend._extract_json('{"a": 1, 坏json}') is None


def test_reason_and_touch_coexist():
    """推理桥与触觉桥共用同一张图，共识保持通过。"""
    eng = _eng()
    eng.reason("权力驱动制度变迁")
    eng.tactile.react_to("我有些担心，想靠近一点")
    report = eng.check()
    assert report.passed
