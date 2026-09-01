"""
SPUM-图论模块 (L0.5 工具层)
============================

L0 本体论的图论表达工具。与经典 L2 图论隔离——
所有概念必须能回溯到 SPUM 核心节点 N001-N013。

核心组件:
    FrameGraph     — 帧内图结构，实现 5 条公理
    dangling       — 悬挂端检测 (δ 密度计算)
    handshaking    — 帧内握手引理 + 欧拉示性数

隔离原则:
    不使用经典图论的谱性质（拉普拉斯、特征值）
    不预设全局坐标系或度量
    不定义孤立节点（度数为 0 的节点在 SPUM 中不存在）
"""

from .graph import FrameGraph
from .handshaking import HandshakingVerifier, euler_characteristic
from .closure import (
    ClosureScore,
    PhiReport,
    PhiConfig,
    greedy_mfas,
    self_consistency,
    steadiness,
    redundancy,
    cyclic_edges,
    find_directed_cycle,
    run_demo,
)

# dangling 模块依赖 torch——惰性导入，未安装 torch 时包仍可加载
try:
    from .dangling import DanglingDetector, compute_dangling
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch 未安装时触发
    DanglingDetector = None
    compute_dangling = None
    _TORCH_AVAILABLE = False

__all__ = [
    "FrameGraph",
    "HandshakingVerifier",
    "euler_characteristic",
    "ClosureScore",
    "PhiReport",
    "PhiConfig",
    "greedy_mfas",
    "self_consistency",
    "steadiness",
    "redundancy",
    "cyclic_edges",
    "find_directed_cycle",
    "run_demo",
]
