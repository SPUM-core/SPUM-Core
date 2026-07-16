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
from .dangling import DanglingDetector, compute_dangling
from .handshaking import HandshakingVerifier, euler_characteristic

__all__ = [
    "FrameGraph",
    "DanglingDetector",
    "compute_dangling",
    "HandshakingVerifier",
    "euler_characteristic",
]
