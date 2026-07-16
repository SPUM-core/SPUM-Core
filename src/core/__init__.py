"""
SPUM 核心模块 — 帧协议自动化 + 评审回滚闭环

组件:
    frame.py — FrameProtocol 帧协议执行器
        FrameState      — 单帧状态 (节点、边、悬挂端、闭合法、层级分布)
        FrameProtocol   — 帧序列管理 (历史、回退、溢出检测)
        FrameLogger     — [帧N] 格式输出
    review.py — 评审回滚闭环
        ReviewResult    — 四维评审结果 (A/B/C/D/LE)
        ReviewBridge    — spum-review → FrameProtocol.rollback() 桥接器
        ReviewVerdict   — 评审判定 (PASS/WARN/ROLLBACK)
        Violation       — 违规记录
        Severity        — 违规严重度
"""

from .frame import FrameState, FrameProtocol, FrameLogger
from .review import (
    ReviewResult, ReviewBridge, ReviewVerdict, Violation, Severity, ReviewAction,
)

__all__ = [
    "FrameState", "FrameProtocol", "FrameLogger",
    "ReviewResult", "ReviewBridge", "ReviewVerdict",
    "Violation", "Severity", "ReviewAction",
]
