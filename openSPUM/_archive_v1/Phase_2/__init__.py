"""
OpenSPUM Phase 2 — 帧演化引擎

包含:
    - FrameUpdateConfig:    帧演化配置
    - FrameLog:             帧运行日志
    - FrameUpdateEngine:    帧演化引擎（级联消解 + 湮灭-创生对偶 + 不变量校验）

依赖:
    - Phase 1: constants, topological_address, node_registry, relation_pool
"""

from .frame_update_engine import (
    FrameUpdateConfig,
    FrameLog,
    FrameUpdateEngine,
)

__all__ = [
    "FrameUpdateConfig",
    "FrameLog",
    "FrameUpdateEngine",
]
