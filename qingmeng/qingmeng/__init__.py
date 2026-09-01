"""
青檬引擎 — 基于 SPUM 内化的复杂系统推理引擎
=============================================

四层解耦架构：
  core     公理内核层（理论不可变）—— 12 晶子 · 五步帧 · 11 条共识
  graph    拓扑推理层（帧快照分析）—— 路径/指标/五形相位
  domain   领域适配层（理论可变）—— 拓扑信号 → 领域语义
  api      接口层（CLI/REST/LLM Bridge）

快速开始:
    from qingmeng import QingmengEngine
    eng = QingmengEngine()

    # 演化一帧（五步帧状态机 + 对偶记账）
    snap = eng.evolve()

    # 共识校验（11 条不变量；违规抛 ConsensusViolationError）
    eng.check(raise_on_fail=True)

    # 五形相位与领域解释
    eng.emergence()          # → S=(水,木,土,金,火)
    eng.interpret("physics") # → σ → 温度语义

与既有实现的关系：
  - spum/ 包（SPUM API v4.0）：AI 之家推理 API
  - e:/qingmeng_spum/qingmeng：青檬引擎渐进重构实现（增量路线）
  - 本包：蓝图四层架构蓝本，逻辑同源、自包含、可独立演进
"""

from __future__ import annotations

from typing import Any, Dict

import networkx as nx

from .core import (
    Core,
    Constants,
    StateEvolutionEngine,
    FrameSnapshot,
    ConsensusValidator,
    ConsensusReport,
    ConsensusViolationError,
    CORE_SIZE,
    ICOSAHEDRON_EDGES,
)
from .graph import Topology, EmergenceDetector, TrajectoryStore
from .domain import (
    DomainRegistry,
    DomainAdapter,
    PhysicsAdapter,
    SociologyAdapter,
    InteractionBoundary,
    BoundaryConfig,
)
from .api.tactile import TactileBridge
from .api.llm_bridge import LLMBridge

__version__ = "0.1.0"


class QingmengEngine:
    """青檬引擎统一入口——四层组合，开箱即用。

    SPUM 内化方式：公理 → 可执行原语（core）；共识要求 → 校验器
    （core.consensus）；跨域知识 → 适配器注册表（domain）。
    """

    def __init__(self) -> None:
        self.core = Core()
        self.graph = self.core.build_icosahedron()
        self.state = StateEvolutionEngine(self.core)
        self.topology = Topology(self.core)
        self.emergence_detector = EmergenceDetector()
        self.validator = ConsensusValidator()
        self.domain = DomainRegistry()
        self.domain.register(PhysicsAdapter())
        self.domain.register(SociologyAdapter())
        self.tactile = TactileBridge(self)
        self.reasoner = LLMBridge(self)
        self.trajectory = TrajectoryStore()   # 观测性：轨迹快照 + 版本回滚
        self.boundary = InteractionBoundary()  # 边界效应：领域耦合度量

    # ── 边界效应（领域耦合） ────────────────────────────────────

    def coupling(self, nodes_a, nodes_b) -> float:
        """计算两组节点之间的耦合强度 ∈ [0,1]（只读度量，不改变图）。"""
        return self.boundary.coupling(self.graph, nodes_a, nodes_b)

    # ── 演化 ─────────────────────────────────────────────────────

    def evolve(self) -> FrameSnapshot:
        """推进一帧（五步帧 + 对偶记账），并记录推理轨迹。"""
        before = self.trajectory.snapshot_state(self)
        snap = self.state.evolve()
        self.trajectory.record("evolve", f"frame {self.state.frame - 1}", self, before=before)
        return snap

    def evolve_many(self, frames: int) -> FrameSnapshot:
        """连续演化多帧，返回最后一帧快照（整批记录一条轨迹）。"""
        before = self.trajectory.snapshot_state(self)
        for _ in range(frames):
            self.state.evolve()
        snap = self.state.snapshot()
        assert snap is not None
        self.trajectory.record("evolve_many", f"{frames} frames", self, before=before)
        return snap

    # ── 共识校验 ─────────────────────────────────────────────────

    def check(self, raise_on_fail: bool = False) -> ConsensusReport:
        """对当前图运行 11 条不变量校验。"""
        return self.validator.validate(
            self.graph, ledger=self.state.ledger, raise_on_fail=raise_on_fail
        )

    # ── 分析 ─────────────────────────────────────────────────────

    def summarize(self) -> Dict[str, Any]:
        """当前帧结构摘要（推理轨迹快照）。"""
        return self.topology.summarize(self.graph)

    def emergence(self) -> Dict:
        """当前帧五形相位向量 S = (水,木,土,金,火)。"""
        return self.emergence_detector.detect(self.graph)

    def interpret(self, domain: str, context: Dict[str, Any] | None = None) -> Dict:
        """用领域适配器解释当前帧结构。"""
        return self.domain.interpret(domain, self.summarize(), context)

    # ── 推理 ─────────────────────────────────────────────────────

    def reason(
        self, prompt: str, max_frames: int = 5, domain: str | None = None
    ):
        """L2 帧推理循环——LLM 提议 → 引擎执行 → 共识校验（辅助推理器）。"""
        before = self.trajectory.snapshot_state(self)
        trace = self.reasoner.reason(prompt, max_frames=max_frames, domain=domain)
        self.trajectory.record("reason", prompt, self, before=before)
        return trace

    # ── 触觉桥 ───────────────────────────────────────────────────

    def touch(self, text: str) -> Dict[str, Any]:
        """触觉桥输入——文本接触产生关系密度，并记录推理轨迹。"""
        before = self.trajectory.snapshot_state(self)
        feeling = self.tactile.react_to(text)
        self.trajectory.record("touch", text, self, before=before)
        return feeling

    # ── 描述 ─────────────────────────────────────────────────────

    def describe(self) -> str:
        return (
            f"青檬引擎 v{__version__} | "
            f"帧={self.state.frame} | {self.core.describe()} | "
            f"{self.validator.describe()}"
        )
