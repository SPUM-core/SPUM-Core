"""
SPUM 帧引擎 — 演化帧 + 推理帧统一接口
========================================

双层帧架构:
    1. 演化帧 (EvolutionFrame): 模拟关系网络的宇宙演化
       - 五步序列: 创生(V⁺) → 连接 → 变化体积 → 判断悬挂 → 删除(V⁻)
       - 操作对象: ⟨P, ε⟩ 关系网络

    2. 推理帧 (ReasoningFrame): 记录 AI 的推理过程
       - 五步序列: 建立关系 → 更新连接 → 评估 → 检查悬挂 → 修剪
       - 操作对象: 概念/节点间的推导关系

两者共享"五步序列、悬挂端检测、帧闭合"的结构，
但操作的对象不同:
    - 演化帧是本体模拟层 (模拟宇宙)
    - 推理帧是元认知层 (记录推理)

用法:
    from spum.frame import FrameEngine

    engine = FrameEngine()
    engine.new_evolution_frame()
    engine.v_plus("A", "B")    # 创建关系
    engine.v_minus("C", "D")   # 删除关系
    engine.commit()            # 完成一帧
    print(engine.last_frame)
"""

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple, Any
from enum import Enum, auto
from .axioms import Axioms, Axiom3


# ══════════════════════════════════════════════════════════════════════
# 帧类型
# ══════════════════════════════════════════════════════════════════════

class FrameType(Enum):
    EVOLUTION = "演化帧"   # 宇宙关系网络模拟
    REASONING = "推理帧"   # AI 推理过程记录


# ══════════════════════════════════════════════════════════════════════
# 帧状态
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FrameState:
    """单帧的完整状态。

    兼容演化帧和推理帧两种模式。
    """
    frame_id: int
    frame_type: FrameType
    # 图状态
    neighbors: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    edge_count: int = 0
    # V⁺/V⁻ 操作记录
    v_plus_events: List[Tuple[str, str]] = field(default_factory=list)
    v_minus_events: List[Tuple[str, str]] = field(default_factory=list)
    # 五步序列进度
    step_1_creation: bool = False
    step_2_connection: bool = False
    step_3_volume: bool = False
    step_4_judge: bool = False
    step_5_deletion: bool = False
    # 悬挂端状态
    dangling_count: int = 0
    dangling_nodes: List[str] = field(default_factory=list)
    is_closed: bool = False
    # 指标
    sigma: float = 0.0
    delta: float = 0.0
    # 备注
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_steps_done(self) -> bool:
        return all([self.step_1_creation, self.step_2_connection,
                    self.step_3_volume, self.step_4_judge, self.step_5_deletion])

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "type": self.frame_type.value,
            "nodes": len(self.neighbors),
            "edges": self.edge_count,
            "sigma": self.sigma,
            "delta": self.delta,
            "dangling": self.dangling_count,
            "is_closed": self.is_closed,
            "all_steps_done": self.all_steps_done,
            "v_plus": len(self.v_plus_events),
            "v_minus": len(self.v_minus_events),
        }


# ══════════════════════════════════════════════════════════════════════
# 帧引擎
# ══════════════════════════════════════════════════════════════════════

class FrameEngine:
    """帧引擎 — 演化帧 + 推理帧统一管理。

    管理帧序列: 帧号递增、悬挂端追踪、栈溢出检测、历史回退。

    用法:
        engine = FrameEngine()

        # 演化帧
        engine.new_evolution_frame()
        engine.v_plus("A", "B")
        engine.commit()
        print(engine.summary())

        # 推理帧
        engine.new_reasoning_frame(notes=["推导暗物质替代方案"])
        engine.add_relation("N005", "N006", "derives_from", "闭合推导")
        engine.set_dangling(2)
        engine.commit()
    """

    def __init__(self):
        self._axioms = Axioms()
        self._frame_counter: int = 0
        self._frames: List[FrameState] = []
        self._current: Optional[FrameState] = None
        self._unclosed_streak: int = 0
        self._rollback_points: List[int] = []
        self._max_frames: int = 1000

    # ── 属性 ─────────────────────────────────────────────────────

    @property
    def current(self) -> Optional[FrameState]:
        return self._current

    @property
    def frame_id(self) -> int:
        return self._frame_counter

    @property
    def history(self) -> List[FrameState]:
        return list(self._frames)

    @property
    def unclosed_streak(self) -> int:
        return self._unclosed_streak

    @property
    def axioms(self) -> Axioms:
        return self._axioms

    @property
    def total_edges(self) -> int:
        """当前所有帧的总边数（跨帧累积，用于守恒校验）。"""
        if self._current:
            return self._current.edge_count
        return self._frames[-1].edge_count if self._frames else 0

    # ── 帧生命周期 ──────────────────────────────────────────────

    def new_evolution_frame(self, notes: Optional[List[str]] = None) -> FrameState:
        """开始新演化帧。"""
        self._frame_counter += 1
        self._current = FrameState(
            frame_id=self._frame_counter,
            frame_type=FrameType.EVOLUTION,
        )
        if notes:
            self._current.notes = notes
        return self._current

    def new_reasoning_frame(self, notes: Optional[List[str]] = None) -> FrameState:
        """开始新推理帧。"""
        self._frame_counter += 1
        self._current = FrameState(
            frame_id=self._frame_counter,
            frame_type=FrameType.REASONING,
        )
        if notes:
            self._current.notes = notes
        return self._current

    def commit(self) -> FrameState:
        """结束当前帧。执行闭合检查和栈溢出检测。

        Returns:
            当前帧状态

        Raises:
            RuntimeError: 没有活跃帧
        """
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 new_*_frame()。")

        # 计算指标
        frame = self._current
        n_nodes = len(frame.neighbors)
        frame.sigma = self._axioms.sigma(n_nodes, frame.edge_count) if n_nodes > 0 else 0.0
        frame.dangling_count = sum(1 for n in frame.neighbors
                                    if len(frame.neighbors[n]) < 2)
        frame.dangling_nodes = sorted(
            n for n in frame.neighbors if len(frame.neighbors[n]) < 2
        )
        frame.delta = frame.dangling_count / max(n_nodes, 1)
        frame.is_closed = frame.dangling_count == 0

        # 更新未闭合连续计数
        if frame.is_closed:
            self._unclosed_streak = 0
            self._rollback_points.append(self._frame_counter)
        else:
            self._unclosed_streak += 1

        self._frames.append(frame)
        result = frame
        self._current = None
        return result

    def is_stack_overflow(self) -> bool:
        """检查是否触发栈溢出（连续 3 帧未闭合）。"""
        return self._unclosed_streak >= 3

    def get_rollback_frame(self) -> Optional[int]:
        """获取最近的可回退闭合帧号。"""
        return self._rollback_points[-1] if self._rollback_points else None

    def rollback(self) -> Optional[FrameState]:
        """回退到最近的闭合帧。

        Returns:
            回退到的帧状态，或 None
        """
        rollback_id = self.get_rollback_frame()
        if rollback_id is None:
            return None

        idx = next(
            (i for i, f in enumerate(self._frames) if f.frame_id == rollback_id),
            None,
        )
        if idx is None:
            return None

        self._frames = self._frames[:idx + 1]
        self._frame_counter = rollback_id
        self._unclosed_streak = 0
        self._current = None
        return self._frames[-1]

    # ── V⁺/V⁻ 操作 ─────────────────────────────────────────────

    def v_plus(self, u: str, v: str) -> dict:
        """创生事件 V⁺ — 创建一条新边。

        Args:
            u: 节点 ID
            v: 节点 ID

        Returns:
            {"event": "V⁺", "u": u, "v": v,
             "new_nodes": [新节点列表]}
        """
        if self._current is None:
            self.new_evolution_frame()

        new_nodes = []
        for node in (u, v):
            if node not in self._current.neighbors:
                new_nodes.append(node)

        ec = [self._current.edge_count]
        self._axioms.axiom1.add_edge(u, v, self._current.neighbors, ec)
        self._current.edge_count = ec[0]
        self._current.v_plus_events.append((u, v))

        return {
            "event": "V⁺",
            "u": u, "v": v,
            "new_nodes": new_nodes,
        }

    def v_minus(self, u: str, v: str) -> dict:
        """湮灭事件 V⁻ — 删除一条边。

        Args:
            u: 节点 ID
            v: 节点 ID

        Returns:
            {"event": "V⁻", "u": u, "v": v,
             "removed_nodes": [因度数归零被删除的节点]}
        """
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 new_*_frame()。")

        removed_nodes = []
        old_deg_u = len(self._current.neighbors.get(u, set()))
        old_deg_v = len(self._current.neighbors.get(v, set()))

        ec = [self._current.edge_count]
        self._axioms.axiom1.remove_edge(u, v, self._current.neighbors, ec)
        self._current.edge_count = ec[0]
        self._current.v_minus_events.append((u, v))

        # 检查悬挂端级联
        for node in (u, v):
            if node not in self._current.neighbors:
                removed_nodes.append(node)

        return {
            "event": "V⁻",
            "u": u, "v": v,
            "removed_nodes": removed_nodes,
        }

    # ── 五步序列（推理帧专用） ──────────────────────────────────

    def step(self, step_number: int, description: str = "") -> None:
        """标记五步帧序列中的一步已完成（推理帧用）。

        Args:
            step_number: 1-5
                1=创生(V⁺), 2=连接, 3=变化体积, 4=判断悬挂, 5=删除(V⁻)
            description: 步骤描述
        """
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 new_*_frame()。")

        step_map = {
            1: "step_1_creation",
            2: "step_2_connection",
            3: "step_3_volume",
            4: "step_4_judge",
            5: "step_5_deletion",
        }
        if step_number not in step_map:
            raise ValueError(f"step_number 必须是 1-5，收到 {step_number}")

        setattr(self._current, step_map[step_number], True)
        if description:
            self._current.notes.append(f"[步骤{step_number}] {description}")

    # ── 帧内容设置（推理帧） ────────────────────────────────────

    def add_relation(self, source: str, target: str,
                     edge_type: str, note: str = "") -> None:
        """添加一条推理关系（推理帧用）。"""
        if self._current is None:
            raise RuntimeError("没有活跃帧。")
        if self._current.frame_type != FrameType.REASONING:
            raise RuntimeError("add_relation 仅用于推理帧。")

        # 在邻居中记录
        rel_key = f"{source}--[{edge_type}]-->{target}"
        self._axioms.axiom1.add_edge(
            source, rel_key,
            self._current.neighbors, [self._current.edge_count]
        )
        self._current.notes.append(f"关系: {rel_key}" + (f" {note}" if note else ""))

    def set_dangling(self, count: int, items: Optional[List[str]] = None) -> None:
        """设置悬挂端数量和列表（推理帧用）。"""
        if self._current is None:
            raise RuntimeError("没有活跃帧。")
        self._current.dangling_count = count
        self._current.dangling_nodes = items or []

    def add_note(self, note: str) -> None:
        """添加帧附注。"""
        if self._current is None:
            raise RuntimeError("没有活跃帧。")
        self._current.notes.append(note)

    # ── 信息方法 ────────────────────────────────────────────────

    def summary(self) -> str:
        """帧序列汇总。"""
        total = len(self._frames)
        closed = sum(1 for f in self._frames if f.is_closed)
        evo = sum(1 for f in self._frames if f.frame_type == FrameType.EVOLUTION)
        rea = sum(1 for f in self._frames if f.frame_type == FrameType.REASONING)
        return (
            f"帧序列: {total} 帧 (演化{evo}/推理{rea})\n"
            f"闭合: {closed}/{total}\n"
            f"连续未闭合: {self._unclosed_streak}\n"
            f"栈溢出: {'是' if self.is_stack_overflow() else '否'}\n"
            f"回退点: {self._rollback_points}"
        )

    def last_frame(self) -> Optional[FrameState]:
        """最近一帧。"""
        return self._frames[-1] if self._frames else self._current

    def format_frame(self, frame: Optional[FrameState] = None) -> str:
        """格式化帧输出。"""
        f = frame or self.last_frame()
        if f is None:
            return "[无帧]"

        type_str = f.frame_type.value
        lines = [
            f"[帧{f.frame_id}] ({type_str})",
            f"  节点: {len(f.neighbors)}  边: {f.edge_count}",
            f"  σ={f.sigma:.4f}  δ={f.delta:.4f}  "
            f"悬挂={f.dangling_count}",
            f"  闭合: {'是' if f.is_closed else '否'}",
        ]
        steps = []
        for i, name in enumerate(Axiom3.STEP_NAMES, 1):
            done = getattr(f, f"step_{i}_creation" if i == 1 else
                          getattr(f, f"step_{i}_connection" if i == 2 else
                          getattr(f, f"step_{i}_volume" if i == 3 else
                          getattr(f, f"step_{i}_judge" if i == 4 else
                          getattr(f, f"step_{i}_deletion", False)))), False)
            # Simplify: just list step names
            _ = done  # unused, just for clarity
        # Actually just show V⁺/V⁻ counts
        lines.append(f"  V⁺={len(f.v_plus_events)}  V⁻={len(f.v_minus_events)}")
        if f.notes:
            lines.append(f"  备注: {'; '.join(f.notes[:3])}")
        return "\n".join(lines)
