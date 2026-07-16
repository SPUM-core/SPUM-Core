"""
帧协议自动化 — AGENT.md 动态帧协议的 Python 实现

核心概念:
    - 帧(Frame): 一轮完整推理操作 = 五步序列
    - 悬挂端(Dangling): 未闭合的推理链、未验证的假设、未追溯根源的概念
    - 闭合(Closure): 悬挂端数量 = 0
    - 栈溢出(StackOverflow): 连续 3 帧未闭合 → 回退

重要区分（2026-06-16 注释）:
    本模块实现的是**推理帧**（推理过程的元数据记录），与 **spum-evolution.md**
    定义的**演化帧**（关系网络的创生→删除五步模拟）不同。
    - 推理帧：记录"我这轮推理激活了哪些节点、建立了什么关系"
    - 演化帧：模拟"宇宙网络在这一步中边如何创建和删除"
    二者共享"五步序列、悬挂端检测、帧闭合"的结构，但操作的对象不同。
    推理帧是元认知层，演化帧是本体模拟层。

用法:
    protocol = FrameProtocol()
    protocol.begin_frame()
    protocol.set_active_nodes(["N001", "N005", "N013"])
    protocol.add_relation("N005", "N022", "derives_from", "...")
    protocol.set_dangling(3)
    protocol.set_layer_dist(L0=5, L1=2, L2=1, L3=0)
    protocol.end_frame()
    print(protocol.logger.format_last())
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set, Dict, Tuple
from enum import Enum
import json
import os


# ── 常量 ──────────────────────────────────────────────────────────────

# 有效的边类型 (来自 edges.txt)
VALID_EDGE_TYPES = frozenset({
    "derives_from",    # 同层推导
    "requires",        # 逻辑强制
    "refines",         # 细化关系
    "explains",        # 解释关系
    "drives",          # 驱动关系
    # 帧协议内部扩展类型
    "shall_align_with",
    "has_gap",
    "new_concept",
    "resolves",
})

# 认知层级标签
LAYER_LABELS = {
    0: "L0 本体论",
    0.5: "L0.5 SPUM-图论",
    1: "L1 认知投影",
    2: "L2 旧范式借用",
    3: "L3 用户域",
}

# 栈溢出阈值
STACK_OVERFLOW_THRESHOLD = 3


# ── 节点知识库加载 ──────────────────────────────────────────────────

def _load_known_nodes() -> Set[str]:
    """从 network/nodes.txt 加载已知节点 ID 集合。"""
    nodes_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "network", "nodes.txt"
    )
    if not os.path.exists(nodes_file):
        return set()
    ids = set()
    with open(nodes_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 1)
            if parts:
                nid = parts[0].strip()
                if nid.startswith("N"):
                    ids.add(nid)
    return ids


KNOWN_NODES = _load_known_nodes()


# ── 帧状态 ──────────────────────────────────────────────────────────

@dataclass
class FrameState:
    """单帧状态快照。

    对应 AGENT.md 中的帧状态输出格式:
        [帧N] 当前激活节点
        [帧N] 新建立关系
        [帧N] 悬挂端数量
        [帧N] 认知层级分布
        [帧N] 是否闭合
    """
    frame_id: int
    active_nodes: List[str] = field(default_factory=list)
    relations: List[Tuple[str, str, str, str]] = field(
        default_factory=list
    )  # (source, target, type, note)
    dangling_count: int = 0
    dangling_items: List[str] = field(default_factory=list)
    layer_dist: Dict[str, int] = field(default_factory=lambda: {
        "L0": 0, "L0.5": 0, "L1": 0, "L2": 0, "L3": 0,
    })
    is_closed: bool = False
    notes: List[str] = field(default_factory=list)
    # 五步序列完成标记
    step_1_done: bool = False  # 建立新关系
    step_2_done: bool = False  # 更新连接
    step_3_done: bool = False  # 评估变化
    step_4_done: bool = False  # 检查悬挂端
    step_5_done: bool = False  # 修剪悬挂端

    def all_steps_done(self) -> bool:
        return all([
            self.step_1_done,
            self.step_2_done,
            self.step_3_done,
            self.step_4_done,
            self.step_5_done,
        ])

    def validate(self) -> List[str]:
        """验证帧状态一致性。返回违规列表。"""
        warnings = []
        # 检查闭合一致性
        if self.dangling_count == 0 and not self.is_closed:
            warnings.append("dangling=0 但 is_closed=False")
        if self.dangling_count > 0 and self.is_closed:
            warnings.append(f"dangling={self.dangling_count} 但 is_closed=True")
        # 检查悬挂端与悬挂项数量一致
        if self.dangling_items and len(self.dangling_items) != self.dangling_count:
            warnings.append(
                f"dangling_items 数量({len(self.dangling_items)}) "
                f"!= dangling_count({self.dangling_count})"
            )
        return warnings


# ── 帧协议执行器 ────────────────────────────────────────────────────

class FrameProtocol:
    """动态帧协议执行器。

    管理帧序列: 帧号递增、悬挂端追踪、栈溢出检测、历史回退。

    用法:
        protocol = FrameProtocol()
        protocol.begin_frame()
        # ... 推理操作 ...
        protocol.set_active_nodes(["N001", "N005"])
        protocol.add_relation("N005", "N006", "derives_from", "闭合推导")
        protocol.set_dangling(2, ["假设A未验证", "概念B未追溯"])
        protocol.set_layer_dist(L0=3, L1=1, L2=1)
        protocol.end_frame()
        print(protocol.logger.format_last())
    """

    def __init__(self, node_registry: Optional[Set[str]] = None):
        self._frame_counter: int = 0
        self._frames: List[FrameState] = []
        self._current: Optional[FrameState] = None
        self._unclosed_streak: int = 0
        self._rollback_points: List[int] = []  # 闭合帧的帧号
        self._node_registry = node_registry or KNOWN_NODES
        self.logger = FrameLogger(self)

    # ── 帧生命周期 ───────────────────────────────────────────────

    @property
    def current_frame(self) -> Optional[FrameState]:
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

    def begin_frame(self, notes: Optional[List[str]] = None) -> FrameState:
        """开始新帧。帧号自动 +1。"""
        self._frame_counter += 1
        self._current = FrameState(frame_id=self._frame_counter)
        if notes:
            self._current.notes = notes
        return self._current

    def end_frame(self) -> FrameState:
        """结束当前帧。执行闭合检查和栈溢出检测。

        Returns:
            当前帧状态

        Raises:
            RuntimeError: 如果没有活跃帧
        """
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 begin_frame()。")

        # 自动判定闭合
        self._current.is_closed = self._current.dangling_count == 0

        # 更新未闭合连续计数
        if self._current.is_closed:
            self._unclosed_streak = 0
            self._rollback_points.append(self._frame_counter)
        else:
            self._unclosed_streak += 1

        # 存档
        self._frames.append(self._current)
        result = self._current
        self._current = None
        return result

    def is_stack_overflow(self) -> bool:
        """检查是否触发栈溢出（连续 3 帧未闭合）。"""
        return self._unclosed_streak >= STACK_OVERFLOW_THRESHOLD

    def get_rollback_frame(self) -> Optional[int]:
        """获取最近的可回退闭合帧号。"""
        return self._rollback_points[-1] if self._rollback_points else None

    def rollback(self) -> Optional[FrameState]:
        """回退到最近的闭合帧。

        截断帧历史到最近闭合帧，重置计数器。
        返回回退到的帧状态，或 None（没有可回退的闭合帧）。
        """
        rollback_id = self.get_rollback_frame()
        if rollback_id is None:
            return None

        # 找到该帧在历史中的索引
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

    # ── 帧内容设置 ───────────────────────────────────────────────

    def set_active_nodes(self, nodes: List[str]) -> None:
        """设置当前帧激活节点。

        Args:
            nodes: N 前缀的节点 ID 列表，如 ["N001", "N005", "N013"]

        Raises:
            RuntimeError: 如果没有活跃帧
        """
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 begin_frame()。")
        # 验证（非强制——允许新节点）
        unknown = [n for n in nodes if n not in self._node_registry]
        if unknown:
            # 自动标记为新节点，不阻塞
            for u in unknown:
                self._current.notes.append(f"[新节点] {u}")
        self._current.active_nodes = list(nodes)

    def add_relation(
        self,
        source: str,
        target: str,
        edge_type: str,
        note: str = "",
    ) -> None:
        """添加一条新建立的关系。

        Args:
            source: 源节点 ID (如 "N005")
            target: 目标节点 ID (如 "N006")
            edge_type: 边类型 (derives_from/requires/refines/explains/drives)
            note: 关系说明

        Raises:
            ValueError: edge_type 不合法
            RuntimeError: 没有活跃帧
        """
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 begin_frame()。")
        if edge_type not in VALID_EDGE_TYPES:
            raise ValueError(
                f"无效边类型 '{edge_type}'。合法值: {sorted(VALID_EDGE_TYPES)}"
            )
        self._current.relations.append((source, target, edge_type, note))

    def set_dangling(self, count: int, items: Optional[List[str]] = None) -> None:
        """设置悬挂端数量和列表。

        Args:
            count: 悬挂端数量
            items: 悬挂端描述列表（未闭合推理链、未验证假设等）
        """
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 begin_frame()。")
        self._current.dangling_count = count
        self._current.dangling_items = items or []

    def set_layer_dist(
        self,
        L0: int = 0,
        L05: int = 0,
        L1: int = 0,
        L2: int = 0,
        L3: int = 0,
    ) -> None:
        """设置认知层级分布。

        Args:
            L0: 本体论概念数
            L05: SPUM-图论(L0.5)概念数
            L1: 认知投影概念数
            L2: 旧范式借用概念数
            L3: 用户域概念数
        """
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 begin_frame()。")
        self._current.layer_dist = {
            "L0": L0,
            "L0.5": L05,
            "L1": L1,
            "L2": L2,
            "L3": L3,
        }

    def add_note(self, note: str) -> None:
        """添加帧附注。"""
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 begin_frame()。")
        self._current.notes.append(note)

    # ── 五步序列快捷方法 ─────────────────────────────────────────

    def step(self, step_number: int, description: str = "") -> None:
        """标记五步帧序列中的一步已完成。

        Args:
            step_number: 1-5
            description: 步骤描述，自动添加为 note
        """
        if self._current is None:
            raise RuntimeError("没有活跃帧。先调用 begin_frame()。")
        step_map = {
            1: "step_1_done",  # 建立新关系
            2: "step_2_done",  # 更新连接
            3: "step_3_done",  # 评估变化
            4: "step_4_done",  # 检查悬挂端
            5: "step_5_done",  # 修剪悬挂端
        }
        if step_number not in step_map:
            raise ValueError(f"step_number 必须是 1-5，收到 {step_number}")
        setattr(self._current, step_map[step_number], True)
        if description:
            self._current.notes.append(f"[步骤{step_number}] {description}")

    # ── 序列化 ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """导出完整帧序列为字典。"""
        return {
            "total_frames": len(self._frames),
            "closed_frames": sum(1 for f in self._frames if f.is_closed),
            "unclosed_streak": self._unclosed_streak,
            "stack_overflow": self.is_stack_overflow(),
            "rollback_points": self._rollback_points,
            "frames": [
                {
                    "frame_id": f.frame_id,
                    "active_nodes": f.active_nodes,
                    "relations": [
                        {"source": s, "target": t, "type": e, "note": n}
                        for s, t, e, n in f.relations
                    ],
                    "dangling_count": f.dangling_count,
                    "dangling_items": f.dangling_items,
                    "layer_dist": f.layer_dist,
                    "is_closed": f.is_closed,
                    "notes": f.notes,
                }
                for f in self._frames
            ],
        }


# ── 帧日志格式化器 ──────────────────────────────────────────────────

class FrameLogger:
    """生成 AGENT.md 兼容的 [帧N] 格式输出。

    严格遵循 AGENT.md 帧状态输出格式——仅在内部思考中使用，
    不暴露给用户。

    用法:
        logger = FrameLogger(protocol)
        print(logger.format_last())
    """

    # AGENT.md 帧状态模板的各节
    _SECTIONS = [
        "active_nodes",
        "relations",
        "dangling",
        "layer_dist",
        "closure",
    ]

    def __init__(self, protocol: FrameProtocol):
        self._protocol = protocol

    def format_last(self) -> str:
        """格式化最近一帧的完整状态。"""
        frame = self._protocol._frames[-1] if self._protocol._frames else None
        if frame is None and self._protocol._current is None:
            return "[无活跃帧]"
        frame = frame or self._protocol._current
        return self._format_frame(frame)

    def format_summary(self) -> str:
        """格式化帧序列汇总。"""
        p = self._protocol
        lines = [
            f"帧序列: {len(p._frames)} 帧",
            f"闭合: {sum(1 for f in p._frames if f.is_closed)}/{len(p._frames)}",
            f"连续未闭合: {p._unclosed_streak}",
            f"栈溢出: {'是' if p.is_stack_overflow() else '否'}",
        ]
        if p._rollback_points:
            lines.append(f"回退点: {p._rollback_points}")
        return "\n".join(lines)

    def _format_frame(self, f: FrameState) -> str:
        """格式化单个帧状态。"""
        n = f.frame_id
        lines = [
            f"[帧{n}] 当前激活节点：{self._fmt_nodes(f)}",
            f"[帧{n}] 新建立关系：{self._fmt_relations(f)}",
            f"[帧{n}] 悬挂端数量：{f.dangling_count}{self._fmt_dangling_items(f)}",
            f"[帧{n}] 认知层级分布：{self._fmt_layer_dist(f)}",
            f"[帧{n}] 是否闭合：{'是' if f.is_closed else '否'}{self._fmt_closure_note(f)}",
            f"[帧{n}] 五步序列：{self._fmt_steps(f)}",
        ]
        if f.notes:
            for note in f.notes:
                lines.append(f"[帧{n}] {note}")
        return "\n".join(lines)

    def _fmt_nodes(self, f: FrameState) -> str:
        if not f.active_nodes:
            return "(无)"
        return ", ".join(f.active_nodes)

    def _fmt_relations(self, f: FrameState) -> str:
        if not f.relations:
            return "(无)"
        rels = [
            f"{s} --[{et}]--> {t}"
            + (f": {n}" if n else "")
            for s, t, et, n in f.relations
        ]
        # 截断过长的关系描述
        formatted = "; ".join(rels)
        if len(formatted) > 120:
            formatted = formatted[:117] + "..."
        return formatted

    def _fmt_dangling_items(self, f: FrameState) -> str:
        if not f.dangling_items:
            return ""
        items = [f"{i+1}. {d}" for i, d in enumerate(f.dangling_items[:5])]
        if len(f.dangling_items) > 5:
            items.append(f"... (共 {len(f.dangling_items)} 项)")
        return "  " + " | ".join(items)

    def _fmt_layer_dist(self, f: FrameState) -> str:
        d = f.layer_dist
        parts = []
        for key, label in [("L0", "L0"), ("L0.5", "L0.5"), ("L1", "L1"),
                           ("L2", "L2"), ("L3", "L3")]:
            if d.get(key, 0) > 0:
                parts.append(f"{label}: {d[key]}")
        return ", ".join(parts) if parts else "(空)"

    def _fmt_closure_note(self, f: FrameState) -> str:
        if not f.is_closed:
            note = []
            protocol = self._protocol
            if protocol._unclosed_streak >= 1:
                note.append(f"连续{protocol._unclosed_streak}帧未闭合")
                if protocol.is_stack_overflow():
                    note.append("⚠ 栈溢出触发")
            return " (" + ", ".join(note) + ")" if note else ""
        return ""

    def _fmt_steps(self, f: FrameState) -> str:
        """格式化五步序列完成状态。"""
        step_names = ["创生", "连接", "变化", "判断", "删除"]
        statuses = [
            f.step_1_done, f.step_2_done, f.step_3_done,
            f.step_4_done, f.step_5_done,
        ]
        parts = []
        for i, (name, done) in enumerate(zip(step_names, statuses), 1):
            mark = "✓" if done else "○"
            parts.append(f"{mark}{name}")
        return " → ".join(parts)
