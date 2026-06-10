"""
spum-review → FrameProtocol 回滚闭环

将 spum-review.md 的四维评分映射到 FrameProtocol 的帧生命周期:
  - A/B/C/D 任一维度 < 50 → 自动触发 rollback()
  - LE >= 2 → 自动触发 rollback()
  - A/B/C/D 任一维度 < 70 但 ≥ 50 → 警告, 标注问题但继续
  - A≥70 且 B≥70 且 C≥70 且 D≥70 且 LE≤1 → 通过

用法:
    from core import FrameProtocol, ReviewBridge, ReviewResult
    bridge = ReviewBridge(protocol)

    # 评审结果 (通常由 LLM 评审流程生成)
    result = ReviewResult(A=85, B=72, C=90, D=78, LE=1, violations=[...])
    action = bridge.submit(result)  # ReviewVerdict.PASS

    result = ReviewResult(A=45, B=80, C=75, D=85, LE=0, violations=[...])
    action = bridge.submit(result)  # ReviewVerdict.ROLLBACK → protocol.rollback()
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto
from .frame import FrameProtocol, FrameState


# ── 评审维度 ──────────────────────────────────────────────────────────

class ReviewVerdict(Enum):
    PASS = "通过"
    WARN = "警告"
    ROLLBACK = "回滚"


class Severity(Enum):
    FATAL = "致命"       # -40
    SEVERE = "严重"     # -20
    WARNING = "警告"    # -10
    REMINDER = "提醒"   # -5

    @property
    def deduction(self) -> int:
        return {Severity.FATAL: 40, Severity.SEVERE: 20,
                Severity.WARNING: 10, Severity.REMINDER: 5}[self]


@dataclass
class Violation:
    """单条违规记录。"""
    severity: Severity
    dimension: str  # A/B/C/D
    description: str
    location: str = ""


@dataclass
class ReviewResult:
    """spum-review.md 四维评审结果。

    对应 spum-review.md 的评审报告格式:
        A 架构纯净度：XX/100  [致命:X 严重:X 警告:X 提醒:X]
        B 推导严密性：XX/100  [循环:X 悬空:X 跃迁:X 模糊词:X]
        C 归约完备度：XX/100
        D 方法论透明度：XX/100
        LE 梯子依赖：X
        判定：通过 / 警告 / 回滚
    """
    A: int  # 架构纯净度 0-100
    B: int  # 推导严密性 0-100
    C: int  # 归约完备度 0-100
    D: int  # 方法论透明度 0-100
    LE: int  # 梯子依赖 0-3
    violations: List[Violation] = field(default_factory=list)
    frame_id: int = 0  # 关联的帧号

    def verdict(self) -> ReviewVerdict:
        """根据 spum-review.md 阈值规则判定。

        规则:
            - 任一维度 < 50 → ROLLBACK
            - LE >= 2 → ROLLBACK
            - 任一维度 < 70 但 >= 50 → WARN
            - 全部 >= 70 且 LE <= 1 → PASS
        """
        dims = {"A": self.A, "B": self.B, "C": self.C, "D": self.D}

        # 致命检查
        for dim, score in dims.items():
            if score < 50:
                return ReviewVerdict.ROLLBACK
        if self.LE >= 2:
            return ReviewVerdict.ROLLBACK

        # 警告检查
        for dim, score in dims.items():
            if score < 70:
                return ReviewVerdict.WARN

        return ReviewVerdict.PASS

    def failing_dimensions(self) -> List[str]:
        """返回低于阈值的维度列表。"""
        result = []
        for dim, score in [("A", self.A), ("B", self.B),
                            ("C", self.C), ("D", self.D)]:
            if score < 50:
                result.append(dim)
        if self.LE >= 2:
            result.append("LE")
        return result

    def warning_dimensions(self) -> List[str]:
        """返回警告区间的维度列表（≥50 但 <70）。"""
        result = []
        for dim, score in [("A", self.A), ("B", self.B),
                            ("C", self.C), ("D", self.D)]:
            if 50 <= score < 70:
                result.append(dim)
        return result

    # ── spum-review.md 格式输出 ────────────────────────────────────

    def format_report(self) -> str:
        """生成与 spum-review.md 兼容的评审报告。"""
        n = self.frame_id
        dims = [
            f"A 架构纯净度：{self.A}/100",
            f"B 推导严密性：{self.B}/100",
            f"C 归约完备度：{self.C}/100",
            f"D 方法论透明度：{self.D}/100",
            f"LE 梯子依赖：{self.LE}",
        ]
        v = self.verdict()
        lines = [
            f"REVIEW REPORT 帧{n}",
            *dims,
            f"判定：{v.value}",
        ]
        if self.violations:
            fatal = [vl for vl in self.violations if vl.severity == Severity.FATAL]
            severe = [vl for vl in self.violations if vl.severity == Severity.SEVERE]
            warn = [vl for vl in self.violations if vl.severity == Severity.WARNING]
            parts = []
            if fatal:
                parts.append(f"致命:{len(fatal)}")
            if severe:
                parts.append(f"严重:{len(severe)}")
            if warn:
                parts.append(f"警告:{len(warn)}")
            lines.append(f"违规：{', '.join(parts)}")
            for vl in self.violations[:5]:
                lines.append(f"  [{vl.severity.value}] [{vl.dimension}] {vl.description}")
        return "\n".join(lines)

    def is_warning(self) -> bool:
        return self.verdict() == ReviewVerdict.WARN

    def is_rollback(self) -> bool:
        return self.verdict() == ReviewVerdict.ROLLBACK


# ── 回滚闭环桥接器 ──────────────────────────────────────────────────

@dataclass
class ReviewAction:
    """单次评审动作的结果。"""
    verdict: ReviewVerdict
    result: ReviewResult
    rollback_frame: Optional[int] = None
    message: str = ""


class ReviewBridge:
    """spum-review 评审 → FrameProtocol 回滚闭环。

    用法:
        bridge = ReviewBridge(protocol)
        action = bridge.submit(result)

        if action.verdict == ReviewVerdict.ROLLBACK:
            # 已自动执行 protocol.rollback()
            print(f"回退到帧 {action.rollback_frame}")
    """

    def __init__(self, protocol: FrameProtocol):
        self._protocol = protocol
        self._history: List[Tuple[ReviewResult, ReviewAction]] = []

    @property
    def protocol(self) -> FrameProtocol:
        return self._protocol

    @property
    def review_history(self) -> List[Tuple[ReviewResult, ReviewAction]]:
        return list(self._history)

    def submit(self, result: ReviewResult) -> ReviewAction:
        """提交评审结果，自动处理回滚。

        Args:
            result: 评审结果

        Returns:
            ReviewAction: 包含判定和自动执行的操作
        """
        # 自动关联当前/最后的帧号
        if result.frame_id == 0:
            if self._protocol._frames:
                result.frame_id = self._protocol._frames[-1].frame_id
            elif self._protocol._current:
                result.frame_id = self._protocol._current.frame_id

        v = result.verdict()

        if v == ReviewVerdict.PASS:
            action = ReviewAction(
                verdict=v,
                result=result,
                message=f"帧{result.frame_id} 评审通过",
            )

        elif v == ReviewVerdict.WARN:
            dims = result.warning_dimensions()
            action = ReviewAction(
                verdict=v,
                result=result,
                message=f"帧{result.frame_id} 评审警告: {', '.join(dims)}",
            )

        else:  # ROLLBACK
            # 执行回滚
            rollback_target = self._protocol.get_rollback_frame()
            if rollback_target is not None:
                rolled_to = self._protocol.rollback()
                msg = (
                    f"帧{result.frame_id} 评审不通过: "
                    f"{', '.join(result.failing_dimensions())} → "
                    f"回退到帧 {rollback_target}"
                )
                action = ReviewAction(
                    verdict=v,
                    result=result,
                    rollback_frame=rollback_target,
                    message=msg,
                )
            else:
                # 无可回退点：重置协议
                msg = (
                    f"帧{result.frame_id} 评审不通过但无可回退闭合帧, "
                    f"协议重置"
                )
                action = ReviewAction(
                    verdict=v,
                    result=result,
                    message=msg,
                )

        self._history.append((result, action))
        return action

    def submit_with_callbacks(
        self,
        result: ReviewResult,
        on_rollback: callable = None,
        on_warn: callable = None,
    ) -> ReviewAction:
        """提交评审并执行回调。

        Args:
            result: 评审结果
            on_rollback: 回滚时调用的回调(ReviewAction)
            on_warn: 警告时调用的回调(ReviewAction)
        """
        action = self.submit(result)
        if action.verdict == ReviewVerdict.ROLLBACK and on_rollback:
            on_rollback(action)
        elif action.verdict == ReviewVerdict.WARN and on_warn:
            on_warn(action)
        return action

    def summary(self) -> str:
        """评审历史摘要。"""
        if not self._history:
            return "无评审记录"

        total = len(self._history)
        passes = sum(1 for _, a in self._history if a.verdict == ReviewVerdict.PASS)
        warns = sum(1 for _, a in self._history if a.verdict == ReviewVerdict.WARN)
        rollbacks = sum(1 for _, a in self._history if a.verdict == ReviewVerdict.ROLLBACK)

        lines = [
            f"评审历史: {total} 次",
            f"  通过: {passes}",
            f"  警告: {warns}",
            f"  回滚: {rollbacks}",
        ]
        return "\n".join(lines)

    def last_result(self) -> Optional[ReviewResult]:
        """最近一次评审结果。"""
        return self._history[-1][0] if self._history else None

    def last_action(self) -> Optional[ReviewAction]:
        """最近一次评审动作。"""
        return self._history[-1][1] if self._history else None
