"""
拓扑地址系统 — SPUM 空间粒子的唯一标识与分化机制

每个空间粒子拥有一个全局唯一的拓扑地址，包含:
    - uid:                  全局唯一标识符（自然数的物理源头）
    - differentiation_step: 分化步数（距离原点的拓扑距离）

地址生成完全确定性:
    - differentiate_from 使用单调递增计数器，非 uuid4
    - 同一进程内，给定调用顺序，UID 严格确定
    - reset_differentiation_counter() 用于测试与引擎初始化
"""

from __future__ import annotations
from dataclasses import dataclass

# 单调递增计数器：保证 differentiate_from 产生确定性 UID
_differentiation_counter: int = 0


def reset_differentiation_counter() -> None:
    """重置分化计数器（用于引擎初始化与测试）。"""
    global _differentiation_counter
    _differentiation_counter = 0


@dataclass(frozen=True)
class TopologicalAddress:
    """空间粒子的拓扑地址。

    frozen=True 保证地址不可变，可用作字典键。

    Attributes:
        uid:                  全局唯一标识字符串
        differentiation_step: 分化步数，0 表示原点（第一粒子）
    """

    uid: str
    differentiation_step: int = 0

    # ------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------
    @classmethod
    def origin(cls) -> TopologicalAddress:
        """创建原点地址（宇宙第一个空间粒子）。

        原点地址的 uid 固定为 'origin'，步数为 0。
        这是整个宇宙网络的拓扑起点。
        """
        return cls(uid="origin", differentiation_step=0)

    @staticmethod
    def differentiate_from(parent: TopologicalAddress) -> TopologicalAddress:
        """从父地址分化出一个新地址。

        新地址的步数 = 父地址步数 + 1。
        uid 由步数前缀 + 单调计数器构成，完全确定性。

        Args:
            parent: 父拓扑地址

        Returns:
            新的子拓扑地址
        """
        global _differentiation_counter
        _differentiation_counter += 1
        step = parent.differentiation_step + 1
        uid = f"N{step}_{_differentiation_counter:012x}"
        return TopologicalAddress(uid=uid, differentiation_step=step)

    # ------------------------------------------------------------
    # 排序支持（用于确定性候选生成）
    # ------------------------------------------------------------
    def __lt__(self, other: TopologicalAddress) -> bool:
        if not isinstance(other, TopologicalAddress):
            return NotImplemented
        return (self.differentiation_step, self.uid) < (
            other.differentiation_step,
            other.uid,
        )

    def __le__(self, other: TopologicalAddress) -> bool:
        if not isinstance(other, TopologicalAddress):
            return NotImplemented
        return (self.differentiation_step, self.uid) <= (
            other.differentiation_step,
            other.uid,
        )
