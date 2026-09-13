"""
Coda — SPUM 元胞自动机的纯信息单元。

本体论约束：
    - 节点不存储任何属性，只存储一个整数：度数 degree
    - 节点的全部"知识"来自与邻居的通信
    - 没有坐标、没有半径、没有体积、没有时间戳
    - 节点不是对象，是通信端点

度数即存在：
    deg = 0  → 潜态（不在网络中）
    deg = 1  → 悬挂态（不满足最小度数≥2）
    deg ≥ 2  → 稳定存在态
    deg ≥ κ → 晶子饱和态（κ = CRYSTALLITE_DEGREE_THRESHOLD）
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Coda:
    """单个 Coda 单元——仅存储度数和邻居引用。

    不存储：
        - 坐标（由投影层只读计算）
        - 半径（由投影层只读计算）
        - 时间戳（帧号由 Simulator 管理）
        - uid（由数组索引隐含）
    """
    degree: int = 0
    neighbors: List[int] = field(default_factory=list)

    @property
    def is_latent(self) -> bool:
        """deg = 0：潜态，未在网络中显化。"""
        return self.degree == 0

    @property
    def is_dangling(self) -> bool:
        """deg = 1：悬挂端，不满足最小度数 ≥ 2。"""
        return self.degree == 1

    @property
    def is_stable(self) -> bool:
        """deg ≥ 2：稳定存在。"""
        return self.degree >= 2

    @property
    def is_crystallite(self) -> bool:
        """deg ≥ κ：晶子饱和。"""
        from .rules import CRYSTALLITE_DEGREE_THRESHOLD
        return self.degree >= CRYSTALLITE_DEGREE_THRESHOLD

    @property
    def dangling_deficit(self) -> int:
        """悬挂亏缺：需要多少边才能达到最小度数 2。"""
        return max(0, 2 - self.degree)
