"""
Apollonian — 阿波罗尼奥斯约束求解引擎。

本体论：
    - 球体的几何（位置、半径）不由内部属性决定
    - 由外部关系（相切约束）唯一确定
    - 移除球体后，空洞几何仍可查询（几何犹存）
"""

from .core.sphere import ApollonianSphere
from .core.network import ApollonianNetwork, FrameSnapshot
from .core.descartes import (
    solve_bend_2d,
    solve_bend_3d,
    solve_position_trilateration,
    solve_radius_from_neighbors,
)

__all__ = [
    "ApollonianSphere",
    "ApollonianNetwork",
    "FrameSnapshot",
    "solve_bend_2d",
    "solve_bend_3d",
    "solve_position_trilateration",
    "solve_radius_from_neighbors",
]
