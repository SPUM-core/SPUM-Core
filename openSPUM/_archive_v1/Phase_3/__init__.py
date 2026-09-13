"""
OpenSPUM Phase 3 — 三维几何聚簇求解器

核心目标:
    在三维空间中为晶子赋予空间位置，通过球体相切约束
    驱动晶子集合自组织为符合拓扑不变量 Σ(6−deg)=12 的闭合骨架。

物理基础 (SPUM 总纲 §2.3):
    每个晶子是一个直径为 4κ 的硬球。
    晶子间只能通过相切面 (tangent planes) 建立簇边。
    当一个闭合格由 12 个晶子构成时，最低能态 = 正二十面体。
    正二十面体满足：
        - 12 个顶点, 每个度数 = 5 (在子图内)
        - Σ(6−deg) = 12 × (6−5) = 12
        - 同胚于球面 (χ = V − E + F = 2)

模块组成:
    - geometry_solver.py:    力学松弛求解器
    - icosahedron_assembly:  正二十面体组装检测
    - validator:             三维拓扑不变量验证
"""

from .geometry_solver import (
    Sphere3D,
    GeometrySolver,
    ForceDirectedRelaxation,
)
from .icosahedron_assembly import (
    IcosahedronDetector,
    map_crystallites_to_icosahedron,
)
from .validator import (
    validate_spum_topology_3d,
    compute_spum_invariant_3d,
)

__all__ = [
    "Sphere3D",
    "GeometrySolver",
    "ForceDirectedRelaxation",
    "IcosahedronDetector",
    "map_crystallites_to_icosahedron",
    "validate_spum_topology_3d",
    "compute_spum_invariant_3d",
]
