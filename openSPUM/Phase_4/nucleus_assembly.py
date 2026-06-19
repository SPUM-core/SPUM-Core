"""
原子核组装 — 12 个永恒粒子的正二十面体锁闭构型

SPUM 定义（元素化学/SPUM-VSPT.md §1.4）：
    12 个永恒粒子以正二十面体对称锁定形成原子核。
    - 中子（z=0）：全部 12 个虚面朝内 → 表面纯实面（48 实面，0 虚面）
    - 质子（z=+1）：10 个朝内 + 2 个朝外 → 表面 46 实面 + 2 虚面缺口

关键数理：
    - 每个永恒粒子：4 实面 + 1 虚面 = 5 面
    - 12-锁闭构型：12×4 = 48 实面对外暴露
    - 中子：0/48 = 0% 虚面
    - 质子：2/48 ≈ 4.17% 虚面

核表面：
    - 实面区：可生长 VSPT → 负电荷（空间汇聚）
    - 虚面区：不可生长 VSPT → 正电荷（空间缺失源）
"""

import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Phase_4.eternal_particle import (
    EternalParticle,
    EternalParticleOrientation,
    Handedness,
    create_eternal_particle,
)


class NucleusType(Enum):
    """原子核类型"""
    NEUTRON = "neutron"   # z=0, 全朝内
    PROTON = "proton"     # z=+1, 10内2外


@dataclass
class NucleusAssembly:
    """原子核 — 12永恒粒子正二十面体锁闭构型。

    Attributes:
        particles:     12 个永恒粒子的字典 {uid: EternalParticle}
        positions:     各粒子的正二十面体顶点坐标 {uid: (x,y,z)}
        ntype:         核类型（中子/质子）
        outward_solid: 对外暴露的实面总数
        outward_vacant: 对外暴露的虚面总数
        charge:        净电荷（拓扑不变量）
        spum_invariant: Σ(6-deg) 验证值
    """
    particles: Dict[str, EternalParticle]
    positions: Dict[str, Tuple[float, float, float]]
    ntype: NucleusType
    outward_solid: int
    outward_vacant: int
    charge: int
    spum_invariant: int = 12

    @property
    def solid_ratio(self) -> float:
        """表面实面占比 = outward_solid / total_possible (=48 对所有 12 粒子)。"""
        # 12 个永恒粒子各有 4 个实面 = 48 个理论实面
        total_possible = len(self.particles) * 4
        return self.outward_solid / max(1, total_possible)

    @property
    def vacant_ratio(self) -> float:
        """表面虚面占比 = outward_vacant / total_possible。

        SPUM-VSPT.md §1.4：
            中子对外等效开口占比：0/48 = 0%
            质子对外等效开口占比：2/48 ≈ 4.17%
        """
        total_possible = len(self.particles) * 4
        return self.outward_vacant / max(1, total_possible)

    @property
    def outward_solid_faces(self) -> int:
        """对外暴露的实面数。"""
        return self.outward_solid

    @property
    def outward_vacant_faces(self) -> int:
        """对外暴露的虚面数。"""
        return self.outward_vacant


def _icosahedron_vertices(edge_length: float = 2.0) -> List[Tuple[float, float, float]]:
    """标准正二十面体的 12 个顶点（复用 Phase 3 逻辑）。"""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    s = edge_length / 2
    return [
        (s, s * phi, 0), (s, -s * phi, 0), (-s, s * phi, 0), (-s, -s * phi, 0),
        (0, s, s * phi), (0, -s, s * phi), (0, s, -s * phi), (0, -s, -s * phi),
        (s * phi, 0, s), (-s * phi, 0, s), (s * phi, 0, -s), (-s * phi, 0, -s),
    ]


def _outward_direction(vertex: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """从顶点指向核外的方向（归一化）。"""
    norm = math.sqrt(sum(v * v for v in vertex))
    if norm < 1e-12:
        return (0.0, 0.0, 0.0)
    return tuple(v / norm for v in vertex)


def build_nucleus(
    ntype: NucleusType = NucleusType.NEUTRON,
    edge_length: float = 2.0,
    particle_radius: float = 1.0,
    outward_particle_indices: Optional[List[int]] = None,
) -> NucleusAssembly:
    """构建 12-永恒粒子原子核。

    Args:
        ntype:                   核类型（中子/质子）
        edge_length:             正二十面体边长
        particle_radius:         永恒粒子半径
        outward_particle_indices: 朝外粒子的顶点索引列表
                                 中子=[]，质子=[0, 1]（默认前两个顶点朝外）

    Returns:
        NucleusAssembly 实例
    """
    vertices = _icosahedron_vertices(edge_length)
    if outward_particle_indices is None:
        outward_particle_indices = [0, 1] if ntype == NucleusType.PROTON else []

    particles = {}
    positions = {}
    outward_solid = 0
    outward_vacant = 0

    for i, v in enumerate(vertices):
        uid = f"ep_{i:02d}"
        orientation = (
            EternalParticleOrientation.OUTWARD
            if i in outward_particle_indices
            else EternalParticleOrientation.INWARD
        )

        # 手性：正二十面体交替分配 L/D（毛球定理）
        handedness = Handedness.L if i % 2 == 0 else Handedness.D

        ep = create_eternal_particle(
            uid=uid,
            orientation=orientation,
            handedness=handedness,
            position=v,
            radius=particle_radius,
        )
        ep.vacant_face_dir = _outward_direction(v) if orientation == EternalParticleOrientation.OUTWARD else (0.0, 0.0, 0.0)

        particles[uid] = ep
        positions[uid] = v

        outward_solid += ep.solid_face_count
        outward_vacant += ep.vacant_face_count

    # 核电荷 = 朝外虚面数（拓扑不变量，SPUM-VSPT.md §4.2）
    charge = outward_vacant  # 质子 +1, 中子 0

    return NucleusAssembly(
        particles=particles,
        positions=positions,
        ntype=ntype,
        outward_solid=outward_solid,
        outward_vacant=outward_vacant,
        charge=charge,
    )
