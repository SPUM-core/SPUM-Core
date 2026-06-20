"""
原子核组装 — 12 个永恒粒子的正二十面体锁闭构型

SPUM 定义（元素化学/SPUM-VSPT.md §1.4）：
    12 个永恒粒子以正二十面体对称锁定形成原子核。
    - 中子（z=0）：全部 12 个虚面朝内 → 表面纯实面（48 实面，0 虚面）
    - 质子（z=+1）：见 ProtonConfig 两种稳定构型

质子构型说明：
    正二十面体 12 粒子锁闭下，可稳定成立的朝外粒子数只有两种：
    - 单隼（SINGLE_TENON）：11 朝内 + 1 朝外 → 44 实 + 1 虚，拓扑电荷 = 1
    - 双隼（DOUBLE_TENON）：10 朝内 + 2 朝外 → 40 实 + 2 虚，拓扑电荷 = 2
    二者均为应力均衡的稳定构型，对应不同的化学活性。

关键数理：
    - 每个永恒粒子：4 实面 + 1 虚面 = 5 面
    - 12-锁闭构型：12×4 = 48 实面对外暴露
    - 中子：0/48 = 0% 虚面
    - 单隼质子：1/48 ≈ 2.08% 虚面
    - 双隼质子：2/48 ≈ 4.17% 虚面

原子核结合（永恒粒子置换模型）：
    一个质子的朝外虚面永恒粒子与一个中子接触时，
    不是简单的表面吸收，而是该永恒粒子置换了中子中的某个永恒粒子，
    形成 12 + 12 - 1 = 23 粒子的复合结构。
    这导致净湮灭通量亏损 ΔΦ > 0 → 结合能以湮灭亏损形式释放。

核表面：
    - 实面区：可生长 VSPT → 电子汇聚（拓扑负区域）
    - 虚面区：不可生长 VSPT → 质子暴露（拓扑正区域）
"""

import math
from enum import Enum
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
    NEUTRON = "neutron"    # z=0, 全朝内
    PROTON = "proton"      # z=+1


class ProtonConfig(Enum):
    """质子构型——两种可稳定成立的朝外粒子数。

    SINGLE_TENON (单隼):
        11 朝内 + 1 朝外 → 44 实 + 1 虚, 拓扑电荷 = 1
        （氢等活性元素的对应构型）

    DOUBLE_TENON (双隼):
        10 朝内 + 2 朝外 → 40 实 + 2 虚, 拓扑电荷 = 2
        （氦等惰性元素的对应构型）
    """
    SINGLE_TENON = "single_tenon"    # 11+1
    DOUBLE_TENON = "double_tenon"    # 10+2


@dataclass
class NucleusAssembly:
    """原子核 — 12永恒粒子正二十面体锁闭构型。

    Attributes:
        particles:     12 个永恒粒子的字典 {uid: EternalParticle}
        positions:     各粒子的正二十面体顶点坐标 {uid: (x,y,z)}
        ntype:         核类型（中子/质子）
        proton_config: 质子构型（仅质子有效，单隼/双隼）
        outward_solid: 对外暴露的实面总数
        outward_vacant: 对外暴露的虚面总数
        charge:        净电荷（拓扑不变量，= 朝外虚面数）
        spum_invariant: Σ(6-deg) 验证值
    """
    particles: Dict[str, EternalParticle]
    positions: Dict[str, Tuple[float, float, float]]
    ntype: NucleusType
    proton_config: Optional[ProtonConfig] = None
    outward_solid: int = 0
    outward_vacant: int = 0
    charge: int = 0
    spum_invariant: int = 12

    @property
    def solid_ratio(self) -> float:
        """表面实面占比 = outward_solid / total_possible (=48 对所有 12 粒子)。"""
        total_possible = len(self.particles) * 4
        return self.outward_solid / max(1, total_possible)

    @property
    def vacant_ratio(self) -> float:
        """表面虚面占比 = outward_vacant / total_possible。"""
        total_possible = len(self.particles) * 4
        return self.outward_vacant / max(1, total_possible)

    @property
    def outward_solid_faces(self) -> int:
        return self.outward_solid

    @property
    def outward_vacant_faces(self) -> int:
        return self.outward_vacant


def _icosahedron_vertices(edge_length: float = 2.0) -> List[Tuple[float, float, float]]:
    """标准正二十面体的 12 个顶点。"""
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
    proton_config: ProtonConfig = ProtonConfig.SINGLE_TENON,
) -> NucleusAssembly:
    """构建 12-永恒粒子原子核。

    Args:
        ntype:          核类型（中子/质子）
        edge_length:    正二十面体边长
        particle_radius:永恒粒子半径
        proton_config:  质子构型（单隼=11+1, 双隼=10+2）

    Returns:
        NucleusAssembly 实例
    """
    vertices = _icosahedron_vertices(edge_length)

    # 根据构型选择朝外粒子索引
    if ntype == NucleusType.NEUTRON:
        outward_indices = set()
    elif proton_config == ProtonConfig.SINGLE_TENON:
        outward_indices = {0}  # 11 朝内 + 1 朝外
    else:  # DOUBLE_TENON
        outward_indices = {0, 1}  # 10 朝内 + 2 朝外

    particles = {}
    positions = {}
    outward_solid = 0
    outward_vacant = 0

    for i, v in enumerate(vertices):
        uid = f"ep_{i:02d}"
        orientation = (
            EternalParticleOrientation.OUTWARD
            if i in outward_indices
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
        ep.vacant_face_dir = (
            _outward_direction(v)
            if orientation == EternalParticleOrientation.OUTWARD
            else (0.0, 0.0, 0.0)
        )

        particles[uid] = ep
        positions[uid] = v

        outward_solid += ep.solid_face_count
        outward_vacant += ep.vacant_face_count

    # 核电荷 = 朝外虚面数（拓扑不变量）
    charge = outward_vacant

    return NucleusAssembly(
        particles=particles,
        positions=positions,
        ntype=ntype,
        proton_config=proton_config if ntype == NucleusType.PROTON else None,
        outward_solid=outward_solid,
        outward_vacant=outward_vacant,
        charge=charge,
    )
