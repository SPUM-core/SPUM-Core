"""
永恒粒子（光子）模型 — VSPT 生长的基本单元

SPUM 定义（元素化学/SPUM-VSPT.md §1.3）：
    每个永恒粒子（光子）= 一个正二十面体形态的凸球体，表面有 5 个三角形面。
    - 4 个实面（Solid Face）：三角形闭合面，可供 VSPT 生长
    - 1 个虚面/开口（Vacant Face）：三角形缺失，不可生长 VSPT

实面与虚面的分布由永恒粒子的朝向决定：
    - 朝内：虚面指向核心 → 对外暴露全部 4 个实面
    - 朝外：虚面指向核外 → 对外暴露 0 个实面 + 1 个虚面缺口

手性（SPUM-VSPT.md §2）：
    毛球定理强制 VSPT 分支场在球面上产生旋向。
    每个永恒粒子的实面 VSPT 生长方向有 L（左旋）或 D（右旋）两种。
"""

import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


class FaceType(Enum):
    """面类型"""
    SOLID = "solid"      # 实面 — 可生长 VSPT
    VACANT = "vacant"    # 虚面/缺口 — 不可生长 VSPT


class EternalParticleOrientation(Enum):
    """永恒粒子朝向"""
    INWARD = "inward"    # 虚面指向核心 — 对外 4 实面
    OUTWARD = "outward"  # 虚面指向核外 — 对外 1 虚面 + 0 实面


class Handedness(Enum):
    """VSPT 手性 (毛球定理强制)"""
    L = -1  # 左旋
    D = +1  # 右旋


@dataclass
class EternalParticle:
    """永恒粒子（光子）= 带缺口的正二十面体凸球体局部。

    Attributes:
        uid:              全局唯一 ID
        orientation:      朝向（朝内/朝外）
        handedness:       VSPT 手性 L/D（由毛球定理决定）
        solid_faces:      4 个实面的 VSPT 树（每个实面一棵）
        vacant_face_dir:  虚面方向（单位向量，指向缺口法线）
        position:         晶子坐标（来自 Phase 3）
        radius:           球体半径
        is_nucleon:       是否为核子（参与原子核组装）
    """
    uid: str
    orientation: EternalParticleOrientation = EternalParticleOrientation.INWARD
    handedness: Handedness = Handedness.L
    solid_faces: List[Optional[str]] = field(default_factory=lambda: [None] * 4)
    vacant_face_dir: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    is_nucleon: bool = False

    @property
    def solid_face_count(self) -> int:
        """对外暴露的实面数。"""
        if self.orientation == EternalParticleOrientation.INWARD:
            return 4  # 全部对外
        else:
            return 0  # 虚面朝外，无实面暴露

    @property
    def vacant_face_count(self) -> int:
        """对外暴露的虚面数。"""
        if self.orientation == EternalParticleOrientation.OUTWARD:
            return 1
        else:
            return 0

    @property
    def solid_ratio(self) -> float:
        """实面占比。"""
        return self.solid_face_count / 5.0

    @property
    def vacant_ratio(self) -> float:
        """虚面占比。"""
        return self.vacant_face_count / 5.0


def create_eternal_particle(
    uid: str,
    orientation: EternalParticleOrientation = EternalParticleOrientation.INWARD,
    handedness: Optional[Handedness] = None,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 1.0,
) -> EternalParticle:
    """创建一个永恒粒子。

    Args:
        uid:         全局唯一 ID
        orientation: 朝向（INWARD=虚面朝内, OUTWARD=虚面朝外）
        handedness:  手性 L/D；None 时自动根据 uid 确定
        position:    三维坐标
        radius:      球体半径

    Returns:
        EternalParticle 实例
    """
    if handedness is None:
        # 基于 uid 确定性选择手性（毛球定理 L/D 二选一）
        handedness = Handedness.L if (hash(uid) % 2 == 0) else Handedness.D

    return EternalParticle(
        uid=uid,
        orientation=orientation,
        handedness=handedness,
        position=position,
        radius=radius,
    )
