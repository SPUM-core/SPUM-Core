"""
OpenSPUM Phase 4 — VSPT 球面生长拓扑

从 12-永恒粒子原子核（Phase 3 正二十面体组装输出）出发，
模拟实面/虚面分布驱动的 VSPT（Vacant-Solid Proliferation Topology）
分支生长，验证 VSPT 三律，还原电子壳层填充与元素周期表。

层级定位:
    Phase 1 (⟨P,ε⟩ 关系网络) → Phase 2 (帧演化)
    → Phase 3 (三维晶子聚簇/正二十面体组装)
    → **Phase 4 (原子核表面 VSPT 生长 → 电子壳层 → 元素分类)**

包含:
    - EternalParticle:   永恒粒子模型——5 面正二十面体局部，4实面+1虚面
    - NucleusAssembly:   原子核组装——12永恒粒子正二十面体锁闭构型
    - VSPTTree:          VSPT 分支树——从实面生长的树状分支结构
    - VSPTEngine:        VSPT 引擎——管理多树并发生长、壳层填充
    - ElectronShell:     电子壳层——2n² 填充序列与元素分类
    - VSPTValidator:     VSPT 三律验证器——k≥3, 120°, ρ∝r⁻³

依赖:
    - Phase 1: constants (κ, τ, 晶子阈值)
    - Phase 3: icosahedron_assembly (正二十面体顶点/边)
"""

from .eternal_particle import (
    FaceType,
    EternalParticle,
    EternalParticleOrientation,
    create_eternal_particle,
)
from .nucleus_assembly import (
    NucleusAssembly,
    NucleusType,
    build_nucleus,
)
from .vspt_growth import (
    VSPTNode,
    VSPTTree,
    VSPTEngine,
    VSPTConfig,
)
from .electron_shell import (
    ElementClass,
    shell_capacity,
    cumulative_capacity,
    classify_element,
    periodic_table_lookup,
    compute_isotope_mass,
    compute_hydrogen_isotope,
    demo_compute_elements,
    MP_SPUM,
    MN_SPUM,
)
from .validator import (
    VSPTValidator,
    validate_k3_law,
    validate_120_angle,
    validate_density_power_law,
)

__all__ = [
    # eternal_particle
    "FaceType",
    "EternalParticle",
    "EternalParticleOrientation",
    "create_eternal_particle",
    # nucleus_assembly
    "NucleusAssembly",
    "NucleusType",
    "build_nucleus",
    # vspt_growth
    "VSPTNode",
    "VSPTTree",
    "VSPTEngine",
    "VSPTConfig",
    # electron_shell
    "ElectronShell",
    "ElementClass",
    "shell_capacity",
    "classify_element",
    "periodic_table_lookup",
    # validator
    "VSPTValidator",
    "validate_k3_law",
    "validate_120_angle",
    "validate_density_power_law",
]
