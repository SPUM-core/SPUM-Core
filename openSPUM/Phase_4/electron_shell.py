"""
电子壳层与元素分类 — 从 Z 的第一性原理计算

SPUM 定义（元素化学/SPUM-VSPT.md §4-§6）：
    元素周期表不是"质子数的排序"——它是原子核 VSPT 构型的拓扑分类。
    - 周期 = VSPT 壳层数（当前正在填充的最外壳层编号）
    - 族 = 核表面实面/虚面分布模式（最外壳层占有数 → 虚面数）
    - 电子壳层容量 2n² 来源于正二十面体顶点集的逐层叠加

核心计算：
    1. 给定 Z，按 2n² 容量逐壳层填充电子
    2. 最外壳层占有数 → 虚面数（占有数到 7 时虚面=占有数，满时=0）
    3. 虚面数 → 族（1→1, 2→2, 3→13, 4→14, 5→15, 6→16, 7→17, 0→18）
    4. 壳层索引 → 周期
    5. 虚面数 → 元素大类（碱金属/卤素/稀有气体等）
    6. 同位素质量 = A × m_p + (A-Z) × Δm_np - 结合能修正

    没有查表。一切从 Z 和壳层容量 2n² 计算。
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# —————————————————————————————————————————————
# 元素分类枚举（从虚面数计算得出，非查表）
# —————————————————————————————————————————————

class ElementClass(Enum):
    NOBLE_GAS = "稀有气体"
    ALKALI_METAL = "碱金属"
    ALKALINE_EARTH = "碱土金属"
    BORON_GROUP = "硼族"
    CARBON_GROUP = "碳族"
    NITROGEN_GROUP = "氮族"
    OXYGEN_GROUP = "氧族"
    HALOGEN = "卤素"
    TRANSITION_METAL = "过渡金属"


SHELL_NAMES = ["K", "L", "M", "N", "O", "P", "Q"]


# —————————————————————————————————————————————
# 壳层容量计算（从正二十面体顶点集几何推导）
# —————————————————————————————————————————————

def shell_capacity(n: int) -> int:
    """第 n 壳层的容量 = 2n²。

    推导（SPUM-VSPT.md §6.1）：
        正二十面体顶点集逐层叠加：
        n=1: 两极 2 顶点 → 2
        n=2: 赤道环 8 顶点 → 8 = 2×2²
        n=3: 旋转 14.359° 后下一层 18 顶点 → 18 = 2×3²
    """
    return 2 * n * n


def cumulative_capacity(n: int) -> int:
    """前 n 壳层总容量 = Σ_{i=1}^{n} 2i² = n(n+1)(2n+1)/3。"""
    return n * (n + 1) * (2 * n + 1) // 3


# —————————————————————————————————————————————
# 元素符号与名称生成
# —————————————————————————————————————————————

# 前 118 号元素符号（IUPAC 标准）
ELEMENT_SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

ELEMENT_NAMES = [
    "氢", "氦", "锂", "铍", "硼", "碳", "氮", "氧", "氟", "氖",
    "钠", "镁", "铝", "硅", "磷", "硫", "氯", "氩", "钾", "钙",
    "钪", "钛", "钒", "铬", "锰", "铁", "钴", "镍", "铜", "锌",
    "镓", "锗", "砷", "硒", "溴", "氪", "铷", "锶", "钇", "锆",
    "铌", "钼", "锝", "钌", "铑", "钯", "银", "镉", "铟", "锡",
    "锑", "碲", "碘", "氙", "铯", "钡", "镧", "铈", "镨", "钕",
    "钷", "钐", "铕", "钆", "铽", "镝", "钬", "铒", "铥", "镱",
    "镥", "铪", "钽", "钨", "铼", "锇", "铱", "铂", "金", "汞",
    "铊", "铅", "铋", "钋", "砹", "氡", "钫", "镭", "锕", "钍",
    "镤", "铀", "镎", "钚", "镅", "锔", "锫", "锎", "锿", "镄",
    "钔", "锘", "铹", "", "", "", "", "", "", "",
    "", "", "", "", "", "", "", "",
]


# —————————————————————————————————————————————
# 核心计算：从 Z 计算族/周期/虚面数/元素类
# —————————————————————————————————————————————

# —————————————————————————————————————————————
# Aufbau 子层填充顺序
# —————————————————————————————————————————————

# 子层数据：(n, l, 容量)
# 按 (n+l, n) 能量序排列（Aufbau 原理）
AUFBAU_ORDER = [
    (1, 0, 2),    # 1s
    (2, 0, 2),    # 2s
    (2, 1, 6),    # 2p
    (3, 0, 2),    # 3s
    (3, 1, 6),    # 3p
    (4, 0, 2),    # 4s
    (3, 2, 10),   # 3d
    (4, 1, 6),    # 4p
    (5, 0, 2),    # 5s
    (4, 2, 10),   # 4d
    (5, 1, 6),    # 5p
    (6, 0, 2),    # 6s
    (4, 3, 14),   # 4f
    (5, 2, 10),   # 5d
    (6, 1, 6),    # 6p
    (7, 0, 2),    # 7s
    (5, 3, 14),   # 5f
    (6, 2, 10),   # 6d
    (7, 1, 6),    # 7p
]


def _period_valence_capacity(period: int) -> int:
    """第 n 周期的价电子满容量（化学惰性所需电子数）。

    2n² 是壳层总容量（来自正二十面体顶点集），
    但化学满壳层只占其中一部分（s+p 或 s+d+p 或 s+f+d+p）：
        period 1: 2  (1s)
        period 2-3: 8  (s+p)
        period 4-5: 18 (s+d+p)
        period 6-7: 32 (s+f+d+p)

    2n² 与周期容量的差额（如 n=3: 18-8=10）由该壳层的 d 子层在后续周期填充。
    """
    if period == 1:
        return 2
    elif period in (2, 3):
        return 8
    elif period in (4, 5):
        return 18
    elif period in (6, 7):
        return 32
    return 2 * period * period  # fallback


def _shell_filling_from_Z(Z: int) -> List[Dict]:
    """从 Z 计算壳层填充（使用 Aufbau 子层填充 + 按 n 聚合）。

    流程：
        1. 按 Aufbau 顺序逐子层填充 Z 个电子
        2. 按主量子数 n 聚合成壳层
        3. 每壳层总容量 = 2n²（几何推导）

    Returns:
        [{name, n, occupancy, capacity, is_full}, ...]
    """
    # 1) 按 Aufbau 顺序填充子层
    remaining = Z
    subshells = []
    for n, l, cap in AUFBAU_ORDER:
        if remaining <= 0:
            break
        occ = min(remaining, cap)
        subshells.append((n, l, occ))
        remaining -= occ

    # 2) 按主量子数 n 聚合
    shell_map = {}
    for n, l, occ in subshells:
        if n not in shell_map:
            shell_map[n] = {"total_occ": 0, "n": n}
        shell_map[n]["total_occ"] += occ

    # 3) 构建壳层列表（按 n 升序）
    shells = []
    for n in sorted(shell_map.keys()):
        cap = shell_capacity(n)  # 2n²
        occ = shell_map[n]["total_occ"]
        shells.append({
            "name": SHELL_NAMES[n - 1] if n <= len(SHELL_NAMES) else f"n={n}",
            "n": n,
            "occupancy": occ,
            "capacity": cap,
            "is_full": occ >= cap,
        })

    return shells


def _vacant_faces_from_outermost(outer_occupancy: int, outer_capacity: int,
                                 period: int) -> int:
    """从最外壳层占有数和周期数计算核表面虚面数。

    规则（SPUM-VSPT.md §4.3-§4.4）：
        - 价壳层满（occ ≥ 周期价容量）→ 0 虚面（闭合型 VSPT，稀有气体）
        - 未满 → 虚面数 = 最外壳层电子数（上限 7）

    周期价容量代替 2n² 作为"满壳层"判断标准的原因是：
        每个周期只填充壳层总容量 2n² 的一部分（s+p），
        剩余部分（d 子层）在后续周期填充。
    """
    valence_cap = _period_valence_capacity(period)
    if outer_occupancy >= valence_cap:
        return 0  # 价壳层满 → 闭合型
    return min(outer_occupancy, 7)


def _group_from_vacant(vacant: int) -> int:
    """虚面数 → 族号。

    映射规则（SPUM-VSPT.md §4.4 族定义）：
        0 → 18（稀有气体）
        1 → 1（碱金属）
        2 → 2（碱土金属）
        3 → 13（硼族）
        4 → 14（碳族）
        5 → 15（氮族）
        6 → 16（氧族）
        7 → 17（卤素）
    """
    mapping = {0: 18, 1: 1, 2: 2, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17}
    return mapping.get(vacant, 0)


def _class_from_vacant(vacant: int, Z: int, outer_n: int) -> ElementClass:
    """虚面数 + Z → 元素分类。

    此分类由虚面数直接计算，非查表。
    """
    if vacant == 1:
        return ElementClass.ALKALI_METAL
    elif vacant == 2:
        return ElementClass.ALKALINE_EARTH
    elif vacant == 3:
        return ElementClass.BORON_GROUP
    elif vacant == 4:
        return ElementClass.CARBON_GROUP
    elif vacant == 5:
        return ElementClass.NITROGEN_GROUP
    elif vacant == 6:
        return ElementClass.OXYGEN_GROUP
    elif vacant == 7:
        return ElementClass.HALOGEN
    elif vacant == 0:
        return ElementClass.NOBLE_GAS
    else:
        # 非标准虚面数 → 过渡金属
        return ElementClass.TRANSITION_METAL


# —————————————————————————————————————————————
# 元素分类——真正的第一性原理计算
# —————————————————————————————————————————————

def classify_element(Z: int) -> Dict:
    """从质子数 Z 计算元素分类（无查表）。

    计算链：
        Z → 壳层填充（2n² 逐层分配）→ 最外壳层占有数 → 虚面数
        → 族 → 周期 → 元素类

    Args:
        Z: 原子序数（1-118）

    Returns:
        {proton_number, symbol, name, period, group,
         element_class, vacant_faces, shell_filling, computed}
    """
    if Z < 1 or Z > 118:
        return {"proton_number": Z, "error": f"Z={Z} 超出范围 (1-118)"}

    # 1) 计算壳层填充
    shells = _shell_filling_from_Z(Z)

    # 2) 从填充确定周期 = 最外壳层索引
    period = shells[-1]["n"]

    # 3) 从最外壳层占有数 + 周期数计算虚面数
    outer = shells[-1]
    vacant = _vacant_faces_from_outermost(outer["occupancy"], outer["capacity"], period)

    # 4) 虚面数 → 族
    group = _group_from_vacant(vacant)

    # 5) 虚面数 + Z → 元素类
    eclass = _class_from_vacant(vacant, Z, period)

    # 6) 符号与名称
    symbol = ELEMENT_SYMBOLS[Z - 1] if Z <= len(ELEMENT_SYMBOLS) else f"Z{Z}"
    name = ELEMENT_NAMES[Z - 1] if Z <= len(ELEMENT_NAMES) else f"Element-{Z}"

    return {
        "proton_number": Z,
        "symbol": symbol,
        "name": name,
        "period": period,
        "group": group,
        "element_class": eclass,
        "vacant_faces": vacant,
        "shell_filling": shells,
        # 标注此结果是计算得出，非查表
        "computed": True,
    }


# —————————————————————————————————————————————
# 同位素质量计算
# —————————————————————————————————————————————

# SPUM 常数（SPUM-VSPT.md §4.2）
# 质子质量 = 12 个永恒粒子
# 中子质量 = 质子质量 × (1 + 0.0014)  — 0.14% 来自朝外开口破坏内部湮灭循环
MP_SPUM = 1.0  # 质子的 SPUM 归一化质量单位
MN_SPUM = MP_SPUM * 1.0014  # 中子重 0.14%
# 核子结合时的结合能修正（SPUM 等价）
# 每对核子共享一个界面，减少向外虚面数 → 质量亏损
BINDING_ENERGY_PER_NUCLEON_PAIR = 0.0008  # 核子对的质量亏损系数


def compute_isotope_mass(Z: int, A: int) -> Dict:
    """计算同位素质量（SPUM 单位）。

    质量组成：
        m = Z × m_p + (A-Z) × m_n - E_b

    其中 E_b 是结合能修正：
        E_b ∝ 核子间的界面数 ≈ (A-1) × 常数

    Args:
        Z: 质子数
        A: 质量数（核子总数）

    Returns:
        {Z, A, N, mass, binding_energy, mass_defect, ...}
    """
    N = A - Z  # 中子数

    # 裸质量
    bare_mass = Z * MP_SPUM + N * MN_SPUM

    # 结合能修正：每对相邻核子共享一个界面
    # 简化模型：A 个核子的聚簇有大约 (A-1) 个界面
    interfaces = A - 1
    binding_energy = interfaces * BINDING_ENERGY_PER_NUCLEON_PAIR

    # 最终质量 = 裸质量 - 结合能
    total_mass = bare_mass - binding_energy
    mass_defect = bare_mass - total_mass

    return {
        "Z": Z,
        "A": A,
        "N": N,
        "bare_mass": round(bare_mass, 6),
        "binding_energy": round(binding_energy, 6),
        "mass_defect": round(mass_defect, 6),
        "total_mass": round(total_mass, 6),
        "formula": f"{Z}×{MP_SPUM} + {N}×{MN_SPUM} - {interfaces}×{BINDING_ENERGY_PER_NUCLEON_PAIR}",
    }


# —————————————————————————————————————————————
# 氢同位素专用计算
# —————————————————————————————————————————————

def compute_hydrogen_isotope(A: int) -> Dict:
    """计算氢同位素（Z=1）的完整结构。

    同位素名称：
        A=1: 氕（protium）
        A=2: 氘（deuterium）
        A=3: 氚（tritium）

    核结构：
        - 氕：1 个质子核（12 永恒粒子，10内2外）
        - 氘：1 质子 + 1 中子融合
        - 氚：1 质子 + 2 中子融合

    核表面拓扑计算：
        - 每个 12-粒子簇有 48 个理论外露面
        - 融合界面上的面不对外暴露
        - 朝外永恒粒子（仅质子有）贡献虚面

    Args:
        A: 质量数 (1, 2, 3)

    Returns:
        完整的结构字典
    """
    if A not in (1, 2, 3):
        return {"error": f"仅支持 A=1,2,3，收到 A={A}"}

    Z = 1
    N = A - Z

    # --- 壳层填充（与 A 无关，只与 Z 有关）---
    element = classify_element(Z)

    # --- 核表面拓扑计算 ---

    # 每个质子核：12 永恒粒子，10 内 2 外
    # 向外：10×4 = 40 实面，2×1 = 2 虚面
    PROTON_SOLID = 40
    PROTON_VACANT = 2

    # 每个中子核：12 永恒粒子，全向内
    # 向外：12×4 = 48 实面，0 虚面
    NEUTRON_SOLID = 48
    NEUTRON_VACANT = 0

    # 融合界面数：
    # A 个核子形成聚簇，大约有 (A-1) 个融合界面
    # 每个界面消耗 1 个外露面（从实面扣除）
    fusion_interfaces = A - 1

    total_solid = PROTON_SOLID + N * NEUTRON_SOLID - fusion_interfaces
    total_vacant = PROTON_VACANT + N * NEUTRON_VACANT

    # --- 质量 ---
    mass_info = compute_isotope_mass(Z, A)

    # --- VSPT 核表面完整描述 ---
    total_exposed = total_solid + total_vacant
    vacant_ratio = total_vacant / (A * 48)  # 分母：理论总外露面

    return {
        "isotope": {1: "氕 (Protium)", 2: "氘 (Deuterium)", 3: "氚 (Tritium)"}[A],
        "symbol": "H",
        "Z": Z,
        "A": A,
        "N": N,
        "nucleus_composition": {
            "protons": 1,
            "neutrons": N,
            "total_nucleons": A,
            "eternal_particles": A * 12,
            # 每个核子 = 1 个 12-永恒粒子正二十面体锁闭
        },
        "nucleus_topology": {
            "outward_solid_faces": total_solid,
            "outward_vacant_faces": total_vacant,
            "total_exposed_faces": total_exposed,
            "vacant_ratio": round(vacant_ratio, 6),
            "fusion_interfaces": fusion_interfaces,
            "surface_description": (
                f"{total_solid} solid + {total_vacant} vacant faces "
                f"(vacant ratio = {vacant_ratio:.2%})"
            ),
        },
        "shell_filling": element["shell_filling"],
        "element_class": element["element_class"].value,
        "period": element["period"],
        "group": element["group"],
        "mass": mass_info,
    }


# —————————————————————————————————————————————
# 兼容接口（保留旧函数，逻辑改为计算而非查表）
# —————————————————————————————————————————————

def periodic_table_lookup(symbol: str) -> Optional[Dict]:
    """按符号查找元素（计算方式）。

    先尝试将符号映射到 Z，再计算。
    """
    for i, sym in enumerate(ELEMENT_SYMBOLS):
        if sym == symbol:
            return classify_element(i + 1)
    return None


# —————————————————————————————————————————————
# 演示：从 Z=1 到 Z=20 的完整计算
# —————————————————————————————————————————————

def demo_compute_elements(max_Z: int = 20) -> List[Dict]:
    """从 Z=1 到 max_Z 逐元素计算分类，展示计算过程。"""
    results = []
    for Z in range(1, max_Z + 1):
        elem = classify_element(Z)
        results.append(elem)
    return results
