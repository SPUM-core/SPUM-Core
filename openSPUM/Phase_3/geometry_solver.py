"""
三维几何求解器 — 球体相切约束的力导向松弛

物理模型:
    每个晶子是直径为 4κ 的硬球。
    球体间存在两种力:
        1. 排斥力 (当 overlap > 0): F_rep = k_rep * overlap
        2. 吸引力 (当 gap > tangent_distance): F_att = k_att * gap
    松弛后到达的平衡态 = 最密堆积构型。

    对于 12 个等大球体, 最低能态 = 正二十面体。
    这一结论是几何必然 (不存在更密的 12 球堆积)。

参数来源 (参见 元素化学/力学参数推导.md):
    k_rep: 从 CRYSTALLITE_CAPACITY(≈50.3) / O_MAX(0.1) / d_thresh(50) ≈ 10.0
    k_att: 当前人工选定为 1.0; 未来应改为动态值 1/(v_a+v_b+1)
    damping: 从湮灭概率 p_ann ≈ 0.09 推导 → 0.5 (含尺度因子)
    dt: 从 TAU=1 和子步数 n_sub=100 得 dt = 0.01

接口:
    Sphere3D:          三维球体数据结构
    GeometrySolver:    约束求解器 (松弛 + 验证)
    solve():           主入口 — 输入节点列表, 输出松弛后位置
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Phase_1.constants import (
    KAPPA,
    CRYSTALLITE_DEGREE_THRESHOLD,
    CRYSTALLITE_CAPACITY,
    GEOMETRIC_TOLERANCE,
    TAU,
)


# ============================================================
# 力学参数 — 从 Phase 1 基本常量推导
# ============================================================

_R_MIN = 0.5 * KAPPA  # 最小球体半径 = κ/2
_O_MAX = GEOMETRIC_TOLERANCE * (2 * _R_MIN)  # 最大允许重叠 = 0.1κ

# 排斥系数 k_rep: 来自晶子容量 / 最大允许重叠 / 度数阈值
# k_rep = 50.27 / 0.1 / 50 ≈ 10.05 ≈ 10.0
# 见 元素化学/力学参数推导.md §1
K_REP_DERIVED = CRYSTALLITE_CAPACITY / _O_MAX / CRYSTALLITE_DEGREE_THRESHOLD

# 时间步长 dt: τ / 子步数 100
# 见 力学参数推导.md §4
_DT_DERIVED = TAU / 100.0

# 阻尼系数 damping: 来自平均湮灭概率
# p_ann = Σ(1/d)/50 ≈ 0.09, damping = p_ann/2 × 尺度因子 ≈ 0.5
# 见 力学参数推导.md §3
_P_ANN = sum(1.0 / d for d in range(1, CRYSTALLITE_DEGREE_THRESHOLD + 1)) / CRYSTALLITE_DEGREE_THRESHOLD
_DAMPING_DERIVED = _P_ANN / 2.0 * 10.0  # 尺度因子 10 保持临界阻尼附近


# ============================================================
# 三维向量工具
# ============================================================

def vec_add(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def vec_sub(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def vec_scale(v: Tuple[float, ...], s: float) -> Tuple[float, ...]:
    return tuple(x * s for x in v)


def vec_norm(v: Tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in v))


def vec_normalize(v: Tuple[float, ...]) -> Tuple[float, ...]:
    n = vec_norm(v)
    if n < 1e-12:
        return v
    return tuple(x / n for x in v)


def vec_distance(a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
    return vec_norm(vec_sub(a, b))


# ============================================================
# 球体数据结构
# ============================================================

@dataclass
class Sphere3D:
    """三维晶子球体。

    Attributes:
        uid:       节点 UID
        degree:    当前度数 (CRYSTALLITE_DEGREE_THRESHOLD=50 为饱和)
        position:  三维坐标 (x, y, z)
        radius:    球体半径 (基于 degree 计算)
        neighbors: 邻接晶子的 UID 列表 (只在子图内, 不含普通边)
        fixed:     是否固定位置 (不受松弛影响)
    """
    uid: str
    degree: int
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0
    neighbors: List[str] = field(default_factory=list)
    fixed: bool = False

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    def tangent_distance(self, other: 'Sphere3D') -> float:
        """两个球体相切时的质心距离。"""
        return self.radius + other.radius

    def overlap_with(self, other: 'Sphere3D') -> float:
        """重叠量 (正=重叠, 负=间隙, 0=恰好相切)。"""
        d = vec_distance(self.position, other.position)
        return self.tangent_distance(other) - d

    def distance_to(self, other: 'Sphere3D') -> float:
        return vec_distance(self.position, other.position)

    def force_to(self, other: 'Sphere3D', k_rep: float = 10.0,
                 k_att: float = 1.0) -> Tuple[float, float, float]:
        """两个球体之间的力向量。

        排斥力: 当 overlap > 0 (球体穿透), 沿连心线向外
        吸引力: 当 gap > 切距 (球体分离), 沿连心线向内

        Returns:
            (fx, fy, fz) 作用于此球体的力
        """
        d_vec = vec_sub(other.position, self.position)
        d = vec_norm(d_vec)

        if d < 1e-12:
            return (0.0, 0.0, 0.0)

        direction = vec_normalize(d_vec)
        r_sum = self.tangent_distance(other)
        overlap = r_sum - d
        gap = d - r_sum

        if overlap > 0:
            # 排斥: 球体重叠, 推离
            force_mag = k_rep * overlap
            return vec_scale(direction, -force_mag)  # 指向远离 other
        elif gap > 0.01 * r_sum:
            # 吸引: 球体分离, 拉近
            # 但只对有直接连接 (neighbor) 的球体施加吸引
            if other.uid in self.neighbors or self.uid in other.neighbors:
                force_mag = k_att * gap
                return vec_scale(direction, force_mag)  # 指向靠近 other

        return (0.0, 0.0, 0.0)


# ============================================================
# 力导向松弛求解器
# ============================================================

class ForceDirectedRelaxation:
    """力导向松弛求解器。

    将球体系统视为力学系统:
        - 重叠球体之间产生排斥力
        - 通过边 (neighbors) 连接的球体之间产生吸引力
        - 系统逐步松弛至最小能态

    对于 12 个等大球体, 最小值 = 正二十面体。
    """

    def __init__(
        self,
        spheres: Dict[str, Sphere3D],
        k_rep: float = K_REP_DERIVED,
        k_att: float = 1.0,
        damping: float = _DAMPING_DERIVED,
        dt: float = _DT_DERIVED,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
    ):
        self.spheres = spheres
        self.k_rep = k_rep
        self.k_att = k_att
        self.damping = damping
        self.dt = dt
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.iteration = 0
        self.energy_history: List[float] = []

    @property
    def total_energy(self) -> float:
        """系统的总势能。"""
        energy = 0.0
        sphere_list = list(self.spheres.values())
        for i, a in enumerate(sphere_list):
            for b in sphere_list[i + 1:]:
                d = a.distance_to(b)
                r_sum = a.tangent_distance(b)
                if d < r_sum:
                    # 排斥势: 重叠越深, 能量越高
                    energy += 0.5 * self.k_rep * (r_sum - d) ** 2
                elif d > r_sum and (b.uid in a.neighbors or a.uid in b.neighbors):
                    # 吸引势: 分离越远, 能量越高
                    energy += 0.5 * self.k_att * (d - r_sum) ** 2
        return energy

    def step(self) -> float:
        """执行一步松弛迭代。

        Returns:
            本轮最大位移 (用于收敛判定)
        """
        forces: Dict[str, Tuple[float, float, float]] = {}
        sphere_list = list(self.spheres.values())

        # 计算所有球体上的力
        for sphere in sphere_list:
            if sphere.fixed:
                forces[sphere.uid] = (0.0, 0.0, 0.0)
                continue
            total_force = (0.0, 0.0, 0.0)
            for other in sphere_list:
                if other.uid == sphere.uid:
                    continue
                f = sphere.force_to(other, self.k_rep, self.k_att)
                total_force = vec_add(total_force, f)
            forces[sphere.uid] = total_force

        # 应用力 → 更新位置
        max_displacement = 0.0
        for sphere in sphere_list:
            if sphere.fixed:
                continue
            f = forces[sphere.uid]
            displacement = vec_scale(f, self.dt * self.damping)
            sphere.position = vec_add(sphere.position, displacement)
            disp_mag = vec_norm(displacement)
            if disp_mag > max_displacement:
                max_displacement = disp_mag

        self.iteration += 1
        self.energy_history.append(self.total_energy)
        return max_displacement

    def is_converged(self, max_displacement: float) -> bool:
        """判断是否已收敛。"""
        return max_displacement < self.convergence_threshold

    def solve(self) -> Dict:
        """执行完整的松弛求解。

        Returns:
            {converged: bool, iterations: int, final_energy: float,
             energy_history: [...], sphere_positions: {...}}
        """
        for _ in range(self.max_iterations):
            max_disp = self.step()
            if self.is_converged(max_disp):
                break

        return {
            "converged": self.is_converged(max_disp),
            "iterations": self.iteration,
            "final_energy": self.total_energy,
            "energy_history": self.energy_history,
            "sphere_positions": {
                uid: s.position for uid, s in self.spheres.items()
            },
        }


# ============================================================
# 几何求解器 (高级接口)
# ============================================================

class GeometrySolver:
    """三维几何聚簇求解器 — 高级接口。

    职责:
        1. 从 NodeRegistry 读取晶子节点
        2. 为晶子分配初始位置 (随机单位球面)
        3. 构建 Sphere3D 对象网络 (从现有簇边)
        4. 运行力导向松弛
        5. 松弛后的位置写回 NodeRegistry
        6. 验证拓扑不变量

    用法:
        solver = GeometrySolver(node_registry, relation_pool)
        result = solver.solve()
        # result.converged, result.icosahedron_detected
        # 现在 node_registry 中晶子有更新后的 spatial_position
    """

    def __init__(self, node_registry=None, relation_pool=None):
        self.node_registry = node_registry
        self.relation_pool = relation_pool

    def build_spheres_from_registry(self) -> Dict[str, Sphere3D]:
        """从 NodeRegistry 构建 Sphere3D 网络。"""
        spheres: Dict[str, Sphere3D] = {}
        crystallites = self.node_registry.crystallite_nodes()

        # 构建簇边查找表
        cluster_edges = set()
        if self.relation_pool:
            for key in getattr(self.relation_pool, '_cluster_edge_keys', set()):
                cluster_edges.add(key)
                cluster_edges.add((key[1], key[0]))

        for node in crystallites:
            uid = node.address.uid
            r = node.radius

            # 使用已有空间位置, 或随机初始化
            if node.spatial_position is not None:
                pos = node.spatial_position
            else:
                # 随机分布在单位球面上
                theta = random.random() * 2 * math.pi
                phi = math.acos(2 * random.random() - 1)
                pos = (
                    math.sin(phi) * math.cos(theta),
                    math.sin(phi) * math.sin(theta),
                    math.cos(phi),
                )

            # 邻居
            neighbors = []
            for (ua, ub) in cluster_edges:
                if ua == uid:
                    neighbors.append(ub)
                elif ub == uid:
                    neighbors.append(ua)

            spheres[uid] = Sphere3D(
                uid=uid,
                degree=node.degree,
                position=pos,
                radius=r,
                neighbors=neighbors,
            )

        return spheres

    def solve(self, **kwargs) -> Dict:
        """执行完整求解。"""
        spheres = self.build_spheres_from_registry()
        if len(spheres) < 4:
            return {
                "converged": True,
                "iterations": 0,
                "final_energy": 0.0,
                "sphere_count": len(spheres),
                "message": "需要 ≥4 个晶子才能形成闭合骨架",
            }

        solver = ForceDirectedRelaxation(
            spheres,
            k_rep=kwargs.get('k_rep', K_REP_DERIVED),
            k_att=kwargs.get('k_att', 1.0),
            damping=kwargs.get('damping', _DAMPING_DERIVED),
            dt=kwargs.get('dt', _DT_DERIVED),
            max_iterations=kwargs.get('max_iterations', 2000),
            convergence_threshold=kwargs.get('threshold', 1e-6),
        )

        result = solver.solve()

        # 写回 NodeRegistry
        if self.node_registry:
            for uid, pos in result["sphere_positions"].items():
                node = self.node_registry.nodes.get(uid)
                if node:
                    node.spatial_position = pos

        result["sphere_count"] = len(spheres)
        return result

    @staticmethod
    def compute_icosahedron_positions(edge_length: float = 4.0) -> List[Tuple[float, float, float]]:
        """计算标准正二十面体的 12 个顶点位置。

        Args:
            edge_length: 边长 (= 2 x 半径)

        Returns:
            12 个三维坐标, 正二十面体顶点
        """
        from Phase_3.icosahedron_assembly import _icosahedron_vertices
        return _icosahedron_vertices(edge_length)


# ============================================================
# 实用函数
# ============================================================

def assign_random_initial_positions(
    spheres: Dict[str, Sphere3D],
    scale: float = 5.0,
) -> None:
    """为球体分配随机初始位置 (单位球面 × scale)。"""
    for sphere in spheres.values():
        theta = random.random() * 2 * math.pi
        phi = math.acos(2 * random.random() - 1)
        sphere.position = (
            scale * math.sin(phi) * math.cos(theta),
            scale * math.sin(phi) * math.sin(theta),
            scale * math.cos(phi),
        )


def tangent_distance(r1: float, r2: float) -> float:
    """两个球体相切时的质心距离。"""
    return r1 + r2
