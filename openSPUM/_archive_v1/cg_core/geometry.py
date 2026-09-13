"""
openSPUM/cg_core/geometry.py

真正的开端：四个刚性球体在虚无中两两相切。

核心命题：
    - 球体不存储位置 — 位置由相切约束三角测量得出
    - 球体不存储度数 — 度数是约束图的计数，推导得出
    - 相切约束是第一性的，几何（位置、体积、空隙）全部派生
    - 移除实体后，约束仍在定义几何

3D 本征起点：
    3D 空间中最小的约束闭包是 4 个两两相切的球体。
    4 个球体围成一个四面体孔隙，
    索迪定理从 4 个曲率唯一确定第 5 个球体的曲率。
    4 个锚点位置 + 曲率 → 三边测量唯一确定第 5 球位置。

层级：
    [L0] 约束图本身（节点+边）— 唯一被存储的实在
    [L1] 位置/半径 — 认知投影层，只读

无位置 / 无度数 / 无外部映射 — 纯约束驱动。
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict

# ── 类型别名 ──

Vec3D = Tuple[float, float, float]


# ── 数据模型 ───────────────────────────────────────────

@dataclass
class Sphere:
    """刚性球体。

    只存储半径（体积的内在度量）。
    不存储位置 — 位置由相切约束求解得出。
    不存储度数 — 度数是约束图推导的派生属性。
    """
    radius: float


@dataclass
class TangencyConstraint:
    """两球相切约束。

    语义：distance(sphere_a, sphere_b) = r_a + r_b
    这是[L0]层面的第一性关系。
    """
    sphere_a: int
    sphere_b: int


# ── 3D 位置求解器（L1 认知投影层） ──────────────────

class PositionSolver3D:
    """3D 位置求解器。

    将约束图投影到三维坐标。
    不修改原始数据 — 只返回求解的坐标映射。

    公理起点（硬编码）：
        正四面体内嵌在 xyz 空间中，4 个顶点构成
        两两距离 = 2r（半径 r 的球体两两相切）。
    """

    # 正四面体顶点（缩放后使 |v_i - v_j| = 2）
    # 初始球体半径 = 1，所以 |v_i - v_j| = 2
    # 正四面体边长 2 → 顶点到原点距离 = √(3/2) ≈ 1.225
    _TETRA_VERTICES: List[Vec3D] = [
        (0.0, 0.0, math.sqrt(1.5)),
        (2.0 / math.sqrt(3.0), 0.0, -1.0 / math.sqrt(6.0)),
        (-1.0 / math.sqrt(3.0), 1.0, -1.0 / math.sqrt(6.0)),
        (-1.0 / math.sqrt(3.0), -1.0, -1.0 / math.sqrt(6.0)),
    ]

    @staticmethod
    def get_tetrahedron_positions(r: float) -> List[Vec3D]:
        """返回正四面体顶点坐标（用于初始 4 球体）。"""
        scale = r  # 球体半径 r → 边长 2r
        return [(v[0] * scale, v[1] * scale, v[2] * scale)
                for v in PositionSolver3D._TETRA_VERTICES]

    @staticmethod
    def solve(spheres: List[Sphere],
              constraints: List[TangencyConstraint]) -> Dict[int, Vec3D]:
        """从约束图求解所有球体的 3D 位置。

        策略：
            1. 前 4 个球体（种子）用正四面体硬编码位置
            2. 后续球体用 3D 三边测量（4 个锚点的球面交点）
        """
        positions: Dict[int, Vec3D] = {}

        # ---- 构建邻接表 ----
        adj: Dict[int, Set[int]] = {i: set() for i in range(len(spheres))}
        for c in constraints:
            adj[c.sphere_a].add(c.sphere_b)
            adj[c.sphere_b].add(c.sphere_a)

        # ---- 种子球体（前 4 个构成正四面体） ----
        if len(spheres) >= 4:
            tetra_pos = PositionSolver3D.get_tetrahedron_positions(spheres[0].radius)
            for i in range(4):
                positions[i] = tetra_pos[i]

        # ---- 按度数排序后续球体 ----
        remaining = [i for i in range(4, len(spheres))]
        remaining.sort(key=lambda i: len(adj[i]), reverse=True)

        def sphere_intersection_3d(
            p1: Vec3D, r1: float,
            p2: Vec3D, r2: float,
            p3: Vec3D, r3: float,
            p4: Vec3D, r4: float
        ) -> Optional[Vec3D]:
            """4 球交点 = 3D 三边测量。

            将 4 个球面方程相减消去二次项，
            得到 3 个线性方程，解 3×3 线性方程组。
            """
            # 以 p1 为参考点，构建线性方程组
            A = [
                [2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1]), 2 * (p2[2] - p1[2])],
                [2 * (p3[0] - p1[0]), 2 * (p3[1] - p1[1]), 2 * (p3[2] - p1[2])],
                [2 * (p4[0] - p1[0]), 2 * (p4[1] - p1[1]), 2 * (p4[2] - p1[2])],
            ]
            b = [
                r1 * r1 - r2 * r2 + p2[0] * p2[0] + p2[1] * p2[1] + p2[2] * p2[2]
                - p1[0] * p1[0] - p1[1] * p1[1] - p1[2] * p1[2],
                r1 * r1 - r3 * r3 + p3[0] * p3[0] + p3[1] * p3[1] + p3[2] * p3[2]
                - p1[0] * p1[0] - p1[1] * p1[1] - p1[2] * p1[2],
                r1 * r1 - r4 * r4 + p4[0] * p4[0] + p4[1] * p4[1] + p4[2] * p4[2]
                - p1[0] * p1[0] - p1[1] * p1[1] - p1[2] * p1[2],
            ]

            # 高斯消元解 3×3
            det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                   - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                   + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
            if abs(det) < 1e-15:
                return None

            inv_det = 1.0 / det
            # 克莱姆法则
            def cramer(col: int) -> float:
                M = [row[:] for row in A]
                for row in range(3):
                    M[row][col] = b[row]
                return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                        - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                        + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])) * inv_det

            return (cramer(0), cramer(1), cramer(2))

        # ---- 迭代求解 ----
        for _ in range(50):
            if not remaining:
                break
            newly = []
            for idx in remaining:
                neighbors = [n for n in adj[idx] if n in positions]
                if len(neighbors) >= 4:
                    # 用前 4 个已求解的邻居做三边测量
                    ns = neighbors[:4]
                    r_self = spheres[idx].radius
                    r_ns = [spheres[n].radius + r_self for n in ns]
                    p_ns = [positions[n] for n in ns]
                    pos = sphere_intersection_3d(
                        p_ns[0], r_ns[0],
                        p_ns[1], r_ns[1],
                        p_ns[2], r_ns[2],
                        p_ns[3], r_ns[3],
                    )
                    if pos is not None:
                        # 验证：与所有 4 个邻居的距离应等于 r+r_n
                        valid = True
                        for k in range(4):
                            d = math.sqrt(sum((pos[i] - p_ns[k][i])**2 for i in range(3)))
                            if abs(d - r_ns[k]) > 1e-6:
                                valid = False
                                break
                        if valid:
                            positions[idx] = pos
                            newly.append(idx)
            for idx in newly:
                remaining.remove(idx)
            if not newly:
                break

        return positions


# ── 索迪定理（3D 笛卡尔定理） ──────────────────────

def soddy_bend_3d(b1: float, b2: float, b3: float, b4: float
                  ) -> Tuple[float, float]:
    """索迪定理：给定四球曲率，求第五球曲率。

    公式：
        S = Σk_{1..4}
        Q = Σk²_{1..4}
        sqrt_term = √(3S² - 6Q)
        k₅ = (S ± sqrt_term) / 2

    Returns:
        (b5_inner, b5_outer)
        inner = 四球围成的缝隙球的曲率（正值）
        outer = 从外部包围四球的大球曲率（可能负值）
    """
    S = b1 + b2 + b3 + b4
    Q = b1 * b1 + b2 * b2 + b3 * b3 + b4 * b4
    disc = 3.0 * S * S - 6.0 * Q
    if disc < -1e-12:
        return (float('nan'), float('nan'))
    disc = max(0.0, disc)
    sqrt_term = math.sqrt(disc)

    b5_inner = (S + sqrt_term) / 2.0
    b5_outer = (S - sqrt_term) / 2.0
    return (b5_inner, b5_outer)


# ── 约束图引擎 ────────────────────────────────────────

class ApollonianGraph:
    """
    约束图：所有球体之间的相切关系。

    几何（位置、距离、空隙）完全由约束图导出。
    这是[L0]层面的唯一实在。
    """

    def __init__(self):
        self.spheres: List[Sphere] = []
        self.constraints: List[TangencyConstraint] = []

    # ── 初始条件 ──

    def add_initial_triple(self, r_a: float, r_b: float, r_c: float):
        """三个刚性球体两两相切（2D 简化版）。

        仅用于 2D 演示。真正的 3D 起点用 add_initial_quad。
        """
        self.spheres = [
            Sphere(r_a),
            Sphere(r_b),
            Sphere(r_c),
        ]
        self.constraints = [
            TangencyConstraint(0, 1),
            TangencyConstraint(1, 2),
            TangencyConstraint(2, 0),
        ]

    def add_initial_quad(self, r_a: float, r_b: float, r_c: float, r_d: float):
        """四个刚性球体在虚无中两两相切（3D 本征起点）。

        4 个球体围成一个四面体孔隙。
        这是 3D 阿波罗尼奥斯填充的最小种子：
            4 个锚点的曲率 → 索迪定理 → 第 5 球曲率
            4 个锚点的位置 + 第 5 球半径 → 三边测量 → 第 5 球位置
        """
        self.spheres = [
            Sphere(r_a),
            Sphere(r_b),
            Sphere(r_c),
            Sphere(r_d),
        ]
        self.constraints = [
            TangencyConstraint(0, 1),
            TangencyConstraint(0, 2),
            TangencyConstraint(0, 3),
            TangencyConstraint(1, 2),
            TangencyConstraint(1, 3),
            TangencyConstraint(2, 3),
        ]

    # ── 邻接查询 ──

    def _build_adjacency(self) -> Dict[int, Set[int]]:
        """从约束列表构建邻接表。"""
        adj: Dict[int, Set[int]] = {i: set() for i in range(len(self.spheres))}
        for c in self.constraints:
            adj[c.sphere_a].add(c.sphere_b)
            adj[c.sphere_b].add(c.sphere_a)
        return adj

    def _are_mutually_tangent(self, *indices: int) -> bool:
        """检查一组球体是否两两相切（通过约束图）。"""
        adj = self._build_adjacency()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if indices[j] not in adj.get(indices[i], set()):
                    return False
        return True

    def get_degree(self, idx: int) -> int:
        """球体的度数 = 邻居数（从约束图推导）。"""
        adj = self._build_adjacency()
        return len(adj.get(idx, set()))

    # ── 缝隙检测（3D：4-clique → 索迪定理） ──

    def find_largest_gap_3d(self) -> Optional[Tuple[float, List[int]]]:
        """
        枚举所有四球相切的四面体孔隙，
        用索迪定理计算空隙球体半径，
        返回最大的空隙。

        索迪定理：
            k₅ = (S + √(3S² - 6Q)) / 2
            其中 S = Σk_{1..4}, Q = Σk²_{1..4}
            正号对应缝隙内切球。
        """
        best_radius = -1.0
        best_quad = None
        n = len(self.spheres)

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if not self._are_mutually_tangent(i, j, k, l):
                            continue

                        bends = [1.0 / self.spheres[idx].radius
                                 for idx in (i, j, k, l)]
                        b_inner, _ = soddy_bend_3d(*bends)
                        if math.isnan(b_inner) or b_inner <= 1e-12:
                            continue
                        r_gap = 1.0 / b_inner
                        if r_gap > best_radius:
                            best_radius = r_gap
                            best_quad = (i, j, k, l)

        if best_quad is None:
            return None
        return best_radius, list(best_quad)

    def find_all_gaps_3d(self) -> List[Tuple[float, List[int]]]:
        """枚举所有可填充的 3D 缝隙，按半径降序排列。"""
        gaps = []
        n = len(self.spheres)

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if not self._are_mutually_tangent(i, j, k, l):
                            continue

                        bends = [1.0 / self.spheres[idx].radius
                                 for idx in (i, j, k, l)]
                        b_inner, _ = soddy_bend_3d(*bends)
                        if math.isnan(b_inner) or b_inner <= 1e-12:
                            continue
                        r_gap = 1.0 / b_inner
                        gaps.append((r_gap, [i, j, k, l]))

        gaps.sort(key=lambda x: x[0], reverse=True)
        return gaps

    # ── 2D 缝隙检测（保留用于兼容） ──

    def find_largest_gap(self) -> Optional[Tuple[float, List[int]]]:
        """2D 缝隙检测（笛卡尔圆定理）。"""
        best_radius = -1.0
        best_triple = None
        n = len(self.spheres)

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if not self._are_mutually_tangent(i, j, k):
                        continue
                    k1 = 1.0 / self.spheres[i].radius
                    k2 = 1.0 / self.spheres[j].radius
                    k3 = 1.0 / self.spheres[k].radius
                    sum_k = k1 + k2 + k3
                    prod_term = k1 * k2 + k2 * k3 + k3 * k1
                    if prod_term <= 0:
                        continue
                    k_gap = sum_k + 2.0 * math.sqrt(prod_term)
                    if k_gap <= 1e-12:
                        continue
                    r_gap = 1.0 / k_gap
                    if r_gap > best_radius:
                        best_radius = r_gap
                        best_triple = (i, j, k)

        if best_triple is None:
            return None
        return best_radius, list(best_triple)

    def find_all_gaps(self) -> List[Tuple[float, List[int]]]:
        """2D 缝隙枚举。"""
        gaps = []
        n = len(self.spheres)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if not self._are_mutually_tangent(i, j, k):
                        continue
                    k1 = 1.0 / self.spheres[i].radius
                    k2 = 1.0 / self.spheres[j].radius
                    k3 = 1.0 / self.spheres[k].radius
                    sum_k = k1 + k2 + k3
                    prod_term = k1 * k2 + k2 * k3 + k3 * k1
                    if prod_term <= 0:
                        continue
                    k_gap = sum_k + 2.0 * math.sqrt(prod_term)
                    if k_gap <= 1e-12:
                        continue
                    r_gap = 1.0 / k_gap
                    gaps.append((r_gap, [i, j, k]))
        gaps.sort(key=lambda x: x[0], reverse=True)
        return gaps

    # ── 缝隙填充 ──

    def fill_gap(self, anchors: List[int], r_gap: float):
        """
        在空隙中填入一个新的刚性球体。

        位置不由我们指定 — 由三边测量从锚点球体唯一确定。

        参数:
            anchors: 锚点球体的索引（3 个用于 2D，4 个用于 3D）
            r_gap: 新球体的半径（由笛卡尔/索迪定理确定）
        """
        new_idx = len(self.spheres)
        self.spheres.append(Sphere(r_gap))

        for anchor in anchors:
            self.constraints.append(TangencyConstraint(new_idx, anchor))

    # ── 演化 ──

    def step(self, max_fills: int = 1, use_3d: bool = True) -> int:
        """执行一步演化：填充最大的 max_fills 个空隙。

        返回本步填充的空隙数。
        """
        if use_3d:
            gaps = self.find_all_gaps_3d()
        else:
            gaps = self.find_all_gaps()
        if not gaps:
            return 0

        filled = 0
        for r_gap, anchors in gaps:
            if filled >= max_fills:
                break
            self.fill_gap(anchors, r_gap)
            filled += 1

        return filled

    # ── 查询 ──

    def solve_positions(self) -> Dict[int, Vec3D]:
        """求解所有球体的 3D 位置。

        这是[L1]认知投影层 — 不修改[L0]约束图。
        位置是派生属性，不是存储属性。
        """
        return PositionSolver3D.solve(self.spheres, self.constraints)

    def summary(self) -> Dict:
        """返回当前状态摘要。"""
        adj = self._build_adjacency()
        degs = [len(adj[i]) for i in range(len(self.spheres))]
        positions = self.solve_positions()
        return {
            "n_spheres": len(self.spheres),
            "n_constraints": len(self.constraints),
            "min_degree": min(degs) if degs else 0,
            "max_degree": max(degs) if degs else 0,
            "avg_degree": sum(degs) / len(degs) if degs else 0.0,
            "n_positions_solved": len(positions),
            "min_radius": min(s.radius for s in self.spheres),
            "max_radius": max(s.radius for s in self.spheres),
        }

    def cavity_query(self, idx: int) -> Optional[Tuple[Vec3D, float]]:
        """空洞查询：查询球体的几何，即使已被移除。

        当前：球体仍在图中时查询其位置和半径。
        未来扩展：移除球体后，从邻居约束重新求解。
        """
        if idx < 0 or idx >= len(self.spheres):
            return None
        radius = self.spheres[idx].radius
        positions = self.solve_positions()
        if idx in positions:
            return (positions[idx], radius)
        return None
