"""
openSPUM/cg_core/growth.py

openSPUM 增长游戏引擎。

规则：
    1. 第 1 个球体：体积 V = 1，无邻居
    2. 第 n 个球体（n ≥ 2）：自动与 min(n-1, 3) 个已有球体相切
       — 选择度数最高的球体作为"锚点"
    3. 每个球体的体积 V = degree + 1
    4. 半径 r = (3V/4π)^(1/3) — 球体体积公式
    5. 每次添加后，重新求解所有位置

本质：
    度数不是外部映射来的指标——
    度数就是邻居数，邻居数定义体积，体积定义空间尺度。
    这是一个纯拓扑生长过程，几何是认知投影。

游戏感：
    · 每步加入一个小球，它自动"咔哒"吸附到最拥挤的区域
    · 被吸附的球体同步膨胀，整个结构随之舒展
    · 观察：度分布如何演化？几何如何自组织？
"""

import math
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass

# ── 数据模型 ──

@dataclass
class Sphere:
    idx: int
    radius: float       # 从体积推导
    volume: int         # = deg + 1
    position: Optional[Tuple[float, float, float]] = None


@dataclass
class Tangency:
    a: int
    b: int


class GrowthEngine:
    """增长游戏引擎。"""

    def __init__(self):
        self.spheres: List[Sphere] = []       # 所有球体
        self.tangencies: List[Tangency] = []   # 相切约束

    # ── 体积/半径转换 ──

    @staticmethod
    def _radius(volume: int) -> float:
        """从体积计算半径。"""
        return ((3.0 * volume) / (4.0 * math.pi)) ** (1.0 / 3.0)

    # ── 度数查询 ──

    def _degree(self, idx: int) -> int:
        return sum(1 for t in self.tangencies if t.a == idx or t.b == idx)

    def _degree_distribution(self) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for i in range(len(self.spheres)):
            d = self._degree(i)
            dist[d] = dist.get(d, 0) + 1
        return dist

    # ── 位置求解器 ──

    @staticmethod
    def _trilateration_3d(
        p1: Tuple[float, float, float], d1: float,
        p2: Tuple[float, float, float], d2: float,
        p3: Tuple[float, float, float], d3: float,
    ) -> Optional[Tuple[Tuple[float, float, float],
                         Tuple[float, float, float]]]:
        """3D 三边测量：从 3 个锚点求未知点的两个候选位置。

        使用 Gram-Schmidt 正交化建立局部坐标系：
            ex: p1 → p2 方向
            ey: p1p3 在垂直于 ex 方向上的分量
            ez: ex × ey

        返回两个解（在 ez 方向对称），调用者选择需要的那个。
        """
        # ex: 从 p1 指向 p2 的单位向量
        ex = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        ex_len = math.sqrt(ex[0]**2 + ex[1]**2 + ex[2]**2)
        if ex_len < 1e-15:
            return None
        ex = (ex[0] / ex_len, ex[1] / ex_len, ex[2] / ex_len)

        # p1p3 向量
        p3v = (p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2])

        # ey: p3v 减去其在 ex 上的投影
        dot_ex_p3 = ex[0] * p3v[0] + ex[1] * p3v[1] + ex[2] * p3v[2]
        ey = (p3v[0] - dot_ex_p3 * ex[0],
              p3v[1] - dot_ex_p3 * ex[1],
              p3v[2] - dot_ex_p3 * ex[2])
        ey_len = math.sqrt(ey[0]**2 + ey[1]**2 + ey[2]**2)
        if ey_len < 1e-15:
            return None
        ey = (ey[0] / ey_len, ey[1] / ey_len, ey[2] / ey_len)

        # ez: ex × ey
        ez = (ex[1] * ey[2] - ex[2] * ey[1],
              ex[2] * ey[0] - ex[0] * ey[2],
              ex[0] * ey[1] - ex[1] * ey[0])

        # x 坐标：在 ex 方向
        x = (d1*d1 - d2*d2 + ex_len*ex_len) / (2.0 * ex_len)

        # y 坐标：在 ey 方向
        y = (d1*d1 - d3*d3 + dot_ex_p3*dot_ex_p3 + ey_len*ey_len
             - 2.0 * dot_ex_p3 * x) / (2.0 * ey_len)

        # z 坐标：在 ez 方向（两个解 ±z）
        z_sq = d1*d1 - x*x - y*y
        if z_sq < -1e-12:
            return None
        z = math.sqrt(max(0.0, z_sq))

        pos_a = (p1[0] + x*ex[0] + y*ey[0] + z*ez[0],
                 p1[1] + x*ex[1] + y*ey[1] + z*ez[1],
                 p1[2] + x*ex[2] + y*ey[2] + z*ez[2])
        pos_b = (p1[0] + x*ex[0] + y*ey[0] - z*ez[0],
                 p1[1] + x*ex[1] + y*ey[1] - z*ez[1],
                 p1[2] + x*ex[2] + y*ey[2] - z*ez[2])
        return (pos_a, pos_b)

    def _solve_positions(self) -> bool:
        """从相切约束求解所有球体的 3D 位置。

        策略：
            - 球 0: 原点
            - 球 1: x 轴上 (0, r0+r1, 0) 
            - 球 2: 三边测量（球 0, 1 做锚点，z=0 平面）
            - 球 3+: 3D 三边测量（3 个已定位邻居做锚点，选外部解）
        """
        n = len(self.spheres)
        if n == 0:
            return True

        positions: Dict[int, Tuple[float, float, float]] = {}
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for t in self.tangencies:
            adj[t.a].add(t.b)
            adj[t.b].add(t.a)

        order = sorted(range(n), key=lambda i: self._degree(i), reverse=True)

        for idx in order:
            if idx in positions:
                continue
            neighbors = [nb for nb in adj[idx] if nb in positions]

            if not positions:
                positions[idx] = (0.0, 0.0, 0.0)

            elif len(positions) == 1:
                # 第二个球：在 x 轴上
                nb = neighbors[0]
                d = self.spheres[idx].radius + self.spheres[nb].radius
                positions[idx] = (d, 0.0, 0.0)

            elif len(positions) == 2:
                # 第三个球：在 xy 平面内用圆-圆交点
                nb0, nb1 = neighbors[0], neighbors[1]
                p0 = positions[nb0]
                p1 = positions[nb1]
                d0 = self.spheres[idx].radius + self.spheres[nb0].radius
                d1 = self.spheres[idx].radius + self.spheres[nb1].radius
                base = p1[0] - p0[0]  # p1 - p0 in x direction
                # 余弦定理
                x = (d0*d0 - d1*d1 + base*base) / (2.0 * base)
                y_sq = d0*d0 - x*x
                y = math.sqrt(max(0.0, y_sq))
                positions[idx] = (x, y, 0.0)

            else:
                # 第 4+ 个球：用 3 个已定位邻居做 3D 三边测量
                r_self = self.spheres[idx].radius
                if len(neighbors) >= 3:
                    anchors = neighbors[:3]
                    p_anchors = [positions[nb] for nb in anchors]
                    d_anchors = [r_self + self.spheres[nb].radius for nb in anchors]

                    candidates = self._trilateration_3d(
                        p_anchors[0], d_anchors[0],
                        p_anchors[1], d_anchors[1],
                        p_anchors[2], d_anchors[2],
                    )
                    if candidates is not None:
                        pos_a, pos_b = candidates
                        # 选距离结构中心更远的那个（外部生长）
                        cx = (sum(p[0] for p in positions.values())
                              / len(positions))
                        cy = (sum(p[1] for p in positions.values())
                              / len(positions))
                        cz = (sum(p[2] for p in positions.values())
                              / len(positions))
                        da = ((pos_a[0] - cx)**2 + (pos_a[1] - cy)**2
                              + (pos_a[2] - cz)**2)
                        db = ((pos_b[0] - cx)**2 + (pos_b[1] - cy)**2
                              + (pos_b[2] - cz)**2)
                        positions[idx] = pos_a if da > db else pos_b
                elif len(neighbors) >= 1:
                    # 退化解：沿第一个邻居方向
                    nb = neighbors[0]
                    d = r_self + self.spheres[nb].radius
                    p = positions[nb]
                    positions[idx] = (p[0] + d, p[1], p[2])

        for idx, pos in positions.items():
            self.spheres[idx].position = pos

        return len(positions) == n

    # ── 核心游戏逻辑 ──

    def add_one(self) -> int:
        """添加一个球体。返回新球体的索引。"""
        n = len(self.spheres)

        if n == 0:
            # 第 1 个球：V=1
            v = 1
            sphere = Sphere(idx=0, radius=self._radius(v), volume=v)
            self.spheres.append(sphere)
            self._solve_positions()
            return 0

        # 选择锚点：度数最高的 min(n, 3) 个球体
        # 按度数降序排列，同度数则索引小的优先（最早球体形成核心）
        degrees = sorted([(self._degree(i), i) for i in range(n)],
                         key=lambda x: (-x[0], x[1]))
        n_anchors = min(n, 3)
        anchors = [degrees[k][1] for k in range(n_anchors)]

        # 新球体的体积 = 锚点数 + 0 = 锚点数
        # （新球体初始只有锚点作为邻居）
        v_new = n_anchors
        new_idx = n
        new_sphere = Sphere(idx=new_idx, radius=self._radius(v_new), volume=v_new)
        self.spheres.append(new_sphere)

        # 添加相切约束
        for anchor in anchors:
            self.tangencies.append(Tangency(new_idx, anchor))

        # ---- 更新所有球体的体积 ----
        # 每个被影响的球体体积 = deg + 1
        self._refresh_volumes()

        # 重新求解位置
        self._solve_positions()

        return new_idx

    def _refresh_volumes(self):
        """重新计算所有球体的体积和半径。"""
        for i in range(len(self.spheres)):
            deg = self._degree(i)
            v = deg + 1
            self.spheres[i].volume = v
            self.spheres[i].radius = self._radius(v)

    # ── 观察 ──

    def summary(self) -> Dict:
        """当前状态摘要。"""
        n = len(self.spheres)
        degrees = [self._degree(i) for i in range(n)]
        volumes = [s.volume for s in self.spheres]
        radii = [s.radius for s in self.spheres]
        solved = sum(1 for s in self.spheres if s.position is not None)

        return {
            "n": n,
            "tangencies": len(self.tangencies),
            "degrees": degrees,
            "volumes": volumes,
            "radii": [f"{r:.4f}" for r in radii],
            "min_r": min(radii),
            "max_r": max(radii),
            "solved": solved,
            "degree_dist": self._degree_distribution(),
        }

    def print_state(self):
        """打印当前状态。"""
        s = self.summary()
        print(f"  [{s['n']:2d}] 球体 n={s['n']}, 约束={s['tangencies']}, "
              f"已求解={s['solved']}")
        print(f"       度数分布: {dict(sorted(s['degree_dist'].items()))}")
        print(f"       体积范围: [{min(s['volumes'])}, {max(s['volumes'])}], "
              f"半径范围: [{s['min_r']:.4f}, {s['max_r']:.4f}]")
        for i in range(s['n']):
            d = s['degrees'][i]
            print(f"       球{i}: deg={d}, V={s['volumes'][i]}, r={s['radii'][i]}", end="")
            pos = self.spheres[i].position
            if pos:
                print(f", ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})", end="")
            print()
