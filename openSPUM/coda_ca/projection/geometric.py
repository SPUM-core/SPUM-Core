"""
几何投影层 — 将 CA 的度数信息映射为可观测的几何形态。

核心原则：
    - 投影层是**只读的**——它不参与演化
    - 投影层不存储状态——每次读取时从度数实时计算
    - 不同投影器可以共存（三维坐标、球谐、光谱……）
    - 投影是可逆的（给定 L0 度数，L1 投影唯一确定）

投影规则：
    - 半径 r(deg) = κ / (κ + deg)  — 度数越大体积越大
    - 坐标由力导向松弛从邻居关系计算
    - 球体形状 = 等度数节点的无偏等向投影

使用方式：
    projector = GeometricProjector(sim.degrees, sim.neighbors)
    pos = projector.compute_positions()  # 力导向松弛
    radii = projector.compute_radii()    # 度数→半径
"""

from typing import List, Optional, Tuple
import math


class GeometricProjector:
    """几何投影器 — 度数→三维几何形态的只读映射。"""

    def __init__(self, degrees: List[int], neighbors: List[List[int]],
                 kappa: float = 12.0):
        self.degrees = degrees
        self.neighbors = neighbors
        self.n = len(degrees)
        self.kappa = kappa

    def compute_radii(self) -> List[float]:
        """度数 → 半径映射。

        公式：r = κ / (κ + deg)
            - deg=0  → r=1.0（潜态单位半径）
            - deg=κ  → r=0.5（晶子半满）
            - deg→∞  → r→0（不可能是晶子饱和点）
        """
        return [
            self.kappa / (self.kappa + max(d, 0))
            for d in self.degrees
        ]

    def compute_positions(self,
                          initial_positions: Optional[List[Tuple[float, float, float]]] = None,
                          iterations: int = 50,
                          learning_rate: float = 0.1) -> List[Tuple[float, float, float]]:
        """力导向松弛 — 从邻居关系计算三维坐标。

        SPUM 版本的特点：
            - 目标距离 = r_i + r_j（球面相切）
            - 没有"弹簧常数"——只有拓扑约束
            - 不收敛到全局最优，只收敛到拓扑自洽

        Args:
            initial_positions: 初始位置（None=斐波那契球面）
            iterations: 松弛迭代次数
            learning_rate: 步长

        Returns:
            每个节点的 (x, y, z) 坐标列表
        """
        import numpy as np

        radii = self.compute_radii()
        pos = self._initial_positions(initial_positions)

        for _ in range(iterations):
            forces = np.zeros((self.n, 3), dtype=np.float64)

            for i in range(self.n):
                if self.degrees[i] == 0:
                    continue
                for j in self.neighbors[i]:
                    if self.degrees[j] == 0:
                        continue
                    vec = pos[j] - pos[i]
                    dist = np.linalg.norm(vec)
                    if dist < 1e-12:
                        continue
                    target_dist = radii[i] + radii[j]
                    # 误差 = 当前距离 - 目标距离
                    error = dist - target_dist
                    force_dir = vec / dist
                    forces[i] += force_dir * error * learning_rate

            # 更新位置
            for i in range(self.n):
                if self.degrees[i] > 0:
                    pos[i] += forces[i]

            # 中心化
            active = [i for i in range(self.n) if self.degrees[i] > 0]
            if active:
                center = np.mean([pos[i] for i in active], axis=0)
                for i in active:
                    pos[i] -= center

        return [tuple(p.astype(float)) for p in pos]

    def compute_surface_area(self, degree: int) -> float:
        """度数 → 等效表面积（认知投影）。"""
        r = self.kappa / (self.kappa + max(degree, 0))
        return 4.0 * math.pi * r * r

    def compute_volume(self, degree: int) -> float:
        """度数 → 等效体积（认知投影）。"""
        r = self.kappa / (self.kappa + max(degree, 0))
        return 4.0 / 3.0 * math.pi * r * r * r

    def compute_spum_invariant_3d(self) -> int:
        """计算 Σ(6−deg) 拓扑不变量（仅在三维投影下）。"""
        return sum(max(0, 6 - d) for d in self.degrees if d > 0)

    def _initial_positions(self,
                           initial: Optional[List[Tuple[float, float, float]]] = None
                           ) -> 'np.ndarray':
        """生成或转换初始位置。"""
        import numpy as np
        if initial is not None:
            return np.array(initial, dtype=np.float64)

        # 斐波那契球面作为默认初态
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        positions = np.zeros((self.n, 3), dtype=np.float64)
        for i in range(self.n):
            y = 1.0 - (2.0 * i + 1.0) / self.n
            r = math.sqrt(1.0 - y * y)
            theta = 2.0 * math.pi / phi * i
            positions[i] = (r * math.cos(theta), y, r * math.sin(theta))
        return positions


class TemporalProjector:
    """时间轴投影器 — 帧序列→连续时间线的只读映射。

    SPUM 中没有外部时间。时间 = 帧计数。
    此投影器将帧序列映射为可观测的时间参数。
    """

    def __init__(self, frame_count: int = 0):
        self.frame_count = frame_count

    def frame_to_time(self, frame: int, tau: float = 1.0) -> float:
        """帧 → 时间。τ 是帧间间隔（默认 1.0）。"""
        return frame * tau

    def velocity(self, prev_pos: List[Tuple[float, float, float]],
                 curr_pos: List[Tuple[float, float, float]],
                 frame_delta: int = 1) -> List[float]:
        """投影速度：位置变化 / 帧数。

        注意：这不是"运动"——运动在 L0 是拓扑连接的重构。
        速度是投影层对拓扑变化的几何解释。
        """
        import numpy as np
        velocities = []
        for p0, p1 in zip(prev_pos, curr_pos):
            v = np.linalg.norm(np.array(p1) - np.array(p0)) / frame_delta
            velocities.append(float(v))
        return velocities


class ChemicalProjector:
    """化学投影器 — 度数/拓扑→元素分类的只读映射。

    VSPT 在 CA 框架中的投影：
        - 度数 → VSPT 分支密度
        - 边结构 → 壳层填充模式
        - 悬挂端 → 化学价（未配对电子）
    """

    def __init__(self, degrees: List[int], neighbors: List[List[int]]):
        self.degrees = degrees
        self.neighbors = neighbors

    def vspt_density(self, degree: int) -> float:
        """度数 → VSPT 虚关系密度。

        ρ ∝ r⁻³, r = κ/(κ+deg)
        => ρ ∝ (κ+deg)³/κ³
        """
        return ((self.kappa + degree) / self.kappa) ** 3

    def valence(self, node: int) -> int:
        """悬挂亏缺 → 化学价。"""
        deg = self.degrees[node]
        if deg == 0:
            return 0
        return max(0, 2 - deg)
