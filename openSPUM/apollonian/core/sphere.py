"""
ApollonianSphere — 几何完全由外部关系定义的球体。

核心原则：
    - 不存储位置和半径
    - 位置和半径在每次访问时从邻居的约束求解
    - 唯一存储的状态：邻居列表（相切关系）
    - 求解使用外部提供的 solved_state（避免递归）

移除球体后，"空洞"仍然存在几何定义：
    邻居的约束仍然等价于位置和半径的解。
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .descartes import (
    solve_bend_3d,
    solve_position_trilateration,
)

# 类型：uid → (position, radius) 的已求解状态
SolvedState = Dict[str, Tuple[Tuple[float, float, float], float]]


class ApollonianSphere:
    """阿波罗尼奥斯球体。

    不存储 position 和 radius——每次从 solved_state 求解。
    存储：uid, neighbor_ids, is_active
    """

    def __init__(self, uid: str = ""):
        self.uid = uid
        self.neighbor_ids: List[str] = []
        self.is_active: bool = True

    # ── 几何求解 ────────────────────────────────────────

    def solve_geometry(
        self,
        solved: SolvedState
    ) -> Optional[Tuple[Tuple[float, float, float], float]]:
        """从已求解的邻居状态求解自身几何。

        Args:
            solved: 当前帧已求解的 {uid: (pos, r)} 映射

        Returns:
            (position, radius) 或 None（求解失败）
        """
        # 收集活跃邻居
        active_neighbors = [
            nid for nid in self.neighbor_ids
            if nid in solved
        ]

        if len(active_neighbors) < 4:
            # 不足 4 个：用邻居几何的"最佳猜测"
            return self._solve_underdetermined(active_neighbors, solved)

        # 取前 4 个邻居
        n4 = active_neighbors[:4]
        positions = [solved[nid][0] for nid in n4]
        radii = [solved[nid][1] for nid in n4]

        # 索迪定理：从 4 个曲率求第 5 个
        bends = [1.0 / max(r, 1e-12) for r in radii]
        b_inner, b_outer = solve_bend_3d(*bends)

        # 尝试两个解，选取与所有 4 个邻居的相切约束一致的解
        r_candidates = []
        if not math.isnan(b_inner) and not math.isinf(b_inner):
            r_candidates.append(1.0 / b_inner)
        if not math.isnan(b_outer) and not math.isinf(b_outer):
            r_candidates.append(1.0 / b_outer)
        # 也添加均值作为后备
        avg_r = sum(radii) / len(radii)
        r_candidates.append(avg_r)

        # 每个候选解：用三边测量求位置，检查相切约束
        best = None
        best_error = float('inf')
        for r_test in r_candidates:
            if r_test <= 0:
                continue
            pos_test = solve_position_trilateration(positions, radii, r_test)
            if pos_test is None:
                continue
            # 检查与 4 个邻居的相切误差
            total_error = 0.0
            valid = True
            for k in range(4):
                d = math.sqrt(sum((pos_test[i] - positions[k][i])**2 for i in range(3)))
                expected = r_test + radii[k]
                err = abs(d - expected) / max(expected, 1e-8)
                if err > 0.5:
                    valid = False
                    break
                total_error += err
            if not valid:
                continue
            if total_error < best_error:
                best_error = total_error
                best = (pos_test, r_test)

        if best is None:
            # 都失败了：用第一个候选
            if r_candidates and r_candidates[0] > 0:
                pos = solve_position_trilateration(positions, radii, r_candidates[0])
                if pos:
                    return (pos, r_candidates[0])
            return None

        return best

    def _solve_underdetermined(
        self,
        neighbor_ids: List[str],
        solved: SolvedState
    ) -> Optional[Tuple[Tuple[float, float, float], float]]:
        """邻居不足 4 个时的退化解。"""
        if not neighbor_ids:
            return ((0.0, 0.0, 0.0), 1.0)

        # 取第一个邻居
        n0 = neighbor_ids[0]
        if n0 not in solved:
            return ((0.0, 0.0, 0.0), 1.0)

        n0_pos, n0_r = solved[n0]

        # 从已有邻居的平均半径估计自身半径
        radii = [solved[nid][1] for nid in neighbor_ids if nid in solved]
        r_self = sum(radii) / len(radii) if radii else 1.0

        # 沿第一个邻居法向放置
        r_sum = n0_r + r_self
        pos = (n0_pos[0] + r_sum, n0_pos[1], n0_pos[2])

        return (pos, r_self)

    # ── 空洞查询 ────────────────────────────────────────

    def get_cavity_geometry(
        self,
        solved: SolvedState
    ) -> Optional[Tuple[Tuple[float, float, float], float]]:
        """查询空洞几何——即使球体已被移除。

        这就是"移除实体，几何犹存"的核心：
        只要邻居的几何还在，这个球体的位置和半径
        就是邻居约束的解——空洞是刚性的。
        """
        return self.solve_geometry(solved)

    # ── 关系操作 ────────────────────────────────────────

    def add_neighbor(self, nid: str):
        if nid not in self.neighbor_ids:
            self.neighbor_ids.append(nid)

    def remove_neighbor(self, nid: str):
        if nid in self.neighbor_ids:
            self.neighbor_ids.remove(nid)

    def remove(self):
        """标记为不活跃——几何信息仍可通过 get_cavity_geometry 查询。"""
        self.is_active = False

    # ── 查询 ────────────────────────────────────────────

    @property
    def degree(self) -> int:
        return len(self.neighbor_ids)

    def __repr__(self) -> str:
        return f"ASphere({self.uid}, deg={self.degree}, act={self.is_active})"
