"""
从 ⟨P, ε⟩ 到 ℝ³ 的诱导坐标嵌入。

算法严格遵循 L0.5_induced_metric.md 的定义：
  1. 12-晶子种子初始化（正二十面体顶点）
  2. BFS传播 + 球面码方向规则
  3. σ调节步长
  4. 多邻居冲突加权质心解决
"""

import numpy as np
import sys
import os
from collections import deque
from typing import List, Tuple, Dict, Optional, Set

_path = os.path.dirname(os.path.abspath(__file__))
if _path not in sys.path:
    sys.path.insert(0, _path)
from icosahedron_seed import generate_icosahedron_seed, get_spherical_code_directions


class InducedMetricEmbedding:
    """
    从 ⟨P, ε⟩ 到 ℝ³ 的诱导坐标嵌入。

    算法：
    1. 12-晶子种子初始化
    2. BFS传播 + 球面码方向规则
    3. σ调节步长
    4. 多邻居冲突加权质心解决
    """

    def __init__(self, kappa: float = 1.0, sigma_0: float = 1.0,
                 epsilon_conflict: float = 0.1):
        self.kappa = kappa
        self.sigma_0 = sigma_0
        self.epsilon_conflict = epsilon_conflict

        self.positions: Dict[int, np.ndarray] = {}   # node_id -> (x, y, z)
        self.sigmas: Dict[int, float] = {}            # node_id -> sigma value
        self.degrees: Dict[int, int] = {}             # node_id -> degree
        self.embedded: Set[int] = set()               # set of embedded node ids
        self.conflict_nodes: Set[int] = set()          # nodes with high conflict residual

        # Internal: mapping from node id to its BFS shell index
        self._shell: Dict[int, int] = {}

    def seed_from_icosahedron(self, node_ids: Optional[List[int]] = None) -> None:
        """用正二十面体的12个顶点初始化种子。

        如果 node_ids 少于12个，则使用前 len(node_ids) 个顶点。
        如果 node_ids 为空，则什么也不做。
        """
        if node_ids is None:
            node_ids = list(range(12))
        if len(node_ids) == 0:
            return
        vertices, _ = generate_icosahedron_seed(self.kappa)
        n_seed = min(len(node_ids), len(vertices))
        for i in range(n_seed):
            nid = node_ids[i]
            self.positions[nid] = vertices[i].copy()
            self.embedded.add(nid)
            self._shell[nid] = 0

    def step_size(self, sigma: float) -> float:
        """
        σ调节步长: κ * (σ₀/σ)^{1/3}

        高σ（低度、密集）→ 步长缩短 → 嵌入压缩
        低σ（高度、稀疏）→ 步长伸长 → 嵌入拉伸
        """
        return self.kappa * (self.sigma_0 / max(sigma, 1e-10)) ** (1.0 / 3.0)

    def _assign_direction(self, node_id: int, neighbor_id: int,
                          adjacency: Dict[int, List[int]]) -> np.ndarray:
        """
        为node_id的一个邻居分配球面码方向。

        策略：从node_id的deg个球面码方向中，选择与所有已嵌入邻居
        已占用方向夹角最大的那个（最大化最小角间距）。

        Args:
            node_id: 已嵌入节点
            neighbor_id: 待嵌入邻居
            adjacency: 邻接表

        Returns:
            单位方向向量
        """
        pos = self.positions[node_id]
        deg = self.degrees.get(node_id, 6)
        directions = get_spherical_code_directions(deg)

        # 收集node_id的所有已嵌入邻居的方向（作为已占用方向）
        taken_dirs = []
        for nb in adjacency.get(node_id, []):
            if nb in self.embedded and nb != neighbor_id:
                vec = self.positions[nb] - pos
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    taken_dirs.append(vec / norm)

        if len(taken_dirs) == 0:
            # 没有已占用的方向，返回第一个方向
            return directions[0]

        # 从所有方向中选与已占用方向最小夹角最大的那个
        best_dir = None
        best_min_angle = -1.0

        for d_idx in range(min(deg, len(directions))):
            dir_vec = directions[d_idx]
            min_angle = float('inf')
            for taken in taken_dirs:
                dot = np.clip(np.dot(dir_vec, taken), -1.0, 1.0)
                angle = np.arccos(dot)
                min_angle = min(min_angle, angle)
            if min_angle > best_min_angle:
                best_min_angle = min_angle
                best_dir = dir_vec

        return best_dir if best_dir is not None else directions[0]

    def get_candidate_position(self, source_id: int, target_id: int,
                               adjacency: Dict[int, List[int]]) -> np.ndarray:
        """
        从已嵌入邻居source_id出发，为target_id计算候选位置。

        方向：使用球面码规则（基于source_id的度数）
        步长：κ * (σ₀/σ)^{1/3}
        """
        step = self.step_size(self.sigmas.get(source_id, self.sigma_0))
        direction = self._assign_direction(source_id, target_id, adjacency)
        return self.positions[source_id] + step * direction

    def bfs_embed(self, adjacency: Dict[int, List[int]],
                  sigma_map: Dict[int, float],
                  degree_map: Dict[int, int],
                  start_node_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        BFS序传播嵌入。

        Args:
            adjacency: dict of node_id -> list of neighbor_ids
            sigma_map: dict of node_id -> sigma value
            degree_map: dict of node_id -> degree
            start_node_ids: list of 12 node ids for the icosahedron seed

        Returns:
            positions: dict of node_id -> (x, y, z) for all embedded nodes

        每步：
        1. 对当前节点的每个未嵌入邻居，计算候选位置
        2. 使用加权质心合并多候选
        3. 检查冲突残差
        4. 标记嵌入不稳定节点
        """
        # 存储sigma、degree和邻接表
        self.sigmas.update(sigma_map)
        self.degrees.update(degree_map)
        self._adjacency = adjacency

        # 初始化种子
        if start_node_ids is None:
            # 取前12个节点作为种子
            all_nodes = list(adjacency.keys())
            if len(all_nodes) >= 12:
                start_node_ids = all_nodes[:12]
            else:
                start_node_ids = all_nodes

        self.seed_from_icosahedron(start_node_ids)

        # BFS传播
        queue = deque(start_node_ids)
        processed = set(start_node_ids)

        while queue:
            current = queue.popleft()

            # 只处理已在邻接表中的节点
            if current not in adjacency:
                continue

            for neighbor in adjacency[current]:
                if neighbor in self.embedded or neighbor in processed:
                    continue

                # 收集neighbor的所有已嵌入邻居的候选位置
                candidates = []
                weights = []
                conflict_sum = 0.0
                num_candidates = 0

                for nb_of_u in adjacency.get(neighbor, []):
                    if nb_of_u in self.embedded:
                        cand = self.get_candidate_position(
                            nb_of_u, neighbor, adjacency)
                        candidates.append(cand)
                        # 权重 w(v) = 1/σ(v)
                        w = 1.0 / max(self.sigmas.get(nb_of_u, self.sigma_0),
                                      1e-10)
                        weights.append(w)
                        num_candidates += 1

                if num_candidates == 0:
                    # 没有已嵌入邻居，跳过（等待后续BFS轮次）
                    continue

                # 加权质心
                weights_arr = np.array(weights)
                pos_u = np.average(candidates, axis=0,
                                   weights=weights_arr)

                # 计算冲突残差
                if num_candidates > 1:
                    conflict_sum = np.sum([
                        np.linalg.norm(pos_u - cand)
                        for cand in candidates
                    ])
                    delta_conflict = conflict_sum / num_candidates
                else:
                    delta_conflict = 0.0

                # 嵌入neighbor
                self.positions[neighbor] = pos_u
                self.embedded.add(neighbor)
                processed.add(neighbor)

                if delta_conflict > self.epsilon_conflict:
                    self.conflict_nodes.add(neighbor)

                # 计算shell层级
                parent_shells = [
                    self._shell.get(nb_of_u, 0)
                    for nb_of_u in adjacency.get(neighbor, [])
                    if nb_of_u in self.embedded and nb_of_u != neighbor
                ]
                self._shell[neighbor] = min(parent_shells) + 1 if parent_shells else 1

                queue.append(neighbor)

        return dict(self.positions)

    def _graph_distance(self, u: int, v: int) -> int:
        """计算两点间的图最短路径距离（BFS）。"""
        if not hasattr(self, '_adjacency') or not self._adjacency:
            # 无邻接表时使用壳层索引差作为近似
            return abs(self._shell.get(u, 0) - self._shell.get(v, 0))
        if u == v:
            return 0
        visited = {u}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            for nb in self._adjacency.get(node, []):
                if nb == v:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        # 不连通时使用壳层索引差
        return abs(self._shell.get(u, 0) - self._shell.get(v, 0))

    def compute_global_distortion(self) -> float:
        """
        计算全局畸变 δ_global。

        δ_global = 1/|P_emb|² *
            Σ|d_G(u,v)/κ - ||φ(u)-φ(v)||/(κ·f_mean)|

        其中 f_mean 为全图平均尺度因子。
        """
        if len(self.embedded) < 2:
            return 0.0

        # 计算全图平均尺度因子 f_mean
        f_values = []
        for nid in self.embedded:
            sigma = self.sigmas.get(nid, self.sigma_0)
            f_values.append((self.sigma_0 / max(sigma, 1e-10)) ** (1.0 / 3.0))
        f_mean = np.mean(f_values) if f_values else 1.0

        n = len(self.embedded)
        node_list = list(self.embedded)
        total_distortion = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                uid = node_list[i]
                vid = node_list[j]

                d_G = self._graph_distance(uid, vid)

                euclidean_dist = np.linalg.norm(
                    self.positions[uid] - self.positions[vid])

                term1 = d_G / self.kappa
                term2 = euclidean_dist / (self.kappa * f_mean)
                total_distortion += abs(term1 - term2)

        delta_global = total_distortion / (n * n)
        return delta_global

    def compute_metric_tensor(self, node_id: int) -> np.ndarray:
        """
        在节点处诱导离散度量张量。

        g_ij(v) ≈ 1/2 * Σ (Δφ_i * Δφ_j) / ||Δφ||²
        求和遍及节点v的所有已嵌入邻居。

        Args:
            node_id: 目标节点

        Returns:
            g: (3, 3) 度量张量矩阵
        """
        if node_id not in self.positions:
            return np.eye(3)

        pos_v = self.positions[node_id]

        # 找所有已嵌入的邻居
        neighbor_positions = []
        for nid, pos in self.positions.items():
            if nid != node_id:
                dist = np.linalg.norm(pos - pos_v)
                if dist < 3.0 * self.step_size(
                        self.sigmas.get(node_id, self.sigma_0)):
                    neighbor_positions.append(pos)

        if len(neighbor_positions) < 3:
            return np.eye(3)

        g = np.zeros((3, 3))
        for pos_u in neighbor_positions:
            delta = pos_u - pos_v
            norm_sq = np.dot(delta, delta)
            if norm_sq < 1e-10:
                continue
            for i in range(3):
                for j in range(3):
                    g[i, j] += 0.5 * delta[i] * delta[j] / norm_sq

        return g

    def get_shell_indices(self) -> Dict[int, int]:
        """返回每个节点的BFS壳层索引。"""
        return dict(self._shell)
