"""
SPUM 粒子数组 — SoA (Structure of Arrays) 布局。

CUDA 设计核心理念：
    每个 CUDA 线程 = 一个空间粒子，SoA 布局保证合并访问。
    当前使用 numpy 后端，接口镜像 GPU 布局：
        d_pos[i]    = float3 = 粒子 i 的三维坐标
        d_radius[i] = float  = 粒子 i 的半径 (从体积计算)
        d_degree[i] = int    = 粒子 i 的度数
        d_initial_volume[i] = float = 粒子 i 创建时的初始体积
        d_initial_degree[i] = int   = 粒子 i 创建时的初始度数
        体积公式: V = initial_volume + (degree - initial_degree)

    删除粒子 → 标记为潜在 (d_active[i] = 0)，不释放槽位。
    创建粒子 → 复用潜在槽位或扩展数组。
"""

import math
import numpy as np
from typing import Tuple, Optional
from .constants import KAPPA, CRYSTALLITE_DEGREE_THRESHOLD


class ParticleArray:
    """SoA 粒子数组 — numpy 后端 (镜像 GPU 布局)。

    所有操作设计为向量化 numpy 操作，
    与 GPU 的 per-thread 逻辑一一对应。

    内存布局:
        pos      (N, 3)  float32  — 位置
        radius   (N,)    float32  — 半径 (从体积计算)
        degree   (N,)    int32    — 度数 (从 connections 计算)
        active   (N,)    bool     — True=活性, False=潜在
        uid      (N,)    object   — 唯一标识字符串
        initial_volume (N,) float32  — 创建时的初始体积 (种子=1, 缝隙=笛卡尔体积)
        initial_degree (N,) int32    — 创建时的初始度数 (种子=0, 缝隙=3)
        体积公式: V = initial_volume + (degree - initial_degree)
        connections: set[tuple[int,int]] — (i,j) 边集合, i<j
    """

    def __init__(self, n: int = 0, max_n: int = 65536):
        self.N = n                     # 当前分配的槽位数
        self.max_n = max_n             # 最大容量
        self.connections = set()       # (min_idx, max_idx) 边集合
        self.tangent = None            # 初始化时分配
        self._alloc(n)

    def _alloc(self, n: int):
        """分配/重分配 SoA 内存。"""
        n = max(n, 1)
        self.N = n
        self.pos = np.zeros((n, 3), dtype=np.float32)
        self.radius = np.zeros(n, dtype=np.float32)
        self.degree = np.zeros(n, dtype=np.int32)
        self.active = np.zeros(n, dtype=np.bool_)
        self.uid = np.empty(n, dtype=object)
        self.initial_volume = np.zeros(n, dtype=np.float32)
        self.initial_degree = np.zeros(n, dtype=np.int32)
        self.tangent = [{} for _ in range(n)]  # tangent[i][j] = (theta, phi)
        self.connections.clear()

    def resize(self, new_n: int):
        """扩容 — 保留已有数据。"""
        if new_n <= self.N:
            return
        old = {
            'pos': self.pos.copy(),
            'radius': self.radius.copy(),
            'degree': self.degree.copy(),
            'active': self.active.copy(),
            'uid': self.uid.copy(),
            'initial_volume': self.initial_volume.copy(),
            'initial_degree': self.initial_degree.copy(),
        }
        old_tangent = self.tangent.copy()  # list of dicts
        old_connections = self.connections.copy()
        self._alloc(new_n)
        for k in old:
            getattr(self, k)[:len(old[k])] = old[k]
        # 恢复 tangent (前 N 个)
        for i in range(len(old_tangent)):
            self.tangent[i] = old_tangent[i]
        self.connections = old_connections

    def add_particle(self, pos: Tuple[float, float, float],
                     uid: str, degree: int = 0,
                     initial_volume: float = 1.0,
                     initial_degree: int = 0) -> int:
        """添加一个粒子到数组中。如果有潜在槽位则复用。

        Args:
            pos: 三维坐标
            uid: 唯一标识
            degree: 初始度数
            initial_volume: 创建时的体积 (种子=1, 缝隙=笛卡尔定理计算值)
            initial_degree: 创建时的度数 (种子=0, 缝隙=3)

        Returns:
            粒子索引
        """
        latent = np.where(~self.active)[0]
        if len(latent) > 0:
            idx = latent[0]
        else:
            idx = self.active.size
            if idx >= self.max_n:
                raise RuntimeError(f"粒子数超上限 {self.max_n}")
            if idx >= self.N:
                new_n = min(self.N * 2 + 1, self.max_n)
                self.resize(new_n)

        self.pos[idx] = pos
        V = float(initial_volume) + max(0, float(degree) - float(initial_degree))
        self.radius[idx] = (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.degree[idx] = degree
        self.active[idx] = True
        self.uid[idx] = uid
        self.initial_volume[idx] = initial_volume
        self.initial_degree[idx] = initial_degree
        return idx

    def remove_particle(self, idx: int):
        """删除粒子 → 标记为潜在，不清除 uid。"""
        self.active[idx] = False
        self.degree[idx] = 0
        # 切断所有连接 (自动清理 tangent)
        self.remove_all_connections(idx)
        if self.tangent[idx] is not None:
            self.tangent[idx].clear()

    def active_count(self) -> int:
        return int(np.sum(self.active))

    def latent_count(self) -> int:
        return int(np.sum(~self.active & (self.uid != None)))

    def degree_histogram(self) -> np.ndarray:
        d = self.degree[self.active]
        d_clipped = np.clip(d, 0, CRYSTALLITE_DEGREE_THRESHOLD)
        hist = np.zeros(CRYSTALLITE_DEGREE_THRESHOLD + 1, dtype=np.int32)
        for i in range(CRYSTALLITE_DEGREE_THRESHOLD + 1):
            hist[i] = int(np.sum(d_clipped == i))
        return hist

    def crystallite_count(self) -> int:
        return int(np.sum(self.degree[self.active] >= CRYSTALLITE_DEGREE_THRESHOLD))

    def dangling_mask(self) -> np.ndarray:
        """返回布尔数组，标记度数 < 3 的活性粒子。（SPUM: deg < 3 为悬挂）"""
        return self.active & (self.degree < 3)

    def spum_invariant(self) -> int:
        """Σ(6 - deg(v)) = 6V - 2E。"""
        V = self.active_count()
        E = int(np.sum(self.degree[self.active])) // 2
        return 6 * V - 2 * E

    def reconstruct_positions(self, origin_idx: int = 0, relax: bool = True):
        """从 tangent 角度数据重建笛卡尔坐标。

        球面角数据是主几何信息 (θ, φ).
        重建算法:
            1. 将 origin_idx 放在 (0,0,0)
            2. BFS: 对每个已放置粒子 i, 未放置的邻接 j,
               用 tangent[i][j] 的方向 + (ri + rj) 计算 pos_j
            3. 未连接的粒子保持原位不变

        此方法应在每帧开始时调用, 以确保 pos 与 tangent 一致。
        """
        if origin_idx < 0 or origin_idx >= self.N or not self.active[origin_idx]:
            # 找第一个活跃替代
            active_pos = np.where(self.active)[0]
            if len(active_pos) == 0:
                return
            origin_idx = int(active_pos[0])

        placed = {origin_idx}
        self.pos[origin_idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # BFS
        queue = [origin_idx]
        qhead = 0
        while qhead < len(queue):
            current = queue[qhead]
            qhead += 1
            r_cur = float(self.radius[current])
            pos_cur = self.pos[current]
            # 遍历所有 tangent 中记录的邻接
            for nbr, (theta, phi) in self.tangent[current].items():
                if nbr in placed:
                    continue
                if not self.active[nbr]:
                    continue
                # 方向向量
                ux = math.sin(theta) * math.cos(phi)
                uy = math.sin(theta) * math.sin(phi)
                uz = math.cos(theta)
                r_nbr = float(self.radius[nbr])
                dist = r_cur + r_nbr
                self.pos[nbr] = np.array([
                    float(pos_cur[0]) + ux * dist,
                    float(pos_cur[1]) + uy * dist,
                    float(pos_cur[2]) + uz * dist,
                ], dtype=np.float32)
                placed.add(nbr)
                queue.append(nbr)

        # 松弛 (可选): BFS 后迭代调整所有粒子的位置, 满足所有邻接的相切约束
        # BFS 只用一个邻居定位, 对其他邻居的约束不成立。松弛修复此偏差。
        # 但对于四面体填充 (所有粒子距原点 1 跳), BFS 已精确, 松弛反而有害。
        if relax:
            for _relax in range(10):
                max_disp = 0.0
                for idx in list(placed):
                    if idx == origin_idx:
                        continue
                    nbrs = list(self.tangent[idx].keys())
                    if len(nbrs) < 1:
                        continue
                    # 计算"平均"期望位置: 用所有邻居的相切约束
                    best_pos = np.zeros(3, dtype=np.float32)
                    w_sum = 0.0
                    r_idx = float(self.radius[idx])
                    for nbr in nbrs:
                        if nbr not in placed or not self.active[nbr]:
                            continue
                        r_nbr = float(self.radius[nbr])
                        d_vec = self.pos[idx] - self.pos[nbr]
                        d = float(np.linalg.norm(d_vec))
                        if d < 1e-12:
                            continue
                        target = self.pos[nbr] + (d_vec / d) * (r_nbr + r_idx)
                        best_pos += target
                        w_sum += 1.0
                    if w_sum < 0.5:
                        continue
                    avg_pos = best_pos / w_sum
                    disp = float(np.linalg.norm(avg_pos - self.pos[idx]))
                    if disp > 0.0001:
                        alpha = 0.5
                        self.pos[idx] = (1.0 - alpha) * self.pos[idx] + alpha * avg_pos
                        max_disp = max(max_disp, disp)
                if max_disp < 0.0001:
                    break

    # ---- 边管理 (persistent connections) ----

    def add_connection(self, i: int, j: int) -> bool:
        if i < 0 or j < 0:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in self.connections:
            return False
        self.connections.add((a, b))
        self.degree[a] += 1
        self.degree[b] += 1
        # 计算和存储 tangent 角度 (从每个粒子视角看向对方)
        self._store_tangent_angles(i, j)
        return True

    def _store_tangent_angles(self, i: int, j: int):
        """计算并存储 i 看 j 和 j 看 i 的切点球面角。

        切点在两球体中心连线上, 距离 i 中心 r_i 处。
        方向: 从 i 指向 j 的单位向量.
        """
        # 方向向量: i → j
        dx = float(self.pos[j][0] - self.pos[i][0])
        dy = float(self.pos[j][1] - self.pos[i][1])
        dz = float(self.pos[j][2] - self.pos[i][2])
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d < 1e-12:
            return
        ux, uy, uz = dx/d, dy/d, dz/d
        # i 看 j: (θ_ij, φ_ij)
        theta_ij = math.acos(max(-1.0, min(1.0, uz)))
        phi_ij = math.atan2(uy, ux)
        self.tangent[i][j] = (theta_ij, phi_ij)
        # j 看 i: 相反方向 (π-θ, φ+π)
        theta_ji = math.acos(max(-1.0, min(1.0, -uz)))
        phi_ji = math.atan2(-uy, -ux)
        self.tangent[j][i] = (theta_ji, phi_ji)

    def remove_connection(self, i: int, j: int) -> bool:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in self.connections:
            return False
        self.connections.discard((a, b))
        self.degree[a] = max(0, self.degree[a] - 1)
        self.degree[b] = max(0, self.degree[b] - 1)
        # 清理 tangent 数据
        if self.tangent[a] and b in self.tangent[a]:
            del self.tangent[a][b]
        if self.tangent[b] and a in self.tangent[b]:
            del self.tangent[b][a]
        return True

    def connected(self, i: int, j: int) -> bool:
        a, b = (i, j) if i < j else (j, i)
        return (a, b) in self.connections

    def remove_all_connections(self, idx: int):
        to_remove = [pair for pair in self.connections if pair[0] == idx or pair[1] == idx]
        for a, b in to_remove:
            self.connections.discard((a, b))
            self.degree[a] = max(0, self.degree[a] - 1)
            self.degree[b] = max(0, self.degree[b] - 1)
            # 清理 tangent
            if self.tangent[a] and b in self.tangent[a]:
                del self.tangent[a][b]
            if self.tangent[b] and a in self.tangent[b]:
                del self.tangent[b][a]

    def clear_connections(self):
        self.connections.clear()
        self.degree[:] = 0
        self.tangent = [{} for _ in range(self.N)]

    def copy_to_host(self):
        return {
            'pos': self.pos.copy(),
            'radius': self.radius.copy(),
            'degree': self.degree.copy(),
            'active': self.active.copy(),
            'uid': self.uid.copy(),
            'initial_volume': self.initial_volume.copy(),
            'initial_degree': self.initial_degree.copy(),
        }

    def snapshot(self, frame_number: int) -> 'FrameSnapshot':
        from .metadata_decoder import FrameSnapshot
        return FrameSnapshot(
            frame_number=frame_number,
            degree_histogram=self.degree_histogram(),
            active_count=self.active_count(),
            latent_count=self.latent_count(),
            crystallite_count=self.crystallite_count(),
            spum_invariant=self.spum_invariant(),
            dangling_count=int(np.sum(self.dangling_mask())),
        )
