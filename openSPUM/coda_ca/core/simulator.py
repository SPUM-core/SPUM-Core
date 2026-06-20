"""
SPUM CA 模拟器 — 无中央调度的并行步进引擎。

核心原则：
    - 没有全局调度器
    - 每一帧，每个节点独立、并行执行同一条规则
    - 节点不知道宇宙整体，只知道自己的邻居
    - 演化是去中心化的、局部的、确定性的

帧循环（对应 SPUM 五步帧序列）：
    1. 收集：每个节点从邻居读取度数
    2. 计算：每个节点应用规则函数计算自己的下一帧度数
    3. 更新：所有节点同时更新度数（同步更新）
    4. 边演化：度数变化后更新边结构
    5. 记录：保存帧快照（只记录度数分布，不记录坐标）

使用方式：
    sim = Simulator(n_codas=1000, rule_name="mean")
    sim.initialize_icosahedron()  # 正二十面体初态
    for _ in range(100):
        sim.step()
        print(sim.summary())
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field

from .coda import Coda
from .rules import get_rule, edges_from_degrees


@dataclass
class FrameSnapshot:
    """单帧快照 — 只记录度数分布和拓扑统计。

    不记录：
        - 坐标（由投影层处理）
        - 半径（由投影层处理）
        - 任何 L1/L2 认知层属性

    本体论纯度：
        帧 = 度数分布 + 边集
    """
    frame: int
    degrees: Tuple[int, ...]          # 所有节点的度数快照
    edge_count: int                    # 总边数
    degree_hist: Dict[int, int]        # 度数直方图（度数→计数）
    n_active: int                      # deg > 0 的节点数
    n_dangling: int                    # deg = 1 的节点数
    n_crystallite: int                 # deg ≥ κ 的节点数
    spum_invariant: int = 0            # Σ(6−deg) over active nodes


class Simulator:
    """SPUM CA 模拟器。

    不存储任何坐标、半径、几何信息。
    仅存储：每个 Coda 的度数 + 邻居列表。
    """

    def __init__(self, n_codas: int = 1000,
                 rule_name: str = "mean",
                 kappa: int = 12,
                 crystallite_threshold: int = 50):
        self.n = n_codas
        self.kappa = kappa
        self.crystallite_threshold = crystallite_threshold
        self.rule_fn: Callable = get_rule(rule_name)
        self.rule_name = rule_name

        # 核心数据：度数数组 + 邻居列表
        self.degrees: List[int] = [0] * n_codas
        self.neighbors: List[List[int]] = [[] for _ in range(n_codas)]

        self.frame: int = 0
        self.history: List[FrameSnapshot] = []

    # ── 初态初始化 ──────────────────────────────────────

    def initialize_icosahedron(self):
        """正二十面体初态：12 节点，每个 deg=5，30 条边。

        顶点位置来自 (_icosahedron_vertices)，
        边由三维最近邻确定（每个顶点有 5 个等距邻居）。
        """
        n = 12
        self.degrees = [0] * self.n
        self.neighbors = [[] for _ in range(self.n)]

        # 计算 12 个顶点（单位球面上的正二十面体）
        import math
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        verts = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                verts.append((0.0, float(s1), float(s2 * phi)))
                verts.append((float(s1), float(s2 * phi), 0.0))
                verts.append((float(s2 * phi), 0.0, float(s1)))
        verts = verts[:12]
        # 归一化到单位球面
        norm = [math.sqrt(x*x + y*y + z*z) for (x, y, z) in verts]
        verts = [(x/n, y/n, z/n) for (x, y, z), n in zip(verts, norm)]

        # 最近邻：每个顶点找 5 个最近邻居（距离 = 边）
        for i in range(12):
            dists = []
            for j in range(12):
                if i == j:
                    continue
                d = (verts[i][0]-verts[j][0])**2 + \
                    (verts[i][1]-verts[j][1])**2 + \
                    (verts[i][2]-verts[j][2])**2
                dists.append((d, j))
            dists.sort()
            for _, j in dists[:5]:  # 取最近的 5 个
                if j not in self.neighbors[i]:
                    self.neighbors[i].append(j)
                if i not in self.neighbors[j]:
                    self.neighbors[j].append(i)

        for i in range(n):
            self.degrees[i] = len(self.neighbors[i])

        self.frame = 0
        self.history = []
        self._record_frame()

    def initialize_star(self, n_surface: int = 12):
        """星形初态：1 中心 + N 表面，中心连接所有表面。

        中心 deg = n_surface，表面 deg = 1。
        这是悬挂端丰富的初态，适合测试悬挂边守恒。
        """
        total = 1 + n_surface
        self.degrees = [0] * self.n
        self.neighbors = [[] for _ in range(self.n)]

        # 中心连接所有表面
        for i in range(n_surface):
            self.neighbors[0].append(1 + i)
            self.neighbors[1 + i].append(0)

        self.degrees[0] = n_surface
        for i in range(n_surface):
            self.degrees[1 + i] = 1

        self.frame = 0
        self.history = []
        self._record_frame()

    def initialize_random_graph(self, edge_density: float = 0.05):
        """确定性随机图初态（非随机：用哈希函数生成确定性拓扑）。"""
        import hashlib
        self.degrees = [0] * self.n
        self.neighbors = [[] for _ in range(self.n)]

        # 确定性"伪随机"连边
        for i in range(self.n):
            for j in range(i + 1, self.n):
                h = hashlib.md5(f"{i},{j}".encode()).hexdigest()
                val = int(h[:8], 16) / 0xFFFFFFFF
                if val < edge_density:
                    self.neighbors[i].append(j)
                    self.neighbors[j].append(i)

        for i in range(self.n):
            self.degrees[i] = len(self.neighbors[i])

        self.frame = 0
        self.history = []
        self._record_frame()

    def initialize_from_sequence(self, n_active: int = 50):
        """链式接入初态：节点逐个接入，度数均衡化。

        见原始 Phase 0 的 _init_sequential。
        """
        self.degrees = [0] * self.n
        self.neighbors = [[] for _ in range(self.n)]

        # 前两个节点连接
        self.neighbors[0].append(1)
        self.neighbors[1].append(0)

        for i in range(2, n_active):
            # 找到度数最低的已有节点
            degs = [(self.degrees[j], j) for j in range(i)]
            degs.sort()
            best_j = degs[0][1]
            self.neighbors[best_j].append(i)
            self.neighbors[i].append(best_j)

        for i in range(self.n):
            self.degrees[i] = len(self.neighbors[i])

        self.frame = 0
        self.history = []
        self._record_frame()

    # ── 帧演化 ──────────────────────────────────────────

    def step(self) -> FrameSnapshot:
        """单步帧演化（同步更新）。

        对应 SPUM 五步帧序列：
            1. 收集：读取邻居度数（隐含在规则函数的调用中）
            2. 计算：应用规则函数
            3. 更新：同步写入新度数
            4. 边演化：度数改变后更新边结构
            5. 记录：保存帧快照
        """
        self.frame += 1

        # Step 1-2: 收集邻居度数 + 计算新度数（同步）
        new_degrees = [0] * self.n
        for i in range(self.n):
            neighbor_degs = [self.degrees[j] for j in self.neighbors[i]
                             if self.degrees[j] > 0 or j < self.n]
            new_degrees[i] = self.rule_fn(self.degrees[i], neighbor_degs)

        # Step 3: 同步更新度数
        self.degrees[:] = new_degrees

        # Step 4: 边演化 — 度数变化后更新边结构
        self._evolve_edges()

        # Step 5: 记录
        return self._record_frame()

    def step_n(self, n: int) -> List[FrameSnapshot]:
        """连续运行 n 帧。"""
        for _ in range(n):
            self.step()
        return self.history[-n:]

    def _evolve_edges(self):
        """边演化：根据新度数更新边结构。

        这是 CA 中的"关系演化"：
            - deg=0 的节点删除所有边（回归潜态）
            - 其他节点保持与度数匹配的边数
            - 使用 edges_from_degrees 进行全局协调

        TODO: 实现纯局部边演化规则（节点只与邻居通信协商）
        """
        self.neighbors = edges_from_degrees(
            self.neighbors, self.degrees, self.n
        )

    # ── 帧记录 ──────────────────────────────────────────

    def _record_frame(self) -> FrameSnapshot:
        """记录当前帧快照。"""
        from collections import Counter
        hist = Counter(self.degrees)
        n_active = sum(1 for d in self.degrees if d > 0)
        n_dangling = sum(1 for d in self.degrees if d == 1)
        n_crystallite = sum(1 for d in self.degrees
                            if d >= self.crystallite_threshold)
        edge_count = sum(len(nbrs) for nbrs in self.neighbors) // 2

        # Σ(6−deg) over active nodes (SPUM 拓扑不变量)
        invariant = sum(max(0, 6 - d) for d in self.degrees if d > 0)

        snap = FrameSnapshot(
            frame=self.frame,
            degrees=tuple(self.degrees),
            edge_count=edge_count,
            degree_hist=dict(hist),
            n_active=n_active,
            n_dangling=n_dangling,
            n_crystallite=n_crystallite,
            spum_invariant=invariant,
        )
        self.history.append(snap)
        return snap

    # ── 查询 ────────────────────────────────────────────

    def summary(self) -> Dict:
        """当前状态摘要（纯度数信息，无几何）。"""
        s = self.history[-1] if self.history else None
        if s is None:
            return {"frame": 0, "n_active": 0, "edge_count": 0}
        return {
            "frame": s.frame,
            "active": s.n_active,
            "dangling": s.n_dangling,
            "crystallite": s.n_crystallite,
            "edges": s.edge_count,
            "mean_deg": round(sum(s.degrees) / max(s.n_active, 1), 2),
            "Σ(6−deg)": s.spum_invariant,
            "rule": self.rule_name,
        }

    def get_active_indices(self) -> List[int]:
        """返回所有 deg > 0 的节点索引。"""
        return [i for i, d in enumerate(self.degrees) if d > 0]

    def get_degree_distribution(self) -> Dict[int, int]:
        """返回度分布字典。"""
        if self.history:
            return self.history[-1].degree_hist
        from collections import Counter
        return dict(Counter(self.degrees))
