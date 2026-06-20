"""
ApollonianNetwork — 约束求解引擎。

核心逻辑：
    每帧构建 solved_state（已求解的几何状态），
    然后检测缝隙并填充。

    不存储位置/半径——全都从约束求解。
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import math

from .sphere import ApollonianSphere, SolvedState


@dataclass
class FrameSnapshot:
    frame: int
    n_active: int
    n_removed: int
    edge_count: int
    n_gaps_filled: int
    degree_hist: Dict[int, int]
    solved: SolvedState = field(default_factory=dict)


class ApollonianNetwork:
    """阿波罗尼奥斯约束网络。"""

    def __init__(self):
        self.spheres: Dict[str, ApollonianSphere] = {}
        self.frame: int = 0
        self.history: List[FrameSnapshot] = []
        self._filled_gap_keys: Set[Tuple[str, ...]] = set()

    # ── 球体管理 ────────────────────────────────────────

    def add_sphere(self, uid: str,
                   neighbors: Optional[List[str]] = None) -> ApollonianSphere:
        sphere = ApollonianSphere(uid=uid)
        if neighbors:
            for nid in neighbors:
                if nid in self.spheres:
                    sphere.add_neighbor(nid)
                    self.spheres[nid].add_neighbor(uid)
        self.spheres[uid] = sphere
        return sphere

    def remove_sphere(self, uid: str):
        if uid not in self.spheres:
            return
        s = self.spheres[uid]
        for nid in s.neighbor_ids:
            if nid in self.spheres:
                self.spheres[nid].remove_neighbor(uid)
        s.remove()

        # 同时释放此球体作为缝隙填充时所对应的缝隙键
        # 使其可被重新填充
        stale = []
        for key in self._filled_gap_keys:
            if uid in key:
                stale.append(key)
        for key in stale:
            self._filled_gap_keys.discard(key)

    def seed_tetrahedron(self):
        """种子初态：4 个互相相切球体（正四面体）。

        4 个半径=1 的球体，两两相切，构成四面体。
        中心在(0,0,0)，顶点在单位球面上，距离=2（两倍半径）。

        这是阿波罗尼奥斯网络的最小种子：
            4 个两两相切的球体围成一个四面体空隙，
            索迪定理确定第 5 个球体的几何。
        """
        import math

        # 正四面体 4 个顶点（单位球面上）
        sqrt3 = math.sqrt(3.0)
        verts = [
            (0.0, 0.0, math.sqrt(2.0)),           # 顶部
            (2.0*math.sqrt(2.0)/3.0, 0.0, -math.sqrt(2.0)/3.0),  # 底部 1
            (-math.sqrt(2.0)/3.0, math.sqrt(6.0)/3.0, -math.sqrt(2.0)/3.0),  # 底部 2
            (-math.sqrt(2.0)/3.0, -math.sqrt(6.0)/3.0, -math.sqrt(2.0)/3.0),  # 底部 3
        ]

        # 归一化到单位球面 + 缩放使得球体半径=1
        # 当 4 个球体半径=1 且两两相切时，中心之间距离=2
        # 所以顶点从原点伸出的方向必须是 |v_i - v_j| = 2
        # 对于正四面体，|v_i - v_j| = 2 且 |v_i| = √(3/2) ≈ 1.225
        scale = math.sqrt(1.5)
        verts = [(v[0]*scale, v[1]*scale, v[2]*scale) for v in verts]

        # 创建 4 个种子球体
        for i in range(4):
            s = ApollonianSphere(uid=f"tet_{i:02d}")
            # 与所有其他 3 个球体相切
            for j in range(4):
                if i != j:
                    s.add_neighbor(f"tet_{j:02d}")
            self.spheres[s.uid] = s

        self.frame = 0
        self.history = []

    def seed_icosahedron(self):
        """正二十面体种子初态（备用）。

        12 个顶点，每个 deg=5。
        不产生四面体空隙（只有三角面），
        用于验证约束求解在完全连通图上的表现。
        """
        import math
        phi = (1.0 + math.sqrt(5.0)) / 2.0

        verts = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                verts.append((0.0, float(s1), float(s2 * phi)))
                verts.append((float(s1), float(s2 * phi), 0.0))
                verts.append((float(s2 * phi), 0.0, float(s1)))
        verts = verts[:12]

        def _norm(v):
            n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
            return (v[0]/n, v[1]/n, v[2]/n)
        verts = [_norm(vv) for vv in verts]

        for i in range(12):
            s = ApollonianSphere(uid=f"seed_{i:02d}")
            dists = []
            for j in range(12):
                if i == j: continue
                d = sum((verts[i][k] - verts[j][k])**2 for k in range(3))
                dists.append((d, f"seed_{j:02d}"))
            dists.sort()
            for _, nid in dists[:5]:
                s.add_neighbor(nid)
            self.spheres[s.uid] = s

        for uid in list(self.spheres.keys()):
            s = self.spheres[uid]
            for nid in list(s.neighbor_ids):
                if nid in self.spheres and uid not in self.spheres[nid].neighbor_ids:
                    self.spheres[nid].add_neighbor(uid)

        self.frame = 0
        self.history = []

    # ── 约束求解 ────────────────────────────────────────

    def _build_solved_state(self) -> SolvedState:
        """构建当前帧的已求解状态。

        为每个活跃球体求解几何（position, radius）。
        求解顺序：度数高的节点优先（它们有更多约束）。

        种子球体（tet_xx）使用硬编码几何作为公理起点
        — 这是系统唯一的外部输入，所有其他几何从此推导。
        """
        solved: SolvedState = {}

        # 0. 种子球体的预设几何（四面体公理）
        import math
        scale = math.sqrt(1.5)
        sqrt3 = math.sqrt(3.0)
        seed_positions = [
            (0.0, 0.0, math.sqrt(2.0)*scale),
            (2.0*math.sqrt(2.0)*scale/3.0, 0.0, -math.sqrt(2.0)*scale/3.0),
            (-math.sqrt(2.0)*scale/3.0, math.sqrt(6.0)*scale/3.0, -math.sqrt(2.0)*scale/3.0),
            (-math.sqrt(2.0)*scale/3.0, -math.sqrt(6.0)*scale/3.0, -math.sqrt(2.0)*scale/3.0),
        ]
        for i, uid in enumerate(["tet_00", "tet_01", "tet_02", "tet_03"]):
            if uid in self.spheres and self.spheres[uid].is_active:
                solved[uid] = (seed_positions[i], 1.0)

        # 1. 求解度数≥4的非种子节点
        high_deg = sorted(
            [uid for uid, s in self.spheres.items()
             if s.is_active and s.degree >= 4 and uid not in solved],
            key=lambda uid: self.spheres[uid].degree,
            reverse=True
        )

        remaining = set(high_deg)
        for _ in range(20):
            if not remaining:
                break
            newly = []
            for uid in list(remaining):
                s = self.spheres[uid]
                n_solved = sum(1 for nid in s.neighbor_ids if nid in solved)
                if n_solved >= 4:
                    geo = s.solve_geometry(solved)
                    if geo is not None:
                        solved[uid] = geo
                        newly.append(uid)
            for uid in newly:
                remaining.discard(uid)

        # 2. 求解剩余的活跃节点
        for uid, s in self.spheres.items():
            if s.is_active and uid not in solved:
                geo = s.solve_geometry(solved)
                if geo is not None:
                    solved[uid] = geo

        return solved

    # ── 缝隙检测与填充 ──────────────────────────────────

    def _find_gaps(self, solved: SolvedState) -> List[Tuple[str, str, str, str]]:
        """找到 4 个互相相切球体围成的缝隙（四面体空隙）。"""
        active = [uid for uid in self.spheres
                  if self.spheres[uid].is_active and uid in solved]
        if len(active) < 4:
            return []

        gaps = []
        checked = set()
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                for k in range(j + 1, len(active)):
                    for l in range(k + 1, len(active)):
                        key = tuple(sorted([active[i], active[j], active[k], active[l]]))
                        if key in checked:
                            continue
                        checked.add(key)

                        a, b, c, d = key
                        sa, sb, sc, sd = (self.spheres[x] for x in key)

                        # 检查是否两两相切
                        if (b in sa.neighbor_ids and c in sa.neighbor_ids
                            and d in sa.neighbor_ids
                            and c in sb.neighbor_ids and d in sb.neighbor_ids
                            and d in sc.neighbor_ids):
                            # 验证这 4 个球之间的所有 6 条边都存在
                            pairs = [(a, b), (a, c), (a, d),
                                     (b, c), (b, d), (c, d)]
                            if all(self.spheres[p[0]].is_active
                                   and self.spheres[p[1]].is_active
                                   and (p[1] in self.spheres[p[0]].neighbor_ids
                                        or p[0] == p[1])
                                   for p in pairs):
                                # 检查此缝隙是否已被填充
                                if key in self._filled_gap_keys:
                                    continue

                                gaps.append(key)
                                self._filled_gap_keys.add(key)

        return gaps

    def _fill_gap(self, gap: Tuple[str, str, str, str],
                  solved: SolvedState) -> Optional[str]:
        """填充一个四面体缝隙。

        索迪定理保证：给定 4 个两两相切球体，
        第 5 个球体的曲率是唯一确定的。
        """
        gap_id = f"gap_f{self.frame:04d}_{len(self.history):04d}"
        nids = list(gap)

        sphere = self.add_sphere(gap_id, neighbors=nids)

        # 验证求解
        geo = sphere.solve_geometry(solved)
        if geo is None:
            self.remove_sphere(gap_id)
            return None

        pos, r = geo
        if r <= 0 or math.isnan(r) or math.isinf(r):
            self.remove_sphere(gap_id)
            return None

        # 检查与 4 个壁的相切偏差
        for nid in nids:
            if nid in solved:
                n_pos, n_r = solved[nid]
                d = math.sqrt(sum((pos[k] - n_pos[k])**2 for k in range(3)))
                expected = r + n_r
                if abs(d - expected) / max(expected, 1e-8) > 0.5:
                    # 偏差过大：缝隙填充失败
                    self.remove_sphere(gap_id)
                    return None

        return gap_id

    # ── 帧演化 ──────────────────────────────────────────

    def step(self) -> FrameSnapshot:
        """单帧演化。"""
        self.frame += 1

        # 1. 约束求解
        solved = self._build_solved_state()

        # 2. 检测缝隙
        gaps = self._find_gaps(solved)

        # 3. 填充缝隙
        filled = []
        for gap in gaps:
            nid = self._fill_gap(gap, solved)
            if nid:
                filled.append(nid)

        # 4. 记录
        snap = self._record_frame(solved, len(filled))
        return snap

    def step_n(self, n: int) -> List[FrameSnapshot]:
        return [self.step() for _ in range(n)]

    # ── 空洞查询 ────────────────────────────────────────

    def probe_cavity(self, uid: str) -> Optional[Tuple[Tuple[float, float, float], float]]:
        """查询空洞几何——即使球体已被移除。"""
        if uid not in self.spheres:
            return None
        # 构建当前已求解状态
        solved = self._build_solved_state()
        sphere = self.spheres[uid]
        return sphere.get_cavity_geometry(solved)

    def demonstrate_cavity_persistence(self, target_uid: str) -> dict:
        """演示空洞持久性：移除球体前后对比。"""
        if target_uid not in self.spheres:
            return {"error": "球体不存在"}

        # 移除前的几何
        before = self.probe_cavity(target_uid)

        # 移除球体
        neighbors_before = list(self.spheres[target_uid].neighbor_ids)
        self.remove_sphere(target_uid)

        # 移除后的几何（空洞）
        after = self.probe_cavity(target_uid)

        return {
            "uid": target_uid,
            "neighbors": neighbors_before,
            "before_removal": before,    # (pos, r)
            "after_removal": after,      # (pos, r) — 空洞
            "cavity_persists": after is not None
        }

    # ── 记录与查询 ──────────────────────────────────────

    def _record_frame(self, solved: SolvedState, n_filled: int) -> FrameSnapshot:
        active = [uid for uid, s in self.spheres.items() if s.is_active]

        degs = [self.spheres[uid].degree for uid in active]
        edge_set = set()
        for uid in active:
            s = self.spheres[uid]
            for nid in s.neighbor_ids:
                if nid in self.spheres and self.spheres[nid].is_active:
                    edge_set.add(tuple(sorted([uid, nid])))

        snap = FrameSnapshot(
            frame=self.frame,
            n_active=len(active),
            n_removed=sum(1 for s in self.spheres.values() if not s.is_active),
            edge_count=len(edge_set),
            n_gaps_filled=n_filled,
            degree_hist=dict(Counter(degs)),
            solved=solved,
        )
        self.history.append(snap)
        return snap

    def summary(self) -> Dict:
        s = self.history[-1] if self.history else None
        if s is None:
            return {"frame": 0, "active": 0}
        return {
            "frame": s.frame,
            "active": s.n_active,
            "removed": s.n_removed,
            "edges": s.edge_count,
            "gaps_filled": s.n_gaps_filled,
        }

    def get_active(self) -> List[str]:
        return [uid for uid, s in self.spheres.items() if s.is_active]

    def clear(self):
        self.spheres.clear()
        self.frame = 0
        self.history.clear()

    def get_degree_distribution(self) -> Dict[int, int]:
        if self.history:
            return self.history[-1].degree_hist
        return {}
