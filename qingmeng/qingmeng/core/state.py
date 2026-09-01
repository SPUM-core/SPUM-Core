"""
青檬引擎 · 状态演化层 — 五步帧状态机
======================================

SPUM 的"时间"是离散帧计数，不是连续流（不变量 X：连续是粗粒化）。
一个完整帧 = 创生 → 连接 → 变化体积 → 判断悬挂 → 删除悬挂。

约束（spum-core.md §2.1）：
  - 一帧内没有循环，没有级联消解——一帧只做一次删除
  - 删除后新产生的悬挂边，是下一帧处理的起点
  - 帧与帧之间没有中间状态——节点要么连接，要么不连接

不变量 II（总边数守恒 · 创生-湮灭对偶）：
  每个创生必是某湮灭的对偶补偿——关系容量重分配，不是无中生有。
  本模块逐帧记账，累计净漂移可持续审计（见 core.consensus）。

用法:
    from qingmeng.core import Core, StateEvolutionEngine
    eng = StateEvolutionEngine(Core())
    snap = eng.evolve()      # 推进一帧
    eng.ledger              # → {'total_created': ..., 'total_annihilated': ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .axioms import Core


# 单帧创生上限（dv/dt ≤ const 在演化层的事务化体现：单帧关系容量上限）
CREATION_MAX_PER_FRAME: int = 2


@dataclass
class FrameSnapshot:
    """单帧状态快照——图状态是帧序列的快照（不变量 III 的观测面）。"""

    frame_id: int
    node_count: int
    edge_count: int
    dangling_count: int
    dangling_density: float
    sigma: float
    created: int = 0                    # 本帧创生边数 V⁺
    annihilated: int = 0                # 本帧删除边数 V⁻
    steps_done: Tuple[bool, ...] = (False,) * 5

    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "dangling_count": self.dangling_count,
            "dangling_density": round(self.dangling_density, 6),
            "sigma": round(self.sigma, 6),
            "duality": {"created": self.created, "annihilated": self.annihilated},
            "steps": {
                "1_创生V+": self.steps_done[0],
                "2_连接": self.steps_done[1],
                "3_变化体积": self.steps_done[2],
                "4_判断悬挂": self.steps_done[3],
                "5_删除V-": self.steps_done[4],
            },
        }


class StateEvolutionEngine:
    """五步帧状态机。

    确定性演化（不变量 VII：约束驱动，非随机）：每帧对悬挂端执行
    最小修补——创生一条边将悬挂端连接到度数最低的非悬挂节点。
    若图无悬挂端（如初始 12 晶子闭合态），帧停滞（闭合子图创生抑制）。
    """

    def __init__(self, core: Core) -> None:
        self.core = core
        self.frame: int = 0
        self._created_total = 0
        self._annihilated_total = 0
        self._history: List[FrameSnapshot] = []

    # ── 账本（不变量 II 审计面） ─────────────────────────────────

    @property
    def ledger(self) -> Dict:
        """累计创生/湮灭 + 净漂移。净漂移单向持续增长 = 对偶失衡。"""
        return {
            "total_created": self._created_total,
            "total_annihilated": self._annihilated_total,
            "net_drift": self._created_total - self._annihilated_total,
        }

    # ── 五步帧 ───────────────────────────────────────────────────

    def evolve(self) -> FrameSnapshot:
        """推进一帧：创生 → 连接 → 变化体积 → 判断悬挂 → 删除。"""
        G = self.core.graph
        steps = [False] * 5
        created = 0
        annihilated = 0

        # ── 1. 创生 V⁺：检测悬挂端并补边（确定性策略） ──────────
        dangling = self.core.dangling_nodes(G)
        if dangling:
            for node in dangling[:CREATION_MAX_PER_FRAME]:
                # 候选：非悬挂、度数最低的非悬挂节点（约束驱动）
                candidates = sorted(
                    (n for n in G.nodes() if n != node and G.degree(n) >= 2),
                    key=lambda n: G.degree(n),
                )
                if candidates and not G.has_edge(node, candidates[0]):
                    G.add_edge(node, candidates[0])
                    created += 1
            steps[0] = True

        # ── 2. 连接：度数重算（无中间状态——图已更新，无额外操作） ──
        steps[1] = True

        # ── 3. 变化体积：σ 局部波动（快照时体现，此处仅标记） ──
        steps[2] = True

        # ── 4. 判断悬挂：统计当前悬挂端 ──
        dangling_after = self.core.dangling_nodes(G)
        steps[3] = True

        # ── 5. 删除 V⁻：一帧只做一次删除（不级联，残余留给下一帧） ──
        #    （初始骨架无悬挂端时，本步为空操作——闭合子图创生抑制）
        dead: List[Tuple[int, int]] = []
        for n in list(dangling_after):
            nbrs = list(G.neighbors(n))
            if not nbrs:
                continue
            dead.append((n, nbrs[0]))
            G.remove_edge(n, nbrs[0])
        # 删除后产生的孤立节点（deg=0）一并移除——度 0 即孤立，逻辑不自洽
        for n in list(G.nodes()):
            if G.degree(n) == 0:
                G.remove_node(n)
        annihilated = len(dead)
        steps[4] = True

        # ── 记账（不变量 II） ──
        self._created_total += created
        self._annihilated_total += annihilated

        snap = FrameSnapshot(
            frame_id=self.frame,
            node_count=G.number_of_nodes(),
            edge_count=G.number_of_edges(),
            dangling_count=len(dangling_after),
            dangling_density=self.core.dangling_density(G),
            sigma=self.core.sigma(G),
            created=created,
            annihilated=annihilated,
            steps_done=tuple(steps),
        )
        self._history.append(snap)
        self.frame += 1
        return snap

    # ── 只读访问 ─────────────────────────────────────────────────

    @property
    def history(self) -> List[FrameSnapshot]:
        return self._history

    def snapshot(self, frame_id: Optional[int] = None) -> Optional[FrameSnapshot]:
        """取某帧快照（默认最新）。"""
        if not self._history:
            return None
        if frame_id is None:
            return self._history[-1]
        return next((s for s in self._history if s.frame_id == frame_id), None)

    # ── 版本回滚（配合 TrajectoryStore.rollback 使用） ───────────

    def restore(self, frame: int, created_total: int, annihilated_total: int) -> None:
        """恢复到指定帧计数与对偶账本，并截断历史快照。

        只还原演化状态——图结构由调用方（轨迹回滚）重建。
        """
        self.frame = frame
        self._created_total = created_total
        self._annihilated_total = annihilated_total
        self._history = [h for h in self._history if h.frame_id < frame]

    def describe(self) -> str:
        return (
            "五步帧状态机 | 创生→连接→变化体积→判断悬挂→删除 | "
            "一帧只删一次，残余留给下一帧 | 创生-湮灭对偶记账"
        )
