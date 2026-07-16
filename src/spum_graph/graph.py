"""
帧内图结构 — 实现 SPUM-图论 5 条核心公理

公理 1: 连接优先于属性 — 节点身份由边集定义，无孤立节点
公理 2: 悬挂端不可消除（有限帧内）— N013 的图论表达
公理 3: 图状态是帧序列的快照 — 无全局图，只有帧状态
公理 4: 完美是极限，不是状态 — δ>0 对任意有限帧
公理 5: 局部闭合充分，全局暂态矛盾允许

用法:
    graph = FrameGraph(frame_id=1)
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B")
    print(graph.degrees())  # {"A": 1, "B": 1}
    print(graph.is_dangling("A"))  # True (degree < 2)
"""

from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, Iterator


class FrameGraph:
    """单帧内的图状态。符合 SPUM-图论公理 1-5。

    关键约束:
        - 无孤立节点：add_node 不暴露给外部，节点必须通过 add_edge 隐式创建
        - 每帧独立：不跨帧缓存任何状态
        - 悬挂端是正常状态：不自动删除，留给调用方按帧协议处理
    """

    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self._neighbors: Dict[str, Set[str]] = defaultdict(set)
        self._edge_count: int = 0

    # ── 公理 1: 连接优先于属性 ──────────────────────────────────

    def add_edge(self, u: str, v: str) -> None:
        """添加无向边。节点由边隐式定义——不存在独立的"创建节点"操作。

        公理1: 孤立节点无法被确认存在，连接是对存在的确认。
        """
        if u == v:
            raise ValueError(f"SPUM-图论不允许自环: {u}→{u}")
        self._neighbors[u].add(v)
        self._neighbors[v].add(u)
        self._edge_count += 1

    def has_node(self, node_id: str) -> bool:
        """节点是否存在取决于它是否有连接。"""
        return node_id in self._neighbors

    # ── 度数 ────────────────────────────────────────────────────

    def degree(self, node_id: str) -> int:
        """返回节点度数。未连接的节点视为不存在（返回 0）。"""
        return len(self._neighbors.get(node_id, set()))

    def degrees(self) -> Dict[str, int]:
        """返回所有节点的度数映射。"""
        return {n: len(adj) for n, adj in self._neighbors.items()}

    # ── 公理 2: 悬挂端检测 ─────────────────────────────────────

    def is_dangling(self, node_id: str) -> bool:
        """度数 < 2 的节点 = 悬挂端。
        公理2: 对任意有限帧，悬挂端集合基数 > 0。
        """
        d = self.degree(node_id)
        return d < 2 if node_id in self._neighbors else False

    def dangling_nodes(self) -> Set[str]:
        """返回当前帧所有悬挂端节点。"""
        return {n for n in self._neighbors if len(self._neighbors[n]) < 2}

    def dangling_density(self) -> float:
        """悬挂端密度 δ = |悬挂端| / |总节点|

        公理4: δ>0 对任意有限帧——完美是极限非状态。
        此方法返回的是当前帧的瞬时 δ 值，不是全局属性。
        """
        n = len(self._neighbors)
        if n == 0:
            return 0.0
        return len(self.dangling_nodes()) / n

    # ── 基础属性 ────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return len(self._neighbors)

    @property
    def edge_count(self) -> int:
        return self._edge_count

    def neighbors(self, node_id: str) -> Set[str]:
        return self._neighbors.get(node_id, set())

    def nodes(self) -> Iterator[str]:
        return iter(self._neighbors)

    # ── 公理 3: 帧快照 ─────────────────────────────────────────

    def snapshot(self) -> Dict:
        """导出当前帧状态的不可变快照。

        公理3: 图状态是帧序列的快照。每帧独立，不跨帧缓存任何状态。
        snapshot 导出的是当前帧张力场的几何快照——不是"全局图"的结构摘要。
        """
        return {
            "frame_id": self.frame_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "dangling_count": len(self.dangling_nodes()),
            "dangling_density": self.dangling_density(),
            "degrees": self.degrees(),
        }

    def __repr__(self) -> str:
        return (f"FrameGraph(frame={self.frame_id}, "
                f"nodes={self.node_count}, edges={self.edge_count}, "
                f"dangling={len(self.dangling_nodes())})")
