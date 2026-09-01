"""
青檬引擎 · 拓扑推理层 — 在 ⟨P, ε⟩ 上的结构分析
================================================

基于 NetworkX 的只读拓扑分析：路径、子图、连通性、指标计算。
本层不修改图——演化只属于 core.state（理论不可变的分层保障）。

关键语义：
  - 无全局图：操作对象是"某一帧的快照"（不变量 III 的观测面）
  - 连接优先于属性：查询从连接结构出发，不依赖节点属性
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

from ..core.axioms import Core


class Topology:
    """对一帧 ⟨P, ε⟩ 快照的只读拓扑分析器。"""

    def __init__(self, core: Core) -> None:
        self.core = core

    # ── 路径与连通性 ─────────────────────────────────────────────

    def shortest_path(self, G: nx.Graph, u: int, v: int) -> Optional[List[int]]:
        """最短路径（无权图 = 最少跳数）。"""
        if not (G.has_node(u) and G.has_node(v)):
            return None
        try:
            return nx.shortest_path(G, u, v)
        except nx.NetworkXNoPath:
            return None

    def diameter(self, G: nx.Graph) -> Optional[int]:
        """图直径（最大最短路径长度）。不连通时返回 None。"""
        if not nx.is_connected(G):
            return None
        return nx.diameter(G)

    def connected_components(self, G: nx.Graph) -> List[List[int]]:
        """连通分量列表（节点 id 已排序）。"""
        return sorted(
            (sorted(c) for c in nx.connected_components(G)),
            key=len,
            reverse=True,
        )

    # ── 结构指标 ─────────────────────────────────────────────────

    def degree_metrics(self, G: nx.Graph) -> Dict:
        """度数统计：均值、最小、分布。"""
        if G.number_of_nodes() == 0:
            return {"avg": 0.0, "min": 0, "max": 0, "distribution": {}}
        degs = [d for _, d in G.degree()]
        dist: Dict[int, int] = {}
        for d in degs:
            dist[d] = dist.get(d, 0) + 1
        return {
            "avg": 2.0 * G.number_of_edges() / G.number_of_nodes(),
            "min": min(degs),
            "max": max(degs),
            "distribution": dict(sorted(dist.items())),
        }

    def bridge_edges(self, G: nx.Graph) -> List[Tuple[int, int]]:
        """桥边（删除后连通分量数增加）——金形修剪的天然目标。"""
        return list(nx.bridges(G))

    def clustering(self, G: nx.Graph) -> float:
        """全局聚类系数——闭合环的密度指标（木形）。"""
        return nx.average_clustering(G)

    def edge_betweenness(self, G: nx.Graph) -> Dict[Tuple[int, int], float]:
        """边介数（高介数边 = 水形主干候选）。"""
        return nx.edge_betweenness_centrality(G)

    # ── 子图 ─────────────────────────────────────────────────────

    def induced_subgraph(self, G: nx.Graph, nodes: List[int]) -> nx.Graph:
        """由给定节点诱导的子图快照。"""
        present = [n for n in nodes if G.has_node(n)]
        return G.subgraph(present).copy()

    def summarize(self, G: nx.Graph) -> Dict:
        """一帧的结构摘要（推理轨迹快照用）。"""
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "sigma": self.core.sigma(G),
            "avg_degree": self.core.avg_degree(G),
            "components": len(list(nx.connected_components(G))),
            "dangling_density": self.core.dangling_density(G),
            "clustering": round(nx.average_clustering(G), 6) if G.number_of_nodes() else None,
        }
