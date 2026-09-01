"""
青檬引擎 · 涌现检测层 — 五形相位检测器
========================================

五形（水木土金火）不是五种元素，而是 ⟨P, ε⟩ 网络在演化中必然涌现的
五种拓扑相位（spum-core.md 第六章）。本层从一帧快照的结构指标中
检测各相位的"活跃度"——为领域适配层提供统一的拓扑信号接口。

相位 → 图论指标:
  水  链式传输        最大无分支路径长度 · 高介数边占比
  木  闭合环骨架      环基数 μ = M − N + C · 全局聚类系数
  土  松散储备池      deg≤2 节点占比 · 碎片化指数 F = 1 − |maxCC|/N
  金  修剪目标        桥边数
  火  密度梯度驱动    Var(deg)（度数不均匀性 = ∇σ 代理）

返回统一为五维向量 S = (S_水, S_木, S_土, S_金, S_火) ∈ [0, 1]。
"""

from __future__ import annotations

from typing import Dict

import networkx as nx


class EmergenceDetector:
    """从一帧 ⟨P, ε⟩ 快照提取五形相位向量 S。"""

    # 各指标的经验归一化上界（使 S 分量落在 [0,1] 便于比较）
    _WATER_PATH_CAP = 16          # 最长链 ≥ 16 视为全水
    _WOOD_MU_CAP = 32             # 环基数 ≥ 32 视为全木
    _FIRE_VAR_CAP = 16.0          # 度数方差 ≥ 16 视为全火

    def detect(self, G: nx.Graph) -> Dict:
        """检测一帧的五形相位向量。

        Returns:
            {
                "vector": {"水": float, "木": float, "土": float, "金": float, "火": float},
                "metrics": {...各指标的原始值...},
            }
        """
        metrics = self._metrics(G)
        vec = {
            "水": self._water(metrics),
            "木": self._wood(metrics),
            "土": self._earth(metrics),
            "金": self._gold(metrics),
            "火": self._fire(metrics),
        }
        return {"vector": vec, "metrics": metrics}

    # ── 原始指标 ─────────────────────────────────────────────────

    def _metrics(self, G: nx.Graph) -> Dict:
        n = G.number_of_nodes()
        m = G.number_of_edges()
        if n == 0:
            return {"n": 0, "m": 0, "mu": 0, "bridges": 0, "low_deg_ratio": 0.0,
                    "frac_index": 0.0, "max_path": 0, "var_deg": 0.0}

        c = nx.number_connected_components(G)
        mu = m - n + c                       # 环基数（木形骨架量）
        bridges = len(list(nx.bridges(G)))   # 桥边数（金形目标）
        degs = [d for _, d in G.degree()]
        low_deg = sum(1 for d in degs if d <= 2)
        max_cc = max((len(c) for c in nx.connected_components(G)), default=0)

        return {
            "n": n,
            "m": m,
            "mu": mu,
            "bridges": bridges,
            "low_deg_ratio": low_deg / n,
            "frac_index": 1.0 - max_cc / n,     # 碎片化指数 F
            "max_path": self._longest_branch(G),
            "var_deg": sum((d - sum(degs) / n) ** 2 for d in degs) / n,
        }

    @staticmethod
    def _longest_branch(G: nx.Graph) -> int:
        """最长无分支路径近似：对度≤2 节点诱导子图求最长路径。"""
        sub = G.subgraph([v for v in G.nodes() if G.degree(v) <= 2])
        best = 0
        for c in nx.connected_components(sub):
            if len(c) == 1:
                best = max(best, 1)
                continue
            sg = sub.subgraph(c)
            # 对链状分量，长度 = 最长端点对路径（简化：分量内节点数）
            best = max(best, len(c))
        return best

    # ── 相位分量（归一化到 [0,1]） ───────────────────────────────

    def _water(self, mt: Dict) -> float:
        return min(1.0, mt["max_path"] / self._WATER_PATH_CAP)

    def _wood(self, mt: Dict) -> float:
        return min(1.0, mt["mu"] / self._WOOD_MU_CAP)

    def _earth(self, mt: Dict) -> float:
        return min(1.0, 0.5 * mt["low_deg_ratio"] + 0.5 * mt["frac_index"])

    def _gold(self, mt: Dict) -> float:
        e = max(mt["m"], 1)
        return min(1.0, mt["bridges"] / e * 6)   # 桥边占比放大系数 6

    def _fire(self, mt: Dict) -> float:
        return min(1.0, mt["var_deg"] / self._FIRE_VAR_CAP)

    def describe(self) -> str:
        return (
            "涌现检测器 | 五形相位 S=(水,木,土,金,火) | "
            "水=链 木=环 土=储备 金=桥 火=∇σ | "
            "从帧快照结构指标提取，输出统一 [0,1] 五维向量"
        )
