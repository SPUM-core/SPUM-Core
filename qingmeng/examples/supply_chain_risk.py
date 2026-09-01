"""
青檬引擎 · 示例：供应链风险传导模拟
=====================================

高价值场景演示（蓝图冷启动建议）：将供应链建模为关系网络，
故障（V⁻）沿拓扑路径传导——风险传播不是"力"或"概率"，而是
连接结构的必然响应。

拓扑映射（SPUM 校准）：
  - 企业 = 节点；供应关系 = 二值边（不变量 VI）
  - 层间耦合 = InteractionBoundary.coupling（拓扑距离 × 密度）
  - 风险传导强度 = propagate（沿最短路径衰减）
  - 结论可审计：共识校验 + 五形向量 + 轨迹 trace_id

运行（在 qingmeng/ 目录下）:
    python examples/supply_chain_risk.py
"""

from __future__ import annotations

import os
import sys

import networkx as nx

# 允许直接从 examples/ 目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qingmeng import QingmengEngine
from qingmeng.core.axioms import CORE_SIZE


def build_supply_chain(engine: QingmengEngine) -> dict:
    """在核心 12 晶子之上挂接供应链子图，返回节点 id 映射。"""
    G = engine.core.graph
    nid = max(G.nodes()) + 1

    ids: dict[str, int] = {}
    for label in ("S1", "S2", "M1", "M2", "D1", "D2", "R1", "R2"):
        ids[label] = nid
        nid += 1

    # 供应链骨架：供应商 → 制造商 → 分销商 → 零售商
    chain = [
        ("S1", "M1"), ("S2", "M1"), ("S2", "M2"),   # 双源供应（冗余）
        ("M1", "D1"), ("M1", "D2"), ("M2", "D1"),   # 制造商多分销
        ("D1", "R1"), ("D2", "R1"), ("D2", "R2"),   # 分销商多零售
    ]
    G.add_edges_from((ids[a], ids[b]) for a, b in chain)
    return ids


def main() -> int:
    eng = QingmengEngine()
    ids = build_supply_chain(eng)
    G = eng.core.graph

    # ── 1. 层间耦合（边界效应） ────────────────────────────────
    print("══ 供应链层间耦合（InteractionBoundary）══")
    layers = {"供应商": ["S1", "S2"], "制造商": ["M1", "M2"],
              "分销商": ["D1", "D2"], "零售商": ["R1", "R2"]}
    names = list(layers)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = eng.coupling([ids[x] for x in layers[names[i]]],
                             [ids[x] for x in layers[names[j]]])
            print(f"  {names[i]} ↔ {names[j]}: 耦合 = {c:.3f}")

    # ── 2. 故障注入：S1 断供（V⁻） ─────────────────────────────
    print("\n══ 故障注入：供应商 S1 断供（V⁻）══")
    # V⁻ 事件与 LLM 桥一致：直接操作二值边（不变量 VI）；
    # 对偶账本由演化帧记账（与 reason/touch 相同的记账约定）
    before = nx.shortest_path_length(G, ids["R1"])
    print(f"  故障前  R1 → S1 最短路径长度 = {before.get(ids['S1'])}")

    G.remove_edge(ids["S1"], ids["M1"])      # 断供
    # S1 断供后 deg=0 —— 度 0 即孤立（逻辑不自洽）：按帧规则隔离移除
    G.remove_node(ids["S1"])
    print(f"  故障后  S1 已隔离（deg=0 → 移除），供应链子图节点={len(ids)}→{len(ids)-1}")

    # 风险传导强度：以受冲击的制造商 M1 为震源，向下游传播
    for target in ("D1", "R1"):
        s = eng.boundary.propagate(G, ids["M1"], ids[target])
        reach = "可达" if s else "不可达"
        print(f"  风险传导 M1 → {target}: 强度 = {s if s else '—'}（{reach}）")

    after = nx.shortest_path_length(G, ids["R1"])
    alt = after.get(ids["S2"])               # 冗余备选路径
    print(f"  故障后  R1 → S2（备选供应）长度 = {alt}（冗余路径仍连通）")

    # ── 3. 共识 + 五形 + 轨迹审计 ─────────────────────────────
    print("\n══ 审计 ══")
    report = eng.check()
    print(f"  共识: {report.passed}（{len(report.checks)} 项检查，"
          f"{len(report.failures)} 项违规）")
    wuxing = eng.emergence().get("vector", {})
    dom = max(wuxing, key=wuxing.get)
    print(f"  五形: { {k: round(v, 3) for k, v in wuxing.items()} }（{dom} 占优）")

    # 最近一条轨迹（观测性生命线——示例手动记录，运行路径走 evolve/touch/reason 自动记录）
    trace_id = eng.trajectory.record("example", "supply_chain_risk", eng)
    rec = eng.trajectory.get(trace_id)
    print(f"  trace_id: {rec.trace_id} | kind={rec.kind} | "
          f"axiom_path={rec.axiom_path}")

    # 风险传导结论（图结构裁决）
    reached = eng.boundary.propagate(G, ids["M1"], ids["R1"])
    print(f"\n结论: 供应商 S1 故障 {'仍可经 M1 传导至零售商 R1（强度 ' + str(reached) + '）' if reached else '已被隔离，无法传导至零售商 R1'} "
          f"——双源冗余将风险限制在局部。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
