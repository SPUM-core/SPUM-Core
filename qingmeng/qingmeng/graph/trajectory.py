"""
青檬引擎 · 推理轨迹层 — 观测性与版本回滚
==========================================

蓝图"观测性是生命线"：每条推理/触觉/演化输出必须携带
`trace_id`、`axiom_path`、`state_snapshots`、`consensus_status`，
供开发者审计；并支持按轨迹回滚引擎状态（版本回滚）。

SPUM 校准：
  - 轨迹是"帧快照序列"（不变量 III 的观测面），不是连续历史
  - 回滚 = 恢复到记录点的 ⟨P, ε⟩ + 账本 + 帧计数（不引入独立实体）
  - 只保存节点/边集合（无权二值，不变量 VI），不保存权重

用法:
    from qingmeng.graph import TrajectoryStore
    store = TrajectoryStore()
    before = store.snapshot_state(engine)
    result = engine.evolve()
    trace_id = store.record("evolve", "frame 5", engine, before=before)
    store.get(trace_id)          # → TraceRecord
    store.rollback(trace_id, engine)   # 恢复到该轨迹发生前
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

# 轨迹环缓冲上限——防止长跑把观测面撑爆
DEFAULT_MAX_RECORDS: int = 500


@dataclass
class TraceRecord:
    """一条推理轨迹的完整审计记录。"""

    trace_id: str
    kind: str                       # evolve / touch / reason
    input: str                      # 触发输入摘要（帧号/文本/prompt）
    frame: int                      # 记录时的帧计数
    axiom_path: List[str]           # 逐条不变量裁决，如 ["I:pass", "III:fail(...)"]
    state_snapshots: List[Dict]     # 结构快照（摘要 + 五形 + 账本）
    consensus: Dict                 # {passed, checks, failures}
    prev_state: Optional[Dict]      # 回滚点：{nodes, edges, frame, created, annihilated}
    seq: int = 0

    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "kind": self.kind,
            "input": self.input,
            "frame": self.frame,
            "axiom_path": self.axiom_path,
            "state_snapshots": self.state_snapshots,
            "consensus": self.consensus,
            "prev_state": {
                "nodes": len(self.prev_state["nodes"]),
                "edges": len(self.prev_state["edges"]),
                "frame": self.prev_state["frame"],
            } if self.prev_state else None,
        }


class TrajectoryStore:
    """轨迹仓库——记录、查询、回滚。

    引擎唯一状态源；轨迹是只读观测面。回滚通过重建 ⟨P, ε⟩ 完成，
    不改动任何其他实体（不变量 I）。
    """

    def __init__(self, max_records: int = DEFAULT_MAX_RECORDS) -> None:
        self.max_records = max_records
        self._records: List[TraceRecord] = []
        self._seq = 0

    # ── 记录 ─────────────────────────────────────────────────────

    @staticmethod
    def snapshot_state(engine: Any) -> Dict:
        """记录引擎当前状态（回滚点）——节点/边集合 + 账本 + 帧。"""
        G = engine.core.graph
        ledger = dict(engine.state.ledger)
        return {
            "nodes": list(G.nodes()),
            "edges": list(G.edges()),
            "frame": engine.state.frame,
            "created": ledger.get("total_created", 0),
            "annihilated": ledger.get("total_annihilated", 0),
        }

    def record(
        self,
        kind: str,
        input_: str,
        engine: Any,
        before: Optional[Dict] = None,
    ) -> str:
        """记录一条轨迹，返回 trace_id。

        Args:
            kind: evolve / touch / reason
            input_: 触发输入摘要
            engine: 引擎实例（取图/账本/共识）
            before: 操作前快照（snapshot_state 产出），供回滚
        """
        report = engine.check()
        self._seq += 1
        rec = TraceRecord(
            trace_id=uuid.uuid4().hex[:10],
            kind=kind,
            input=input_,
            frame=engine.state.frame,
            axiom_path=[
                f"{c['invariant']}:{'pass' if c['passed'] else 'fail(' + c['detail'][:40] + ')'}"
                for c in report.checks
            ],
            state_snapshots=[{
                "frame": engine.state.frame,
                "summary": engine.summarize(),
                "wuxing": engine.emergence().get("vector", {}),
                "ledger": dict(engine.state.ledger),
            }],
            consensus=report.to_dict(),
            prev_state=before,
            seq=self._seq,
        )
        self._records.append(rec)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records:]
        return rec.trace_id

    # ── 查询 ─────────────────────────────────────────────────────

    def get(self, trace_id: str) -> Optional[TraceRecord]:
        return next((r for r in self._records if r.trace_id == trace_id), None)

    def recent(self, n: int = 20) -> List[TraceRecord]:
        return self._records[-n:]

    def __len__(self) -> int:
        return len(self._records)

    # ── 版本回滚 ─────────────────────────────────────────────────

    def rollback(self, trace_id: str, engine: Any) -> Optional[TraceRecord]:
        """将引擎恢复到指定轨迹发生前的状态。

        重建图（节点/边集合）→ 恢复帧计数与对偶账本 → 截断历史快照。
        轨迹记录本身保留（观测面不可被回滚抹除）。
        """
        rec = self.get(trace_id)
        if rec is None or rec.prev_state is None:
            return None
        prev = rec.prev_state
        G: nx.Graph = engine.core.graph
        G.clear()
        G.add_nodes_from(prev["nodes"])
        G.add_edges_from(prev["edges"])
        engine.state.restore(
            frame=prev["frame"],
            created_total=prev["created"],
            annihilated_total=prev["annihilated"],
        )
        return rec

    def reset(self) -> None:
        self._records.clear()
        self._seq = 0
