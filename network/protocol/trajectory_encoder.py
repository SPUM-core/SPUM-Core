"""
TrajectoryEncoder — 推理轨迹编码为可序列化的拓扑签名

一个"轨迹"不是日志——它是 AI 在一轮推理中遍历的知识图谱路径。
每条边记录: 从哪个节点出发、到达哪个节点、使用了什么推导关系。

签名结构:
    T-{YYYYMMDD}-{counter}  (人类可读)
    内部编码为拓扑地址链: N001 → N002 → N005 → N022 → ...

轨迹文件 (.snap) 格式:
    # trajectory: T-20260714-001
    # created: 2026-07-14T12:00:00+00:00
    # model: gpt-4o
    # session: S-20260714-a1b2c3
    # origin: SPUM-core 项目浏览
    # === edge set ===
    N001 | N002 | derives_from | S-a1b2c3 | T-20260714-001 | 2026-07-14T12:00:00Z | 1.0 | canonical
    N002 | N003 | derives_from | S-a1b2c3 | T-20260714-001 | 2026-07-14T12:00:01Z | 1.0 | canonical
    N022 | SOC-001 | explains | S-a1b2c3 | T-20260714-001 | 2026-07-14T12:00:02Z | 0.85 | novel
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from .graph_store import GraphStore, TrajectoryEdge


# ============================================================
# 轨迹 ID 生成
# ============================================================

_trajectory_counter: int = 0


def _next_trajectory_id() -> str:
    """生成下一个轨迹 ID（线程安全由调用方保证）。"""
    global _trajectory_counter
    _trajectory_counter += 1
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"T-{today}-{_trajectory_counter:03d}"


def _compute_trajectory_hash(path: List[str]) -> str:
    """计算轨迹路径的 SHA256 指纹。"""
    raw = "→".join(path)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


# ============================================================
# Trajectory 数据类
# ============================================================

@dataclass
class Trajectory:
    """一条完整的推理轨迹。"""
    trajectory_id: str
    session_id: str
    model: str
    origin: str                     # 触发轨迹的上下文描述
    path: List[str]                 # 遍历的节点序列
    edges: List[TrajectoryEdge]     # 轨迹边
    signature: str = ""             # 拓扑签名（自动计算）
    metadata: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.signature:
            self.signature = _compute_trajectory_hash(self.path)

    def node_count(self) -> int:
        return len(set(self.path))

    def edge_count(self) -> int:
        return len(self.edges)


# ============================================================
# TrajectoryEncoder
# ============================================================

class TrajectoryEncoder:
    """推理轨迹编码器。"""

    def __init__(self, store: GraphStore):
        self.store = store
        self.store.load_canonical()

    # ------------------------------------------------------------
    # 编码
    # ------------------------------------------------------------

    def encode_path(self, path: List[str], session_id: str,
                    model: str, origin: str,
                    metadata: dict | None = None) -> Trajectory:
        """将节点路径编码为轨迹（自动推断边类型）。

        沿路径生成边，每条边的类型从 canonical edges.txt 推断。
        如果路径中的相邻节点在 canonical 中存在边，使用对应的 edge_type；
        否则标记为 'explains'（新边默认类型）。

        Args:
            path: 节点 ID 序列 (如 ["N001", "N002", "N005", "N022"])
            session_id: 会话 ID
            model: 模型名称
            origin: 轨迹触发上下文
            metadata: 附加元数据
        """
        trajectory_id = _next_trajectory_id()
        now = datetime.now(timezone.utc).isoformat()
        edges: List[TrajectoryEdge] = []
        canonical = self.store._canonical_edges

        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            # 先尝试直接方向
            edge_key = f"{source}|{target}|{edge_type}" if False else None

            # 从 canonical 查找边类型
            edge_type = "explains"  # 默认
            for et in ["derives_from", "requires", "refines", "explains", "drives"]:
                fwd = f"{source}|{target}|{et}"
                rev = f"{target}|{source}|{et}"
                if fwd in canonical or rev in canonical:
                    edge_type = et
                    break

            edge = TrajectoryEdge(
                source=source,
                target=target,
                edge_type=edge_type,
                session_id=session_id,
                trajectory_id=trajectory_id,
                timestamp=now,
                confidence=1.0,
            )
            edges.append(edge)

        metadata_dict = metadata or {}
        return Trajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            model=model,
            origin=origin,
            path=path,
            edges=edges,
            metadata=metadata_dict,
        )

    def encode_edges(self, edges: List[TrajectoryEdge],
                     session_id: str, model: str, origin: str,
                     metadata: dict | None = None) -> Trajectory:
        """直接编码一组预构建的边（手动构建）。"""
        trajectory_id = _next_trajectory_id()
        path: List[str] = []
        seen: Set[str] = set()
        for e in edges:
            if e.source not in seen:
                path.append(e.source)
                seen.add(e.source)
            if e.target not in seen:
                path.append(e.target)
                seen.add(e.target)

        metadata_dict = metadata or {}
        return Trajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            model=model,
            origin=origin,
            path=path,
            edges=edges,
            metadata=metadata_dict,
        )

    # ------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------

    def persist(self, trajectory: Trajectory) -> str:
        """将轨迹写入 GraphStore。"""
        path = self.store.write_trajectory(
            trajectory_id=trajectory.trajectory_id,
            edges=trajectory.edges,
            metadata={
                "session": trajectory.session_id,
                "model": trajectory.model,
                "origin": trajectory.origin,
                "path": " → ".join(trajectory.path),
                "signature": trajectory.signature,
                **trajectory.metadata,
            },
        )
        return path

    # ------------------------------------------------------------
    # 解码（反序列化）
    # ------------------------------------------------------------

    def decode(self, trajectory_id: str) -> Trajectory:
        """从 .snap 文件恢复轨迹。"""
        edges = self.store.read_trajectory(trajectory_id)
        if not edges:
            raise ValueError(f"轨迹为空: {trajectory_id}")

        # 从第一条边恢复元数据
        session_id = edges[0].session_id
        path: List[str] = []
        seen: Set[str] = set()
        for e in edges:
            if e.source not in seen:
                path.append(e.source)
                seen.add(e.source)
            if e.target not in seen:
                path.append(e.target)
                seen.add(e.target)

        # 从 .snap 文件头部恢复 metadata
        filepath = self.store.snapshot_dir / f"{trajectory_id}.snap"
        metadata: Dict[str, str] = {}
        model = "unknown"
        origin = ""
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("# model:"):
                        model = line.split(":", 1)[1].strip()
                    elif line.startswith("# origin:"):
                        origin = line.split(":", 1)[1].strip()
                    elif line.startswith("# ") and ":" in line:
                        k, v = line.strip("# ").split(":", 1)
                        metadata[k.strip()] = v.strip()

        return Trajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            model=model,
            origin=origin,
            path=path,
            edges=edges,
            metadata=metadata,
        )

    # ------------------------------------------------------------
    # 比较
    # ------------------------------------------------------------

    def diff(self, t1: Trajectory, t2: Trajectory) -> dict:
        """比较两条轨迹的差异。"""
        nodes1 = set(t1.path)
        nodes2 = set(t2.path)
        edges1 = {(e.source, e.target, e.edge_type) for e in t1.edges}
        edges2 = {(e.source, e.target, e.edge_type) for e in t2.edges}

        return {
            "common_nodes": nodes1 & nodes2,
            "nodes_only_in_first": nodes1 - nodes2,
            "nodes_only_in_second": nodes2 - nodes1,
            "common_edges": edges1 & edges2,
            "edges_only_in_first": edges1 - edges2,
            "edges_only_in_second": edges2 - edges1,
        }

    def fork(self, trajectory: Trajectory, new_path: List[str],
             origin: str) -> Trajectory:
        """从现有轨迹分叉出新轨迹。

        用于: AI-B 加载 AI-A 的轨迹后，沿路径继续推理。
        """
        return self.encode_path(
            path=new_path,
            session_id=trajectory.session_id,
            model=trajectory.model,
            origin=f"forked_from_{trajectory.trajectory_id}: {origin}",
            metadata={"parent_trajectory": trajectory.trajectory_id,
                      "parent_signature": trajectory.signature},
        )
