"""
GraphStore — 持久化边集存储

职责:
    1. 读取 canonical edges.txt 作为合法边类型基准
    2. 读取/write trajectory 快照文件（.snap 格式）
    3. 验证写入边是否符合 edges.txt 定义的边类型集
    4. 合并多条轨迹 → 统一拓扑快照

存储结构:
    network/
    ├── edges.txt                ← 规范边集（33 核心边 + 15 模块桥接边）
    └── protocol/
        ├── graph_store.py       ← 本文件
        ├── snapshots/           ← 轨迹快照（.snap）
        │   ├── T-20260714-001.snap
        │   └── T-20260714-002.snap
        └── merged/              ← 合并后的全局快照
            └── current.snap
"""

from __future__ import annotations
import csv
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ============================================================
# 数据类型
# ============================================================

@dataclass(frozen=True)
class CanonicalEdge:
    """规范边（来自 edges.txt）"""
    source: str
    target: str
    edge_type: str   # derives_from / requires / refines / explains / drives
    description: str = ""

    def key(self) -> str:
        return f"{self.source}|{self.target}|{self.edge_type}"

    def reversed(self) -> bool:
        """如果边在 edges.txt 中是反向存储的，返回 True。
        例如 edges.txt 存的是 N013 | N009 | drives，则 source=N013 target=N009 为标准方向。
        """
        return False


@dataclass(frozen=True)
class TrajectoryEdge:
    """轨迹边（由 AI 推理产生）"""
    source: str          # 起始节点 ID
    target: str          # 目标节点 ID
    edge_type: str       # 边类型（与 canonical 一致）
    session_id: str      # 创建该边的会话 ID
    trajectory_id: str   # 所属轨迹 ID
    timestamp: str       # ISO 8601
    confidence: float = 1.0   # 推理置信度 (0.0-1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_line(self) -> str:
        return (f"{self.source} | {self.target} | {self.edge_type} "
                f"| {self.session_id} | {self.trajectory_id} "
                f"| {self.timestamp} | {self.confidence}")

    @classmethod
    def from_line(cls, line: str) -> Optional["TrajectoryEdge"]:
        parts = [p.strip() for p in line.strip().split("|")]
        if len(parts) < 7:
            return None
        try:
            return cls(
                source=parts[0],
                target=parts[1],
                edge_type=parts[2],
                session_id=parts[3],
                trajectory_id=parts[4],
                timestamp=parts[5],
                confidence=float(parts[6]),
            )
        except (ValueError, IndexError):
            return None


# ============================================================
# 边类型验证集
# ============================================================

VALID_EDGE_TYPES: Set[str] = {
    "derives_from", "requires", "refines", "explains", "drives",
}

# 节点 ID 格式: N001-N022, SOC-*, ECON-*, LING-*, FI-*, AGT-*, VSPT-*, R*
NODE_ID_PATTERN = re.compile(r"^[A-Z]+-\d+$|^N\d{3}$|^R\d$")


def is_valid_node_id(node_id: str) -> bool:
    """验证节点 ID 格式合法性。"""
    return bool(NODE_ID_PATTERN.match(node_id))


# ============================================================
# GraphStore
# ============================================================

class GraphStore:
    """持久化边集存储。"""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.canonical_path = self.base_dir / "edges.txt"
        self.snapshot_dir = self.base_dir / "protocol" / "snapshots"
        self.merged_dir = self.base_dir / "protocol" / "merged"

        # 运行时缓存
        self._canonical_edges: Dict[str, CanonicalEdge] = {}
        self._canonical_loaded: bool = False

    # ------------------------------------------------------------
    # 规范边加载
    # ------------------------------------------------------------

    def load_canonical(self) -> Dict[str, CanonicalEdge]:
        """加载 edges.txt 作为合法边类型基准。"""
        if self._canonical_loaded:
            return self._canonical_edges

        path = self.canonical_path
        if not path.exists():
            raise FileNotFoundError(
                f"规范边文件不存在: {path}\n"
                f"SPUM 网络协议依赖 network/edges.txt 作为合法边类型基准。"
            )

        edges: Dict[str, CanonicalEdge] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if not line or line.startswith("#"):
                    continue

                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 3:
                    continue

                source, target, edge_type = parts[0], parts[1], parts[2]
                description = parts[3] if len(parts) > 3 else ""

                if edge_type not in VALID_EDGE_TYPES:
                    continue

                edge = CanonicalEdge(
                    source=source,
                    target=target,
                    edge_type=edge_type,
                    description=description,
                )
                edges[edge.key()] = edge

                # 反向边也注册（允许双向遍历）
                reverse_key = f"{target}|{source}|{edge_type}"
                if reverse_key not in edges:
                    edges[reverse_key] = edge

        self._canonical_edges = edges
        self._canonical_loaded = True
        print(f"[GraphStore] 已加载 {len(edges)} 条规范边（含反向）")
        return edges

    def validate_edge(self, source: str, target: str, edge_type: str) -> Tuple[bool, str]:
        """验证一条边是否合法。

        Returns:
            (is_valid, reason)
        """
        if edge_type not in VALID_EDGE_TYPES:
            return False, f"非法边类型: {edge_type}，合法类型: {VALID_EDGE_TYPES}"

        if not is_valid_node_id(source) and not source.startswith("AGT-"):
            return False, f"非法源节点 ID: {source}"

        if not is_valid_node_id(target) and not target.startswith("AGT-"):
            return False, f"非法目标节点 ID: {target}"

        # 检查 canonical 中是否存在
        key = f"{source}|{target}|{edge_type}"
        if not self._canonical_loaded:
            self.load_canonical()

        if key in self._canonical_edges:
            return True, "canonical"

        # 反向 canonical 检查
        reverse_key = f"{target}|{source}|{edge_type}"
        if reverse_key in self._canonical_edges:
            return True, "canonical_reversed"

        # 不在 canonical 中 → 新边（合法但标记为 novel）
        return True, "novel"

    # ------------------------------------------------------------
    # 轨迹快照读写
    # ------------------------------------------------------------

    def write_trajectory(self, trajectory_id: str, edges: List[TrajectoryEdge],
                         metadata: dict | None = None) -> str:
        """将一组轨迹边写入 .snap 文件。

        Args:
            trajectory_id: 轨迹 ID (如 "T-20260714-001")
            edges: 轨迹边列表
            metadata: 附加元数据 (session_id, model, 描述等)

        Returns:
            写入的文件路径
        """
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # 验证所有边
        for e in edges:
            self.validate_edge(e.source, e.target, e.edge_type)

        filepath = self.snapshot_dir / f"{trajectory_id}.snap"

        with open(filepath, "w", encoding="utf-8") as f:
            # 元数据头
            f.write(f"# trajectory: {trajectory_id}\n")
            f.write(f"# created: {datetime.now(timezone.utc).isoformat()}\n")
            if metadata:
                for k, v in metadata.items():
                    f.write(f"# {k}: {v}\n")
            f.write("# === edge set ===\n")

            # 边集
            for e in edges:
                is_valid, reason = self.validate_edge(e.source, e.target, e.edge_type)
                status = "canonical" if "canonical" in reason else "novel"
                f.write(f"{e.to_line()} | {status}\n")

        print(f"[GraphStore] 已写入轨迹: {filepath} ({len(edges)} 条边)")
        return str(filepath)

    def read_trajectory(self, trajectory_id: str) -> List[TrajectoryEdge]:
        """从 .snap 文件读取轨迹边。"""
        filepath = self.snapshot_dir / f"{trajectory_id}.snap"
        if not filepath.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {filepath}")

        edges: List[TrajectoryEdge] = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 去掉尾部状态标记
                clean = line.rsplit("|", 1)[0].strip()
                edge = TrajectoryEdge.from_line(clean)
                if edge:
                    edges.append(edge)

        return edges

    def list_trajectories(self) -> List[str]:
        """列出所有轨迹快照 ID。"""
        if not self.snapshot_dir.exists():
            return []
        files = sorted(self.snapshot_dir.glob("*.snap"))
        return [f.stem for f in files]

    # ------------------------------------------------------------
    # 合并
    # ------------------------------------------------------------

    def merge_all(self, output_name: str = "current") -> str:
        """合并所有轨迹快照为一个统一拓扑快照。"""
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        trajectories = self.list_trajectories()

        if not trajectories:
            print("[GraphStore] 无轨迹可合并")
            return ""

        # 按 session_id 去重（保留置信度最高的版本）
        seen: Set[Tuple[str, str, str, str]] = set()
        merged: List[TrajectoryEdge] = []

        for tid in trajectories:
            edges = self.read_trajectory(tid)
            for e in edges:
                dedup_key = (e.source, e.target, e.edge_type, e.session_id)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    merged.append(e)

        filepath = self.merged_dir / f"{output_name}.snap"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# merged snapshot: {output_name}\n")
            f.write(f"# created: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"# source_trajectories: {len(trajectories)}\n")
            f.write(f"# total_edges: {len(merged)}\n")
            f.write("# === edge set ===\n")
            for e in merged:
                is_valid, reason = self.validate_edge(e.source, e.target, e.edge_type)
                status = "canonical" if "canonical" in reason else "novel"
                f.write(f"{e.to_line()} | {status}\n")

        print(f"[GraphStore] 已合并 {len(trajectories)} 条轨迹 → {filepath} ({len(merged)} 条去重边)")
        return str(filepath)

    def get_merged_edge_count(self) -> int:
        """获取合并后的边数。"""
        current = self.merged_dir / "current.snap"
        if not current.exists():
            return 0
        count = 0
        with open(current, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    count += 1
        return count

    # ------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------

    def stats(self) -> dict:
        """返回存储统计信息。"""
        trajectories = self.list_trajectories()
        total_traj_edges = 0
        for tid in trajectories:
            total_traj_edges += len(self.read_trajectory(tid))

        merged_count = self.get_merged_edge_count()

        return {
            "canonical_edges": len(self._canonical_edges),
            "trajectory_count": len(trajectories),
            "total_trajectory_edges": total_traj_edges,
            "merged_edges": merged_count,
            "snapshot_dir": str(self.snapshot_dir),
            "merged_dir": str(self.merged_dir),
        }
