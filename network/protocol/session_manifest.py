"""
SessionManifest — 会话元数据 + 跨实例追踪

每个 AI 会话（加载 SPUM 规则并与用户交互的实例）生成一份 Manifest。
Manifest 记录了:
    - 该会话的身份 (session_id = 拓扑地址)
    - 它继承了哪些前序轨迹的边集
    - 它贡献了哪些新边
    - 它的推理路径
    - 分叉关系（哪个父轨迹产生了这个会话）

Manifest 存储在 network/protocol/manifests/ 目录，供后续 AI 实例加载。
"""

from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set


# ============================================================
# 会话 ID 生成（拓扑地址风格）
# ============================================================

_session_counter: int = 0


def _generate_session_id(model: str) -> str:
    """生成全局唯一的会话 ID。

    格式: S-{模型前缀}-{日期}-{计数}
    示例: S-AGNT-20260714-001

    使用单调递增计数器 + 模型前缀 + 日期，保证确定性。
    与 TopologicalAddress.differentiate_from 风格一致。
    """
    global _session_counter
    _session_counter += 1
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    # 模型缩写
    prefix = model[:4].upper() if model else "UNKN"
    return f"S-{prefix}-{today}-{_session_counter:03d}"


def _compute_manifest_hash(data: dict) -> str:
    """计算 Manifest 内容的 SHA256 指纹（用于完整性验证）。"""
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ============================================================
# SessionManifest
# ============================================================

@dataclass
class SessionManifest:
    """会话元数据。"""

    session_id: str
    model: str
    origin: str                          # 触发会话的上下文
    created_at: str

    # 继承信息
    parent_trajectories: List[str] = field(default_factory=list)   # 加载了哪些前序轨迹
    loaded_edges: int = 0                                          # 继承了多少条边

    # 贡献信息
    edges_written: int = 0                   # 写入了多少条新边
    trajectories_created: List[str] = field(default_factory=list)  # 生成了哪些轨迹

    # 推理路径快照
    inference_path: List[str] = field(default_factory=list)        # 遍历的节点序列
    path_depth: int = 0                                            # 路径长度

    # 完整性
    hash: str = ""

    def __post_init__(self):
        if not self.hash:
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        d = asdict(self)
        d.pop("hash", None)
        return _compute_manifest_hash(d)

    # ------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SessionManifest":
        return cls(**d)

    # ------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------

    @classmethod
    def create(cls, model: str, origin: str,
               parent_trajectories: List[str] | None = None) -> "SessionManifest":
        session_id = _generate_session_id(model)
        return cls(
            session_id=session_id,
            model=model,
            origin=origin,
            created_at=datetime.now(timezone.utc).isoformat(),
            parent_trajectories=parent_trajectories or [],
        )


# ============================================================
# ManifestStore
# ============================================================

class ManifestStore:
    """Manifest 持久化存储。"""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.manifest_dir = self.base_dir / "protocol" / "manifests"

    def write(self, manifest: SessionManifest) -> str:
        """写入 Manifest。"""
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.manifest_dir / f"{manifest.session_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
        return str(filepath)

    def read(self, session_id: str) -> Optional[SessionManifest]:
        """读取 Manifest。"""
        filepath = self.manifest_dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SessionManifest.from_dict(data)

    def list_all(self) -> List[str]:
        """列出所有会话 ID。"""
        if not self.manifest_dir.exists():
            return []
        files = sorted(self.manifest_dir.glob("*.json"))
        return [f.stem for f in files]

    def lineage(self, session_id: str) -> List[SessionManifest]:
        """追溯会话的血统链（父轨迹 → 当前会话）。"""
        manifests: List[SessionManifest] = []
        current = self.read(session_id)
        if not current:
            return manifests

        manifests.append(current)
        # 通过 parent_trajectories 反向查找父会话
        # 这里简化实现：从 manifest 目录中查找包含当前 session 轨迹的父会话
        for sid in self.list_all():
            if sid == session_id:
                continue
            m = self.read(sid)
            if m and current.session_id in m.trajectories_created:
                manifests.insert(0, m)

        return manifests

    def stats(self) -> dict:
        """统计信息。"""
        all_sessions = self.list_all()
        total_edges_written = 0
        total_edges_loaded = 0
        models: Set[str] = set()

        for sid in all_sessions:
            m = self.read(sid)
            if m:
                total_edges_written += m.edges_written
                total_edges_loaded += m.loaded_edges
                models.add(m.model)

        return {
            "total_sessions": len(all_sessions),
            "total_edges_written": total_edges_written,
            "total_edges_loaded": total_edges_loaded,
            "unique_models": sorted(models),
            "manifest_dir": str(self.manifest_dir),
        }
