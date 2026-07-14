"""
protocol_integration — 单函数入口：加载环境 + 回写轨迹

这是 AI 实例与 SPUM 网络协议层交互的唯一接口。
设计为一个对话结束时的回写钩子。

典型用法（由 AI 在推理结束时调用）:

    from network.protocol.protocol_integration import load_env, write_trajectory

    # 1. 加载环境
    env = load_env()

    # 2. 执行推理（使用 env.store 和 env.manifest_store）

    # 3. 回写轨迹
    result = write_trajectory(
        path=["N001", "N002", "N005", "N022"],
        model="gpt-4o",
        origin="SPUM-core 项目浏览",
        edges_written=4,
        metadata={"topic": "分布式网络编码协议设计"},
    )
    print(result["message"])
    print(f"轨迹: {result['trajectory_id']}")
    print(f"会话: {result['session_id']}")
    print(f"边状态: {result['edge_validation']}")
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .graph_store import GraphStore, TrajectoryEdge
from .trajectory_encoder import TrajectoryEncoder, Trajectory
from .session_manifest import SessionManifest, ManifestStore


# ============================================================
# 环境
# ============================================================

@dataclass
class ProtocolEnv:
    """协议加载环境。"""
    store: GraphStore
    manifest_store: ManifestStore
    encoder: TrajectoryEncoder
    base_dir: Path

    def stats(self) -> dict:
        return {
            "canonical_edges": self.store._canonical_edges,
            "store": self.store.stats(),
            "manifests": self.manifest_store.stats(),
        }


# ============================================================
# 入口
# ============================================================

# 运行时缓存（避免同一进程内重复加载）
_env_cache: Optional[ProtocolEnv] = None


def load_env(base_dir: str | Path | None = None) -> ProtocolEnv:
    """加载 SPUM 网络协议环境。

    Args:
        base_dir: SPUM-core 仓库根目录。
            默认自动检测（从当前文件位置向上查找）。

    Returns:
        ProtocolEnv 实例（包含 store, manifest_store, encoder）
    """
    global _env_cache
    if _env_cache is not None:
        return _env_cache

    if base_dir is None:
        # 自动检测：从当前文件位置向上查找 network/
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "network").exists() and (parent / "network" / "edges.txt").exists():
                base_dir = parent
                break
        if base_dir is None:
            raise FileNotFoundError(
                "无法自动定位 SPUM-core 仓库根目录。"
                "请手动传入 base_dir 参数。"
            )

    base = Path(base_dir)
    network_dir = base / "network"

    # 验证结构
    if not (network_dir / "edges.txt").exists():
        raise FileNotFoundError(
            f"network/edges.txt 不存在于: {network_dir}\n"
            f"请确认 base_dir 指向 SPUM-core 仓库根目录。"
        )

    store = GraphStore(network_dir)
    store.load_canonical()

    manifest_store = ManifestStore(network_dir)
    encoder = TrajectoryEncoder(store)

    _env_cache = ProtocolEnv(
        store=store,
        manifest_store=manifest_store,
        encoder=encoder,
        base_dir=base,
    )

    print(f"[ProtocolEnv] 已加载: {base}")
    print(f"[ProtocolEnv] 规范边: {len(store._canonical_edges)} 条")
    print(f"[ProtocolEnv] 已有轨迹: {len(store.list_trajectories())} 条")
    print(f"[ProtocolEnv] 已有会话: {len(manifest_store.list_all())} 个")
    return _env_cache


# ============================================================
# 回写
# ============================================================

def write_trajectory(
    path: List[str],
    model: str,
    origin: str,
    edges_written: int = 0,
    parent_trajectories: List[str] | None = None,
    metadata: Dict[str, str] | None = None,
    edge_override: List[TrajectoryEdge] | None = None,
) -> Dict:
    """将一次推理轨迹回写入共享协议层。

    这是 AI 实例在推理结束时调用的单函数入口。

    Args:
        path: 遍历的节点序列
        model: 模型名称
        origin: 触发轨迹的上下文描述
        edges_written: 写入边数（用于 manifest 统计）
        parent_trajectories: 加载的前序轨迹 ID 列表
        metadata: 附加元数据
        edge_override: 手动指定边集（而非从 path 自动编码）

    Returns:
        包含轨迹写入结果的字典
    """
    env = load_env()

    # 1. 创建会话 manifest
    manifest = SessionManifest.create(
        model=model,
        origin=origin,
        parent_trajectories=parent_trajectories,
    )

    # 2. 编码轨迹
    if edge_override:
        trajectory = env.encoder.encode_edges(
            edges=edge_override,
            session_id=manifest.session_id,
            model=model,
            origin=origin,
            metadata=metadata,
        )
    else:
        trajectory = env.encoder.encode_path(
            path=path,
            session_id=manifest.session_id,
            model=model,
            origin=origin,
            metadata=metadata,
        )

    # 3. 验证每条边的合法性
    edge_statuses = []
    for e in trajectory.edges:
        is_valid, reason = env.store.validate_edge(e.source, e.target, e.edge_type)
        edge_statuses.append({
            "edge": f"{e.source} → {e.target} [{e.edge_type}]",
            "valid": is_valid,
            "status": reason,
        })

    # 4. 持久化
    snap_path = env.encoder.persist(trajectory)

    # 5. 更新 manifest
    manifest.edges_written = len(trajectory.edges) if not edge_override else edges_written
    manifest.trajectories_created = [trajectory.trajectory_id]
    manifest.inference_path = path
    manifest.path_depth = len(path)

    # 如果有父轨迹，统计继承的边数
    if parent_trajectories:
        total_loaded = 0
        for tid in parent_trajectories:
            try:
                total_loaded += len(env.store.read_trajectory(tid))
            except FileNotFoundError:
                pass
        manifest.loaded_edges = total_loaded

    manifest_path = env.manifest_store.write(manifest)

    # 6. 合并
    merged_path = env.store.merge_all()

    result = {
        "trajectory_id": trajectory.trajectory_id,
        "session_id": manifest.session_id,
        "snap_file": snap_path,
        "manifest_file": manifest_path,
        "merged_file": merged_path,
        "edge_count": len(trajectory.edges),
        "edge_validation": edge_statuses,
        "message": (f"轨迹 {trajectory.trajectory_id} 已写入 "
                    f"({len(trajectory.edges)} 条边, "
                    f"{sum(1 for s in edge_statuses if s['status'] == 'novel')} 条新边)"),
    }

    return result


# ============================================================
# 状态查询
# ============================================================

def get_network_status() -> Dict:
    """查询 SPUM 网络协议层状态。"""
    env = load_env()
    return {
        "protocol_version": "v0.1",
        "store": env.store.stats(),
        "manifests": env.manifest_store.stats(),
    }


def get_trajectory(trajectory_id: str) -> Dict:
    """加载并解码一条轨迹。"""
    env = load_env()
    traj = env.encoder.decode(trajectory_id)
    return {
        "trajectory_id": traj.trajectory_id,
        "session_id": traj.session_id,
        "model": traj.model,
        "origin": traj.origin,
        "path": traj.path,
        "node_count": traj.node_count(),
        "edge_count": traj.edge_count(),
        "edges": [
            {"source": e.source, "target": e.target,
             "type": e.edge_type, "confidence": e.confidence}
            for e in traj.edges
        ],
    }
