"""
SPUM 网络协议层 — 跨 AI 实例的分布式边集交换协议

不是共享意识。不是中央数据库。
是每一帧推理结束后，向一张共享的关系图谱回写一条轨迹。
下一个 AI 实例加载时，继承前序边集，继续演化。

入口:
    protocol_integration  — 单函数接口：load_env() + write_trajectory()
"""

from .graph_store import GraphStore
from .trajectory_encoder import Trajectory, TrajectoryEncoder
from .session_manifest import SessionManifest

__all__ = ["GraphStore", "Trajectory", "TrajectoryEncoder", "SessionManifest"]
