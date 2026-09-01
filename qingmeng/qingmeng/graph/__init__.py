"""青檬引擎 · 拓扑推理层 —— 帧快照上的结构分析与涌现检测。

分层保障：本层只读分析，不修改图（演化只属于 core.state）。
模块:
    topology    路径/连通性/指标/子图 —— 只读拓扑分析
    emergence   五形相位检测 —— 从结构指标提取 S=(水,木,土,金,火)
    trajectory  推理轨迹快照与版本回滚 —— 观测性（trace_id/axiom_path）
"""

from .topology import Topology
from .emergence import EmergenceDetector
from .trajectory import TrajectoryStore, TraceRecord, DEFAULT_MAX_RECORDS

__all__ = [
    "Topology",
    "EmergenceDetector",
    "TrajectoryStore",
    "TraceRecord",
    "DEFAULT_MAX_RECORDS",
]
