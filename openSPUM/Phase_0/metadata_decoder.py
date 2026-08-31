"""
FrameSnapshot — GPU 帧快照的元数据容器。

设计原则:
    帧快照 (约 200 bytes) 是 GPU → CPU 的唯一传输数据。
    GPU 不输出"物质解读"，只输出拓扑统计量。
    CPU 侧从这些统计量中解读出: 晶子数量、相位、Σ(6-deg) 等。

度数阈值 (SPUM2611 v5.0):
    CRYSTALLITE_DEGREE_THRESHOLD = 42 (T5 稳定解 deg(O) = 42，
    中心粒子与 12 顶点 + 30 边中点全部轨道位置相切)。
    12 仍是 Σ(6-deg) = 12 的计数事实 (T2)，非闭锁输入。
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .constants import CRYSTALLITE_DEGREE_THRESHOLD


@dataclass
class FrameSnapshot:
    """单帧快照 — GPU 输出的元数据 (约 200 bytes)。"""
    frame_number: int
    degree_histogram: np.ndarray        # [0..42] 度分布 (槽数 = 阈值+1)
    active_count: int                   # 当前活性粒子数
    latent_count: int                   # 当前潜在粒子数
    crystallite_count: int              # degree >= 42 (T5) 的粒子数
    spum_invariant: int                 # Σ(6 - deg) = 6V - 2E (T2: 闭合子图 = 12)
    dangling_count: int                 # 本帧新悬挂粒子数

    # 可选字段 (用于深度分析)
    degree_mean: Optional[float] = None
    degree_max: Optional[int] = None
    second_moment: Optional[float] = None  # Σ(deg²)/V


@dataclass
class FrameLog:
    """人类可读的帧日志 — 从 FrameSnapshot 解码。"""
    frame_number: int
    particle_count: int
    edge_count: int
    spum_invariant: int
    dangling_count: int
    crystallite_count: int
    latent_count: int
    degree_mean: float

    # 定性标签
    phase_label: str = ""               # 种子/集群/晶子/饱和
    invariant_stable: bool = False      # Σ(6-deg) 是否接近 12 (T2 计数事实)
    crystallite_forming: bool = False   # 是否有晶子形成 (deg >= 42, T5)

    def brief(self) -> str:
        """单行摘要。"""
        return (
            f"帧 {self.frame_number:>4d}: "
            f"V={self.particle_count:>4d} "
            f"E={self.edge_count:>5d} "
            f"Σ(6-deg)={self.spum_invariant:>4d} "
            f"悬挂={self.dangling_count:>3d} "
            f"晶子={self.crystallite_count:>2d} "
            f"潜在={self.latent_count:>3d} "
            f"均值度={self.degree_mean:.2f} "
            f"[{self.phase_label}]"
        )


class CUDAMetadataDecoder:
    """从 FrameSnapshot 解码出 FrameLog。

    每个 FrameSnapshot 对应一帧的 GPU 输出。
    解码器不做"物理解读"——只做拓扑统计。
    """

    @staticmethod
    def decode(snapshot: FrameSnapshot) -> FrameLog:
        hist = snapshot.degree_histogram
        n_active = snapshot.active_count
        if n_active > 0:
            total_deg = int(np.sum(np.arange(len(hist)) * hist.astype(np.int64)))
            edge_count = total_deg // 2
            degree_mean = total_deg / n_active
        else:
            total_deg = 0
            edge_count = 0
            degree_mean = 0.0

        # 定性标签
        phase_label = CUDAMetadataDecoder._classify_phase(
            n_active, snapshot.crystallite_count, snapshot.spum_invariant
        )

        return FrameLog(
            frame_number=snapshot.frame_number,
            particle_count=n_active,
            edge_count=edge_count,
            spum_invariant=snapshot.spum_invariant,
            dangling_count=snapshot.dangling_count,
            crystallite_count=snapshot.crystallite_count,
            latent_count=snapshot.latent_count,
            degree_mean=round(degree_mean, 3),
            phase_label=phase_label,
            invariant_stable=abs(snapshot.spum_invariant - 12) <= 6,
            crystallite_forming=snapshot.crystallite_count > 0,
        )

    @staticmethod
    def _classify_phase(n_active: int, crystallite_count: int,
                        spum_invariant: int) -> str:
        # 12 是 Σ(6-deg)=12 的计数事实 (T2)，非闭锁输入——
        # 作为"种子期/活跃网络"的最小规模判据仍成立
        if n_active < 12:
            return "种子期"
        if crystallite_count > 0:
            return "成核期"
        if abs(spum_invariant - 12) <= 6:
            return "集群期"
        return "增长期"

    @staticmethod
    def decode_series(snapshots: List[FrameSnapshot]) -> List[FrameLog]:
        """解码连续帧序列。"""
        return [CUDAMetadataDecoder.decode(s) for s in snapshots]
