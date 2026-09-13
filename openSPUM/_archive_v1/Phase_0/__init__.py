"""
OpenSPUM Phase 0 — CUDA 空间粒子引擎

核心架构（v2 — 无级联、无湮灭-创生对偶）：
    每 CUDA 线程 = 1 个 ⟨P, ε⟩ 空间粒子。
    一帧 = 5 步，严格遵循 SPUM 系统总纲 §4：
        创生 → 连接 → 变化体积 → 判断悬挂边 → 删除悬挂边

    当前后端：
        numpy (CPU) — 参考实现，镜像 GPU 的 SoA 结构
        CUDA (GPU)  — 后端就绪，随 CUDA Toolkit 安装自动启用

    帧快照 (约 200 bytes/帧) 作为元数据输出给 CPU：
        → 度分布直方图、晶子计数、悬挂计数、Σ(6-deg)

使用方式:
    from Phase_0.gpu_engine import SPUMEngine
    
    engine = SPUMEngine(max_particles=65536)
    for i in range(100):
        snapshot = engine.run_frame()
        log = engine.decode(snapshot)
        print(log.brief())
"""

from .constants import (
    KAPPA, TAU, CRYSTALLITE_DEGREE_THRESHOLD,
    GEOMETRIC_TOLERANCE, MIN_GAP_RATIO, MAX_PARTICLES,
)
from .metadata_decoder import FrameSnapshot, CUDAMetadataDecoder

__all__ = [
    "KAPPA", "TAU",
    "CRYSTALLITE_DEGREE_THRESHOLD",
    "GEOMETRIC_TOLERANCE", "MIN_GAP_RATIO", "MAX_PARTICLES",
    "FrameSnapshot", "CUDAMetadataDecoder",
]
