"""
SPUMEngine — 第 0 阶段 GPU 空间粒子引擎主类。

CPU 后端 + 元数据解码。

使用方式:
    engine = SPUMEngine(seed_geometry="sequential", max_particles=50000)
    
    for i in range(200):
        snapshot = engine.run_frame()
        log = engine.decode(snapshot)
        print(log.brief())

    engine.plot_history()  # 或导出快照列表
"""

import math
import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from .constants import (
    KAPPA, GEOMETRIC_TOLERANCE, CRYSTALLITE_DEGREE_THRESHOLD, MAX_PARTICLES
)
from .particle_array import ParticleArray
from .frame_kernels import (
    run_full_frame, step1_create, step2_connect,
    step3_volume, step3b_enforce_impenetrability,
    step4_dangling, step5_purge,
    _init_sequential,
)
from .metadata_decoder import FrameSnapshot, FrameLog, CUDAMetadataDecoder


@dataclass
class EngineConfig:
    """引擎配置。"""
    max_particles: int = MAX_PARTICLES
    seed_geometry: str = "sequential"    # 初态: star | sequential
    n_surface: int = 42                   # star/sequential 模式: coda 数 (T5 稳定解 42)
    pre_growth_frames: int = 5            # 前 N 帧无悬挂修剪 (先增长)
    gap_enabled: bool = True              # 是否启用缝隙创生
    verbose: bool = False                 # 每帧打印摘要


class SPUMEngine:
    """SPUM GPU 空间粒子引擎 (CPU 参考后端)。

    核心帧循环:
        for each frame:
            step1_create()    → 缝隙检测 + 潜在活化
            step2_connect()   → 相切/角邻接连接
            step3_volume()    → degree → radius
            step4_dangling()  → 标记悬挂 (pre-growth 后启用)
            step5_purge()     → 回归潜在池 (pre-growth 后启用)
            record_snapshot() → FrameSnapshot (约 200 bytes)
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.particles = ParticleArray(max_n=self.config.max_particles)
        self.frame_number = 0
        self.history: List[FrameSnapshot] = []
        self.decoder = CUDAMetadataDecoder()

        # 初始化粒子初态
        self._initialize_seeds()

    @property
    def _in_pregrowth(self) -> bool:
        """是否处于 pre-growth 阶段 (无修剪)。"""
        return self.frame_number <= self.config.pre_growth_frames

    def _initialize_seeds(self):
        """确定性初态初始化。

        两种模式 (SPUM2611 v5.0):
            star: 1 中心 coda (deg=42, T5 稳定解) + 42 表面 coda (deg=1)
            sequential: 链式接入, 每个 coda 从切线计算坐标

        注: 正二十面体是推导的**输出** (主定理: 计数 → 空间)，
        不作为输入种子构型。
        """
        if self.config.seed_geometry == "star":
            self._init_star()
        elif self.config.seed_geometry == "sequential":
            self._init_sequential()
        else:
            raise ValueError(f"Unknown seed_geometry: {self.config.seed_geometry}")

    def _init_star(self):
        """星形初态: 1 中心 coda (deg=n_surface) + N 表面 coda (deg=1)。

        中心在原点, 表面 coda 在球面半径 = center_r + 1 处。
        所有表面 coda 与中心初始连接。
        """
        from .frame_kernels import _star_surface

        n_surf = self.config.n_surface
        positions, radii = _star_surface(n_surf)

        # 中心 coda (先 degree=0, 连接建立后自动设置)
        self.particles.add_particle(
            pos=(0.0, 0.0, 0.0),
            uid=f"cent_0000",
            degree=0,
        )
        self.particles.radius[0] = radii[0]

        # 表面 coda
        for i in range(n_surf):
            self.particles.add_particle(
                pos=(float(positions[i + 1][0]),
                     float(positions[i + 1][1]),
                     float(positions[i + 1][2])),
                uid=f"surf_{i:04d}",
                degree=0,
            )
            self.particles.radius[1 + i] = radii[1 + i]

        # 建立初始中心-表面连接 (persistent)
        for i in range(n_surf):
            self.particles.add_connection(0, 1 + i)

    def _init_sequential(self):
        """顺序构建: coda 逐个接入, 从切线计算坐标。

        所有 coda 初始 r=1, coda 0 为原点。
        coda i 连接到 coda (i-1), 双方体积+1。
        坐标由切线位置决定。
        """
        n = self.config.n_surface

        # 调用核心函数计算位置/半径/边
        positions, radii, edges = _init_sequential(n)

        # 创建粒子
        for i in range(n):
            self.particles.add_particle(
                pos=(float(positions[i][0]),
                     float(positions[i][1]),
                     float(positions[i][2])),
                uid=f"coda_{i:04d}",
                degree=0,
            )
            self.particles.radius[i] = radii[i]

        # 建立连接
        for i, j in edges:
            self.particles.add_connection(i, j)

    def run_frame(self) -> FrameSnapshot:
        """执行一帧完整演化。

        pre-growth 阶段 (前 pre_growth_frames 帧):
            跳过 Step 4-5 (悬挂检测和删除)
            只执行: 创生 → 连接 → 体积

        之后:
            完整 5 步 + star_mode

        Returns:
            FrameSnapshot — 当前帧快照
        """
        self.frame_number += 1

        star_mode = (self.config.seed_geometry == "star")

        # sequential 模式: 不跳过悬挂, 无 star_mode 连接
        # star 模式: pre-growth 阶段跳过悬挂+断连, 第一帧做 KNN
        no_purge = self._in_pregrowth and star_mode
        allow_disconnect = not self._in_pregrowth or not star_mode
        pregrowth_knn = 0

        stats = run_full_frame(
            self.particles,
            star_mode=star_mode,
            no_purge=no_purge,
            pregrowth_knn=pregrowth_knn,
            allow_disconnect=allow_disconnect,
        )

        # 记录快照
        snapshot = self.particles.snapshot(self.frame_number)
        self.history.append(snapshot)

        if self.config.verbose:
            log = self.decode(snapshot)
            print(log.brief())

        return snapshot

    def run_frames(self, n: int) -> List[FrameSnapshot]:
        """连续运行多帧。"""
        return [self.run_frame() for _ in range(n)]

    def decode(self, snapshot: FrameSnapshot) -> FrameLog:
        """将帧快照解码为人类可读日志。"""
        return self.decoder.decode(snapshot)

    def decode_history(self) -> List[FrameLog]:
        """解码全部历史帧。"""
        return [self.decode(s) for s in self.history]

    @property
    def active_count(self) -> int:
        return self.particles.active_count()

    @property
    def latent_count(self) -> int:
        return self.particles.latent_count()

    @property
    def spum_invariant(self) -> int:
        return self.particles.spum_invariant()

    def summary(self) -> Dict:
        """引擎状态摘要。"""
        return {
            "frame": self.frame_number,
            "active": self.active_count,
            "latent": self.latent_count,
            "max_particles": self.config.max_particles,
            "invariant": self.spum_invariant,
            "history_length": len(self.history),
        }

    def plot_history(self, show: bool = True):
        """快速可视化历史 — V + Σ(6-deg) + 晶子数 时序。

        需要 matplotlib。
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        logs = self.decode_history()
        frames = [l.frame_number for l in logs]
        V = [l.particle_count for l in logs]
        inv = [l.spum_invariant for l in logs]
        crys = [l.crystallite_count for l in logs]
        dang = [l.dangling_count for l in logs]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))

        ax1.plot(frames, V, 'b-', linewidth=1)
        ax1.set_title('活性粒子数 V')
        ax1.set_xlabel('帧')
        ax1.grid(alpha=0.3)

        ax2.plot(frames, inv, 'r-', linewidth=1)
        ax2.axhline(y=12, color='gray', linestyle='--', alpha=0.5, label='Σ(6-deg)=12')
        ax2.set_title('SPUM 不变量 Σ(6-deg)')
        ax2.set_xlabel('帧')
        ax2.legend()
        ax2.grid(alpha=0.3)

        ax3.plot(frames, crys, 'g-', linewidth=1)
        ax3.set_title(f'晶子数 (degree >= {CRYSTALLITE_DEGREE_THRESHOLD})')
        ax3.set_xlabel('帧')
        ax3.grid(alpha=0.3)

        ax4.plot(frames, dang, 'orange', linewidth=1)
        ax4.set_title('每帧新悬挂数')
        ax4.set_xlabel('帧')
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        path = f"spum_phase0_history_f{self.frame_number}.png"
        fig.savefig(path, dpi=150)
        print(f"图表已保存: {path}")
        if show:
            plt.show()
        plt.close()
