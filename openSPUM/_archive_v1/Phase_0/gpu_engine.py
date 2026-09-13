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
    run_full_frame, step1_create, step1b_gap_fill, step2_connect,
    step3_volume, step3b_enforce_impenetrability,
    step4_dangling, step5_purge, enforce_tangency,
    _init_sequential,
)
from .metadata_decoder import FrameSnapshot, FrameLog, CUDAMetadataDecoder
from .gpu_backend import cuda_available, device_name, active_backend, resolve


@dataclass
class EngineConfig:
    """引擎配置。"""
    max_particles: int = MAX_PARTICLES
    seed_geometry: str = "sequential"    # 初态: star | sequential | multi_center
    n_surface: int = 42                   # star/sequential 模式: coda 数 (T5 稳定解 42)
    n_centers: int = 8                    # multi_center 模式: 中心簇数
    surface_per_center: int = 12          # multi_center 模式: 每簇表面 coda 数
    cluster_radius: float = 60.0          # multi_center 模式: 簇壳半径
    pre_growth_frames: int = 5            # 前 N 帧无悬挂修剪 (先增长)
    gap_enabled: bool = True              # 是否启用缝隙创生
    gap_max_checks: int = 100000          # step1_create 最大三体检查数 (多中心可用更高值)
    backend: str = "auto"                 # auto|cuda|cpu — 后端选择 (显卡部署)
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
        # 后端解析: 'cuda' 强制 GPU, 'cpu' 强制 CPU, 'auto' 探测降级
        self.backend = resolve(self.config.backend)
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
            multi_center: C 个独立星簇分布球面, 每簇中心+表面 coda

        注: 正二十面体是推导的**输出** (主定理: 计数 → 空间)，
        不作为输入种子构型。
        """
        if self.config.seed_geometry == "star":
            self._init_star()
        elif self.config.seed_geometry == "sequential":
            self._init_sequential()
        elif self.config.seed_geometry == "multi_center":
            self._init_multi_center()
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

    def _init_multi_center(self):
        """多中心分布式初态: C 个独立星簇均匀分布于球面。

        每个簇 = 1 中心 + M 表面 coda (局部 star 缩放), 簇间由
        cluster_radius 隔离, 保证跨簇不相切、不被全局创生染指。
        目的是让晶子 (deg≥42) 分散成独立簇, 而不是聚成单中心
        过饱和团簇, 为帧内涌现 12 晶子闭环创造时机窗口。

        诚实边界 (防伪加载):
            正二十面体是推导**输出**, 不作为初态输入。中心数 C≠12,
            中心布点用 `_fibonacci_sphere` (非正二十面体顶点), 避免作弊。
            多中心仍是"无差别缝隙创生"在每个簇局部的重演, 命中
            "恰好 12"是时机窗口而非保证 (参见 plan §5)。
        """
        from .frame_kernels import _fibonacci_sphere

        n_centers = self.config.n_centers
        M = self.config.surface_per_center
        R = self.config.cluster_radius

        # C 个中心方向: 斐波那契球面均匀分布 (非正二十面体)
        center_dirs = _fibonacci_sphere(n_centers)

        idx = 0  # 粒子索引计数器
        for I in range(n_centers):
            center_dir = center_dirs[I]
            center_pos = tuple(float(x) * R for x in center_dir)

            # 簇内表面 coda 方向: 围绕中心局部的斐波那契球面 (尺度远
            # 小于簇间距, 保证跨簇隔离)。surface_r 由中心度决定。
            surf_dirs = _fibonacci_sphere(M)
            center_r = float((1 + M) * KAPPA)  # 中心连 M 个表面后
            surface_r = 2.0                    # 1 + 1 (仅连中心)

            # 中心
            self.particles.add_particle(
                pos=center_pos,
                uid=f"cent_{I:04d}",
                degree=0,
            )
            center_i = idx
            self.particles.radius[center_i] = center_r
            idx += 1

            # 簇内表面 coda (在中心表面切线位置, 相对偏移)
            for j in range(M):
                surface_dist = center_r + surface_r
                surf_pos = (
                    center_pos[0] + surf_dirs[j][0] * surface_dist,
                    center_pos[1] + surf_dirs[j][1] * surface_dist,
                    center_pos[2] + surf_dirs[j][2] * surface_dist,
                )
                self.particles.add_particle(
                    pos=(float(surf_pos[0]), float(surf_pos[1]), float(surf_pos[2])),
                    uid=f"surf_{I:04d}_{j:04d}",
                    degree=0,
                )
                self.particles.radius[idx] = surface_r
                self.particles.add_connection(center_i, idx)
                idx += 1

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
        multi_center = (self.config.seed_geometry == "multi_center")
        # pre-growth 行为 (no_purge): star 与 multi_center 都先增长挂载
        seed_like = star_mode or multi_center

        # sequential 模式: 不跳过悬挂, 无 star_mode 连接
        # star/multi_center 模式: pre-growth 阶段跳过悬挂+断连, 先增长
        no_purge = self._in_pregrowth and seed_like
        allow_disconnect = not self._in_pregrowth or not seed_like
        pregrowth_knn = 0

        if self.backend == "cuda":
            stats = self._run_frame_cuda(
                star_mode=star_mode, multi_center=multi_center,
                no_purge=no_purge, allow_disconnect=allow_disconnect,
                pregrowth_knn=pregrowth_knn,
            )
        else:
            stats = run_full_frame(
                self.particles,
                star_mode=star_mode,
                multi_center=multi_center,
                no_purge=no_purge,
                pregrowth_knn=pregrowth_knn,
                allow_disconnect=allow_disconnect,
                gap_max_checks=self.config.gap_max_checks,
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

    # ── GPU 加速帧路径 ─────────────────────────────────
    # CUDA 只加速 O(N²) 几何候选计算 (cdist), 度数装配仍在 CPU。
    # 与 CPU 帧语义一致 → 结果逐帧可对照。
    def _run_frame_cuda(self, star_mode, multi_center, no_purge,
                        allow_disconnect, pregrowth_knn) -> Dict:
        from . import gpu_accel
        from .constants import GEOMETRIC_TOLERANCE

        p = self.particles
        active_before = p.active_count()

        # Step 3: 变化体积 — 与 CPU 完全一致 (step3_volume 已向量化, 无需 GPU)
        step3_volume(p)

        # Step 3c: 相邻相切强制 (BFS 重构坐标) — 与 CPU 一致
        enforce_tangency(p)

        # Step 1: 创生 — 在空隙中填入新球体 (逻辑密集, CPU)
        created = step1_create(p, max_checks=self.config.gap_max_checks)
        if created == 0:
            # 与 CPU run_full_frame 一致: 三体无缝隙时尝试球面空隙填充
            created = step1b_gap_fill(p, max_per_frame=20)

        # Step 2: 连接。
        #   GPU: 用 CUDA cdist 预计算几何相切候选对集合 (O(N²) 热区)。
        #   CPU: 调用与 CPU 路径完全相同的 step2_connect —— 度数装配、
        #        角距排序、封顶、多簇逻辑全部保留 → 严格语义等价。
        tangent_set = set()
        if star_mode or multi_center:
            tangent_set = {
                (a, b) for a, b in gpu_accel.tangent_pairs_gpu(
                    p.pos, p.radius, p.active, tol=GEOMETRIC_TOLERANCE)
            }
        connected = step2_connect(
            p, star_mode=star_mode, multi_center=multi_center,
            pregrowth_knn=pregrowth_knn, skip_center=False,
            tangent_pairs=tangent_set if tangent_set else None)

        # Step 3b: 不可入性 — 与 CPU 完全一致 (球面角度滑动消解, 非 overlap 断连)
        overlaps_resolved = step3b_enforce_impenetrability(
            p, allow_disconnect=allow_disconnect)

        if no_purge:
            purged = 0
            dangling_count = 0
        else:
            dangling = step4_dangling(p, seed_only=False)
            dangling_count = int(np.sum(dangling))
            purged = step5_purge(p, dangling)

        reincarnated = 0
        active_after = p.active_count()

        return {
            "created": created,
            "connected": connected,
            "overlaps_resolved": overlaps_resolved,
            "dangling_marked": dangling_count,
            "purged": purged,
            "reincarnated": reincarnated,
            "active_before": active_before,
            "active_after": active_after,
            "degree_histogram": p.degree_histogram(),
        }

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
