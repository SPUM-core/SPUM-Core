"""
SPUM 统一宇宙模拟器 — 完整网络演化引擎的统一入口。

架构:
    UniverseSimulator
        ├── SPUMEngine (Phase_0)      — 五步帧内核 (创生/连接/体积/悬挂/删除)
        ├── EmergenceObserver         — 每帧涌现观测 (不完美/守恒/晶子/闭环/σ场)
        └── Timeline                  — 涌现里程碑记录 (首次晶子帧/首次闭环候选)

用法:
    sim = UniverseSimulator(seed_geometry="star", n_surface=42)
    sim.run(100)
    sim.report()

与旧管线 (Phase_1→2→3→4 分离) 的区别:
    这里只有一条循环 — 同一张 ⟨P,ε⟩ 网络逐帧演化,
    观测层在每一帧实时读取涌现, 不做任何阶段切换。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from Phase_0.gpu_engine import SPUMEngine, EngineConfig
from Phase_0.constants import CRYSTALLITE_DEGREE_THRESHOLD
from .emergence import observe_universe


@dataclass
class UniverseConfig:
    """宇宙模拟器配置 (透传 SPUMEngine 配置)。"""
    seed_geometry: str = "star"        # 初态: star | sequential | multi_center
    n_surface: int = 42                # 表面 coda 数 (T5 稳定解 42)
    n_centers: int = 8                 # multi_center: 中心簇数
    surface_per_center: int = 12       # multi_center: 每簇表面 coda 数
    cluster_radius: float = 60.0       # multi_center: 簇壳半径
    gap_max_checks: int = 100000       # step1_create 三体检查上限 (多中心可加大)
    max_particles: int = 65536
    pre_growth_frames: int = 5
    gap_enabled: bool = True
    observe_every: int = 1             # 每 N 帧观测一次


@dataclass
class UniverseReport:
    """涌现观测报告 — 全历史汇总。"""
    frames_run: int = 0
    final: Dict = field(default_factory=dict)
    milestones: Dict = field(default_factory=dict)
    # 不完美定理 (图论版) 双判据:
    local_closure_reached: bool = False   # 局部闭合可达 (42 团簇无悬挂)
    never_static: bool = True             # 全局静止不可达 (V 持续增长)
    invariant_trace: List[int] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        m = self.milestones
        f = self.final
        lines = [
            f"帧数            = {self.frames_run}",
            f"活性粒子 V      = {f.get('V', '?')}   "
            f"总边 E = {f.get('E', '?')}",
            f"σ = V/E         = {f.get('sigma', '?')}",
            f"Σ(6−deg)        = {f.get('invariant', '?')}",
            f"局部闭合        = {f.get('local_closed', '?')}   "
            f"(可达: {self.local_closure_reached})",
            f"全局静止        = 不可达   "
            f"(V 持续增长: {self.never_static})",
            f"晶子数 (deg≥{CRYSTALLITE_DEGREE_THRESHOLD}) = {f.get('crystallites', '?')}",
            f"火形指数 Φ      = {f.get('fire_index', '?')}   "
            f"σ梯度 = {f.get('sigma_gradient', '?')}",
            f"开口占比        = {f.get('opening_ratio', '?')}   "
            f"(界 ≤ 1/3: {f.get('opening_bound_ok', '?')})",
            f"12晶子闭环      = {m.get('ring_first_frame', None)} "
            f"(候选帧: {m.get('ring_candidate_frames', [])})",
            f"首次晶子帧      = {m.get('first_crystallite_frame', None)}",
        ]
        events = m.get("ring_emergence_events", [])
        if events:
            desc = ", ".join(
                f"帧{e['frame']}/{e['cluster_id']}" for e in events)
            lines.append(f"多簇闭环涌现    = {desc}")
        return lines


class UniverseSimulator:
    """完整网络演化模拟器。"""

    def __init__(self, config: Optional[UniverseConfig] = None):
        self.config = config or UniverseConfig()
        engine_cfg = EngineConfig(
            max_particles=self.config.max_particles,
            seed_geometry=self.config.seed_geometry,
            n_surface=self.config.n_surface,
            pre_growth_frames=self.config.pre_growth_frames,
            gap_enabled=self.config.gap_enabled,
            gap_max_checks=self.config.gap_max_checks,
            verbose=False,
        )
        # 多中心专属配置
        for attr in ("n_centers", "surface_per_center", "cluster_radius"):
            setattr(engine_cfg, attr, getattr(self.config, attr))
        self.engine = SPUMEngine(config=engine_cfg)
        self.history: List[Dict] = []
        self._milestones: Dict = {
            "first_crystallite_frame": None,
            "ring_candidate_frames": [],   # 晶子子图 ≥12 的帧
            "ring_first_frame": None,      # 12晶子闭环成形的首帧
            "ring_geometrically_confirmed": None,
            "ring_emergence_events": [],   # [(frame, cluster_id)] 各簇闭环涌现
        }

    # ── 运行 ──────────────────────────────────────────────

    def run(self, n_frames: int) -> List[Dict]:
        """运行 n 帧, 每 observe_every 帧做一次涌现观测。

        边缘不完美的正确观测点:
            帧前悬挂 (dangling_before) — 上一帧删除悬挂后, 其邻居
            度数降级再生的新悬挂。一帧只删一次, 删除必然留下新残留,
            因此 dangling_before 每帧 > 0 即"删除再生悬挂"的实证。
        """
        for _ in range(n_frames):
            p = self.engine.particles
            dangling_before = int(np.sum(p.active & (p.degree < 3)))
            self.engine.run_frame()
            if self.engine.frame_number % self.config.observe_every == 0:
                obs = observe_universe(p)
                obs["frame"] = self.engine.frame_number
                obs["dangling_before"] = dangling_before
                self.history.append(obs)
                self._update_milestones(obs)
        return self.history

    def _update_milestones(self, obs: Dict):
        n_crys = obs["crystallites"]
        if n_crys > 0 and self._milestones["first_crystallite_frame"] is None:
            self._milestones["first_crystallite_frame"] = obs["frame"]

        ring = obs.get("ring", {})
        if ring.get("n_crystallites", 0) >= 12:
            self._milestones["ring_candidate_frames"].append(obs["frame"])
        if ring.get("ring_formed"):
            if self._milestones["ring_first_frame"] is None:
                self._milestones["ring_first_frame"] = obs["frame"]
            # 多中心: 记录 (帧, 簇) 涌现事件 (去重同簇)
            cid = ring.get("cluster_id")
            if cid is not None:
                events = self._milestones["ring_emergence_events"]
                if not any(e["frame"] == obs["frame"] and e["cluster_id"] == cid
                           for e in events):
                    events.append({"frame": obs["frame"], "cluster_id": cid})
            ico = ring.get("icosahedron")
            if ico and ico.get("is_icosahedron"):
                self._milestones["ring_geometrically_confirmed"] = obs["frame"]

    # ── 查询 ──────────────────────────────────────────────

    def report(self) -> UniverseReport:
        """汇总报告。"""
        rep = UniverseReport(frames_run=self.engine.frame_number)
        if self.history:
            rep.final = self.history[-1]
            rep.invariant_trace = [h["invariant"] for h in self.history]
            # 局部闭合可达: 至少一帧无悬挂 (42 团簇闭合)
            rep.local_closure_reached = any(
                h["dangling"] == 0 for h in self.history)
            # 全局静止不可达: V 严格单调增长 (网络永不停机)
            Vs = [h["V"] for h in self.history]
            rep.never_static = all(
                Vs[i] < Vs[i + 1] for i in range(len(Vs) - 1))
        rep.milestones = dict(self._milestones)
        return rep

    def print_report(self):
        """打印报告 (演示入口)。"""
        rep = self.report()
        print("=" * 62)
        print("SPUM 完整网络演化 — 涌现观测报告")
        print("=" * 62)
        for line in rep.summary_lines():
            print("  " + line)
        print("-" * 62)
        print("  逐帧不变量 Σ(6−deg) 轨迹:")
        trace = rep.invariant_trace
        if trace:
            # 压缩显示: 只显示首尾 + 变化点
            shown = []
            prev = None
            for v in trace:
                if v != prev:
                    shown.append(v)
                    prev = v
            if len(shown) > 12:
                shown = shown[:6] + ["..."] + shown[-6:]
            print("    " + " → ".join(str(v) for v in shown))
        print("=" * 62)
