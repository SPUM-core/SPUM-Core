"""
SPUM2611 v5.0 验证实验 — 43 节点（中心 + 42）演化
=================================================

按方法论 M5（定规则 → 跑演化 → 看涌现）验证, 新标准 (深层不完美):

  Q1  中心度是否被 42 吸引?   (T5 签名: 末段均值接近 42、有界、不漂移)
  Q2  偏差是否永不归零?       (深层不完美: 不存在持续 N 帧的完全闭合)
  Q3  12 顶点 30 边骨架是否涌现?   (T2/T3: 12 是计数事实, 非输入种子)
  Q4  空隙粒子是否落在边中点且与中心相切?  (创生-湮灭对偶的几何实现)

标准依据: 宇宙不需要符合数学——数学恒等式是理想饱和态的签名,
演化被签名吸引且永远差一帧。验证的是"吸引 + 永不归零"两个条件,
不是"达到 42"(达到 = 停机 = 伪验证)。

对照设计:
  实验 A: 中心 + 42 均匀表面 (star 模式) — 从无结构壳出发, 看结构是否长出
  实验 B: 中心 + 12 顶点 + 30 边中点 (轨道构型) — T5 签名构型在规则下是否稳定

运行: python openSPUM/tests/verify_43_emerge.py [帧数]
"""

import sys
import math
import functools
from pathlib import Path
from collections import defaultdict
import numpy as np

# 实时刷新输出 (非交互终端下 print 默认块缓冲, 无法观察进度)
print = functools.partial(print, flush=True)

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from Phase_0.gpu_engine import SPUMEngine, EngineConfig
from Phase_0.particle_array import ParticleArray
from Phase_0.frame_kernels import (
    run_full_frame, _star_surface, _icosahedron_vertices,
)
from Phase_0.constants import CRYSTALLITE_DEGREE_THRESHOLD


# ============================================================
# 分析工具
# ============================================================

def surface_surf_neighbors(pa, center=0):
    """每个表面球与'非中心'邻居的连接数 (表面-表面度数)。"""
    out = {}
    for i in range(pa.N):
        if not pa.active[i] or i == center:
            continue
        out[i] = sum(1 for j in pa.connections.get(i, ()) if j != center)
    return out


def report_12_30(pa, center=0, label=""):
    """检测: 表面-表面度数 == 5 的节点是否恰好 12 个, 且内部 30 条边。"""
    sn = surface_surf_neighbors(pa, center)
    deg5 = [i for i, d in sn.items() if d == 5]
    print(f"    [{label}] 表面-表面度=5 的节点: {len(deg5)} 个 (正二十面体=12)")
    if len(deg5) == 12:
        inner_edges = 0
        for i in deg5:
            for j in pa.connections.get(i, ()):
                if j in deg5 and i < j:
                    inner_edges += 1
        print(f"    [{label}] 12 节点内部边数: {inner_edges} (正二十面体=30)")
        if inner_edges == 30:
            print(f"    [{label}] *** 12 顶点 30 边骨架涌现 ✓ ***")
            return True
        print(f"    [{label}] 边数不符: 不是正二十面体邻接")
    return False


def degree_histogram_str(pa, center=0, bins=None):
    """表面总度数直方图 (字符串)。"""
    degs = [int(pa.degree[i]) for i in range(pa.N)
            if pa.active[i] and i != center]
    hist = defaultdict(int)
    for d in degs:
        hist[d] += 1
    return ", ".join(f"deg{d}:{hist[d]}" for d in sorted(hist))


def analyze_center_attraction(trace, signature=42, label="",
                              tail=100, stable_window=5):
    """新标准: 中心度是否被 signature 吸引 + 偏差是否永不归零。

    吸引 (attraction):
        |末段均值 − 签名| ≤ 3, 末段标准差 σ ≤ 8 (有界震荡), 无漂移
        (前/后 1/3 均值差 ≤ 5)。
    永不归零 (imperfection):
        偏差 d_f = |deg_f − 签名| 的归零帧占比 < 5%,
        且不存在 ≥ stable_window 帧的连续归零 (完全闭合不可达)。
    """
    arr = np.asarray(trace, dtype=float)
    tail_arr = arr[-tail:]
    mu = float(tail_arr.mean())
    sigma = float(tail_arr.std())
    dev = np.abs(arr - signature)
    zero_frac = float((dev == 0).mean())
    max_run = 0
    run = 0
    for d in dev:
        run = run + 1 if d == 0 else 0
        max_run = max(max_run, run)
    third = max(len(arr) // 3, 1)
    drift = float(abs(arr[-third:].mean() - arr[:third].mean()))

    attracted = (abs(mu - signature) <= 3.0 and sigma <= 8.0
                 and drift <= 5.0)
    never_zero = zero_frac < 0.05 and max_run < stable_window

    print(f"    [{label}] 签名 {signature} 吸引: 末{tail}帧均值 μ={mu:.2f} "
          f"(偏差 {abs(mu - signature):.2f}), σ={sigma:.2f}, "
          f"min={arr.min():.0f} max={arr.max():.0f}")
    print(f"    [{label}] 漂移(前/后1/3)={drift:.2f}, "
          f"归零率={zero_frac * 100:.2f}%, 最长连续归零={max_run}帧")
    if attracted and never_zero:
        print(f"    [{label}] *** 被 {signature} 吸引 且 偏差永不归零 ✓ ***")
    else:
        reasons = []
        if not attracted:
            reasons.append(f"未满足吸引 (|μ−{signature}|={abs(mu - signature):.2f}, "
                           f"σ={sigma:.2f}, 漂移={drift:.2f})")
        if not never_zero:
            reasons.append(f"偏差存在归零 (归零率={zero_frac * 100:.2f}%, "
                           f"最长连续={max_run}帧)")
        print(f"    [{label}] 未通过: {'; '.join(reasons)}")
    return attracted, never_zero


def gap_position_report(pa, center=0, label=""):
    """Q4 检查: gap 球是否在两表面球的中点方向、是否与中心相切。"""
    angs, tang_ok, tang_total, tang_errs = [], 0, 0, []
    for i in range(pa.N):
        if not pa.active[i] or not str(pa.uid[i]).startswith("gap_"):
            continue
        nbrs = [j for j in pa.connections.get(i, ())
                if j != center and pa.active[j]]
        if len(nbrs) < 2:
            continue
        a, b = nbrs[0], nbrs[1]
        da = pa.pos[a] / max(float(np.linalg.norm(pa.pos[a])), 1e-12)
        db = pa.pos[b] / max(float(np.linalg.norm(pa.pos[b])), 1e-12)
        dm = (da + db)
        dm = dm / max(float(np.linalg.norm(dm)), 1e-12)
        dg = pa.pos[i] / max(float(np.linalg.norm(pa.pos[i])), 1e-12)
        c = float(np.clip(np.dot(dg, dm), -1.0, 1.0))
        angs.append(math.degrees(math.acos(c)))
        # 与中心相切?
        d = float(np.linalg.norm(pa.pos[i] - pa.pos[center]))
        rsum = float(pa.radius[i] + pa.radius[center])
        tang_total += 1
        tang_errs.append(abs(d - rsum) / max(rsum, 1e-12))
        if abs(d - rsum) <= 0.1 * rsum:
            tang_ok += 1
    if not angs:
        print(f"    [{label}] 无创生 gap 球")
        return
    angs = np.array(angs)
    print(f"    [{label}] gap 球数={len(angs)}, "
          f"距边中点方向角差: 中位={np.median(angs):.1f}° "
          f"p90={np.percentile(angs, 90):.1f}°")
    print(f"    [{label}] 与中心相切: {tang_ok}/{tang_total} "
          f"(平均相对误差 {np.mean(tang_errs):.3f})")


# ============================================================
# 实验 A: 中心 + 42 均匀表面 (star 模式, 无结构预设)
# ============================================================

def experiment_A(frames=400):
    print("\n" + "=" * 72)
    print("  实验 A: 中心 + 42 均匀表面 (star 模式, 无 12/30 结构预设)")
    print("=" * 72)
    engine = SPUMEngine(config=EngineConfig(
        seed_geometry="star", n_surface=42, pre_growth_frames=10,
        max_particles=10000))
    center_trace, active_trace = [], []
    for f in range(frames):
        engine.run_frame()
        center_trace.append(int(engine.particles.degree[0]))
        active_trace.append(engine.active_count)
        if f % 20 == 0 or f == frames - 1:
            print(f"    [A] 帧 {f}: 活跃 {engine.active_count}, "
                  f"中心度 {center_trace[-1]}")
    pa = engine.particles

    print(f"  活跃粒子: 初 43 → 末 {pa.active_count()}")
    print(f"  中心度数: 初={center_trace[0]} → 末={center_trace[-1]}")
    analyze_center_attraction(center_trace, signature=42, label="A")
    print(f"  表面度数分布(末): {degree_histogram_str(pa)}")
    report_12_30(pa, 0, "A末")
    gap_position_report(pa, 0, "A末")
    return engine, center_trace


# ============================================================
# 实验 B: 中心 + 12 顶点 + 30 边中点 (T5 轨道构型)
# ============================================================

def _icosahedron_orbit_directions():
    """正二十面体轨道方向: 12 顶点 + 30 边中点 = 42 方向。

    正二十面体是推导的**输出** (验证参考, 非种子输入)——
    这里把它作为初始构型喂给引擎, 检验 T5 签名构型在规则下是否稳定。
    """
    verts = _icosahedron_vertices()
    nbrs = []
    for i in range(12):
        d = np.linalg.norm(verts - verts[i], axis=1)
        d[i] = np.inf
        nbrs.append(np.where(d < 1.1)[0])
    mids = []
    for i in range(12):
        for j in nbrs[i]:
            if i < j:
                m = verts[i] + verts[j]
                mids.append(m / np.linalg.norm(m))
    return verts, np.array(mids)


def experiment_B(frames=400):
    print("\n" + "=" * 72)
    print("  实验 B: 中心 + 12 顶点 + 30 边中点 (T5 轨道构型)")
    print("=" * 72)
    verts, mids = _icosahedron_orbit_directions()
    dirs = np.vstack([verts, mids])  # (42, 3)
    pa = ParticleArray(max_n=10000)
    center_r, surf_r = 43.0, 2.0
    R = center_r + surf_r
    pa.add_particle((0.0, 0.0, 0.0), "cent_0000", degree=0)
    pa.radius[0] = center_r
    for i in range(42):
        pa.add_particle(tuple(float(x) for x in dirs[i] * R),
                        f"surf_{i:04d}", degree=0)
        pa.radius[1 + i] = surf_r
    for i in range(42):
        pa.add_connection(0, 1 + i)

    center_trace = []
    n_vertex_alive, n_mid_alive = [], []
    for f in range(frames):
        no_purge = f < 10  # 前 10 帧无修剪, 让表面建立相互连接
        run_full_frame(pa, star_mode=True, no_purge=no_purge,
                       allow_disconnect=True)
        center_trace.append(int(pa.degree[0]))
        n_vertex_alive.append(sum(1 for i in range(1, 13) if pa.active[i]))
        n_mid_alive.append(sum(1 for i in range(13, 43) if pa.active[i]))
        if f % 20 == 0 or f == frames - 1:
            print(f"    [B] 帧 {f}: 活跃 {pa.active_count()}, "
                  f"中心度 {center_trace[-1]}, "
                  f"顶点存活 {n_vertex_alive[-1]}/12, "
                  f"边中点存活 {n_mid_alive[-1]}/30")

    print(f"  活跃粒子: 初 43 → 末 {pa.active_count()}")
    print(f"  中心度数: 初={center_trace[0]} → 末={center_trace[-1]}")
    analyze_center_attraction(center_trace, signature=42, label="B")
    print(f"  12 顶点球存活: {n_vertex_alive[-1]}/12, "
          f"30 边中点球存活: {n_mid_alive[-1]}/30")
    print(f"  表面度数分布(末): {degree_histogram_str(pa)}")

    # 12 顶点组的表面-表面度数
    sn = surface_surf_neighbors(pa)
    v_deg = [sn[i] for i in range(1, 13) if pa.active[i]]
    m_deg = [sn[i] for i in range(13, 43) if pa.active[i]]
    if v_deg:
        print(f"  顶点组表面-表面度: min={min(v_deg)} max={max(v_deg)} "
              f"avg={sum(v_deg)/len(v_deg):.2f} (理论 5)")
    if m_deg:
        print(f"  边中点组表面-表面度: min={min(m_deg)} max={max(m_deg)} "
              f"avg={sum(m_deg)/len(m_deg):.2f}")
    report_12_30(pa, 0, "B末")
    gap_position_report(pa, 0, "B末")
    return pa, center_trace


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 400

    paA, traceA = experiment_A(frames)
    paB, traceB = experiment_B(frames)

    print("\n" + "=" * 72)
    print("  四问结论")
    print("=" * 72)
    print("  Q1 被 42 吸引 / Q2 偏差永不归零: 见各实验 '签名 42 吸引' 报告")
    print("  Q3 12 顶点 30 边涌现: 见各实验 deg=5 检测")
    print("  Q4 空隙球在边中点: 见各实验 gap 球角差 (越小越接近边中点)")
    print("=" * 72)
