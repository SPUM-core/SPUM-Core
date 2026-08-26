"""
不完美定理的标准演示 — 量子真空涨落的认知投影机制
=================================================

演示目的：
    将「不完美定理」（GT-002 / spum-core §2.3）从文字断言落地为可运行、
    可复现的图论演化演示，并展示它与「量子真空涨落」的关系：

    [L0] 本体：帧事件确定发生。每帧删除悬挂边，必生新悬挂边。
               节点要么连接，要么不连接——没有随机性，没有"涨落"。
    [L1] 认知：观察者读不到完整拓扑，把帧间残留的、不可消除的活动
               投影为"真空涨落"——与连续是离散的投影同一类。

演示内容：
    1. 在均匀随机网络（模拟近均匀 σ 的"真空"区域）上执行五步帧演化
    2. 每帧记录：δ(τ) 悬挂端密度、V⁺/V⁻ 事件数、d_topo 帧间距离、σ
    3. 验证不完美定理：δ(τ) 在全部帧中永不归零
    4. 涨落谱分析：ΔV(τ)=V⁺+V⁻ 活动量序列的功率谱——离散、帧频截断
    5. 输出图集 + 数值报告

运行：
    python src/spum_graph/demo_imperfection.py [--n 200] [--p 0.015]
        [--frames 500] [--seed 42] [--out src/spum_graph/imperfection_demo]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import FrameGraph  # noqa: E402


# ============================================================
# 1. 初始"真空"网络
# ============================================================

def seed_vacuum(n: int, p: float, rng: np.random.Generator) -> FrameGraph:
    """生成均匀随机网络（ER 图）模拟近均匀 σ 的真空区域。

    取最大连通分量作为初始帧——避免全图分裂为孤立碎片，
    但保留悬挂端（真空本就从不完美出发）。
    """
    g = FrameGraph(0)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                g.add_edge(str(i), str(j))

    # 取最大连通分量（BFS）
    nodes = list(g.nodes())
    visited = set()
    components = []
    for s in nodes:
        if s in visited:
            continue
        comp = []
        stack = [s]
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            stack.extend(g.neighbors(v))
        components.append(comp)
    largest = max(components, key=len)

    g2 = FrameGraph(0)
    for v in largest:
        for u in g.neighbors(v):
            if v < u:  # 无向边只加一次
                g2.add_edge(v, u)
    return g2


# ============================================================
# 2. 单帧五步演化
# ============================================================

def one_frame(g: FrameGraph, rng: np.random.Generator, max_create: int) -> tuple:
    """执行一帧五步演化。

    五步帧：创生 V⁺ → 连接（度数重算）→ 变化体积（σ 重算）
            → 判断（查悬挂）→ 删除 V⁻

    一帧内没有级联消解——删除后新产生的悬挂端留到下一帧。

    Returns:
        (new_graph, stats)  stats = {v_plus, v_minus, d_topo, sigma}
    """
    frame_id = g.frame_id + 1
    g_new = FrameGraph(frame_id)

    # 拷贝上一帧的边集
    for v in g.nodes():
        for u in g.neighbors(v):
            if v < u:
                g_new.add_edge(v, u)

    m_before = g_new.edge_count
    v_plus = 0

    # ── 步骤 1：创生 V⁺ ──────────────────────────────────────
    # 悬挂端 = 单侧张力未配平边界 = 创生热点（∇σ 密度梯度流入口）。
    # 只要存在悬挂端，创生必沿悬挂端延伸（确定性，非概率）——
    # 机制：悬挂端 v（度 1）接上 u（度 1）→ v 度 2 配平、u 为悬挂
    #      → 删除 u → v 度 2→1 成为新悬挂。悬挂端沿链转移，永不消除。
    # 这正是不完美定理（GT-002）的引擎：删除必生新悬挂，演化永不停。
    n_create = int(rng.integers(1, max_create + 1))
    nodes = list(g_new.nodes())
    for _ in range(n_create):
        if not nodes:
            break
        dangling = [v for v in nodes if g_new.degree(v) < 2]
        if dangling:
            # 沿悬挂端延伸（链端生长）——不完美定理的引擎
            v = dangling[rng.integers(0, len(dangling))]
            u = f"n{frame_id}_{v_plus}"
            g_new.add_edge(v, u)
            v_plus += 1
        else:
            # 无悬挂端（种子有悬挂端 + 延伸确定性 → 本分支实际不可达）：
            # 附着新节点到低度节点——创生流入未饱和区域
            low = [v for v in nodes if g_new.degree(v) <= 2]
            pool = low if low else nodes
            v = pool[rng.integers(0, len(pool))]
            u = f"n{frame_id}_{v_plus}"
            g_new.add_edge(v, u)
            v_plus += 1

    # ── 步骤 2：连接（度数重算，隐式完成）────────────────────
    # ── 步骤 3：变化体积（σ 重算，见下）───────────────────────

    # ── 步骤 4-5：判断并删除悬挂边 V⁻ ────────────────────────
    v_minus = 0
    dangling = list(g_new.dangling_nodes())
    for v in dangling:
        for u in list(g_new.neighbors(v)):
            # 从 g_new 移除边 (v, u)
            g_new._neighbors[v].discard(u)
            g_new._neighbors[u].discard(v)
            g_new._edge_count -= 1
            v_minus += 1
    # 度数降为 0 的节点同步删除（不再存在——关系定义存在）
    zero = [v for v in list(g_new._neighbors) if not g_new._neighbors[v]]
    for v in zero:
        del g_new._neighbors[v]

    m_after = g_new.edge_count
    # 帧间拓扑距离 d_topo = 对称差边数 / max(|E_τ|, |E_{τ+1}|)
    d_topo = (v_plus + v_minus) / max(m_before, 1)

    n_nodes = g_new.node_count
    sigma = n_nodes / m_after if m_after > 0 else float("inf")

    stats = {
        "v_plus": v_plus,
        "v_minus": v_minus,
        "d_topo": d_topo,
        "sigma": sigma,
    }
    return g_new, stats


# ============================================================
# 3. 涨落谱分析
# ============================================================

def power_spectrum(seq: np.ndarray, fs: float = 1.0) -> tuple:
    """δ(τ) 序列的功率谱（实信号，单边）。

    Returns:
        (freqs, power)  频率、归一化功率
    """
    x = np.asarray(seq, dtype=float)
    x = x - x.mean()
    n = len(x)
    spec = np.fft.rfft(x)
    power = np.abs(spec) ** 2
    # 归一化（相对功率）
    power = power / power.sum() if power.sum() > 0 else power
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, power


def autocorr(seq: np.ndarray, max_lag: int) -> np.ndarray:
    """自相关函数（零均值归一化）。"""
    x = np.asarray(seq, dtype=float)
    x = x - x.mean()
    n = len(x)
    out = np.correlate(x, x, mode="full")[n - 1:]
    out = out / (out[0] if out[0] != 0 else 1.0)
    return out[:max_lag]


# ============================================================
# 4. 绘图
# ============================================================

def plot_delta(delta_seq, vp, vm, out_path, frames):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    t = np.arange(len(delta_seq))

    axes[0].plot(t, delta_seq, lw=0.7, color="#1f77b4")
    axes[0].axhline(min(delta_seq), color="#d62728", ls="--", lw=1.0,
                    label=f"min δ = {min(delta_seq):.4f}")
    axes[0].set_ylabel("δ (dangling density)")
    axes[0].set_title("Imperfection Theorem: δ(τ) never reaches zero "
                      "(vacuum fluctuation as L1 projection)")
    axes[0].legend(loc="upper right")

    axes[1].plot(t, vp, lw=0.6, color="#2ca02c", label="V⁺ (creation)")
    axes[1].plot(t, vm, lw=0.6, color="#d62728", label="V⁻ (annihilation)")
    axes[1].set_ylabel("frame events")
    axes[1].set_xlabel("frame τ")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_spectrum(freqs, power, freqs_d, power_d, out_path):
    """双面板：左 = ΔV 活动量涨落谱（主），右 = δ 悬挂端密度谱（参考）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].stem(freqs, power, linefmt="#1f77b4", markerfmt="",
                 basefmt=" ")
    axes[0].set_xlabel("frequency (cycles/frame)")
    axes[0].set_ylabel("normalized power")
    axes[0].set_title("ΔV frame-activity spectrum — the fluctuation\n"
                      "(discrete, frame-rate bounded)")
    nyq = 0.5
    axes[0].axvline(nyq, color="#d62728", ls="--", lw=1.0,
                    label=f"Nyquist = {nyq}")
    axes[0].legend(loc="upper right")

    axes[1].stem(freqs_d, power_d, linefmt="#2ca02c", markerfmt="",
                 basefmt=" ")
    axes[1].set_xlabel("frequency (cycles/frame)")
    axes[1].set_ylabel("normalized power")
    axes[1].set_title("δ(τ) dangling-density spectrum (reference)")
    axes[1].axvline(nyq, color="#d62728", ls="--", lw=1.0)
    axes[1].set_xlim(0, 0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_autocorr(ac, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stem(range(len(ac)), ac, linefmt="#ff7f0e", markerfmt="",
            basefmt=" ")
    ax.set_xlabel("lag (frames)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Autocorrelation of ΔV(τ) — frame activity fluctuation")
    ax.axhline(0, color="gray", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================
# 5. 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="不完美定理几何化演示（真空涨落机制）")
    ap.add_argument("--n", type=int, default=200, help="初始节点数")
    ap.add_argument("--p", type=float, default=0.015, help="ER 连接概率")
    ap.add_argument("--frames", type=int, default=500, help="演化帧数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--max-create", type=int, default=2, help="每帧最大创生边数")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent / "imperfection_demo"),
                    help="输出目录")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ── 演化 ──────────────────────────────────────────────
    g = seed_vacuum(args.n, args.p, rng)
    delta_seq, vp_seq, vm_seq, dtopo_seq, sigma_seq = [], [], [], [], []
    n_seq, m_seq = [], []

    for frame in range(args.frames):
        g, stats = one_frame(g, rng, args.max_create)
        delta_seq.append(g.dangling_density())
        vp_seq.append(stats["v_plus"])
        vm_seq.append(stats["v_minus"])
        dtopo_seq.append(stats["d_topo"])
        sigma_seq.append(stats["sigma"])
        n_seq.append(g.node_count)
        m_seq.append(g.edge_count)

    delta_arr = np.array(delta_seq)
    # 帧活动量 ΔV = V⁺ + V⁻——涨落的定量体现（每帧关系重分配的事件数）
    dv_arr = np.array(vp_seq) + np.array(vm_seq)

    # ── 涨落谱（对 ΔV 活动量序列）──────────────────────────
    freqs, power = power_spectrum(dv_arr)
    ac = autocorr(dv_arr, max_lag=min(60, len(dv_arr) // 2))
    # δ 谱作参考（悬挂端密度）
    freqs_d, power_d = power_spectrum(delta_arr)

    # ── 统计量 ────────────────────────────────────────────
    nonzero_frames = int(np.sum(delta_arr > 0))
    zero_frames = int(np.sum(delta_arr == 0))
    nonzero_bins = int(np.sum(power > 1e-6))
    peak_freq = freqs[np.argmax(power)] if len(freqs) else 0.0

    stats = {
        "frames": args.frames,
        "min_delta": float(delta_arr.min()),
        "mean_delta": float(delta_arr.mean()),
        "max_delta": float(delta_arr.max()),
        "std_delta": float(delta_arr.std()),
        "nonzero_frames": nonzero_frames,
        "zero_frames": zero_frames,
        "total_v_plus": int(sum(vp_seq)),
        "total_v_minus": int(sum(vm_seq)),
        "mean_dv": float(np.mean(dv_arr)),
        "std_dv": float(np.std(dv_arr)),
        "mean_d_topo": float(np.mean(dtopo_seq)),
        "mean_sigma": float(np.mean(sigma_seq)),
        "final_nodes": n_seq[-1],
        "final_edges": m_seq[-1],
        "spectrum_peak_freq": float(peak_freq),
        "spectrum_nonzero_bins": nonzero_bins,
        "autocorr_lag1": float(ac[1]) if len(ac) > 1 else 0.0,
    }

    # ── 绘图 ──────────────────────────────────────────────
    plot_delta(delta_arr, vp_seq, vm_seq, out_dir / "fig1_delta_fluctuation.png",
               args.frames)
    plot_spectrum(freqs, power, freqs_d, power_d,
                  out_dir / "fig2_spectrum.png")
    plot_autocorr(ac, out_dir / "fig3_autocorr.png")

    # ── 数值报告 ──────────────────────────────────────────
    report = _build_report(args, stats, freqs, power)
    report_path = out_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\n输出目录: {out_dir.resolve()}")


def _build_report(args, stats, freqs, power) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("不完美定理演示 — 量子真空涨落的认知投影机制")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"参数：N={args.n}, p={args.p}, 帧数={args.frames}, "
                 f"seed={args.seed}, max_create={args.max_create}")
    lines.append("")
    lines.append("── 不完美定理验证（GT-002）────────────────────────")
    lines.append(f"  δ 最小值       = {stats['min_delta']:.4f}")
    lines.append(f"  δ 均值         = {stats['mean_delta']:.4f}")
    lines.append(f"  δ 最大值       = {stats['max_delta']:.4f}")
    lines.append(f"  δ 标准差       = {stats['std_delta']:.4f}")
    lines.append(f"  δ > 0 帧占比   = {stats['nonzero_frames']}/{stats['frames']} "
                 f"({100.0 * stats['nonzero_frames'] / stats['frames']:.1f}%)")
    lines.append(f"  δ = 0 帧数     = {stats['zero_frames']}")
    conclusion = ("✅ 成立：δ 永不归零" if stats["zero_frames"] == 0
                  else "❌ 失败：存在 δ = 0 帧")
    lines.append(f"  结论           = {conclusion}")
    lines.append("")
    lines.append("── 创生-湮灭对偶（R2）─────────────────────────────")
    lines.append(f"  V⁺ 总事件      = {stats['total_v_plus']}")
    lines.append(f"  V⁻ 总事件      = {stats['total_v_minus']}")
    lines.append(f"  V⁺/V⁻ 比       = "
                 f"{stats['total_v_plus'] / max(stats['total_v_minus'], 1):.3f}")
    lines.append("")
    lines.append("── 真空状态量 ────────────────────────────────────")
    lines.append(f"  σ 均值         = {stats['mean_sigma']:.4f}")
    lines.append(f"  d_topo 均值    = {stats['mean_d_topo']:.4f}")
    lines.append(f"  终态节点数     = {stats['final_nodes']}")
    lines.append(f"  终态边数       = {stats['final_edges']}")
    lines.append("")
    lines.append("── 涨落谱分析（ΔV 活动量的认知投影）──────────────")
    lines.append(f"  ΔV 均值        = {stats['mean_dv']:.3f} 事件/帧")
    lines.append(f"  ΔV 标准差      = {stats['std_dv']:.3f}（涨落幅度）")
    lines.append(f"  谱峰频率       = {stats['spectrum_peak_freq']:.4f} "
                 f"(cycles/frame)")
    lines.append(f"  非零谱 bin 数  = {stats['spectrum_nonzero_bins']} / "
                 f"{len(freqs)}")
    lines.append(f"  Nyquist 频率   = 0.5 cycles/frame（帧采样上限）")
    lines.append(f"  自相关 lag-1   = {stats['autocorr_lag1']:.4f}")
    lines.append("")
    lines.append("── 解释（两层结构）───────────────────────────────")
    lines.append("  [L0] 本体·不完美：每帧事件确定，节点要么连接要")
    lines.append("        么不连接。删除悬挂边必生新悬挂边（GT-002）")
    lines.append("        ——δ(τ) 永不归零，演化永不停。")
    lines.append("  [L1] 认知·涨落：观察者读不到完整拓扑，把每帧")
    lines.append("        残留的 V⁺/V⁻ 活动量 ΔV 的统计波动投影为")
    lines.append("        '真空涨落'——谱离散、帧频截断，零点能有限，")
    lines.append("        无需重整化。涨落是认知现象，机制是不完美定理。")
    lines.append("")
    lines.append("── 诚实标注 ──────────────────────────────────────")
    lines.append("  本演示为图论层面（无球体几何）；Δμ 锚点漂移因真空")
    lines.append("  无类别锚点而省略；卡西米尔力的 δ→F 定量对齐为未来工作。")
    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
