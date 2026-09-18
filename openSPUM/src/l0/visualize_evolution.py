# -*- coding: utf-8 -*-
"""visualize_evolution.py — OpenSPUM L0 演化通用可视化模块。

依赖：matplotlib (Agg 后端) + networkx + numpy
对应锚点：docs/L0L1L2_Architecture.md §7.5（方向 4 σ 场梯度涌现读数）

设计
----
  · 通用：覆盖任意 seed/cap/dmin/vminus/vplus 配置
  · 双入口：
      visualize_run(...)        — 跑帧演化 + 渲染
      visualize_from_stats(...) — 从已有 stats 列表渲染（让 probe_sigma_emergence 直接复用）
  · CLI：python visualize_evolution.py seed=icosa nframes=10 vminus=dense out=...

面板（3×3 网格，单张 PNG）
----
  (0,0) V / E 时序           (0,1) σ_global/min/max    (0,2) ★ |∇σ|_mean/max（方向 4 核心）
  (1,0) deg_min/mean/max+χ   (1,1) σ_max/σ_min ratio   (1,2) 度分布热图（X=t,Y=度数,色=节点数）
  (2,0..2) 网络快照对比：t=0 / t=N/2 / t=N，节点按 σ 着色

性能
----
  · sigma_stats 不修改，单独跑 deg_hist（解耦）
  · 网络快照在 V>500 时退到 random_layout；V>2000 时跳过快照
  · deepcopy RotNet 仅在快照帧执行（默认 t=0 与 t=N）
"""
import sys, os, copy
from collections import Counter
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_L1 = os.path.join(os.path.dirname(_HERE), "l1")
if _L1 not in sys.path:
    sys.path.insert(0, _L1)

import matplotlib
matplotlib.use("Agg")  # 无显示环境下输出 PNG
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np

# Windows CJK 字体配置（必须在 plt import 之后；按可用性优先级）
# 注意：rcParams 在 plt import 时被加载为默认值，此前设置无效
_CJK_CANDIDATES = ["Microsoft YaHei", "SimHei", "SimSun", "Microsoft JhengHei"]
_resolved_cjk = None
for _f in _CJK_CANDIDATES:
    try:
        _path = _fm.findfont(_f, fallback_to_default=False)
        _resolved_cjk = _f
        break
    except Exception:
        continue
if _resolved_cjk:
    matplotlib.rcParams["font.sans-serif"] = [_resolved_cjk, "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正确渲染

from combinatorial_proto import RotNet, SEEDS
from l0_core import L0Core
from probe_sigma_emergence import sigma_stats
from l1_projection import sigma_of

FIGURES_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "docs", "figures"))


# ============================================================
# 采集 / 转换工具
# ============================================================
def _collect_with_hist(net):
    """sigma_stats + 度直方图。不修改 sigma_stats（已冻结）。"""
    s = sigma_stats(net)
    s["deg_hist"] = Counter(net.deg(v) for v in net.ids())
    return s


def net_to_nx(net):
    """RotNet → networkx.Graph，带 deg 属性。"""
    rot = net.rot if isinstance(net, RotNet) else net
    g = nx.Graph()
    for v in rot:
        g.add_node(v, deg=len(rot[v]))
    for v in rot:
        for w in rot[v]:
            if v < w:
                g.add_edge(v, w)
    return g


def _layout(g, n):
    """按规模选 layout。V 大时 spring 太慢，退到 random。"""
    if n <= 1:
        return {list(g.nodes())[0]: (0.5, 0.5)}
    if n <= 300:
        k = 2.0 / np.sqrt(n) if n > 1 else 0.1
        return nx.spring_layout(g, seed=42, k=k, iterations=30)
    if n <= 2000:
        return nx.random_layout(g, seed=42)
    return None  # 跳过


def snapshot_axes(net, ax, title=""):
    """在 ax 上画网络快照，节点按 σ 着色、大小按 deg 缩放。"""
    g = net_to_nx(net)
    ids = list(g.nodes())
    n = len(ids)
    if n == 0:
        ax.text(0.5, 0.5, "(空图)", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return

    pos = _layout(g, n)
    if pos is None:
        ax.text(0.5, 0.5, f"V={n} 太大，跳过快照",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        return

    degs = [g.nodes[v]["deg"] for v in ids]
    sigs = [sigma_of(d) for d in degs]
    sizes = [max(15, 8 + 3 * d) for d in degs]

    nodes = nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=sigs, node_size=sizes,
        cmap="viridis", edgecolors="black", linewidths=0.3,
    )
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, width=0.5)

    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if nodes is not None:
        plt.colorbar(nodes, ax=ax, fraction=0.04, pad=0.04, label="σ_v")


# ============================================================
# 渲染
# ============================================================
def visualize_from_stats(stats, snapshots=None, out=None, dpi=120,
                         title_prefix="OpenSPUM L0 演化"):
    """从 stats 列表渲染多面板 PNG。

    stats: sigma_stats 输出列表（含 deg_hist 字段则用度分布热图，否则占位）
    snapshots: dict {t: RotNet} 关键帧网络快照（可选）
    out: 输出 PNG 路径；默认 docs/figures/_evo.png
    """
    if out is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        out = os.path.join(FIGURES_DIR, "_evo.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    n = len(stats)
    ts = list(range(n))

    Vs = [s["V"] for s in stats]
    Es = [s["E"] for s in stats]
    sg = [s["sigma_global"] for s in stats]
    s_min = [s["sigma_min"] for s in stats]
    s_max = [s["sigma_max"] for s in stats]
    grad_mean = [s["grad_mean"] for s in stats]
    grad_max = [s["grad_max"] for s in stats]
    deg_min = [s["deg_min"] for s in stats]
    deg_max = [s["deg_max"] for s in stats]
    deg_mean = [s["deg_mean"] for s in stats]
    chi_vals = [s["chi"] for s in stats]
    chi_numeric = [c if isinstance(c, (int, float)) else float("nan") for c in chi_vals]
    ratios = [s["sigma_ratio"] if s["sigma_ratio"] != float("inf") else None for s in stats]

    fig = plt.figure(figsize=(16, 11), dpi=dpi)
    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.45, wspace=0.35,
        height_ratios=[1, 1, 1.3],
    )

    # ── (0,0) V / E 时序
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.plot(ts, Vs, "o-", label="V", color="steelblue", markersize=4, lw=1.5)
    ax00.plot(ts, Es, "s-", label="E", color="coral", markersize=4, lw=1.5)
    ax00.set_xlabel("帧 t")
    ax00.set_ylabel("数量")
    ax00.set_title("V / E 时序", fontsize=11)
    ax00.legend(fontsize=9)
    ax00.grid(alpha=0.3)

    # ── (0,1) σ_global/min/max + 1/3 参考线
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(ts, sg, "o-", label="σ_global", color="black", markersize=4, lw=1.5)
    ax01.plot(ts, s_min, "v-", label="σ_min", color="steelblue", markersize=4, lw=1.5)
    ax01.plot(ts, s_max, "^-", label="σ_max", color="coral", markersize=4, lw=1.5)
    ax01.axhline(y=1.0/3.0, color="gray", ls="--", lw=0.8, alpha=0.7,
                label="1/3 (deg=6 平坦)")
    ax01.set_xlabel("帧 t")
    ax01.set_ylabel("σ")
    ax01.set_title("σ 场演化", fontsize=11)
    ax01.legend(fontsize=9, loc="best")
    ax01.grid(alpha=0.3)

    # ── (0,2) ★ σ 场梯度涌现（方向 4 核心）
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.plot(ts, grad_mean, "o-", label="|grad σ|_mean",
              color="darkred", markersize=4, lw=1.5)
    ax02.plot(ts, grad_max, "s-", label="|grad σ|_max",
              color="darkorange", markersize=4, lw=1.5)
    ax02.axhline(y=0, color="black", lw=0.5, ls="--")
    ax02.set_xlabel("帧 t")
    ax02.set_ylabel("|grad σ|")
    ax02.set_title("★ σ 场梯度涌现（方向 4）", fontsize=11)
    ax02.legend(fontsize=9, loc="best")
    ax02.grid(alpha=0.3)

    # ── (1,0) 度数 + χ
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(ts, deg_max, "^-", label="deg_max", color="coral", markersize=4, lw=1.5)
    ax10.plot(ts, deg_mean, "o-", label="deg_mean", color="black", markersize=4, lw=1.5)
    ax10.plot(ts, deg_min, "v-", label="deg_min", color="steelblue", markersize=4, lw=1.5)
    ax10.axhline(y=6, color="gray", ls="--", lw=0.8, alpha=0.7,
                label="κ=6 平坦")
    ax10.set_xlabel("帧 t")
    ax10.set_ylabel("度数")
    ax10.set_title("度数范围 + χ", fontsize=11)
    ax10.legend(fontsize=9, loc="upper left")
    ax10.grid(alpha=0.3)

    chi_unique = set(c for c in chi_vals if c != "?")
    if chi_unique:
        ax10b = ax10.twinx()
        ax10b.plot(ts, chi_numeric, "k:", label="χ", alpha=0.5)
        ax10b.set_ylabel("χ", color="gray")
        ax10b.legend(loc="upper right", fontsize=9)
    else:
        ax10.text(0.02, 0.96, "χ: 跳过审计（性能）",
                  transform=ax10.transAxes,
                  fontsize=8, color="gray", va="top")

    # ── (1,1) σ_max/σ_min ratio
    ax11 = fig.add_subplot(gs[1, 1])
    finite_ts = [t for t, r in zip(ts, ratios) if r is not None]
    finite_rs = [r for r in ratios if r is not None]
    if finite_rs:
        ax11.plot(finite_ts, finite_rs, "o-",
                  label="σ_max/σ_min", color="purple", markersize=4, lw=1.5)
    ax11.set_xlabel("帧 t")
    ax11.set_ylabel("σ_max / σ_min")
    ax11.set_title("σ 不均匀性 (ratio)", fontsize=11)
    ax11.axhline(y=1, color="gray", ls="--", lw=0.8, alpha=0.7,
                 label="1 (均匀)")
    ax11.legend(fontsize=9)
    ax11.grid(alpha=0.3)

    # ── (1,2) 度分布热图（X=t, Y=度数, 色=节点数）
    ax12 = fig.add_subplot(gs[1, 2])
    has_hist = "deg_hist" in stats[0]
    if has_hist:
        # 全局度范围
        all_degs = set()
        for s in stats:
            all_degs.update(s["deg_hist"].keys())
        d_min = min(all_degs)
        d_max = max(all_degs)
        n_deg = d_max - d_min + 1
        mat = np.zeros((n_deg, n), dtype=float)
        for j, s in enumerate(stats):
            for d, c in s["deg_hist"].items():
                mat[d - d_min, j] = c
        # 按列归一化（每帧节点数不同，看相对分布）
        col_sum = mat.sum(axis=0, keepdims=True)
        col_sum[col_sum == 0] = 1.0
        mat_norm = mat / col_sum

        im = ax12.imshow(mat_norm, aspect="auto", origin="lower",
                         cmap="YlOrRd", interpolation="nearest",
                         extent=[-0.5, n - 0.5, d_min - 0.5, d_max + 0.5])
        ax12.set_xlabel("帧 t")
        ax12.set_ylabel("度数")
        ax12.set_title("度分布热图（按帧归一）", fontsize=11)
        ax12.axhline(y=6, color="cyan", ls="--", lw=0.8, alpha=0.6,
                     label="κ=6")
        plt.colorbar(im, ax=ax12, fraction=0.04, pad=0.04, label="占比")
    else:
        ax12.text(0.5, 0.5, "(度分布热图\n需 stats[i]['deg_hist'])",
                  transform=ax12.transAxes, ha="center", va="center",
                  fontsize=10, color="gray")
        ax12.set_title("度分布（占位）", fontsize=11)
        ax12.axis("off")

    # ── (2, 0..2) 网络快照
    if snapshots:
        snap_ts = sorted(snapshots.keys())[:3]
        for i, t in enumerate(snap_ts):
            ax = fig.add_subplot(gs[2, i])
            idx = min(t, n - 1)
            s = stats[idx]
            title = (f"t={t}  V={s['V']}  |grad σ|_mean={s['grad_mean']:.4f}  "
                     f"σ_ratio={s['sigma_ratio']:.2f}")
            snapshot_axes(snapshots[t], ax, title=title)

    fig.suptitle(title_prefix, fontsize=13, y=0.995)
    meta = (f"生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"matplotlib {matplotlib.__version__} | networkx {nx.__version__}")
    fig.text(0.99, 0.005, meta, ha="right", va="bottom",
             fontsize=7, color="gray")

    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"已渲染: {out}")
    return out


# ============================================================
# 跑演化 + 渲染
# ============================================================
def visualize_run(seed="icosa", nframes=10, cap=12, dmin=3,
                  vminus="dense", vplus="any", out=None,
                  snapshot_frames=None, dpi=120):
    """跑帧演化 + 渲染多面板图。

    snapshot_frames: list of int，要保存 RotNet 副本的帧序号。
                     默认 [0, nframes]；V 大时 deepcopy 慢，应少存。
    """
    if snapshot_frames is None:
        snapshot_frames = [0, nframes]
    snapshot_frames = set(snapshot_frames)

    if seed not in SEEDS:
        raise ValueError(f"未知 seed={seed}；可选: {list(SEEDS.keys())}")

    net = RotNet(SEEDS[seed]())
    core = L0Core(net, cap=cap, dmin=dmin, vminus=vminus, vplus=vplus)

    snapshots = {}
    if 0 in snapshot_frames:
        snapshots[0] = copy.deepcopy(core.net)

    stats = [_collect_with_hist(core.net)]
    for t in range(1, nframes + 1):
        core.frame()
        stats.append(_collect_with_hist(core.net))
        if t in snapshot_frames:
            snapshots[t] = copy.deepcopy(core.net)

    if out is None:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        out = os.path.join(
            FIGURES_DIR,
            f"evo_{seed}_{vminus}_{vplus}_cap{cap}_dmin{dmin}_N{nframes}.png",
        )

    title = (f"OpenSPUM L0 演化  seed={seed} cap={cap} dmin={dmin} "
             f"vminus={vminus} vplus={vplus} (N={nframes})")
    return visualize_from_stats(stats, snapshots=snapshots, out=out,
                                dpi=dpi, title_prefix=title)


# ============================================================
# selftest
# ============================================================
def selftest():
    """最小冒烟：3 帧演化 + 默认快照 + PNG 落盘。"""
    print("=" * 78)
    print("[visualize_evolution] selftest")
    print("=" * 78)
    ok = True
    out = os.path.join(FIGURES_DIR, "_selftest.png")
    try:
        path = visualize_run(seed="icosa", nframes=3, cap=12, dmin=3,
                             vminus="dense", vplus="any", out=out, dpi=80)
        ok &= os.path.exists(path) and os.path.getsize(path) > 1000
        print(f"  ① 3 帧演化图生成  {path}  "
              f"({os.path.getsize(path) // 1024} KB)  "
              f"[{'OK' if ok else 'FAIL'}]")
    except Exception as e:
        ok = False
        print(f"  ① 失败: {e}  [FAIL]")
    print(f"\n  selftest: {'全部通过' if ok else '存在失败项'}")
    return ok


if __name__ == "__main__":
    kv = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v

    if kv.get("selftest", "0") == "1":
        selftest()
    else:
        seed = kv.get("seed", "icosa")
        nf = int(kv.get("nframes", 10))
        cap = int(kv.get("cap", 12))
        vm = kv.get("vminus", "dense")
        vp = kv.get("vplus", "any")
        dmin = int(kv.get("dmin", 3))
        out = kv.get("out", None)
        dpi = int(kv.get("dpi", 120))

        # 默认 3 帧快照：t=0 / t=N/2 / t=N
        snaps = [0, max(1, nf // 2), nf]

        visualize_run(seed=seed, nframes=nf, cap=cap, dmin=dmin,
                      vminus=vm, vplus=vp, out=out, dpi=dpi,
                      snapshot_frames=snaps)
