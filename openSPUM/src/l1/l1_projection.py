# -*- coding: utf-8 -*-
"""l1_projection.py — L1 投影层：从 id + 环序生成网络与局部量（不参与 L0 规则）。

架构基准: docs/L0L1L2_Architecture.md §6.3
路径来源: §7 分岔 N4 —— 采纳 **N4-B 局部角容量投影**，排除 **N4-A 全局嵌入求解**

一句话
------
**只要知道「谁连着谁」（id + 环序），投影层就能形成网络。**

本模块**没有**任何全局嵌入、迭代或坐标：每个量只依赖 v 自身、v 的邻居，
以及含 v 的面的三个顶点的 `deg`。§6.3 的四项在此逐条实现：

  ① 网络骨架   ：节点 = id，边 = 「谁连着谁」（`edge_list`，直接取自旋转系统）
  ② 局部方向场 ：`local_frame` —— 半径 r(deg)，deg 个邻接方向按**环序**铺在该点自身球面上
                  （`ring_sphere`：环序闭合纬线环，§7 分岔 N6 采纳；fibonacci 保留作对照）
  ③ 局部量     ：`sigma_of` / `grad_sigma` / `angle_defect`（几何角亏 2π−A_v）
  ④ 12/42 识别 ：`crystallites`（纯组合判据，委托 `combinatorial_proto.find_crystallites`）

局部曲率读数 `A_v` / `δ_v`（纯局部 O(F)）
----------------------------------------
  把每条边视为「两端球相切」后，三角形 (i,j,k) 的三边长即 r_i+r_j、r_j+r_k、r_k+r_i，
  三个内角由余弦定理唯一确定。围绕 v 的三角形在 v 处张角之和记为 `A_v`，则
  `δ_v = 2π − A_v` 是 v 处的**内禀曲率符号**读数（`δ_v > 0` 凸折、`δ_v < 0` 鞍点）。
  ⚠️ **语义修订（2026-09-13 分岔 N7）**：`A_v > 2π` **不蕴含「嵌入不可能」**。
     `A_v` 等价于 v 的环序邻居方向构成的**闭球面多边形总长**，闭多边形的总长可以
     > 2π（实测 t=1 有 12 个 `A_v/2π = 1.3820` 的点，同时存在精确相切嵌入：边残差
     3.6e-15、非边最小间隙 +3.85）。2π 界只对「凸多面体 / 可平面展开」成立，而
     **鞍点（负曲率）顶点在 R³ 中完全可嵌入**。故 `viol` **降级为曲率读数，不是
     可行性判据**（详见 §7 分岔 N7；N4 判决 1 的「必要条件」部分由此被证伪）。
  ⚠️ `def_sum`（Σ(2π−A_v)）是**聚合审计读数**，不是求解输入：纯三角剖分帧恒 = 4π
     （高斯-博内自动成立，见 §7 分岔 N4 读数表注）。当 `nh > 0`（存在洞）时
     角只由三角面贡献，`defect` 不完整 —— 故 `project` 另给 `defect_complete`。

用法
----
  python l1_projection.py selftest=1
  python l1_projection.py seed=icosa nframes=4 cap=12 dmin=3 vminus=dense
      （逐帧打印 L1 摘要；帧演化委托 l0_core，本模块自身不参与 L0 规则）
"""

import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_L0 = os.path.join(os.path.dirname(_HERE), "l0")
if _L0 not in sys.path:
    sys.path.insert(0, _L0)

from combinatorial_proto import RotNet, SEEDS, find_crystallites  # noqa: E402

KAPPA = 6          # 平坦阈值：角亏 δ(v) = κ − deg(v)；三角剖分 κ=6
R_CRIT = 2 * KAPPA  # T8 临界半径 D_max = 4κ ⇒ r ≤ 2κ


# ============================================================
# §6.3 ② 局部方向场：半径律 + 球面方向
# ============================================================
def fibonacci_sphere(n):
    """n 个近似均匀分布的单位方向（球面螺旋）。n <= 0 返回空表。

    保留作**对照**：螺旋首尾不闭合，与环序（循环序）不匹配 —— 见分岔 N6。
    """
    out = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        rr = math.sqrt(max(0.0, 1.0 - z * z))
        phi = math.pi * (3.0 - math.sqrt(5.0)) * i
        out.append((rr * math.cos(phi), rr * math.sin(phi), z))
    return out


def ring_sphere(n, z0=None):
    """n 个铺在一条**闭合纬线**上的单位方向（环序方向场，分岔 N6 采纳）。

    phi_i = 2πi/n 按索引闭合 ⇒ 与 rot[v] 的**循环环序**天然匹配（fibonacci 螺旋
    首尾不闭合）。纬线高度取 z0 = −1/√n：该值下 deg=5 的 n 个方向两两夹角恰为
    正二十面体的 63.435°（故 icosa 的树边边长精确 = 2r(5)，见 §7 分岔 N6）。
    """
    if n <= 0:
        return []
    z0 = (-1.0 / math.sqrt(n)) if z0 is None else z0
    rr = math.sqrt(max(0.0, 1.0 - z0 * z0))
    return [(rr * math.cos(2.0 * math.pi * i / n),
             rr * math.sin(2.0 * math.pi * i / n), z0) for i in range(n)]


def radius_of(deg, kappa=KAPPA):
    """A2 面积预算下限 r(deg) = (κ/2)·√(deg/π)（§A.13.1）。

    由「每个切点在球面独占面积 κ²」+「临界直径 D_max = 4κ」联立得到。
    该半径下球面**恰好**铺满 deg 个切点（A2 利用率 ≡ 1）；
    r(deg) 触 R_crit = 2κ 时 deg = π(4κ)²/κ² = 16π ≈ 50.27。
    """
    return (kappa / 2.0) * math.sqrt(deg / math.pi)


def sigma_of(deg):
    """局部 σ_v = 2/deg_v（由全局 σ = |P|/|ε| 与 ⟨deg⟩ = 2/σ 局部化）。"""
    return 2.0 / deg if deg > 0 else float("inf")


def local_frame(net, v, kappa=KAPPA, mode="ring"):
    """节点 v 的局部标架：(r_v, [(邻居, 单位方向), ...])。

    方向按 **rot[v] 的环序**依次取 `ring_sphere(deg)`（默认，分岔 N6 采纳）——
    环序是纯组合数据，故「方向场」完全由「谁连着谁」决定；各点各自一套局部标架，
    点间无需对齐（对齐由 L2 的平行移动承担）。`mode="fib"` 取 fibonacci 螺旋作对照。
    """
    ring = list(net.rot[v])
    dirs = ring_sphere(len(ring)) if mode == "ring" else fibonacci_sphere(len(ring))
    return radius_of(len(ring), kappa), list(zip(ring, dirs))


# ============================================================
# §6.3 ① 网络骨架
# ============================================================
def edge_list(net):
    """去重无向边表（网络骨架的边），确定性升序。纯 id，不需坐标。"""
    es = set()
    for v in net.ids():
        for w in net.rot[v]:
            es.add((v, w) if v < w else (w, v))
    return sorted(es)


# ============================================================
# §6.3 ③ 局部量
# ============================================================
def angle_defect(net, kappa=KAPPA):
    """几何角亏 `δ_v = 2π − A_v`（**内禀曲率读数**，分岔 N7 修订）。

    A_v = 所有含 v 的**三角面**在 v 处的内角之和。三角面 (i,j,k) 的三边长取
    r_i+r_j、r_j+r_k、r_k+r_i，内角由余弦定理给出 —— 只依赖三点 deg，纯局部。
    `A_v` 亦等价于 v 的环序邻居方向构成的**闭球面多边形总长**。

    ⚠️ `δ_v > 0` = 凸折、`δ_v < 0` = 鞍点，**两者都可嵌入 R³**（分岔 N7 实测证伪
    「`A_v > 2π` ⇒ 不可嵌入」：t=1 的 12 个越界点同时拥有精确相切嵌入）。故
    `viol` 只是「平面角总和 > 2π」的曲率读数，**不是可行性判据**。

    返回 {"ang", "defect", "viol", "ntri", "nf", "nh", "def_sum"}：
      viol = [v : A_v > 2π]（曲率读数，**非**可行性判据）
      ntri[v] = 含 v 的三角面数；nh = 非三角面（洞）数
    """
    ids = net.ids()
    r = {v: radius_of(net.deg(v), kappa) for v in ids}
    ang = {v: 0.0 for v in ids}
    ntri = {v: 0 for v in ids}
    nf = nh = 0
    for cyc in net.faces():
        vs = [d[0] for d in cyc]
        if len(vs) != 3 or len(set(vs)) != 3:
            nh += 1
            continue
        i, j, k = vs
        s_ij = r[i] + r[j]
        s_jk = r[j] + r[k]
        s_ki = r[k] + r[i]
        for (a, sab, sac, sbc) in ((i, s_ij, s_ki, s_jk),
                                   (j, s_ij, s_jk, s_ki),
                                   (k, s_ki, s_jk, s_ij)):
            c = (sab * sab + sac * sac - sbc * sbc) / (2.0 * sab * sac)
            ang[a] += math.acos(max(-1.0, min(1.0, c)))
        ntri[i] += 1
        ntri[j] += 1
        ntri[k] += 1
        nf += 1
    defect = {v: 2.0 * math.pi - ang[v] for v in ids}
    viol = sorted(v for v in ids if ang[v] > 2.0 * math.pi + 1e-9)
    return {"ang": ang, "defect": defect, "viol": viol, "ntri": ntri,
            "nf": nf, "nh": nh, "def_sum": sum(defect.values())}


def grad_sigma(net):
    """∇σ 的局部化：分量 = mean_{w∈rot[v]}(σ_w) − σ_v。纯局部。"""
    ids = net.ids()
    sg = {v: sigma_of(net.deg(v)) for v in ids}
    out = {}
    for v in ids:
        nb = net.rot[v]
        if not nb:
            continue
        out[v] = sum(sg[w] for w in nb) / len(nb) - sg[v]
    return out


# ============================================================
# §6.3 ④ 12 晶子闭环 / 42 团簇（纯组合，委托 L0 侧判据）
# ============================================================
def crystallites(net):
    """5-正则闭合诱导子图分量（含 12 点正二十面体判定）。纯组合，无几何。"""
    return find_crystallites(net)


# ============================================================
# 一次性投影
# ============================================================
def project(net, kappa=KAPPA):
    """从 id + 环序一次性产出全部 L1 局部量（无求解、无迭代、无坐标）。"""
    ids = net.ids()
    ad = angle_defect(net, kappa)
    gs = grad_sigma(net)
    nodes = {}
    for v in ids:
        dv = net.deg(v)
        nodes[v] = {"deg": dv, "r": radius_of(dv, kappa), "sigma": sigma_of(dv),
                    "grad_sigma": gs.get(v), "defect": ad["defect"][v],
                    "n_incident_tri": ad["ntri"][v]}
    edges = edge_list(net)
    V, E = len(ids), len(edges)
    return {"nodes": nodes, "edges": edges, "V": V, "E": E,
            "sigma_global": (V / E) if E else float("inf"),
            "viol": ad["viol"], "nf": ad["nf"], "nh": ad["nh"],
            "def_sum": ad["def_sum"], "defect_complete": ad["nh"] == 0,
            "crystallites": crystallites(net)}


# ============================================================
# 自检
# ============================================================
def selftest():
    def close(a, b, tol=1e-9):
        return abs(a - b) <= tol

    # ---- 方向场 ----
    for n in (0, 1, 5, 12):
        ds = fibonacci_sphere(n)
        assert len(ds) == n
        for (x, y, z) in ds:
            assert close(x * x + y * y + z * z, 1.0, 1e-9)

    # ---- tetra：V=4 E=6 F=4，全 deg=3，等边 ⇒ A_v=π、defect=π、Σ=4π ----
    t = RotNet(SEEDS["tetra"]())
    ad = angle_defect(t)
    assert (len(t.ids()), t.E(), ad["nf"], ad["nh"]) == (4, 6, 4, 0)
    assert all(t.deg(v) == 3 for v in t.ids())
    assert all(close(ad["ang"][v], math.pi, 1e-9) for v in t.ids())
    assert all(close(ad["defect"][v], math.pi, 1e-9) for v in t.ids())
    assert close(ad["def_sum"] / (2 * math.pi), 2.0, 1e-9)   # Σ=4π
    assert ad["viol"] == []

    # ---- icosa：V=12 E=30 F=20，全 deg=5 ⇒ A_v=5π/3、defect=π/3、Σ=4π ----
    ic = RotNet(SEEDS["icosa"]())
    ad = angle_defect(ic)
    assert (len(ic.ids()), ic.E(), ad["nf"], ad["nh"]) == (12, 30, 20, 0)
    assert all(ic.deg(v) == 5 for v in ic.ids())
    assert all(close(ad["ang"][v] / (2 * math.pi), 5.0 / 6.0, 1e-9) for v in ic.ids())
    assert all(close(ad["defect"][v], math.pi / 3.0, 1e-9) for v in ic.ids())
    assert close(ad["def_sum"] / (2 * math.pi), 2.0, 1e-9)   # Σ=4π
    assert ad["viol"] == []
    assert close(radius_of(5), 3.7846988, 1e-6)   # 3·√(5/π)
    assert close(sigma_of(5), 0.4, 1e-12)
    assert all(close(g, 0.0, 1e-12) for g in grad_sigma(ic).values())

    # ---- 网络骨架 / 局部标架 ----
    assert len(edge_list(ic)) == ic.E() == 30
    r5, frame = local_frame(ic, 0)
    assert len(frame) == ic.deg(0) == 5 and close(r5, radius_of(5))
    assert [w for (w, _) in frame] == list(ic.rot[0])

    # ---- 分岔 N6：环序闭合纬线环方向场（采纳）+ fibonacci 对照 ----
    assert ring_sphere(0) == []
    for n in (1, 2, 3, 5, 12, 50):
        ds = ring_sphere(n)
        assert len(ds) == n
        for (x, y, z) in ds:
            assert close(x * x + y * y + z * z, 1.0, 1e-9)
            assert close(z, -1.0 / math.sqrt(n), 1e-12)
    d5 = ring_sphere(5)
    c5 = sum(d5[0][k] * d5[1][k] for k in range(3))
    assert close(c5, 1.0 / math.sqrt(5.0), 1e-12)      # 63.4349° = 正二十面体键角
    assert frame[0][1] == ring_sphere(5)[0]            # 默认即 ring，且按环序
    assert local_frame(ic, 0, mode="fib")[1][0][1] == fibonacci_sphere(5)[0]

    # ---- project 结构自洽 ----
    p = project(ic)
    assert p["V"] == 12 and p["E"] == 30 and len(p["nodes"]) == 12
    assert close(p["sigma_global"], 0.4, 1e-12) and p["defect_complete"]
    assert p["viol"] == [] and p["crystallites"] and p["crystallites"][0]["icosa"]

    # ---- L0 采纳轨迹：曲率读数 `viol` 从 0 变非空（纯读数；语义见分岔 N7）----
    import l0_core as m                                   # 仅自检/演示需要
    c = m.L0Core(m.RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                 vminus="dense", vplus="any")
    assert angle_defect(c.net)["viol"] == []               # t=0：等度 ⇒ 全凸折
    c.frame()
    ad1 = angle_defect(c.net)
    assert len(ad1["viol"]) == 12                          # t=1：12 个鞍点（A_v>2π）
    assert close(max(ad1["ang"].values()) / (2 * math.pi), 1.3820, 1e-3)

    print("l1_projection selftest: 全部通过")
    print("  tetra: A_v=π、defect=π、Σ=4π；icosa: A_v=5π/3、defect=π/3、Σ=4π、∇σ≡0")
    print(f"  icosa 局部标架: r(5)={radius_of(5):.6f}、5 个方向按环序 ✓；骨架 30 条边 ✓")
    print(f"  L0 采纳轨迹: t=0 鞍点 0 → t=1 鞍点 {len(ad1['viol'])}"
          f"（max A_v/2π={max(ad1['ang'].values())/(2*math.pi):.4f}，曲率读数）")


# ============================================================
# CLI
# ============================================================
def _demo(seed="icosa", nframes=4, cap=12, dmin=3, vminus="dense",
          vplus="any", kappa=KAPPA):
    import l0_core as m
    c = m.L0Core(m.RotNet(SEEDS[seed]()), cap=cap, dmin=dmin,
                 vminus=vminus, vplus=vplus, kappa=kappa)
    print(f"== L1 局部投影（seed={seed} cap={cap} dmin={dmin} "
          f"{vminus}/{vplus} κ={kappa}）==")
    print(f"r(deg)=(κ/2)√(deg/π)；R_crit=2κ={R_CRIT}；"
          f"饱和 deg_sat=16π≈{math.pi*(4*kappa)**2/kappa**2:.2f}")
    print(f"{'t':>3} {'V':>5} {'E':>6} {'σ=V/E':>7} {'r_max':>8} {'r/Rcrit':>8} "
          f"{'A_v/2π max':>10} {'越界':>5} {'Σdef/2π':>8} {'洞':>4}")
    print("-" * 76)
    for t in range(nframes + 1):
        p = project(c.net, kappa)
        degs = [n["deg"] for n in p["nodes"].values()]
        rmax = max((n["r"] for n in p["nodes"].values()), default=0.0)
        amax = 0.0
        if p["defect_complete"] and p["nodes"]:
            ad = angle_defect(c.net, kappa)
            amax = max(ad["ang"].values()) / (2 * math.pi)
        amax_s = f"{amax:>10.4f}" if p["defect_complete"] else f"{'—':>10}"
        defsum_s = f"{p['def_sum']/(2*math.pi):>8.4f}" if p["defect_complete"] else f"{'—':>8}"
        print(f"{t:>3} {p['V']:>5} {p['E']:>6} {p['sigma_global']:>7.4f} "
              f"{rmax:>8.5f} {rmax/R_CRIT:>8.5f} {amax_s} "
              f"{len(p['viol']):>5} {defsum_s} {p['nh']:>4}"
              + (f"  maxdeg={max(degs)}" if degs else ""))
        if t < nframes:
            c.frame()
    print("（— = 该帧存在洞，角只由三角面贡献 ⇒ 角亏聚合读数不完整，"
          "越界判据 viol 仍有效）")


def _kv(argv):
    d = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            d[k] = v
    return d


def main(argv):
    kv = _kv(argv)
    if "selftest" in kv:
        selftest()
        return
    if "seed" in kv or "nframes" in kv:
        _demo(seed=kv.get("seed", "icosa"), nframes=int(kv.get("nframes", 4)),
              cap=int(kv.get("cap", 12)), dmin=int(kv.get("dmin", 3)),
              vminus=kv.get("vminus", "dense"), vplus=kv.get("vplus", "any"),
              kappa=int(kv.get("kappa", KAPPA)))
        return
    print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
