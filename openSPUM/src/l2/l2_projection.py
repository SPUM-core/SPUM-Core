# -*- coding: utf-8 -*-
"""l2_projection.py — L2 全局投影层：全局坐标 + 全局读数（**工件层，永不回写 L0**）。

架构基准: docs/L0L1L2_Architecture.md §6.4
路径来源: §7 分岔 N5（L2 全局坐标怎么生成）——采纳 **N5-B 对齐传播 + N5-C 工件精修**，
          排除 N5-A 全局刚性相切嵌入（§7 分岔 N4 已判不可行）
          方向场来自 §7 分岔 N6（环序闭合纬线环 `ring_sphere`，见 `l1_projection`）

一句话
------
**L2 不求解几何，只把 L1 的局部量沿环序拼成一个可看的全局工件，并读出全局量。**

为什么要"拼"而不能"解"（分岔 N5 的判决）
----------------------------------------
  纯三角剖分满足 `E = 3V − 6`（刚性余量恒为 0），逐帧一般**没有全局相切嵌入**
  （N4 已排除全局嵌入求解）。注意：不可嵌入性的证据是**刚性计数 + 求解器不可靠
  + 范式选择**，**不是**局部角判据 —— §7 分岔 N7 已证伪「`A_v > 2π` ⇒ 不可嵌入」
  （`A_v > 2π` 只是负曲率/鞍点读数，鞍点顶点在 R³ 中完全可嵌入）。
  于是 L2 只能给出**工件**，并如实报告失真：
    · **holonomy**（非树边闭合差）= 曲率无法被平面容纳的度量 —— 这是读数，不是 bug；
    · **边残差**、**非边重叠对数** = 松弛后的残留冲突。
  §6.4 的全局坐标**明确不主张为"真实几何"**。

§6.4 的四项在此逐条实现（全部只读 net；L2 永不改 L0）
----------------------------------------------------
  ① 全局坐标   ：`aligned_tree`（沿 BFS 生成树做平行移动积分，树边精确满足
                  `|p_u−p_w| = r_u + r_w`，`O(E)` 无迭代）+ `relax`（位置约束投影精修）
  ② 全局 σ 场  ：`sigma_field`；**锚点 = 引力代理** `anchors`（σ 局部汇 / 源）
  ③ 红移       ：`redshift_proxy` —— 沿 BFS 最短路累积的 σ 相对变化（**代理读数**）
  ④ 可视化导出 ：`export_obj` / `export_json`（离线查看；节点带 r、σ、deg、defect）
  ⑤ 径向幂次谱 ：`gravity_spectrum` / `radial_bins` / `hop_integral` / `power_fit`
                  （**阶段六第③项**：把 ②③ 的代理读数升级为可证伪的四路幂次读数
                  σ、ΔN=∫σdr、∇σ、∇ΔN；目标 `a=c²∇ΔN ∝ r⁻²`；零假设对照
                  `null_control`；可分辨性由 `synthetic_calibration` 标定）
  ⑥ 有限窗口   ：`window_ball`（跳数球窗）/ `window_prefix`（`V ≤ V_window` 规模窗）
                  / `radial_window`（按箱内样本数自动选拟合窗口）/ `window_readouts`
                  / `window_stability`（**阶段六第④项**：兑现 N1-C「有限窗口由投影层
                  选取」；判词只看 `σ_window` 末级相对残差，`slope_σ` 与锚点密度
                  只作旁证——实测二者在窗口上分别「多级为 None」与「被边界效应主导」）

用法
----
  python l2_projection.py selftest=1
  python l2_projection.py seed=icosa nframes=4 cap=12 dmin=3 iters=60
      （逐帧打印 L2 摘要；帧演化委托 l0_core，本模块自身不参与 L0 规则）
  python l2_projection.py spectrum=1 seed=icosa nframes=4 cap=12 dmin=3 nbins=8
      （逐帧打印四路径向幂次斜率与判决）
  python l2_projection.py window=1 seed=icosa nframes=2 sizes=8,16,32
      （逐帧打印有限窗口的稳定性扫描：窗口读数对窗口大小是否稳定）
"""

import json
import math
import os
import sys
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for _sub in ("l0", "l1"):
    _p = os.path.join(_SRC, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from combinatorial_proto import RotNet, SEEDS  # noqa: E402
import l1_projection as L1  # noqa: E402

I3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


# ============================================================
# 0. 3x3 旋转工具（最小旋转 = 沿边的平行移动）
# ============================================================
def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def mat_apply(M, u):
    return (M[0][0] * u[0] + M[0][1] * u[1] + M[0][2] * u[2],
            M[1][0] * u[0] + M[1][1] * u[1] + M[1][2] * u[2],
            M[2][0] * u[0] + M[2][1] * u[1] + M[2][2] * u[2])


def rot_from_uv(u, v):
    """最小旋转把单位向量 u 转到单位向量 v（Rodrigues，行主序 3x3）。"""
    k = _cross(u, v)
    s = _norm(k)
    c = _dot(u, v)
    if s < 1e-12:
        if c > 0:
            return I3
        a = (1.0, 0.0, 0.0) if abs(u[0]) < 0.9 else (0.0, 1.0, 0.0)
        ax = _cross(u, a)
        n = _norm(ax)
        ax = tuple(x / n for x in ax)
        return tuple(tuple(2 * ax[i] * ax[j] - (1.0 if i == j else 0.0)
                           for j in range(3)) for i in range(3))
    kx = ((0.0, -k[2], k[1]), (k[2], 0.0, -k[0]), (-k[1], k[0], 0.0))
    kk = tuple(tuple(sum(kx[i][t] * kx[t][j] for t in range(3))
                     for j in range(3)) for i in range(3))
    f = (1.0 - c) / (s * s)
    return tuple(tuple((1.0 if i == j else 0.0) + kx[i][j] + kk[i][j] * f
                       for j in range(3)) for i in range(3))


# ============================================================
# §6.4 ① 全局坐标（投影工件）：对齐传播 + 工件精修
# ============================================================
def _radius(net, kappa=L1.KAPPA):
    return {v: L1.radius_of(net.deg(v), kappa) for v in net.ids()}


def _frames(net, kappa=L1.KAPPA, mode="ring"):
    return {v: dict(L1.local_frame(net, v, kappa, mode)[1]) for v in net.ids()}


def aligned_tree(net, root=None, kappa=L1.KAPPA, mode="ring"):
    """沿 BFS 生成树做**平行移动积分**，得到全局坐标（确定性、`O(E)`、无迭代）。

    规则（纯 id + 环序）：BFS 从 root 出发；处理边 i→w 时
      g = M_i · dir_i(w)                              （i 的全局标架下的 i→w 方向）
      p_w = p_i + (r_i + r_w)·g                       （树边精确满足相切目标长）
      M_w = R(u→−g)，其中 u = dir_w(i)                （把 w 标架的「指向 i」转到 −g）

    返回 {"pos", "frames", "tree_edges", "holonomy"}：
      holonomy[(i,w)] = 用 i 的标架预测 w 的位置与树位置之差（非树边）。
      该量**非零是几何事实**（离散曲率无法在平面容纳），不是误差 —— 见模块 docstring。
    """
    ids = net.ids()
    if not ids:
        return {"pos": {}, "frames": {}, "tree_edges": set(), "holonomy": {}}
    r = _radius(net, kappa)
    fr = _frames(net, kappa, mode)
    root = ids[0] if root is None else root
    pos = {root: (0.0, 0.0, 0.0)}
    M = {root: I3}
    tree_edges = set()
    q = deque([root])
    while q:
        i = q.popleft()
        for (w, dl) in fr[i].items():
            if w in pos:
                continue
            g = mat_apply(M[i], dl)
            pos[w] = tuple(pos[i][k] + (r[i] + r[w]) * g[k] for k in range(3))
            M[w] = rot_from_uv(fr[w][i], tuple(-x for x in g))
            tree_edges.add((i, w) if i < w else (w, i))
            q.append(w)
    hol = {}
    for (i, w) in L1.edge_list(net):
        if (i, w) in tree_edges:
            continue
        g = mat_apply(M[i], fr[i][w])
        pred = tuple(pos[i][k] + (r[i] + r[w]) * g[k] for k in range(3))
        hol[(i, w)] = math.dist(pos[w], pred)
    return {"pos": pos, "frames": M, "tree_edges": tree_edges, "holonomy": hol}


def relax(net, pos, iters=60, w_rep=0.5, nonedge=True, center=True,
          kappa=L1.KAPPA):
    """工件精修：位置约束投影（Gauss-Seidel），确定性、无步长参数。

    ① 边：距离约束 `|p_u−p_w| = r_u + r_w`，两端各移一半；
    ② 非边：`|p_u−p_w| < r_u + r_w` 时各移一半推开（`w_rep` 权重）；
    ③ 每轮质心归零（抑制整体漂移）。
    这只是**把工件压实得更像样**，不改变「不可精确嵌入」的性质。

    ⚠️ **初始化依赖（分岔 N7）**：PBD 只降到**最近的**局部极小，残差只是上界，
    **不能**当作「不可嵌入」的证据——实测同一张 t=1 的图，换 `root` 就能命中
    精确解（边残差 3.6e-15）或陷局部极小（残差 1.24、非边重叠）。
    """
    ids = net.ids()
    if not ids:
        return {}
    r = _radius(net, kappa)
    edges = L1.edge_list(net)
    p = {v: tuple(pos[v]) for v in ids}
    nb = {v: set() for v in ids}
    for (u, w) in edges:
        nb[u].add(w)
        nb[w].add(u)
    others = ([(u, w) for u in ids for w in ids if w > u and w not in nb[u]]
              if nonedge else [])
    for _ in range(iters):
        for (u, w) in edges:
            d = tuple(p[w][k] - p[u][k] for k in range(3))
            L = _norm(d) or 1e-9
            c = 0.5 * (L - (r[u] + r[w])) / L
            p[u] = tuple(p[u][k] + c * d[k] for k in range(3))
            p[w] = tuple(p[w][k] - c * d[k] for k in range(3))
        for (u, w) in others:
            d = tuple(p[w][k] - p[u][k] for k in range(3))
            L = _norm(d) or 1e-9
            tgt = r[u] + r[w]
            if L < tgt:
                c = 0.5 * w_rep * (tgt - L) / L
                p[u] = tuple(p[u][k] - c * d[k] for k in range(3))
                p[w] = tuple(p[w][k] + c * d[k] for k in range(3))
        if center:
            cen = tuple(sum(p[v][k] for v in ids) / len(ids) for k in range(3))
            p = {v: tuple(p[v][k] - cen[k] for k in range(3)) for v in ids}
    return p


def distortion(net, pos, tree=None, kappa=L1.KAPPA):
    """失真读数：边残差、holonomy、非边重叠对数（**读数，不是要消掉的误差**）。"""
    ids = net.ids()
    r = _radius(net, kappa)
    edges = L1.edge_list(net)
    nb = {v: set() for v in ids}
    for (u, w) in edges:
        nb[u].add(w)
        nb[w].add(u)
    res = [abs(math.dist(pos[u], pos[w]) - (r[u] + r[w])) for (u, w) in edges]
    tree = tree or aligned_tree(net, kappa=kappa)
    hol = list(tree["holonomy"].values())
    ov = 0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            u, w = ids[a], ids[b]
            if w in nb[u]:
                continue
            if math.dist(pos[u], pos[w]) < r[u] + r[w]:
                ov += 1
    tree_res = [abs(math.dist(pos[u], pos[w]) - (r[u] + r[w]))
                for (u, w) in tree["tree_edges"]]
    def mx(xs):
        return max(xs) if xs else 0.0
    def mn(xs):
        return sum(xs) / len(xs) if xs else 0.0
    return {"edge_res_mean": mn(res), "edge_res_max": mx(res),
            "tree_res_max": mx(tree_res), "holonomy_mean": mn(hol),
            "holonomy_max": mx(hol), "n_holonomy": len(hol),
            "nonedge_overlaps": ov}


# ============================================================
# §6.4 ② 全局 σ 场 / 锚点（引力代理）
# ============================================================
def sigma_field(net):
    """全局 σ 读数：`σ_global = V/E`、局部 `σ_v`、直方图、`∇σ_v`。"""
    ids = net.ids()
    E = net.E()
    sg = {v: L1.sigma_of(net.deg(v)) for v in ids}
    vals = sorted(sg.values())
    n = len(vals)
    return {"V": len(ids), "E": E, "sigma_global": (len(ids) / E) if E else float("inf"),
            "sigma_min": vals[0] if vals else 0.0,
            "sigma_median": vals[n // 2] if vals else 0.0,
            "sigma_max": vals[-1] if vals else 0.0,
            "sigma": sg, "grad_sigma": L1.grad_sigma(net)}


def anchors(net):
    """**引力代理 = σ 局部极值点**（§7.4 / N015 的「锚点」）。

      sinks[v]：σ_v 高于邻居均值（`∇σ_v < 0`）——σ 汇聚处 = 关系密集的内向拉点；
      sources[v]：反向。
    等度图（如 icosa t=0）上 `∇σ ≡ 0` ⇒ 两者皆空（无锚点）。
    """
    gs = L1.grad_sigma(net)
    sinks = sorted(v for v, g in gs.items() if g < -0.0)
    sources = sorted(v for v, g in gs.items() if g > 0.0)
    return {"sinks": sinks, "sources": sources}


# ============================================================
# §6.4 ③ 红移（代理读数）
# ============================================================
def redshift_proxy(net, root=None):
    """沿 BFS 最短路的 σ 相对变化累积（**代理读数，非物理预言**）。

    `z_v = Σ_{沿 v→root 的路径} (σ_cur − σ_next)/σ_next`。
    σ 高（关系密、温度低）区 => z 为正；等度图上 `σ ≡ const` ⇒ `z ≡ 0`。
    """
    ids = net.ids()
    if not ids:
        return {}
    sg = {v: L1.sigma_of(net.deg(v)) for v in ids}
    root = ids[0] if root is None else root
    z = {root: 0.0}
    q = deque([root])
    while q:
        i = q.popleft()
        for w in net.rot[i]:
            if w in z:
                continue
            z[w] = z[i] + (sg[w] - sg[i]) / sg[i]
            q.append(w)
    return z


# ============================================================
# §6.4 ④ 可视化导出（离线查看）
# ============================================================
def export_obj(net, pos, path, kappa=L1.KAPPA):
    """导出 Wavefront OBJ：`v` = 节点位置（工件），`l` = 边。"""
    ids = net.ids()
    idx = {v: i + 1 for i, v in enumerate(ids)}
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# OpenSPUM L2 全局投影工件（{len(ids)} 节点 {net.E()} 边）\n")
        f.write("# 注意：坐标是投影工件，不是「真实几何」\n")
        for v in ids:
            p = pos[v]
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for (u, w) in L1.edge_list(net):
            f.write(f"l {idx[u]} {idx[w]}\n")
    return path


def export_json(net, pos=None, path=None, kappa=L1.KAPPA):
    """导出 JSON（节点属性 + 边 + 全局读数）；返回该 dict。"""
    l1 = L1.project(net, kappa)
    ids = net.ids()
    gs = L1.grad_sigma(net)
    z = redshift_proxy(net)
    obj = {
        "note": "L2 全局投影：坐标为工件（并非真实几何）；数值全部由 id + 环序读出",
        "V": l1["V"], "E": l1["E"], "sigma_global": l1["sigma_global"],
        "nodes": [{"id": v, "deg": l1["nodes"][v]["deg"], "r": l1["nodes"][v]["r"],
                   "sigma": l1["nodes"][v]["sigma"], "grad_sigma": gs.get(v),
                   "defect": l1["nodes"][v]["defect"],
                   "redshift": z.get(v),
                   "pos": list(pos[v]) if pos and v in pos else None}
                  for v in ids],
        "edges": [list(e) for e in l1["edges"]],
        "anchors": anchors(net),
        "viol": l1["viol"], "nh": l1["nh"], "def_sum": l1["def_sum"],
    }
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
    return obj


# ============================================================
# 一次性全局投影
# ============================================================
def global_projection(net, iters=60, kappa=L1.KAPPA, mode="ring"):
    """从 id + 环序一次性产出 §6.4 全部读数（只读；**不写 L0**）。

    返回 `distortion`（精修后坐标的残留）与 `distortion_raw`（对齐传播的原始读数：
    树边恒精确 `~1e-16`，边残差/ holonomy 是未精修的失真）。
    """
    tree = aligned_tree(net, kappa=kappa, mode=mode)
    raw = tree["pos"]
    pos = relax(net, raw, iters=iters, kappa=kappa) if iters else raw
    return {"pos": pos, "raw_pos": raw, "tree_edges": tree["tree_edges"],
            "holonomy": tree["holonomy"],
            "distortion": distortion(net, pos, tree, kappa),
            "distortion_raw": distortion(net, raw, tree, kappa),
            "sigma": sigma_field(net), "anchors": anchors(net),
            "redshift": redshift_proxy(net)}


# ============================================================
# 自检
# ============================================================
def selftest():
    def close(a, b, tol=1e-9):
        return abs(a - b) <= tol

    # ---- 旋转工具 ----
    u = (1.0, 0.0, 0.0)
    assert all(close(x, y, 1e-12) for x, y in zip(mat_apply(rot_from_uv(u, u), u), u))
    v = (0.0, 1.0, 0.0)
    assert all(close(x, y, 1e-12) for x, y in zip(mat_apply(rot_from_uv(u, v), u), v))
    w = (0.0, 0.0, 1.0)
    assert all(close(x, y, 1e-12) for x, y in zip(mat_apply(rot_from_uv(u, w), u), w))
    nv = (-1.0, 0.0, 0.0)                       # 反向：180°，仍须把 u 送成 nv
    assert all(close(x, y, 1e-12) for x, y in zip(mat_apply(rot_from_uv(u, nv), u), nv))

    # ---- icosa t=0：等度对称图，解析锚点 ----
    ic = RotNet(SEEDS["icosa"]())
    r5 = L1.radius_of(5)
    g = global_projection(ic, iters=0)          # 只看对齐传播
    pos = g["pos"]
    assert len(pos) == 12 and all(all(map(math.isfinite, p)) for p in pos.values())
    d0 = g["distortion_raw"]
    # ① 树边精确：构造保证（t=0 目标是 2r(5)，故 11 条树边全部 = 2r(5)）
    assert d0["tree_res_max"] < 1e-12
    sp = sorted(round(math.dist(pos[a], pos[b]), 9)
                for (a, b) in g["tree_edges"])
    assert len(sp) == 11
    assert close(sp[0], 2 * r5, 1e-9) and close(sp[-1], 2 * r5, 1e-9)
    # ② holonomy 非零 = 曲率不可容纳（读数，不是误差）
    assert d0["n_holonomy"] == 30 - 11 and close(d0["holonomy_mean"], 11.9701, 1e-3)
    # ③ 等度 ⇒ σ 均匀 ⇒ 无锚点、红移 ≡ 0
    sf = g["sigma"]
    assert close(sf["sigma_global"], 0.4, 1e-12) and close(sf["sigma_min"], 0.4, 1e-12)
    assert g["anchors"] == {"sinks": [], "sources": []}
    assert all(close(z, 0.0, 1e-12) for z in g["redshift"].values())

    # ---- 精修后：残差显著下降并**收敛到一个非零平台**（= 不可精确嵌入的必然残留）----
    g2 = global_projection(ic, iters=60)
    d2 = g2["distortion"]
    assert d2["edge_res_mean"] < 0.14 and d2["edge_res_max"] < 0.40
    assert all(all(map(math.isfinite, p)) for p in g2["pos"].values())
    g2b = global_projection(ic, iters=150)
    d2b = g2b["distortion"]
    # 收敛判据取**残差读数**（位置存在缓慢整体旋转漂移——工件层无害）
    assert close(d2["edge_res_mean"], d2b["edge_res_mean"], 1e-6)
    assert d2b["edge_res_max"] < 0.40

    # ---- 确定性：同输入两次结果逐点相同 ----
    g3 = global_projection(ic, iters=30)
    for v in ic.ids():
        assert g3["pos"][v] == relax(ic, aligned_tree(ic)["pos"], iters=30)[v]

    # ---- icosa t=1：异质出现 ⇒ 锚点非空、树边仍精确 ----
    import l0_core as m
    c = m.L0Core(m.RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                 vminus="dense", vplus="any")
    c.frame()
    g4 = global_projection(c.net, iters=30)
    d4 = g4["distortion"]
    assert g4["distortion_raw"]["tree_res_max"] < 1e-12   # 树边仍精确（构造）
    assert d4["n_holonomy"] == c.net.E() - (c.net.V() - 1)
    assert g4["anchors"]["sinks"] or g4["anchors"]["sources"]
    assert len(L1.angle_defect(c.net)["viol"]) == 12  # 与 L1/N4 判决 1 一致

    # ---- 导出（写临时目录后清理）----
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p_obj = export_obj(ic, g2["pos"], os.path.join(td, "icosa.obj"))
        p_js = os.path.join(td, "icosa.json")
        export_json(ic, g2["pos"], p_js)
        with open(p_obj, encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        assert sum(1 for s in lines if s.startswith("v ")) == 12
        assert sum(1 for s in lines if s.startswith("l ")) == 30
        with open(p_js, encoding="utf-8") as f:
            js = json.load(f)
        assert js["V"] == 12 and len(js["nodes"]) == 12 and len(js["edges"]) == 30
        assert js["nodes"][0]["pos"] is not None and js["nodes"][0]["r"] > 0

    # ---- §6.4 ⑤ 径向幂次谱：夹具 A–F ----
    verify_fixtures()
    # ---- §6.4 ⑥ 有限窗口算子：夹具 A–G ----
    verify_window_fixtures()

    print("l2_projection selftest: 全部通过")
    print(f"  icosa t=0 对齐传播：11 条树边全部精确 = 2r(5)={2*r5:.6f}；"
          f"holonomy {d0['n_holonomy']} 条 均值={d0['holonomy_mean']:.4f}（= 曲率读数）")
    print(f"  精修 60 轮：边残差 均值={d2['edge_res_mean']:.4f} 最大={d2['edge_res_max']:.4f}"
          f"（工件口径：不追求精确）")
    print(f"  等度图：σ≡0.4、无锚点、红移≡0；t=1：锚点 {len(g4['anchors']['sinks'])}"
          f"/{len(g4['anchors']['sources'])} 汇/源、树边仍精确")


# ============================================================
# §6.4 ⑤ 径向幂次谱（引力幂次检验）—— 阶段六第③项
# ============================================================
# 权威口径（`物理学/基本相互作用/引力.md` 引理 3 + 推论 3a）：
#   局域密度   σ(r) = σ₀ + σ₁·Ṁ/r²
#   跳数积分   ΔN(r) = ∫_r^∞ [σ(r') − σ₀] dr' ∝ Ṁ/r
#   可观测量   a = c²·∇(ΔN) ∝ r⁻²     ← 牛顿（平方反比）
#   ❌ 陷阱路径（已废弃，仅作对照读数）：a = c²·∇σ ∝ r⁻³
# 四路读数（若 σ ∝ r^p，则 ΔN ∝ r^(p+1)、∇σ ∝ r^(p−1)、∇ΔN ∝ r^p）：
#   slope_σ = −2（引理 3）、slope_ΔN = −1（推论 3a）、
#   slope_∇σ = −3（陷阱对照）、slope_∇ΔN = −2（`a` 的幂次 ⇒ newton）
# ⚠️ 分箱 / 积分 / 拟合都是**工件层读数**：坐标是投影工件（§6.4 ①），σ_v = 2/deg_v
#    是纯组合量。本谱只把「代理读数」升级为**可证伪读数**，不主张测量真实几何。
# ⚠️ σ 的幂次口径**两说并列**（登记为文档冲突，见 §7 阶段六第③项）：
#    `引力.md` 引理 3 给 σ ∝ r⁻²；`spum-evolution.md` §4.2 的 `a = −2kc²m/r³` 隐含
#    σ 被直接当势用（陷阱路径）。本实现**不裁决**，四路读数并列给出。


def power_fit(r, y, min_pts=3):
    """log-log 最小二乘：{"slope","intercept","npts"}；不足 / 退化 → slope=None。

    只取 `r_i > 0 且 y_i > 0` 的点。⚠️ 导数路（∇σ/∇ΔN）恒为负，调用方须先取幅值。
    """
    # --- S1 收集有效点 ---
    pts = [(math.log(r[i]), math.log(y[i]))
           for i in range(len(r)) if r[i] > 0 and y[i] > 0]
    # --- S2 有效点不足 ---
    if len(pts) < min_pts:
        return {"slope": None, "intercept": None, "npts": len(pts)}
    # --- S3 均值与协方差 ---
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(v for _, v in pts) / len(pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (v - my) for x, v in pts)
    # --- S4 退化（r 全相等）---
    if sxx == 0.0:
        return {"slope": None, "intercept": None, "npts": len(pts)}
    # --- S5 斜率 / 截距 ---
    slope = sxy / sxx
    return {"slope": slope, "intercept": my - slope * mx, "npts": len(pts)}


def radial_derivative(r, y):
    """中心差分（端点单侧）→ {"r": [...], "d": [...]}；`m < 2` → 空。"""
    m = len(r)
    if m < 2:
        return {"r": [], "d": []}
    d = [0.0] * m
    for i in range(m):
        if i == 0:
            den = r[1] - r[0]
            d[i] = (y[1] - y[0]) / den if den else 0.0
        elif i == m - 1:
            den = r[m - 1] - r[m - 2]
            d[i] = (y[m - 1] - y[m - 2]) / den if den else 0.0
        else:
            den = r[i + 1] - r[i - 1]
            d[i] = (y[i + 1] - y[i - 1]) / den if den else 0.0
    return {"r": list(r), "d": d}


def hop_integral(r, sigma, sigma0):
    """跳数积分离散化 `ΔN(r_k) = Σ_{j>k}(σ_j − σ₀)·w_j` → {"r","dN"}。

    箱宽取**右端点**约定：`w_j = r_j − r_{j−1}`（`j ≥ 1`）、`w_0 = r_1 − r_0`；
    `m == 1` 退化时 `w_0 = 1.0`、`dN = [0.0]`（最外箱之外无内容）。
    """
    m = len(r)
    if m == 0:
        return {"r": [], "dN": []}
    # --- S1 箱宽（m==1 退化取 1.0）---
    w = [1.0] * m
    for j in range(1, m):
        w[j] = r[j] - r[j - 1]
    if m >= 2:
        w[0] = r[1] - r[0]
    # --- S2 尾部求和 ---
    dN = [0.0] * m
    for k in range(m):
        s = 0.0
        for j in range(k + 1, m):
            s += (sigma[j] - sigma0) * w[j]
        dN[k] = s
    return {"r": list(r), "dN": dN}


def radial_bins(net, pos, center=None, nbins=8, kappa=L1.KAPPA):
    """径向上按「到原点的距离」分箱、取箱内均值 → {"center","center_id","r","n","sigma","deg"}。

    `center=None` → 质心；否则取该**节点 id** 的 `pos` 作原点。空箱剔除；
    `r_max − r_min < 1e-12`（等度球壳，如 icosa t=0）→ **单箱退化**。
    """
    ids = net.ids()
    if not ids:
        return {"center": None, "center_id": None, "r": [], "n": [], "sigma": [], "deg": []}
    # --- S1 原点与距离 ---
    if center is None:
        c = tuple(sum(pos[v][k] for v in ids) / len(ids) for k in range(3))
    else:
        c = pos[center]
    dist = {v: math.dist(pos[v], c) for v in ids}
    sg = {v: L1.sigma_of(net.deg(v)) for v in ids}
    # --- S2 单箱退化 ---
    rmin = min(dist.values())
    rmax = max(dist.values())
    if rmax - rmin < 1e-12:
        return {"center": c, "center_id": center, "r": [rmin], "n": [len(ids)],
                "sigma": [sum(sg[v] for v in ids) / len(ids)],
                "deg": [sum(net.deg(v) for v in ids) / len(ids)]}
    # --- S3 等宽分箱（最后一箱闭区间，防浮点漏项）---
    h = (rmax - rmin) / nbins
    buckets = [[] for _ in range(nbins)]
    for v in ids:
        i = min(nbins - 1, int((dist[v] - rmin) / h))
        buckets[i].append(v)
    # --- S4 箱内均值（只留非空箱；`r` 取箱内实际距离均值，不用箱中点）---
    out = {"center": c, "center_id": center, "r": [], "n": [], "sigma": [], "deg": []}
    for b in buckets:
        if not b:
            continue
        out["r"].append(sum(dist[v] for v in b) / len(b))
        out["n"].append(len(b))
        out["sigma"].append(sum(sg[v] for v in b) / len(b))
        out["deg"].append(sum(net.deg(v) for v in b) / len(b))
    return out


def gravity_spectrum(net, pos=None, center=None, nbins=8, iters=60, kappa=L1.KAPPA):
    """径向幂次谱顶层入口：四路拟合 + 判决（`a` 的幂次靠近 −2 还是 −3）。

    ⚠️ **退化守卫**：若 σ 剖面的极差低于噪声地板（等度图，如 icosa t=0），
    则四路一律判为**无信号**（`slope=None`）——否则 1e-16 级浮点噪声会被非均匀
    箱距放大成**伪幂次**（实测 ∇σ 路曾给出伪斜率 −0.75）。这是零假设对照的硬要求。
    """
    # --- S1 坐标（缺省即取全局投影工件）---
    if pos is None:
        pos = global_projection(net, iters=iters, kappa=kappa)["pos"]
    # --- S2 径向分箱 ---
    bins = radial_bins(net, pos, center, nbins, kappa)
    # --- S3 背景密度 σ₀ = V/E ---
    sigma0 = sigma_field(net)["sigma_global"]
    # --- S4 退化守卫（σ 剖面无结构 ⇒ 无信号）---
    prof = bins["sigma"]
    flat = (not prof) or (max(prof) - min(prof) < 1e-9 * max(1.0, abs(sigma0)))
    if flat:
        null = {"slope": None, "intercept": None, "npts": 0}
        return {"center": bins["center"], "center_id": bins["center_id"], "bins": bins,
                "sigma0": sigma0, "degenerate": True, "dN": None,
                "grad_sigma": None, "grad_dN": None,
                "fits": {"sigma": dict(null), "dN": dict(null),
                         "grad_sigma": dict(null), "grad_dN": dict(null)},
                "verdict": {"a_exponent": None, "label": "undefined",
                            "newton_error": None, "trap_error": None}}
    # --- S5 跳数积分 ΔN = ∫_r^∞ (σ − σ₀) dr ---
    dN = hop_integral(bins["r"], bins["sigma"], sigma0)
    # --- S6 两路径向导数 ---
    grad_sigma = radial_derivative(bins["r"], bins["sigma"])
    grad_dN = radial_derivative(dN["r"], dN["dN"])
    # --- S7 四路拟合（导数路取幅值——幂次只读形状）---
    fits = {"sigma": power_fit(bins["r"], bins["sigma"]),
            "dN": power_fit(dN["r"], dN["dN"]),
            "grad_sigma": power_fit(grad_sigma["r"], [abs(v) for v in grad_sigma["d"]]),
            "grad_dN": power_fit(grad_dN["r"], [abs(v) for v in grad_dN["d"]])}
    # --- S8 判决（**容差带**：只有真的靠近 −2 / −3 才给判词）---
    a_exp = fits["grad_dN"]["slope"]
    if a_exp is None:
        label, newton_error, trap_error = "undefined", None, None
    else:
        newton_error, trap_error = abs(a_exp + 2), abs(a_exp + 3)
        if newton_error <= 0.35:
            label = "newton"
        elif trap_error <= 0.35:
            label = "trap"
        else:
            label = "neither"        # 幂次既不近 −2 也不近 −3 ⇒ 本轨迹无该信号
    return {"center": bins["center"], "center_id": bins["center_id"], "bins": bins,
            "sigma0": sigma0, "degenerate": False, "dN": dN,
            "grad_sigma": grad_sigma, "grad_dN": grad_dN,
            "fits": fits,
            "verdict": {"a_exponent": a_exp, "label": label,
                        "newton_error": newton_error, "trap_error": trap_error}}


def null_control(kappa=L1.KAPPA):
    """**零假设对照**：等度图 icosa t=0（σ≡0.4、无锚点、红移≡0 ⇒ 全路退化）。

    返回 {"degenerate","sigma_spread","n_anchors","z_max","spectrum"}。
    """
    net = RotNet(SEEDS["icosa"]())
    pos = global_projection(net, iters=0, kappa=kappa)["pos"]
    sp = gravity_spectrum(net, pos=pos, nbins=8, iters=0, kappa=kappa)
    sf = sigma_field(net)
    anch = anchors(net)                     # ⚠️ 返回 dict（非二元组）
    z = redshift_proxy(net)
    spread = sf["sigma_max"] - sf["sigma_min"]
    return {"degenerate": spread < 1e-12, "sigma_spread": spread,
            "n_anchors": len(anch["sinks"]) + len(anch["sources"]),
            "z_max": max((abs(v) for v in z.values()), default=0.0), "spectrum": sp}


def synthetic_calibration(rs=None, fit_slice=(2, 6)):
    """**合成标定**：注入 σ∝r⁻² / σ∝r⁻³，证明四路拟合能分辨幂次。

    拟合只用 `rs[fit_slice]` 子区间以压低右端点离散化的边缘效应。
    返回 {"r","fit_slice","cases":[{"p","expect","got"}]}。
    """
    if rs is None:
        rs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    a, b = fit_slice
    cases = []
    for p in (-2, -3):
        sigma = [r ** float(p) for r in rs]
        dN = hop_integral(rs, sigma, 0.0)
        gs = radial_derivative(rs, sigma)
        gd = radial_derivative(dN["r"], dN["dN"])
        got = {"sigma": power_fit(rs[a:b], sigma[a:b])["slope"],
               "dN": power_fit(rs[a:b], dN["dN"][a:b])["slope"],
               "grad_sigma": power_fit(rs[a:b], [abs(v) for v in gs["d"]][a:b])["slope"],
               "grad_dN": power_fit(rs[a:b], [abs(v) for v in gd["d"]][a:b])["slope"]}
        cases.append({"p": p,
                      "expect": {"sigma": float(p), "dN": float(p + 1),
                                 "grad_sigma": float(p - 1), "grad_dN": float(p)},
                      "got": got})
    return {"r": list(rs), "fit_slice": [a, b], "cases": cases}


# ============================================================
# §6.4 ⑥ 有限窗口算子（阶段六第④项：兑现 N1-C）
# ============================================================
# N1-C（§7 分岔 N1）：V 无界是纯组合 L0 的结构性质；「观测到的有限宇宙」由
# 投影层选取窗口。本小节把该判词兑现为**显式窗口算子**，并交付「窗口读数对
# 窗口大小稳定」的可证伪判据。
#   · 窗口构造：`window_ball`（跳数球窗）/ `window_prefix`（V ≤ V_window 规模窗）
#   · 拟合窗口：`radial_window`（按箱内样本数自动选取 —— 裁决第③项遗留 (i)：
#               `fit_slice` 只留给 `synthetic_calibration` 的合成标定口径，
#               采纳轨迹上的拟合窗口改用「`r>0` 且箱内样本数 ≥ `min_bin_pts`」自动口径）
#   · 窗口读数：`window_readouts`（σ_window / ⟨deg⟩ / 边界占比 / 分量数 / 锚点密度 / 四路幂次谱）
#   · 稳定性  ：`window_stability`（逐级 gap + 判词）
# ⚠️ 窗口是**只读读数工具**：只按节点集合构造新 `RotNet`，不 `prune`、不写回 `net`、
#    不参与 L0 规则（L2 永不回写 L0）。
# ⚠️ **边界效应是窗口读数的固有成分**：子图边界节点的度数必然低于母图（邻居被截），
#    故 `σ_window = V/E` 系统性偏高、`r_v` 与方向场在边界退化。这不是 bug，是读数本身；
#    判据只看「随窗口增大是否收敛」。

def _subnet(net, nodes):
    """按节点集合取子图：`rot` **保序截断**（不重排、不 prune）→ 新 `RotNet`。

    保序是关键：L1/L2 的方向场与平行移动完全由 `rot` 的**环序**决定，
    截断保持相对顺序 ⇒ 子图仍是合法旋转系统（只是边界节点度数下降）。
    """
    S = set(nodes)
    return RotNet({v: [w for w in net.rot[v] if w in S] for v in nodes})


def _segments(g):
    """含 None 的序列 → 相邻差的绝对值列表（任一为 None 则该项为 None）。"""
    out = []
    for i in range(len(g) - 1):
        a, b = g[i], g[i + 1]
        out.append(None if (a is None or b is None) else abs(b - a))
    return out


def window_ball(net, center=None, radius=2):
    """**跳数球窗**：以 `center` 为心、在 net 中拓扑距离 ≤ `radius` 的节点集 → 子图。

    `center=None` → 取 `min(net.ids())`（确定性）。球窗天然连通（BFS 只走连通部分），
    故子图每个节点 `deg ≥ 1`（`radius=0` 时为单点窗）。
    """
    # --- S1 空图 ---
    ids = net.ids()
    if not ids:
        return RotNet({})
    # --- S2 原点（确定性：缺省取最小 id）---
    c = min(ids) if center is None else center
    if c not in net.rot:
        raise ValueError("window_ball: center 不在图中")
    # --- S3 BFS 至 radius（按 rot 的给定顺序扩展 ⇒ 确定性）---
    dist = {c: 0}
    q = deque([c])
    while q:
        u = q.popleft()
        if dist[u] >= int(radius):
            continue
        for w in net.rot[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    # --- S4 子图（升序 id，保序截断）---
    return _subnet(net, sorted(dist))


def window_prefix(net, v_window, center=None):
    """**规模窗**（`V ≤ V_window` 子图）：按 BFS 序取前 `v_window` 个节点。

    BFS 序由 `rot` 的给定顺序决定 ⇒ 同输入同窗口（确定性）。`v_window ≤ 0` → 空图。
    """
    # --- S1 边界：空图 / 非正规模 ---
    ids = net.ids()
    if not ids or int(v_window) <= 0:
        return RotNet({})
    # --- S2 原点 ---
    c = min(ids) if center is None else center
    if c not in net.rot:
        raise ValueError("window_prefix: center 不在图中")
    # --- S3 完整 BFS 序（出队即定序；入队即标记，不重复入队）---
    order, seen, q = [], {c}, deque([c])
    while q:
        u = q.popleft()
        order.append(u)
        for w in net.rot[u]:
            if w not in seen:
                seen.add(w)
                q.append(w)
    # --- S4 前缀截取 + 子图 ---
    return _subnet(net, sorted(order[:int(v_window)]))


def radial_window(bins, min_bin_pts=3):
    """**径向拟合窗口**：`radial_bins` 的箱序列 → 可拟合子序列（按箱内样本数自动选）。

    保留条件：`r_k > 0` **且** `n_k ≥ min_bin_pts`。返回 `{"r","sigma","deg","kept","npts"}`，
    其中 `kept` 是被保留箱的**位置下标**。`m == 0` → 全空（`npts=0`）。
    """
    # --- S1 空箱序列 ---
    m = len(bins["r"])
    if m == 0:
        return {"r": [], "sigma": [], "deg": [], "kept": [], "npts": 0}
    # --- S2 保留判据（r>0 ∧ 样本数达标）---
    kept = [k for k in range(m)
            if bins["r"][k] > 0 and bins["n"][k] >= int(min_bin_pts)]
    # --- S3 同步抽取三列 ---
    return {"r": [bins["r"][k] for k in kept],
            "sigma": [bins["sigma"][k] for k in kept],
            "deg": [bins["deg"][k] for k in kept],
            "kept": kept, "npts": len(kept)}


def _n_components(sub):
    """子图连通分量数（纯 BFS，确定性）。"""
    seen, n = set(), 0
    for v in sub.ids():
        if v in seen:
            continue
        n += 1
        q = deque([v])
        seen.add(v)
        while q:
            u = q.popleft()
            for w in sub.rot[u]:
                if w not in seen:
                    seen.add(w)
                    q.append(w)
    return n


def window_readouts(net, sub, nbins=8, iters=60, kappa=L1.KAPPA):
    """**窗口读数**：对子图 `sub`（母图 `net`）求一组边界敏感的内禀读数。

    返回固定键：`v_window` / `e_window` / `sigma_window`(=V/E) / `deg_mean`(=2E/V) /
    `n_boundary`（子图度数 < 母图度数的节点数）/ `frac_boundary` / `n_comp` /
    `anchor_density` / `spectrum`（四路幂次谱，可为 None）。
    空窗 → 除计数外全部 None（不抛异常）。
    """
    # --- S1 空窗 ---
    Vw = sub.V()
    if Vw == 0:
        return {"v_window": 0, "e_window": 0, "sigma_window": None, "deg_mean": None,
                "n_boundary": 0, "frac_boundary": None, "n_comp": 0,
                "anchor_density": None, "spectrum": None}
    # --- S2 规模读数 ---
    Ew = sub.E()
    sigma_window = (Vw / Ew) if Ew else None
    deg_mean = 2.0 * Ew / Vw
    # --- S3 边界占比（边界效应 = 窗口偏离母图的直接度量）---
    n_boundary = sum(1 for v in sub.ids() if net.deg(v) > sub.deg(v))
    # --- S4 分量数 ---
    n_comp = _n_components(sub)
    # --- S5 锚点密度（引力代理；等度图恒 0）---
    anc = anchors(sub)                      # ⚠️ 返回 dict
    anchor_density = (len(anc["sinks"]) + len(anc["sources"])) / Vw
    # --- S6 四路幂次谱（内部自算工件坐标）---
    spectrum = gravity_spectrum(sub, nbins=nbins, iters=iters, kappa=kappa)
    return {"v_window": Vw, "e_window": Ew, "sigma_window": sigma_window,
            "deg_mean": deg_mean, "n_boundary": n_boundary,
            "frac_boundary": n_boundary / Vw, "n_comp": n_comp,
            "anchor_density": anchor_density, "spectrum": spectrum}


def window_stability(net, sizes=(8, 16, 32), nbins=8, iters=60, kappa=L1.KAPPA,
                     min_bin_pts=3, tol_rel=0.05):
    """**窗口稳定性扫描**：逐级读数 + 逐级残差 + 判词（阶段六第④项验证判据）。

    对 `sizes` 中每个规模取 `window_prefix` → `window_readouts`，给出逐级残差。
    **判词只看 `σ_window = V/E` 的末级*相对*残差**（尺度无关，不靠绝对值拍容差）：
      `rel_last = |σ_last − σ_prev| / σ_last`
      · `rel_last ≤ tol_rel` → `stable`；否则 `unstable`；序列含 None/0 → `undefined`。
    **旁证（如实报，不参与判词；实测依据见各条）**：
      · `slope_σ` —— 窗口上分箱样本稀少 ⇒ **多级为 None**（实测 t=0 全窗、t=1 的 V=8/32 皆 None），
        只在可算时作旁证，**不可作判据**；
      · `anchor_density` —— 被**窗口边界效应主导**：边界节点度数下降 ⇒ `σ_v = 2/deg` 出现梯度 ⇒
        等度子图也长出锚点（实测球窗 6 节点 `anchor_density = 1.0`，而全窗 = 0），**不可作判据**；
      · `frac_boundary` —— 边界占比，解释 `σ_window` 偏离母图的来源（实测随窗口单调下降）。
    ⚠️ 本函数只**如实报读数与判词**，不主张窗口一定稳定。
    **实测边界（seed=icosa `dense/any`，sizes=8,16,24,32）**：
      · t=0（V=12）：`V_window ≥ 12` 截断到全图 ⇒ 去重后剩 2 级，`rel_last = 0.3333` ⇒ `unstable`；
      · t=1（V=32）`rel_last = 0.0547`、t=2（V=36）`rel_last = 0.1346`（σ 序列非单调）⇒ `unstable`；
      · t=3（V=66）`rel_last = 0.0042` ⇒ `stable`（窗口 24/32 的 σ 已收敛到 0.405）
      ⇒ **稳定性是有条件的**：`V_window ≪ V` 时成立；V 与窗口同量级时边界效应主导。
    """
    # --- S1 输入边界 ---
    sizes = sorted({int(s) for s in sizes})
    if not sizes or net.V() == 0:
        return {"rows": [], "gaps": {k: [] for k in
                                     ("sigma", "frac_boundary", "anchor_density", "slope_sigma")},
                "gap_last": {}, "rel_last": None, "trend": "undefined",
                "n_truncated": 0,
                "verdict": "undefined", "reason": "sizes 为空或图为空"}
    # --- S2 逐级读数（谱路用自动拟合窗口 radial_window；**按实际窗口规模去重**）---
    # ⚠️ `V_window ≥ V` 时 `window_prefix` 截断到全图 ⇒ 多级读数完全相同，会造出
    #    伪 0 残差（实测 seed=icosa t=0、sizes=(8,16,24,32) 曾因此误判 stable）。
    #    故按**实际** `v_window` 去重，重复级不计入。
    rows = []
    for s in sizes:
        sub = window_prefix(net, s)
        if rows and sub.V() == rows[-1]["v_window"]:
            continue
        r = window_readouts(net, sub, nbins=nbins, iters=iters, kappa=kappa)
        spec = r["spectrum"]
        rw = radial_window(spec["bins"], min_bin_pts) if spec else {"r": [], "sigma": [], "npts": 0}
        ft = power_fit(rw["r"], rw["sigma"]) if rw["npts"] > 0 else {"slope": None}
        rows.append({"size": s, "v_window": r["v_window"], "e_window": r["e_window"],
                     "sigma_window": r["sigma_window"], "frac_boundary": r["frac_boundary"],
                     "n_comp": r["n_comp"], "anchor_density": r["anchor_density"],
                     "slope_sigma": ft["slope"],
                     "truncated": r["v_window"] >= net.V(),
                     "label": spec["verdict"]["label"] if spec else None})
    # --- S3 逐级残差（含 None 安全）---
    gaps = {"sigma": _segments([r["sigma_window"] for r in rows]),
            "frac_boundary": _segments([r["frac_boundary"] for r in rows]),
            "anchor_density": _segments([r["anchor_density"] for r in rows]),
            "slope_sigma": _segments([r["slope_sigma"] for r in rows])}
    gap_last = {k: (v[-1] if v else None) for k, v in gaps.items()}
    # --- S4 判词（σ_window 的相对末级残差；趋势作旁证）---
    sig = [r["sigma_window"] for r in rows]
    if len(rows) < 2 or any((x is None or x == 0.0) for x in sig):
        verdict, rel_last, trend = "undefined", None, "undefined"
        reason = "σ_window 序列含 None/0、或窗口数不足 2 ⇒ 无法判稳定"
    else:
        rel = [g / sig[i + 1] for i, g in enumerate(gaps["sigma"]) if g is not None]
        rel_last = rel[-1] if rel else None
        trend = ("converging" if all(rel[i] >= rel[i + 1] for i in range(len(rel) - 1))
                 else "non-monotone") if rel_last is not None else "undefined"
        if rel_last is None:
            verdict, reason = "undefined", "σ_window 的逐级残差全为 None"
        elif rel_last <= tol_rel:
            verdict = "stable"
            reason = ("σ_window 末级相对残差 %.4f ≤ tol_rel=%.2f（相对残差趋势 %s）"
                      "⇒ 窗口读数已稳定" % (rel_last, tol_rel, trend))
        else:
            verdict = "unstable"
            reason = ("σ_window 末级相对残差 %.4f > tol_rel=%.2f（相对残差趋势 %s）"
                      "⇒ 边界效应未消，窗口未稳定" % (rel_last, tol_rel, trend))
    # --- S5 汇总 ---
    return {"rows": rows, "gaps": gaps, "gap_last": gap_last,
            "rel_last": rel_last, "trend": trend,
            "n_truncated": sum(1 for r in rows if r["truncated"]),
            "verdict": verdict, "reason": reason}


def _spectrum_demo(seed="icosa", nframes=4, cap=12, dmin=3, nbins=8, iters=60,
                   kappa=L1.KAPPA):
    """CLI：逐帧打印四路幂次斜率与判决（附可分辨性标定 + 零假设对照）。"""
    import l0_core as m
    print("== 可分辨性标定（合成注入，拟合须复原注入幂次）==")
    for case in synthetic_calibration()["cases"]:
        got = " ".join(f"{k}={case['got'][k]:+.3f}(期望{case['expect'][k]:+.0f})"
                       for k in ("sigma", "dN", "grad_sigma", "grad_dN"))
        print(f"  σ∝r^{case['p']}: {got}")
    nc = null_control()
    print(f"== 零假设对照（等度图 icosa t=0）：σ 极差={nc['sigma_spread']:.2e}、"
          f"锚点={nc['n_anchors']}、z_max={nc['z_max']:.2e} ⇒ 全路无信号 ==")
    c = m.L0Core(m.RotNet(SEEDS[seed]()), cap=cap, dmin=dmin,
                 vminus="dense", vplus="any", kappa=kappa)
    print(f"== L2 径向幂次谱（seed={seed} cap={cap} dmin={dmin} dense/any "
          f"nbins={nbins} 松弛 {iters} 轮）==")
    print("坐标为投影工件；slope∇ΔN = `a` 的幂次：−2 ⇒ newton（权威口径），"
          "−3 ⇒ trap（已废弃路径，仅作对照）")
    print(f"{'t':>3} {'V':>5} {'E':>6} {'箱数':>5} {'slopeσ':>9} {'slopeΔN':>9} "
          f"{'slope∇σ':>9} {'slope∇ΔN':>10} {'判决':>10}")
    print("-" * 84)
    for t in range(nframes + 1):
        sp = gravity_spectrum(c.net, nbins=nbins, iters=iters, kappa=kappa)
        cells = []
        for k in ("sigma", "dN", "grad_sigma", "grad_dN"):
            x = sp["fits"][k]["slope"]
            cells.append(f"{x:>9.4f}" if x is not None else f"{'--':>9}")
        print(f"{t:>3} {c.net.V():>5} {c.net.E():>6} {len(sp['bins']['r']):>5} "
              f"{cells[0]} {cells[1]} {cells[2]} {cells[3]} "
              f"{sp['verdict']['label']:>10}")
        if t < nframes:
            c.frame()
    print("（`--` 有两种来源：σ 剖面无结构（如等度图 t=0）、或非空箱数 < 3 不足以拟合；"
          "判决只在斜率真的靠近 −2 / −3（±0.35）时给 newton / trap）")


def verify_fixtures():
    """§4 夹具 A–F：全过返回 True，否则 AssertionError。"""
    # --- 夹具 A：power_fit 精确解 ---
    r = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    y = [3.0, 0.75, 0.1875, 0.046875, 0.01171875, 0.0029296875]
    fa = power_fit(r, y)
    assert fa["npts"] == 6 and abs(fa["slope"] + 2.0) < 1e-9
    assert abs(power_fit(r, [3.0 * x ** -3 for x in r])["slope"] + 3.0) < 1e-9
    # --- 夹具 B：hop_integral 整数解（手算）---
    assert hop_integral([1.0, 2.0, 3.0], [4.0, 1.0, 1.0], 0.0)["dN"] == [2.0, 1.0, 0.0]
    # --- 夹具 C：radial_derivative 手算 ---
    d = radial_derivative([1.0, 2.0, 4.0], [4.0, 1.0, 0.25])["d"]
    assert all(abs(p - q) < 1e-12 for p, q in zip(d, [-3.0, -1.25, -0.375]))
    # --- 夹具 D：零假设对照（等度图退化）---
    nc = null_control()
    assert nc["degenerate"] and nc["sigma_spread"] < 1e-12
    assert nc["n_anchors"] == 0 and nc["z_max"] < 1e-12
    sp0 = nc["spectrum"]
    assert sp0["degenerate"] is True and sum(sp0["bins"]["n"]) == 12
    assert all(sp0["fits"][k]["slope"] is None
               for k in ("sigma", "dN", "grad_sigma", "grad_dN"))
    assert sp0["verdict"]["label"] == "undefined"
    # --- 夹具 E：合成标定可分辨 r⁻²/r⁻³ ---
    for case in synthetic_calibration()["cases"]:
        for k in ("sigma", "dN", "grad_sigma", "grad_dN"):
            assert abs(case["got"][k] - case["expect"][k]) < 0.25, \
                (case["p"], k, case["got"][k])
        # 判词带（±0.35）须能正确分类 ⇒ 可分辨性成立
        assert abs(case["got"]["grad_dN"] + 2) < 0.35 if case["p"] == -2 \
            else abs(case["got"]["grad_dN"] + 3) < 0.35
    # --- 夹具 F：结构断言（icosa t=1）---
    import l0_core as m
    c = m.L0Core(m.RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                 vminus="dense", vplus="any")
    c.frame()
    sp = gravity_spectrum(c.net, nbins=8)
    assert sum(sp["bins"]["n"]) == c.net.V()
    assert all(k in sp["fits"] for k in ("sigma", "dN", "grad_sigma", "grad_dN"))
    assert sp["verdict"]["label"] in {"newton", "trap", "neither", "undefined"}
    return True


# ============================================================
# §6.4 ⑥ 夹具与 CLI（有限窗口算子）
# ============================================================
def verify_window_fixtures():
    """§6.4 ⑥ 夹具 A–G（窗口算子）：全过返回 True，否则 AssertionError。

    期望值**手工可算或已实测标定**：icosa t=0 = 12 节点 5-正则（V=12 / E=30 / σ=0.4）；
    球窗 r=1 = 中心 + 5 个一阶邻居，邻居在子图内度数降为 3（连中心 1 + 5-环上 2）
    ⇒ 5 个边界节点、E = 5（中心边）+ 5（5-环）= 10 ⇒ σ_window = 0.6。
    """
    ic = RotNet(SEEDS["icosa"]())
    # --- 夹具 A：跳数球窗规模（r=0/1/2）---
    assert window_ball(ic, 0, 0).V() == 1 and window_ball(ic, 0, 0).E() == 0
    b1 = window_ball(ic, 0, 1)
    assert b1.V() == 6 and b1.E() == 10
    b2 = window_ball(ic, 0, 2)
    assert b2.V() == 11 and b2.E() == 25
    # --- 夹具 B：规模窗 = BFS 前缀；V=6 时与球窗 r=1 **同集合** ---
    p6 = window_prefix(ic, 6)
    assert p6.V() == 6 and p6.ids() == b1.ids()
    assert window_prefix(ic, 0).V() == 0 and window_prefix(ic, -3).V() == 0
    assert window_prefix(ic, 999).V() == 12          # 截断到全图
    # --- 夹具 C：全窗读数 = 母图本身（边界占比 0；等度 ⇒ 无锚点）---
    rf = window_readouts(ic, window_prefix(ic, 12))
    assert (rf["v_window"], rf["e_window"]) == (12, 30)
    assert abs(rf["sigma_window"] - 0.4) < 1e-12 and abs(rf["deg_mean"] - 5.0) < 1e-12
    assert rf["n_boundary"] == 0 and rf["frac_boundary"] == 0.0
    assert rf["n_comp"] == 1 and rf["anchor_density"] == 0.0
    # --- 夹具 D：球窗 r=1（5 个边界节点；等度子图亦因边界长出锚点 —— 如实读数）---
    rd = window_readouts(ic, b1)
    assert abs(rd["sigma_window"] - 0.6) < 1e-12 and rd["n_boundary"] == 5
    assert abs(rd["frac_boundary"] - 5.0 / 6.0) < 1e-12 and rd["n_comp"] == 1
    assert rd["anchor_density"] == 1.0
    # --- 夹具 E：radial_window 按「r>0 ∧ 箱内样本数」筛箱 ---
    rw = radial_window({"r": [0.0, 1.0, 2.0], "n": [4, 1, 5],
                        "sigma": [0.4, 0.4, 0.4], "deg": [5, 5, 5]}, min_bin_pts=3)
    assert rw["kept"] == [2] and rw["npts"] == 1 and rw["r"] == [2.0]
    assert radial_window({"r": [], "n": [], "sigma": [], "deg": []})["npts"] == 0
    # --- 夹具 F：**判据能拒绝**（反向控制）——t=0 窗口扫描必判 unstable ---
    st0 = window_stability(ic, sizes=(6, 12))
    assert st0["verdict"] == "unstable" and abs(st0["rel_last"] - 0.5) < 1e-12
    # --- 夹具 H：`V_window ≥ V` 的**截断去重**（防伪 0 残差）---
    #     全图仅 12 节点 ⇒ sizes 中 16/24/32 全部截断到 12，只计一次 ⇒ rows 长度 2
    sth = window_stability(ic, sizes=(8, 16, 24, 32))
    assert len(sth["rows"]) == 2 and sth["n_truncated"] == 1
    assert sth["verdict"] == "unstable" and abs(sth["rel_last"] - 1.0 / 3.0) < 1e-12
    # --- 夹具 G：t=1 扫描 —— 相对残差递减（trend=converging）但末级仍超容差 ---
    import l0_core as m
    c = m.L0Core(m.RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                 vminus="dense", vplus="any")
    c.frame()
    st1 = window_stability(c.net, sizes=(8, 16, 32))
    assert st1["verdict"] == "unstable" and st1["trend"] == "converging"
    assert st1["gaps"]["sigma"][0] > st1["gaps"]["sigma"][1]
    return True


def _window_demo(seed="icosa", nframes=2, cap=12, dmin=3, sizes=(8, 16, 32),
                 nbins=8, iters=60, kappa=L1.KAPPA):
    """CLI：逐帧打印窗口稳定性扫描（窗口读数对窗口大小是否稳定？）。"""
    import l0_core as m
    c = m.L0Core(m.RotNet(SEEDS[seed]()), cap=cap, dmin=dmin,
                 vminus="dense", vplus="any", kappa=kappa)
    print(f"== L2 有限窗口算子（§6.4 ⑥；seed={seed} cap={cap} dmin={dmin} dense/any "
          f"窗口 {list(sizes)} nbins={nbins} 松弛 {iters} 轮）==")
    print("窗口 = BFS 前缀子图（V ≤ V_window，只读；不改 L0、不 prune）；"
          "判词只看 σ_window 末级**相对**残差")
    for t in range(nframes + 1):
        st = window_stability(c.net, sizes=sizes, nbins=nbins, iters=iters, kappa=kappa)
        print(f"-- t={t}  V={c.net.V()} E={c.net.E()}  判词={st['verdict']}  "
              f"rel_last={st['rel_last']}  趋势={st['trend']}")
        print(f"   {'V_win':>6} {'E_win':>6} {'σ_win':>8} {'边界占比':>9} "
              f"{'锚点密度':>9} {'slope_σ':>9} {'谱判词':>10}")
        for row in st["rows"]:
            sg = (f"{row['slope_sigma']:>9.3f}" if row["slope_sigma"] is not None
                  else f"{'--':>9}")
            print(f"   {row['v_window']:>6} {row['e_window']:>6} "
                  f"{row['sigma_window']:>8.4f} {row['frac_boundary']:>9.4f} "
                  f"{row['anchor_density']:>9.3f} {sg} {str(row['label']):>10}")
        print(f"   依据：{st['reason']}")
        if t < nframes:
            c.frame()
    print("（旁证不参与判词：slope_σ 在窗口上常因箱内样本稀少而为 `--`；"
          "锚点密度被边界效应主导 —— 等度子图也会长出锚点）")


# ============================================================
# CLI
# ============================================================
def _demo(seed="icosa", nframes=4, cap=12, dmin=3, iters=60, kappa=L1.KAPPA):
    import l0_core as m
    c = m.L0Core(m.RotNet(SEEDS[seed]()), cap=cap, dmin=dmin,
                 vminus="dense", vplus="any", kappa=kappa)
    print(f"== L2 全局投影（seed={seed} cap={cap} dmin={dmin} dense/any κ={kappa} "
          f"松弛 {iters} 轮）==")
    print("坐标为**投影工件**（对齐传播 + 位置约束投影）；holonomy = 曲率读数，非误差")
    print(f"{'t':>3} {'V':>5} {'E':>6} {'σ=V/E':>7} {'σmax':>7} {'锚点±':>9} "
          f"{'z_max':>8} {'树残差':>9} {'边残差':>16} {'holonomy':>14} {'重叠':>5}")
    print("-" * 104)
    for t in range(nframes + 1):
        g = global_projection(c.net, iters=iters, kappa=kappa)
        d, dr, sf = g["distortion"], g["distortion_raw"], g["sigma"]
        zmax = max((abs(z) for z in g["redshift"].values()), default=0.0)
        na = len(g["anchors"]["sinks"]) + len(g["anchors"]["sources"])
        print(f"{t:>3} {sf['V']:>5} {sf['E']:>6} {sf['sigma_global']:>7.4f} "
              f"{sf['sigma_max']:>7.4f} {na:>9} {zmax:>8.4f} "
              f"{dr['tree_res_max']:>9.2e} "
              f"{d['edge_res_mean']:>7.4f}/{d['edge_res_max']:<8.4f} "
              f"{dr['holonomy_mean']:>6.4f}/{dr['holonomy_max']:<7.4f} "
              f"{d['nonedge_overlaps']:>5}")
        if t < nframes:
            c.frame()
    print("（树残差 = 对齐传播(未精修)的生成树边残差，构造上恒 ~1e-16；"
          "边残差/重叠 = 精修后残留；holonomy = 非树边闭合差 = 曲率读数）")


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
    if "spectrum" in kv:
        _spectrum_demo(seed=kv.get("seed", "icosa"), nframes=int(kv.get("nframes", 4)),
                       cap=int(kv.get("cap", 12)), dmin=int(kv.get("dmin", 3)),
                       nbins=int(kv.get("nbins", 8)), iters=int(kv.get("iters", 60)),
                       kappa=int(kv.get("kappa", L1.KAPPA)))
        return
    if "window" in kv:
        raw = kv.get("sizes", "8,16,32")
        _window_demo(seed=kv.get("seed", "icosa"), nframes=int(kv.get("nframes", 2)),
                     cap=int(kv.get("cap", 12)), dmin=int(kv.get("dmin", 3)),
                     sizes=tuple(int(x) for x in raw.split(",") if x.strip()),
                     nbins=int(kv.get("nbins", 8)), iters=int(kv.get("iters", 60)),
                     kappa=int(kv.get("kappa", L1.KAPPA)))
        return
    if "seed" in kv or "nframes" in kv:
        _demo(seed=kv.get("seed", "icosa"), nframes=int(kv.get("nframes", 4)),
              cap=int(kv.get("cap", 12)), dmin=int(kv.get("dmin", 3)),
              iters=int(kv.get("iters", 60)), kappa=int(kv.get("kappa", L1.KAPPA)))
        return
    print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
