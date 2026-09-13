# -*- coding: utf-8 -*-
"""l3_projection.py — L3 投影层：帧序列（跨帧）投影，用 id 对齐 + 环序结构。

架构基准: docs/L0L1L2_架构设计.md §6.5（本轮新立 —— 原文档**无** §6.5/L3 定义，见 §7 分岔 N8）
指标语义: 图论/spum-图论指标.md GT-005(δ) / GT-006(Δμ) / GT-007(d_topo) / GT-008(σ)

一句话
------
**L1/L2 投影「一帧之内」，L3 投影「帧与帧之间」——跨帧的锚是 id，帧内的结构仍由环序给出。**

L3 只读**帧快照序列**，不参与 L0 规则；且**不依赖 L2 坐标**——§6.5 的四项读数全部
坐标无关，只吃 id + 环序 ⇒ 与 L2 工件解耦（L2 的全局坐标可省，L3 照样成立）。§6.5 四项：

  ① 帧间网络骨架 `frame_trace` / `attr_series`
      以 id 对齐相邻帧 → alive / born / dead / persistent；持久 id 的
      `deg / r(deg) / σ_v` 沿时间轴的序列（`r`、`σ` 取自 §6.3 局部量）
  ② 张力场指标层（GT-005~008，坐标无关）
      `delta`(δ_k) / `anchor_drift`(Δμ) / `dtopo`(d_topo, d_norm, d_w) / `sigma_global`(σ)
  ③ 双层闭合判据 `closure_series`：`δ<θ_δ ∧ Δμ<θ_μ`，且需**连续 k 帧**（N013/N016 修正：
      单靠 δ 下降不能判闭合——δ 对锚点漂移完全无感）
  ④ 导出 `export_csv` / `export_json`（逐帧指标序列）

⚠️ 口径（与 GT-005 的差异，本轮实测发现）：注册定义 δ = |{deg<2}|/|V| 在 L0 采纳动力学下
   **恒为 0**（帧末修剪已删掉 deg<dmin 的节点，dmin≥3）⇒ 必须同时给泛化读数
   δ_{dmin}（修剪边界）与 δ_{κ}（平坦阈值 κ=6，正曲率端）。闭合判据默认用 δ_{κ}，
   因为 δ_2 在三角剖分里恒成立、对闭合无信息量。

用法
----
  python l3_projection.py selftest=1
  python l3_projection.py seed=icosa nframes=6 cap=12 dmin=3 vminus=dense vplus=any
  python l3_projection.py seed=icosa nframes=6 cap=12 dmin=3 csv=series.csv json=series.json
"""

import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for _d in ("l0", "l1"):
    _p = os.path.join(_SRC, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from combinatorial_proto import RotNet, SEEDS  # noqa: E402
import l0_core as m                            # noqa: E402
import l1_projection as L1                     # noqa: E402

KAPPA = L1.KAPPA          # 平坦阈值 κ=6（deg<κ ⇒ 正角亏端）
THETA_DELTA = 0.03        # GT-006 建议阈值（从 97.1% 正确率推算）
THETA_MU = 2.0            # GT-006 建议阈值（N016 需大幅收紧）
K_CLOSED = 5              # 连续帧数（短暂波动不误判为闭合）
W_PLUS = 1.0              # d_w 权重：V⁺
W_MINUS = 2.0             # d_w 权重：V⁻（GT-007 默认 1:2）
_TOL = 1e-12


# ============================================================
# 基础：帧快照 / 边集
# ============================================================
def snap(net):
    """帧快照（深拷贝旋转系统，保留 id 计数器）——L3 以 id 对齐，故必须冻结每帧。"""
    c = RotNet({v: list(r) for v, r in net.rot.items()})
    c.nid = net.nid
    return c


def edge_set(net):
    """无向边集合（id 对，规范化 (min,max)）——「谁连着谁」的集合形式。"""
    return {(v, w) if v < w else (w, v)
            for v in net.ids() for w in net.rot[v]}


def trace_frames(core, nframes):
    """跑 L0 取帧序列：返回 (nets, events)。

    nets[0] = 帧初；nets[τ] = 第 τ 帧末快照。events[τ] = 第 τ 次 frame() 的事件计数
    （即把 nets[τ] 变成 nets[τ+1] 的那一帧）。L3 只读，不改 L0。
    """
    nets = [snap(core.net)]
    events = []
    for _ in range(int(nframes)):
        events.append(dict(core.frame()))
        nets.append(snap(core.net))
    return nets, events


# ============================================================
# §6.5 ① 帧间网络骨架：id 对齐
# ============================================================
def frame_trace(nets):
    """逐帧 id 对齐读数（跨帧的锚 = id）。"""
    out = []
    for tau, net in enumerate(nets):
        ids = set(net.ids())
        if tau == 0:
            born, dead = sorted(ids), []
        else:
            prev = set(nets[tau - 1].ids())
            born = sorted(ids - prev)
            dead = sorted(prev - ids)
        persist = len(ids) - len(born)
        out.append({"t": tau, "V": len(ids), "E": net.E(),
                    "born": born, "dead": dead,
                    "n_born": len(born), "n_dead": len(dead),
                    "n_persist": persist})
    return out


def attr_series(nets, ids=None):
    """持久 id 的属性时间序列：`deg / r(deg) / σ_v`（r、σ 取自 §6.3 局部量）。

    `ids=None` ⇒ 取**全帧存活**的 id（id 持久性最强的一批）；缺席帧记 `None`。
    """
    if ids is None:
        live = [set(n.ids()) for n in nets]
        ids = sorted(set.intersection(*live)) if live else []
    out = {}
    for v in ids:
        dg, rr, sg = [], [], []
        for net in nets:
            if v in net.rot:
                d = net.deg(v)
                dg.append(d)
                rr.append(L1.radius_of(d))
                sg.append(L1.sigma_of(d))
            else:
                dg.append(None)
                rr.append(None)
                sg.append(None)
        out[v] = {"deg": dg, "r": rr, "sigma": sg}
    return out


# ============================================================
# §6.5 ② 张力场指标层（GT-005~008，坐标无关）
# ============================================================
def delta(net, k=2):
    """GT-005 悬挂端密度 δ_k = |{v : deg(v) < k}| / |V|。

    注册定义取 `k=2`（公理下限）；`k=dmin` 为修剪边界读数，`k=KAPPA`（=6）为
    三角剖分的正角亏端读数。空图返回 `None`。
    """
    ids = net.ids()
    if not ids:
        return None
    return sum(1 for v in ids if net.deg(v) < k) / len(ids)


def anchors_of(net):
    """锚点 = σ 局部极值（坐标无关；与 §6.4 `anchors` 同口径）。

    `σ_v = 2/deg_v` ⇒ `∇σ_v < 0 ⟺ deg_v < mean_{w~v} deg_w`（局部最小度 = σ 局部极大）。
    等度图（如正二十面体）两者皆空 —— 此时不存在"引力中心"。
    """
    gs = L1.grad_sigma(net)
    return {"sinks": sorted(v for v, g in gs.items() if g < -_TOL),
            "sources": sorted(v for v, g in gs.items() if g > _TOL)}


def anchor_drift(prev, cur):
    """GT-006 锚点漂移率 Δμ(τ) = (1/C)·Σ_c |μ_c(τ) − μ_c(τ−1)|。

    μ_c = 类别 c 的**度质心**（GT-006 的一维口径；本层锚点类别 = 单点 ⇒ μ_c = deg(c)），
    类别按 **id 对齐**。`churn` = 锚点集 Jaccard 距离（锚点被重新定义/诞生/消失的度量）。

    ⚠️ 无共同锚点时 `Δμ ≡ 0` —— 此时必须看 `churn`（=1.0 表示锚点集完全换血），
    不可据 `Δμ=0` 判"稳定"。
    """
    A = set(anchors_of(prev)["sinks"]) | set(anchors_of(prev)["sources"])
    B = set(anchors_of(cur)["sinks"]) | set(anchors_of(cur)["sources"])
    common, union = A & B, A | B
    churn = 0.0 if not union else 1.0 - len(common) / len(union)
    if not common:
        return {"dmu": 0.0, "churn": churn, "n_prev": len(A),
                "n_cur": len(B), "n_common": 0}
    dmu = sum(abs(cur.deg(c) - prev.deg(c)) for c in sorted(common)) / len(common)
    return {"dmu": dmu, "churn": churn, "n_prev": len(A),
            "n_cur": len(B), "n_common": len(common)}


def dtopo(prev, cur, w_plus=W_PLUS, w_minus=W_MINUS):
    """GT-007 帧间拓扑距离。三个口径（全部由边集对称差给出，无需事件计数）：

      d_topo = |E_τ Δ E_{τ+1}| / max(|E_τ|, |E_{τ+1}|)
      d_norm = |E_τ Δ E_{τ+1}| / (|E_τ| + |E_{τ+1}|)
      d_w    = w⁺·|V⁺| + w⁻·|V⁻|，其中 V⁺ = E_cur−E_prev、V⁻ = E_prev−E_cur（边集口径）
    """
    ep, ec = edge_set(prev), edge_set(cur)
    plus, minus = len(ec - ep), len(ep - ec)
    sym = plus + minus
    den, tot = max(len(ep), len(ec)), len(ep) + len(ec)
    return {"d_topo": (sym / den) if den else 0.0,
            "d_norm": (sym / tot) if tot else 0.0,
            "d_w": w_plus * plus + w_minus * minus,
            "n_plus": plus, "n_minus": minus, "sym_diff": sym}


def sigma_global(net):
    """GT-008 全局 σ = |V| / |E| = 2 / ⟨deg⟩。"""
    E = net.E()
    return (net.V() / E) if E else float("inf")


# ============================================================
# §6.5 ③ 双层闭合判据
# ============================================================
def closure_series(rows, theta_delta=THETA_DELTA, theta_mu=THETA_MU,
                   k=K_CLOSED, key_delta="delta_kappa", key_mu="dmu"):
    """双层判据：`闭合(τ) = (δ(τ) < θ_δ) ∧ (Δμ(τ) < θ_μ)`，并统计**连续 k 帧**的段。

    N013 修正：单靠 δ 不能判闭合（δ 对锚点漂移无感）⇒ 必须两项同时成立。
    第 0 帧无 Δμ ⇒ 恒记 `False`。
    """
    flags = []
    for r in rows:
        d, mu = r.get(key_delta), r.get(key_mu)
        flags.append(bool(d is not None and mu is not None
                          and d < theta_delta and mu < theta_mu))
    segs, s = [], None
    for i, f in enumerate(flags):
        if f and s is None:
            s = i
        elif not f and s is not None:
            segs.append((s, i - 1))
            s = None
    if s is not None:
        segs.append((s, len(flags) - 1))
    long_segs = [(a, b) for (a, b) in segs if b - a + 1 >= k]
    return {"flags": flags, "segments": segs, "long_segments": long_segs,
            "closed": bool(long_segs), "k": k,
            "theta_delta": theta_delta, "theta_mu": theta_mu}


# ============================================================
# 逐帧指标序列
# ============================================================
def series(nets, events=None, k_delta=KAPPA):
    """把帧序列投影成一维指标序列（每帧一行，读起来像时间序列）。"""
    rows = []
    for tau, net in enumerate(nets):
        ids = net.ids()
        V, E = net.V(), net.E()
        r = {"t": tau, "V": V, "E": E, "sigma": sigma_global(net),
             "mean_deg": (2.0 * E / V) if V else 0.0,
             "max_deg": max((net.deg(v) for v in ids), default=0),
             "delta2": delta(net, 2), "delta_dmin": None, "delta_kappa": delta(net, k_delta)}
        if tau == 0:
            r.update({"dmu": None, "churn": None, "n_anchors": 0,
                      "d_topo": None, "d_norm": None, "d_w": None,
                      "n_plus": None, "n_minus": None,
                      "event_born": None, "event_dead": None, "event_broken": None})
        else:
            ad = anchor_drift(nets[tau - 1], net)
            dt = dtopo(nets[tau - 1], net)
            r.update({"dmu": ad["dmu"], "churn": ad["churn"],
                      "n_anchors": ad["n_cur"],
                      "d_topo": dt["d_topo"], "d_norm": dt["d_norm"], "d_w": dt["d_w"],
                      "n_plus": dt["n_plus"], "n_minus": dt["n_minus"],
                      "event_born": None, "event_dead": None, "event_broken": None})
            if events:
                ev = events[tau - 1]
                r["event_born"] = ev.get("born")
                r["event_dead"] = ev.get("dead")
                r["event_broken"] = ev.get("broken")
        rows.append(r)
    return rows


_COLS = ["t", "V", "E", "sigma", "mean_deg", "max_deg",
         "delta2", "delta_dmin", "delta_kappa", "n_anchors",
         "dmu", "churn", "d_topo", "d_norm", "d_w", "n_plus", "n_minus",
         "event_born", "event_dead", "event_broken"]


# ============================================================
# §6.5 ④ 导出
# ============================================================
def export_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_COLS)
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in _COLS])


def export_json(path, rows, closure=None, trace=None):
    obj = {"rows": rows, "closure": closure}
    if trace is not None:
        obj["trace"] = trace
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


# ============================================================
# 入口
# ============================================================
def global_evolution(core, nframes=6, k_delta=KAPPA, theta_delta=THETA_DELTA,
                     theta_mu=THETA_MU, k_closed=K_CLOSED):
    """L3 入口：跑 L0 取帧序列 → ① 对齐 ② 指标 ③ 闭合判据（不做任何坐标求解）。"""
    nets, events = trace_frames(core, nframes)
    tr = frame_trace(nets)
    rows = series(nets, events, k_delta)
    for r, t in zip(rows, tr):
        r["delta_dmin"] = delta(nets[r["t"]], core.dmin)
    cl = closure_series(rows, theta_delta, theta_mu, k_closed)
    return {"rows": rows, "closure": cl, "trace": tr,
            "attrs": attr_series(nets),
            "V": [t["V"] for t in tr], "E": [t["E"] for t in tr]}


# ============================================================
# 自检
# ============================================================
def selftest():
    def close(a, b, tol=1e-9):
        return abs(a - b) <= tol

    # ---- 基础：边集 / 快照 / δ / σ ----
    ic = RotNet(SEEDS["icosa"]())
    assert len(edge_set(ic)) == ic.E() == 30
    s = snap(ic)
    assert s.rot == ic.rot and s.rot is not ic.rot and s.nid == ic.nid
    s.rot[0] = [99]                      # 改快照不得影响原网（深拷贝）
    assert ic.rot[0] != [99]
    assert close(sigma_global(ic), 12 / 30, 1e-12)
    assert delta(ic, 2) == 0.0
    assert delta(ic, 5) == 0.0           # 全 deg=5 ⇒ 无 <5
    assert close(delta(ic, 6), 1.0, 1e-12)   # 全 deg=5 ⇒ 全为「正角亏端」
    te = RotNet(SEEDS["tetra"]())
    assert close(delta(te, 4), 1.0, 1e-12)   # 全 deg=3 < 4
    assert delta(RotNet({}), 2) is None      # 空图

    # ---- 等度图无锚点；同帧自比 Δμ=0、churn=0 ----
    an = anchors_of(ic)
    assert an["sinks"] == [] and an["sources"] == []
    ad = anchor_drift(ic, ic)
    assert ad["dmu"] == 0.0 and ad["churn"] == 0.0 and ad["n_common"] == 0

    # ---- d_topo 三口径：自比恒 0；|E| 相同时 d_topo = d_norm = sym/|E| ----
    dt = dtopo(ic, ic)
    assert dt["d_topo"] == 0.0 and dt["d_norm"] == 0.0 and dt["d_w"] == 0.0
    # 构造：删掉 icosa 的一条边 → |V⁻|=1，|V⁺|=0
    cur = snap(ic)
    u, w = sorted(edge_set(ic))[0]
    cur.rot[u] = [x for x in cur.rot[u] if x != w]
    cur.rot[w] = [x for x in cur.rot[w] if x != u]
    dt = dtopo(ic, cur)
    assert dt["n_minus"] == 1 and dt["n_plus"] == 0 and dt["sym_diff"] == 1
    assert close(dt["d_topo"], 1 / 30, 1e-12)      # max(30,29)=30
    assert close(dt["d_norm"], 1 / 59, 1e-12)      # 30+29
    assert close(dt["d_w"], W_MINUS, 1e-12)        # w⁻=2

    # ---- 闭合判据：构造 5 行（δ 小但 Δμ 大 ⇒ 不闭合；两项都小 ⇒ 闭合）----
    def row(d, mu):
        return {"delta_kappa": d, "dmu": mu}
    bad = [row(0.0, 9.0)] * 6
    cb = closure_series(bad, k=5)
    assert cb["closed"] is False and cb["long_segments"] == []
    # δ 全为 0（对锚点漂移完全无感，N013 结构性复现）
    assert all(f is False for f in cb["flags"])
    good = [row(0.0, 1.0)] * 7
    cg = closure_series(good, k=5)
    assert cg["closed"] is True and cg["long_segments"] == [(0, 6)]
    cg2 = closure_series([row(0.0, 0.0)] * 3, k=5)
    assert cg2["closed"] is False                   # 不足 k 帧
    assert cg2["segments"] == [(0, 2)]

    # ---- L0 采纳轨迹：id 对齐 + 逐帧指标 + 确定性 ----
    core = m.L0Core(m.RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                    vminus="dense", vplus="any")
    out = global_evolution(core, nframes=4)
    rows, tr = out["rows"], out["trace"]
    assert [t["V"] for t in tr[:3]] == [12, 32, 36]        # 与 §7.4 冻结读数一致
    # t=0 → t=1：12 个老 id 全持久、新增 20
    assert tr[1]["n_persist"] == 12 and tr[1]["n_born"] == 20 and tr[1]["n_dead"] == 0
    # 树的根帧全为 born（无前帧）
    assert tr[0]["n_born"] == 12 and tr[0]["n_dead"] == 0
    # 序列行自洽：d_topo / d_norm 由 n_plus、n_minus 与两帧 |E| 唯一确定
    assert rows[0]["dmu"] is None and rows[0]["d_topo"] is None
    for i in range(1, len(rows)):
        sym = rows[i]["n_plus"] + rows[i]["n_minus"]
        e0, e1 = rows[i - 1]["E"], rows[i]["E"]
        assert close(rows[i]["d_topo"], sym / max(e0, e1), 1e-12)
        assert close(rows[i]["d_norm"], sym / (e0 + e1), 1e-12)
        assert close(rows[i]["d_w"], W_PLUS * rows[i]["n_plus"]
                     + W_MINUS * rows[i]["n_minus"], 1e-12)
        assert rows[i]["V"] == tr[i]["V"] and rows[i]["E"] == tr[i]["E"]
        assert rows[i]["delta_dmin"] == 0.0   # 仅对 nframes=4 成立；大 V⁻ 帧例外见下
    # 属性序列：全帧存活 id 的 deg/r/σ 三者一致（σ = 2/deg）
    for v, a in out["attrs"].items():
        for d, rr, sg in zip(a["deg"], a["r"], a["sigma"]):
            assert close(rr, L1.radius_of(d), 1e-12)
            assert close(sg, 2.0 / d, 1e-12)
    # 确定性：同配置两次逐帧一致
    c2 = m.L0Core(m.RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                  vminus="dense", vplus="any")
    out2 = global_evolution(c2, nframes=4)
    assert [r["V"] for r in out2["rows"]] == [r["V"] for r in rows]
    assert [r["d_topo"] for r in out2["rows"]] == [r["d_topo"] for r in rows]

    # ---- 第④项「不完美定理的持续验证」（10 帧采纳轨迹）----
    # 边缘臂：帧末修剪只删一层 ⇒ 大 V⁻ 帧（t=9，dead=5）仍残留悬挂端，δ_dmin 并非恒 0。
    core9 = m.L0Core(m.RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                     vminus="dense", vplus="any")
    out9 = global_evolution(core9, nframes=9)
    dd = [r["delta_dmin"] for r in out9["rows"]]
    dk = [r["delta_kappa"] for r in out9["rows"]]
    assert all(x == 0.0 for x in dd[:9])          # t=0..8：无大 V⁻ ⇒ 残留臂全 0
    assert close(dd[9], 1.0 / 558.0, 1e-12)       # t=9：1 个残留 / V=558 ≈ 0.001792
    assert all(x > 0.5 for x in dk[1:])           # 泛化臂 δ_κ 恒非零（0.51~1.00）
    # 双层 AND 判据（δ_κ<θ_δ ∧ Δμ<θ_μ 连续 k 帧）在 10 帧上从未满足
    assert out9["closure"]["closed"] is False
    assert out9["closure"]["long_segments"] == []
    assert not any(out9["closure"]["flags"])

    # ---- 导出：CSV / JSON 往返 ----
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cp, jp = os.path.join(d, "s.csv"), os.path.join(d, "s.json")
        export_csv(cp, rows)
        export_json(jp, rows, out["closure"], tr)
        with open(cp, encoding="utf-8") as f:
            got = list(csv.reader(f))
        assert got[0] == _COLS and len(got) == len(rows) + 1
        with open(jp, encoding="utf-8") as f:
            obj = json.load(f)
        assert obj["rows"][0]["V"] == rows[0]["V"] and "closure" in obj

    print("l3_projection selftest: 全部通过")
    print("  id 对齐：t=0→t=1 持久 12 / 新增 20 / 消失 0（V 12→32→36，与 §7.4 冻结读数一致）")
    print("  δ 读数：δ_2 ≡ 0（帧末修剪已删 deg<dmin）、δ_5 = 0、δ_6 = 1.0（全正角亏端）")
    print("  闭合判据：δ 全 0 而 Δμ=9 ⇒ 不闭合（N013「δ 对锚点漂移无感」结构性复现）")
    print("  导出：CSV 21 列 + JSON（rows/closure/trace）往返一致；两次运行逐帧一致")


def _fmt(x, w, p):
    return "—".rjust(w) if x is None else f"{x:{w}.{p}f}"


def _demo(rows, closure):
    print("  t     V      E   σ=V/E  ⟨deg⟩  maxdeg  δ_6     Δμ   churn  d_topo  d_w    V⁺/V⁻   事件(b/d/br)")
    for r in rows:
        ev = "—" if r["event_born"] is None else \
            f"{r['event_born']}/{r['event_dead']}/{r['event_broken']}"
        pm = "—" if r["n_plus"] is None else f"{r['n_plus']}/{r['n_minus']}"
        print(f"  {r['t']:<4} {r['V']:>4} {r['E']:>6}  {r['sigma']:.4f} "
              f"{r['mean_deg']:6.3f} {r['max_deg']:>6}  {r['delta_kappa']:.4f} "
              f"{_fmt(r['dmu'], 6, 3)} {_fmt(r['churn'], 6, 3)} {_fmt(r['d_topo'], 7, 3)} "
              f"{_fmt(r['d_w'], 6, 1)} {pm:>7}   {ev}")
    c = closure
    print(f"  闭合判据（δ_κ<{c['theta_delta']} ∧ Δμ<{c['theta_mu']} 连续 {c['k']} 帧）："
          f"闭合={c['closed']}；连续段={c['segments']}；达标段={c['long_segments']}")


def _kv(args):
    out = {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def main(argv):
    kv = _kv(argv)
    if kv.get("selftest") == "1":
        selftest()
        return 0
    seed = kv.get("seed", "icosa")
    nframes = int(kv.get("nframes", 6))
    kw = dict(cap=int(kv.get("cap", 12)), dmin=int(kv.get("dmin", 3)),
              drive=kv.get("drive", "slack"), pairing=kv.get("pairing", "none"),
              vminus=kv.get("vminus", "dense"), vplus=kv.get("vplus", "any"))
    core = m.L0Core(m.RotNet(SEEDS[seed]()), **kw)
    out = global_evolution(core, nframes,
                           theta_delta=float(kv.get("theta_d", THETA_DELTA)),
                           theta_mu=float(kv.get("theta_mu", THETA_MU)),
                           k_closed=int(kv.get("k", K_CLOSED)))
    print(f"l3_projection · seed={seed} nframes={nframes} " +
          " ".join(f"{a}={b}" for a, b in kw.items()))
    _demo(out["rows"], out["closure"])
    if kv.get("csv"):
        export_csv(kv["csv"], out["rows"])
        print(f"  → CSV: {kv['csv']}")
    if kv.get("json"):
        export_json(kv["json"], out["rows"], out["closure"], out["trace"])
        print(f"  → JSON: {kv['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
