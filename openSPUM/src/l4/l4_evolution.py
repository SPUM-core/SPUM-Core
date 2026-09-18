# -*- coding: utf-8 -*-
"""l4_evolution.py — L4 演化观察层：净湮灭 / 集中现象 / 不完美定理（跨帧只读观察）。

架构基准: docs/L0L1L2_Architecture.md §6.6（阶段五「演化观察」；本轮新立）
本体语义: spum-evolution.md §四「净湮灭效应：引力的拓扑本质」/ §六「不完美定理」

一句话
------
**L1/L2 投影「一帧之内」、L3 投影「帧与帧之间」、L4 观察「事件落在哪个区域」。**

L4 只读 L0 的帧序列，**不参与 L0 规则、不回写任何状态**；四组读数全部坐标无关、
只吃 id + 环序（不依赖 L2 坐标）。分区依据是**度数**（纯组合量）：

    over  : deg > κ          过密区（负曲率端）
    flat  : deg == κ
    under : deg < κ          稀疏区（正曲率端）
    zone  : 全 over → "core"；含 ≥1 个 over → "sh"（核的边界）；其余 → "bg"（背景）

读数（对应 §7 构建路线「阶段五」四项）：

  ① 分区事件账 `event_account` / `annihilation_profile`
       **按事件位点**（不是按边端点！见下「口径」）把 V⁺ 与 V⁻ 分到 core/sh/bg，
       给出逐分区净 ΔE 与净湮灭率 ρ；（`rho > 0` = 净湮灭，`rho < 0` = 净创生）
  ② 集中现象 `concentration_series`：max_deg / n_sat / gini / σ 方差 / 三分区规模
  ③ 不完美定理 `imperfection_series`：帧末残留悬挂端、帧内删除数、L0-A9 同帧断边+删点
  ④ 入口 `observe(core, nframes)` + `export_csv` / `export_json`
  ⑤ 结构闭合审计 `closure_of` / `closure_series`
       —— **阶段五第③项「闭合子图内部与外部」的落地**（判决见下）

⚠️ **阶段五第③项判决（探针取证 + 实测）**：采纳轨迹上**每一帧都是单一连通闭曲面**
   （`n_comp ≡ 1`、`boundary_edges ≡ 0`、`χ ≡ 2`；icosa 与 patch 两种子皆然）⇒
   「闭合子图内部 vs 外部」在**分量层面退化**：不存在分量级的内外之分。
   其非平凡内容即下方 ① 的 `core`/`bg` 分区——**过密核 = 局部闭合内部（`core`）、
   背景 = 外部（`bg`）**。可冻结的恒等式：闭合连通（χ=2）帧上
   `Σ(6−deg) = 12 + 2·holes`，其中 `holes = 3V − 6 − E` 是偏离纯三角剖分的「边亏损」。
   ⇒ 第③项**归并**到 ① 的度数分区，不另立一套「内外」判据。

⚠️ **口径（本轮实测确立，防一个构造性假象）**：V⁺ **必须按「位点」分区**——
   一次创生（锥化）总是「在一张面的三个角上长出**一个全新元胞**」，故若按**边端点**
   分区，每条新边的两端里必有一端是新点（帧初不存在 ⇒ 必落 "under"）⇒
   `V⁺core ≡ 0` 将成为**定义强制的假象**，而非实测结论。按位点（= 新元胞的三个角）
   分区后，V⁺core 才可能非零（cap=64 轨迹 t≥7 即出现 96/8/160/788）。

⚠️ **口径二（必读）：V⁻ 的分区分布是规则强制的，V⁺ 的分区分布才是经验读数。**
   `vminus="dense"` 的湮灭判据是「**两端皆 over**」，故按构造 `V⁻sh ≡ V⁻bg ≡ 0`、
   `V⁻core` 就是全部断边 ⇒ `rho_bg = −1.000` 是**恒等式**，不是发现。
   真正可检验的是 **V⁺ 的分区分布**（一次锥化落在「全过密角的面」还是「含过密角的面」
   还是「背景面」），以及由它决定的核心净符号：
   `净core = V⁺core − V⁻core`，`rho_core > 0 ⟺ V⁺core < V⁻core`（核心净湮灭）。
   读数：cap=12 时 `V⁺core ≡ 0` ⇒ `rho_core = +1.0000`（纯净湮灭）；
   cap=64 时 `V⁺core` 自 t=7 起非零（96/8/160/788）⇒ `rho_core = +0.9284`。
   ⇒ **核心净湮灭不是恒等式**（大 cap 会削弱它），而背景净创生是恒等式。

⚠️ **不完美读数在采纳动力学下大多退化**（与 L3 的 δ 臂同族）：帧末 `deg < dmin`
   基本恒为 0（同帧创生把被断边压低的点救回），仅 t=9 出现 1 例；`dead > 0` 亦稀疏。
   L4 只如实记录，不据此改 L0。

用法
----
  python l4_evolution.py selftest=1
  python l4_evolution.py seed=icosa nframes=10 cap=12 dmin=3 vminus=dense vplus=any
  python l4_evolution.py seed=icosa nframes=10 cap=12 csv=acc.csv json=obs.json
"""

import csv
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for _d in ("l0", "l1"):
    _p = os.path.join(_SRC, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from combinatorial_proto import RotNet, SEEDS, find_crystallites  # noqa: E402
import l0_core as m                            # noqa: E402
import l1_projection as L1                     # noqa: E402

KAPPA = L1.KAPPA          # 平坦阈值 κ=6（deg>κ ⇒ 负曲率 / 过密端）
_ZONES = ("core", "sh", "bg")


# ============================================================
# 基础：帧快照 / 边集（与 L3 同口径，各自定义以保持层间解耦）
# ============================================================
def snap(net):
    """帧快照（深拷贝旋转系统，保留 id 计数器）——L4 以帧为观察单位。"""
    c = RotNet({v: list(r) for v, r in net.rot.items()})
    c.nid = net.nid
    return c


def edge_set(net):
    """无向边集合（id 对，规范化 (min,max)）。"""
    return {(v, w) if v < w else (w, v)
            for v in net.ids() for w in net.rot[v]}


# ============================================================
# §6.6 分区（纯组合：只看度数）
# ============================================================
def partition(net, kappa=KAPPA):
    """按度数三分区。返回 `{v: label}`，label ∈ {"over","flat","under"}。

    `deg > κ` → "over"（过密 / 负曲率端）；`deg == κ` → "flat"；`deg < κ` → "under"。
    空图返回 `{}`。
    """
    out = {}
    for v in net.ids():
        d = net.deg(v)
        out[v] = "over" if d > kappa else ("flat" if d == kappa else "under")
    return out


def zone(labels):
    """把「若干位点的标签」压成一个区：全为 "over" → "core"；含 ≥1 个 "over" → "sh"；
    其余（**含空列表**）→ "bg"。

    ⚠️ 空列表必须回 "bg"：`all(...)` 对空序列恒为 True，不能直接用。
    """
    if not labels:
        return "bg"
    if all(x == "over" for x in labels):
        return "core"
    if any(x == "over" for x in labels):
        return "sh"
    return "bg"


def _zeros():
    return {z: 0 for z in _ZONES}


# ============================================================
# §6.6 ① 分区事件账（净湮灭效应）
# ============================================================
def event_account(nets, kappa=KAPPA, cap=None):
    """逐帧按**事件位点**分区记账。

    `nets` 为帧快照序列（`nets[0]` = 帧初）；返回长度 `len(nets)-1` 的行列表，
    第 i 行描述「把 `nets[i]` 变成 `nets[i+1]`」的那一帧（`t = i+1`）。

    字段：
      `t`       帧序号（从 1 起）
      `n_plus`  创生位点分区计数（**位点 = 新元胞的三个角**，标签取**帧初**快照）
      `n_minus` 断边位点分区计数（位点 = 断边两端，两端须在帧末仍存活）
      `net`     = `n_plus − n_minus`（逐分区）
      `n_follow` 随删点消失的边数（至少一端在帧末已不存在）——单列，不进分区账
      `viol_sat` 创生位点中「任一角的**帧初**度数 ≥ cap」的个数（`cap=None` → None）
                 ——这是「创生抑制」的**闸门读数**：期望恒 0（见 §6.6 口径）
    """
    rows = []
    for i in range(len(nets) - 1):
        prev, cur = nets[i], nets[i + 1]
        prev_lab = partition(prev, kappa)          # 整帧只用这一份「帧初标签」
        n_plus, n_minus = _zeros(), _zeros()
        n_follow = 0
        viol_sat = 0 if cap is not None else None

        # --- 创生位点：新元胞（帧初不存在）的三角即其位点 ---
        for v in cur.ids():
            if v in prev.rot:
                continue
            corners = list(cur.rot[v])
            n_plus[zone([prev_lab.get(x, "under") for x in corners])] += 1
            if cap is not None and any(
                    x in prev.rot and prev.deg(x) >= cap for x in corners):
                viol_sat += 1

        # --- 断边位点：两端在帧末仍存活者落入分区账，其余记 n_follow ---
        for (u, w) in sorted(edge_set(prev) - edge_set(cur)):
            if u not in cur.rot or w not in cur.rot:
                n_follow += 1
                continue
            n_minus[zone([prev_lab.get(u, "under"), prev_lab.get(w, "under")])] += 1

        rows.append({"t": i + 1, "n_plus": n_plus, "n_minus": n_minus,
                     "net": {z: n_plus[z] - n_minus[z] for z in _ZONES},
                     "n_follow": n_follow, "viol_sat": viol_sat})
    return rows


def annihilation_profile(rows):
    """把 `event_account` 的行汇总成净湮灭画像。

    逐分区 z：`P[z] = Σ n_plus[z]`、`M[z] = Σ n_minus[z]`、
    `rho[z] = (M[z] − P[z]) / (M[z] + P[z])`（分母 0 → 0.0）、`dE[z] = Σ net[z]`。

    符号约定：`rho > 0` = **净湮灭**（断边多于创生）；`rho < 0` = **净创生**。
    空 `rows` → 计数全 0、`rho` 全 0.0、`n_frames = 0`。
    """
    prof = {"P": _zeros(), "M": _zeros(), "dE": _zeros(),
            "rho": {z: 0.0 for z in _ZONES}, "n_frames": len(rows)}
    for r in rows:
        for z in _ZONES:
            prof["P"][z] += r["n_plus"][z]
            prof["M"][z] += r["n_minus"][z]
            prof["dE"][z] += r["net"][z]
    for z in _ZONES:
        tot = prof["M"][z] + prof["P"][z]
        prof["rho"][z] = (prof["M"][z] - prof["P"][z]) / tot if tot else 0.0
    return prof


# ============================================================
# §6.6 ② 集中现象
# ============================================================
def gini(xs):
    """基尼系数（口径与 `L0Core.gini` 逐位一致）：xs 升序、n=len、tot=Σ。
    n==0 或 tot==0 → 0.0；否则 `2·Σ(i+1)·x_i / (n·tot) − (n+1)/n`。
    """
    xs = sorted(xs)
    n = len(xs)
    tot = sum(xs)
    if n == 0 or tot == 0:
        return 0.0
    acc = sum((i + 1) * x for i, x in enumerate(xs))
    return (2.0 * acc) / (n * tot) - (n + 1.0) / n


def concentration_series(nets, cap=None):
    """逐帧集中度读数。每帧一行：

      `t` `V` `E` `max_deg` `mean_deg`(=2E/V，V=0 → 0.0)
      `n_over` `n_flat` `n_under`（三分区节点数）
      `n_sat`（`deg ≥ cap` 的节点数；`cap=None` → None）
      `gini`（度数基尼）`sigma_var`（σ_v = 2/deg 的**总体**方差）

    空图行：`max_deg=0`、`mean_deg=gini=sigma_var=0.0`、三分区计数 0、`n_sat` 随 cap。
    """
    rows = []
    for t, net in enumerate(nets):
        ids = net.ids()
        V, E = len(ids), net.E()
        degs = sorted(net.deg(v) for v in ids)
        n_over = sum(1 for d in degs if d > KAPPA)
        n_flat = sum(1 for d in degs if d == KAPPA)
        n_sat = sum(1 for d in degs if d >= cap) if cap is not None else None
        if V == 0:
            max_deg, mean_deg, sv = 0, 0.0, 0.0
        else:
            max_deg, mean_deg = degs[-1], (2.0 * E / V)
            xs = [L1.sigma_of(d) for d in degs]
            mu = sum(xs) / V
            sv = sum((x - mu) ** 2 for x in xs) / V
        rows.append({"t": t, "V": V, "E": E, "max_deg": max_deg,
                     "mean_deg": mean_deg, "n_over": n_over, "n_flat": n_flat,
                     "n_under": V - n_over - n_flat, "n_sat": n_sat,
                     "gini": gini(degs), "sigma_var": sv})
    return rows


# ============================================================
# §6.6 ③ 不完美定理
# ============================================================
def imperfection_series(nets, events=None, dmin=3):
    """逐帧不完美读数（**边缘臂双口径**；中心臂 Δμ 见 L3 `anchor_drift`）。每帧一行：

      `t` `n_under_end`(= |{v ∈ nets[t] : deg < dmin}|，帧末残留悬挂端)
      `delta_dmin`(残留臂密度 = `n_under_end` / V；V=0 → 0.0)
      `delta_kappa`(泛化臂密度 = |{v : deg < κ}| / V；κ=6，**L3 / GT-005 同口径**)
      `n_pruned`(= events[t−1]["dead"]，`events=None` 或 `t==0` → None)
      `n_broken`(= events[t−1]["broken"]，同上 → None)
      `n_under_start`(上一帧的帧末残留；`t==0` → None)
      `a9`(= 同帧既断边又删点，L0-A9；数据缺 → None)

    `events[i]` 对应「把 `nets[i]` 变成 `nets[i+1]`」的那一帧。

    ⚠️ **两口径缺一不可（阶段五第④项「不完美定理的持续验证」的落地口径）**：
      · `delta_dmin` 是**注册口径**（§4.3 修剪边界），但在采纳动力学下**几乎恒 0** ——
        帧内修剪只删一层，被削到 deg<dmin 的邻居会被**同帧创生**救回；只有发生大 V⁻
        （`dead>0`）的帧才残留（实测 10 帧仅 t=9 一例，= 1/558 ≈ 0.001792）。
        ⇒ 它不是「每帧必现」，而是「**大 V⁻ 帧必现**」。
      · `delta_kappa` 是**泛化口径**（正角亏端），在采纳轨迹上**恒非零**（实测 0.51~1.00）
        ⇒ 「边缘不完美」的**持续**承载靠 `delta_kappa`，不是 `delta_dmin`。
      故「不完美定理」的持续验证 = **边缘臂（`delta_kappa`）与中心臂（L3 Δμ）的 AND
      判据持续不满足**（判词由 L3 `closure_series` 给出，本层只供数，不判阈值）。
    """
    rows = []
    for t, net in enumerate(nets):
        ids = net.ids()
        V = len(ids)
        n_under_end = sum(1 for v in ids if net.deg(v) < dmin)
        n_under_kappa = sum(1 for v in ids if net.deg(v) < KAPPA)
        ev = events[t - 1] if (events is not None and t > 0) else None
        n_pruned = ev.get("dead") if ev else None
        n_broken = ev.get("broken") if ev else None
        rows.append({"t": t, "n_under_end": n_under_end,
                     "delta_dmin": (n_under_end / V) if V else 0.0,
                     "delta_kappa": (n_under_kappa / V) if V else 0.0,
                     "n_pruned": n_pruned, "n_broken": n_broken,
                     "n_under_start": rows[-1]["n_under_end"] if t > 0 else None,
                     "a9": bool(n_broken and n_pruned) if ev else None})
    return rows


# ============================================================
# §6.6 ③′ 结构闭合审计（阶段五第③项「闭合子图内部与外部」落地）
# ============================================================
def closure_of(net):
    """单帧快照的**结构闭合审计**（纯组合、只读、坐标无关）。

    返回键：
      `n_comp`         连通分量个数
      `comp_sizes`     各分量顶点数（按 (顶点数降序, χ 升序) 排序）
      `chi`            各分量 χ = V_c − E_c + F_c（与 `comp_sizes` 同序同长；
                       面归其首个顶点 `vs[0]` 所在的分量）
      `boundary_edges` 只被 < 2 个面包含的边数（**闭曲面 → 0**）
      `sum6deg`        Σ(6 − deg(v))；闭连通三角剖分 → 12，含 k≥4 洞 → 12 + 2·`holes`
      `holes`          d = 3V − 6 − E（偏离纯三角剖分的「边亏损」；V=0 → 0）
      `crystallites`   5-正则分量顶点数列表（`find_crystallites` 口径；<12 点被滤掉）

    边界情形：空图 → `n_comp=0`、诸列表空、`sum6deg=0`、`holes=0`；含孤立顶点
    （`rot=[]`）时它自成一个 χ=1 的分量，故 `n_comp` 与 `χ` 会相应变化。
    """
    comp_id, comps = {}, []
    for v in net.ids():
        if v in comp_id:
            continue
        comp, st = [], [v]
        comp_id[v] = len(comps)
        while st:
            u = st.pop()
            comp.append(u)
            for w in net.rot[u]:
                if w not in comp_id:
                    comp_id[w] = len(comps)
                    st.append(w)
        comps.append(comp)

    f_c = [0] * len(comps)
    ecount = defaultdict(int)
    for cyc in net.faces():
        vs = net.face_vertices(cyc)
        f_c[comp_id[vs[0]]] += 1
        k = len(vs)
        for i in range(k):
            a, b = vs[i], vs[(i + 1) % k]
            ecount[(a, b) if a < b else (b, a)] += 1
    e_c = [0] * len(comps)
    for (a, _b) in ecount:
        e_c[comp_id[a]] += 1

    rec = sorted(({"size": len(c), "chi": len(c) - e_c[i] + f_c[i]}
                  for i, c in enumerate(comps)),
                 key=lambda r: (-r["size"], r["chi"]))
    V, E = net.V(), net.E()
    return {"n_comp": len(comps),
            "comp_sizes": [r["size"] for r in rec],
            "chi": [r["chi"] for r in rec],
            "boundary_edges": sum(1 for c in ecount.values() if c < 2),
            "sum6deg": sum(6 - net.deg(v) for v in net.ids()),
            "holes": 3 * V - 6 - E if V else 0,
            "crystallites": [c["n"] for c in find_crystallites(net)]}


def closure_series(nets):
    """逐帧结构闭合审计：第 i 项 = `{"frame": i, **closure_of(nets[i])}`。

    用键 `frame`（从 0 起）而非事件账的 `t`（从 1 起）——前者是**帧状态**读数、
    后者是**帧事件**读数，二者不得混用。
    """
    return [{"frame": i, **closure_of(n)} for i, n in enumerate(nets)]


# ============================================================
# §6.6 ④ 入口
# ============================================================
def observe(core, nframes, kappa=KAPPA, cap=None):
    """L4 入口：跑 L0 取帧序列，一次产出五组读数（**只调用 `core.frame()`**）。"""
    if cap is None:
        cap = getattr(core, "cap", None)
    dmin = getattr(core, "dmin", 3)
    nets, events = [snap(core.net)], []
    for _ in range(int(nframes)):
        events.append(dict(core.frame()))
        nets.append(snap(core.net))
    accounts = event_account(nets, kappa, cap)
    return {"accounts": accounts, "profile": annihilation_profile(accounts),
            "concentration": concentration_series(nets, cap),
            "imperfection": imperfection_series(nets, events, dmin),
            "closure": closure_series(nets),
            "V": [n.V() for n in nets], "E": [n.E() for n in nets],
            "nets": nets, "events": events}


# ============================================================
# §6.6 ④ 导出
# ============================================================
ACC_COLS = ("t", "plus_core", "plus_sh", "plus_bg",
            "minus_core", "minus_sh", "minus_bg",
            "net_core", "net_sh", "net_bg", "n_follow", "viol_sat")


def flatten_accounts(rows):
    """把分区事件账摊平成一维行（供 CSV/JSON 用）。"""
    out = []
    for r in rows:
        out.append({"t": r["t"],
                    **{f"plus_{z}": r["n_plus"][z] for z in _ZONES},
                    **{f"minus_{z}": r["n_minus"][z] for z in _ZONES},
                    **{f"net_{z}": r["net"][z] for z in _ZONES},
                    "n_follow": r["n_follow"], "viol_sat": r["viol_sat"]})
    return out


def export_csv(path, rows, cols):
    """写 CSV（`None` 写空串）。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(cols))
        for r in rows:
            w.writerow(["" if r.get(c) is None else r.get(c) for c in cols])


def export_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


# ============================================================
# 自检
# ============================================================
def selftest():
    def close(a, b, tol=1e-12):
        return abs(a - b) <= tol

    ic = RotNet(SEEDS["icosa"]())

    # ---- 夹具 A：分区（icosa 全 deg=5 < 6 ⇒ 全 "under"）----
    lab = partition(ic, kappa=6)
    assert len(lab) == 12 and set(lab.values()) == {"under"}
    cnt = {k: sum(1 for v in lab if lab[v] == k) for k in ("over", "flat", "under")}
    assert cnt == {"over": 0, "flat": 0, "under": 12}
    assert partition(RotNet({})) == {}

    # ---- 夹具 B：压区（含空列表边界）----
    assert zone(["over", "over", "over"]) == "core"
    assert zone(["over", "flat", "under"]) == "sh"
    assert zone(["under", "flat"]) == "bg"
    assert zone([]) == "bg"                       # ⚠️ all() 对空序列恒 True，必须挡住

    # ---- 夹具 C：事件账第一帧（icosa 全 under ⇒ 20 个新元胞落 bg）----
    core = m.L0Core(RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                    vminus="dense", vplus="any")
    out = observe(core, nframes=10)
    acc, prof = out["accounts"], out["profile"]
    assert len(acc) == 10
    a1 = acc[0]
    assert a1["t"] == 1
    assert a1["n_plus"] == {"core": 0, "sh": 0, "bg": 20}
    assert a1["n_minus"] == {"core": 0, "sh": 0, "bg": 0}
    assert a1["net"] == {"core": 0, "sh": 0, "bg": 20}
    assert a1["n_follow"] == 0 and a1["viol_sat"] == 0

    # ---- 夹具 D：净湮灭画像（构造行）----
    d = annihilation_profile([{"n_plus": {"core": 0, "sh": 4, "bg": 0},
                               "n_minus": {"core": 30, "sh": 0, "bg": 0},
                               "net": {"core": -30, "sh": 4, "bg": 0}}])
    assert d["P"] == {"core": 0, "sh": 4, "bg": 0}
    assert d["M"] == {"core": 30, "sh": 0, "bg": 0}
    assert d["rho"] == {"core": 1.0, "sh": -1.0, "bg": 0.0}
    assert d["dE"] == {"core": -30, "sh": 4, "bg": 0} and d["n_frames"] == 1
    e = annihilation_profile([])
    assert e["n_frames"] == 0 and e["rho"] == {"core": 0.0, "sh": 0.0, "bg": 0.0}

    # ---- 夹具 E：集中现象首行（icosa 等度 ⇒ gini=0、σ 方差 0）----
    c0 = out["concentration"][0]
    assert (c0["t"], c0["V"], c0["E"], c0["max_deg"], c0["mean_deg"]) == (0, 12, 30, 5, 5.0)
    assert (c0["n_over"], c0["n_flat"], c0["n_under"], c0["n_sat"]) == (0, 0, 12, 0)
    assert close(c0["gini"], 0.0)            # 等度 ⇒ 基尼恒 0
    assert close(c0["sigma_var"], 0.0, 1e-30)   # 浮点残差 ~1e-33，非逐位 0
    assert close(gini([5] * 12), 0.0)
    assert close(gini([2, 2, 2, 2]), 0.0)

    # ---- 夹具 F：不完美读数（无前帧 ⇒ None）----
    f = imperfection_series([ic, ic], [{"born": 0, "dead": 0, "broken": 0}], dmin=3)
    assert f[0]["n_pruned"] is None and f[0]["a9"] is None and f[0]["n_under_start"] is None
    assert f[1]["n_under_end"] == 0 and f[1]["n_pruned"] == 0 and f[1]["n_broken"] == 0
    assert f[1]["a9"] is False
    f2 = imperfection_series([ic])              # 无 events、单帧
    assert f2[0]["n_pruned"] is None and f2[0]["a9"] is None
    assert f2[0]["delta_kappa"] == 1.0 and f2[0]["delta_dmin"] == 0.0   # 全 deg=5<κ
    assert imperfection_series([RotNet({})])[0]["delta_kappa"] == 0.0  # 空图

    # ---- 夹具 G：结构闭合审计（阶段五第③项）----
    cA = closure_of(ic)
    assert cA == {"n_comp": 1, "comp_sizes": [12], "chi": [2],
                  "boundary_edges": 0, "sum6deg": 12, "holes": 0,
                  "crystallites": [12]}
    cB = closure_of(RotNet(SEEDS["patch"]()))
    assert (cB["n_comp"], cB["comp_sizes"], cB["chi"]) == (1, [19], [2])
    assert (cB["boundary_edges"], cB["sum6deg"], cB["holes"]) == (0, 30, 9)
    assert cB["crystallites"] == []
    assert closure_of(RotNet({})) == {"n_comp": 0, "comp_sizes": [], "chi": [],
                                      "boundary_edges": 0, "sum6deg": 0,
                                      "holes": 0, "crystallites": []}
    cE = closure_of(RotNet({**ic.rot, 999: []}))   # icosa + 孤立顶点 ⇒ 两分量
    assert (cE["n_comp"], cE["comp_sizes"], cE["chi"]) == (2, [12, 1], [2, 1])
    assert (cE["boundary_edges"], cE["sum6deg"], cE["holes"]) == (0, 18, 3)
    assert closure_series([]) == []

    # ---- 空输入边界 ----
    empty = RotNet({})
    ce = concentration_series([empty])[0]
    assert ce["V"] == 0 and ce["max_deg"] == 0 and ce["gini"] == 0.0
    assert ce["n_over"] == ce["n_flat"] == ce["n_under"] == 0
    assert concentration_series([empty], cap=12)[0]["n_sat"] == 0
    assert event_account([]) == [] and event_account([ic]) == []

    # ---- 采纳轨迹读数：V 轨迹与 §7.4 冻结值逐位一致 ----
    assert out["V"] == [12, 32, 36, 66, 82, 98, 154, 226, 432, 558, 960]

    # ---- ① 净湮灭：核心区净湮灭、背景区净创生 ----
    assert prof["rho"]["core"] > 0 > prof["rho"]["bg"]      # core 净湮灭 / bg 净创生
    assert prof["P"]["core"] < prof["P"]["core"] + prof["P"]["sh"]
    # t=2 帧：30 条断边全落在 core（帧初 12 个 deg=10 枢纽互断）
    a2 = acc[1]
    assert a2["n_minus"] == {"core": 30, "sh": 0, "bg": 0}
    assert a2["n_plus"]["sh"] == 4

    # ---- 结构闸门：创生位点含饱和角者恒 0（cap 是承重结构，§7.4 X1）----
    assert all(r["viol_sat"] == 0 for r in acc)

    # ---- 口径二：V⁻ 分区是规则强制（dense 判据两端过密 ⇒ sh/bg 无断边）----
    assert all(r["n_minus"]["sh"] == 0 and r["n_minus"]["bg"] == 0 for r in acc)
    # 而 V⁺ 分区是读数：cap=64 时 V⁺core 自 t=7 起非零 ⇒ 核心净湮灭不是恒等式
    coreb = m.L0Core(RotNet(SEEDS["icosa"]()), cap=64, dmin=3,
                     vminus="dense", vplus="any")
    ob = observe(coreb, nframes=7)
    assert sum(r["n_plus"]["core"] for r in ob["accounts"]) > 0
    assert ob["profile"]["rho"]["core"] > 0

    # ---- 逐帧恒等式：ΔE = Σ_{新元胞} deg_cur − (Σ n_minus + n_follow) ----
    for i, r in enumerate(acc):
        prev, cur = out["nets"][i], out["nets"][i + 1]
        new = [v for v in cur.ids() if v not in prev.rot]
        assert cur.E() - prev.E() == (sum(cur.deg(v) for v in new)
                                      - sum(r["n_minus"].values()) - r["n_follow"])
        assert sum(r["n_plus"].values()) == len(new)

    # ---- ③ 不完美：帧末残留稀疏（采纳动力学下大多被同帧创生救回）----
    imp = out["imperfection"]
    assert sum(x["n_under_end"] for x in imp) == 1          # 10 帧仅 t=9 残留 1 例
    assert any(x["a9"] for x in imp[1:]) or all(
        (x["n_broken"] or 0) * (x["n_pruned"] or 0) == 0 for x in imp[1:])
    # ---- 边缘臂双口径（第④项）：残留臂退化、泛化臂恒非零 ----
    assert close(imp[9]["delta_dmin"], 1.0 / 558.0, 1e-12)  # t=9 唯一非零
    assert all(x["delta_dmin"] == 0.0 for x in imp if x["t"] != 9)
    assert close(imp[5]["delta_kappa"], 0.5102, 1e-3)       # 泛化臂最小帧 t=5
    assert all(x["delta_kappa"] > 0.5 for x in imp[1:])     # 恒非零 ⇒ 持续承载

    # ---- ⑤ 结构闭合：全程单一闭曲面 ⇒ 第③项「内部与外部」在分量层面退化 ----
    clo = out["closure"]
    assert len(clo) == 11 and [x["frame"] for x in clo] == list(range(11))
    assert all(x["n_comp"] == 1 and x["boundary_edges"] == 0 and x["chi"] == [2]
               for x in clo)
    assert all(x["sum6deg"] == 12 + 2 * x["holes"] for x in clo)   # χ=2 恒等式
    assert clo[0]["crystallites"] == [12] and all(
        x["crystallites"] == [] for x in clo[1:])                  # 5-正则仅种子
    assert (clo[2]["sum6deg"], clo[2]["holes"]) == (72, 30)        # 轨迹点 t=2

    # ---- 确定性：两次运行逐位一致 ----
    core2 = m.L0Core(RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                     vminus="dense", vplus="any")
    out2 = observe(core2, nframes=10)
    assert out2["accounts"] == acc and out2["V"] == out["V"]
    assert out2["concentration"] == out["concentration"]
    assert out2["closure"] == clo

    # ---- 导出往返 ----
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cp, jp = os.path.join(d, "a.csv"), os.path.join(d, "o.json")
        flat = flatten_accounts(acc)
        export_csv(cp, flat, ACC_COLS)
        export_json(jp, {"profile": prof, "concentration": out["concentration"]})
        with open(cp, encoding="utf-8") as f:
            got = list(csv.reader(f))
        assert got[0] == list(ACC_COLS) and len(got) == len(flat) + 1
        with open(jp, encoding="utf-8") as f:
            obj = json.load(f)
        assert obj["profile"]["n_frames"] == 10

    # ---- 对照：patch 种子亦复现「核心净湮灭 / 背景净创生」----
    corep = m.L0Core(RotNet(SEEDS["patch"]()), cap=12, dmin=3,
                     vminus="dense", vplus="any")
    outp = observe(corep, nframes=10)
    pp = outp["profile"]
    assert pp["rho"]["core"] > 0 > pp["rho"]["bg"]
    assert all(r["viol_sat"] == 0 for r in outp["accounts"])
    assert all(x["n_comp"] == 1 and x["boundary_edges"] == 0 and x["chi"] == [2]
               and x["sum6deg"] == 12 + 2 * x["holes"] for x in outp["closure"])

    print("l4_evolution selftest: 全部通过")
    print("  夹具 A–G：分区 / 压区(含空表) / 首帧事件账 / 净湮灭画像 / 集中首行 / "
          "不完美双口径(含空图) / 结构闭合五例")
    print(f"  净湮灭画像（icosa cap=12 dense/any 10 帧）："
          f"rho_core=+{prof['rho']['core']:.3f}（净湮灭） "
          f"rho_bg={prof['rho']['bg']:+.3f}（净创生） "
          f"rho_sh={prof['rho']['sh']:+.3f}")
    print(f"  闸门读数：viol_sat ≡ 0（{len(acc)} 帧）；核心净 ΔE={prof['dE']['core']} "
          f"背景净 ΔE=+{prof['dE']['bg']}")
    print(f"  口径二：V⁻sh ≡ V⁻bg ≡ 0（规则强制，rho_bg=−1 是恒等式）；"
          f"cap=64 时 V⁺core={sum(r['n_plus']['core'] for r in ob['accounts'])}>0 "
          f"⇒ 核心净湮灭非恒等式")
    print("  恒等式：ΔE = Σ_新元胞 deg − (Σ断边 + 随删点消失) 逐帧成立")
    print(f"  结构闭合：n_comp≡1 / boundary_edges≡0 / χ≡2（全程，icosa+patch）"
          f"⇒「闭合子图内部与外部」分量层面退化（归并到 core/bg 分区）")
    print(f"  闭合恒等式：Σ(6−deg)=12+2·holes 逐帧成立（t=10: {clo[-1]['sum6deg']}"
          f" = 12+2×{clo[-1]['holes']}）")
    print(f"  不完美定理（第④项）：边缘臂 δ_dmin 仅 t=9 非零（=1/558，被同帧创生掩盖）、"
          f"δ_κ∈[{min(x['delta_kappa'] for x in imp[1:]):.4f},"
          f"{max(x['delta_kappa'] for x in imp[1:]):.4f}] 恒非零；"
          f"中心臂 Δμ 见 L3 ⇒ 双层 AND 判据持续不满足")


def _demo(out):
    prof = out["profile"]
    print("  t     V      E  maxd nSat nOver |  V+core V+sh V+bg |  V-core V-sh V-bg |"
          " 净core 净sh 净bg | follow viol")
    conc = out["concentration"]
    for r, c in zip(out["accounts"], conc[1:]):
        p, mm = r["n_plus"], r["n_minus"]
        print(f"  {r['t']:<4} {c['V']:>4} {c['E']:>6} {c['max_deg']:>5} {str(c['n_sat']):>4} "
              f"{c['n_over']:>5} | {p['core']:>6} {p['sh']:>4} {p['bg']:>4} | "
              f"{mm['core']:>7} {mm['sh']:>4} {mm['bg']:>4} | "
              f"{r['net']['core']:>6} {r['net']['sh']:>4} {r['net']['bg']:>4} | "
              f"{r['n_follow']:>6} {str(r['viol_sat']):>4}")
    print(f"  净湮灭画像：rho_core={prof['rho']['core']:+.4f}  rho_sh={prof['rho']['sh']:+.4f}  "
          f"rho_bg={prof['rho']['bg']:+.4f}   （>0 净湮灭 / <0 净创生）")
    print(f"  净 ΔE：core={prof['dE']['core']}  sh={prof['dE']['sh']}  bg={prof['dE']['bg']}")
    imp = out["imperfection"]
    print("  不完美（边缘臂双口径）：δ_dmin 非零帧 "
          + (",".join(f"t{x['t']}={x['delta_dmin']:.6f}" for x in imp
                      if x["delta_dmin"]) or "无")
          + "（残留臂几乎恒 0）| δ_κ∈["
          + f"{min(x['delta_kappa'] for x in imp[1:]):.4f},"
          + f"{max(x['delta_kappa'] for x in imp[1:]):.4f}]（泛化臂恒非零）")
    clo = out["closure"]
    ok = all(x["n_comp"] == 1 and x["boundary_edges"] == 0 and x["chi"] == [2]
             and x["sum6deg"] == 12 + 2 * x["holes"] for x in clo)
    print("  结构闭合：n_comp=" + ",".join(str(x["n_comp"]) for x in clo)
          + " | boundary_edges=" + ",".join(str(x["boundary_edges"]) for x in clo)
          + " | χ=" + ",".join(str(x["chi"][0]) if len(x["chi"]) == 1
                               else f"n={x['n_comp']}" for x in clo)
          + f" ⇒ {'恒为单一闭曲面（内/外在分量层面退化）' if ok else '出现分片/边界'}")


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
    nframes = int(kv.get("nframes", 10))
    kw = dict(cap=int(kv.get("cap", 12)), dmin=int(kv.get("dmin", 3)),
              drive=kv.get("drive", "slack"), pairing=kv.get("pairing", "none"),
              vminus=kv.get("vminus", "dense"), vplus=kv.get("vplus", "any"))
    core = m.L0Core(m.RotNet(SEEDS[seed]()), **kw)
    out = observe(core, nframes, kappa=int(kv.get("kappa", KAPPA)))
    print(f"l4_evolution · seed={seed} nframes={nframes} " +
          " ".join(f"{a}={b}" for a, b in kw.items()))
    _demo(out)
    if kv.get("csv"):
        export_csv(kv["csv"], flatten_accounts(out["accounts"]), ACC_COLS)
        print(f"  → CSV: {kv['csv']}")
    if kv.get("json"):
        export_json(kv["json"], {"profile": out["profile"],
                                 "concentration": out["concentration"],
                                 "imperfection": out["imperfection"],
                                 "closure": out["closure"]})
        print(f"  → JSON: {kv['json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
