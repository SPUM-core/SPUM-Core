# -*- coding: utf-8 -*-
"""l0_count.py — 计数投影算子 Π_c：把「数」显式拆成 **预设** 与 **读数** 两级，禁止混列。

架构基准: docs/L0L1L2_架构设计.md §6.2「计数链审计」

要解决什么（§6.2 的口径不自洽）
------------------------------
§6.2 原文：「以下数字**全部从图中数出来，不预设**」。但其中 12 的推出实际依赖
**两个预设**：`χ=2`（子图闭合）与「全 `deg=5`」（均匀饱和）——**审计口径 ≠ 推导口径**。
更露骨的是**接吻数 12**：§6.2 把「组合 12」与「几何 12」当成同一个数使用，而
「组合 12 → 几何 12」这一步的**投影算子没有落地件**。

修法（最小形式定义）
--------------------
把预设作为**显式入参**、读数作为**返回值**，两级分离：

    Π_c : (G, obj_class) × presets → {value, preset, preset_ok, kind}

  · **预设不成立 ⇒ `value=None`、`kind="undefined"`**（不是 0，也不是"碰巧等于 12"），
    并报出**是哪个预设**挡住的（`preset`）⇒ 投影步可审计、可复算。
  · `kind="readout"` 的数值，**只在所声明的预设下**成立；脱离该预设即为越界。

由此，同一个「12」被强制拆成三档（层级标注规范）：

| 档 | 内容 | 例 |
|---|---|---|
| **预设** | L0 公理 / 操作假设，**不是数出来的** | `χ=2`（闭合）、`tri_only`（无洞，即真三角剖分）、边-面规则、均匀饱和（全 `deg=C`） |
| **读数** | 对预设对象执行计数操作得到 | `count(G,"V",closed_chi2=True,uniform_C=5)=12`；`30/20/42/50/62` |
| **越界（投影）** | 需 L0 之外的投影算子才得到 | 三维接吻数 12、`60°`、`16π`（见 `BOUNDARY`） |

数论入口就在这条缝里：闭合 + 均匀饱和 ⇒ `N(6−C)=12`，即 **(6−C) | 12** ⇒ `C∈{3,4,5}`
⇒ `N∈{4,6,12}`（见 `forced_N`）。**决定正则三角剖分解的是 12 的约数表，不是 12 本身**；
全程不碰 π / 角度 / 半径。

容量内生性判定（§4b，2026-09-14）
--------------------------------
上表把「均匀饱和 C」当**预设**用。但 §7.2 实测结论是「**cap 是承重结构，不是内存
参数**」——容量是承重的。那它从哪来？`no_intrinsic_capacity` 给判定：

  纯组合 L0 **不可能**内生容量。见证族 = 双锥 n（`seed_bipyramid`）：对任意 n ≥ 4
  都合法（χ=2、无洞、link 全 cycle）且 **E = 3V−6**（极大平面图 —— 平面性推到饱和
  极限），却含 deg = n 的顶点 ⇒ **组合层没有「太满」这个状态**。
  ⇒ 容量不是组合量；要它必须外接**排他性**（角半径 / 球影，附录 A §A.3），或声明为
  来自投影层的外部输入。这正是「cap 承重」的根源 —— 承重，因为它**不在**组合层里。

用法
----
  python l0_count.py selftest=1
  python l0_count.py seed=icosa obj=V closed=1 tri=1 uniform=5
  python l0_count.py seed=patch obj=sum6deg closed=1 tri=1
  python l0_count.py table=1
  python l0_count.py capacity=1 [D=64]      # 容量内生性判定（双锥见证族压力测试）
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from combinatorial_proto import (  # noqa: E402
    RotNet, SEEDS, audit, seed_bipyramid,
)

# 可计数的对象类（读数侧）
OBJ_CLASSES = ("V", "E", "F", "deg_hist", "sum6deg", "holes", "n_shells")

# 越界（投影）项：在 L0 内**没有**算子，须有 L0 之外的投影层才谈得上。
# 如实登记，防止它们与「读数」混列 —— 这正是 §6.2 与 knowledge.md 的病根。
BOUNDARY = (
    ("接吻数 12", "三维等大球几何容量", "L0 内无算子：组合 12 → 几何 12 的投影层未落地"),
    ("60°", "K₃ 内角（180°/3）", "角度投影：L0 只记「三边两两相邻」"),
    ("16π", "中心球面可辨区域数", "π 是认知压缩因子，非 L0 常数"),
    ("关系容量（deg 上界）", "排他性占据（球面容量 / 角半径 r(deg)）",
     "L0 内**不可能**有算子：组合层对任意 deg 都有合法见证（双锥族，"
     "见 no_intrinsic_capacity）—— 故 cap 只能外接，不能内生。"
     "**L0-B14 加固**（2026-09-15，`probe_r1.py`）：不仅见证族存在，"
     "且 r-邻域可检查的局部不变量（κ=6−deg、球内三角数、link 类型、"
     "|N₂|）系统性地无法区分 deg≤5 与 deg≥6——双锥极点与正二十面体"
     "顶点的 1-邻域结构同构。唯一路径：A3（均匀性，全局假设）+ "
     "Euler 恒等式 ⇒ (6−d)|12 ⇒ d∈{3,4,5}；无 A3 则只约束均值。"),
)


# ============================================================
# §1 预设门控：预设不成立 ⇒ 读数 undefined
# ============================================================
def _undef(preset):
    """预设不成立 → 明文 undefined（**不是** 0，也不是"碰巧的数"）。"""
    return {"value": None, "preset": preset, "preset_ok": False, "kind": "undefined"}


def _gated(G, a, presets):
    """按固定顺序核验预设 → `(ok, 挡住的名字, 已应用的名字列表)`。

    · `closed_chi2=bool`：要求 `χ==2` 与否，**无默认值**（不传 = 不设该预设）；
    · `tri_only=bool` ：要求**无非三角面**（即无洞）与否。⚠️ **`χ=2` 不含"无边界"**——
      开放 patch 的外边界在旋转系统里被一个 12 边形**面**包住，故 `χ` 仍为 2；
      "无边界"在旋转系统里表现为**所有面皆为三角**。C1 的 12 需要 `closed_chi2 ∧ tri_only`。
    · `uniform_C=C`   ：要求全 `deg==C`；
    · `dmin=k`        ：要求全 `deg>=k`。
    """
    applied = []
    if "closed_chi2" in presets:
        applied.append("closed_chi2")
        if (a["chi"] == 2) != bool(presets["closed_chi2"]):
            return False, "closed_chi2", applied
    if "tri_only" in presets:
        applied.append("tri_only")
        if (a["n_holes"] == 0) != bool(presets["tri_only"]):
            return False, "tri_only", applied
    degs = None
    if "uniform_C" in presets:
        name = f"uniform_C={int(presets['uniform_C'])}"
        applied.append(name)
        degs = degs if degs is not None else [G.deg(v) for v in G.ids()]
        if any(d != int(presets["uniform_C"]) for d in degs):
            return False, name, applied
    if "dmin" in presets:
        name = f"dmin={int(presets['dmin'])}"
        applied.append(name)
        degs = degs if degs is not None else [G.deg(v) for v in G.ids()]
        if any(d < int(presets["dmin"]) for d in degs):
            return False, name, applied
    return True, None, applied


# ============================================================
# §2 读数侧（只读 G，复用 combinatorial_proto.audit，不另写一套点数逻辑）
# ============================================================
def n_shells(G, cap=5):
    """全 `deg == cap` 的**连通分量**数（闭合饱和壳）。

    ⚠️ 是**分量**数，不是「度为 cap 的顶点数」：icosa 得 1（整体一个 5-正则分量），
    两张 icosa 的不交并得 2。
    """
    seen, n = set(), 0
    for s in G.ids():
        if s in seen:
            continue
        comp, q = {s}, [s]
        while q:
            u = q.pop()
            for w in G.rot[u]:
                if w not in comp:
                    comp.add(w)
                    q.append(w)
        seen |= comp
        if comp and all(G.deg(v) == cap for v in comp):
            n += 1
    return n


def _readout(G, a, obj_class):
    """对预设对象执行计数操作 → 数值（纯整数/直方图，无浮点）。"""
    if obj_class == "V":
        return a["V"]
    if obj_class == "E":
        return a["E"]
    if obj_class == "F":
        return a["F"]
    if obj_class == "deg_hist":
        return dict(a["hist"])
    if obj_class == "sum6deg":
        return a["S"]
    if obj_class == "holes":
        return a["n_holes"]
    if obj_class == "n_shells":
        return n_shells(G)
    raise ValueError(f"unknown obj_class: {obj_class}")


# ============================================================
# §3 Π_c —— 计数投影算子（对外唯一入口）
# ============================================================
def count(G, obj_class, **presets):
    """计数投影算子 Π_c。

    Π_c(G, obj_class; presets) → `{"value", "preset", "preset_ok", "kind"}`

    · 预设全部成立：`kind="readout"`，`value` = 读数，`preset` = 已应用的预设串；
    · 有预设不成立：`kind="undefined"`，`value=None`，`preset` = 挡住的那个预设；
    · 不传任何预设：`preset=None`，返回**原样读数**（不声称任何预设下的成立性）。
    """
    if obj_class not in OBJ_CLASSES:
        raise ValueError(f"obj_class 须 ∈ {OBJ_CLASSES}，收到 {obj_class!r}")
    a = audit(G)                                   # 只读一次
    ok, blocked, applied = _gated(G, a, presets)
    if not ok:
        return _undef(blocked)
    return {"value": _readout(G, a, obj_class),
            "preset": "+".join(applied) if applied else None,
            "preset_ok": True, "kind": "readout"}


# ============================================================
# §4 数论入口：整除强制定理（纯算术，不依赖几何）
# ============================================================
def forced_N(C):
    """整除强制定理：闭合（`χ=2`）三角剖分 + 均匀饱和（全 `deg=C`）下，整数节点数 N 被
    `N(6−C) = 12` 强制，即 **(6−C) | 12**。返回 N（int）；无正整数解 → `None`。

    结合三角剖分约束 `C ≥ 3`：`C=3→N=4`（`K₄`）、`C=4→N=6`（八面体图）、`C=5→N=12`
    （二十面体图）。**决定解存在性的是 12 的约数表 {1,2,3,4,6,12}，不是 12 本身。**
    全程不出现 π / 角度 / 半径。
    """
    d = 6 - C
    if C < 3 or d <= 0 or 12 % d != 0:
        return None
    return 12 // d


def forced_table(lo=0, hi=8):
    """`C → N` 对照（None = 无正整数解）。供审计与文档引用。"""
    return {C: forced_N(C) for C in range(lo, hi + 1)}


# ============================================================
# §4b 容量内生性判定：纯组合 L0 **不可能**内生「关系容量」上界
# ============================================================
def no_intrinsic_capacity(D=64):
    """**判定**：纯组合 L0 里不存在「关系容量」（度数上界）的内生判据。

    四段，每段可复算：

      (a) **见证族**：双锥 n（`seed_bipyramid`）对任意 n ≥ 4 合法 ——
          χ=2、无洞、link 全 cycle（无 path/chord/disc）、握手成立，且
          **d = 3V−6−E = 0**（极大平面图：平面性已推到饱和极限）；同时含
          deg = n 的顶点（两极），Σ(6−deg) ≡ 12。
          ⇒ **任意度数都能出现在合法构型里**：组合层没有「太满」这个状态。
          （n=3 是退化特例——三角双锥=两个四面体叠合，赤道三角形是**分离三角**
           ⇒ `chord=3`；故见证族从 n=4 起。n=4 即八面体，全 deg=4。）
      (b) **压力测试**：n = 4..D 逐一过 `audit`，无一被任何**组合判据**拦下。
      (c) **对照**：守恒量 Σ(6−deg)=6χ 是**全局恒等式**，不含单个 deg 的上界；
          §7.4 N1 判决独立记录「cap 只钉局部度数，钉不住 V」。
      (d) **L0-B14 加固**（2026-09-15，`probe_r1.py`）：局部规则类方法
          **系统性地**无法区分 deg≤5 与 deg≥6——四个 r-邻域可检查的不变量
          （κ=6−deg、球内三角数、link 类型、|N₂|）中，κ 与三角数只是 deg
          的定义运算，link 类型恒为 cycle（两组重叠），|N₂| 在两组重叠。
          **双锥极点（deg=6）与正二十面体顶点（deg=5）的 1-邻域结构同构**
          （均为 cycle link），局部不可区分。此外 R1（禁止 K₆）在球面
          三角剖分中恒真（K₅/K₆ 已被 Kuratowski 平面性排除），不增加约束。

    ⇒ 「容量」不是组合量：要它必须外接**排他性**（v1.x 的角半径 r(deg) /
    球影排他，附录 A §A.3），或把它声明为来自投影层的外部输入。
    这正是 §7.2 实测结论「**cap 是承重结构，不是内存参数**」的根源 ——
    承重，恰恰因为它**不在** L0 的组合层里。

    唯一组合路径（非局部规则）：A3（均匀性，**全局**假设）+ Euler 恒等式
    `d·V=6V−12` ⇒ `(6−d)|12` ⇒ `d∈{3,4,5}`；无 A3 则 Euler 只约束均值
    `⟨deg⟩<6`，不约束个体。

    返回 `{"rows", "D", "over", "ok"}`；`over` = 越过三个已知容量候选
    （12 / 42 / 16π≈50.27）后仍合法的见证。
    """
    rows = []
    ok = True
    for n in range(4, D + 1):
        G = RotNet(seed_bipyramid(n))
        a = audit(G)
        degs = [G.deg(v) for v in G.ids()]
        row = {"n": n, "V": a["V"], "E": a["E"], "F": a["F"], "chi": a["chi"],
               "S": a["S"], "d3v6": a["d3v6"], "maxdeg": max(degs),
               "min": min(degs), "holes": a["n_holes"], "cycle": a["cyc"],
               "path": a["path"], "chord": a["chord"], "disc": a["disc"]}
        row["ok"] = (a["chi"] == 2 and a["n_holes"] == 0 and a["path"] == 0
                     and a["chord"] == 0 and a["disc"] == 0
                     and a["d3v6"] == 0 and a["S"] == 12
                     and a["V"] == n + 2 and a["E"] == 3 * n and a["F"] == 2 * n
                     and max(degs) == n and min(degs) == 4)
        ok &= row["ok"]
        rows.append(row)
    # 对照：越过 12（接吻数）/ 42（团簇）/ 16π≈50.27（球面容量）后仍有合法构型
    over = [(name, th, next(r for r in rows if r["n"] > th))
            for name, th in (("接吻数 12", 12), ("42 团簇", 42), ("16π≈50.27", 51))
            if th < D]
    return {"rows": rows, "D": D, "over": over, "ok": ok}


# ============================================================
# §5 控制实验 + 自检
# ============================================================
def verify():
    """夹具断言：正向读数 + **反向控制**（预设不成立必须 undefined）。"""
    # ① K₄：闭合 + 无洞（真三角剖分）⇒ Σ(6−deg) = 12（与图论 C1 一致）
    k4 = RotNet(SEEDS["tetra"]())
    r = count(k4, "sum6deg", closed_chi2=True, tri_only=True)
    assert r["kind"] == "readout" and r["value"] == 12, r

    # ② icosa：闭合 + 无洞 + 均匀饱和 C=5 ⇒ V=12 / E=30 / F=20，且是 1 个饱和壳
    ic = RotNet(SEEDS["icosa"]())
    for obj, want in (("V", 12), ("E", 30), ("F", 20), ("sum6deg", 12)):
        r = count(ic, obj, closed_chi2=True, tri_only=True, uniform_C=5)
        assert r["kind"] == "readout" and r["value"] == want, (obj, r)
    r = count(ic, "n_shells")
    assert r["kind"] == "readout" and r["value"] == 1, r

    # ③ **反向控制 a（分离"闭合"与"无边界"）**：开放 patch 的 `χ` 也是 2（外边界是一个
    #    12 边形**面**），单靠 closed_chi2 **拦不住**它 —— 必须叠加 tri_only 才 undefined。
    pt = RotNet(SEEDS["patch"]())
    r_chi = count(pt, "sum6deg", closed_chi2=True)
    assert r_chi["kind"] == "readout" and r_chi["preset_ok"], r_chi   # χ=2 成立，读数为 30
    r = count(pt, "sum6deg", closed_chi2=True, tri_only=True)
    assert r["kind"] == "undefined" and r["value"] is None, r
    assert r["preset"] == "tri_only" and not r["preset_ok"], r

    # ④ **反向控制 b**：icosa 不是 4-正则 ⇒ 均匀预设不成立 ⇒ undefined
    r = count(ic, "V", uniform_C=4)
    assert r["kind"] == "undefined" and r["preset"] == "uniform_C=4", r

    # ⑤ 无预设 ⇒ 原样读数（不声称任何预设下的成立性）
    r = count(ic, "V")
    assert r["kind"] == "readout" and r["value"] == 12 and r["preset"] is None, r

    # ⑥ 数论入口：整除强制解
    assert [forced_N(c) for c in (3, 4, 5)] == [4, 6, 12]
    assert all(forced_N(c) is None for c in (-1, 0, 1, 2, 6, 7, 12)), forced_table()

    # ⑦ **容量内生性判定**：纯组合 L0 无度数上界（见证族 = 双锥）
    nc = no_intrinsic_capacity(D=64)
    assert nc["ok"], [r for r in nc["rows"] if not r["ok"]][:3]
    last = nc["rows"][-1]
    assert last["n"] == 64 and last["maxdeg"] == 64 and last["d3v6"] == 0, last
    # 守恒量 Σ(6−deg) 全程 ≡ 12 ⇒ 它不含任何 deg 上界
    assert {r["S"] for r in nc["rows"]} == {12}, sorted({r["S"] for r in nc["rows"]})
    # 反向控制：越过 12 / 42 / 16π≈50.27 三个候选容量后**仍有**合法构型
    assert len(nc["over"]) == 3, nc["over"]
    for name, th, row in nc["over"]:
        assert row["ok"] and row["maxdeg"] > th, (name, th, row)
    return True


def selftest():
    verify()
    print("l0_count selftest: 全部通过")
    print("  ① 预设门控：闭合 + 无洞 + 均匀饱和 C=5 ⇒ 读数 V=12 / E=30 / F=20，饱和壳 n_shells=1")
    print("  ② 反向控制 a：开放 patch 的 χ 也是 2（外边界是一个 12 边形**面**），"
          "closed_chi2 拦不住 ⇒ 叠加 tri_only 才 undefined")
    print("  ③ 反向控制 b：icosa + uniform_C=4 ⇒ undefined（预设不成立 ≢ 读数）")
    print("  ④ 无预设：原样读数，preset=None（不声称预设下的成立性）")
    print("  ⑤ 数论入口 forced_N：C=3,4,5 → 4,6,12；(6−C)∤12 或 C<3 一律 None")
    print("  ⑥ 越界（投影）项——L0 内无算子：" + "；".join(
        f"{n}（{why}）" for n, _, why in BOUNDARY))
    print("  ⑦ 容量内生性判定：双锥见证族 n=4..64 全部合法（χ=2、无洞、E=3V−6 极大平面）"
          "⇒ 组合层无 deg 上界；越过 12/42/16π≈50.27 后仍合法")
    return True


# ============================================================
# §6 CLI
# ============================================================
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
    if "capacity" in kv:
        D = int(kv.get("D", 64))
        nc = no_intrinsic_capacity(D)
        print("=" * 78)
        print(f"[容量内生性判定] 纯组合 L0 有度数上界吗？见证族 = 双锥 n（n = 4..{D}）")
        print("=" * 78)
        show = [n for n in (4, 5, 12, 13, 42, 43, 51, D) if 4 <= n <= D]
        for n in show:
            r = nc["rows"][n - 4]
            print(f"  双锥 n={r['n']:<3} V={r['V']:3d} E={r['E']:4d} F={r['F']:3d} "
                  f"χ={r['chi']} Σ(6−deg)={r['S']:2d} 3V−6−E={r['d3v6']} "
                  f"*maxdeg={r['maxdeg']:3d} 洞={r['holes']} "
                  f"link[c{r['cycle']}/p{r['path']}/h{r['chord']}/d{r['disc']}] "
                  f"[{'OK' if r['ok'] else 'FAIL'}]")
        print("-" * 78)
        print("  全部合法：χ=2、无洞、link 全 cycle、**E = 3V−6（极大平面图）** ——")
        print("  平面性已推到饱和极限，度数仍无上界。")
        for name, th, row in nc["over"]:
            print(f"  越过 {name}（={th}）：双锥 n={row['n']} 仍合法 "
                  f"（maxdeg={row['maxdeg']}）⇒ 该数不是组合上界")
        print(f"  守恒量 Σ(6−deg) 全程 ≡ 12 ⇒ 含上界的是它吗：不是（全局恒等式）。")
        print("-" * 78)
        print("  ▶ 判定：组合层没有「太满」这个状态 ⇒ **容量不是组合量**。")
        print("     要它必须外接**排他性**（角半径 / 球影，附录 A §A.3），"
              "或声明为来自投影层的外部输入。")
        print("     这就是 §7.2「cap 是承重结构，不是内存参数」的根源。")
        print(f"  [{('OK' if nc['ok'] else 'FAIL')}]")
        return
    if "seed" in kv or "obj" in kv or "table" in kv:
        if "table" in kv:
            print("forced_N(C)（整除强制定理：N(6−C)=12 ⟺ (6−C)|12，C≥3）")
            for C, N in forced_table().items():
                print(f"  C={C:<2} → N={N if N is not None else 'None（无正整数解）'}")
            return
        seed = kv.get("seed", "icosa")
        obj = kv.get("obj", "V")
        pre = {}
        if kv.get("closed") == "1":
            pre["closed_chi2"] = True
        if kv.get("tri") == "1":
            pre["tri_only"] = True
        if "uniform" in kv:
            pre["uniform_C"] = int(kv["uniform"])
        if "dmin" in kv:
            pre["dmin"] = int(kv["dmin"])
        G = RotNet(SEEDS[seed]())
        r = count(G, obj, **pre)
        print(f"seed={seed}  obj={obj}  presets={pre if pre else '（无）'}")
        print(f"  → value={r['value']}  preset={r['preset']}  "
              f"preset_ok={r['preset_ok']}  kind={r['kind']}")
        print("越界（投影）项（L0 内无算子）：")
        for n, what, why in BOUNDARY:
            print(f"  · {n}（{what}）—— {why}")
        return
    print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
