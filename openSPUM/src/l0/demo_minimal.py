# -*- coding: utf-8 -*-
"""demo_minimal.py — 三场景最小教学 demo（只读探针，不改任何 L0 规则）。

回应「先有关系与饱和规则，再投影出空间」的三个最小实验。全程只跑
combinatorial_proto 的纯组合旋转系统——没有坐标、没有角度、没有 π：
圆、球、花环都是投影层（L1/L2）的读法，本脚本一个都不计算。

  场景 A  七节点花环   seed_patch(R=1)：7 点；中心 deg6、6 个外围 deg3；
          6 个三角面 + 1 个六边形洞。χ=2 但是**开放圆盘**，不是球。
  场景 B  正二十面体   seed_icosa：12/30/20，全 deg5，Σ(6−deg)=12，
          自身即 12 晶子闭环。没有「第 13 个中心」——1+12 辐射构造在算术上
          就不是闭曲面（闭合三角剖分强制 F=2E/3 ⇒ χ=−1，奇数不可能）。
  场景 C  穿刺→再生    删一个 deg5 点 → 五边形洞（开口 5/16=0.3125 ≤ 1/3）
          → 下一帧洞锥化补入新点（id=12，不是 0）→ 回到 12/30/20、晶子12=1
          → 再下一帧 V⁺=0：饱和壳**锁定停机**，而不是被删除。

用法：
  python demo_minimal.py selftest=1
"""

import sys

from combinatorial_proto import (RotNet, audit, print_row, run_frame,
                                 seed_icosa, seed_patch)


# ============================================================
# 断言小工具
# ============================================================
def report(checks, ok, notes=()):
    """checks: [(名称, 实测, 期望), ...]；notes: 教学注记（不断言）。"""
    for name, got, want in checks:
        flag = "OK" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{flag}] {name}: {got!r}（期望 {want!r}）")
    for n in notes:
        print(f"  注：{n}")
    return ok


# ============================================================
# 场景 A：七节点花环（二维七圆花环）
# ============================================================
def scene_a(ok):
    print("-" * 78)
    print("场景 A：七节点花环 seed_patch(R=1) —— C=6 只能活在开放边界里")
    net = RotNet(seed_patch(1))
    a = audit(net)
    print_row("A0 ", a)
    checks = [
        ("V,E,F = 7,12,7（6 三角面 + 1 六边形洞面）",
         (a["V"], a["E"], a["F"]), (7, 12, 7)),
        ("χ=2（圆盘也是 2 —— χ=2 不蕴含闭合）", a["chi"], 2),
        ("六边形洞", a["max_hole"], 6),
        ("边界点 link=path", a["path"], 6),
        ("度分布：中心 deg6 / 外围 deg3", a["hist"], {3: 6, 6: 1}),
        ("Σ(6−deg)=18（开放盘，不是闭合壳的 12）", a["S"], 18),
        ("d=3V−6−E=3（未饱和到平面极限）", a["d3v6"], 3),
    ]
    notes = [
        "规则层没有「二维/三维」开关。全 deg6 的有限闭合三角剖分不存在：",
        "Σ(6−6)·V=0 ≠ 12。所以 C=6 花环必然挂着一个六边形边界洞——",
        "「六角密铺」是开放贴片；能闭合的均匀解在 C=5（见场景 B）。",
    ]
    return report(checks, ok, notes)


# ============================================================
# 场景 B：正二十面体（12 晶子闭环，无中心）
# ============================================================
def scene_b(ok):
    print("-" * 78)
    print("场景 B：正二十面体 seed_icosa —— 12 个点就是壳本身，没有中心")
    net = RotNet(seed_icosa())
    a = audit(net)
    print_row("B0 ", a)
    n_ic = sum(1 for c in a["cryst"] if c["icosa"])

    # 反例：「1 中心 + 12 外围，中心连全部外围」是否可能？
    #   V = 13；E = 30（壳边）+ 12（辐条）= 42
    #   若仍是闭合三角剖分，3F=2E 强制 F=28 ⇒ χ = 13−42+28 = −1
    #   闭合曲面 χ=2−2g 必为偶数 ⇒ −1 不可能；
    #   且 30 条壳边各自已属于 2 个壳面，任何 (hub,i,j) 面都让它属于第 3 个面。
    V_h, E_h = 13, 30 + 12
    F_h = 2 * E_h // 3
    chi_h = V_h - E_h + F_h

    checks = [
        ("V,E,F", (a["V"], a["E"], a["F"]), (12, 30, 20)),
        ("χ=2 且无洞（闭合球面）", (a["chi"], a["n_holes"]), (2, 0)),
        ("全体 deg5", a["hist"], {5: 12}),
        ("Σ(6−deg)=12", a["S"], 12),
        ("12 点 5-正则晶子闭环", n_ic, 1),
        ("1+12 辐射构造的 χ", chi_h, -1),
        ("χ 为奇数 ⇒ 不可能是闭合曲面", bool(chi_h % 2), True),
    ]
    notes = [
        "闭合 + 全 deg5 ⇒ V=12/(6−5)=12，唯一整数解；12 点互为壳，无中心。",
        "三角剖分中每点关联的三角面数恰等于度数——「数面饱和」就是度数容量，",
        "不需要第二套计数器；r(deg)∝√deg，deg12 的假想中心只会更大不会更小。",
        f"加第 13 点连全部 12 点：E={E_h}，闭合剖分强制 F={F_h}，χ={chi_h}，",
        "且 30 条壳边会从「属 2 面」变成「属 3 面」，违反边-面规则。",
    ]
    return report(checks, ok, notes)


# ============================================================
# 场景 C：穿刺 → 洞锥化再生 → 壳锁定
# ============================================================
def scene_c(ok):
    print("-" * 78)
    print("场景 C：穿刺→再生 —— 删一个壳点，下一帧新点补洞，再下一帧停机")
    net = RotNet(seed_icosa())
    v = max(net.ids(), key=lambda x: (net.deg(x), -x))  # 全 deg5 ⇒ 字典序最小 = 0
    d0 = net.deg(v)
    net.remove_vertex(v)
    a0 = audit(net)
    print_row("C- ", a0)

    born1, dead1 = run_frame(net, mode="hole", dmin=3)
    a1 = audit(net)
    r1 = dict(a1)
    r1["born"], r1["dead"] = len(born1), len(dead1)
    print_row("Cf1", r1)

    born2, dead2 = run_frame(net, mode="hole", dmin=3)
    a2 = audit(net)
    r2 = dict(a2)
    r2["born"], r2["dead"] = len(born2), len(dead2)
    print_row("Cf2", r2)

    n_ic1 = sum(1 for c in a1["cryst"] if c["icosa"])
    checks = [
        ("穿刺点度数", d0, 5),
        ("穿刺后 V,E,F", (a0["V"], a0["E"], a0["F"]), (11, 25, 16)),
        ("五边形洞", a0["max_hole"], 5),
        ("边界点 link=path", a0["path"], 5),
        ("Σ(6−deg)=16（被刺破的壳）", a0["S"], 16),
        ("开口占比 5/16=0.3125 ≤ 1/3", round(a0["open_ratio"], 4), 0.3125),
        ("f1 补入新点 id（新实体，不是旧 id 0）", born1, [12]),
        ("f1 无修剪", dead1, []),
        ("f1 回到 12/30/20", (a1["V"], a1["E"], a1["F"]), (12, 30, 20)),
        ("f1 晶子12 重新计数为 1", n_ic1, 1),
        ("f2 无创生无删除（饱和壳锁定停机）", (born2, dead2), ([], [])),
        ("旧 id 0 已不存在（id = 存在的时间延伸）", 0 in net.ids(), False),
    ]
    notes = [
        "闭合壳上唯一能制造开口的操作是 V⁻ 删点（创生类操作 Δχ=0 且不造洞）。",
        "补洞新点 id=12：组合槽位与旧点相同，但实体是新的——帧间同一性靠 id 锚定。",
        "饱和在 SPUM 里是「锁定/创生抑制」（无洞可锥化 ⇒ V⁺=0），不是把节点删掉。",
    ]
    return report(checks, ok, notes)


# ============================================================
# 入口
# ============================================================
def selftest():
    print("=" * 78)
    print("[demo_minimal] 三场景最小教学 demo（纯组合旋转系统，零坐标零角度）")
    print("=" * 78)
    ok = True
    ok = scene_a(ok)
    ok = scene_b(ok)
    ok = scene_c(ok)
    print("=" * 78)
    print(f"三场景断言总体: {'全部通过' if ok else '存在失败项'}")
    print("说明：圆、球、花环、角亏都不出现在规则里——它们是 L1/L2 投影层的读法。")
    return ok


def main():
    kv = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v
    ok = selftest()
    if kv.get("selftest", "0") == "1":
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
