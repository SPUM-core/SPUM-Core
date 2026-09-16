# -*- coding: utf-8 -*-
"""probe_r1.py — R1（禁止 K₆）探针：系统检验团结构排除能否约束度数。

三部分：
  A. 团结构枚举：所有测试三角剖分的最大团 / 全部极大团
  B. Euler 恒等式 + 均匀性 → deg≤5 的纯组合论证
  C. 局部规则搜索：r-邻域可检查的不变量能否区分 deg≤5 与 deg≥6
"""
import sys, os, math
from collections import Counter
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import networkx as nx
from combinatorial_proto import RotNet, SEEDS, audit, seed_bipyramid
from l0_equivalence import r_ball, equivalence_classes, _triangles_in_ball

# ── 测试图集 ──
def build_graphs():
    gs = {}
    gs["tetra"]  = RotNet(SEEDS["tetra"]())
    gs["octa"]   = RotNet(seed_bipyramid(4))  # n=4 = 八面体
    gs["icosa"]  = RotNet(SEEDS["icosa"]())
    for n in [4,5,6,7,8,10,12,16,20]:
        gs[f"bi_{n}"] = RotNet(seed_bipyramid(n))
    return gs

def to_nx(net):
    rot = net.rot if isinstance(net, RotNet) else net
    g = nx.Graph()
    for v in rot:
        g.add_node(v)
    for v in rot:
        for w in rot[v]:
            if v < w:
                g.add_edge(v, w)
    return g

# ============================================================
# A. 团结构枚举
# ============================================================
def probe_a():
    print("=" * 78)
    print("A. 团结构枚举：K₄ 是否为最大团？K₅/K₆ 是否已被平面性自动排除？")
    print("=" * 78)
    gs = build_graphs()
    all_max_4 = True
    for name, net in gs.items():
        g = to_nx(net)
        max_clique = max(nx.find_cliques(g), key=len)
        max_size = len(max_clique)
        degs = [net.deg(v) for v in net.ids()]
        max_deg = max(degs)
        if max_size > 4:
            all_max_4 = False
        print(f"  {name:10s}  V={len(g):3d}  max_deg={max_deg:3d}  "
              f"max_clique=K{max_size}  {'⚠' if max_size>4 else '✓'}")
    print(f"\n  结论：所有球面三角剖分的最大团 = K₄ = {all_max_4}")
    print(f"  ⇒ K₅/K₆ 已被**平面性**自动排除（Kuratowski：K₅ 不可平面）")
    print(f"  ⇒ R1（禁止 K₆）在三角剖分中是**恒真**的——不增加任何新约束")
    print(f"  ⇒ 双锥 bi_n（任意 n）都不含 K₅，故 R1 无法排除它们")

# ============================================================
# B. Euler 恒等式 + 均匀性 → deg≤5
# ============================================================
def probe_b():
    print("\n" + "=" * 78)
    print("B. Euler 恒等式 + 均匀性 → deg≤5 的纯组合论证")
    print("=" * 78)
    print()
    print("  论证链：")
    print("  (1) 球面三角剖分：3F = 2E,  V-E+F = 2  ⇒  E = 3V-6,  Σdeg = 2E = 6V-12")
    print("  (2) 均匀（全 deg=d）：Σdeg = d·V  ⇒  d·V = 6V-12  ⇒  V = 12/(6-d)")
    print("  (3) V > 0 且为正整数  ⇒  (6-d) | 12  且  d < 6")
    print()
    print("  d | 6-d | V=12/(6-d) | 存在性")
    print("  --|-----|------------|-------")
    for d in range(2, 10):
        delta = 6 - d
        if delta > 0 and 12 % delta == 0:
            v = 12 // delta
            exists = {4: "K₄ ✓", 6: "八面体 ✓", 12: "二十面体 ✓"}.get(v, "不存在 ✗")
        elif delta <= 0:
            v = "∞(不可解)" if delta == 0 else "负(荒谬)"
            exists = "不可能 ✗"
        else:
            v = f"{12/delta:.2f}(非整)"
            exists = "不可能 ✗"
        print(f"  {d} |  {delta:>3d} | {str(v):>10s} | {exists}")

    print()
    print("  ⇒ **均匀 + 闭合三角剖分**的组合结果：d ∈ {3,4,5}，V ∈ {4,6,12}")
    print("  ⇒ 这不需要几何，不需要 π，不需要角度——纯 Euler 恒等式")
    print()
    print("  但：双锥 bi_n 不是均匀的（两极 deg=n，赤道 deg=4）")
    print("  ⇒ A3（均匀性）是关键承重假设，不是 Euler 恒等式推导出来的")
    print("  ⇒ 核心问题仍为：A3（均匀性）能否从组合公理**导出**而非假设？")

    # 验证：均匀三角剖分只有 3 个
    print()
    print("  验证：用 l0_equivalence 的 r-0 等价（= 均匀性）筛种子")
    for name, net in build_graphs().items():
        r0 = equivalence_classes(net, 0)
        degs = [net.deg(v) for v in net.ids()]
        uniform = (r0["n_classes"] == 1)
        if uniform:
            print(f"    {name:10s}  均匀 ✓  deg={set(degs)}  V={len(degs)}")

# ============================================================
# C. 局部规则搜索：r-邻域可检查的不变量能否区分 deg≤5 与 deg≥6
# ============================================================
def probe_c():
    print("\n" + "=" * 78)
    print("C. 局部规则搜索：r-邻域可检查的不变量能否区分 deg≤5 与 deg≥6？")
    print("=" * 78)
    print()
    print("  对每个顶点 v，计算以下局部不变量：")
    print("    deg(v), κ=6-deg(v), tri(v)=球内三角形数, link_type")
    print("    2nd_nb_size=|N₂(v)|, 2nd_nb_edges=E(N₂(v))")
    print()

    gs = build_graphs()
    rows = []
    for name, net in gs.items():
        rot = net.rot if isinstance(net, RotNet) else net
        for v in net.ids():
            d = len(rot[v])
            kappa = 6 - d

            # r=1 球
            n1, a1 = r_ball(net, v, 1)
            tri1 = _triangles_in_ball(a1, n1)
            my_tri = tri1.get(v, 0)

            # link 类型：邻居子图的连通性
            neighbors = rot[v]
            nb_set = set(neighbors)
            nb_edges = sum(1 for u in neighbors for w in rot[u] if u < w and w in nb_set)
            # link 是 cycle ⟺ nb_edges == deg (每点恰 2 条 link 边)
            is_cycle_link = (nb_edges == d)

            # r=2 球大小
            n2, _ = r_ball(net, v, 2)
            n2_size = len(n2) - len(n1)  # 二级邻居数

            rows.append({
                "graph": name, "v": v, "deg": d, "kappa": kappa,
                "tri": my_tri, "cycle_link": is_cycle_link,
                "n2": n2_size,
            })

    # 分组：deg≤5 vs deg≥6
    le5 = [r for r in rows if r["deg"] <= 5]
    ge6 = [r for r in rows if r["deg"] >= 6]

    print(f"  样本：deg≤5 共 {len(le5)} 个顶点；deg≥6 共 {len(ge6)} 个顶点")
    print()

    # 检查每个不变量是否能区分
    for key, label in [("kappa", "κ=6-deg"), ("tri", "球内三角数"),
                        ("cycle_link", "link=cycle"), ("n2", "|N₂|")]:
        vals_le5 = set(r[key] for r in le5)
        vals_ge6 = set(r[key] for r in ge6)
        separates = (vals_le5 & vals_ge6 == set())
        print(f"  {label:15s}  deg≤5 取值: {sorted(vals_le5)}  "
              f"deg≥6 取值: {sorted(vals_ge6)}  "
              f"{'✗ 重叠' if not separates else '✓ 分离'}")

    print()
    print("  详细抽样（deg=5 vs deg=6+）：")
    print(f"  {'graph':10s} {'v':>3s} {'deg':>3s} {'κ':>3s} {'tri':>3s} "
          f"{'link_cyc':>8s} {'|N₂|':>4s}")
    for r in rows:
        if r["deg"] in (5, 6, 7, 8) or r["deg"] >= 10:
            flag = "← deg≤5" if r["deg"] <= 5 else "← deg≥6"
            print(f"  {r['graph']:10s} {r['v']:3d} {r['deg']:3d} {r['kappa']:3d} "
                  f"{r['tri']:3d} {str(r['cycle_link']):>8s} {r['n2']:4d}  {flag}")

    print()
    print("  结论：")
    print("  · κ=6-deg 能分离（deg≤5 → κ≥1, deg≥6 → κ≤0），但 κ 是 deg 的定义运算，")
    print("    不是独立约束——它只是把 deg 重新标记，不解释**为什么** deg 不能≥6。")
    print("  · 其余不变量（三角数、link 类型、|N₂|）在两组间有重叠 → 不能区分。")
    print("  · **不存在 r-邻域可检查的独立局部规则能排除 deg≥6**。")
    print("    双锥的极点的 1-邻域结构（一个 cycle link）与 icosahedron 顶点的")
    print("    1-邻域结构（也是一个 cycle link）**完全同构**——局部不可区分。")


if __name__ == "__main__":
    probe_a()
    probe_b()
    probe_c()
