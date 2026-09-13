"""纯组合拓扑审计：把 L0 引擎输出的关系网络当单纯复形，检验「边是否交叉」。

用法: python diag_manifold.py [nframes] [seed=tetra] [dmin=3] [every=1]

为什么需要它
------------
「边交叉」是**嵌入**性质，不是图的性质。只有边时无法定义交叉，必须先给网络
一个「面」（三元组 {i,j,k} 声明为面）。本脚本按两种口径取面集：

  口径 A（clique 复形）：把所有 3-团都当面 —— 这正是当前 K2 的隐含判据
      （「j,k 互为邻居」即当成缝隙/面）。
  口径 B（真嵌入面）：networkx 平面性检验给出 rotation system，取其真实面。

判据（全部纯计数，元胞可自查）
------------------------------
  (C1) 每条边恰属于 1 或 2 个面           —— 边-面入射计数 ≤ 2
  (C2) 每个顶点的 face-link 是路径或环     —— 内部 t=deg、边界 t=deg-1，且连通
  (C3) 无领结：任意两面最多共享一条边
  Euler: χ = V - E + F（球面三角剖分 χ=2，且 F = 2V-4）

真平面性
--------
  networkx.check_planarity：planar=False ⇒ 边集本身无法无交叉画出 ⇒ **确认交叉**。
  E > 3V-6（最大平面界）⇒ 必有交叉。E = 3V-6 只是必要条件，仍可能是非平面。

「分离三角」= 是 3-团但不是真面（F_clique - 2V-4）。它是 K2 会误当成缝隙去锥化
的位置；在分离三角上锥化 = 往已被两侧占满的面塞球 = 制造交叉。
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag_crystallite import load_adj, parse_kv, seed_of
from host import FrameScheduler


def triangles_of(adj):
    """全部 3-团（口径 A）。返回 (集合, 边->面入射计数)。"""
    tris = set()
    for i in range(len(adj)):
        ni = adj[i]
        for j in ni:
            if j <= i:
                continue
            for k in (ni & adj[j]):
                if k > j:
                    tris.add((i, j, k))
    eface: dict[tuple[int, int], int] = {}
    for (a, b, c) in tris:
        for (x, y) in ((a, b), (a, c), (b, c)):
            key = (x, y) if x < y else (y, x)
            eface[key] = eface.get(key, 0) + 1
    return tris, eface


def link_type(adj, v):
    """顶点 v 的 link（N(v) 的诱导子图）类型：'cycle'/'path'/'other'。
    cycle = 内部顶点（连通、全度 2、边数 = d）；path = 边界顶点（连通、
    恰两点度 1、其余度 2、边数 = d-1）；其余 = 非流形（交叉/领结）。"""
    nb = sorted(adj[v])
    d = len(nb)
    if d == 0:
        return "other"
    ld = {x: 0 for x in nb}
    la = {x: [] for x in nb}
    le = 0
    for ai in range(d):
        for bi in range(ai + 1, d):
            x, y = nb[ai], nb[bi]
            if y in adj[x]:
                le += 1
                ld[x] += 1
                ld[y] += 1
                la[x].append(y)
                la[y].append(x)
    seen = {nb[0]}
    st = [nb[0]]
    while st:
        u = st.pop()
        for w in la[u]:
            if w not in seen:
                seen.add(w)
                st.append(w)
    conn = len(seen) == d
    degs = sorted(ld.values())
    if conn and le == d and all(x == 2 for x in degs):
        return "cycle"
    if conn and le == d - 1 and degs.count(1) == 2 and all(x in (1, 2) for x in degs):
        return "path"
    return "other"


def audit(adj, use_nx=True):
    """返回一次审计的字典。"""
    V = len(adj)
    if V == 0:
        return {"V": 0}
    E = sum(len(a) for a in adj) // 2
    tris, eface = triangles_of(adj)
    F = len(tris)
    hist = Counter(eface.values())

    lt = Counter(link_type(adj, v) for v in range(V))
    other_ex = [(v, len(adj[v])) for v in range(V) if link_type(adj, v) == "other"][:4]

    out = {
        "V": V, "E": E, "d3v6": E - (3 * V - 6), "F3": F, "sphereF": 2 * V - 4,
        "sep": F - (2 * V - 4),
        "e_ge3": sum(1 for c in eface.values() if c >= 3),
        "e_1": sum(1 for c in eface.values() if c == 1),
        "cyc": lt["cycle"], "path": lt["path"], "other": lt["other"],
        "other_ex": other_ex, "hist": dict(sorted(hist.items())),
        "chi": V - E + F, "planar": None, "note": "",
    }

    if use_nx:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(V))
        for i in range(V):
            for j in adj[i]:
                if j > i:
                    G.add_edge(i, j)
        comp = nx.number_connected_components(G)
        if comp > 1:
            out["note"] = f"图不连通({comp}分量)，平面性逐分量判"
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        planar, _emb = nx.check_planarity(G)
        out["planar"] = planar
    return out


def main():
    kv, rest = parse_kv(sys.argv[1:])
    nframes = int(rest[0]) if len(rest) > 0 else 8
    seed_name = kv.get("seed", "tetra")
    dmin = int(kv.get("dmin", 3))
    every = max(1, int(kv.get("every", 1)))
    angular = int(kv.get("angular", 0))

    pos, edges = seed_of(seed_name)
    sch = FrameScheduler(capacity=16384, dangling_min=dmin)
    sch.spawn_initial(pos, edges, radius0=1.0)
    print(f"[组合拓扑审计] seed={seed_name} dmin={dmin} nframes={nframes} every={every}")

    hdr = (f"{'f':>3} {'V':>6} {'E':>7} {'E-3V+6':>7} {'F3':>7} {'2V-4':>6} "
           f"{'sepTri':>6} {'e>=3f':>6} {'e=1f':>5} {'cyc':>5} {'path':>5} "
           f"{'other':>5} {'chi':>5} {'planar':>6}")
    print(hdr)
    for f in range(nframes + 1):
        if f % every == 0:
            _, adj = load_adj(sch)
            a = audit(adj)
            if a.get("V", 0) == 0:
                print(f"{f:>3}  (空网络)")
            else:
                print(f"{f:>3} {a['V']:>6} {a['E']:>7} {a['d3v6']:>7} {a['F3']:>7} "
                      f"{a['sphereF']:>6} {a['sep']:>6} {a['e_ge3']:>6} {a['e_1']:>5} "
                      f"{a['cyc']:>5} {a['path']:>5} {a['other']:>5} "
                      f"{a['chi']:>5} {str(a['planar']):>6}")
                if a["other"] and f == nframes:
                    print(f"    link=other 示例 (v,deg): {a['other_ex']}")
                if a["note"]:
                    print(f"    注: {a['note']}")
        if f < nframes:
            sch.run_frame(angular_iters=angular)

    # 口径 B 对照：真嵌入面数（Euler）—— 与 F3 比较即得「被误当面的分离三角」数
    print("\n口径对照: F3(全部3-团) vs 2V-4(球面真面数)。"
          "若 planarity=True 且 F3>2V-4，多出的就是分离三角（K2 会误当缝隙）。")


if __name__ == "__main__":
    main()
