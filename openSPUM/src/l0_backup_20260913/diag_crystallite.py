"""晶子/闭环拓扑诊断：L0 帧序列自然演化（无外力）产出的实际拓扑结构。

用法: python diag_crystallite.py [nframes] [slide] [angular] [full] [dump]
                                   [seed=tetra|triangle|chain|single]
                                   [dmin=3] [trace=1]

报告：
  - V / E / 度直方图 / Σ(6−deg)（应与 6V−2E 恒等）
  - 饱和中心（deg=max）的邻居互连数 M：M=24 → 立方八面体(fcc)摆法；M=30 → 正二十面体
  - 连通分量规模分布
  - 正二十面体诱导子图搜索（12 顶点 / 30 边 / 全体内度 5），full=1 时穷举

种子探测（seed=/dmin= 由用户裁定"先做种子探测"）：
  - seed=triangle/chain/single 为开放种子（有边界/悬挂端），tetra 为闭合种子
  - dmin=2 把 K6 悬挂阈值降回公理下限（deg<2 才不自洽），观察 deg-2 边界能否存活
  - trace=1 逐帧打印 V/E/Σ(6−deg)/开放度(3V−6−E)/悬挂再生，用于判定边界演化
"""

import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from host import FrameScheduler


def tetra_seed():
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                 dtype=np.float64)
    v /= np.linalg.norm(v[0])
    v *= np.sqrt(6.0) / 2.0
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return v, edges


def triangle_seed():
    """3 个半径 1 的球两两相切（等边三角形，边长 2，外接圆半径 2/√3）。
    全体 deg=2 —— 一个纯边界环（顶点无内部三角面）。"""
    v = np.array([[1, 0, 0],
                  [-0.5, np.sqrt(3) / 2, 0],
                  [-0.5, -np.sqrt(3) / 2, 0]], dtype=np.float64)
    v *= (2.0 / np.sqrt(3.0))
    edges = [(0, 1), (1, 2), (0, 2)]
    return v, edges


def chain_seed():
    """2 个半径 1 的球相切（中心距 2）。两端 deg=1 —— 最小悬挂端。"""
    v = np.array([[-1.0, 0, 0], [1.0, 0, 0]], dtype=np.float64)
    return v, [(0, 1)]


def single_seed():
    """单个孤立球（deg=0）——公理下不可确认存在，预期被 K7 立即清除。"""
    return np.array([[0.0, 0, 0]], dtype=np.float64), []


def patch_seed(R: int = 2):
    """三角晶格六边形贴片（间距 2 = 半径 1 球相切，R=2 → 19 节点）。
    角点 deg=3、边点 deg=4、内部 deg=6 —— 一个"平铺但带真实边界"的开放种子，
    用于检验边界能否在演化中存活（open = 3V−6−E 是否 > 0）。"""
    a = 2.0
    pts, key2i = [], {}
    for q in range(-R, R + 1):
        for r in range(max(-R, -q - R), min(R, -q + R) + 1):
            key2i[(q, r)] = len(pts)
            pts.append((a * (q + r / 2.0), a * r * np.sqrt(3) / 2.0, 0.0))
    edges = set()
    for (q, r), i in key2i.items():
        for dq, dr in ((1, 0), (0, 1), (1, -1)):
            j = key2i.get((q + dq, r + dr))
            if j is not None:
                edges.add((min(i, j), max(i, j)))
    return np.array(pts, dtype=np.float64), sorted(edges)


def seed_of(name: str):
    return {"tetra": tetra_seed, "triangle": triangle_seed,
            "chain": chain_seed, "single": single_seed,
            "patch": patch_seed}[name]()


def parse_kv(argv):
    kv, rest = {}, []
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v
        else:
            rest.append(a)
    return kv, rest


def load_adj(sch):
    """返回 (全局索引数组, 邻接表 adj[local] -> set(local))。"""
    act = sch.active.get()
    idx = np.where(act)[0]
    remap = -np.ones(act.shape[0], dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    nb_idx = sch.nb_idx.get()
    nb_cnt = sch.nb_count.get()
    adj = [set() for _ in range(len(idx))]
    for i in idx:
        a = int(remap[i])
        for s in range(int(nb_cnt[i])):
            j = int(nb_idx[i, s])
            if j < 0 or not act[j]:
                continue
            b = int(remap[j])
            if a != b:
                adj[a].add(b)
                adj[b].add(a)
    return idx, adj


def components(adj):
    seen = [False] * len(adj)
    comps = []
    for s in range(len(adj)):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        c = []
        while stack:
            u = stack.pop()
            c.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(c)
    comps.sort(key=len, reverse=True)
    return comps


def find_icosahedron(adj, limit_nodes=None):
    """精确搜索：12 顶点诱导子图，全体内度 = 5（正二十面体 = 12 顶点唯一的
    5-正则平面三角剖分；连通且 5-正则 12 点即正二十面体）。

    以 v 为种子：枚举 adj[v] 中 5 点子集 R（v 的 5 个内部邻居）；
    R 必须自身成 5-环；余下 6 点 U 必须是 R 的邻居并满足整体 5-正则。
    """
    found = []
    n = 0
    for v in range(len(adj)):
        nbrs = sorted(adj[v])
        if len(nbrs) < 5:
            continue
        # 必要预筛：N(v) 内部边数 ≥ 5（5-环）
        if sum(1 for a in range(len(nbrs)) for b in range(a + 1, len(nbrs))
               if nbrs[b] in adj[nbrs[a]]) < 5:
            continue
        if limit_nodes is not None and n >= limit_nodes:
            break
        n += 1
        for R in combinations(nbrs, 5):
            Rs = set(R)
            if any(len(adj[x] & Rs) != 2 for x in R):
                continue
            U = set()
            for x in R:
                U |= (adj[x] - Rs - {v})
            if len(U) != 6:
                continue
            S = {v} | Rs | U
            if len(S) == 12 and all(len(adj[x] & S) == 5 for x in S):
                if S not in found:
                    found.append(S)
    return found


def dump_center_geometry(sch, idx, adj, deg, which=0):
    """Dump the spherical geometry of one saturated (deg=max) node:
    its neighbor directions, all pairwise angles, and whether the
    geometric tangency count agrees with the graph edge count M."""
    import math
    dmax = int(deg.max()) if len(deg) else 0
    cand = [v for v in range(len(adj)) if deg[v] == dmax]
    if not cand:
        print("no deg=max node")
        return
    g2l = {int(idx[x]): x for x in range(len(idx))}
    v = cand[which % len(cand)]
    gi = int(idx[v])
    nb_cnt_raw = sch.nb_count.get()
    nb_idx_raw = sch.nb_idx.get()
    nb_dir_raw = sch.nb_dir.get()
    row_i = nb_idx_raw[gi]
    row_d = nb_dir_raw[gi]
    n = int(nb_cnt_raw[gi])
    ns = []
    for s in range(n):
        j = int(row_i[s])
        if j < 0:
            continue
        th = float(row_d[s, 0]); ph = float(row_d[s, 1])
        u = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph),
                      math.cos(th)])
        ns.append((j, u, th, ph))
    print(f"\n--- dump deg={dmax} node  local={v} global={gi}  nb={len(ns)} ---")

    def linked(g1, g2):
        l1 = g2l.get(g1); l2 = g2l.get(g2)
        return l1 is not None and l2 is not None and l2 in adj[l1]

    ang = []
    for a in range(len(ns)):
        for b in range(a + 1, len(ns)):
            c = float(np.clip(np.dot(ns[a][1], ns[b][1]), -1.0, 1.0))
            ang.append((math.degrees(math.acos(c)), ns[a][0], ns[b][0]))
    ang.sort()
    geo_tan = sum(1 for a, _, _ in ang if a <= 61.0)
    graph_M = sum(1 for _, i1, i2 in ang if linked(i1, i2))
    print(f"pairwise angles (66), sorted; tangent(<=61deg) count={geo_tan} "
          f"graph_M={graph_M}")
    print("  " + " ".join(f"{a:5.1f}" for a, _, _ in ang))
    shell_deg = {j: 0 for j, _, _, _ in ns}
    geo_deg = {j: 0 for j, _, _, _ in ns}
    for a, i1, i2 in ang:
        if linked(i1, i2):
            shell_deg[i1] += 1
        if a <= 61.0:
            geo_deg[i1] += 1
    print("shell graph-deg dist:",
          dict(sorted(Counter(shell_deg.values()).items())))
    print("shell GEOM-deg dist (<=61deg):",
          dict(sorted(Counter(geo_deg.values()).items())))


def live_stats(sch):
    """当前存活粒子的 (V, E, 度数数组)。"""
    act = sch.active.get()
    cnt = sch.nb_count.get()
    d = cnt[act].astype(np.int64)
    return int(d.size), int(d.sum()) // 2, d


def trace_line(f, V, E, d, dmin):
    """单帧边界追踪行。open = 3V-6-E > 0 表示存在边界（非闭合球面三角剖分）。"""
    n0 = int((d == 0).sum())
    n1 = int((d == 1).sum())
    n2 = int((d == 2).sum())
    below = int((d < dmin).sum())   # 本帧结束后残留、待下一帧 K6 清除的悬挂端
    return (f"  f{f:3d}  V={V:6d}  E={E:7d}  S6={int((6 - d).sum()):6d}  "
            f"open={3 * V - 6 - E:5d}  deg0={n0:3d} deg1={n1:3d} deg2={n2:5d}  "
            f"below_dmin={below:5d}")


def main():
    kv, rest = parse_kv(sys.argv[1:])
    nframes = int(rest[0]) if len(rest) > 0 else 20
    slide = float(rest[1]) if len(rest) > 1 else None
    angular = int(rest[2]) if len(rest) > 2 else 0
    full = bool(int(rest[3])) if len(rest) > 3 else False
    seed_name = kv.get("seed", "tetra")
    dmin = int(kv.get("dmin", 3))
    trace = kv.get("trace", "0") == "1"
    do_dump = (len(rest) > 4 and rest[4] == "dump") or kv.get("dump") == "1"

    pos, edges = seed_of(seed_name)
    sch = FrameScheduler(capacity=8192, radius_slope=slide, dangling_min=dmin)
    sch.spawn_initial(pos, edges, radius0=1.0)
    law = "r≡1" if slide is None else f"r=1+{slide}·deg"
    print(f"[L0 拓扑诊断] nframes={nframes} seed={seed_name} dmin={dmin} "
          f"半径律={law} angular_iters={angular}")

    prevV = int(sch.n_active.get()[0])
    for f in range(nframes):
        sch.run_frame(angular_iters=angular)
        if trace:
            V, E, d = live_stats(sch)
            regen = max(0, prevV - V)  # 本帧净清除的颗粒（含悬挂端消失）
            print(trace_line(f + 1, V, E, d, dmin) + f"  cleared~{regen}")
            prevV = V if V > 0 else prevV

    V = int(sch.n_active.get()[0])
    print(f"帧数={sch.frame_number}  V={V}")

    idx, adj = load_adj(sch)
    deg = np.array([len(a) for a in adj])
    E = int(deg.sum()) // 2
    print(f"E={E}  Sigma(6-deg)={int((6-deg).sum())}  6V-2E={6*len(idx)-2*E}")
    print(f"度分布: {dict(sorted(zip(*np.unique(deg, return_counts=True))))}")
    print(f"σ=V/E={len(idx)/max(E,1):.4f}  avg_deg={2*E/max(len(idx),1):.3f}")

    comps = components(adj)
    print(f"连通分量数={len(comps)}  前8大={[len(c) for c in comps[:8]]}")

    dmax = int(deg.max()) if len(deg) else 0
    Ms = []
    for v in range(len(adj)):
        if deg[v] != dmax or dmax < 3:
            continue
        nb = sorted(adj[v])
        Ms.append(sum(1 for a in range(len(nb)) for b in range(a + 1, len(nb))
                      if nb[b] in adj[nb[a]]))
    print(f"饱和中心(deg={dmax}) 数={len(Ms)}  "
          f"邻居互连数M分布={dict(sorted(Counter(Ms).items()))}")
    print("  （立方八面体/fcc 摆法 M=24；正二十面体摆法 M=30）")

    # M vs degree：每个度数下"邻居壳的互连数"均值/区间
    # 参照：度数 d 的规则壳最大互连 M=d(d-1)/2 × ... 这里给关键参考值
    #   d=4 四面体 M=6；d=6 八面体 M=12；d=12 二十面体 M=30；d=12 fcc M=24
    ref = {4: 6, 6: 12, 12: 30}
    by_deg = {}
    for v in range(len(adj)):
        d = deg[v]
        if d < 3:
            continue
        nb = sorted(adj[v])
        m = sum(1 for a in range(len(nb)) for b in range(a + 1, len(nb))
                if nb[b] in adj[nb[a]])
        by_deg.setdefault(d, []).append(m)
    print("deg  count  M_mean  M_min  M_max   max_possible  ref")
    for d in sorted(by_deg):
        arr = by_deg[d]
        mp = d * (d - 1) // 2
        print(f"{d:3d}  {len(arr):5d}  {sum(arr)/len(arr):6.2f}  "
              f"{min(arr):5d}  {max(arr):5d}   {mp:5d}         {ref.get(d,'')}")

    lim = None if full else 400
    cands = find_icosahedron(adj, limit_nodes=lim)
    print(f"正二十面体诱导子图（12点5-正则）数={len(cands)}"
          f"{'' if full else f'（仅扫前{lim}个候选种子）'}")
    for c in cands[:3]:
        print(f"   候选 全局索引={sorted(int(idx[x]) for x in c)}")

    if do_dump:
        dump_center_geometry(sch, idx, adj, deg, 0)


if __name__ == "__main__":
    main()
