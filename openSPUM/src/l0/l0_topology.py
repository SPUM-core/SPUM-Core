# -*- coding: utf-8 -*-
"""l0_topology.py — **纯组合投影入口**（2026-09-15 裁决）。

从**边表**读拓扑与维度，全程无坐标、无距离、无优化器。

投影链路（2026-09-15 裁决）
--------------------
  纯组合投影链路（不经过 l0_project.py 的力导向布局）：

    combinatorial_proto.py (RotNet / rot 旋转系统)
        │  面是一等公民：tri_is_face / cone_tri_face / flip_edge
        │  三元闭环自动生成面（2-单纯形）；面即刚性，刚性即形状
        ▼
    l0_topology.py (本文件)
        │  combinatorics(faces=...)  面由外部声明（多元模式）
        │  growth_dim / spectral     维度 = 谱读数，不是输入参数
        ▼
    l1_projection.py (N4-B 局部角容量投影)
        │  ring_sphere / sigma_of / angle_defect
        │  全程无全局嵌入、无迭代、无坐标
        ▼
    l0_equivalence.py (r-邻域等价)
           equivalence_classes / stable_r

  l0_project.py 已降级为**几何对照装置**——它的 solve() / minimal_dim()
  用 scipy L-BFGS-B 做力导向布局（坐标+距离），不得作为投影入口。
  本文件不再 import l0_project.py；边表生成器已自带。

三条读数（全部是图的函数，与坐标无关）
------------------------------------
  R1 曲率    : K_v = 1 − deg(v)/6，Σ K_v = V − E/3。
               ⇒ 仅当网络是**闭合三角剖分**时，Σ K_v = χ（离散高斯-博内；等价 Σ(6−deg)=6χ）。
  R2 闭合判定: 纯边表判定「是否为闭合 2-流形三角剖分」：
                 (a) 每条边恰在 2 个三角形里（无边界、无领结）；
                 (b) 每个顶点的 link（邻居按三角形连边）是**单环**。
               两条都成立 ⇒ χ = V − E + F 有拓扑意义。χ=2 ⇔ 拓扑球面。
  R3 维度    : **无坐标**维度读数（三条独立路径）
               (a) 生长维数 d_g：N(r)=距离≤r 的节点数，log N ~ d_g · log r；
               (b) 谱隙 λ1 与谱维数 d_s：拉普拉斯计数函数 N(λ) ~ λ^{d_s/2}；
               (c) 直径 diam。expander 判据：λ1 不趋零 + diam ~ log V ⇒ 无低维投影。

严格标注（防伪结论）
------------------
  ⚠ 「闭合三角剖分 ⇒ 单点度 ≤ 12」**为假**。反例双锥 bi_n：
       V=n+2, E=3n, F=2n, χ=2，每边 2 面、每点 link 单环，而两极 deg=n 无上界。
     闭合只给出 Σ(6−deg)=6χ（0 层级节点恰 12 个），**不给出单点度数上界**。
     12 需要 A3（同尺度等价，现为假设）或 G2/G3（球面排他）中至少一条。
  ⚠ 「Σ(1−k/6)=2 ⇒ 球面」只在闭合三角剖分上成立；对任意边表它只是 V−E/3。

用法
----
  python l0_topology.py selftest=1
  python l0_topology.py graph=icosa          （正二十面体：闭合三角剖分，χ=2）
  python l0_topology.py graph=subicosa       （细分一次：42 点，度 5/6，仍 χ=2）
  python l0_topology.py graph=bipyramid n=5  （双锥：合法球面剖分，两极 deg=n）
  python l0_topology.py graph=grid m=6       （2D 方格：无三角形 ⇒ 非三角剖分）
  python l0_topology.py graph=rand50         （50 点 2/3 关系 → 随机）
  python l0_topology.py graph=reg6 V=42      （6-正则随机图）
"""

import os
import sys
from collections import Counter, deque

import numpy as np

# ── 边表生成器（2026-09-15 从 l0_project.py 迁入，断掉对力导向布局的依赖）──


def build_regular(V, d, seed=20260914, tries=800):
    """d-正则随机图。优先 networkx（大 d 也可行），退化到配置模型。返回边表或 None。"""
    if 0 < d < V and (V * d) % 2 == 0:
        try:
            import networkx as nx
            G = nx.random_regular_graph(d, V, seed=seed)
            return sorted((min(u, v), max(u, v)) for u, v in G.edges())
        except Exception:
            pass
    rng = np.random.default_rng(seed * 1000 + d)
    for _ in range(tries):
        stubs = np.repeat(np.arange(V), d)
        rng.shuffle(stubs)
        edges, ok = set(), True
        for i in range(0, len(stubs), 2):
            a, b = int(stubs[i]), int(stubs[i + 1])
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok:
            return sorted(edges)
    return None


def build_rand23(V, seed=20260914, tries=2000):
    """每点随机 2/3 条关系。"""
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        target = rng.integers(2, 4, size=V)
        stubs = np.repeat(np.arange(V), target)
        rng.shuffle(stubs)
        edges, ok = set(), True
        for i in range(0, len(stubs), 2):
            a, b = int(stubs[i]), int(stubs[i + 1])
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok:
            return sorted(edges)
    return None


# ============================================================
# 0. 内置图（全部只产边表，无坐标）
# ============================================================
def g_icosa():
    import itertools
    phi = (1 + 5 ** 0.5) / 2
    raw = []
    for s1 in (1, -1):
        for s2 in (1, -1):
            raw += [(0, s1, s2 * phi), (s1, s2 * phi, 0), (s2 * phi, 0, s1)]
    pts = np.unique(np.round(raw, 6), axis=0)   # 仅用于定出邻接，随后即丢弃
    E = [(i, j) for i, j in itertools.combinations(range(len(pts)), 2)
         if abs(np.linalg.norm(pts[i] - pts[j]) - 2.0) < 1e-6]
    return len(pts), E


def g_subicosa():
    """正二十面体 1→4 三角细分（纯组合：边中点作新顶点）。"""
    import itertools
    V, E = g_icosa()
    adj = [set() for _ in range(V)]
    for a, b in E:
        adj[a].add(b)
        adj[b].add(a)
    faces = [(a, b, c) for a, b, c in itertools.combinations(range(V), 3)
             if b in adj[a] and c in adj[a] and c in adj[b]]
    mid, nxt = {}, V
    for a, b in E:
        mid[(min(a, b), max(a, b))] = nxt
        nxt += 1
    m = lambda a, b: mid[(min(a, b), max(a, b))]
    newE = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            newE.add((u, m(u, v)))
            newE.add((v, m(u, v)))
        newE.add((m(a, b), m(b, c)))
        newE.add((m(b, c), m(c, a)))
        newE.add((m(c, a), m(a, b)))
    return nxt, sorted(newE)


def g_bipyramid(n):
    """双锥 bi_n：两极 0,1 + 腰环 2..n+1。合法球面三角剖分，两极 deg=n。"""
    P, Q = 0, 1
    ring = list(range(2, n + 2))
    E = []
    for i, v in enumerate(ring):
        w = ring[(i + 1) % n]
        E += [(P, v), (Q, v), (min(v, w), max(v, w))]
    return n + 2, sorted(set(E))


def g_grid(m):
    """m×m 方格（4-邻接）。空间性但无三角形 ⇒ 非三角剖分。"""
    V = m * m
    E = []
    for i in range(m):
        for j in range(m):
            v = i * m + j
            if i + 1 < m:
                E.append((v, v + m))
            if j + 1 < m:
                E.append((v, v + 1))
    return V, E


# ============================================================
# 1. 组合拓扑：三角形 / 边缘关系 / link / χ
# ============================================================
def combinatorics(V, edges, faces=None):
    """faces=None ⇒ 二元模式（把 3-团当 2-单纯形：flag 复形假设）。
       faces=[...] ⇒ 多元模式（2-单纯形由外部**声明**，不从二元关系推断）。"""
    adj = [set() for _ in range(V)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    if faces is None:
        tri = set()
        for u in range(V):
            nb = sorted(adj[u])
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    if nb[j] in adj[nb[i]]:
                        tri.add(tuple(sorted((u, nb[i], nb[j]))))
        tri = sorted(tri)
    else:
        tri = sorted(tuple(sorted(f)) for f in faces)

    etri = Counter()                    # 每条边落在几个三角形里
    link = [set() for _ in range(V)]    # link 图：邻居之间若成三角形则连边
    for a, b, c in tri:
        for x, y in ((a, b), (b, c), (c, a)):
            etri[(min(x, y), max(x, y))] += 1
        link[a].add((min(b, c), max(b, c)))
        link[b].add((min(a, c), max(a, c)))
        link[c].add((min(a, b), max(a, b)))

    # (a) 每条边恰在 2 个三角形里
    bad_edges = [e for e in edges if etri[(min(e[0], e[1]), max(e[0], e[1]))] != 2]
    # (b) 每个顶点的 link 是单环（连通 + 每点 link-度 2 + 覆盖全部邻居）
    bad_links = []
    for v in range(V):
        lk = [e for e in link[v]]
        deg_lk = Counter()
        for x, y in lk:
            deg_lk[x] += 1
            deg_lk[y] += 1
        nb = sorted(adj[v])
        if not nb:
            continue
        ok = (len(lk) == len(nb)) and all(deg_lk[x] == 2 for x in nb)
        if ok:                          # 连通性
            seen, st = {nb[0]}, [nb[0]]
            while st:
                x = st.pop()
                for y in nb:
                    if y not in seen and (min(x, y), max(x, y)) in set(lk):
                        seen.add(y)
                        st.append(y)
            ok = len(seen) == len(nb)
        if not ok:
            bad_links.append(v)

    F = len(tri)
    chi = V - len(edges) + F
    closed = (not bad_edges) and (not bad_links)
    return {"F": F, "chi": chi, "closed": closed,
            "bad_edges": bad_edges, "bad_links": bad_links}


def curvature(V, edges):
    deg = np.zeros(V, dtype=int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    K = 1.0 - deg / 6.0
    return deg, K, float(K.sum())


# ============================================================
# 2. 无坐标维度读数
# ============================================================
def _powfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None, None
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    sol, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ sol
    ssr = float(((ly - pred) ** 2).sum())
    sst = float(((ly - ly.mean()) ** 2).sum())
    return float(sol[0]), (1.0 - ssr / sst if sst > 0 else 0.0)


def growth_dim(V, edges):
    """生长维数 d_g：N(r) 幂律拟合；同时给直径与 N(r) 剖面。"""
    adj = [[] for _ in range(V)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    prof, diam, cnt = None, 0, 0
    for s in range(V):
        dist = [-1] * V
        dist[s] = 0
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = dist[x] + 1
                    q.append(y)
        mx = max(dist)
        diam = max(diam, mx)
        if prof is None:
            prof = [0.0] * (mx + 1)
        elif len(prof) < mx + 1:
            prof += [0.0] * (mx + 1 - len(prof))
        for r in range(mx + 1):
            prof[r] += sum(1 for d in dist if 0 <= d <= r)
        cnt += 1
    prof = [p / cnt for p in prof]
    rs = [r for r in range(1, len(prof)) if 0 < prof[r] <= 0.5 * V]
    ns = [prof[r] for r in rs]
    slope, r2 = _powfit(rs, ns)
    return {"d_g": slope, "r2": r2, "diam": diam, "prof": prof}


def spectral(V, edges):
    """谱隙 λ1 与谱维数 d_s（拉普拉斯计数函数幂律）。"""
    A = np.zeros((V, V))
    for u, v in edges:
        A[u, v] = A[v, u] = 1.0
    L = np.diag(A.sum(1)) - A
    w = np.sort(np.linalg.eigvalsh(L))
    pos = w[w > 1e-9]
    if len(pos) == 0:
        return {"lam1": 0.0, "d_s": None, "slope": None, "r2": None}
    lam1 = float(pos[0])
    hi = float(pos.max())
    xs = np.linspace(lam1 * 1.05, hi * 0.6, 24)
    ys = np.array([int((pos <= x).sum()) for x in xs], float)
    m = ys >= 1
    slope, r2 = _powfit(xs[m], ys[m])
    return {"lam1": lam1, "slope": slope, "r2": r2,
            "d_s": (2.0 * slope if slope is not None else None)}


# ============================================================
# 3. 报告
# ============================================================
def report(name, V, edges, faces=None):
    print("=" * 92)
    print(f"【{name}】  V={V}  E={len(edges)}")
    deg, K, sK = curvature(V, edges)
    print(f"  [R1 曲率]  度直方 = {dict(sorted(Counter(deg.tolist()).items()))}")
    print(f"            K_v = 1 − deg/6   ΣK = {sK:.4f}   （= V − E/3 = {V - len(edges)/3:.4f}）")

    cb = combinatorics(V, edges, faces)
    mode = "多元（面=声明）" if faces is not None else "二元（面=3-团推论）"
    print(f"  [R2 闭合]  胞腔模式 = {mode}    2-单纯形数 F = {cb['F']}")
    if cb["closed"]:
        print(f"            ✅ 闭合 2-流形三角剖分    χ = V−E+F = {cb['chi']}"
              f"    {'⇔ 拓扑球面' if cb['chi'] == 2 else '⇔ 非球面'}")
        print(f"            6χ = {6 * cb['chi']}   （应等于 Σ(6−deg) = {6 * V - 2 * len(edges)}）")
    else:
        print(f"            ✗ 非闭合三角剖分"
              f"（坏边 {len(cb['bad_edges'])} 条 / 坏 link {len(cb['bad_links'])} 点）")
        print(f"            ⇒ χ 读数**无意义**；ΣK 只是 V−E/3，不该解释为球面判据")

    gd = growth_dim(V, edges)
    sp = spectral(V, edges)
    dgtxt = f"{gd['d_g']:.3f} (R²={gd['r2']:.3f})" if gd["d_g"] is not None else "—（无幂律区）"
    dstxt = f"{sp['d_s']:.3f} (R²={sp['r2']:.3f})" if sp["d_s"] is not None else "—"
    print(f"  [R3 维度]  生长维数 d_g = {dgtxt}   直径 = {gd['diam']}")
    print(f"            谱隙 λ1 = {sp['lam1']:.4f}   谱维数 d_s = {dstxt}")
    n2 = gd["prof"][2] if len(gd["prof"]) > 2 else float("nan")
    print(f"            N(1)={gd['prof'][1]:.1f}  N(2)={n2:.1f}  （expander: N 近翻倍）")
    return {"V": V, "E": len(edges), "sK": sK, "closed": cb["closed"],
            "chi": cb["chi"], "d_g": gd["d_g"], "diam": gd["diam"],
            "lam1": sp["lam1"], "d_s": sp["d_s"]}


# ============================================================
# 3b. 用户命题专演：二元关系不够 ⇒ 需多元关系
# ============================================================
def demo_polyadic():
    print("=" * 92)
    print("[demo] 二元关系不够 —— 同一条边表、不同胞腔 ⇒ 不同拓扑")
    print("=" * 92)

    # (A) K4 骨架：边表完全固定，只改「声明哪些 2-单纯形」
    V = 4
    E = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    cases = [
        ("不声明任何面（纯 1-复形）", []),
        ("声明 1 个面 (0,1,2)", [(0, 1, 2)]),
        ("声明 2 个面 (0,1,2)(0,1,3)", [(0, 1, 2), (0, 1, 3)]),
        ("声明全部 4 个面（= flag 复形）",
         [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]),
    ]
    print(f"  1-骨架固定 = K4：V=4, E=6, 边表 = {E}")
    print(f"  {'声明的 2-单纯形':<32}{'F':>3}{'χ=V−E+F':>12}   闭合?")
    for name, fs in cases:
        cb = combinatorics(V, E, fs)
        print(f"  {name:<32}{cb['F']:>3}{cb['chi']:>12}   {'是' if cb['closed'] else '否'}")
    print("  ⇒ 同一条边表，χ 从 −2 到 +2 全取得到 —— **拓扑不由二元关系决定**。")

    # (B) 用户的 o 例子：ABCDEFA 环，加不加公共关系 o
    Ecyc = [(i, (i + 1) % 6) for i in range(6)]
    Econe = Ecyc + [(6, i) for i in range(6)]
    Fcone = [(6, i, (i + 1) % 6) for i in range(6)]
    Edbl = Econe + [(7, i) for i in range(6)]
    Fdbl = Fcone + [(7, i, (i + 1) % 6) for i in range(6)]
    print()
    print("  [用户的 o 例子] ABCDEFA 环，逐级加公共关系")
    for tag, VV, EE, fs in (("ABCDEFA 环（无 o）", 6, Ecyc, []),
                            ("+ 公共关系 o（锥）", 7, Econe, Fcone),
                            ("+ 公共关系 o, o'（双锥）", 8, Edbl, Fdbl)):
        cb = combinatorics(VV, EE, fs)
        print(f"    {tag:<22} V={VV} E={len(EE):>2} F={cb['F']:>2}  χ={cb['chi']:>2}  "
              f"闭合={'是' if cb['closed'] else '否'}"
              f"（{len(cb['bad_edges'])} 条边非双面）")
    print("    ⇒ 加了 o 才能把环「封」住（χ=1 带边界 → χ=2 无边界）。**你的观察成立。**")

    # (C) 但：封住 ≠ 度数被锁
    print()
    print("  [关键分岔] 封住之后，度数被锁了吗？")
    print(f"    {'结构':<14}{'V':>4}{'E':>5}{'F':>5}{'χ':>4}   两极 deg")
    for n in (5, 6, 12, 50):
        VV, EE = g_bipyramid(n)
        cb = combinatorics(VV, EE, None)
        print(f"    {'双锥 bi_%d' % n:<14}{VV:>4}{len(EE):>5}{cb['F']:>5}{cb['chi']:>4}   {n}")
    print("    ⇒ 多元关系锁住了**胞腔**（χ 定了），**没锁住度数**（n 任意）。")

    # (D) 12 的纯组合出处：再加 A3「同尺度等价」
    print()
    print("  [12 的纯组合出处] 追加 A3「同尺度等价」—— 大小=关系量 ⇒ 同尺度 ⇔ 所有点同度 d：")
    print("       闭合三角剖分 ⇒ Σ(6−deg) = 6χ = 12 ⇒ (6 − d)·V = 12")
    names = {3: "正四面体", 4: "正八面体", 5: "正二十面体"}
    for d in range(2, 6):
        if 12 % (6 - d) == 0:
            print(f"       d={d}  ⇒  V = 12/(6−{d}) = {12 // (6 - d):>2}   {names.get(d, '')}")
    print("    ⇒ 正则闭合三角剖分只有这三种；**V = 12 出现在 d = 5（正二十面体）**。")
    print("    ⇒ bi_n 被 A3 排除（两极 deg=n ≠ 环点 deg=4）。")
    print("    ⇒ 这是 12 的**唯一纯组合来路**，代价是 A3 必须由假设升为公理。")



def main():
    kv = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v
        else:
            kv[a] = "1"

    if kv.get("selftest"):
        print("=" * 92)
        print("[selftest] 纯组合拓扑读数（无坐标、无距离、无优化器）")
        print("=" * 92)
        V, E = g_icosa()
        r1 = report("正二十面体（闭合三角剖分）", V, E)
        V, E = g_bipyramid(5)
        r2 = report("双锥 bi_5（两极 deg=5）", V, E)
        V, E = g_bipyramid(50)
        r3 = report("双锥 bi_50（两极 deg=50 —— 反例：闭合但度数无上界）", V, E)
        ok = (r1["closed"] and r1["chi"] == 2 and
              r2["closed"] and r2["chi"] == 2 and
              r3["closed"] and r3["chi"] == 2)
        print("=" * 92)
        print(f"[selftest] 三项均为闭合球面剖分（χ=2）：{ok}")
        print("  ⚠ 三者同时为真 ⇒ 「闭合 ⇒ 度 ≤ 12」不成立（bi_50 两极 deg=50）。")
        return

    if kv.get("demo") == "polyadic":
        demo_polyadic()
        return

    g = kv.get("graph", "icosa")
    if g == "icosa":
        V, E = g_icosa();       name = "正二十面体"
    elif g == "subicosa":
        V, E = g_subicosa();    name = "正二十面体细分一次"
    elif g == "bipyramid":
        n = int(kv.get("n", 5)); V, E = g_bipyramid(n); name = f"双锥 bi_{n}"
    elif g == "grid":
        m = int(kv.get("m", 6)); V, E = g_grid(m);      name = f"{m}×{m} 方格"
    elif g == "rand50":
        V = int(kv.get("V", 50)); E = build_rand23(V);  name = f"{V} 点 · 每点 2/3 关系（随机）"
    elif g == "reg6":
        V = int(kv.get("V", 42)); d = int(kv.get("d", 6))
        E = build_regular(V, d);  name = f"{V} 点 · {d}-正则随机图"
    else:
        raise SystemExit(f"未知 graph={g}")

    if E is None:
        raise SystemExit("边表生成失败")
    print("=" * 92)
    print("[L0 纯组合拓扑检测器]  输入 = 边表；无坐标 / 无距离 / 无优化器")
    print("=" * 92)
    report(name, V, E)


if __name__ == "__main__":
    main()
