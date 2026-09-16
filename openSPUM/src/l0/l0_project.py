# -*- coding: utf-8 -*-
"""l0_project.py — **几何对照装置**（2026-09-15 正式降级）。

降级裁决（2026-09-15）
----------------------
本文件曾是 L0 的投影入口，但 `solve()` / `minimal_dim()` 用 scipy L-BFGS-B
做**力导向布局**——它预设了坐标空间，借用了欧氏距离，与 SPUM 公理
「关系是唯一本原，空间是关系的投影」冲突。

纯组合投影链路已完整落地，不依赖本文件：

    combinatorial_proto.py (RotNet / rot 旋转系统)
        → 面是一等公民（tri_is_face / cone_tri_face）
        → l0_topology.py (growth_dim / spectral / combinatorics)
        → l1_projection.py (ring_sphere / 局部角容量)
        → l0_equivalence.py (r-邻域等价 / stable_r)

上述链路全程无坐标、无距离、无优化器。本文件保留作**几何对照**：
当需要把纯组合读数与欧氏嵌入结果并排比较时，`solve()` / `readout()`
仍可运行；但**不得作为投影链路的入口**。

原投影机制描述（R1–R4）保留作设计参照，但 R4「维度=扫出来的最小可行维」
已被 `l0_topology.py` 的 `growth_dim` / `spectral`（谱读数）替代——
维度是图的函数，不是坐标空间的参数。

边表生成器 `build_regular` / `build_rand23` 已迁移至 `l0_topology.py`
自带，`l0_topology.py` 不再 import 本文件。
"""

import sys
from collections import Counter

import numpy as np

TOL_E = 0.05      # 边长残差（归一化后）
TOL_O = 0.02      # 互不入穿透
D0_NORM = 1.0


# ============================================================
# 0. 纯关系输入（只有边表）
# ============================================================
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
    """每点随机 2/3 条关系（上一实验用的 50 节点图）。"""
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
# 1. 投影求解：把关系压成点集（维度是参数，扫出来的）
# ============================================================
def _prep(V, edges):
    n = V
    E = np.array(edges, dtype=int) if edges else np.zeros((0, 2), dtype=int)
    deg = np.zeros(n, dtype=int)
    for (u, v) in edges:
        deg[u] += 1
        deg[v] += 1
    w = deg.astype(float)
    if len(E):
        t = w[E[:, 0]] + w[E[:, 1]]
        t = t / t.mean()
    else:
        t = np.zeros(0)
    I, J = np.triu_indices(n, k=1)
    A = np.zeros((n, n), dtype=bool)
    for (u, v) in edges:
        A[u, v] = A[v, u] = True
    return E, t, w, I, J, A


def _components(V, edges):
    """连通分量（纯关系读出的，与几何无关）。"""
    adj = [[] for _ in range(V)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    seen = [False] * V
    comps = []
    for s in range(V):
        if seen[s]:
            continue
        seen[s] = True
        stack, c = [s], []
        while stack:
            x = stack.pop()
            c.append(x)
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    stack.append(y)
        comps.append(sorted(c))
    return comps


def _spectral_connected(n, edges, dim):
    """单个连通分量的拉普拉斯谱嵌入（n×dim）。"""
    A = np.zeros((n, n))
    for (u, v) in edges:
        A[u, v] = A[v, u] = 1.0
    d = A.sum(axis=1)
    L = np.diag(d) - A
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-9))
    Ln = L * dinv[:, None] * dinv[None, :]
    w, vec = np.linalg.eigh(Ln)
    idx = np.argsort(w)
    w, vec = w[idx], vec[:, idx]
    nz = int((w > 1e-9).sum())
    start = n - nz                      # 跳过零特征值（该分量只有一个）
    k = min(dim, max(1, nz))
    X = vec[:, start:start + k].copy()
    if X.shape[1] < dim:
        X = np.hstack([X, np.zeros((n, dim - X.shape[1]))])
    if len(edges):
        E = np.array(edges, dtype=int)
        Lm = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
        m = Lm.mean()
        if m > 1e-12:
            X = X / m
    return X


def _spectral_init(V, edges, dim):
    """拉普拉斯特征向量初始化：对环/网格等规则图几乎是精确解（环 → 圆）。

    多分量时逐分量嵌入，再沿第 0 轴拉开（跨分量全为非关系 ⇒ 必须互不入）。
    """
    comps = _components(V, edges)
    if len(comps) <= 1:
        return _spectral_connected(V, edges, dim).ravel()
    X = np.zeros((V, dim))
    for comp in comps:
        pos = {v: i for i, v in enumerate(comp)}
        sub = [(pos[u], pos[v]) for (u, v) in edges if u in pos and v in pos]
        X[np.array(comp)] = _spectral_connected(len(comp), sub, dim)
    spacing = 6.0 * (float(np.abs(X).max()) + 1.0)
    off = 0.0
    for comp in comps:
        X[np.array(comp), 0] += off
        off += spacing
    return X.ravel()


def solve(V, edges, dim, mu=200.0, restarts=12, seed=20260914, maxiter=8000,
          spectral=True):
    """多起点 L-BFGS。能量 = Σ_关系 (L−t)² + μ Σ_非关系 max(0, t−L)²。

    起点 0 = 拉普拉斯谱初始化（对规则图近乎精确）；其余为随机起点。
    """
    n = V
    E, t, w, I, J, A = _prep(V, edges)
    ne = len(E)
    neq = len(I)

    def fg(p):
        X = p.reshape(n, dim)
        g = np.zeros_like(X)
        e = 0.0
        if ne:
            du = X[E[:, 0]] - X[E[:, 1]]
            L = np.sqrt((du * du).sum(axis=1) + 1e-18)
            e += float(((L - t) ** 2).sum())
            c = 2.0 * (L - t) / L
            np.add.at(g, E[:, 0], c[:, None] * du)
            np.add.at(g, E[:, 1], -c[:, None] * du)
        if mu > 0 and neq:
            dp = X[I] - X[J]
            D = np.sqrt((dp * dp).sum(axis=1) + 1e-18)
            tt = w[I] + w[J]
            tt = tt / tt.mean() if tt.mean() > 0 else tt
            viol = np.maximum(0.0, tt - D)
            # 只惩罚「非关系」对
            viol = np.where(A[I, J], 0.0, viol)
            e += float(mu * (viol ** 2).sum())
            c2 = -2.0 * mu * viol / D
            np.add.at(g, I, c2[:, None] * dp)
            np.add.at(g, J, -c2[:, None] * dp)
        return e, g.ravel()

    from scipy.optimize import minimize
    starts = []
    if spectral and ne:
        starts.append(_spectral_init(V, edges, dim))
    for r in range(restarts):
        rng = np.random.default_rng(seed + 104729 * r + 7 * dim)
        starts.append((rng.normal(size=(n, dim)) * 0.7).ravel())
    best = None
    for x0 in starts:
        res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-14})
        if best is None or res.fun < best.fun:
            best = res
    return best.x.reshape(n, dim), float(best.fun)


# ============================================================
# 2. 读数：冲突 / 重合 / 有效维度
# ============================================================
def readout(V, edges, X, dim):
    n = V
    E, t, w, I, J, A = _prep(V, edges)
    out = {"dim": dim}
    if len(E):
        L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
        res = np.abs(L - t)
        out["edge_res"] = float(res.max())
        out["conflict"] = int((res > TOL_E).sum())
    else:
        out["edge_res"] = 0.0
        out["conflict"] = 0
    D = np.linalg.norm(X[I] - X[J], axis=1)
    tt = w[I] + w[J]
    tt = tt / tt.mean() if tt.mean() > 0 else tt
    pen = np.where(A[I, J], -1e9, tt - D)      # 只看非关系对
    out["overlap"] = float(max(0.0, pen.max()))
    out["coincide"] = int(((D < 0.5 * tt) & (~A[I, J])).sum())
    Y = X - X.mean(axis=0)
    lam = np.linalg.svd(Y, compute_uv=False) ** 2
    out["eff_dim"] = float((lam.sum() ** 2) / (lam ** 2).sum()) if lam.sum() > 0 else 0.0
    out["ok"] = (out["edge_res"] <= TOL_E and out["overlap"] <= TOL_O)
    return out


def minimal_dim(V, edges, dims=5, **kw):
    """扫 dim = 1..dims，返回 (最小可行维 D*, 该维读数, 逐维读数)。"""
    rows, first = [], None
    for dim in range(1, dims + 1):
        X, e = solve(V, edges, dim, **kw)
        r = readout(V, edges, X, dim)
        r["energy"] = e
        rows.append(r)
        if r["ok"] and first is None:
            first = dim
    return first, rows


# ============================================================
# 3. 报告
# ============================================================
def struct_line(V, edges):
    deg = np.zeros(V, dtype=int)
    for (u, v) in edges:
        deg[u] += 1
        deg[v] += 1
    return (f"V={V} E={len(edges)} 关系量直方={dict(sorted(Counter(deg.tolist()).items()))}")


def show(V, edges, dims=5, kw=None, tag=""):
    kw = kw or {}
    print(f"  ── {tag}：{struct_line(V, edges)}")
    D, rows = minimal_dim(V, edges, dims=dims, **kw)
    for r in rows:
        flag = "✅" if r["ok"] else "  "
        print(f"     dim={r['dim']}  {flag} 边残差max={r['edge_res']:.4f} "
              f"冲突边={r['conflict']:3d}  互不入穿透={r['overlap']:.4f} "
              f"重合对={r['coincide']:3d}  有效维={r['eff_dim']:.3f}")
    print(f"     ⇒ **最小可行维 D\\* = {D}**"
          f"{'（扫描上限内无解）' if D is None else ''}")
    return D, rows


def main():
    kv = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v
        else:
            kv[a] = "1"
    kw = {"mu": float(kv.get("mu", 200.0)),
          "restarts": int(kv.get("restarts", 12)),
          "seed": int(kv.get("seed", 20260914))}
    V = int(kv.get("V", 40))
    dims = int(kv.get("dims", 5))

    print("=" * 96)
    print("[L0 关系投影机制 v0]  输入 = 纯关系（边表）；输出 = 投影 + 最小可行维 D*")
    print("  R1 节点压成点（体量 = 关系量，作标量）  R2 关系 ⇒ 相接  R3 非关系 ⇒ 互不入")
    print("=" * 96)

    if kv.get("selftest"):
        print("\n[selftest] 已知结构的最小可行维")
        cases = {
            "C4 四边环(度2)": (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
            "C6 六边环(度2)": (6, [(i, (i + 1) % 6) for i in range(6)]),
            "K4 四面体(度3)": (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
            "正二十面体(度5)": (12, None),
        }
        ico = None
        for name, (n, ed) in cases.items():
            if ed is None:
                continue
            D, _ = show(n, ed, dims=4, kw=kw, tag=name)
            print()
        ok = True
        print("[selftest] 结论：C4/C6 应在 dim=2 首次可行，K4 应在 dim=3 首次可行")
        return

    if kv.get("scan"):
        dmax = int(kv.get("dmax", 14))
        print(f"\n[扫描] d-正则随机图 V={V}，d = 2..{dmax} ⇒ 最小可行维 D*(d)")
        print(f"  （关系量 = 度数 d；节点数固定 {V}；dims 扫到 {dims}）")
        print(f"  {'d':>3} {'E':>6}  {'D*':>4}   逐维读数")
        for d in range(2, dmax + 1):
            ed = build_regular(V, d, seed=kw["seed"])
            if ed is None:
                print(f"  {d:>3} {'—':>6}   配对失败")
                continue
            D, rows = minimal_dim(V, ed, dims=dims, **kw)
            detail = "  ".join(
                f"dim{r['dim']}:{'✅' if r['ok'] else '✗'}"
                f"(边{r['edge_res']:.2f},穿{r['overlap']:.2f})" for r in rows)
            print(f"  {d:>3} {len(ed):>6}  {str(D):>4}   {detail}")
        return

    if kv.get("graph") == "reg50":
        ed = build_rand23(int(kv.get("V", 50)), seed=kw["seed"])
        show(int(kv.get("V", 50)), ed, dims=dims, kw=kw,
             tag="50 节点 · 每点 2/3 条关系")
        return

    d = int(kv.get("d", 5))
    ed = build_regular(V, d, seed=kw["seed"])
    if ed is None:
        raise SystemExit("配对失败")
    show(V, ed, dims=dims, kw=kw, tag=f"{V} 节点 · {d}-正则随机图")


if __name__ == "__main__":
    main()
