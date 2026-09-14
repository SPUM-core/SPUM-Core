# -*- coding: utf-8 -*-
"""combinatorial_proto.py — L0 纯组合原型：纯 ID 计数的关系网络元胞机。

范式裁定（用户 2026-09-11）
--------------------------
  L0 只用 id 建立关系网络；几何是网络输出后的解析（投影层 L2）。
  元胞 = 一个 id；元胞的规则 = 计数「它和谁连着」；「连着的多少决定它
  还能连多少」；边不能交叉。几何只在输出后解析。
  本文件不含任何浮点/坐标/距离——纯整数旋转系统。

「边是否交叉」怎么判（本原型要回答的核心问题）
---------------------------------------------
  交叉是**嵌入**性质，不是图的性质。只有「谁和谁连着」时交叉无定义。
  解法：把**面**提升为一等公民，用「旋转系统」（每个元胞的环序邻居表）
  承载 —— 它是纯整数数据，不是几何数据。

  判据（全部是局部整数检查，每个元胞只查自己的环序）：
    (C2) 每个元胞的 link 是路径或环    —— 否则非流形（交叉/领结）
    (C3) 无领结：任意两面最多共享一条边
    Euler 不变量 chi = V - E + F        —— 球面 2；下降即出现把手(=交叉)
    容量界 E <= 3V-6                    —— 平面图必要条件

  **最小充分判据**（本原型的核心）：
    三角形 (i,j,k) 可安全锥化  <=>  三处局部后继检查同时成立
        succ_j(i)==k 且 succ_k(j)==i 且 succ_i(k)==j
    只成立一部分 => 该 3-团是「分离三角」（两侧都已被占满），在其上锥化 =
    往已占满的位置塞球 = 制造交叉（chi 下降）。这正是旧 CUDA K2 的缺陷。

用法
----
  python combinatorial_proto.py nframes=6 seed=tetra mode=all dmin=3 unguarded=0
  python combinatorial_proto.py selftest=1
  python combinatorial_proto.py seed=icosa puncture=1          （单点穿刺 → 单开口）

  seed      : tetra | icosa | patch（R=2 → 19 元胞的开放三角晶格贴片）
  mode      : all（每个合法三角面都创生）| one（每帧只创生一次）
              | hole（洞锥化，闭合开口）| edge（洞边锥化，边界生长）
  dmin      : 删除阈值（公理 deg<2；三维三角剖分工程下限 3）
  unguarded : 1 = 跳过判据、把所有 3-团当面锥化（复现旧引擎的交叉缺陷，作对照）
  puncture  : 1 = 先执行一次 V- 单点穿刺（删最大度元胞）制造单开口

输出全是整数计数：
  V E F chi d=3V-6-E S=Sigma(6-deg) 度直方图 洞数/最大洞 开口占比 晶子候选。
"""

import sys
from collections import Counter


# ============================================================
# 0. 种子夹具：有向面表 → 旋转系统
#    面表是纯组合数据（种子输入），不含几何；审计与规则全部纯组合。
# ============================================================
def rotation_from_faces(faces):
    """faces: 有向三角面 [(x,y,z), ...]（面按 x->y->z->x 定向）。

    succ[(x,y)] = z  <=>  dart (x,y) 所在面后继 dart 是 (y,z)。
    返回 rot[v] = v 的环序邻居表，满足 succ_v(w) = succ[(v,w)]。
    """
    succ = {}
    for (x, y, z) in faces:
        for (a, b, c) in ((x, y, z), (y, z, x), (z, x, y)):
            old = succ.get((a, b))
            if old is not None and old != c:
                raise ValueError(
                    f"面表定向不一致: dart ({a},{b}) 后继 {old} vs {c}")
            succ[(a, b)] = c

    verts = sorted({a for (a, _) in succ})
    rot = {}
    for v in verts:
        outs = sorted(b for (a, b) in succ if a == v)
        if not outs:
            rot[v] = []
            continue
        w0 = outs[0]
        cyc = [w0]
        w = w0
        while True:
            w = succ.get((v, w))
            if w is None:
                raise ValueError(f"顶点 {v} 的环序不闭合（不是合法旋转系统）")
            if w == w0:
                break
            cyc.append(w)
        rot[v] = cyc
    return rot


class RotNet:
    """旋转系统 = L0 的全部状态（纯 id 整数，无几何、无浮点）。"""

    def __init__(self, rot=None):
        self.rot = dict(rot) if rot else {}
        self.nid = (max(self.rot) + 1) if self.rot else 0

    # ---- 计数原语（每个元胞只需自己的 rot）----
    def ids(self):
        return sorted(self.rot)

    def V(self):
        return len(self.rot)

    def deg(self, v):
        return len(self.rot[v])

    def E(self):
        return sum(len(r) for r in self.rot.values()) // 2

    # ---- 环序后继（纯局部查询）----
    def succ(self, v, a):
        r = self.rot[v]
        return r[(r.index(a) + 1) % len(r)]

    def pred(self, v, a):
        r = self.rot[v]
        return r[(r.index(a) - 1) % len(r)]

    def adj(self, a, b):
        return b in self.rot.get(a, ())

    # ---- 面追踪（审计与去重校验用；规则本身不做全局追踪）----
    def dart_next(self, d):
        v, w = d
        return (w, self.succ(w, v))

    def face_of(self, d, cap=1 << 20):
        f = [d]
        e = self.dart_next(d)
        n = 0
        while e != d:
            f.append(e)
            e = self.dart_next(e)
            n += 1
            if n > cap:
                raise RuntimeError(f"面追踪不收敛于 dart {d}（旋转系统损坏）")
        return f

    def faces(self):
        """返回全部 dart 环（每个环 = 一个面）。"""
        seen = set()
        out = []
        for v in self.ids():
            for w in self.rot[v]:
                if (v, w) in seen:
                    continue
                f = self.face_of((v, w))
                seen.update(f)
                out.append(f)
        return out

    def face_vertices(self, cycle):
        return [d[0] for d in cycle]

    def is_face_cycle(self, vs):
        """给定顶点环序 vs，判断它当前是否仍是（同一方向的）一个面。"""
        if len(vs) < 3 or not self.adj(vs[0], vs[1]):
            return False
        return self.face_vertices(self.face_of((vs[0], vs[1]))) == list(vs)

    # ---- 交叉判据（核心；每个元胞只查自己的环序）----
    def tri_is_face(self, i, j, k):
        """三角形 (i,j,k) 是否是**真面**（可安全锥化）。

          在 j 处：i 的后继 == k
          在 k 处：j 的后继 == i
          在 i 处：k 的后继 == j
        三者同时成立 <=> dart(i->j) 所在面恰为 (i->j)->(j->k)->(k->i)。
        """
        if not (self.adj(i, j) and self.adj(j, k) and self.adj(k, i)):
            return False
        return (self.succ(j, i) == k and self.succ(k, j) == i
                and self.succ(i, k) == j)

    # ---- 三个 χ 保持操作 ----
    def cone_tri_face(self, i, j, k):
        """A 类（面锥化）：在**真**三角面上锥化，新增 deg-3 元胞。
        ΔV=+1 ΔE=+3 ΔF=+2 Δchi=0；Δd(k>=4 开口)=0（面上无开口）。"""
        if not self.tri_is_face(i, j, k):
            return None
        w = self.nid
        self.nid += 1
        ri = self.rot[i]
        ri.insert(ri.index(k) + 1, w)      # succ_i(k)=w, succ_i(w)=j
        rj = self.rot[j]
        rj.insert(rj.index(i) + 1, w)      # succ_j(i)=w, succ_j(w)=k
        rk = self.rot[k]
        rk.insert(rk.index(j) + 1, w)      # succ_k(j)=w, succ_k(w)=i
        self.rot[w] = [i, k, j]
        return w

    def cone_hole(self, vs):
        """B 类（洞锥化）：在 k>=4 边形洞上锥化，新增 deg-k 元胞，闭合该洞。
        ΔV=+1 ΔE=+k ΔF=+(k-1) Δchi=0；Δd = -(k-3)（开口收拢）。"""
        k = len(vs)
        if k < 4 or not self.is_face_cycle(vs):
            return None
        w = self.nid
        self.nid += 1
        self.rot[w] = [vs[0]] + list(reversed(vs[1:]))
        for t in range(k):
            a, b = vs[(t - 1) % k], vs[(t + 1) % k]
            r = self.rot[vs[t]]
            r.insert(r.index(a) + 1, w)    # succ(a)=w, succ(w)=b
        return w

    def cone_edge(self, i, j):
        """边锥化：在边 {i,j} 所在的面上插入 deg-2 元胞（边界生长）。
        ΔV=+1 ΔE=+2 ΔF=+1 Δchi=0；Δd = +1（边被折线替换，洞增一边）。"""
        if self.deg(i) < 2 or self.deg(j) < 2 or not self.adj(i, j):
            return None
        k = self.pred(i, j)
        m = self.succ(j, i)
        w = self.nid
        self.nid += 1
        ri = self.rot[i]
        ri.insert(ri.index(k) + 1, w)      # succ_i(k)=w, succ_i(w)=j
        rj = self.rot[j]
        rj.insert(rj.index(i) + 1, w)      # succ_j(i)=w, succ_j(w)=m
        self.rot[w] = [i, j]
        return w

    # ---- 重连：保 χ 保 V 保 E 的对合 ----
    def flip_delta(self, i, j):
        """若边 {i,j} 可翻转，返回 (k, l, ΔΨ)；否则 None。

        可翻转条件：{i,j} 是边；两侧的两个面都是三角面（k = succ_j(i)、
        l = succ_i(j)，且 succ_k(j)==i、succ_l(i)==j）；且 k≠l、k,l 不互邻
        （否则新边 {k,l} 已存在，翻转会产生重边）。

        注：deg(i)=3 时其 link 是三角形 ⇒ 其三个邻居两两互邻 ⇒ k,l 必相邻
        ⇒ 自动被拒。故翻转**不会**把任何顶点降到 deg<3，无需额外守卫。

        ΔΨ = 2(d_k+d_l−d_i−d_j) + 4（Ψ = Σ(6−deg)²）。翻转移动度数而非边数：
        deg(i),deg(j) 各 −1，deg(k),deg(l) 各 +1 ⇒ ΔV=ΔE=ΔF=Δχ=0，Σ(6−deg) 不变。
        """
        if i == j or not self.adj(i, j):
            return None
        k, l = self.succ(j, i), self.succ(i, j)
        if k == l or self.adj(k, l):
            return None
        if self.succ(k, j) != i or self.succ(l, i) != j:
            return None
        dpsi = 2 * ((self.deg(k) + self.deg(l))
                    - (self.deg(i) + self.deg(j))) + 4
        return (k, l, dpsi)

    def flip_edge(self, i, j):
        """重连（边翻转 / 2-2 Pachner 移动）：删边 {i,j}，加边 {k,l}。

        ΔV=0 ΔE=0 ΔF=0 Δchi=0，且仍为三角剖分。它是**对合**（翻两次复原）
        —— 与「锥化 / 删点」互逆一样，这是动作集自逆的第二种来源：
        在三角剖分上「顶点度数 d ↔ 面边数 d」（庞加莱对偶），deg-3 ↔ 三角形。

        局部手术（rot 环序）：i 删 j；j 删 i；k 在 j 后插入 l；l 在 i 后插入 k。
        """
        r = self.flip_delta(i, j)
        if r is None:
            return None
        k, l = r[0], r[1]
        self.rot[i].remove(j)
        self.rot[j].remove(i)
        rk = self.rot[k]
        rk.insert(rk.index(j) + 1, l)
        rl = self.rot[l]
        rl.insert(rl.index(i) + 1, k)
        return (i, j, k, l)

    def force_cone_tri(self, i, j, k):
        """野蛮锥化：用在**非真面**的 3-团（分离三角）上。

        真面有确定定向，分离三角没有——旧引擎的判据是「j,k 互邻即缝隙」，
        天然不带定向。此处用给定的（字典序）三元组当定向，即「在已被两侧
        占满的位置塞球」：新元胞仍与 i,j,k 相邻，但环序不再对应任何面
        （chi 下降 / 出现领结）。"""
        if not (self.adj(i, j) and self.adj(j, k) and self.adj(k, i)):
            return None
        w = self.nid
        self.nid += 1
        for (a, b) in ((i, k), (j, i), (k, j)):
            r = self.rot[a]
            if b in r:
                r.insert(r.index(b) + 1, w)
            else:
                r.append(w)
        self.rot[w] = [i, k, j]
        return w

    # ---- 删除（V-；一帧只删一层，无级联）----
    def break_edge(self, u, v):
        """断边（V⁻ 的**边事件**形式；§3.2「湮灭 = 删边」的落地）。

        删一条边 → 两侧的两个面合并为一个面。
        ΔV=0 ΔE=−1 ΔF=−1 Δχ=0；合并后若为 k≥4 边形则产生开口。
        要求两端 deg ≥ 2：deg=1 的一端断开后会变成孤立元胞，面追踪无定义。

        注：§2.4 的「三个保 χ 的组合操作 + 一个删除操作」表未列此项，但它与
        `remove_vertex` 一样保 χ，是 §3.2 候选类型中「湮灭」的直接实现。
        """
        if u == v or not self.adj(u, v):
            return None
        if self.deg(u) < 2 or self.deg(v) < 2:
            return None
        self.rot[u].remove(v)
        self.rot[v].remove(u)
        return (u, v)

    def remove_vertex(self, v):
        """删除单个元胞及其全部关联边（其 d 个面合并成 1 个 d 边形）。
        ΔV=-1 ΔE=-d ΔF=-(d-1) => Δchi = 0；Δd = d-3（d>3 时产生开口）。"""
        for w in list(self.rot[v]):
            if w in self.rot and v in self.rot[w]:
                self.rot[w].remove(v)
        del self.rot[v]

    def prune(self, dmin=2):
        dead = [v for v in self.ids() if self.deg(v) < dmin]
        for v in dead:
            self.remove_vertex(v)
        return dead


# ============================================================
# 1. 种子
# ============================================================
def seed_tetra():
    """正四面体（闭合球面三角剖分）：V=4 E=6 F=4 chi=2 d=0。"""
    return rotation_from_faces([(0, 1, 3), (1, 0, 2), (2, 0, 3), (1, 2, 3)])


def seed_icosa():
    """正二十面体（闭合球面三角剖分）：V=12 E=30 F=20 chi=2 d=0。
    顶点：0=顶 T，1=底 B，2..6=上环 a_i，7..11=下环 c_i。"""
    T, B = 0, 1
    A = [2 + i for i in range(5)]
    C = [7 + i for i in range(5)]
    faces = []
    for i in range(5):
        faces.append((T, A[i], A[(i + 1) % 5]))
        faces.append((B, C[(i + 1) % 5], C[i]))
        faces.append((A[i], C[i], A[(i + 1) % 5]))
        faces.append((A[(i + 1) % 5], C[i], C[(i + 1) % 5]))
    return rotation_from_faces(faces)


def seed_patch(R=2):
    """三角晶格六边形贴片（R=2 → 19 元胞）的开放种子。

    环序 = 6 个晶格方向（逆时针）中实际存在邻居的子序列；子序列保持
    环序，故每个元胞的 rot 是一个合法环序。R=2 时 V=19 E=42，
    外边界为 12 边形 → 洞 1 个（开放）。
    """
    dirs = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]
    keys = []
    for q in range(-R, R + 1):
        for r in range(max(-R, -q - R), min(R, -q + R) + 1):
            keys.append((q, r))
    idx = {k: i for i, k in enumerate(keys)}
    rot = {}
    for k in keys:
        q, r = k
        rot[idx[k]] = [idx[(q + dq, r + dr)] for (dq, dr) in dirs
                       if (q + dq, r + dr) in idx]
    return rot


SEEDS = {"tetra": seed_tetra, "icosa": seed_icosa, "patch": seed_patch}


# ============================================================
# 2. 帧规则
# ============================================================
def all_triangles(net):
    """全部 3-团（确定性顺序）。仅野蛮路径使用。"""
    out = []
    for i in net.ids():
        ni = set(net.rot[i])
        for j in sorted(ni):
            if j <= i:
                continue
            for k in sorted(ni & set(net.rot[j])):
                if k > j:
                    out.append((i, j, k))
    return out


def adjacent_hole_edge(net, vs):
    """洞 vs 上第一条两端 deg>=2 的边 (i,j)（确定性）。"""
    k = len(vs)
    for t in range(k):
        i, j = vs[t], vs[(t + 1) % k]
        if net.deg(i) >= 2 and net.deg(j) >= 2:
            return (i, j)
    return None


def run_frame(net, mode="all", dmin=2, unguarded=False):
    """一帧：创生(V+) → 判断悬挂 → 删除(V-)。一帧只做一次删除，无级联。

    创生候选在帧首快照（模拟 L1 请求队列），逐个以判据复核后执行。
    返回 (born_ids, dead_ids)。
    """
    f0 = net.faces()
    cand_tri = [net.face_vertices(c) for c in f0 if len(c) == 3]
    cand_hole = [net.face_vertices(c) for c in f0 if len(c) >= 4]

    born = []
    if unguarded:
        # 旧判据「j,k 互邻即缝隙」：3-团一律锥化。若它确是真面（两个定向
        # 之一），按正确定向锥化；否则（分离三角）无定向可用 → 野蛮插入。
        for (i, j, k) in all_triangles(net):
            if net.tri_is_face(i, j, k):
                w = net.cone_tri_face(i, j, k)
            elif net.tri_is_face(i, k, j):
                w = net.cone_tri_face(i, k, j)
            else:
                w = net.force_cone_tri(i, j, k)
            if w is not None:
                born.append(w)
    elif mode == "all":
        for (i, j, k) in cand_tri:
            w = net.cone_tri_face(i, j, k)
            if w is not None:
                born.append(w)
    elif mode == "one":
        for (i, j, k) in cand_tri:
            w = net.cone_tri_face(i, j, k)
            if w is not None:
                born.append(w)
                break
    elif mode == "hole":
        for vs in cand_hole:
            w = net.cone_hole(vs)
            if w is not None:
                born.append(w)
    elif mode == "edge":
        for vs in cand_hole:
            if not net.is_face_cycle(vs):
                continue
            e = adjacent_hole_edge(net, vs)
            if e is None:
                continue
            w = net.cone_edge(*e)
            if w is not None:
                born.append(w)
                break
    else:
        raise ValueError(f"unknown mode: {mode}")

    dead = net.prune(dmin)
    return born, dead


# ============================================================
# 3. 审计（全局只读；规则本身是局部的）
# ============================================================
def holes_of(net):
    """洞 = 顶点不重复的 k>=4 边面。返回边长列表（降序）。"""
    hs = []
    for c in net.faces():
        vs = net.face_vertices(c)
        if len(vs) >= 4 and len(set(vs)) == len(vs):
            hs.append(len(vs))
    return sorted(hs, reverse=True)


def link_type(net, v):
    """元胞 v 的 link（N(v) 的诱导子图）类型：cycle / path / chord / disc。

    注：旋转系统在「面意义」上保证 link 恒为一个环；这里的诱导子图判据是
    更严的量：
      cycle = 干净内部元胞（诱导子图恰为一个环）
      path  = 边界元胞（诱导子图为一条未被弦闭合的路径）
      chord = N(v) 内部有多余弦 => v 处存在分离三角（面已被两侧占满）
      disc  = N(v) 诱导子图不连通 => 真正的非流形（领结/交叉）
    """
    nb = sorted(net.rot[v])
    d = len(nb)
    if d == 0:
        return "disc"
    ld = {x: 0 for x in nb}
    la = {x: [] for x in nb}
    le = 0
    for ai in range(d):
        for bi in range(ai + 1, d):
            x, y = nb[ai], nb[bi]
            if y in net.rot[x]:
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
    if len(seen) != d:
        return "disc"
    degs = sorted(ld.values())
    if le == d and all(x == 2 for x in degs):
        return "cycle"
    if le == d - 1 and degs.count(1) == 2 and all(x in (1, 2) for x in degs):
        return "path"
    return "chord"


def find_crystallites(net):
    """在全局 deg==5 的元胞诱导子图上找连通分量，报告 12 点 5-正则分量
    （= 正二十面体：12 顶点唯一的 5-正则平面三角剖分）。"""
    deg5 = {v for v in net.ids() if net.deg(v) == 5}
    seen = set()
    out = []
    for v in sorted(deg5):
        if v in seen:
            continue
        comp = []
        st = [v]
        seen.add(v)
        while st:
            u = st.pop()
            comp.append(u)
            for w in net.rot[u]:
                if w in deg5 and w not in seen:
                    seen.add(w)
                    st.append(w)
        if len(comp) < 12:
            continue
        cs = set(comp)
        m = sum(1 for u in comp for w in net.rot[u] if w in cs) // 2
        reg = all(sum(1 for w in net.rot[u] if w in cs) == 5 for u in comp)
        out.append({"n": len(comp), "m": m, "regular": reg,
                    "icosa": (len(comp) == 12 and m == 30 and reg)})
    return out


def count_separating_triangles(net):
    """分离三角数 = 是 3-团、但两个定向都不是真面 的三角形个数。
    这些正是旧判据「j,k 互邻即缝隙」会误当缝隙去锥化的位置。"""
    n = 0
    for (i, j, k) in all_triangles(net):
        if not (net.tri_is_face(i, j, k) or net.tri_is_face(i, k, j)):
            n += 1
    return n


def audit(net):
    ids = net.ids()
    V, E = net.V(), net.E()
    cyc = net.faces()
    n_iso = sum(1 for v in ids if net.deg(v) == 0)
    F = len(cyc) + n_iso
    degs = [net.deg(v) for v in ids]
    holes = holes_of(net)
    chi = V - E + F
    lt = Counter(link_type(net, v) for v in ids)
    t3 = len(all_triangles(net))
    return {
        "V": V, "E": E, "F": F, "chi": chi, "genus": (2 - chi) / 2,
        "d3v6": 3 * V - 6 - E, "S": sum(6 - d for d in degs),
        "hist": dict(sorted(Counter(degs).items())),
        "n_holes": len(holes), "max_hole": holes[0] if holes else 0,
        "open_ratio": (holes[0] / F) if (holes and F) else 0.0,
        "T3": t3, "sep": count_separating_triangles(net),
        "cyc": lt["cycle"], "path": lt["path"],
        "chord": lt["chord"], "disc": lt["disc"],
        "cryst": find_crystallites(net),
    }


# ============================================================
# 4. 打印
# ============================================================
def fmt_hist(hist):
    return " ".join(f"{k}:{v}" for k, v in hist.items())


def print_row(tag, a):
    if a.get("V", 0) == 0:
        print(f"{tag}  (空网络)")
        return
    ic = sum(1 for c in a["cryst"] if c["icosa"])
    bd = ""
    if "born" in a:
        bd = f" | V+={a['born']:4d} V-={a['dead']:3d}"
    print(f"{tag} V={a['V']:4d} E={a['E']:5d} F={a['F']:5d} chi={a['chi']:4d} "
          f"g={a['genus']:.0f} d={a['d3v6']:3d} S={a['S']:3d}{bd} | "
          f"3团={a['T3']:4d} 分离={a['sep']:4d} | "
          f"洞={a['n_holes']}(max{a['max_hole']}) 开口={a['open_ratio']:.3f} "
          f"晶子12={ic}")
    print(f"     deg[{fmt_hist(a['hist'])}]  "
          f"link[cycle={a['cyc']} path={a['path']} chord={a['chord']} "
          f"disc={a['disc']}]")


def selftest():
    print("=" * 78)
    print("[自检] 种子计数与不变量")
    print("=" * 78)
    ok = True
    for name, want in (("tetra", (4, 6, 4, 2)), ("icosa", (12, 30, 20, 2))):
        net = RotNet(SEEDS[name]())
        a = audit(net)
        got = (a["V"], a["E"], a["F"], a["chi"])
        flag = "OK" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  {name:6s} V,E,F,chi = {got}  期望 {want}  [{flag}]  "
              f"S={a['S']} d={a['d3v6']} 分离={a['sep']} "
              f"link(cyc/chord/disc)={a['cyc']}/{a['chord']}/{a['disc']}")
    # 正二十面体：全体 deg=5，S=12，且自身就是 12 点 5-正则
    net = RotNet(seed_icosa())
    a = audit(net)
    all5 = set(a["hist"].keys()) == {5}
    ic = sum(1 for c in a["cryst"] if c["icosa"])
    print(f"  icosa  全体deg5={all5}  S=6V-2E={a['S']}=={6 * a['V'] - 2 * a['E']}  "
          f"自检晶子12={ic}")
    if not (all5 and a["S"] == 12 and ic == 1):
        ok = False
    # patch：V=19 E=42，外边界 12 边形洞 → d=9；边界元胞 link=path（12 个）
    net = RotNet(seed_patch(2))
    a = audit(net)
    p_ok = (a["V"], a["E"], a["chi"], a["max_hole"], a["path"]) == (19, 42, 2, 12, 12)
    if not p_ok:
        ok = False
    print(f"  patch  V={a['V']} E={a['E']} F={a['F']} chi={a['chi']} d={a['d3v6']} "
          f"洞={a['n_holes']}(max{a['max_hole']}) 开口={a['open_ratio']:.3f} "
          f"link(path边界={a['path']})  [{'OK' if p_ok else 'FAIL'}]")
    # 重连（边翻转）：对合性 + 保 V/E/F/chi + ΔΨ 公式自洽
    #   注：rot[v] 是**循环**邻居表，起点元素任意 ⇒ 比较须取循环规范形（旋转到最小元素打头），
    #   否则会把「同一环序换个起点」误判为不一致。
    def _cyc(r):
        if not r:
            return ()
        m = min(range(len(r)), key=lambda t: r[t])
        return tuple(r[m:]) + tuple(r[:m])

    net = RotNet(seed_icosa())
    a0 = audit(net)
    s0 = {v: _cyc(r) for v, r in net.rot.items()}
    edges = sorted({(v, w) if v < w else (w, v)
                    for v in net.ids() for w in net.rot[v]})
    fl = [(i, j) for (i, j) in edges if net.flip_delta(i, j) is not None]
    f_ok = bool(fl)
    detail = ""
    if f_ok:
        i, j = fl[0]
        k, l, dpsi = net.flip_delta(i, j)
        p0 = sum((6 - net.deg(v)) ** 2 for v in net.ids())
        net.flip_edge(i, j)
        a1 = audit(net)
        p1 = sum((6 - net.deg(v)) ** 2 for v in net.ids())
        net.flip_edge(k, l)
        s2 = {v: _cyc(r) for v, r in net.rot.items()}
        f_ok = ((a1["V"], a1["E"], a1["F"], a1["chi"])
                == (a0["V"], a0["E"], a0["F"], a0["chi"])
                and p1 - p0 == dpsi and s2 == s0 and a1["S"] == 12)
        detail = (f"icosa 可翻边 {len(fl)}/{len(edges)}；首条 ({i},{j})->({k},{l}) "
                  f"ΔΨ={dpsi}（实测 {p1 - p0}）；翻两次复原（环序等价）={s2 == s0}")
    if not f_ok:
        ok = False
    print(f"  [{'OK' if f_ok else 'FAIL'}] 重连 flip_edge：{detail}")
    print(f"  自检总体: {'全部通过' if ok else '存在失败项'}")
    return ok


def main():
    kv, rest = {}, []
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v
        else:
            rest.append(a)

    if kv.get("selftest", "0") == "1":
        selftest()
        return

    seed = kv.get("seed", "tetra")
    mode = kv.get("mode", "all")
    dmin = int(kv.get("dmin", 3))
    nframes = int(kv.get("nframes", rest[0] if rest else 4))
    unguarded = kv.get("unguarded", "0") == "1"
    puncture = kv.get("puncture", "0") == "1"
    if seed not in SEEDS:
        raise SystemExit(f"未知种子 {seed}，可选 {sorted(SEEDS)}")

    net = RotNet(SEEDS[seed]())
    print("=" * 78)
    print(f"[纯组合原型] seed={seed} mode={mode} dmin={dmin} "
          f"nframes={nframes} unguarded={int(unguarded)} puncture={int(puncture)}")
    print("  一帧 = 创生(全部合法位置) → 判断悬挂 → 删除(只删一层)")
    print("=" * 78)
    if puncture:
        # V- 单点穿刺：删一个最大度元胞 → 其 d 个三角面合并成一个 d 边形洞。
        # 这是闭合种子上唯一能产生「开口」的路径（所有创生操作 Δchi=0 且不造洞）。
        v = max(net.ids(), key=lambda x: (net.deg(x), -x))
        d0 = net.deg(v)
        net.remove_vertex(v)
        a = audit(net)
        print(f"[穿刺] 删除元胞 {v}（deg={d0}）→ 洞 {d0} 边形；"
              f"Δd 预期 = deg-3 = {d0 - 3}")
        print_row("op ", a)
        print(f"     开口占比 = max洞/F = {a['max_hole']}/{a['F']} = "
              f"{a['open_ratio']:.4f}  （判据IV 阈值 1/3 = 0.3333）")
        print("-" * 78)
    print_row("f0 ", audit(net))
    for f in range(1, nframes + 1):
        born, dead = run_frame(net, mode=mode, dmin=dmin, unguarded=unguarded)
        a = audit(net)
        a["born"] = len(born)
        a["dead"] = len(dead)
        print_row(f"f{f:<2d}", a)
        if net.V() == 0:
            print("  (网络已清空，后续帧无对象)")
            break
    print("-" * 78)
    print("说明：chi=2 且 genus=0 → 球面嵌入（无交叉）；chi<2 → 出现把手(=交叉)。")
    print("      d=3V-6-E=0 → 闭合三角剖分；d>0 → 存在开口/边界。")


if __name__ == "__main__":
    main()
