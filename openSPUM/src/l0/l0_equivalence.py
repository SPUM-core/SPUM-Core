# -*- coding: utf-8 -*-
"""l0_equivalence.py — 「同尺度等价」的可计算化：根化 r-邻域子图同构。

架构基准
--------
docs/L0_闭合与隔绝_χ2的局部来源.md §5 A3a、§6 候选断言 L0-B9、§8 待裁定第 2 项。

要解决什么
----------
A3（同尺度等价）在 `l0_kissing` 里用的是最粗版「同壳层 ⇒ 同度」（= r=0 等价）。
同度是必要不充分条件：12 点全 deg=3 的图里，三棱柱的点球含三角形、K₃,₃ 的点球
二部无三角，二者局域构型不同但度数相同。

本模块把「同尺度等价」形式化为可计算量：

    两个节点 u, v 在尺度 r 上等价  ⟺  它们的**根化 r-邻域子图同构**。

  · r-邻域子图 G_r(v) = 距离 v ≤ r 的节点在原图中诱导出的子图；
  · **根化** = v 自身被标记为根（同构必须把根映到根）；
  · r=0  ⇒ 只看 deg(v) ⇒ 最粗（即 l0_kissing 的 A3）；
  · r 越大 ⇒ 区分越细；r 超过图直径后不再变化。

两条独立读数（必须一致，否则读数作废）
------------------------------------
  ① **规范证书**（本模块纯 Python 实现，不依赖 networkx）：
     根化 Weisfeiler-Lehman 细化 → 序列化签名。
  ② **VF2 精确同构**（networkx.is_isomorphic，根节点属性匹配）：
     独立交叉验证，权威性更高。

一致性判据：证书说「等价」 ⇒ VF2 必须也说「等价」（证书不得过合并）。
证书说「不等价」时 VF2 可能等价（证书可过拆分，由 VF2 合并）。

谁来决定 r
----------
r 是观测者选定的**尺度参数**，节点本身不「知道」该看几层。本模块给出一个
原则性读数 `stable_r(net)`：使等价类划分停止细化的最小 r（partition(r)
== partition(r+1)）。超过此 r，再多跳数不带来新的区分信息——这是图自身
的「内禀分辨率」。在 SPUM 语义下，一个节点的「尺度身份」由最小可区分它的
r 决定；对距离正则图（如正二十面体），stable_r = 0（任何 r 都只有 1 类）。

用法
----
  python l0_equivalence.py selftest=1
  python l0_equivalence.py seed=icosa r=2
  python l0_equivalence.py seed=prism_k33        # 三棱柱 ⊔ K₃,₃ 演示
"""

import os
import sys
from collections import deque

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from combinatorial_proto import RotNet, SEEDS  # noqa: E402

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:  # pragma: no cover
    _HAS_NX = False


# ============================================================
# §0 接口适配：接受 RotNet 或裸 rot dict
# ============================================================
def _adj(net):
    return net.rot if isinstance(net, RotNet) else net


def _ids(net):
    return net.ids() if isinstance(net, RotNet) else sorted(net)


# ============================================================
# §1 r-邻域子图（根化球）
# ============================================================
def r_ball(net, root, r):
    """返回 (nodes, adj)：G_r(root) 的节点集与内部邻接表。

    nodes = {v : dist(root, v) ≤ r}；adj[v] = v 在该子图中的邻居列表
    （只保留两端都在 nodes 内的边）。根节点 root 自身一定在 nodes 中。
    """
    rot = _adj(net)
    dist = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        if dist[u] >= r:
            continue
        for w in rot[u]:
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    nodes = set(dist)
    adj = {v: [w for w in rot[v] if w in nodes] for v in nodes}
    return nodes, adj


def diameter(net):
    """图直径（最大最短路）；多分量取各分量直径的最大值。"""
    rot = _adj(net)
    ids = _ids(net)
    diam = 0
    for src in ids:
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for w in rot[u]:
                if w not in d:
                    d[w] = d[u] + 1
                    q.append(w)
        if d:
            diam = max(diam, max(d.values()))
    return diam


# ============================================================
# §2 规范证书：根化 Weisfeiler-Lehman（纯 Python，无 networkx）
# ============================================================
def _triangles_in_ball(adj, nodes):
    """球内每节点关联的三角形数（只计三边都在球内的三角）。"""
    tri = {v: 0 for v in nodes}
    nset = set(nodes)
    for v in nodes:
        nb = [w for w in adj[v] if w in nset and w > v]
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                if nb[j] in set(adj[nb[i]]):
                    tri[v] += 1
                    tri[nb[i]] += 1
                    tri[nb[j]] += 1
    return tri


def rooted_wl_certificate(adj, root, nodes, root_deg=None):
    """根化 r-球的规范证书（可哈希）。

    算法
    ----
    初始着色（二维特征，使 1-WL 能区分正则图）：
      每个节点 v → (球内度数 deg_ball(v), 球内关联三角数 tri_ball(v))
      根节点额外用其**原图度数** root_deg 替代球内度数（r=0 时球内无边，
      度数必须取自原图，否则所有节点证书相同）。
    迭代 1-WL：每节点新色 = (旧色, 邻居旧色的有序多元组)，再压缩为整数。
    收敛后，证书 = (根签名, 全图 (色, 邻居色多元组) 的有序多元组)。

    为什么需要三角特征：三棱柱与 K₃,₃ 都是 3-正则 6 点图，纯度数初始色下
    1-WL 永不细化（均匀 d-正则图对 1-WL 不可区分）。加入三角数后，三棱柱
    每点关联 1 个三角、K₃,₃ 为 0，初始色即分开。

    注：r≥1 时根的所有邻居都在球内，root_deg 与球内度数一致；r=0 时只有
    root_deg 携带信息。

    ⚠️ 1-WL 对一般图不完备（存在非同构但 WL 同色的对）；故 VF2 是权威交叉
       验证。本证书是**可靠不变量**（VF2 等价 ⇒ 证书等价），但不必备
       （证书等价 ⇏ VF2 等价，后者计入 collisions）。
    """
    nodes = sorted(nodes)
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    tri = _triangles_in_ball(adj, nodes)
    init_color = [None] * n
    color = [0] * n
    for i, v in enumerate(nodes):
        if v == root:
            d = root_deg if root_deg is not None else len(adj[v])
        else:
            d = len(adj[v])
        init_color[i] = (d, tri[v])
        color[i] = (d, tri[v])

    for _ in range(n + 1):                  # 至多 n 轮必收敛
        new = [None] * n
        for i, v in enumerate(nodes):
            neigh = tuple(sorted(color[idx[w]] for w in adj[v]))
            new[i] = (color[i], neigh)
        # 规范压缩：按 (色, 邻居色多元组) 的字典序分配整数标签，
        # 使标签独立于节点 ID 顺序（否则同构图因根的位置不同得到不同证书）。
        distinct = sorted(set(new))
        label_map = {c: i for i, c in enumerate(distinct)}
        compressed = [label_map[c] for c in new]
        if compressed == color:
            break
        color = compressed

    # 最终签名 = (WL 整数标签, 初始色元组 (deg, tri))。
    # 必须保留初始色：单点球（r=0）的 WL 标签恒为 0，度数信息全靠初始色携带。
    def sig(i):
        return (color[i], init_color[i])

    ri = idx[root]
    root_sig = (sig(ri), tuple(sorted(sig(idx[w]) for w in adj[root])))
    full = tuple(sorted(sig(i) for i in range(n)))
    return (root_sig, full)


# ============================================================
# §3 VF2 交叉验证（networkx）
# ============================================================
def _ball_nx(adj, root, nodes, root_deg):
    g = nx.Graph()
    for v in nodes:
        g.add_node(v, is_root=(v == root),
                   d=len(adj[v]),
                   root_d=(root_deg if v == root else None))
    for v in nodes:
        for w in adj[v]:
            if v < w:
                g.add_edge(v, w)
    return g


def vf2_equivalent(net, u, v, r):
    """权威判定：G_r(u) 与 G_r(v) 是否根化同构。

    节点匹配条件：is_root 一致 + 球内度数一致 + 根的原图度数一致
    （r=0 时球内度数恒为 0，根的原图度数是唯一结构信息）。
    """
    if not _HAS_NX:
        raise RuntimeError("VF2 交叉验证需要 networkx")
    rot = _adj(net)
    nu, au = r_ball(net, u, r)
    nv, av = r_ball(net, v, r)
    if len(nu) != len(nv):
        return False
    gu = _ball_nx(au, u, nu, root_deg=len(rot[u]))
    gv = _ball_nx(av, v, nv, root_deg=len(rot[v]))

    def nm(a, b):
        return (a["is_root"] == b["is_root"]
                and a["d"] == b["d"]
                and a["root_d"] == b["root_d"])

    return nx.is_isomorphic(gu, gv, node_match=nm)


# ============================================================
# §4 等价类划分
# ============================================================
def _cert_labels(net, r):
    ids = _ids(net)
    rot = _adj(net)
    certs = {}
    labels = []
    class_of = {}
    for v in ids:
        nodes, adj = r_ball(net, v, r)
        c = rooted_wl_certificate(adj, v, nodes, root_deg=len(rot[v]))
        if c not in class_of:
            class_of[c] = len(class_of)
        labels.append(class_of[c])
        certs[v] = c
    return labels, certs


def _vf2_labels(net, r):
    """并查集 + 逐对 VF2 → 权威等价类标签。"""
    ids = _ids(net)
    parent = list(range(len(ids)))
    idx = {v: i for i, v in enumerate(ids)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if vf2_equivalent(net, ids[i], ids[j], r):
                union(i, j)

    # 压缩并重新编号
    roots = [find(i) for i in range(len(ids))]
    remap = {}
    labels = []
    for rr in roots:
        if rr not in remap:
            remap[rr] = len(remap)
        labels.append(remap[rr])
    return labels


def equivalence_classes(net, r, vf2=True):
    """返回尺度 r 上的等价类划分。

    返回 dict：
      r            : 尺度
      labels       : 每个节点的类 id（按 net.ids() 顺序），**VF2 权威标签**
      cert_labels  : 规范证书给出的标签（可能过合并或过拆分）
      n_classes    : VF2 等价类数
      classes      : 等价类节点集合的列表（VF2，规范形式，便于跨 r 比较）
      consistent   : 证书**可靠性**（VF2 同 ⇒ cert 同）；False 即证书有 bug
      sound_breaks : VF2 说等价但证书说不等价的节点对（应为空 = 证书过拆分 bug）
      collisions   : 证书说等价但 VF2 说不等价的节点对（1-WL 碰撞，非 bug）
    """
    ids = _ids(net)
    cert_labels, _ = _cert_labels(net, r)
    if vf2 and _HAS_NX:
        vf2_labels = _vf2_labels(net, r)
        labels = vf2_labels
        sound_breaks = []     # VF2 同 ⇒ cert 同 应恒成立
        collisions = []       # cert 同但 VF2 不同（1-WL 碰撞，预期内）
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                vf2_same = (vf2_labels[i] == vf2_labels[j])
                cert_same = (cert_labels[i] == cert_labels[j])
                if vf2_same and not cert_same:
                    sound_breaks.append((ids[i], ids[j]))
                elif cert_same and not vf2_same:
                    collisions.append((ids[i], ids[j]))
        consistent = (len(sound_breaks) == 0)
    else:
        labels = cert_labels
        consistent = None
        sound_breaks = []
        collisions = []

    # 规范划分：按类 id 收集节点集合
    groups = {}
    for v, lab in zip(ids, labels):
        groups.setdefault(lab, []).append(v)
    classes = sorted(frozenset(vs) for vs in groups.values())
    return {
        "r": r,
        "labels": labels,
        "cert_labels": cert_labels,
        "n_classes": len(classes),
        "classes": classes,
        "consistent": consistent,
        "sound_breaks": sound_breaks,
        "collisions": collisions,
    }


# ============================================================
# §5 谁定 r：stable_r —— 划分停止细化的最小尺度
# ============================================================
def stable_r(net, max_r=None):
    """内禀分辨率：使等价类划分停止细化的最小 r。

    partition(r) == partition(r+1)  ⇒  再多跳数不带来新的区分。
    返回稳定时的 r（即最后一次发生细化的 r）。若 r=0 已稳定则返回 0。
    """
    if max_r is None:
        max_r = diameter(net)
    prev = None
    stable_at = 0
    for r in range(max_r + 1):
        cur = frozenset(equivalence_classes(net, r)["classes"])
        if prev is not None and cur == prev:
            return stable_at
        stable_at = r
        prev = cur
    return max_r


# ============================================================
# §6 演示用种子（无坐标、仅邻接；不要求合法旋转系统）
# ============================================================
def seed_prism():
    """三棱柱 C₃×K₂：6 点全 deg=3，含 2 个三角形面。"""
    return {
        0: [1, 2, 3], 1: [0, 2, 4], 2: [0, 1, 5],
        3: [0, 4, 5], 4: [1, 3, 5], 5: [2, 3, 4],
    }


def seed_k33():
    """K₃,₃：6 点全 deg=3，二部图，无三角形。"""
    return {
        0: [3, 4, 5], 1: [3, 4, 5], 2: [3, 4, 5],
        3: [0, 1, 2], 4: [0, 1, 2], 5: [0, 1, 2],
    }


def seed_prism_k33():
    """三棱柱 ⊔ K₃,₃：12 点全 deg=3 —— A3a 的标准演示。"""
    p = seed_prism()
    k = seed_k33()
    rot = dict(p)
    for v, nb in k.items():
        rot[v + 6] = [w + 6 for w in nb]
    return rot


def seed_grid3x3():
    """3×3 方格：4 角 deg=2、4 边 deg=3、中心 deg=4 —— r=0 即分 3 类。"""
    rot = {}
    for i in range(3):
        for j in range(3):
            v = i * 3 + j
            nb = []
            if i > 0:
                nb.append(v - 3)
            if i < 2:
                nb.append(v + 3)
            if j > 0:
                nb.append(v - 1)
            if j < 2:
                nb.append(v + 1)
            rot[v] = nb
    return rot


# ============================================================
# §7 自检
# ============================================================
def selftest():
    ok = True
    line = "-" * 78
    print("=" * 78)
    print("[l0_equivalence] 同尺度等价 = 根化 r-邻域子图同构")
    print("=" * 78)

    # ---- ① 证书与 VF2 一致性：所有测试图 × r=0..3 ----
    print("① 证书 ⇒ VF2 一致性（证书过合并即读数作废）")
    graphs = {
        "tetra": RotNet(SEEDS["tetra"]()),
        "icosa": RotNet(SEEDS["icosa"]()),
        "bi_5": RotNet(SEEDS["bipyramid"](5)),
        "bi_12": RotNet(SEEDS["bipyramid"](12)),
        "prism": seed_prism(),
        "k33": seed_k33(),
        "prism_k33": seed_prism_k33(),
        "grid3x3": seed_grid3x3(),
        "patch_R2": SEEDS["patch"](2),
    }
    all_consistent = True
    for name, g in graphs.items():
        for r in range(4):
            rep = equivalence_classes(g, r)
            if rep["consistent"] is False:
                all_consistent = False
                ok = False
                print(f"   ❌ {name} r={r}: 证书可靠性破坏（VF2 同但证书不同）"
                      f" {rep['sound_breaks']}")
            elif rep["collisions"]:
                print(f"   ⚠ {name} r={r}: 1-WL 碰撞 {len(rep['collisions'])} 对"
                      f"（证书过合并，非 bug，由 VF2 细化）")
    print(f"   9 图 × 4 尺度 全部一致 = {all_consistent}  "
          f"[{'OK' if all_consistent else 'FAIL'}]")

    # ---- ② A3a 核心演示：三棱柱 ⊔ K₃,₃ ----
    print(line)
    print("② A3a 演示：三棱柱 ⊔ K₃,₃（12 点全 deg=3）")
    g = seed_prism_k33()
    r0 = equivalence_classes(g, 0)
    r1 = equivalence_classes(g, 1)
    # 三棱柱节点 = {0..5}，K₃,₃ = {6..11}
    prism_set = frozenset(range(6))
    k33_set = frozenset(range(6, 12))
    c0_ok = (r0["n_classes"] == 1 and r0["consistent"] is True)
    c1_ok = (r1["n_classes"] == 2 and r1["consistent"] is True
             and prism_set in r1["classes"] and k33_set in r1["classes"])
    ok &= (c0_ok and c1_ok)
    print(f"   r=0：{r0['n_classes']} 类（VF2） 期望 1  [{'OK' if c0_ok else 'FAIL'}]"
          f"  —— 同度 = r=0 等价，必要不充分")
    print(f"   r=1：{r1['n_classes']} 类（VF2） 期望 2  [{'OK' if c1_ok else 'FAIL'}]"
          f"  —— 三棱柱点球含三角 / K₃,₃ 点球二部无三角")

    # ---- ③ 正二十面体距离正则：任何 r 恒为 1 类 ----
    print(line)
    print("③ 正二十面体：距离正则 ⇒ 任何 r 恒为 1 类（不能用来演示区分）")
    gi = RotNet(SEEDS["icosa"]())
    icosa_ok = all(equivalence_classes(gi, r)["n_classes"] == 1 for r in range(4))
    ok &= icosa_ok
    print(f"   r=0..3 等价类数 = {[equivalence_classes(gi, r)['n_classes'] for r in range(4)]}"
          f"  期望 [1,1,1,1]  [{'OK' if icosa_ok else 'FAIL'}]")

    # ---- ④ 双锥 bi_5：r=0 起即 2 类（两极 deg=5、赤道 deg=4）----
    print(line)
    print("④ 双锥 bi_5：r=0 即分 2 类（两极 deg=5 / 赤道 deg=4）")
    gb = RotNet(SEEDS["bipyramid"](5))
    b0 = equivalence_classes(gb, 0)
    bip_ok = (b0["n_classes"] == 2 and b0["consistent"] is True)
    ok &= bip_ok
    print(f"   r=0：{b0['n_classes']} 类  期望 2  [{'OK' if bip_ok else 'FAIL'}]")

    # ---- ⑤ 3×3 方格：r=0 分 3 类（deg ∈ {2,3,4}）----
    print(line)
    print("⑤ 3×3 方格：r=0 分 3 类（角 deg=2 / 边 deg=3 / 心 deg=4）")
    gr = seed_grid3x3()
    g0 = equivalence_classes(gr, 0)
    grid_ok = (g0["n_classes"] == 3 and g0["consistent"] is True)
    ok &= grid_ok
    print(f"   r=0：{g0['n_classes']} 类  期望 3  [{'OK' if grid_ok else 'FAIL'}]")

    # ---- ⑥ stable_r：内禀分辨率 ----
    print(line)
    print("⑥ stable_r（划分停止细化的最小尺度）")
    sr_prism = stable_r(seed_prism())
    sr_pk = stable_r(seed_prism_k33())
    sr_icosa = stable_r(RotNet(SEEDS["icosa"]()))
    sr_grid = stable_r(seed_grid3x3())
    print(f"   三棱柱 stable_r = {sr_prism}（顶点传递，r=0/1 均 1 类）")
    print(f"   三棱柱⊔K₃,₃ stable_r = {sr_pk}（r=1 完成跨分量区分）")
    print(f"   正二十面体 stable_r = {sr_icosa}（距离正则，r=0 即稳定）")
    print(f"   3×3 方格 stable_r = {sr_grid}")
    # 正二十面体必须 stable_r == 0（距离正则）
    sr_ok = (sr_icosa == 0)
    ok &= sr_ok
    print(f"   正二十面体 stable_r=0（距离正则）  [{'OK' if sr_ok else 'FAIL'}]")

    # ---- ⑦ 反向控制：证书不恒真 ----
    #    构造两个 deg 相同但 1-邻域不同构的节点，证书必须能区分（r=1）。
    print(line)
    print("⑦ 反向控制：证书不恒真（同度但 1-邻域不同构必须被区分）")
    # 取一个正二十面体 + 一个三棱柱的不交并：icosa 点 deg=5、prism 点 deg=3，
    # r=0 已按 deg 分两类；关键是 r=1 时 icosa 内部恒 1 类、prism 内部恒 1 类。
    # 更强的反向控制：在同一 deg=3 族内，prism 与 k33 的 r=1 证书必须不同。
    # 直接比较两图根节点的 r=1 证书
    np_, ap = r_ball(seed_prism(), 0, 1)
    nk_, ak = r_ball(seed_k33(), 0, 1)
    cert_p = rooted_wl_certificate(ap, 0, np_, root_deg=3)
    cert_k = rooted_wl_certificate(ak, 0, nk_, root_deg=3)
    rc_ok = (cert_p != cert_k)
    ok &= rc_ok
    print(f"   prism 根 r=1 证书 ≠ K₃,₃ 根 r=1 证书 = {rc_ok}"
          f"  [{'OK' if rc_ok else 'FAIL'}]")

    print(line)
    print(f"l0_equivalence selftest: {'全部通过' if ok else '存在失败项'}")
    return ok


# ============================================================
# §8 CLI
# ============================================================
def _kv(argv):
    d = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            d[k] = v
    return d


SEEDS_EXTRA = {"prism": seed_prism, "k33": seed_k33,
               "prism_k33": seed_prism_k33, "grid3x3": seed_grid3x3}


def main(argv):
    kv = _kv(argv)
    if kv.get("selftest", "0") == "1":
        selftest()
        return

    seed = kv.get("seed", "prism_k33")
    r = int(kv.get("r", 1))
    if seed in SEEDS:
        net = RotNet(SEEDS[seed]())
    elif seed in SEEDS_EXTRA:
        net = SEEDS_EXTRA[seed]()
    else:
        print(f"未知 seed={seed}；可选 {list(SEEDS) + list(SEEDS_EXTRA)}")
        return
    print(f"== seed={seed}  r={r}  V={len(_ids(net))} ==")
    rep = equivalence_classes(net, r)
    print(f"VF2 等价类数 = {rep['n_classes']}（证书 = "
          f"{len(set(rep['cert_labels']))}）  一致 = {rep['consistent']}")
    print(f"标签 = {rep['labels']}")
    print(f"stable_r = {stable_r(net)}")


if __name__ == "__main__":
    main(sys.argv[1:])
