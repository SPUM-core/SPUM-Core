# -*- coding: utf-8 -*-
"""l0_locality.py — L0 局部性审计：把 **L0-A10** 从「代码审查 + 指针分析」升级为**自动断言**。

架构基准: docs/L0L1L2_Architecture.md §7 阶段六第④项 + §9 断言表 L0-A10

要解决什么
----------
L0-A10 原文：「所有 kernel 访问跳度 ≤ 二阶邻居」，此前判据只是**代码审查 + 指针分析**
（人工结论，不可复算）。本模块把它变成**程序算出来的读数**：对 `l0_gpu.py` 的每个
索引工件，逐元素测量「索引基准实体 → 被访问实体」的**图距离**，与登记界比对。

跳度的精确定义（本模块采用的口径）
--------------------------------
对索引工件 `A`（把「单元 `i`」映射到「图实体 `A[i]`」）：

    hop(A) = max_i  dist( base(i), ent(A[i]) )

  · `base(i)`：单元的**基准节点**。dart 单元的基准 = 它的 owner 节点（`owner_of_dart`）；
    节点单元的基准 = 自身；面环内位置的基准 = **上一个位置的顶点**（单步推进口径）。
  · `ent(x)`：实体 → 节点集合。节点 `v` → `{v}`；dart `d=(u,w)` → `{u, w}`。
  · `dist(x, S) = max_{s∈S} dist(x, s)` —— **取实体所触及的最远节点**。这是最强的局部性
    陈述：访问触及的**每一个**节点都在界内（若它成立，则「最近节点」口径必然成立）。
    若 x 与 S 不在同一连通分量 → `inf`（不可达）。
    （实测：本口径下 `dart_next` 在非三角面出现时确实取到 2，界不是空话；若改用 min
    则最多取到 1，二阶上界永不被触达 ⇒ 断言退化。）

**为什么按「单次访问」而不按「累积遍历」计量**：`enumerate_faces` 的指针跳跃与
`face_vertices` 的面环游走是**显式的 O(log N) / O(环长) 累积算法步骤**，不是隐藏的
局部性假设。A10 约束的是 kernel 的**每一次**内存访问；遍历的每一**步**仍须 ≤ 2 跳。

豁免项（如实登记，不冒充通过）
----------------------------
`face_rep`（面环最小 dart 的**全局归约**）与 `argsort` / `searchsorted` / `bincount`
（全局排序与桶归约）都是 **O(N log N) 的全局通信**，跳度可以任意大 ⇒ 登记为 `exempt`，
**如实报实测值**，不参与判词。这是 L0-A10 的真实边界，必须写明而非掩盖。
注意：`face_rep` 的实测值在这些小图上很小（面环长 3 时 ≤ 1），那是因为**面小**，
不是因为该工件局部 —— 全局性在它的**中间步骤**（指针跳跃 / `unique` / `argsort`），
本审计只覆盖最终工件。

可证伪性（判据不是恒真）
----------------------
`reverse_control()` 构造**跨分量工件**（两分量图上取另一分量的节点）⇒ 距离不可达
（`inf`）⇒ 断言必须**失败**。若它反而通过，说明测量无区分力。

用法
----
  python l0_locality.py selftest=1
  python l0_locality.py seed=icosa nframes=1
"""

import os
import sys
import warnings
from collections import deque, namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
for _sub in ("l0",):
    _p = os.path.join(_SRC, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")            # 屏蔽 cupy 的 CUDA 探测告警（本模块只用 numpy 后端）
from combinatorial_proto import RotNet, SEEDS  # noqa: E402
import l0_gpu as G  # noqa: E402

G.cp = np  # ⚠️ 复用 GPU 索引原语：本模块只跑**纯索引**部分（排序/查找/CSR），无需 CUDA

_Item = namedtuple("_Item", "name base entities declared note")


# ============================================================
# §1 图距离（全源 BFS；L0 图规模小，直接算距离表）
# ============================================================
def node_dist(net):
    """全源最短路距离 → `{u: {v: dist}}`（同分量必在，跨分量缺键 = 不可达）。"""
    dist = {}
    for src in net.ids():
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for w in net.rot[u]:
                if w not in d:
                    d[w] = d[u] + 1
                    q.append(w)
        dist[src] = d
    return dist


def _max_hop(base_nodes, entities, dist):
    """逐单元测跳度 → `(max_hop, argmax_index)`；不可达记 `inf`。

    单元内先取「实体触及节点的最大距离」（严格口径），单元间再取最大。
    实体为单节点时（①②⑤）无歧义；实体为 dart 时取两端点的最远者。
    """
    worst, arg = 0, None
    for i, (b, ents) in enumerate(zip(base_nodes, entities)):
        row = dist.get(b, {})
        for e in ents:
            d = row.get(e)
            if d is None:                    # 跨分量 ⇒ 不可达
                return float("inf"), i
            if d > worst:
                worst, arg = d, i
    return worst, arg


# ============================================================
# §2 索引工件构造（**复用 l0_gpu 的纯索引原语**，不重写一套副本）
# ============================================================
def build_artifacts(net):
    """构造待审计的索引工件（全部来自 `l0_gpu` 的既有原语）。"""
    ids, rot_ptr, rot_idx, deg = G.build_csr(net.rot)
    own = G.owner_of_dart(rot_ptr)
    rev = G.reverse_dart(rot_ptr, rot_idx)
    dn = G.dart_next_index(rot_ptr, rot_idx, rev)
    _fid, n_faces, frep = G.enumerate_faces(rot_ptr, rot_idx, rev)
    fv_ptr, fv_idx, fv_len = G.face_vertices(rot_ptr, rot_idx, rev, frep)
    return {"ids": ids, "rot_ptr": rot_ptr, "rot_idx": rot_idx, "deg": deg,
            "own": own, "rev": rev, "dn": dn, "frep": frep, "n_faces": n_faces,
            "fv_ptr": fv_ptr, "fv_idx": fv_idx, "fv_len": fv_len}


def audit_items(a):
    """→ `_Item` 列表（登记界；`declared=None` = 豁免项，不参与判词）。"""
    nd = int(a["rot_idx"].size)
    own = [int(x) for x in a["own"]]
    tgt = [int(x) for x in a["rot_idx"]]

    def dart_nodes(d):                       # dart → 端点节点集合
        return (own[d], tgt[d])

    items = []
    # ① rot_idx：dart → 目标节点（一阶邻居）
    items.append(_Item("rot_idx（dart→目标节点）", own,
                       [(tgt[d],) for d in range(nd)], 1, "CSR 邻居表：目标即一阶邻居"))
    # ② owner_of_dart：dart → 自身 owner
    items.append(_Item("owner_of_dart（dart→owner）", own,
                       [(own[d],) for d in range(nd)], 0, "自指：搜索定位 owner 段"))
    # ③ reverse_dart：dart → 反向 dart 的端点集
    #    d=(u,v) ⇒ rev[d]=(v,u)，触及 {v,u}；基准 u，最远节点 v 与 u 相邻 ⇒ 界 = 1
    items.append(_Item("reverse_dart（dart→rev 端点）", own,
                       [dart_nodes(int(a["rev"][d])) for d in range(nd)], 1,
                       "rev[d]=(v,u)：基准 u 到两端点的最远距离 = dist(u,v) = 1（u,v 共边）"))
    # ④ dart_next_index：dart → 后继 dart 的端点集 ← **这就是「二阶邻居」的来源**
    items.append(_Item("dart_next_index（dart→next 端点）", own,
                       [dart_nodes(int(a["dn"][d])) for d in range(nd)], 2,
                       "d=(u,v) → (v,w)：w ∈ N(v) ⇒ dist(u,w) ≤ 2（三角面时取 1，"
                       "非三角面时取到 2 —— 实测 t=2 触达）"))
    # ⑤ face_vertices：**面环内单步**（位置 j 的顶点 vs 位置 j−1 的顶点）
    fb, fe = [], []
    for f in range(int(a["fv_len"].size)):
        s, e = int(a["fv_ptr"][f]), int(a["fv_ptr"][f + 1])
        vs = [int(x) for x in a["fv_idx"][s:e]]
        for j, v in enumerate(vs):
            fb.append(v if j == 0 else vs[j - 1])
            fe.append((v,))
    items.append(_Item("face_vertices（面环内单步）", fb, fe, 2,
                       "环上相邻位置由**同一条面边**连接 ⇒ 严格界 = 1；登记 2 为保守上界"
                       "（面环游走本身是显式累积步骤，按单步计量）"))
    # ⑥ face_rep：面环最小 dart 的**全局归约**（豁免）
    #    `enumerate_faces` 的 face_rep = unique(rep)（每面一个最小 dart，按面 id 索引）；
    #    基准取该面的首顶点，实体取归约结果的端点集。
    #    ⚠️ 豁免的理由是**中间步骤**（指针跳跃 `rep[nxt]`、`cp.unique`、`argsort` 都是
    #    全局通信），本审计只能测**最终工件** —— 故实测值小（面环长 3 时 ≤1）**不代表**
    #    该工件局部，只有明确登记为豁免、不参与判词，才不掩盖这一点。
    nf = int(a["n_faces"])
    fb_rep = [int(a["fv_idx"][int(a["fv_ptr"][f])]) for f in range(nf)]
    items.append(_Item("face_rep（环最小 dart 归约）", fb_rep,
                       [dart_nodes(int(a["frep"][f])) for f in range(nf)], None,
                       "豁免：中间步骤（指针跳跃 unique/argsort）为全局通信，跳度可任意大；"
                       "此处实测的是最终工件，值小不代表局部"))
    return items


# ============================================================
# §3 审计与判词
# ============================================================
def audit_locality(net):
    """逐工件测跳度并与登记界比对 → `{"rows", "verdict", "n_fail", "n_exempt"}`。

    · 非豁免项：`ok = (measured ≤ declared)`；
    · 豁免项：`ok = None`（如实报实测值，不参与判词）；
    · `verdict = "pass"` 当且仅当**所有非豁免项** `ok` 为真。
    """
    dist = node_dist(net)
    a = build_artifacts(net)
    rows, n_fail = [], 0
    for it in audit_items(a):
        m, arg = _max_hop(it.base, it.entities, dist)
        if it.declared is None:
            rows.append({"name": it.name, "declared": None, "measured": m,
                         "ok": None, "note": it.note, "argmax": arg})
            continue
        ok = (m <= it.declared)
        n_fail += 0 if ok else 1
        rows.append({"name": it.name, "declared": it.declared, "measured": m,
                     "ok": ok, "note": it.note, "argmax": arg})
    return {"rows": rows, "verdict": "pass" if n_fail == 0 else "fail",
            "n_fail": n_fail,
            "n_exempt": sum(1 for r in rows if r["declared"] is None)}


# ============================================================
# §4 控制实验（证明判据**不是恒真**）
# ============================================================
def reverse_control():
    """**反向控制**：跨分量工件必须被判超界（`inf`）—— 否则测量无区分力。

    构造：两张 icosa 的不交并（两分量），审计工件取「另一分量」的节点 ⇒ 距离不可达。
    """
    r1 = SEEDS["icosa"]()
    n = max(r1) + 1
    rot = dict(r1)
    for v, ws in r1.items():
        rot[v + n] = [w + n for w in ws]
    net = RotNet(rot)
    dist = node_dist(net)
    ids = net.ids()
    other = min(ids)                    # 第一分量的小 id 节点
    far = max(ids)                      # 第二分量的大 id 节点
    m, _ = _max_hop([other], [(far,)], dist)
    return {"reachable": m != float("inf"), "measured": m,
            "declared": 2, "rejected": m > 2,
            "note": "跨分量对不可达 ⇒ 跳度 inf ⇒ 断言必拒（否则判据恒真）"}


def selftest():
    """自检：① 单分量图审计全过；② 反向控制必须被拒；③ 界非空话（二阶上界被实际触达）。"""
    net = RotNet(SEEDS["icosa"]())
    rep0 = audit_locality(net)
    assert rep0["verdict"] == "pass" and rep0["n_fail"] == 0
    # 反向控制：测量有区分力
    rc = reverse_control()
    assert rc["rejected"] and not rc["reachable"]
    # ② 逐帧复测：判词须恒 `pass`（局部性不随演化退化），且各工件实测须落在其登记界内
    import l0_core as m
    c = m.L0Core(m.RotNet(SEEDS["icosa"]()), cap=12, dmin=3,
                 vminus="dense", vplus="any")
    dn_seen, rv_seen, fv_seen = [], [], []

    def _row(rep, key):
        return next(r for r in rep["rows"] if r["name"].startswith(key))["measured"]

    for _ in range(3):
        c.frame()
        rep = audit_locality(c.net)
        assert rep["verdict"] == "pass"
        dn_seen.append(_row(rep, "dart_next_index"))
        rv_seen.append(_row(rep, "reverse_dart"))
        fv_seen.append(_row(rep, "face_vertices"))
    # ③ 界必须被**实际触达**，否则「二阶邻居」是空话：
    #    dart_next 的二阶界只在面为非三角（洞/长面）时取到 —— 实测 t=2 触达。
    assert max(dn_seen) == 2 and all(v <= 2 for v in dn_seen), dn_seen
    #    一阶工件必须**恒为 1**（不是 ≤1 的废话）
    assert set(rv_seen) == {1} and set(fv_seen) == {1}, (rv_seen, fv_seen)
    print("l0_locality selftest: 全部通过")
    print(f"  ① 单分量 icosa：{len(rep0['rows'])} 项工件，非豁免全过、"
          f"豁免 {rep0['n_exempt']} 项（如实报实测值）")
    print(f"  ② 反向控制：跨分量对测得 {rc['measured']} ⇒ 被拒（判据非恒真）")
    print(f"  ③ t=1..3（V={c.net.V()}）：dart_next 逐帧实测 {dn_seen}"
          f"（界 2，被实际触达）；reverse_dart/face_vertices 恒为 {set(rv_seen)}")
    return True


# ============================================================
# §5 CLI
# ============================================================
def _report(net, label=""):
    rep = audit_locality(net)
    print(f"== L0-A10 局部性审计 {label} V={net.V()} E={net.E()} "
          f"判词={rep['verdict']}（非豁免失败 {rep['n_fail']} 项、豁免 {rep['n_exempt']} 项）==")
    print(f"{'工件':<34} {'登记界':>7} {'实测跳度':>9} {'判定':>6}  说明")
    print("-" * 104)
    for r in rep["rows"]:
        dec = "--" if r["declared"] is None else str(r["declared"])
        m = "inf" if r["measured"] == float("inf") else str(r["measured"])
        ok = "豁免" if r["ok"] is None else ("通过" if r["ok"] else "**超界**")
        print(f"{r['name']:<34} {dec:>7} {m:>9} {ok:>6}  {r['note']}")
    print("（豁免项 = 全局归约/排序原语：跳度可任意大，如实报值、不参与判词）")


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
    if "seed" in kv or "nframes" in kv:
        import l0_core as m
        seed = kv.get("seed", "icosa")
        c = m.L0Core(m.RotNet(SEEDS[seed]()), cap=int(kv.get("cap", 12)),
                     dmin=int(kv.get("dmin", 3)), vminus="dense", vplus="any")
        for t in range(int(kv.get("nframes", 1)) + 1):
            _report(c.net, f"seed={seed} t={t}")
            if t < int(kv.get("nframes", 1)):
                c.frame()
        rc = reverse_control()
        print(f"== 反向控制（跨分量）== 实测跳度 {rc['measured']}、"
              f"可达={rc['reachable']}、被拒={rc['rejected']}")
        return
    print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
