# -*- coding: utf-8 -*-
"""l0_kissing.py — 从「关系定义存在」涌现 12（接吻数）。

问题
----
一个空间粒子，最多能被多少个**同尺度**邻居同时确认？

方法（第三代：定规则 → 跑演化 → 看涌现）
--------------------------------------
不预设 12，不预设三维，不预设球/半径/角度/π/坐标。只从「关系定义存在」
出发，**遍历所有可能**，看哪一个可行者能存活；极大者即答案。

规则（全部只取自「关系定义存在」）
--------------------------------
  A1 关系相互     ：确认是双向的 —— 边无向。
  A2 存在即被确认 ：每节点 deg ≥ 2 —— 无孤立、无悬挂（否则关系未被确认）。
  A3 同尺度等价   ：同一壳层的空间粒子等价 ⇒ 度数相同（均匀）。
                    ⚠️ 这是 **r=0 等价**（只看关系量 deg），必要不充分——
                    同度节点的 r-邻域未必同构（例：三棱柱与 K₃,₃ 全 deg=3，
                    r=1 起分两类）。精确定义见 `l0_equivalence.py`（根化
                    r-邻域子图同构 + VF2 交叉验证）；本模块的 A3 是其最粗情形。★
                    ⚠️ **承重假设**（L0-B14 裁决，2026-09-15）：A3 不能从
                    纯组合公理导出——`probe_r1.py` 系统检验了 R1（禁止 K₆）、
                    r-邻域局部不变量、Euler+均匀性三条路径，结论是无局部规则
                    能区分 deg≤5 与 deg≥6（双锥极点与正二十面体顶点的 1-邻域
                    结构同构）。A3 是把 deg≤5 从「假设」变成「结论」的**唯一**
                    组合路径，代价是它自身必须由假设升为公理。无 A3 则 Euler
                    只约束均值 ⟨deg⟩<6，不约束个体。
  A4 闭合         ：壳层自身闭合，每条边恰属两面 ⇒ 球面三角剖分（χ=2、无洞）。★
  A5 极大         ：遍历所有可能，取可行之极大者。

推导（纯整数计数；无浮点、无 π、无角度、无半径）
----------------------------------------------
  三角饱和（A4）  ⇒ 3F = 2E
  均匀（A3）      ⇒ 2E = k·r          （k 个粒子，每个 deg = r）
  Euler（A4）     ⇒ k − E + F = 2
  ─────────────────────────────────────────────
  ⇒ k(6 − r) = 12  ⇒  (6 − r) | 12，r ≥ 3
  ⇒ r ∈ {3, 4, 5}  ⇒  k ∈ {4, 6, 12}
  ⇒ **极大 k = 12**  ← 涌现，不是预设

「6」来自 Euler/三角饱和的整数代数（E = 3V − 6），**不是几何度数**。

构造验证（实际旋转系统 + audit 复核）
------------------------------------
  k=4  → tetra（全 deg=3）    k=6 → octa（全 deg=4）    k=12 → icosa（全 deg=5）

反向控制（证明 12 是「均匀 + 闭合」的产物，不是任意限制）
-------------------------------------------------------
  (a) 去掉均匀 ⇒ k=13 可行（细分 icosa 的一个面）⇒ 12 非普适上界
  (b) 去掉闭合 ⇒ 开放贴片 V 随 R 无界增长    ⇒ 闭合是 12 的必要条件
  (c) k=13 无均匀解 ⇒ 穷举表内缺席（不是"没试"）

进一步改规则（探索区，见 selftest ⑥⑦⑧）
--------------------------------------
  ⑥ 变 χ（闭合的拓扑类） ⇒ **Σ(6−deg) = 6χ**
       χ= 2 球面 → 12 = 6χ（k∈{4,6,12}）
       χ= 0 环面 → 强制 r=6（平坦）、k 无界（构造：六边形环面 L=3）
       χ< 0 曲面 → r>6（负曲率）
     ⇒ 「12」不是魔法数，就是 **6χ**；那个 6 是**平坦度数**，不是几何假设。
  ⑦ 变饱和（面型） ⇒ **Σ(6−deg) = 6χ + 2(2E−3F)**
       仅「全三角饱和」时退化为 6χ；开口/多边形面额外贡献 2(2E−3F)
       （开放 patch = 30）⇒ 12 不是自动的，饱和是它的必要条件。
  ⑧ 松 A3（均匀 ⇒ **度数差 ≤ 1**） ⇒ k 不再被钉死
       a 个度 r + b 个度 r+1；Σ(6−deg)=6χ 仍成立，但**节点数 k 无界**：
         {3,4} 族：3a+2b=12 ⇒ k∈[4,6]      （双锥 n=3..4）
         {4,5} 族：2a+  b=12 ⇒ k∈[6,12]     （双锥 n=4..5）
         {5,6} 族：  a    =12 ⇒ k∈[12,∞)     ★ a ≡ 12 恒成立
     三族的**端点**（a=0 或 b=0）恰是 ② 的均匀解 {4,6,12}——{5,6} 族 k=12 行
     即 (a=12, b=0) ⇒ 无 deg-6 节点 ⇒ 就是均匀 icosa（松弛族∩严格族）。
     存活的不变量是 **Σ(6−deg)=12**（总曲率缺陷），不是「12 个节点」。
     「12 个节点」只是该不变量在 {5,6} 族里的**特例读数**（其余节点皆平坦
     deg=6 ⇒ 缺陷数 = 节点数）。⚠️ 计数可行 ≠ 可实现（大 k 为开放项）。
  ⑨ {5,6} 族 **k=13 行不可实现**（自包含证明，无外部定理；见 selftest ⑨ / CLI k13=1）
       计数可行：V=13 E=33 F=22 χ=2 Σ(6−deg)=12  ← 行 (a,b)=(12,1)
       但骨架被唯一确定，且该骨架的**面型计数自相矛盾**：
         deg-6 节点 v 唯一 ⇒ 邻接 6-环 u0..u5（各 deg=5，相邻者相切）
           ⇒ 每 u 除 v 与环上两邻外余 2 条边；环序迫使 A_i = B_{i+1}
           ⇒ 恰 6 个"外节点" x，且 u_i ~ x_{i-1}, x_i；x_i ~ u_i, u_{i+1}
        割开 u 环 ⇒ 外侧是盘 V=12 E=27 F=16；x 盘内度 5（3 条非 u 边全在 X 内）
           ⇒ X 上 3-正则（9 条边）⇒ X 的三角数 ≤ 2（6 顶点 3-正则标号图 70 个
             全枚举，同构仅两类——三角棱柱 C₃×K₂ 三角数 2、K₃,₃ 三角数 0）
        面型计数：无 uuu；uuw 恰 6 ⇒ n2+n3 = 10；顶点-面关联
           6·1 + 2n2 + 3n3 = 6×5 = 30 ⇒ n3 = 4
        而 www 面必是 X 的三角 ⇒ n3 ≤ 2 < 4  ⇒ **矛盾，行不存在** ∎
       反向控制：k=12（icosa）无 deg-6 节点 ⇒ 推导不适用 ⇒ 可实现（② 已构造）；
                 k=13 的**非均匀**实现存在（细分 icosa 一面）⇒ 排除只针对 {5,6} 行。
       ⚠️ k ≥ 14（b ≥ 2）未证——开放项，不得由本行外推。

用法
----
  python l0_kissing.py selftest=1
  python l0_kissing.py sweep=1 K=16       # χ=2 穷举
  python l0_kissing.py chi=1              # 改 χ：看 Σ(6−deg)=6χ
  python l0_kissing.py quasi=1 K=32       # 松 A3：度数差 ≤ 1
  python l0_kissing.py k13=1              # {5,6} 族 k=13 行：不可实现证明
  python l0_kissing.py k=12
"""

import itertools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from combinatorial_proto import (  # noqa: E402
    RotNet, rotation_from_faces, audit, seed_tetra, seed_icosa, seed_patch,
    seed_bipyramid,
)

# 规则表（只取自「关系定义存在」；供打印与审计引用）
RULES = (
    ("A1", "关系相互",     "确认是双向的 —— 边无向"),
    ("A2", "存在即被确认", "每节点 deg ≥ 2 —— 无孤立、无悬挂"),
    ("A3", "同尺度等价",   "同壳层粒子等价 ⇒ 度数相同（均匀）= r=0 等价（必要不充分，见 l0_equivalence.py）"),
    ("A4", "闭合",         "每条边恰属两面 ⇒ 球面三角剖分（χ=2、无洞）"),
    ("A5", "极大",         "遍历所有可能，取可行之极大者"),
)


# ============================================================
# §1 种子：k=6 的正八面体（k=4 / k=12 复用既有种子）
# ============================================================
def seed_octa():
    """正八面体（闭合球面三角剖分）：V=6 E=12 F=8 χ=2，全 deg=4。

    顶点：0=顶 T，1=底 B，2..5=赤道环 a_i（纯 id，无坐标）。
    """
    T, B = 0, 1
    A = [2, 3, 4, 5]
    faces = []
    for i in range(4):
        faces.append((T, A[i], A[(i + 1) % 4]))
        faces.append((B, A[(i + 1) % 4], A[i]))
    return rotation_from_faces(faces)


SHELL_SEEDS = {4: seed_tetra, 6: seed_octa, 12: seed_icosa}


def seed_torus6(L=3):
    """平坦六边形环面（χ=0）：L×L 周期格，**全 deg=6**。

    这是「零曲率」闭合壳层——推导式 `k(6−r)=6χ` 在 χ=0 时强制 `r=6`：
    公式里那个 6 正是「平坦度数」，不是几何假设。
    """
    dirs = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]
    keys = [(q, r) for q in range(L) for r in range(L)]
    idx = {k: i for i, k in enumerate(keys)}
    rot = {}
    for k in keys:
        q, r = k
        rot[idx[k]] = [idx[((q + dq) % L, (r + dr) % L)] for (dq, dr) in dirs]
    return rot



# ============================================================
# §2 判据：闭合 / 均匀（纯整数检查，复用 audit）
# ============================================================
def degrees(net):
    return sorted(net.deg(v) for v in net.ids())


def is_closed_shell(net):
    """A4：χ=2 且无洞（所有面皆为三角）——"无边界"的正确判据。

    ⚠️ `χ=2` **单独不蕴含无边界**：开放贴片的外边界在旋转系统里被一个
    k 边形**面**包住，故其 χ 亦为 2。无边界必须看"有无非三角面"。
    """
    a = audit(net)
    return a["chi"] == 2 and a["n_holes"] == 0


def is_uniform(net):
    """A3：所有节点度数相同。"""
    d = degrees(net)
    return bool(d) and len(set(d)) == 1


def is_uniform_closed_shell(net):
    return is_closed_shell(net) and is_uniform(net)


# ============================================================
# §3 遍历所有可能：穷举 (k, r) 并施加 A3+A4
# ============================================================
def enumerate_uniform(chi=2, K=16, rmin=3):
    """遍历所有可能的 (k, r)：k 个粒子、全 deg=r 的闭合三角剖分可定向曲面。

    判据全部是整数计数（三角饱和 + Euler），无浮点：
      · 2E = k·r 须为偶      · 3F = 2E（三角饱和）  · χ = k−E+F（闭合拓扑类）

    ⇒ k(6 − r) = 6χ。χ=2（球面）时即 k(6−r)=12。
    """
    rows = []
    for k in range(1, K + 1):
        for r in range(rmin, k):          # 简单图：r ≤ k−1
            if (k * r) % 2:
                continue
            E = k * r // 2
            if (2 * E) % 3:
                continue
            F = 2 * E // 3
            if k - E + F != chi:
                continue
            rows.append({"k": k, "r": r, "E": E, "F": F})
    return rows


def enumerate_shells(K=16, rmin=3):
    """球面（χ=2）特例：均匀闭合壳层。"""
    return enumerate_uniform(chi=2, K=K, rmin=rmin)


def chi_sweep(K=16):
    """改一条规则（闭合的拓扑类 χ），看「拓扑常数」怎么变：Σ(6−deg) = 6χ。"""
    return {chi: enumerate_uniform(chi=chi, K=K) for chi in (2, 0, -2, -4)}


def enumerate_quasi_uniform(chi=2, K=32, rmin=3):
    """**松弛 A3**（全等 → 度数差 ≤ 1）：穷举 (k, r, a, b)。

    a 个节点度 r，b 个节点度 r+1（差 1）；a+b = k；
    Σ(6−deg) = 6χ；三角饱和 ⇒ 3F=2E（故 2E 须为偶且被 3 整除）。
    判据纯整数。χ=2 时结果：
      deg∈{5,6}：a ≡ 12 **恒成立**（k = 12+b，b ≥ 0）★
      deg∈{4,5}：2a + b = 12 ⇒ k ∈ [6, 12]
      deg∈{3,4}：3a + 2b = 12 ⇒ k ∈ [4, 6]
      deg∈{6,7}：−b = 12 ⇒ 无解
    """
    rows = []
    for k in range(4, K + 1):
        for r in range(rmin, k):
            for b in range(0, k + 1):
                a = k - b
                if a * (6 - r) + b * (5 - r) != 6 * chi:
                    continue
                s2 = a * r + b * (r + 1)          # = 2E
                if s2 % 2 or s2 % 3:
                    continue
                rows.append({"k": k, "r": r, "a": a, "b": b, "E": s2 // 2})
    return rows


def quasi_families(chi=2, K=32):
    """把松弛穷举按 r 分组：`{r: [(k, a, b), ...]}`（a 个度 r + b 个度 r+1）。"""
    fam = {}
    for row in enumerate_quasi_uniform(chi=chi, K=K):
        fam.setdefault(row["r"], []).append((row["k"], row["a"], row["b"]))
    return {r: sorted(v) for r, v in fam.items()}


def construct_shell(k):
    """构造 k 个粒子的均匀闭合壳层；不可构造 ⇒ None。"""
    fn = SHELL_SEEDS.get(k)
    return RotNet(fn()) if fn else None


def subdivide_face(net):
    """细分一个真三角面（新增 deg-3 粒子）——用于反向控制 (a)：破坏均匀性。"""
    for c in net.faces():
        vs = net.face_vertices(c)
        if len(vs) == 3 and net.tri_is_face(vs[0], vs[1], vs[2]):
            return net.cone_tri_face(vs[0], vs[1], vs[2])
    return None


# ============================================================
# §3b {5,6} 族的 k=13 行：计数可行，但不可实现（自包含证明）
# ============================================================
def max_triangles_3reg6():
    """6 顶点 3-正则**简单图**里最多能有几个三角（穷举完备）。

    3-正则 ⇒ 9 条边；C(15,9) = 5005 个候选全枚举，留 3-正则者并数三角。
    标号图共 70 个，**同构只有两类**：三角棱柱 C₃×K₂ 与 K₃,₃，
    三角数分别 2 与 0 ⇒ 上界 2（与标签无关，穷举即完备）。
    """
    pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    n_graph, max_tri = 0, 0
    for comb in itertools.combinations(range(15), 9):
        es = {pairs[c] for c in comb}
        deg = [0] * 6
        for i, j in es:
            deg[i] += 1
            deg[j] += 1
        if any(d != 3 for d in deg):
            continue
        n_graph += 1
        tri = sum(1 for a, b, c in itertools.combinations(range(6), 3)
                  if (a, b) in es and (a, c) in es and (b, c) in es)
        max_tri = max(max_tri, tri)
    return n_graph, max_tri


def exclude_k13_row():
    """{5,6} 族 **k=13 行**（a=12, b=1）的不可实现性判定。

    前提：13 节点、闭合无洞、全三角的剖分，度数 = 12 个 5 + 1 个 6。
    下面每一步都是**必然而非假设**；最后一步矛盾 ⇒ 该行不存在。
    全过程只用计数恒等式与环序（无外部定理、无几何、无 π）。

      ① 计数自洽：V=13 E=33 F=22 χ=2 Σ(6−deg)=12（可行 ≠ 可实现）
      ② 唯一 deg-6 节点 v ⇒ 邻接环 u0..u5 长 6、相邻者相切；每 u 全 deg=5
      ③ 环序 u_i=(v,u_{i+1},A_i,B_i,u_{i-1}) 且 △(u_i,u_{i+1},A_i)
         ⇒ A_i 与 u_{i+1} 相切 ⇒ A_i = B_{i+1} ⇒ 恰 6 个外节点 x，
           u_i ~ x_{i-1}, x_i；x_i ~ u_i, u_{i+1}（u-x 边 12 条）
      ④ 割开 u 环 ⇒ 外侧是盘 V=12 E=27 F=16；x 盘内度 5（3 条非 u 边全在 X）
         ⇒ X 上 3-正则（9 条 x-x 边）
      ⑤ 面型：无 uuu；uuw 恰 6 ⇒ n2+n3 = F_disk − 6 = 10
      ⑥ 顶点-面关联：6·1 + 2n2 + 3n3 = 6×5 = 30（x 盘内度 5 ⇒ 各 5 个环绕面）
         ⇒ n3 = 4
      ⑦ 但 www 面必是 X 的三角 ⇒ n3 ≤ max_tri(X) = 2 < 4  矛盾 ∎
    """
    k, a, b = 13, 12, 1
    deg_sum = a * 5 + b * 6
    E = deg_sum // 2
    F = 2 * E // 3
    S = a * (6 - 5) + b * (6 - 6)                 # Σ(6−deg)

    ring = 6                                      # ② deg(v) = r+1 = 6 ⇒ link(v) 是 6-环
    n_x = ring                                    # ③ 外节点数 = 环长
    ux_edges = 2 * ring                           # 每个 u 余 2 条边，另一端在 X

    disk_V = ring + n_x
    disk_F = F - ring                             # 含 v 的面恰 ring 个
    disk_E = disk_V + disk_F - 1                  # 盘：V − E + F = 1
    xx_edges = disk_E - ring - ux_edges           # 盘内边 = 环 + u-x + x-x
    n_graph, max_tri = max_triangles_3reg6()

    n_uuw = ring                                  # △(u_i,u_{i+1},x_i) 恰 ring 个
    n2_n3 = disk_F - n_uuw
    x_faces = n_x * 5                             # x 盘内度 5 ⇒ 各环绕 5 个面
    n3 = (x_faces - n_uuw) - 2 * n2_n3            # 6·1 + 2n2 + 3n3 = x_faces
    n2 = n2_n3 - n3
    return {
        "V": k, "E": E, "F": F, "chi": k - E + F, "S": S,
        "ring": ring, "n_x": n_x, "ux": ux_edges, "xx": xx_edges,
        "disk_V": disk_V, "disk_E": disk_E, "disk_F": disk_F,
        "n_uuw": n_uuw, "n2": n2, "n3": n3,
        "x_faces": x_faces, "n_graph": n_graph, "max_tri": max_tri,
    }


# ============================================================
# §4 自检（正向 + 三重反向控制）
# ============================================================
def verify():
    ok = True
    line = "-" * 78
    print("=" * 78)
    print("[l0_kissing] 从「关系定义存在」涌现 12 —— 遍历所有可能，看谁存活")
    print("=" * 78)

    # ---- ① 遍历所有可能 ----
    rows = enumerate_shells(K=16)
    ks = sorted({r["k"] for r in rows})
    p1 = ks == [4, 6, 12]
    ok &= p1
    print(f"① 遍历 (k,r) ∈ [1..16]×[3..k]：可行者 {len(rows)} 个")
    for r in rows:
        print(f"      k={r['k']:2d} 全deg={r['r']}  E={r['E']:3d} F={r['F']:2d} "
              f"χ={r['k'] - r['E'] + r['F']}")
    print(f"   可行 k 集合 = {ks}   期望 [4, 6, 12]   [{'OK' if p1 else 'FAIL'}]")
    print(f"   ▶ 极大可行 k = {max(ks)}  ← 涌现（不是预设）")

    # ---- ② 构造验证三个可行壳 ----
    print(line)
    print("② 构造验证（实际旋转系统 + audit 复核）")
    want = {4: (4, 6, 4, 3), 6: (6, 12, 8, 4), 12: (12, 30, 20, 5)}
    for k in (4, 6, 12):
        net = construct_shell(k)
        a = audit(net)
        got = (a["V"], a["E"], a["F"], a["S"])
        w = (want[k][0], want[k][1], want[k][2], 6 * want[k][0] - 2 * want[k][1])
        good = (got == w and is_uniform_closed_shell(net)
                and a["chi"] == 2 and a["n_holes"] == 0
                and set(degrees(net)) == {want[k][3]})
        ok &= good
        print(f"   k={k:2d}  V,E,F,Σ(6−deg) = {got}  期望 {w}  "
              f"全deg={set(degrees(net))}  均匀闭合={is_uniform_closed_shell(net)}  "
              f"[{'OK' if good else 'FAIL'}]")

    # ---- ③ 反向控制 a：去掉均匀 ⇒ k>12 可行 ----
    print(line)
    print("③ 反向控制 a（去掉 A3 均匀）：细分 icosa 一个面 ⇒ k=13，闭合但非均匀")
    net = RotNet(seed_icosa())
    subdivide_face(net)
    a = audit(net)
    ca = (a["V"] == 13 and is_closed_shell(net) and not is_uniform(net))
    ok &= ca
    print(f"   V={a['V']} E={a['E']} F={a['F']} χ={a['chi']} 洞={a['n_holes']}  "
          f"deg 直方={a['hist']}")
    print(f"   闭合={is_closed_shell(net)}  均匀={is_uniform(net)}  "
          f"⇒ k=13 可行 ⇒ 12 不是普适上界（只是「均匀饱和」下的极大）  "
          f"[{'OK' if ca else 'FAIL'}]")

    # ---- ④ 反向控制 b：去掉闭合 ⇒ 无界 ----
    print(line)
    print("④ 反向控制 b（去掉 A4 闭合）：开放贴片 V 随 R 增长 ⇒ 无界")
    vs = [RotNet(seed_patch(R)).V() for R in (1, 2, 3, 4)]
    cb = all(vs[i] < vs[i + 1] for i in range(len(vs) - 1))
    ok &= cb
    print(f"   patch R=1..4 ⇒ V = {vs}   单调增长={cb}  "
          f"⇒ 闭合是 12 的必要条件  [{'OK' if cb else 'FAIL'}]")

    # ---- ⑤ 反向控制 c：k=13 无均匀解 ----
    print(line)
    print("⑤ 反向控制 c：k=13 在穷举表内缺席（不是「没试」）")
    cc = 13 not in ks
    ok &= cc
    print(f"   13 ∈ 可行集 = {13 in ks}   期望 False  [{'OK' if cc else 'FAIL'}]")

    # ---- ⑥ 改规则：变闭合的拓扑类 χ ----
    print(line)
    print("⑥ 改规则（变 χ）：由 Euler k−E+F=χ 与三角饱和 ⇒ Σ(6−deg) = 6χ")
    sweep = chi_sweep(K=16)
    chi2 = sorted({r["k"] for r in sweep[2]})
    zero = sweep[0]
    zero_rs = {r["r"] for r in zero}
    zero_ks = sorted({r["k"] for r in zero})
    neg = [(r["k"], r["r"]) for r in sweep[-2]]
    d_ok = (chi2 == [4, 6, 12] and zero_rs == {6} and min(zero_ks) == 7
            and bool(neg) and min(r for _, r in neg) > 6)
    ok &= d_ok
    print(f"   χ= 2（球面）：k={chi2}   r={sorted({r['r'] for r in sweep[2]})}"
          f"   ⇒ Σ(6−deg)=12 = 6χ")
    print(f"   χ= 0（环面）：k={zero_ks}   r={sorted(zero_rs)}"
          f"   ⇒ 强制 r=6（平坦），k 无界")
    print(f"   χ=−2（双环）：(k,r)={neg}   ⇒ r>6（负曲率）")
    print(f"   ▶ 洞察：「12」不是魔法数，就是 **6χ**；χ=0 把公式里的 6"
          f"逼回它本来的身份——平坦度数  [{'OK' if d_ok else 'FAIL'}]")
    net = RotNet(seed_torus6(3))
    a = audit(net)
    t_ok = (a["chi"] == 0 and a["n_holes"] == 0 and set(degrees(net)) == {6}
            and is_uniform(net))
    ok &= t_ok
    print(f"   构造 六边形环面 L=3：V={a['V']} E={a['E']} F={a['F']} χ={a['chi']} "
          f"全deg={set(degrees(net))} 闭合={a['n_holes'] == 0}  "
          f"[{'OK' if t_ok else 'FAIL'}]")

    # ---- ⑦ 改规则：变饱和（面型）----
    print(line)
    print("⑦ 改规则（变 A4 的「全三角」饱和）：Σ(6−deg) = 6χ + 2(2E−3F)")
    s7 = True
    for tag, net in (("tetra  闭合饱和", RotNet(seed_tetra())),
                     ("icosa  闭合饱和", RotNet(seed_icosa())),
                     ("patch  开放(12边形面)", RotNet(seed_patch(2)))):
        a = audit(net)
        S = a["S"]
        sixchi = 6 * a["chi"]
        pred = sixchi + 2 * (2 * a["E"] - 3 * a["F"])
        good = (S == pred)
        s7 &= good
        print(f"   {tag:22s} χ={a['chi']}  Σ(6−deg)={S:3d}   6χ={sixchi:3d}   "
              f"公式={pred:3d}  [{'OK' if good else 'FAIL'}]")
    ok &= s7
    print("   ▶ 12 = 6χ **仅在「全三角饱和」下**成立；非饱和面（开口/多边形）"
          "额外贡献 2(2E−3F) ⇒ 12 不是自动的")

    # ---- ⑧ 松 A3：均匀 ⇒ 度数差 ≤ 1 ----
    print(line)
    print("⑧ 松 A3（均匀 ⇒ 度数差 ≤ 1）：a 个度 r + b 个度 r+1，判据 Σ(6−deg)=6χ")
    fam = quasi_families(chi=2, K=32)
    ks34 = [k for k, _, _ in fam.get(3, [])]
    ks45 = [k for k, _, _ in fam.get(4, [])]
    ks56 = [k for k, _, _ in fam.get(5, [])]
    a56 = sorted({a for _, a, _ in fam.get(5, [])})
    e_ok = (sorted(fam.keys()) == [3, 4, 5]
            and ks34 == [4, 5, 6]
            and ks45 == list(range(6, 13))
            and ks56 == list(range(12, 33))
            and a56 == [12])
    ok &= e_ok
    print(f"   r=3 族 {{3,4}}：k={ks34}      3a+2b=12  ⇒ k∈[4,6]")
    print(f"   r=4 族 {{4,5}}：k={ks45}      2a+  b=12  ⇒ k∈[6,12]")
    print(f"   r=5 族 {{5,6}}：k={ks56[0]}..{ks56[-1]}（{len(ks56)} 个）"
          f"   a ≡ {a56} 恒成立  ★")
    print(f"   r≥7 无解（两系数同负）⇒ 松弛后 k **不再被钉死**，"
          f"k=12 只是 {{5,6}} 族的起点  [{'OK' if e_ok else 'FAIL'}]")

    # {5,6} 族的 k=12 行 = (a=12, b=0)：**没有** deg-6 节点 ⇒ 12 个节点全 deg=5，
    # 正是均匀 icosa（松弛族与严格族在此重合；② 已构造过同一个壳）。
    row12 = [r for r in fam.get(5, []) if r[0] == 12]
    net12 = RotNet(seed_icosa())
    a12 = audit(net12)
    d12 = degrees(net12)
    e12 = (row12 == [(12, 12, 0)]
           and a12["V"] == 12 and a12["E"] == 30 and a12["F"] == 20
           and a12["chi"] == 2 and a12["n_holes"] == 0 and a12["S"] == 12
           and set(d12) == {5} and is_uniform(net12)
           and sum(1 for x in d12 if x == 5) == 12)
    ok &= e12
    print(f"   构造 {{5,6}} k=12 行 {row12[0] if row12 else '—'} ⇒ 无 deg-6 节点：")
    print(f"     V={a12['V']} E={a12['E']} F={a12['F']} χ={a12['chi']} "
          f"洞={a12['n_holes']} 度直方={a12['hist']} Σ(6−deg)={a12['S']} "
          f"均匀={is_uniform(net12)}  "
          f"⇒ 该实现即**均匀 icosa**，与 ② 的 k=12 同一壳  [{'OK' if e12 else 'FAIL'}]")
    print("   ▶ 三族的**端点**（a=0 或 b=0）恰是 ② 的均匀解 {4,6,12}；"
          "内部成员（a,b 皆 >0）才是松弛**新增**的实现。")

    b_ok = True
    for n in (3, 4, 5):
        net = RotNet(seed_bipyramid(n))
        a = audit(net)
        d = degrees(net)
        good = (a["chi"] == 2 and a["n_holes"] == 0 and a["S"] == 12
                and max(d) - min(d) <= 1)
        b_ok &= good
        print(f"   构造 双锥 n={n}: V={a['V']} E={a['E']} F={a['F']} χ={a['chi']} "
              f"度直方={a['hist']} Σ(6−deg)={a['S']} 均匀={is_uniform(net)}  "
              f"[{'OK' if good else 'FAIL'}]")
    ok &= b_ok
    print("   ▶ 存活的不变量是 **Σ(6−deg)=12**（总曲率缺陷），不是「12 个节点」；")
    print("     「12」仅在 {5,6} 族里读作「五配位节点数 a」（其余节点平坦 ⇒ 缺陷数=节点数）。")
    print("     ⚠️ 计数可行 ≠ 可实现：除 n=3/4/5 双锥外的成员未构造验证（开放项）。")

    # ---- ⑨ {5,6} 族的 k=13 行：计数可行，但不可实现（自包含证明）----
    print(line)
    print("⑨ {5,6} 族 k=13 行 (a=12, b=1)：计数可行 ⇒ 骨架唯一 ⇒ 面型计数矛盾")
    r13 = [r for r in fam.get(5, []) if r[0] == 13]
    d = exclude_k13_row()
    f_ok = (r13 == [(13, 12, 1)]
            and d["V"] == 13 and d["E"] == 33 and d["F"] == 22
            and d["chi"] == 2 and d["S"] == 12)
    ok &= f_ok
    print(f"   ① 穷举表**内有**该行 {r13}（不是「缺席」）⇒ 计数自洽："
          f"V={d['V']} E={d['E']} F={d['F']} χ={d['chi']} Σ(6−deg)={d['S']}  "
          f"[{'OK' if f_ok else 'FAIL'}]")
    print(f"   ② 唯一 deg-6 节点 v ⇒ 邻接 6-环 u0..u5（环长 {d['ring']}），每 u 全 deg=5")
    print(f"   ③ 环序（A_i = B_{{i+1}}）⇒ 恰 {d['n_x']} 个外节点 x；u-x 边 {d['ux']} 条")
    print(f"   ④ 割开 u 环 ⇒ 外侧是盘 V={d['disk_V']} E={d['disk_E']} F={d['disk_F']}；"
          f"x 盘内度 5 ⇒ X 上 3-正则（{d['xx']} 条 x-x 边）")
    print(f"   ⑤ 面型：无 uuu；uuw 恰 {d['n_uuw']} ⇒ n2+n3 = {d['disk_F']} − "
          f"{d['n_uuw']} = {d['n2'] + d['n3']}")
    print(f"   ⑥ 顶点-面关联：6·1 + 2·{d['n2']} + 3·{d['n3']} = {d['x_faces']} "
          f"⇒ n2={d['n2']} n3={d['n3']}")
    g_ok = (d["n3"] == 4 and d["max_tri"] == 2 and d["n_graph"] == 70
            and d["n3"] > d["max_tri"])
    ok &= g_ok
    print(f"   ⑦ 而 www 面必是 X 的三角 ⇒ n3 ≤ max_tri(6 顶点 3-正则图) = "
          f"{d['max_tri']}（穷举 {d['n_graph']} 个标号图，同构仅三角棱柱/K₃,₃）  "
          f"{d['n3']} > {d['max_tri']}  ⇒ **矛盾 ∎ 行不存在**  "
          f"[{'OK' if g_ok else 'FAIL'}]")

    net = RotNet(seed_icosa())
    h_ok = 6 not in degrees(net)                  # k=12 行：无 deg-6 ⇒ 推导不适用
    net = RotNet(seed_icosa())
    subdivide_face(net)
    a = audit(net)
    i_ok = (a["V"] == 13 and is_closed_shell(net) and not is_uniform(net))
    ok &= (h_ok and i_ok)
    print(f"   反向控制 ① k=12 行 (a=12,b=0) 无 deg-6 节点 = {h_ok}"
          f" ⇒ 推导不适用（② 已构造 icosa）")
    print(f"   反向控制 ② k=13 的**非均匀**实现存在（细分 icosa 一面）= {i_ok}"
          f" ⇒ 排除只针对 {{5,6}} 行，不是「k=13 不可能」")
    print("   ⚠️ k ≥ 14（b ≥ 2）未证——开放项，不得由本行外推。")

    print(line)
    print(f"l0_kissing selftest: {'全部通过' if ok else '存在失败项'}")
    return ok


# ============================================================
# §5 CLI
# ============================================================
def main():
    kv = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            kv[k] = v

    if kv.get("selftest", "0") == "1":
        verify()
        return

    K = int(kv.get("K", 16))
    if kv.get("sweep", "0") == "1":
        print("=" * 78)
        print(f"[遍历] 穷举所有可能 (k,r) ∈ [1..{K}]×[3..k]，判据 = 均匀 + 闭合")
        print("=" * 78)
        rows = enumerate_shells(K=K)
        for k in range(1, K + 1):
            hit = [r for r in rows if r["k"] == k]
            if hit:
                r = hit[0]
                print(f"  k={k:2d}  全deg={r['r']}  E={r['E']:3d} F={r['F']:2d}  "
                      f"◆ 可行（均匀闭合三角剖分）")
        ks = sorted({r["k"] for r in rows})
        print("-" * 78)
        print(f"  可行 k 集合 = {ks}    极大 = {max(ks)}")
        print("  说明：k(6−r)=12 ⇒ (6−r)|12 ⇒ r∈{3,4,5} ⇒ k∈{4,6,12}。")
        print("        6 来自 Euler/三角饱和的整数代数（E=3V−6），非几何度数。")
        return

    if kv.get("chi", "0") == "1":
        print("=" * 78)
        print(f"[改规则] 变闭合的拓扑类 χ ⇒ Σ(6−deg) = 6χ，穷举 (k,r) ∈ [1..{K}]")
        print("=" * 78)
        for chi, rows in chi_sweep(K=K).items():
            ks = sorted({r["k"] for r in rows})
            rs = sorted({r["r"] for r in rows})
            print(f"  χ={chi:2d}：k={ks}  r={rs}")
        print("-" * 78)
        print("  χ= 2 球面  ⇒ Σ=12=6χ     （12 不是魔法数，就是 6χ）")
        print("  χ= 0 环面  ⇒ 强制 r=6（平坦），k 无界")
        print("  χ< 0 曲面  ⇒ r>6（负曲率）")
        net = RotNet(seed_torus6(3))
        a = audit(net)
        print(f"  实例 六边形环面 L=3：V={a['V']} E={a['E']} F={a['F']} "
              f"χ={a['chi']} 全deg={set(degrees(net))}")
        return

    if kv.get("quasi", "0") == "1":
        print("=" * 78)
        print(f"[松 A3] 度数差 ≤ 1 ⇒ a 个度 r + b 个度 r+1，Σ(6−deg)=6χ，"
              f"k ≤ {K}")
        print("=" * 78)
        for r, rows in quasi_families(chi=2, K=K).items():
            ks = [k for k, _, _ in rows]
            print(f"  r={r} 族 {{{r},{r + 1}}}：k={ks[0]}..{ks[-1]}（{len(ks)} 个）"
                  f"   解 ({r}a + {5 - r}b = 12)")
        print("-" * 78)
        print("  存活的不变量：Σ(6−deg) = 12（总曲率缺陷），而非「12 个节点」。")
        print("  {5,6} 族：a ≡ 12 —— 12 = 五配位节点数（富勒烯式读数）。")
        print("  {5,6} 族 k=12 行 =(a=12,b=0) ⇒ 均匀 icosa（松弛族∩严格族）。")
        print("  {4,5}/{3,4} 族：12 被 a,b 分摊 ⇒ 节点数不再等于 12。")
        print("  ⚠️ 计数可行 ≠ 可实现：表中大 k 成员的存在性未构造验证（开放项）。")
        return

    if kv.get("k13", "0") == "1":
        print("=" * 78)
        print("[{5,6} 族 k=13 行] a=12（deg5）+ b=1（deg6）：计数可行，但不可实现")
        print("=" * 78)
        d = exclude_k13_row()
        print(f"  ① 计数自洽：V={d['V']} E={d['E']} F={d['F']} χ={d['chi']} "
              f"Σ(6−deg)={d['S']}   ← 行 (a,b)=(12,1)，与准均匀穷举表一致")
        print(f"  ② 唯一 deg-6 节点 v ⇒ 邻接 6-环 u0..u5（环长 {d['ring']}），每 u 全 deg=5")
        print(f"  ③ 环序迫使 A_i = B_{{i+1}} ⇒ 恰 {d['n_x']} 个外节点 x，"
              f"u-x 边 {d['ux']} 条；x_i ~ u_i, u_{{i+1}}")
        print(f"  ④ 割开 u 环 ⇒ 外侧是盘 V={d['disk_V']} E={d['disk_E']} "
              f"F={d['disk_F']}；x 盘内度 5 ⇒ X 上 3-正则（{d['xx']} 条 x-x 边）")
        print(f"  ⑤ 面型：无 uuu；uuw 恰 {d['n_uuw']} ⇒ n2+n3 = {d['disk_F']} − "
              f"{d['n_uuw']} = {d['n2'] + d['n3']}")
        print(f"  ⑥ 顶点-面关联：6·1 + 2·{d['n2']} + 3·{d['n3']} = {d['x_faces']} "
              f"⇒ n2={d['n2']} n3={d['n3']}")
        print(f"  ⑦ 但 www 面必是 X 的三角 ⇒ n3 ≤ max_tri(6 顶点 3-正则图) = "
              f"{d['max_tri']}（穷举 {d['n_graph']} 个标号图完备；同构仅两类："
              f"三角棱柱（2 三角）与 K₃,₃（0 三角））")
        print(f"     {d['n3']} > {d['max_tri']}  ⇒ **矛盾 ∎ 该行不存在**")
        print("-" * 78)
        print("  反向控制：k=12 行 (a=12,b=0) 无 deg-6 节点 ⇒ 推导不适用（可实现）；")
        print("           k=13 的**非均匀**实现存在（细分 icosa 一面）⇒ 排除只针对 {5,6} 行。")
        print("  ⚠️ k ≥ 14（b ≥ 2）未证——开放项，不得由本行外推。")
        return

    if "k" in kv:
        k = int(kv["k"])
        net = construct_shell(k)
        if net is None:
            print(f"k={k} 无均匀闭合壳层解（可行集 = "
                  f"{sorted({r['k'] for r in enumerate_shells()})}）")
            return
        a = audit(net)
        print(f"k={k}  V={a['V']} E={a['E']} F={a['F']} χ={a['chi']} "
              f"Σ(6−deg)={a['S']} 全deg={set(degrees(net))} "
              f"均匀闭合={is_uniform_closed_shell(net)}")
        return

    print("用法: python l0_kissing.py selftest=1 | sweep=1 K=16 | chi=1 | "
          "quasi=1 K=32 | k13=1 | k=12")


if __name__ == "__main__":
    main()
