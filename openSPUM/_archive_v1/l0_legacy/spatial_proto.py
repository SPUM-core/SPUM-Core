# -*- coding: utf-8 -*-
"""spatial_proto.py — 最小 3D 创生原型：纯关系 L0 + 3D 投影 L1（无面结构）。

范式裁定（用户 2026-09-12，两轮修正后）
--------------------------------------
  1. L0 本体是**纯关系**：只有 id 与邻接。没有面、没有旋转系统、没有平面性。
  2. L1 投影**必然是 3D**（2D 接吻数=6 < 3D=12）；2D 只能是 3D 再切片，绝非本体。
  3. 创生 = 在 3D 里找一个「可容球位置」：新球与任意 **≥3** 个已存在球同时相切
     （含口袋位/内侧位），边 = 它与**所有**相切球的关系（不只 3 条）。

关键修正（相对上一版 spatial_proto 的错误）
------------------------------------------
  (A) 球体**非等大**：半径随度数增长。上一版用 r≡1.0 等大球 → 接吻数 12 成了
      人为上限；这是「等大球」的结论，不是 SPUM 的结论。上一版另一稿用 r=√deg，
      形式对但标度是拍脑袋的。本版半径律**从公理推导**（见 §0）。
  (B) 关系阈值是 **50**（容量 E+F = 30+20），演化实际截断 **42**。上一版用
      「2d−3」平面界当判据是层次错误：21 根本不违反关系阈值 50。

本版范式裁定（用户 2026-09-12）：**约束必须由模拟自然涌现，不是人为规则。**
--------------------------------------------------------------------
  图论允许节点度数无限增长（边集上限 10^6 量级）——这是图论与现实的鸿沟。
  填平鸿沟不能靠外部阈值，只能靠公理层的两条输入在运行中自洽收敛：

    A2 排他占据：每个切点（关系）在球面独占面积 κ²（κ = 最小长度刻度）
    T8 临界直径：D_max = 4κ（由 A2 排他深度 κ 与 T0 生长率上界联立导出：
                 生长资源恰好维持新增表面 V⁺ = κ·dS/dt ⟹ D = 4κ）

  联立即得度数上限（T8）：πD²/κ² = π(4κ)²/κ² = **16π ≈ 50.27**。
  ── 16π / 50 / 42 / 12 在代码里**一个都不出现**：运行中由饱和判据自然给出。

  本原型追踪：
      - max_deg 能否**突破等大球接吻数 12**、向涌现阈值 16π 收敛
      - 饱和球（面积预算耗尽）是否是「大球 + 一圈更小层级球」的层级构型

用法
----
  python spatial_proto.py selftest=1
  python spatial_proto.py seed=tetra nframes=8 max_spheres=150
  python spatial_proto.py seed=tetra nframes=6 max_spheres=150 trace=1

  只做创生（V⁺）+ 帧末全局重排（保持「边相切 + 全对不重叠」自洽），
  不做悬挂删除（V⁻）——本原型只回答「非等大 3D 创生能否逼近关系容量 42/50」。
"""

import sys
import time
from collections import Counter

import numpy as np

# 计算后端：优先 CuPy（显卡）。O(N²) 全对距离只应在设备端算——CPU 串行撑不住
# （这正是本原型规模化的关键）。无 GPU 时自动回退 NumPy（同一段代码，xp = cp|np）。
try:
    import cupy as _cp
    _XP = _cp
    _HAS_GPU = True
except Exception:
    _cp = None
    _XP = np
    _HAS_GPU = False


def _to_host(a):
    """设备端数组 → CPU numpy（NumPy 后端则原样返回）。"""
    return _cp.asnumpy(a) if _HAS_GPU else np.asarray(a)


# ============================================================
# 0. 公理层输入 → 半径律与临界半径（代码中无 12 / 42 / 50 魔数）
#
#    全部输入只有两条，且都在公理/定理层：
#      A2 排他占据 —— 每个切点（关系）在球面独占面积 κ²，κ 是最小长度刻度。
#      T8 临界直径 —— D_max = 4κ（A2 排他深度 κ 与 T0 生长率上界联立的结果）。
#
#    推导（SPUM2611 §八）：
#      球面 πD² ≥ deg·κ²      ⟹  D ≥ √(deg/π)·κ      （容纳 deg 个切点的下限）
#      下限与上限 D ≤ 4κ 恰好重合 ⟹ deg = π(4κ)²/κ² = 16π
#    ── 16π 不写进代码：它是两式联立的结果，由 saturated() 在运行中给出。
# ============================================================
KAPPA = 1.0                     # 最小长度刻度（A2）
R_CRIT = 2.0 * KAPPA            # 临界半径 = D_max/2，D_max = 4κ（T8）


def radius_of(deg):
    """A2 面积预算下限：r(deg) = (κ/2)·√(deg/π)。度数决定体积的几何必然。"""
    return 0.5 * KAPPA * float(np.sqrt(max(int(deg), 1) / np.pi))


def saturated(deg):
    """面积预算是否耗尽：下限半径是否已抵达临界半径（D ≥ D_max = 4κ）。

    等价于 deg ≥ π·R_CRIT²/(κ²/4)·… 的具体数值由 KAPPA/R_CRIT 算出，
    代码不写 16π / 50 —— 阈值是推导结果，不是输入。
    """
    return radius_of(deg) >= R_CRIT - 1e-12


# ============================================================
# 0.5 球面角容量原语 + 最小可创生尺度（对标 particle_state.cuh）
#
#    两条**局部**涌现判据（都不写度数魔数）：
#      (a) r_min —— 最小可创生粒子半径 = 一度球下限 r(1)。度数为 0 即不存在，
#          故不存在比一度球更小的可创生粒子。缝隙可容球 < r_min 则不活化：
#          这是「体积成本」的下限，阻断缝隙无限细分创生（几何发散的根）。
#      (b) 角容量 —— 邻居 other 在 self 球面独占角半径 α = asin(r_other/(r_self+r_other))；
#          任意邻居对 (j,k) 的球面角距必须 ≥ α_j + α_k（排他占据的球面形式）。
#          球面被切点填满 ⟹ 任何新方向都被拒 ⟹ 度数饱和**在运行中涌现**。
# ============================================================
R_MIN = radius_of(1)              # 最小可创生半径 = 一度球下限（度 0 不存在）


def angular_radius(r_self, r_other):
    """邻居 other 在 self 球面上的角半径 α = asin(r_other/(r_self+r_other))。"""
    return float(np.arcsin(np.clip(r_other / (r_self + r_other), -1.0, 1.0)))


def contact_angle(r_i, r_j, r_k):
    """i 的两个邻居 j,k 互不穿透所需的最小球面角距（α 近似 = α_j + α_k）。"""
    return angular_radius(r_i, r_j) + angular_radius(r_i, r_k)


def angular_distance(u, v):
    """两单位向量的球面大圆角距。"""
    return float(np.arccos(np.clip(float(np.dot(u, v)), -1.0, 1.0)))


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def gap_spawn_radius(R, rn, theta):
    """球面缝隙可容新球半径（对标 particle_state.cuh::gap_spawn_radius）。

    中心球 R 的球面上，两邻居方向之间夹角 2θ（缝隙中心到邻居角距 θ），
    新球与中心球及邻居（近似取半径 rn）同时相切时的半径：
      r_new = R·(R+rn)·(1−cosθ) / ( (R+rn)·cosθ − R + rn )
    返回 ≤ 0 表示几何上缝隙不可容。
    """
    B = R + rn
    c = float(np.cos(theta))
    denom = B * c - R + rn
    if denom < 1e-8:
        return -1.0
    return float(R * B * (1.0 - c) / denom)


def fibonacci_sphere(n):
    """斐波那契/黄金角球面螺旋：n 个近似均匀分布的单位方向。"""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)              # 极角（均匀 cos 采样）
    theta = np.pi * (1.0 + np.sqrt(5.0)) * i        # 方位角（黄金角）
    s = np.sin(phi)
    return np.stack([s * np.cos(theta), s * np.sin(theta), np.cos(phi)], axis=1)


def shadow_capacity(R, r, n_candidates=20000):
    """球影排他容量（下界）：半径 R 球面可容纳多少个半径 r 的小球。

    两小球在 R 球面上的球影不重叠 ⟺ 方向角距 ≥ 2α，α = asin(r/(R+r))。
    用黄金角螺旋生成候选方向，贪心接受 pairwise 满足角距者——返回的是
    一个**真实可验证的构型**（下界），不是面积上限。
    """
    alpha = angular_radius(R, r)
    cos_min = float(np.cos(2.0 * alpha))
    accepted = []
    for d in fibonacci_sphere(n_candidates):
        if accepted and max(float(np.dot(a, d)) for a in accepted) > cos_min:
            continue
        accepted.append(d)
    return len(accepted)


def hetero_capacity_report():
    """度异质容量实验：大球 + 小球能否逼近半径极限 16π≈50。"""
    r1 = radius_of(1)
    r3 = radius_of(3)
    r10 = radius_of(10)
    sat_deg = 4.0 * np.pi * R_CRIT ** 2 / KAPPA ** 2     # = 16π，半径极限
    print("[度异质容量实验] 球影排他（大球 + 小球）")
    print(f"  半径极限（A2+T8）：deg={sat_deg:.2f} 时 r(deg) 触 R_crit={R_CRIT:.3f}")
    cap3 = shadow_capacity(R_CRIT, r3)
    cap1 = shadow_capacity(R_CRIT, r1)
    cap_hom = shadow_capacity(radius_of(16), r10)
    print(f"  异质：R=R_crit={R_CRIT} 大球 + r(3)={r3:.3f} 小球 → 可容 {cap3}")
    print(f"  异质：R=R_crit={R_CRIT} 大球 + r(1)={r1:.3f} 小球 → 可容 {cap1}")
    print(f"  同质：R=r(16)={radius_of(16):.3f} 球 + r(10)={r10:.3f} 邻居 → 可容 {cap_hom}")
    if max(cap3, cap1) >= sat_deg:
        print(f"  → 结论：异质球影容量 {max(cap3, cap1)} ≥ 半径极限 {sat_deg:.1f}，"
              f"球影不阻塞 50，绑定约束是半径极限（50 原则上可达）")
    else:
        print(f"  → 结论：球影容量 {max(cap3, cap1)} < 半径极限 {sat_deg:.1f}，球影仍阻塞 50")


# ============================================================
# 1. 种子夹具（拓扑相切 + 半径由度数决定）
# ============================================================
def seed_tetra():
    """正四面体 4 球，两两相切。deg=3，r=√3，中心距=2√3。"""
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                 dtype=np.float64)
    v /= np.linalg.norm(v[0])
    v *= np.sqrt(6.0) / 2.0          # 单位：中心距 2（等大球 r=1 相切）
    v *= radius_of(3)                 # 缩放到 r=√3 相切（中心距 2√3）
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return v, edges


def seed_icosa():
    """正二十面体 12 球（校验夹具）：顶点 5-正则，30 边。deg=5，r=√5。"""
    phi = (1 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=np.float64)
    d01 = np.linalg.norm(v[0] - v[1])
    v *= (2.0 / d01) * radius_of(5)  # 缩放到 r=√5 相切（中心距 2√5）
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(np.linalg.norm(v[i] - v[j]) - 2.0 * radius_of(5)) < 1e-6:
                edges.append((i, j))
    return v, edges


def seed_of(name):
    return {"tetra": seed_tetra, "icosa": seed_icosa}[name]()


# ============================================================
# 2. 三维球心定位：新球与三球同时相切（三球面交，至多两个解）
#    解 = 同一缝隙的两侧（外/内口袋位）——不是「两个面」。
# ============================================================
def trilaterate(c1, c2, c3, d1, d2, d3):
    """求 x 满足 |x - ci| = di (i=1,2,3)。返回至多两个 3D 解。"""
    c1 = np.asarray(c1, float)
    c2 = np.asarray(c2, float)
    c3 = np.asarray(c3, float)
    e2 = c2 - c1
    d12 = np.linalg.norm(e2)
    if d12 < 1e-12:
        return []
    e2 = e2 / d12
    i = float(np.dot(e2, c3 - c1))
    ey = c3 - c1 - i * e2
    j = float(np.linalg.norm(ey))
    if j < 1e-12:
        return []                    # 三球共线退化
    e3 = ey / j
    ez = np.cross(e2, e3)
    x = (d1 * d1 - d2 * d2 + d12 * d12) / (2.0 * d12)
    y = (d1 * d1 - d3 * d3 + i * i + j * j - 2.0 * i * x) / (2.0 * j)
    z2 = d1 * d1 - x * x - y * y
    if z2 < -1e-9:
        return []
    z2 = max(z2, 0.0)
    z = float(np.sqrt(z2))
    base = c1 + x * e2 + y * e3
    if z < 1e-9:
        return [base]
    return [base + z * ez, base - z * ez]


# ============================================================
# 3. 帧：创生（V⁺）
# ============================================================
def _angular_capacity_ok(pos, r, adj, m, c_dir, r_new, tol_ang):
    """角容量判据（局部、球面）：新球在相切球 m 球面上的方向 c_dir 不得侵犯
    m 的任一已占邻居——角距 ≥ contact_angle(r_m, r_n, r_new)。

    关键：m 接纳新邻居后度数 +1，其半径随之增长到 radius_of(deg_m + 1)。
    判据用「增长后半径」而非当前半径——否则当前半径的球面恰好 deg_m·κ²
    （A2 面积预算满），会永远拒绝第 deg_m+1 个邻居，造成**人为提前饱和**。
    半径随度数增长是 radius_of 的自然推论，这里只是让接纳判据与它自洽。
    """
    rm = radius_of(len(adj[m]) + 1)     # 预期增长后半径（度数 +1）
    pm = np.asarray(pos[m], dtype=np.float64)
    for n in adj[m]:
        n_dir = unit(np.asarray(pos[n], dtype=np.float64) - pm)
        if angular_distance(c_dir, n_dir) < contact_angle(rm, r[n], r_new) - tol_ang:
            return False
    return True


def create_frame(pos, r, adj, tol, max_new=None, cluster_tol=1e-3, pref=False):
    """一帧创生。返回 (new_positions, new_edges, new_radii)（确定性顺序）。

    只扫描 A 类缝隙（三球 i,j,k 两两相切）。新球半径 = radius_of(3)
    （出生度 3 的 A2 面积预算下限，对标 degree_radius(SPUM_BIRTH_DEG)），
    出生度 = 与 r_cap 球相切的球数（≥3）。缝隙容不下 r_cap 球时 trilaterate
    无解 → 该缝隙自然不活化——这是「体积成本下限」在 A 类通道的涌现形式。

    三条涌现判据（**无任何度数魔数**）：
      (a) r_min 活化    r_cap ≥ R_MIN —— 体积成本下限，阻断缝隙无限细分创生；
      (b) 笛卡尔排他    与「帧首球 + 本帧已加新球」不重叠；
      (c) 角容量        在每个相切球球面上，新方向与已占邻居角距 ≥ contact_angle
                         —— 球面被切点填满 ⟹ 新方向被拒 ⟹ 度数饱和在运行中涌现。

    规模化实现：候选枚举是局部的 O(N·deg²)（每球只查自己的邻居对）；而
    「候选 vs 帧首球」的排他 + 相切检测是 O(M·N) 全对距离——在设备端（CuPy）
    分块批量计算，取代原逐个候选在 Python 里做 O(N) 距离扫描与 np.asarray 拷贝。
    无 GPU 时同一段代码以 NumPy 运行（xp = np），仍比旧版少一层 Python 循环。
    """
    xp = _XP
    n0 = len(pos)
    P0 = np.asarray(pos, dtype=np.float64)     # CPU 引用（trilaterate 用）
    R0 = np.asarray(r, dtype=np.float64)

    # ---- 阶段 1：枚举 A 类缝隙候选（CPU，局部 O(N·deg²)） ----
    tri = []          # (i, j, k)
    Cs = []           # 候选坐标 (3,)
    for i in range(n0):
        ni = sorted(adj[i])
        for u in range(len(ni)):
            j = ni[u]
            if j <= i:
                continue
            for v in range(u + 1, len(ni)):
                k = ni[v]
                if k <= i or k not in adj[j]:
                    continue                     # 非 A 类缝隙（不两两相切）
                r_cap = radius_of(3)             # 出生度 3 的 A2 面积预算下限
                if r_cap < R_MIN:                # (a) r_min 活化（radius_of 单调，恒不触发；保留语义）
                    continue
                for c in trilaterate(P0[i], P0[j], P0[k],
                                     r_cap + R0[i], r_cap + R0[j], r_cap + R0[k]):
                    tri.append((i, j, k))
                    Cs.append(c)
    if not tri:
        return [], [], []
    M = len(tri)
    r_cap = radius_of(3)
    C = xp.asarray(np.asarray(Cs, dtype=np.float64))    # (M,3)
    Pg = xp.asarray(P0)                                 # (n0,3)
    Rg = xp.asarray(R0)                                 # (n0,)
    tgt0 = r_cap + Rg                                   # (n0,)

    # ---- 阶段 2：候选 vs 帧首球 排他 + 相切检测（设备端，分块 O(M·N)） ----
    frame_ok = np.ones(M, dtype=bool)          # True = 与帧首球无重叠
    frame_tangent = [None] * M                 # 每候选的帧首球相切索引
    B = 4096
    for s in range(0, M, B):
        e = min(s + B, M)
        D = xp.linalg.norm(C[s:e, None, :] - Pg[None, :, :], axis=2)  # (b,n0)
        ov_np = _to_host(D < (tgt0[None, :] - tol))
        tg_np = _to_host(xp.abs(D - tgt0[None, :]) <= tol)
        frame_ok[s:e] = ~ov_np.any(axis=1)
        for m in range(e - s):
            frame_tangent[s + m] = np.where(tg_np[m])[0].tolist()

    # ---- 阶段 2.5（可选）：偏好连接 —— 优先处理「相切列表含高度球」的候选 ----
    # 度异质实验：让大球先长，把创生偏置到高度球周围，测 max_deg 能否突破
    # 同质球影容量 ~16。这只是**处理顺序**的偏置，不引入任何度数魔数或外部规则。
    if pref:
        deg_existing = np.array([len(a) for a in adj], dtype=np.int64)
        order = sorted(range(M),
                       key=lambda m: (-int(deg_existing[frame_tangent[m]].max())
                                      if frame_tangent[m] else 0))
        Cs = [Cs[m] for m in order]
        frame_ok = frame_ok[order]
        frame_tangent = [frame_tangent[m] for m in order]

    # ---- 阶段 3：顺序收尾（新球互斥 + 角容量 + 聚类去重，O(新球数²) 小集合） ----
    cur_pos = [np.array(p, dtype=np.float64) for p in pos]
    cur_rad = list(r)
    cur_adj = [set(a) for a in adj]
    new_positions = []
    new_edges = []
    new_radii = []
    for idx in range(M):
        if not frame_ok[idx]:
            continue
        c = Cs[idx]
        tangent = list(frame_tangent[idx])
        # (b) 与已接受新球排他（顺序小集合）；恰好相切者并入 tangent
        if new_positions:
            NP = np.asarray(new_positions, dtype=np.float64)
            NR = np.asarray(new_radii, dtype=np.float64)
            dnew = np.linalg.norm(NP - c, axis=1)
            if (dnew < r_cap + NR - tol).any():
                continue
            for t in np.where(np.abs(dnew - (r_cap + NR)) <= tol)[0]:
                tangent.append(n0 + int(t))     # 新球全局 id
        if len(tangent) < 3:
            continue
        # (c) 角容量判据：在每个相切球球面上不侵犯已占邻居
        ok = True
        for m in tangent:
            c_dir = unit(c - np.asarray(cur_pos[m], dtype=np.float64))
            if not _angular_capacity_ok(cur_pos, cur_rad, cur_adj, m, c_dir, r_cap,
                                        tol):
                ok = False
                break
        if not ok:
            continue
        if any(np.linalg.norm(c - q) < cluster_tol for q in new_positions):
            continue                     # 位置聚类去重
        nid = n0 + len(new_positions)
        new_positions.append(c)
        new_edges.append(tangent)
        new_radii.append(r_cap)
        cur_pos.append(c)
        cur_rad.append(r_cap)
        cur_adj.append(set(tangent))
        for m in tangent:
            cur_adj[m].add(nid)
        if max_new is not None and len(new_positions) >= max_new:
            break
    return new_positions, new_edges, new_radii


# ============================================================
# 4. 嵌入求解：**体积是区间对象**（SPUM2611 §八）——约束的涌现之地
#
#    §八：V(d) ∈ [V_min(d), V_max]，V_min(d) ∝ d^{3/2}，V_max 由 D_max = 4κ 钉死。
#    即：半径不是单值 r_min(deg)——把区间压成一点会让相切系统过约束
#    （E ≈ 3N 恰在刚性阈值，实测 432/444 条边不相切、通用无解）；半径是区间
#    [r_lo(deg), R_CRIT] 内的**自由变量**，与位置联合求解。
#
#    未知量 = 3N(位置) + N(半径) = 4N，相切方程 = E；E ≲ 3N 时系统有余量，
#    自洽嵌入存在。**度数上限因此不是写出来的，而是「区间耗尽 + 嵌入不可解」**：
#    当 r_lo(deg) 逼到 R_CRIT，或邻居排布容不下时，残差居高不下 → 生长被几何拒绝。
#
#    三条约束全部来自公理层，无任何幻觉魔数：
#      (1) 边相切     |p_i − p_j| = r_i + r_j
#      (2) 非边排他   |p_i − p_j| ≥ r_i + r_j
#      (3) 区间       r_lo(deg_i) ≤ r_i ≤ R_CRIT   （A2 面积预算 + T8 临界直径）
# ============================================================
def solve_embedding(pos, deg, adj, r_prev=None, iters=600, tol=1e-4, step=0.5):
    """联合求 (p, r)。返回 (p, r, 残差)。残差 = max(边相切误差, 非边重叠, 区间越界)。

    全部 O(N²) 全对距离在设备端（CuPy）计算；CPU 只做度循环与标量控制流。
    无 GPU 时自动回退 NumPy（同一段代码，xp = cp | np）。
    """
    xp = _XP
    n = len(pos)
    p = xp.asarray(np.asarray(pos, dtype=np.float64))
    r_lo = xp.asarray(np.array([radius_of(int(x)) for x in deg], dtype=np.float64))
    if r_prev is None:
        r = xp.minimum(r_lo, R_CRIT).copy()
    else:
        rp = xp.asarray(np.asarray(r_prev, dtype=np.float64))
        if rp.size < n:                      # 本帧新增球：用下限半径补齐
            rp = xp.concatenate([rp, r_lo[rp.size:]])
        r = xp.clip(rp[:n], r_lo, R_CRIT)

    ea, eb = [], []
    for a in range(n):
        for b in adj[a]:
            if a < b:
                ea.append(a)
                eb.append(b)
    ea = xp.asarray(ea, dtype=xp.int64)
    eb = xp.asarray(eb, dtype=xp.int64)

    # 非边掩码：直接由边表在设备端构造，避免 CPU 侧 O(N²) 布尔阵
    nonedge = xp.ones((n, n), dtype=bool)
    if ea.size:
        nonedge[ea, eb] = False
        nonedge[eb, ea] = False
    nonedge[xp.arange(n), xp.arange(n)] = False

    resid_edge = resid_over = resid_lo = 0.0
    deg_arr = xp.maximum(xp.asarray([len(a) for a in adj], dtype=xp.float64), 1.0)
    for _ in range(iters):
        # (1) 边相切：位置与半径各承担一半修正（半径按度数归一，防高变球过冲）
        if ea.size:
            dd = p[ea] - p[eb]
            L = xp.maximum(xp.linalg.norm(dd, axis=1), 1e-9)
            err = L - (r[ea] + r[eb])                 # >0 太远
            corr = (err * (step * 0.5) / L)[:, None] * dd
            xp.add.at(p, ea, -corr)
            xp.add.at(p, eb, corr)
            dr = xp.zeros(n, dtype=xp.float64)
            xp.add.at(dr, ea, err)
            xp.add.at(dr, eb, err)
            r = xp.clip(r + (step * 0.5) * dr / deg_arr, r_lo, R_CRIT)
        r = xp.clip(r, r_lo, R_CRIT)

        # (2) 非边排他：先缩半径（下限 r_lo），缩不动再推位置
        D = p[:, None, :] - p[None, :, :]
        L2 = xp.sqrt((D * D).sum(axis=2))
        L2[xp.arange(n), xp.arange(n)] = xp.inf
        mind = r[:, None] + r[None, :]
        viol = nonedge & (L2 < mind - tol)
        if xp.any(viol).item():
            deficit = xp.where(viol, mind - L2, 0.0)
            sh = 0.3 * deficit.max(axis=1)
            r = xp.clip(r - xp.minimum(sh, r - r_lo), r_lo, R_CRIT)
            mind = r[:, None] + r[None, :]
            viol2 = nonedge & (L2 < mind - tol)
            if xp.any(viol2).item():
                c2 = xp.where(viol2,
                              (L2 - mind) / xp.maximum(xp.where(viol2, L2, 1.0), 1e-9)
                              * (step * 0.5), 0.0)
                c2 = xp.nan_to_num(c2, nan=0.0, posinf=0.0, neginf=0.0)
                p -= (c2[:, :, None] * D).sum(axis=1)

        # 残差
        if ea.size:
            resid_edge = float(xp.max(xp.abs(
                xp.linalg.norm(p[ea] - p[eb], axis=1) - (r[ea] + r[eb]))).item())
        else:
            resid_edge = 0.0
        D = p[:, None, :] - p[None, :, :]
        L2 = xp.sqrt((D * D).sum(axis=2))
        L2[xp.arange(n), xp.arange(n)] = xp.inf
        mind = r[:, None] + r[None, :]
        ov = nonedge & (L2 < mind - tol)
        resid_over = float(xp.max(xp.where(ov, mind - L2, 0.0)).item()) \
            if xp.any(ov).item() else 0.0
        resid_lo = float(xp.max(xp.maximum(r_lo - r, 0.0)).item())
        if max(resid_edge, resid_over, resid_lo) < tol:
            break
    return _to_host(p), _to_host(r), max(resid_edge, resid_over, resid_lo)


# ============================================================
# 5. 测量
# ============================================================
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


def geo_audit(pos, r, adj, tol=1e-4):
    """几何一致性审计：边是否真相切、非边是否真不重叠（非等大半径）。

    O(N²) 全对距离在设备端向量化，替代原来的 Python 双重循环。
    """
    xp = _XP
    n = len(pos)
    P = xp.asarray(np.asarray(pos, dtype=np.float64))
    R = xp.asarray(np.asarray(r, dtype=np.float64))

    D = xp.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    triu = xp.triu(xp.ones((n, n), dtype=bool), k=1)

    ea_l, eb_l = [], []
    for a in range(n):
        for b in adj[a]:
            if a < b:
                ea_l.append(a)
                eb_l.append(b)
    adjm = xp.zeros((n, n), dtype=bool)
    if ea_l:
        ea = xp.asarray(ea_l, dtype=xp.int64)
        eb = xp.asarray(eb_l, dtype=xp.int64)
        adjm[ea, eb] = True
        adjm[eb, ea] = True

    bad_edge = int(xp.sum(
        triu & adjm & (xp.abs(D - (R[:, None] + R[None, :])) > tol)).item())
    bad_overlap = int(xp.sum(
        triu & (~adjm) & (D < (R[:, None] + R[None, :]) - tol)).item())
    print(f"[几何审计] 边不相切={bad_edge}  非边重叠={bad_overlap}")


def report(pos, r, adj, tag):
    n = len(pos)
    deg = np.array([len(a) for a in adj])
    E = int(deg.sum()) // 2
    print(f"[{tag}] V={n}  E={E}")
    print(f"  度分布: {dict(sorted(zip(*np.unique(deg, return_counts=True))))}")
    if n == 0:
        return
    comps = components(adj)
    print(f"  连通分量数={len(comps)}  前6大={[len(c) for c in comps[:6]]}")

    dmax = int(deg.max())
    sat_thresh = 4.0 * np.pi * R_CRIT ** 2 / (KAPPA ** 2)   # = 16π，由 A2+T8 联立算出
    n_sat = int(sum(1 for x in deg if saturated(int(x) + 1)))
    print(f"  max_deg={dmax}  饱和阈值={sat_thresh:.2f}"
          f"（=πD_max²/κ²，由 A2 κ + T8 D_max=4κ 联立；代码未写此数）"
          f"  已达饱和球={n_sat}")
    # 饱和中心（deg 最大）的邻居度数构成：验证「大球 + 更小层级球」层级构型
    sat = [v for v in range(n) if deg[v] == dmax]
    if sat:
        v = sat[0]
        nb_deg = sorted(deg[b] for b in adj[v])
        nb_r_min = min(float(r[b]) for b in adj[v])
        print(f"  最大度球(deg={dmax}, r={float(r[v]):.3f}, "
              f"r_lo={radius_of(dmax):.3f}, R_crit={R_CRIT:.3f}) 邻居度数: "
              f"min={nb_deg[0]} median={nb_deg[len(nb_deg)//2]} max={nb_deg[-1]} "
              f"  邻居半径比 min={nb_r_min / float(r[v]):.3f}")
    # 体积区间松弛：实际半径相对下限浮了多少（§八 区间对象）
    r_arr = np.asarray(r, dtype=np.float64)
    r_lo = np.array([radius_of(int(x)) for x in deg])
    slack = r_arr - r_lo
    print(f"  区间松弛（r − r_lo）: mean={slack.mean():.4f} max={slack.max():.4f}  "
          f"触上限(r=R_crit)球数={int((r_arr >= R_CRIT - 1e-9).sum())}")


# ============================================================
# 6. 主流程
# ============================================================
def parse_kv(argv):
    kv, rest = {}, []
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v
        else:
            rest.append(a)
    return kv, rest


def run(kv):
    global _XP
    if kv.get("nogpu", "0") == "1":
        _XP = np                          # 强制 CPU 后端（对比基准）
    seed_name = kv.get("seed", "tetra")
    nframes = int(kv.get("nframes", 8))
    max_spheres = int(kv.get("max_spheres", 150))
    tol = float(kv.get("tol", 1e-6))
    realign = kv.get("realign", "1") == "1"
    trace = kv.get("trace", "0") == "1"
    pref = kv.get("pref", "0") == "1"
    solv_iters = int(kv.get("solv_iters", 600))
    solv_step = float(kv.get("solv_step", 0.5))

    v, edges = seed_of(seed_name)
    pos = [np.array(p, dtype=np.float64) for p in v]
    adj = [set() for _ in pos]
    for (a, b) in edges:
        adj[a].add(b)
        adj[b].add(a)
    r = [radius_of(len(a)) for a in adj]
    seed_n = len(pos)                    # 初始节点区域（彼此的确认从这里锚定）

    print(f"[最小 3D 创生原型] seed={seed_name} "
          f"backend={'GPU(CuPy)' if _XP is not np else 'CPU(NumPy)'}  "
          f"r(deg)=(κ/2)√(deg/π) [A2 面积预算]  "
          f"R_crit={R_CRIT:.3f}(D_max=4κ) [T8]  R_MIN={R_MIN:.3f}(一度球)  "
          f"tol={tol} nframes={nframes} "
          f"max_spheres={max_spheres} realign={realign} pref={pref}")
    for f in range(nframes):
        t0 = time.perf_counter()
        np_new, ne, nr = create_frame(pos, r, adj, tol,
                                      max_new=max_spheres - len(pos), pref=pref)
        cre_ms = (time.perf_counter() - t0) * 1000.0
        if not np_new:
            print(f"  帧 {f + 1}: 无可创生位置（饱和），提前停止")
            break
        for c, es, rr in zip(np_new, ne, nr):
            nid = len(pos)
            pos.append(c)
            r.append(rr)               # 保持 r 与 pos 同步（创生半径 = 缝隙可容）
            adj.append(set(es))
            for m in es:
                adj[m].add(nid)
        deg_cur = [len(a) for a in adj]
        solv_ms = float("nan")
        if realign:
            # 体积区间 [r_lo(deg), R_CRIT] 内联合求 (p, r)；残差 = 几何可嵌入性
            t0 = time.perf_counter()
            p_arr, r_arr, resid = solve_embedding(pos, deg_cur, adj, r_prev=r,
                                                  iters=solv_iters,
                                                  step=solv_step)
            solv_ms = (time.perf_counter() - t0) * 1000.0
            pos = [np.array(x, dtype=np.float64) for x in p_arr]
            r = list(r_arr)
        else:
            r = [radius_of(d) for d in deg_cur]
            resid = float("nan")
        if trace:
            deg = np.array([len(a) for a in adj])
            seed_max = int(max(len(adj[i]) for i in range(seed_n)))
            print(f"  帧 {f + 1}: V={len(pos)}  E={int(deg.sum()) // 2}  "
                  f"本帧+{len(np_new)}  max_deg={int(deg.max())}  "
                  f"初始区max_deg={seed_max}  "
                  f"嵌入残差={resid:.2e}  创生={cre_ms:.0f}ms  求解={solv_ms:.0f}ms")
        if len(pos) >= max_spheres:
            print(f"  帧 {f + 1}: 达到 max_spheres={max_spheres}，停止")
            break

    geo_audit(pos, r, adj)
    report(pos, r, adj, "终态")


def selftest():
    # 1) 半径律（A2 面积预算）+ 涌现阈值（A2 与 T8 联立，代码未写该数）
    assert radius_of(1) < radius_of(4) < radius_of(50)
    thr_exact = 4.0 * np.pi * R_CRIT ** 2 / KAPPA ** 2      # = 16π
    assert not saturated(int(np.floor(thr_exact)) - 1)
    assert saturated(int(np.ceil(thr_exact)))
    print(f"[selftest] 半径律 r=(κ/2)√(deg/π) 单调升 OK；"
          f"涌现阈值 πD_max²/κ² = {thr_exact:.3f}（A2+T8 联立，非写入）")

    # 2) 种子计数
    v, edges = seed_tetra()
    assert len(v) == 4 and len(edges) == 6
    v, edges = seed_icosa()
    deg = Counter()
    for (a, b) in edges:
        deg[a] += 1
        deg[b] += 1
    assert len(v) == 12 and len(edges) == 30 and set(deg.values()) == {5}
    print("[selftest] 种子计数 OK：tetra (4,6)，icosa (12,30,全5)")

    # 3) 三球面交正确性：tetra 面 (0,1,2) 应找回顶点 3（等大 r=1 情形，d=2）
    v, _ = seed_tetra()
    rr = radius_of(3)
    sols = trilaterate(v[0], v[1], v[2], 2 * rr, 2 * rr, 2 * rr)
    assert len(sols) == 2, f"期望 2 解，得到 {len(sols)}"
    d3 = min(np.linalg.norm(s - v[3]) for s in sols)
    assert d3 < 1e-9, f"应找回第 4 顶点，误差 {d3}"
    print("[selftest] 三球面交 OK")

    # 4) tetra 单帧创生：4 面各出 1 个镜像位（A 类缝隙 + r_min + 角容量）
    pos = [np.array(p, dtype=np.float64) for p in v]
    adj = [set() for _ in pos]
    for (a, b) in seed_tetra()[1]:
        adj[a].add(b)
        adj[b].add(a)
    r = [radius_of(3)] * 4
    np_new, ne, nr = create_frame(pos, r, adj, 1e-6)
    assert len(np_new) == 4, f"tetra 单帧应创生 4 球，实际 {len(np_new)}"
    # 出生半径 = radius_of(3)（A2 面积预算，出生度 3）
    assert all(abs(rr - radius_of(3)) < 1e-6 for rr in nr), "tetra 出生半径应为 r(3)"
    print("[selftest] tetra 单帧创生 OK：4 镜像位（A 类缝隙 + r_min + 角容量）")

    # 5) r_min / 角容量 / 缝隙可容半径（gap_spawn_radius 对标设计文档 §3.4 笛卡尔精确解）
    assert R_MIN == radius_of(1)
    assert abs(angular_radius(1.0, 1.0) - np.arcsin(0.5)) < 1e-12   # 等大球 α = 30°
    assert abs(contact_angle(1.0, 1.0, 1.0) - 2.0 * np.arcsin(0.5)) < 1e-12
    # B 类面洞笛卡尔精确解（§3.4 表）：等大球 R=rn=1，θ=arccos(1/3)（四面体面）→ r_max=2.0
    assert abs(gap_spawn_radius(1.0, 1.0, np.arccos(1.0 / 3.0)) - 2.0) < 1e-9
    # 二十面体面 θ≈37.38° → r_max≈0.258（§3.4 表，度 12 饱和涌现的数值来源）
    r_ico = gap_spawn_radius(1.0, 1.0, np.deg2rad(37.38))
    assert abs(r_ico - 0.258) < 1e-2
    print("[selftest] r_min / 角容量 / 缝隙可容半径（gap_spawn_radius）OK")

    print("[selftest] 全部通过")


def main():
    kv, _ = parse_kv(sys.argv[1:])
    if kv.get("selftest") == "1":
        selftest()
    elif kv.get("hetcap") == "1":
        hetero_capacity_report()
    else:
        run(kv)


if __name__ == "__main__":
    main()
