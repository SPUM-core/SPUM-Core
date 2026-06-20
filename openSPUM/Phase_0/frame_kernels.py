"""
SPUM 帧内核 — 5 步帧逻辑的 numpy 实现（镜像 GPU per-thread 操作）。

每个函数对应一个 CUDA kernel 的逻辑。
当前使用 numpy 向量化操作实现，CUDA 后端就绪时只需替换 backend。

帧结构（SPUM 系统总纲 §4）:
    Step 1: 创生 — 检测缝隙 → 从潜在池活化粒子填充缝隙
    Step 2: 连接 — 检测相切关系 → 建立边
    Step 3: 变化体积 — degree → radius 映射
    Step 4: 判断悬挂 — degree < 2 → 标记
    Step 5: 删除悬挂 — 标记的粒子回归潜在池
"""

import math
import numpy as np
from typing import Dict, Tuple
from .constants import KAPPA, GEOMETRIC_TOLERANCE, CRYSTALLITE_DEGREE_THRESHOLD, MIN_GAP_RATIO


# ============================================================
# 工具函数
# ============================================================

def _vec3_sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])

def _vec3_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def _vec3_norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def _vec3_cross(a, b):
    return (a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0])


def _find_center(particles) -> int:
    """找到中心 coda (uid 以 cent_ 开头) 的索引。"""
    for i in range(particles.N):
        if particles.active[i] and str(particles.uid[i]).startswith('cent_'):
            return i
    return -1


def _three_sphere_gap(pos_a, r_a, pos_b, r_b, pos_c, r_c, ref_pos=None, loose=False):
    """笛卡尔定理 + 三边测量: 三球内切缝隙球的位置和半径。

    输入:
        pos_a/b/c: 三球中心 (array-like)
        r_a/b/c: 三球半径
        ref_pos: 参考位置 (可选)。用于消歧——返回离 ref_pos 更近的一侧。
                 不传时返回正 z 侧。
        loose: 如果 True, 放宽相切检查 (允许最多 5 倍 GEOMETRIC_TOLERANCE 偏差).

    返回:
        (gap_pos, gap_radius) 或 None
    """
    d_ab = _vec3_norm(_vec3_sub(pos_a, pos_b))
    d_bc = _vec3_norm(_vec3_sub(pos_b, pos_c))
    d_ca = _vec3_norm(_vec3_sub(pos_c, pos_a))

    # 相切检查
    tol = GEOMETRIC_TOLERANCE
    tang_ok = (abs(d_ab - (r_a + r_b)) < tol * max(r_a + r_b, 1e-8) and
               abs(d_bc - (r_b + r_c)) < tol * max(r_b + r_c, 1e-8) and
               abs(d_ca - (r_c + r_a)) < tol * max(r_c + r_a, 1e-8))
    if not tang_ok:
        if loose:
            # 宽松: 允许最多 5 倍容差
            loose_tol = tol * 5.0
            if not (abs(d_ab - (r_a + r_b)) < loose_tol * max(r_a + r_b, 1e-8) and
                    abs(d_bc - (r_b + r_c)) < loose_tol * max(r_b + r_c, 1e-8) and
                    abs(d_ca - (r_c + r_a)) < loose_tol * max(r_c + r_a, 1e-8)):
                return None
        else:
            return None

    # 曲率 → 笛卡尔定理
    k1 = 1.0 / r_a if r_a > 1e-12 else 1e12
    k2 = 1.0 / r_b if r_b > 1e-12 else 1e12
    k3 = 1.0 / r_c if r_c > 1e-12 else 1e12

    sum_k = k1 + k2 + k3
    sum_kk = k1*k1 + k2*k2 + k3*k3
    discriminant = 3.0 * (sum_k * sum_k - 2.0 * sum_kk)
    if discriminant < 0:
        return None
    k4 = (sum_k + math.sqrt(discriminant)) / 2.0  # 内切缝隙球
    k4 = float(k4)
    if k4 <= 0:
        return None

    gap_r = 1.0 / k4

    # 三边测量 (trilateration): 求三个球 (A,B,C, 半径=ri+grap_r) 的交点
    A = np.array(pos_a, dtype=float)
    B = np.array(pos_b, dtype=float)
    C = np.array(pos_c, dtype=float)
    d_a = float(r_a) + gap_r
    d_b = float(r_b) + gap_r
    d_c = float(r_c) + gap_r

    # B-A, C-A
    B0 = B - A
    C0 = C - A

    x_b = np.linalg.norm(B0)
    if x_b < 1e-12:
        return None
    ex = B0 / x_b

    x_c = np.dot(C0, ex)
    proj_perp = C0 - x_c * ex
    y_c = np.linalg.norm(proj_perp)
    if y_c < 1e-12:
        return None
    ey = proj_perp / y_c
    ez = np.cross(ex, ey)  # 单位法向

    # 解三边测量
    x = (d_a*d_a - d_b*d_b + x_b*x_b) / (2.0 * x_b)
    y = (d_a*d_a - d_c*d_c + x_c*x_c + y_c*y_c - 2.0*x_c*x) / (2.0 * y_c)
    z_sq = d_a*d_a - x*x - y*y
    if z_sq < 0:
        return None
    z = math.sqrt(max(0.0, z_sq))

    # 两个解: +z 和 -z
    pos_z = A + x*ex + y*ey + z*ez
    neg_z = A + x*ex + y*ey - z*ez

    # 消歧: 用 ref_pos (如果提供)
    if ref_pos is not None:
        ref = np.array(ref_pos, dtype=float)
        if np.linalg.norm(pos_z - ref) < np.linalg.norm(neg_z - ref):
            return (tuple(pos_z.astype(np.float32)), gap_r)
        else:
            return (tuple(neg_z.astype(np.float32)), gap_r)
    else:
        # 默认: 返回正 z 侧
        return (tuple(pos_z.astype(np.float32)), gap_r)


def _icosahedron_vertices() -> np.ndarray:
    """正二十面体的 12 个顶点 (单位球面, 确定性)。

    边长为 1.051, 每个顶点 5 个等距邻居。
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # 黄金比例
    verts = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            verts.append((0.0, float(s1), float(s2 * phi)))
            verts.append((float(s1), float(s2 * phi), 0.0))
            verts.append((float(s2 * phi), 0.0, float(s1)))
    arr = np.array(verts[:12], dtype=np.float32)
    # 归一化到单位球面
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / norms


def _fibonacci_sphere(n: int) -> np.ndarray:
    """确定性的斐波那契球面均匀分布 (无随机)。"""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    positions = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        y = 1.0 - (2.0 * i + 1.0) / n
        r = math.sqrt(1.0 - y * y)
        theta = 2.0 * math.pi / phi * i
        positions[i] = (r * math.cos(theta), y, r * math.sin(theta))
    return positions


def _star_surface(n_surface: int = 12) -> Tuple[np.ndarray, float]:
    """中心 + 表面 coda, 表面精确在中心表面切线位置。

    r_center = 1 + n_surface (连接 n_surface 个表面后)
    r_surface = 1 + 1 = 2 (仅连接中心)
    distance = r_center + r_surface (相切)

    Returns:
        (positions, radii)
        positions: (n_surface+1, 3) 中心+所有表面
        radii: (n_surface+1,)
    """
    center_r = float((1 + n_surface) * KAPPA)  # r=13 (for n=12)
    surface_r = 2.0  # 1 + 1 (初始仅连接中心)

    # 表面均匀分布在单位球面
    surface_dir = _fibonacci_sphere(n_surface)

    # 每个表面在中心表面切线位置
    surface_dist = center_r + surface_r

    positions = np.zeros((n_surface + 1, 3), dtype=np.float32)
    positions[0] = (0.0, 0.0, 0.0)
    positions[1:] = surface_dir * surface_dist

    radii = np.zeros(n_surface + 1, dtype=np.float32)
    radii[0] = center_r
    radii[1:] = surface_r

    return positions, radii


def _init_sequential(n_coda: int = 50) -> Tuple[np.ndarray, np.ndarray, list]:
    """顺序构建网络: coda 按 ID 逐个接入。

    规则:
        1. 所有 coda 初始体积=1 (r=1), 无连接
        2. coda 0 为坐标原点
        3. coda i 连接到 coda (i-1), 双方体积+1
        4. 新 coda 从切线位置计算自己的坐标

    Returns:
        (positions, radii, edges)
        edges: [(i,j), ...] 待建立的 persistent 连接
    """
    pos = np.zeros((n_coda, 3), dtype=np.float32)
    radii = np.ones(n_coda, dtype=np.float32)
    edges = []

    # 方向: Fibonacci 球面 (N-1 个方向, 确保 3D 覆盖)
    dirs = _fibonacci_sphere(n_coda - 1)

    # coda 0 在原点, coda 1 在 +Z 方向
    pos[0] = (0.0, 0.0, 0.0)
    edges.append((0, 1))
    pos[1] = (0.0, 0.0, 2.0)  # r1 + r0 = 1+1 = 2

    # 后续 coda: 逐个接入度数最低的已有 coda
    # 目的: 均衡网络, 避免单点度数过高
    for i in range(2, n_coda):
        # 按度数升序排列已有 coda
        degs = [(sum(1 for e in edges if e[0] == j or e[1] == j), j)
                for j in range(i)]
        degs.sort()
        best_j = degs[0][1]  # 度数最低的

        edges.append((best_j, i))

        # 从 best_j 沿 Fibonacci 方向放置
        deg_i = sum(1 for e in edges if e[0] == i or e[1] == i)
        deg_j = sum(1 for e in edges if e[0] == best_j or e[1] == best_j)
        r_i = float((1 + deg_i) * KAPPA)
        r_j = float((1 + deg_j) * KAPPA)
        pos[i] = pos[best_j] + dirs[i - 2] * (r_i + r_j)

    # 最终半径
    for i in range(n_coda):
        d = sum(1 for e in edges if e[0] == i or e[1] == i)
        radii[i] = float((1 + d) * KAPPA)

    return pos, radii, edges


def _compute_surface_adjacency(surface_pos: np.ndarray,
                                surface_radii: np.ndarray) -> np.ndarray:
    """计算表面 coda 之间的角邻接矩阵 (向量化)。

    两个表面 coda 在球面上相邻的距离条件是:
        角距 < 角半径(coda_i) + 角半径(coda_j)
    其中 角半径 = arcsin(r / R), R = 到中心的距离 (逐粒子)

    返回 (n, n) 布尔邻接矩阵。
    """
    n = len(surface_pos)
    if n < 2:
        return np.zeros((n, n), dtype=bool)

    # 逐粒子径向距离: (n,)
    Rs = np.linalg.norm(surface_pos, axis=1)
    Rs = np.maximum(Rs, 1e-12)

    # 角半径 (向量化)
    angular_radii = np.arcsin(np.clip(surface_radii / Rs, -1.0, 1.0))  # (n,)

    # 角距矩阵: acos(dot(pi, pj) / (Ri * Rj))
    dots = (surface_pos @ surface_pos.T) / (Rs[:, None] * Rs[None, :])
    dots = np.clip(dots, -1.0, 1.0)
    angular_dists = np.arccos(dots)  # (n, n)

    # 角半径和矩阵
    angular_sum = angular_radii[:, None] + angular_radii[None, :]  # (n, n)

    # 邻接: 角距 < 角半径和, 且非对角线
    adj = angular_dists < angular_sum
    np.fill_diagonal(adj, False)

    return adj


# ============================================================
# 初始化: 密堆分布 (FCC) — 100 个体积为 1 的球体
# ============================================================

def init_close_packed(particles, n_total: int = 100):
    """在原点生成 FCC 密堆分布的球体, 全部 volume=1。

    FCC 晶格常数:
        最近邻距离 = 2 * r (相切)
        r = (3/4π)^(1/3) ≈ 0.620 (单位体积球半径)
        晶格常数 a = sqrt(2) * 2r ≈ 1.754
    """
    r_unit = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    a = math.sqrt(2.0) * 2.0 * r_unit

    # FCC 基矢
    basis = [(0.0, 0.0, 0.0),
             (0.5, 0.5, 0.0),
             (0.5, 0.0, 0.5),
             (0.0, 0.5, 0.5)]

    points = []
    cell_range = 6  # 6^3 = 216 cells × 4 = 864 候选点
    for ix in range(-cell_range, cell_range + 1):
        for iy in range(-cell_range, cell_range + 1):
            for iz in range(-cell_range, cell_range + 1):
                for b in basis:
                    x = (ix + b[0]) * a
                    y = (iy + b[1]) * a
                    z = (iz + b[2]) * a
                    points.append((x, y, z))

    # 按到原点距离排序, 取最近 n_total 个
    points.sort(key=lambda p: p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    points = points[:n_total]

    # 添加粒子 (全部 volume=1, degree=0)
    for i, pos in enumerate(points):
        uid = "cent_0000" if i == 0 else f"fcc_{i:04d}"
        particles.add_particle(pos=pos, uid=uid, degree=0,
                               initial_volume=1.0, initial_degree=0)

    # 计算半径 (V=1 → r ≈ 0.620)
    step3_volume(particles)

    # 连接所有相切的球体
    nn_dist = 2.0 * r_unit  # 最近邻距离 = 2r
    tol = 0.05
    n = particles.N
    for i in range(n):
        if not particles.active[i]:
            continue
        for j in range(i + 1, n):
            if not particles.active[j]:
                continue
            d = float(np.linalg.norm(particles.pos[i] - particles.pos[j]))
            if abs(d - nn_dist) < tol * nn_dist:
                particles.add_connection(i, j)


# ============================================================
# Step 1: 创生 — 缝隙检测 + 潜在活化
# ============================================================

def step1_create(particles, max_checks: int = 100000) -> int:
    """检测活性粒子间的三体缝隙，在每个缝隙中创建最大球体。

    使用空间哈希 (grid-based) 来替代 O(N³) 的全遍历。
    仅在相邻单元格内检测三元组。如果候选数超过 max_checks,
    跳过此帧 (由 step1b 球面填充处理)。

    Returns:
        本帧活化的粒子数
    """
    active_idx = np.where(particles.active)[0]
    n_active = len(active_idx)
    if n_active < 3:
        return 0

    MIN_GAP_V = 0.3
    MIN_GAP_R = (3.0 * MIN_GAP_V / (4.0 * math.pi)) ** (1.0 / 3.0)

    # 空间哈希: cell_size 基于表面粒子中位数半径
    cell_size = max(1.0 * float(np.median(particles.radius[active_idx])), 0.5)

    # 将每个粒子分配到网格单元格
    cell_map = {}  # (cx, cy, cz) → [active_idx indices]
    for ii in range(n_active):
        ai = active_idx[ii]
        pos = particles.pos[ai]
        cx = int(math.floor(pos[0] / cell_size))
        cy = int(math.floor(pos[1] / cell_size))
        cz = int(math.floor(pos[2] / cell_size))
        key = (cx, cy, cz)
        if key not in cell_map:
            cell_map[key] = []
        cell_map[key].append(ii)

    # 预估总运算量: 同一格 C(n,2) + 邻格 n_i * n_j
    total_checks = 0
    processed = set()
    for (cx, cy, cz), members in cell_map.items():
        n = len(members)
        total_checks += n * (n - 1) // 2  # within-cell pairs
        for dcx in [-1, 0, 1]:
            for dcy in [-1, 0, 1]:
                for dcz in [-1, 0, 1]:
                    if dcx == 0 and dcy == 0 and dcz == 0:
                        continue
                    nk = (cx + dcx, cy + dcy, cz + dcz)
                    if nk in cell_map and nk not in processed:
                        total_checks += n * len(cell_map[nk])
        processed.add((cx, cy, cz))

    # 如果运算量太大, 跳过 (缝隙由 step1b 处理)
    if total_checks > max_checks:
        return 0

    # 收集候选: 只检查同一格或相邻格的三元组
    candidates = []
    used_active = set()
    processed.clear()

    for (cx, cy, cz), members in cell_map.items():
        if len(members) < 3:
            continue

        # 本格的三元组
        for i_idx in range(len(members)):
            ai = active_idx[members[i_idx]]
            if not particles.active[ai]:
                continue
            r_i = float(particles.radius[ai])
            pos_i = tuple(particles.pos[ai])
            for j_idx in range(i_idx + 1, len(members)):
                aj = active_idx[members[j_idx]]
                if not particles.active[aj]:
                    continue
                r_j = float(particles.radius[aj])
                pos_j = tuple(particles.pos[aj])
                d_ij = np.linalg.norm(particles.pos[ai] - particles.pos[aj])
                if abs(d_ij - (r_i + r_j)) > GEOMETRIC_TOLERANCE * (r_i + r_j):
                    continue
                for k_idx in range(j_idx + 1, len(members)):
                    ak = active_idx[members[k_idx]]
                    if not particles.active[ak]:
                        continue
                    r_k = float(particles.radius[ak])
                    pos_k = tuple(particles.pos[ak])
                    gap = _three_sphere_gap(pos_i, r_i, pos_j, r_j, pos_k, r_k)
                    if gap is None:
                        continue
                    gap_pos, gap_r = gap
                    if gap_r < MIN_GAP_R:
                        continue
                    if ai in used_active or aj in used_active or ak in used_active:
                        continue
                    candidates.append((gap_r, gap_pos, ai, aj, ak))

        # 相邻格: 只检查与未处理格的对
        for dcx in [-1, 0, 1]:
            for dcy in [-1, 0, 1]:
                for dcz in [-1, 0, 1]:
                    if dcx == 0 and dcy == 0 and dcz == 0:
                        continue
                    nk = (cx + dcx, cy + dcy, cz + dcz)
                    if nk not in cell_map or nk in processed:
                        continue
                    nbrs = cell_map[nk]
                    for i_idx in range(len(members)):
                        ai = active_idx[members[i_idx]]
                        if not particles.active[ai]:
                            continue
                        r_i = float(particles.radius[ai])
                        pos_i = tuple(particles.pos[ai])
                        for j_idx in range(len(nbrs)):
                            aj = active_idx[nbrs[j_idx]]
                            if not particles.active[aj]:
                                continue
                            r_j = float(particles.radius[aj])
                            pos_j = tuple(particles.pos[aj])
                            d_ij = np.linalg.norm(particles.pos[ai] - particles.pos[aj])
                            if abs(d_ij - (r_i + r_j)) > GEOMETRIC_TOLERANCE * (r_i + r_j):
                                continue
                            # 找第三个粒子: 可在本格或邻居格
                            for k_idx in range(i_idx + 1, len(members)):
                                ak = active_idx[members[k_idx]]
                                if ak == aj:
                                    continue
                                if not particles.active[ak]:
                                    continue
                                r_k = float(particles.radius[ak])
                                pos_k = tuple(particles.pos[ak])
                                gap = _three_sphere_gap(pos_i, r_i, pos_j, r_j, pos_k, r_k)
                                if gap is None:
                                    continue
                                gap_pos, gap_r = gap
                                if gap_r < MIN_GAP_R:
                                    continue
                                if ai in used_active or aj in used_active or ak in used_active:
                                    continue
                                candidates.append((gap_r, gap_pos, ai, aj, ak))
        processed.add((cx, cy, cz))

    # 按 gap_r 降序
    candidates.sort(key=lambda x: -x[0])

    activated = 0
    for gap_r, gap_pos, ai, aj, ak in candidates:
        if activated >= 20:  # 每帧最多 20 个新缝隙粒子
            break
        if ai in used_active or aj in used_active or ak in used_active:
            continue
        new_idx = int(np.sum(particles.active))
        if new_idx >= particles.max_n:
            break
        if new_idx >= particles.N:
            particles.resize(min(particles.N * 2 + 1, particles.max_n))

        V_gap = (4.0 / 3.0) * math.pi * (gap_r ** 3)
        particles.pos[new_idx] = gap_pos
        particles.radius[new_idx] = gap_r
        particles.degree[new_idx] = 0
        particles.active[new_idx] = True
        particles.uid[new_idx] = f"gap_{particles.active_count()}"
        particles.initial_volume[new_idx] = V_gap
        particles.initial_degree[new_idx] = 3

        particles.add_connection(ai, new_idx)
        particles.add_connection(aj, new_idx)
        particles.add_connection(ak, new_idx)

        used_active.add(ai)
        used_active.add(aj)
        used_active.add(ak)
        activated += 1

    return activated


def step1b_gap_fill(particles, max_per_frame: int = 20) -> int:
    """Apollonian 递归填充: 中心介导的两层扫描。

    第一层: 中心 + surface_i + surface_j:
        当两表面球相切时, 计算它们与中心之间的缝隙球。
        这是"三球定界"的直接实现, 新球位于表面壳层上。
    
    第二层: surface_i + surface_j + surface_k (当第一层无候选时):
        三个两两相切的表面球之间的缝隙。
        这是更深的递归层级。

    新球半径上限: max_gap_r = 2.0 (V ≈ 33), 防止巨大缝隙球破坏壳层。
    新球体积 >= MIN_GAP_V (截断递归深度, V=0.3)。

    Returns:
        本帧添加的粒子数
    """
    center_idx = _find_center(particles)
    if center_idx < 0:
        return 0

    center_pos = particles.pos[center_idx]
    center_r = float(particles.radius[center_idx])

    active_idx = np.where(particles.active)[0]
    surf_idx_list = [i for i in active_idx if i != center_idx]
    if len(surf_idx_list) < 2:
        return 0

    MIN_GAP_V = 0.3
    MIN_GAP_R = (3.0 * MIN_GAP_V / (4.0 * math.pi)) ** (1.0 / 3.0)
    MAX_GAP_V = 33.0
    MAX_GAP_R = (3.0 * MAX_GAP_V / (4.0 * math.pi)) ** (1.0 / 3.0)
    # MAX_GAP_R ≈ 1.99

    # 检测表面-表面相切
    tol = GEOMETRIC_TOLERANCE
    n = len(surf_idx_list)
    tangent_adj = {}
    for ii in range(n):
        i = surf_idx_list[ii]
        pi = particles.pos[i]
        ri = float(particles.radius[i])
        for jj in range(ii + 1, n):
            j = surf_idx_list[jj]
            d = float(np.linalg.norm(pi - particles.pos[j]))
            rsum = ri + float(particles.radius[j])
            if abs(d - rsum) <= tol * rsum:
                tangent_adj[(ii, jj)] = True
                tangent_adj[(jj, ii)] = True

    if len(tangent_adj) < 1:
        return 0

    candidates = []

    # 第一层: 中心介导 (中心 + surface_i + surface_j)
    for ii in range(n):
        i = surf_idx_list[ii]
        for jj in range(ii + 1, n):
            if (ii, jj) not in tangent_adj:
                continue
            j = surf_idx_list[jj]
            gap = _three_sphere_gap(
                center_pos, center_r,
                particles.pos[i], float(particles.radius[i]),
                particles.pos[j], float(particles.radius[j])
            )
            if gap is None:
                continue
            gap_pos, gap_r = gap
            if gap_r < MIN_GAP_R or gap_r > MAX_GAP_R:
                continue
            candidates.append((gap_r, gap_pos, i, j, -1))  # k=-1 表示中心介导

    # 第二层: 纯表面 (surface_i + surface_j + surface_k)
    if not candidates:
        for ii in range(n):
            i = surf_idx_list[ii]
            for jj in range(ii + 1, n):
                if (ii, jj) not in tangent_adj:
                    continue
                for kk in range(jj + 1, n):
                    if (ii, kk) not in tangent_adj or (jj, kk) not in tangent_adj:
                        continue
                    j = surf_idx_list[jj]
                    k = surf_idx_list[kk]
                    gap = _three_sphere_gap(
                        particles.pos[i], float(particles.radius[i]),
                        particles.pos[j], float(particles.radius[j]),
                        particles.pos[k], float(particles.radius[k])
                    )
                    if gap is None:
                        continue
                    gap_pos, gap_r = gap
                    if gap_r < MIN_GAP_R or gap_r > MAX_GAP_R:
                        continue
                    candidates.append((gap_r, gap_pos, i, j, k))

    if not candidates:
        return 0

    # 按 gap_r 升序 (最小缝隙优先, 从内到外)
    candidates.sort(key=lambda x: x[0])

    added = 0
    used = set()
    for gap_r, gap_pos, i, j, k in candidates:
        if added >= max_per_frame:
            break
        if i in used or j in used:
            continue
        if k >= 0 and k in used:
            continue

        new_idx = int(np.sum(particles.active))
        if new_idx >= particles.max_n:
            break
        if new_idx >= particles.N:
            particles.resize(min(particles.N * 2 + 1, particles.max_n))

        V_gap = (4.0 / 3.0) * math.pi * (gap_r ** 3)
        particles.pos[new_idx] = np.array(gap_pos, dtype=np.float32)
        particles.radius[new_idx] = gap_r
        particles.degree[new_idx] = 0
        particles.active[new_idx] = True
        particles.uid[new_idx] = f"gap_{particles.active_count()}"
        particles.initial_volume[new_idx] = V_gap

        if k < 0:
            # 中心介导: 连接中心 + i + j
            particles.initial_degree[new_idx] = 3
            particles.add_connection(center_idx, new_idx)
            particles.add_connection(new_idx, i)
            particles.add_connection(new_idx, j)
            used.add(i); used.add(j)
        else:
            # 纯表面: 连接 i + j + k (补全四面体)
            particles.initial_degree[new_idx] = 3
            particles.add_connection(new_idx, i)
            particles.add_connection(new_idx, j)
            particles.add_connection(new_idx, k)
            particles.add_connection(i, j)
            particles.add_connection(j, k)
            particles.add_connection(k, i)
            used.add(i); used.add(j); used.add(k)

        added += 1

    return added

def step2_connect(particles, star_mode: bool = False,
                  pregrowth_knn: int = 0,
                  skip_center: bool = False) -> int:
    """检测相切关系，更新度数 (degree)。

    连接模式:
        1. 几何相切 (通用): 任意两球体 distance ≈ r_i + r_j
        2. 表面角邻接 (star_mode): 表面球体在球面上的角重叠
        3. 最近邻连接 (pregrowth_knn > 0): 表面球体连接 k 个最近邻

    Args:
        particles: ParticleArray
        star_mode: 启用表面角邻接
        pregrowth_knn: pre-growth 阶段, 每个表面 coda 连接 k 个最近表面邻居
        skip_center: 跳过中心粒子的几何相切检测 (用于避免中心成为超枢纽)

    Returns:
        本帧新建的边数
    """
    active_idx = np.where(particles.active)[0]
    n_active = len(active_idx)
    if n_active < 2:
        return 0

    total_edges = 0
    tol = GEOMETRIC_TOLERANCE

    # 识别表面和中心粒子 (star_mode 或 pregrowth_knn 时需要)
    if star_mode or pregrowth_knn > 0:
        center_idx = None
        for i in range(particles.N):
            if particles.active[i] and str(particles.uid[i]).startswith('cent_'):
                center_idx = i
                break
        surf_mask = np.array([
            particles.active[i] and i != center_idx
            for i in range(particles.N)
        ])
        surf_act_idx = np.where(surf_mask)[0]
        n_surf = len(surf_act_idx)
    else:
        surf_act_idx = np.array([], dtype=int)
        n_surf = 0
        center_idx = None

    # === 最近邻连接 (pre-growth) ===
    # 给表面 coda 建立 k 个最近邻连接 → 度数增长
    # 每个连接只计一次 (不重复)
    if pregrowth_knn > 0 and n_surf >= 2 and center_idx is not None:
        R = np.linalg.norm(particles.pos[surf_act_idx[0]])

        # 计算所有表面 coda 之间的角距
        n = n_surf
        ang_dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dot = np.dot(particles.pos[surf_act_idx[i]],
                             particles.pos[surf_act_idx[j]]) / (R * R)
                dot = np.clip(dot, -1.0, 1.0)
                ang_dists[i, j] = ang_dists[j, i] = math.acos(dot)

        # 建立无向图连接: 每个 coda 连接 k 个最近邻
        # 用集合避免重复计数; 仅添加不存在的新连接
        edges_added = 0
        for i in range(n):
            idx_i = surf_act_idx[i]
            nearest = np.argsort(ang_dists[i])[1:pregrowth_knn + 1]
            for j in nearest:
                if j < n:
                    idx_j = surf_act_idx[j]
                    if particles.add_connection(idx_i, idx_j):
                        edges_added += 1
        total_edges += edges_added

    # === 表面几何相切 (star_mode, capped) ===
    # 检测表面球体之间的几何相切: distance ≈ r_i + r_j
    # 每个粒子每帧最多连接 k_max 个最近邻几何相切粒子,
    # 且总度不超过 max_deg (防止累积过度连接)。
    if star_mode and n_surf >= 2 and center_idx is not None:
        k_max = 6
        max_deg = 8  # 全局上限: 每个表面粒子最多 8 条表面-表面边
        tol = GEOMETRIC_TOLERANCE
        edges_added = 0
        n = n_surf
        # 建立角距+几何相切列表: 每个粒子 i, 收集相切粒子 j 的角距
        surf_pos = particles.pos[surf_act_idx]
        Rs = np.linalg.norm(surf_pos, axis=1)
        for i in range(n):
            idx_i = surf_act_idx[i]
            if not particles.active[idx_i]:
                continue
            if int(particles.degree[idx_i]) >= max_deg:
                continue
            pi = particles.pos[idx_i]
            ri = float(particles.radius[idx_i])
            Ri = float(Rs[i])
            if Ri < 1e-12:
                continue
            # 收集所有相切候选
            candidates_i = []  # [(ang_dist, j_idx)]
            for j in range(n):
                if i >= j:
                    continue
                idx_j = surf_act_idx[j]
                if not particles.active[idx_j]:
                    continue
                if int(particles.degree[idx_j]) >= max_deg:
                    continue
                d = float(np.linalg.norm(pi - particles.pos[idx_j]))
                rsum = ri + float(particles.radius[idx_j])
                if abs(d - rsum) <= tol * rsum:
                    Rj = float(Rs[j])
                    if Rj < 1e-12:
                        continue
                    dot = float(np.dot(pi, particles.pos[idx_j])) / (Ri * Rj)
                    dot = max(-1.0, min(1.0, dot))
                    ang = math.acos(dot)
                    candidates_i.append((ang, j))
            # 按角距升序, 取前 k_max
            candidates_i.sort(key=lambda x: x[0])
            for ang, j in candidates_i[:k_max]:
                idx_j = surf_act_idx[j]
                if particles.add_connection(idx_i, idx_j):
                    edges_added += 1
        total_edges += edges_added

    return total_edges


# ============================================================
# Step 3: 变化体积 — degree → radius
# ============================================================

def step3_volume(particles):
    """每个粒子根据当前度数更新体积和半径。

    V = initial_volume + (degree - initial_degree)
    r = (3V/4π)^(1/3)

    种子球 (初始):  initial_volume=1, initial_degree=0 → V = 1 + degree
    缝隙球 (笛卡尔): initial_volume=V_gap, initial_degree=3 → V = V_gap + (deg - 3)
    缝隙球初始 deg=3 时 V = V_gap，不会自动达到 V=4。

    SPUM 公理 (§2): 空间体量由度数决定, 但初始体积由诞生条件决定。
    """
    active = particles.active
    iv = particles.initial_volume.astype(np.float32)
    id_ = particles.initial_degree.astype(np.int32)
    deg = particles.degree

    # V = initial_volume + (degree - initial_degree)
    V = iv + (deg - id_).astype(np.float32)
    V[V < 0.5] = 0.5  # 避免负体积导致半径异常

    # r = (3V/4π)^(1/3)
    particles.radius[active] = (3.0 * V[active] / (4.0 * math.pi)) ** (1.0 / 3.0)


# ============================================================
# Step 3c: 相邻相切强制 — 无背景空间
# ============================================================

def enforce_tangency(particles, relax=True):
    """强制所有相邻球体精确相切。

    SPUM 公理: 不存在背景空间。所有相邻球体必相切。
    坐标系统是连接拓扑的投影，不是预先存在的容器。

    实现: 从 tangent 角度数据重建笛卡尔坐标。
    每个粒子存储邻接在该粒子球面上的切点球面角 (θ, φ).
    BFS 从第一个活跃粒子起, 利用 tangent 角度 + 半径重建全局坐标。

    参数:
        relax: 是否运行松弛步骤。对四面体填充建议 False (BFS 即精确)。
              对中心-表面拓扑建议 True (BFS 后还需调整).
    """
    particles.reconstruct_positions(relax=relax)


# ============================================================
# Step 3b: 不可入性 — 重叠强制消解
# ============================================================

def _spherical_to_cart(theta, phi):
    """(θ, φ) → 单位向量 (x, y, z)"""
    st = math.sin(theta)
    return (st * math.cos(phi), st * math.sin(phi), math.cos(theta))

def _cart_to_spherical(x, y, z):
    """单位向量 → (θ, φ)"""
    theta = math.acos(max(-1.0, min(1.0, z)))
    phi = math.atan2(y, x)
    return (theta, phi)

def _angular_distance(theta1, phi1, theta2, phi2):
    """两球面点的大圆角距。"""
    c = math.sin(theta1) * math.sin(theta2) * math.cos(phi1 - phi2) + math.cos(theta1) * math.cos(theta2)
    return math.acos(max(-1.0, min(1.0, c)))

def _rotate_away_from(vx, vy, vz, ref_x, ref_y, ref_z, delta):
    """沿大圆弧将单位向量 v 旋转远离 ref 的角度 delta。

    公式: Rodrigues 旋转, 轴 = v × ref / |v × ref|
    """
    # 旋转轴: v × ref (归一化)
    cx = vy * ref_z - vz * ref_y
    cy = vz * ref_x - vx * ref_z
    cz = vx * ref_y - vy * ref_x
    cn = math.sqrt(cx*cx + cy*cy + cz*cz)
    if cn < 1e-12:
        return (vx, vy, vz)  # 共线
    cx /= cn; cy /= cn; cz /= cn

    cos_d = math.cos(delta)
    sin_d = math.sin(delta)
    dot = cx * vx + cy * vy + cz * vz

    # v_rot = v*cos_d + (axis×v)*sin_d + axis*(axis·v)*(1-cos_d)
    cross_x = cy * vz - cz * vy
    cross_y = cz * vx - cx * vz
    cross_z = cx * vy - cy * vx

    rx = vx * cos_d + cross_x * sin_d + cx * dot * (1.0 - cos_d)
    ry = vy * cos_d + cross_y * sin_d + cy * dot * (1.0 - cos_d)
    rz = vz * cos_d + cross_z * sin_d + cz * dot * (1.0 - cos_d)

    rn = math.sqrt(rx*rx + ry*ry + rz*rz)
    if rn < 1e-12:
        return (vx, vy, vz)
    return (rx/rn, ry/rn, rz/rn)

def _target_alpha_on_sphere(R_parent, r_j, r_k):
    """两球在母体球面上的目标角距。

    两球 j, k 同时与母体 i 相切:
        i 中心到 j 中心 = R_i + r_j
        i 中心到 k 中心 = R_i + r_k
        j 与 k 相切: jk 中心距 = r_j + r_k

    余弦定理求 α:
        cos(α) = [(R_i+r_j)² + (R_i+r_k)² - (r_j+r_k)²] / [2(R_i+r_j)(R_i+r_k)]
    """
    Rj = R_parent + r_j
    Rk = R_parent + r_k
    cos_α = (Rj*Rj + Rk*Rk - (r_j + r_k)**2) / (2.0 * Rj * Rk)
    return math.acos(max(-1.0, min(1.0, cos_α)))


# ============================================================
# Step 3b: 不可入性 — 球面滑动消解相入
# ============================================================

def step3b_enforce_impenetrability(particles, max_iter: int = 10,
                                   allow_disconnect: bool = True) -> int:
    """纯局部球面滑动消解相入。

    每帧每 coda 独立运行:
        coda_i 检查每对邻居 (j, k) 在 i 球面上的角距。
        如果角距 < 物理半径所需的目标角距 → 沿大圆弧推开 j 和 k。
        更新 tangent[i][j] 和 tangent[i][k] (纯局部修改)。

    Args:
        particles: ParticleArray
        max_iter: 每个 coda 的最大滑动轮数

    Returns:
        本帧的推离操作数
    """
    tol = GEOMETRIC_TOLERANCE
    resolved = 0

    for i in range(particles.N):
        if not particles.active[i]:
            continue
        nbrs = particles.tangent[i]
        nbr_list = list(nbrs.keys())
        if len(nbr_list) < 2:
            continue

        r_i = float(particles.radius[i])

        for _ in range(max_iter):
            any_push = False
            for a_idx in range(len(nbr_list)):
                j = nbr_list[a_idx]
                if j not in nbrs or not particles.active[j]:
                    continue
                for b_idx in range(a_idx + 1, len(nbr_list)):
                    k = nbr_list[b_idx]
                    if k not in nbrs or not particles.active[k]:
                        continue

                    theta_j, phi_j = nbrs[j]
                    theta_k, phi_k = nbrs[k]

                    # 当前角距
                    α = _angular_distance(theta_j, phi_j, theta_k, phi_k)
                    if α < 1e-12:
                        continue

                    # 所需目标角距
                    r_j = float(particles.radius[j])
                    r_k = float(particles.radius[k])
                    α_req = _target_alpha_on_sphere(r_i, r_j, r_k)

                    if α < α_req * (1.0 - tol):
                        δ = min(0.2, (α_req - α) / 2.0)

                        v_j = _spherical_to_cart(theta_j, phi_j)
                        v_k = _spherical_to_cart(theta_k, phi_k)

                        v_j_new = _rotate_away_from(v_j[0], v_j[1], v_j[2],
                                                    v_k[0], v_k[1], v_k[2], δ)
                        v_k_new = _rotate_away_from(v_k[0], v_k[1], v_k[2],
                                                    v_j[0], v_j[1], v_j[2], δ)

                        # 更新 i 的视角
                        nbrs[j] = _cart_to_spherical(v_j_new[0], v_j_new[1], v_j_new[2])
                        nbrs[k] = _cart_to_spherical(v_k_new[0], v_k_new[1], v_k_new[2])

                        # 传播: 更新 j 看 i 和 k 看 i 的视角 (方向取反)
                        # i 在 j 球面上的位置 = (π - θ_j, φ_j + π)
                        θ_j_new = math.acos(max(-1.0, min(1.0, v_j_new[2])))
                        φ_j_new = math.atan2(v_j_new[1], v_j_new[0])
                        φ_j_opp = φ_j_new + math.pi
                        if φ_j_opp > math.pi: φ_j_opp -= 2.0 * math.pi
                        elif φ_j_opp < -math.pi: φ_j_opp += 2.0 * math.pi
                        particles.tangent[j][i] = (math.pi - θ_j_new, φ_j_opp)
                        θ_k_new = math.acos(max(-1.0, min(1.0, v_k_new[2])))
                        φ_k_new = math.atan2(v_k_new[1], v_k_new[0])
                        φ_k_opp = φ_k_new + math.pi
                        if φ_k_opp > math.pi: φ_k_opp -= 2.0 * math.pi
                        elif φ_k_opp < -math.pi: φ_k_opp += 2.0 * math.pi
                        particles.tangent[k][i] = (math.pi - θ_k_new, φ_k_opp)

                        any_push = True
                        resolved += 1

            if not any_push:
                break

        # 挤出检测: 仅母体对有孩子的邻居执行 (不互相挤)
        # 子球被挤出母体球面 → 释放到更高壳层
        if allow_disconnect and len(nbr_list) >= 3:
            # 检查 i 是否是"母体" (度数大或有中心的 uid 前缀)
            is_parent = bool(str(particles.uid[i]).startswith('cent_'))
            if not is_parent:
                continue
            for j in nbr_list:
                if j not in nbrs or not particles.active[j]:
                    continue
                overlap_count = 0
                total = 0
                for k in nbr_list:
                    if k == j or k not in nbrs or not particles.active[k]:
                        continue
                    total += 1
                    r_j = float(particles.radius[j])
                    r_k = float(particles.radius[k])
                    α_req = _target_alpha_on_sphere(r_i, r_j, r_k)
                    θ_j, φ_j = nbrs[j]
                    θ_k, φ_k = nbrs[k]
                    α = _angular_distance(theta_j, phi_j, theta_k, phi_k)
                    if α < α_req * (1.0 - tol):
                        overlap_count += 1
                if total >= 2 and overlap_count > total * 0.2:
                    particles.remove_connection(i, j)
                    break

    return resolved


# ============================================================
# Step 4: 判断悬挂
# ============================================================

def step4_dangling(particles, seed_only: bool = False) -> np.ndarray:
    """标记度数 < 3 的活性粒子为悬挂。

    SPUM 公理: 度数 < 3 的节点是悬挂节点，无法在三角剖分中闭合。

    Args:
        seed_only: 仅标记初始种子粒子 (cent_, fcc_), 不标记缝隙球体

    Returns:
        (N,) 布尔数组，True=悬挂
    """
    if seed_only:
        # 只标记种子球体 (cent_, fcc_), 跳过缝隙球体 (gap_)
        result = np.zeros(particles.N, dtype=np.bool_)
        for i in range(particles.N):
            if not particles.active[i]:
                continue
            uid = str(particles.uid[i])
            if uid.startswith('cent_') or uid.startswith('fcc_'):
                if particles.degree[i] < 3:
                    result[i] = True
        return result
    return particles.active & (particles.degree < 3)


# ============================================================
# Step 5: 删除悬挂
# ============================================================

def step5_purge(particles, dangling_mask: np.ndarray) -> int:
    """悬挂粒子回归潜在池。

    SPUM 公理 (§4):
        删除的节点失去坐标，回归潜在状态，作为下一帧的创生元。
    附加: 表面 coda 被删除时, 中心也失去该连接。

    Returns:
        本帧删除的悬挂粒子数
    """
    purge_count = int(np.sum(dangling_mask))
    if purge_count == 0:
        return 0

    # 找到中心 coda
    center_idx = None
    for i in range(particles.N):
        if particles.active[i] and str(particles.uid[i]).startswith('cent_'):
            center_idx = i
            break

    # 标记为潜在; 如果是表面 coda, 也断中心连接
    for i in np.where(dangling_mask)[0]:
        particles.active[i] = False
        particles.remove_all_connections(i)
        if center_idx is not None and str(particles.uid[i]).startswith('surf_'):
            pass  # remove_all_connections 已处理

    return purge_count


# ============================================================
# 再生日: 删除的球体在网络外围重新出现
# ============================================================

def reincarnate_to_minimum(particles, min_active: int = 100, batch_mode: bool = False):
    """维持网络最低粒子数。

    无方向优化，直接使用 Fibonacci 候选方向顺序填充。
    如果是批量填充 (batch_mode=True)，跳过 enforce_tangency 加速。

    Args:
        batch_mode: True = 不调用 enforce_tangency (由外层统一调)

    Returns:
        本帧再生的球体数
    """
    active = particles.active_count()
    if active >= min_active:
        return 0

    center_idx = None
    for i in range(particles.N):
        if particles.active[i] and str(particles.uid[i]).startswith('cent_'):
            center_idx = i
            break
    if center_idx is None:
        return 0

    need = min(min_active - active, 20)
    center_r = float(particles.radius[center_idx])
    unit_vol_r = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

    # 预计算 Fibonacci 候选方向
    n_cand = min(500, need * 5 + 50)
    candidates = _fibonacci_sphere(n_cand)

    # 跳过已有表面方向
    used_dirs = set()
    for i in range(particles.N):
        if particles.active[i] and i != center_idx:
            norm = float(np.linalg.norm(particles.pos[i]))
            if norm > 1e-12:
                d = tuple(np.round(particles.pos[i] / norm, 3))
                used_dirs.add(d)

    # 计算表面粒子的平均径向距离 (将新粒子放在相同球壳上)
    surf_pos_list = np.array([particles.pos[i] for i in range(particles.N)
                              if particles.active[i] and i != center_idx], dtype=np.float32)
    avg_surf_R = float(np.mean(np.linalg.norm(surf_pos_list, axis=1))) if len(surf_pos_list) > 0 else (center_r + unit_vol_r)

    added = 0
    cand_idx = 0

    # 收集现有表面 (仅一次)
    surface_indices = [i for i in range(particles.N)
                       if particles.active[i] and i != center_idx]
    active_count = len(surface_indices) + 1  # +1 for center

    for _ in range(need):
        # 找未使用的方向
        best_dir = None
        for _ in range(n_cand * 2):
            if cand_idx >= n_cand:
                cand_idx = 0
            d = tuple(np.round(candidates[cand_idx], 3))
            cand_idx += 1
            if d not in used_dirs:
                best_dir = np.array(d, dtype=np.float32)
                break

        if best_dir is None:
            # 所有方向都用完了 — 允许重复
            best_dir = candidates[cand_idx % n_cand]

        new_pos = best_dir * avg_surf_R
        new_idx = active_count  # 直接使用跟踪的计数
        if new_idx >= particles.max_n:
            break
        if new_idx >= particles.N:
            particles.resize(min(particles.N * 2 + 1, particles.max_n))

        particles.pos[new_idx] = new_pos
        particles.radius[new_idx] = unit_vol_r
        particles.degree[new_idx] = 0
        particles.active[new_idx] = True
        particles.uid[new_idx] = f"rebirth_{new_idx:04d}"
        particles.initial_volume[new_idx] = 1.0
        particles.initial_degree[new_idx] = 0

        # 连接到中心 + 2 个最近表面 (确保初始 deg ≥ 3)
        particles.add_connection(center_idx, new_idx)
        k = 0
        for si in surface_indices:
            if k >= 2: break
            if not particles.connected(new_idx, si):
                particles.add_connection(new_idx, si)
                k += 1

        surface_indices.append(new_idx)
        active_count += 1
        used_dirs.add(tuple(np.round(best_dir, 3)))
        added += 1

    # 非批量模式才调用 enforce_tangency
    if added > 0 and not batch_mode:
        enforce_tangency(particles)

    return added


# ============================================================
# 完整帧
# ============================================================

def run_full_frame(particles, star_mode: bool = False,
                   no_purge: bool = False,
                   pregrowth_knn: int = 0,
                   allow_disconnect: bool = True,
                   skip_center: bool = False,
                   gap_fill_per_frame: int = 20,
                   seed_purge_only: bool = False,
                   min_active: int = 0) -> Dict:
    """一帧 = 5 步 + 不可入性强制。

    SPUM 公理体系:
        §4: 一帧 = 创生 → 连接 → 体积 → 不可入 → 悬挂 → 删除
        §不可入: 所有球体均不可入 (distance ≥ r1 + r2)
        §无背景: 所有相邻球体必相切 — 坐标是拓扑的投影

    Args:
        particles: ParticleArray 实例
        star_mode: 是否启用表面角邻接连接
        no_purge: 是否跳过悬挂检测和删除 (pre-growth 阶段)
        pregrowth_knn: pre-growth 阶段, 每个表面 coda 连接 k 个最近邻
        allow_disconnect: 是否允许重叠时断连 (False=预增长, 只标记)
        skip_center: 是否禁止中心建立新连接
        gap_fill_per_frame: 每帧最大缝隙填充数
        seed_purge_only: 仅删除初始种子粒子 (cent_, fcc_), 跳过缝隙球体
        min_active: 维持最小活性粒子数 (<=0 表示不限制)

    Returns:
        {created, connected, overlaps_resolved, dangling_marked, purged,
         reincarnated, active_before, active_after, degree_histogram}
    """
    active_before = particles.active_count()

    # Step 3: 变化体积 (degree → radius)
    # 用户逻辑: 先增长体积, 空隙自然出现
    step3_volume(particles)

    # Step 3c: 相邻相切强制 — 体积变化后, 坐标重构产生空隙
    enforce_tangency(particles)

    # Step 1: 创生 — 在空隙中填入新球体
    created = step1_create(particles)

    # Step 1b: 如果无 3 体缝隙, 尝试球面空隙填充
    if created == 0:
        created = step1b_gap_fill(particles, max_per_frame=gap_fill_per_frame)

    # Step 2: 连接 (表面角邻接 + 最近邻)
    connected = step2_connect(particles, star_mode=star_mode,
                              pregrowth_knn=pregrowth_knn,
                              skip_center=skip_center)

    # Step 3b: 不可入性 — 重叠球体被挤压
    overlaps_resolved = step3b_enforce_impenetrability(particles,
                                                       allow_disconnect=allow_disconnect)

    if no_purge:
        purged = 0
        dangling_count = 0
    else:
        # Step 4: 判断悬挂
        dangling = step4_dangling(particles, seed_only=seed_purge_only)
        dangling_count = int(np.sum(dangling))

        # Step 5: 删除悬挂
        purged = step5_purge(particles, dangling)

    # Step 6: 再生 — 维持最低粒子数
    if min_active > 0 and particles.active_count() < min_active:
        reincarnated = reincarnate_to_minimum(particles, min_active=min_active)
    else:
        reincarnated = 0

    active_after = particles.active_count()

    return {
        "created": created,
        "connected": connected,
        "overlaps_resolved": overlaps_resolved,
        "dangling_marked": dangling_count,
        "purged": purged,
        "reincarnated": reincarnated,
        "active_before": active_before,
        "active_after": active_after,
        "degree_histogram": particles.degree_histogram(),
    }
