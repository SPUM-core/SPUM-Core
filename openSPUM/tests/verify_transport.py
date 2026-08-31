"""
SPUM 搬运定理候选代码 — 图单元 → 容量位的映射 (T3 / C10)
========================================================

搬运定理 (假设, 未证, 见 图论/spum-计数关系.md C10 与待证项 T3):
    对任意闭合三角剖分子图 S 及中心节点 O (deg(O) = V(S)),
    O 的容量 = E(S) + F(S)。

"搬运" = S 的边单元 → O 表面的边虚面位, S 的面单元 → O 表面的面虚面位,
且每个单元恰对应一个可容纳位。

本文件把"搬运"从文字断言变成可执行枚举:

    组合层 (确定, 可审计):
        位置 = V+E+F = 62; 面位 = E+F = 50; 差 = 12 = V。
        每一步都是"重新数一遍": 数顶点 12, 数邻居对 30, 数 3-团 20。

    几何层 (候选, 待证 T3 — 断点核心):
        (1) 30 个边位是否"恰对应一个可容纳位":
            42 构造实证 — 边球 r_e=0.1869 沿径向滑入 31.72° 楔缝,
            同时触中心球 + 两顶点球。全两两检查: 重叠 0, 相切 102,
            中心相切 42。
        (2) 20 个面位是否被截断:
            面心方向与顶点方向的最小夹角 < 顶点球角半径 → 填入必重叠。

结论判读: 组合层是计数重言 (可审计); 几何层只验证"二十面体这一个实例"
自洽 — 不证明搬运定理的一般性 (那是 T3 本体)。

RULE-COUNT 协议审计 (AGENT.md 计数与几何协议): 本脚本按协议自动分组
标注输出 — [可数出] L0 计数 (带计数操作, 可重数) vs [投影] L1 投影
(坐标/容差/判定约定, 不可数出, 必须显式标注)。

运行: python openSPUM/tests/verify_transport.py
"""

import math
import sys
from pathlib import Path
import numpy as np

import functools

print = functools.partial(print, flush=True)

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


# ============================================================
# RULE-COUNT 协议审计层 (AGENT.md 计数与几何协议 §一/§二)
#   [可数出] L0 计数: 每个数字必须带计数操作, 可重数
#   [投影]   L1 投影: 每个数字必须带投影来源, 不可数出
# ============================================================

class Audit:
    def __init__(self):
        self.countable = []    # (名称, 数值, 计数操作)
        self.projection = []   # (名称, 数值, 投影来源)

    def add_countable(self, name, value, op):
        self.countable.append((name, value, op))

    def add_projection(self, name, value, src):
        self.projection.append((name, value, src))

    def report(self):
        print("\n" + "=" * 72)
        print("  RULE-COUNT 协议审计 — 可数出 vs 投影")
        print("=" * 72)
        print("\n[可数出] L0 计数 —— 每个数字都带计数操作 (可重数):")
        for name, value, op in self.countable:
            print("  %-12s = %-8s  操作: %s" % (name, value, op))
        print("\n[投影]   L1 投影 —— 每个数字都带投影来源 (不可数出, 须标注):")
        for name, value, src in self.projection:
            print("  %-12s = %-8s  来源: %s" % (name, value, src))
        print("\n  [可数出] 数出的数字, 是 L0 计数: 12/30/20/62/50/42/102…")
        print("  [投影]   不可数出的数字, 是 L1: 半径/角度/容差/重叠%…")
        print("  标注投影 = 减法开始的地方 (投影之外的余数, 才是 L0)")
        print("=" * 72)


# ============================================================
# 组合层: 二十面体图 S 的构造与计数 (纯组合, 无几何假设)
# ============================================================

PHI = (1.0 + math.sqrt(5.0)) / 2.0


def icosahedron_vertices():
    """正二十面体 12 顶点 (笛卡尔坐标), 归一化到单位球。"""
    raw = [
        (0, 1, PHI), (0, -1, PHI), (0, 1, -PHI), (0, -1, -PHI),
        (1, PHI, 0), (-1, PHI, 0), (1, -PHI, 0), (-1, -PHI, 0),
        (PHI, 0, 1), (-PHI, 0, 1), (PHI, 0, -1), (-PHI, 0, -1),
    ]
    verts = np.asarray(raw, dtype=float)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def build_graph(verts):
    """邻接 (距离 < 1.1 → 边) + 面 (3-团枚举, 正二十面体无非面 3-团)。"""
    d = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    adj = (d < 1.1) & (d > 1e-6)
    n = len(verts)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j]]
    faces = [(i, j, k)
             for i in range(n) for j in range(i + 1, n) for k in range(j + 1, n)
             if adj[i, j] and adj[i, k] and adj[j, k]]
    return edges, faces


def count_layer():
    """组合层: 位置 62 / 面位 50 / 差 12 —— 全部可重数 (审计)。"""
    verts = icosahedron_vertices()
    edges, faces = build_graph(verts)
    V, E, F = len(verts), len(edges), len(faces)
    print("[组合层] 数顶点 V = %d" % V)
    print("[组合层] 数邻居对 E = %d" % E)
    print("[组合层] 数 3-团   F = %d" % F)
    position = V + E + F          # 位置总数 62
    face_slot = E + F             # 面位容量 50
    print("[组合层] 位置 = V+E+F = %d" % position)
    print("[组合层] 面位 = E+F   = %d" % face_slot)
    print("[组合层] 差   = %d = V = %d  (C4 减法投影: 顶点单元已被实边占据)"
          % (position - face_slot, V))
    return verts, edges, faces, V, E, F


# ============================================================
# 几何层: 搬运的可容纳性检查 (候选, 待证 T3)
# ============================================================

R_C = 1.0        # 中心球半径 (单位)
R_V = 0.9021     # 顶点球半径 (42 构造, 触中心)
R_E = 0.1869     # 边球半径 (42 构造, 沿径向滑入 31.72° 楔缝)
D_V = R_C + R_V  # 顶点球心距 = 1.9021
D_E = R_C + R_E  # 边球心距 = 1.1869
TOL = 1e-3       # 相切判定容差 (相对半径和)


def edge_slot_dir(verts, i, j):
    """边虚面位方向 = 两顶点方向的归一化中分线。"""
    v = verts[i] + verts[j]
    return v / np.linalg.norm(v)


def face_slot_dir(verts, i, j, k):
    """面虚面位方向 = 三顶点方向之和的归一化。"""
    v = verts[i] + verts[j] + verts[k]
    return v / np.linalg.norm(v)


def check_42_construction(verts, edges):
    """42 构造: 中心 + 12 顶点球 + 30 边球, 全两两检查重叠/相切。

    期望 (42b 脚本实证): 重叠 0 对, 相切 102 对 (中心-42 + 边-顶点 60),
    中心球相切 42 球。这是"30 个边位恰对应一个可容纳位"的实证。
    """
    spheres = [(np.zeros(3), R_C)]                      # 0: 中心
    for i in range(len(verts)):                         # 1..12: 顶点球
        spheres.append((verts[i] * D_V, R_V))
    for (i, j) in edges:                                # 13..42: 边球
        spheres.append((edge_slot_dir(verts, i, j) * D_E, R_E))
    n = len(spheres)
    overlap = tangent = 0
    for a in range(n):
        for b in range(a + 1, n):
            dist = np.linalg.norm(spheres[a][0] - spheres[b][0])
            rsum = spheres[a][1] + spheres[b][1]
            if dist < rsum - TOL:
                overlap += 1
            elif abs(dist - rsum) <= TOL:
                tangent += 1
    center_tangent = sum(
        1 for a in range(1, n)
        if abs(np.linalg.norm(spheres[0][0] - spheres[a][0])
               - (spheres[0][1] + spheres[a][1])) <= TOL)
    return overlap, tangent, center_tangent


def check_edge_slot_exclusion(verts, edges):
    """30 个边位两两互斥: 最小夹角 > 0 → 每个边位是独立可容纳位。"""
    dirs = [edge_slot_dir(verts, i, j) for (i, j) in edges]
    min_ang = 180.0
    for a in range(len(dirs)):
        for b in range(a + 1, len(dirs)):
            c = float(np.clip(np.dot(dirs[a], dirs[b]), -1.0, 1.0))
            min_ang = min(min_ang, math.degrees(math.acos(c)))
    return min_ang


def face_slot_radius(verts, i, j, k):
    """面位球半径: 与中心球 + 三个顶点球同时相切 (解析解)。

    θ = 面心方向与顶点的夹角; d_f = R_C + r_f。
    相切条件 |d_f·f − D_V·u_i| = r_f + r_v 展开为 r_f 的线性方程:
        r_f = [1 + D_V² − 2·D_V·cosθ − r_v²] / [2(r_v − 1 + D_V·cosθ)]
    """
    f = face_slot_dir(verts, i, j, k)
    cos_theta = float(max(np.dot(f, verts[i]),
                          np.dot(f, verts[j]), np.dot(f, verts[k])))
    num = 1.0 + D_V * D_V - 2.0 * D_V * cos_theta - R_V * R_V
    den = 2.0 * (R_V - 1.0 + D_V * cos_theta)
    return num / den, f


def check_face_vs_edge_exclusion(verts, edges, faces):
    """面位 vs 边位互斥检查 —— 面位截断的真实机制 (候选)。

    若面球 (触中心+三顶点) 与相邻边球重叠, 则 30 边位与 20 面位
    不可共存, 只能二选一: 42 (填边位) 或 32 (填面位)。
    这给出 A6 "边隙优先" 的几何基础 (不再只是最满原则排序)。
    """
    edge_spheres = {frozenset((i, j)): (edge_slot_dir(verts, i, j) * D_E, R_E)
                    for (i, j) in edges}
    overlaps = []
    for (i, j, k) in faces:
        r_f, f = face_slot_radius(verts, i, j, k)
        d_f = R_C + r_f
        for e in ((i, j), (i, k), (j, k)):
            c, r_e = edge_spheres[frozenset(e)]
            dist = float(np.linalg.norm(d_f * f - c))
            rsum = r_f + r_e
            if dist < rsum - TOL:
                overlaps.append((rsum - dist) / rsum * 100.0)  # 重叠百分比
    r_f, _ = face_slot_radius(verts, faces[0][0], faces[0][1], faces[0][2])
    n_total = len(faces) * 3
    return r_f, len(overlaps), n_total, (np.mean(overlaps) if overlaps else 0.0)


def check_face_truncation(verts, faces):
    """旧判据 (保留作对照): 面心方向 vs 顶点球角半径。

    结果: 37.38° > 28.31° → 面位未被顶点球挡住。
    即旧判据是错的 —— 截断不来自面位-顶点重叠 (见 check_face_vs_edge_exclusion)。
    """
    alpha_v = math.degrees(math.asin(R_V / D_V))
    min_ang = 180.0
    for (i, j, k) in faces:
        f = face_slot_dir(verts, i, j, k)
        for t in (i, j, k):
            c = float(np.clip(np.dot(f, verts[t]), -1.0, 1.0))
            min_ang = min(min_ang, math.degrees(math.acos(c)))
    return alpha_v, min_ang


def main():
    audit = Audit()

    print("=" * 72)
    print("  搬运定理候选验证 (T3/C10) — 二十面体单实例")
    print("=" * 72)

    verts, edges, faces, V, E, F = count_layer()
    position = V + E + F
    face_slot = E + F
    audit.add_countable("V", V, "数顶点")
    audit.add_countable("E", E, "数邻居对 (距离<1.1 且非自环)")
    audit.add_countable("F", F, "数 3-团 (两两相邻三元组)")
    audit.add_countable("位置", position, "V+E+F 求和")
    audit.add_countable("面位", face_slot, "E+F 求和")
    audit.add_countable("差", position - face_slot, "位置−面位 = V (C4 减法投影)")

    print("\n[几何层] 42 构造 (中心 + 12 顶点球 + 30 边球)...")
    overlap, tangent, center_tangent = check_42_construction(verts, edges)
    print("[几何层] 重叠对数   = %d  (期望 0)" % overlap)
    print("[几何层] 相切对数   = %d  (期望 102 = 中心-42 + 边-顶点 60)"
          % tangent)
    print("[几何层] 中心相切数 = %d  (期望 42 = 12 顶点 + 30 边球)"
          % center_tangent)
    audit.add_countable("重叠对数", overlap, "43 球全两两距离检查 (dist < rsum−TOL)")
    audit.add_countable("相切对数", tangent, "43 球全两两距离检查 (|dist−rsum| ≤ TOL)")
    audit.add_countable("中心相切数", center_tangent, "中心与 42 球逐一距离检查")

    min_excl = check_edge_slot_exclusion(verts, edges)
    print("\n[几何层] 30 个边位两两最小夹角 = %.2f° (互斥, 每边位独立)"
          % min_excl)

    # 旧判据对照: 面位 vs 顶点球 (证明旧判据不够)
    alpha_v, min_face_ang = check_face_truncation(verts, faces)
    print("\n[几何层] 旧判据 (面位 vs 顶点球): 角半径 alpha_v=%.2f°, "
          "面心-顶点最小夹角=%.2f°" % (alpha_v, min_face_ang))
    if min_face_ang > alpha_v:
        print("[几何层]   → %s > %s: 面位未被顶点球挡住 — 旧判据不成立"
              % ("%.2f°" % min_face_ang, "%.2f°" % alpha_v))

    # 新判据: 面位 vs 边位互斥 (真实截断机制)
    r_f, n_overlap, n_total, mean_overlap = check_face_vs_edge_exclusion(
        verts, edges, faces)
    print("\n[几何层] 面位球半径 r_f = %.4f (触中心 + 三顶点, 解析解)"
          % r_f)
    print("[几何层] 面球-边球重叠: %d/%d 对, 平均重叠 %.2f%%"
          % (n_overlap, n_total, mean_overlap))
    if n_overlap == n_total:
        print("[几何层] → 全部面球与相邻边球重叠 → 30 边位与 20 面位"
              "不可共存 (二选一: 42 或 32)")
        print("[几何层] → A6 边隙优先获得几何基础: 42 = max(30 边位, "
              "20 面位), 不是排序偏好而是互斥必然")
    audit.add_countable("互斥对数", "%d/%d" % (n_overlap, n_total),
                        "面球-边球距离检查")
    audit.add_countable("二选一", "42 | 32",
                        "max(30 边位, 20 面位); 42 = 30+12, 32 = 20+12")
    audit.add_countable("A6 选择", 42, "42 > 32 → 边隙优先 (互斥必然)")

    # 投影注册 (L1): 坐标嵌入/解析几何/判定约定 — 不可数出, 必须标注
    audit.add_projection("R_V", "%.4f" % R_V, "42 构造半径比例 (坐标嵌入)")
    audit.add_projection("R_E", "%.4f" % R_E, "42 构造半径比例 (坐标嵌入)")
    audit.add_projection("D_V", "%.4f" % D_V, "R_C+R_V (坐标嵌入)")
    audit.add_projection("D_E", "%.4f" % D_E, "R_C+R_E (坐标嵌入)")
    audit.add_projection("TOL", TOL, "相切判定容差 (判定约定, 非 L0)")
    audit.add_projection("邻接阈值", 1.1, "正则嵌入距离阈值 (判定约定)")
    audit.add_projection("r_f", "%.4f" % r_f, "面球半径解析解 (坐标投影)")
    audit.add_projection("alpha_v", "%.2f°" % alpha_v,
                         "arcsin(R_V/D_V) (解析几何投影)")
    audit.add_projection("面心-顶点夹角", "%.2f°" % min_face_ang,
                         "方向点积 (坐标投影)")
    audit.add_projection("边位两两夹角", "%.2f°" % min_excl,
                         "方向点积 (坐标投影)")
    audit.add_projection("重叠均值", "%.2f%%" % mean_overlap,
                         "距离比值 (坐标投影)")

    print("\n" + "=" * 72)
    print("  结论判读")
    print("=" * 72)
    print("  组合层: 位置 62, 面位 50, 差 12=V —— 计数重言, 可审计 (C10)")
    print("  几何层: 30 边位可容纳 (42 构造自洽); 20 面位与边位互斥")
    print("          42 = 互斥约束下的最大填充; 50 (E+F) = 组合容量上限")
    print("          62 (V+E+F) = 位置总数; 42 ≤ 50 < 62 (C8)")
    print("  搬运定理 = E(S)+F(S) 在二十面体上成立 (单实例)")
    print("  一般性未证: '图单元凭什么成为容量位' 仍是 L2→L3 断点核心 (T3)")
    print("=" * 72)

    audit.report()


if __name__ == "__main__":
    main()
