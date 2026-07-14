#!/usr/bin/env python3
"""
SPUM 认知投影渲染器 v0.2 — 分辨率扫描
从 12 节点细分到 10^6 节点，测量 π_eff = 表面积/直径² 向 π = 3.14159... 的收敛

核心追问：
    当粗粒化尺度增加（节点数 N → ∞），π_eff 是否收敛到 π？
    如果是，收敛速率是多少？
    它与"空间几何发生学"中的"粗粒化定理"是否一致？
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import time


# ===================================================================
# 1. 正二十面体基底（单位球面）
# ===================================================================
def icosahedron_base():
    """返回单位球面上的正二十面体顶点 + 20 个三角形面。"""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # 归一化到单位球面
    norm = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norm

    # 20 个三角形面（顶点索引），定义正二十面体的拓扑
    faces = np.array([
        [0, 1, 4], [0, 4, 5], [0, 5, 1], [1, 5, 9], [1, 9, 8],
        [1, 8, 4], [4, 8, 10], [4, 10, 5], [5, 10, 11], [5, 11, 9],
        [2, 3, 6], [2, 6, 7], [2, 7, 3], [3, 7, 11], [3, 11, 10],
        [3, 10, 6], [6, 10, 8], [6, 8, 7], [7, 8, 9], [7, 9, 11]
    ])
    return verts, faces


# ===================================================================
# 2. 球面细分（Loop 细分 → 投影到球面）
# ===================================================================
def subdivide(vertices, faces):
    """
    一次球面细分：
    每个三角面 → 4 个小三角面（插入边中点 → 投影到单位球面）
    
    返回新的顶点列表和面列表。
    """
    new_verts = list(vertices)
    edge_map = {}

    def get_edge_midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key not in edge_map:
            mid = (new_verts[i] + new_verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid)  # 投影到单位球面
            edge_map[key] = len(new_verts)
            new_verts.append(mid)
        return edge_map[key]

    new_faces = []
    for f in faces:
        i, j, k = f
        a = get_edge_midpoint(i, j)
        b = get_edge_midpoint(j, k)
        c = get_edge_midpoint(k, i)
        new_faces.append([i, a, c])
        new_faces.append([j, b, a])
        new_faces.append([k, c, b])
        new_faces.append([a, b, c])

    return np.array(new_verts), np.array(new_faces)


# ===================================================================
# 3. 测量 π_eff = 表面积 / 直径²
# ===================================================================
def measure_pi_eff(verts):
    """
    从 3D 点集计算 π_eff。
    
    对于单位球面上的点：
        - 凸包表面积 → 4πR² = 4π（当 N→∞）
        - 直径 = 2R = 2
        - π_eff = 表面积 / 4 → π（当 N→∞）
    
    直径计算优化：
        - N < 5000: 精确计算 pdist(凸包顶点)
        - N ≥ 5000: 直径 = 2 × max(||v||)，对球面分布精确成立
    """
    hull = ConvexHull(verts, qhull_options='QJ')
    surface_area = hull.area
    
    # 直径计算
    n_hull = len(hull.vertices)
    if n_hull < 5000:
        # 小样本：精确计算
        hull_pts = verts[hull.vertices]
        diam = np.max(pdist(hull_pts))
    else:
        # 大样本：对球面分布，直径 = 2 × 最大半径
        # 这是精确的，因为球面上最远两点是通过球心的对径点
        max_r = np.max(np.linalg.norm(verts, axis=1))
        diam = 2.0 * max_r
    
    if diam < 1e-12:
        return np.nan
    return surface_area / (diam ** 2)


# ===================================================================
# 4. 分辨率扫描主循环
# ===================================================================
def run_resolution_scan(max_levels=8):
    """从 Level 0 迭代细分到 max_levels，每层测量 π_eff。"""
    verts, faces = icosahedron_base()

    results = []
    for level in range(max_levels + 1):
        n = len(verts)
        t0 = time.time()
        pi_eff = measure_pi_eff(verts)
        dt = time.time() - t0
        err = abs(pi_eff - np.pi)
        err_pct = (pi_eff - np.pi) / np.pi * 100

        results.append({
            'level': level,
            'N': n,
            'faces': len(faces),
            'pi_eff': pi_eff,
            'error': err,
            'error_pct': err_pct,
            'time': dt,
            'verts': verts if level <= 5 else None,  # 只保留小 N 的坐标用于绘图
        })

        print(f"Level {level:2d} | N={n:7d} | F={len(faces):8d} | "
              f"π_eff={pi_eff:.8f} | error={err:.8f} ({err_pct:+.4f}%) | "
              f"time={dt:.2f}s")

        # 继续细分
        if level < max_levels:
            verts, faces = subdivide(verts, faces)

    return results


# ===================================================================
# 5. 绘图
# ===================================================================
def plot_results(results):
    """生成收敛曲线图。"""
    levels = np.array([r['level'] for r in results])
    nodes = np.array([r['N'] for r in results])
    pi_vals = np.array([r['pi_eff'] for r in results])
    errors = np.array([r['error'] for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) π_eff vs 细分层级
    ax = axes[0, 0]
    ax.plot(levels, pi_vals, 'bo-', markersize=6, label='π_eff')
    ax.axhline(y=np.pi, color='r', linestyle='--', linewidth=1.5,
               label=f'π = {np.pi:.8f}')
    ax.set_xlabel('细分层级', fontsize=11)
    ax.set_ylabel('π_eff = 表面积 / 直径²', fontsize=11)
    ax.set_title('(a) π_eff 随细分层级的变化', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    # 标注每个点的值
    for lv, pv in zip(levels, pi_vals):
        ax.annotate(f'{pv:.6f}', (lv, pv),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=7)

    # (b) π_eff vs 节点数（半对数）
    ax = axes[0, 1]
    ax.semilogx(nodes, pi_vals, 'bo-', markersize=6, label='π_eff')
    ax.axhline(y=np.pi, color='r', linestyle='--', linewidth=1.5,
               label=f'π = {np.pi:.8f}')
    ax.set_xlabel('节点数 N (对数)', fontsize=11)
    ax.set_ylabel('π_eff', fontsize=11)
    ax.set_title('(b) π_eff 随节点数的收敛（半对数）', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for n, pv in zip(nodes, pi_vals):
        ax.annotate(f'{pv:.6f}', (n, pv),
                    textcoords="offset points", xytext=(5, 10),
                    ha='left', fontsize=7)

    # (c) 绝对误差 |π_eff - π| vs 节点数（双对数）
    ax = axes[1, 0]
    ax.loglog(nodes, errors, 'ro-', markersize=6, label='|π_eff - π|')
    # 拟合一条 N^{-0.5} 参考线
    ref_n = np.array([nodes[1], nodes[-1]])
    ref_err = errors[1] * (ref_n / nodes[1]) ** (-0.5)
    ax.loglog(ref_n, ref_err, 'k--', alpha=0.5, label='N^{-0.5} (参考)')
    ax.set_xlabel('节点数 N (对数)', fontsize=11)
    ax.set_ylabel('|π_eff - π|', fontsize=11)
    ax.set_title('(c) 收敛速率（双对数）', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for n, e in zip(nodes, errors):
        ax.annotate(f'{e:.2e}', (n, e),
                    textcoords="offset points", xytext=(5, 10),
                    ha='left', fontsize=7)

    # (d) 误差百分比
    err_pcts = np.array([r['error_pct'] for r in results])
    ax = axes[1, 1]
    ax.semilogx(nodes, err_pcts, 'go-', markersize=6, label='相对误差 %')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('节点数 N (对数)', fontsize=11)
    ax.set_ylabel('相对误差 (%)', fontsize=11)
    ax.set_title('(d) 相对误差', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for n, ep in zip(nodes, err_pcts):
        ax.annotate(f'{ep:+.4f}%', (n, ep),
                    textcoords="offset points", xytext=(5, 10),
                    ha='left', fontsize=7)

    plt.tight_layout()
    plt.savefig('resolution_scan.png', dpi=150)
    print(f"\n[图] 已保存: resolution_scan.png")


# ===================================================================
# 6. 执行
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SPUM 认知投影渲染器 v0.2 — 分辨率扫描")
    print("核心追问: 当粗粒化尺度增加, π_eff 如何收敛到 π = 3.14159...?")
    print("=" * 70)
    print(f"{'Level':<6} {'N':<8} {'F':<10} {'π_eff':<14} {'error':<14} {'时间':<8}")
    print("-" * 70)

    results = run_resolution_scan(max_levels=7)

    print("=" * 70)
    print(f"\n初始 π_eff = {results[0]['pi_eff']:.8f} (12节点)")
    print(f"最终 π_eff = {results[-1]['pi_eff']:.8f} ({results[-1]['N']}节点)")
    print(f"π 真实值   = {np.pi:.8f}")
    print(f"\n收敛比率: "
          f"{abs(results[-1]['pi_eff'] - np.pi) / abs(results[0]['pi_eff'] - np.pi):.2e}")
    print("=" * 70)

    plot_results(results)
