#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPUM 认知投影渲染器 v0.1
Cognitive Projection Renderer for SPUM

方法论定位：
  底层输入：纯拓扑关系网络 ⟨P, ε⟩（正二十面体邻接矩阵）
  认知投影算子：MDS 嵌入 + 不同的距离度量与目标维度
  输出测量：表观几何常数 π_eff = 周长/直径 (2D) 或 表面积/直径² (3D)

核心追问：
  当同一个拓扑网络被不同"摄像机"渲染时，π 是恒定的，还是随投影参数变化的？
  如果是后者，则 π 不是拓扑不变量，而是认知投影算子的特征值。

依赖：
  numpy, scipy, scikit-learn, matplotlib, networkx
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import networkx as nx
import matplotlib.pyplot as plt

# ---------- 1. 基底：正二十面体的纯拓扑关系 ----------
def build_icosahedron_graph():
    """
    构建正二十面体的纯拓扑图（12 节点，30 边，每个节点度数 5）。
    返回：邻接矩阵 A（12x12，0/1），以及用于参考的经典 3D 坐标（仅用于生成欧氏距离基准）。
    """
    # 经典黄金比例坐标（边长 = 2）
    phi = (1 + np.sqrt(5)) / 2
    coords_3d = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # 纯拓扑邻接矩阵（基于距离阈值，边长严格为 2）
    eucl_dist = squareform(pdist(coords_3d))
    A = (eucl_dist < 2.1).astype(int)  # 阈值 2.1 能完美识别边
    np.fill_diagonal(A, 0)  # 无自环
    
    # 验证度数（应该全是 5）
    degrees = np.sum(A, axis=1)
    assert np.all(degrees == 5), f"度数不是 5，实际为 {degrees}"
    print(f"[基底] 正二十面体加载完成。节点数=12，边数={np.sum(A)//2}，度数={degrees[0]}")
    
    return A, coords_3d

# ---------- 2. 距离矩阵生成器（不同“认知度量”） ----------
def compute_distance_matrix(A, metric_type='euclidean', coords_3d=None):
    """
    根据不同的度量类型，生成节点间的距离矩阵 D（12x12）。
    
    metric_type:
      - 'euclidean' : 经典欧氏距离（需要 coords_3d）
      - 'graph'      : 纯拓扑最短路径跳数（图距离）
      - 'resistance' : 电阻距离（图拉普拉斯伪逆，可选项，暂不实现）
    """
    n = A.shape[0]
    if metric_type == 'euclidean':
        if coords_3d is None:
            raise ValueError("欧氏距离需要提供 coords_3d")
        return squareform(pdist(coords_3d))
    
    elif metric_type == 'graph':
        G = nx.from_numpy_array(A)
        # 全对最短路径（跳数）
        dist_matrix = nx.floyd_warshall_numpy(G)
        return dist_matrix
    
    else:
        raise ValueError(f"未知度量类型: {metric_type}")

# ---------- 3. 认知投影渲染器 ----------
def render_projection(dist_matrix, target_dim, random_state=42):
    """
    将距离矩阵通过度量 MDS 嵌入到 target_dim 维欧氏空间。
    返回：
      - coords: 嵌入后的坐标 (N x target_dim)
      - stress: MDS 应力值（衡量投影失真程度）
    """
    mds = MDS(
        n_components=target_dim,
        dissimilarity='precomputed',
        random_state=random_state,
        normalized_stress='auto',  # 自动归一化应力
        max_iter=300
    )
    coords = mds.fit_transform(dist_matrix)
    stress = mds.stress_  # 未归一化应力，但可用于相对比较
    return coords, stress

# ---------- 4. 几何常数测量仪（输出端的“表观 π”） ----------
def measure_pi_eff(coords, mds_stress=0.0):
    """
    从嵌入坐标中测量"表观 π"。
    
    2D:  π_eff = 凸包周长 / 直径
    3D:  π_eff = 凸包表面积 / 直径²
    理由：对于理想球体/圆，这两个比值分别等于 π。
    """
    n, d = coords.shape
    if n < d + 1:
        return np.nan, np.nan, mds_stress
    
    try:
        hull = ConvexHull(coords)
    except Exception as e:
        # 退化情况（所有点共面或共线）
        return np.nan, np.nan, mds_stress
    
    # 直径：凸包点之间的最大成对距离
    hull_points = coords[hull.vertices]
    diam = np.max(pdist(hull_points))
    if diam < 1e-12:
        return np.nan, np.nan, mds_stress
    
    if d == 2:
        # hull.area 对于 2D 返回周长，hull.volume 返回面积
        perimeter = hull.area
        pi_eff = perimeter / diam
        return pi_eff, 0.0, mds_stress  # 体积暂不输出
    
    elif d == 3:
        # hull.area 对于 3D 返回表面积
        surface_area = hull.area
        pi_eff = surface_area / (diam ** 2)
        volume = hull.volume
        return pi_eff, volume, mds_stress
    
    else:
        return np.nan, np.nan, mds_stress

# ---------- 5. 主实验：扫描认知投影参数 ----------
def run_cognitive_scan():
    """运行完整的认知投影扫描实验。"""
    # 加载基底
    A, coords_3d = build_icosahedron_graph()
    
    # 定义实验条件
    metrics = [
        ('欧氏距离 (Euclidean)', 'euclidean'),
        ('图距离 (Graph hops)', 'graph')
    ]
    target_dims = [2, 3]
    
    results = []
    fig, axes = plt.subplots(len(metrics), len(target_dims), figsize=(10, 8))
    if len(metrics) == 1:
        axes = axes.reshape(1, -1)
    if len(target_dims) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (metric_label, metric_key) in enumerate(metrics):
        # 生成距离矩阵
        D = compute_distance_matrix(A, metric_key, coords_3d)
        
        for j, dim in enumerate(target_dims):
            # 投影
            coords, stress = render_projection(D, dim)
            # 测量
            pi_eff, volume, stress_val = measure_pi_eff(coords, mds_stress=stress)
            
            # 记录
            results.append({
                '度量': metric_label,
                '目标维度': dim,
                'π_eff': pi_eff,
                '应力': stress_val,
                '坐标': coords
            })
            
            # 绘制嵌入结果（2D 投影或 3D 投影）
            ax = axes[i, j]
            if dim == 2:
                ax.scatter(coords[:, 0], coords[:, 1], c='steelblue', s=60, edgecolors='k')
                # 绘制凸包
                try:
                    hull = ConvexHull(coords)
                    for simplex in hull.simplices:
                        ax.plot(coords[simplex, 0], coords[simplex, 1], 'r-', lw=1.5)
                except:
                    pass
                ax.set_aspect('equal', adjustable='box')
                ax.set_title(f"{metric_label}\n投影到 2D | π_eff = {pi_eff:.4f}")
            else:
                # 3D 投影：用散点图 + 颜色深度表示
                ax = fig.add_subplot(len(metrics), len(target_dims), i*len(target_dims)+j+1, projection='3d')
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                          c='steelblue', s=40, edgecolors='k', alpha=0.8)
                # 绘制凸包的边（只画部分，避免太乱）
                try:
                    hull = ConvexHull(coords)
                    for simplex in hull.simplices:
                        # 对于 3D 三角形面，画边
                        for k in range(3):
                            start = simplex[k]
                            end = simplex[(k+1)%3]
                            ax.plot3D(*zip(coords[start], coords[end]), color='r', lw=1.0, alpha=0.4)
                except:
                    pass
                ax.set_title(f"{metric_label}\n投影到 3D | π_eff = {pi_eff:.4f}")
                # 设置等比例
                ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig('cognitive_projection_pi_scan.png', dpi=150)
    print("\n[渲染] 图片已保存为 cognitive_projection_pi_scan.png")
    
    # 打印汇总表格
    print("\n" + "="*60)
    print("认知投影扫描结果汇总")
    print("="*60)
    print(f"{'度量':<20} | {'目标维度':<10} | {'π_eff':<12} | {'应力':<12}")
    print("-"*60)
    for r in results:
        pi_str = f"{r['π_eff']:.4f}" if not np.isnan(r['π_eff']) else "N/A"
        print(f"{r['度量']:<20} | {r['目标维度']}D{' '*7} | {pi_str:<12} | {r['应力']:<12.2f}")
    
    return results

# ---------- 6. 执行 ----------
if __name__ == "__main__":
    print("="*60)
    print("SPUM 认知投影渲染器 v0.1")
    print("核心追问：同一个拓扑网络，切换认知投影参数后，π 还恒定吗？")
    print("="*60)
    
    results = run_cognitive_scan()
    
    print("\n" + "="*60)
    print("初步结论（方法论层面）：")
    print("如果 π_eff 随度量或维度变化，则 π 不是拓扑不变量。")
    print("它只是特定认知投影算子（欧氏-三维-粗粒化）的输出特征值。")
    print("这正是 SPUM 空间几何发生学第十二章要验证的硬实验。")
    print("="*60)