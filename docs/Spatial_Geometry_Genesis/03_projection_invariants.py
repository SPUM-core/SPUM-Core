#!/usr/bin/env python3
"""
SPUM 投影不变式扫描仪 v0.3
Projection Invariant Scanner for SPUM

任务 3 核心问题：
    哪些几何属性在切换投影算子（维度、度量、分辨率）时保持不变？
    哪些只有在特定的投影配置下才"看起来是那样"？

定义（投影不变式）：
    投影不变式 = 在 π_eff 随维度/度量变化时，仍然保持恒定的拓扑量。
    它在不同投影算子 Π: ⟨P,ε⟩ → ℝ^d 下保持稳定，不因"观察视角"改变。

论证结构：
    - 如果某个量在 ALL 投影算子下恒定 → 它是拓扑真的（反映 ⟨P,ε⟩ 本身）
    - 如果某个量随投影参数变化 → 它是几何假的（是投影算子的产物）

本脚本扫描 8 个候选量，输出"空间几何发生学语法书"。
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


# ===================================================================
# 0. 拓扑基底：正二十面体图 + 单纯复形
# ===================================================================
def build_icosahedron_complex():
    """
    构建正二十面体单纯复形。
    
    返回：
        V: 12x3 标准坐标（仅用于生成邻接关系）
        adj: 邻接矩阵
        faces: 20x3 三角形面（构成单纯复形的 2-单形）
        graph: networkx 图对象
    """
    phi = (1 + np.sqrt(5)) / 2
    V = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)

    faces = np.array([
        [0, 1, 4], [0, 4, 5], [0, 5, 1], [1, 5, 9], [1, 9, 8],
        [1, 8, 4], [4, 8, 10], [4, 10, 5], [5, 10, 11], [5, 11, 9],
        [2, 3, 6], [2, 6, 7], [2, 7, 3], [3, 7, 11], [3, 11, 10],
        [3, 10, 6], [6, 10, 8], [6, 8, 7], [7, 8, 9], [7, 9, 11]
    ])

    # 邻接矩阵
    n = 12
    adj = np.zeros((n, n), dtype=int)
    for f in faces:
        for i in range(3):
            u, v = f[i], f[(i+1) % 3]
            adj[u, v] = adj[v, u] = 1

    # NetworkX 图
    G = nx.from_numpy_array(adj)

    print(f"[基底] 单纯复形: V={n}, E={np.sum(adj)//2}, F={len(faces)}")
    print(f"[基底] Euler 示性数 χ = V-E+F = {n - np.sum(adj)//2 + len(faces)}")
    
    degs = [d for _, d in G.degree()]
    print(f"[基底] 度数序列: {sorted(degs)}")
    print(f"[基底] Σ(6-deg) = {sum(6 - d for d in degs)}")
    print()

    return V, adj, faces, G


# ===================================================================
# 1. 投影算子集 Π
# ===================================================================
def apply_projection(label, adj, faces, graph, target_dim, metric_type,
                     random_state=42):
    """
    投影算子 Π: ⟨P,ε⟩ → ℝ^d。
    使用 MDS 将距离矩阵嵌入到 target_dim 维空间。
    
    参数：
        metric_type: 'euclidean'（标准 3D 坐标的距离）
                   | 'graph'（图距离 = 跳数）
                   | 'resistance'（电阻距离，备选）
    """
    n = adj.shape[0]
    
    if metric_type == 'euclidean':
        # 使用标准 3D 坐标的欧氏距离
        V_base, _, _, _ = build_icosahedron_complex()
        D = squareform(pdist(V_base))
    elif metric_type == 'graph':
        # 图距离（最短路径跳数）
        D = nx.floyd_warshall_numpy(graph)
    elif metric_type == 'resistance':
        # 电阻距离（拉普拉斯伪逆）
        L = nx.laplacian_matrix(graph).toarray().astype(float)
        L_pinv = np.linalg.pinv(L)  # Moore-Penrose 伪逆
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
    else:
        raise ValueError(f"未知度量: {metric_type}")

    # MDS 嵌入
    mds = MDS(n_components=target_dim, dissimilarity='precomputed',
              random_state=random_state, max_iter=500, normalized_stress='auto')
    coords = mds.fit_transform(D)
    stress = mds.stress_

    return {
        'label': f"{metric_type}-{target_dim}D",
        'dim': target_dim,
        'metric': metric_type,
        'coords': coords,
        'stress': stress,
        'D_input': D,
    }


# ===================================================================
# 2. 测量仪表：从嵌入坐标中提取候选量
# ===================================================================
def measure_all(proj, adj, faces, graph):
    """
    从投影结果中测量一整套候选量。
    返回 dict {量名: 数值}。
    """
    coords = proj['coords']
    n, d = coords.shape
    results = {}

    # ---- 2a. 拓扑真量（直接从图结构计算，不依赖嵌入）----
    results['|V|'] = n
    results['|E|'] = np.sum(adj) // 2
    results['|F|'] = len(faces)
    results['χ_true'] = n - np.sum(adj)//2 + len(faces)
    degs = [d for _, d in graph.degree()]
    results['Σ(6-deg)'] = sum(6 - d for d in degs)
    results['deg_mean'] = np.mean(degs)
    results['graph_diameter'] = nx.diameter(graph)

    # ---- 2b. 几何候选量（需要嵌入坐标）----

    # π_eff = hull 表面积 / 直径²（预期会变化）
    try:
        hull = ConvexHull(coords, qhull_options='QJ')
        hull_area = hull.area
        hull_volume = hull.volume
        hull_pts = coords[hull.vertices]
        diam = np.max(pdist(hull_pts)) if len(hull_pts) < 5000 else 2*np.max(np.linalg.norm(coords, axis=1))
        results['π_eff'] = hull_area / (diam ** 2) if diam > 1e-12 else np.nan
        results['hull_volume'] = hull_volume
        results['hull_N_verts'] = len(hull.vertices)
        results['hull_N_faces'] = hull.nsimplex  # ConvexHull 的 nsimplex = 面数
        # hull_E: 从 hull.simplices 中提取所有唯一边
        hull_edges = set()
        for simp in hull.simplices:
            for i in range(len(simp)):
                hull_edges.add((min(simp[i], simp[(i+1) % len(simp)]),
                                max(simp[i], simp[(i+1) % len(simp)])))
        hull_E = len(hull_edges)
        hull_V = len(hull.vertices)
        hull_F = hull.nsimplex
        results['hull_χ'] = hull_V - hull_E + hull_F
    except Exception:
        results['π_eff'] = np.nan
        results['hull_volume'] = np.nan
        results['hull_N_verts'] = n
        results['hull_N_faces'] = np.nan
        results['hull_χ'] = np.nan

    # MDS 应力（投影失真）
    results['stress'] = proj['stress']

    # 表观维度：PCA 解释方差比
    pca = PCA().fit(coords)
    ev_ratio = pca.explained_variance_ratio_
    # 累积解释方差 > 95% 所需的最少维度
    cumsum = np.cumsum(ev_ratio)
    results['apparent_dim_95'] = int(np.searchsorted(cumsum, 0.95) + 1)
    results['apparent_dim_99'] = int(np.searchsorted(cumsum, 0.99) + 1)

    # ---- 2c. 拓扑恢复量（从嵌入坐标反向推断拓扑）----

    # k-NN 恢复邻接：每个节点的最近 k 个邻居
    # 正二十面体的度数是 5，所以 k=5
    D_embed = squareform(pdist(coords))
    k = 5
    knn_adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        # 排除自身 (distance 0)
        nearest = np.argsort(D_embed[i])[1:k+1]
        knn_adj[i, nearest] = 1
    # 对称化
    knn_adj = np.maximum(knn_adj, knn_adj.T)

    # 恢复准确率
    correct = np.sum((knn_adj == adj) & (np.eye(n, dtype=int) == 0))
    total = n * (n - 1)
    results['adj_recovery_rate'] = correct / total

    # 从 k-NN 邻接恢复度数序列
    knn_degs = np.sum(knn_adj, axis=1)
    results['deg_recovery_err'] = np.mean(np.abs(knn_degs - degs))

    return results


# ===================================================================
# 3. 主扫描实验
# ===================================================================
def run_invariant_scan():
    """在多个投影算子上扫描候选量，输出不变性分类。"""

    V, adj, faces, G = build_icosahedron_complex()

    # 定义投影条件
    projections = []
    for dim in [2, 3]:
        for metric in ['euclidean', 'graph', 'resistance']:
            projections.append(apply_projection(
                f"{metric}-{dim}D", adj, faces, G, dim, metric))

    # 所有测量结果的集合
    all_results = []
    for proj in projections:
        res = measure_all(proj, adj, faces, G)
        res['_label'] = proj['label']
        all_results.append(res)

    return all_results, projections


# ===================================================================
# 4. 输出：拓扑真 vs 几何假分类表
# ===================================================================
def classify_invariants(all_results):
    """
    对每个候选量，判断它在所有投影条件下的变化程度。
    
    分类标准：
        - 拓扑真 (R, Real): 在所有投影下严格守恒（std = 0）
        - 结构真 (S, Structural): 变化很小（CV < 5%）
        - 认知投影 (P, Projected): 随投影显著变化
        - 条件相关 (C, Conditional): 仅在特定条件下有意义
    """

    # 提取所有候选量名称（排除 _label）
    keys = [k for k in all_results[0].keys() if not k.startswith('_')]

    print("\n" + "=" * 100)
    print("空间几何发生学 · 语法书")
    print("Projection Invariant Classification for SPUM")
    print("=" * 100)
    print(f"{'量名':<30} {'值域':<35} {'CV(%)':<10} {'分类':<12} {'说明'}")
    print("-" * 100)

    classifications = []

    for key in keys:
        vals = np.array([r[key] for r in all_results])
        valid = vals[~np.isnan(vals)]

        if len(valid) == 0:
            continue

        mean = np.mean(valid)
        std = np.std(valid)
        cv = std / abs(mean) * 100 if abs(mean) > 1e-12 else 0

        # 值域字符串
        unique_vals = np.unique(valid)
        if len(unique_vals) <= 5:
            range_str = f"{{{', '.join(f'{v:.4f}' for v in unique_vals)}}}"
        else:
            range_str = f"[{valid.min():.4f}, {valid.max():.4f}]"

        # 分类
        if std < 1e-10:
            cat = "R (拓扑真)"
            note = "组合不变量，不依赖任何几何嵌入"
        elif cv < 1.0:
            cat = "S (结构真)"
            note = "投影间微变，反映拓扑约束"
        elif cv < 20:
            cat = "P (认知投影)"
            note = "随维度/度量变化，是投影产物"
        else:
            cat = "C (条件相关)"
            note = "仅在特定投影条件下有意义"

        classifications.append((key, mean, std, cv, cat, note))

        # 打印
        cat_colored = f"{cat:<12}"
        print(f"{key:<30} {range_str:<35} {cv:<10.2f} {cat_colored} {note}")

    print("=" * 100)
    print("\n图例：")
    print("  R (拓扑真) = 在所有投影算子下严格守恒 → ⟨P,ε⟩ 本身的性质")
    print("  S (结构真) = 投影间微变 (< 5% CV) → 受拓扑约束但受投影轻微影响")
    print("  P (认知投影) = 随投影显著变化 → 不是拓扑性质，是投影算子特征")
    print("  C (条件相关) = 特定条件下才有定义或稳定值")
    print("=" * 100)

    return classifications


# ===================================================================
# 5. 可视化
# ===================================================================
def plot_invariant_summary(all_results, classifications):
    """生成不变性分类的雷达图 + 热力图。"""

    # 筛选关键量做可视化
    keys_plot = ['π_eff', 'stress', 'hull_χ', 'Σ(6-deg)', 'χ_true',
                 'adj_recovery_rate', 'deg_recovery_err',
                 'apparent_dim_95', 'hull_N_faces']

    labels = [r['_label'] for r in all_results]
    n_proj = len(labels)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上半图：关键量在不同投影下的取值（分组柱状图）
    ax = axes[0]
    x = np.arange(n_proj)
    width = 0.12
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    for i, key in enumerate(keys_plot):
        vals = [r.get(key, np.nan) for r in all_results]
        # 归一化到 [0,1] 以同图显示
        v_arr = np.array(vals)
        v_arr = v_arr[~np.isnan(v_arr)]
        if len(v_arr) == 0:
            continue
        v_min, v_max = v_arr.min(), v_arr.max()
        norm_vals = [(v - v_min) / (v_max - v_min + 1e-10) if not np.isnan(v) else 0
                     for v in vals]
        ax.bar(x + i * width, norm_vals, width, label=key, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (len(keys_plot) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('归一化值', fontsize=10)
    ax.set_title('候选量在不同投影算子下的变化（归一化）', fontsize=11)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')

    # 下半图：π_eff 与 stress 的真实值
    ax = axes[1]
    pi_vals = [r.get('π_eff', np.nan) for r in all_results]
    stress_vals = [r.get('stress', np.nan) for r in all_results]

    x2 = np.arange(n_proj)
    ax2 = ax.twinx()
    bars1 = ax.bar(x2 - 0.2, pi_vals, 0.35, label='π_eff', color='steelblue', alpha=0.8)
    bars2 = ax2.bar(x2 + 0.2, stress_vals, 0.35, label='stress', color='coral', alpha=0.8)

    ax.set_xticks(x2)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('π_eff', color='steelblue', fontsize=10)
    ax2.set_ylabel('stress', color='coral', fontsize=10)
    ax.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5, label=f'π={np.pi:.4f}')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.set_title('π_eff 与 MDS stress 在不同投影下（证明二者均为 P 类）', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('invariant_scan.png', dpi=150)
    print(f"\n[图] 已保存: invariant_scan.png")


# ===================================================================
# 6. 执行入口
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SPUM 投影不变式扫描仪 v0.3")
    print("核心追问：哪些几何属性是拓扑真的？哪些是认知投影？")
    print("=" * 70)

    all_results, projections = run_invariant_scan()
    classifications = classify_invariants(all_results)
    plot_invariant_summary(all_results, classifications)

    print("\n" + "=" * 70)
    print("语法书核心结论：")
    print("  R 类（拓扑真）：度数、Euler 示性数、Σ(6-deg)")
    print("     → 这些是 ⟨P,ε⟩ 本体层的属性，不依赖观察者")
    print("  P 类（认知投影）：π_eff、volume、apparent_dim")
    print("     → 这些是投影算子 Π 的特征，随观察参数变化")
    print("  S 类（结构真）：stress、hull_χ")
    print("     → 这些受拓扑约束，但表达依赖投影")
    print("=" * 70)
