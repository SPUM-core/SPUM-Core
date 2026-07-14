#!/usr/bin/env python3
"""
SPUM 锐边界挑战 v0.3b
Sharp Boundary Challenge for SPUM

核心追问：
    是否存在一个量 Q，它部分依赖于嵌入坐标，
    但在所有投影算子（维度 × 度量）下保持不变？
    
    如果找到 → 锐边界不成立，语法书需要引入 S 类
    如果找不到 → 锐边界定理成立，语法书的 R/P/C 分类是完备的

新增候选不变量（超越 03 的测量集）：
    1. 持久同调 Betti 曲线（β₀·β₁ 的尺度演化）
    2. 距离秩相关（Kendall τ）
    3. 最小生成树 (MST) 结构
    4. k-NN 图谱间隔（λ₁ - λ₂）
    5. 图直径的嵌入恢复比
    6. 中心性序数相关
    7. 距离矩阵的固有秩
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components, laplacian
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau, spearmanr
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


# ===================================================================
# 0. 基底（复用 03 的构建函数）
# ===================================================================
def build_icosahedron_complex():
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
    n = 12
    adj = np.zeros((n, n), dtype=int)
    for f in faces:
        for i in range(3):
            u, v = f[i], f[(i+1) % 3]
            adj[u, v] = adj[v, u] = 1
    G = nx.from_numpy_array(adj)
    return V, adj, faces, G


# ===================================================================
# 1. 投影算子
# ===================================================================
def apply_projection(adj, graph, target_dim, metric_type, random_state=42):
    n = adj.shape[0]
    V_base, _, _, _ = build_icosahedron_complex()
    
    if metric_type == 'euclidean':
        D = squareform(pdist(V_base))
    elif metric_type == 'graph':
        D = nx.floyd_warshall_numpy(graph)
    elif metric_type == 'resistance':
        L = nx.laplacian_matrix(graph).toarray().astype(float)
        L_pinv = np.linalg.pinv(L)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                D[i, j] = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
    else:
        raise ValueError(f"未知度量: {metric_type}")

    mds = MDS(n_components=target_dim, dissimilarity='precomputed',
              random_state=random_state, max_iter=500, normalized_stress='auto')
    coords = mds.fit_transform(D)
    
    return {
        'label': f"{metric_type}-{target_dim}D",
        'dim': target_dim,
        'metric': metric_type,
        'coords': coords,
        'stress': mds.stress_,
        'D_input': D,
    }


# ===================================================================
# 2. 扩展测量集
# ===================================================================

def measure_persistent_homology(coords):
    """
    简易 Vietoris-Rips 持久同调扫描。
    
    对距离阈值 ε ∈ [0, max_dist]，逐步构建 ε-邻接图，
    追踪 β₀ (连通分量数) 和 β₁ (独立环数) 的演化。
    
    返回 persistence 曲线（β₀ 和 β₁ 作为 ε 的函数）。
    """
    n = len(coords)
    D = squareform(pdist(coords))
    max_dist = np.max(D)
    
    # 所有唯一边界距离（横轴采样点）
    thresholds = np.unique(D.ravel())
    thresholds = thresholds[thresholds > 0]  # 排除对角
    thresholds = np.sort(thresholds)
    # 均匀采样 50 个点
    if len(thresholds) > 50:
        thresholds = np.linspace(thresholds[0], thresholds[-1], 50)
    
    betas = []
    
    # 构建 Diag 矩阵表示
    for eps in thresholds:
        # ε-邻接图
        eps_adj = (D < eps).astype(int)
        np.fill_diagonal(eps_adj, 0)
        
        # β₀ = 连通分量数
        n_components, labels = connected_components(csr_matrix(eps_adj), directed=False)
        
        # β₁ = E - V + C (对图而言，独立环数)
        n_edges = np.sum(eps_adj) // 2
        beta_1 = n_edges - n + n_components
        
        betas.append((eps, n_components, max(0, beta_1)))
    
    return np.array(thresholds), np.array(betas)


def measure_mst_structure(coords):
    """
    最小生成树结构。
    
    MST 的性质（边数、权重分布、度分布）在不同投影下可能守恒。
    """
    D = squareform(pdist(coords))
    # 转换为 scipy sparse 格式（需要上三角）
    mst = minimum_spanning_tree(csr_matrix(D))
    mst_dense = mst.toarray()
    
    # MST 边权重列表
    edge_weights = mst_dense[mst_dense > 0]
    
    # MST 节点度数
    mst_degs = np.sum(mst_dense > 0, axis=1) + np.sum(mst_dense.T > 0, axis=1)
    
    return {
        'mst_n_edges': len(edge_weights),
        'mst_total_weight': np.sum(edge_weights),
        'mst_weight_std': np.std(edge_weights) if len(edge_weights) > 0 else 0,
        'mst_max_deg': np.max(mst_degs),
        'mst_deg_std': np.std(mst_degs),
    }


def measure_spectral_properties(coords, true_adj, k=5):
    """
    从嵌入坐标重构 k-NN 图，测量其谱性质。
    
    谱间隔（spectral gap）λ₁ - λ₂ 反映图的连通性强度。
    如果谱间隔在不同投影下稳定，就是候选不变量。
    """
    n = len(coords)
    D = squareform(pdist(coords))
    
    # k-NN 邻接
    knn_adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        nearest = np.argsort(D[i])[1:k+1]
        knn_adj[i, nearest] = 1.0
    knn_adj = np.maximum(knn_adj, knn_adj.T)
    
    # 拉普拉斯矩阵
    L = laplacian(csr_matrix(knn_adj), normed=True).toarray()
    eigenvalues = np.sort(np.linalg.eigvalsh(L))
    
    # 谱间隔（Fiedler 间隔）
    spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
    # 代数连通性（Fiedler 值）
    algebraic_connectivity = eigenvalues[1] if len(eigenvalues) > 1 else 0
    
    # 同样对真实邻接计算
    L_true = laplacian(csr_matrix(true_adj), normed=True).toarray()
    eig_true = np.sort(np.linalg.eigvalsh(L_true))
    gap_true = eig_true[1] - eig_true[0] if len(eig_true) > 1 else 0
    
    return {
        'spectral_gap_knn': spectral_gap,
        'algebraic_connectivity': algebraic_connectivity,
        'spectral_gap_true': gap_true,
        'spectral_gap_error': abs(spectral_gap - gap_true),
    }


def measure_distance_rank_correlation(proj):
    """
    距离秩相关：输入距离矩阵与输出欧氏距离矩阵之间的 Kendall τ。
    
    如果不同投影的 τ 都高且稳定，说明"距离序结构"是准不变量。
    """
    D_in = np.array(proj['D_input'])  # 已是方阵
    D_out = squareform(pdist(proj['coords']))
    
    # 向量化上三角（排除对角线）
    triu_idx = np.triu_indices_from(D_in, k=1)
    vec_in = D_in[triu_idx]
    vec_out = D_out[triu_idx]
    
    tau, _ = kendalltau(vec_in, vec_out)
    rho, _ = spearmanr(vec_in, vec_out)
    
    return {
        'kendall_tau': tau,
        'spearman_rho': rho,
    }


def measure_centrality_stability(coords, true_G):
    """
    中心性序数稳定性：从嵌入重构图中计算的中心性排序与真实图的相关性。
    
    如果排序几乎不变，说明中心性结构是准不变量。
    """
    n = len(coords)
    D = squareform(pdist(coords))
    k = 5
    
    # k-NN 图
    knn_adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        nearest = np.argsort(D[i])[1:k+1]
        knn_adj[i, nearest] = 1
    knn_adj = np.maximum(knn_adj, knn_adj.T)
    
    # 重构图的 NetworkX
    G_knn = nx.from_numpy_array(knn_adj)
    
    try:
        # 度中心性排序比较
        deg_true = sorted([d for _, d in true_G.degree()])
        deg_knn = sorted([d for _, d in G_knn.degree()])
        deg_rho, _ = spearmanr(deg_true, deg_knn)
        
        # 介数中心性
        bc_true = nx.betweenness_centrality(true_G)
        bc_knn = nx.betweenness_centrality(G_knn)
        bc_rho, _ = spearmanr(list(bc_true.values()), list(bc_knn.values()))
        
        # 特征向量中心性
        try:
            ec_true = nx.eigenvector_centrality(true_G, max_iter=1000, tol=1e-6)
            ec_knn = nx.eigenvector_centrality(G_knn, max_iter=1000, tol=1e-6)
            ec_rho, _ = spearmanr(list(ec_true.values()), list(ec_knn.values()))
        except Exception:
            ec_rho = np.nan
    except Exception:
        deg_rho, bc_rho, ec_rho = np.nan, np.nan, np.nan
    
    return {
        'centrality_deg_rho': deg_rho,
        'centrality_bc_rho': bc_rho,
        'centrality_ec_rho': ec_rho,
    }


def measure_geodesic_recovery(coords, true_graph):
    """
    测地恢复：从嵌入坐标重构图中恢复的图直径和最短路径。
    """
    n = len(coords)
    D = squareform(pdist(coords))
    k = 5
    
    knn_adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        nearest = np.argsort(D[i])[1:k+1]
        knn_adj[i, nearest] = 1
    knn_adj = np.maximum(knn_adj, knn_adj.T)
    
    G_knn = nx.from_numpy_array(knn_adj)
    
    # 图直径
    try:
        true_diam = nx.diameter(true_graph)
        knn_diam = nx.diameter(G_knn)
        diam_error = abs(knn_diam - true_diam) / true_diam
    except Exception:
        knn_diam = np.nan
        diam_error = np.nan
    
    # 最短路径矩阵的 Spearman 相关
    try:
        true_spl = dict(nx.all_pairs_shortest_path_length(true_graph))
        knn_spl = dict(nx.all_pairs_shortest_path_length(G_knn))
        
        true_vec = []
        knn_vec = []
        for i in range(n):
            for j in range(i+1, n):
                true_vec.append(true_spl[i][j])
                knn_vec.append(knn_spl[i][j])
        spl_rho, _ = spearmanr(true_vec, knn_vec)
    except Exception:
        spl_rho = np.nan
    
    return {
        'diam_recovered': knn_diam,
        'diam_error': diam_error,
        'spl_rank_rho': spl_rho,
    }


# ===================================================================
# 3. 综合测量
# ===================================================================
def measure_all_extended(proj, adj, faces, G):
    """完整测量，包含持久同调等新候选量。"""
    coords = proj['coords']
    n, d = coords.shape
    results = {}
    
    # --- 已有的参考量 ---
    results['|V|'] = n
    results['|E|'] = np.sum(adj) // 2
    results['|F|'] = len(faces)
    results['χ_true'] = n - np.sum(adj)//2 + len(faces)
    degs = [d for _, d in G.degree()]
    results['Σ(6-deg)'] = sum(6 - d for d in degs)
    
    # π_eff
    try:
        hull = ConvexHull(coords, qhull_options='QJ')
        hull_area = hull.area
        hull_pts = coords[hull.vertices]
        diam_hull = np.max(pdist(hull_pts)) if len(hull_pts) < 5000 else 2*np.max(np.linalg.norm(coords, axis=1))
        results['π_eff'] = hull_area / (diam_hull ** 2) if diam_hull > 1e-12 else np.nan
    except Exception:
        results['π_eff'] = np.nan
    
    # --- 新增候选量 ---
    
    # (1) 持久同调 Betti 曲线
    thresholds, betas = measure_persistent_homology(coords)
    # β₀: 从多分量到单分量的转变阈值
    # β₁: 最大环数
    beta0_at_max = betas[-1, 1]  # 最后应 = 1
    beta1_max = np.max(betas[:, 2]) if len(betas) > 0 else 0
    # β₀ → 1 的临界阈值（最后一个合并事件）
    beta0_crit = np.nan
    for t, b0, b1 in betas:
        if b0 == 1:
            beta0_crit = t
            break
    results['β₀_final'] = beta0_at_max
    results['β₁_max'] = beta1_max
    results['β₀_crit_threshold'] = beta0_crit
    
    # (2) 距离秩相关
    rank = measure_distance_rank_correlation(proj)
    results.update(rank)
    
    # (3) MST 结构
    mst = measure_mst_structure(coords)
    results.update(mst)
    
    # (4) k-NN 谱性质
    spec = measure_spectral_properties(coords, adj, k=5)
    results.update(spec)
    
    # (5) 中心性稳定性
    cent = measure_centrality_stability(coords, G)
    results.update(cent)
    
    # (6) 测地恢复
    geo = measure_geodesic_recovery(coords, G)
    results.update(geo)
    
    return results


# ===================================================================
# 4. 主扫描
# ===================================================================
def run_extended_scan():
    V, adj, faces, G = build_icosahedron_complex()
    
    # 投影条件（6 种）
    proj_configs = []
    for dim in [2, 3]:
        for metric in ['euclidean', 'graph', 'resistance']:
            proj_configs.append((dim, metric))
    
    all_results = []
    projs = []
    for dim, metric in proj_configs:
        proj = apply_projection(adj, G, dim, metric)
        projs.append(proj)
        res = measure_all_extended(proj, adj, faces, G)
        res['_label'] = proj['label']
        all_results.append(res)
    
    return all_results, projs


# ===================================================================
# 5. 分类与输出
# ===================================================================
def classify_extended(all_results):
    keys = [k for k in all_results[0].keys() if not k.startswith('_')]
    
    print("\n" + "=" * 110)
    print("锐边界挑战 · 扩展候选量扫描结果")
    print("Sharp Boundary Challenge — Extended Invariant Scan")
    print("=" * 110)
    print(f"{'候选量名':<32} {'值域':<30} {'CV(%)':<10} {'分类':<12} {'判定'}")
    print("-" * 110)
    
    s_candidates = []  # 可能的 S 类
    r_count = 0
    p_count = 0
    c_count = 0
    
    for key in sorted(keys):
        vals = np.array([r[key] for r in all_results])
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            continue
        
        mean = np.mean(valid)
        std = np.std(valid)
        cv = std / abs(mean) * 100 if abs(mean) > 1e-12 else 0
        
        unique_vals = np.unique(valid)
        if len(unique_vals) <= 5:
            range_str = f"{{{', '.join(f'{v:.4f}' for v in unique_vals)}}}"
        else:
            range_str = f"[{valid.min():.4f}, {valid.max():.4f}]"
        
        # 分类
        if std < 1e-10:
            cat = "R (拓扑真)"
            note = "严格守恒"
            r_count += 1
        elif cv < 1.0:
            cat = "S (结构真)"
            note = "CV < 1%"
            s_candidates.append((key, cv, mean, std))
        elif cv < 5.0:
            cat = "P (认知投影)"
            note = "CV < 5%"
            p_count += 1
        elif cv < 20:
            cat = "P (认知投影)"
            note = "CV < 20%"
            p_count += 1
        else:
            cat = "C (条件相关)"
            note = "CV ≥ 20%"
            c_count += 1
        
        print(f"{key:<32} {range_str:<30} {cv:<10.2f} {cat:<12} {note}")
    
    print("=" * 110)
    print(f"汇总: R={r_count} S={len(s_candidates)} P={p_count} C={c_count}")
    
    if s_candidates:
        print("\n⚠️ 发现 S 类候选量 — 锐边界可能不成立!")
        for name, cv, mean, std in s_candidates:
            print(f"  {name}: CV={cv:.4f}% mean={mean:.6f} std={std:.6f}")
    else:
        print("\n✅ 未发现 S 类量 — 锐边界定理成立")
        print("   所有嵌入依赖量均随投影参数变化,")
        print("   严格守恒仅限于纯组合量。")
    
    print("=" * 110)
    return s_candidates


# ===================================================================
# 6. 持久同调可视化
# ===================================================================
def plot_persistence(all_results, projs):
    """绘制各投影下的持久同调曲线。"""
    n_proj = len(projs)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    
    # 重新计算并绘制每个投影的持久曲线
    for idx, (proj, res) in enumerate(zip(projs, all_results)):
        coords = proj['coords']
        thresholds, betas = measure_persistent_homology(coords)
        
        ax = axes[idx]
        ax.plot(thresholds, betas[:, 1], 'b-', label='β₀', linewidth=2)
        ax.plot(thresholds, betas[:, 2], 'r-', label='β₁', linewidth=2)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.4)
        ax.set_xlabel('距离阈值 ε')
        ax.set_ylabel('Betti 数')
        ax.set_title(proj['label'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 标注 β₀_crit
        if not np.isnan(res.get('β₀_crit_threshold', np.nan)):
            crit = res['β₀_crit_threshold']
            ax.axvline(x=crit, color='blue', linestyle=':', alpha=0.5)
            ax.annotate(f'β₀→1 @ {crit:.3f}',
                       xy=(crit, 6), fontsize=7,
                       ha='center')
    
    plt.tight_layout()
    plt.savefig('persistence_curves.png', dpi=150)
    print(f"\n[图] 持久同调曲线: persistence_curves.png")


# ===================================================================
# 7. 执行入口
# ===================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SPUM 锐边界挑战 v0.3b")
    print("核心追问：是否存在嵌入依赖的投影不变式？")
    print("=" * 70)
    
    all_results, projs = run_extended_scan()
    s_candidates = classify_extended(all_results)
    plot_persistence(all_results, projs)
    
    print("\n" + "=" * 70)
    print("执行完成。检查 persistence_curves.png 查看 Betti 演化。")
    print("=" * 70)
