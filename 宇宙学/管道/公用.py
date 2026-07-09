"""
SPUM 宇宙学 — 管道公用工具

共享工具函数：图构建、σ 计算、谱线测量（不含 ΛCDM）。
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


# ══════════════════════════════════════
# 1. ⟨P, ε⟩ 子网构建
# ══════════════════════════════════════

def compute_similarity_matrix(fingerprints: np.ndarray,
                               metric: str = 'cosine') -> np.ndarray:
    """
    计算指纹向量之间的相似度矩阵。
    
    参数
    ----------
    fingerprints : (n_spectra, n_features) 归一化指纹向量
    metric : 'cosine' 或 'euclidean'
    
    返回
    -------
    sim : (n_spectra, n_spectra) 相似度矩阵，值域 [0, 1]
    """
    n = fingerprints.shape[0]
    
    if metric == 'cosine':
        norms = np.linalg.norm(fingerprints, axis=1, keepdims=True)
        norms[norms == 0] = 1
        fp_norm = fingerprints / norms
        sim = fp_norm @ fp_norm.T
        sim = np.clip(sim, 0, 1)
    else:
        from scipy.spatial.distance import pdist, squareform
        dist = squareform(pdist(fingerprints, metric='euclidean'))
        sim = 1.0 - dist / dist.max()
        sim = np.clip(sim, 0, 1)
    
    return sim


def build_graph(similarity: np.ndarray,
                threshold: float = 0.85,
                mode: str = 'threshold',
                k: int = 10) -> np.ndarray:
    """
    从相似度矩阵构建 ⟨P, ε⟩ 邻接矩阵。
    
    模式：
      'threshold' — ε = {(i, j) | sim[i, j] > threshold}
      'knn'       — ε = {(i, j) | j 在 i 的 top-k 最近邻中}
      'mknn'      — 互 kNN: i 和 j 互为 top-k 最近邻才连边
    
    参数
    ----------
    similarity : 相似度矩阵
    threshold : 阈值模式下的连通阈值
    mode : 'threshold', 'knn', 'mknn'
    k : kNN 模式的邻居数
    
    返回
    -------
    adjacency : 邻接矩阵（布尔）
    """
    n = similarity.shape[0]
    
    if mode == 'threshold':
        adj = similarity > threshold
        np.fill_diagonal(adj, False)
        return adj
    
    elif mode in ('knn', 'mknn'):
        # 对相似度降序排列取 top-k
        # 注意：自身相似度最高（对角线），需要排除
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            # 获取 top-k 最近邻（排除自身）
            order = np.argsort(similarity[i])[::-1]
            neighbors = order[order != i][:k]
            adj[i, neighbors] = True
        
        if mode == 'mknn':
            # 互 kNN：i→j 且 j→i
            adj = adj & adj.T
        
        return adj
    
    else:
        raise ValueError(f"不支持的图构建模式: {mode}")


# ══════════════════════════════════════
# 2. σ 计算
# ══════════════════════════════════════

def compute_sigma(adjacency: np.ndarray,
                  mode: str = 'global') -> np.ndarray:
    """
    计算每个节点所在子网的 σ = |P|/|ε|。
    
    参数
    ----------
    adjacency : 邻接矩阵（布尔）
    mode : 'global' — 整个连通分量的 σ
           'local' — 节点一阶邻域的局部 σ
    
    返回
    -------
    sigma_values : (n_nodes,) 每个节点的 σ 值
    """
    n = adjacency.shape[0]
    
    if mode == 'global':
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(adjacency, directed=False)
        
        sigma_values = np.zeros(n)
        for comp_id in range(n_components):
            mask = (labels == comp_id)
            nodes_in_comp = np.sum(mask)
            edges_in_comp = np.sum(adjacency[mask][:, mask]) / 2
            sigma_values[mask] = nodes_in_comp / max(edges_in_comp, 1)
        
        return sigma_values
    
    elif mode == 'local':
        # 局部 σ：节点 i 的一阶邻域子图
        sigma_values = np.zeros(n)
        for i in range(n):
            neighbors = np.where(adjacency[i])[0]
            if len(neighbors) < 2:
                sigma_values[i] = len(neighbors) + 1  # 小邻域 → 高 σ
                continue
            sub_nodes = len(neighbors) + 1  # i + neighbors
            sub_edges = 0
            for ni in neighbors:
                sub_edges += np.sum(adjacency[ni][neighbors])
            sub_edges = sub_edges // 2 + np.sum(adjacency[i][neighbors])
            sigma_values[i] = sub_nodes / max(sub_edges, 1)
        
        return sigma_values
    
    else:
        raise ValueError(f"不支持的 σ 模式: {mode}")


# ══════════════════════════════════════
# 3. 谱线测量（无 ΛCDM 假设）
# ══════════════════════════════════════

# 重要谱线的实验室波长（Å）
# 来源：NIST Atomic Spectra Database（实验室测量，非宇宙学）
LINE_LAB = {
    'Hβ':    4861.3,
    '[OIII]': 4958.9,
    '[OIII]2': 5006.8,
    'HeI':   5875.6,
    '[OI]':  6301.5,
    '[NII]': 6548.1,
    'Hα':    6562.8,
    '[NII]2': 6583.5,
    '[SII]': 6716.4,
    '[SII]2': 6730.8,
    'CaII_K': 3933.7,
    'CaII_H': 3968.5,
    'NaI':   5890.0,
    'NaI2':  5895.6,
}


def measure_line_shift(flux: np.ndarray, wavelength: np.ndarray,
                        line_name: str,
                        search_width: float = 50.0) -> Optional[float]:
    """
    测量指定谱线的重心偏移。
    
    不假设偏移来自速度——只测量"谱线重心相对于实验室波长的位移"。
    
    参数
    ----------
    flux : 通量数组
    wavelength : 波长数组
    line_name : 谱线名称（在 LINE_LAB 中）
    search_width : 搜索窗半宽（Å）
    
    返回
    -------
    delta_lambda : 重心偏移值（Å），正值 = 红移，负值 = 蓝移
                   若未检出则返回 None
    """
    if line_name not in LINE_LAB:
        return None
    
    lam0 = LINE_LAB[line_name]
    
    # 搜索窗口
    mask = (wavelength >= lam0 - search_width) & (wavelength <= lam0 + search_width)
    if not np.any(mask):
        return None
    
    lam_local = wavelength[mask]
    flux_local = flux[mask]
    
    # 局部连续谱基线（取窗口两端 10% 的中值）
    n = len(flux_local)
    n_edge = max(1, n // 10)
    baseline_l = np.median(flux_local[:n_edge])
    baseline_r = np.median(flux_local[-n_edge:])
    baseline = np.interp(lam_local, [lam_local[0], lam_local[-1]],
                         [baseline_l, baseline_r])
    
    # 净通量（基线以上）
    net_flux = flux_local - baseline
    
    # 谱线存在性检验（信噪比 > 3）
    noise = np.std(flux_local[:n_edge] + flux_local[-n_edge:])
    peak_snr = np.max(net_flux) / max(noise, 1e-10)
    if peak_snr < 1.0:
        return None
    
    net_flux_positive = np.maximum(net_flux, 0)
    total = np.trapezoid(net_flux_positive, lam_local)
    
    if total <= 0:
        return None
    
    # 重心
    centroid = np.trapezoid(net_flux_positive * lam_local, lam_local) / total
    delta_lambda = centroid - lam0
    
    return delta_lambda


def measure_line_ratio(flux: np.ndarray, wavelength: np.ndarray,
                        line_a: str, line_b: str) -> Optional[float]:
    """
    测量两条谱线的强度比。
    
    参数
    ----------
    flux, wavelength : 光谱
    line_a, line_b : 谱线名称
    
    返回
    -------
    ratio : flux_a / flux_b (等值宽度比)
    """
    integrals = []
    for line in [line_a, line_b]:
        if line not in LINE_LAB:
            return None
        
        lam0 = LINE_LAB[line]
        mask = (wavelength >= lam0 - 20) & (wavelength <= lam0 + 20)
        if not np.any(mask):
            return None
        
        lam_local = wavelength[mask]
        flux_local = flux[mask]
        
        # 基线
        left = np.median(flux_local[:len(flux_local)//5])
        right = np.median(flux_local[-len(flux_local)//5:])
        baseline = np.interp(lam_local, [lam_local[0], lam_local[-1]], [left, right])
        
        net = flux_local - baseline
        net = np.maximum(net, 0)
        integral = np.trapezoid(net, lam_local)
        integrals.append(integral)
    
    if integrals[1] <= 0:
        return None
    return integrals[0] / integrals[1]


# ══════════════════════════════════════
# 4. 空间分布
# ══════════════════════════════════════

def spatial_kde_density(ra: np.ndarray, dec: np.ndarray,
                         bandwidth_deg: float = 0.1) -> np.ndarray:
    """
    用 KDE 估计光谱在 RA/Dec 平面上的空间分布密度。
    
    参数
    ----------
    ra, dec : 位置数组（度）
    bandwidth_deg : KDE 带宽（度）
    
    返回
    -------
    density : 每个点的空间密度估计值
    """
    pos = np.column_stack([ra, dec])
    
    # 简单高斯 KDE（scipy）
    n = len(ra)
    
    # 自适应宽带：取最近邻距离
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    
    # 第 10 近邻距离作为局部宽带
    k = min(10, n - 1)
    distances, _ = tree.query(pos, k=k + 1)
    local_bandwidth = distances[:, k]  # 第 k 近邻距离
    
    # 以局部宽带进行高斯权重求和
    density = np.zeros(n)
    for i in range(n):
        d = np.sqrt(np.sum((pos - pos[i]) ** 2, axis=1))
        w = np.exp(-d ** 2 / (2 * local_bandwidth[i] ** 2))
        density[i] = np.sum(w)
    
    return density


# ══════════════════════════════════════
# 5. 报告输出
# ══════════════════════════════════════

def write_pipeline_log(output_path: str, results: Dict):
    """将管道运行结果写入 JSON 日志。"""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
