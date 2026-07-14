#!/usr/bin/env python3
"""
SPUM 拓扑保真度追踪器 v0.4
帧演化中追踪 F = k-NN 邻接恢复率，检验网络是否更"3D 可理解"。
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'openSPUM'))

import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

from Phase_0.gpu_engine import SPUMEngine, EngineConfig


def measure_F(particles, k=5):
    """从粒子快照计算 F = k-NN adj_recovery_rate。"""
    active_idx = np.where(particles.active)[0]
    n = len(active_idx)
    if n < 4:
        return None

    A_true = np.zeros((n, n), dtype=np.int8)
    for ii, i in enumerate(active_idx):
        for jj, j in enumerate(active_idx):
            if jj <= ii: continue
            if particles.connected(i, j):
                A_true[ii, jj] = A_true[jj, ii] = 1

    n_edges = np.sum(A_true) // 2
    if n_edges == 0:
        return None

    G = nx.from_numpy_array(A_true)
    try:
        spl = dict(nx.all_pairs_shortest_path_length(G))
    except nx.NetworkXError:
        return None

    D = np.full((n, n), np.inf, dtype=np.float32)
    np.fill_diagonal(D, 0)
    for i in range(n):
        for j in spl[i]:
            D[i, j] = spl[i][j]

    if np.any(np.isinf(D)):
        # 断开——在最大连通分量上重试
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        if len(comps) < 1 or len(comps[0]) < 4:
            return {'F': -2.0, 'n': 0, 'E': 0}
        sub_idx = np.array(sorted(comps[0]), dtype=int)
        sub_n = len(sub_idx)
        A_sub = A_true[np.ix_(sub_idx, sub_idx)]
        G_sub = nx.from_numpy_array(A_sub)
        try:
            spl_sub = dict(nx.all_pairs_shortest_path_length(G_sub))
        except nx.NetworkXError:
            return {'F': -2.0, 'n': 0, 'E': 0}
        D_sub = np.full((sub_n, sub_n), np.inf, dtype=np.float32)
        np.fill_diagonal(D_sub, 0)
        for i in range(sub_n):
            for j in spl_sub[i]:
                D_sub[i, j] = spl_sub[i][j]
        if np.any(np.isinf(D_sub)):
            return {'F': -2.0, 'n': 0, 'E': 0}
        try:
            mds = MDS(n_components=2, dissimilarity='precomputed',
                      random_state=42, max_iter=200, normalized_stress='auto')
            coords = mds.fit_transform(D_sub)
            stress = mds.stress_
        except Exception:
            return {'F': -2.0, 'n': 0, 'E': 0}
        D_embed = squareform(pdist(coords))
        k_use = min(k, sub_n - 1)
        knn_adj = np.zeros((sub_n, sub_n), dtype=np.int8)
        for i in range(sub_n):
            nearest = np.argsort(D_embed[i])[1:k_use+1]
            knn_adj[i, nearest] = 1
        knn_adj = np.maximum(knn_adj, knn_adj.T)
        correct = np.sum((knn_adj == A_sub) & (np.eye(sub_n, dtype=bool) == 0))
        total = sub_n * (sub_n - 1)
        F = correct / total
        triu = np.triu_indices(sub_n, k=1)
        tau, _ = kendalltau(D_sub[triu], D_embed[triu])
        return {'F': F, 'stress': stress, 'tau': tau, 'n': sub_n, 'E': np.sum(A_sub)//2}

    try:
        mds = MDS(n_components=2, dissimilarity='precomputed',
                  random_state=42, max_iter=200, normalized_stress='auto')
        coords = mds.fit_transform(D)
        stress = mds.stress_
    except Exception:
        return None

    D_embed = squareform(pdist(coords))
    knn_adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        nearest = np.argsort(D_embed[i])[1:k+1]
        knn_adj[i, nearest] = 1
    knn_adj = np.maximum(knn_adj, knn_adj.T)

    correct = np.sum((knn_adj == A_true) & (np.eye(n, dtype=bool) == 0))
    total = n * (n - 1)
    F = correct / total

    triu = np.triu_indices(n, k=1)
    tau, _ = kendalltau(D[triu], D_embed[triu])

    return {'F': F, 'stress': stress, 'tau': tau, 'n': n, 'E': n_edges}


def run_experiment(name, config, n_frames=31):
    """运行单一实验，返回 F 历史。"""
    engine = SPUMEngine(config=config)
    history = []
    t_start = time.time()

    for frame in range(n_frames):
        if frame > 0:
            t0 = time.time()
            _ = engine.run_frame()

        result = measure_F(engine.particles, k=5)
        if result and result.get('F', -1) >= 0:
            result['frame'] = frame
            history.append(result)

    print(f"  {name}: {len(history)} 帧, "
          f"N {history[0]['n']}→{history[-1]['n']}, "
          f"F {history[0]['F']:.4f}→{history[-1]['F']:.4f} "
          f"[{time.time()-t_start:.0f}s]")
    return history


def plot_results(all_histories, labels):
    """多实验 F 对比图。"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    colors = ['b', 'g', 'r', 'm']

    for hist, label, c in zip(all_histories, labels, colors):
        frames = [r['frame'] for r in hist]
        Fs = [r['F'] for r in hist]
        axes[0].plot(frames, Fs, 'o-', color=c, markersize=4, label=label)
        taus = [r['tau'] for r in hist]
        axes[1].plot(frames, taus, 's-', color=c, markersize=4, label=label)

    axes[0].set_ylabel('F (adj_recovery_rate)', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('拓扑保真度追踪 — 多初态比较', fontsize=12)

    axes[1].set_ylabel("Kendall τ", fontsize=11)
    axes[1].set_xlabel('帧数', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'fidelity_tracking.png')
    plt.savefig(out, dpi=150)
    print(f"[图] {out}")


if __name__ == "__main__":
    print("=" * 60)
    print("SPUM 拓扑保真度追踪器 v0.4")
    print("=" * 60)

    configs = [
        ("Star(n=50)", EngineConfig(
            seed_geometry="star", n_surface=50,
            pre_growth_frames=5, max_particles=5000, verbose=False)),
    ]

    all_histories = []
    labels = []
    for name, cfg in configs:
        hist = run_experiment(name, cfg)
        all_histories.append(hist)
        labels.append(name)

    plot_results(all_histories, labels)
    print("=" * 60)
