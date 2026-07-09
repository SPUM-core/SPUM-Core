"""
P1 — 光谱指纹子网

从原始光谱构建 ⟨P, ε⟩，计算 σ，检验 σ 的空间分布结构。

输入：原始光谱（RawSpectrum）
输出：光谱指纹子网的分析日志 + 可视化

用法：
    python p1_光谱指纹子网.py [--mock] [--ra 194.95] [--dec 27.98]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import argparse
from datetime import datetime

from 原始数据.spectral_loader import (
    load_spectra_by_coords, build_dual_fingerprints, RawSpectrum
)
from 公用 import (
    compute_similarity_matrix, build_graph, compute_sigma,
    spatial_kde_density, write_pipeline_log
)


def run_p1(ra: float = 194.95, dec: float = 27.98,
           radius_deg: float = 0.5, use_mock: bool = False,
           output_dir: str = None) -> dict:
    """
    执行 P1 管道。
    
    步骤：
    1. 加载原始光谱
    2. 构建双通道指纹（连续谱指纹用于 σ，全谱指纹保留）
    3. 计算连续谱指纹的相似度矩阵
    4. 构建 ⟨P, ε⟩ 图
    5. 计算每个节点的 σ（从连续谱指纹）
    6. 计算空间密度分布
    7. 独立性置换检验：验证 σ 与谱线偏移无内禀耦合
    8. 检验 σ 是否呈空间结构
    """
    print("=" * 60)
    print(f"P1 — 光谱指纹子网")
    print(f"    {datetime.now().isoformat()}")
    print(f"    中心: RA={ra:.2f}°, Dec={dec:.2f}°, 半径={radius_deg}°")
    print(f"    {'模拟数据' if use_mock else 'SDSS 真实数据'}")
    print("=" * 60)
    
    # Step 1: 加载原始光谱
    print(f"\n[1/8] 加载原始光谱...")
    spectra = load_spectra_by_coords(ra, dec, radius_deg,
                                      max_spectra=500, use_mock=use_mock)
    print(f"      加载 {len(spectra)} 条光谱")
    
    if len(spectra) == 0:
        return {"status": "error", "message": "未加载到光谱"}
    
    # Step 2: 双通道指纹
    print(f"\n[2/8] 构建双通道指纹...")
    dataset = build_dual_fingerprints(spectra)
    fp_cont = dataset['fingerprint_continuum']   # (n, 50) ← 用于 σ
    fp_full = dataset['fingerprint_full']        # (n, 200) ← 用于偏移测量
    metadata = dataset['metadata']
    print(f"      连续谱指纹: {fp_cont.shape}")
    print(f"      全谱指纹:   {fp_full.shape}")
    
    # Step 3: 相似度矩阵（用连续谱指纹）
    print(f"\n[3/8] 计算连续谱指纹相似度...")
    sim = compute_similarity_matrix(fp_cont, metric='cosine')
    print(f"      相似度范围: [{sim.min():.4f}, {sim.max():.4f}]")
    
    # Step 4: 构建 ⟨P, ε⟩
    k = max(5, len(spectra) // 20)
    print(f"\n[4/8] 构建 ⟨P, ε⟩ (mutual kNN, k={k})...")
    adj = build_graph(sim, mode='mknn', k=k)
    n_edges = np.sum(adj) / 2
    print(f"      边数: {int(n_edges)}")
    
    # Step 5: 计算 σ
    print(f"\n[5/7] 计算 σ...")
    sigma_global = compute_sigma(adj, mode='global')
    sigma_local = compute_sigma(adj, mode='local')
    
    print(f"      σ (全局): 均值={np.mean(sigma_global):.3f}, "
          f"标准差={np.std(sigma_global):.3f}")
    print(f"      σ (局部): 均值={np.mean(sigma_local):.3f}, "
          f"标准差={np.std(sigma_local):.3f}")
    
    # Step 6: 空间分布
    print(f"\n[6/8] 计算空间密度...")
    ra_arr = np.array([m['ra'] for m in metadata])
    dec_arr = np.array([m['dec'] for m in metadata])
    spatial_density = spatial_kde_density(ra_arr, dec_arr)
    
    # Step 7: 独立性置换检验
    # 验证 σ（来自连续谱）与谱线偏移（来自全谱）无内禀耦合
    print(f"\n[7/8] 独立性置换检验...")
    print(f"      验证 σ(连续谱) 与谱线偏移(全谱) 之间无内禀耦合")
    
    from scipy.stats import spearmanr
    
    # 用全谱指纹中 Hα 大致位置附近的 bin 模拟偏移量
    # 5450-7600Å 范围内总通量占比作为"红端富集度"代理
    halpha_region = (fp_full[:, 100:180].sum(axis=1) / 
                     fp_full.sum(axis=1).clip(min=1e-10))
    
    rho_real, p_real = spearmanr(sigma_local, halpha_region)
    print(f"      真实数据: ρ(σ, 红端富集度) = {rho_real:.4f} (p={p_real:.4f})")
    
    # 置换检验：打乱 σ 标签 200 次
    n_perm = 200
    rho_perm = np.zeros(n_perm)
    for perm in range(n_perm):
        shuffled = np.random.permutation(sigma_local)
        rho_perm[perm], _ = spearmanr(shuffled, halpha_region)
    
    # 如果真实 ρ 超出 95% 置换分布 → 独立性不成立
    ci_low = np.percentile(rho_perm, 2.5)
    ci_high = np.percentile(rho_perm, 97.5)
    independence_violated = ci_low <= rho_real <= ci_high
    
    print(f"      打乱 95% 区间: [{ci_low:.4f}, {ci_high:.4f}]")
    if independence_violated:
        print(f"      ❌ 真实 ρ 在打乱区间内 → 独立性失败！")
    else:
        print(f"      ✅ 真实 ρ 超出打乱区间 → 独立性成立")
    
    # Step 8: 检验 σ 的空间结构
    print(f"\n[8/8] 检验 σ 空间结构...")
    rho_sigma_spatial, p_sigma_spatial = spearmanr(sigma_local, spatial_density)
    
    print(f"      ρ(σ_local, spatial_density) = {rho_sigma_spatial:.4f} "
          f"(p = {p_sigma_spatial:.4f})")
    
    # 核心/外围 σ 比较
    median_density = np.median(spatial_density)
    core_mask = spatial_density > median_density
    outer_mask = ~core_mask
    
    sigma_core_mean = np.mean(sigma_local[core_mask]) if np.any(core_mask) else 0
    sigma_outer_mean = np.mean(sigma_local[outer_mask]) if np.any(outer_mask) else 0
    
    print(f"      核心 σ (高密度区): {sigma_core_mean:.4f}")
    print(f"      外围 σ (低密度区): {sigma_outer_mean:.4f}")
    print(f"      核心/外围比: {sigma_core_mean / max(sigma_outer_mean, 0.001):.3f}")
    
    # 装配结果
    results = {
        "pipeline": "P1 — 光谱指纹子网 (v3, 连续谱σ)",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "ra": ra, "dec": dec, "radius_deg": radius_deg,
            "use_mock": use_mock,
        },
        "independence_test": {
            "spearman_rho_continuum_vs_full": float(rho_real),
            "permutation_95_ci": [float(ci_low), float(ci_high)],
            "independence_holds": bool(not independence_violated),
        },
        "sigma_global": {
            "mean": float(np.mean(sigma_global)),
            "std": float(np.std(sigma_global)),
        },
        "sigma_local": {
            "mean": float(np.mean(sigma_local)),
            "std": float(np.std(sigma_local)),
        },
        "sigma_spatial_correlation": {
            "spearman_rho": float(rho_sigma_spatial),
            "p_value": float(p_sigma_spatial),
        },
        "core_vs_outer": {
            "sigma_core_mean": float(sigma_core_mean),
            "sigma_outer_mean": float(sigma_outer_mean),
            "ratio": float(sigma_core_mean / max(sigma_outer_mean, 0.001)),
        },
        "conclusion": (
            "σ 呈空间结构" 
            if abs(rho_sigma_spatial) > 0.15 
            else "σ 空间结构不显著"
        ),
    }
    
    # 输出
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "p1_结果.json")
        write_pipeline_log(log_path, results)
        print(f"\n结果已保存: {log_path}")
    
    print("\n" + "=" * 60)
    print(f"结论: {results['conclusion']}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="P1 — 光谱指纹子网")
    parser.add_argument('--ra', type=float, default=194.95, help='RA (度)')
    parser.add_argument('--dec', type=float, default=27.98, help='Dec (度)')
    parser.add_argument('--radius', type=float, default=0.5, help='搜索半径 (度)')
    parser.add_argument('--mock', action='store_true', help='使用模拟数据')
    parser.add_argument('--output', type=str, nargs='?',
                        const='../验证/', default=None, help='输出目录')
    args = parser.parse_args()
    
    run_p1(
        ra=args.ra, dec=args.dec,
        radius_deg=args.radius, use_mock=args.mock,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
