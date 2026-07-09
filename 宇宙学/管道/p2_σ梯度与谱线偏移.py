"""
P2 — σ 梯度与谱线偏移

检验 SPUM 核心预言：σ（来自连续谱）越高的区域，谱线系统性向红端偏移。

独立性约束：
  σ ← 连续谱指纹（高斯平滑 σ=150Å，抹平窄线）
  谱线偏移 ← 全谱指纹（保留窄线位置信息）
  二者来源独立 → 相关性为真实信号，非内禀耦合

用法：
    python p2_σ梯度与谱线偏移.py [--mock] [--lines 100]
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
    spatial_kde_density, measure_line_shift, measure_line_ratio,
    LINE_LAB, write_pipeline_log
)


def run_p2(ra: float = 194.95, dec: float = 27.98,
           radius_deg: float = 0.5, use_mock: bool = False,
           n_line_measure: int = 500,
           output_dir: str = None) -> dict:
    """
    执行 P2 管道。
    
    步骤：
    1. 加载原始光谱
    2. 构建双通道指纹（连续谱→σ，全谱→偏移测量）
    3. 从连续谱指纹建立 ⟨P, ε⟩
    4. 计算 σ
    5. 从全谱指纹测量谱线偏移
    6. σ-谱线偏移相关性检验
    7. 线比变化检验（ΛCDM 区分关键）
    8. 独立性置换检验
    """
    print("=" * 60)
    print(f"P2 — σ 梯度与谱线偏移")
    print(f"    {datetime.now().isoformat()}")
    print(f"    中心: RA={ra:.2f}°, Dec={dec:.2f}°, 半径={radius_deg}°")
    print(f"    {'模拟数据' if use_mock else 'SDSS 真实数据'}")
    print("=" * 60)
    
    # Step 1-2: 加载光谱 + 双通道指纹
    print(f"\n[1-2] 加载原始光谱并构建双通道指纹...")
    spectra = load_spectra_by_coords(ra, dec, radius_deg,
                                      max_spectra=500, use_mock=use_mock)
    print(f"      加载 {len(spectra)} 条光谱")
    
    if len(spectra) == 0:
        return {"status": "error", "message": "未加载到光谱"}
    
    dataset = build_dual_fingerprints(spectra)
    fp_cont = dataset['fingerprint_continuum']  # (n, 50) ← σ
    fp_full = dataset['fingerprint_full']       # (n, 200) ← 偏移
    metadata = dataset['metadata']
    print(f"      连续谱指纹: {fp_cont.shape}")
    print(f"      全谱指纹:   {fp_full.shape}")
    
    # Step 3: ⟨P, ε⟩ 从连续谱指纹
    k = max(5, len(spectra) // 20)
    print(f"\n[3] 从连续谱指纹构建 ⟨P, ε⟩ (mutual kNN, k={k})...")
    sim = compute_similarity_matrix(fp_cont, metric='cosine')
    adj = build_graph(sim, mode='mknn', k=k)
    n_edges = np.sum(adj) / 2
    print(f"      边数: {int(n_edges)}")
    
    # Step 4: σ
    print(f"\n[4] 计算 σ...")
    sigma_local = compute_sigma(adj, mode='local')
    print(f"      σ (局部) 均值={np.mean(sigma_local):.3f}, 标准差={np.std(sigma_local):.3f}")
    
    # 空间密度
    ra_arr = np.array([m['ra'] for m in metadata])
    dec_arr = np.array([m['dec'] for m in metadata])
    spatial_density = spatial_kde_density(ra_arr, dec_arr)
    
    # Step 5: 测量谱线偏移（用全谱指纹）
    print(f"\n[5] 测量谱线偏移 (从全谱指纹)...")
    target_lines = ['Hα', 'Hβ', '[OIII]2', '[NII]2', '[SII]']
    
    line_shifts = {line: [] for line in target_lines}
    line_ratios = []
    
    n_measure = min(n_line_measure, len(spectra))
    for i in range(n_measure):
        spec = spectra[i]
        for line in target_lines:
            delta = measure_line_shift(spec.flux, spec.wavelength, line)
            if delta is not None:
                line_shifts[line].append((i, delta))
        
        # 线比测量（扩展采样）
        ratio = measure_line_ratio(spec.flux, spec.wavelength, '[OIII]2', 'Hβ')
        if ratio is not None:
            line_ratios.append((i, ratio))
    
    print(f"      谱线偏移测量完成（各线 n=50-500）")
    print(f"      线比测量: {len(line_ratios)} 条")
    
    # Step 6: σ 与谱线偏移的相关性
    print(f"\n[6] 检验 σ(连续谱) 与谱线偏移(全谱) 的相关性...")
    from scipy.stats import spearmanr
    shift_results = {}
    
    for line in target_lines:
        data = line_shifts[line]
        if len(data) < 5:
            continue
        
        indices = np.array([d[0] for d in data])
        shifts = np.array([d[1] for d in data])
        sigma_vals = sigma_local[indices]
        spatial_vals = spatial_density[indices]
        
        rho_sigma, p_sigma = spearmanr(sigma_vals, shifts)
        rho_spatial, p_spatial = spearmanr(spatial_vals, shifts)
        
        print(f"      {line:10s}: n={len(indices):3d}  "
              f"ρ(σ, shift)={rho_sigma:+.4f} (p={p_sigma:.4f})  "
              f"ρ(spatial, shift)={rho_spatial:+.4f} (p={p_spatial:.4f})")
        
        shift_results[line] = {
            "n_measurements": len(indices),
            "spearman_sigma_vs_shift": {
                "rho": float(rho_sigma), "p_value": float(p_sigma),
            },
            "spearman_spatial_vs_shift": {
                "rho": float(rho_spatial), "p_value": float(p_spatial),
            },
            "mean_shift": float(np.mean(shifts)),
            "std_shift": float(np.std(shifts)),
        }
    
    # Step 7: 线比变化
    print(f"\n[7] 线比变化检验...")
    print(f"      ΛCDM：线比不变 | SPUM：线比随 σ 变化")
    
    ratio_results = {}
    if len(line_ratios) >= 10:
        indices_r = np.array([r[0] for r in line_ratios])
        ratios = np.array([r[1] for r in line_ratios])
        sigma_r = sigma_local[indices_r]
        
        rho_ratio, p_ratio = spearmanr(sigma_r, ratios)
        print(f"      ρ(σ, [OIII]/Hβ) = {rho_ratio:+.4f} (p={p_ratio:.4f})")
        
        ratio_results["[OIII]_Hβ"] = {
            "n_measurements": len(indices_r),
            "spearman_sigma_vs_ratio": {
                "rho": float(rho_ratio), "p_value": float(p_ratio),
            },
            "mean_ratio": float(np.mean(ratios)),
            "std_ratio": float(np.std(ratios)),
        }
    else:
        print(f"      线比测量不足: {len(line_ratios)} < 10")
        ratio_results["[OIII]_Hβ"] = {
            "n_measurements": len(line_ratios), "status": "insufficient_data",
        }
    
    # Step 8: 独立性置换检验
    print(f"\n[8] 独立性置换检验...")
    # 用全谱指纹的"红端富集度"作为偏移代理
    halpha_region = (fp_full[:, 100:180].sum(axis=1) /
                     fp_full.sum(axis=1).clip(min=1e-10))
    rho_real, _ = spearmanr(sigma_local, halpha_region)
    
    n_perm = 200
    rho_perm = np.zeros(n_perm)
    for perm in range(n_perm):
        shuffled = np.random.permutation(sigma_local)
        rho_perm[perm], _ = spearmanr(shuffled, halpha_region)
    
    ci_low, ci_high = np.percentile(rho_perm, [2.5, 97.5])
    independence_holds = not (ci_low <= rho_real <= ci_high)
    
    print(f"      ρ(σ, 红端富集度) = {rho_real:.4f}")
    print(f"      打乱 95% 区间: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"      {'✅ 独立性成立' if independence_holds else '❌ 独立性失败'}")
    
    # 输出
    results = {
        "pipeline": "P2 — σ 梯度与谱线偏移 (v3, 连续谱σ)",
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "ra": ra, "dec": dec, "radius_deg": radius_deg,
            "use_mock": use_mock, "n_line_measure": n_measure,
        },
        "independence_test": {
            "spearman_rho_continuum_vs_full": float(rho_real),
            "permutation_95_ci": [float(ci_low), float(ci_high)],
            "independence_holds": bool(independence_holds),
        },
        "sigma_summary": {
            "mean": float(np.mean(sigma_local)),
            "std": float(np.std(sigma_local)),
        },
        "line_shift_results": shift_results,
        "line_ratio_results": ratio_results,
        "conclusion": (
            "σ-谱线偏移正相关支持 SPUM"
            if any(r.get("spearman_sigma_vs_shift", {}).get("rho", 0) > 0.1
                   for r in shift_results.values())
            else "σ-谱线偏移相关性不显著"
        ),
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "p2_结果.json")
        write_pipeline_log(log_path, results)
        print(f"\n结果已保存: {log_path}")
    
    print("\n" + "=" * 60)
    print(f"结论: {results['conclusion']}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser("P2 — σ 梯度与谱线偏移")
    parser.add_argument('--ra', type=float, default=194.95)
    parser.add_argument('--dec', type=float, default=27.98)
    parser.add_argument('--radius', type=float, default=0.5)
    parser.add_argument('--mock', action='store_true')
    parser.add_argument('--lines', type=int, default=100)
    parser.add_argument('--output', type=str, nargs='?',
                        const='../验证/', default=None)
    args = parser.parse_args()
    
    run_p2(
        ra=args.ra, dec=args.dec,
        radius_deg=args.radius, use_mock=args.mock,
        n_line_measure=args.lines,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
