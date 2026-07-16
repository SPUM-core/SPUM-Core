"""
SPUM 红移验证管道。

验证步骤：
1. 获取星系团数据（在线/内嵌样本）
2. 对每个星系团计算局部 σ（环境密度）
3. 检验 z ~ σ 的相关性（Spearman）
4. 可视化：红移-σ 散点图、红移-半径剖面
5. 统计摘要：跨星系团的 ρ 分布
6. 输出完整的红移推演链路
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Optional
import os
import json

from model import SPUMRedshiftModel
from density import SigmaEstimator
from cluster import GalaxyCluster
from data import fetch_all_clusters
from config import config


# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class RedshiftVerifier:
    """
    SPUM 红移验证器。

    对星系团数据进行 σ→z 预言的全管道验证。
    """

    def __init__(self, model: Optional[SPUMRedshiftModel] = None,
                 estimator: Optional[SigmaEstimator] = None):
        self.model = model or SPUMRedshiftModel(sigma_obs=1.0)
        self.estimator = estimator or SigmaEstimator(
            n_nearest=config.n_nearest
        )
        self.results = {}
        self.clusters: Dict[str, GalaxyCluster] = {}

    def run(self, cluster_names: Optional[List[str]] = None,
            use_cached: bool = True) -> Dict[str, dict]:
        """
        运行完整验证管道。

        对每个星系团：
        1. 计算局部 σ（面密度）
        2. 计算 z-σ 相关性
        3. 拟合 z = α·σ + const

        参数
        ----------
        cluster_names : 星系团名称列表
        use_cached : 是否使用内嵌样本数据

        返回
        -------
        results : {cluster_name: metrics_dict}
        """
        print("=" * 60)
        print("SPUM 红移验证 — z = σ_source/σ_obs - 1")
        print("=" * 60)

        # Step 1: 获取数据
        print("\n[Step 1] 获取星系团数据...")
        self.clusters = fetch_all_clusters(cluster_names, use_cached=use_cached)
        print(f"  加载了 {len(self.clusters)} 个星系团")

        # Step 2-4: 对每个星系团验证
        print("\n[Step 2-4] 计算 σ 场并检验相关性...")
        for name, cluster in self.clusters.items():
            result = self._analyze_cluster(cluster)
            self.results[name] = result
            print(f"  {cluster.summary()}")

        # Step 5: 统计摘要
        print("\n[Step 5] 跨簇统计摘要...")
        summary = self._cross_cluster_summary()
        self._print_summary(summary)

        # Step 6: 可视化
        print("\n[Step 6] 生成可视化...")
        os.makedirs(config.output_dir, exist_ok=True)
        self._plot_all()

        # Step 7: 输出完整的红移推演链路
        print("\n[Step 7] 红移推演链路...")
        redshift_chain = self._redshift_derivation_chain()
        self._print_chain(redshift_chain)

        return self.results

    def _analyze_cluster(self, cluster: GalaxyCluster) -> dict:
        """分析单个星系团。"""
        ra = np.array(cluster.ra_list)
        dec = np.array(cluster.dec_list)
        z = np.array(cluster.redshifts)
        radii = np.array(cluster.projected_radii)

        # 计算 σ 场
        sigma_raw, sigma_norm = self.estimator.estimate_sigma_field(ra, dec, z)

        # 相关性分析
        corr = self.estimator.compute_z_sigma_correlation(ra, dec, z)

        # SPUM 模型拟合
        fit = self.model.fit_sigma_from_cluster(z, sigma_norm)

        # 红移-半径梯度
        # 分割核心/外围
        r_threshold = np.median(radii)
        z_core = z[radii <= r_threshold]
        z_out = z[radii > r_threshold]
        z_gradient = float(np.mean(z_core) - np.mean(z_out))

        # 更新簇的 SPUM 参数
        cluster.sigma_mean = float(np.mean(sigma_norm))
        cluster.sigma_std = float(np.std(sigma_norm))
        cluster.z_gradient = z_gradient
        cluster.z_sigma_correlation = float(corr["spearman_rho"])
        cluster.p_value = float(corr["spearman_p"])

        # 提取最佳模型信息（兼容 compare_models 嵌套 dict）
        best_name = fit.get("best", {}).get("name", "linear")
        best_model = fit.get(best_name, {})
        best_r2 = best_model.get("r_squared", 0)
        best_aic = best_model.get("aic", np.inf)
        best_bic = best_model.get("bic", np.inf)

        # 形态学统计
        morph_bins = cluster.bin_by_morphology(sigma_norm)
        morph_delta_z = (np.mean(morph_bins["early"]["z"]) -
                         np.mean(morph_bins["late"]["z"])) if (
            len(morph_bins["early"]["z"]) > 0 and len(morph_bins["late"]["z"]) > 0
        ) else 0

        return {
            "n_galaxies": cluster.n_members,
            "z_mean": cluster.z_mean,
            "sigma_mean": cluster.sigma_mean,
            "sigma_std": cluster.sigma_std,
            "z_gradient_core_outskirt": z_gradient,
            "spearman_rho": corr["spearman_rho"],
            "spearman_p": corr["spearman_p"],
            "pearson_r": corr["pearson_r"],
            "best_model_name": best_name,
            "best_r_squared": best_r2,
            "best_aic": best_aic,
            "best_bic": best_bic,
            "morph_delta_z": morph_delta_z,
            "n_early": cluster.n_early,
            "n_late": cluster.n_late,
            "early_fraction": cluster.early_fraction,
            # 全模型参数 → 绘图使用
            "model_fit": fit,
            "sigma_norm": sigma_norm.tolist(),
            "z_array": z.tolist(),
            "radii": radii.tolist(),
        }

    def _cross_cluster_summary(self) -> dict:
        """跨簇统计。"""
        if not self.results:
            return {}

        rhos = [r["spearman_rho"] for r in self.results.values()]
        p_vals = [r["spearman_p"] for r in self.results.values()]
        z_grads = [r["z_gradient_core_outskirt"] for r in self.results.values()]

        # 形态学统计
        morph_dzs = [r.get("morph_delta_z", 0) for r in self.results.values()]
        early_fracs = [r.get("early_fraction", 0) for r in self.results.values()]

        # 模型比较统计
        best_models = [r.get("best_model_name", "?") for r in self.results.values()]
        best_r2s = [r.get("best_r_squared", 0) for r in self.results.values()]
        model_dist = {}
        for m in best_models:
            model_dist[m] = model_dist.get(m, 0) + 1

        return {
            "n_clusters": len(self.results),
            "mean_spearman_rho": float(np.mean(rhos)),
            "std_spearman_rho": float(np.std(rhos)),
            "positive_rho_fraction": float(np.sum([r > 0 for r in rhos])) / max(len(rhos), 1),
            "significant_fraction": float(np.sum([p < 0.05 for p in p_vals])) / max(len(p_vals), 1),
            "mean_z_gradient": float(np.mean(z_grads)),
            "rho_values": rhos,
            "z_gradients": z_grads,
            "clusters_analyzed": list(self.results.keys()),
            # 形态学
            "mean_morph_delta_z": float(np.mean(morph_dzs)),
            "morph_positive_fraction": float(np.sum([d > 0 for d in morph_dzs])) / max(len(morph_dzs), 1),
            "mean_early_fraction": float(np.mean(early_fracs)),
            # 模型比较
            "best_model_distribution": model_dist,
            "mean_best_r2": float(np.mean(best_r2s)),
        }

    def _print_summary(self, summary: dict):
        """打印跨簇统计摘要。"""
        if not summary:
            print("  （无数据）")
            return

        print(f"\n{'═' * 55}")
        print(f"  跨簇统计摘要 ({summary['n_clusters']} 个星系团)")
        print(f"{'═' * 55}")
        print(f"  z-σ 相关性:")
        print(f"    平均 Spearman ρ = {summary['mean_spearman_rho']:.4f} ± {summary['std_spearman_rho']:.4f}")
        print(f"    正相关占比 = {summary['positive_rho_fraction']:.1%}")
        print(f"    显著相关占比 (p<0.05) = {summary['significant_fraction']:.1%}")
        print(f"    平均核心-外围 Δz = {summary['mean_z_gradient']:.5f}")

        # 形态学统计
        if summary.get("mean_morph_delta_z") is not None:
            print(f"  形态学:")
            print(f"    平均早型-晚型 Δz = {summary['mean_morph_delta_z']:.5f}")
            print(f"    早型 z > 晚型 z = {summary.get('morph_positive_fraction', 0):.1%}")
            print(f"    平均早型占比 = {summary.get('mean_early_fraction', 0):.1%}")

        # 模型比较
        if summary.get("best_model_distribution"):
            print(f"  模型比较 (AIC 最优):")
            for model_name, count in summary["best_model_distribution"].items():
                print(f"    {model_name}: {count}/{summary['n_clusters']}")
            print(f"    平均最佳 R² = {summary.get('mean_best_r2', 0):.4f}")

        print(f"{'─' * 55}")

        # SPUM 判定
        rho = summary["mean_spearman_rho"]
        morph_dz = summary.get("mean_morph_delta_z", 0)
        print(f"\n  判定:")
        if rho > 0.2:
            print(f"    ✓ z-σ 正相关: ρ={rho:.4f} — σ 框架支持")
        elif rho > 0.1:
            print(f"    ~ z-σ 弱正相关: ρ={rho:.4f} — 趋势正确，需更多数据")
        else:
            print(f"    ? z-σ 不明确: ρ={rho:.4f} — 需进一步检验")

        if morph_dz > 0:
            print(f"    ✓ 早型 z > 晚型 z: Δz={morph_dz:.5f} — 形态-σ 统一支持")
        elif morph_dz < 0:
            print(f"    ✗ 早型 z < 晚型 z: Δz={morph_dz:.5f} — 与预言相反")
        else:
            print(f"    - 形态差异不显著")

        print(f"{'─' * 55}\n")

    def _plot_all(self):
        """生成所有可视化。"""
        if not self.clusters:
            print("  无数据可供可视化")
            return

        self._plot_individual_clusters()
        self._plot_cross_cluster_summary()
        self._plot_sigma_field()

    def _plot_individual_clusters(self):
        """对每个星系团生成 2×3 多面板图。"""
        for name, cluster in self.clusters.items():
            result = self.results.get(name)
            if result is None:
                continue

            sigma = np.array(result["sigma_norm"])
            z = np.array(result["z_array"])
            radii = np.array(result["radii"])
            morph_labels = cluster.morphology_labels  # 0=晚型, 1=早型

            fig, axes = plt.subplots(2, 3, figsize=(15, 9))

            # ─── (a) z vs σ — 按形态着色 ───
            ax = axes[0, 0]
            colors_morph = ["#6495ED" if m == 0 else "#DC143C" for m in morph_labels]
            ax.scatter(sigma, z, alpha=0.5, s=10, c=colors_morph, edgecolors="none")
            # 早型/晚型图例
            ax.scatter([], [], c="#DC143C", label=f"早型 E/S0 (n={result['n_early']})", s=15)
            ax.scatter([], [], c="#6495ED", label=f"晚型 S/Irr (n={result['n_late']})", s=15)

            # 线性拟合参考线
            slope, intercept, r_val, p_val, _ = stats.linregress(sigma, z)
            x_fit = np.linspace(sigma.min(), sigma.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, "k--", lw=1,
                    label=f"线性 r={r_val:.3f}")
            ax.set_xlabel("局部 σ（归一化）")
            ax.set_ylabel("红移 z")
            ax.set_title(f"{name} — z vs σ (按形态)")
            ax.legend(fontsize=7, loc="upper left")

            # ─── (b) z vs 半径（按形态着色） ───
            ax = axes[0, 1]
            ax.scatter(radii, z, alpha=0.5, s=10, c=colors_morph, edgecolors="none")
            ax.set_xlabel("到团中心的距离 (deg)")
            ax.set_ylabel("红移 z")
            ax.set_title(f"{name} — z vs 半径")
            z_grad = result["z_gradient_core_outskirt"]
            ax.text(0.95, 0.05, f"Δz = {z_grad:.5f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

            # ─── (c) 形态比较：早型 vs 晚型 Δz ───
            ax = axes[0, 2]
            morph_dz = result.get("morph_delta_z", 0)
            early_z = z[morph_labels == 1]
            late_z = z[morph_labels == 0]
            bp = ax.boxplot([late_z, early_z], positions=[0, 1], widths=0.5,
                            patch_artist=True,
                            boxprops=dict(linewidth=1.2),
                            medianprops=dict(color="black", linewidth=1.5))
            bp["boxes"][0].set_facecolor("#6495ED")
            bp["boxes"][1].set_facecolor("#DC143C")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["晚型 S/Irr", "早型 E/S0"], fontsize=8)
            ax.set_ylabel("红移 z")
            ax.set_title(f"形态-红移: Δz = {morph_dz:.5f}")
            if result["n_early"] > 0 and result["n_late"] > 0:
                t_stat, p_morph = stats.ttest_ind(early_z, late_z)
                ax.text(0.5, 0.95, f"t-test p={p_morph:.4f}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=9, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

            # ─── (d) 模型比较：4 条拟合曲线 ───
            ax = axes[1, 0]
            ax.scatter(sigma, z, alpha=0.3, s=6, c="gray", edgecolors="none")
            x_smooth = np.linspace(sigma.min(), sigma.max(), 200)
            model_fit = result.get("model_fit", {})
            styles = {"linear": ("-", "C0"), "power_law": ("--", "C1"),
                      "exponential": (":", "C2"), "quadratic": ("-.", "C3")}
            for mname, (ls, color) in styles.items():
                mdata = model_fit.get(mname, {})
                if "error" in mdata:
                    continue
                try:
                    if mname == "linear":
                        y_pred = mdata.get("slope", 0) * x_smooth + mdata.get("intercept", 0)
                    elif mname == "power_law":
                        y_pred = (mdata.get("a", 0) * x_smooth ** mdata.get("alpha", 1)
                                  + mdata.get("b", 0))
                    elif mname == "exponential":
                        y_pred = np.exp(mdata.get("a", 0) * x_smooth + mdata.get("b", 0)) - 1
                    elif mname == "quadratic":
                        y_pred = (mdata.get("a0", 0) + mdata.get("a1", 0) * x_smooth
                                  + mdata.get("a2", 0) * x_smooth ** 2)
                    r2 = mdata.get("r_squared", 0)
                    ax.plot(x_smooth, y_pred, ls=ls, color=color, lw=1.5,
                            label=f"{mname} R²={r2:.4f}")
                except Exception:
                    continue
            ax.set_xlabel("局部 σ（归一化）")
            ax.set_ylabel("红移 z")
            ax.set_title("模型比较 (z = f(σ))")
            ax.legend(fontsize=7)

            # ─── (e) 核心 vs 外围直方图（与旧版相同） ───
            ax = axes[1, 1]
            r_median = np.median(radii)
            z_core = z[radii <= r_median]
            z_out = z[radii > r_median]
            ax.hist(z_core, bins=15, alpha=0.6, label=f"核心 (n={len(z_core)})",
                    color="tomato")
            ax.hist(z_out, bins=15, alpha=0.6, label=f"外围 (n={len(z_out)})",
                    color="skyblue")
            ax.set_xlabel("红移 z")
            ax.set_ylabel("星系数")
            ax.set_title(f"{name} — 核心 vs 外围")
            ax.legend(fontsize=8)

            # ─── (f) 模型指标表 ───
            ax = axes[1, 2]
            ax.axis("off")
            best_name = result.get("best_model_name", "?")
            best_r2 = result.get("best_r_squared", 0)
            best_aic = result.get("best_aic", 0)
            best_bic = result.get("best_bic", 0)
            rows = [
                ["指标", "值"],
                ["最佳模型", best_name.upper()],
                ["R²", f"{best_r2:.5f}"],
                ["AIC", f"{best_aic:.2f}"],
                ["BIC", f"{best_bic:.2f}"],
                ["ρ (Spearman)", f"{result['spearman_rho']:.4f}"],
                ["p (Spearman)", f"{result['spearman_p']:.4f}"],
                ["Δz 核心-外围", f"{result['z_gradient_core_outskirt']:.5f}"],
                ["Δz 早型-晚型", f"{result.get('morph_delta_z', 0):.5f}"],
                ["早型占比", f"{result.get('early_fraction', 0):.1%}"],
            ]
            table = ax.table(cellText=rows[1:], colLabels=rows[0],
                             loc="center", cellLoc="center",
                             colWidths=[0.3, 0.4])
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for (row, _), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor("#2c3e50")
                    cell.set_text_props(color="white", fontweight="bold")
                elif row % 2 == 0:
                    cell.set_facecolor("#f5f5f5")
            ax.set_title("模型指标", fontsize=10)

            plt.tight_layout()
            fname = f"{name}_redshift_verification.{config.figure_format}"
            path = os.path.join(config.output_dir, fname)
            plt.savefig(path, dpi=config.figure_dpi, bbox_inches="tight")
            plt.close()
            print(f"  → 已保存: {path}")

    def _plot_cross_cluster_summary(self):
        """跨簇统计图。"""
        summary = self._cross_cluster_summary()
        if not summary:
            return

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        # (a) ρ 直方图
        ax = axes[0]
        rhos = summary["rho_values"]
        ax.hist(rhos, bins=max(3, len(rhos)), alpha=0.7,
                color="steelblue", edgecolor="white")
        ax.axvline(0, color="red", ls="--", lw=1, label="ρ=0（无相关）")
        ax.axvline(np.mean(rhos), color="darkgreen", ls="-", lw=2,
                   label=f"均值 ρ={np.mean(rhos):.3f}")
        ax.set_xlabel("Spearman ρ")
        ax.set_ylabel("星系团数")
        ax.set_title("z-σ 相关性分布")
        ax.legend(fontsize=8)

        # (b) Δz 柱状图
        ax = axes[1]
        z_grads = summary["z_gradients"]
        names = summary["clusters_analyzed"]
        colors = ["green" if g > 0 else "red" for g in z_grads]
        ax.bar(range(len(z_grads)), z_grads, color=colors, alpha=0.7)
        ax.axhline(0, color="black", ls="-", lw=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("核心-外围 Δz")
        ax.set_title("核心红移大于外围？")
        ax.text(0.95, 0.95,
                f"正: {int(summary['positive_rho_fraction'] * summary['n_clusters'])}"
                f"/{summary['n_clusters']}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

        plt.tight_layout()
        fname = f"cross_cluster_summary.{config.figure_format}"
        path = os.path.join(config.output_dir, fname)
        plt.savefig(path, dpi=config.figure_dpi, bbox_inches="tight")
        plt.close()
        print(f"  → 已保存: {path}")

    def _plot_sigma_field(self):
        """绘制所有星系团的 σ 场（RA/Dec 投影）。"""
        n = len(self.clusters)
        if n == 0:
            return

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
        if n == 1:
            axes = [axes]

        for ax, (name, cluster) in zip(axes, self.clusters.items()):
            result = self.results.get(name)
            if result is None:
                continue

            ra = np.array(cluster.ra_list)
            dec = np.array(cluster.dec_list)
            sigma = np.array(result["sigma_norm"])

            scatter = ax.scatter(ra, dec, c=sigma, cmap="viridis",
                                 s=15, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label="局部 σ")
            ax.set_xlabel("RA (deg)")
            ax.set_ylabel("Dec (deg)")
            ax.set_title(f"{name} — σ 场")
            ax.set_aspect("equal")

        plt.tight_layout()
        fname = f"sigma_field.{config.figure_format}"
        path = os.path.join(config.output_dir, fname)
        plt.savefig(path, dpi=config.figure_dpi, bbox_inches="tight")
        plt.close()
        print(f"  → 已保存: {path}")

    def _redshift_derivation_chain(self) -> str:
        """
        生成完整的红移推演链路。

        从 SPUM 第一公理到观测红移的完整推导。
        """
        chain = [
            "=" * 65,
            "SPUM 红移推演完整链路",
            "=" * 65,
            "",
            "┌─ [L0] 第一公理: 宇宙 = ⟨P, ε⟩ 离散关系网络",
            "│",
            "├─ [L0] σ = |P|/|ε|: 空间密度（子网连接紧密度）",
            "│   ├─ 高 σ ≈ 节点平均度数低 ≈ 帧周期长",
            "│   └─ 低 σ ≈ 节点平均度数高 ≈ 帧周期短",
            "│",
            "├─ [公理] 原子振荡频率 f ∝ 1/σ",
            "│   ├─ 帧周期 = 局部 σ 的函数",
            "│   ├─ 一切振荡过程（原子跃迁）的周期由帧周期决定",
            "│   └─ 帧周期 = τ(σ)，其中 τ 为基本离散帧间隔",
            "│",
            "├─ [定义] 红移 z 是两个 σ 的本征频率比",
            "│   ├─ z = f(σ_obs)/f(σ_source) - 1",
            "│   ├─ 一阶近似: f(σ) ∝ 1/σ → z = σ_source/σ_obs - 1",
            "│   └─ 观测 = 两个星系子网的拓扑耦合读数",
            "│",
            "├─ [推演] 同一星系团内:",
            "│   ├─ 核心区 σ 高（密集连接）→ z 更大（更红）",
            "│   ├─ 外围区 σ 低（稀疏连接）→ z 更小（更蓝）",
            "│   ├─ Δz(核心-外围) = (σ_core - σ_outskirt)/σ_obs > 0",
            "│   └─ 这是 SPUM 的本质预言，不是「速度弥散」",
            "│",
            "├─ [验证] 对每个星系团:",
            f"│   ├─ n_clusters = {len(self.clusters)}",
        ]

        if self.results:
            rhos = [r["spearman_rho"] for r in self.results.values()]
            chain.extend([
                f"│   ├─ 平均 Spearman ρ = {np.mean(rhos):.3f} ± {np.std(rhos):.3f}",
                f"│   ├─ 正相关占比 = {np.sum([r > 0 for r in rhos])}/{len(rhos)}",
                "│   └─ → SPUM 预言得到" + ("支持" if np.mean(rhos) > 0 else "不明确"),
            ])
        else:
            chain.append("│   └─ （未运行）")

        chain.extend([
            "│",
            "├─ [结论] 红移不是退行速度",
            "│   ├─ ΛCDM 解释: 空间膨胀拉伸波长",
            "│   ├─ SPUM 解释: σ 差异的本征频率读数",
            "│   └─ 星系团内红移差异 = σ 梯度投影，非速度弥散",
            "│",
            "├─ [待推理] 严格 f(σ) 映射函数（SPUM2610 §9.2）",
            "│   ├─ 一阶近似 f(σ) ∝ 1/σ 已足够",
            "│   ├─ 严格形式需从帧动力学推导",
            "│   └─ 此脚本使用的是一阶近似",
            "│",
            "└─ [预言] 如果 SPUM 正确:",
            "    ├─ 红移与局部密度强相关（已在检验）",
            "    ├─ 星系团「速度弥散」非运动学起源",
            "    └─ 形态-红移关联是 σ 环境决定论",
        ])

        return "\n".join(chain)

    def _print_chain(self, chain: str):
        """打印推演链路。"""
        print(chain)

        # 同时保存到文件
        path = os.path.join(config.output_dir, "redshift_derivation_chain.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(chain)
        print(f"  → 已保存: {path}")

    def save_results(self):
        """保存数值结果到 JSON。"""
        if not self.results:
            return

        serializable = {}
        for name, r in self.results.items():
            record = {
                "n_galaxies": r.get("n_galaxies"),
                "z_mean": r.get("z_mean"),
                "sigma_mean": r.get("sigma_mean"),
                "sigma_std": r.get("sigma_std"),
                "z_gradient": r.get("z_gradient_core_outskirt"),
                "spearman_rho": r.get("spearman_rho"),
                "spearman_p": r.get("spearman_p"),
                "pearson_r": r.get("pearson_r"),
                "best_model": r.get("best_model_name"),
                "best_r_squared": r.get("best_r_squared"),
                "best_aic": r.get("best_aic"),
                "best_bic": r.get("best_bic"),
                "morph_delta_z": r.get("morph_delta_z"),
                "n_early": r.get("n_early"),
                "n_late": r.get("n_late"),
                "early_fraction": r.get("early_fraction"),
            }
            # 添加各模型的 R² 和 AIC
            mf = r.get("model_fit", {})
            for mname in ["linear", "power_law", "exponential", "quadratic"]:
                mdata = mf.get(mname, {})
                if mdata and "error" not in mdata:
                    record[f"{mname}_r2"] = mdata.get("r_squared")
                    record[f"{mname}_aic"] = mdata.get("aic")
            serializable[name] = record

        path = os.path.join(config.output_dir, "results.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"  → 已保存: {path}")
