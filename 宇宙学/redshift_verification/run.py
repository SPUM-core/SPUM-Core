#!/usr/bin/env python3
"""
SPUM 红移验证 — 主入口。

验证 SPUM 宇宙学的核心预言：
  红移 z = σ_source / σ_obs - 1

用法:
    python run.py                       # 全部星系团（默认：内嵌样本）
    python run.py --online              # 尝试 SDSS API 获取数据
    python run.py --clusters Coma Virgo  # 指定星系团

输出:
    results/ 目录下包含:
    - {cluster}_redshift_verification.png  # 每个星系团的 z-σ 图
    - cross_cluster_summary.png            # 跨簇统计
    - sigma_field.png                      # σ 场投影
    - redshift_derivation_chain.txt        # 完整推演链路
    - results.json                         # 数值结果
"""

import argparse
import sys
import os

# 确保包路径正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verify import RedshiftVerifier
from config import config


def main():
    parser = argparse.ArgumentParser(
        description="SPUM 红移验证 — z = σ_source/σ_obs - 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python run.py                    全部星系团（内嵌样本）
    python run.py --online           尝试 SDSS 数据
    python run.py --clusters Coma    仅 Coma 星系团
        """
    )
    parser.add_argument("--online", action="store_true",
                        help="尝试在线 SDSS 数据（默认：内嵌样本）")
    parser.add_argument("--clusters", nargs="+", default=None,
                        help="指定星系团名称（多个用空格分隔）")
    parser.add_argument("--n-nearest", type=int, default=5,
                        help="第 N 近邻用于密度估计（默认：5）")
    parser.add_argument("--output", type=str, default="results",
                        help="输出目录（默认：results）")

    args = parser.parse_args()
    config.n_nearest = args.n_nearest
    config.output_dir = args.output

    # 运行验证
    verifier = RedshiftVerifier()
    results = verifier.run(
        cluster_names=args.clusters,
        use_cached=not args.online,
    )

    # 保存结果
    verifier.save_results()

    # 最终判定
    if results:
        rhos = [r["spearman_rho"] for r in results.values()]
        mean_rho = sum(rhos) / len(rhos)
        n_pos = sum(1 for r in rhos if r > 0)

        print("\n" + "=" * 60)
        print("SPUM 红移验证 — 最终判定")
        print("=" * 60)
        print(f"  ✓ 检验了 {len(results)} 个星系团")
        print(f"  ✓ 平均 Spearman ρ = {mean_rho:.4f}")
        print(f"  ✓ {n_pos}/{len(rhos)} 个星系团支持 z 与 σ 正相关")
        if mean_rho > 0.2:
            print(f"\n  ▶ 判定: SPUM 预言得到强支持")
            print(f"    红移是 σ 差异的本征读数，非退行速度。")
        elif mean_rho > 0:
            print(f"\n  ▶ 判定: SPUM 预言得到弱支持")
        else:
            print(f"\n  ▶ 判定: 不明确，需更多数据")
        print("=" * 60)
    else:
        print("\n  ✗ 无有效数据")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
