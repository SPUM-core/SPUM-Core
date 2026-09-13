"""
晶子涌现验证脚本

验证: 在网络规模增大到 10^5 边量级时，
     是否自动涌现度数 >= 50 的晶子节点。

用法: python openSPUM/tests/verify_crystallite.py
"""

import sys
import os
import time
from collections import Counter
from pathlib import Path

# 确保路径正确
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from Phase_1.seed_epoch_engine import SeedEpochConfig, SeedEpochEngine


def verify_crystallite(target_edges: int, seed_duration: int, label: str):
    """运行一次种子创生期并打印晶子涌现统计。"""
    config = SeedEpochConfig(target_edges=target_edges,
                              seed_duration=seed_duration)
    engine = SeedEpochEngine(config=config)

    t0 = time.time()
    final_edges = engine.run_seed_epoch()
    elapsed = time.time() - t0

    nodes = engine.node_registry.nodes
    total_degree = sum(n.degree for n in nodes.values())
    crystallites = [n for n in nodes.values() if n.is_crystallite]
    dangling = [n for n in nodes.values() if n.is_dangling]

    print(f"\n{'=' * 60}")
    print(f"  验证: {label}")
    print(f"  target_edges={target_edges}, seed_duration={seed_duration}")
    print(f"{'=' * 60}")
    print(f"  运行耗时:       {elapsed:.2f}s")
    print(f"  总边数:         {final_edges}")
    print(f"  总节点数:       {len(nodes)}")
    print(f"  总度数之和:     {total_degree}  (应 = {2 * final_edges})")
    assert total_degree == 2 * final_edges, "度数和不等于 2 倍边数!"

    print(f"  晶子节点 (deg>=50): {len(crystallites)}")
    print(f"  悬挂节点 (deg<2):   {len(dangling)}")

    if crystallites:
        crystallites.sort(key=lambda n: n.degree, reverse=True)
        print(f"  最大度数:         {crystallites[0].degree}")
        print(f"  晶子率:           {len(crystallites) / len(nodes) * 100:.2f}%")
        print(f"  前 10 晶子度数:   {[n.degree for n in crystallites[:10]]}")

        # 按度数区间统计
        buckets = {"50-59": 0, "60-69": 0, "70-79": 0, "80-89": 0, "90+": 0}
        for c in crystallites:
            if c.degree >= 90:
                buckets["90+"] += 1
            elif c.degree >= 80:
                buckets["80-89"] += 1
            elif c.degree >= 70:
                buckets["70-79"] += 1
            elif c.degree >= 60:
                buckets["60-69"] += 1
            else:
                buckets["50-59"] += 1
        print(f"  晶子度数分布:   {buckets}")
    else:
        print("  *** 未涌现晶子! ***")

    # 度数分布概要
    deg_dist = Counter(n.degree for n in nodes.values())
    print(f"\n  度数分布快照 (低端):")
    for deg in sorted(deg_dist)[:8]:
        print(f"    degree={deg:>3}: {deg_dist[deg]:>6} 节点")
    if len(deg_dist) > 8:
        print(f"    ... (共 {len(deg_dist)} 种度数)")
        print(f"  度数分布 (高端):")
        for deg in sorted(deg_dist)[-5:]:
            print(f"    degree={deg:>3}: {deg_dist[deg]:>6} 节点")


if __name__ == "__main__":
    # 验证 1: 10^4 量级 (快)
    verify_crystallite(20000, 300, "10^4 量级 (快验证)")

    # 验证 2: 10^5 量级 (核心)
    verify_crystallite(100000, 500, "10^5 量级 (核心验证)")

    print(f"\n{'=' * 60}")
    print("  晶子涌现验证完成。")
    print(f"{'=' * 60}")
