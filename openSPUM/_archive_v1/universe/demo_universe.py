"""
SPUM 完整网络演化 — 演示程序。

初态可选:
    star         — 1 中心 + N 表面 coda (单中心过饱和团簇)
    sequential   — 链式接入
    multi_center — C 个独立星簇分布式布点 (晶子分散成独立簇,
                    争取帧内涌现恰好 12 晶子闭环)

观测涌现: 悬挂端残留 (不完美)、Σ(6−deg) 不变量、晶子涌现、
12 晶子闭环 (连通分量级)、σ 空间密度与火形梯度。

运行:
    python openSPUM/universe/demo_universe.py            # star 30 帧
    python openSPUM/universe/demo_universe.py 40 multi_center
    python openSPUM/universe/demo_universe.py 40 multi_center 8 12
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from universe.simulator import UniverseSimulator, UniverseConfig


def main(n_frames: int = 30, seed_geometry: str = "star", n_surface: int = 42,
         n_centers: int = 8, surface_per_center: int = 12,
         gap_max_checks: int = 1000000):
    cfg = UniverseConfig(
        seed_geometry=seed_geometry,
        n_surface=n_surface,
        n_centers=n_centers,
        surface_per_center=surface_per_center,
        gap_max_checks=gap_max_checks,
        pre_growth_frames=5,
    )
    sim = UniverseSimulator(cfg)

    if seed_geometry == "multi_center":
        print(f"初态: multi_center "
              f"({n_centers} 簇 × {surface_per_center} coda/簇)  "
              f"→  运行 {n_frames} 帧\n")
    else:
        print(f"初态: {seed_geometry} ({n_surface} 表面 coda)  "
              f"→  运行 {n_frames} 帧\n")

    sim.run(n_frames)
    sim.print_report()

    # 多中心: 显示各簇闭环涌现事件
    events = sim.report().milestones.get("ring_emergence_events", [])
    if events:
        print("\n  多簇闭环涌现事件 (帧/簇):")
        for e in events:
            print(f"    - 帧 {e['frame']} / {e['cluster_id']}")

    # 早期演化轨迹
    print("\n  早期演化轨迹 (前 12 观测帧):")
    print("  " + " | ".join([
        f"帧{h['frame']}:V={h['V']},悬前={h['dangling_before']},Σ={h['invariant']},晶={h['crystallites']}"
        for h in sim.history[:12]
    ]))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    geo = sys.argv[2] if len(sys.argv) > 2 else "star"
    if geo == "multi_center":
        nc = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        spc = int(sys.argv[4]) if len(sys.argv) > 4 else 12
        main(n_frames=n, seed_geometry=geo, n_centers=nc,
             surface_per_center=spc, gap_max_checks=2000000)
    else:
        main(n_frames=n, seed_geometry=geo)