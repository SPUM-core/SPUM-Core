"""
晶子闭环排查诊断 — 检查演化中晶子子图为何不能形成 30 边/度5 正二十面体。

诊断项:
    D1. 晶子总数与最大连通分量
    D2. 最大分量内的子图度分布 (是否全为 5?)
    D3. 中心粒子 (cent_) 是否进入晶子子图 (破坏正则性?)
    D4. 晶子度数是否被 42 硬顶 (排他性是否生效?)
    D5. 有无节点超过 42 (创生步骤绕过排他性?)
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

from Phase_0.constants import CRYSTALLITE_DEGREE_THRESHOLD, KAPPA
from universe.simulator import UniverseSimulator, UniverseConfig


def diag(seed_geometry="star", n_surface=42, n_centers=8, surface_per_center=12,
         n_frames=40, pre_growth=5):
    cfg = UniverseConfig(
        seed_geometry=seed_geometry, n_surface=n_surface,
        n_centers=n_centers, surface_per_center=surface_per_center,
        pre_growth_frames=pre_growth, gap_max_checks=2000000,
    )
    sim = UniverseSimulator(cfg)
    sim.run(n_frames)

    p = sim.engine.particles
    active = p.active
    deg = p.degree
    is_crys = active & (deg >= CRYSTALLITE_DEGREE_THRESHOLD)
    n_crys = int(np.sum(is_crys))
    print(f"=== 诊断: {seed_geometry} {n_frames} 帧 ===")
    print(f"帧 {sim.engine.frame_number}  V={int(np.sum(active))}  "
          f"晶子数={n_crys}  晶子度范围="
          f"{int(np.min(deg[is_crys])) if n_crys else '-'}.."
          f"{int(np.max(deg[is_crys])) if n_crys else '-'}")

    # D5: 超过 42 的节点
    over = active & (deg > CRYSTALLITE_DEGREE_THRESHOLD)
    print(f"[D5] deg>42 节点数 = {int(np.sum(over))}  "
          f"(最大 deg = {int(np.max(deg[active])) if np.sum(active) else 0})")

    if n_crys < 2:
        print("晶子不足, 无法构成闭环")
        return

    # 晶子诱导子图
    idx_set = set(int(i) for i in np.where(is_crys)[0])
    adj = {i: set() for i in idx_set}
    for (a, b) in p.connections:
        if a in idx_set and b in idx_set:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

    # 连通分量
    visited = set()
    comps = []
    for start in adj:
        if start in visited:
            continue
        stack = [start]
        comp = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    stack.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)

    print(f"[D1] 连通分量数 = {len(comps)}  最大分量 = {len(comps[0])} 晶子")
    for ci, comp in enumerate(comps[:5]):
        sub_deg = sorted((len(adj[i]) for i in comp), reverse=True)
        # 分量内边
        comp_set = set(comp)
        n_edges = sum(
            1 for (a, b) in p.connections
            if a in comp_set and b in comp_set) // 2
        n_center = sum(
            1 for i in comp if str(p.uid[i]).startswith("cent_"))
        print(f"  分量{ci}: n={len(comp)} 边={n_edges} "
              f"子图度分布={sub_deg[:10]}{'...' if len(sub_deg) > 10 else ''} "
              f"中心数={n_center}")
        # D2: 度5 计数
        deg5 = sum(1 for i in comp if len(adj[i]) == 5)
        print(f"        deg=5 计数 = {deg5}  Σ(6−deg) = "
              f"{sum(6 - len(adj[i]) for i in comp)}")
        # D3: 中心参与?
        for i in comp:
            if str(p.uid[i]).startswith("cent_"):
                print(f"        中心 {p.uid[i]} 在分量内, 子图度 = {len(adj[i])}")
                break

    # 晶子连接: 全部晶子对的全局 (非分量) 检查
    print(f"[D3] 晶子子图包含中心粒子: "
          f"{any(str(p.uid[i]).startswith('cent_') for i in idx_set)}")
    # 晶子的外部连接 (非晶子邻居)
    if n_crys:
        ext = []
        for i in idx_set:
            n_ext = deg[int(i)] - len(adj[int(i)])
            ext.append(n_ext)
        print(f"[D3] 晶子平均外部边(非晶子邻居) = {np.mean(ext):.1f}  "
              f"晶子平均总度 = {np.mean([deg[int(i)] for i in idx_set]):.1f}")
    return sim


if __name__ == "__main__":
    print("=" * 70)
    print("场景 A: star (42 表面) 40 帧")
    diag("star", n_surface=42, n_frames=40)
    print("=" * 70)
    print("场景 B: multi_center (8 簇 × 12) 40 帧")
    diag("multi_center", n_centers=8, surface_per_center=12, n_frames=40)
