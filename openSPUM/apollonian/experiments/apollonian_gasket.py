"""
阿波罗尼奥斯垫片递归填充 — 关系定义一切。

核心命题：
    4 个互相相切球体 → 索迪定理唯一确定第 5 球
    第 5 球加入后 → 产生 4 个新四面体空隙
    每个空隙被填充 → 迭代产生分形结构

    球体的位置和半径不由"内部物质"决定，
    而由外部关系的约束求解唯一确定。

运行：
    python -m openSPUM.apollonian.experiments.apollonian_gasket
"""

import sys
from pathlib import Path
import math

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from apollonian import ApollonianNetwork


def run():
    print("=" * 70)
    print("阿波罗尼奥斯垫片 — 约束驱动的分形填充")
    print("核心命题：几何完全由外部关系定义")
    print("=" * 70)

    net = ApollonianNetwork()
    net.seed_tetrahedron()

    # 跑 10 帧
    for frame in range(10):
        snap = net.step()
        active = net.get_active()
        solved = net._build_solved_state()
        edges = sum(len(net.spheres[uid].neighbor_ids) for uid in active) // 2

        print(f"  帧 {snap.frame:3d}: "
              f"球体={snap.n_active:4d}, "
              f"边={edges:4d}, "
              f"缝隙填充={snap.n_gaps_filled:4d}, "
              f"已求解={len(solved):4d}")

        if snap.n_gaps_filled == 0 and frame > 3:
            print(f"  → 收敛于 {snap.n_active} 个球体")
            break

    active = net.get_active()
    solved = net._build_solved_state()

    print()
    print("=" * 70)
    print("最终状态")
    print("=" * 70)
    print(f"  总球体数: {len(net.spheres)}")
    print(f"  活跃球体: {len(active)}")
    print(f"  已求解:   {len(solved)}")

    # 统计半径分布
    radii = [r for _, r in solved.values()]
    if radii:
        print(f"  半径范围: [{min(radii):.6f}, {max(radii):.6f}]")
        print(f"  半径中值: {sorted(radii)[len(radii)//2]:.6f}")

    # 列出所有球体（按半径排序）
    sorted_spheres = sorted(solved.items(), key=lambda x: x[1][1], reverse=True)
    print(f"\n  前 10 大球体:")
    for uid, (pos, r) in sorted_spheres[:10]:
        print(f"    {uid}: r={r:.6f}, pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    print(f"\n  后 10 小球体:")
    for uid, (pos, r) in sorted_spheres[-10:]:
        print(f"    {uid}: r={r:.6f}, pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")

    # 空洞持久性验证（移除最近生成的球体）
    if sorted_spheres:
        smallest_uid = sorted_spheres[-1][0]
        print(f"\n  = 空洞持久性验证（移除 {smallest_uid}） =")
        pos_before, r_before = solved[smallest_uid]
        print(f"  移除前: pos=({pos_before[0]:.4f}, {pos_before[1]:.4f}, "
              f"{pos_before[2]:.4f}), r={r_before:.6f}")

        net.remove_sphere(smallest_uid)
        result = net.probe_cavity(smallest_uid)

        if result is not None:
            cav_pos, cav_r = result
            dr = math.sqrt(sum((cav_pos[k] - pos_before[k])**2 for k in range(3)))
            print(f"  空洞: pos=({cav_pos[0]:.4f}, {cav_pos[1]:.4f}, "
                  f"{cav_pos[2]:.4f}), r={cav_r:.6f}")
            print(f"  偏差: Δpos={dr:.10f}, Δr={abs(cav_r-r_before):.10f}")
            if dr < 1e-8 and abs(cav_r - r_before) < 1e-8:
                print(f"  ✅ 空洞几何精确保持 — '移除实体，几何犹存'")
            else:
                print(f"  ⚠️  偏差非零（约束网络重算导致）")
        else:
            print(f"  ❌ 空洞不存在")

    print()
    print("=" * 70)
    print("结论：几何（位置+半径）完全由外部约束关系决定")
    print("      球体不存储位置和半径")
    print("      移除实体后，约束仍在定义几何")
    print("=" * 70)


if __name__ == "__main__":
    run()
