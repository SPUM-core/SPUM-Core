"""
空洞持久性实验 — 验证"移除实体，几何犹存"。

核心实验：
    1. 构建四面体种子（4个互相相切球体）
    2. 用索迪定理填充缝隙 → 第5个球体
    3. 移除第5个球体 → 空洞几何持续存在
    4. 对比移除前后的位置和半径

运行：
    python -m openSPUM.apollonian.experiments.cavity_persistence
"""

import sys
from pathlib import Path
import math

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from apollonian import ApollonianNetwork


def test_soddy_theorem():
    """验证索迪定理（数学核心）。"""
    print("=" * 60)
    print("索迪定理验证")
    print("=" * 60)

    from apollonian.core.descartes import solve_bend_3d

    # 4 个等大球体（半径=1, 曲率=1）
    b_inner, b_outer = solve_bend_3d(1.0, 1.0, 1.0, 1.0)
    print(f"  四球曲率: 1.0, 1.0, 1.0, 1.0")
    print(f"  缝隙球曲率: {b_inner:.6f}  (半径={1/b_inner:.6f})")
    print(f"  包围球曲率: {b_outer:.6f}  (半径={1/abs(b_outer):.6f})")

    # 验证：5 个曲率应满足 Σb² = (Σb)²/3
    bends = [1.0, 1.0, 1.0, 1.0, b_inner]
    sb = sum(bends)
    sb2 = sum(b*b for b in bends)
    print(f"  验证: 3Σb²/(Σb)² = {3*sb2/(sb*sb):.10f} (应为 1.0)")
    assert abs(3*sb2 - sb*sb) < 1e-10, "索迪定理验证失败"
    print("  ✅ 索迪定理验证通过\n")


def test_tetrahedron_seed():
    """验证四面体种子初态。"""
    print("=" * 60)
    print("四面体种子初态")
    print("=" * 60)

    net = ApollonianNetwork()
    net.seed_tetrahedron()

    active = net.get_active()
    print(f"  种子球体数: {len(active)}")
    assert len(active) == 4

    for uid in active:
        s = net.spheres[uid]
        print(f"    {uid}: deg={s.degree}, 邻居={s.neighbor_ids}")

    # 约束求解
    solved = net._build_solved_state()
    print(f"  成功求解: {len(solved)}/{len(active)}")

    for uid in sorted(solved.keys()):
        pos, r = solved[uid]
        print(f"    {uid}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}), r={r:.4f}")

    print("  ✅ 四面体种子初态创建成功\n")
    return net


def test_apollonian_gap_fill(net: ApollonianNetwork):
    """验证缝隙填充：从 4 个种子球体生成第 5 个。"""
    print("=" * 60)
    print("阿波罗尼奥斯缝隙填充")
    print("=" * 60)

    # 第 1 帧：检测并填充缝隙
    snap1 = net.step()
    print(f"  帧 1: active={snap1.n_active}, gaps_filled={snap1.n_gaps_filled}")

    if snap1.n_gaps_filled > 0:
        # 找到新球体
        new_spheres = [uid for uid in net.spheres
                       if net.spheres[uid].is_active and uid not in
                       ["tet_00", "tet_01", "tet_02", "tet_03"]]
        for uid in new_spheres:
            s = net.spheres[uid]
            print(f"    新球体: {uid}, deg={s.degree}")
            print(f"      邻居: {s.neighbor_ids}")

        # 求解新球体的几何
        solved = net._build_solved_state()
        for uid in new_spheres:
            if uid in solved:
                pos, r = solved[uid]
                print(f"      位置: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
                print(f"      半径: {r:.6f}")
                print(f"      预期半径（索迪定理）: {1/(2+math.sqrt(6)):.6f}")
        print("  ✅ 缝隙填充成功 — 第 5 球体由索迪定理确定")
    else:
        print("  ⚠️  无缝隙被填充")
        # 调试：检查四面体是否有 4-clique
        active = net.get_active()
        print(f"  活跃节点数: {len(active)}")
        for uid in active:
            s = net.spheres[uid]
            print(f"    {uid}: deg={s.degree}, neighbors={s.neighbor_ids}")
    print()


def test_cavity_persistence(net: ApollonianNetwork, target_uid: str = ""):
    """验证空洞持久性。"""
    print("=" * 60)
    print("空洞持久性 — 移除实体，几何犹存")
    print("=" * 60)

    if not target_uid:
        # 默认移除 gap_f0001_0000（缝隙球）
        # 这个球体由 4 个大球的约束关系唯一确定
        # 移除它后，4 个大球仍旧锁定着那个精确的几何位置
        target_uid = "gap_f0001_0000"
        if target_uid not in net.spheres or not net.spheres[target_uid].is_active:
            # 回退：按度数移除
            active = net.get_active()
            if not active:
                print("  无活跃球体")
                return
            target_uid = max(active, key=lambda uid: net.spheres[uid].degree)

    # 移除前的几何
    solved_before = net._build_solved_state()
    if target_uid in solved_before:
        pos_before, r_before = solved_before[target_uid]
        print(f"  目标: {target_uid}")
        print(f"  移除前: pos=({pos_before[0]:.4f}, {pos_before[1]:.4f}, "
              f"{pos_before[2]:.4f}), r={r_before:.6f}")
        print(f"  邻居: {net.spheres[target_uid].neighbor_ids}")
    else:
        print(f"  目标 {target_uid} 未在已求解状态中")
        return

    # 移除球体
    neighbors_before = list(net.spheres[target_uid].neighbor_ids)
    net.remove_sphere(target_uid)
    print(f"  已移除 {target_uid}")

    # 查询空洞几何
    result = net.probe_cavity(target_uid)

    if result is not None:
        cav_pos, cav_r = result
        print(f"  空洞存在: pos=({cav_pos[0]:.4f}, {cav_pos[1]:.4f}, "
              f"{cav_pos[2]:.4f}), r={cav_r:.6f}")

        # 对比
        dr = math.sqrt(sum((cav_pos[k] - pos_before[k])**2 for k in range(3)))
        print(f"  位置偏差: {dr:.6f}")
        print(f"  半径偏差: {abs(cav_r - r_before):.6f}")
        if dr < 1e-6 and abs(cav_r - r_before) < 1e-6:
            print("  ✅ 空洞几何与移除前完全一致 — '移除实体，几何犹存' 验证通过")
        else:
            print("  ⚠️  空洞几何有偏差（约束网络重算导致）")
    else:
        print("  ❌ 空洞不存在（约束求解失败）")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("空洞持久性实验")
    print("验证 SPUM 核心命题：移除实体，几何犹存")
    print("=" * 60)

    # 1. 索迪定理验证
    test_soddy_theorem()

    # 2. 四面体种子
    net = test_tetrahedron_seed()

    # 3. 缝隙填充
    test_apollonian_gap_fill(net)

    # 4. 空洞持久性
    test_cavity_persistence(net)

    # 5. 后续帧
    print("=" * 60)
    print("后续帧演化")
    print("=" * 60)
    for _ in range(5):
        snap = net.step()
        s = net.summary()
        print(f"  帧 {snap.frame}: active={snap.n_active}, "
              f"edges={snap.edge_count}, gaps_filled={snap.n_gaps_filled}")

    active = net.get_active()
    solved = net._build_solved_state()
    print(f"\n 最终: {len(active)} 个活跃球体, {len(solved)} 个已求解")
    print(f" 度分布: {net.get_degree_distribution()}")
    print()

    print("=" * 60)
    print("实验完成")
    print("=" * 60)
