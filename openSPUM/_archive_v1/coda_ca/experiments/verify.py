"""
验证实验 — Coda CA 的基本行为验证。

验证项：
    1. 正二十面体初态保持：deg=5 节点在均值规则下是否稳定？
    2. 悬挂边动态平衡：悬挂边比例是否在稳态下收敛？
    3. Σ(6−deg) 不变量：闭合子图是否保持拓扑常数 12？
    4. 三种规则的比较：mean / median / gradient

运行：
    python -m openSPUM.coda_ca.experiments.verify
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from coda_ca import Simulator


def test_icosahedron_stability():
    """验证正二十面体在均值规则下的稳定性。

    正二十面体 12 个顶点，每个 deg=5。
    在均值规则下，所有邻居度数都是 5，均值 = 5。
    因此 deg → 5，应保持不变。
    """
    print("\n--- Test 1: 正二十面体稳定性 ---")
    sim = Simulator(n_codas=100, rule_name="mean")
    sim.initialize_icosahedron()

    for frame in range(20):
        snap = sim.step()
        active_deg5 = sum(1 for d in snap.degrees if d == 5)
        print(f"  帧 {frame+1}: deg=5 节点 = {active_deg5}/12, "
              f"总边数 = {snap.edge_count}")

    snap = sim.history[-1]
    assert snap.edge_count >= 20, "边数不应大幅下降"
    print("  ✅ 正二十面体稳定性测试通过")


def test_dangling_equilibrium():
    """验证悬挂边比例是否收敛到稳态。

    使用序列初态（度数不均衡），观察悬挂边比例
    是否在几帧内收敛到稳定值。
    """
    print("\n--- Test 2: 悬挂边动态平衡 ---")
    sim = Simulator(n_codas=500, rule_name="mean")
    sim.initialize_from_sequence(n_active=100)

    ratios = []
    for frame in range(100):
        snap = sim.step()
        if snap.n_active > 0:
            ratios.append(snap.n_dangling / snap.n_active)

    # 检查后 50 帧的方差（应收敛）
    late_ratios = ratios[-50:]
    if late_ratios:
        variance = sum((r - sum(late_ratios)/len(late_ratios))**2
                       for r in late_ratios) / len(late_ratios)
        print(f"  悬挂比例（后50帧均值）: {sum(late_ratios)/len(late_ratios):.4f}")
        print(f"  后50帧方差: {variance:.6f}")
        if variance < 0.01:
            print("  ✅ 悬挂边比例收敛到稳态")
        else:
            print("  ⚠️  悬挂边比例仍在波动（可能需要更多帧）")

    snap = sim.history[-1]
    print(f"  最终状态: active={snap.n_active}, "
          f"dangling={snap.n_dangling}, "
          f"edges={snap.edge_count}")


def test_spum_invariant():
    """验证 Σ(6−deg) 拓扑不变量。

    正二十面体闭合子图：
        Σ(6−5) for 12 vertices = 12 * 1 = 12
    应始终保持 Σ(6−deg) = 12。
    """
    print("\n--- Test 3: Σ(6−deg) 拓扑不变量 ---")
    sim = Simulator(n_codas=100, rule_name="mean")
    sim.initialize_icosahedron()

    for frame in range(10):
        snap = sim.step()
        print(f"  帧 {frame+1}: Σ(6−deg) = {snap.spum_invariant}")
        assert snap.spum_invariant == 12, \
            f"Σ(6−deg) 应为 12，得到 {snap.spum_invariant}"

    print("  ✅ Σ(6−deg) = 12 保持恒定")


def test_star_evolution():
    """星形初态演化：中心 deg=50，表面 deg=1。

    表面节点 deg=1（悬挂端），均值规则下应被激活到 deg=2。
    中心 deg 应相应下降（均值趋向）。
    """
    print("\n--- Test 4: 星形初态演化 ---")
    n_surface = 12
    sim = Simulator(n_codas=100, rule_name="mean")
    sim.initialize_star(n_surface=n_surface)

    print(f"  帧 0: center deg={sim.degrees[0]}, "
          f"surface degs={sim.degrees[1:1+n_surface]}")

    for frame in range(20):
        snap = sim.step()

    final_center = sim.degrees[0]
    surface_degs = sim.degrees[1:1+n_surface]
    print(f"  帧 20: center deg={final_center}, "
          f"surface degs min={min(surface_degs)}, "
          f"max={max(surface_degs)}")
    print(f"  悬挂端数: {snap.n_dangling}")

    dangling_surface = sum(1 for d in surface_degs if d == 1)
    if dangling_surface == 0:
        print("  ✅ 所有表面悬挂端已被激活")
    else:
        print(f"  ⚠️  仍有 {dangling_surface}/{n_surface} 表面悬挂端未激活")


def compare_rules_behavior():
    """比较三种规则下的演化行为。"""
    print("\n--- Test 5: 规则行为比较 ---")
    for rule in ["mean", "median", "gradient"]:
        sim = Simulator(n_codas=500, rule_name=rule)
        sim.initialize_from_sequence(n_active=100)

        for _ in range(50):
            sim.step()

        snap = sim.history[-1]
        degs = [d for d in snap.degrees if d > 0]
        mean_deg = sum(degs) / len(degs) if degs else 0
        max_deg = max(degs) if degs else 0

        print(f"  {rule:>8}: active={snap.n_active:>4}, "
              f"dangling={snap.n_dangling:>3}, "
              f"mean_deg={mean_deg:>5.2f}, "
              f"max_deg={max_deg:>3}")


if __name__ == "__main__":
    print("=" * 60)
    print("SPUM CA — 验证实验")
    print("=" * 60)

    test_icosahedron_stability()
    test_spum_invariant()
    test_star_evolution()
    test_dangling_equilibrium()
    compare_rules_behavior()

    print("\n" + "=" * 60)
    print("全部验证完成")
    print("=" * 60)
