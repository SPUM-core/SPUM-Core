"""
SPUM 方向动力学层演示 — 反弹与惯性

场景 1: 弹性反弹 — 1g 小球 (lock=1) 撞 100kg 大球 (lock=1e5, 闭锁边界)
    - A 沿 +1 方向抽取 → 前方断供 → 缺口矢量强制反转 → 弹回
    - 动量账 Σ lock·dir 守恒 (弹性)
    - 大球获得微小同向位移 (动量转移, 与 2m1/(m1+m2) 一致)

场景 2: 惯性 = 改写需帧数 ∝ 锁定度
    - 同样撞击, lock 分别为 1/10/100 → 反转帧数递增

场景 3: 非弹性 — capture_ratio=0.5
    - 部分关系流被散射为无向涨落 (热), 反弹减弱, 动量账仍守恒

运行: python openSPUM/kinetics/demo_rebound.py
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from kinetics import Body, KineticsEngine


def scenario1_elastic_rebound():
    print("=" * 62)
    print("场景 1: 弹性反弹 (1g 撞 100kg, 闭锁边界)")
    print("=" * 62)
    eng = KineticsEngine()
    a = Body(name="A", lock=1.0, direction=1.0, position=0.0)
    b = Body(name="B", lock=1e5, direction=0.0, position=10.0, closed=True)
    eng.add(a).add(b)

    p0 = sum(x.momentum for x in eng.bodies)
    for frame in range(1, 21):
        eng.step()
        s = eng.last()
        print(f"  frame {frame:>2}: A dir={a.direction:+.4f} pos={a.position:6.2f}"
              f" | B dir={b.direction:+.8f} pos={b.position:8.4f}"
              f" | Σp={s.momentum_total:+.6f}")

    p1 = sum(x.momentum for x in eng.bodies)
    print(f"\n  初始动量账 = {p0:+.6f}   末态动量账 = {p1:+.6f}"
          f"   守恒 = {abs(p0-p1) < 1e-6}")
    # 理论值: A 弹回 (完全弹性, m1<<m2 → v_A'≈-v0); B 获得 2m1/(m1+m2)·v0
    vB_theory = 2.0 * 1.0 / (1.0 + 1e5)
    print(f"  B 末态速度 = {b.direction:+.8f}  理论 2m1/(m1+m2) = {vB_theory:+.8f}")
    print(f"  A 末态方向 = {a.direction:+.4f} (应 ≈ -1.0, 弹回)")


def scenario2_inertia_frames():
    print("\n" + "=" * 62)
    print("场景 2: 惯性 = 改写需帧数 ∝ 锁定度")
    print("=" * 62)
    for lock in (1, 10, 100):
        eng = KineticsEngine()
        a = Body(name="A", lock=float(lock), direction=1.0, position=0.0)
        b = Body(name="B", lock=1e5, direction=0.0, position=10.0, closed=True)
        eng.add(a).add(b)
        # 反转期间位置冻结: 检测 A 位置停住的帧数 = 反转帧数
        reversal_frames = 0
        prev_pos = a.position
        for _ in range(500):
            eng.step()
            if a.position == prev_pos and a.direction != 1.0:
                reversal_frames += 1
            if a.direction <= -0.999:
                break
            prev_pos = a.position
        print(f"  lock={lock:>4} → 反转帧数 = {reversal_frames:>3}   "
              f"(∝ lock: {reversal_frames/max(lock,1):.1f} 帧/单位)")


def scenario3_inelastic():
    print("\n" + "=" * 62)
    print("场景 3: 非弹性 (capture_ratio=0.5) — 部分关系流散射为热")
    print("=" * 62)
    eng = KineticsEngine(capture_ratio=0.5)
    a = Body(name="A", lock=1.0, direction=1.0, position=0.0)
    b = Body(name="B", lock=1e5, direction=0.0, position=10.0, closed=True)
    eng.add(a).add(b)

    p0 = sum(x.momentum for x in eng.bodies)
    k0 = sum(x.kinetic for x in eng.bodies)
    for _ in range(20):
        eng.step()
    s = eng.last()
    p1 = sum(x.momentum for x in eng.bodies)
    print(f"  A 末态方向 = {a.direction:+.4f} (弹回减弱, |v|<1)")
    print(f"  末态动量账 = {p1:+.6f} vs 初始 {p0:+.6f} → 守恒 = {abs(p0-p1) < 1e-6}")
    print(f"  散射为热的关系流 = {s.heat_total:.4f} (>0, 动量账仍守恒)")
    print(f"  动能账 ½Σlock·dir²: {k0:.4f} → {s.kinetic_total:.4f}  损失 = {s.heat_total:.4f} = 热")


if __name__ == "__main__":
    scenario1_elastic_rebound()
    scenario2_inertia_frames()
    scenario3_inelastic()
