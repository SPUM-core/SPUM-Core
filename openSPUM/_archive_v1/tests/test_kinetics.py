"""
SPUM 方向动力学层验证 — 反弹与惯性的可证伪断言。

断言清单 (对应惯性.md 的核心命题):
    C1. 弹性反弹: 缺口矢量在闭锁边界断供后强制反转, 动量账守恒
    C2. 碰撞结算: B 获得 2·m1/(m1+m2)·v0 (动量转移与标准值一致)
    C3. 惯性 ∝ 锁定度: 反转帧数严格正比 lock (改写需帧数 = 弛豫)
    C4. 非弹性: capture_ratio<1 → 反转减弱, 动量账仍守恒 (一阶矩)
    C5. 热 = 二阶矩损失: 动能账损失精确等于 heat_total
    C6. 小力无死区: 亚阈值注入仍产生方向改写 (F=ma 无死区)

运行: python -m pytest openSPUM/tests/test_kinetics.py -v
      或直接: python openSPUM/tests/test_kinetics.py
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from kinetics import Body, KineticsEngine


def _setup(m1=1.0, m2=1e5, capture=1.0):
    """标准碰撞初态: A 向右抽取, 前方 B 为闭锁边界。"""
    eng = KineticsEngine(capture_ratio=capture)
    a = Body(name="A", lock=m1, direction=1.0, position=0.0)
    b = Body(name="B", lock=m2, direction=0.0, position=10.0, closed=True)
    eng.add(a).add(b)
    return eng, a, b


# ── C1: 弹性反弹 + 动量账守恒 ─────────────────────────────

def test_elastic_rebound_momentum_conserved():
    eng, a, b = _setup()
    p0 = sum(x.momentum for x in eng.bodies)
    for _ in range(40):
        eng.step()
    p1 = sum(x.momentum for x in eng.bodies)
    assert abs(p0 - p1) < 1e-6, f"动量账不守恒: {p0} -> {p1}"
    assert a.direction <= -0.999, "A 未弹回 (方向未反转)"


# ── C2: 碰撞结算与标准值一致 ──────────────────────────────

def test_momentum_transfer_matches_2m1_over_sum():
    eng, a, b = _setup(m1=1.0, m2=1e5)
    for _ in range(40):
        eng.step()
    vB_theory = 2.0 * 1.0 / (1.0 + 1e5)
    assert abs(b.direction - vB_theory) < 1e-8, \
        f"B 动量转移偏离理论: {b.direction} vs {vB_theory}"


# ── C3: 惯性 ∝ 锁定度 ─────────────────────────────────────

def test_reversal_frames_proportional_to_lock():
    frames = []
    for lock in (1, 10, 100):
        eng, a, b = _setup(m1=lock)
        n = 0
        prev_pos = a.position
        for _ in range(500):
            eng.step()
            if a.position == prev_pos and a.direction != 1.0:
                n += 1
            if a.direction <= -0.999:
                break
            prev_pos = a.position
        frames.append(n)
    assert frames == [1, 10, 100], f"反转帧数未正比 lock: {frames}"


# ── C4: 非弹性 — 动量账仍守恒 ─────────────────────────────

def test_inelastic_momentum_still_conserved():
    eng, a, b = _setup(capture=0.5)
    p0 = sum(x.momentum for x in eng.bodies)
    for _ in range(40):
        eng.step()
    p1 = sum(x.momentum for x in eng.bodies)
    assert abs(p0 - p1) < 1e-6, "非弹性时动量账不守恒"
    assert abs(a.direction + 0.5) < 1e-6, "非弹性反转应减半"


# ── C5: 热 = 二阶矩损失 ───────────────────────────────────

def test_heat_equals_kinetic_loss():
    eng, a, b = _setup(capture=0.5)
    k0 = sum(x.kinetic for x in eng.bodies)
    for _ in range(40):
        eng.step()
    s = eng.last()
    k1 = sum(x.kinetic for x in eng.bodies)
    loss = k0 - k1
    assert abs(loss - s.heat_total) < 1e-9, \
        f"热 ≠ 动能损失: {s.heat_total} vs {loss}"


# ── C6: 亚阈值注入无死区 (F=ma 线性响应) ─────────────────

def test_sub_threshold_injection_still_moves():
    """极小注入 (dir=+0.01) 不接触闭锁体时, 位置持续前进 (无死区)。"""
    eng = KineticsEngine()
    a = Body(name="A", lock=1.0, direction=0.01, position=0.0)
    b = Body(name="B", lock=1e5, direction=0.0, position=1e6, closed=True)
    eng.add(a).add(b)
    for _ in range(50):
        eng.step()
    assert a.position > 0.4, "亚阈值注入应持续产生位移 (无死区)"


if __name__ == "__main__":
    fns = [
        test_elastic_rebound_momentum_conserved,
        test_momentum_transfer_matches_2m1_over_sum,
        test_reversal_frames_proportional_to_lock,
        test_inelastic_momentum_still_conserved,
        test_heat_equals_kinetic_loss,
        test_sub_threshold_injection_still_moves,
    ]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} 通过")
    sys.exit(0 if ok == len(fns) else 1)
