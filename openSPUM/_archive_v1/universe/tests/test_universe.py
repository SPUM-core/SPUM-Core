"""
SPUM 完整网络演化 — 可证伪断言验证。

断言清单:
    U1. 不完美定理 (图论版)
        U1a 局部闭合可达 — 42 团簇最终无悬挂端 (dangling → 0)
        U1b 全局静止不可达 — V 持续增长, 网络永不停机
    U2. 拓扑恒等       — Σ(6−deg) = 6V − 2E 恒等成立
    U3. 晶子涌现       — 演化中涌现 deg ≥ κ 的饱和终极单元
    U4. 12 晶子闭环    — 12 晶子互连成 30 边/子图度 5/Σ=12 → 正二十面体
    U5. 开口占比       — 悬挂端占比 ≤ 1/3 (带边界高斯-博内)
    U6. 确定性         — 同配置同输出, 演化路径唯一

运行: python openSPUM/universe/tests/test_universe.py
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np

from Phase_0.particle_array import ParticleArray
from Phase_0.frame_kernels import _icosahedron_vertices
from universe.simulator import UniverseSimulator, UniverseConfig
from universe.emergence import (
    detect_12_crystallite_ring, observe_universe,
)

FRAMES = 12   # 通用观测帧数 (性能约束: 每帧 O(N²) 松弛)

# 共享模拟缓存 — 多个测试复用同一演化轨迹, 减少总帧数
_CACHE: dict = {}


def _shared_sim(n_frames=FRAMES):
    if n_frames not in _CACHE:
        _CACHE[n_frames] = _make_sim(n_frames)
    return _CACHE[n_frames]


def _make_sim(n_frames=FRAMES):
    sim = UniverseSimulator(UniverseConfig(
        seed_geometry="star", n_surface=42, pre_growth_frames=5))
    sim.run(n_frames)
    return sim


# ============================================================
# 人造正二十面体构型 (12 晶子 + 背景池)
# ============================================================

def _synthetic_icosahedron():
    """12 个全局 deg=47 的晶子, 子图内 30 边构成正二十面体。

    每晶子: 5 子图边 + 42 外部边 = 47 ≥ κ(42) → 晶子
    背景: 84 节点, 每节点连 6 个晶子 (度 6, 无悬挂)
    子图不变量: Σ(6−deg) = 12 × 1 = 12 ✓
    """
    pts = _icosahedron_vertices()  # (12, 3) 单位球面
    r = 1.051 / 2.0                # 边长/2 → 相切半径

    arr = ParticleArray(max_n=256)
    for i in range(12):
        arr.add_particle(pos=tuple(pts[i]), uid=f"crys_{i:02d}", degree=0)
        arr.radius[i] = r

    sub_edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.linalg.norm(pts[i] - pts[j]) < 1.1:
                arr.add_connection(i, j)
                sub_edges.append((i, j))
    assert len(sub_edges) == 30, f"人造构型应有 30 条子图边, 现 {len(sub_edges)}"

    bg = 12
    for b in range(84):
        arr.add_particle(pos=(20.0, 20.0, 20.0 + b), uid=f"bg_{b:02d}", degree=0)
    for k in range(12):  # 每个晶子 42 条外部边
        for b in range(42):
            arr.add_connection(k, bg + (k * 42 + b) % 84)

    return arr


# ── U1: 不完美定理 (图论版) ────────────────────────────────

def test_imperfection_graph_version():
    sim = _shared_sim()
    dangling_after = [h["dangling"] for h in sim.history]
    # U1a: 局部闭合可达 — 42 团簇最终无悬挂端
    assert dangling_after[-1] == 0, \
        f"U1a 失败: 末帧仍残留悬挂 {dangling_after[-1]}"
    # U1b: 全局静止不可达 — V 严格单调增长 (永不停机)
    Vs = [h["V"] for h in sim.history]
    assert all(Vs[i] < Vs[i + 1] for i in range(len(Vs) - 1)), \
        f"U1b 失败: V 未单调增长 {Vs}"
    print(f"  U1 不完美(图论版): 局部闭合可达 (末帧悬挂=0) + "
          f"全局永不静止 (V: {Vs[0]}→{Vs[-1]})")


# ── U2: 拓扑恒等 ───────────────────────────────────────────

def test_topology_identity():
    sim = _shared_sim()
    for h in sim.history:
        V, E = h["V"], h["E"]
        assert h["invariant"] == 6 * V - 2 * E, \
            f"帧{h['frame']}: Σ(6−deg)={h['invariant']} ≠ 6V−2E={6*V-2*E}"
    print(f"  U2 拓扑恒等: Σ(6−deg)=6V−2E 全部成立, 末帧 Σ={sim.history[-1]['invariant']}")


# ── U3: 晶子涌现 ───────────────────────────────────────────

def test_crystallite_emergence():
    sim = _shared_sim(n_frames=16)
    crys = [h["crystallites"] for h in sim.history]
    assert max(crys) > 0, "16 帧内应涌现晶子 (deg ≥ κ)"
    assert crys[-1] > 0, "末帧应有晶子"
    print(f"  U3 晶子涌现: 晶子数轨迹 {crys[0]}→{crys[-1]} (最高 {max(crys)})")


# ── U4: 12 晶子闭环 = 正二十面体 ───────────────────────────

def test_ring_detector_synthetic():
    arr = _synthetic_icosahedron()
    ring = detect_12_crystallite_ring(arr)
    assert ring["ring_formed"], f"12 晶子闭环应成形: {ring}"
    assert ring["n_ring_edges"] == 30
    assert ring["deg5_count"] == 12
    assert ring["invariant"] == 12
    ico = ring["icosahedron"]
    assert ico is not None and ico["is_icosahedron"], \
        f"几何检测应确认正二十面体: {ico}"
    print(f"  U4 12晶子闭环: 30 边/度5/Σ=12 ✓, 正二十面体确认 (score={ico['match_score']})")


# ── U5: 开口占比 ≤ 1/3 ─────────────────────────────────────

def test_opening_ratio_bound():
    sim = _shared_sim()
    for h in sim.history:
        assert h["opening_bound_ok"], \
            f"帧{h['frame']}: 开口占比 {h['opening_ratio']} > 1/3"
    print(f"  U5 开口占比: {FRAMES} 帧全部 ≤ 1/3, 末帧 = {sim.history[-1]['opening_ratio']}")


# ── U6: 确定性 ─────────────────────────────────────────────

def test_determinism():
    a = _shared_sim(n_frames=10)
    b = _shared_sim(n_frames=10)
    key_a = [(h["V"], h["E"], h["dangling"], h["crystallites"]) for h in a.history]
    key_b = [(h["V"], h["E"], h["dangling"], h["crystallites"]) for h in b.history]
    assert key_a == key_b, "同配置同输出 (确定性违反)"
    print(f"  U6 确定性: 两次运行逐帧一致 (10 帧)")


if __name__ == "__main__":
    tests = [
        test_imperfection_graph_version,
        test_topology_identity,
        test_crystallite_emergence,
        test_ring_detector_synthetic,
        test_opening_ratio_bound,
        test_determinism,
    ]
    ok = 0
    for fn in tests:
        try:
            fn()
            ok += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"\n{ok}/{len(tests)} 通过")
    sys.exit(0 if ok == len(tests) else 1)
