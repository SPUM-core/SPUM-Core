"""
饱和区验证 步骤 A: 拐点定位
=====================================================

对 b 从 0.0 到 1.0 加密采样 (0.6-1.0 段步长 0.05),
测量稳态 delta 均值/标准差、E_R 末值。

判据说明 (v2, 修正):
    发现 ann_rate=5 使 delta(b) 呈整数阶梯 (round(b*5)),
    逐段斜率判据被量化污染 -> 改用双判据:
    判据 B1 (方差跃升): std/delta 首次超过 0.2 (线性段为 0);
    判据 B2 (E_R 耗尽): E_R 首次低于初始值的一半 (耗损显著)。
    两个判据都指向的 b 区间即饱和拐点。

运行: python openSPUM/tests/stepA_saturation.py
"""

from __future__ import annotations
import sys
import math
from pathlib import Path

_root = str(Path(__file__).resolve().parent)  # openSPUM/tests/
if _root not in sys.path:
    sys.path.insert(0, _root)

from archimedes_vacuum_weight import (
    build_network, select_region, run_frame_boundary, region_stats,
)


def main() -> None:
    b_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.65, 0.68, 0.7, 0.72, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    n_r, frames, warmup, growth, ann_rate = 40, 40, 8, 3, 5

    W = 72
    print("=" * W)
    print("  饱和区验证 步骤 A: 拐点定位 (delta vs b)")
    print("=" * W)
    print(f"  参数: |R|={n_r}, ann_rate={ann_rate}, frames={frames}, "
          f"warmup={warmup}, growth={growth}")
    print()
    print(f"  {'b':<7}{'delta':<9}{'std':<9}{'std/delta':<10}"
          f"{'E_R末':<9}{'E_R耗损%':<10}")
    print(f"  {'─'*7}{'─'*9}{'─'*9}{'─'*10}{'─'*9}{'─'*10}")

    rows = []
    for b in b_values:
        reg, pool = build_network()
        R = select_region(reg, n_r)
        e0 = region_stats(pool, reg, R)["E_in"]
        deltas = []
        e_in = None
        for f in range(frames):
            r = run_frame_boundary(pool, reg, R, b, f,
                                   growth=growth, ann_rate=ann_rate)
            if f >= warmup:
                deltas.append(r["delta_in"])
        st = region_stats(pool, reg, R)
        e_in = st["E_in"]
        m = sum(deltas) / len(deltas)
        s = math.sqrt(sum((x - m) ** 2 for x in deltas) / len(deltas))
        rows.append((b, m, s, e_in, e0))

    for b, m, s, e_in, e0 in rows:
        sd = s / m if m > 1e-9 else 0.0
        dep = (e0 - e_in) / e0 * 100
        print(f"  {b:<7.2f}{m:<9.3f}{s:<9.3f}{sd:<10.2f}{e_in:<9d}{dep:<10.1f}")

    # ---- 双判据拐点 ----
    e0 = rows[0][4]
    hit_var = hit_dep = None
    for b, m, s, e_in, e0_ in rows:
        sd = s / m if m > 1e-9 else 0.0
        dep = (e0_ - e_in) / e0_
        if hit_var is None and sd > 0.20:
            hit_var = b
        if hit_dep is None and dep > 0.50:
            hit_dep = b
    print()
    print(f"  初始 E_R = {e0}")
    print(f"  判据 B1 (方差跃升 std/delta>0.20): 首次出现于 b = {hit_var}")
    print(f"  判据 B2 (E_R 耗损过半):            首次出现于 b = {hit_dep}")
    if hit_var is not None and hit_dep is not None:
        b_lo = min(hit_var, hit_dep) - 0.05
        b_hi = max(hit_var, hit_dep)
        print(f"  拐点区间: b ≈ {b_lo:.2f} - {b_hi:.2f}")
    print("=" * W)


if __name__ == "__main__":
    main()
