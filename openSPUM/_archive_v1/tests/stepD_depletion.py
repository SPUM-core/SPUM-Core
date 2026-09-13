"""
饱和区验证 步骤 D: 耗尽时间常数分离
=====================================================

持续保持 b=1 (全截断/超导态) 长时演化, 追踪 E_R(t)、delta(t)、sigma_R(t)
的耗尽包络, 提取耗尽时间常数 tau_dep。

物理设定:
    真空涨落率正比于样品内部结构量: ann_rate(t) = round(E_R(t)/32)
    (E_R=162 时 = 5/帧, 与主实验一致)
    -> E_R(t) 呈指数衰减, tau_dep = 32 帧 (理论值)

判据 (来自改进方案 5 步骤 D):
    实验需确认 tau_dep >> tau_th (热弛豫时间)。
    模拟给出 tau_dep 的帧数, 换算公式:
        tau_dep(物理) = tau_dep(帧) x 帧长
    帧长由调制频率标定; 若调制频率 10 mHz (周期 100 s),
    一个完整帧对应样品热平衡一个调制周期。

运行: python openSPUM/tests/stepD_depletion.py
"""

from __future__ import annotations
import sys
import math
from pathlib import Path

_root = str(Path(__file__).resolve().parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from archimedes_vacuum_weight import (
    build_network, select_region, run_frame_boundary, region_stats,
    _internal_edge_list,
)


def main() -> None:
    n_r, frames, growth = 40, 300, 3
    k_dep = 32.0  # ann_rate = round(E_R / k_dep)

    W = 72
    print("=" * W)
    print("  饱和区验证 步骤 D: 耗尽时间常数分离 (b=1 持续)")
    print("=" * W)
    print(f"  参数: |R|={n_r}, b=1.0, frames={frames}, growth={growth}, "
          f"ann_rate=round(E_R/{k_dep:.0f})")
    print()

    reg, pool = build_network()
    R = select_region(reg, n_r)
    st0 = region_stats(pool, reg, R)
    e0 = st0["E_in"]
    print(f"  初始: E_R={e0}, sigma_R={st0['sigma_in']:.4f}, "
          f"sigma_out={st0['sigma_out']:.4f}")
    print()

    e_series, d_series, sig_series = [], [], []
    for f in range(frames):
        e_cur = len(_internal_edge_list(pool, R))
        ann_rate = max(1, int(round(e_cur / k_dep)))
        r = run_frame_boundary(pool, reg, R, 1.0, f,
                               growth=growth, ann_rate=ann_rate)
        st = region_stats(pool, reg, R)
        e_series.append(st["E_in"])
        d_series.append(r["delta_in"])
        sig_series.append(st["sigma_in"])

    # ---- 时间常数提取 (v2: 渗漏平衡修正) ----
    # 实测发现: E_R 不归零, 而是衰减到渗漏平衡点 E_R* (级联回流)
    # -> 拟合相对平衡点的弛豫: E_R(t) - E_R* ~ exp(-t/tau_dep)
    e_star = e_series[-1]  # 平衡点 (末帧)
    d_star = d_series[-1]  # 平衡 delta
    e_star_est = sum(e_series[-50:]) / 50.0
    d_star_est = sum(d_series[-50:]) / 50.0

    t_half = t_37 = None
    for i, e in enumerate(e_series):
        delta_e = e - e_star_est
        if delta_e <= 0:
            continue
        if t_half is None and delta_e <= (e0 - e_star_est) / 2.0:
            t_half = i + 1
        if t_37 is None and delta_e <= (e0 - e_star_est) / math.e:
            t_37 = i + 1
            break

    # 相对平衡的指数拟合: ln(E_R(t) - E_R*) 线性段
    fit_n = min(t_37 if t_37 else frames, 40)
    xs = list(range(fit_n))
    ys = [math.log(max(e - e_star_est, 1e-9)) for e in e_series[:fit_n]]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    tau_fit = -1.0 / slope if slope < 0 else float("nan")

    print(f"  {'帧':<6}{'E_R':<9}{'ann':<6}{'delta':<9}{'sigma_R':<10}")
    print(f"  {'─'*6}{'─'*9}{'─'*6}{'─'*9}{'─'*10}")
    marks = set([0, 10, 30, 50, 80, 120, 160, 200, 240, 299])
    for i, (e, d, s) in enumerate(zip(e_series, d_series, sig_series)):
        if i in marks or (t_half is not None and i == t_half - 1):
            tag = ""
            if t_half is not None and i == t_half - 1:
                tag = " <- 相对平衡半衰"
            ann = max(1, int(round(e / k_dep)))
            print(f"  {i + 1:<6}{e:<9}{ann:<6}{d:<9.1f}{s:<10.4f}{tag}")

    print()
    print(f"  渗漏平衡点: E_R* = {e_star_est:.0f} "
          f"(初始 {e0} 的 {e_star_est / e0 * 100:.0f}%)")
    print(f"  平衡 delta: {d_star_est:.2f} / 帧 (初始 ~5 -> 衰减 {100 * (1 - d_star_est / 5.0):.0f}%)")
    print(f"  sigma_R: {sig_series[0]:.3f} -> 平衡 {sig_series[-1]:.3f} "
          f"(放大 {sig_series[-1] / sig_series[0]:.1f}x)")
    print(f"  E_R 相对平衡半衰期 t_1/2 = {t_half} 帧")
    print(f"  E_R 相对平衡 1/e 帧数 = {t_37} 帧")
    if t_37:
        print(f"  -> 耗尽时间常数 tau_dep = {t_37} 帧 (E_R 衰减到平衡差值的 1/e)")
    print(f"  指数拟合: tau_fit = {tau_fit:.1f} 帧")
    print(f"  [机制] 耗尽非归零: 级联回流补偿在 E_R* 处与湮灭平衡 "
          f"(渗漏平衡)")

    # ---- 与热弛豫时间 tau_th 的分离 ----
    print()
    print("  [tau_dep 与 tau_th 分离]")
    print("  换算: tau_dep(物理) = tau_dep(帧) x 帧长")
    print("  帧长标定: 调制频率 f_mod, 周期 P = 1/f_mod;")
    print("    一个完整帧 = 样品完成一次热平衡所需时长")
    print("    若 f_mod = 10 mHz (P = 100 s), 帧长 ~ P/2 = 50 s")
    if t_37:
        tau_phys = t_37 * 50.0
        print(f"    -> tau_dep(物理) ~= {t_37} 帧 x 50 s = "
              f"{tau_phys / 60:.1f} min = {tau_phys / 3600:.3f} h")
        print("  判据: 需 tau_dep >> tau_th (热弛豫时间)")
        print("    若样品热时间常数 tau_th ~ 1-10 s, 则:")
        print(f"    tau_dep/tau_th ~= {(t_37 * 50.0) / 10.0:.0f} "
              f"(下限, 以 tau_th=10 s 计) >> 1 (满足分离)")
    print("=" * W)


if __name__ == "__main__":
    main()
