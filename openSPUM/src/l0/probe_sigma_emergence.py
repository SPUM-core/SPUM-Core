# -*- coding: utf-8 -*-
"""probe_sigma_emergence.py — 方向 4 实验：σ 场梯度是否从帧演化中自发涌现？

理论预测（SPUM）
----------------
  σ = |P|/|ε| = 2/deg（每节点局部）
  ∇σ = mean(σ_neighbors) - σ_v（局部梯度）
  引力 = 净湮灭驱动的空间流；∇σ 梯度场是它的读数（宏观表现）

  若帧演化自发产生 σ 不均匀 ⇒ 引力的拓扑起源被实验确认。

实验设计
--------
  种子：正二十面体（均匀 deg=5, σ=0.4, ∇σ=0）
  动力学：vminus="dense" + vplus="any"（采纳的非对称动力学）
  读数：每帧快照跑 l1_projection.project()，记录：
    · σ_global = V/E
    · σ 分布（直方图、max/min ratio）
    · mean|∇σ|（平均梯度幅度）
    · max|∇σ|（最大梯度幅度）
    · deg 分布（min/max/mean）
    · V/E/χ（结构验证）

  判据：mean|∇σ| 从 ~0（均匀种子）增长到显著正值 ⇒ σ 梯度自发涌现
  反向控制：均匀种子 t=0 的 mean|∇σ| 必须精确为 0
"""
import sys, os, math
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_L1 = os.path.join(os.path.dirname(_HERE), "l1")
if _L1 not in sys.path:
    sys.path.insert(0, _L1)

from combinatorial_proto import RotNet, SEEDS, audit
from l0_core import L0Core
from l1_projection import project, sigma_of, grad_sigma, angle_defect, KAPPA


def sigma_stats(net):
    """从 RotNet 直接读 σ 场统计（不跑完整 project，更快）。"""
    ids = net.ids()
    degs = [net.deg(v) for v in ids]
    sigs = [sigma_of(d) for d in degs]
    sg = {v: sigma_of(net.deg(v)) for v in ids}
    grads = []
    for v in ids:
        nb = net.rot[v]
        if nb:
            grads.append(sum(sg[w] for w in nb) / len(nb) - sg[v])
    s_min, s_max = min(sigs), max(sigs)
    s_mean = sum(sigs) / len(sigs) if sigs else 0
    g_mean = sum(abs(g) for g in grads) / len(grads) if grads else 0
    g_max = max((abs(g) for g in grads), default=0)
    V = len(ids)
    E = net.E()
    return {
        "V": V, "E": E, "chi": "?",
        "deg_min": min(degs) if degs else 0,
        "deg_max": max(degs) if degs else 0,
        "deg_mean": sum(degs)/len(degs) if degs else 0,
        "sigma_global": V / E if E else 0,
        "sigma_min": s_min, "sigma_max": s_max,
        "sigma_ratio": s_max / s_min if s_min > 0 else float("inf"),
        "grad_mean": g_mean, "grad_max": g_max,
    }


def run(seed_name="icosa", nframes=120, cap=12, dmin=3,
        vminus="dense", vplus="any", verbose=True):
    """跑帧演化 + 每帧 σ 读数。返回 stats 列表。"""
    if seed_name in SEEDS:
        net = RotNet(SEEDS[seed_name]())
    else:
        raise ValueError(f"未知 seed={seed_name}")

    core = L0Core(net, cap=cap, dmin=dmin,
                  vminus=vminus, vplus=vplus)
    all_stats = []

    # t=0 快照
    s0 = sigma_stats(core.net)
    all_stats.append(s0)
    if verbose:
        print(f"  {'t':>3s} {'V':>4s} {'E':>5s} {'χ':>3s} "
              f"{'deg_min':>3s} {'deg_max':>3s} {'deg_mean':>6s} "
              f"{'σ_glob':>7s} {'σ_min':>6s} {'σ_max':>6s} {'σ_ratio':>7s} "
              f"{'|∇σ|_mean':>10s} {'|∇σ|_max':>10s}")
        print(f"  {'---':>3s} {'---':>4s} {'---':>5s} {'---':>3s} "
              f"{'---':>3s} {'---':>3s} {'---':>6s} "
              f"{'---':>7s} {'---':>6s} {'---':>6s} {'---':>7s} "
              f"{'---':>10s} {'---':>10s}")

    def _print(t, s):
        chi_str = str(s['chi'])
        print(f"  {t:>3d} {s['V']:4d} {s['E']:5d} {chi_str:>3s} "
              f"{s['deg_min']:3d} {s['deg_max']:3d} {s['deg_mean']:6.2f} "
              f"{s['sigma_global']:7.4f} {s['sigma_min']:6.4f} "
              f"{s['sigma_max']:6.4f} {s['sigma_ratio']:7.2f} "
              f"{s['grad_mean']:10.6f} {s['grad_max']:10.6f}")

    if verbose:
        _print(0, s0)

    for t in range(1, nframes + 1):
        fr = core.frame()
        s = sigma_stats(core.net)
        all_stats.append(s)
        if verbose and (t <= 20 or t % 10 == 0):
            _print(t, s)

    return all_stats


def analyze(stats):
    """分析 σ 梯度涌现趋势。"""
    print("\n" + "=" * 78)
    print("σ 场梯度涌现分析")
    print("=" * 78)

    t0 = stats[0]
    tN = stats[-1]

    # 反向控制：t=0 的 grad_mean 必须为 0（均匀种子）
    ctrl_ok = abs(t0["grad_mean"]) < 1e-12
    print(f"\n  反向控制：t=0 grad_mean = {t0['grad_mean']:.2e} "
          f"{'✓ (均匀种子)' if ctrl_ok else '✗ (非零!)'}")

    # σ 不均匀性增长
    ratio_0 = t0["sigma_ratio"]
    ratio_N = tN["sigma_ratio"]
    grad_0 = t0["grad_mean"]
    grad_N = tN["grad_mean"]
    grad_max_N = tN["grad_max"]

    print(f"\n  σ 不均匀性：")
    print(f"    σ_max/σ_min  t=0: {ratio_0:.4f}  →  t={len(stats)-1}: {ratio_N:.4f}")
    print(f"    |∇σ|_mean   t=0: {grad_0:.6e}  →  t={len(stats)-1}: {grad_N:.6e}")
    print(f"    |∇σ|_max    t={len(stats)-1}: {grad_max_N:.6e}")

    # 趋势判定
    grad_grew = grad_N > 10 * max(grad_0, 1e-15)
    ratio_grew = ratio_N > ratio_0 + 0.01

    print(f"\n  判定：")
    print(f"    |∇σ|_mean 显著增长（>10× t=0）= {grad_grew}")
    print(f"    σ_ratio 增长（>0.01）= {ratio_grew}")

    # 最大 |∇σ| 出现的帧
    max_grad_t = max(range(len(stats)), key=lambda i: stats[i]["grad_mean"])
    print(f"\n  最大 |∇σ|_mean 出现在 t={max_grad_t} "
          f"(值={stats[max_grad_t]['grad_mean']:.6e})")

    # σ_global 趋势
    sg_0 = t0["sigma_global"]
    sg_N = tN["sigma_global"]
    print(f"\n  σ_global：t=0 {sg_0:.4f} → t={len(stats)-1} {sg_N:.4f}")

    # 结构验证（chi 标注为 "?" 时跳过）
    chi_vals = [s["chi"] for s in stats if s["chi"] != "?"]
    chi_stable = all(c == 2 for c in chi_vals) if chi_vals else "skipped"
    print(f"\n  结构验证：χ=2 全程稳定 = {chi_stable}")

    emerged = grad_grew and ratio_grew and ctrl_ok
    print(f"\n  ★ σ 场梯度自发涌现 = {emerged}")

    return emerged


def selftest():
    """正反向控制自检。"""
    ok = True
    print("=" * 78)
    print("[probe_sigma_emergence] selftest")
    print("=" * 78)

    # ① 正向控制：均匀种子 t=0 的 grad_mean 精确为 0
    ic = RotNet(SEEDS["icosa"]())
    s0 = sigma_stats(ic)
    ok &= abs(s0["grad_mean"]) < 1e-12
    print(f"  ① icosa t=0 grad_mean={s0['grad_mean']:.2e} "
          f"[{'OK' if abs(s0['grad_mean']) < 1e-12 else 'FAIL'}]")

    # ② 正向控制：非均匀图（双锥 bi_5）t=0 的 grad_mean > 0
    from combinatorial_proto import seed_bipyramid
    bi = RotNet(seed_bipyramid(5))
    sb = sigma_stats(bi)
    ok &= sb["grad_mean"] > 0
    print(f"  ② bi_5 t=0 grad_mean={sb['grad_mean']:.6f} "
          f"[{'OK' if sb['grad_mean'] > 0 else 'FAIL'}] "
          f"(两极 σ={sigma_of(5):.4f} / 赤道 σ={sigma_of(4):.4f})")

    # ③ 短跑 5 帧确认不崩溃
    try:
        stats = run("icosa", nframes=5, verbose=False)
        ok &= len(stats) == 6
        print(f"  ③ 5 帧短跑 V={[s['V'] for s in stats]} "
              f"[{'OK' if len(stats)==6 else 'FAIL'}]")
    except Exception as e:
        ok = False
        print(f"  ③ 5 帧短跑崩溃: {e} [FAIL]")

    # ④ σ 单调性检查：sigma_of 是 deg 的减函数
    ok &= sigma_of(3) > sigma_of(5) > sigma_of(6)
    print(f"  ④ σ(3)={sigma_of(3):.4f} > σ(5)={sigma_of(5):.4f} > σ(6)={sigma_of(6):.4f} "
          f"[{'OK' if sigma_of(3) > sigma_of(5) > sigma_of(6) else 'FAIL'}]")

    print(f"\n  selftest: {'全部通过' if ok else '存在失败项'}")
    return ok


if __name__ == "__main__":
    kv = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v

    if kv.get("selftest", "0") == "1":
        selftest()
    else:
        seed = kv.get("seed", "icosa")
        nf = int(kv.get("nframes", 120))
        cap = int(kv.get("cap", 12))
        vm = kv.get("vminus", "dense")
        vp = kv.get("vplus", "any")
        dmin = int(kv.get("dmin", 3))
        stats = run(seed, nframes=nf, cap=cap, dmin=dmin,
                    vminus=vm, vplus=vp, verbose=True)
        analyze(stats)
