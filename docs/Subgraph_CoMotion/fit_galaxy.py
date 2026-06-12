#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SPUM 子图协动 — 星系旋转曲线拟合器
====================================

实现 SPUM2610 第 8 章公式 8.3 (SPUM_Subgraph_CoMotion.md 公式 3):

    v_tot(r) = sqrt( G * M_bar(r) / r + V_co^2 * r^2 / (r^2 + r_co^2) )

其中 M_bar(r) = Upsilon * L(r) + M_gas(r)，仅 3 个自由参数:
  - Upsilon: 恒星质光比
  - V_co:    子图协动渐近速度
  - r_co:    特征半径

依赖:
  pip install numpy scipy

用法:
  # 单星系拟合
  python fit_galaxy.py --data NGC3198_rotmod.dat

  # 从 SPARC 目录批量拟合
  python fit_galaxy.py --data-dir /path/to/sparc/ --out results.json

  # 指定参数边界与初值
  python fit_galaxy.py --data NGC3198_rotmod.dat \
      --ups-min 0.1 --ups-max 2.0 \
      --vco-min 20 --vco-max 400 \
      --rco-min 0.5 --rco-max 30
"""

import argparse, json, os, sys
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ── 物理常数 ──────────────────────────────────────────────────────────
G = 4.302e-6  # 引力常数 (kpc·km²/s² / M_sun)

# ── 默认拟合边界 ─────────────────────────────────────────────────────
DEFAULT_BOUNDS = {
    "ups_min": 0.1, "ups_max": 2.0,
    "vco_min": 20, "vco_max": 400,
    "rco_min": 0.5, "rco_max": 30,
}


# ══════════════════════════════════════════════════════════════════════
# 模型函数
# ══════════════════════════════════════════════════════════════════════

def v_bar(r: np.ndarray, m_bar: np.ndarray) -> np.ndarray:
    """重子物质贡献: v_bar = sqrt(G * M_bar / r)。"""
    m_safe = np.maximum(m_bar, 0)
    return np.sqrt(G * m_safe / np.maximum(r, 1e-6))


def v_co(r: np.ndarray, V_co: float, r_co: float) -> np.ndarray:
    """子图协动贡献: v_co = V_co * r / sqrt(r^2 + r_co^2)。"""
    return V_co * r / np.sqrt(r**2 + r_co**2)


def v_tot_model(
    r: np.ndarray,
    Upsilon: float,
    V_co: float,
    r_co: float,
    l_cum: np.ndarray,
    m_gas_cum: np.ndarray,
) -> np.ndarray:
    """SPUM 子图协动模型总旋转速度 (公式 3)。

    Args:
        r: 半径数组 (kpc)
        Upsilon: 恒星质光比
        V_co: 子图协动渐近速度 (km/s)
        r_co: 特征半径 (kpc)
        l_cum: 累积光度 L(r) (L_sun)，仅来自恒星
        m_gas_cum: 累积气体质量 M_gas(r) (M_sun)

    Returns:
        总旋转速度 (km/s)
    """
    m_bar = Upsilon * l_cum + m_gas_cum
    return np.sqrt(
        G * m_bar / np.maximum(r, 1e-6)
        + (V_co**2) * r**2 / (r**2 + r_co**2)
    )


def v_nfw_model(
    r: np.ndarray,
    Upsilon: float,
    M200: float,
    c: float,
    l_cum: np.ndarray,
    m_gas_cum: np.ndarray,
    H0: float = 70.0,
) -> np.ndarray:
    """NFW 暗物质晕模型。

    Args:
        M200: 晕质量 (M_sun)
        c: 浓度参数 c = R200 / r_s
    """
    m_bar = Upsilon * l_cum + m_gas_cum
    v_bar_sq = G * m_bar / np.maximum(r, 1e-6)

    rho_crit = 3 * (H0 * 1e3 / 3.086e19)**2 / (8 * np.pi * G) * 1e9
    rho_crit_Msun_kpc3 = rho_crit
    R200 = (M200 / (4 * np.pi / 3 * 200 * rho_crit_Msun_kpc3))**(1.0/3.0)
    r_s = R200 / c
    fc = np.log(1 + c) - c / (1 + c)
    x = np.maximum(r / r_s, 1e-6)
    v_halo_sq = (G * M200 / r) * (np.log(1 + x) - x / (1 + x)) / fc

    return np.sqrt(v_bar_sq + v_halo_sq)


# ══════════════════════════════════════════════════════════════════════
# 数据加载 (SPARC 格式)
# ══════════════════════════════════════════════════════════════════════

def load_sparc_dat(filepath: str) -> dict:
    """加载 SPARC .dat 文件。

    SPARC 文件通常包含列: Rad, Vobs, errV, Vgas, Vdisk, Sbulge, Sdisk, ...
    也支持包含 SBdisk 列 (面亮度 I) 和 Sgas (气体面密度) 的格式。

    Returns:
        {"r": ..., "v_obs": ..., "e_v": ..., "v_gas": ...,
         "i_disk": ..., "sigma_gas": ..., "filename": ...}
    """
    dat = np.loadtxt(filepath)
    ncols = dat.shape[1]

    d = {"filename": os.path.basename(filepath)}

    # 自动识别列 (基于常见 SPARC 列顺序)
    # 标准 SPARC _rotmod.dat: Rad Vobs errV Vgas Vdisk Sbulge [Sdisk]
    if ncols >= 3:
        d["r"] = dat[:, 0]       # kpc
        d["v_obs"] = dat[:, 1]   # km/s
        d["e_v"] = dat[:, 2]     # km/s
    if ncols >= 5:
        d["v_gas"] = dat[:, 3]   # km/s
        d["v_disk"] = dat[:, 4]  # km/s (stellar disk contribution)
    if ncols >= 6:
        d["sbulge"] = dat[:, 5]  # bulge surface brightness

    # 清理无效点
    mask = (d["r"] > 0) & (d["v_obs"] > 0) & (d["e_v"] > 0)
    for key in ["r", "v_obs", "e_v"]:
        d[key] = d[key][mask]
    if "v_gas" in d:
        d["v_gas"] = d["v_gas"][mask]
    if "v_disk" in d:
        d["v_disk"] = d["v_disk"][mask]

    # 误差下限
    d["e_v"] = np.maximum(d["e_v"], 0.5)

    return d


def compute_l_cum(r: np.ndarray, i_disk: np.ndarray) -> np.ndarray:
    """从面亮度 I (L_sun/pc^2) 计算累积光度 L(r) (L_sun)。

    使用梯形积分，r 需从 0 到 r_max 单调递增。
    单位转换: kpc^2 → pc^2 乘以 1e6。
    """
    i_safe = np.maximum(i_disk, 0)
    integrand = 2 * np.pi * r * i_safe * 1e6
    return np.array([np.trapz(integrand[:i+1], r[:i+1]) for i in range(len(r))])


def compute_m_gas_cum(r: np.ndarray, v_gas: np.ndarray) -> np.ndarray:
    """从气体速度估算累积气体质量。

    M_gas(r) ≈ v_gas(r)^2 * r / G (球对称近似)
    """
    m = v_gas**2 * np.maximum(r, 1e-6) / G
    return m


# ══════════════════════════════════════════════════════════════════════
# 拟合
# ══════════════════════════════════════════════════════════════════════

def fit_spum(
    r: np.ndarray,
    v_obs: np.ndarray,
    e_v: np.ndarray,
    l_cum: np.ndarray,
    m_gas_cum: np.ndarray,
    bounds: dict = None,
) -> dict:
    """拟合 SPUM 子图协动模型。

    Returns:
        {"Upsilon": ..., "V_co": ..., "r_co": ...,
         "chi2_red": ..., "v_fit": ..., "success": bool, "message": str}
    """
    bnd = bounds or DEFAULT_BOUNDS
    p0 = _guess_init(r, v_obs)

    try:
        popt, pcov = curve_fit(
            lambda r, U, V, R: v_tot_model(r, U, V, R, l_cum, m_gas_cum),
            r, v_obs,
            sigma=e_v,
            p0=p0,
            bounds=(
                [bnd["ups_min"], bnd["vco_min"], bnd["rco_min"]],
                [bnd["ups_max"], bnd["vco_max"], bnd["rco_max"]],
            ),
            method="trf",
            max_nfev=2000,
        )
        v_fit = v_tot_model(r, *popt, l_cum, m_gas_cum)
        chi2 = np.sum(((v_obs - v_fit) / e_v)**2)
        dof = len(r) - 3
        chi2_red = chi2 / dof if dof > 0 else np.inf

        return {
            "Upsilon": float(popt[0]),
            "V_co": float(popt[1]),
            "r_co": float(popt[2]),
            "chi2_red": float(chi2_red),
            "chi2": float(chi2),
            "dof": int(dof),
            "v_fit": v_fit.tolist(),
            "success": True,
            "message": "ok",
        }

    except Exception as e:
        return {
            "Upsilon": None, "V_co": None, "r_co": None,
            "chi2_red": None, "v_fit": None,
            "success": False, "message": str(e),
        }


def fit_nfw(
    r: np.ndarray,
    v_obs: np.ndarray,
    e_v: np.ndarray,
    l_cum: np.ndarray,
    m_gas_cum: np.ndarray,
) -> dict:
    """拟合 NFW 暗物质晕模型。"""
    p0 = [0.5, 1e11, 10.0]
    try:
        popt, pcov = curve_fit(
            lambda r, U, M, c: v_nfw_model(r, U, M, c, l_cum, m_gas_cum),
            r, v_obs,
            sigma=e_v,
            p0=p0,
            bounds=([0.1, 1e9, 1], [2.0, 1e14, 100]),
            method="trf",
            max_nfev=2000,
        )
        v_fit = v_nfw_model(r, *popt, l_cum, m_gas_cum)
        chi2 = np.sum(((v_obs - v_fit) / e_v)**2)
        dof = len(r) - 3
        return {
            "Upsilon": float(popt[0]),
            "M200": float(popt[1]),
            "c": float(popt[2]),
            "chi2_red": float(chi2 / dof),
            "success": True,
        }
    except Exception as e:
        return {"success": False, "message": str(e)}


def _guess_init(r: np.ndarray, v_obs: np.ndarray) -> list:
    """基于数据特征估算初值。"""
    v_max = np.max(v_obs)
    # V_co 初值: v_max * 0.9 (大部分星系 v_co ~ v_max)
    # r_co 初值: 约外区半径的一半
    r_max = np.max(r)
    return [
        min(0.6, max(0.2, v_max / 200)),   # Upsilon 初值
        v_max * 0.9,                         # V_co 初值
        r_max / 3,                           # r_co 初值
    ]


# ══════════════════════════════════════════════════════════════════════
# 批量拟合
# ══════════════════════════════════════════════════════════════════════

def batch_fit_sparc(data_dir: str, pattern: str = "*_rotmod.dat") -> list:
    """批量拟合 SPARC 目录下所有星系。

    Returns:
        list of dict，每个 dict 包含星系拟合结果
    """
    import glob
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    results = []

    for fpath in files:
        name = os.path.basename(fpath).replace("_rotmod.dat", "")
        print(f"  {name}...", end="", flush=True)

        try:
            d = load_sparc_dat(fpath)

            # 面亮度: 用 v_disk² / (G * r) 反推 (SPARC 中 v_disk 是恒星盘贡献)
            l_cum = (d["v_disk"]**2 * d["r"] / G) if "v_disk" in d else np.zeros_like(d["r"])
            m_gas_cum = compute_m_gas_cum(d["r"], d.get("v_gas", np.zeros_like(d["r"])))

            spum_fit = fit_spum(d["r"], d["v_obs"], d["e_v"], l_cum, m_gas_cum)
            spum_fit["galaxy"] = name

            if spum_fit["success"] and spum_fit["chi2_red"] < 100:
                print(f" chi2={spum_fit['chi2_red']:.2f}")
                results.append(spum_fit)
            else:
                print(f" FAIL ({spum_fit.get('message', 'high chi2')[:40]})")

        except Exception as e:
            print(f" ERROR: {e}")

    return results


def summarize(results: list) -> dict:
    """汇总批量拟合结果。"""
    chi2s = [r["chi2_red"] for r in results if r["success"]]
    vcos = [r["V_co"] for r in results if r["success"]]
    rcos = [r["r_co"] for r in results if r["success"]]
    upss = [r["Upsilon"] for r in results if r["success"]]

    if not chi2s:
        return {"n_valid": 0}

    return {
        "n_total": len(results),
        "n_valid": len(chi2s),
        "chi2_med": float(np.median(chi2s)),
        "chi2_mean": float(np.mean(chi2s)),
        "chi2_q25": float(np.percentile(chi2s, 25)),
        "chi2_q75": float(np.percentile(chi2s, 75)),
        "V_co_med": float(np.median(vcos)),
        "r_co_med": float(np.median(rcos)),
        "Upsilon_med": float(np.median(upss)),
    }


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="SPUM 子图协动星系旋转曲线拟合器 v1.0"
    )
    ap.add_argument("--data", help="单个星系 .dat 文件路径")
    ap.add_argument("--data-dir", help="SPARC 数据目录路径 (批量拟合)")
    ap.add_argument("--nfw", action="store_true", help="同时拟合 NFW 模型比较")
    ap.add_argument("--out", default=None, help="输出 JSON 文件路径")
    ap.add_argument("--ups-min", type=float, default=0.1)
    ap.add_argument("--ups-max", type=float, default=2.0)
    ap.add_argument("--vco-min", type=float, default=20)
    ap.add_argument("--vco-max", type=float, default=400)
    ap.add_argument("--rco-min", type=float, default=0.5)
    ap.add_argument("--rco-max", type=float, default=30)
    args = ap.parse_args()

    bounds = {
        "ups_min": args.ups_min, "ups_max": args.ups_max,
        "vco_min": args.vco_min, "vco_max": args.vco_max,
        "rco_min": args.rco_min, "rco_max": args.rco_max,
    }

    # ── 批量模式 ──
    if args.data_dir:
        print(f"批量拟合: {args.data_dir}")
        results = batch_fit_sparc(args.data_dir)
        summary = summarize(results)
        print(f"\n=== 汇总 ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"summary": summary, "results": results}, f, indent=2)
            print(f"已保存: {args.out}")
        return

    # ── 单星系模式 ──
    if not args.data:
        ap.error("需要 --data 或 --data-dir")

    print(f"拟合: {args.data}")
    d = load_sparc_dat(args.data)

    # 从 SPARC 数据反推 l_cum
    if "v_disk" in d:
        l_cum = d["v_disk"]**2 * d["r"] / G
    else:
        l_cum = np.zeros_like(d["r"])
    m_gas_cum = compute_m_gas_cum(d["r"], d.get("v_gas", np.zeros_like(d["r"])))

    # SPUM 拟合
    fit = fit_spum(d["r"], d["v_obs"], d["e_v"], l_cum, m_gas_cum, bounds)
    print(f"\nSPUM 子图协动 (3参数):")
    if fit["success"]:
        print(f"  Upsilon  = {fit['Upsilon']:.4f}")
        print(f"  V_co     = {fit['V_co']:.2f} km/s")
        print(f"  r_co     = {fit['r_co']:.2f} kpc")
        print(f"  chi2_red = {fit['chi2_red']:.4f}")
    else:
        print(f"  失败: {fit['message']}")

    # NFW 比较
    if args.nfw:
        nfw_fit = fit_nfw(d["r"], d["v_obs"], d["e_v"], l_cum, m_gas_cum)
        print(f"\nNFW 暗物质晕 (3参数):")
        if nfw_fit["success"]:
            print(f"  Upsilon  = {nfw_fit['Upsilon']:.4f}")
            print(f"  M200     = {nfw_fit['M200']:.2e} M_sun")
            print(f"  c        = {nfw_fit['c']:.2f}")
            print(f"  chi2_red = {nfw_fit['chi2_red']:.4f}")
            if fit["success"]:
                delta_chi2 = fit["chi2_red"] - nfw_fit["chi2_red"]
                winner = "SPUM" if fit["chi2_red"] < nfw_fit["chi2_red"] else "NFW"
                print(f"  Δχ²_red  = {delta_chi2:+.4f} (优势方: {winner})")
        else:
            print(f"  失败: {nfw_fit['message']}")

    if args.out:
        out = {"SPUM": fit}
        if args.nfw:
            out["NFW"] = nfw_fit
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\n已保存: {args.out}")


if __name__ == "__main__":
    main()
