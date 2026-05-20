import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os
import glob
from tqdm import tqdm

# ============================================
# 物理常数（单位：kpc, km/s, M_sun）
# ============================================
G = 4.30091e-6          # kpc·(km/s)² / M_sun
# 临界密度（H0=70 km/s/Mpc）: rho_crit = 3 H0^2/(8πG) = 1.36e2 M_sun/kpc³
rho_crit = 1.36e2

# ============================================
# 数据读取函数（复用之前的）
# ============================================
def load_rotmod(file_path):
    r, v_obs, err_v, v_gas, v_disk, v_bulge, sb, sigma_gas = [], [], [], [], [], [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) >= 7:
                r.append(float(parts[0]))
                v_obs.append(float(parts[1]))
                err_v.append(float(parts[2]))
                v_gas.append(float(parts[3]))
                v_disk.append(float(parts[4]))
                v_bulge.append(float(parts[5]))
                sb.append(float(parts[6]))
                if len(parts) >= 8:
                    sigma_gas.append(float(parts[7]))
                else:
                    sigma_gas.append(np.nan)
    r = np.array(r)
    v_obs = np.array(v_obs)
    err_v = np.array(err_v)
    v_gas = np.array(v_gas)
    sb = np.array(sb)
    sigma_gas = np.array(sigma_gas)
    # 清洗
    mask = (r > 0) & (v_obs > 0) & (err_v > 0) & np.isfinite(sb) & np.isfinite(v_gas)
    r, v_obs, err_v, v_gas, sb = r[mask], v_obs[mask], err_v[mask], v_gas[mask], sb[mask]
    sigma_gas = sigma_gas[mask] if len(sigma_gas) > 0 else np.full_like(r, np.nan)
    err_v = np.maximum(err_v, 0.5)
    if np.all(np.isnan(sigma_gas)):
        sigma_gas = None
    return r, v_obs, err_v, v_gas, sb, sigma_gas

# ============================================
# 重子质量计算（恒星+气体）
# ============================================
def cumulative_baryon_mass(r, sb, v_gas, upsilon, sigma_gas=None):
    r_safe = np.maximum(r, 1e-3)
    dr = np.gradient(r_safe)
    dL = sb * (2 * np.pi * r_safe * dr) * 1e6
    M_star = upsilon * np.cumsum(dL)
    if sigma_gas is not None and not np.all(np.isnan(sigma_gas)):
        dM_gas = sigma_gas * (2 * np.pi * r_safe * dr) * 1e6
        M_gas = np.cumsum(dM_gas)
    else:
        v_gas_safe = np.clip(v_gas, 0, 300)
        M_gas = (v_gas_safe**2 * r_safe) / G
    M_gas = np.minimum(M_gas, 1e11)
    M_gas = np.nan_to_num(M_gas)
    return M_star + M_gas

# ============================================
# NFW 暗物质晕速度平方
# ============================================
def v_nfw2(r, M200, c):
    """
    r: kpc
    M200: 晕质量 (M_sun) 以内，r200 由 M200 = (4π/3)*200*rho_crit*r200^3
    c: 浓度参数
    """
    r200 = (M200 / ((4*np.pi/3)*200*rho_crit))**(1/3)
    r_s = r200 / c
    x = r / r_s
    # NFW 质量分布 M(r) = M200 * [ln(1+x) - x/(1+x)] / [ln(1+c) - c/(1+c)]
    factor = np.log(1+c) - c/(1+c)
    M_r = M200 * (np.log(1+x) - x/(1+x)) / factor
    v2 = G * M_r / np.maximum(r, 1e-3)
    return v2

# ============================================
# 重子速度平方
# ============================================
def v_baryon2(r, sb, v_gas, upsilon, sigma_gas):
    M_bar = cumulative_baryon_mass(r, sb, v_gas, upsilon, sigma_gas)
    return G * M_bar / np.maximum(r, 1e-3)

# ============================================
# NFW 模型总旋转速度
# ============================================
def nfw_rotation_curve(r, sb, v_gas, upsilon, M200, c, sigma_gas=None):
    v2_bar = v_baryon2(r, sb, v_gas, upsilon, sigma_gas)
    v2_nfw = v_nfw2(r, M200, c)
    return np.sqrt(v2_bar + v2_nfw)

# ============================================
# 单星系 NFW 拟合
# ============================================
def fit_nfw_galaxy(file_path):
    r, v_obs, err_v, v_gas, sb, sigma_gas = load_rotmod(file_path)
    if len(r) < 5:
        return None
    def model(r, upsilon, M200, c):
        return nfw_rotation_curve(r, sb, v_gas, upsilon, M200, c, sigma_gas)
    v_max = np.max(v_obs)
    r_max = r[-1]
    M200_guess = 10 * (v_max**2 * r_max) / G
    c_guess = 10.0
    p0 = [0.5, M200_guess, c_guess]
    bounds = ([0.1, 1e8, 2], [2.0, 1e13, 30])
    try:
        popt, pcov = curve_fit(model, r, v_obs, sigma=err_v, p0=p0, bounds=bounds, maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        v_fit = model(r, *popt)
        chi2 = np.sum(((v_obs - v_fit) / err_v)**2)
        dof = len(r) - 3
        chi2_red = chi2 / dof if dof>0 else np.nan
        return {
            'upsilon': popt[0], 'upsilon_err': perr[0],
            'M200': popt[1], 'M200_err': perr[1],
            'c': popt[2], 'c_err': perr[2],
            'chi2_red': chi2_red, 'n_points': len(r), 'chi2': chi2
        }
    except Exception as e:
        return None

# ============================================
# 批量对比主函数
# ============================================
def compare_models(data_dir='.', spum_csv_path='SPUM_batch_results.csv'):
    # 读取 SPUM 拟合结果
    if not os.path.exists(spum_csv_path):
        print(f"错误：找不到 SPUM 结果文件 {spum_csv_path}")
        return None
    spum_df = pd.read_csv(spum_csv_path)
    spum_df = spum_df[np.isfinite(spum_df['chi2_red']) & (spum_df['chi2_red'] > 0)]
    if len(spum_df) == 0:
        print("SPUM 结果文件中没有有效拟合")
        return None

    # 获取所有 rotmod 文件
    files = glob.glob(os.path.join(data_dir, '*_rotmod.dat'))
    print(f"找到 {len(files)} 个 rotmod 文件")

    results = []
    for f in tqdm(files):
        gal = os.path.basename(f).replace('_rotmod.dat', '')
        spum_row = spum_df[spum_df['galaxy'] == gal]
        if len(spum_row) == 0:
            continue
        # 读取数据获取点数
        r, v_obs, err_v, v_gas, sb, sigma_gas = load_rotmod(f)
        n_points = len(r)
        if n_points < 5:
            continue
        # SPUM 的卡方和参数数
        chi2_spum_red = spum_row['chi2_red'].values[0]
        chi2_spum = chi2_spum_red * (n_points - 5) if n_points > 5 else np.nan
        if np.isnan(chi2_spum):
            continue
        k_spum = 5
        # NFW 拟合
        nfw_res = fit_nfw_galaxy(f)
        if nfw_res is None:
            continue
        chi2_nfw = nfw_res['chi2']
        k_nfw = 3
        # 计算 AIC 和 BIC
        aic_spum = 2*k_spum + chi2_spum
        bic_spum = k_spum * np.log(n_points) + chi2_spum
        aic_nfw = 2*k_nfw + chi2_nfw
        bic_nfw = k_nfw * np.log(n_points) + chi2_nfw
        delta_aic = aic_spum - aic_nfw
        delta_bic = bic_spum - bic_nfw
        results.append({
            'galaxy': gal,
            'n_points': n_points,
            'chi2_spum_red': chi2_spum_red,
            'chi2_nfw_red': nfw_res['chi2_red'],
            'aic_spum': aic_spum,
            'aic_nfw': aic_nfw,
            'bic_spum': bic_spum,
            'bic_nfw': bic_nfw,
            'delta_aic': delta_aic,
            'delta_bic': delta_bic,
            'spum_better_aic': delta_aic < 0,
            'spum_better_bic': delta_bic < 0,
            'nfw_params': f"{nfw_res['M200']:.2e},{nfw_res['c']:.2f}"
        })

    if len(results) == 0:
        print("没有成功比较的星系")
        return None

    df = pd.DataFrame(results)
    df.to_csv('SPUM_vs_NFW_comparison.csv', index=False)

    # 统计
    n_total = len(df)
    n_aic = df['spum_better_aic'].sum()
    n_bic = df['spum_better_bic'].sum()
    print(f"\n对比完成，共 {n_total} 个星系成功比较")
    print(f"SPUM 在 AIC 上优于 NFW 的星系数: {n_aic} ({100*n_aic/n_total:.1f}%)")
    print(f"SPUM 在 BIC 上优于 NFW 的星系数: {n_bic} ({100*n_bic/n_total:.1f}%)")
    print(f"平均 ΔAIC = {df['delta_aic'].mean():.2f} (负值表示 SPUM 更优)")
    print(f"平均 ΔBIC = {df['delta_bic'].mean():.2f}")
    return df

if __name__ == "__main__":
    # 设置路径：数据文件在当前目录（sparc_data），SPUM 结果在上一级目录
    DATA_DIR = "."
    SPUM_CSV = "../SPUM_batch_results.csv"
    compare_models(DATA_DIR, SPUM_CSV)
