import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os
import glob
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ============================================
# 物理常数（单位：kpc, km/s, M_sun）
# ============================================
G = 4.30091e-6          # kpc·(km/s)² / M_sun
c = 299792.458          # km/s

# ============================================
# 1. 数据读取（兼容6列或更多，尝试读取气体面密度）
# ============================================
def load_sparc_rotmod(file_path):
    """
    读取 SPARC rotmod 文件
    返回: r, v_obs, err_v, v_gas, v_disk, v_bulge, sb, sigma_gas
          sigma_gas 可能为 None（如果文件没有该列）
    """
    r, v_obs, err_v, v_gas, v_disk, v_bulge, sb, sigma_gas = [], [], [], [], [], [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:   # 至少7列
                r.append(float(parts[0]))
                v_obs.append(float(parts[1]))
                err_v.append(float(parts[2]))
                v_gas.append(float(parts[3]))
                v_disk.append(float(parts[4]))
                v_bulge.append(float(parts[5]))
                sb.append(float(parts[6]))
                # 如果有第8列（气体面密度），则读取
                if len(parts) >= 8:
                    sigma_gas.append(float(parts[7]))
                else:
                    sigma_gas.append(np.nan)
            else:
                continue
    # 转为numpy数组
    if len(r) == 0:
        return (None, None, None, None, None, None, None, None)
    r = np.array(r)
    v_obs = np.array(v_obs)
    err_v = np.array(err_v)
    v_gas = np.array(v_gas)
    v_disk = np.array(v_disk)
    v_bulge = np.array(v_bulge)
    sb = np.array(sb)
    sigma_gas = np.array(sigma_gas) if sigma_gas else np.full_like(r, np.nan)

    # 数据清洗
    mask = (r > 0) & (v_obs > 0) & (err_v > 0) & np.isfinite(sb) & np.isfinite(v_gas)
    if np.sum(mask) == 0:
        return (None, None, None, None, None, None, None, None)
    r = r[mask]
    v_obs = v_obs[mask]
    err_v = err_v[mask]
    v_gas = v_gas[mask]
    v_disk = v_disk[mask]
    v_bulge = v_bulge[mask]
    sb = sb[mask]
    sigma_gas = sigma_gas[mask] if len(sigma_gas) > 0 else np.full_like(r, np.nan)
    err_v = np.maximum(err_v, 0.5)

    # 如果所有 sigma_gas 都是 nan，则设为 None
    if np.all(np.isnan(sigma_gas)):
        sigma_gas = None
    return (r, v_obs, err_v, v_gas, v_disk, v_bulge, sb, sigma_gas)

# ============================================
# 2. 累积重子质量（优先积分气体面密度）
# ============================================
def cumulative_baryon_mass(r, sb_disk, v_gas, upsilon, sigma_gas=None):
    """
    计算累积重子质量（恒星+气体）
    恒星质量：从面亮度积分
    气体质量：如果提供了 sigma_gas (Msun/pc^2)，则积分；否则用 v_gas²r/G 估算
    """
    r_safe = np.maximum(r, 1e-3)
    dr = np.gradient(r_safe)
    # 恒星质量：面亮度积分，单位转换：面积 kpc^2 -> pc^2 乘 1e6
    dL = sb_disk * (2 * np.pi * r_safe * dr) * 1e6   # L_sun
    L_cum = np.cumsum(dL)
    M_star = upsilon * L_cum

    # 气体质量
    if sigma_gas is not None and not np.all(np.isnan(sigma_gas)):
        # 积分气体面密度（单位 Msun/pc^2）
        dM_gas = sigma_gas * (2 * np.pi * r_safe * dr) * 1e6   # Msun
        M_gas = np.cumsum(dM_gas)
        M_gas = np.minimum(M_gas, 1e11)          # 上限 1e11 Msun
        M_gas = np.nan_to_num(M_gas)
    else:
        # 从气体速度反推（近似）
        v_gas_safe = np.clip(v_gas, 0, 300)
        M_gas = (v_gas_safe**2 * r_safe) / G
        M_gas = np.minimum(M_gas, 1e11)
        M_gas = np.nan_to_num(M_gas)
    return M_star + M_gas

# ============================================
# 3. SPUM 模型函数
# ============================================
def spum_rotation_curve(r, sb_disk, v_gas, upsilon, lambda_grav, V_base, r0_base, beta_light, sigma_gas=None):
    r_safe = np.maximum(r, 1e-3)
    M_bar = cumulative_baryon_mass(r_safe, sb_disk, v_gas, upsilon, sigma_gas)

    # 短程链式引力
    a_grav = (G * M_bar / r_safe**2) * (1 + r_safe/lambda_grav) * np.exp(-r_safe/lambda_grav)
    a_grav = np.nan_to_num(a_grav)

    # 底图协同运动
    v_base = V_base * r_safe / np.sqrt(r_safe**2 + r0_base**2)
    a_base = v_base**2 / r_safe
    a_base = np.nan_to_num(a_base)

    # 光调制（弱效应）
    sb_smooth = gaussian_filter1d(sb_disk, sigma=1)
    dI_dr = np.gradient(sb_smooth, r_safe)
    a_light = beta_light * np.abs(dI_dr)
    a_light = np.nan_to_num(a_light)

    a_total = a_grav + a_base + a_light
    a_total = np.clip(a_total, 0, 1e6)
    return np.sqrt(r_safe * a_total)

# ============================================
# 4. 智能初值估计（根据星系最大速度和气体丰度）
# ============================================
def get_initial_params(r, v_obs, v_gas):
    v_max = np.nanmax(v_obs)
    v_gas_med = np.nanmedian(v_gas)
    # 边界定义
    bounds = ([0.1, 0.5, 5.0, 0.5, 0.0],
              [2.0, 30.0, 400.0, 30.0, 0.1])
    # 气体丰度高或矮星系，采用较小质光比和底图速度
    if v_gas_med > 50 or v_max < 80:
        upsilon_init = 0.3
        lambda_init = 5.0
        V_base_init = max(0.5 * v_max, bounds[0][2] + 1)
    else:
        upsilon_init = 0.6
        lambda_init = 8.0
        V_base_init = max(0.7 * v_max, bounds[0][2] + 1)
    V_base_init = min(V_base_init, bounds[1][2] - 1)
    r0_init = 10.0
    beta_init = 0.005
    return [upsilon_init, lambda_init, V_base_init, r0_init, beta_init], bounds

# ============================================
# 5. 单星系拟合（多策略回退）
# ============================================
def fit_one_galaxy(file_path, output_dir='./SPUM_batch_results', make_plot=True):
    gal = os.path.basename(file_path).replace('_rotmod.dat', '')
    data = load_sparc_rotmod(file_path)
    if data[0] is None:
        print(f"⚠️ {gal}: 数据读取失败")
        return None
    r, v_obs, err_v, v_gas, _, _, sb, sigma_gas = data
    if len(r) < 4:
        print(f"⚠️ {gal}: 有效数据点不足4个，跳过")
        return None

    p0, bounds = get_initial_params(r, v_obs, v_gas)

    # 定义模型（绑定当前星系的 sb, v_gas, sigma_gas）
    def model(r, upsilon, lam, Vb, r0, beta):
        return spum_rotation_curve(r, sb, v_gas, upsilon, lam, Vb, r0, beta, sigma_gas)

    # 多策略拟合
    strategies = [
        (p0, bounds, 5),   # 标准五参数
        (p0[:4] + [0.0], ([0.1,0.5,5.0,0.5], [2.0,30.0,400.0,30.0]), 4),  # 固定beta=0
        ([0.3, 5.0, max(0.5*v_obs.max(), 10.0), 10.0, 0.0],
         ([0.1,0.5,5.0,0.5,0.0], [2.0,30.0,400.0,30.0,0.0]), 5),  # 激进初值
    ]
    success = False
    popt, pcov = None, None
    for params, bnd, nparams in strategies:
        try:
            if nparams == 4:
                def model4(r, upsilon, lam, Vb, r0):
                    return model(r, upsilon, lam, Vb, r0, 0.0)
                popt4, pcov4 = curve_fit(model4, r, v_obs, sigma=err_v,
                                         p0=params, bounds=bnd, maxfev=10000)
                popt = list(popt4) + [0.0]
                pcov = np.zeros((5,5))
                pcov[:4,:4] = pcov4
            else:
                popt, pcov = curve_fit(model, r, v_obs, sigma=err_v,
                                       p0=params, bounds=bnd, maxfev=10000)
            success = True
            break
        except Exception:
            continue
    if not success:
        print(f"❌ {gal}: 所有拟合策略均失败")
        return None

    # 计算拟合优度
    v_fit = model(r, *popt)
    chi2 = np.sum(((v_obs - v_fit) / err_v)**2)
    dof = len(r) - len(popt)
    if dof > 0:
        chi2_red = chi2 / dof
    else:
        chi2_red = np.nan
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*5
    upsilon_fit, lambda_fit, V_base_fit, r0_fit, beta_fit = popt
    upsilon_err, lambda_err, V_base_err, r0_err, beta_err = perr

    # 可选绘图
    if make_plot:
        os.makedirs(output_dir, exist_ok=True)
        # 计算分量
        M_bar = cumulative_baryon_mass(r, sb, v_gas, upsilon_fit, sigma_gas)
        r_safe = np.maximum(r, 1e-3)
        a_grav_only = (G * M_bar / r_safe**2) * (1 + r_safe/lambda_fit) * np.exp(-r_safe/lambda_fit)
        v_grav = np.sqrt(r_safe * np.nan_to_num(a_grav_only))
        v_base = V_base_fit * r_safe / np.sqrt(r_safe**2 + r0_fit**2)
        sb_smooth = gaussian_filter1d(sb, sigma=1)
        dI_dr = np.gradient(sb_smooth, r_safe)
        a_light_only = beta_fit * np.abs(dI_dr)
        v_light = np.sqrt(r_safe * np.nan_to_num(a_light_only))

        plt.figure(figsize=(9,6))
        plt.errorbar(r, v_obs, yerr=err_v, fmt='ko', capsize=2, markersize=3, label='SPARC data')
        plt.plot(r, v_fit, 'r-', lw=2.5, label='SPUM total')
        plt.plot(r, v_grav, 'b--', lw=2, label='Short-range gravity')
        plt.plot(r, v_base, 'g-.', lw=2, label='Base-map co-rotation')
        plt.plot(r, v_light, 'm:', lw=2, label='Photon modulation')
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Rotation velocity (km/s)')
        plt.title(f'{gal}   χ²/DoF = {chi2_red:.2f}')
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{gal}_fit.png'), dpi=120)
        plt.close()

    return {
        'galaxy': gal,
        'chi2_red': chi2_red,
        'n_points': len(r),
        'upsilon': upsilon_fit,
        'upsilon_err': upsilon_err,
        'lambda_grav': lambda_fit,
        'lambda_err': lambda_err,
        'V_base': V_base_fit,
        'V_base_err': V_base_err,
        'r0_base': r0_fit,
        'r0_err': r0_err,
        'beta_light': beta_fit,
        'beta_err': beta_err,
        'has_sigma_gas': sigma_gas is not None
    }

# ============================================
# 6. 批量拟合主函数
# ============================================
def batch_fit(data_dir='.', output_dir='./SPUM_batch_results', make_plots=True):
    files = glob.glob(os.path.join(Rotmod_LTG, '*_rotmod.dat'))
    if not files:
        print(f"错误：在 {data_dir} 中没有找到 *_rotmod.dat 文件")
        return None
    print(f"找到 {len(files)} 个星系文件，开始批量拟合...")
    results = []
    for f in tqdm(files):
        res = fit_one_galaxy(f, output_dir, make_plot=make_plots)
        if res is not None:
            results.append(res)
    if not results:
        print("❌ 没有任何星系拟合成功")
        return None

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'SPUM_batch_results.csv'), index=False)
    # 统计有效结果（chi2_red 有限且 < 100）
    valid = df[np.isfinite(df['chi2_red']) & (df['chi2_red'] < 100)]
    print(f"\n✅ 成功拟合 {len(results)}/{len(files)} 个星系")
    print(f"📊 有效拟合（χ²<100）: {len(valid)} 个")
    if len(valid) > 0:
        print(f"📊 平均 χ²/DoF = {valid['chi2_red'].mean():.2f} ± {valid['chi2_red'].std():.2f}")
        print(f"📊 平均 λ_grav = {valid['lambda_grav'].mean():.2f} ± {valid['lambda_grav'].std():.2f} kpc")
        print(f"📊 平均 V_base = {valid['V_base'].mean():.1f} ± {valid['V_base'].std():.1f} km/s")
        print(f"📊 平均 r0_base = {valid['r0_base'].mean():.2f} ± {valid['r0_base'].std():.2f} kpc")
        print(f"📊 平均 β_light = {valid['beta_light'].mean():.4f} ± {valid['beta_light'].std():.4f}")
    print(f"📁 结果保存至: {output_dir}/SPUM_batch_results.csv")
    return df

# ============================================
# 7. 主程序入口
# ============================================
if __name__ == "__main__":
    # 请修改为你的数据目录（包含所有 *_rotmod.dat 文件）
    DATA_DIR = "./sparc_data"        # 例如：解压后的 Rotmod_LTG 文件夹
    OUTPUT_DIR = "./SPUM_batch_results"

    df = batch_fit(DATA_DIR, OUTPUT_DIR, make_plots=True)
