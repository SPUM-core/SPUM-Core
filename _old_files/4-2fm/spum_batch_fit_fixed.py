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
# 物理常数
# ============================================
G = 4.30091e-6          # kpc·(km/s)² / M_sun
c = 299792.458          # km/s

# ============================================
# 1. 稳健数据读取（修复索引错误）
# ============================================
def load_sparc_rotmod(file_path):
    r, v_obs, err_v, v_gas, v_disk, sb = [], [], [], [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6:
                r.append(float(parts[0]))
                v_obs.append(float(parts[1]))
                err_v.append(float(parts[2]))
                v_gas.append(float(parts[3]))
                v_disk.append(float(parts[4]))
                sb.append(float(parts[5]) if len(parts) > 5 else 0.0)
    # 转换为 numpy 数组
    r = np.array(r)
    v_obs = np.array(v_obs)
    err_v = np.array(err_v)
    v_gas = np.array(v_gas)
    v_disk = np.array(v_disk)
    sb = np.array(sb)
    v_bulge = np.zeros_like(r)   # bulge 速度设为0

    # 数据清洗
    mask = (r > 0) & (v_obs > 0) & (err_v > 0) & np.isfinite(sb) & np.isfinite(v_gas)
    if np.sum(mask) == 0:
        return None, None, None, None, None, None, None
    r = r[mask]
    v_obs = v_obs[mask]
    err_v = err_v[mask]
    v_gas = v_gas[mask]
    v_disk = v_disk[mask]
    sb = sb[mask]
    v_bulge = v_bulge[mask]
    # 防止误差为0
    err_v = np.maximum(err_v, 0.5)
    return r, v_obs, err_v, v_gas, v_disk, v_bulge, sb

# ============================================
# 2. 累积重子质量（安全）
# ============================================
def cumulative_baryon_mass(r, sb_disk, v_gas, upsilon):
    r_safe = np.maximum(r, 1e-3)
    dr = np.gradient(r_safe)
    dL = sb_disk * (2 * np.pi * r_safe * dr) * 1e6
    M_star = upsilon * np.cumsum(dL)
    v_gas_safe = np.clip(v_gas, 0, 300)
    M_gas = (v_gas_safe**2 * r_safe) / G
    M_gas = np.minimum(M_gas, 1e11)
    M_gas = np.nan_to_num(M_gas)
    return M_star + M_gas

# ============================================
# 3. SPUM 模型（数值稳定）
# ============================================
def spum_rotation_curve(r, sb_disk, v_gas, upsilon, lambda_grav, V_base, r0_base, beta_light):
    r_safe = np.maximum(r, 1e-3)
    M_bar = cumulative_baryon_mass(r_safe, sb_disk, v_gas, upsilon)
    a_grav = (G * M_bar / r_safe**2) * (1 + r_safe/lambda_grav) * np.exp(-r_safe/lambda_grav)
    a_grav = np.nan_to_num(a_grav, nan=0.0, posinf=0.0)
    v_base = V_base * r_safe / np.sqrt(r_safe**2 + r0_base**2)
    a_base = v_base**2 / r_safe
    a_base = np.nan_to_num(a_base)
    sb_smooth = gaussian_filter1d(sb_disk, sigma=1)
    dI_dr = np.gradient(sb_smooth, r_safe)
    a_light = beta_light * np.abs(dI_dr)
    a_light = np.nan_to_num(a_light)
    a_total = a_grav + a_base + a_light
    a_total = np.clip(a_total, 0, 1e6)
    return np.sqrt(r_safe * a_total)

# ============================================
# 4. 智能初值（确保在边界内）
# ============================================
def get_initial_params(r, v_obs, v_gas):
    v_max = np.nanmax(v_obs)
    v_gas_med = np.nanmedian(v_gas)
    # 边界定义
    bounds = ([0.1, 0.5, 5.0, 0.5, 0.0],
              [2.0, 30.0, 400.0, 30.0, 0.1])
    # 根据气体丰度和最大速度调整初值
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
# 5. 单星系拟合（带多次回退）
# ============================================
def fit_one_galaxy(file_path, output_dir='./results', make_plot=True):
    gal = os.path.basename(file_path).replace('_rotmod.dat', '')
    r, v_obs, err_v, v_gas, _, _, sb = load_sparc_rotmod(file_path)
    if r is None or len(r) < 4:
        print(f"⚠️ {gal}: 有效数据点不足4个，跳过")
        return None

    p0, bounds = get_initial_params(r, v_obs, v_gas)

    def model(r, upsilon, lam, Vb, r0, beta):
        return spum_rotation_curve(r, sb, v_gas, upsilon, lam, Vb, r0, beta)

    # 多策略尝试
    strategies = [
        (p0, bounds, 5),
        (p0[:4] + [0.0], ([0.1,0.5,5.0,0.5], [2.0,30.0,400.0,30.0]), 4),
        ([0.3, 5.0, max(0.5*v_obs.max(), 10.0), 10.0, 0.0],
         ([0.1,0.5,5.0,0.5,0.0], [2.0,30.0,400.0,30.0,0.0]), 5),
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
        except Exception as e:
            continue
    if not success:
        print(f"❌ {gal}: 所有拟合策略均失败")
        return None

    v_fit = model(r, *popt)
    chi2 = np.sum(((v_obs - v_fit) / err_v)**2)
    dof = len(r) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else np.inf
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*5

    if make_plot:
        os.makedirs(output_dir, exist_ok=True)
        M_bar = cumulative_baryon_mass(r, sb, v_gas, popt[0])
        r_safe = np.maximum(r, 1e-3)
        a_grav = (G * M_bar / r_safe**2) * (1 + r_safe/popt[1]) * np.exp(-r_safe/popt[1])
        v_grav = np.sqrt(r_safe * np.nan_to_num(a_grav))
        v_base = popt[2] * r_safe / np.sqrt(r_safe**2 + popt[3]**2)
        sb_smooth = gaussian_filter1d(sb, sigma=1)
        a_light = popt[4] * np.abs(np.gradient(sb_smooth, r_safe))
        v_light = np.sqrt(r_safe * np.nan_to_num(a_light))

        plt.figure(figsize=(9,6))
        plt.errorbar(r, v_obs, yerr=err_v, fmt='ko', capsize=2, markersize=3, label='SPARC')
        plt.plot(r, v_fit, 'r-', lw=2.5, label='SPUM total')
        plt.plot(r, v_grav, 'b--', lw=2, label='Short-range gravity')
        plt.plot(r, v_base, 'g-.', lw=2, label='Base-map co-rotation')
        plt.plot(r, v_light, 'm:', lw=2, label='Photon modulation')
        plt.xlabel('Radius (kpc)')
        plt.ylabel('V (km/s)')
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
        'upsilon': popt[0], 'upsilon_err': perr[0],
        'lambda_grav': popt[1], 'lambda_err': perr[1],
        'V_base': popt[2], 'V_base_err': perr[2],
        'r0_base': popt[3], 'r0_err': perr[3],
        'beta_light': popt[4], 'beta_err': perr[4],
    }

# ============================================
# 6. 批量拟合主函数
# ============================================
def batch_fit(data_dir='.', output_dir='./SPUM_batch_results'):
    files = glob.glob(os.path.join(data_dir, '*_rotmod.dat'))
    if not files:
        print(f"错误：在 {data_dir} 中没有找到 *_rotmod.dat 文件")
        return
    print(f"找到 {len(files)} 个星系文件，开始批量拟合...")
    results = []
    for f in tqdm(files):
        res = fit_one_galaxy(f, output_dir, make_plot=True)
        if res:
            results.append(res)
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, 'SPUM_batch_results.csv'), index=False)
        print(f"\n✅ 成功拟合 {len(results)}/{len(files)} 个星系")
        print(f"📊 平均 χ²/DoF = {df['chi2_red'].mean():.2f} ± {df['chi2_red'].std():.2f}")
        print(f"📊 平均 λ_grav = {df['lambda_grav'].mean():.2f} ± {df['lambda_grav'].std():.2f} kpc")
        print(f"📊 平均 V_base = {df['V_base'].mean():.1f} ± {df['V_base'].std():.1f} km/s")
        print(f"📁 结果保存在: {output_dir}")
    else:
        print("❌ 没有任何星系拟合成功，请检查数据目录和格式")

# ============================================
# 运行
# ============================================
if __name__ == "__main__":
    DATA_DIR = "./sparc_data"      # 修改为你的数据目录
    OUTPUT_DIR = "./SPUM_batch_results"
    batch_fit(DATA_DIR, OUTPUT_DIR)
