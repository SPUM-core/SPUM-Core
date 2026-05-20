import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import os
import glob
import pandas as pd
from tqdm import tqdm

# ============================================
# 物理常数（单位：kpc, km/s, M_sun）
# ============================================
G = 4.30091e-6          # kpc·(km/s)² / M_sun
c = 299792.458          # km/s

# ============================================
# 1. 稳健的 SPARC rotmod 数据读取（兼容6列和7列）
# ============================================
def load_sparc_rotmod(file_path):
    """
    读取 SPARC rotmod 文件，返回：
    r, v_obs, err_v, v_gas, v_disk, v_bulge, sb_disk
    """
    r, v_obs, err_v, v_gas, v_disk, sb = [], [], [], [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 6:   # 至少6列
                r.append(float(parts[0]))
                v_obs.append(float(parts[1]))
                err_v.append(float(parts[2]))
                v_gas.append(float(parts[3]))
                v_disk.append(float(parts[4]))
                # 面亮度：如果有第6列则使用，否则设为0（某些文件只有5列？实际最少6列）
                sb.append(float(parts[5]) if len(parts) > 5 else 0.0)
            else:
                continue   # 跳过格式不正确的行
    # 转换为 numpy 数组
    r = np.array(r)
    v_obs = np.array(v_obs)
    err_v = np.array(err_v)
    v_gas = np.array(v_gas)
    v_disk = np.array(v_disk)
    sb = np.array(sb)
    # bulge 速度设为0（大部分星系无 bulge 或已包含在 v_disk 中）
    v_bulge = np.zeros_like(r)

    # 数据清洗：移除无效点（半径>0，速度>0，误差>0，面亮度有限，气体速度有限）
    mask = (r > 0) & (v_obs > 0) & (err_v > 0) & np.isfinite(sb) & np.isfinite(v_gas)
    # 避免误差为0导致除零
    err_v = np.maximum(err_v, 0.5)
    return (r[mask], v_obs[mask], err_v[mask],
            v_gas[mask], v_disk[mask], v_bulge[mask], sb[mask])

# ============================================
# 2. 改进的累积重子质量计算（防爆炸）
# ============================================
def cumulative_baryon_mass(r, sb_disk, v_gas, upsilon):
    """
    返回累积重子质量（恒星+气体），单位 M_sun
    恒星质量：从面亮度积分，单位转换已处理
    气体质量：基于气体速度估算，并加上安全限制
    """
    r_safe = np.maximum(r, 1e-3)   # 避免 r=0
    # 恒星质量积分
    dr = np.gradient(r_safe)
    dL = sb_disk * (2 * np.pi * r_safe * dr) * 1e6   # L_sun
    L_cum = np.cumsum(dL)
    M_star = upsilon * L_cum

    # 气体质量：v_gas² r / G，但加上安全限制
    v_gas_safe = np.clip(v_gas, 0, 300)          # 气体速度不超过300 km/s
    M_gas = (v_gas_safe**2 * r_safe) / G
    M_gas = np.minimum(M_gas, 1e11)              # 最大气体质量限制 1e11 M_sun
    M_gas = np.nan_to_num(M_gas, nan=0.0, posinf=0.0)

    return M_star + M_gas

# ============================================
# 3. SPUM 核心模型（数值稳定版）
# ============================================
def spum_rotation_curve(r, sb_disk, v_gas, upsilon, lambda_grav, V_base, r0_base, beta_light):
    """
    计算 SPUM 旋转速度
    """
    r_safe = np.maximum(r, 1e-3)
    # 重子质量
    M_bar = cumulative_baryon_mass(r_safe, sb_disk, v_gas, upsilon)

    # 1) 短程链式引力
    a_grav = (G * M_bar / r_safe**2) * (1 + r_safe / lambda_grav) * np.exp(-r_safe / lambda_grav)
    a_grav = np.nan_to_num(a_grav, nan=0.0, posinf=0.0, neginf=0.0)

    # 2) 底图协同运动
    v_base = V_base * r_safe / np.sqrt(r_safe**2 + r0_base**2)
    a_base = v_base**2 / r_safe
    a_base = np.nan_to_num(a_base, nan=0.0, posinf=0.0)

    # 3) 光调制（弱效应）
    sb_smooth = gaussian_filter1d(sb_disk, sigma=1)
    dI_dr = np.gradient(sb_smooth, r_safe)
    a_light = beta_light * np.abs(dI_dr)
    a_light = np.nan_to_num(a_light, nan=0.0, posinf=0.0)

    a_total = a_grav + a_base + a_light
    a_total = np.clip(a_total, 0, 1e6)   # 防止负值或过大
    v_tot = np.sqrt(r_safe * a_total)
    return v_tot

# ============================================
# 4. 智能初值估计（基于星系观测特征）
# ============================================
def estimate_initial_params(r, v_obs, v_gas):
    v_max = np.nanmax(v_obs)
    v_gas_med = np.nanmedian(v_gas)
    # 气体丰度高的矮星系采用较低质光比和底图速度
    if v_gas_med > 50 or v_max < 80:
        upsilon_init = 0.3
        lambda_init = 5.0
        V_base_init = 0.5 * v_max
    else:
        upsilon_init = 0.6
        lambda_init = 8.0
        V_base_init = 0.7 * v_max
    r0_init = 10.0
    beta_init = 0.005
    return [upsilon_init, lambda_init, V_base_init, r0_init, beta_init]

# ============================================
# 5. 单星系拟合（带失败回退）
# ============================================
def fit_single_galaxy(file_path, output_dir="./SPUM_fit_results", make_plot=True):
    # 提取星系名
    galaxy_name = os.path.basename(file_path).replace("_rotmod.dat", "")
    os.makedirs(output_dir, exist_ok=True)

    # 读取并清洗数据
    r, v_obs, err_v, v_gas, v_disk, v_bulge, sb = load_sparc_rotmod(file_path)
    if len(r) < 5:
        print(f"⚠️ {galaxy_name}: 有效数据点不足5个，跳过")
        return None

    # 获取初值
    p0 = estimate_initial_params(r, v_obs, v_gas)

    # 定义拟合函数（绑定当前星系的数据）
    def model_func(r, upsilon, lambda_grav, V_base, r0_base, beta_light):
        return spum_rotation_curve(r, sb, v_gas, upsilon, lambda_grav, V_base, r0_base, beta_light)

    # 物理边界（放宽但合理）
    bounds = ([0.1, 1.0, 20.0, 1.0, 0.0],
              [2.0, 30.0, 400.0, 30.0, 0.1])

    # 尝试拟合
    success = False
    popt, pcov = None, None
    try:
        popt, pcov = curve_fit(model_func, r, v_obs, sigma=err_v,
                               p0=p0, bounds=bounds, maxfev=20000, method='trf')
        success = True
    except Exception as e:
        print(f"⚠️ {galaxy_name}: 初次拟合失败 ({str(e)})，尝试回退策略（固定 beta=0）")
        # 回退：固定光调制系数为0
        def model_func_no_beta(r, upsilon, lambda_grav, V_base, r0_base):
            return spum_rotation_curve(r, sb, v_gas, upsilon, lambda_grav, V_base, r0_base, 0.0)
        bounds_no_beta = ([0.1, 1.0, 20.0, 1.0], [2.0, 30.0, 400.0, 30.0])
        p0_no_beta = p0[:4]
        try:
            popt_no_beta, pcov_no_beta = curve_fit(model_func_no_beta, r, v_obs, sigma=err_v,
                                                   p0=p0_no_beta, bounds=bounds_no_beta,
                                                   maxfev=20000, method='trf')
            popt = list(popt_no_beta) + [0.0]   # beta=0
            pcov = np.zeros((5,5))
            pcov[:4,:4] = pcov_no_beta
            success = True
        except Exception as e2:
            print(f"❌ {galaxy_name}: 回退拟合也失败 ({str(e2)})，跳过")
            return None

    if not success:
        return None

    # 计算拟合优度
    v_fit = model_func(r, *popt)
    chi2 = np.sum(((v_obs - v_fit) / err_v)**2)
    dof = len(r) - len(popt)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    # 提取参数及标准差
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*5
    upsilon_fit, lambda_fit, V_base_fit, r0_fit, beta_fit = popt
    upsilon_err, lambda_err, V_base_err, r0_err, beta_err = perr

    # 生成拟合图（可选）
    if make_plot:
        # 计算各分量用于绘图
        M_bar = cumulative_baryon_mass(r, sb, v_gas, upsilon_fit)
        r_safe = np.maximum(r, 1e-3)
        a_grav_only = (G * M_bar / r_safe**2) * (1 + r_safe/lambda_fit) * np.exp(-r_safe/lambda_fit)
        v_grav = np.sqrt(r_safe * np.nan_to_num(a_grav_only))
        v_base = V_base_fit * r_safe / np.sqrt(r_safe**2 + r0_fit**2)
        sb_smooth = gaussian_filter1d(sb, sigma=1)
        dI_dr = np.gradient(sb_smooth, r_safe)
        a_light_only = beta_fit * np.abs(dI_dr)
        v_light = np.sqrt(r_safe * np.nan_to_num(a_light_only))

        plt.figure(figsize=(10, 6))
        plt.errorbar(r, v_obs, yerr=err_v, fmt='ko', capsize=2, markersize=3,
                     label='SPARC data')
        plt.plot(r, v_fit, 'r-', lw=2.5, label='SPUM total')
        plt.plot(r, v_grav, 'b--', lw=2, label='Short-range gravity')
        plt.plot(r, v_base, 'g-.', lw=2, label='Base-map co-rotation')
        plt.plot(r, v_light, 'm:', lw=2, label='Photon modulation')
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Rotation velocity (km/s)')
        plt.title(f'{galaxy_name}  |  χ²/DoF = {chi2_red:.2f}')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{galaxy_name}_fit.png"), dpi=150)
        plt.close()

    # 返回结果字典
    return {
        'galaxy': galaxy_name,
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
        'beta_err': beta_err
    }

# ============================================
# 6. 批量拟合所有星系
# ============================================
def batch_fit(data_dir='.', output_dir='./SPUM_fit_results'):
    # 查找所有 rotmod 文件
    rotmod_files = glob.glob(os.path.join(data_dir, '*_rotmod.dat'))
    if not rotmod_files:
        print(f"错误：在目录 {data_dir} 中未找到任何 *_rotmod.dat 文件")
        return None

    print(f"找到 {len(rotmod_files)} 个星系文件，开始批量拟合...")
    results = []
    for fpath in tqdm(rotmod_files, desc="拟合进度"):
        res = fit_single_galaxy(fpath, output_dir, make_plot=True)
        if res is not None:
            results.append(res)

    # 保存结果汇总
    df = pd.DataFrame(results)
    if len(df) > 0:
        csv_path = os.path.join(output_dir, 'SPUM_batch_results.csv')
        df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"\n✅ 成功拟合 {len(df)} / {len(rotmod_files)} 个星系")
        print(f"📊 平均约化卡方: {df['chi2_red'].mean():.2f} ± {df['chi2_red'].std():.2f}")
        print(f"📊 平均引力链长 λ: {df['lambda_grav'].mean():.2f} ± {df['lambda_grav'].std():.2f} kpc")
        print(f"📊 平均底图速度 V_base: {df['V_base'].mean():.1f} ± {df['V_base'].std():.1f} km/s")
        print(f"📊 平均底图尺度 r0: {df['r0_base'].mean():.2f} ± {df['r0_base'].std():.2f} kpc")
        print(f"📊 平均光调制系数 β: {df['beta_light'].mean():.4f} ± {df['beta_light'].std():.4f}")
        print(f"📁 结果已保存至: {csv_path}")
        print(f"📁 拟合图片保存在: {output_dir}")
    else:
        print("❌ 没有成功拟合任何星系")

    return df

# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    # 请将 data_dir 修改为你的 rotmod 文件所在目录
    DATA_DIR = "./sparc_data"      # 例如，解压后的 Rotmod_LTG 文件夹
    OUTPUT_DIR = "./SPUM_batch_results"

    # 运行批量拟合
    results_df = batch_fit(DATA_DIR, OUTPUT_DIR)
