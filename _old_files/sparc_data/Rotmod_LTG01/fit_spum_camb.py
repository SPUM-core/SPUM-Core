import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# ============================================
# 1. 读取真实数据（NGC3198）
# ============================================
def load_sparc_rotmod(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                data.append([float(x) for x in parts[:6]])
    data = np.array(data)
    r = data[:, 0]
    v_obs = data[:, 1]
    err_v = data[:, 2]
    v_gas = data[:, 3]
    v_disk = data[:, 4]
    sb_disk = data[:, 5]
    return r, v_obs, err_v, v_gas, sb_disk

# 修改文件名
filename = "NGC3198_rotmod.dat"
r, v_obs, err_v, v_gas, sb_disk = load_sparc_rotmod(filename)

# 常数
G = 4.30091e-6          # kpc (km/s)^2 / M_sun
c = 299792.458          # km/s

# ============================================
# 2. 累积质量函数（同前）
# ============================================
def cumulative_stellar_mass(r, sb, upsilon):
    dr = np.gradient(r)
    dL = sb * (2 * np.pi * r * dr) * 1e6
    L_cum = np.cumsum(dL)
    M_star_cum = upsilon * L_cum
    return M_star_cum

def cumulative_gas_mass(r, v_gas):
    M_gas_cum = (v_gas**2 * r) / G
    M_gas_cum = np.minimum(M_gas_cum, 1e11)
    return M_gas_cum

# ============================================
# 3. SPUM 模型（与之前相同）
# ============================================
def spum_rotation_curve(r, sb, v_gas, upsilon, lambda_grav, V_base, r0_base, beta_light):
    M_star = cumulative_stellar_mass(r, sb, upsilon)
    M_gas = cumulative_gas_mass(r, v_gas)
    M_bar = M_star + M_gas
    
    r_safe = np.where(r == 0, 1e-6, r)
    a_grav = (G * M_bar / r_safe**2) * (1 + r_safe/lambda_grav) * np.exp(-r_safe/lambda_grav)
    a_grav = np.nan_to_num(a_grav, nan=0.0, posinf=0.0)
    
    v_base = V_base * r / np.sqrt(r**2 + r0_base**2)
    a_base = v_base**2 / r_safe
    a_base = np.nan_to_num(a_base, nan=0.0)
    
    sb_smooth = gaussian_filter1d(sb, sigma=1)
    dI_dr = np.gradient(sb_smooth, r)
    a_light = beta_light * np.abs(dI_dr)
    a_light = np.nan_to_num(a_light, nan=0.0)
    
    a_total = a_grav + a_base + a_light
    a_total = np.maximum(a_total, 0)
    v_tot = np.sqrt(r_safe * a_total)
    return v_tot

# ============================================
# 4. 拟合（调整初值和边界以适应 NGC3198）
# ============================================
def fit_spum(r, v_obs, err_v, sb, v_gas):
    def model_func(r, upsilon, lambda_grav, V_base, r0_base, beta_light):
        return spum_rotation_curve(r, sb, v_gas, upsilon, lambda_grav, V_base, r0_base, beta_light)
    
    # NGC3198 典型参数初值
    upsilon_init = 0.5          # M/L
    lambda_init = 8.0           # kpc (链式传递尺度)
    V_base_init = 100.0         # km/s (外区平盘速度约150，底图贡献一部分)
    r0_base_init = 10.0         # kpc
    beta_init = 0.01            # 光调制系数
    
    p0 = [upsilon_init, lambda_init, V_base_init, r0_base_init, beta_init]
    
    # 边界放宽
    bounds = ([0.2, 2.0, 50.0, 2.0, 0.0],
              [1.5, 30.0, 200.0, 30.0, 0.5])
    
    try:
        popt, pcov = curve_fit(model_func, r, v_obs, sigma=err_v, p0=p0, bounds=bounds, maxfev=10000)
        perr = np.sqrt(np.diag(pcov))
        success = True
    except Exception as e:
        print(f"拟合失败: {e}")
        popt = p0
        perr = [np.nan]*5
        success = False
    
    return popt, perr, success

# ============================================
# 5. 执行拟合
# ============================================
popt, perr, success = fit_spum(r, v_obs, err_v, sb_disk, v_gas)

if success:
    upsilon_fit, lambda_fit, V_base_fit, r0_fit, beta_fit = popt
    print("="*60)
    print("SPUM 拟合结果 (NGC3198)")
    print("="*60)
    print(f"质量光度比 M/L     = {upsilon_fit:.3f} ± {perr[0]:.3f}")
    print(f"引力链长 λ_grav    = {lambda_fit:.2f} ± {perr[1]:.2f} kpc")
    print(f"底图渐近速度 V_base= {V_base_fit:.1f} ± {perr[2]:.1f} km/s")
    print(f"底图特征尺度 r0    = {r0_fit:.2f} ± {perr[3]:.2f} kpc")
    print(f"光调制系数 β      = {beta_fit:.4f} ± {perr[4]:.4f}")
    
    # 计算拟合曲线和分量
    v_fit = spum_rotation_curve(r, sb_disk, v_gas, upsilon_fit, lambda_fit, V_base_fit, r0_fit, beta_fit)
    
    # 分量
    M_star = cumulative_stellar_mass(r, sb_disk, upsilon_fit)
    M_gas = cumulative_gas_mass(r, v_gas)
    M_bar = M_star + M_gas
    r_safe = np.where(r == 0, 1e-6, r)
    a_grav_only = (G * M_bar / r_safe**2) * (1 + r_safe/lambda_fit) * np.exp(-r_safe/lambda_fit)
    a_grav_only = np.nan_to_num(a_grav_only)
    v_grav = np.sqrt(r_safe * a_grav_only)
    
    v_base = V_base_fit * r / np.sqrt(r**2 + r0_fit**2)
    
    sb_smooth = gaussian_filter1d(sb_disk, sigma=1)
    dI_dr = np.gradient(sb_smooth, r)
    a_light_only = beta_fit * np.abs(dI_dr)
    v_light = np.sqrt(r_safe * a_light_only)
    
    chi2 = np.sum(((v_obs - v_fit) / err_v)**2)
    dof = len(r) - 5
    chi2_red = chi2 / dof if dof > 0 else np.nan
    print(f"约化卡方 χ²/DoF   = {chi2_red:.2f}")
    print("="*60)
    
    # 绘图
    plt.figure(figsize=(10, 7))
    plt.errorbar(r, v_obs, yerr=err_v, fmt='ko', capsize=2, label='观测数据 (SPARC)', zorder=3)
    plt.plot(r, v_fit, 'r-', lw=2.5, label='SPUM 总拟合', zorder=5)
    plt.plot(r, v_grav, 'b--', lw=2, label='短程引力 (链式)', zorder=4)
    plt.plot(r, v_base, 'g-.', lw=2, label='底图协同运动', zorder=4)
    plt.plot(r, v_light, 'm:', lw=2, label='光调制效应', zorder=4)
    
    plt.xlabel('Radius R (kpc)')
    plt.ylabel('Rotation Velocity V (km/s)')
    plt.title(f'SPUM Rotation Curve Fit | NGC3198')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('NGC3198_SPUM_fit.png', dpi=150)
    plt.show()
else:
    print("拟合失败，请检查数据或调整初值/边界。")
