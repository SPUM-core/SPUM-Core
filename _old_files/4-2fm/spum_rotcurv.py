import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# ============================================
# 物理常数与单位
# ============================================
G = 4.30091e-6          # kpc * (km/s)^2 / M_sun
c = 299792.458          # km/s

# ============================================
# 数据加载函数（请根据实际文件格式修改）
# ============================================
def load_sparc_data(filename):
    """
    加载 SPARC 数据文件，格式：半径(kpc) 速度(km/s) 误差(km/s) 面亮度(Lsun/pc^2) 气体速度(km/s) ...
    返回：r, v_obs, err_v, I_band, v_gas
    """
    # 示例：从文本文件读取
    data = np.genfromtxt(filename, comments='#')
    # 假设列索引：0:r, 1:v_obs, 2:err, 3:I_36, 4:v_gas
    r = data[:, 0]
    v_obs = data[:, 1]
    err_v = data[:, 2]
    I_36 = data[:, 3]
    v_gas = data[:, 4]
    # 去除无效点
    mask = (r > 0) & (v_obs > 0) & (err_v > 0) & (I_36 > 0) & (v_gas > 0)
    return r[mask], v_obs[mask], err_v[mask], I_36[mask], v_gas[mask]

# 如果没有真实文件，可以生成模拟数据（基于 NGC3198 典型参数）
def generate_mock_data():
    # 典型参数
    Mstar = 3.8279e10   # M_sun
    h = 5.84            # kpc
    Vflat = 150.0       # km/s
    r = np.linspace(0.5, 30, 50)
    # 模拟面亮度
    I_36 = Mstar / (2 * np.pi * h**2) * np.exp(-r/h)  # Lsun/pc^2
    # 模拟气体速度（简单模型）
    v_gas = 25 * np.ones_like(r)
    # 模拟观测速度（真实星系旋转曲线）
    v_obs = Vflat * (1 - np.exp(-r/h)) + np.random.normal(0, 3, size=len(r))
    err_v = np.full_like(r, 3.0)
    return r, v_obs, err_v, I_36, v_gas

# ============================================
# 重子质量计算（累积）
# ============================================
def cumulative_baryon_mass(r, I_36, v_gas, upsilon=0.5):
    """
    计算累积重子质量
    r: kpc
    I_36: 面亮度 (Lsun/pc^2)
    v_gas: 气体速度 (km/s)
    upsilon: 恒星质量光度比 (M_sun/L_sun)
    返回 M_star_cum, M_gas_cum, M_bar_cum
    """
    # 1. 恒星质量：从面亮度积分
    dr = np.gradient(r)
    # 注意：I_36 单位 Lsun/pc^2，面积元 2πr dr 单位 kpc^2，需转换为 pc^2: 1 kpc^2 = 1e6 pc^2
    dL = I_36 * (2 * np.pi * r * dr) * 1e6   # Lsun
    L_cum = np.cumsum(dL)
    M_star_cum = upsilon * L_cum              # M_sun
    
    # 2. 气体质量：从气体速度反推（近似，实际应使用面密度积分）
    # 假设气体分布也是盘状，采用简化：M_gas_cum = v_gas^2 * r / G
    M_gas_cum = (v_gas**2 * r) / G            # M_sun
    # 避免极端值
    M_gas_cum = np.minimum(M_gas_cum, 1e11)
    
    M_bar_cum = M_star_cum + M_gas_cum
    return M_star_cum, M_gas_cum, M_bar_cum

# ============================================
# SPUM 模型函数
# ============================================
def spum_rotation_curve(r, M_bar_cum, I_36, upsilon, lambda_grav, V_base, r0_base, beta_light):
    """
    计算 SPUM 旋转曲线
    参数:
        r: 半径数组 (kpc)
        M_bar_cum: 累积重子质量 (M_sun)
        I_36: 面亮度 (Lsun/pc^2)
        upsilon: 恒星质量光度比 (固定，由拟合给出)
        lambda_grav: 引力链式传递特征长度 (kpc)
        V_base: 底图协同运动的渐近速度 (km/s)
        r0_base: 底图特征尺度 (kpc)
        beta_light: 光调制耦合系数 (km²/s²/kpc / (Lsun/pc^2/kpc))
    返回:
        v_tot: 总旋转速度 (km/s)
    """
    # 1. 短程引力加速度 (链式传递)
    # 注意：这里使用累积质量直接计算，忽略了卷积细节，但近似可用
    a_grav = (G * M_bar_cum / r**2) * (1 + r/lambda_grav) * np.exp(-r/lambda_grav)
    # 处理 r=0 避免除零
    a_grav = np.nan_to_num(a_grav, nan=0.0, posinf=0.0)
    
    # 2. 底图协同运动加速度 (离心加速度，但此处直接作为向心加速度？注意符号)
    # 底图贡献是向心的，即 a_base = v_base^2 / r，其中 v_base = V_base * r / sqrt(r^2 + r0_base^2)
    v_base = V_base * r / np.sqrt(r**2 + r0_base**2)
    a_base = v_base**2 / r
    a_base = np.nan_to_num(a_base, nan=0.0)
    
    # 3. 光调制加速度：正比于面亮度梯度的绝对值
    # 计算面亮度梯度
    dI_dr = np.gradient(I_36, r)
    # 取绝对值并平滑
    a_light = beta_light * np.abs(dI_dr)
    a_light = gaussian_filter1d(a_light, sigma=1)  # 轻度平滑
    # 物理上，光调制产生的加速度应该是向内的，所以取正（因为 dI/dr 通常负，绝对值后为正）
    
    # 总加速度
    a_total = a_grav + a_base + a_light
    # 防止负值
    a_total = np.maximum(a_total, 0)
    
    # 旋转速度
    v_tot = np.sqrt(r * a_total)
    return v_tot

# ============================================
# 拟合封装
# ============================================
def fit_spum(r, v_obs, err_v, I_36, v_gas):
    # 先计算重子累积质量（需要 upsilon，但 upsilon 也是拟合参数，所以先给初值，在目标函数中重新计算）
    # 将 upsilon 作为拟合参数之一
    def model_func(r, upsilon, lambda_grav, V_base, r0_base, beta_light):
        # 计算累积重子质量
        _, _, M_bar = cumulative_baryon_mass(r, I_36, v_gas, upsilon)
        v_model = spum_rotation_curve(r, M_bar, I_36, upsilon, lambda_grav, V_base, r0_base, beta_light)
        return v_model
    
    # 初始猜测值
    p0 = [0.5,        # upsilon
          10.0,       # lambda_grav (kpc)
          120.0,      # V_base (km/s)
          10.0,       # r0_base (kpc)
          0.01]       # beta_light (适当量级)
    
    # 边界
    bounds = ([0.1, 1.0, 50.0, 1.0, 0.0],
              [2.0, 50.0, 200.0, 50.0, 0.5])
    
    # 拟合
    popt, pcov = curve_fit(model_func, r, v_obs, sigma=err_v, p0=p0, bounds=bounds, maxfev=5000)
    perr = np.sqrt(np.diag(pcov))
    
    # 计算拟合曲线和分量
    upsilon_fit, lambda_fit, V_base_fit, r0_fit, beta_fit = popt
    _, _, M_bar_fit = cumulative_baryon_mass(r, I_36, v_gas, upsilon_fit)
    v_fit = spum_rotation_curve(r, M_bar_fit, I_36, upsilon_fit, lambda_fit, V_base_fit, r0_fit, beta_fit)
    
    # 分解各分量
    a_grav = (G * M_bar_fit / r**2) * (1 + r/lambda_fit) * np.exp(-r/lambda_fit)
    a_grav = np.nan_to_num(a_grav)
    v_grav = np.sqrt(r * a_grav)
    v_base = V_base_fit * r / np.sqrt(r**2 + r0_fit**2)
    a_light = beta_fit * np.abs(np.gradient(I_36, r))
    a_light = gaussian_filter1d(a_light, sigma=1)
    v_light = np.sqrt(r * a_light)
    
    return popt, perr, v_fit, v_grav, v_base, v_light

# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    # 加载数据（请替换为真实文件路径）
    # r, v_obs, err_v, I_36, v_gas = load_sparc_data("NGC3198_sparc.txt")
    # 使用模拟数据演示
    r, v_obs, err_v, I_36, v_gas = generate_mock_data()
    
    # 拟合
    popt, perr, v_fit, v_grav, v_base, v_light = fit_spum(r, v_obs, err_v, I_36, v_gas)
    
    # 打印结果
    print("=" * 60)
    print("SPUM 拟合结果")
    print("=" * 60)
    print(f"质量光度比 M/L   = {popt[0]:.3f} ± {perr[0]:.3f}")
    print(f"引力链长 λ_grav  = {popt[1]:.2f} ± {perr[1]:.2f} kpc")
    print(f"底图渐近速度 V_base = {popt[2]:.1f} ± {perr[2]:.1f} km/s")
    print(f"底图特征尺度 r0    = {popt[3]:.2f} ± {perr[3]:.2f} kpc")
    print(f"光调制系数 β      = {popt[4]:.4f} ± {perr[4]:.4f}")
    print("=" * 60)
    
    # 绘图
    plt.figure(figsize=(10, 7))
    plt.errorbar(r, v_obs, yerr=err_v, fmt='ko', capsize=2, label='观测数据')
    plt.plot(r, v_fit, 'r-', lw=2.5, label='SPUM 总拟合')
    plt.plot(r, v_grav, 'b--', lw=2, label='短程引力 (链式)')
    plt.plot(r, v_base, 'g-.', lw=2, label='底图协同运动')
    plt.plot(r, v_light, 'm:', lw=2, label='光调制效应')
    plt.xlabel('半径 R (kpc)', fontsize=12)
    plt.ylabel('旋转速度 V (km/s)', fontsize=12)
    plt.title('SPUM 旋转曲线拟合 (链式引力 + 底图 + 光调制)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
