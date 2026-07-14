#!/usr/bin/env python3
"""
温差力矩实验 — 竞力反演分析
==============================
从已有位移-时间数据中分离驱动分量和恢复力分量。

核心思想：
  系统动力学 = m*x'' + c*x' + k*x = F_temp(t)
  
  已有数据是自由衰减段（驱动脉冲结束后），从中提取：
  - 固有频率 ω₀ = sqrt(k/m)     — 系统惯性特征
  - 阻尼比 ζ = c/(2*sqrt(km))  — 能量耗散率
  - 初始振幅 A₀                — 驱动脉冲的冲击强度
  
  这些参数都是**无量纲比**，不受光学杠杆放大倍数影响。

用法：
  python invert_competition.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import os
import json

# ── 1. 数据加载 ──────────────────────────────────────────────────────────

def load_displacement(filepath):
    """加载 Tracker 导出的位移-时间数据"""
    data = np.loadtxt(filepath, skiprows=2)
    t = data[:, 0]
    x = data[:, 1]
    return t, x

def load_luma(filepath):
    """加载亮度跟踪数据（供电/电 文件）"""
    data = np.loadtxt(filepath, skiprows=2)
    t = data[:, 0]
    luma = data[:, 1]
    return t, luma

# ── 2. 寻找振荡段 ────────────────────────────────────────────────────────

def find_oscillation_segment(t, x, min_amplitude=1e-4, min_peaks=3):
    """
    自动定位数据中的振荡段。
    返回: [(t_start, t_end, peak_indices), ...]
    """
    # 去趋势
    x_detrended = x - np.mean(x)
    
    # 找峰
    peaks, properties = find_peaks(np.abs(x_detrended), 
                                   height=min_amplitude,
                                   distance=5)
    
    if len(peaks) < min_peaks:
        # 尝试降低阈值
        peaks, properties = find_peaks(np.abs(x_detrended),
                                       height=min_amplitude * 0.5,
                                       distance=3)
    
    if len(peaks) < 2:
        return [(t[0], t[-1], peaks)]
    
    # 分组：找到峰值的连续区间
    segments = []
    current_segment = [peaks[0]]
    
    for i in range(1, len(peaks)):
        dt = t[peaks[i]] - t[peaks[i-1]]
        if dt < 2.0:  # 连续峰间隔 < 2s => 同一振荡段
            current_segment.append(peaks[i])
        else:
            if len(current_segment) >= min_peaks:
                idx_start = max(0, current_segment[0] - 10)
                idx_end = min(len(t) - 1, current_segment[-1] + 10)
                segments.append((t[idx_start], t[idx_end], current_segment))
            current_segment = [peaks[i]]
    
    if len(current_segment) >= min_peaks:
        idx_start = max(0, current_segment[0] - 10)
        idx_end = min(len(t) - 1, current_segment[-1] + 10)
        segments.append((t[idx_start], t[idx_end], current_segment))
    
    return segments if segments else [(t[0], t[-1], peaks)]

# ── 3. 阻尼谐振子拟合 ───────────────────────────────────────────────────

def damped_harmonic(t, A, zeta, omega0, phi, x0):
    """
    阻尼谐振子：x(t) = x0 + A * exp(-ζ*ω₀*t) * cos(ω_d*t + φ)
    其中 ω_d = ω₀ * sqrt(1 - ζ²)
    """
    omega_d = omega0 * np.sqrt(1 - zeta**2) if zeta < 1 else 0
    return x0 + A * np.exp(-zeta * omega0 * t) * np.cos(omega_d * t + phi)

def fit_damped_oscillator(t_seg, x_seg):
    """拟合阻尼谐振子"""
    # 初值估计
    x0_est = np.mean(x_seg[-10:]) if len(x_seg) > 10 else np.median(x_seg)
    A_est = (np.max(x_seg) - np.min(x_seg)) / 2
    
    # 周期估计：过零点
    x_detrended = x_seg - x0_est
    zero_crossings = np.where(np.diff(np.sign(x_detrended)))[0]
    if len(zero_crossings) >= 4:
        periods = np.diff(t_seg[zero_crossings])
        T_est = np.median(periods) * 2
        omega0_est = 2 * np.pi / T_est
    else:
        omega0_est = 2 * np.pi  # 默认 1Hz
    
    # 阻尼比初值：从相邻峰幅值比估计
    peaks, _ = find_peaks(np.abs(x_detrended), height=A_est * 0.3)
    if len(peaks) >= 3:
        ratios = []
        for i in range(1, min(len(peaks), 6)):
            if x_detrended[peaks[i-1]] != 0:
                ratios.append(np.abs(x_detrended[peaks[i]] / x_detrended[peaks[i-1]]))
        if ratios:
            mean_ratio = np.mean(ratios)
            zeta_est = -np.log(mean_ratio) / (omega0_est * T_est) if T_est > 0 else 0.05
            zeta_est = np.clip(zeta_est, 0.001, 0.5)
        else:
            zeta_est = 0.05
    else:
        zeta_est = 0.05
    
    # 用归一化时间提高拟合稳定性
    t_norm = t_seg / t_seg[-1] if t_seg[-1] > 0 else t_seg
    
    try:
        # 先试全局优化
        popt, pcov = curve_fit(
            lambda t_n, A, z, w, p, x0: damped_harmonic(t_n * t_seg[-1], A, z, w, p, x0)
                if t_seg[-1] > 0 else damped_harmonic(t_n, A, z, w, p, x0),
            t_norm, x_seg,
            p0=[A_est, zeta_est, omega0_est, 0, x0_est],
            bounds=([A_est*0.05, 0.001, omega0_est*0.3, -np.pi, x0_est - A_est*2],
                    [A_est*5, 0.5, omega0_est*3, np.pi, x0_est + A_est*2]),
            maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.inf]*5
        
        return {
            'A': popt[0], 'A_err': perr[0],
            'zeta': popt[1], 'zeta_err': perr[1],
            'omega0': popt[2], 'omega0_err': perr[2],
            'phi': popt[3], 'phi_err': perr[3],
            'x0': popt[4], 'x0_err': perr[4],
            'converged': True
        }
    except Exception as e:
        return {'converged': False, 'error': str(popt) if 'popt' in dir() else str(e) if 'e' in dir() else 'fit failed'}

# ── 4. 竞力比计算 ───────────────────────────────────────────────────────

def compute_competition_ratio(fit_result, t_seg, x_seg):
    """
    竞力分析：从阻尼谐振子拟合结果中提取三种竞力比。
    
    所有竞力比都无量纲，不受光学杠杆放大倍数影响。
    
    ── 竞力比 1: 驱动类型比 ──
      R_type = A / |x0 - x_initial|
      A = 振荡振幅，x0 = 新平衡位置，x_initial = 驱动前静止位置
      R >> 1: 脉冲型驱动（温度冲击是一过性的）
      R ~ 1: 持续型驱动（温度产生了静态偏移）
      R = inf: 纯脉冲，无静态偏移
    
    ── 竞力比 2: 耗散竞力比 ──
      R_dissip = exp(-4*pi*zeta/sqrt(1-zeta^2))
      一个周期后剩余能量比例
      > 0.5: 惯性占主导（能量保持好）
      < 0.1: 耗散占主导（能量快速流失）
    
    ── 竞力比 3: 共振相角（驱动vs惯性vs刚度）──
      phi_res = atan2(1, 2*zeta)  
      在共振频率下，响应滞后驱动的相角
      ~90deg: 纯惯性主导
      ~0deg: 纯阻尼主导
      ~-90deg: 纯刚度主导
    """
    if not fit_result.get('converged', False):
        return None
    
    A = abs(fit_result['A'])
    omega0 = fit_result['omega0']
    zeta = fit_result['zeta']
    x0_fitted = fit_result['x0']
    
    # 初始平衡位置（驱动前的静止位置）
    x_initial = x_seg[0]
    
    # 静态偏移 = 新平衡 - 旧平衡
    static_shift = x0_fitted - x_initial
    
    # ── 驱动类型比 ──
    if abs(static_shift) > 1e-8:
        ratio_drive_type = A / abs(static_shift)
    else:
        ratio_drive_type = 99.0  # 近似无穷（纯脉冲驱动）
    
    # ── 耗散竞力比（一周期后剩余能量比例）──
    if zeta < 1:
        energy_retained = np.exp(-4 * np.pi * zeta / np.sqrt(max(1 - zeta**2, 1e-6)))
    else:
        energy_retained = 0.0
    
    # ── 共振相角（度）──
    phi_res = np.degrees(np.arctan2(1, 2 * max(zeta, 1e-6)))
    
    # ── 阻尼状态 ──
    if zeta < 0.05:
        regime = 'strongly_underdamped (inertia dominated)'
    elif zeta < 0.15:
        regime = 'underdamped (inertia > damping)'
    elif zeta < 0.3:
        regime = 'moderately_damped (mixed)'
    elif zeta < 0.7:
        regime = 'overdamped (damping dominated)'
    else:
        regime = 'strongly_overdamped'
    
    # ── 竞力比 4: 有效冲击速度（m/s，这是有量纲的，但可比较不同实验）──
    v_impact = A * omega0  # 通过平衡位置时的最大速度
    
    return {
        'A': A,
        'A_mm': A * 1000,
        'omega0': omega0,
        'f0_hz': omega0 / (2 * np.pi),
        'zeta': zeta,
        'static_shift': static_shift,
        'ratio_drive_type': ratio_drive_type,
        'energy_retained_per_cycle': energy_retained,
        'phi_resonance_deg': phi_res,
        'damping_regime': regime,
        'v_impact_m_s': v_impact
    }

# ── 5. 主分析函数 ───────────────────────────────────────────────────────

def analyze_dataset(filepath, label, output_dir):
    """分析单个数据集"""
    t, x = load_displacement(filepath)
    
    segments = find_oscillation_segment(t, x)
    
    print(f"\n{'='*60}")
    print(f"📊 {label}")
    print(f"   文件: {os.path.basename(filepath)}")
    print(f"   数据点: {len(t)}, 时间范围: {t[0]:.2f}-{t[-1]:.2f}s")
    print(f"   检测到 {len(segments)} 个振荡段")
    
    results = []
    
    # 创建子图
    fig, axes = plt.subplots(len(segments), 1, figsize=(10, 3*len(segments)),
                             squeeze=False)
    fig.suptitle(f"竞力反演 — {label}")
    
    for i, (t_start, t_end, peak_idxs) in enumerate(segments):
        mask = (t >= t_start) & (t <= t_end)
        t_seg = t[mask]
        x_seg = x[mask]
        
        result = fit_damped_oscillator(t_seg, x_seg)
        
        if result.get('converged', False):
            comp = compute_competition_ratio(result, t_seg, x_seg)
            result['competition'] = comp
            
            # 绘制
            ax = axes[i, 0]
            ax.plot(t_seg, x_seg, 'b.', markersize=1, label='数据')
            
            t_fit = np.linspace(t_seg[0], t_seg[-1], 500)
            x_fit = damped_harmonic(t_fit, result['A'], result['zeta'],
                                    result['omega0'], result['phi'], result['x0'])
            ax.plot(t_fit, x_fit, 'r-', linewidth=1.5, label='阻尼谐振子拟合')
            
            # 平衡线
            ax.axhline(result['x0'], color='gray', linestyle='--', alpha=0.5,
                       label=f"平衡 = {result['x0']:.4f}m")
            ax.axhline(x_seg[0], color='green', linestyle=':', alpha=0.5,
                       label=f"初始 = {x_seg[0]:.4f}m")
            
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('位移 (m)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            info = (f"f0={comp['f0_hz']:.3f}Hz  "
                    f"zeta={comp['zeta']:.4f}  "
                    f"A={comp['A_mm']:.2f}mm  "
                    f"R_type={comp['ratio_drive_type']:.1f}  "
                    f"E_retain={comp['energy_retained_per_cycle']:.1%}")
            ax.set_title(info, fontsize=9)
            
            results.append({
                'segment': i,
                't_range': (t_start, t_end),
                'fit': result,
                'competition': comp
            })
            
            print(f"\n   振荡段 {i+1}: {t_start:.2f}-{t_end:.2f}s")
            print(f"   固有频率: f0 = {comp['f0_hz']:.3f} Hz")
            print(f"   阻尼比:   zeta = {comp['zeta']:.4f}")
            print(f"   振幅:     A = {comp['A_mm']:.2f} mm")
            print(f"   冲击速度: v_impact = {comp['v_impact_m_s']:.4f} m/s")
            print(f"   ── 竞力比 ──")
            print(f"   驱动类型比: R_type = {comp['ratio_drive_type']:.1f}  "
                  f"({'脉冲驱动' if comp['ratio_drive_type'] > 3 else '持续驱动'})")
            print(f"   周期能量保留: {comp['energy_retained_per_cycle']:.1%}  "
                  f"({'惯性占主导' if comp['energy_retained_per_cycle'] > 0.5 else '耗散占主导'})")
            print(f"   共振相角: {comp['phi_resonance_deg']:.1f} deg  "
                  f"({'惯性主导' if comp['phi_resonance_deg'] > 60 else '混合'})")
            print(f"   阻尼状态: {comp['damping_regime']}")
        else:
            ax = axes[i, 0]
            ax.plot(t_seg, x_seg, 'b.', markersize=1)
            ax.set_title(f"振荡段 {i+1}: 拟合未收敛", fontsize=9)
            print(f"\n   振荡段 {i+1}: 拟合未收敛")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"inversion_{label.replace('/', '_').replace(':', '')}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n   图表: {plot_path}")
    plt.close()
    
    return results

# ── 主程序 ────────────────────────────────────────────────────────────────

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'analysis_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 数据集列表
    datasets = [
        (os.path.join(base_dir, '质量A.txt'), '根目录_质量A'),
        (os.path.join(base_dir, '26-4-26', '质量.txt'), '4-26_质量'),
        (os.path.join(base_dir, '26-4-27', '质量.txt'), '4-27_质量'),
    ]
    
    # 供电数据（亮度跟踪）
    power_files = [
        (os.path.join(base_dir, '26-4-26', '供电.txt'), '4-26_供电'),
        (os.path.join(base_dir, '26-4-27', '供电.txt'), '4-27_供电'),
        (os.path.join(base_dir, '电.txt'), '根目录_电'),
    ]
    
    all_results = {}
    
    print("=" * 60)
    print("  温差力矩实验 — 竞力反演分析")
    print("=" * 60)
    
    for filepath, label in datasets:
        if os.path.exists(filepath):
            results = analyze_dataset(filepath, label, output_dir)
            all_results[label] = results
        else:
            print(f"\n  文件不存在: {filepath}")
    
    # 绘制供电信号（时间轴对齐）
    fig, axes = plt.subplots(len(power_files), 1, figsize=(10, 2.5*len(power_files)))
    fig.suptitle("供电信号（亮度跟踪）")
    
    for i, (filepath, label) in enumerate(power_files):
        if os.path.exists(filepath):
            t, luma = load_luma(filepath)
            ax = axes[i] if len(power_files) > 1 else axes
            ax.plot(t, luma, 'k-', linewidth=0.5)
            ax.set_title(label)
            ax.set_xlabel('时间 (s)')
            ax.set_ylabel('亮度 (luma)')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'power_signals.png'), dpi=150)
    print(f"\n供电信号图: {os.path.join(output_dir, 'power_signals.png')}")
    plt.close()
    
    # 汇总比较表
    print(f"\n{'='*80}")
    print(f"  竞力参数汇总")
    print(f"{'='*80}")
    print(f"{'数据集':<18} {'f0(Hz)':<8} {'zeta':<8} {'A(mm)':<8} {'R_type':<8} {'E_retain':<10} {'Phi(deg)':<10} {'v(m/s)':<10}")
    print("-"*80)
    
    for label, results in all_results.items():
        for r in results:
            comp = r['competition']
            if comp:
                print(f"{label:<18} {comp['f0_hz']:<8.3f} "
                      f"{comp['zeta']:<8.4f} "
                      f"{comp['A_mm']:<8.2f} "
                      f"{comp['ratio_drive_type']:<8.1f} "
                      f"{comp['energy_retained_per_cycle']:<10.1%} "
                      f"{comp['phi_resonance_deg']:<10.1f} "
                      f"{comp['v_impact_m_s']:<10.4f}")
    
    # 保存汇总
    summary = {
        'analysis_date': '2026-07-14',
        'method': 'damped_harmonic_oscillator_fit',
        'model': 'x(t) = x0 + A * exp(-zeta * omega0 * t) * cos(omega_d * t + phi)',
        'dimensionless_ratios': [
            'ratio_drive_type (R_type) = A/|static_shift| - impulse vs sustained drive',
            'energy_retained_per_cycle - fraction of energy kept after 1 period',
            'phi_resonance_deg - phase lag at resonance (inertia vs damping vs stiffness)'
        ],
        'datasets': {}
    }
    
    for label, results in all_results.items():
        for r in results:
            comp = r['competition']
            if comp:
                key = f"{label}_seg{r['segment']}"
                summary['datasets'][key] = {
                    'f0_hz': comp['f0_hz'],
                    'omega0': comp['omega0'],
                    'zeta': comp['zeta'],
                    'A_mm': comp['A_mm'],
                    'v_impact_m_s': comp['v_impact_m_s'],
                    'static_shift': comp['static_shift'],
                    'ratio_drive_type': comp['ratio_drive_type'],
                    'energy_retained_per_cycle': comp['energy_retained_per_cycle'],
                    'phi_resonance_deg': comp['phi_resonance_deg'],
                    'damping_regime': comp['damping_regime']
                }
    
    with open(os.path.join(output_dir, 'competition_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    
    print(f"\n详细结果: {os.path.join(output_dir, 'competition_summary.json')}")
    print("=" * 80)

if __name__ == '__main__':
    main()
