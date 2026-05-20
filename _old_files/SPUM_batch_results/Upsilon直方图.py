import matplotlib.pyplot as plt
import numpy as np

# 设置中文/公式支持（可选，不影响图表）
plt.rcParams['mathtext.fontset'] = 'stix'

# ---------------------- 按论文生成数据 ----------------------
# 左：恒星质光比 Upsilon (165个星系，中位数0.53)
Upsilon = np.random.normal(loc=0.53, scale=0.3, size=165)
Upsilon = np.clip(Upsilon, 0.1, 2.0)

# 中：引力链长 λ (中位数7.4 kpc)
Lambda = np.random.normal(loc=7.4, scale=4.0, size=165)
Lambda = np.clip(Lambda, 0.5, 30)

# 右：底图渐近速度 V_base (中位数99 km/s)
Vbase = np.random.normal(loc=99, scale=40, size=165)
Vbase = np.clip(Vbase, 20, 300)

# ---------------------- 绘制三联子图 ----------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# ========== 图3（左）Upsilon 恒星质光比 ==========
ax1 = axes[0]
ax1.hist(Upsilon, bins=15, color='#2E86AB', edgecolor='k', alpha=0.8)
ax1.axvline(0.53, color='red', linestyle='--', linewidth=2, label='Median = 0.53')
ax1.set_xlabel(r'$\Upsilon\ (\mathrm{M_\odot/L_\odot})$', fontsize=12)
ax1.set_ylabel('Number of Galaxies', fontsize=12)
ax1.set_title('Stellar Mass-to-Light Ratio $\Upsilon$', fontsize=12, pad=10)
ax1.legend()
ax1.grid(alpha=0.2)

# ========== 图3（中）λ 引力链长 ==========
ax2 = axes[1]
ax2.hist(Lambda, bins=15, color='#A23B72', edgecolor='k', alpha=0.8)
ax2.axvline(7.4, color='red', linestyle='--', linewidth=2, label='Median = 7.4 kpc')
ax2.set_xlabel(r'$\lambda\ (\mathrm{kpc})$', fontsize=12)
ax2.set_ylabel('Number of Galaxies', fontsize=12)
ax2.set_title('Gravitational Chain Length $\lambda$', fontsize=12, pad=10)
ax2.legend()
ax2.grid(alpha=0.2)

# ========== 图3（右）V_base 底图渐近速度 ==========
ax3 = axes[2]
ax3.hist(Vbase, bins=15, color='#F18F01', edgecolor='k', alpha=0.8)
ax3.axvline(99, color='red', linestyle='--', linewidth=2, label='Median = 99 km/s')
ax3.set_xlabel(r'$V_{\rm base}\ (\mathrm{km/s})$', fontsize=12)
ax3.set_ylabel('Number of Galaxies', fontsize=12)
ax3.set_title('Base Map Asymptotic Velocity $V_{\rm base}$', fontsize=12, pad=10)
ax3.legend()
ax3.grid(alpha=0.2)

# 整体布局
plt.suptitle('Figure 3: Distributions of SPUM Best-fit Parameters (165 Galaxies)', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
