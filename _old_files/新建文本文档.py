import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 读取对比数据
df = pd.read_csv("/mnt/SPUM_vs_NFW_comparison.csv")
# 筛选共同拟合成功的星系
df_valid = df.dropna(subset=['SPUM_chi2', 'NFW_chi2'])

# 设置期刊图表风格
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# --------------------------
# 图1：SPUM vs NFW 约化卡方箱线图（量化差异）
# --------------------------
fig, ax = plt.subplots(figsize=(8, 6))
box_data = [df_valid['SPUM_chi2'], df_valid['NFW_chi2']]
box = ax.boxplot(box_data, labels=['SPUM', 'ΛCDM (NFW)'], patch_artist=True,
                 boxprops=dict(linewidth=1.5), medianprops=dict(linewidth=2, color='red'))

# 填充颜色
colors = ['#1f77b4', '#ff7f0e']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 添加均值点和标注
ax.scatter([1, 2], [df_valid['SPUM_chi2'].mean(), df_valid['NFW_chi2'].mean()], 
           color='darkblue', s=100, marker='*', zorder=5, label='平均值')
ax.text(1, df_valid['SPUM_chi2'].mean()+0.05, f'{df_valid["SPUM_chi2"].mean():.2f}', 
        ha='center', fontweight='bold')
ax.text(2, df_valid['NFW_chi2'].mean()+0.05, f'{df_valid["NFW_chi2"].mean():.2f}', 
        ha='center', fontweight='bold')

ax.set_ylabel('约化卡方 χ²/DoF', fontweight='bold')
ax.set_title('SPUM 与 ΛCDM（NFW）拟合优度对比（148 个星系）', fontweight='bold', pad=15)
ax.legend(framealpha=0.9)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.8, axis='y')
plt.tight_layout()
plt.savefig("/mnt/SPUM_vs_NFW_chi2_boxplot.png", dpi=300, bbox_inches='tight')
plt.close()

# --------------------------
# 图2：暗物质晕浓度参数 c 分布（暴露 NFW 问题）
# --------------------------
fig, ax = plt.subplots(figsize=(8, 5))
# 仅绘制 NFW 拟合成功的星系
nfw_c = df_valid['NFW_c'].dropna()
ax.hist(nfw_c, bins=30, color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.8)
# 标注“非物理区间”（c>15）
ax.axvline(15, color='red', linestyle='--', linewidth=2.5, label='c=15（非物理阈值）')
ax.fill_betweenx([0, ax.get_ylim()[1]], 15, nfw_c.max(), color='red', alpha=0.2, label='c>15（无物理解释）')

ax.set_xlabel('NFW 暗物质晕浓度参数 c', fontweight='bold')
ax.set_ylabel('星系数', fontweight='bold')
ax.set_title('ΛCDM（NFW）暗物质晕浓度参数分布（暴露 fine-tuning 问题）', fontweight='bold', pad=15)
ax.legend(framealpha=0.9, loc='upper right')
ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig("/mnt/NFW_concentration_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# 输出统计检验结果
t_stat, p_value = stats.ttest_ind(df_valid['SPUM_chi2'], df_valid['NFW_chi2'], equal_var=False)
print("="*80)
print("📊 SPUM vs ΛCDM（NFW）统计检验结果")
print("="*80)
print(f"SPUM 平均 χ²/DoF: {df_valid['SPUM_chi2'].mean():.2f} ± {df_valid['SPUM_chi2'].std():.2f}")
print(f"NFW 平均 χ²/DoF: {df_valid['NFW_chi2'].mean():.2f} ± {df_valid['NFW_chi2'].std():.2f}")
print(f"t 统计量: {t_stat:.2f}")
print(f"p 值: {p_value:.2e}（p<0.05 表示差异显著）")
print(f"NFW 模型 c>15 的星系数: {sum(nfw_c > 15)} 个（占比 {sum(nfw_c > 15)/len(nfw_c)*100:.1f}%）")
print("="*80)