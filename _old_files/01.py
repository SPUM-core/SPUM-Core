import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置英文显示避免字体问题
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取对比结果 CSV 文件
df = pd.read_csv('SPUM_vs_NFW_comparison.csv')

# 筛选有效的 delta_aic
delta_aic = df['delta_aic'][np.isfinite(df['delta_aic'])]

# 定义分类区间
conditions = [
    delta_aic < -10,
    (delta_aic >= -10) & (delta_aic < -2),
    (delta_aic >= -2) & (delta_aic <= 2),
    delta_aic > 2
]
labels = [
    'Strong SPUM advantage (ΔAIC < -10)',
    'Moderate SPUM advantage (-10 ≤ ΔAIC < -2)',
    'Indistinguishable (|ΔAIC| ≤ 2)',
    'NFW advantage (ΔAIC > 2)'
]
colors = ['#2e8b57', '#3cb371', '#ffd700', '#cd5c5c']

# 计算每个区间的数量和百分比
counts = [np.sum(cond) for cond in conditions]
percentages = [100 * c / len(delta_aic) for c in counts]

# 打印统计结果
print(f"Total valid galaxies: {len(delta_aic)}")
for label, cnt, pct in zip(labels, counts, percentages):
    print(f"{label}: {cnt} ({pct:.1f}%)")

# 选择绘图类型：饼图 (pie) 或水平条形图 (barh)
plot_type = 'pie'  # 可改为 'barh'

if plot_type == 'pie':
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=90, textprops={'fontsize': 10}
    )
    # 美化 autopct 文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title(r'Distribution of $\Delta$AIC categories', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('fig6_pie.png', dpi=150)
    plt.show()

else:  # horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # 让第一类在最上面
    ax.set_xlabel('Number of galaxies')
    ax.set_title(r'Number of galaxies in each $\Delta$AIC category')
    # 在条末端添加数字和百分比
    for i, (cnt, pct) in enumerate(zip(counts, percentages)):
        ax.text(cnt + 0.5, i, f'{cnt} ({pct:.1f}%)', va='center')
    plt.tight_layout()
    plt.savefig('fig6_barh.png', dpi=150)
    plt.show()
