#!/usr/bin/env python3
"""完整拟合数据报告"""
import sys, os, json, csv, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from comprehensive_fitting import (
    MitBihSource, ButPpgSource, PthXlSource, SyntheticSource,
    DataSource, FittingSample, CalibrationEngine, FittingReport
)
from collections import Counter

# ═══════ 1. 各数据源详细统计 ═══════
print("=" * 70)
print("  SPUM 综合拟合 — 详细数据报告")
print("=" * 70)

sources = [
    ("MIT-BIH (ECG+心律)", MitBihSource(), ["火(梯度)"]),
    ("PTB-XL (ECG+诊断)", PthXlSource(), ["火(梯度)"]),
    ("BUT PPG (血糖/血压/SpO2)", ButPpgSource(), ["土(储备)", "水(流通)", "金(修剪)"]),
    ("合成数据", SyntheticSource(n_reps=3), ["木(约束)", "火(梯度)", "土(储备)", "金(修剪)", "水(流通)"]),
]

all_samples = []
for name, src, dims in sources:
    samples = src.load()
    all_samples.extend(samples)
    print(f"\n{'─'*70}")
    print(f"  [{name}]")
    print(f"    样本数: {len(samples)}")
    print(f"    影响维度: {dims}")
    
    # 按 label_category 统计
    cats = Counter(s.label_category for s in samples)
    print(f"    类别分布: {dict(cats)}")
    
    # 按 label 统计
    labels = Counter(s.label for s in samples)
    print(f"    标注分布: {dict(labels)}")
    
    # 每类均值 ΔF
    by_cat = {}
    for s in samples:
        by_cat.setdefault(s.label_category, []).append(s.delta_F)
    for cat, df_list in sorted(by_cat.items()):
        mean = np.mean(df_list, axis=0)
        std = np.std(df_list, axis=0)
        print(f"    {cat} (n={len(df_list)}):")
        print(f"      ΔF均=[{mean[0]:.3f},{mean[1]:.3f},{mean[2]:.3f},{mean[3]:.3f},{mean[4]:.3f}]")
        print(f"      ΔF标准差=[{std[0]:.3f},{std[1]:.3f},{std[2]:.3f},{std[3]:.3f},{std[4]:.3f}]")

# ═══════ 2. 总体统计 ═══════
print(f"\n{'='*70}")
print(f"  总体统计")
print(f"{'─'*70}")
print(f"  总计: {len(all_samples)} 样本")
all_cats = Counter(s.label_category for s in all_samples)
print(f"  类别: {dict(all_cats)}")

engine = CalibrationEngine(all_samples)
results = engine.calibrate_all()

print(f"\n{'='*70}")
print(f"  五维分离度 (Fisher Ratio)")
print(f"{'─'*70}")
total_sep = 0
n_dims = 0
for dim_name in ["木(约束)", "火(梯度)", "土(储备)", "金(修剪)", "水(流通)"]:
    r = results[dim_name]
    total_sep += r.separation
    n_dims += 1
    # 分离度解释
    if r.separation > 1.0:
        note = "✅ 极好 — 临床标注可清晰区分"
    elif r.separation > 0.3:
        note = "✅ 良好"
    elif r.separation > 0.1:
        note = "🟡 可用 — 需更多数据增强"
    elif r.separation > 0.05:
        note = "⚠️ 弱 — 需补充数据"
    else:
        note = "❌ 不足 — 数据缺失或标注不充分"
    
    print(f"  {dim_name}: 分离度={r.separation:.4f}  {note}")
    print(f"    样本={r.n_samples}, 标注类别={r.n_labels}")
    if r.sensitivity:
        for g, sens in sorted(r.sensitivity.items()):
            g_short = g.split("/")[-1] if "/" in g else g
            print(f"      {g_short}: 灵敏度={sens:.2f}")

avg_sep = total_sep / n_dims
print(f"\n  ─平均分离度: {avg_sep:.4f}")
if avg_sep > 0.2:
    print(f"  ─判定: ✅ 拟合通过 (超过80%阈值)")
elif avg_sep > 0.1:
    print(f"  ─判定: 🟡 拟合可接受")
else:
    print(f"  ─判定: ⚠️ 拟合不足")

# ═══════ 3. 模板更新建议 ═══════
print(f"\n{'='*70}")
print(f"  证型模板更新建议")
print(f"{'─'*70}")
for dim_name, r in results.items():
    if r.template_updates:
        for g, update in r.template_updates.items():
            print(f"  {g}: {update['dim']} {update['old_sig']}→{update['suggested_sig']}")
    else:
        print(f"  {dim_name}: 无建议更新")

# ═══════ 4. 五形 ΔF 综合分布 ═══════
print(f"\n{'='*70}")
print(f"  五形 ΔF 综合分布 (NORMAL_BASELINE)")
print(f"{'─'*70}")
from syndrome_decoder import NORMAL_BASELINE
dim_names = ["木(约束)", "火(梯度)", "土(储备)", "金(修剪)", "水(流通)"]
print(f"  Baseline: [{', '.join(f'{v:.3f}' for v in NORMAL_BASELINE)}]")
print()

# 按大类展示
for cat in sorted(set(s.label_category for s in all_samples)):
    items = [s for s in all_samples if s.label_category == cat]
    if not items: continue
    means = np.mean([s.delta_F for s in items], axis=0)
    offsets = means - np.array(NORMAL_BASELINE)
    print(f"  [{cat}] n={len(items)}")
    for j, d in enumerate(dim_names):
        arrow = "↑" if offsets[j] > 0.05 else ("↓" if offsets[j] < -0.05 else "→")
        print(f"    {d}: {means[j]:.4f}  ({offsets[j]:+.4f}) {arrow}")
    print()

# ═══════ 5. 完整报告JSON ═══════
report = FittingReport([s[1] for s in sources]).generate()
report_path = os.path.join(os.path.dirname(__file__), "fitting_report.json")
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
print(f"{'─'*70}")
print(f"  完整报告已保存: {report_path}")
print(f"{'='*70}")
