"""
export_vis_data.py — 将 .npy + JSON + 预测报告合并为 vis_data.json
供前端可视化页面使用。

用法: python export_vis_data.py
输出: 经济学/vis_data.json
"""
import json, os, re, sys, numpy as np
from collections import defaultdict
from datetime import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(DATA_DIR, '_vol_cache')
JSON_PATH = os.path.join(DATA_DIR, 'mingge_results.json')
REPORT_PATH = os.path.join(DATA_DIR, 'predict_report.txt')
OUT_PATH = os.path.join(DATA_DIR, 'vis_data.json')

WX = ['木', '火', '土', '金', '水']

# ============================================================
#  1. 读取/计算 S₀
# ============================================================
def load_s0():
    """从 JSON 加载 S₀，若不存在则从 wuxing_predict 导入计算"""
    import sys as _sys
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, encoding='utf-8') as f:
            data = json.load(f)
        stocks = []
        for r in data['results']:
            s0 = tuple(r['s0'])
            label = _vec_label(s0)
            mx = max(s0) if max(s0) > 0 else 1
            stocks.append({
                'sym': r['sym'], 'name': r['name'], 'listing': r['listing'],
                's0': list(s0), 'label': label,
                's0_normalized': [round(v/mx, 2) for v in s0]
            })
        return stocks
    # Fallback: 直接计算
    print('⚠ mingge_results.json 不存在，重新计算 S₀...')
    _sys.path.insert(0, DATA_DIR)
    from wuxing_predict import calc_s0, STOCKS
    stocks = []
    for sym, name, list_str in STOCKS:
        s0 = calc_s0(list_str)
        label = _vec_label(s0)
        mx = max(s0) if max(s0) > 0 else 1
        stocks.append({
            'sym': sym, 'name': name, 'listing': list_str,
            's0': list(s0), 'label': label,
            's0_normalized': [round(v/mx, 2) for v in s0]
        })
    return stocks

def _vec_label(vec):
    max_i = max(range(5), key=lambda i: vec[i])
    missing = [WX[i] for i, v in enumerate(vec) if v == 0]
    label = f'{WX[max_i]}旺'
    if missing:
        label += f"缺{''.join(missing)}"
    return label

# ============================================================
#  2. 解析预测报告
# ============================================================
def parse_report():
    """从 predict_report.txt 提取配对预测结果"""
    pairs = []
    if not os.path.exists(REPORT_PATH):
        print('⚠ predict_report.txt 不存在，跳过配对数据')
        return pairs, {}
    with open(REPORT_PATH, encoding='utf-8') as f:
        text = f.read()

    # 匹配行: "  土旺缺火金    中信证券  紫金矿业  +0.5%  +54.7%  +0.2%  +24.4%"
    pat = re.compile(
        r'^\s{2}([\u4e00-\u9fff]+)\s+([\u4e00-\u9fff\w]+)\s+([\u4e00-\u9fff\w]+)'
        r'\s+([+-]?[\d.]+%|N\/A)\s+([+-]?[\d.]+%|N\/A)'
        r'\s+([+-]?[\d.]+%|N\/A)\s+([+-]?[\d.]+%|N\/A)'
    )
    for line in text.split('\n'):
        m = pat.search(line)
        if not m:
            continue
        label, target, source, r1, x1, r5, x5 = m.groups()
        def pct(v):
            if v == 'N/A': return None
            return float(v.replace('%', ''))
        pairs.append({
            'label': label, 'target': target, 'source': source,
            'h1_ridge': pct(r1), 'h1_xgb': pct(x1),
            'h5_ridge': pct(r5), 'h5_xgb': pct(x5),
        })
    return pairs, text

# ============================================================
#  3. 读取波动率数据（抽样减少体积）
# ============================================================
def load_vol(sample_step=5):
    """从 _vol_cache 读取波动率，每 sample_step 日抽样"""
    vol_data = {}
    if not os.path.exists(CACHE_DIR):
        print('⚠ _vol_cache 目录不存在')
        return vol_data
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.npy')]
    for fname in files:
        try:
            sym = fname.replace('.npy', '')
            obj = np.load(os.path.join(CACHE_DIR, fname), allow_pickle=True).item()
            dates = obj['dates']
            vol = obj.get('vol', None)
            if vol is None:
                # 从 returns 计算波动率
                rets = obj['returns']
                vol_arr = np.full(len(rets), np.nan)
                for i in range(20, len(rets)):
                    if not (np.isnan(rets[i-20:i]).any() or np.isinf(rets[i-20:i]).any()):
                        vol_arr[i] = np.std(rets[i-20:i])
                vol = vol_arr
            # 抽样 + 过滤 NaN
            idx = range(0, len(dates), sample_step)
            sampled_dates = [dates[i] for i in idx if i < len(vol) and not np.isnan(vol[i]) and not np.isinf(vol[i])]
            sampled_vol = [float(vol[i]) for i in idx if i < len(vol) and not np.isnan(vol[i]) and not np.isinf(vol[i])]
            # 只保留最近 3 年（约 750 交易日，抽样后 ~150 点）
            cut = min(len(sampled_dates), 200)
            vol_data[sym] = {
                'dates': sampled_dates[-cut:],
                'vol': sampled_vol[-cut:]
            }
        except Exception as e:
            print(f'  ✗ {fname}: {e}')
    return vol_data

# ============================================================
#  4. 构建分组数据
# ============================================================
def build_groups(stocks):
    vec_groups = defaultdict(list)
    for s in stocks:
        vec_groups[tuple(s['s0'])].append(s['name'])
    groups = []
    for vec, members in sorted(vec_groups.items(), key=lambda x: -len(x[1])):
        label = _vec_label(vec)
        groups.append({
            'label': label, 's0': list(vec), 'members': members, 'size': len(members)
        })
    return groups

# ============================================================
#  5. 主流程
# ============================================================
def main():
    print('导出 vis_data.json...')

    # S₀
    stocks = load_s0()
    print(f'  S₀: {len(stocks)} 只股票')

    # 分组
    groups = build_groups(stocks)
    print(f'  分组: {len(groups)} 组')

    # 配对
    pairs, report_text = parse_report()
    print(f'  配对: {len(pairs)} 对')

    # 波动率
    vol_data = load_vol(sample_step=5)
    print(f'  波动率: {len(vol_data)} 只')

    # 构建 name -> sym 映射
    name_to_sym = {s['name']: s['sym'] for s in stocks}
    for p in pairs:
        p['target_sym'] = name_to_sym.get(p['target'], '')
        p['source_sym'] = name_to_sym.get(p['source'], '')

    # 聚合统计
    xgb_h1 = [p['h1_xgb'] for p in pairs if p['h1_xgb'] is not None]
    ridge_h1 = [p['h1_ridge'] for p in pairs if p['h1_ridge'] is not None]
    stats = {
        'total_stocks': len(stocks),
        'total_groups': len(groups),
        'multi_groups': sum(1 for g in groups if g['size'] >= 2),
        'total_pairs': len(pairs),
        'xgb_avg': round(np.mean(xgb_h1), 2) if xgb_h1 else 0,
        'ridge_avg': round(np.mean(ridge_h1), 2) if ridge_h1 else 0,
        'xgb_max': round(max(xgb_h1), 2) if xgb_h1 else 0,
        'xgb_min': round(min(xgb_h1), 2) if xgb_h1 else 0,
    }

    out = {
        'stocks': stocks,
        'groups': groups,
        'pairs': pairs,
        'vol_data': vol_data,
        'stats': stats,
        'export_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    file_size = os.path.getsize(OUT_PATH)
    print(f'\n✅ 已导出: {OUT_PATH}')
    print(f'   大小: {file_size/1024:.1f} KB')
    print(f'   股票: {stats["total_stocks"]} | 分组: {stats["total_groups"]} | 配对: {stats["total_pairs"]}')
    print(f'   XGBoost H+1 均值: {stats["xgb_avg"]}%')

if __name__ == '__main__':
    main()
