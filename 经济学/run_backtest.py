"""
run_backtest.py — SPUM 五形回测主入口
======================================

完整的回测框架：S₀ 同命格预测策略 vs 传统技术指标基线。

流程:
  1. 加载已有数据（vis_data.json + _vol_cache）
  2. 构建 S₀ 跨预测信号
  3. 对每只股票运行回测（S₀ + 全部技术指标基线）
  4. 聚合成多策略对比报告
  5. 计算截面 IC + 时序 IC
  6. 导出回测指标到 vis_data.json

用法:
  python run_backtest.py                     # 完整运行
  python run_backtest.py --quick             # 快速模式（抽样 20 只）
  python run_backtest.py --only-s0           # 只跑 S₀ 策略
  python run_backtest.py --export-only       # 只导出已有结果
"""
import json, os, sys, math, traceback
from collections import defaultdict
from datetime import datetime

import numpy as np

# 项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import (
    BacktestEngine, compute_portfolio_metrics, compute_ic_series,
    compute_ic_metrics, compare_strategies, compute_cross_sectional_ic,
    StrategyResult, load_s0_pairs, load_stock_data, load_vol_data,
)
from benchmark_strategies import compute_all_benchmarks, strategy_descriptions

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(DATA_DIR, '_vol_cache')
OUT_PATH = os.path.join(DATA_DIR, 'backtest_report.txt')
JSON_PATH = os.path.join(DATA_DIR, 'vis_data.json')

WX = ['木', '火', '土', '金', '水']

# ============================================================
#  数据加载
# ============================================================

def load_cache_data():
    """从 _vol_cache 加载完整的波动率数据（含原始 returns）"""
    data = {}
    if not os.path.exists(CACHE_DIR):
        print(f'  ⚠ _vol_cache 不存在，请先运行 wuxing_predict.py')
        return data
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith('.npy'):
            continue
        sym = fname.replace('.npy', '')
        try:
            obj = np.load(os.path.join(CACHE_DIR, fname), allow_pickle=True).item()
            data[sym] = obj
        except Exception as e:
            print(f'  ✗ 加载 {sym}: {e}')
    return data


def load_price_data(sym: str, max_days: int = 1000) -> list[float]:
    """
    加载个股价格数据。优先从缓存取 returns 反推，否则下载。
    返回收盘价列表（最新在前）。
    """
    # 尝试从 _vol_cache 反推价格
    cache_path = os.path.join(CACHE_DIR, f'{sym}.npy')
    if os.path.exists(cache_path):
        try:
            obj = np.load(cache_path, allow_pickle=True).item()
            rets = obj.get('returns', [])
            # 从 returns 反推价格序列
            prices = [100.0]
            for r in rets[-max_days:]:
                if not (np.isnan(r) or np.isinf(r)):
                    prices.append(prices[-1] * (1 + r))
                else:
                    prices.append(prices[-1])
            return prices[1:]  # 去掉初始 100
        except Exception:
            pass
    
    # 尝试下载
    try:
        import akshare as ak
        df = ak.stock_zh_a_daily(symbol=sym)
        if df is not None and len(df) > 20:
            df = df.sort_values('date')
            return df['close'].values[-max_days:].tolist()
    except Exception:
        pass
    
    return []


# ============================================================
#  S₀ 信号构建
# ============================================================

def build_s0_signal_for_stock(
    target_sym: str,
    pairs: list[dict],
    vol_data: dict,
    cache_data: dict,
    lookback: int = 20,
) -> tuple[list[str], list[float]]:
    """
    为单只股票构建 S₀ 跨预测信号。
    
    逻辑:
      1. 找到所有以该股票为目标的配对（其他同命格股票 → 本股）
      2. 对每个配对，取信号源最近 lookback 日的 vol 变化
      3. 用配对的 h1_xgb 作为权重加权
      4. 信号 = 加权平均的 vol 变化方向
    
    参数:
        target_sym: 目标股票代码
        pairs:      所有配对列表
        vol_data:   vis_data.json 中的 vol_data
        cache_data: _vol_cache 中的原始数据
        lookback:   回顾窗口
    
    返回:
        (dates, signals) 可用于 BacktestEngine
    """
    # 找到此股票的名字
    stocks = load_stock_data()
    target_name = None
    for s in stocks:
        if s['sym'] == target_sym:
            target_name = s['name']
            break
    if not target_name:
        return [], []
    
    # 找到所有以它为目标的配对
    relevant = [p for p in pairs if p.get('target') == target_name and p.get('h1_xgb') is not None]
    if not relevant:
        return [], []
    
    # 获取目标股的 vol 序列
    target_vol = vol_data.get(target_sym, {})
    target_dates = target_vol.get('dates', [])
    if len(target_dates) < lookback + 10:
        return [], []
    
    # 对每个日期，计算加权信号
    n = len(target_dates)
    signals = [0.0] * n
    dates = target_dates[:]
    
    for i in range(lookback, n):
        date = dates[i]
        weighted_sum = 0.0
        weight_total = 0.0
        
        for p in relevant:
            source_name = p['source']
            source_sym = None
            for s in stocks:
                if s['name'] == source_name:
                    source_sym = s['sym']
                    break
            if not source_sym:
                continue
            
            source_vol = vol_data.get(source_sym, {})
            source_dates = source_vol.get('dates', [])
            source_vals = source_vol.get('vol', [])
            
            # 找同日期
            try:
                idx = source_dates.index(date)
            except ValueError:
                continue
            
            if idx < lookback:
                continue
            
            # 信号源最近 lookback 日的 vol 变化
            recent = source_vals[max(0, idx - lookback):idx + 1]
            if len(recent) < 5:
                continue
            
            recent_mean = sum(recent) / len(recent)
            prior = source_vals[max(0, idx - lookback * 2):max(0, idx - lookback)]
            if len(prior) < 5:
                continue
            prior_mean = sum(prior) / len(prior)
            
            if prior_mean > 0:
                vol_change = (recent_mean - prior_mean) / prior_mean
            else:
                vol_change = 0.0
            
            weight = max(0.0, p['h1_xgb'])  # XGBoost improvement 作为权重
            weighted_sum += vol_change * weight
            weight_total += weight
        
        if weight_total > 0:
            signals[i] = max(-1.0, min(1.0, weighted_sum / weight_total * 2))
    
    return dates, signals


def build_all_s0_signals(
    pairs: list[dict],
    vol_data: dict,
    cache_data: dict,
) -> dict[str, tuple[list[str], list[float]]]:
    """为所有有配对数据的股票构建 S₀ 信号"""
    stocks = load_stock_data()
    result = {}
    
    for s in stocks:
        sym = s['sym']
        if sym not in vol_data:
            continue
        dates, signals = build_s0_signal_for_stock(sym, pairs, vol_data, cache_data)
        if dates and any(abs(s) > 0.001 for s in signals):
            result[sym] = (dates, signals)
    
    return result


# ============================================================
#  回测运行器
# ============================================================

def run_backtest_single_stock(
    sym: str,
    name: str,
    s0_dates: list[str],
    s0_signals: list[float],
    prices: list[float],
    price_dates: list[str],
) -> list[StrategyResult]:
    """
    对单只股票运行全部策略的回测。
    
    返回:
        StrategyResult 列表（S₀ + 所有技术指标基线）
    """
    # 对齐日期：取 S₀ 信号与价格序列的交集
    date_set = set(s0_dates) & set(price_dates)
    if len(date_set) < 50:
        return []
    
    date_sorted = sorted(date_set)
    
    # 对齐 S₀ 信号
    s0_date_to_sig = dict(zip(s0_dates, s0_signals))
    s0_aligned = [s0_date_to_sig[d] for d in date_sorted]
    
    # 对齐价格
    price_date_to_val = dict(zip(price_dates, prices))
    prices_aligned = [price_date_to_val[d] for d in date_sorted]
    
    engine = BacktestEngine(initial_capital=1_000_000)
    results = []
    
    # 1. S₀ 策略
    if any(abs(s) > 0.01 for s in s0_aligned):
        r = engine.run(
            date_sorted, s0_aligned, prices_aligned,
            strategy_name=f'S₀ {name[:6]}'
        )
        if r.metrics:
            results.append(r)
    
    # 2. 技术指标基线
    benchmarks = compute_all_benchmarks(date_sorted, prices_aligned)
    for bname, bsignal in benchmarks.items():
        bname_short = f'{bname[:12]} {name[:6]}'
        r = engine.run(date_sorted, bsignal, prices_aligned, strategy_name=bname_short)
        if r.metrics and r.metrics.get('total_days', 0) >= 50:
            results.append(r)
    
    return results


# ============================================================
#  IC 分析
# ============================================================

def compute_ic_analysis(s0_signals: dict, vol_data: dict) -> dict:
    """
    计算 S₀ 信号的时序 IC 分析。
    
    对每只股票：预测的 vol 变化方向 vs 实际的 vol 变化方向。
    """
    all_ics = []
    
    for sym, (dates, signals) in s0_signals.items():
        vol = vol_data.get(sym, {})
        vol_vals = vol.get('vol', [])
        vol_dates = vol.get('dates', [])
        
        if len(dates) < 50 or len(vol_vals) < 50:
            continue
        
        # 对齐
        d_map = dict(zip(vol_dates, vol_vals))
        aligned_vol = [d_map.get(d) for d in dates]
        
        # 计算实际 vol 变化
        actual_changes = []
        for i in range(1, len(aligned_vol)):
            if aligned_vol[i] is not None and aligned_vol[i-1] is not None and aligned_vol[i-1] > 0:
                actual_changes.append((aligned_vol[i] - aligned_vol[i-1]) / aligned_vol[i-1])
            else:
                actual_changes.append(0.0)
        
        pred_changes = [signals[i] for i in range(1, len(signals))]
        
        n = min(len(pred_changes), len(actual_changes))
        if n < 30:
            continue
        
        # Spearman IC（滚动 20 日不适用截面数据，这里用整体秩相关）
        from scipy.stats import spearmanr
        try:
            ic, pval = spearmanr(pred_changes[:n], actual_changes[:n])
            all_ics.append({
                'sym': sym,
                'ic': round(float(ic), 4),
                'pval': round(float(pval), 4),
                'n_days': n,
            })
        except Exception:
            continue
    
    if not all_ics:
        return {'n_stocks': 0, 'ic_mean': 0, 'ic_pos_ratio': 0}
    
    ic_vals = [x['ic'] for x in all_ics]
    return {
        'n_stocks': len(all_ics),
        'ic_mean': round(float(np.mean(ic_vals)), 4),
        'ic_median': round(float(np.median(ic_vals)), 4),
        'ic_std': round(float(np.std(ic_vals)), 4),
        'ic_pos_ratio': round(sum(1 for x in ic_vals if x > 0) / len(ic_vals) * 100, 1),
        'details': sorted(all_ics, key=lambda x: -abs(x['ic']))[:20],
    }


# ============================================================
#  聚合报告
# ============================================================

def aggregate_strategy_results(all_results: dict[str, list[StrategyResult]]) -> list[StrategyResult]:
    """
    跨股票聚合策略结果。对每个策略名称，合并所有股票的 daily_returns。
    """
    # 按策略名称分组
    strategy_groups = defaultdict(list)
    
    for sym, results in all_results.items():
        for r in results:
            # 提取基础策略名（去掉股票名后缀）
            base_name = r.name.rsplit(' ', 1)[0] if ' ' in r.name else r.name
            strategy_groups[base_name].append(r.daily_returns)
    
    aggregated = []
    for sname, returns_list in strategy_groups.items():
        if len(returns_list) < 2:
            continue
        
        # 等权混合所有股票的每日收益
        max_len = max(len(r) for r in returns_list)
        mixed = []
        for i in range(max_len):
            day_returns = [r[i] for r in returns_list if i < len(r)]
            mixed.append(float(np.mean(day_returns) if day_returns else 0))
        
        metrics = compute_portfolio_metrics(mixed)
        aggregated.append(StrategyResult(
            name=sname,
            daily_returns=mixed,
            cumulative_returns=[],
            signals=[], positions=[], trades=[],
            metrics=metrics,
        ))
    
    return aggregated


# ============================================================
#  主流程
# ============================================================

def main():
    print('=' * 80)
    print('SPUM 五形回测框架 v1.0')
    print(f'运行时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print('=' * 80)
    
    quick_mode = '--quick' in sys.argv
    only_s0 = '--only-s0' in sys.argv
    export_only = '--export-only' in sys.argv
    
    # 1. 加载数据
    print('\n[1/5] 加载数据...')
    pairs = load_s0_pairs()
    stocks = load_stock_data()
    vol_data = load_vol_data()
    cache_data = load_cache_data()
    
    print(f'  股票: {len(stocks)}只  配对: {len(pairs)}对')
    print(f'  Vol 数据: {len(vol_data)}只  Cache: {len(cache_data)}只')
    
    if export_only:
        print('\n[导出模式] 直接从已有数据生成报告...')
        _export_backtest_metrics(pairs, stocks, {})
        return
    
    # 2. 构建 S₀ 信号
    print('\n[2/5] 构建 S₀ 同命格跨预测信号...')
    s0_signals = build_all_s0_signals(pairs, vol_data, cache_data)
    print(f'  有效 S₀ 信号: {len(s0_signals)} 只股票')
    if s0_signals:
        avg_signal_days = sum(len(v[0]) for v in s0_signals.values()) // len(s0_signals)
        print(f'  平均信号长度: {avg_signal_days} 日')
    
    # 3. 逐股回测
    print('\n[3/5] 逐股回测...')
    all_results = {}
    stock_count = 0
    max_stocks = 20 if quick_mode else 999
    
    stock_list = stocks[:max_stocks]
    for idx, s in enumerate(stock_list):
        sym = s['sym']
        name = s['name']
        
        if sym not in s0_signals:
            continue
        
        s0_dates, s0_sig = s0_signals[sym]
        
        # 加载价格数据
        prices = load_price_data(sym)
        if len(prices) < 100:
            continue
        
        # 构造价格日期
        # 从 cache_data 获取 dates
        cache_entry = cache_data.get(sym, {})
        cache_dates = cache_entry.get('dates', [])
        if len(cache_dates) < len(prices):
            # 用占位日期
            price_dates = [f'day{i}' for i in range(len(prices))]
        else:
            price_dates = cache_dates[-len(prices):]
        
        results = run_backtest_single_stock(
            sym, name, s0_dates, s0_sig, prices, price_dates
        )
        
        if results:
            all_results[sym] = results
            stock_count += 1
        
        if (idx + 1) % 10 == 0:
            print(f'  进度: {idx + 1}/{len(stock_list)}  ({stock_count} 只有效)')
    
    print(f'  完成: {stock_count} 只股票有回测结果')
    
    # 4. 聚合与对比
    print('\n[4/5] 聚合策略结果...')
    aggregated = aggregate_strategy_results(all_results)
    
    report = compare_strategies(aggregated)
    print('\n' + report)
    
    # 5. IC 分析
    print('\n[5/5] IC 分析...')
    ic_analysis = compute_ic_analysis(s0_signals, vol_data)
    
    print(f'  截面 IC (配对级): ', end='')
    cross_ic = compute_cross_sectional_ic(pairs)
    print(f'n={cross_ic.get("n", 0)}, IC={cross_ic.get("ic_mean", 0):.4f}')
    
    print(f'  时序 IC (个股级): ', end='')
    print(f'n={ic_analysis.get("n_stocks", 0)}只, '
          f'IC均值={ic_analysis.get("ic_mean", 0):.4f}, '
          f'正向率={ic_analysis.get("ic_pos_ratio", 0):.1f}%')
    
    if ic_analysis.get('details'):
        print(f'  最佳 5 只:')
        for d in ic_analysis['details'][:5]:
            print(f'    {d["sym"]} IC={d["ic"]:+.4f} p={d["pval"]:.3f} n={d["n_days"]}d')
    
    # 写报告
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write(report)
        f.write('\n\n')
        f.write(f'{"=" * 80}\n')
        f.write('截面 IC 分析\n')
        f.write(f'{"=" * 80}\n')
        f.write(json.dumps(cross_ic, ensure_ascii=False, indent=2))
        f.write('\n\n')
        f.write(f'{"=" * 80}\n')
        f.write('时序 IC 分析\n')
        f.write(f'{"=" * 80}\n')
        f.write(json.dumps({
            'n_stocks': ic_analysis.get('n_stocks'),
            'ic_mean': ic_analysis.get('ic_mean'),
            'ic_median': ic_analysis.get('ic_median'),
            'ic_std': ic_analysis.get('ic_std'),
            'ic_pos_ratio': ic_analysis.get('ic_pos_ratio'),
        }, ensure_ascii=False, indent=2))
    
    print(f'\n✅ 报告已写入 {OUT_PATH}')
    
    # 更新 vis_data.json
    _export_backtest_metrics(pairs, stocks, ic_analysis, cross_ic, aggregated)
    
    return stock_count


def _export_backtest_metrics(pairs, stocks, ic_analysis=None, cross_ic=None, aggregated=None):
    """将回测指标写入 vis_data.json"""
    if not os.path.exists(JSON_PATH):
        print('  ⚠ vis_data.json 不存在，跳过导出')
        return
    
    with open(JSON_PATH, encoding='utf-8') as f:
        data = json.load(f)
    
    # 添加回测指标
    backtest_metrics = {
        'ic_cross_sectional': cross_ic or compute_cross_sectional_ic(pairs),
        'ic_time_series': {
            'n_stocks': ic_analysis.get('n_stocks', 0) if ic_analysis else 0,
            'ic_mean': ic_analysis.get('ic_mean', 0) if ic_analysis else 0,
            'ic_median': ic_analysis.get('ic_median', 0) if ic_analysis else 0,
            'ic_pos_ratio': ic_analysis.get('ic_pos_ratio', 0) if ic_analysis else 0,
        },
        'export_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'version': '1.0',
    }
    
    # 添加策略对比
    if aggregated:
        strategies = []
        for r in sorted(aggregated, key=lambda x: x.metrics.get('sharpe', 0), reverse=True):
            if r.metrics:
                strategies.append({
                    'name': r.name,
                    'sharpe': r.metrics.get('sharpe', 0),
                    'total_return': r.metrics.get('total_return', 0),
                    'max_drawdown': r.metrics.get('max_drawdown', 0),
                    'calmar': r.metrics.get('calmar', 0),
                    'hit_rate': r.metrics.get('hit_rate', 0),
                    'annual_return': r.metrics.get('annual_return', 0),
                    'annual_vol': r.metrics.get('annual_vol', 0),
                })
        backtest_metrics['strategy_comparison'] = strategies
    
    data['backtest'] = backtest_metrics
    
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'  ✅ 回测指标已导出到 vis_data.json')
    
    # 打印基线描述
    print(f'\n基线策略说明:')
    desc = strategy_descriptions()
    for name, d in desc.items():
        print(f'  {name:<24} {d}')


if __name__ == '__main__':
    try:
        n = main()
        if n is not None and n > 0:
            print(f'\n✅ 回测完成: {n} 只股票')
    except Exception as e:
        print(f'\n❌ FATAL: {e}')
        traceback.print_exc()
        sys.exit(1)
