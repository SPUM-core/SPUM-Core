'''
daily_phase 引擎验证脚本
========================

两部分验证:
  1. 合成数据语义验证(无需网络) — 六类已知模式, 检验五相位识别与
     predict_next_direction 是否符合 SPUM 语义。
  2. 真实数据回测(需 akshare)   — 002415 新浪源, backtest + calibrate(fast)。

用法:
  python verify_daily_phase.py          # 合成 + 真实
  python verify_daily_phase.py --synthetic   # 仅合成
  python verify_daily_phase.py --real        # 仅真实
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from daily_phase import DailyPhaseDetector, backtest, calibrate, PHASE_CONFIG_DEFAULT

SAT = PHASE_CONFIG_DEFAULT['saturation_threshold']


def mk_df(close, vol=None):
    n = len(close)
    if vol is None:
        vol = np.full(n, 1e6)
    close = np.asarray(close, dtype=float)
    high = np.maximum(close, np.concatenate([[close[0]], close[:-1]])) * 1.01
    low = np.minimum(close, np.concatenate([[close[0]], close[:-1]])) * 0.99
    return pd.DataFrame({'close': close, 'high': high, 'low': low, 'volume': vol},
                        index=pd.date_range('2024-01-01', periods=n))


def synthetic_validation():
    print('=' * 78)
    print('一、合成数据语义验证 (seed=42, 每场景 300 帧)')
    print('  注: 百分位检测器=相对异常检测——异常须落在检测窗口尾部')
    print('      (火=ROC5, 金=ATR5/ATR20, 土=40日段, 水=VPT 21日)')
    print('=' * 78)
    rng = np.random.default_rng(42)
    N = 300
    det = DailyPhaseDetector()
    n_pass = 0
    n_total = 0

    def check(name, cond, desc):
        nonlocal n_pass, n_total
        n_total += 1
        mark = 'PASS' if cond else 'FAIL'
        if cond:
            n_pass += 1
        print(f'  [{mark}] {name}: {desc}')

    # 1) 平稳 + 末段10日陡涨放量 → fire 应饱和 (动量+量能双异常)
    base = 10 + np.cumsum(rng.normal(0, 0.02, N - 10))
    surge = base[-1] * np.exp(np.linspace(0, 0.2, 10))
    close1 = np.concatenate([base, surge])
    vol1 = np.concatenate([np.full(N - 10, 1e6), np.full(10, 4e6)])
    p1 = det.detect(mk_df(close1, vol1))
    check('陡涨放量→火亢', p1.fire.value >= SAT, f'fire={p1.fire.value:.2%}')

    # 2) 平稳 + 末段12日宽幅震荡 → wood 低(骨架失稳), metal 高(ATR5/20放大)
    flat = 20 + np.cumsum(rng.normal(0, 0.01, N - 12))
    churn = 20 + np.cumsum(rng.normal(0, 0.25, 12))
    p2 = det.detect(mk_df(np.concatenate([flat, churn])))
    check('高波动→木弱', p2.wood.value < 0.3, f'wood={p2.wood.value:.2%}')
    check('高波动→金亢', p2.metal.value >= SAT, f'metal={p2.metal.value:.2%}')

    # 3) 先宽幅震荡 + 末段42日窄幅横盘 → earth 应高(土盈)
    churn0 = 30 + np.cumsum(rng.normal(0, 0.12, N - 42))
    side1 = churn0[-1] + np.sin(np.arange(42) / 20) * 0.03 + rng.normal(0, 0.004, 42)
    p3 = det.detect(mk_df(np.concatenate([churn0, side1])))
    check('先动后静→土盈', p3.earth.value >= SAT, f'earth={p3.earth.value:.2%}')

    # 4) 平稳 + 末段5日放量缓涨 → water 应高 (VPT 21日窗口尾部极值)
    close4 = 10 + np.cumsum(rng.normal(0, 0.02, N - 5))
    close4 = np.concatenate([close4, close4[-1] * np.exp(np.linspace(0, 0.06, 5))])
    vol4 = np.concatenate([np.full(N - 5, 1e6), np.full(5, 8e6)])
    p4 = det.detect(mk_df(close4, vol4))
    check('放量涨→水盈', p4.water.value >= 0.7, f'water={p4.water.value:.2%}')

    # 5) 平稳 + 末段8日波动骤升 → metal 应高 (金亢)
    close5 = 40 + np.cumsum(rng.normal(0, 0.01, N - 8))
    close5 = np.concatenate([close5, close5[-1] + np.cumsum(rng.normal(0, 0.5, 8))])
    p5 = det.detect(mk_df(close5))
    check('波动骤升→金亢', p5.metal.value >= SAT, f'metal={p5.metal.value:.2%}')

    # 6) 平稳 + 末段放量阴跌(跌幅逐日加大, 末日光头大阴线) → fire 高, water 低
    #    注: VPT 检测 diff 百分位——须避免均匀下跌(增量并列)稀释百分位
    close6 = 50 + np.cumsum(rng.normal(0, 0.01, N - 8))
    tail6 = close6[-1] * np.array([0.99, 0.98, 0.965, 0.95, 0.93, 0.90, 0.86, 0.80])
    close6 = np.concatenate([close6, tail6])
    vol6 = np.concatenate([np.full(N - 8, 1e6), np.full(8, 4e6)])
    p6 = det.detect(mk_df(close6, vol6))
    check('放量跌→火形高', p6.fire.value >= SAT, f'fire={p6.fire.value:.2%}')
    check('放量跌→水枯', p6.water.value < 0.3, f'water={p6.water.value:.2%}')

    print(f'\n  合成验证: {n_pass}/{n_total} 通过\n')
    return n_pass == n_total


def rule_diagnostics(codes=('002415', '600519', '000858', '300750', '600887'),
                     window=120, max_rows=None):
    '''
    相位组合后验方向诊断。
    按饱和组合分类(火/水/金/木/土子集), 统计后一日实际涨/跌/横盘占比。
    用于数据校准 predict_next_direction 规则语义。
    '''
    print('=' * 78)
    print('三、相位组合后验方向诊断 (跨股票汇总)')
    print('=' * 78)
    import akshare as ak
    from collections import defaultdict
    stats = defaultdict(lambda: {'n': 0, 'up': 0, 'down': 0, 'flat': 0, 'ret': []})
    det = DailyPhaseDetector()
    for code in codes:
        symbol = f'sz{code}' if code.startswith(('0', '3')) else f'sh{code}'
        try:
            df = ak.stock_zh_a_daily(symbol=symbol)
        except Exception as e:
            print(f'  [{code}] 获取失败: {e}')
            continue
        df = df.rename(columns={'日期': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        if max_rows:
            df = df.tail(max_rows)
        close = df['close'].values
        for i in range(window, len(close) - 1):
            profile = det.detect(df.iloc[i - window:i])
            saturated = frozenset(profile.saturated_phases())
            actual = close[i + 1] / close[i] - 1
            s = stats[saturated]
            s['n'] += 1
            if actual > 0:
                s['up'] += 1
            elif actual < -0.003:
                s['down'] += 1
            else:
                s['flat'] += 1
            s['ret'].append(actual)
    names = {'火': 'fire', '水': 'water', '木': 'wood', '金': 'metal', '土': 'earth'}
    print(f'\n  {"饱和组合":18s} {"n":>6s} {"涨":>6s} {"跌":>6s} {"横":>6s} {"均值日收益":>10s}')
    for combo, s in sorted(stats.items(), key=lambda kv: -kv[1]['n']):
        if s['n'] < 30:
            continue
        n = s['n']
        ret = np.array(s['ret'])
        combo_str = '+'.join(sorted(combo)) if combo else '(无饱和)'
        print(f'  {combo_str:18s} {n:6d} {s["up"] / n:6.1%} {s["down"] / n:6.1%} '
              f'{s["flat"] / n:6.1%} {ret.mean():+10.4%}')
    return stats


def multi_backtest(codes=('002415', '600519', '000858', '300750', '600887'),
                   window=120, min_confidence=0.3, config=None):
    '''
    跨股票汇总回测: 每只股票独立滚动回测, 汇总方向准确率。
    用于检验规则修正的稳健性(防单股过拟合)。
    '''
    print('=' * 78)
    print('四、跨股票汇总回测 (默认参数, window=120)')
    print('=' * 78)
    import akshare as ak
    rows = []
    total = {'n': 0, 'correct': 0, 'up_n': 0, 'up_c': 0, 'down_n': 0, 'down_c': 0}
    for code in codes:
        symbol = f'sz{code}' if code.startswith(('0', '3')) else f'sh{code}'
        try:
            df = ak.stock_zh_a_daily(symbol=symbol)
        except Exception as e:
            print(f'  [{code}] 获取失败: {e}')
            continue
        df = df.rename(columns={'日期': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        r = backtest(df, window=window, min_confidence=min_confidence, config=config)
        up = r['dirs'].get('↗', {})
        dn = r['dirs'].get('↘', {})
        rows.append((code, len(df), r['accuracy'], r['signals'],
                     up.get('acc', 0), up.get('n', 0), dn.get('acc', 0), dn.get('n', 0)))
        total['n'] += r['signals']
        total['correct'] += r['correct']
        total['up_n'] += up.get('n', 0)
        total['up_c'] += up.get('correct', 0)
        total['down_n'] += dn.get('n', 0)
        total['down_c'] += dn.get('correct', 0)
    print(f'\n  {"代码":8s} {"帧数":>6s} {"acc":>6s} {"信号":>6s} '
          f'{"↗acc":>6s} {"↗n":>6s} {"↘acc":>6s} {"↘n":>6s}')
    for code, nf, acc, sig, upa, upn, dna, dnn in rows:
        print(f'  {code:8s} {nf:6d} {acc:6.1%} {sig:6d} {upa:6.1%} {upn:6d} {dna:6.1%} {dnn:6d}')
    print(f'\n  汇总: n={total["n"]} acc={total["correct"] / max(total["n"], 1):.1%}  '
          f'↗acc={total["up_c"] / max(total["up_n"], 1):.1%} (n={total["up_n"]})  '
          f'↘acc={total["down_c"] / max(total["down_n"], 1):.1%} (n={total["down_n"]})')
    return total


def real_validation(code='002415', source='sina'):
    print('=' * 78)
    print(f'二、真实数据回测 (code={code}, 源={source})')
    print('=' * 78)
    import akshare as ak
    symbol = f'sz{code}' if code.startswith(('0', '3')) else f'sh{code}'
    if source == 'sina':
        df = ak.stock_zh_a_daily(symbol=symbol)
    else:
        df = ak.stock_zh_index_daily(symbol=symbol)
    df = df.rename(columns={'日期': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    print(f'  数据: {len(df)} 帧  {df.index[0].date()} ~ {df.index[-1].date()}')

    # 当前相位
    det = DailyPhaseDetector()
    profile = det.detect(df)
    vals = {k: round(getattr(profile, k).value, 3) for k in
            ['fire', 'water', 'wood', 'metal', 'earth']}
    d, c, r = profile.predict_next_direction()
    print(f'  当前相位={vals}  预测={d} conf={c} | {r}')

    # 滚动回测 (默认参数)
    bt = backtest(df, window=120, min_confidence=0.3)
    print(f'\n  回测(默认参数): acc={bt["accuracy"]:.1%}  signals={bt["signals"]}')
    for k, v in sorted(bt['dirs'].items()):
        print(f'    {k}: n={v["n"]} correct={v["correct"]} acc={v["acc"]:.0%}')

    # 参数校准 (fast)
    best = calibrate(df, window_base=200, fast=True)
    print(f'\n  校准(fast): params=[{best["params"]}]')
    print(f'    score={best["score"]:.3f}  acc={best["accuracy"]:.1%}  signals={best["signals"]}')
    print(f'    方向明细: {best["dirs"]}')

    # 用最优参数复跑
    bt2 = backtest(df, window=200, min_confidence=0.3, config=best['config'])
    print(f'\n  回测(最优参数): acc={bt2["accuracy"]:.1%}  signals={bt2["signals"]}')
    for k, v in sorted(bt2['dirs'].items()):
        print(f'    {k}: n={v["n"]} correct={v["correct"]} acc={v["acc"]:.0%}')
    return bt2


if __name__ == '__main__':
    only = sys.argv[1][2:] if len(sys.argv) > 1 else 'all'
    ok = True
    if only in ('all', 'synthetic'):
        ok = synthetic_validation()
    if only in ('all', 'real'):
        ok = real_validation() and ok
    if only in ('all', 'rules'):
        rule_diagnostics()
    if only in ('all', 'multi'):
        multi_backtest()
    sys.exit(0 if ok else 1)
