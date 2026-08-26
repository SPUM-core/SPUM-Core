'''
地支周期因子实证检验 v2 —— 跨市场合并
=======================================

理论假设 (来自 SPUM 桥接):
  地支 12 = 拓扑常数 12 = 闭合结构 12 折叠
  天干 10 = 五形 × V⁺/V⁻ 极性
  → 同地支年 / 同五行年 / 同天干年的市场相位状态应有统计相似性

v1 (单市场 A股 36年) 结论: 全部不显著, 但每地支组仅 3 样本, 功效极低。
v2 改进: 跨市场合并 —— 每个独立市场是独立演化, 按地支年分组后
         样本量从 3 增至 30+, 统计功效大幅提升。
         关键: 收益率/波动率用市场内百分位 (消除市场间水平差异),
              相位饱和占比天然是市场内相对量, 可直接合并。

数据现实 (2026-08 实测):
  东财全球指数接口不可用, 新浪全球指数仅返回最近 1000 帧 (4年)。
  → 跨市场独立演化扩展失败, 降级为 A股 4 宽基指数合并:
    上证(1990~) 深证成指(1993~) 沪深300(2005~) 创业板指(2010~)
  诚实标注: 4 宽基高度相关, 非独立样本——功效增益有限,
            每地支组 ~8 样本, 结果仅作探索性参考。

用法:
  python verify_ganzhi_cycle.py
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from daily_phase import DailyPhaseDetector

# ---------- 干支历法 ----------
TIANGAN = '甲乙丙丁戊己庚辛壬癸'          # 10
DIZHI = '子丑寅卯辰巳午未申酉戌亥'        # 12
DZ_WUXING = {'子': '水', '丑': '土', '寅': '木', '卯': '木', '辰': '土',
             '巳': '火', '午': '火', '未': '土', '申': '金', '酉': '金',
             '戌': '土', '亥': '水'}
TG_WUXING = {'甲': '木', '乙': '木', '丙': '火', '丁': '火', '戊': '土',
             '己': '土', '庚': '金', '辛': '金', '壬': '水', '癸': '水'}


def ganzhi(year: int):
    '''1984 = 甲子年'''
    tg = TIANGAN[(year - 1984) % 10]
    dz = DIZHI[(year - 1984) % 12]
    return tg, dz


# 市场定义: (名称, 加载方式, 代码/中文名)
# 方式: 'sina'=stock_zh_index_daily, 'tx'=stock_zh_index_daily_tx
MARKETS = [
    ('上证指数', 'sina', 'sh000001'),
    ('深证成指', 'tx', 'sz399001'),
    ('沪深300', 'tx', 'sh000300'),
    ('创业板指', 'tx', 'sz399006'),
]


def load_index(kind: str, sym: str) -> pd.DataFrame:
    import akshare as ak
    if kind == 'sina':
        df = ak.stock_zh_index_daily(symbol=sym)
    else:
        df = ak.stock_zh_index_daily_tx(symbol=sym)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    if 'volume' not in df.columns:
        df['volume'] = df.get('amount', 0)
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
    return df


def annual_stats(df: pd.DataFrame) -> pd.DataFrame:
    '''按年聚合: 收益率 / 波动率 / 振幅 / 干支标注'''
    rows = []
    for year, gy in df.groupby(df.index.year):
        if len(gy) < 60:
            continue
        c = gy['close'].values.astype(float)
        ret = c[-1] / c[0] - 1 if c[0] != 0 else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            daily = np.diff(c) / c[:-1]
        daily = daily[~np.isnan(daily)]
        if len(daily) < 20:
            continue
        vol = np.std(daily) * np.sqrt(252)
        amp = (gy['high'].max() - gy['low'].min()) / c[0] if c[0] != 0 else np.nan
        tg, dz = ganzhi(year)
        rows.append({'year': year, 'tg': tg, 'dz': dz, 'wx': DZ_WUXING[dz],
                     'ret': ret, 'vol': vol, 'amp': amp})
    return pd.DataFrame(rows)


def market_relative(annual: pd.DataFrame) -> pd.DataFrame:
    '''市场内标准化: ret/vol/amp → 市场内百分位 (0~1)'''
    a = annual.copy()
    for col in ('ret', 'vol', 'amp'):
        s = a[col]
        a[col + '_pct'] = s.rank(pct=True)
    return a


def year_phase_profile(df: pd.DataFrame, det: DailyPhaseDetector,
                       window=120, step=5):
    '''每年内抽样检测相位, 返回各相位饱和天数占比 (市场内相对量)'''
    out = {}
    for year, gy in df.groupby(df.index.year):
        n = len(gy)
        if n < window + 40:
            continue
        sats = {k: 0 for k in ('火', '水', '木', '金', '土')}
        total = 0
        thr = det.config['saturation_threshold']
        for i in range(window, n, step):
            prof = det.detect(gy.iloc[i - window:i])
            for p in (prof.fire, prof.water, prof.wood, prof.metal, prof.earth):
                if p.is_saturated(thr):
                    sats[p.label] += 1
            total += 1
        if total > 0:
            out[year] = {k: v / total for k, v in sats.items()}
    return out


def kw_test(group_vals, n_perm=500):
    '''Kruskal-Wallis + 置换检验: (H, p, p_perm, k, n, ε²) 或 None

    置换检验: 将组标签随机打乱 n_perm 次, 计算 H ≥ 真实 H 的比例。
    若 p_perm 明显大于渐近 p → 渐近 p 因小样本/非独立被低估, 结果不可信。
    '''
    try:
        from scipy import stats
    except ImportError:
        return None
    groups = []
    for v in group_vals:
        g = np.asarray(v, dtype=float)
        g = g[~np.isnan(g)]
        if len(g) >= 2 and np.ptp(g) > 0:
            groups.append(g)
    if len(groups) < 2 or sum(len(g) for g in groups) < 8:
        return None
    H, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eps2 = (H - k + 1) / (n - k) if n > k else 0.0
    # 置换检验
    allv = np.concatenate(groups)
    labels = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    rng = np.random.default_rng(2026)
    cnt = 0
    for _ in range(n_perm):
        lab = rng.permutation(labels)
        gs = [allv[lab == i] for i in range(max(lab) + 1)]
        gs = [g for g in gs if len(g) > 0]
        if len(gs) >= 2 and stats.kruskal(*gs)[0] >= H:
            cnt += 1
    p_perm = (cnt + 1) / (n_perm + 1)
    return {'H': float(H), 'p': float(p), 'p_perm': float(p_perm),
            'k': k, 'n': n, 'eps2': float(eps2)}


def line(name, res, note=''):
    if res is None:
        print(f'    {name}: 样本不足, 跳过')
        return
    sig = ' ★显著' if res['p'] < 0.05 else ' 不显著'
    perm = f' perm_p={res["p_perm"]:.3f}' if res.get('p_perm') is not None else ''
    print(f'    {name}: H={res["H"]:.2f} p={res["p"]:.3f}{sig}{perm} '
          f'(k={res["k"]}, n={res["n"]}, ε²={res["eps2"]:.3f}){note}')


def run_checks(pool: pd.DataFrame, phase, label: str):
    print(f'\n{"=" * 74}')
    print(f'  {label}   (合并年观测: {len(pool)})')
    print(f'{"=" * 74}')

    # 绝对指标 (跨市场注意水平差异)
    print('\n  ── 绝对指标 ── (直接合并, 含市场混杂)')
    for col, name in (('ret', '年收益率'), ('vol', '年波动率')):
        print(f'\n  {name}:')
        for key, kn in (('dz', '地支'), ('wx', '五行'), ('tg', '天干')):
            grouped = [pool.loc[pool[key] == g, col].values
                       for g in sorted(pool[key].unique())]
            line(f'{kn}({len(grouped)}组)', kw_test(grouped))

    # 相对指标 (市场内百分位, 消除混杂) —— 主要检验
    print('\n  ── 市场内相对指标 ── (主要检验, 消除市场水平差异)')
    for col, name in (('ret_pct', '年收益百分位'), ('vol_pct', '年波动百分位')):
        print(f'\n  {name}:')
        for key, kn in (('dz', '地支'), ('wx', '五行'), ('tg', '天干')):
            grouped = [pool.loc[pool[key] == g, col].values
                       for g in sorted(pool[key].unique())]
            line(f'{kn}({len(grouped)}组)', kw_test(grouped))

    # 相位饱和占比 (市场内相对量)
    if phase:
        pf = pd.DataFrame(phase).T
        pf.index = pf.index.astype(int)
        ann = pool.set_index('year')
        print('\n  ── 五形相位饱和天数占比 ──')
        for ph in ('火', '水', '木', '金', '土'):
            series = pf[ph].dropna()
            y = [y for y in series.index if y in ann.index]
            series = series.loc[y]
            grp = ann.loc[y]
            for key, kn in (('dz', '地支'), ('wx', '五行')):
                grouped = [series[grp[key] == g].values
                           for g in sorted(grp[key].unique())]
                line(f'{ph}形·{kn}({len(grouped)}组)', kw_test(grouped))

    # 地支均值表 (市场内百分位均值)
    print('\n  ── 地支分组均值 (市场内收益/波动百分位) ──')
    print(f'    {"地支":4s} {"五行":4s} {"样本":>4s} {"收益pct":>9s} {"波动pct":>9s}')
    for dz in DIZHI:
        sub = pool[pool['dz'] == dz]
        if len(sub) == 0:
            continue
        print(f'    {dz:4s} {DZ_WUXING[dz]:4s} {len(sub):4d} '
              f'{sub["ret_pct"].mean():9.2%} {sub["vol_pct"].mean():9.2%}')

    # 同地支年份明细表 —— 周期一致性证伪工具
    # 若地支周期成立, 同地支(相隔12年)各年份应相似; 若差异大 → 特定年份伪影
    print('\n  ── 同地支年份明细 (收益pct按年) —— 周期一致性证伪 ──')
    print('  若地支周期为真, 同地支各年份应相似; 若各年份差异大 → 年份伪影')
    for dz in DIZHI:
        sub = pool[pool['dz'] == dz]
        if len(sub) == 0:
            continue
        yrs = sorted(sub['year'].unique())
        detail = ' | '.join(
            f"{y}:{sub.loc[sub['year'] == y, 'ret_pct'].mean():.0%}" for y in yrs)
        print(f'    {dz}({DZ_WUXING[dz]}): {detail}')


def main():
    det = DailyPhaseDetector()
    pool = pd.DataFrame()
    phase_all = {}
    loaded = []
    for name, kind, sym in MARKETS:
        try:
            df = load_index(kind, sym)
        except Exception as e:
            print(f'[跳过] {name}: {str(e)[:80]}')
            continue
        n = len(df)
        years = int(df.index.year.max() - df.index.year.min())
        ann = annual_stats(df)
        ann.insert(0, 'market', name)
        pool = pd.concat([pool, ann], ignore_index=True)
        prof = year_phase_profile(df, det)
        for y, p in prof.items():
            phase_all[(name, int(y))] = p
        loaded.append((name, n, years, len(ann)))
        print(f'[载入] {name}: {n}帧 {years}年 {len(ann)}个年观测')

    if pool.empty:
        print('无任何市场数据')
        return

    pool = market_relative(pool)
    # 相位宽表: index=(market,year), 转回按 market 合并
    phase = {}
    for (mk, y), p in phase_all.items():
        phase.setdefault(mk, {})[y] = p

    markets = [m for m, _, _, _ in loaded]
    print(f'\n有效市场: {len(markets)} 个, 年观测合计 {len(pool)}')
    run_checks(pool, None, '合并检验')
    # 相位检验 (单独传宽表)
    print(f'\n{"=" * 74}')
    print(f'  五形相位饱和占比检验 (跨市场合并)')
    print(f'{"=" * 74}')
    ann_idx = pool.set_index(['market', 'year'])
    for ph in ('火', '水', '木', '金', '土'):
        rows = []
        for mk in markets:
            if mk not in phase:
                continue
            for y, p in phase[mk].items():
                if p is None or ph not in p or (mk, y) not in ann_idx.index:
                    continue
                rows.append({'market': mk, 'year': y, 'val': p[ph]})
        if len(rows) < 40:
            continue
        pp = pd.DataFrame(rows)
        ann2 = pool.set_index('year')
        print(f'\n  {ph}形饱和占比:')
        for key, kn in (('dz', '地支'), ('wx', '五行')):
            ppk = pp.copy()
            ppk[key] = ppk['year'].map(ann2[key].to_dict())
            grouped = [ppk.loc[ppk[key] == g, 'val'].values
                       for g in sorted(ppk[key].dropna().unique())]
            line(f'{kn}({len(grouped)}组)', kw_test(grouped))


if __name__ == '__main__':
    main()
