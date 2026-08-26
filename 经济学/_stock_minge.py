# Source Generated with Decompyle++
# File: _stock_minge.cpython-311.pyc (Python 3.11)

'''
SPUM 奇门遁甲 × 股票命格分析
=============================

以上市时间（转农历）为先天股票命格，归类同命格股票，分析波动规律。

核心逻辑:
  1. 获取A股列表 + 上市日期
  2. 公历 → 农历转换 (zhdate)
  3. 计算天干地支 (年柱+月柱+日柱)
  4. 按日柱(命格)分组
  5. 下载各组价格数据 → 分析波动规律
'''
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime, date
from collections import defaultdict
from zhdate import ZhDate
TIANGAN = [
    '甲',
    '乙',
    '丙',
    '丁',
    '戊',
    '己',
    '庚',
    '辛',
    '壬',
    '癸']
DIZHI = [
    '子',
    '丑',
    '寅',
    '卯',
    '辰',
    '巳',
    '午',
    '未',
    '申',
    '酉',
    '戌',
    '亥']
SHENGXIAO = [
    '鼠',
    '牛',
    '虎',
    '兔',
    '龙',
    '蛇',
    '马',
    '羊',
    '猴',
    '鸡',
    '狗',
    '猪']
MONTH_BRANCH = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    0,
    1]
MONTH_BRANCH_NAME = [
    '寅',
    '卯',
    '辰',
    '巳',
    '午',
    '未',
    '申',
    '酉',
    '戌',
    '亥',
    '子',
    '丑']

def year_ganzhi(year: int):
    '''年柱: (天干, 地支)'''
    return (TIANGAN[(year - 4) % 10], DIZHI[(year - 4) % 12])


def month_ganzhi(year: int, month: int):
    '''月柱: (天干, 地支) — month is lunar month (1-12)'''
    yg = (year - 4) % 10
    stem = ((yg % 5) * 2 + month) % 10
    branch = (month + 1) % 12
    return (TIANGAN[stem - 1], DIZHI[branch])


def day_ganzhi(d: date):
    '''日柱: (天干, 地支)
    参考: 1900-01-01 = 甲戌日 (天干=0, 地支=10)
    '''
    ref = date(1900, 1, 1)
    days = (d - ref).days
    stem = days % 10
    branch = (days + 10) % 12
    return (TIANGAN[stem], DIZHI[branch])


def ganzhi_for_listing(listing_date: date):
    '''完整四柱计算 (年柱, 月柱, 日柱)'''
    (yg, yz) = year_ganzhi(listing_date.year)
    from datetime import datetime as dt_module
    lunar = ZhDate.from_datetime(dt_module.combine(listing_date, dt_module.min.time()))
    mg = (((((listing_date.year - 4) % 10) % 5) * 2) + lunar.lunar_month) % 10
    mz = DIZHI[(lunar.lunar_month + 1) % 12]
    (dg, dz) = day_ganzhi(listing_date)
    return {
        'year': f'{yg}{yz}',
        'month': f'{TIANGAN[mg]}{mz}',
        'day': f'{dg}{dz}',
        'lunar': f'{lunar.lunar_year}年{lunar.lunar_month}月{lunar.lunar_day}日',
        'shengxiao': SHENGXIAO[(listing_date.year - 4) % 12],
        'year_stem': yg,
        'year_branch': yz,
        'day_stem': dg,
        'day_branch': dz }

STOCKS = [
    ('sz300750', '宁德时代', '2018-06-11'),
    ('sz002415', '海康威视', '2010-05-28'),
    ('sz300124', '汇川技术', '2010-09-28'),
    ('sz002475', '立讯精密', '2010-09-15'),
    ('sz000725', '京东方A', '2001-01-12'),
    ('sh688981', '中芯国际', '2020-07-16'),
    ('sh603501', '韦尔股份', '2017-05-04'),
    ('sh603986', '兆易创新', '2016-08-18'),
    ('sz002230', '科大讯飞', '2008-05-12'),
    ('sz300033', '同花顺', '2009-12-25'),
    ('sh600570', '恒生电子', '2003-12-16'),
    ('sz300059', '东方财富', '2010-03-19'),
    ('sh600519', '贵州茅台', '2001-08-27'),
    ('sh600887', '伊利股份', '1996-03-12'),
    ('sz000333', '美的集团', '2013-09-18'),
    ('sz000651', '格力电器', '1996-11-18'),
    ('sz000858', '五粮液', '1998-04-27'),
    ('sh603288', '海天味业', '2014-02-11'),
    ('sh601888', '中国中免', '2009-10-15'),
    ('sz002304', '洋河股份', '2009-11-06'),
    ('sh600809', '山西汾酒', '1994-01-06'),
    ('sh600882', '妙可蓝多', '1995-12-06'),
    ('sh600276', '恒瑞医药', '2000-10-18'),
    ('sz300760', '迈瑞医疗', '2018-10-16'),
    ('sh603259', '药明康德', '2018-05-08'),
    ('sz000538', '云南白药', '1993-12-15'),
    ('sz300015', '爱尔眼科', '2009-10-30'),
    ('sh600196', '复星医药', '1998-08-07'),
    ('sz002007', '华兰生物', '2004-06-25'),
    ('sh600085', '同仁堂', '1997-06-25'),
    ('sh600036', '招商银行', '2002-04-09'),
    ('sh601318', '中国平安', '2007-03-01'),
    ('sh601398', '工商银行', '2006-10-27'),
    ('sh600030', '中信证券', '2003-01-06'),
    ('sh601166', '兴业银行', '2007-02-05'),
    ('sh600016', '民生银行', '2000-12-19'),
    ('sh600031', '三一重工', '2003-07-03'),
    ('sh600309', '万华化学', '2001-01-05'),
    ('sh601899', '紫金矿业', '2008-04-25'),
    ('sh600585', '海螺水泥', '2002-02-07'),
    ('sz000338', '潍柴动力', '2007-04-30'),
    ('sh600690', '海尔智家', '1993-11-19'),
    ('sh601012', '隆基绿能', '2012-04-11'),
    ('sz300274', '阳光电源', '2011-11-02'),
    ('sz002594', '比亚迪', '2011-06-30'),
    ('sh601633', '长城汽车', '2011-09-28'),
    ('sh600104', '上汽集团', '1997-11-25'),
    ('sz002460', '赣锋锂业', '2010-08-10'),
    ('sh600438', '通威股份', '2004-03-02'),
    ('sz002129', '中环股份', '2007-04-20'),
    ('sz000002', '万科A', '1991-01-29'),
    ('sh600048', '保利发展', '2006-07-31'),
    ('sh600900', '长江电力', '2003-11-18'),
    ('sh601668', '中国建筑', '2009-07-29'),
    ('sz002352', '顺丰控股', '2010-02-05'),
    ('sh601857', '中国石油', '2007-11-05'),
    ('sh600941', '中国移动', '2022-01-05'),
    ('sz001979', '招商蛇口', '2015-12-30')]

def load_stock_data(sym, max_retries=2):
    '''从TX源加载股票数据，返回(df, listing_date)'''
    import akshare as ak
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist_tx(symbol=sym)
            if df is not None and len(df) > 20:
                df = df.sort_values('date')
                df.index = pd.to_datetime(df['date'])
                if 'volume' not in df.columns and 'amount' in df.columns:
                    df['volume'] = df['amount']
                listing = datetime.strptime(df['date'].iloc[0], '%Y-%m-%d').date()
                return (df, listing)
        except:
            time.sleep(2)
    return (None, None)


def calc_volatility_metrics(df):
    '''计算一组波动指标'''
    if df is None or len(df) < 20:
        return {}
    close = df['close'].values
    returns = np.diff(close) / close[:-1]
    daily_vol = float(np.std(returns))
    annual_vol = float(np.std(returns) * np.sqrt(252))
    max_drawdown = float((close.min() - close.max()) / close.max())
    avg_return = float(np.mean(returns))
    if np.std(returns) > 0:
        sharpe_ratio = float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    else:
        sharpe_ratio = 0
    skewness = float(pd.Series(returns).skew())
    kurtosis = float(pd.Series(returns).kurtosis())
    return {
        'daily_vol': daily_vol,
        'annual_vol': annual_vol,
        'max_drawdown': max_drawdown,
        'avg_return': avg_return,
        'sharpe_ratio': sharpe_ratio,
        'skewness': skewness,
        'kurtosis': kurtosis }


def calc_group_correlation(group_dfs):
    '''计算组内股票的平均相关系数'''
    if len(group_dfs) < 2:
        return 0
    daily_returns = []
    for (name, df) in group_dfs:
        if df is None or len(df) < 50:
            continue
        r = pd.Series(np.diff(df['close'].values) / df['close'].values[:-1], index = df.index[1:], name = name)
        daily_returns.append(r)
    if len(daily_returns) < 2:
        return 0
    ret_df = pd.concat(daily_returns, axis = 1).dropna()
    if ret_df.shape[1] < 2 or ret_df.shape[0] < 10:
        return 0
    corr = ret_df.corr().values
    n = corr.shape[0]
    upper_tri = corr[np.triu_indices(n, k = 1)]
    if len(upper_tri) > 0:
        return float(np.mean(upper_tri))
    else:
        return 0


def run_phase_analysis(df, name):
    '''对个股运行五形相位回测'''
    import sys
    sys.path.insert(0, 'f:/spum-core/经济学')
    from daily_phase import backtest
    try:
        r = backtest(df, window = 120, min_confidence = 0.3)
        up = r['dirs'].get('↗', { })
        dn = r['dirs'].get('↘', { })
        return {
            'phase_acc': r['accuracy'],
            'phase_signals': r['signals'],
            'phase_up_acc': up.get('acc', 0),
            'phase_dn_acc': dn.get('acc', 0),
            'phase_up_n': up.get('n', 0),
            'phase_dn_n': dn.get('n', 0) }
    except:
        return { }


def main():
    print('======================================================================')
    print('SPUM 奇门遁甲 × 股票命格分析')
    print('======================================================================')
    records = []
    total = len(STOCKS)
    for (i, (sym, name, listing_str)) in enumerate(STOCKS):
        listing_date = datetime.strptime(listing_str, '%Y-%m-%d').date()
        gz = ganzhi_for_listing(listing_date)
        (df, actual_listing) = load_stock_data(sym)
        if actual_listing and actual_listing < listing_date:
            gz = ganzhi_for_listing(actual_listing)
            listing_used = actual_listing
        else:
            listing_used = listing_date
        if df is not None:
            metrics = calc_volatility_metrics(df)
        else:
            metrics = { }
        if df is not None:
            phase = run_phase_analysis(df, name)
        else:
            phase = { }
        rec = {
            'sym': sym,
            'name': name,
            'listing': listing_used,
            'lunar': gz['lunar'],
            'year_gz': gz['year'],
            'month_gz': gz['month'],
            'day_gz': gz['day'],
            'year_stem': gz['year_stem'],
            'day_stem': gz['day_stem'],
            'day_branch': gz['day_branch'],
            'shengxiao': gz['shengxiao'],
            **metrics,
            **phase }
        records.append(rec)
        pg = f'[{i + 1}/{total}]'
        status = f'{name:8s} | {listing_used} | 日柱={gz["day"]} | 年={gz["year"]}'
        if metrics:
            vol_str = f'vol={metrics.get("daily_vol", 0):.3f}'
        else:
            vol_str = 'no data'
        print(f'{pg} {status} | {vol_str}')
        sys.stdout.flush()
    df_all = pd.DataFrame(records)
    print('\n======================================================================')
    print('按日柱(命格)分组 — 波动率统计')
    print('======================================================================')
    day_groups = df_all.groupby('day_gz')
    group_stats = []
    for (day_gz, group) in day_groups:
        n = len(group)
        vols = group['daily_vol'].dropna().values
        if len(vols) == 0:
            continue
        mean_vol = np.mean(vols)
        std_vol = np.std(vols)
        stocks_list = ', '.join(group['name'].tolist())
        print(f'\n【{day_gz}】({n}只)')
        print(f'  平均日波动率: {mean_vol:.4f} ± {std_vol:.4f}')
        print(f'  年化波动率:   {mean_vol * np.sqrt(252):.4f}')
        print(f'  成分: {stocks_list[:120]}')
        group_stats.append({
            'day_gz': day_gz,
            'count': n,
            'mean_vol': mean_vol,
            'std_vol': std_vol })
    print('\n======================================================================')
    print('按年生肖分组 — 波动率统计')
    print('======================================================================')
    sx_groups = df_all.groupby('shengxiao')
    for (sx, group) in sx_groups:
        n = len(group)
        vols = group['daily_vol'].dropna().values
        if len(vols) == 0:
            continue
        mean_vol = np.mean(vols)
        stocks_list = ', '.join(group['name'].tolist())
        print(f'\n【{sx}】({n}只)')
        print(f'  平均日波动率: {mean_vol:.4f} (年化{mean_vol * np.sqrt(252):.4f})')
        print(f'  成分: {stocks_list[:120]}')
    print('\n======================================================================')
    print('五形相位准确率 vs 日柱命格')
    print('======================================================================')
    phase_cols = ['phase_acc', 'phase_up_acc', 'phase_dn_acc']
    phase_summary = df_all.groupby('day_gz')[phase_cols].mean()
    print(phase_summary.to_string())
    print('\n======================================================================')
    print('同命格组内相关系数')
    print('======================================================================')
    top_groups = df_all.groupby('day_gz').filter(lambda x: len(x) >= 3)
    if len(top_groups) > 0:
        for (day_gz, group) in top_groups.groupby('day_gz'):
            g_dfs = []
            for (_, row) in group.iterrows():
                (df_stock, _) = load_stock_data(row['sym'])
                if df_stock is not None:
                    g_dfs.append((row['name'], df_stock.tail(500)))
            corr = calc_group_correlation(g_dfs)
            print(f'  {day_gz} ({len(g_dfs)}只): 组内平均相关系数 = {corr:.4f}')
    print('\n======================================================================')
    print('日柱命格波动率排名 (高→低)')
    print('======================================================================')
    gs_df = pd.DataFrame(group_stats).sort_values('mean_vol', ascending = False)
    print(f"{'日柱':<6} {'数量':<6} {'平均日波动率':<12} {'年化波动率':<12} {'标准差':<10}")
    print('--------------------------------------------------')
    for (_, r) in gs_df.iterrows():
        print(f'{r["day_gz"]:<6} {r["count"]:<6} {r["mean_vol"]:.4f}       {r["mean_vol"] * np.sqrt(252):.4f}       {r["std_vol"]:.4f}')
    output_path = 'f:/spum-core/经济学/mingge_results.json'
    df_all.to_json(output_path, orient = 'records', force_ascii = False)
    print(f'\n结果已保存: {output_path}')
    print(f'总股票数: {len(df_all)}, 覆盖日柱: {len(group_stats)}')

if __name__ == '__main__':
    main()
