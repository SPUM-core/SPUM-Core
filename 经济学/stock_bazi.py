'''
股票先天八字五行向量构建
========================

核心思想 (SPUM 桥接 + 传统干支):
  股票的"出生时刻"= 上市时刻。像人的生辰八字一样, 上市时刻的四柱干支
  (年柱/月柱/日柱/时柱) 构成股票的"先天八字"——上市帧的相位编码。

流程:
  1. 上市日期(公历) + 统一假设上市时刻 09:30
  2. 换算真太阳时: 均时差(Equation of Time) + 经度修正(深交所 114.06°E, 基准 120°E)
  3. lunar_python 排四柱八字 (年/月/日/时)
  4. 先天五行向量 S_bazi = (水, 木, 火, 土, 金), 两套版本:
     - 明字版 (默认列 水/木/火/土/金): 四柱 8 字等权计数归一化
     - 藏干版 (后缀 _hide 列): 天干权重 1.0 + 地支藏干分级权重
       本气 0.7 / 中气 0.2 / 余气 0.1, 加权求和后归一化

股票池: 深交所主板, 按上市年份分层抽样 (1991~2023 每年 3 只 → ~99 只),
        行业信息齐全 (用于相关性分析的行业混杂控制)。

诚实标注:
  - 上市时刻无历史精确记录, 统一假设 09:30 (A股开盘时刻)
  - 真太阳时修正对本项目时柱影响小 (巳时范围 09:00-11:00, ±40 分钟修正不越界)
  - 藏干权重 (本气 0.7/中气 0.2/余气 0.1) 为传统子平通用简化,
    未采用按节气当权天数的精细方案; 未含纳音/旺衰

用法:
  python stock_bazi.py [--n-per-year 3] [--seed 42]
  输出: bazi_data.csv (含 明字版 5 列 + 藏干版 5 列)
'''
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '_lib'))

import argparse
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------- 干支五行 ----------
TG_WX = {'甲': '木', '乙': '木', '丙': '火', '丁': '火', '戊': '土',
         '己': '土', '庚': '金', '辛': '金', '壬': '水', '癸': '水'}
DZ_WX = {'子': '水', '丑': '土', '寅': '木', '卯': '木', '辰': '土',
         '巳': '火', '午': '火', '未': '土', '申': '金', '酉': '金',
         '戌': '土', '亥': '水'}
# 地支藏干 (顺序 = 本气, 中气, 余气)
DZ_HIDE = {'子': '癸', '丑': '己癸辛', '寅': '甲丙戊', '卯': '乙', '辰': '戊乙癸',
           '巳': '丙庚戊', '午': '丁己', '未': '己丁乙', '申': '庚壬戊',
           '酉': '辛', '戌': '戊辛丁', '亥': '壬甲'}
# 藏干分级权重 (本气 / 中气 / 余气)
HIDE_WEIGHTS = [0.7, 0.2, 0.1]
WX_ORDER = ['水', '木', '火', '土', '金']


def true_solar_time(year, month, day, hour, minute, second, longitude=114.06):
    '''真太阳时 = 平太阳时 + 均时差 + 经度修正 (基准 120°E = UTC+8)
    经度修正: 每 1° 东经 4 分钟; 均时差用常用近似公式 (±16 分钟量级)'''
    n = (datetime(year, month, day) - datetime(year, 1, 1)).days + 1
    b = 2 * math.pi / 365 * (n - 81)
    eot = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)  # 分钟
    lng_min = (longitude - 120.0) * 4.0  # 分钟
    t = datetime(year, month, day, hour, minute, second) + timedelta(minutes=eot + lng_min)
    return t


def make_bazi(year, month, day, hour, minute, second, longitude):
    '''返回 (八字四柱字符串, 明字版 S dict, 藏干版 S_hide dict)'''
    from lunar_python import Solar
    tst = true_solar_time(year, month, day, hour, minute, second, longitude)
    solar = Solar.fromYmdHms(tst.year, tst.month, tst.day, tst.hour, tst.minute, tst.second)
    lunar = solar.getLunar()
    ec = lunar.getEightChar()
    pillars = [ec.getYear(), ec.getMonth(), ec.getDay(), ec.getTime()]  # 如 '庚午'

    # ---- 明字版: 8 字等权 ----
    chars = [ch for p in pillars for ch in p]
    count = {k: 0 for k in WX_ORDER}
    for ch in chars:
        wx = TG_WX.get(ch, DZ_WX.get(ch))
        if wx:
            count[wx] += 1
    total = sum(count.values())
    sbazi = {k: v / total for k, v in count.items()} if total else count

    # ---- 藏干版: 天干 1.0 + 地支藏干分级权重 (本气0.7/中气0.2/余气0.1) ----
    hides = [ec.getYearHideGan(), ec.getMonthHideGan(),
             ec.getDayHideGan(), ec.getTimeHideGan()]
    wsum = {k: 0.0 for k in WX_ORDER}
    for p, h in zip(pillars, hides):
        tg = p[0]
        if tg in TG_WX:
            wsum[TG_WX[tg]] += 1.0          # 天干权重 1.0
        dz = p[1]
        for j, ch in enumerate(h):          # 藏干顺序 = 本气→中气→余气
            wx = TG_WX.get(ch)
            if wx and j < len(HIDE_WEIGHTS):
                wsum[wx] += HIDE_WEIGHTS[j]
    wtotal = sum(wsum.values())
    sbazi_h = {k: v / wtotal for k, v in wsum.items()} if wtotal else wsum

    lunar_str = lunar.toString()
    return {'bazi': '/'.join(pillars), 'lunar': lunar_str,
            'S': sbazi, 'S_hide': sbazi_h}


def build_universe(n_per_year=3, seed=42):
    import akshare as ak
    df = ak.stock_info_sz_name_code()
    df = df.rename(columns={'A股代码': 'code', 'A股简称': 'name',
                            'A股上市日期': 'list_date', '所属行业': 'industry'})
    df = df[['code', 'name', 'list_date', 'industry']]
    df['list_date'] = pd.to_datetime(df['list_date'])
    # 主板: 00 开头
    df = df[df['code'].str.startswith('00')].copy()
    df = df[(df['list_date'] >= '1991-01-01') & (df['list_date'] <= '2023-12-31')]
    df['year'] = df['list_date'].dt.year
    rng = np.random.default_rng(seed)
    picked = []
    for y, g in df.groupby('year'):
        if len(g) <= n_per_year:
            picked.append(g)
        else:
            picked.append(g.sample(n_per_year, random_state=int(rng.integers(0, 100000))))
    return pd.concat(picked).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-year', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    uni = build_universe(args.n_per_year, args.seed)
    print(f'股票池: {len(uni)} 只 (深交所主板, 上市年份 {uni["year"].min()}~{uni["year"].max()})')

    rows = []
    for _, r in uni.iterrows():
        ld = r['list_date']
        res = make_bazi(ld.year, ld.month, ld.day, 9, 30, 0, longitude=114.06)
        row = {
            'code': r['code'], 'name': r['name'], 'industry': r['industry'],
            'list_date': ld.strftime('%Y-%m-%d'), 'lunar': res['lunar'],
            'bazi': res['bazi'],
        }
        row.update(res['S'])                          # 明字版: 水/木/火/土/金
        row.update({f'{k}_hide': v for k, v in res['S_hide'].items()})  # 藏干版
        rows.append(row)
    out = pd.DataFrame(rows)
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bazi_data.csv')
    out.to_csv(outfile, index=False, encoding='utf-8-sig')
    print(f'已写入 {outfile}')
    # 五行向量分布概览 (两套版本)
    for tag, cols in [('明字版', WX_ORDER), ('藏干版', [f'{w}_hide' for w in WX_ORDER])]:
        print(f'\n五行向量统计 {tag} (全池):')
        for wx in WX_ORDER:
            c = f'{wx}_hide' if tag == '藏干版' else wx
            print(f'  {wx}: 均值 {out[c].mean():.3f} 范围 [{out[c].min():.2f}, {out[c].max():.2f}]')
    print('\n示例 (前 8 只):')
    cols = ['code', 'name', 'list_date', 'lunar', 'bazi'] + WX_ORDER + [f'{w}_hide' for w in WX_ORDER]
    print(out[cols].head(8).to_string(index=False))


if __name__ == '__main__':
    main()
