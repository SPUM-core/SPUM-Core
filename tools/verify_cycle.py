# -*- coding: utf-8 -*-
"""spum_cycle.py 修复等价性验证脚本"""
import sys, os, io, contextlib
from pathlib import Path

ECON = Path(__file__).resolve().parent.parent / '经济学'
sys.path.insert(0, str(ECON))
os.chdir(ECON)

import py_compile
py_compile.compile(str(ECON / 'spum_cycle.py'), doraise=True)
print('[PASS] py_compile')

import importlib.util
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

old = load('o', str(ECON / '__pycache__' / 'spum_cycle.cpython-311.pyc'))
new = load('n', str(ECON / 'spum_cycle.py'))
print('[PASS] import both')

# 1. 公共 API 一致
api_old = set(n for n in dir(old) if not n.startswith('_'))
api_new = set(n for n in dir(new) if not n.startswith('_'))
assert api_old == api_new, f'API mismatch: old-only={api_old-api_new} new-only={api_new-api_old}'
print(f'[PASS] public API identical ({len(api_old)} names)')

# 2. dataclass 字段一致
import dataclasses
for clsname in ('PhaseState', 'CycleProfile'):
    fo = [(f.name, getattr(f.type, '__name__', str(f.type)), f.default is dataclasses.MISSING, f.default_factory is not dataclasses.MISSING)
          for f in dataclasses.fields(getattr(old, clsname))]
    fn = [(f.name, getattr(f.type, '__name__', str(f.type)), f.default is dataclasses.MISSING, f.default_factory is not dataclasses.MISSING)
          for f in dataclasses.fields(getattr(new, clsname))]
    assert fo == fn, f'{clsname} fields mismatch: {fo} vs {fn}'
print('[PASS] dataclass fields identical')

# 3. 常量一致
assert old.DEFAULT_INDUSTRY == new.DEFAULT_INDUSTRY
assert old.INDUSTRY_MAP == new.INDUSTRY_MAP
assert old.STD_INDICATORS == new.STD_INDICATORS
assert set(old.TRANSFORMS) == set(new.TRANSFORMS)
print('[PASS] constants identical')

# 4. 函数签名一致
import inspect
def norm_sig(s):
    return s.replace('o.', '').replace('n.', '').replace('m.', '')

for fn in ('analyze_company', 'print_cycle_report'):
    so = norm_sig(str(inspect.signature(getattr(old, fn))))
    sn = norm_sig(str(inspect.signature(getattr(new, fn))))
    assert so == sn, f'{fn} sig mismatch: {so} vs {sn}'
for clsname, meth in (('CyclePhaseDetector', '__init__'), ('CyclePhaseDetector', 'extract'),
                      ('CyclePhaseDetector', 'profile_from_indicators'), ('CyclePhaseDetector', 'profile_from_market'),
                      ('CyclePhaseDetector', '_default_profile'), ('StockPhaseProjector', 'project')):
    so = norm_sig(str(inspect.signature(getattr(getattr(old, clsname), meth))))
    sn = norm_sig(str(inspect.signature(getattr(getattr(new, clsname), meth))))
    assert so == sn, f'{clsname}.{meth} sig mismatch: {so} vs {sn}'
print('[PASS] signatures identical')

# 5. 合成 OHLCV 数据
import numpy as np
import pandas as pd

def make_ohlcv(seed, n, start=10.0, drift=0.0003, vol=0.02):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    open_ = np.concatenate([[start], close[:-1]]) * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    vol_ = rng.uniform(1e6, 1e7, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': vol_})

def close_dict(m):
    return {
        'fire': (m.fire.value, m.fire.label),
        'water': (m.water.value, m.water.label),
        'wood': (m.wood.value, m.wood.label),
        'metal': (m.metal.value, m.metal.label),
        'earth': (m.earth.value, m.earth.label),
    }

fails = 0
def cmp_profile(name, po, pn):
    global fails
    do, dn = close_dict(po), close_dict(pn)
    for k in do:
        if abs(do[k][0] - dn[k][0]) > 1e-9 or do[k][1] != dn[k][1]:
            print(f'  FAIL {name}: {k} {do[k]} vs {dn[k]}')
            fails += 1

# 6. profile_from_market 对比（多种行情场景）
for seed in range(8):
    df = make_ohlcv(seed, 300)
    df_short = make_ohlcv(seed, 40)
    df_mid = make_ohlcv(seed, 200)
    for industry in ('tech', 'consumer', 'manufacturing', 'real_estate', 'unknown_xyz'):
        po = old.CyclePhaseDetector(industry).profile_from_market(df)
        pn = new.CyclePhaseDetector(industry).profile_from_market(df)
        cmp_profile(f'market-{seed}-{industry}', po, pn)
        po = old.CyclePhaseDetector(industry).profile_from_market(df_short)
        pn = new.CyclePhaseDetector(industry).profile_from_market(df_short)
        cmp_profile(f'market-short-{seed}-{industry}', po, pn)
        po = old.CyclePhaseDetector(industry).profile_from_market(df_mid)
        pn = new.CyclePhaseDetector(industry).profile_from_market(df_mid)
        cmp_profile(f'market-mid-{seed}-{industry}', po, pn)
print('[DONE] profile_from_market 对比')

# 7. extract / profile_from_indicators 对比
rng = np.random.default_rng(42)
dates = pd.date_range('2024-01-01', periods=10)
base_cols = {'主营业务收入增长率': None, '净利润增长率': None, '销售毛利率': None, '主营业务成本率': None,
             '资产负债率': None, '净资产收益率': None, '总资产周转率': None, '流动比率': None, '三项费用比重': None}
ind_ok = pd.DataFrame({c: rng.uniform(-20, 30, 10) if c != '流动比率' else rng.uniform(0.5, 3, 10)
                       for c in base_cols}, index=dates)
ind_ok.columns = [c + '(单位)' if i % 3 == 0 else c for i, c in enumerate(ind_ok.columns)]
ind_partial = ind_ok.copy()
ind_partial['销售毛利率(单位)'] = np.nan
ind_partial['主营业务成本率(单位)'] = rng.uniform(55, 80, 10)
ind_empty = pd.DataFrame(index=dates)
for name, ind in (('full', ind_ok), ('partial-cost', ind_partial), ('empty', ind_empty)):
    eo = old.CyclePhaseDetector('tech').extract(ind)
    en = new.CyclePhaseDetector('tech').extract(ind)
    assert set(eo) == set(en)
    for k in eo:
        if eo[k] != en[k]:
            print(f'  FAIL extract-{name}: {k} {eo[k]} vs {en[k]}')
            fails += 1
    for industry in ('tech', 'consumer', 'manufacturing', 'real_estate'):
        for pm in (0.0, 0.12):
            po = old.CyclePhaseDetector(industry).profile_from_indicators(eo, pm)
            pn = new.CyclePhaseDetector(industry).profile_from_indicators(en, pm)
            cmp_profile(f'indicators-{name}-{industry}-pm{pm}', po, pn)
print('[DONE] extract / profile_from_indicators 对比')

# 8. _default_profile 对比
for industry in ('tech', 'unknown_xyz'):
    cmp_profile('default-' + industry, old.CyclePhaseDetector(industry)._default_profile(),
                new.CyclePhaseDetector(industry)._default_profile())
print('[DONE] _default_profile 对比')

# 9. CycleProfile 方法对比（含边界）
cases = [
    dict(fire=0.9, water=0.4, wood=0.3, metal=0.2, earth=0.1),   # fire 饱和
    dict(fire=0.3, water=0.9, wood=0.4, metal=0.3, earth=0.2),   # water 饱和
    dict(fire=0.2, water=0.3, wood=0.88, metal=0.3, earth=0.3),  # wood 饱和
    dict(fire=0.2, water=0.3, wood=0.3, metal=0.9, earth=0.3),   # metal 饱和
    dict(fire=0.2, water=0.3, wood=0.3, metal=0.3, earth=0.9),   # earth 饱和
    dict(fire=0.2, water=0.8, wood=0.3, metal=0.3, earth=0.3),   # water 饱和
    dict(fire=0.7, water=0.6, wood=0.6, metal=0.6, earth=0.6),   # 火占优 d>=0.75
    dict(fire=0.65, water=0.64, wood=0.64, metal=0.64, earth=0.6),  # 火占优 d<0.75, 无缺位
    dict(fire=0.1, water=0.6, wood=0.7, metal=0.5, earth=0.65),  # 火缺位+土占优
    dict(fire=0.3, water=0.3, wood=0.7, metal=0.1, earth=0.3),   # 金缺位+木占优
    dict(fire=0.5, water=0.5, wood=0.5, metal=0.5, earth=0.5),   # 均等 -> 火占优(首个max)
    dict(fire=0.29, water=0.29, wood=0.29, metal=0.29, earth=0.29),  # 全缺位, dominant=火
]
for i, c in enumerate(cases):
    po = old.CycleProfile(**{k: old.PhaseState(v, 'x') for k, v in c.items()})
    pn = new.CycleProfile(**{k: new.PhaseState(v, 'x') for k, v in c.items()})
    assert po.dominant_phase() == pn.dominant_phase()
    assert po.saturated_phases() == pn.saturated_phases()
    assert po.absent_phases() == pn.absent_phases()
    assert po.predict_transition() == pn.predict_transition()
    assert po.summary() == pn.summary()
print('[PASS] CycleProfile methods 一致 (12 边界用例)')

# 10. PhaseState 方法
for v in (0.0, 0.29, 0.3, 0.59, 0.6, 0.84, 0.85, 1.0):
    a, b = old.PhaseState(v, '火'), new.PhaseState(v, '火')
    assert (a.is_saturated(), a.is_dominant(), a.is_absent()) == (b.is_saturated(), b.is_dominant(), b.is_absent())
print('[PASS] PhaseState methods 一致')

# 11. StockPhaseProjector.project 对比
for i, c in enumerate(cases):
    po = old.CycleProfile(**{k: old.PhaseState(v, 'x') for k, v in c.items()})
    pn = new.CycleProfile(**{k: new.PhaseState(v, 'x') for k, v in c.items()})
    ro = old.StockPhaseProjector().project(po)
    rn = new.StockPhaseProjector().project(pn)
    for k in ro:
        if isinstance(ro[k], float):
            assert abs(ro[k] - rn[k]) < 1e-9, f'project {k}: {ro[k]} vs {rn[k]}'
        else:
            assert ro[k] == rn[k], f'project {k}: {ro[k]} vs {rn[k]}'
print('[PASS] StockPhaseProjector.project 一致')

# 12. analyze_company 对比
for seed in (7, 99):
    df = make_ohlcv(seed, 300)
    for industry in ('tech', 'consumer'):
        # indicators 路径
        ro = old.analyze_company('600519', '贵州茅台', industry, df_market=df, indicators=ind_ok)
        rn = new.analyze_company('600519', '贵州茅台', industry, df_market=df, indicators=ind_ok)
        for k in ro:
            if isinstance(ro[k], dict) and k != 'profile':
                assert ro[k] == rn[k], f'analyze {k}: {ro[k]} vs {rn[k]}'
            elif k == 'profile':
                cmp_profile(f'analyze-profile-{seed}-{industry}', ro[k], rn[k])
            elif isinstance(ro[k], float):
                assert abs(ro[k] - rn[k]) < 1e-9
            elif isinstance(ro[k], list):
                assert ro[k] == rn[k]
            else:
                assert ro[k] == rn[k], f'analyze {k}: {ro[k]} vs {rn[k]}'
        # 纯行情路径
        ro = old.analyze_company('000001', '', industry, df_market=df)
        rn = new.analyze_company('000001', '', industry, df_market=df)
        assert ro['error'] == rn['error'] if 'error' in ro else True
        if 'error' not in ro:
            for k in ro:
                if k == 'profile':
                    cmp_profile(f'analyze-market-{seed}-{industry}', ro[k], rn[k])
                elif isinstance(ro[k], dict):
                    assert ro[k] == rn[k], f'analyze-market {k}'
        # 无数据路径
        ro = old.analyze_company('600000')
        rn = new.analyze_company('600000')
        assert ro == rn == {'error': '需要财务数据或行情数据'}
        # 短行情 (<60) -> _default_profile
        df_short = make_ohlcv(seed, 40)
        ro = old.analyze_company('600000', industry=industry, df_market=df_short)
        rn = new.analyze_company('600000', industry=industry, df_market=df_short)
        cmp_profile(f'analyze-short-{seed}-{industry}', ro['profile'], rn['profile'])
print('[DONE] analyze_company 对比')

# 13. print_cycle_report 对比（stdout 捕获）
buf_o, buf_n = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(buf_o):
    old.print_cycle_report(old.analyze_company('600519', '贵州茅台', 'tech', df_market=make_ohlcv(1, 300), indicators=ind_ok))
with contextlib.redirect_stdout(buf_n):
    new.print_cycle_report(new.analyze_company('600519', '贵州茅台', 'tech', df_market=make_ohlcv(1, 300), indicators=ind_ok))
assert buf_o.getvalue() == buf_n.getvalue(), f'print_cycle_report mismatch:\nOLD:\n{buf_o.getvalue()}\nNEW:\n{buf_n.getvalue()}'
print('[PASS] print_cycle_report 输出一致')
buf_o, buf_n = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(buf_o):
    old.print_cycle_report({'error': '测试错误'})
with contextlib.redirect_stdout(buf_n):
    new.print_cycle_report({'error': '测试错误'})
assert buf_o.getvalue() == buf_n.getvalue()
print('[PASS] print_cycle_report error 分支一致')

assert fails == 0, f'{fails} profile comparisons FAILED'
print(f'\n===== ALL PASS ({fails} failures) =====')
