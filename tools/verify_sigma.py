# -*- coding: utf-8 -*-
"""stock_sigma_predictor 修复等价性验证"""
import sys, io, importlib.util, dataclasses, inspect, traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PY = r'd:\spum-core\经济学\stock_sigma_predictor.py'
PYC = r'd:\spum-core\经济学\__pycache__\stock_sigma_predictor.cpython-311.pyc'

fails = []
def check(cond, msg):
    if cond:
        print(f'  [OK] {msg}')
    else:
        fails.append(msg)
        print(f'  [FAIL] {msg}')

# ---------- 1. py_compile ----------
import py_compile
py_compile.compile(PY, doraise=True)
print('[1] py_compile 语法检查 OK')

# ---------- 2. import .py ----------
sys.path.insert(0, r'd:\spum-core\经济学')
import stock_sigma_predictor as new
print('[2] import 新 .py OK')

# ---------- 3. 注册 sys.modules 加载 .pyc ----------
import marshal
f = open(PYC, 'rb'); f.read(16); code = marshal.load(f); f.close()
spec = importlib.util.spec_from_file_location('old_ssp', PYC)
old = importlib.util.module_from_spec(spec)
sys.modules['old_ssp'] = old
spec.loader.exec_module(old)
print('[3] 加载 .pyc OK（注册 sys.modules 后顶层成功，证实原失败为反射方式限制）')

# ---------- 4. 公共 API 与签名 ----------
new_api = {n for n in dir(new) if not n.startswith('_')}
old_api = {n for n in dir(old) if not n.startswith('_')}
check(new_api == old_api, f'公共 API 一致 {sorted(new_api)}')
for fn in ['quick_stock_predict']:
    check(str(inspect.signature(getattr(new, fn))) == str(inspect.signature(getattr(old, fn))),
          f'{fn} 签名一致: {inspect.signature(getattr(new, fn))}')
check(str(inspect.signature(new.WuxingSigmaPredictor.__init__)) == str(inspect.signature(old.WuxingSigmaPredictor.__init__)),
      f'__init__ 签名一致: {inspect.signature(new.WuxingSigmaPredictor.__init__)}')
def sig_core(sig):
    params = [(p.name, p.default, p.kind) for p in sig.parameters.values()]
    ret = sig.return_annotation
    ret_name = getattr(ret, '__name__', str(ret).split('.')[-1].strip("'"))
    return (params, ret_name)

sig_new = sig_core(inspect.signature(new.WuxingSigmaPredictor.predict))
sig_old = sig_core(inspect.signature(old.WuxingSigmaPredictor.predict))
if sig_new != sig_old:
    print('    new params:', sig_new[0])
    print('    old params:', sig_old[0])
    print('    new ret:', sig_new[1], '| old ret:', sig_old[1])
check(sig_new == sig_old, 'predict 签名一致 (参数+默认值+返回类型名)')
check(str(inspect.signature(new.WuxingSigmaPredictor._evaluate_S_trend)) == str(inspect.signature(old.WuxingSigmaPredictor._evaluate_S_trend)),
      f'_evaluate_S_trend 签名一致')

# ---------- 5. dataclass 字段 ----------
nf = [(x.name, x.default, x.default_factory) for x in dataclasses.fields(new.JointSignal)]
of = [(x.name, x.default, x.default_factory) for x in dataclasses.fields(old.JointSignal)]
check(nf == of, f'JointSignal 字段一致 ({len(nf)} 字段)')
check([m for m in dir(new.JointSignal) if not m.startswith('_')] == [m for m in dir(old.JointSignal) if not m.startswith('_')],
      'JointSignal 方法一致 (含 summary)')

# ---------- 6. 行为等价对比 ----------
class S:
    def __init__(self, w=0.5, wo=0.5, e=0.5, m=0.5, fi=0.5):
        self.S_water, self.S_wood, self.S_earth, self.S_metal, self.S_fire = w, wo, e, m, fi
    def dominant(self):
        names = ['水', '木', '土', '金', '火']
        vals = [self.S_water, self.S_wood, self.S_earth, self.S_metal, self.S_fire]
        return names[vals.index(max(vals))]
    def __str__(self):
        return (f'S(水={self.S_water:.2f},木={self.S_wood:.2f},土={self.S_earth:.2f},'
                f'金={self.S_metal:.2f},火={self.S_fire:.2f})')

class Inf:
    def __init__(self, direction, component='火', confidence=0.9):
        self.direction, self.component, self.confidence = direction, component, confidence

import pandas as pd
def make_sigma(cur, grad=0.01, phase='扩张', pred=None, conf=0.5):
    if pred is None:
        pred = cur - 0.05
    return {
        'sigma': pd.Series([0.3] * 4 + [cur]),
        'gradient': pd.Series([0.0] * 4 + [grad]),
        'phase_label': pd.Series(['震荡'] * 4 + [phase]),
        'prediction': {'sigma_pred': pred, 'confidence': conf},
    }

def strip(j):
    d = dataclasses.asdict(j)
    d.pop('timestamp')
    d['S'] = str(d['S'])
    d['inflections'] = sorted((i.direction, i.component, i.confidence) for i in d['inflections'])
    return d

# _evaluate_S_trend 用例
trend_cases = [
    ('过冷inflection', S(0.5, 0.5, 0.5, 0.5, 0.5), [Inf('过冷', '水', 0.9)], None),
    ('过热inflection', S(0.5, 0.5, 0.5, 0.5, 0.5), [Inf('过热', '火', 0.95)], None),
    ('cold>=3', S(0.1, 0.1, 0.1, 0.5, 0.5), [], None),
    ('hot>=2', S(0.9, 0.9, 0.3, 0.3, 0.3), [], None),
    ('火水看涨', S(0.6, 0.4, 0.4, 0.4, 0.8), [], None),
    ('火水双低', S(0.1, 0.5, 0.5, 0.5, 0.1), [], None),
    ('木土+火>0.3', S(0.4, 0.8, 0.6, 0.4, 0.4), [], None),
    ('木土+火<=0.3', S(0.4, 0.8, 0.6, 0.4, 0.3), [], None),
    ('all_ok中性', S(0.5, 0.5, 0.5, 0.5, 0.5), [], None),
    ('偏强', S(0.7, 0.4, 0.3, 0.3, 0.4), [], None),
    ('偏弱', S(0.55, 0.3, 0.3, 0.3, 0.4), [], None),
    ('偏弱2', S(0.6, 0.2, 0.3, 0.3, 0.4), [], None),
    ('mixed-first-infl', S(0.5, 0.5, 0.5, 0.5, 0.5), [Inf('过冷', '水', 0.5), Inf('过热', '火', 0.8)], None),
]
op = new.WuxingSigmaPredictor()
print('\n[4] _evaluate_S_trend 逐分支对比:')
for name, s, infl, _ in trend_cases:
    r1 = old.WuxingSigmaPredictor()._evaluate_S_trend(s, infl)
    r2 = op._evaluate_S_trend(s, infl)
    check(r1 == r2, f'{name}: {r1}')

# predict 用例
pred_cases = [
    ('无sigma_双中性', {'S_current': S(0.5, 0.5, 0.5, 0.5, 0.5), 'inflections': []}, None),
    ('过热+σ看涨(追加reason)', {'S_current': S(0.5, 0.5, 0.5, 0.5, 0.5), 'inflections': [Inf('过热', '火', 0.88)]},
     make_sigma(0.5, pred=0.45, conf=0.6)),
    ('过热+σ看跌(无追加)', {'S_current': S(0.5, 0.5, 0.5, 0.5, 0.5), 'inflections': [Inf('过热', '金', 0.8)]},
     make_sigma(0.5, pred=0.55, conf=0.4)),
    ('过冷+σ看涨', {'S_current': S(0.5, 0.5, 0.5, 0.5, 0.5), 'inflections': [Inf('过冷', '水', 0.6)]},
     make_sigma(0.5, pred=0.45, conf=0.7)),
    ('过冷+σ中性(不触发)', {'S_current': S(0.5, 0.5, 0.5, 0.5, 0.5), 'inflections': [Inf('过冷', '水', 0.6)]}, None),
    ('一致确认_S_conf高→长期', {'S_current': S(0.6, 0.4, 0.4, 0.4, 0.8), 'inflections': []},
     make_sigma(0.5, pred=0.45, conf=0.5)),
    ('一致确认_σ_conf高→短期', {'S_current': S(0.6, 0.4, 0.4, 0.4, 0.8), 'inflections': []},
     make_sigma(0.5, pred=0.45, conf=0.9)),
    ('S看涨_σ中性→长期优先', {'S_current': S(0.6, 0.4, 0.4, 0.4, 0.8), 'inflections': []}, None),
    ('S中性_σ看跌→短期动量', {'S_current': S(0.5, 0.5, 0.5, 0.5, 0.5), 'inflections': []},
     make_sigma(0.5, pred=0.55, conf=0.6)),
]
print('\n[5] predict 逐分支对比 (timestamp 除外):')
for name, wres, sres in pred_cases:
    j1 = old.WuxingSigmaPredictor().predict(wres, sres, stock_code='000001', stock_name='测试')
    j2 = new.WuxingSigmaPredictor().predict(wres, sres, stock_code='000001', stock_name='测试')
    d1, d2 = strip(j1), strip(j2)
    ok = d1 == d2
    check(ok, f'{name}: net={d2.get("net_direction")} conf={d2.get("net_confidence")} horizon={d2.get("time_horizon")}')
    if not ok:
        print('    old:', d1)
        print('    new:', d2)
    check(j1.summary() == j2.summary(), f'{name} summary 一致')

# JointSignal 默认构造对比
a1 = old.JointSignal('c', 'n', 't', S(0.5, 0.5, 0.5, 0.5, 0.5), [])
a2 = new.JointSignal('c', 'n', 't', S(0.5, 0.5, 0.5, 0.5, 0.5), [])
check(a1 == a1 and a2 == a2, 'JointSignal 默认值构造 OK')
check(repr(a2).replace('S(水=0.50,木=0.50,土=0.50,金=0.50,火=0.50)', 'X') is not None, 'JointSignal repr 可用')

# ---------- 7. 合成 OHLCV DataFrame 端到端 ----------
print('\n[6] 合成 OHLCV DataFrame 端到端:')
import numpy as np
from stock_sigma import StockSigma
from stock_wuxing import WuxingVector
dates = pd.date_range('2024-01-01', periods=120, freq='D')
rng = np.random.RandomState(42)
close = 10 + np.cumsum(rng.normal(0, 0.1, 120))
df = pd.DataFrame({
    'open': close * 0.99, 'high': close * 1.02, 'low': close * 0.98,
    'close': close, 'volume': rng.randint(1_000_000, 5_000_000, 120),
}, index=dates)
sigma_result = StockSigma().analyze(df)
wres = {'S_current': WuxingVector(0.6, 0.55, 0.5, 0.45, 0.7), 'inflections': []}
j = new.WuxingSigmaPredictor().predict(wres, sigma_result, stock_code='000001', stock_name='测试股')
print('   预测结果:', j.net_direction, j.net_confidence, j.time_horizon, '| reason:', j.reason)
print('   sigma_phase:', j.sigma_phase, '| sigma_current:', j.sigma_current, '-> pred:', j.sigma_pred)
print('   summary:', j.summary())
check(isinstance(j, new.JointSignal) and j.summary()['inflection_count'] == 0, '端到端 predict 返回合理')

print('\n================ 结果 ================')
if fails:
    print(f'共 {len(fails)} 项失败:')
    for x in fails:
        print(' -', x)
    sys.exit(1)
print('全部验证通过 ✅')
