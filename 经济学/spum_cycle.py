# Source Generated with Decompyle++
# File: spum_cycle.cpython-311.pyc (Python 3.11)

'''
SPUM 商业周期相位分析器
=======================

核心主义 — 从 ⟨P, ε⟩ 公理推导，不用统计拟合。

一家公司是 G_econ 中的一个子图。它的发展不是"价格曲线"，
而是子图的拓扑相位沿 火→水→木→金→土→火 的离散帧演化。

五形相位的帧间流转：
  火形 ∇σ → 驱动水形链输送土形储备
  水形链 → 将土形输送到木形生长前沿
  木形吸收 → 环基数 μ 上升 → 骨架扩展 → 木僵
  木僵 → 金形修剪激活 → 拆除桥边
  金形释放 → 回归土形储备 → 新的 σ 不均匀 → 火形再生

每个公司子图在任意帧都处在这五个相位中的某个占优位置。
占优相位的饱和（D_max）就是拐点。

行业适配: 五形耦合链是通用拓扑公理, 不随行业改变。
          行业差异仅在于哪种财务指标能最好地揭示相位占优度。
          详见 industry_mapping.py。
'''
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
from industry_mapping import INDUSTRY_MAP, STD_INDICATORS, TRANSFORMS, DEFAULT_INDUSTRY


@dataclass
class PhaseState:
    '''
    单相位的拓扑状态。

    值域:
      [0.0, 0.3)  相位基本不存在
      [0.3, 0.6)  相位活跃但未占优
      [0.6, 0.85) 相位占优
      [0.85, 1.0] 相位饱和 → D_max → 必然向下一相位过渡
    '''
    value: float = 0.0
    label: str = ''

    def is_saturated(self) -> bool:
        return self.value >= 0.85

    def is_dominant(self) -> bool:
        return self.value >= 0.6

    def is_absent(self) -> bool:
        return self.value < 0.3


@dataclass
class CycleProfile:
    '''
    公司五形相位剖析。

    不是 S 向量（健康度量），而是相位占优度。
    占优相位 + 饱和检测 → 下一帧相位过渡方向。
    '''
    fire: PhaseState
    water: PhaseState
    wood: PhaseState
    metal: PhaseState
    earth: PhaseState

    def dominant_phase(self) -> str:
        phases = {
            '火': self.fire,
            '水': self.water,
            '木': self.wood,
            '金': self.metal,
            '土': self.earth}
        return max(phases, key = lambda k: phases[k].value)

    def saturated_phases(self) -> List[str]:
        return [p.label for p in (self.fire, self.water, self.wood, self.metal, self.earth) if p.is_saturated()]

    def absent_phases(self) -> List[str]:
        return [p.label for p in (self.fire, self.water, self.wood, self.metal, self.earth) if p.is_absent()]

    def predict_transition(self) -> Tuple[str, str, float]:
        saturated = self.saturated_phases()
        if '火' in saturated:
            return ('火', '金', 0.85)
        if '木' in saturated:
            return ('木', '金', 0.8)
        if '金' in saturated:
            return ('金', '土', 0.75)
        if '土' in saturated:
            return ('土', '火', 0.7)
        if '水' in saturated:
            return ('水', '木', 0.7)
        dominant = self.dominant_phase()
        d_val = getattr(self, {
            '火': 'fire',
            '水': 'water',
            '木': 'wood',
            '金': 'metal',
            '土': 'earth'}[dominant]).value
        if d_val >= 0.75:
            next_phase = {
                '火': '金',
                '水': '木',
                '木': '金',
                '金': '土',
                '土': '火'}[dominant]
            return (dominant, next_phase, 0.5 + (d_val - 0.75) * 2)
        absent = self.absent_phases()
        if '火' in absent and self.earth.is_dominant():
            return ('土', '火', 0.5)
        if '金' in absent and self.wood.is_dominant():
            return ('木', '金', 0.55)
        return ('', '', 0.0)

    def summary(self) -> dict:
        return {
            '火': f"{self.fire.value:.2f}{'⚠' if self.fire.is_saturated() else ''}",
            '水': f"{self.water.value:.2f}{'⚠' if self.water.is_saturated() else ''}",
            '木': f"{self.wood.value:.2f}{'⚠' if self.wood.is_saturated() else ''}",
            '金': f"{self.metal.value:.2f}{'⚠' if self.metal.is_saturated() else ''}",
            '土': f"{self.earth.value:.2f}{'⚠' if self.earth.is_saturated() else ''}",
            '占优': self.dominant_phase(),
            '饱和': self.saturated_phases(),
            '缺位': self.absent_phases()}


class CyclePhaseDetector:
    '''
    从财报/行情数据识别公司所处的五形相位和占优度。

    每个相位的占优度由 industry_mapping.py 中的行业映射定义。
    相位占优度 ≠ 财务健康 — 它是"这个相位在当前子图中
    是否占优"的程度。值越高越接近饱和 (D_max)。
    '''

    def __init__(self, industry: str = 'default'):
        self.industry = industry
        if industry in INDUSTRY_MAP:
            self.map = INDUSTRY_MAP[industry]
        else:
            self.map = INDUSTRY_MAP.get(DEFAULT_INDUSTRY, {})
        self.name = self.map.get('name', industry)
        self.note = self.map.get('note', '')

    def extract(self, indicators: pd.DataFrame) -> dict:
        """
        从 akShare 财务指标 DataFrame 中提取所有可用指标的最新值。

        indicators: stock_financial_analysis_indicator 的返回值
                    (列=指标名, 行=日期)

        返回: { 'revenue_growth': 6.54, 'gross_margin': 0.0, ... }
        """
        result = {}
        for key, keyword in STD_INDICATORS.items():
            found = False
            for col in indicators.columns:
                if keyword in str(col):
                    vals = indicators[col].dropna()
                    if len(vals) > 0:
                        result[key] = float(vals.iloc[-1])
                        found = True

                if not found:
                    result[key] = 0.0
        if result.get('gross_margin', 0.0) <= 0 and result.get('cost_rate', 0.0) > 0:
            result['gross_margin'] = 1.0 - result['cost_rate'] / 100.0
            result['gross_margin'] *= 100.0
        return result

    def profile_from_indicators(self, extracted: dict, price_momentum: float = 0.0) -> CycleProfile:
        """
        从提取的指标数据识别五形相位。

        Args:
            extracted: extract() 的输出 — { 'revenue_growth': float, ... }
            price_momentum: 60天价格动量, 用于补充火形判断

        Returns:
            CycleProfile
        """
        thresholds = self.map.get('default_thresholds', {})
        phase_configs = self.map.get('phases', {})
        phase_values = {}
        for phase_name, config in phase_configs.items():
            primary = config.get('primary', {})
            ind_key = primary.get('indicator', '')
            transform_name = primary.get('transform', '')
            raw_val = extracted.get(ind_key, 0.0)
            transform_fn = TRANSFORMS.get(transform_name)
            if transform_fn:
                val = transform_fn(raw_val, thresholds)
            else:
                val = 0.5
            fallback = config.get('fallback', {})
            if val == 0 and fallback:
                fb_key = fallback.get('indicator', '')
                fb_transform = fallback.get('transform', '')
                raw_fb = extracted.get(fb_key, 0.0)
                if raw_fb > 0:
                    fb_fn = TRANSFORMS.get(fb_transform)
                    if fb_fn:
                        val = fb_fn(raw_fb, thresholds)
            secondary = config.get('secondary', {})
            if secondary:
                sec_key = secondary.get('indicator', '')
                sec_transform = secondary.get('transform', '')
                sec_weight = secondary.get('weight', 0.3)
                raw_sec = extracted.get(sec_key, 0.0)
                if raw_sec > 0 and sec_transform:
                    sec_fn = TRANSFORMS.get(sec_transform)
                    if sec_fn:
                        sec_val = sec_fn(raw_sec, thresholds)
                        val = val * (1 - sec_weight) + sec_val * sec_weight
            phase_values[phase_name] = float(np.clip(val, 0, 1))
        if price_momentum > 0:
            phase_values['fire'] = min(1.0, phase_values.get('fire', 0.0) + 0.05)
        profiles = {
            'fire': PhaseState(phase_values.get('fire', 0.5), '火'),
            'water': PhaseState(phase_values.get('water', 0.5), '水'),
            'wood': PhaseState(phase_values.get('wood', 0.5), '木'),
            'metal': PhaseState(phase_values.get('metal', 0.5), '金'),
            'earth': PhaseState(phase_values.get('earth', 0.5), '土')}
        return CycleProfile(**profiles)

    def profile_from_market(self, df: pd.DataFrame) -> CycleProfile:
        '''
        仅有行情数据时的近似识别。精度低于财报方式。
        '''
        close = df['close']
        if len(close) < 60:
            return self._default_profile()
        returns = close.pct_change().dropna()
        mom_60 = close.iloc[-1] / close.iloc[-60] - 1
        mom_252 = close.iloc[-1] / close.iloc[-252] - 1 if len(close) > 252 else mom_60
        vol = returns.std() * np.sqrt(252)
        if mom_60 > 0.3 and vol > 0.4:
            fire_val = 1.0
        elif mom_60 < 0.15:
            fire_val = max(0, mom_60 / 0.15) * 0.5
        else:
            fire_val = mom_60 / 0.3
        water_val = 0.7 if mom_252 > 0.2 and vol < 0.3 else max(0, mom_252 / 0.5)
        if mom_252 > 0.1 and vol < 0.25:
            wood_val = 0.8
        elif vol > 0.5:
            wood_val = 0.0
        else:
            wood_val = max(0, 0.5 - vol * 0.5)
        sharpe = (returns.mean() / max(returns.std(), 0.001)) * np.sqrt(252)
        if sharpe > 1:
            metal_val = 0.8
        elif sharpe > 0:
            metal_val = 0.5
        else:
            metal_val = 0.3
        max_dd = (close / close.expanding().max()).min()
        if max_dd > -0.1:
            earth_val = 0.7
        elif max_dd > -0.3:
            earth_val = 0.5
        elif max_dd > -0.5:
            earth_val = 0.3
        else:
            earth_val = 0.0
        return CycleProfile(fire = PhaseState(float(np.clip(fire_val, 0, 1)), '火'), water = PhaseState(float(np.clip(water_val, 0, 1)), '水'), wood = PhaseState(float(np.clip(wood_val, 0, 1)), '木'), metal = PhaseState(float(np.clip(metal_val, 0, 1)), '金'), earth = PhaseState(float(np.clip(earth_val, 0, 1)), '土'))

    def _default_profile(self) -> CycleProfile:
        return CycleProfile(fire = PhaseState(0.5, '火'), water = PhaseState(0.5, '水'), wood = PhaseState(0.5, '木'), metal = PhaseState(0.5, '金'), earth = PhaseState(0.5, '土'))


class StockPhaseProjector:
    '''
    股价相位投影器。

    核心: 股价不是预测目标, 是相位过渡在货币边上的投影。

    投影规则 (来自 五形耦合动力学):
      ┌────────────────┬─────────────────┬──────────────────────┐
      │ 相位过渡        │ 投影(股价方向)  │ 业务含义              │
      ├────────────────┼─────────────────┼──────────────────────┤
      │ 火亢 → 金修剪   │ 短期↗ 中期↘     │ 最后一段冲顶 → 回落   │
      │ 木僵 → 金修剪   │ 短期→ 中期↘     │ 护城河被拆 → 价值回归  │
      │ 金亢 → 土枯竭   │ ↘↘             │ 效率溢价消散 → 下跌   │
      │ 土盈 → 火再生   │ ↗↗             │ 新增长曲线开启         │
      │ 水盈 → 木锁定   │ → 或 ↗          │ 渠道成熟 → 进入稳态   │
      │ 火衰 → 水/土    │ ↘ 或 →          │ 增长熄火 → 下落/筑底  │
      │ 金缺 + 木僵     │ ↘↘             │ 僵化 + 无效率 → 危机   │
      │ 健康流转         │ →              │ 各相位正常交替 → 稳定  │
      └────────────────┴─────────────────┴──────────────────────┘
    '''

    def project(self, profile: CycleProfile) -> dict:
        '''预测未来 1-3 帧的股价投影'''
        (from_phase, to_phase, confidence) = profile.predict_transition()
        direction = '→'
        strength = 0
        reason = ''
        if from_phase == '火':
            direction = '↗中期↘'
            strength = 0.8
            reason = '火亢: 市场驱动力饱和, 短期仍有最后一段冲顶, 中期金形修剪将触发回调'
        elif from_phase == '木':
            direction = '→中期↘'
            strength = 0.7
            reason = '木僵: 护城河已形成但过度锁定, 金形修剪即将拆解冗余环结构'
        elif from_phase == '金':
            direction = '↘'
            strength = 0.75
            reason = '金亢: 效率已达极限, 过度修剪将耗尽土形储备, 基本面承压'
        elif from_phase == '土':
            direction = '↗'
            strength = 0.65
            reason = '土盈: 储备充裕, 火形梯度即将从储备池中重新激活新增长'
        elif from_phase == '水':
            direction = '→'
            strength = 0.5
            reason = '水盈: 渠道已铺满, 木形锁定中, 股价从增长驱动转为价值驱动'
        elif not from_phase:
            saturated = profile.saturated_phases()
            absent = profile.absent_phases()
            if len(saturated) == 0 and len(absent) == 0:
                direction = '→'
                strength = 0.3
                reason = '健康流转: 各相位正常交替, 无明显拐点'
            elif len(saturated) > 0:
                direction = '↘'
                strength = 0.6
                reason = f'''{','.join(saturated)}趋于饱和, 基本面承压'''
            else:
                direction = '↗'
                strength = 0.4
                reason = f'''{','.join(absent)}缺位, 有补位空间, 谨慎乐观'''
        return {
            'direction': direction,
            'strength': float(np.clip(strength, 0, 1)),
            'confidence': float(np.clip(confidence * 0.8 + strength * 0.2, 0, 1)),
            'from_phase': from_phase,
            'to_phase': to_phase,
            'reason': reason}


def analyze_company(code: str, name: str = '', industry: str = 'default', df_market: Optional[pd.DataFrame] = None, indicators: Optional[pd.DataFrame] = None) -> dict:
    '''
    完整商业周期相位分析。

    输出不是"涨跌预测", 而是:
      1. 公司子图当前处于五形相位的哪个阶段
      2. 哪个相位接近 D_max (饱和)
      3. 下一帧将向哪个相位过渡
      4. 这个过渡在股价上的投影

    使用方式::

        from spum_cycle import analyze_company
        result = analyze_company(\'600519\', \'贵州茅台\', \'consumer\',
                                 df_market=price_df, indicators=fin_df)
        print(result[\'phase_summary\'])
    '''
    detector = CyclePhaseDetector(industry)
    if indicators is not None:
        extracted = detector.extract(indicators)
        price_momentum = 0.0
        if df_market is not None:
            close = df_market['close']
            if len(close) > 60:
                price_momentum = close.iloc[-1] / close.iloc[-60] - 1
        profile = detector.profile_from_indicators(extracted, price_momentum = price_momentum)
    elif df_market is not None:
        profile = detector.profile_from_market(df_market)
    else:
        return {'error': '需要财务数据或行情数据'}
    (from_p, to_p, trans_conf) = profile.predict_transition()
    projector = StockPhaseProjector()
    projection = projector.project(profile)
    return {
        'code': code,
        'name': name or code,
        'industry': industry,
        'profile': profile,
        'phase_summary': profile.summary(),
        'dominant_phase': profile.dominant_phase(),
        'saturated_phases': profile.saturated_phases(),
        'absent_phases': profile.absent_phases(),
        'transition': {'from': from_p, 'to': to_p, 'confidence': trans_conf},
        'price_projection': projection,
        'extracted_indicators': extracted if indicators is not None else None,
        'industry_note': detector.note}


def print_cycle_report(result: dict):
    '''打印简洁的周期分析报告'''
    if 'error' in result:
        print(f'''错误: {result['error']}''')
        return None
    name = result['name']
    code = result['code']
    profile = result['profile']
    proj = result['price_projection']
    trans = result['transition']
    summary = result['phase_summary']
    print(f'''''')
    print(f'''  {'======================================================'}''')
    print(f'''  SPUM 拓扑周期分析: {name} ({code})''')
    print(f'''  {'======================================================'}''')
    print('  五形相位:')
    phases = [
        ('火', profile.fire),
        ('水', profile.water),
        ('木', profile.wood),
        ('金', profile.metal),
        ('土', profile.earth)]
    for label, p in phases:
        bar = '█' * int(p.value * 20) + '░' * (20 - int(p.value * 20))
        mark = ' ⚠ D_max' if p.is_saturated() else ''
        print(f'''    {label}: {p.value:.2f} {bar}{mark}''')
    print(f'''  占优相位: {summary['占优']}''')
    if summary['饱和']:
        print(f'''  ⚠ 饱和(D_max): {', '.join(summary['饱和'])}''')
    if summary['缺位']:
        print(f'''  △ 缺位: {', '.join(summary['缺位'])}''')
    print(f'''''')
    print('  相位过渡预测:')
    if trans['from']:
        print(f'''    {trans['from']} → {trans['to']} (信心: {trans['confidence']:.0%})''')
    else:
        print('    无明确过渡信号 — 相位健康流转中')
    print(f'''''')
    print('  股价投影 (1-3帧):')
    print(f'''    方向: {proj['direction']}''')
    print(f'''    强度: {proj['strength']:.0%}''')
    print(f'''    信心: {proj['confidence']:.0%}''')
    print(f'''    原因: {proj['reason']}''')
    print(f'''  {'======================================================'}''')
    print(f'''''')
