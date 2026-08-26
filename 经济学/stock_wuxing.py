# Source Generated with Decompyle++
# File: stock_wuxing.cpython-311.pyc (Python 3.11)

'''
SPUM × 公司五形向量分析器
===========================

理论基底: SPUM 青囊总纲 + 五形公理

核心思想:
  一家公司不是"股票曲线"——它是 G_econ 中的一个实体子图,
  其发展由五个拓扑相位 (S_水, S_木, S_土, S_金, S_火) 刻画。

  价格不是 σ 的输入——价格是 S 五维向量在货币边上的投影。

映射:
  五形    拓扑相位          公司子系统                   财务指标代理
  ───    ───────          ─────────                   ─────────────
  水形    链式传输          收入流/供应链/渠道          营收增长、周转率
  木形    闭合环骨架        护城河/技术壁垒/品牌         毛利率、研发比、份额
  土形    分散储备池        现金储备/人才/未分配资源     现金比、负债率、自由现金流
  金形    修剪与更新        成本控制/重组/效率          费用率、资产周转率、ROIC
  火形    密度梯度驱动      市场需求/竞争压力/创新       TAM 增速、市占率变化

依赖: akshare (可选, 用于自动获取财务数据)
'''
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime, date
import warnings


@dataclass
class WuxingConfig:
    moat_min_gross_margin: float = 0.2
    moat_reference_rnd_ratio: float = 0.05
    water_growth_window: int = 4
    water_max_organic_growth: float = 0.3
    earth_min_cash_ratio: float = 0.15
    earth_max_debt_ratio: float = 0.6
    metal_reference_sga_ratio: float = 0.15
    metal_min_asset_turnover: float = 0.3
    fire_min_revenue_growth: float = 0.05
    fire_saturation_penetration: float = 0.6
    d_max_threshold: float = 0.85
    d_min_threshold: float = 0.15
    industry_benchmark: str = 'default'


INDUSTRY_BENCHMARKS = {
    'tech': {
        'moat_reference_rnd_ratio': 0.1,
        'moat_min_gross_margin': 0.4,
        'metal_reference_sga_ratio': 0.2,
        'water_max_organic_growth': 0.5,
        'fire_saturation_penetration': 0.5 },
    'finance': {
        'moat_reference_rnd_ratio': 0.01,
        'moat_min_gross_margin': 0.15,
        'metal_reference_sga_ratio': 0.3,
        'water_max_organic_growth': 0.15,
        'earth_max_debt_ratio': 0.95,
        'fire_saturation_penetration': 0.7 },
    'manufacturing': {
        'moat_reference_rnd_ratio': 0.03,
        'moat_min_gross_margin': 0.15,
        'metal_reference_sga_ratio': 0.1,
        'water_max_organic_growth': 0.2,
        'fire_saturation_penetration': 0.65 },
    'consumer': {
        'moat_reference_rnd_ratio': 0.02,
        'moat_min_gross_margin': 0.25,
        'metal_reference_sga_ratio': 0.25,
        'water_max_organic_growth': 0.25,
        'fire_saturation_penetration': 0.55 } }


@dataclass
class WuxingVector:
    S_water: float = 0.5
    S_wood: float = 0.5
    S_earth: float = 0.5
    S_metal: float = 0.5
    S_fire: float = 0.5

    def as_dict(self) -> dict:
        return {
            '水': self.S_water,
            '木': self.S_wood,
            '土': self.S_earth,
            '金': self.S_metal,
            '火': self.S_fire }

    def as_array(self) -> np.ndarray:
        return np.array([
            self.S_water,
            self.S_wood,
            self.S_earth,
            self.S_metal,
            self.S_fire ])

    def magnitude(self) -> float:
        '''五维向量模长'''
        return float(np.linalg.norm(self.as_array()))

    def dominant(self) -> str:
        '''占优分量名称'''
        names = [
            '水',
            '木',
            '土',
            '金',
            '火' ]
        vals = self.as_array()
        return names[int(np.argmax(vals))]

    def deficiency(self) -> str:
        '''最弱分量名称'''
        names = [
            '水',
            '木',
            '土',
            '金',
            '火' ]
        vals = self.as_array()
        return names[int(np.argmin(vals))]

    def is_near_d_max(self, threshold: float = 0.85) -> Dict[str, bool]:
        '''各分量是否接近 D_max (增长天花板)'''
        return {
            '水': self.S_water >= threshold,
            '木': self.S_wood >= threshold,
            '土': self.S_earth >= threshold,
            '金': self.S_metal >= threshold,
            '火': self.S_fire >= threshold }


HEALTHY_RANGES = {
    'water': (0.25, 0.75),
    'wood': (0.3, 0.8),
    'earth': (0.25, 0.7),
    'metal': (0.2, 0.7),
    'fire': (0.25, 0.8) }


@dataclass
class InflectionSignal:
    component: str
    direction: str
    value: float
    threshold: float
    description: str
    confidence: float


class CompanyWuxing:
    """
    公司五形向量分析器。

    通过财务数据和市场数据计算公司的 S 五维向量,
    识别各分量的 D_max 逼近状态, 预测发展拐点。

    用法::

        analyzer = CompanyWuxing(name='贵州茅台', industry='consumer')

        # 方式1: 自动获取
        data = analyzer.fetch_data('600519')

        # 方式2: 手动传入
        analyzer.load_financials(balance_sheet_df, income_df)
        analyzer.load_market_data(price_df)

        # 分析
        result = analyzer.analyze()
        analyzer.print_report(result)
    """
    
    def __init__(self, name: str = '', code: str = '', industry: str = 'default', config: Optional[WuxingConfig] = None):
        self.name = name
        self.code = code
        self.industry = industry
        base_cfg = config or WuxingConfig()
        bench = INDUSTRY_BENCHMARKS.get(industry, { })
        for k, v in bench.items():
            setattr(base_cfg, k, v)
        self.cfg = base_cfg
        self._balance = None
        self._income = None
        self._indicators = None
        self._market = None

    
    def fetch_data(self, code: str, start_year: int = 2020) -> bool:
        '''
        通过 akshare 自动获取财务数据和行情数据。

        数据结构: stock_financial_analysis_indicator 返回的 DataFrame
        中, 每列为一个财务指标, 每行为一个报告期。

        返回 True 表示至少部分数据获取成功。
        '''
        import akshare as ak
        success = False
        
        try:
            self._indicators = ak.stock_financial_analysis_indicator(symbol = code, start_year = str(start_year))
            success = True
        except Exception as e:
            warnings.warn(f'财务分析指标获取失败: {e}')
        self.code = code
        return success

    
    def load_financials(self, balance: Optional[pd.DataFrame] = None, income: Optional[pd.DataFrame] = None, indicators: Optional[pd.DataFrame] = None):
        '''手动加载财务数据'''
        if balance is not None:
            self._balance = balance
        if income is not None:
            self._income = income
        if indicators is not None:
            self._indicators = indicators

    
    def load_market_data(self, df: pd.DataFrame):
        '''加载行情数据 (用于火形补充)'''
        self._market = df

    
    def analyze(self) -> dict:
        """
        执行完整五形分析。

        返回: {
            'S_current': WuxingVector,
            'inflections': [InflectionSignal],
            'indicators': { 各分量的原始指标 },
            'H': { 健康区间 },
            'S_history': { 各分量的时间序列 } (若有面板数据)
        }
        """
        cfg = self.cfg
        (S_water, water_indicators) = self._calc_water()
        (S_wood, wood_indicators) = self._calc_wood()
        (S_earth, earth_indicators) = self._calc_earth()
        (S_metal, metal_indicators) = self._calc_metal()
        (S_fire, fire_indicators) = self._calc_fire()
        S = WuxingVector(S_water = float(np.clip(S_water, 0, 1)), S_wood = float(np.clip(S_wood, 0, 1)), S_earth = float(np.clip(S_earth, 0, 1)), S_metal = float(np.clip(S_metal, 0, 1)), S_fire = float(np.clip(S_fire, 0, 1)))
        inflections = self._detect_inflections(S, cfg)
        return {
            'S_current': S,
            'inflections': inflections,
            'indicators': {
                'water': water_indicators,
                'wood': wood_indicators,
                'earth': earth_indicators,
                'metal': metal_indicators,
                'fire': fire_indicators },
            'H': HEALTHY_RANGES }

    
    def _get_indicator(self, keyword: str, fallback: str = '') -> float:
        '''
        从财务指标 DataFrame 中提取指标的最新值。

        搜索列名中包含 keyword 的列, 取最后一个非空值。
        若未找到或全空, 尝试 fallback 关键词。
        返回 0 表示未找到。
        '''
        if self._indicators is None or self._indicators.empty:
            return 0.0
        ind = self._indicators
        for kw in (keyword, fallback):
            if not kw:
                continue
            for col in ind.columns:
                if kw in str(col):
                    vals = ind[col].dropna()
                    if len(vals) > 0:
                        return float(vals.iloc[-1])
        return 0.0

    
    def _get_gross_margin(self) -> float:
        '''
        获取毛利率。直接列不可用时从主营业务成本率推算。

        毛利率 = 1 - 主营业务成本率(%)
        '''
        gm = self._get_indicator('销售毛利率')
        if gm > 0:
            return gm / 100.0
        cost_rate = self._get_indicator('主营业务成本率')
        if cost_rate > 0:
            return 1.0 - cost_rate / 100.0
        return 0.0

    
    def _calc_water(self) -> Tuple[float, dict]:
        '''
        水形: 收入流强度。

        指标:
          - 主营业务收入增长率 → 营收增长
          - 总资产周转率 → 收入转化效率

        0 = 收入枯竭, 1 = 收入过载
        '''
        cfg = self.cfg
        indicators = { }
        revenue_growth_pct = self._get_indicator('主营业务收入增长率')
        if revenue_growth_pct:
            revenue_growth = revenue_growth_pct / 100.0
        else:
            revenue_growth = 0.0
        indicators['revenue_growth'] = revenue_growth
        turnover = self._get_indicator('总资产周转率')
        indicators['asset_turnover'] = turnover
        if revenue_growth == 0 and self._market is not None:
            close = self._market['close']
            if len(close) > 252:
                annual_return = close.iloc[-1] / close.iloc[-252] - 1
                revenue_growth = max(0, min(annual_return, 0.5))
                indicators['revenue_growth_approx'] = revenue_growth
        max_growth = cfg.water_max_organic_growth
        if revenue_growth <= 0:
            S = 0.0
        elif revenue_growth >= max_growth:
            S = 1.0
        else:
            S = revenue_growth / max_growth
        if 0 < turnover < 0.3:
            S *= 0.7
        indicators['S_raw'] = float(S)
        return float(S), indicators

    
    def _calc_wood(self) -> Tuple[float, dict]:
        '''
        木形: 护城河强度。

        指标:
          - 销售毛利率 → 护城河深度
          - 销售毛利率的历史波动 → 护城河稳定性

        0 = 无壁垒, 1 = 护城河僵化
        '''
        cfg = self.cfg
        indicators = { }
        gross_margin = self._get_gross_margin()
        indicators['gross_margin'] = gross_margin
        gross_std = 0.0
        if self._indicators is not None and gross_margin > 0:
            cost_col = None
            for c in self._indicators.columns:
                if '主营业务成本率' in str(c):
                    cost_col = c
                    break
            gm_col = None
            for c in self._indicators.columns:
                if '销售毛利率' in str(c):
                    gm_col = c
                    break
            hist_col = gm_col or cost_col
            if hist_col:
                vals = self._indicators[hist_col].dropna()
                if len(vals) > 3:
                    if gm_col:
                        gross_std = float(vals.std()) / 100.0
                    else:
                        hist_gm = vals.apply(lambda x: 1 - x / 100.0)
                        gross_std = float(hist_gm.std())
                    indicators['gross_margin_std'] = gross_std
        if gross_margin == 0 and self._market is not None:
            close = self._market['close']
            if len(close) > 60:
                vol = close.pct_change().std()
                ret = close.iloc[-1] / close.iloc[-60] - 1
                moat_score = max(0, min(ret / 0.15 - vol * 5, 1.0))
                gross_margin = 0.2 + moat_score * 0.4
                indicators['gross_margin_approx'] = gross_margin
        min_margin = cfg.moat_min_gross_margin
        if gross_margin <= min_margin:
            S_wood = 0.0
        elif gross_margin >= 0.8:
            S_wood = 1.0
        else:
            S_wood = (gross_margin - min_margin) / (0.8 - min_margin)
        if gross_std > 0.05:
            S_wood *= max(0.5, 1 - gross_std * 3)
        indicators['S_raw'] = float(S_wood)
        return float(S_wood), indicators

    
    def _calc_earth(self) -> Tuple[float, dict]:
        '''
        土形: 储备充足度。

        指标:
          - 资产负债率 (逆向)
          - 流动比率
          - 现金流量比率

        0 = 枯竭, 1 = 冗余
        '''
        cfg = self.cfg
        indicators = { }
        debt_ratio_pct = self._get_indicator('资产负债率')
        if debt_ratio_pct:
            debt_ratio = debt_ratio_pct / 100.0
        else:
            debt_ratio = 0.0
        indicators['debt_ratio'] = debt_ratio
        current_ratio = self._get_indicator('流动比率')
        indicators['current_ratio'] = current_ratio
        if debt_ratio == 0 and self._market is not None:
            close = self._market['close']
            if len(close) > 252:
                max_dd = (close / close.expanding().max()).min()
                debt_ratio = 0.3 + (1 - min(1, max(0, 1 - max_dd))) * 0.3
                current_ratio = 1.0 + (1 - max_dd) * 0.5
                indicators['debt_ratio_approx'] = debt_ratio
                indicators['current_ratio_approx'] = current_ratio
        max_debt = cfg.earth_max_debt_ratio
        if debt_ratio >= max_debt:
            debt_score = 0.0
        elif debt_ratio <= 0.2:
            debt_score = 0.9
        else:
            debt_score = 1 - (debt_ratio - 0.2) / (max_debt - 0.2)
        if current_ratio > 0:
            if current_ratio < 1:
                cr_score = current_ratio
            elif current_ratio > 5:
                cr_score = 0.5
            else:
                cr_score = min(1.0, current_ratio / 2.5)
            S = 0.5 * debt_score + 0.5 * cr_score
        else:
            S = debt_score
        indicators['S_raw'] = float(S)
        return float(S), indicators

    
    def _calc_metal(self) -> Tuple[float, dict]:
        '''
        金形: 修剪与效率。

        指标:
          - 三项费用比重 (逆向: 越低越高效)
          - 总资产周转率 (越高越高效)
          - 净资产收益率 ROE

        0 = 无修剪 (臃肿), 1 = 过亢 (过度削减)
        '''
        cfg = self.cfg
        indicators = { }
        fee_ratio_pct = self._get_indicator('三项费用比重')
        if fee_ratio_pct:
            fee_ratio = fee_ratio_pct / 100.0
        else:
            fee_ratio = 0.0
        indicators['fee_ratio'] = fee_ratio
        turnover = self._get_indicator('总资产周转率')
        indicators['asset_turnover'] = turnover
        roe_pct = self._get_indicator('净资产收益率')
        if roe_pct:
            roe = roe_pct
        else:
            roe = 0.0
        indicators['roe'] = roe
        if fee_ratio == 0 and self._market is not None:
            close = self._market['close']
            if len(close) > 252:
                sharpe = close.pct_change().mean() / max(close.pct_change().std(), 0.001) * np.sqrt(252)
                fee_ratio = max(0.05, 0.25 - sharpe * 0.02)
                indicators['fee_ratio_approx'] = fee_ratio
        if fee_ratio <= 0.05:
            S = 0.9
        elif fee_ratio >= 0.4:
            S = 0.1
        elif fee_ratio == 0:
            S = 0.5
        else:
            S = 1 - (fee_ratio - 0.05) / 0.35
        if roe > 15:
            S = min(0.85, S + 0.1)
        indicators['S_raw'] = float(S)
        return float(S), indicators

    
    def _calc_fire(self) -> Tuple[float, dict]:
        '''
        火形: 市场驱动力。

        指标:
          - 主营业务收入增长率 (历史 1 年)
          - 净利润增长率
          - 价格动量 (短期)

        0 = 熄火, 1 = 过冲
        '''
        cfg = self.cfg
        indicators = { }
        revenue_growth_pct = self._get_indicator('主营业务收入增长率')
        if revenue_growth_pct:
            revenue_growth = revenue_growth_pct / 100.0
        else:
            revenue_growth = 0.0
        profit_growth_pct = self._get_indicator('净利润增长率')
        if profit_growth_pct:
            profit_growth = profit_growth_pct / 100.0
        else:
            profit_growth = 0.0
        indicators['revenue_growth'] = revenue_growth
        indicators['profit_growth'] = profit_growth
        price_momentum = 0.0
        if self._market is not None:
            close = self._market['close']
            if len(close) > 60:
                mom_1m = close.iloc[-1] / close.iloc[-20] - 1
                mom_3m = close.iloc[-1] / close.iloc[-60] - 1
                price_momentum = mom_3m * 0.7 + mom_1m * 0.3
                indicators['price_momentum_1m'] = float(mom_1m)
                indicators['price_momentum_3m'] = float(mom_3m)
        if revenue_growth > 0:
            growth_used = revenue_growth
        elif price_momentum > 0:
            growth_used = max(0, price_momentum * 0.5)
            indicators['growth_used'] = 'price_momentum_approx'
        else:
            growth_used = 0.0
        min_growth = cfg.fire_min_revenue_growth
        max_growth = 0.4
        if growth_used <= min_growth:
            S = 0.0
        elif growth_used >= max_growth:
            S = 1.0
        else:
            S = (growth_used - min_growth) / (max_growth - min_growth)
        indicators['S_raw'] = float(S)
        return float(S), indicators

    
    def _detect_inflections(self, S: WuxingVector, cfg: WuxingConfig) -> List[InflectionSignal]:
        '''
        检测各分量是否接近或超过 D_max。

        当一个分量接近上限 (≥ threshold):
          - 它在当前帧已经饱和
          - 下一帧必然开始下降 (或被金形修剪)
          - 这就是拐点

        当一个分量接近下限 (≤ d_min_threshold):
          - 它严重不足
          - 要么被其他分量补充 (木生火), 要么触发危机 (V⁻ 级联)
        '''
        signals = []
        d_max = cfg.d_max_threshold
        d_min = cfg.d_min_threshold
        name_map = {
            '水': ('S_water', '收入流'),
            '木': ('S_wood', '护城河'),
            '土': ('S_earth', '储备'),
            '金': ('S_metal', '效率'),
            '火': ('S_fire', '市场驱动') }
        comps = {
            '水': S.S_water,
            '木': S.S_wood,
            '土': S.S_earth,
            '金': S.S_metal,
            '火': S.S_fire }
        for cname, val in comps.items():
            (attr, desc) = name_map[cname]
            if val >= d_max:
                confidence = min(0.95, 0.5 + (val - d_max) / (1 - d_max))
                signals.append(InflectionSignal(component = cname, direction = '过热', value = val, threshold = d_max, description = f'{desc}接近D_max: {desc}已达饱和, 继续投入的边际收益趋零', confidence = float(confidence)))
                continue
            if val <= d_min:
                confidence = min(0.9, 0.5 + (d_min - val) / d_min)
                signals.append(InflectionSignal(component = cname, direction = '过冷', value = val, threshold = d_min, description = f'{desc}严重不足: {desc}已近枯竭, 可能引发系统级风险', confidence = float(confidence)))
        return signals

    
    def print_report(self, result: dict):
        '''打印五形分析报告'''
        S = result['S_current']
        inflections = result['inflections']
        indicators = result['indicators']
        print('========================================================')
        title = f'  SPUM 五形分析: {self.name or self.code}'
        print(title)
        print('========================================================')
        print(f'  行业: {self.industry}  |  代码: {self.code}')
        print(f'  {"─" * 50}')
        print('  S 五维向量:')
        print(f'    水 (收入流):   {S.S_water:.2f}  {"▄" * int(S.S_water * 20)}{"▁" * (20 - int(S.S_water * 20))}')
        print(f'    木 (护城河):   {S.S_wood:.2f}  {"▄" * int(S.S_wood * 20)}{"▁" * (20 - int(S.S_wood * 20))}')
        print(f'    土 (储备):     {S.S_earth:.2f}  {"▄" * int(S.S_earth * 20)}{"▁" * (20 - int(S.S_earth * 20))}')
        print(f'    金 (效率):     {S.S_metal:.2f}  {"▄" * int(S.S_metal * 20)}{"▁" * (20 - int(S.S_metal * 20))}')
        print(f'    火 (市场驱动): {S.S_fire:.2f}  {"▄" * int(S.S_fire * 20)}{"▁" * (20 - int(S.S_fire * 20))}')
        print(f'  占优分量: {S.dominant()}  |  最弱分量: {S.deficiency()}')
        if inflections:
            print(f'  {"─" * 50}')
            print(f'  ⚠ 拐点信号 ({len(inflections)} 个):')
            for inf in inflections:
                icon = '🔥' if inf.direction == '过热' else '🧊'
                print(f'    {icon} {inf.component}形 {inf.direction}')
                print(f'       当前={inf.value:.2f}, 阈值={inf.threshold}')
                print(f'       {inf.description}')
                print(f'       信心: {inf.confidence:.0%}')
        else:
            print('  ✅ 无拐点信号 —— 各分量均在健康区间内')
        print(f'  {"─" * 50}')
        print('  原始指标:')
        for phase, inds in indicators.items():
            phase_name = {
                'water': '水',
                'wood': '木',
                'earth': '土',
                'metal': '金',
                'fire': '火' }.get(phase, phase)
            items = ', '.join([ f'{k}={v:.3f}' for k, v in inds.items() ])
            print(f'    {phase_name}: {items}')
        print(f'  {"─" * 50}')
        print(f'  S 模长: {S.magnitude():.2f}  |  健康区间 H: 各分量 [0.20~0.80]')
        print('========================================================')
        return None


def quick_wuxing(code: str, name: str = '', industry: str = 'default', start: str = '20220101') -> dict:
    '''
    快速分析: 尝试获取财务数据 + 行情数据 → 五形分析。
    若无财务数据, 用行情数据近似。
    '''
    import akshare as ak
    analyzer = CompanyWuxing(name = name or code, code = code, industry = industry)
    try:
        from stock_sigma import StockSigma
        sigma = StockSigma()
        market_df = sigma.fetch_a_stock(code, start = start, source = 'tx')
        analyzer.load_market_data(market_df)
    except Exception:
        try:
            if code.startswith('0'):
                symbol = f'sh{code}' if code.startswith('00') else f'sz{code}'
            df = ak.stock_zh_index_daily(symbol = f'sh{code}')
            df.index = pd.to_datetime(df['date'])
            df = df[df.index >= start].copy()
            analyzer.load_market_data(df)
        except Exception as e:
            warnings.warn(f'行情数据获取失败: {e}')
    try:
        analyzer.fetch_data(code)
    except Exception:
        pass
    result = analyzer.analyze()
    analyzer.print_report(result)
    return result
