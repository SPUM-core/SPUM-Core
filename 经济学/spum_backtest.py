# Source Generated with Decompyle++
# File: spum_backtest.cpython-311.pyc (Python 3.11)

'''
SPUM 相位过渡回测引擎
=====================

Phase 2 — 历史回测框架。

"真值"定义（SPUM 方式）:
  相位过渡的真值是 S 向量本身的演化，不是股价。
  如果在帧 t 预测"火→金"，真值是帧 t+n 的 S 向量
  是否确实朝金形占优方向移动。

评估指标:
  - 相位过渡命中率: 预测的过渡在后续 1-3 帧内是否实现
  - 相位方向准确率: 饱和相位的值在后续帧是否确实下降
  - 拐点预警提前量: 从 D_max 信号到实际反转的帧数

帧定义: 1 帧 = 1 年 (4 个季度财报窗口)
'''
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
import json
import os
import time
from spum_cycle import CyclePhaseDetector, CycleProfile, PhaseState, StockPhaseProjector
from industry_mapping import INDUSTRY_MAP

@dataclass
class FrameRecord:
    year: int
    profile: CycleProfile
    predicted_transition: Tuple[str, str, float]
    price_return: float = 0.0

@dataclass
class TransitionEvent:
    year: int
    from_phase: str
    to_phase: str
    strength: float

class BacktestEngine:
    '''
    逐帧相位过渡回测。
    
    对每只股票, 以每年为帧:
      1. 从该年及之前的数据计算 S 向量
      2. 记录预测的相位过渡 (火→金, 木→金, 等)
      3. 在后续帧中检查: 预测的过渡是否实际发生
    '''
    
    def __init__(self, industry: str):
        self.industry = industry
        self.map = INDUSTRY_MAP.get(industry, { })
        self.name = self.map.get('name', industry)
        self.detector = CyclePhaseDetector(industry)

    
    def backtest(self, store: 'DataStore', codes: List[str], start_year: int = 2018, end_year: int = 2025, min_frame_years: int = 4) -> 'BacktestReport':
        '''
        批量回测。
        
        Args:
            store: DataStore 实例 (已加载数据的存储)
            codes: 待回测的股票代码列表
            start_year: 起始年份
            end_year: 结束年份
            min_frame_years: 至少需要多少年数据才能开始判断
        
        Returns:
            BacktestReport
        '''
        results = []
        for code in codes:
            r = self._run_single(code, store, start_year, end_year, min_frame_years)
            results.append(r)
        return BacktestReport(self.industry, self.name, results)

    
    def _run_single(self, code: str, store: 'DataStore', start_year: int, end_year: int, min_frame_years: int) -> 'SingleResult':
        '''单只股票回测'''
        frames = []
        transition_hits = []
        phase_direction_checks = []
        for year in range(start_year, end_year + 1):
            ind_data = store.get_financial(code, year)
            if ind_data is None:
                continue
            profile = self._compute_profile(ind_data)
            from_p, to_p, conf = profile.predict_transition()
            price_return = store.get_price_return(code, year, year + 1)
            frames.append(FrameRecord(year = year, profile = profile, predicted_transition = (from_p, to_p, conf), price_return = price_return))
            for phase_name in profile.saturated_phases():
                phase_attr = {
                    '火': 'fire',
                    '水': 'water',
                    '木': 'wood',
                    '金': 'metal',
                    '土': 'earth' }[phase_name]
                phase_val = getattr(profile, phase_attr).value
                future_val = self._get_future_phase(code, store, year + 1, year + 4, phase_attr)
                if future_val is not None:
                    decreased = future_val < phase_val * 0.9
                    phase_direction_checks.append({
                        'year': year,
                        'phase': phase_name,
                        'from_value': phase_val,
                        'to_value': future_val,
                        'decreased': decreased,
                        'hit': decreased })
            if from_p and to_p and conf >= 0.3:
                actual_to = self._check_transition(code, store, year + 1, year + 4, to_p)
                hit = actual_to is not None
                transition_hits.append({
                    'predicted_from': from_p,
                    'predicted_to': to_p,
                    'year': year,
                    'confidence': conf,
                    'actual_to': actual_to if actual_to else 'none',
                    'hit': hit,
                    'price_return': price_return })
        if not frames:
            return SingleResult(code, self.industry, [], [], [], 0.0)
        predicted_transition_years = set()
        for t in transition_hits:
            if t['hit']:
                predicted_transition_years.add(t['year'])
        avg_confidence = np.mean([f.predicted_transition[2] for f in frames if f.predicted_transition[2] > 0]) if frames else 0.0
        return SingleResult(code, self.industry, frames, transition_hits, phase_direction_checks, avg_confidence)

    
    def _compute_profile(self, ind_data: pd.DataFrame) -> CycleProfile:
        '''从财务数据计算 S 向量'''
        extracted = self.detector.extract(ind_data)
        return self.detector.profile_from_indicators(extracted)

    
    def _get_future_phase(self, code: str, store: 'DataStore', from_year: int, to_year: int, phase_attr: str) -> Optional[float]:
        '''获取后续帧中某相位值的均值'''
        vals = []
        for y in range(from_year, to_year + 1):
            d = store.get_financial(code, y)
            if d is None:
                continue
            p = self._compute_profile(d)
            vals.append(getattr(p, phase_attr).value)
        if not vals:
            return None
        return float(np.mean(vals))

    
    def _check_transition(self, code: str, store: 'DataStore', from_year: int, to_year: int, target_phase: str) -> Optional[str]:
        '''
        检查某个过渡预测是否在后续帧中实现。
        
        判断标准: 在后续帧中, target_phase 是否成为占优相位
        '''
        phase_attr = {
            '火': 'fire',
            '水': 'water',
            '木': 'wood',
            '金': 'metal',
            '土': 'earth' }[target_phase]
        for y in range(from_year, to_year + 1):
            d = store.get_financial(code, y)
            if d is None:
                continue
            p = self._compute_profile(d)
            dominant = p.dominant_phase()
            if dominant == target_phase:
                return target_phase
        return None


@dataclass
class SingleResult:
    code: str
    industry: str
    frames: List[FrameRecord]
    transition_hits: List[dict]
    phase_direction_checks: List[dict]
    avg_confidence: float

    @property
    def transition_accuracy(self) -> float:
        if self.transition_hits:
            return sum(1 for t in self.transition_hits if t['hit']) / len(self.transition_hits)
        return 0.0

    @property
    def direction_accuracy(self) -> float:
        if self.phase_direction_checks:
            return sum(1 for d in self.phase_direction_checks if d['hit']) / len(self.phase_direction_checks)
        return 0.0

    @property
    def n_transitions(self) -> int:
        return len(self.transition_hits)

    @property
    def n_hits(self) -> int:
        return sum(1 for t in self.transition_hits if t['hit'])


@dataclass
class BacktestReport:
    industry: str
    industry_name: str
    results: List[SingleResult]

    @property
    def overall_accuracy(self) -> float:
        total_hits = sum(r.n_hits for r in self.results)
        total_trans = sum(r.n_transitions for r in self.results)
        return total_hits / max(total_trans, 1)

    @property
    def overall_direction_accuracy(self) -> float:
        vals = [r.direction_accuracy for r in self.results if r.phase_direction_checks]
        return float(np.mean(vals)) if vals else 0.0

    def summary(self) -> dict:
        return {
            'industry': f'{self.industry_name} ({self.industry})',
            'stocks': len(self.results),
            'total_transition_events': sum(r.n_transitions for r in self.results),
            'transition_hits': sum(r.n_hits for r in self.results),
            'transition_accuracy': f'{self.overall_accuracy:.1%}',
            'direction_accuracy': f'{self.overall_direction_accuracy:.1%}',
            'avg_confidence': f'{np.mean([r.avg_confidence for r in self.results if r.avg_confidence]):.1%}',
        }

    def transition_breakdown(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            for t in r.transition_hits:
                rows.append({
                    'code': r.code,
                    'year': t['year'],
                    'from': t['predicted_from'],
                    'to': t['predicted_to'],
                    'confidence': t['confidence'],
                    'hit': t['hit'],
                    'price_return': t['price_return'] })
        return pd.DataFrame(rows)

    def direction_breakdown(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            for d in r.phase_direction_checks:
                rows.append({
                    'code': r.code,
                    'year': d['year'],
                    'phase': d['phase'],
                    'from_value': d['from_value'],
                    'to_value': d['to_value'],
                    'decreased': d['decreased'] })
        return pd.DataFrame(rows)

    def print_report(self):
        s = self.summary()
        print('')
        print(f'  {"=" * 54}')
        print(f'  [回测报告] {s["industry"]}')
        print(f'  {"=" * 54}')
        print(f'  股票数量:        {s["stocks"]}')
        print(f'  过渡事件总数:    {s["total_transition_events"]}')
        print(f'  命中次数:        {s["transition_hits"]}')
        print(f'  过渡命中率:      {s["transition_accuracy"]}')
        print(f'  方向准确率:      {s["direction_accuracy"]}')
        print(f'  平均信心:        {s["avg_confidence"]}')
        print('')
        df = self.transition_breakdown()
        if len(df) > 0:
            print('  ── 过渡类型精度 ──')
            for (f, t), g in df.groupby(['from', 'to']):
                hit_rate = g['hit'].mean()
                n = len(g)
                print(f'    {f}→{t}: {hit_rate:.0%} ({n}次)')
        dd = self.direction_breakdown()
        if len(dd) > 0:
            print('  ── 相位方向精度 ──')
            for phase, g in dd.groupby('phase'):
                acc = g['decreased'].mean()
                n = len(g)
                print(f'    {phase}下降: {acc:.0%} ({n}次)')
        print(f'  {"=" * 54}')
        print('')


class DataStore:
    '''
    历史数据存储池。
    
    一只股票只抓一次, 按年份切片:
      get_financial(code, year) → 截止该年末的所有可用财务指标
      
    缓存: economics/_cache/{code}.json
    '''
    
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '_cache')
        self.cache_dir = cache_dir
        self._pool = { }
        os.makedirs(cache_dir, exist_ok = True)

    
    def _load_all(self, code: str) -> Optional[pd.DataFrame]:
        '''一次性加载某只股票的全部财务数据（池化）'''
        if code in self._pool:
            return self._pool[code]
        cache_path = os.path.join(self.cache_dir, f'{code}.json')
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                if isinstance(cache_data, dict) and 'index' in cache_data:
                    df = pd.DataFrame(cache_data['data'], index = cache_data['index'], columns = cache_data['columns'])
                    if not df.empty:
                        df.index = pd.to_datetime(df.index)
                        for col in df.select_dtypes(include = ['object']).columns:
                            df[col] = pd.to_numeric(df[col], errors = 'coerce')
                        self._pool[code] = df
                        return df
        except Exception:
            pass
        try:
            import akshare as ak
            df = ak.stock_financial_analysis_indicator(symbol = code, start_year = '2016')
            if df is None or df.empty:
                return None
            if '日期' in df.columns:
                df.index = pd.to_datetime(df['日期'])
                df = df.drop(columns = ['日期'])
            try:
                df.index = [str(i) for i in df.index]
                cache_data = {
                    'index': list(df.index),
                    'columns': list(df.columns),
                    'data': df.values.tolist() }
                with open(cache_path, 'w') as f:
                    json.dump(cache_data, f)
            except Exception:
                pass
            df.index = pd.to_datetime(df.index)
            self._pool[code] = df
            return df
        except Exception:
            return None

    
    def get_financial(self, code: str, year: int) -> Optional[pd.DataFrame]:
        """
        获取截止某年的财务指标。
        
        例: get_financial('300750', 2021) → 截止 2021-12-31 的数据。
        如果该年无数据, 取最近的前一年。
        """
        full = self._load_all(code)
        if full is None or full.empty:
            return None
        mask = full.index <= pd.Timestamp(f'{year}-12-31')
        sliced = full[mask].copy()
        if sliced.empty:
            earliest = full.index.min().year
            if earliest > year:
                return None
            mask2 = full.index <= pd.Timestamp(f'{earliest + 1}-12-31')
            sliced = full[mask2].copy()
        return sliced

    
    def get_price_return(self, code: str, from_year: int, to_year: int) -> float:
        '''计算某段时间的股价涨跌幅'''
        try:
            import akshare as ak
            if code.startswith('0') or code.startswith('3'):
                symbol = f'sz{code}'
            else:
                symbol = f'sh{code}'
            df = ak.stock_zh_index_daily(symbol = symbol)
            if df is None or len(df) == 0:
                return 0.0
            df = df.sort_values('date')
            mask_from = pd.to_datetime(df['date']) >= pd.Timestamp(f'{from_year}-01-01')
            mask_to = pd.to_datetime(df['date']) <= pd.Timestamp(f'{to_year}-12-31')
            window = df[mask_from & mask_to]
            if len(window) < 2:
                return 0.0
            start_price = float(window['close'].iloc[0])
            end_price = float(window['close'].iloc[-1])
            return end_price / start_price - 1
        except Exception:
            return 0.0

    
    def prefetch(self, codes: List[str], start_year: int, end_year: int):
        '''预加载多只股票的完整数据到内存池'''
        print(f'  预加载 {len(codes)} 只股票...')
        for i, code in enumerate(codes):
            self._load_all(code)
            if (i + 1) % 5 == 0:
                print(f'    ... {i + 1}/{len(codes)}')
        print('  ✅ 预加载完成')
        return None


INDUSTRY_STOCKS = {
    'tech': [
        '300750',
        '宁德时代',
        '002415',
        '海康威视',
        '000725',
        '京东方A',
        '002230',
        '科大讯飞',
        '300124',
        '汇川技术',
        '002049',
        '紫光国微',
        '688981',
        '中芯国际',
        '300274',
        '阳光电源',
        '002459',
        '晶澳科技',
        '688012',
        '中微公司'],
    'consumer': [
        '600519',
        '贵州茅台',
        '000858',
        '五粮液',
        '002304',
        '洋河股份',
        '600887',
        '伊利股份',
        '002714',
        '牧原股份',
        '603288',
        '海天味业',
        '600809',
        '山西汾酒',
        '000568',
        '泸州老窖',
        '002568',
        '百润股份',
        '600882',
        '妙可蓝多'],
    'manufacturing': [
        '000333',
        '美的集团',
        '000651',
        '格力电器',
        '600690',
        '海尔智家',
        '601899',
        '紫金矿业',
        '600585',
        '海螺水泥',
        '600309',
        '万华化学',
        '601012',
        '隆基绿能',
        '002129',
        '中环股份',
        '600031',
        '三一重工',
        '000338',
        '潍柴动力'],
    'real_estate': [
        '600048',
        '保利发展',
        '000002',
        '万科A'] }

def run_industry_backtest(industry: str, start_year: int = 2018, end_year: int = 2025, prefetch: bool = True, cache_dir: str = None) -> BacktestReport:
    """
    运行单个行业的完整回测。
    
    使用方式::
    
        from spum_backtest import run_industry_backtest
        report = run_industry_backtest('tech')
        report.print_report()
    """
    stocks = INDUSTRY_STOCKS.get(industry, [])
    codes = stocks[::2]
    if not codes:
        print(f'⚠ 未找到行业 {industry} 的股票列表')
        return BacktestReport(industry, industry, [])
    store = DataStore(cache_dir)
    engine = BacktestEngine(industry)
    report = engine.backtest(store, codes, start_year, end_year)
    return report


def run_all_backtests(start_year: int = 2018, end_year: int = 2025) -> Dict[str, BacktestReport]:
    '''运行所有支持行业的回测'''
    reports = { }
    for industry in ('tech', 'consumer', 'manufacturing', 'real_estate'):
        print(f'\n  ═══ {INDUSTRY_MAP[industry]["name"]} ({industry}) ═══')
        report = run_industry_backtest(industry, start_year, end_year, prefetch = False)
        reports[industry] = report
        report.print_report()
    return reports
