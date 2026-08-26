# Source Generated with Decompyle++
# File: stock_sigma_trader.cpython-311.pyc (Python 3.11)

'''
SPUM σ 振荡 → 交易信号引擎
================================

从 σ 分析结果生成可执行的买卖信号, 含仓位管理和风控。

三层架构:
  Layer 1: 单股信号生成 (σ → 买卖动作 + 仓位)
  Layer 2: 多股组合信号 (板块共识 → 仓位调整)
  Layer 3: 风控过滤 (止损/回撤/集中度)

输出格式: JSON 序列化, 兼容 PTrade/QMT 的 Python 策略接口。

依赖: stock_sigma.py, stock_sigma_multi.py
'''
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from datetime import datetime, date
import warnings


@dataclass
class TradingConfig:
    '''交易配置'''
    max_position_pct: float = 0.3
    max_total_pct: float = 0.8
    min_confidence: float = 0.5
    daily_stop_loss: float = -0.02
    max_drawdown: float = -0.15
    min_sigma_change: float = 0.01
    position_tiers: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.5, 0.1), (0.6, 0.2), (0.75, 0.3)]
    )
    require_crossvote: bool = True
    crossvote_bonus: float = 0.1
    crossvote_penalty: float = 0.5
    output_format: str = 'json'


@dataclass
class TradeSignal:
    '''单笔交易信号 (JSON 可序列化)'''
    timestamp: str
    stock_code: str
    stock_name: str = ''
    action: str = 'hold'
    sigma_current: float = 0.0
    sigma_pred: float = 0.0
    sigma_gradient: float = 0.0
    confidence: float = 0.0
    confidence_enhanced: float = 0.0
    position_pct: float = 0.0
    stop_loss: float = -0.02
    take_profit: float = 0.05
    max_hold_days: int = 10
    sector: str = ''
    sector_phase: str = ''
    crossvote: float = 0.0
    vote_label: str = ''
    reason: str = ''
    phase: str = ''

    def to_dict(self) -> dict:
        '''转为字典 (JSON 兼容)'''
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}

    def to_json(self, indent: int = 2) -> str:
        '''转为 JSON 字符串'''
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class PortfolioState:
    '''投资组合状态 (用于风控)'''
    total_value: float = 1000000.0
    cash: float = 1000000.0
    positions: Dict[str, dict] = field(default_factory=dict)
    peak_value: float = 1000000.0
    daily_pnl: float = 0.0
    is_frozen: bool = False


class SignalGenerator:
    '''
    σ → 交易信号转换器。

    将 StockSigma.analyze() 和 MultiStockGraph 的输出
    转换为可执行的买卖信号。
    '''

    def __init__(self, config: Optional[TradingConfig] = None):
        self.cfg = config if config else TradingConfig()

    def generate(self, stock_result: dict, stock_code: str = '', stock_name: str = '', sector: str = '', multi_enhanced: Optional[dict] = None, current_price: float = 0.0) -> TradeSignal:
        '''
        从单股 σ 分析结果生成交易信号。

        参数
        ----------
        stock_result : dict
            StockSigma.analyze() 的返回结果
        stock_code : str
            股票代码
        stock_name : str
            股票名称
        sector : str
            所属板块
        multi_enhanced : dict or None
            MultiStockGraph.enhance_prediction() 的返回结果
        current_price : float
            当前价格 (用于止盈止损计算)

        返回
        -------
        TradeSignal
        '''
        sigma = stock_result['sigma']
        gradient = stock_result['gradient']
        phase_label = stock_result['phase_label']
        pred = stock_result['prediction']
        now = datetime.now().isoformat(timespec='seconds')
        signal = TradeSignal(
            timestamp=now,
            stock_code=stock_code,
            stock_name=stock_name,
            action='hold',
            sigma_current=float(sigma.iloc[-1]),
            sigma_pred=pred['sigma_pred'],
            sigma_gradient=float(gradient.iloc[-1]),
            confidence=pred['confidence'],
            confidence_enhanced=pred['confidence'],
            phase=str(phase_label.iloc[-1]),
            sector=sector,
            reason='',
        )
        if multi_enhanced:
            signal.sector_phase = multi_enhanced.get('sector_phase', '')
            signal.crossvote = multi_enhanced.get('crossvote', 0.0)
            signal.vote_label = multi_enhanced.get('vote_label', '')
            signal.confidence_enhanced = multi_enhanced.get('confidence_enhanced', pred['confidence'])
            if multi_enhanced.get('divergence_warn'):
                signal.action = 'hold'
                signal.reason = 'σ 背离预警: 个股脱离板块共识, 不交易'
                return signal
        conf = signal.confidence_enhanced
        sigma_t = signal.sigma_current
        sigma_t1 = signal.sigma_pred
        delta_sigma = sigma_t1 - sigma_t
        if abs(delta_sigma) < self.cfg.min_sigma_change:
            signal.action = 'hold'
            signal.reason = f'σ 变化太小 (Δ={delta_sigma:.3f}), 不交易'
            return signal
        if conf < self.cfg.min_confidence:
            signal.action = 'hold'
            signal.reason = f'信心不足 ({conf:.0%}), 不交易'
            return signal
        if delta_sigma < 0:
            signal.action = 'enter_long'
            signal.reason = f'σ {sigma_t:.3f}→{sigma_t1:.3f} (Δ={delta_sigma:.3f}), {signal.phase}末期, 板块{signal.sector_phase}'
        elif delta_sigma > 0:
            signal.action = 'enter_short'
            signal.reason = f'σ {sigma_t:.3f}→{sigma_t1:.3f} (Δ={delta_sigma:.3f}), {signal.phase}末期, 板块{signal.sector_phase}'
        signal.position_pct = self._calc_position(conf, signal.crossvote)
        signal.stop_loss = self.cfg.daily_stop_loss
        signal.take_profit = -(self.cfg.daily_stop_loss) * 2.5
        if signal.crossvote >= 0.6:
            signal.reason += f' | ★ 板块强共识 ({signal.crossvote:.0%})'
        elif signal.crossvote <= -0.3:
            signal.reason += f' | ⚠ 板块反向 ({signal.crossvote:.0%})'
        return signal

    def _calc_position(self, confidence: float, crossvote: float) -> float:
        '''
        基于信心和板块验证计算仓位。

        仓位阶梯:
          ≥0.75 → 30%
          ≥0.60 → 20%
          ≥0.50 → 10%
          <0.50 → 0% (不交易)

        板块调整:
          强共识 (crossvote ≥ 0.6) → +10%
          反向 (crossvote ≤ -0.3) → 减半
        '''
        tiers = sorted(self.cfg.position_tiers, key=lambda x: -x[0])
        base_pct = 0.0
        for threshold, pct in tiers:
            if confidence >= threshold:
                base_pct = pct
                break
        if crossvote >= 0.6:
            base_pct = min(self.cfg.max_position_pct, base_pct + self.cfg.crossvote_bonus)
        elif crossvote <= -0.3:
            base_pct *= self.cfg.crossvote_penalty
        return round(base_pct, 2)


class PortfolioSignalAggregator:
    '''
    组合信号聚合器。

    将多个个股信号聚合成组合级别的交易计划,
    确保总仓位不超限, 按信心排序执行。
    '''

    def __init__(self, config: TradingConfig, portfolio: Optional[PortfolioState] = None):
        self.cfg = config
        self.portfolio = portfolio if portfolio else PortfolioState()

    def aggregate(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        '''
        聚合并排序信号:
          1. 按信心排序
          2. 按总仓位上限裁剪
          3. 已有持仓的信号优先级更高
          4. 风控检查
        '''
        if not signals:
            return []
        ranked = sorted(signals, key=lambda s: s.confidence_enhanced, reverse=True)
        held_codes = set(self.portfolio.positions.keys())
        held = [s for s in ranked if s.stock_code in held_codes]
        new = [s for s in ranked if s.stock_code not in held_codes]
        ranked = held + new
        ranked = [s for s in ranked if self._pass_risk_check(s)]
        total_pct = sum(
            self.portfolio.positions.get(s.stock_code, {}).get('position_pct', 0)
            for s in ranked
        )
        final = []
        for s in ranked:
            if total_pct >= self.cfg.max_total_pct:
                s.action = 'hold'
                s.reason += f' | 总仓位已达上限 ({total_pct:.0%})'
                s.position_pct = 0.0
            final.append(s)
            total_pct += s.position_pct
        return final

    def _pass_risk_check(self, signal: TradeSignal) -> bool:
        '''风控检查'''
        if self.portfolio.is_frozen:
            signal.reason += ' | 风控冻结'
            signal.action = 'hold'
            signal.position_pct = 0.0
            return False
        dd = (self.portfolio.total_value - self.portfolio.peak_value) / max(self.portfolio.peak_value, 1)
        if dd <= self.cfg.max_drawdown:
            signal.reason += f' | 最大回撤触发 ({dd:.1%})'
            signal.action = 'exit_long'
            signal.position_pct = 0.0
            self.portfolio.is_frozen = True
            return False
        pos = self.portfolio.positions.get(signal.stock_code)
        if pos and signal.action == 'hold':
            entry_price = pos.get('avg_price', 0)
            current_price = pos.get('current_price', entry_price)
            if entry_price > 0:
                pnl = (current_price - entry_price) / entry_price
                if pnl <= self.cfg.daily_stop_loss:
                    signal.action = 'exit_long'
                    signal.reason = f'触发止损 ({pnl:.1%})'
                    return True
        return True


class PTradeStrategyTemplate:
    '''
    PTrade 策略模板。

    使用方式: 将此文件放入 PTrade 的策略目录,
    在 PTrade 的 Python 策略编辑器中 import 后调用。

    PTrade API 参考:
      - order(stock_code, amount, price_type=5)
      - order_target_percent(stock_code, percent, price_type=5)
      - get_position(stock_code)
      - get_portfolio()
      - log.info(msg)
    '''

    def __init__(self, signal_generator: SignalGenerator, aggregator: PortfolioSignalAggregator):
        self.sg = signal_generator
        self.ag = aggregator

    def on_sigma_signal(self, stock_result: dict, stock_code: str, stock_name: str, sector: str = '', multi_enhanced: Optional[dict] = None):
        '''
        PTrade 回调: 每次 σ 分析完成后调用此方法。

        典型用法::

            # 在 PTrade 策略中
            def handlebar(context):
                analyzer = StockSigma()
                result = analyzer.analyze(df)
                trader.on_sigma_signal(result, '000001', '平安银行')
        '''
        signal = self.sg.generate(stock_result, stock_code, stock_name, sector, multi_enhanced)
        if signal.action == 'enter_long':
            target = signal.position_pct
            print(f'[PTrade] 买入 {stock_code} ({stock_name}): 仓位 {target:.0%}, 信心 {signal.confidence_enhanced:.0%}')
            print(f'         原因: {signal.reason}')
        elif signal.action == 'exit_long':
            print(f'[PTrade] 卖出 {stock_code} ({stock_name}): 平仓')
            print(f'         原因: {signal.reason}')
        elif signal.action == 'enter_short':
            print(f'[PTrade] 融券卖出 {stock_code} ({stock_name}): 仓位 {signal.position_pct:.0%}')
            print(f'         原因: {signal.reason}')
        elif signal.action == 'hold':
            pass
        return signal


class QMTStrategyTemplate:
    '''
    QMT (迅投) 策略模板。

    QMT API 参考:
      - passorder(23, 1, stock_code, price_type, price, amount, ...)
      - get_market_data(['close'], stock_code)
      - get_position(stock_code)
      - log.debug(msg)

    典型用法::

        # 在 QMT 策略的 handlebar 中
        def handlebar(ContextInfo):
            trader = QMTStrategyTemplate(signal_gen, aggregator)
            trader.execute(ContextInfo, signals)
    '''

    def __init__(self, signal_generator: SignalGenerator, aggregator: PortfolioSignalAggregator):
        self.sg = signal_generator
        self.ag = aggregator

    def execute(self, context, signals: List[TradeSignal]):
        '''
        在 QMT 的 handlebar 中执行信号。

        context : QMT 的 ContextInfo 对象
        signals : List[TradeSignal]
        '''
        for sig in signals:
            if sig.action == 'enter_long':
                print(f'[QMT] 买入 {sig.stock_code}: 仓位 {sig.position_pct:.0%}, 信心 {sig.confidence_enhanced:.0%}')
            elif sig.action == 'exit_long':
                print(f'[QMT] 卖出 {sig.stock_code}: 平仓')
            elif sig.action == 'enter_short':
                print(f'[QMT] 融券卖出 {sig.stock_code}')
            elif sig.action == 'exit_short':
                print(f'[QMT] 买券还券 {sig.stock_code}')


def export_signals(signals: List[TradeSignal], format: str = 'json', filepath: Optional[str] = None) -> Optional[str]:
    '''
    导出信号到文件或返回字符串。

    参数
    ----------
    signals : List[TradeSignal]
        信号列表
    format : str
        'json' | 'csv'
    filepath : str or None
        文件路径, None 则返回字符串
    '''
    if format == 'json':
        data = [s.to_dict() for s in signals]
        output = json.dumps(data, ensure_ascii=False, indent=2)
    elif format == 'csv':
        import io
        buf = io.StringIO()
        fields = ['timestamp', 'stock_code', 'stock_name', 'action', 'confidence_enhanced', 'position_pct', 'sigma_current', 'sigma_pred', 'sector_phase', 'crossvote', 'reason']
        buf.write(','.join(fields) + '\n')
        for s in signals:
            row = [str(getattr(s, f, '')) for f in fields]
            buf.write(','.join(row) + '\n')
        output = buf.getvalue()
    else:
        raise ValueError(f'不支持的输出格式: {format}')
    if filepath:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(output)
        return None
    return output


def print_signal_summary(signals: List[TradeSignal]):
    '''简洁打印信号列表'''
    if not signals:
        print('无信号')
        return
    enters = [s for s in signals if 'enter' in s.action]
    exits = [s for s in signals if 'exit' in s.action]
    holds = [s for s in signals if s.action == 'hold']
    print('============================================================')
    print('  SPUM σ 交易信号摘要')
    print('============================================================')
    print(f'  买入: {len(enters)}  卖出: {len(exits)}  观望: {len(holds)}')
    print('  ' + '─' * 56)
    if enters:
        print('\n  🟢 买入信号')
        for s in enters:
            dir_arrow = '↑' if 'long' in s.action else '↓'
            print(f'    {dir_arrow} {s.stock_code:>8s} {s.stock_name or "":8s} 仓位 {s.position_pct:.0%} 信心 {s.confidence_enhanced:.0%} [{s.vote_label}]')
    if exits:
        print('\n  🔴 卖出信号')
        for s in exits:
            print(f'    ✕ {s.stock_code:>8s} {s.stock_name or "":8s}  {s.reason[:40]}')
    print('\n' + '=' * 60)
