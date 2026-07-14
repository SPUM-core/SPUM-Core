"""
backtest_engine.py — SPUM 五形回测核心引擎
===========================================

职责:
  1. 组合仿真（基于信号的多空仓位）
  2. 绩效指标计算（夏普/最大回撤/IC序列/命中率）
  3. 多策略对比框架

用法:
  from backtest_engine import BacktestEngine, compute_ic_series, compute_portfolio_metrics
"""
import json, math, os
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
#  数据容器
# ============================================================

@dataclass
class StrategyResult:
    """单个策略的回测结果"""
    name: str
    daily_returns: list[float]          # 每日收益率序列
    cumulative_returns: list[float]     # 累计收益率序列
    signals: list[float]                # 每日信号值（-1~1）
    positions: list[float]              # 每日仓位
    trades: list[dict]                  # 交易记录
    metrics: dict                       # 绩效指标
    ic_series: list[float] = field(default_factory=list)         # IC 序列
    ic_metrics: dict = field(default_factory=dict)               # IC 汇总


def np_mean(arr):
    if isinstance(arr, np.ndarray):
        return float(np.mean(arr)) if arr.size > 0 else 0.0
    return sum(arr) / len(arr) if arr else 0.0

def np_std(arr, ddof=1):
    n = arr.size if isinstance(arr, np.ndarray) else len(arr)
    if n < 2:
        return 0.0
    m = np_mean(arr)
    if isinstance(arr, np.ndarray):
        return float(np.std(arr, ddof=ddof))
    return math.sqrt(sum((x - m) ** 2 for x in arr) / (n - ddof))


# ============================================================
#  绩效指标计算
# ============================================================

def compute_portfolio_metrics(daily_returns: list[float], rf_annual: float = 0.02) -> dict:
    """
    从每日收益率序列计算全套绩效指标。
    
    参数:
        daily_returns: 每日收益率序列（小数，如 0.01 代表 1%）
        rf_annual:     年化无风险利率（默认 2%）
    
    返回:
        metrics dict
    """
    ret = np.asarray(daily_returns, dtype=float)
    if ret.size < 5:
        return {
            'total_return': 0, 'annual_return': 0, 'annual_vol': 0,
            'sharpe': 0, 'max_drawdown': 0, 'calmar': 0,
            'hit_rate': 0, 'profit_factor': 0,
            'avg_gain': 0, 'avg_loss': 0, 'win_days': 0, 'loss_days': 0,
        }

    n = len(ret)
    trading_days = 252  # A 股年化交易日

    # 累计收益
    total_return = float(np.prod(1 + ret) - 1)
    annual_return = float((1 + total_return) ** (trading_days / n) - 1)

    # 年化波动
    annual_vol = float(np_std(ret) * math.sqrt(trading_days))

    # 夏普比率
    rf_daily = (1 + rf_annual) ** (1 / trading_days) - 1
    excess = ret - rf_daily
    sharpe = float(np_mean(excess) / (np_std(excess) + 1e-10) * math.sqrt(trading_days))

    # 最大回撤
    cum = np.cumprod(1 + ret)
    running_max = np.maximum.accumulate(cum)
    drawdown = (cum - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    # 卡尔玛比率
    calmar = annual_return / (abs(max_drawdown) + 1e-10)

    # 命中率 & 盈亏比
    win = ret[ret > 0]
    loss = ret[ret < 0]
    hit_rate = len(win) / n if n > 0 else 0
    avg_gain = float(np_mean(win)) if len(win) > 0 else 0
    avg_loss = float(np_mean(loss)) if len(loss) > 0 else 0
    profit_factor = abs(avg_gain / (avg_loss + 1e-10)) if avg_loss != 0 else float('inf')

    return {
        'total_return': round(total_return * 100, 2),       # %
        'annual_return': round(annual_return * 100, 2),
        'annual_vol': round(annual_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'max_drawdown': round(max_drawdown * 100, 2),       # %
        'calmar': round(calmar, 3),
        'hit_rate': round(hit_rate * 100, 1),               # %
        'profit_factor': round(profit_factor, 3),
        'avg_gain': round(avg_gain * 100, 3),               # %
        'avg_loss': round(avg_loss * 100, 3),
        'win_days': int(len(win)),
        'loss_days': int(len(loss)),
        'total_days': n,
    }


def compute_ic_series(predictions: list[float], actuals: list[float], window: int = 20) -> list[float]:
    """
    计算滚动 IC（Information Coefficient）序列。
    IC = Spearman 秩相关系数，衡量预测值与实际值的排序一致性。
    
    参数:
        predictions:  预测值序列（截面数据——多个股票在同一个时间点的预测）
        actuals:      实际值序列
        window:       IC 计算窗口（滚动）
    
    返回:
        IC 序列（长度 = len - window + 1）
    """
    n = min(len(predictions), len(actuals))
    if n < window + 1:
        return []
    # 转化为排列秩
    ic_list = []
    for i in range(window, n):
        p = predictions[i - window:i]
        a = actuals[i - window:i]
        rp = pd.Series(p).rank().values
        ra = pd.Series(a).rank().values
        rp_m = np_mean(rp)
        ra_m = np_mean(ra)
        num = sum((rp[j] - rp_m) * (ra[j] - ra_m) for j in range(window))
        den = math.sqrt(sum((rp[j] - rp_m)**2 for j in range(window)) *
                        sum((ra[j] - ra_m)**2 for j in range(window))) + 1e-10
        ic_list.append(num / den)
    return ic_list


def compute_ic_metrics(ic_series: list[float]) -> dict:
    """从 IC 序列计算汇总指标"""
    if not ic_series or len(ic_series) < 2:
        return {'ic_mean': 0, 'ic_std': 0, 'ic_ir': 0, 'ic_pos_ratio': 0}
    ic_mean = float(np_mean(ic_series))
    ic_std = float(np_std(ic_series))
    return {
        'ic_mean': round(ic_mean, 4),
        'ic_std': round(ic_std, 4),
        'ic_ir': round(ic_mean / (ic_std + 1e-10), 4),   # Information Ratio
        'ic_pos_ratio': round(sum(1 for x in ic_series if x > 0) / len(ic_series) * 100, 1),
    }


# ============================================================
#  回测引擎
# ============================================================

@dataclass
class TradeRecord:
    date: str
    sym: str
    direction: int        # 1=long, -1=short
    entry_price: float
    size: float           # 仓位比例
    reason: str = ''


class BacktestEngine:
    """
    轻量回测引擎。支持多信号源策略的成本感知组合仿真。
    
    用法:
        engine = BacktestEngine(initial_capital=1_000_000, commission_rate=0.0003)
        result = engine.run(
            dates=[...],
            signals=[...],        # 每日信号 -1~1
            prices=[...],          # 每日价格（用于计算收益率）
            strategy_name='S₀ 同命格'
        )
    """
    
    def __init__(self, initial_capital: float = 1_000_000,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
    
    def run(self, dates: list[str], signals: list[float], prices: list[float],
            benchmark_returns: Optional[list[float]] = None,
            strategy_name: str = 'Strategy') -> StrategyResult:
        """
        运行单策略回测。
        
        参数:
            dates:        交易日列表
            signals:      每日信号值 [-1, 1]（滞后一期生效）
            prices:       每日价格（用于计算实际收益率）
            strategy_name: 策略名称
        
        返回:
            StrategyResult
        """
        n = min(len(dates), len(signals), len(prices))
        if n < 20:
            return StrategyResult(
                name=strategy_name, daily_returns=[], cumulative_returns=[],
                signals=[], positions=[], trades=[], metrics={}
            )
        
        dates = dates[:n]
        signals = signals[:n]
        prices = prices[:n]
        
        capital = self.initial_capital
        position = 0.0          # 当前持仓比例
        daily_returns = []
        cumulative_returns = [1.0]
        trade_log = []
        
        for i in range(1, n):
            # 信号滞后一期（避免前视偏差）
            prev_signal = signals[i - 1]
            
            # 信号 → 目标仓位
            target_pos = max(-1.0, min(1.0, prev_signal))
            
            # 计算交易成本
            delta_pos = target_pos - position
            if abs(delta_pos) > 0.001:
                cost = abs(delta_pos) * (
                    self.commission_rate + self.slippage
                )
            else:
                cost = 0.0
            
            # 当日收益率（考虑仓位和成本）
            ret = (prices[i] / prices[i - 1]) - 1 if prices[i - 1] > 0 else 0
            portfolio_ret = position * ret - cost
            
            daily_returns.append(portfolio_ret)
            cumulative_returns.append(cumulative_returns[-1] * (1 + portfolio_ret))
            
            # 记录调仓
            if abs(delta_pos) > 0.02 and target_pos != position:
                trade_log.append({
                    'date': dates[i], 'direction': 'long' if target_pos > 0 else 'short',
                    'delta': round(delta_pos, 3), 'pos_after': round(target_pos, 3),
                    'cost': round(cost * 100, 3),
                })
            
            position = target_pos
        
        # 绩效指标
        metrics = compute_portfolio_metrics(daily_returns)
        
        return StrategyResult(
            name=strategy_name,
            daily_returns=daily_returns,
            cumulative_returns=cumulative_returns,
            signals=signals,
            positions=[max(-1, min(1, s)) for s in signals],
            trades=trade_log,
            metrics=metrics,
        )


# ============================================================
#  多策略对比
# ============================================================

def compare_strategies(results: list[StrategyResult]) -> str:
    """
    生成多策略对比表格文字版。
    
    参数:
        results: StrategyResult 列表
    
    返回:
        格式化的对比报告字符串
    """
    if not results:
        return '（无回测结果）'
    
    lines = []
    lines.append(f'{"=" * 90}')
    lines.append(f'SPUM 五形回测框架 — 多策略对比')
    lines.append(f'生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'{"=" * 90}')
    
    # 指标列表
    metric_keys = [
        ('total_return', '总收益率', '%'),
        ('annual_return', '年化收益率', '%'),
        ('annual_vol', '年化波动', '%'),
        ('sharpe', '夏普比率', ''),
        ('max_drawdown', '最大回撤', '%'),
        ('calmar', '卡尔玛比率', ''),
        ('hit_rate', '命中率', '%'),
        ('profit_factor', '盈亏比', ''),
    ]
    
    # 表头
    header = f'{"策略":<24}'
    for label, _, unit in metric_keys:
        header += f' {label:<10}'
    header += ' IC均值 IC-IR'
    lines.append(header)
    lines.append('-' * 90)
    
    for r in sorted(results, key=lambda x: x.metrics.get('sharpe', 0), reverse=True):
        if not r.metrics:
            continue
        row = f'{r.name:<24}'
        for key, _, unit in metric_keys:
            val = r.metrics.get(key, 0)
            if unit == '%':
                row += f' {val:>+8.1f}%'
            elif key == 'sharpe' or key == 'calmar':
                row += f' {val:>+10.3f}'
            else:
                row += f' {val:>10.3f}'
        ic_mean = r.ic_metrics.get('ic_mean', 0)
        ic_ir = r.ic_metrics.get('ic_ir', 0)
        row += f' {ic_mean:>+7.4f} {ic_ir:>+6.2f}'
        lines.append(row)
    
    lines.append('-' * 90)
    
    # 最佳策略摘要
    best_sharpe = max(results, key=lambda r: r.metrics.get('sharpe', -999))
    best_return = max(results, key=lambda r: r.metrics.get('total_return', -999))
    best_ic = max(results, key=lambda r: r.ic_metrics.get('ic_mean', -999))
    
    lines.append(f'\n【最佳夏普】 {best_sharpe.name}  Sharpe={best_sharpe.metrics.get("sharpe", 0):.3f}')
    lines.append(f'【最佳收益】 {best_return.name}  Return={best_return.metrics.get("total_return", 0):.2f}%')
    lines.append(f'【最佳 IC 】 {best_ic.name}  IC_mean={best_ic.ic_metrics.get("ic_mean", 0):.4f}')
    
    # 交易统计
    lines.append(f'\n{"=" * 90}')
    lines.append(f'交易明细')
    for r in results:
        if r.trades:
            lines.append(f'\n{r.name} ({len(r.trades)} 笔调仓):')
            for t in r.trades[:10]:  # 只显示前 10 笔
                lines.append(f'  {t["date"]} {t["direction"]:<5} Δ={t["delta"]:<+7.3f} '
                           f'仓位={t["pos_after"]:<+7.3f} 成本={t["cost"]:.3f}%')
            if len(r.trades) > 10:
                lines.append(f'  ... 还有 {len(r.trades) - 10} 笔')
    
    return '\n'.join(lines)


# ============================================================
#  工具：从 vis_data.json 提取 S₀ 配对信号
# ============================================================

def load_s0_pairs() -> list[dict]:
    """从 vis_data.json 加载配对数据"""
    path = os.path.join(DATA_DIR, 'vis_data.json')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data.get('pairs', [])


def load_stock_data() -> list[dict]:
    """从 vis_data.json 加载股票 S₀ 数据"""
    path = os.path.join(DATA_DIR, 'vis_data.json')
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data.get('stocks', [])


def load_vol_data() -> dict:
    """从 vis_data.json 加载波动率数据"""
    path = os.path.join(DATA_DIR, 'vis_data.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data.get('vol_data', {})


# ============================================================
#  简易截面 IC 计算（用于 S₀ 配对信号）
# ============================================================

def compute_cross_sectional_ic(pairs: list[dict]) -> dict:
    """
    计算 S₀ 配对信号的截面 IC。
    
    对每个配对 (source→target)：
    - 预测值 = h1_xgb（信号源对目标的预测提升 %）
    - 实际值 = 配对实际是否有效（我们用 h1_ridge 作为实际代理）
    
    返回:
        IC 汇总 + 按命格分组的 IC
    """
    valid = [p for p in pairs if p.get('h1_xgb') is not None and p.get('h1_ridge') is not None]
    if len(valid) < 5:
        return {'ic_mean': 0, 'ic_std': 0, 'ic_ir': 0, 'n': 0}
    
    preds = np.array([p['h1_xgb'] for p in valid])
    actuals = np.array([p['h1_ridge'] for p in valid])
    
    # Spearman 秩相关
    rp = pd.Series(preds).rank().values
    ra = pd.Series(actuals).rank().values
    rp_m = np_mean(rp)
    ra_m = np_mean(ra)
    n = len(preds)
    num = sum((rp[i] - rp_m) * (ra[i] - ra_m) for i in range(n))
    den = math.sqrt(sum((rp[i] - rp_m)**2 for i in range(n)) *
                    sum((ra[i] - ra_m)**2 for i in range(n))) + 1e-10
    ic = num / den
    
    # 按命格分组
    by_label = defaultdict(list)
    for p in valid:
        by_label[p['label']].append((p['h1_xgb'], p['h1_ridge']))
    
    label_ics = {}
    for label, pairs_list in by_label.items():
        if len(pairs_list) < 3:
            continue
        pp = np.array([x[0] for x in pairs_list])
        aa = np.array([x[1] for x in pairs_list])
        rpp = pd.Series(pp).rank().values
        raa = pd.Series(aa).rank().values
        rpp_m = np_mean(rpp)
        raa_m = np_mean(raa)
        nn = len(pairs_list)
        n_num = sum((rpp[j] - rpp_m) * (raa[j] - raa_m) for j in range(nn))
        n_den = math.sqrt(sum((rpp[j] - rpp_m)**2 for j in range(nn)) *
                          sum((raa[j] - raa_m)**2 for j in range(nn))) + 1e-10
        label_ics[label] = round(n_num / n_den, 4)
    
    return {
        'ic_mean': round(float(ic), 4),
        'ic_std': round(float(np_std(list(label_ics.values())) if label_ics else 0), 4),
        'ic_ir': round(float(ic / (np_std(list(label_ics.values())) + 1e-10)), 4) if label_ics else 0,
        'n': len(valid),
        'by_label': label_ics,
    }


if __name__ == '__main__':
    # 快速自检
    pairs = load_s0_pairs()
    print(f'配对数据: {len(pairs)} 对')
    if pairs:
        ic = compute_cross_sectional_ic(pairs)
        print(f'截面 IC: {ic}')
    
    # 示例：随机信号对比
    np.random.seed(42)
    n = 252
    dates = [f'2025-01-{i+1:02d}' for i in range(n)]
    prices = [100 * (1 + sum(np.random.randn(j+1) * 0.01 for j in range(i))) / (i+1) for i in range(n)]
    
    engine = BacktestEngine()
    
    # 策略 1: 随机信号
    sig1 = np.random.randn(n) * 0.3
    sig1 = np.clip(sig1, -1, 1)
    r1 = engine.run(dates, sig1.tolist(), prices, strategy_name='随机信号')
    print(f'随机信号    Sharpe={r1.metrics.get("sharpe", 0):.3f} '
          f'Return={r1.metrics.get("total_return", 0):.1f}%')
    
    # 策略 2: 动量（正信号——价格涨则做多）
    sig2 = []
    for i in range(n):
        if i < 5:
            sig2.append(0)
        else:
            mom = (prices[i] / prices[i-5] - 1) * 3
            sig2.append(max(-1, min(1, mom)))
    r2 = engine.run(dates, sig2, prices, strategy_name='5日动量')
    print(f'5日动量     Sharpe={r2.metrics.get("sharpe", 0):.3f} '
          f'Return={r2.metrics.get("total_return", 0):.1f}%')
