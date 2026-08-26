# Source Generated with Decompyle++
# File: stock_sigma_optimizer.cpython-311.pyc (Python 3.11)

"""
SPUM σ 振荡分析器 — 参数调优 & 回测管道
=============================================

对 StockSigma 模型做滚动回测 + 网格搜索参数优化:
  - 逐帧滚动预测方向 vs 实际价格方向 → 计算命中率
  - 多组参数网格搜索 → 找最优组合
  - 输出收益曲线、最大回撤、夏普比

用法::

    from stock_sigma_optimizer import ParameterOptimizer

    opt = ParameterOptimizer()
    result = opt.optimize(df)          # 自动网格搜索
    opt.report(result)                 # 打印最优参数 + 性能

    # 用最佳参数做最终预测
    best = result['best_params']
    analyzer = StockSigma(SigmaConfig(**best))
    final = analyzer.analyze(df)
"""
import numpy as np
import pandas as pd
from itertools import product
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import time
import warnings


@dataclass
class BacktestResult:
    '''单次滚动回测的结果。'''
    params: dict
    hit_rate: float
    total_trades: int
    cumulative_return: float
    sharpe: float
    max_drawdown: float
    daily_returns: np.ndarray
    predictions: np.ndarray
    actuals: np.ndarray
    sigma_preds: np.ndarray
    confidences: np.ndarray


class BacktestEngine:
    '''
    滚动回测引擎。

    逐帧模拟: 用截至当日的数据计算 σ, 预测下一帧方向,
    与实际方向对比, 累积收益。
    '''

    def __init__(self, min_train: int = 60, verbose: bool = False):
        self.min_train = min_train
        self.verbose = verbose

    def run(self, df: pd.DataFrame, params: dict) -> BacktestResult:
        '''
        在给定参数下执行完整滚动回测。

        参数
        ----------
        df : pd.DataFrame
            必须含 close, volume, high, low
        params : dict
            与 SigmaConfig 字段对应

        返回
        -------
        BacktestResult
        '''
        from stock_sigma import StockSigma, SigmaConfig
        cfg = SigmaConfig(**params)
        analyzer = StockSigma(cfg)
        close = df['close'].values
        n = len(close)
        predictions = np.zeros(n)
        actuals = np.zeros(n)
        sigma_preds = np.zeros(n)
        confidences = np.zeros(n)
        daily_returns = np.zeros(n)
        full = analyzer.analyze(df)
        sigma_full = full['sigma'].values
        grad_full = full['gradient'].values
        sigma_eq = full['sigma_eq']
        trade_count = 0
        for t in range(self.min_train, n - 1):
            sigma_t = sigma_full[t]
            grad_t = grad_full[t]
            pred = self._predict_one(sigma_t, grad_t, sigma_eq, cfg)
            pred_dir = pred['direction']
            sigma_pred = pred['sigma_pred']
            conf = pred['confidence']
            actual_ret = (close[t + 1] - close[t]) / close[t]
            if actual_ret > 0:
                actual_dir = 1
            elif actual_ret < 0:
                actual_dir = -1
            else:
                actual_dir = 0
            if conf >= 0.5 and pred_dir != 0:
                trade_count += 1
                if pred_dir == actual_dir:
                    daily_returns[t] = actual_ret
                else:
                    daily_returns[t] = -actual_ret
            else:
                daily_returns[t] = 0.0
            predictions[t] = pred_dir
            actuals[t] = actual_dir
            sigma_preds[t] = sigma_pred
            confidences[t] = conf
        total_return = np.sum(daily_returns)
        n_trades = trade_count
        hit = (daily_returns > 0).sum()
        hit_rate = hit / max(n_trades, 1)
        non_zero = daily_returns[daily_returns != 0]
        if len(non_zero) > 1:
            sharpe = float(np.sqrt(252) * np.mean(non_zero) / max(np.std(non_zero), 1e-10))
        else:
            sharpe = 0.0
        cum = np.cumsum(daily_returns)
        peak = np.maximum.accumulate(cum)
        drawdown = cum - peak
        max_dd = float(np.min(drawdown))
        if self.verbose:
            print(f'    回测完成: 命中率 {hit_rate:.1%} ({n_trades} 笔)')
        return BacktestResult(params = params, hit_rate = hit_rate, total_trades = n_trades,
                              cumulative_return = total_return, sharpe = sharpe, max_drawdown = max_dd,
                              daily_returns = daily_returns, predictions = predictions, actuals = actuals,
                              sigma_preds = sigma_preds, confidences = confidences)

    def _predict_one(self, sigma_t: float, grad_t: float, sigma_eq: float, cfg) -> dict:
        '''单帧预测 (与 StockSigma.predict_next_frame 逻辑一致)'''
        sigma_t1 = sigma_t + cfg.alpha * grad_t - cfg.beta * (sigma_t - sigma_eq)
        delta = sigma_t1 - sigma_t
        delta = np.clip(delta, -(cfg.delta_max), cfg.delta_max)
        sigma_t1 = sigma_t + delta
        near_peak = sigma_t < cfg.threshold_low
        near_trough = sigma_t > cfg.threshold_high
        grad_near_zero = abs(grad_t) < 0.02
        if (near_peak or near_trough) and grad_near_zero:
            confidence = 0.75
        elif abs(grad_t) > 0.05:
            confidence = 0.6
        else:
            confidence = 0.4
        if sigma_t1 < sigma_t - 0.01:
            direction = 1
        elif sigma_t1 > sigma_t + 0.01:
            direction = -1
        else:
            direction = 0
        return {
            'sigma_pred': float(np.clip(sigma_t1, 0, 1)),
            'direction': direction,
            'confidence': confidence }


class ParameterOptimizer:
    """
    对 StockSigma 参数做网格搜索优化。

    基本用法::

        opt = ParameterOptimizer()
        results = opt.optimize(df)          # 全量搜索
        opt.report(results)                 # 打印报告

    快速模式 (减少搜索组合)::

        results = opt.optimize(df, mode='fast')
    """
    DEFAULT_GRID = {
        'alpha': [
            0.25,
            0.35,
            0.45,
            0.55,
            0.65],
        'beta': [
            0.1,
            0.15,
            0.2,
            0.25,
            0.3],
        'w_price': [
            0.3,
            0.4,
            0.5],
        'w_volume': [
            0.25,
            0.35],
        'w_volatility': [
            0.15,
            0.25,
            0.35],
        'threshold_low': [
            0.2,
            0.25,
            0.3],
        'threshold_high': [
            0.7,
            0.75,
            0.8],
        'delta_max': [
            0.1,
            0.15,
            0.2],
        'window': [
            15,
            20,
            25] }
    FAST_GRID = {
        'alpha': [
            0.3,
            0.45,
            0.6],
        'beta': [
            0.1,
            0.2,
            0.3],
        'w_price': [
            0.3,
            0.4,
            0.5],
        'w_volume': [
            0.25,
            0.35],
        'w_volatility': [
            0.25],
        'threshold_low': [
            0.25],
        'threshold_high': [
            0.75],
        'delta_max': [
            0.15],
        'window': [
            20] }

    def __init__(self, min_train: int = 60, verbose: bool = True):
        self.engine = BacktestEngine(min_train = min_train, verbose = verbose)
        self.verbose = verbose

    def optimize(self, df: pd.DataFrame, mode: str = 'full', custom_grid: Optional[dict] = None,
                 metric: str = 'hit_rate', top_n: int = 5) -> dict:
        """
        执行参数网格搜索。

        参数
        ----------
        df : pd.DataFrame
            日线数据 (close, volume, high, low)
        mode : str
            'full' = 全量网格 (2160+ 组合)
            'fast' = 快速模式 (243 组合)
        custom_grid : dict or None
            自定义搜索空间
        metric : str
            优化目标: 'hit_rate' | 'sharpe' | 'total_return'
        top_n : int
            返回前 N 个最优组合

        返回
        -------
        dict : { 'results': [BacktestResult], 'best_params': dict, ... }
        """
        grid = custom_grid or (self.FAST_GRID if mode == 'fast' else self.DEFAULT_GRID)
        keys = list(grid.keys())
        values = list(grid.values())
        combinations = list(product(*values))
        total = len(combinations)
        if self.verbose:
            print(f'  σ 参数调优: {mode} 模式')
            print(f'  搜索空间: {len(keys)} 个参数 × {total} 种组合')
            print(f'  数据: {len(df)} 帧, 回测起点: 第 {self.engine.min_train} 帧')
            print(f'  优化目标: {metric}')
            print(f'  {"──────────────────────────────────────────────────"}')
        results = []
        start_time = time.time()
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            w_sum = params.get('w_price', 0.4) + params.get('w_volume', 0.35) + params.get('w_volatility', 0.25)
            if abs(w_sum - 1.0) > 0.05:
                continue
            try:
                bt = self.engine.run(df, params)
                score = getattr(bt, metric, bt.hit_rate)
                results.append((score, bt))
            except Exception as e:
                if self.verbose:
                    warnings.warn(f'  组合 {i + 1}/{total} 失败: {e}')
            if self.verbose and (i + 1) % max(1, total // 10) == 0:
                elapsed = time.time() - start_time
                pct = (i + 1) / total * 100
                print(f'  进度: {pct:.0f}% ({i + 1}/{total}), 耗时 {elapsed:.0f}s')
        results.sort(key = lambda x: x[0], reverse = True)
        elapsed = time.time() - start_time
        if results:
            best_score, best_bt = results[0]
        else:
            best_score, best_bt = 0, None
        if self.verbose:
            print(f'  {"──────────────────────────────────────────────────"}')
            print(f'  完成: {len(results)} 组合有效')
            print(f'  耗时: {elapsed:.0f}s')
            print(f'  最优 {metric}: {best_score:.3f}')
        return {
            'results': [bt for _, bt in results[:top_n]],
            'all_results': results,
            'best_params': best_bt.params if best_bt else {},
            'best_score': best_score,
            'metric': metric,
            'grid_size': (len(keys), total, len(results)) }

    def report(self, opt_result: dict):
        '''打印优化报告'''
        best = opt_result['best_params']
        best_score = opt_result['best_score']
        metric = opt_result['metric']
        (grid_keys, grid_total, grid_valid) = opt_result['grid_size']
        print('\n========================================================')
        print('  SPUM σ 参数调优报告')
        print('========================================================')
        print(f'''  搜索: {grid_keys} 参数 × {grid_total} 组合 → {grid_valid} 有效''')
        print(f'''  目标: {metric}''')
        print(f'''  最优 {metric}: {best_score:.4f}''')
        print(f'''  {'──────────────────────────────────────────────────'}''')
        print('  最佳参数:')
        for k, v in best.items():
            print(f'''    {k:20s} = {v}''')
        print(f'''  {'──────────────────────────────────────────────────'}''')
        top_results = opt_result.get('results', [])
        if len(top_results) > 1:
            print(f'''  Top {len(top_results)} 对比:''')
            print(f'''  {'名次':>4s} {'命中率':>8s} {'收益':>10s} {'夏普':>8s} {'回撤':>8s} {'交易':>6s}''')
            for i, bt in enumerate(top_results):
                print(f'''  {i + 1:>4d} {bt.hit_rate:>7.1%} {bt.cumulative_return:>+9.2%} {bt.sharpe:>7.2f} {bt.max_drawdown:>7.2%} {bt.total_trades:>5d}''')
        print('========================================================')

    def plot(self, opt_result: dict, df: pd.DataFrame, top_n: int = 3):
        '''可视化 Top N 参数组合的收益曲线'''

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn('请安装 matplotlib')
            return None

        top = opt_result.get('results', [])[:top_n]
        if not top:
            print('无结果可绘图')
            return None
        (fig, axes) = plt.subplots(2, 1, figsize = (14, 8))
        ax1 = axes[0]
        start_idx = self.engine.min_train
        dates = df.index[start_idx:]
        for i, bt in enumerate(top):
            cum = np.cumsum(bt.daily_returns[start_idx:])
            label = f'P{i + 1}: α={bt.params.get("alpha", 0):.2f} β={bt.params.get("beta", 0):.2f} 命中={bt.hit_rate:.1%}'
            ax1.plot(dates, cum, lw = 1.5, label = label)
        ax1.axhline(0, color = 'gray', lw = 0.5)
        ax1.set_ylabel('累计收益')
        ax1.set_title(f'σ 振荡模型 — Top {top_n} 收益曲线对比')
        ax1.legend(loc = 'upper left')
        ax1.grid(alpha = 0.3)
        ax2 = axes[1]
        best_bt = top[0]
        from stock_sigma import StockSigma, SigmaConfig
        analyzer = StockSigma(SigmaConfig(**best_bt.params))
        result = analyzer.analyze(df)
        ax2_twin = ax2.twinx()
        ax2.plot(dates, df['close'].values[start_idx:], color = '#1a73e8', lw = 1, alpha = 0.7, label = '价格')
        ax2_twin.plot(dates, result['sigma'].values[start_idx:], color = '#e8710a', lw = 2, alpha = 0.85, label = 'σ (最佳参数)')
        preds = best_bt.predictions[start_idx:]
        up_idx = np.where(preds == 1)[0]
        dn_idx = np.where(preds == -1)[0]
        close_vals = df['close'].values[start_idx:]
        ax2.scatter(dates[up_idx], close_vals[up_idx], color = 'green', s = 8, alpha = 0.3, label = '预测↑')
        ax2.scatter(dates[dn_idx], close_vals[dn_idx], color = 'red', s = 8, alpha = 0.3, label = '预测↓')
        ax2.set_ylabel('价格')
        ax2_twin.set_ylabel('σ')
        ax2.set_xlabel('日期')
        ax2.legend(loc = 'upper left')
        ax2.grid(alpha = 0.3)
        plt.tight_layout()
        plt.show()


def quick_optimize(df: pd.DataFrame, mode: str = 'fast', metric: str = 'hit_rate') -> dict:
    """
    一键优化: 加载数据 → 网格搜索 → 最优参数。

    参数
    ----------
    df : pd.DataFrame
        日线数据 (close, volume, high, low)
    mode : str
        'fast' | 'full'
    metric : str
        'hit_rate' | 'sharpe' | 'total_return'

    返回
    -------
    dict : 含 'best_params', 'results', 'best_score'
    """
    opt = ParameterOptimizer(verbose = True)
    result = opt.optimize(df, mode = mode, metric = metric)
    opt.report(result)
    return result
