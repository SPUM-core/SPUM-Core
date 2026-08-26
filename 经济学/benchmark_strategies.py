# Source Generated with Decompyle++
# File: benchmark_strategies.cpython-311.pyc (Python 3.11)

'''
benchmark_strategies.py — 传统技术指标策略基线
===============================================

提供一组可复用的传统技术指标策略，作为 S₀ 五形命格策略的对比基线。

每个策略输出信号序列 [-1, 1]，可与 BacktestEngine 直接对接。

用法:
  from benchmark_strategies import compute_all_benchmarks
  results = compute_all_benchmarks(dates, close_prices)
'''
import numpy as np
import math

def np_mean(arr):
    return sum(arr) / len(arr) if arr else 0.0


def np_std(arr, ddof = 1):
    if len(arr) < 2:
        return 0.0
    m = np_mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / (len(arr) - ddof))


def sma(values: list[float], window: int) -> list[float]:
    '''简单移动平均'''
    out = []
    for i in range(len(values)):
        if i < window - 1:
            out.append(float('nan'))
            continue
        out.append(float(np_mean(values[(i - window) + 1:i + 1])))
    return out


def ema(values: list[float], window: int) -> list[float]:
    '''指数移动平均'''
    out = []
    multiplier = 2 / (window + 1)
    for i in range(len(values)):
        if i == 0:
            out.append(values[i])
            continue
        out.append((values[i] - out[-1]) * multiplier + out[-1])
    return out


def rsi(values: list[float], window: int = 14) -> list[float]:
    '''相对强弱指标 RSI'''
    if len(values) < window + 1:
        return [float('nan')] * len(values)
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    rsi_vals = [float('nan')]
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(len(deltas)):
        gain = max(deltas[i], 0)
        loss = abs(min(deltas[i], 0))
        if i < window:
            avg_gain += gain
            avg_loss += loss
            if i == window - 1:
                avg_gain /= window
                avg_loss /= window
                rs = avg_gain / (avg_loss + 1e-10)
                rsi_vals.append(100 - 100 / (1 + rs))
            else:
                rsi_vals.append(float('nan'))
        else:
            avg_gain = (avg_gain * (window - 1) + gain) / window
            avg_loss = (avg_loss * (window - 1) + loss) / window
            rs = avg_gain / (avg_loss + 1e-10)
            rsi_vals.append(100 - 100 / (1 + rs))
    return rsi_vals


def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9):
    '''MACD 指标，返回 (macd_line, signal_line, histogram)'''
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(values))]
    signal_line = ema(macd_line, signal)
    histogram = [macd_line[i] - signal_line[i] for i in range(len(values))]
    return (macd_line, signal_line, histogram)


def bollinger_bands(values: list[float], window: int = 20, num_std: float = 2.0):
    '''布林带，返回 (upper, middle, lower)'''
    middle = sma(values, window)
    upper = []
    lower = []
    for i in range(len(values)):
        if math.isnan(middle[i]):
            upper.append(float('nan'))
            lower.append(float('nan'))
            continue
        band = num_std * np_std(values[(i - window) + 1:i + 1]) if i >= window - 1 else 0
        upper.append(middle[i] + band)
        lower.append(middle[i] - band)
    return (upper, middle, lower)


def ma_crossover_signal(prices: list[float], fast: int = 5, slow: int = 20) -> list[float]:
    '''
    MA 金叉死叉信号。
    快 MA > 慢 MA → +1（做多）；快 MA < 慢 MA → -1（做空）。
    信号强度 = 差值 / 价格（归一化到 [-1, 1]）
    '''
    sma_fast = sma(prices, fast)
    sma_slow = sma(prices, slow)
    signals = []
    for i in range(len(prices)):
        if math.isnan(sma_fast[i]) or math.isnan(sma_slow[i]) or prices[i] == 0:
            signals.append(0.0)
            continue
        diff = (sma_fast[i] - sma_slow[i]) / prices[i]
        signals.append(max(-1.0, min(1.0, diff * 10)))
    return signals


def rsi_signal(prices: list[float], window: int = 14, oversold: float = 30, overbought: float = 70) -> list[float]:
    '''
    RSI 超买超卖信号。
    RSI < oversold → +1（超卖，做多）
    RSI > overbought → -1（超买，做空）
    中间区域线性插值
    '''
    rsi_vals = rsi(prices, window)
    signals = []
    for r in rsi_vals:
        if math.isnan(r):
            signals.append(0.0)
            continue
        if r < oversold:
            signals.append(1.0)
            continue
        if r > overbought:
            signals.append(-1.0)
            continue
        mid = (oversold + overbought) / 2
        s = ((mid - r) / (overbought - oversold)) * 2
        signals.append(float(max(-1.0, min(1.0, s))))
    return signals


def macd_signal(prices: list[float], fast: int = 12, slow: int = 26, signal_period: int = 9) -> list[float]:
    '''
    MACD 信号。
    柱状图 > 0 → +1（多头）；柱状图 < 0 → -1（空头）。
    信号强度 = 柱状图 / 价格
    '''
    (_, _, histogram) = macd(prices, fast, slow, signal_period)
    signals = []
    for i in range(len(prices)):
        if math.isnan(histogram[i]) or prices[i] == 0:
            signals.append(0.0)
            continue
        s = (histogram[i] / prices[i]) * 50
        signals.append(max(-1.0, min(1.0, s)))
    return signals


def bollinger_signal(prices: list[float], window: int = 20, num_std: float = 2.0) -> list[float]:
    '''
    布林带均值回归信号。
    价格 > 上轨 → -1（做空，回归）
    价格 < 下轨 → +1（做多，回归）
    中间区线性衰减
    '''
    (upper, middle, lower) = bollinger_bands(prices, window, num_std)
    signals = []
    for i in range(len(prices)):
        if math.isnan(upper[i]) or middle[i] == 0:
            signals.append(0.0)
            continue
        if prices[i] >= upper[i]:
            signals.append(-1.0)
            continue
        if prices[i] <= lower[i]:
            signals.append(1.0)
            continue
        width = upper[i] - lower[i]
        offset = ((prices[i] - middle[i]) / (width + 1e-10)) * 2
        signals.append(float(max(-1.0, min(1.0, -offset))))
    return signals


def compute_all_benchmarks(dates: list[str], prices: list[float]) -> dict:
    '''
    计算所有技术指标基线策略的信号。
    
    参数:
        dates:  交易日列表
        prices: 收盘价列表
    
    返回:
        dict: {策略名称: 信号列表}
    '''
    return {
        'MA(5,20) 金叉死叉': ma_crossover_signal(prices, 5, 20),
        'MA(10,30) 金叉死叉': ma_crossover_signal(prices, 10, 30),
        'RSI(14) 超买超卖': rsi_signal(prices, 14, 30, 70),
        'RSI(7) 超买超卖': rsi_signal(prices, 7, 25, 75),
        'MACD(12,26,9)': macd_signal(prices, 12, 26, 9),
        '布林带(20,2σ)': bollinger_signal(prices, 20, 2),
        '布林带(20,2.5σ)': bollinger_signal(prices, 20, 2.5) }


def strategy_descriptions() -> dict:
    '''返回各策略的中文描述'''
    return {
        'MA(5,20) 金叉死叉': '快线(5日)上穿慢线(20日)做多，下穿做空。趋势跟踪。',
        'MA(10,30) 金叉死叉': '快线(10日)上穿慢线(30日)做多，下穿做空。趋势跟踪。',
        'RSI(14) 超买超卖': 'RSI<30超卖做多，RSI>70超买做空。均值回归。',
        'RSI(7) 超买超卖': 'RSI(7)<25超卖做多，>75超买做空。短周期均值回归。',
        'MACD(12,26,9)': 'MACD柱状图>0做多，<0做空。趋势跟踪。',
        '布林带(20,2σ)': '价格突破上轨做空，跌破下轨做多。均值回归。',
        '布林带(20,2.5σ)': '价格突破上轨(2.5σ)做空，跌破下轨做多。更宽松的均值回归。' }

if __name__ == '__main__':
    np.random.seed(42)
    n = 300
    prices = [
        100]
    for i in range(1, n):
        prices.append(prices[-1] * (1 + np.random.randn() * 0.015))
    signals = compute_all_benchmarks([f'day{i}' for i in range(n)], prices)
    for name, sig in signals.items():
        nz = sum(1 for s in sig if abs(s) > 0.01)
        print(f'{name:<24} 非零信号: {nz}/{len(sig)}')
    print('\n✅ benchmark_strategies.py 自检通过')
