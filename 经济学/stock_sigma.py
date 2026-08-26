# Source Generated with Decompyle++
# File: stock_sigma.cpython-311.pyc (Python 3.11)

'''
SPUM × 股票市场 σ 振荡分析器
================================

理论基底: SPUM 经济学模块 (ECON-017~019)
  - 价格曲线 = σ 分布的阻尼振荡在货币边上的投影
  - 正常波动 = σ 自反馈系统的确定性振荡 (受 dv/dt ≤ const 约束)
  - 拐点 = ∇σ 方向逆转 + σ 触及天花板/地板

映射关系:
  SPUM 概念        股票市场对应        数值代理
  ─────────        ──────────          ─────────
  σ                市场密度            价格/成交量/波动率的复合归一化
  ∇σ               多空梯度            σ(t) - σ(t-1)
  dv/dt ≤ const    单帧变化上限        日最大 σ 变化 (默认 0.15)
  扩张期           价格上涨             σ 递减 (密度上升)
  顶峰              价格顶部             σ 接近最小值 (饱和)
  收缩期           价格下跌             σ 递增 (密度下降)
  谷底              价格底部             σ 接近最大值 (稀疏)

依赖: numpy, pandas, matplotlib (可选: akshare)
'''
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import warnings


@dataclass
class SigmaConfig:
    '''σ 分析器配置 (修复自 dataclass 反编译损坏)'''
    window: int = 25
    delta_max: float = 0.15
    w_price: float = 0.3
    w_volume: float = 0.35
    w_volatility: float = 0.35
    phase_smooth: int = 3
    threshold_low: float = 0.2
    threshold_high: float = 0.75
    confirm_frames: int = 2
    alpha: float = 0.25
    beta: float = 0.3
    sigma_eq_decay: float = 0.005
    cascade_threshold: float = 0.35
    price_limit: float = 0.1


class StockSigma:
    """
    SPUM σ 振荡分析器

    将股票价格曲线映射为 ⟨P, ε⟩ 经济子图中的 σ 密度振荡,
    通过 σ 轨迹识别周期相位, 基于拓扑约束预测拐点。

    用法::

        analyzer = StockSigma()
        df = analyzer.fetch_a_stock('000001')  # 平安银行
        result = analyzer.analyze(df)
        analyzer.plot(result)
    """

    def __init__(self, config: Optional[SigmaConfig] = None):
        if config:
            self.cfg = config
        else:
            self.cfg = SigmaConfig()
        self._label_map = {
            -1: '收缩',
            0: '震荡',
            1: '扩张' }

    def analyze(self, df: pd.DataFrame) -> dict:
        """
        对输入的股票数据执行完整 σ 分析。

        参数
        ----------
        df : pd.DataFrame
            必须包含列: 'close', 'volume' (和可选的 'high', 'low')
            推荐索引为日期

        返回
        -------
        dict : {
            'sigma':         pd.Series  — 复合 σ 时间序列
            'sigma_price':   pd.Series  — 价格子分量 σ
            'sigma_volume':  pd.Series  — 成交量子分量 σ
            'sigma_volatility': pd.Series — 波动率子分量 σ
            'gradient':      pd.Series  — ∇σ 梯度
            'phase':         pd.Series  — 周期相位 (1=扩张,0=震荡,-1=收缩)
            'phase_label':   pd.Series  — 相位中文标签
            'sigma_eq':      float      — 当前均衡 σ
            'divergence':    pd.Series  — 价格-σ 背离信号
            'cascade_warn':  pd.Series  — V⁻ 级联预警
            'prediction':    dict       — 下一帧预测结果
        }
        """
        df = df.copy()
        s_price = self._calc_sigma_price(df)
        s_volume = self._calc_sigma_volume(df)
        s_volatility = self._calc_sigma_volatility(df)
        sigma = (self.cfg.w_price * s_price
                 + self.cfg.w_volume * s_volume
                 + self.cfg.w_volatility * s_volatility)
        gradient = self._calc_gradient(sigma)
        sigma_eq = self._calc_equilibrium(sigma)
        phase = self._detect_phase(sigma, gradient)
        phase_label = pd.Series([self._label_map.get(p, '未知') for p in phase],
                                index=phase.index, name='phase_label')
        divergence = self._detect_divergence(df['close'].values, sigma.values)
        cascade_warn = self._detect_cascade(gradient)
        prediction = self.predict_next_frame(sigma.values, gradient.values, sigma_eq)
        return {
            'sigma': sigma,
            'sigma_price': s_price,
            'sigma_volume': s_volume,
            'sigma_volatility': s_volatility,
            'gradient': gradient,
            'phase': phase,
            'phase_label': phase_label,
            'sigma_eq': sigma_eq,
            'divergence': divergence,
            'cascade_warn': cascade_warn,
            'prediction': prediction }

    def _calc_sigma_price(self, df: pd.DataFrame) -> pd.Series:
        '''
        价格 σ: 收盘价在近期区间中的相对位置。

        SPUM 映射: 价格是高是低不是绝对值 —— 是 σ 分布的瞬时快照。
        σ_price 高 = 价格在区间低位 (稀疏, 空间大)
        σ_price 低 = 价格在区间高位 (饱和, 天花板近)

        公式: σ_price = (high_max - close) / (high_max - low_min)
        '''
        high = df.get('high', df['close']).rolling(self.cfg.window).max()
        low = df.get('low', df['close']).rolling(self.cfg.window).min()
        close = df['close']
        sigma = (high - close) / ((high - low) + 1e-10)
        return sigma.clip(0, 1).fillna(0.5)

    def _calc_sigma_volume(self, df: pd.DataFrame) -> pd.Series:
        '''
        成交量 σ: 成交活跃度的归一化。

        SPUM 映射: 高成交量 = 低 σ (密集交易, 连接饱和)
                   低成交量 = 高 σ (稀疏交易, 连接不足)

        公式: σ_volume = 1 - (volume / volume_ma) 归一化到 [0, 1]
        '''
        vol = df['volume']
        vol_ma = vol.rolling(self.cfg.window).mean().replace(0, np.nan)
        ratio = vol / vol_ma
        sigma = 2 / (1 + np.exp(ratio / 3))
        return pd.Series(sigma.clip(0, 1), name='sigma_volume').fillna(0.5)

    def _calc_sigma_volatility(self, df: pd.DataFrame) -> pd.Series:
        '''
        波动率 σ: 波动率归一化。

        SPUM 映射: 高波动 = 高 σ (不稳定, 连接频繁断裂/重连 = "热")
                   低波动 = 低 σ (稳定, 连接牢固 = "冷")

        公式: σ_volatility = sigmoid(ATR / ATR_ma)
        '''
        high = df.get('high', df['close'])
        low = df.get('low', df['close'])
        close = df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(self.cfg.window).mean()
        atr_ma = atr.rolling(self.cfg.window * 5).mean().replace(0, np.nan)
        ratio = (atr / atr_ma).fillna(1)
        sigma = 2 / (1 + np.exp(-(ratio - 1) / 0.5))
        return sigma.clip(0, 1).fillna(0.5)

    def _calc_gradient(self, sigma: pd.Series) -> pd.Series:
        '''
        计算 ∇σ: σ 的一阶差分 (中心差分).

        ∇σ > 0 → σ 上升 → 密度降低 → 价格趋弱
        ∇σ < 0 → σ 下降 → 密度升高 → 价格趋强

        受 dv/dt ≤ const 约束: |∇σ| ≤ delta_max
        '''
        grad = sigma.diff().clip(-(self.cfg.delta_max), self.cfg.delta_max)
        return grad.fillna(0)

    def _calc_equilibrium(self, sigma: pd.Series) -> float:
        '''
        计算自适应均衡 σ_eq。

        σ_eq 是长期 σ 的指数加权平均 —— 代表该股票的"自然密度"。
        由于不完美定理, σ 永远不会等于 σ_eq, 只会围绕它振荡。

        公式: σ_eq = EWMA(sigma, span=long_window)
        '''
        span = min(self.cfg.window * 10, len(sigma) // 2)
        if span < 2:
            return 0.5
        ewma = sigma.ewm(span=span).mean()
        return float(ewma.iloc[-1])

    def _detect_phase(self, sigma: pd.Series, gradient: pd.Series) -> pd.Series:
        '''
        识别周期相位:

          1 (扩张):  σ 递减中 (∇σ < 0, 且 σ > threshold_low)
          0 (震荡):  无明显趋势 (∇σ ≈ 0, 或 σ 在极值区徘徊)
         -1 (收缩):  σ 递增中 (∇σ > 0, 且 σ < threshold_high)

        相位切换需要连续 confirm_frames 帧确认 ——
        对应 SPUM 一帧内无级联消解, 新相位需多帧确认。
        '''
        cfg = self.cfg
        raw = pd.Series(0, index=sigma.index)
        expansion = (gradient < 0) & (sigma > cfg.threshold_low)
        raw[expansion] = 1
        contraction = (gradient > 0) & (sigma < cfg.threshold_high)
        raw[contraction] = -1
        smooth = raw.rolling(cfg.confirm_frames, min_periods=1).apply(
            (lambda x: 1 if (x > 0).all() else (-1 if (x < 0).all() else 0)), raw=True)
        return smooth.astype(int)

    def _detect_divergence(self, price: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        '''
        检测价格-σ 背离 —— SPUM 中最重要的反转信号。

        背离的拓扑本质:
          价格和 σ 应该反向运动 (价格↑ → σ↓, 价格↓ → σ↑)
          当它们同向运动时, 说明 σ 信号失真 —— 趋势不可持续

        顶背离 (bearish divergence):
          价格创新高, 但 σ 未创新低 (该饱和却没饱和) → 虚假繁荣

        底背离 (bullish divergence):
          价格创新低, 但 σ 未创新高 (该稀疏却没稀疏) → 虚假恐慌

        返回: [-1, 1] 的序列
          > 0 = 底背离强度 (看涨信号)
          < 0 = 顶背离强度 (看跌信号)
          0  = 无背离
        '''
        n = len(price)
        divergence = np.zeros(n)
        window = self.cfg.window
        for i in range(window, n):
            p_high = price[i] >= np.max(price[i - window:i + 1])
            p_low = price[i] <= np.min(price[i - window:i + 1])
            s_low = sigma[i] <= np.min(sigma[i - window:i + 1])
            s_high = sigma[i] >= np.max(sigma[i - window:i + 1])
            if p_high and sigma[i] > np.min(sigma[i - window:i]):
                s_min = np.min(sigma[i - window:i])
                divergence[i] = -(sigma[i] - s_min) / max(sigma[i] - s_min, 1e-06)
                divergence[i] = np.clip(divergence[i], -1, 0)
            if p_low and sigma[i] < np.max(sigma[i - window:i]):
                s_max = np.max(sigma[i - window:i])
                divergence[i] = (s_max - sigma[i]) / max(s_max - sigma[i], 1e-06)
                divergence[i] = np.clip(divergence[i], 0, 1)
        return divergence

    def _detect_cascade(self, gradient: pd.Series) -> pd.Series:
        '''
        V⁻ 级联预警: 当 ∇σ 正方向超过 cascade_threshold 时预警。

        σ 单帧飙升 = 密度骤降 = 大量连接断裂 = 恐慌抛售
        对应 ECON-020: 经济危机 = V⁻ 级联
        '''
        threshold = self.cfg.cascade_threshold
        cascade = (gradient > threshold).astype(int)
        cascade.name = 'cascade_warn'
        return cascade

    def predict_next_frame(self, sigma: np.ndarray, grad: np.ndarray, sigma_eq: float) -> dict:
        '''
        基于 σ 阻尼振荡模型预测下一帧。

        模型 (SPUM 形式):
          σ(t+1) = σ(t) + α·∇σ(t) - β·(σ(t) - σ_eq)

          第一项: 当前密度
          第二项: 动量 —— ∇σ 的惯性延续 (V⁺ 事件残量)
          第三项: 回复力 —— 向均衡的拓扑拉力 (守恒律)

          约束: |σ(t+1) - σ(t)| ≤ delta_max (dv/dt ≤ const)

        返回: dict
          sigma_pred  : 预测的 σ 值
          direction   : 预测方向 (1=涨, -1=跌, 0=震荡)
          confidence  : 信心度 [0, 1]
          sigma_range : [σ_min, σ_max] 置信区间
        '''
        cfg = self.cfg
        sigma_t = sigma[-1]
        grad_t = grad[-1]
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
        grad_std = float(np.std(grad[-cfg.window:])) if len(grad) > cfg.window else 0.03
        sigma_range = (max(0, sigma_t1 - 1.5 * grad_std), min(1, sigma_t1 + 1.5 * grad_std))
        return {
            'sigma_pred': float(np.clip(sigma_t1, 0, 1)),
            'direction': direction,
            'direction_label': {
                1: '看涨',
                -1: '看跌',
                0: '震荡' }.get(direction, '未知'),
            'confidence': float(confidence),
            'sigma_range': (float(sigma_range[0]), float(sigma_range[1])) }

    def fetch_a_stock(self, code: str, start: str = '20250101', end: Optional[str] = None,
                      max_retries: int = 3, source: str = 'auto') -> pd.DataFrame:
        """
        获取 A 股日线数据 (多源支持 + 自动重试)。

        参数
        ----------
        code : str
            股票代码, 如 '000001' (平安银行), '600519' (茅台)
        start : str
            起始日期, 格式 'YYYYMMDD'
        end : str or None
            结束日期, 默认今天
        max_retries : int
            网络请求最大重试次数 (默认 3)
        source : str
            数据源: 'auto' | 'em' (东方财富) | 'tx' (腾讯)

        返回
        -------
        pd.DataFrame : 包含 close, volume, high, low
        """
        import time
        try:
            import akshare as ak
        except ImportError:
            raise ImportError('请安装 akshare: pip install akshare -U\n或使用 load_local() 加载本地 CSV')
        if end:
            end_date = end
        else:
            end_date = pd.Timestamp.today().strftime('%Y%m%d')
        apis = []
        if source == 'auto':
            apis = [
                ('em', lambda: self._fetch_em(ak, code, start, end_date)),
                ('tx', lambda: self._fetch_tx(ak, code, start, end_date))]
        elif source == 'em':
            apis = [
                ('em', lambda: self._fetch_em(ak, code, start, end_date))]
        elif source == 'tx':
            apis = [
                ('tx', lambda: self._fetch_tx(ak, code, start, end_date))]
        else:
            raise ValueError(f'不支持的数据源: {source}')
        last_error = None
        for name, fetcher in apis:
            for attempt in range(max_retries):
                try:
                    df = fetcher()
                    if len(df) > 0:
                        return self._standardize_columns(df)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
        raise ConnectionError(
            f'无法获取 {code} 数据 (已尝试 {len(apis)} 个数据源, 每个 {max_retries} 次)\n'
            f'  最后错误: {last_error}\n'
            f'  提示: 可使用 load_local() 加载本地 CSV 数据')

    def _fetch_em(self, ak, code: str, start: str, end: str) -> pd.DataFrame:
        '''东方财富数据源'''
        return ak.stock_zh_a_hist(symbol=code, period='daily', start_date=start, end_date=end, adjust='qfq')

    def _fetch_tx(self, ak, code: str, start: str, end: str) -> pd.DataFrame:
        '''腾讯数据源'''
        df = ak.stock_zh_a_hist_tx(symbol=code, start_date=start, end_date=end, adjust='qfq')
        col_map = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume' }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if 'date' in df.columns:
            df.index = pd.to_datetime(df.pop('date'))
        df.index.name = 'date'
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''统一列名为英文'''
        col_map = {
            '日期': None,
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': None,
            '涨跌幅': None,
            '涨跌额': None,
            '换手率': None }
        rename = {k: v for k, v in col_map.items() if v is not None}
        drop_cols = [k for k, v in col_map.items() if v is None and k in df.columns]
        df = df.rename(columns=rename)
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        if '日期' in df.columns:
            df.index = pd.to_datetime(df.pop('日期'))
        df.index.name = 'date'
        for col in ('close', 'volume', 'high', 'low'):
            if col not in df.columns:
                if col == 'volume':
                    df['volume'] = 0
                else:
                    raise ValueError(f'数据缺少必要列: {col}')
        return df.sort_index()

    def load_local(self, filepath: str, date_col: str = 'date', date_fmt: Optional[str] = None,
                   **kwargs) -> pd.DataFrame:
        '''
        从本地 CSV 加载数据。

        参数
        ----------
        filepath : str
            CSV 文件路径
        date_col : str
            日期列名
        date_fmt : str or None
            日期格式, 默认自动推断
        **kwargs : dict
            传递给 pd.read_csv 的参数
        '''
        df = pd.read_csv(filepath, **kwargs)
        if date_col in df.columns:
            if date_fmt:
                df.index = pd.to_datetime(df[date_col], format=date_fmt)
            else:
                df.index = pd.to_datetime(df[date_col])
            df = df.drop(columns=[date_col])
        df.index.name = 'date'
        required = set()
        required.update({'high', 'volume', 'low', 'close'})
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'数据缺少必要列: {missing}')
        if 'volume' not in df.columns or df['volume'].isna().all():
            df['volume'] = 0
        return df.sort_index()

    def plot(self, result: dict, df: pd.DataFrame, title: str = 'SPUM σ 振荡分析'):
        '''
        四面板可视化:
          1. 价格 + σ 叠加
          2. ∇σ 梯度
          3. 周期相位
          4. 背离信号 + 级联预警 + 预测
        '''
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            warnings.warn('请安装 matplotlib 以使用可视化功能')
            return None

        sigma = result['sigma']
        grad = result['gradient']
        phase = result['phase']
        div = result['divergence']
        cascade = result['cascade_warn']
        pred = result['prediction']
        sigma_eq = result['sigma_eq']
        (fig, axes) = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        dates = df.index if hasattr(df.index, 'date') else df.index
        ax1 = axes[0]
        color_price = '#1a73e8'
        color_sigma = '#e8710a'
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(dates, df['close'].values, color=color_price, linewidth=1.5, label='价格')
        line2 = ax1_twin.plot(dates, sigma.values, color=color_sigma, linewidth=2, alpha=0.85, label='σ (密度)')
        ax1_twin.axhline(sigma_eq, color=color_sigma, linestyle='--', alpha=0.4, linewidth=1,
                         label=f'σ_eq={sigma_eq:.3f}')
        ax1_twin.axhline(self.cfg.threshold_low, color='green', linestyle=':', alpha=0.3, label='低位阈')
        ax1_twin.axhline(self.cfg.threshold_high, color='red', linestyle=':', alpha=0.3, label='高位阈')
        pred_date = dates[-1] + pd.Timedelta(days=1) if hasattr(dates, 'freq') else dates[-1]
        ax1_twin.scatter([pred_date], [pred['sigma_pred']], color='purple', s=80, zorder=5,
                         label=f'预测 σ={pred["sigma_pred"]:.3f}')
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        ax1.set_ylabel('价格')
        ax1_twin.set_ylabel('σ')
        ax1.set_title(title, fontsize=13, fontweight='bold')
        ax2 = axes[1]
        colors_grad = ['red' if v > 0 else ('green' if v < 0 else 'gray') for v in grad.values]
        ax2.bar(dates, grad.values, color=colors_grad, width=0.8, alpha=0.7)
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.axhline(self.cfg.delta_max, color='red', linestyle='--', alpha=0.5,
                    label=f'dv/dt max={self.cfg.delta_max}')
        ax2.axhline(-(self.cfg.delta_max), color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('∇σ')
        ax2.legend(loc='upper left')
        cascade_idx = np.where(cascade.values == 1)[0]
        if len(cascade_idx) > 0:
            ax2.scatter(dates[cascade_idx], grad.values[cascade_idx], color='red', s=120,
                        marker='v', label='⚠ V⁻ 级联预警', zorder=5)
            ax2.legend(loc='upper left')
        ax3 = axes[2]
        phase_fill = ax3.fill_between(dates, 0, phase.values, where=phase.values > 0,
                                      color='green', alpha=0.4, label='扩张')
        ax3.fill_between(dates, 0, phase.values, where=phase.values < 0,
                         color='red', alpha=0.4, label='收缩')
        ax3.set_ylim(-1.5, 1.5)
        ax3.set_yticks([-1, 0, 1])
        ax3.set_yticklabels(['收缩', '震荡', '扩张'])
        ax3.set_ylabel('相位')
        ax3.legend(loc='upper left')
        ax4 = axes[3]
        colors_div = ['green' if v > 0 else ('red' if v < 0 else 'gray') for v in div]
        ax4.bar(dates, div, color=colors_div, width=0.8, alpha=0.6, label='价格-σ 背离')
        ax4.axhline(0, color='black', linewidth=0.5)
        direction = pred['direction']
        conf = pred['confidence']
        dir_text = pred['direction_label']
        ax4.text(0.02, 0.95, f'下一帧预测: {dir_text} (信心 {conf:.0%})', transform=ax4.transAxes,
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_ylabel('背离强度')
        ax4.set_xlabel('日期')
        ax4.legend(loc='upper left')
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return fig


def quick_analysis(code: str, start: str = '20250101', end: Optional[str] = None,
                   config: Optional[SigmaConfig] = None) -> dict:
    """
    一键分析: 获取数据 → 计算 σ → 检测相位 → 预测。

    参数
    ----------
    code : str
        A 股代码, 如 '000001', '600519'
    start : str
        起始日期 YYYYMMDD
    end : str or None
        结束日期, 默认今天
    config : SigmaConfig or None
        σ 分析器配置

    返回
    -------
    dict : analyze() 的完整结果
    """
    analyzer = StockSigma(config)
    df = analyzer.fetch_a_stock(code, start, end)
    result = analyzer.analyze(df)
    result['df'] = df
    return result


def print_summary(result: dict):
    '''打印分析结果摘要'''
    pred = result['prediction']
    latest_phase = result['phase_label'].iloc[-1]
    latest_sigma = result['sigma'].iloc[-1]
    print('========================================================')
    print('  SPUM σ 振荡分析报告')
    print('========================================================')
    print(f'  当前相位:    {latest_phase}')
    print(f'  当前 σ:      {latest_sigma:.4f}')
    print(f'  均衡 σ_eq:   {result["sigma_eq"]:.4f}')
    print(f'  最近 ∇σ:    {result["gradient"].iloc[-1]:+.4f}')
    print(f'  ∇σ 均值:     {result["gradient"].mean():+.4f}')
    print(f'  ∇σ 标准差:   {result["gradient"].std():.4f}')
    print('  ─────────────────────────────')
    print('  下一帧预测:')
    print(f'    方向:      {pred["direction_label"]}')
    print(f'    预测 σ:    {pred["sigma_pred"]:.4f}')
    print(f'    信心:      {pred["confidence"]:.0%}')
    print(f'    σ 区间:    [{pred["sigma_range"][0]:.3f}, {pred["sigma_range"][1]:.3f}]')
    print('  ─────────────────────────────')
    phase_counts = result['phase'].value_counts()
    print('  相位分布:')
    for k, v in {
        1: '扩张',
        0: '震荡',
        -1: '收缩' }.items():
        cnt = phase_counts.get(k, 0)
        print(f'    {v}: {cnt} 帧 ({(cnt / len(result["phase"])) * 100:.0f}%)')
    cascade_count = result['cascade_warn'].sum()
    if cascade_count > 0:
        print(f'  ⚠  V⁻ 级联预警: 发生 {int(cascade_count)} 次')
    else:
        print('  ✅ V⁻ 级联: 未触发')
    div = result['divergence']
    bearish = (div < -0.3).sum()
    bullish = (div > 0.3).sum()
    if bearish > 0:
        print(f'  ⚠  顶背离信号: {int(bearish)} 次')
    if bullish > 0:
        print(f'  ⚠  底背离信号: {int(bullish)} 次')
    print('========================================================')
