'''
SPUM 多频自适应相位分析器
========================

核心修正 v2: 百分位阈值替换固定阈值。

SPUM 原理: 相位占优度是该指标在当前系统(市场)的"异常程度"。
  异常程度 = 当前值在历史分布中的百分位排名。

  火亢 = 动量在历史前 10%
  木僵 = 波动率在历史后 10%
  金亢 = 波动率变化在历史前 10%
  土盈 = 横盘程度在历史前 10%

这使得检测器可以适配任意时间尺度(分钟/小时/日/周), 无需重设阈值。
'''
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass

PHASE_CONFIG_DEFAULT = {
    'saturation_threshold': 0.8,
    'up_fire_min': 0.55,
    'up_earth_min': 0.3,
    'up_earth_max': 0.8,
    'down_fire_min': 0.5,
    'flat_metal_threshold': 0.3,
    'flat_earth_fire_max': 0.4,
}


@dataclass
class PhaseState:
    value: float = 0.0
    label: str = ''

    def is_saturated(self, threshold: float) -> bool:
        return self.value >= threshold

    def is_dominant(self) -> bool:
        return self.value >= 0.6

    def is_absent(self) -> bool:
        return self.value < 0.3


@dataclass
class DailyProfile:
    fire: PhaseState
    water: PhaseState
    wood: PhaseState
    metal: PhaseState
    earth: PhaseState
    config: dict = None

    def dominant_phase(self) -> str:
        phases = {'火': self.fire, '水': self.water, '木': self.wood, '金': self.metal, '土': self.earth}
        return max(phases, key=lambda k: phases[k].value)

    def saturated_phases(self) -> List[str]:
        thr = self.config.get('saturation_threshold', 0.85)
        return [p.label for p in (self.fire, self.water, self.wood, self.metal, self.earth)
                if p.is_saturated(thr)]

    def predict_next_direction(self) -> Tuple[str, float, str]:
        '''
        数据校准后的预测规则 (v3, 依据 5 股 16 年跨股票诊断):
          火亢单独(无量确认)=动量延续→↗  54% 续涨 (v2 误判↘回调)
          火+水=量价极端=过热反转→↘     48.7% 跌, 均值-0.23%
          金亢单独=波动放大=突破偏上→↗  50.5% 涨, 均值+0.26% (v2 误判→横盘)
          土+火=储备被驱动→↗            53.3% 涨 (最强信号)
        '''
        cfg = self.config
        saturated = self.saturated_phases()
        fire, water, wood, metal, earth = (self.fire.value, self.water.value,
                                           self.wood.value, self.metal.value, self.earth.value)
        num_sat = len(saturated)
        if '火' in saturated and '水' in saturated and '金' in saturated:
            return ('↘', 0.55, '三饱和(水+火+金)=量价极端,弱回调')
        if '火' in saturated and '水' in saturated:
            return ('↘', 0.6, '火亢+水盈=量价驱动到极致,过热反转')
        if '火' in saturated and '木' in saturated:
            return ('↗', 0.5, '火亢+木僵=骨架锁定下的动量延续')
        if '火' in saturated and num_sat == 1:
            return ('↗', 0.55, f'火亢单独(前{fire:.0%})=动量延续')
        up_fire_min = cfg.get('up_fire_min', 0.65)
        up_earth_min = cfg.get('up_earth_min', 0.3)
        up_earth_max = cfg.get('up_earth_max', 0.7)
        if up_earth_min <= earth <= up_earth_max and fire >= up_fire_min:
            return ('↗', 0.5, f'火形={fire:.0%}+土形={earth:.0%},储备驱动看涨')
        if '水' in saturated and fire >= cfg.get('down_fire_min', 0.5):
            return ('↗', 0.4, '水盈(资金流入)+火形尚可,温和看涨')
        if '金' in saturated and num_sat == 1:
            return ('↗', 0.5, '金亢单独=波动放大,突破方向偏上')
        if '土' in saturated and num_sat == 1 and fire < cfg.get('flat_earth_fire_max', 0.4):
            return ('→', 0.25, '土盈+火弱,储备无驱动=横盘')
        return ('→', 0.05, '无明确相位信号,不做预测')


def _percentile_rank(values, current: float) -> float:
    '''计算当前值在历史序列中的百分位(0~1)'''
    if len(values) < 10:
        return 0.5
    below = np.sum(values < current)
    equal = np.sum(values == current)
    rank = (below + 0.5 * equal) / len(values)
    return float(np.clip(rank, 0, 1))


class DailyPhaseDetector:
    '''
    自适应百分位相位检测器。

    detect(df, config=None) → DailyProfile

    所有相位值 = 当前指标在滚动历史窗口中的百分位排名(0~1)。
    适配任意K线颗粒度: 分钟/小时/日/周。
    '''

    def __init__(self, config: dict = None):
        self.config = config or PHASE_CONFIG_DEFAULT.copy()

    def detect(self, df: pd.DataFrame) -> DailyProfile:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        vol = df['volume'].values
        n = len(close)
        if n < 40:
            return self._default()
        roc = np.zeros(n - 5)
        vol_ratio = np.zeros(n - 5)
        for i in range(5, n):
            roc[i - 5] = close[i] / max(close[i - 5], 0.001) - 1
            vol_ma5 = np.mean(vol[i - 4:i + 1])
            vol_ma20 = np.mean(vol[max(0, i - 19):i + 1])
            vol_ratio[i - 5] = vol_ma5 / max(vol_ma20, 1)
        roc_pct = _percentile_rank(roc, roc[-1])
        vr_pct = _percentile_rank(vol_ratio, vol_ratio[-1])
        if roc[-1] > 0:
            fire_val = roc_pct * (0.6 + 0.4 * vr_pct)
        else:
            fire_val = 1 - roc_pct
        fire = PhaseState(float(np.clip(fire_val, 0, 1)), '火')
        vpt = np.zeros(n)
        for i in range(1, n):
            ret = (close[i] - close[i - 1]) / max(close[i - 1], 0.001)
            vpt[i] = vpt[i - 1] + ret * vol[i]
        if n >= 21:
            vpt_slice = vpt[-21:]
        else:
            vpt_slice = vpt
        vpt_diff = np.diff(vpt_slice)
        water_val = _percentile_rank(vpt_diff, vpt_diff[-1]) if len(vpt_diff) > 0 else 0.5
        water = PhaseState(float(np.clip(water_val, 0, 1)), '水')
        ma20 = np.mean(close[-20:]) if n >= 20 else np.mean(close)
        std20 = np.std(close[-20:]) if n >= 20 else np.std(close)
        bb_width = 4 * std20 / max(ma20, 0.001)
        bb_hist = []
        for i in range(20, n):
            m = np.mean(close[i - 19:i + 1])
            s = np.std(close[i - 19:i + 1])
            bb_hist.append(4 * s / max(m, 0.001))
        bb_hist = np.array(bb_hist)
        bb_pct = _percentile_rank(bb_hist, bb_width)
        wood_val = 1 - bb_pct
        wood = PhaseState(float(np.clip(wood_val, 0, 1)), '木')
        tr = np.zeros(n)
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        atr_hist = []
        for i in range(21, n):
            atr_hist.append(np.mean(tr[i - 4:i + 1]) / max(np.mean(tr[i - 20:i + 1]), 0.001))
        atr_hist = np.array(atr_hist)
        atr_cur = np.mean(tr[-5:]) / max(np.mean(tr[-21:]), 0.001) if n >= 21 else 1
        metal_val = _percentile_rank(atr_hist, atr_cur)
        metal = PhaseState(float(np.clip(metal_val, 0, 1)), '金')
        window = min(40, n - 1)
        recent = close[-window:]
        recent_amp = (recent.max() - recent.min()) / max(recent.mean(), 0.001)
        amp_hist = []
        for i in range(window, n - window + 1):
            seg = close[i:i + window]
            amp_hist.append((seg.max() - seg.min()) / max(seg.mean(), 0.001))
        amp_hist = np.array(amp_hist) if amp_hist else np.array([recent_amp])
        # 土形 = 段内相对振幅的补: 窄幅横盘(低振幅)→土形高(储备蓄势)
        #       宽幅震荡(高振幅)→土形低(火形活跃, 非储备)
        earth_val = 1 - _percentile_rank(amp_hist, recent_amp)
        earth = PhaseState(float(np.clip(earth_val, 0, 1)), '土')
        return DailyProfile(fire=fire, water=water, wood=wood, metal=metal,
                            earth=earth, config=self.config)

    def _default(self) -> DailyProfile:
        return DailyProfile(fire=PhaseState(0.5, '火'), water=PhaseState(0.5, '水'),
                            wood=PhaseState(0.5, '木'), metal=PhaseState(0.5, '金'),
                            earth=PhaseState(0.5, '土'), config=self.config)


def backtest(df: pd.DataFrame, window: int = 120, min_confidence: float = 0.3,
             config: dict = None) -> dict:
    """
    滚动窗口回测。

    Args:
        df: OHLCV DataFrame, index=date
        window: 检测窗口数(期)
        min_confidence: 最低预测信心阈值
        config: 可选个股校准参数

    Returns:
        {
            'accuracy': float,       # 方向命中率
            'signals': int,          # 有把握的信号数
            'dirs': {                # 各方向明细
                '↗': {'n': N, 'correct': N, 'acc': float},
                ...
            },
            'profiles': [],          # 每日相位记录
        }
    """
    detector = DailyPhaseDetector(config=config)
    close = df['close'].values
    dirs = {}
    total, correct = 0, 0
    profiles = []
    for i in range(window, len(close) - 1):
        chunk = df.iloc[i - window:i]
        profile = detector.detect(chunk)
        direction, confidence, reason = profile.predict_next_direction()
        actual_ret = close[i + 1] / close[i] - 1
        if confidence < min_confidence:
            profiles.append({
                'date': df.index[i],
                'direction': direction,
                'confidence': confidence,
                'hit': None,
                'profile': profile,
            })
            continue
        if direction == '↗':
            hit = actual_ret > 0
        elif direction == '↘':
            hit = actual_ret < 0
        elif direction == '→':
            hit = abs(actual_ret) < 0.003
        else:
            hit = False
        total += 1
        if hit:
            correct += 1
        if direction not in dirs:
            dirs[direction] = {'n': 0, 'correct': 0}
        dirs[direction]['n'] += 1
        if hit:
            dirs[direction]['correct'] += 1
        profiles.append({
            'date': df.index[i],
            'direction': direction,
            'confidence': confidence,
            'hit': hit,
            'actual_ret': float(actual_ret),
            'reason': reason,
            'profile': profile,
        })
    return {
        'accuracy': correct / max(total, 1),
        'signals': total,
        'correct': correct,
        'dirs': {k: {**v, 'acc': v['correct'] / max(v['n'], 1)} for k, v in dirs.items()},
        'profiles': profiles,
    }


def calibrate(df: pd.DataFrame, window_base: int = 200, fast: bool = False) -> dict:
    """
    对个股进行参数网格搜索, 返回最佳配置。

    搜索空间 (fast=True 时减半):
      - saturation_threshold: [0.80, 0.85, 0.90]
      - up_fire_min: [0.55, 0.65, 0.75]
      - up_earth_max: [0.60, 0.70, 0.80]

    评分 = 准确率 × log(信号数+5) × (↘准确率 / ↗准确率均衡因子)

    Args:
        df: OHLCV DataFrame
        window_base: 回测检测窗口
        fast: 快速模式 (搜索 3×2×2=12 组合)

    Returns:
        {'config': dict, 'score': float, 'accuracy': float, 'signals': int}
    """
    if fast:
        sats = [0.8, 0.88]
        fires = [0.55, 0.7]
        earths = [0.65, 0.8]
    else:
        sats = [0.8, 0.85, 0.9]
        fires = [0.55, 0.65, 0.75]
        earths = [0.6, 0.7, 0.8]
    best = {'config': None, 'score': -1, 'accuracy': 0, 'signals': 0, 'dirs': {}, 'params': ''}
    total = len(sats) * len(fires) * len(earths)
    i = 0
    for sat in sats:
        for fmin in fires:
            for emax in earths:
                cfg = PHASE_CONFIG_DEFAULT.copy()
                cfg['saturation_threshold'] = sat
                cfg['up_fire_min'] = fmin
                cfg['up_earth_max'] = emax
                result = backtest(df, window=window_base, min_confidence=0.3, config=cfg)
                acc = result['accuracy']
                sig = result['signals']
                dirs = result.get('dirs', {})
                up_acc = dirs.get('↗', {}).get('acc', 0)
                dn_acc = dirs.get('↘', {}).get('acc', 0)
                balance = min(up_acc, dn_acc) / max(up_acc, dn_acc, 0.01)
                score = acc * np.log(sig + 5) * (0.5 + 0.5 * balance)
                i += 1
                if score > best['score']:
                    best.update({
                        'config': cfg,
                        'score': score,
                        'accuracy': acc,
                        'signals': sig,
                        'dirs': {k: f"{v['acc']:.0%}" for k, v in dirs.items()},
                        'params': f'sat={sat} fmin={fmin} emax={emax}',
                    })
    return best
