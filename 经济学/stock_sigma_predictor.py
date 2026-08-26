# Source Generated with Decompyle++
# File: stock_sigma_predictor.cpython-311.pyc (Python 3.11)
# 修复记录: 由 marshal+dis 逐指令还原 + 注册 sys.modules 后反射验证。
#   - JointSignal dataclass: 反编译损坏 -> 按字节码重建全部 17 字段 + summary 方法
#   - predict/_evaluate_S_trend: 6 处 "Decompyle incomplete" 与丢失分支已还原
#   - None.x 变量名丢失: 还原为 inf/S/dominant 等正确局部变量
#   - quick_stock_predict 签名: 以反射为准 (code='', name='', industry='default', start='20220101')
#   - 顶层失败根源: importlib 手工加载 .pyc 未注册 sys.modules, 导致 dataclass 类体
#     _process_class 取 cls.__module__ 模块字典失败; .py 正常 import 无此问题。

'''
SPUM 股价预测器: 五形 S 向量 + σ 振荡 → 价格投影
===================================================

三层架构:
  Layer 1 (实体层): CompanyWuxing — S 五维向量描述公司拓扑健康
  Layer 2 (市场层): StockSigma — σ 振荡描述市场价格动态
  Layer 3 (投影层): WuxingSigmaPredictor — S×σ 联合推演

核心假设:
  价格 = f(S向量, σ振荡状态)
  其中 f 的约束由 dv/dt ≤ const 和 D_max 决定。
'''
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings


@dataclass
class JointSignal:
    '''联合信号: S×σ 融合预测'''
    stock_code: str
    stock_name: str
    timestamp: str
    S: 'WuxingVector'
    inflections: List
    sigma_current: float = 0.0
    sigma_pred: float = 0.0
    sigma_gradient: float = 0.0
    sigma_phase: str = ''
    sigma_confidence: float = 0.0
    net_direction: str = '中性'
    net_confidence: float = 0.0
    time_horizon: str = '短期'
    S_direction: str = '中性'
    S_confidence: float = 0.0
    sigma_direction: str = '中性'
    reason: str = ''

    def summary(self) -> dict:
        return {
            'stock': f'{self.stock_name}({self.stock_code})',
            'S_vector': str(self.S),
            'S_direction': self.S_direction,
            'S_confidence': f'{self.S_confidence:.0%}',
            'sigma_direction': self.sigma_direction,
            'net_direction': self.net_direction,
            'net_confidence': f'{self.net_confidence:.0%}',
            'inflection_count': len(self.inflections),
            'reason': self.reason,
            'horizon': self.time_horizon,
        }


class WuxingSigmaPredictor:
    '''
    S × σ 联合预测器。

    将 S 五形向量提供的实体层信息与 σ 振荡提供的市场层信息融合。

    判据规则:
      ┌─────────────┬──────────────┬──────────────────────┐
      │ S 趋势      │ σ 趋势       │ 联合预测             │
      ├─────────────┼──────────────┼──────────────────────┤
      │ 看涨        │ 看涨         │ ★★★ 强看涨          │
      │ 看涨        │ 震荡         │ ★★  看涨 (σ确认)    │
      │ 看跌        │ 看跌         │ ★★★ 强看跌          │
      │ 看跌        │ 震荡         │ ★★  看跌 (σ确认)    │
      │ 中性/过热   │ 看涨         │ ⚠ 短期冲顶的可能    │
      │ 中性/过冷   │ 看跌         │ ⚠ 短期超卖的可能    │
      │ 过热(S≥0.85)│ 看涨         │ ☢ 拐点预警           │
      │ 过冷(S≤0.15)│ 看跌         │ ☢ 反转预警           │
      │ 背离        │ 背离         │ 观望                 │
      └─────────────┴──────────────┴──────────────────────┘
    '''

    def __init__(self, sigma_analyzer=None):
        self.sigma = sigma_analyzer

    def predict(self, wuxing_result: dict, sigma_result: Optional[dict] = None,
                stock_code: str = '', stock_name: str = '') -> JointSignal:
        '''
        执行 S × σ 联合预测。

        参数:
          wuxing_result: CompanyWuxing.analyze() 的结果
          sigma_result: StockSigma.analyze() 的结果 (可选)
          stock_code, stock_name: 标识

        返回:
          JointSignal
        '''
        S = wuxing_result['S_current']
        inflections = wuxing_result['inflections']
        S_dir, S_conf = self._evaluate_S_trend(S, inflections)
        sigma_dir = '中性'
        sigma_conf = 0.0
        sigma_cur = 0.0
        sigma_pred = 0.0
        sigma_grad = 0.0
        sigma_phase = ''
        if sigma_result:
            sigma_cur = float(sigma_result['sigma'].iloc[-1])
            sigma_grad = float(sigma_result['gradient'].iloc[-1])
            sigma_phase = str(sigma_result['phase_label'].iloc[-1])
            pred = sigma_result['prediction']
            sigma_pred = pred['sigma_pred']
            sigma_conf = pred['confidence']
            sigma_dir = '看涨' if sigma_pred < sigma_cur else '看跌'
        now = datetime.now().isoformat(timespec='seconds')
        joint = JointSignal(
            stock_code=stock_code,
            stock_name=stock_name,
            timestamp=now,
            S=S,
            inflections=inflections,
            sigma_current=float(sigma_cur),
            sigma_pred=float(sigma_pred),
            sigma_gradient=float(sigma_grad),
            sigma_phase=sigma_phase,
            sigma_confidence=sigma_conf,
            S_direction=S_dir,
            S_confidence=S_conf,
            sigma_direction=sigma_dir,
        )
        overheat = [i for i in inflections if i.direction == '过热']
        overcold = [i for i in inflections if i.direction == '过冷']
        if overheat:
            comps = ','.join([i.component for i in overheat])
            joint.S_direction = '过热拐点'
            joint.net_direction = '看跌'
            joint.net_confidence = max(i.confidence for i in overheat)
            joint.time_horizon = '中期'
            joint.reason = f'{comps}形接近D_max, 拐点将至, 再投入边际收益趋零'
            if sigma_dir == '看涨':
                joint.reason += ' (σ仍在看涨, 短期可能最后一段冲顶)'
            return joint
        if overcold and sigma_dir != '中性':
            comps = ','.join([i.component for i in overcold])
            joint.net_direction = sigma_dir
            joint.net_confidence = min(0.85, sigma_conf + 0.2)
            joint.time_horizon = '中期'
            joint.reason = f'{comps}形严重不足 + σ{sigma_dir}确认'
            return joint
        if S_dir == sigma_dir and S_dir != '中性':
            joint.net_direction = S_dir
            joint.net_confidence = min(0.9, 0.5 * S_conf + 0.5 * sigma_conf + 0.15)
            joint.time_horizon = '长期' if S_conf > sigma_conf else '短期'
            joint.reason = f'S{S_dir} + σ{sigma_dir} 一致确认'
        elif S_dir == '中性' and sigma_dir == '中性':
            joint.net_direction = '中性'
            joint.net_confidence = 0.0
            joint.reason = 'S和σ均无明确方向'
        elif S_dir != '中性':
            joint.net_direction = S_dir
            joint.net_confidence = S_conf * 0.7
            joint.time_horizon = '长期'
            joint.reason = f'S{S_dir}, σ{sigma_dir}反向→短期噪声, 长期S趋势优先'
        elif sigma_dir != '中性':
            joint.net_direction = sigma_dir
            joint.net_confidence = sigma_conf * 0.6
            joint.time_horizon = '短期'
            joint.reason = f'σ{sigma_dir}, S中性→短期动量为主'
        return joint

    def _evaluate_S_trend(self, S: 'WuxingVector', inflections: list) -> Tuple[str, float]:
        '''
        从 S 向量判断基本面趋势。

        规则:
          - 火 > 0.7 + 水 > 0.5 → 看涨 (成长驱动)
          - 木 > 0.8 + 土 > 0.6 → 看涨 (护城河+储备) 但注意是否过热
          - 火 < 0.15 + 水 < 0.15 → 看跌 (增长枯竭)
          - 木 < 0.2 + 金 < 0.3 → 看跌 (无护城河且效率低)
          - 只有一个分量偏低但其它正常 → 中性偏弱
          - 全部在 [0.25, 0.75] → 中性
        '''
        for inf in inflections:
            if inf.direction == '过热':
                return ('过热拐点', inf.confidence)
            if inf.direction == '过冷':
                return ('过冷', min(0.7, inf.confidence))
        cold = sum(1 for v in (S.S_water, S.S_wood, S.S_earth, S.S_metal, S.S_fire) if v <= 0.15)
        hot = sum(1 for v in (S.S_water, S.S_wood, S.S_earth, S.S_metal, S.S_fire) if v >= 0.85)
        if cold >= 3:
            return ('看跌', 0.7)
        if hot >= 2:
            return ('过热拐点', 0.75)
        if S.S_fire > 0.7 and S.S_water > 0.5:
            return ('看涨', min(0.8, 0.5 + S.S_fire * 0.3))
        if S.S_fire < 0.15 and S.S_water < 0.15:
            return ('看跌', 0.65)
        if S.S_wood > 0.7 and S.S_earth > 0.5:
            if S.S_fire > 0.3:
                return ('看涨', 0.65)
            return ('中性', 0.5)
        all_ok = all(0.2 <= v <= 0.8 for v in (S.S_water, S.S_wood, S.S_earth, S.S_metal, S.S_fire))
        if all_ok:
            return ('中性', 0.5)
        dominant = S.dominant()
        s_map = {
            '水': 'water',
            '木': 'wood',
            '土': 'earth',
            '金': 'metal',
            '火': 'fire' }
        d_val = getattr(S, f'S_{s_map[dominant]}', 0.5)
        if d_val > 0.6:
            return ('偏强', 0.55)
        return ('偏弱', 0.5)


def quick_stock_predict(code: str = '', name: str = '', industry: str = 'default',
                        start: str = '20220101') -> dict:
    '''
    一键执行: 获取数据 → 五形分析 → σ振荡 → 联合预测。

    返回 summary dict。
    '''
    from stock_wuxing import CompanyWuxing
    from stock_sigma import StockSigma
    from stock_sigma_multi import IndexSectorDemo
    print(f'>> {name or code}: 五形分析...')
    wuxing = CompanyWuxing(name=name or code, code=code, industry=industry)
    wuxing.fetch_data(code)
    wuxing_result = wuxing.analyze()
    S = wuxing_result['S_current']
    print(f'   S向量: {S}')
    print(f'>> {name or code}: σ振荡分析...')
    try:
        sigma = StockSigma()
        df = sigma.fetch_a_stock(code, start=start, source='tx')
        sigma_result = sigma.analyze(df)
        print(f'   σ: {sigma_result["sigma"].iloc[-1]:.3f} → {sigma_result["prediction"]["sigma_pred"]:.3f}')
    except Exception as e:
        print(f'   σ分析失败: {e}')
        sigma_result = None
    predictor = WuxingSigmaPredictor()
    joint = predictor.predict(wuxing_result, sigma_result, stock_code=code, stock_name=name or code)
    print(f'>> 联合预测: {joint.net_direction} (信心: {joint.net_confidence:.0%}, 跨度: {joint.time_horizon})')
    print(f'   原因: {joint.reason}')
    print(f'   拐点: {len(joint.inflections)} 个')
    return joint.summary()
