# Source Generated with Decompyle++
# File: industry_mapping.cpython-311.pyc (Python 3.11)

"""
SPUM × 行业适配映射表
=====================

五形耦合链（火→金→土→火→水→木→火）是通用拓扑公理，
不随行业改变。行业适配只是将财务报表中的可用指标，
映射为五个相位的占优度估计。

耦合链最适配的行业: 科技(火), 消费(木), 制造(金), 地产(土)
不适配的行业(放弃): 金融/银行 — 报表结构特殊, 五形相位不适用

格式说明:
  industry: {
    'phases': {                    # 五个相位的指标映射
      'fire': {
        'primary': 'keyword',      # 主要指标（列名关键词）
        'fallback': 'keyword',     # 后备指标
        'transform': 'fn_name',    # 转换函数: raw_value → [0,1]
        'weight': float,           # 该指标在相位判断中的权重
      },
      ...
    },
    'thresholds': {                # 行业特定的阈值覆盖
      'fire_saturation': 0.85,
      ...
    },
    'phase_weights': {             # 各相位在完整识别中的相对重要性
      'fire': 1.0,
      ...
    },
  }
"""

def _pct(raw):
    '''百分比值 → 小数: 10.5 → 0.105'''
    return raw / 100


def _identity(raw):
    '''原样返回'''
    return raw


def _clip(raw, lo=0.0, hi=1.0):
    '''裁剪到[lo, hi]'''
    return max(lo, min(hi, raw))


def _positive(raw):
    '''截断负数: -5 → 0'''
    return max(0, raw)

TRANSFORMS = {}


def register(name):
    def deco(fn):
        TRANSFORMS[name] = fn
        return fn
    return deco


@register('growth_to_fire')
def growth_to_fire(growth_pct, thresh):
    '''营收增长率% → 火形占优度
    <5% → 0, >30% → 1'''
    g = _positive(growth_pct) / 100.0
    if g <= 0.03:
        return 0.0
    if g >= 0.3:
        return 1.0
    return (g - 0.03) / 0.27


@register('growth_to_water')
def growth_to_water(growth_pct, thresh):
    '''营收增长率% → 水形占优度
    水形看传输强度, 用增长率+周转率联合'''
    g = _positive(growth_pct) / 100.0
    if g >= 0.4:
        return 1.0
    return _clip(g / 0.4)


@register('margin_to_wood')
def margin_to_wood(margin_pct, thresh):
    '''毛利率% → 木形占优度 (环锁定深度)
    <15% → 0 (无环), >75% → 1 (强锁)'''
    m = margin_pct / 100.0
    if m <= 0.15:
        return 0.0
    if m >= 0.75:
        return 1.0
    return (m - 0.15) / 0.6


@register('debt_to_earth')
def debt_to_earth(debt_pct, thresh):
    '''资产负债率% → 土形占优度 (储备充足度)
    土形是储备池: 低负债=储备充足=高土形'''
    d = debt_pct / 100.0
    if d >= 0.85:
        return 0.0
    if d <= 0.2:
        return 1.0
    return 1 - (d - 0.2) / 0.65


@register('roe_to_metal')
def roe_to_metal(roe_pct, thresh):
    '''ROE% → 金形占优度 (修剪效率)
    高ROE = 高效 = 金形修剪充分
    <3% → 0, >20% → 1'''
    r = roe_pct
    if r <= 3:
        return 0.0
    if r >= 20:
        return 1.0
    return (r - 3) / 17


@register('cost_rate_to_margin')
def cost_rate_to_margin(cost_rate_pct, thresh):
    '''主营业务成本率% → 毛利率近似 → 木形
    毛利率 ≈ 1 - 成本率'''
    gm = 1.0 - cost_rate_pct / 100.0
    if gm <= 0.15:
        return 0.0
    if gm >= 0.75:
        return 1.0
    return (gm - 0.15) / 0.6


@register('profit_growth_to_fire')
def profit_growth_to_fire(profit_growth_pct, thresh):
    '''净利润增长率% → 火形补充'''
    g = _positive(profit_growth_pct) / 100.0
    if g <= 0.03:
        return 0.0
    if g >= 0.3:
        return 1.0
    return (g - 0.03) / 0.27


@register('turnover_to_water')
def turnover_to_water(turnover, thresh):
    '''总资产周转率 → 水形补充
    <0.1 → 0, >1.0 → 1'''
    if turnover <= 0.1:
        return 0.0
    if turnover >= 1.0:
        return 1.0
    return (turnover - 0.1) / 0.9


@register('debt_to_earth_realestate')
def debt_to_earth_realestate(debt_pct, thresh):
    '''地产行业资产负债率 → 土形
    地产高杠杆是常态, 阈值上移
    >90% → 枯竭, <50% → 充足'''
    d = debt_pct / 100.0
    if d >= 0.9:
        return 0.0
    if d <= 0.5:
        return 1.0
    return 1 - (d - 0.5) / 0.4


@register('turnover_to_metal')
def turnover_to_metal(turnover, thresh):
    '''周转率 → 金形 (效率)
    <0.2 → 0, >0.8 → 1'''
    if turnover <= 0.2:
        return 0.0
    if turnover >= 0.8:
        return 1.0
    return (turnover - 0.2) / 0.6


@register('current_ratio_to_earth')
def current_ratio_to_earth(cr, thresh):
    '''流动比率 → 土形修正
    <0.8 → 0 (枯竭), >3.0 → 1 (冗余)'''
    if cr <= 0.8:
        return 0.0
    if cr >= 3.0:
        return 1.0
    return (cr - 0.8) / 2.2


@register('zero_transform')
def zero_transform(raw, thresh):
    '''找不到指标时返回 0.5 (中性)'''
    return 0.5
STD_INDICATORS = {
    'revenue_growth': '主营业务收入增长率',
    'profit_growth': '净利润增长率',
    'gross_margin': '销售毛利率',
    'cost_rate': '主营业务成本率',
    'debt_ratio': '资产负债率',
    'roe': '净资产收益率',
    'turnover': '总资产周转率',
    'current_ratio': '流动比率',
    'fee_ratio': '三项费用比重' }
INDUSTRY_MAP = {
    'tech': {
        'name': '科技',
        'ticker_suffix': [
            'SZ',
            'SH',
            'BJ'],
        'default_thresholds': {
            'fire_saturation': 0.85,
            'water_saturation': 0.85,
            'wood_saturation': 0.85 },
        'phases': {
            'fire': {
                'label': '火',
                'description': '∇σ梯度/市场驱动力',
                'primary': {
                    'indicator': 'revenue_growth',
                    'transform': 'growth_to_fire' },
                'secondary': {
                    'indicator': 'profit_growth',
                    'transform': 'profit_growth_to_fire',
                    'weight': 0.3 } },
            'water': {
                'label': '水',
                'description': '链式传输/营收渠道',
                'primary': {
                    'indicator': 'revenue_growth',
                    'transform': 'growth_to_water' },
                'secondary': {
                    'indicator': 'turnover',
                    'transform': 'turnover_to_water',
                    'weight': 0.3 } },
            'wood': {
                'label': '木',
                'description': '环锁定/护城河',
                'primary': {
                    'indicator': 'cost_rate',
                    'transform': 'cost_rate_to_margin' },
                'fallback': {
                    'indicator': 'gross_margin',
                    'transform': 'margin_to_wood' } },
            'metal': {
                'label': '金',
                'description': '桥边修剪/效率',
                'primary': {
                    'indicator': 'roe',
                    'transform': 'roe_to_metal' },
                'secondary': {
                    'indicator': 'turnover',
                    'transform': 'turnover_to_metal',
                    'weight': 0.3 } },
            'earth': {
                'label': '土',
                'description': '储备池/缓冲',
                'primary': {
                    'indicator': 'debt_ratio',
                    'transform': 'debt_to_earth' },
                'secondary': {
                    'indicator': 'current_ratio',
                    'transform': 'current_ratio_to_earth',
                    'weight': 0.25 } } } },
    'consumer': {
        'name': '消费',
        'ticker_suffix': [
            'SZ',
            'SH',
            'BJ'],
        'default_thresholds': {
            'fire_saturation': 0.8,
            'wood_saturation': 0.85,
            'metal_saturation': 0.85 },
        'phases': {
            'fire': {
                'label': '火',
                'description': '市场需求梯度',
                'primary': {
                    'indicator': 'revenue_growth',
                    'transform': 'growth_to_fire' } },
            'water': {
                'label': '水',
                'description': '分销渠道/供应链',
                'primary': {
                    'indicator': 'revenue_growth',
                    'transform': 'growth_to_water' },
                'secondary': {
                    'indicator': 'turnover',
                    'transform': 'turnover_to_water',
                    'weight': 0.4 } },
            'wood': {
                'label': '木',
                'description': '品牌护城河/环锁定',
                'primary': {
                    'indicator': 'cost_rate',
                    'transform': 'cost_rate_to_margin' },
                'fallback': {
                    'indicator': 'gross_margin',
                    'transform': 'margin_to_wood' } },
            'metal': {
                'label': '金',
                'description': '运营效率/修剪',
                'primary': {
                    'indicator': 'roe',
                    'transform': 'roe_to_metal' },
                'secondary': {
                    'indicator': 'turnover',
                    'transform': 'turnover_to_metal',
                    'weight': 0.3 } },
            'earth': {
                'label': '土',
                'description': '现金储备/缓冲',
                'primary': {
                    'indicator': 'debt_ratio',
                    'transform': 'debt_to_earth' },
                'secondary': {
                    'indicator': 'current_ratio',
                    'transform': 'current_ratio_to_earth',
                    'weight': 0.25 } } } },
    'manufacturing': {
        'name': '制造',
        'ticker_suffix': [
            'SZ',
            'SH',
            'BJ'],
        'default_thresholds': {
            'fire_saturation': 0.8,
            'metal_saturation': 0.85 },
        'phases': {
            'fire': {
                'label': '火',
                'description': '市场驱动',
                'primary': {
                    'indicator': 'revenue_growth',
                    'transform': 'growth_to_fire' } },
            'water': {
                'label': '水',
                'description': '供应链/渠道',
                'primary': {
                    'indicator': 'revenue_growth',
                    'transform': 'growth_to_water' },
                'secondary': {
                    'indicator': 'turnover',
                    'transform': 'turnover_to_water',
                    'weight': 0.4 } },
            'wood': {
                'label': '木',
                'description': '技术壁垒/环锁定',
                'primary': {
                    'indicator': 'cost_rate',
                    'transform': 'cost_rate_to_margin' },
                'fallback': {
                    'indicator': 'gross_margin',
                    'transform': 'margin_to_wood' } },
            'metal': {
                'label': '金',
                'description': '生产效率/修剪',
                'primary': {
                    'indicator': 'roe',
                    'transform': 'roe_to_metal' },
                'secondary': {
                    'indicator': 'turnover',
                    'transform': 'turnover_to_metal',
                    'weight': 0.4 } },
            'earth': {
                'label': '土',
                'description': '储备/缓冲',
                'primary': {
                    'indicator': 'debt_ratio',
                    'transform': 'debt_to_earth' },
                'secondary': {
                    'indicator': 'current_ratio',
                    'transform': 'current_ratio_to_earth',
                    'weight': 0.25 } } } },
    'real_estate': {
        'name': '地产',
        'ticker_suffix': [
            'SZ',
            'SH'],
        'default_thresholds': {
            'earth_saturation': 0.9,
            'fire_saturation': 0.8 },
        'phases': {
            'fire': {
                'label': '火',
                'description': '市场需求/销售热度',
                'primary': {
                    'indicator': 'revenue_growth',
                    'transform': 'growth_to_fire' } },
            'water': {
                'label': '水',
                'description': '项目周转/销售渠道',
                'primary': {
                    'indicator': 'turnover',
                    'transform': 'turnover_to_water' } },
            'wood': {
                'label': '木',
                'description': '土储/环锁定',
                'primary': {
                    'indicator': 'cost_rate',
                    'transform': 'cost_rate_to_margin' } },
            'metal': {
                'label': '金',
                'description': '降本/去杠杆/修剪',
                'primary': {
                    'indicator': 'roe',
                    'transform': 'roe_to_metal' } },
            'earth': {
                'label': '土',
                'description': '杠杆储备/土储',
                'primary': {
                    'indicator': 'debt_ratio',
                    'transform': 'debt_to_earth_realestate' } } } } }
DEFAULT_INDUSTRY = 'tech'
