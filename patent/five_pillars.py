#!/usr/bin/env python3
"""
SPUM 五项合参 — 望闻问切 + 八字
==================================
轻量版，单文件，无外部重型依赖。

五柱融合架构：
  八字(生日) → 先天五行体质基线 ──┐
                                 ├──→ 融合判定 → 综合病历
  望(面/舌/毛/体) → 外观五行偏性 ──┘        ↑
  闻(症状描述) → 关键帧事件标记 ──────────────┘
  问(拟合) → 证型确认 ──────────────────────┘
  切(ECG ΔF) → ΔF向量 → 证型匹配 ──────────┘

依赖：无（纯 Python 标准库）
接口预留：望诊 data_only 模式，接入识图 AI 时仅需替换 _parse_vision_result()
"""

import datetime
import re
from typing import Dict, Optional, List, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════
# 常量表
# ═══════════════════════════════════════════════════════════

# 天干 → 五行
TIAN_GAN_WUXING = {
    0: "木", 1: "木",  # 甲乙
    2: "火", 3: "火",  # 丙丁
    4: "土", 5: "土",  # 戊己
    6: "金", 7: "金",  # 庚辛
    8: "水", 9: "水",  # 壬癸
}
TIAN_GAN_NAMES = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]

# 地支 → 五行
DI_ZHI_WUXING = {
    0: "水", 1: "土", 2: "木", 3: "木", 4: "土", 5: "火",
    6: "火", 7: "土", 8: "金", 9: "金", 10: "土", 11: "水",
}
DI_ZHI_NAMES = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]

# 地支 → 月令索引（正月寅=2, 二月卯=3, ...）
DI_ZHI_MONTH = {2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 0: 0, 1: 1}

# 日干 × 月令 → 旺衰（0=休囚, 1=相, 2=旺）
# [日干木(0,1), 火(2,3), 土(4,5), 金(6,7), 水(8,9)]
# 月令索引: 寅卯(2,3)=春木旺, 巳午(5,6)=夏火旺, 申酉(8,9)=秋金旺, 亥子(0,11)=冬水旺, 丑辰未戌(1,4,7,10)=四季土旺
WANG_SHUAI = {
    # (日干五行, 月令五行) → 状态: 2旺 1相 0休囚死
    ("木", "木"): 2, ("木", "火"): 1, ("木", "土"): 0, ("木", "金"): 0, ("木", "水"): 1,
    ("火", "木"): 1, ("火", "火"): 2, ("火", "土"): 1, ("火", "金"): 0, ("火", "水"): 0,
    ("土", "木"): 0, ("土", "火"): 1, ("土", "土"): 2, ("土", "金"): 1, ("土", "水"): 0,
    ("金", "木"): 0, ("金", "火"): 0, ("金", "土"): 1, ("金", "金"): 2, ("金", "水"): 1,
    ("水", "木"): 1, ("水", "火"): 0, ("水", "土"): 0, ("水", "金"): 1, ("水", "水"): 2,
}

# 五行偏性 → 性格倾向
WUXING_PERSONALITY = {
    "木": "条达直接，不耐压抑，易怒易郁。决断力强，但易刚愎。",
    "火": "热情外向，急躁易喜，社交活跃。情感波动大，喜极则伤心。",
    "土": "敦厚稳重，思虑周全，耐力好。偏保守，不易应变。",
    "金": "果断刚毅，规则感强，有洁癖倾向。偏冷漠，不善表达情感。",
    "水": "灵活善变，直觉敏锐，适应力强。偏深沉，易多思多虑。",
}

# 五行偏性 → 情志易感（该五行人易受哪种情志所伤）
WUXING_EMOTION_VULNERABILITY = {
    "木": "怒", "火": "喜",
    "土": "思", "金": "悲",
    "水": "恐",
}

# 面色 → 五行偏性
FACE_COLOR_MAP = {
    "青": "木", "赤": "火", "红": "火",
    "黄": "土",
    "白": "金", "苍白": "金",
    "黑": "水", "晦暗": "水",
}

# 体毛特征 → 五行偏性（按部位 × 疏密 × 软硬）
# 不同部位映射不同脏腑：
#   头发 = 肾之华在发 + 肝(血之余) + 胃(前额)
#   眉毛 = 肝(眉为肝之余)
#   鼻毛 = 肺(鼻为肺之窍)
#   胸毛 = 心(上焦)
#   腋毛 = 心/肾
#   阴毛 = 肝/肾(下焦)
#   四肢体毛 = 肺(皮毛)
BODY_HAIR_MAP = {
    # ── 头发 ──
    "头发_浓密": {"肾": 0.5, "肝": 0.3, "胃": 0.0},
    "头发_稀疏": {"肾": -0.5, "肝": -0.3, "胃": 0.0},
    "头发_细软": {"肾": -0.3, "肝": 0.0, "胃": 0.0},
    "头发_粗硬": {"肾": 0.5, "肝": 0.3, "胃": 0.0},
    "头发_早白": {"肾": -0.5, "肝": 0.3},  # 肾亏+肝郁
    "头发_油腻": {"胃": 0.5, "脾": 0.3},   # 脾胃湿热
    "头发_干枯": {"肾": -0.5, "肝": -0.3},  # 肝肾阴虚
    # 头发分布模式
    "发_前额脱发": {"胃": -0.5, "大肠": -0.3},  # 阳明经气虚
    "发_头顶脱发": {"肾": -0.8, "肝": -0.3},    # 肾精亏虚
    "发_两鬓斑白": {"肝": 0.5, "胆": 0.3},      # 肝胆郁热
    "发_整体稀疏": {"肾": -0.5, "肝": -0.3, "血": -0.5},  # 血虚+肾亏
    "发_发际线后移": {"胃": -0.3, "大肠": -0.3},
    # 眉毛
    "眉_浓密": {"肝": 0.5},
    "眉_稀疏": {"肝": -0.5},
    "眉_脱落": {"肝": -0.8, "肾": -0.3},  # 肝肾亏虚
    "眉_焦黄": {"肝": -0.3, "脾": 0.3},    # 脾虚湿热
    # 鼻毛
    "鼻毛_浓密": {"肺": 0.5},
    "鼻毛_稀疏": {"肺": -0.3},
    "鼻毛_脱落": {"肺": -0.5},
    "鼻毛_长出鼻孔": {"肺": 0.5, "肾": -0.3},  # 肺气实+肾不纳气
    # 胸毛
    "胸毛_浓密": {"心": 0.5, "肾": 0.3},
    "胸毛_稀疏": {"心": -0.3},
    "胸毛_无": {"心": 0.0, "肾": 0.0},  # 正常
    # 腋毛
    "腋毛_浓密": {"心": 0.3, "肾": 0.5},
    "腋毛_稀疏": {"肾": -0.3},
    "腋毛_脱落": {"肾": -0.5, "心": -0.3},  # 心肾两虚
    # 阴毛
    "阴毛_浓密": {"肝": 0.5, "肾": 0.3},    # 下焦气盛
    "阴毛_稀疏": {"肝": -0.3, "肾": -0.3},  # 肝肾不足
    "阴毛_脱落": {"肝": -0.5, "肾": -0.5},  # 肝肾大亏
    "阴毛_早白": {"肾": -0.5, "肝": 0.3},
    # 四肢体毛
    "体毛_浓密": {"肺": 0.5, "肝": 0.3},    # 肺气实
    "体毛_稀疏": {"肺": -0.3, "血": -0.3},  # 肺气虚+血虚
    "体毛_粗硬": {"肺": 0.5},
    "体毛_细软": {"肺": -0.3},
}  # fmt: skip

# 脏腑 → 五行映射（用于体毛偏移聚合）
ORGAN_WUXING = {
    "肝": "木", "胆": "木",
    "心": "火", "小肠": "火",
    "脾": "土", "胃": "土",
    "肺": "金", "大肠": "金",
    "肾": "水", "膀胱": "水",
    "血": "木",  # 肝主血
}

# 体型 → 五行偏性
BODY_TYPE_MAP = {
    "消瘦": "火", "肥胖": "土",
    "壮实": "木", "纤细": "水",
    "匀称": "土",
}

# 舌象 → 五行偏性（基础规则）
TONGUE_MAP = {
    "淡红": {}, "红": {"火": 0.5}, "绛": {"火": 1.0},
    "紫暗": {"水": 0.5, "金": 0.3}, "淡白": {"火": -0.5, "土": -0.3},
    "黄苔": {"土": 0.5}, "白苔": {"金": 0.3}, "腻苔": {"土": 0.8},
    "少苔": {"水": -0.5, "火": 0.3}, "剥苔": {"水": -1.0},
}

# 常见症状 → 五行偏性偏移
SYMPTOM_WUXING_MAP = {
    "头痛": {"木": 0.3, "火": 0.2}, "头晕": {"水": -0.3, "土": 0.2},
    "失眠": {"火": 0.5, "水": -0.3}, "多梦": {"火": 0.3, "木": 0.2},
    "心悸": {"火": 0.5, "木": 0.3}, "胸闷": {"木": 0.3, "土": 0.3},
    "胁痛": {"木": 0.5}, "腹胀": {"土": 0.5},
    "食欲不振": {"土": -0.5}, "便秘": {"金": 0.5, "火": 0.3},
    "腹泻": {"土": -0.5, "水": 0.3}, "尿频": {"水": 0.5},
    "腰酸": {"水": -0.5}, "膝软": {"水": -0.5},
    "怕冷": {"火": -0.5, "水": 0.3}, "怕热": {"火": 0.5, "水": -0.3},
    "多汗": {"火": 0.3, "土": -0.3}, "自汗": {"土": -0.5},
    "口干": {"火": 0.3, "水": -0.5}, "口苦": {"火": 0.5, "木": 0.3},
    "咳嗽": {"金": 0.5, "土": -0.3}, "痰多": {"土": 0.5},
    "烦躁": {"火": 0.5, "木": 0.3}, "抑郁": {"木": -0.5, "火": -0.3},
    "疲劳": {"土": -0.5, "水": -0.3}, "气短": {"金": -0.5, "土": -0.3},
}


# ═══════════════════════════════════════════════════════════
# 模块一：八字 — 先天五行体质基线
# ═══════════════════════════════════════════════════════════

def _is_leap(year: int) -> bool:
    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)


def _days_between(y1: int, m1: int, d1: int, y2: int, m2: int, d2: int) -> int:
    """返回 y2-m2-d2 减去 y1-m1-d1 的天数（可正可负）。"""
    def _to_days(y: int, m: int, d: int) -> int:
        mdays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        days = (y - 1) * 365 + (y - 1) // 4 - (y - 1) // 100 + (y - 1) // 400
        for i in range(m - 1):
            days += mdays[i]
        if m > 2 and _is_leap(y):
            days += 1
        return days + d - 1
    return _to_days(y2, m2, d2) - _to_days(y1, m1, d1)


def _get_day_stem_branch(year: int, month: int, day: int) -> Tuple[int, int]:
    """计算日柱的天干(0-9)和地支(0-11)。

    参考点：2000-01-01 = 戊午日（天干=4, 地支=6）
    """
    days = _days_between(2000, 1, 1, year, month, day)
    stem = (4 + days) % 10
    branch = (6 + days) % 12
    return stem, branch


def _get_hour_branch(hour: int) -> int:
    """时支：23-0=子(0), 1-2=丑(1), ..., 21-22=亥(11)"""
    return (hour + 1) // 2 % 12


def _get_month_stem(year_stem: int, month: int) -> int:
    """计算月干（五虎遁）。month=1..12"""
    # 甲己之年丙作首，乙庚之岁戊为头，丙辛之岁寻庚上，
    # 丁壬壬寅顺水流，若问戊癸何处起，甲寅之上好追求。
    # month 1(寅)对应月干偏移
    offset = [2, 4, 6, 8, 0]  # 甲(0)→丙(2), 乙(1)→戊(4), 丙(2)→庚(6), 丁(3)→壬(8), 戊(4)→甲(0)
    return (offset[year_stem % 5] + month + 1) % 10


def bazi_profile(year: int, month: int, day: int, hour: Optional[int] = None) -> Dict:
    """八字 → 先天五行体质基线。

    参数：
        year/month/day: 公历出生日期
        hour: 出生时辰（0-23），可选

    返回：
        {
            "先天五行": {"木": 0.3, "火": 0.0, ...},  # 正=偏旺, 负=偏弱
            "性格": "...",
            "情志易伤": "喜",
            "日干": "甲", "日干五行": "木",
            "旺衰": "旺",
        }
    """
    # 年柱
    year_stem = (year - 4) % 10
    year_branch = (year - 4) % 12

    # 月柱
    month_stem = _get_month_stem(year_stem, month)

    # 日柱
    day_stem, day_branch = _get_day_stem_branch(year, month, day)

    # 时柱（可选）
    if hour is not None:
        hour_branch = _get_hour_branch(hour)
        hour_stem = (day_stem % 5 * 2 + hour_branch) % 10  # 五鼠遁
    else:
        hour_stem, hour_branch = None, None

    # 日干五行
    self_wuxing = TIAN_GAN_WUXING[day_stem]

    # 月令地支 → 五行
    month_branch = (month + 1) % 12  # 正月寅=2 → 地支索引
    month_wuxing = DI_ZHI_WUXING[month_branch]

    # 旺衰
    wangshuai = WANG_SHUAI.get((self_wuxing, month_wuxing), 0)

    # 先天五行基线：日干旺衰 × 0.5 作为该行偏性
    baseline = {"木": 0.0, "火": 0.0, "土": 0.0, "金": 0.0, "水": 0.0}
    baseline[self_wuxing] = (wangshuai - 1) * 0.5  # 旺=+0.5, 相=0.0, 休囚=-0.5

    return {
        "日干": TIAN_GAN_NAMES[day_stem],
        "日干五行": self_wuxing,
        "旺衰": ["休囚", "相", "旺"][wangshuai],
        "先天五行": baseline,
        "性格": WUXING_PERSONALITY[self_wuxing],
        "情志易伤": WUXING_EMOTION_VULNERABILITY[self_wuxing],
        "_八字": {
            "年柱": f"{TIAN_GAN_NAMES[year_stem]}{DI_ZHI_NAMES[year_branch]}",
            "月柱": f"{TIAN_GAN_NAMES[month_stem]}{DI_ZHI_NAMES[month_branch]}",
            "日柱": f"{TIAN_GAN_NAMES[day_stem]}{DI_ZHI_NAMES[day_branch]}",
            "时柱": f"{TIAN_GAN_NAMES[hour_stem]}{DI_ZHI_NAMES[hour_branch]}" if hour_stem is not None else None,
        },
    }


# ═══════════════════════════════════════════════════════════
# 模块二：望诊 — 外观五行偏性（接口预留）
# ═══════════════════════════════════════════════════════════

def inspect_appearance(
    face_color: Optional[str] = None,
    tongue: Optional[str] = None,
    body_hair: Optional[List[str]] = None,
    body_type: Optional[str] = None,
    vision_raw: Optional[Dict] = None,
) -> Dict:
    """望诊 → 外观五行偏性。

    参数：
        face_color: 面色（青/赤/红/黄/白/苍白/黑/晦暗）
        tongue:    舌象（淡红/红/绛/紫暗/淡白/苔色等）
        body_hair: 体毛特征列表，每项按"部位_描述"格式，如：
                   ["头发_稀疏", "发_前额脱发", "眉_浓密", "鼻毛_长出鼻孔",
                    "胸毛_浓密", "腋毛_脱落", "阴毛_稀疏", "体毛_粗硬"]
        body_type: 体型（消瘦/肥胖/壮实/纤细/匀称）
        vision_raw: 预留接口，接入识图AI的原始输出

    返回：
        {"外观五行": {"木": 0.0, ...}, "体毛": [...], "特征列表": [...]}
    """
    bias = {"木": 0.0, "火": 0.0, "土": 0.0, "金": 0.0, "水": 0.0}
    features = []
    confidence = 0

    # 面色
    if face_color and face_color in FACE_COLOR_MAP:
        wx = FACE_COLOR_MAP[face_color]
        bias[wx] += 0.4
        features.append(f"面色{face_color} → {wx}偏盛")
        confidence += 0.3

    # 体毛（关键特征 — 按部位×疏密×软硬）
    hair_features = []
    if body_hair:
        for h in body_hair:
            if h in BODY_HAIR_MAP:
                organ_bias = BODY_HAIR_MAP[h]
                for organ, val in organ_bias.items():
                    wx = ORGAN_WUXING.get(organ)
                    if wx:
                        bias[wx] += val * 0.35
                hair_features.append(h)
                features.append(f"体毛: {h}")
        if hair_features:
            confidence += 0.35

    # 舌象
    if tongue and tongue in TONGUE_MAP:
        tongue_bias = TONGUE_MAP[tongue]
        for wx, val in tongue_bias.items():
            bias[wx] += val * 0.3
        features.append(f"舌象{tongue}")
        confidence += 0.2

    # 体型
    if body_type and body_type in BODY_TYPE_MAP:
        wx = BODY_TYPE_MAP[body_type]
        bias[wx] += 0.3
        features.append(f"体型{body_type} → {wx}偏盛")
        confidence += 0.2

    return {
        "外观五行": bias,
        "体毛": hair_features,
        "特征列表": features,
        "置信度": min(confidence, 1.0),
        "_数据源": "人工录入" if confidence > 0 else "无数据",
    }


# ═══════════════════════════════════════════════════════════
# 模块三：闻诊 — 症状→事件帧标记
# ═══════════════════════════════════════════════════════════

def inquiry_symptoms(symptom_text: str) -> Dict:
    """症状描述 → 关键帧事件标记。

    参数：
        symptom_text: 症状描述（逗号/空格分隔，如"头痛,失眠,心悸"）

    返回：
        {
            "匹配症状": [...],
            "五行偏移": {"木": 0.0, ...},
            "疑似证型": ["肝阳上亢"],
            "事件帧": "失眠+心悸 → 心火亢盛"
        }
    """
    bias = {"木": 0.0, "火": 0.0, "土": 0.0, "金": 0.0, "水": 0.0}
    matched = []

    # 切分症状
    tokens = re.split(r'[,，、\s]+', symptom_text.strip())

    for token in tokens:
        if not token:
            continue
        for sym, wx_bias in SYMPTOM_WUXING_MAP.items():
            if sym in token:
                for wx, val in wx_bias.items():
                    bias[wx] += val
                if sym not in matched:
                    matched.append(sym)
                break

    # 生成事件帧
    event_frame = ""
    if matched:
        top_wx = max(bias, key=bias.get)
        event_frame = f"{'+'.join(matched)} → {top_wx}扰动"

    return {
        "匹配症状": matched,
        "五行偏移": bias,
        "事件帧": event_frame,
        "症状数": len(matched),
    }


# ═══════════════════════════════════════════════════════════
# 模块四：融合 — P0~P2 置信度层级体系
# ═══════════════════════════════════════════════════════════
#
# 置信度层级（从高到低）：
#
#   P0 ── 人体本身（ground truth，不参与计算，为所有测量的锚点）
#         中医的起点是"人"，不是数据。所有仪器测量都是对人的投影。
#
#   P1 ── 仪器直接投影（最高置信度）
#         ① 脉诊仪(ECG) → 人体心电生理的直接投影
#         ② 望诊(照片/语气/面色/舌象) → 人体外观的直接投影
#         ③ 体态、声频等 → 体的机械振动的直接投影
#         特点：不经过患者主观转述，测量-人体的映射关系最直接
#
#   P1.5 ─ P1 的延伸（略低于 P1）
#         ① 患者说话语气、语调（非内容） → 气机状态的间接投影
#         ② 颜面部微表情 → 情志的直接投影
#         这类数据接近直接投影但受社交修饰影响
#
#  修正项 ── 问诊（低于 P1，用于修正 P1 的解读方向）
#         ① 患者主诉（自述症状）
#         ② 发病前后时间线
#         ③ 生活/饮食/情志等诱因
#         特点：经过患者主观转述和认知过滤，不能独立推翻 P1，
#                但能调整 P1 的解读权重和方向
#
#   P2 ── 八字/先天（最低置信度）
#         ① 先天五行体质基线
#         ② 旺衰/格局推断
#         ③ 性格倾向/易感情志
#         特点：仅提供"背景参考"，不直接参与辨证，
#                仅在 P1 已有结果后用于"比对印证"或"预警提示"
#
# 核心原则 — 中医西医不可通约：
#   ① 中医辨证与西医诊断属于两种范式，不存在"西医确诊 = 中医金标准"
#   ② 脉诊仪测的是中医脉象（浮沉迟数虚实），不是心电图ST段
#   ③ 五色望诊对应的是中医五色辨证（青风赤火热黄湿白燥黑寒），不是皮肤科分型
#   ④ 任何以西医生化指标/影像学结果校正中医证型的做法，本质是范式归约暴力
#   ⑤ 置信度只在本范式内有序：不能在中医框架内说"这个辨证置信度低于西医诊断"
#   ⑥ 临床路径上可以交叉参考，但自信度评估必须各自独立
# ═══════════════════════════════════════════════════════════

# P1 直接投影的默认置信度基础值
CONFIDENCE_P1_BASE = 0.85       # 脉诊(ECG)直接投影
CONFIDENCE_P1_VISION = 0.75     # 望诊(照片)直接投影，略低（受光照/角度影响）
CONFIDENCE_P1_TONE = 0.70       # 语气/语气语调，略低（受社交修饰）

# 修正项(问诊)最大修正幅度
CONFIDENCE_CORRECTION_MAX = 0.15   # 问诊最多只能 ±0.15 修正 P1 结果
CONFIDENCE_CORRECTION_WEIGHT = 0.4 # 问诊中每项匹配的五行修正权重

# P2(八字)背景参考权重
CONFIDENCE_P2_WEIGHT = 0.15     # 八字只作为 0.15 的背景叠加


def layer_p1_pulse(pulse_result: Optional[Dict]) -> Dict:
    """
    P1 层：脉诊仪 ECG → 人体心电生理的直接投影

    输入：pulse_diagnosis.spum_pulse_diagnosis() 的输出
    输出：P1 层结构化判定
    """
    if not pulse_result:
        return {"可用": False, "置信度": 0.0, "ΔF": None, "证型": None}

    delta_F = None
    # 优先用几何五形的耦合结果
    geo = pulse_result.get("五形几何分析（双层耦合）", {})
    if geo and geo.get("delta_F_coupled") is not None:
        delta_F = geo["delta_F_coupled"]
    elif "五维修正向量 ΔF" in pulse_result:
        raw = pulse_result["五维修正向量 ΔF"]
        if isinstance(raw, dict):
            delta_F = np.array([float(v) for v in raw.values()])
        else:
            delta_F = np.array([float(v) for v in raw])

    syndrome = pulse_result.get("辨证诊断", {})
    sqi = pulse_result.get("信号质量 SQI", 0.5)

    # 置信度 = SQI 校正后的 P1 基值
    confidence = CONFIDENCE_P1_BASE * min(1.0, sqi / 0.7)

    return {
        "可用": True,
        "层级": "P1",
        "置信度": round(confidence, 4),
        "ΔF": delta_F,
        "证型": syndrome.get("主证"),
        "辨证置信": syndrome.get("置信度", 0.0),
        "综合摘要": syndrome.get("综合摘要", ""),
    }


def layer_p1_vision(inspection: Optional[Dict]) -> Dict:
    """
    P1 层：望诊(面色/舌象/体毛/体态) → 人体外观的直接投影

    输入：inspect_appearance() 的输出
    输出：P1 层结构化判定
    """
    if not inspection:
        return {"可用": False, "置信度": 0.0}

    wuxing = inspection.get("外观五行", {})
    if not wuxing or all(v == 0.0 for v in wuxing.values()):
        return {"可用": False, "置信度": 0.0, "外观五行": wuxing}

    return {
        "可用": True,
        "层级": "P1",
        "置信度": CONFIDENCE_P1_VISION,
        "外观五行": wuxing,
    }


def layer_correction_inquiry(inquiry: Optional[Dict]) -> Dict:
    """
    修正层：问诊(症状主诉) → 修正项

    只做方向性调整，不独立输出判定。
    最大修正幅度不超过 CONFIDENCE_CORRECTION_MAX
    """
    if not inquiry:
        return {"可用": False, "偏移": {"木": 0.0, "火": 0.0, "土": 0.0, "金": 0.0, "水": 0.0}}

    bias = inquiry.get("五行偏移", {})
    if not bias or all(v == 0.0 for v in bias.values()):
        return {"可用": False, "偏移": {"木": 0.0, "火": 0.0, "土": 0.0, "金": 0.0, "水": 0.0}}

    n_matched = inquiry.get("症状数", 0)
    # 匹配症状越多，修正幅度越大
    correction_strength = min(1.0, n_matched / 5.0) * CONFIDENCE_CORRECTION_MAX

    # 归一化偏移方向
    total = max(0.001, sum(abs(v) for v in bias.values()))
    direction = {k: (v / total) * correction_strength for k, v in bias.items()}

    return {
        "可用": True,
        "层级": "修正项",
        "置信度": round(correction_strength / CONFIDENCE_CORRECTION_MAX, 4),
        "偏移": direction,
        "匹配症状": inquiry.get("匹配症状", []),
        "事件帧": inquiry.get("事件帧", ""),
    }


def layer_p2_bazi(bazi: Optional[Dict]) -> Dict:
    """
    P2 层：八字 → 先天体质背景参考

    仅用于"比对印证"或"预警提示"，不直接参与 P1 的辨证修正。
    """
    if not bazi:
        return {"可用": False, "置信度": 0.0}

    return {
        "可用": True,
        "层级": "P2",
        "置信度": CONFIDENCE_P2_WEIGHT,
        "日主五行": bazi.get("日干五行"),
        "五柱五行": bazi.get("先天五行"),
    }


def hierarchical_fusion(
    p1_pulse: Dict,
    p1_vision: Dict,
    correction: Dict,
    p2_bazi: Dict,
) -> Dict:
    """
    分层融合 — P1 主判定 → 修正项调整 → P2 背景参考

    流程：
      1. P1（脉诊+望诊）独立输出主判定和 ΔF
      2. 修正项(问诊)在 ±CORRECTION_MAX 范围内调整 ΔF
      3. P2(八字)仅做比对输出，不修改 ΔF

    中医西医不可通约保护：
      - 所有置信度只在中医范式内比较
      - 不引入西医生化指标作为"金标准"
      - P1 的 DNA 来自望闻问切和八字，不来自实验室检查
    """
    result = {
        "融合层级": {},
        "P1主判定": None,
        "修正项": None,
        "P2背景参考": None,
        "综合ΔF": None,
        "最终判定": None,
        "置信度": 0.0,
        "范式声明": "本系统为中医辨证体系，不可通约于西医诊断范式",
    }

    # ── 步骤1：P1 主判定 ──
    p1_available = []
    main_delta_F = np.zeros(5)
    p1_confidence = 0.0

    if p1_pulse.get("可用"):
        p1_available.append("脉诊")
        if p1_pulse["ΔF"] is not None:
            main_delta_F = np.array(p1_pulse["ΔF"])
            p1_confidence = p1_pulse["置信度"]
        result["P1主判定"] = {
            "来源": "脉诊(ECG直接投影)",
            "ΔF": [round(float(v), 4) for v in p1_pulse["ΔF"]] if p1_pulse["ΔF"] is not None else None,
            "证型": p1_pulse["证型"],
            "辨证置信": p1_pulse["辨证置信"],
            "置信度": p1_pulse["置信度"],
            "综合摘要": p1_pulse["综合摘要"],
        }

    if p1_vision.get("可用"):
        p1_available.append("望诊")
        vision_wx = p1_vision["外观五行"]
        # 将望诊五行转换为 ΔF 方向的微调（仅 ±0.1 量级）
        vision_adjust = np.array([vision_wx.get("木", 0) * 0.1,
                                   vision_wx.get("火", 0) * 0.1,
                                   vision_wx.get("土", 0) * 0.1,
                                   vision_wx.get("金", 0) * 0.1,
                                   vision_wx.get("水", 0) * 0.1])
        if not p1_pulse.get("可用"):
            # 仅有望诊时，独立作为主判定
            main_delta_F = vision_adjust
            p1_confidence = p1_vision["置信度"]
        else:
            # 脉诊+望诊融合（望诊权重 0.3）
            main_delta_F = main_delta_F * 0.7 + vision_adjust * 0.3
            p1_confidence = max(p1_confidence, p1_vision["置信度"])

        if result["P1主判定"] is None:
            result["P1主判定"] = {
                "来源": "望诊(外观直接投影)",
                "ΔF": [round(float(v), 4) for v in vision_adjust],
                "置信度": p1_vision["置信度"],
            }
        else:
            result["P1主判定"]["来源"] = "+".join(p1_available)

    result["融合层级"]["P1"] = {
        "可用来源": p1_available,
        "置信度": round(p1_confidence, 4),
    }

    # ── 步骤2：修正项(问诊)调整 ──
    if correction.get("可用"):
        offset = correction["偏移"]
        corr_vec = np.array([offset.get("木", 0), offset.get("火", 0),
                              offset.get("土", 0), offset.get("金", 0),
                              offset.get("水", 0)])
        main_delta_F = main_delta_F + corr_vec
        main_delta_F = np.clip(main_delta_F, -1.0, 1.0)
        result["修正项"] = {
            "偏移方向": {k: round(v, 4) for k, v in offset.items() if abs(v) > 0.001},
            "症状": correction.get("匹配症状", []),
            "事件帧": correction.get("事件帧", ""),
            "置信度": correction.get("置信度", 0),
        }
        result["融合层级"]["修正项"] = {
            "可用": True,
            "修正幅度": round(float(np.max(np.abs(corr_vec))), 4),
        }
    else:
        result["融合层级"]["修正项"] = {"可用": False}

    # ── 步骤3：P2(八字)背景参考 ──
    if p2_bazi.get("可用"):
        bazi_wx = p2_bazi.get("五柱五行", {})
        result["P2背景参考"] = {
            "日主五行": p2_bazi.get("日主五行"),
            "先天体质偏性": bazi_wx,
            "置信度": p2_bazi["置信度"],
            "说明": "P2 仅作背景参考，不直接参与 ΔF 修正",
        }
        result["融合层级"]["P2"] = {
            "可用": True,
            "置信度": p2_bazi["置信度"],
        }
    else:
        result["融合层级"]["P2"] = {"可用": False}

    # ── 结果输出 ──
    result["综合ΔF"] = [round(float(v), 4) for v in main_delta_F]

    # 最终判定
    dim_names = ["木", "火", "土", "金", "水"]
    max_dim = dim_names[np.argmax(np.abs(main_delta_F))]
    result["最终判定"] = {
        "主要偏性": max_dim,
        "置信度": round(p1_confidence, 4),
        "方向": "亢" if main_delta_F[dim_names.index(max_dim)] > 0 else "虚",
    }

    # 整体置信度 = P1 置信度 + 修正项增量
    total_conf = p1_confidence
    if correction.get("可用"):
        total_conf += correction.get("置信度", 0) * 0.1
    result["置信度"] = round(min(1.0, total_conf), 4)

    return result


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def spum_five_pillars(birth_year: int, birth_month: int, birth_day: int,
                      birth_hour: Optional[int] = None,
                      face_color: Optional[str] = None,
                      tongue: Optional[str] = None,
                      body_hair: Optional[List[str]] = None,
                      body_type: Optional[str] = None,
                      symptoms: Optional[str] = None,
                      pulse_result: Optional[Dict] = None) -> Dict:
    """五项合参主入口。

    参数：
        birth_year/month/day: 公历出生日期（必须）
        birth_hour: 出生时辰 0-23（可选）
        face_color/tongue/body_hair/body_type: 望诊项（可选）
        body_hair: 体毛列表，每项如"头发_稀疏""发_前额脱发""胸毛_浓密"等
        symptoms: 症状文本（可选，如"头痛,失眠"）
        pulse_result: 切诊结果（来自 pulse_diagnosis.py，可选）

    返回：
        综合病历 dict
    """
    report = {
        "八字": None,
        "望诊": None,
        "闻诊": None,
        "切诊": None,
        "融合": None,
        "个人信息": {
            "出生日期": f"{birth_year}-{birth_month:02d}-{birth_day:02d}",
            "有出生时辰": birth_hour is not None,
        },
    }

    # 八字（必须只要有生日）
    report["八字"] = bazi_profile(birth_year, birth_month, birth_day, birth_hour)

    # 望诊（可选）
    if any(x is not None for x in [face_color, tongue, body_hair, body_type]):
        report["望诊"] = inspect_appearance(face_color, tongue, body_hair, body_type)

    # 闻诊（可选）
    if symptoms:
        report["闻诊"] = inquiry_symptoms(symptoms)

    # 切诊（可选，外部传入）
    if pulse_result:
        report["切诊"] = pulse_result

    # 融合（分层置信度体系）
    report["融合"] = hierarchical_fusion(
        p1_pulse=layer_p1_pulse(pulse_result),
        p1_vision=layer_p1_vision(report["望诊"]),
        correction=layer_correction_inquiry(report["闻诊"]),
        p2_bazi=layer_p2_bazi(report["八字"]),
    )

    return report


# ═══════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("SPUM 五项合参 — 自测")
    print("=" * 50)

    # 测试八字
    print("\n▸ 八字测试（1990-07-15）：")
    bp = bazi_profile(1990, 7, 15)
    print(f"  日干: {bp['日干']}({bp['日干五行']}) {bp['旺衰']}")
    print(f"  先天五行: {bp['先天五行']}")
    print(f"  性格: {bp['性格']}")
    print(f"  情志易伤: {bp['情志易伤']}")

    # 测试望诊（多部位体毛 + 面色）
    print("\n▸ 望诊测试（面色赤 + 多部位体毛）：")
    ins = inspect_appearance(
        face_color="赤",
        body_hair=["头发_稀疏", "发_头顶脱发", "眉_浓密", "鼻毛_长出鼻孔",
                   "胸毛_浓密", "腋毛_脱落", "阴毛_稀疏", "体毛_粗硬"],
    )
    print(f"  外观五行: {ins['外观五行']}")
    print(f"  体毛特征: {ins['体毛']}")

    # 测试闻诊
    print("\n▸ 闻诊测试（头痛,失眠,心悸）：")
    inq = inquiry_symptoms("头痛,失眠,心悸")
    print(f"  匹配症状: {inq['匹配症状']}")
    print(f"  五行偏移: {inq['五行偏移']}")
    print(f"  事件帧: {inq['事件帧']}")

    # 测试融合
    print("\n▸ 融合测试（八字+望诊+闻诊）：")
    fusion = five_pillars_fusion(bazi=bp, inspection=ins, inquiry=inq)
    print(f"  综合五行: {fusion['综合五行']}")
    print(f"  主偏: {fusion['主偏']}")
    print(f"  参与合参: {fusion['参与合参']}")

    print("\n✅ 五项合参系统自测完成")
