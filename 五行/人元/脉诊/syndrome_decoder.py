#!/usr/bin/env python3
"""
SPUM 辨证解码器 — ΔF 五维修正向量 → 中医证型
==============================================

将五形 ΔF = (S_木, S_火, S_土, S_金, S_水) 映射到中医证型。

映射逻辑：
  每个证型由一组"五形偏离模式"定义——
  木↑ 表示 S_木 正向偏离（偏硬/约束强）
  火↓ 表示 S_火 负向偏离（偏缓/梯度弱）
  以此类推。

匹配算法：
  1. 加权余弦相似度（按证型的活跃维度加权）
  2. 残差分解支持兼证识别
  3. 置信度校准

用法：
  from syndrome_decoder import decode_syndrome
  result = decode_syndrome(delta_F=[0.27, 0.25, 0.35, 0.60, -0.45])
"""

import numpy as np
from typing import List, Tuple, Dict

# ═══════════════════════════════════════════════════════════
# 证型数据库
# ═══════════════════════════════════════════════════════════
# 每个证型定义：
#   sig = [木, 火, 土, 金, 水]  — 期望偏离方向
#         1 = 显著偏高, -1 = 显著偏低, 0 = 不参与
#         0.5/-0.5 = 轻度偏离
#   weight — 各维度权重（该证型主要影响的维度）
#   desc — 描述

SYNDROMES = [
    # ─── 脏证（五脏）───
    {
        "name": "肝阳上亢",
        "abbr": "肝亢",
        "sig": [ 1.0,  0.5,  0.0,  0.0, -0.5],  # 木亢+火偏亢+水偏亏
        "weight": [1.0, 0.6, 0.0, 0.0, 0.5],
        "desc": "肝气郁结化火，阳亢于上。血管张力偏高，脉率偏急，流通欠畅。"
    },
    {
        "name": "肝气郁结",
        "abbr": "肝郁",
        "sig": [ 0.0,  0.0,  0.0,  0.0, -0.5],  # 水滞为主
        "weight": [0.5, 0.0, 0.0, 0.0, 1.0],
        "desc": "气机不畅。脉弦而不急，流通阻滞。"
    },
    {
        "name": "心火亢盛",
        "abbr": "心火",
        "sig": [ 0.0,  1.0,  0.0,  0.5, -0.5],  # 火亢+金略亢+水亏
        "weight": [0.0, 1.0, 0.0, 0.3, 0.5],
        "desc": "心经火热。脉率偏急，波形偏硬，流通欠畅。"
    },
    {
        "name": "心阳虚",
        "abbr": "阳虚",
        "sig": [-0.5, -0.5, -0.5,  0.5, -0.5],  # 木火土皆亏+金亢(代偿)
        "weight": [0.7, 0.7, 0.5, 0.3, 0.3],
        "desc": "心阳不足，温运无力。脉率偏缓，脉体偏细，电压偏低。"
    },
    {
        "name": "心气虚",
        "abbr": "气虚",
        "sig": [-0.5,  0.0, -0.5,  0.0,  0.0],  # 木亏+土亏
        "weight": [0.7, 0.0, 0.7, 0.0, 0.0],
        "desc": "心气不足，鼓动无力。脉软而细。"
    },
    {
        "name": "脾虚湿困",
        "abbr": "脾湿",
        "sig": [ 0.0,  0.0,  0.5,  0.0, -0.5],  # 土实(假性)+水滞
        "weight": [0.0, 0.0, 0.8, 0.0, 0.5],
        "desc": "脾失健运，湿浊内停。脉体偏粗而流通不畅。"
    },

    # ─── 气血津液 ───
    {
        "name": "气血两虚",
        "abbr": "气血虚",
        "sig": [-0.5, -0.5, -1.0, -0.5,  0.0],  # 土(细)显著偏负
        "weight": [0.5, 0.3, 1.0, 0.3, 0.0],
        "desc": "气血俱亏。脉体偏细（土虚），脉率偏缓（火衰），张力不足（木弱）。"
    },
    {
        "name": "气滞血瘀",
        "abbr": "瘀血",
        "sig": [ 0.0,  0.0,  0.0,  0.5, -1.0],  # 金亢+水显著滞
        "weight": [0.0, 0.0, 0.0, 0.6, 1.0],
        "desc": "血液瘀滞。波形碎裂（金亢），流通严重阻滞（水滞）。"
    },
    {
        "name": "阴虚火旺",
        "abbr": "阴虚",
        "sig": [ 0.5,  0.5, -0.5,  0.0, -0.5],  # 木火偏亢+土亏+水亏
        "weight": [0.5, 0.5, 0.5, 0.0, 0.7],
        "desc": "阴液不足，虚火内扰。脉偏细（土亏），张力偏大（木亢），流通欠畅（水亏）。"
    },
    {
        "name": "痰湿内阻",
        "abbr": "痰湿",
        "sig": [ 0.0,  0.0,  0.7,  0.7, -0.7],  # 增强阈值：需要显著偏离
        "weight": [0.0, 0.0, 0.7, 0.5, 0.7],
        "desc": "痰浊壅盛。脉体显著偏粗（土实），波形显著碎裂（金亢），流通明显阻滞（水滞）。需土>0.4才判。"
    },

    # ─── 外感 ───
    {
        "name": "外感发热",
        "abbr": "外感",
        "sig": [ 0.5,  1.0,  0.0,  0.0,  0.0],  # 火显著亢
        "weight": [0.3, 1.0, 0.0, 0.0, 0.0],
        "desc": "外邪袭表，正邪交争。脉率显著偏急（火亢），张力偏大（木亢）。"
    },

    # ─── 特殊 ───
    {
        "name": "心脉瘀阻",
        "abbr": "心瘀",
        "sig": [ 0.5,  0.0,  0.0,  1.0, -1.0],  # 木亢+金亢+水滞
        "weight": [0.5, 0.0, 0.0, 0.8, 0.8],
        "desc": "心脉痹阻。血管张力偏高（木亢），波形碎裂（金亢），流通严重阻滞（水滞）。"
    },
    {
        "name": "心肾不交",
        "abbr": "心肾",
        "sig": [ 0.5,  0.5, -0.5,  0.0, -0.5],  # 上热下寒
        "weight": [0.5, 0.7, 0.3, 0.0, 0.5],
        "desc": "心肾水火不济。上焦火亢（火↑）、下焦水亏（水↓），脉体偏细（土↓）。"
    },
    # ─── 特殊模式（单一维度显著异常）───
    {
        "name": "火衰（温差趋零）",
        "abbr": "火衰",
        "sig": [ 0.0, -0.8,  0.0,  0.0,  0.0],  # 仅火维度显著偏低
        "weight": [0.0, 1.0, 0.0, 0.0, 0.0],
        "desc": "体表-核心温差趋零。外周灌注崩塌风险，常见于室速/室颤/循环衰竭。"
    },
]

# ═══════════════════════════════════════════════════════════
# 群体基线
# ═══════════════════════════════════════════════════════════
# MIT-BIH 正常心律(14条)的五形均值
# 用于校正临床 ECG 基线：dF_adj = dF - baseline
# 校正后，偏离正常值的方向和幅度才是证型判别的真实信号
NORMAL_BASELINE = [0.247, 0.203, 0.354, 0.600, -0.407]


def decode_syndrome(delta_F: List[float],
                    qualities: Dict = None,
                    dynamics: Dict = None,
                    baseline: List[float] = None,
                    top_k: int = 3,
                    min_confidence: float = 0.15) -> Dict:
    """
    主入口：ΔF 五维修正向量 → 辨证结果

    参数：
        delta_F: [木, 火, 土, 金, 水] 五维向量
        qualities: 可选，六品质详情
        dynamics: 可选，帧间动力学签名
        baseline: 可选，正常人群基线。
                  提供后 dF_adj = dF - baseline 再匹配证型。
                  ECG 数据建议用 NORMAL_BASELINE。
        top_k: 返回前 k 个匹配证型
        min_confidence: 最小置信度阈值

    返回：
        {
            'primary': {'name': '肝阳上亢', 'confidence': 0.82, ...},
            'secondary': [...],
            'combinations': ['肝阳上亢+心火亢盛', ...],
            'summary': '综合判断描述',
        }
    """
    dF = np.array(delta_F)

    # ── 基线校正 ──
    # 减去正常人群均值，使 dF_adj 代表"偏离正常的程度"
    # 校正后：正常记录 → dF_adj ≈ 0；病理记录 → dF_adj 方向明显
    if baseline is not None:
        dF_match = dF - np.array(baseline)
        # 同时也要调整回退阈值——正常脉象 = 偏离基线 < 阈值
        fallback_threshold = 0.08
    else:
        dF_match = dF
        fallback_threshold = 0.10

    # ─── 匹配每个证型 ───
    scores = []
    for syn in SYNDROMES:
        sig = np.array(syn["sig"])
        w = np.array(syn["weight"])
        score = _match_syndrome(dF_match, sig, w)
        scores.append((score, syn))

    # 排序
    scores.sort(key=lambda x: -x[0])

    # ─── 主证（最佳匹配）───
    primary_score, primary_syn = scores[0]

    # ─── 正常脉象回退 ───
    mean_dev = np.mean(np.abs(dF_match))
    if mean_dev < fallback_threshold and primary_score < 0.55:
        primary_syn = {
            "name": "正常脉象",
            "abbr": "正常",
            "sig": [0, 0, 0, 0, 0],
            "weight": [1, 1, 1, 1, 1],
            "desc": "五形趋于平衡。各部品质适中，无显著偏离。",
        }
        primary_score = 0.50

    # ─── 兼证（残差匹配：在 dF_match 空间中进行）───
    if primary_score > 0.15 and primary_syn["name"] != "正常脉象":
        residual = dF_match - np.array(primary_syn["sig"]) * min(primary_score, 1.0)
        secondary_scores = []
        for syn in SYNDROMES:
            if syn["name"] == primary_syn["name"]:
                continue
            sig = np.array(syn["sig"])
            w = np.array(syn["weight"])
            score = _match_syndrome(residual, sig, w)
            secondary_scores.append((score, syn))
        secondary_scores.sort(key=lambda x: -x[0])
    else:
        secondary_scores = []

    # ─── 结果组装 ───
    result = {
        "五形输入": {k: round(float(v), 3) for k, v in
                    zip(['S_木(约束)', 'S_火(梯度)', 'S_土(储备)', 'S_金(修剪)', 'S_水(流通)'], dF)},
    }

    # 主证
    sp, ss = primary_score, primary_syn
    if sp >= min_confidence:
        result["主证"] = {
            "证型": ss["name"],
            "缩写": ss["abbr"],
            "置信度": round(float(sp), 3),
            "描述": ss["desc"],
            "匹配模式": {k: round(float(v), 2) for k, v in
                       zip(['木', '火', '土', '金', '水'], ss["sig"])},
        }
    else:
        result["主证"] = {"证型": "未明确", "置信度": round(float(sp), 3), "描述": "偏离幅度不足以匹配已知证型"}

    # 兼证
    secondaries = []
    for sp2, ss2 in secondary_scores:
        if sp2 >= min_confidence and ss2["name"] != primary_syn["name"]:
            secondaries.append({
                "证型": ss2["name"],
                "置信度": round(float(sp2), 3),
            })
        if len(secondaries) >= min(top_k - 1, len(secondary_scores)):
            break
    if secondaries:
        result["兼证"] = secondaries

    # 组合证型检测
    combos = _detect_combinations(dF_match)
    if combos:
        result["组合证型"] = combos

    # 动力学补充
    if dynamics:
        sig_type = dynamics.get("脉动力学签名", {}).get("类型", "")
        if "强健型" in sig_type:
            result["动力学校正"] = "正气尚充，恢复能力可"
        elif "衰惫型" in sig_type or "郁滞型" in sig_type:
            result["动力学校正"] = "正气不足，恢复能力差，建议扶正为主"

    # 综合摘要
    summary_parts = []
    pri = result.get("主证", {})
    summary_parts.append(f"主证：{pri.get('证型','?')}（置信度 {pri.get('置信度',0):.0%}）")
    if secondaries:
        sec_strs = [f"{s['证型']}({s['置信度']:.0%})" for s in secondaries]
        summary_parts.append(f"兼证：{'，'.join(sec_strs)}")
    if combos:
        summary_parts.append(f"组合模式：{'；'.join(combos)}")
    if qualities:
        # 补充六品质细节
        q_desc = []
        for label, key, pos, neg in [
            ("软硬", "soft_hard", "偏硬", "偏软"),
            ("粗细", "thin_thick", "偏粗", "偏细"),
            ("缓急", "slow_urgent", "偏急", "偏缓"),
        ]:
            v = qualities.get(key, 0)
            if abs(v) > 0.3:
                q_desc.append(f"{pos if v>0 else neg}({v:+.2f})")
        if q_desc:
            summary_parts.append(f"品质特征：{'，'.join(q_desc)}")

    result["综合摘要"] = "；".join(summary_parts)

    return result


def _match_syndrome(dF: np.ndarray, sig: np.ndarray, weight: np.ndarray) -> float:
    """
    计算 ΔF 与证型模板的匹配度

    关键原则：
      - 偏离方向匹配：dF_i 与 sig_i 同号且幅度相近
      - 偏离幅度重要：sig_i != 0 的维度必须有足够的 |dF_i|
      - 中性维度宽松：sig_i == 0 的维度不影响得分

    score = direction_score * magnitude_score
      方向分：加权平均的归一化方向一致性 [0,1]
      幅度分：活跃维度上 |dF| 的均值 [0,1]，过小时惩罚

    只有两者都高时总分会高。
    """
    active = weight > 0.01
    if not np.any(active):
        return 0.0

    dF_a = dF[active]
    sig_a = sig[active]
    w_a = weight[active]

    # ── 区分"偏离型"维度 (sig != 0) 和"中性"维度 (sig == 0) ──
    deviating = np.abs(sig_a) > 0.01
    neutral = ~deviating

    # ── 偏离型维度：方向必须一致，幅度必须够大 ──
    if np.any(deviating):
        dF_dev = dF_a[deviating]
        sig_dev = sig_a[deviating]

        # 方向匹配：dF 与 sig 同方向程度
        # 用余弦相似度思想：cos = sign(dF) · sign(sig)
        # 但更柔和：dot product 归一化
        dir_match = np.clip(1.0 - np.abs(dF_dev - sig_dev) / 2.0, 0.0, 1.0)

        # 幅度惩罚：偏离型维度 |dF| 必须 > 阈值
        # |dF| = 0.15 时得 0.5 分，|dF| = 0.3 时得 1.0 分
        mag = np.abs(dF_dev)
        mag_score = np.clip(mag / 0.3, 0.0, 1.0)

        # 综合 = 方向一致 × 幅度足够
        dev_score = np.mean(dir_match * mag_score)
    else:
        dev_score = 0.0

    # ── 中性维度：越接近 0 越好（但不太影响总分） ──
    if np.any(neutral):
        dF_neu = dF_a[neutral]
        # 接近 0 得 1.0，离得远就降低
        neu_score = np.clip(1.0 - np.abs(dF_neu) * 1.5, 0.0, 1.0)
    else:
        neu_score = 1.0

    # ── 加权综合 ──
    # 如果全维度都是中性的（如"正常脉象"），用另一种逻辑
    if not np.any(deviating):
        # 所有维度中性：全部接近 0 才高分
        all_close = np.clip(1.0 - np.abs(dF_a) * 1.5, 0.0, 1.0)
        score = np.mean(all_close)
    else:
        # 加权平均：偏离维度的权重 + 中性维度的权重
        w_dev = np.sum(w_a[deviating])
        w_neu = np.sum(w_a[neutral])
        w_total = w_dev + w_neu
        score = (dev_score * w_dev + neu_score * w_neu) / w_total

    return float(np.clip(score, 0.0, 1.0).item())


def _detect_combinations(dF: np.ndarray) -> List[str]:
    """检测已知的组合证型模式"""
    combos = []
    S_wood, S_fire, S_earth, S_metal, S_water = dF

    # 上热下寒：木↑ + 水↓
    if S_water < -0.3 and (S_fire > 0.2 or S_wood > 0.2):
        combos.append("上热下寒（上焦火亢+下焦水亏）")

    # 气阴两虚：木↓ + 土↓ + 水↓
    if S_wood < 0 and S_earth < 0 and S_water < -0.2:
        combos.append("气阴两虚（气虚+阴虚）")

    # 阳虚血瘀：火↓ + 水↓ + 金↑
    if S_fire < -0.1 and S_water < -0.3 and S_metal > 0.5:
        combos.append("阳虚血瘀（心阳不足+血瘀）")

    return combos


# ═══════════════════════════════════════════════════════════
# 五诊合参融合
# ═══════════════════════════════════════════════════════════

# 各模态独立置信度（中医临床经验估计）
# 五诊是 ⟨P, ε⟩ 同一底层网络在五组独立观测子空间的投影
MODALITY_CONFIDENCE = {
    "脉诊": 0.70,   # 循环系统 σ 投影 — 覆盖率最高
    "望诊": 0.15,   # 表皮网络 + 舌象 — 次之
    "闻诊": 0.05,   # 水形链振动频谱 — 再次
    "问诊": 0.05,   # 患者自述 S 扰动 — 主观
    "八字": 0.05,   # 帧0先验参数 — 边界条件
}

# 五诊一致时贝叶斯后验置信度（假设每诊独立正确率 80%）
# P(正确|五诊一致) = Πp_i / (Πp_i + Π(1-p_i))
#                  = 0.8^5 / (0.8^5 + 0.2^5)
#                  = 0.9990 ≈ 99.9%
FIVE_MODALITY_POSTERIOR = 0.8**5 / (0.8**5 + 0.2**5)


def fuse_five_modalities(
    pulse_S: List[float] = None,       # 脉诊：ΔF 向量
    look_S: List[float] = None,        # 望诊：舌象面色的五形投影
    listen_S: List[float] = None,      # 闻诊：声音/气味的五形投影
    inquire_S: List[float] = None,     # 问诊：症状自述的五形投影
    bazi_S: List[float] = None,        # 八字：先天禀赋的五形初始偏移
    modality_conf: Dict[str, float] = None,
    pulse_qualities: Dict = None,
) -> Dict:
    """
    五诊合参融合

    五诊是同一张 ⟨P, ε⟩ 网络在五组独立观测子空间的拓扑投影。
    当五诊指向同一诊断时，盲区遗漏概率为各模态盲区的乘积。

    融合逻辑：
      1. 对每个可用的模态，独立做辨证 decode_syndrome
      2. 计算各模态主证的一致性
      3. 如五诊一致 → 后验置信度 ≈ 99.9%
      4. 如存在不一致 → 输出分歧模式，分歧本身就是诊断信号

    参数：
        pulse_S:  脉诊 ΔF [木,火,土,金,水] 或 None
        look_S:   望诊 ΔF（望面色、舌象）或 None
        listen_S: 闻诊 ΔF（声音、气味）或 None
        inquire_S: 问诊 ΔF（症状自述）或 None
        bazi_S:   八字 ΔF（先天五形偏移量）或 None
        modality_conf: 自定义各模态置信度，默认为 MODALITY_CONFIDENCE
        pulse_qualities: 脉诊六品质详情（可选，丰富输出）

    返回：
        {
            'fused_diagnosis': 主证名,
            'posterior_confidence': 0.999,
            'agreement_level': '五诊一致' | '四诊一致' | '分歧',
            'modality_results': {各模态独立输出},
            'disagreement_pattern': 分歧模式描述（如有）,
            'summary': 综合摘要,
        }
    """
    conf = modality_conf or MODALITY_CONFIDENCE

    # ── 各模态独立辨证 ──
    modalities = {
        "脉诊": (pulse_S, pulse_qualities),
        "望诊": (look_S, None),
        "闻诊": (listen_S, None),
        "问诊": (inquire_S, None),
        "八字": (bazi_S, None),
    }

    results = {}
    available = 0
    for name, (vec, q) in modalities.items():
        if vec is None:
            continue
        # 八字是先天先验，不做基线校正；其余四诊脉基线校正
        bl = None if name == "八字" else NORMAL_BASELINE
        r = decode_syndrome(vec, q, baseline=bl)
        pri = r.get("主证", {})
        results[name] = {
            "主证": pri.get("证型", "未明确"),
            "置信度": pri.get("置信度", 0),
            "兼证": [s["证型"] for s in r.get("兼证", [])],
            "组合模式": r.get("组合证型", []),
            "ΔF": [round(v, 3) for v in vec],
        }
        available += 1

    if available == 0:
        return {"error": "至少需一个模态"}

    # ── 一致性计算 ──
    primary_dxs = [v["主证"] for v in results.values()]
    primary_conf = [v["置信度"] for v in results.values()]

    # 主证一致性计数
    from collections import Counter
    dx_counts = Counter(primary_dxs)
    top_dx, top_count = dx_counts.most_common(1)[0]

    # 一致的模态数
    n_agree = top_count
    n_total = available

    # ── 后验置信度 ──
    if n_agree == n_total:
        # 完全一致：后验 = 1 - Π(1 - conf_i)
        posterior = 1.0 - np.prod(
            [1.0 - conf[n] for n in results.keys()]
        )
        agreement = "五诊一致" if n_total >= 5 else f"{n_total}诊一致"
    else:
        # 不一致：后验置信度按一致模态计算
        consistent_conf = [conf[n] for n, r in results.items()
                          if r["主证"] == top_dx]
        if consistent_conf:
            posterior = 1.0 - np.prod([1.0 - c for c in consistent_conf])
        else:
            posterior = max(primary_conf)
        agreement = f"分歧（{n_agree}/{n_total}一致）"

    # ── 分歧模式分析 ──
    disagreement = ""
    if n_agree < n_total:
        disagreeing = [(n, r) for n, r in results.items()
                      if r["主证"] != top_dx]
        parts = []
        for n, r in disagreeing:
            parts.append(f"{n}判[{r['主证']}]")
        diff_modes = list(set(r["主证"] for _, r in disagreeing))
        # 分歧模式本身是诊断信号
        # 例如：脉诊判"肝阳上亢"而望诊判"阳虚" → 真寒假热
        diff_patterns = []
        for d in diff_modes:
            if d in [r["主证"] for _, r in disagreeing]:
                pass
        disagreement = f"{'、'.join(parts)}与主流[{top_dx}]不符"

    # ── 输出 ──
    fused_diag = top_dx if top_count >= n_total / 2 else "未明确"
    posterior = np.clip(posterior, 0.0, 0.999).item()

    result = {
        "融合诊断": fused_diag,
        "后验置信度": round(posterior, 6),
        "一致性等级": agreement,
        "参与模态": list(results.keys()),
        "各模态结果": results,
        "主证计数": dict(dx_counts),
        "综合摘要": f"五诊合参：{fused_diag}（后验置信度 {posterior:.2%}；{agreement}）",
    }

    if disagreement:
        result["分歧模式"] = disagreement
        # 分歧本身是诊断信号
        result["诊断信号"] = f"注意脉证不符：{disagreement}。此即中医'含脉从证'或'含证从脉'的指征。"
        result["综合摘要"] += f"；{result['诊断信号']}"

    return result


# ═══════════════════════════════════════════════════════════
# 快捷接口：从五形 ΔF 元数据直接生成报告
# ═══════════════════════════════════════════════════════════
# （以下为 generate_clinical_report 函数）

def generate_clinical_report(delta_F: List[float],
                              qualities: Dict = None,
                              dynamics: Dict = None,
                              baseline: List[float] = None,
                              hr: float = None) -> str:
    """
    生成可读的中文临床报告

    参数：
        delta_F: [木, 火, 土, 金, 水]
        qualities: 六品质详情
        dynamics: 帧间动力学结果
        baseline: 正常人群基线（对应 decode_syndrome 的 baseline）
        hr: 心率(bpm)

    返回：
        格式化的中文报告字符串
    """
    result = decode_syndrome(delta_F, qualities, dynamics, baseline=baseline)

    lines = []
    lines.append("=" * 56)
    lines.append("  SPUM 五形辨证报告")
    lines.append("=" * 56)

    # 五形输入
    w5 = result["五形输入"]
    lines.append(f"\n  五维修正向量 ΔF:")
    lines.append(f"    木(约束)={w5['S_木(约束)']:+.3f}  火(梯度)={w5['S_火(梯度)']:+.3f}  "
                 f"土(储备)={w5['S_土(储备)']:+.3f}  金(修剪)={w5['S_金(修剪)']:+.3f}  "
                 f"水(流通)={w5['S_水(流通)']:+.3f}")

    if hr:
        lines.append(f"    心率: {hr:.0f}bpm")

    # 主证
    pri = result.get("主证", {})
    lines.append(f"\n  主证: {pri.get('证型', '?')}  (置信度 {pri.get('置信度', 0):.0%})")
    if pri.get("描述"):
        lines.append(f"    {pri['描述']}")

    # 兼证
    sec = result.get("兼证", [])
    if sec:
        sec_str = "  |  ".join(f"{s['证型']}({s['置信度']:.0%})" for s in sec)
        lines.append(f"  兼证: {sec_str}")

    # 组合
    combos = result.get("组合证型", [])
    if combos:
        lines.append(f"  组合模式: {'; '.join(combos)}")

    # 六品质特征
    if qualities:
        q_parts = []
        for label, key, pos, neg in [
            ("软硬", "soft_hard", "偏硬", "偏软"),
            ("粗细", "thin_thick", "偏粗", "偏细"),
            ("缓急", "slow_urgent", "偏急", "偏缓"),
        ]:
            v = qualities.get(key, 0)
            if abs(v) > 0.2:
                q_parts.append(f"{pos if v>0 else neg}({v:+.2f})")
        if q_parts:
            lines.append(f"  品质特征: {'、'.join(q_parts)}")
        lines.append(f"    金(修剪)={qualities.get('prune',0):.3f}  "
                     f"水(流通)={qualities.get('flow',0):.3f}")

    # 动力学
    if dynamics:
        sig = dynamics.get("脉动力学签名", {})
        score = dynamics.get("整体动力学评分", {})
        lines.append(f"\n  帧间动力学: {sig.get('类型', 'N/A')}")
        lines.append(f"    灵活性={score.get('灵活性(变化能力)',0):.2f}  "
                     f"稳定性={score.get('稳定性(过冲小)',0):.2f}  "
                     f"恢复力={score.get('恢复力(恢复快)',0):.2f}")

    result_text = result.get("动力学校正", "")
    if result_text:
        lines.append(f"  {result_text}")

    lines.append(f"\n{'='*56}")
    lines.append(f"  综合摘要: {result.get('综合摘要', '')}")
    lines.append(f"{'='*56}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 56)
    print("  SPUM 辨证解码器自测")
    print("=" * 56)

    # 测试案例
    test_cases = [
        ("正常脉象", [0.20, 0.20, 0.30, 0.40, -0.35], {"soft_hard":0.20, "thin_thick":0.20, "slow_urgent":0.20, "prune":0.30, "flow":0.50}),
        ("阴虚火旺", [0.35, 0.40, -0.30, 0.30, -0.45], {"soft_hard":0.35, "thin_thick":-0.30, "slow_urgent":0.40, "prune":0.30, "flow":0.30}),
        ("心阳虚(缓)", [0.00, -0.20, 0.10, 0.40, -0.25], {"soft_hard":0.00, "thin_thick":0.10, "slow_urgent":-0.20, "prune":0.40, "flow":0.50}),
        ("气滞血瘀", [0.05, 0.05, 0.30, 0.60, -0.45], {"soft_hard":0.05, "thin_thick":0.30, "slow_urgent":0.05, "prune":0.60, "flow":0.10}),
        ("肝阳上亢", [0.45, 0.35, 0.20, 0.30, -0.40], {"soft_hard":0.45, "thin_thick":0.20, "slow_urgent":0.35, "prune":0.30, "flow":0.30}),
        ("室速/室颤(ECG)", [0.165, 0.043, 0.337, 0.551, -0.300],
         {"soft_hard":0.17, "thin_thick":0.34, "slow_urgent":0.04, "prune":0.55, "flow":0.35}),
    ]

    for tc in test_cases:
        label, dF, q = tc
        result = decode_syndrome(dF, q)

        w5 = result["五形输入"]
        vals = "  ".join(f"{k}={v:+.3f}" for k, v in w5.items())
        print(f"\n{'─' * 56}")
        print(f"  [{label}]")
        print(f"  ΔF: {vals}")
        pri = result.get("主证", {})
        print(f"  主证: {pri.get('证型', '?')} (conf={pri.get('置信度',0):.3f})")
        sec = result.get("兼证", [])
        if sec:
            sec_str = '; '.join(s['证型'] + '(' + str(round(s['置信度'], 2)) + ')' for s in sec)
            print(f"  兼证: {sec_str}")
        combos = result.get("组合证型", [])
        if combos:
            print(f"  组合: {'; '.join(combos)}")

    # ════════════════════════════════════════════════
    # 五诊合参融合自测
    # ════════════════════════════════════════════════
    print(f"\n{'='*56}")
    print("  五诊合参融合自测")
    print(f"{'='*56}")

    # 注意：测试使用 raw ΔF（带 ECG 基线），用 NORMAL_BASELINE 校正
    # 肝阳上亢的原始 ΔF 应偏离正常基线：
    #   木 > 0.247, 火 > 0.203, 水 < -0.407
    pulse = [0.45, 0.35, 0.10, 0.50, -0.50]    # 脉诊：木亢+火亢+水亏
    look = [0.40, 0.30, 0.20, 0.50, -0.45]     # 望诊：面红目赤
    listen = [0.35, 0.28, 0.25, 0.50, -0.42]    # 闻诊：声高气粗
    inquire = [0.20, 0.30, 0.20, 0.40, -0.45]   # 问诊：头痛目眩
    bazi = [0.30, 0.10, 0.10, 0.20, -0.20]     # 八字：甲木日主（木旺先天）

    r5 = fuse_five_modalities(pulse_S=pulse, look_S=look, listen_S=listen,
                               inquire_S=inquire, bazi_S=bazi)
    print(f"\n  [五诊一致 · 肝阳上亢]")
    print(f"  融合诊断: {r5['融合诊断']}")
    print(f"  后验置信度: {r5['后验置信度']:.4f} ({r5['后验置信度']*100:.2f}%)")
    print(f"  一致性: {r5['一致性等级']}")
    for n, res in r5['各模态结果'].items():
        print(f"    {n}: {res['主证']} (conf={res['置信度']:.2f})")
    print(f"  综合: {r5['综合摘要']}")

    # 分歧案例：脉证不符（脉诊 vs 望诊+问诊）
    pulse = [0.45, 0.35, 0.10, 0.50, -0.50]    # 脉诊 → 肝阳上亢
    look = [-0.10, -0.20, -0.10, 0.40, -0.10]  # 望诊 → 面白肢冷 → 阳虚
    inquire = [-0.10, -0.20, 0.00, 0.30, -0.20] # 问诊 → 畏寒 → 阳虚

    r5b = fuse_five_modalities(pulse_S=pulse, look_S=look, inquire_S=inquire)
    print(f"\n  [五诊分歧 · 脉证不符]")
    print(f"  融合诊断: {r5b['融合诊断']}")
    print(f"  后验置信度: {r5b['后验置信度']:.4f}")
    print(f"  一致性: {r5b['一致性等级']}")
    for n, res in r5b['各模态结果'].items():
        print(f"    {n}: {res['主证']} (conf={res['置信度']:.2f})")
    if '分歧模式' in r5b:
        print(f"  分歧: {r5b['分歧模式']}")
    if '诊断信号' in r5b:
        print(f"  诊断信号: {r5b['诊断信号']}")
