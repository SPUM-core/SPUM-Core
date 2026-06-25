#!/usr/bin/env python3
"""
SPUM 五形脉诊算法 v3.0 — 几何五形 + 时序趋势耦合
=================================================================
单路 ECG 信号 → 五段几何拆解 → 五形向量 → 趋势耦合 → 辨证诊断。

  Layer 1（静态几何）：单波峰五段拆解 → 五形即时快照
    上升爬坡段 = 木：斜率+毛刺 → 肝胆疏泄
    峰值最高点 = 火：幅值+宽度 → 心气阳热
    峰顶平稳段 = 土：平台长度 → 脾胃中气
    下坡回落段 = 金：衰减快慢 → 肺气收敛
    低谷回归基线 = 水：谷底深度+静息 → 肾封藏

  Layer 2（时序趋势）：连续多波峰序列 → 趋势修正向量
    各维趋势斜率/波动率 → 区分一过性扰动(外因) vs 本虚(脏腑)

  耦合规则：
    单峰失衡 + 多峰趋势平稳 = 一过性外因扰动，短期调治
    单峰失衡 + 多峰趋势同向偏移 = 脏腑长期本虚，固本为主

输入：单路 ECG 信号 ≥30s @ ≥250Hz
输出：双层五形向量 + 趋势判定 + 辨证诊断

依赖：numpy scipy
安装：pip install numpy scipy

用法：
  python pulse_diagnosis.py --self-test

版本：v3.0 / 2026-06-25
"""

import numpy as np
from scipy import signal
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

from syndrome_decoder import decode_syndrome, NORMAL_BASELINE


# ============================================================
# Layer 1：信号预处理（同 v2.0）
# ============================================================

def design_bandpass(lowcut: float = 0.5, highcut: float = 45.0,
                    fs: float = 360.0, order: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a


def preprocess_ecg(ecg: np.ndarray, fs: float = 360.0) -> np.ndarray:
    b_bp, a_bp = design_bandpass(0.5, 45.0, fs, 4)
    filtered = signal.filtfilt(b_bp, a_bp, ecg)
    if fs >= 100:
        try:
            w0 = 50.0 / (fs / 2.0)
            b_n, a_n = signal.iirnotch(w0, 30.0)
            filtered = signal.filtfilt(b_n, a_n, filtered)
        except Exception:
            pass
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    return filtered


def compute_sqi(ecg: np.ndarray, fs: float, r_peaks: np.ndarray) -> float:
    noise = ecg - signal.medfilt(ecg, kernel_size=int(fs * 0.1) + 1)
    snr = 10 * np.log10(np.var(ecg) / (np.var(noise) + 1e-8))
    sqi_snr = min(snr / 20.0, 1.0)
    duration_sec = len(ecg) / fs
    expected_beats = duration_sec * 1.2
    detection_rate = len(r_peaks) / max(expected_beats, 1)
    sqi_rate = min(detection_rate / 0.95, 1.0)
    if len(r_peaks) > 3:
        rri = np.diff(r_peaks)
        rri_ok = np.sum((rri >= 0.3 * fs) & (rri <= 1.5 * fs))
        art_ratio = 1.0 - rri_ok / max(len(rri), 1)
    else:
        art_ratio = 0.5
    sqi_art = max(1.0 - art_ratio * 1.5, 0.0)
    sqi = 0.25 * sqi_snr + 0.40 * sqi_rate + 0.35 * sqi_art
    return float(np.clip(sqi, 0.0, 1.0))


def detect_r_peaks(ecg: np.ndarray, fs: float = 360.0) -> np.ndarray:
    b, a = design_bandpass(5.0, 15.0, fs, 2)
    filtered = signal.filtfilt(b, a, ecg)
    diff = np.diff(filtered)
    diff = np.pad(diff, (1, 0), mode='constant')
    squared = diff ** 2
    window = int(0.12 * fs) | 1
    integrated = signal.convolve(squared, np.ones(window) / window, mode='same')
    refractory = int(0.2 * fs)
    peaks = signal.find_peaks(integrated, distance=refractory,
                              height=np.median(integrated) * 6.0)[0]
    if len(peaks) == 0:
        abs_sig = np.abs(ecg)
        peaks = signal.find_peaks(abs_sig, distance=refractory,
                                  height=np.percentile(abs_sig, 95))[0]
    refined = []
    for p in peaks:
        s = max(0, p - int(0.04 * fs))
        e = min(len(ecg), p + int(0.04 * fs))
        refined.append(s + np.argmax(np.abs(ecg[s:e])))
    return np.array(sorted(set(refined)), dtype=np.int64)


# ============================================================
# Layer 2：波峰提取
# ============================================================

def extract_raw_beats(ecg: np.ndarray, r_peaks: np.ndarray,
                      fs: float) -> list:
    """提取每个 R 峰周围的原始 beat 段（供几何五形分析使用）"""
    half_before = int(0.25 * fs)
    half_after = int(0.45 * fs)
    beats = []
    for r in r_peaks:
        start = r - half_before
        end = r + half_after
        if start < 0 or end > len(ecg):
            continue
        beat = ecg[start:end] - ecg[start]
        beats.append(beat)
    return beats


# ═══════════════════════════════════════════════════════════
# Layer 3b：单波几何五段 → 五形静态向量（五段几何法）
# ═══════════════════════════════════════════════════════════
# 单心跳主峰曲线天然五段几何结构：
#   上升爬坡段 = 木：斜率、微小震荡 → 肝胆疏泄强弱
#   峰值最高点 = 火：电压幅值、顶点宽度 → 心气、阳热势能
#   峰顶平稳段 = 土：平台持续长度 → 脾胃中气承载能力
#   下坡回落段 = 金：衰减快慢 → 肺气收敛功能
#   低谷回归基线 = 水：谷底深度、静息时长 → 肾封藏之本
#
# 输入：单波 beat = ecg[start:end]（~0.7s @ 360Hz ≈ 252 点）
#      beat[0] = 基线，R 峰位于 int(0.25 * fs) ≈ 90 点处
# ═══════════════════════════════════════════════════════════

def _segment_beat(beat: np.ndarray, fs: float) -> Dict:
    """
    将单波拆分为五段几何关键点。

    返回：
        {
            'r_peak_idx': int,       # R 峰位置（相对 beat 起始）
            'r_amp': float,           # R 峰幅值
            'rise': slice,            # 上升段：起点→R 峰
            'apex': slice,            # 峰顶：R 峰周围 ±0.02s
            'plateau': slice,         # 平台段：峰顶后→T 波前
            'fall': slice,            # 回落段：T 波下降支
            'trough': slice,          # 低谷段：T 波结束→beat 末尾
            'baseline': float,        # 基线值（beat[0]）
            't_rise_peak': float,     # 上升段时间(秒)
            't_peak_end': float,      # 从峰顶到回落结束时间(秒)
        }
    """
    n = len(beat)
    half_before = int(0.25 * fs)  # R 峰在 beat 中的位置
    r_peak_idx = half_before
    r_amp = beat[r_peak_idx]

    # 上升段：起点到 R 峰
    rise = slice(0, r_peak_idx)

    # 峰顶：R 峰周围 ±0.02s
    apex_half = max(1, int(0.02 * fs))
    apex = slice(max(0, r_peak_idx - apex_half),
                 min(n, r_peak_idx + apex_half + 1))

    # 平台段：峰顶后 0.02s → 0.12s（ST 段 + T 波起始）
    plateau_start = min(n, r_peak_idx + apex_half + 1)
    plateau_end = min(n, r_peak_idx + int(0.12 * fs))
    plateau = slice(plateau_start, plateau_end)

    # 回落段：平台后 → 0.30s（T 波下降支）
    fall_start = plateau_end
    fall_end = min(n, r_peak_idx + int(0.30 * fs))
    fall = slice(fall_start, fall_end)

    # 低谷段：回落后到末尾（TP 段）
    trough = slice(fall_end, n)

    return {
        'r_peak_idx': r_peak_idx,
        'r_amp': r_amp,
        'rise': rise,
        'apex': apex,
        'plateau': plateau,
        'fall': fall,
        'trough': trough,
        'baseline': float(beat[0]),
        't_rise_peak': r_peak_idx / fs,
        't_peak_end': (fall_end - r_peak_idx) / fs,
    }


def extract_single_beat_wuxing(beat: np.ndarray, fs: float) -> Dict[str, float]:
    """
    单波几何五段拆分 → 五形基础向量（Layer 1 静态快照）。

    返回：
        {
            'S_木': float,   # 上升段：斜率 + 毛刺度 → 肝胆疏泄
            'S_火': float,   # 峰顶区：幅值 + 宽度 → 心气阳热
            'S_土': float,   # 平台段：持续长度 + 起伏 → 中气承载
            'S_金': float,   # 回落段：衰减率 → 肺气收敛
            'S_水': float,   # 低谷段：谷底深度 + 静息时长 → 肾封藏
        }
    """
    seg = _segment_beat(beat, fs)
    r_amp = seg['r_amp']

    # ─── 木：上升段 — 爬坡斜率 + 微小震荡 —──
    rise_seg = beat[seg['rise']]
    if len(rise_seg) > 3:
        # 上升斜率（归一化）
        rise_t = seg['t_rise_peak']
        rise_slope = r_amp / (rise_t + 1e-10)
        slope_norm = np.clip((rise_slope - 200) / 300, -1.0, 1.0)
        # 微小震荡（上升段加速度方差 — 毛刺度）
        grad1_rise = np.gradient(rise_seg) * fs
        grad2_rise = np.gradient(grad1_rise) * fs
        micro_spike = np.std(grad2_rise) / (np.std(grad1_rise) + 1e-10)
        spike_norm = np.clip(micro_spike / 15.0, 0.0, 1.0)
        # 木 = 斜率正相关 + 毛刺负相关（毛刺多则疏泄不畅）
        S_wood = np.clip(0.6 * slope_norm - 0.4 * spike_norm, -1.0, 1.0)
    else:
        S_wood = 0.0

    # ─── 火：峰顶区 — 幅值 + 顶点宽度 —──
    apex_seg = beat[seg['apex']]
    # 幅值归一化
    amp_norm = np.clip((r_amp - 0.8) / 1.5, -1.0, 1.0)
    # 顶点宽度（95% 峰值处的宽度）
    peak_95 = r_amp * 0.95
    above_95 = np.where(apex_seg >= peak_95)[0]
    if len(above_95) >= 2:
        apex_width = (above_95[-1] - above_95[0]) / fs
    else:
        apex_width = 0.01
    width_norm = np.clip((0.04 - apex_width) / 0.03, -1.0, 1.0)
    # 火 = 幅值 + 宽度组合
    S_fire = np.clip(0.5 * amp_norm + 0.5 * width_norm, -1.0, 1.0)

    # ─── 土：平台段 + 搏幅 + 面积（中气承载能力）───
    # 粗脉→高幅+大面积→中气壅滞；细脉→低幅+小面积→中气不足
    plat_seg = beat[seg['plateau']]
    if len(plat_seg) > 3:
        # 平台段长度（ST 段持续时长）
        plat_len = len(plat_seg) / fs
        plat_len_norm = np.clip((plat_len - 0.04) / 0.06, -1.0, 1.0)
        # 平台起伏度（标准差/均值 — 越平坦土越稳）
        plat_std = np.std(plat_seg) / (np.mean(np.abs(plat_seg)) + 1e-10)
        plat_flat = np.clip(1.0 - plat_std * 3.0, 0.0, 1.0)
        # 搏幅因子（粗脉的关键：R 峰-谷底的全振幅）
        trough_val = np.min(beat[seg['trough']]) if len(beat[seg['trough']]) > 0 else seg['baseline']
        peak_to_peak = r_amp - trough_val
        amp_factor = np.clip((peak_to_peak - 1.0) / 1.5, -1.0, 1.0)
        # 收缩面积（从 onset 到回落结束的曲线下面积）
        sys_end = min(len(beat), seg['fall'].stop)
        area_sys = np.sum(np.abs(beat[:sys_end] - seg['baseline'])) / fs
        area_factor = np.clip((area_sys - 0.15) / 0.2, -1.0, 1.0)
        # 土 = 平台(0.2) + 搏幅(0.55) + 面积(0.25)
        S_earth = np.clip(
            0.20 * plat_len_norm +
            0.10 * (plat_flat * 2 - 1) +
            0.45 * amp_factor +
            0.25 * area_factor,
            -1.0, 1.0
        )
    else:
        S_earth = 0.0

    # ─── 金：回落段 — 衰减快慢 —──
    fall_seg = beat[seg['fall']]
    if len(fall_seg) > 3:
        # 回落段下降率（负斜率绝对值）
        fall_amp_range = fall_seg[0] - fall_seg[-1]
        fall_t = len(fall_seg) / fs
        fall_rate = fall_amp_range / (fall_t + 1e-10)
        rate_norm = np.clip((fall_rate - 100) / 200, -1.0, 1.0)
        # 拖尾长度（从峰值 50% 到 10% 的时间）
        half_peak = r_amp * 0.5
        seg_50pct = np.where(fall_seg >= half_peak)[0]
        ten_pct = r_amp * 0.1
        seg_10pct = np.where(fall_seg >= ten_pct)[0]
        if len(seg_50pct) > 0 and len(seg_10pct) > 0:
            tail_t = (seg_10pct[-1] - seg_50pct[0]) / fs
        else:
            tail_t = 0.1
        tail_norm = np.clip((0.08 - tail_t) / 0.06, -1.0, 1.0)
        # 金 = 衰减率（快=收敛强）+ 拖尾短
        S_metal = np.clip(0.5 * rate_norm + 0.5 * tail_norm, -1.0, 1.0)
    else:
        S_metal = 0.0

    # ─── 水：低谷段 — 谷底深度 + 静息时长 —──
    trough_seg = beat[seg['trough']]
    if len(trough_seg) > 3:
        # 谷底深度（最低点相对基线）
        trough_min = np.min(trough_seg) - seg['baseline']
        trough_norm = np.clip((-trough_min - 0.05) / 0.15, -1.0, 1.0)
        # 静息时长（低谷段保持接近基线的时间）
        near_baseline = np.where(np.abs(trough_seg - seg['baseline']) < r_amp * 0.05)[0]
        rest_len = len(near_baseline) / fs
        rest_norm = np.clip((rest_len - 0.08) / 0.10, -1.0, 1.0)
        S_water = np.clip(0.5 * trough_norm + 0.5 * rest_norm, -1.0, 1.0)
    else:
        S_water = 0.0

    return {
        'S_木': float(np.clip(S_wood, -1.0, 1.0)),
        'S_火': float(np.clip(S_fire, -1.0, 1.0)),
        'S_土': float(np.clip(S_earth, -1.0, 1.0)),
        'S_金': float(np.clip(S_metal, -1.0, 1.0)),
        'S_水': float(np.clip(S_water, -1.0, 1.0)),
    }


# ═══════════════════════════════════════════════════════════
# Layer 2a：多波时序差分 → 趋势修正向量
# ═══════════════════════════════════════════════════════════
# 截取连续心跳序列，提取多组单峰五形向量，做时序差分运算。
# 判断失衡是临时扰动还是一过性。
#
# 趋势判定规则：
#   木趋势：爬坡斜率持续忽陡忽缓、毛刺增多 → 木气郁滞
#           多峰斜率稳定平缓 → 木本亏虚
#   火趋势：峰值高低反复大幅浮动 → 心火时亢时衰
#           全段峰值持续偏低 → 心气耗损
#   土趋势：平台长度交替变化 → 中气升降失常
#           全段无平缓平台 → 中长期中气下陷
#   金趋势：回落速度逐峰变慢 → 肺气持续耗散
#           每一波回落都急促陡峭 → 收敛太过
#   水趋势：谷底持续变浅、静息缩短 → 肾精耗散
#           谷底规律性周期波动 → 肾阴阳失衡
# ═══════════════════════════════════════════════════════════

# 五形名称常量
WUXING_KEYS = ['S_木', 'S_火', 'S_土', 'S_金', 'S_水']



def compute_wuxing_trends(wuxing_list: List[Dict]) -> Dict:
    """
    多波五形序列 → 每行的趋势修正向量（Layer 2）。

    参数：
        wuxing_list: [{'S_木':..., 'S_火':..., ...}, ...] 连续多搏
                     长度不够 5 搏时返回全零趋势。

    返回：
        {
            'trend':        {'S_木': float, ...},  # 趋势斜率(每搏)
            'fluctuation':  {'S_木': float, ...},  # 波动率(标准差)
            'mean':         {'S_木': float, ...},  # 均值
            'trend_class':  {key: '平/升/降/振荡/亏虚'},  # 趋势定性
            'n_beats':      int,                    # 参与计算搏数
        }
    """
    n = len(wuxing_list)
    if n < 5:
        # 数据不足，返回零向量（一致使用 ndarray 类型）
        return {
            'trend': np.zeros(5),
            'fluctuation': np.zeros(5),
            'mean': np.zeros(5),
            'trend_class': {k: '数据不足' for k in WUXING_KEYS},
            'n_beats': n,
        }

    result = {}
    x = np.arange(n)

    for k in WUXING_KEYS:
        vals = np.array([b[k] for b in wuxing_list], dtype=float)

        # 均值（基线水平）
        mean_val = float(np.mean(vals))

        # 波动率（标准差 — 反映振荡幅度）
        fluc = float(np.std(vals))

        # 趋势斜率（线性回归 — 反映方向性偏移）
        if np.std(x) > 0 and np.std(vals) > 1e-10:
            slope = float(np.polyfit(x, vals, 1)[0])
        else:
            slope = 0.0

        # ─── 趋势定性分类 ───
        abs_slope = abs(slope)
        if abs_slope < 0.015:
            if fluc < 0.05:
                trend_class = '平'  # 稳定
            else:
                trend_class = '振荡'  # 波动大但无方向
        elif slope > 0:
            trend_class = '升'
        else:
            trend_class = '降'

        # 升/降的幅度级别
        if abs_slope >= 0.05 and fluc < abs_slope * 1.5:
            pass  # 强趋势

        result[k] = {
            '均值': round(mean_val, 3),
            '趋势斜率': round(slope, 4),
            '波动率': round(fluc, 3),
            '趋势分类': trend_class,
        }

    # ─── 综合趋势向量（提取用于耦合计算的数值向量）───
    trend_vec = np.array([result[k]['趋势斜率'] for k in WUXING_KEYS])
    fluc_vec = np.array([result[k]['波动率'] for k in WUXING_KEYS])
    mean_vec = np.array([result[k]['均值'] for k in WUXING_KEYS])

    return {
        'trend': trend_vec,
        'fluctuation': fluc_vec,
        'mean': mean_vec,
        'detail': result,
        'n_beats': n,
    }


# ═══════════════════════════════════════════════════════════
# Layer 2b：双层耦合运算逻辑
# ═══════════════════════════════════════════════════════════
# 耦合规则：
#   单峰失衡 + 多峰趋势平稳 = 一过性外因扰动
#   单峰失衡 + 多峰趋势同向偏移 = 脏腑长期本虚
#
# 输入：
#   delta_F_static: 单波/聚合五形向量 [S_木, S_火, S_土, S_金, S_水]
#   trend_result:   compute_wuxing_trends() 的返回值
#
# 输出：
#   delta_F_coupled: 耦合修正后的五形向量
#   分类：transient（一过性）/ chronic（本虚）/ mixed
# ═══════════════════════════════════════════════════════════

# 趋势显著性阈值
TREND_THRESHOLD = 0.025     # |斜率| > 此值视为有趋势
FLUC_THRESHOLD = 0.08       # 波动率 > 此值视为振荡

# 耦合融合权重
W_STATIC = 0.6   # 静态向量权重
W_TREND = 0.4    # 趋势修正权重


def couple_static_trend(
    delta_F_static: np.ndarray,
    trend_result: Dict,
) -> Dict:
    """
    双层耦合运算 → 融合五形向量 + 趋势判定。

    参数：
        delta_F_static: 五形静态向量 [S_木, S_火, S_土, S_金, S_水]
        trend_result: compute_wuxing_trends() 返回值

    返回：
        {
            'delta_F_coupled': ndarray,    # 融合向量
            'delta_F_static':  ndarray,    # 静态（原始）
            'delta_F_trend':   ndarray,    # 趋势修正量
            'classification':   str,       # 'normal'/'transient'/'chronic'/'mixed'
            'per_dim': {
                'S_木': { 'static': float, 'trend': float, 'coupled': float, 'class': '平/升/降/振荡' },
                ...
            },
            'graded_diagnosis': str,       # 综合诊断建议
        }
    """
    trend_vec = trend_result.get('trend', np.zeros(5))
    fluc_vec = trend_result.get('fluctuation', np.zeros(5))
    n_beats = trend_result.get('n_beats', 0)

    # 趋势修正向量 = 趋势斜率 × W_TREND（趋势方向的正则化）
    delta_trend = np.array([np.clip(t / 0.1, -1.0, 1.0) for t in trend_vec])

    # 融合：静态 + 趋势修正
    delta_coupled = W_STATIC * delta_F_static + W_TREND * delta_trend

    # ─── 每维分类 ───
    per_dim = {}
    chronic_dims = 0
    nz_dims = 0  # 非零静态维度数

    for i, k in enumerate(WUXING_KEYS):
        s = float(delta_F_static[i])
        t = float(trend_vec[i])
        f = float(fluc_vec[i])
        c = float(delta_coupled[i])

        # 趋势定性
        abs_t = abs(t)
        if abs_t < TREND_THRESHOLD:
            if f > FLUC_THRESHOLD:
                tc = '振荡'
            else:
                tc = '平'
        elif t > 0:
            tc = f'升({abs_t:.3f}/搏)'
        else:
            tc = f'降({abs_t:.3f}/搏)'

        # 本虚判定：静态非零 + 趋势同向偏移
        if abs(s) > 0.1 and abs_t > TREND_THRESHOLD:
            # 静态方向和趋势方向是否一致
            static_dir = 1 if s > 0 else -1
            trend_dir = 1 if t > 0 else -1
            if static_dir * trend_dir > 0:  # 同向
                chronic_dims += 1
            nz_dims += 1
        elif abs(s) > 0.1:
            nz_dims += 1

        per_dim[k] = {
            'static': round(s, 3),
            'trend': round(t, 4),
            'coupled': round(c, 3),
            'class': tc,
        }

    # ─── 全局分类 ───
    max_static = np.max(np.abs(delta_F_static))
    max_trend = np.max(np.abs(trend_vec))

    if max_static < 0.1 and max_trend < TREND_THRESHOLD:
        classification = 'normal'
        graded_diagnosis = '正常脉象，无明显失衡'
    elif max_static >= 0.1 and chronic_dims == 0 and nz_dims > 0:
        classification = 'transient'
        graded_diagnosis = '一过性失调（外因扰动为主），短期调治即可'
    elif chronic_dims >= 1 and chronic_dims >= nz_dims * 0.5:
        classification = 'chronic'
        chronic_names = [WUXING_KEYS[i] for i in range(5)
                         if abs(delta_F_static[i]) > 0.1
                         and abs(trend_vec[i]) > TREND_THRESHOLD]
        graded_diagnosis = (f'脏腑本虚（{", ".join(chronic_names)}长期偏移），'
                            f'需固本为主、分期调理')
    elif chronic_dims >= 1:
        classification = 'mixed'
        graded_diagnosis = '混合型（部分本虚 + 部分一过性扰动），对症+固本兼顾'
    elif nz_dims == 0 and max_trend > TREND_THRESHOLD:
        classification = 'latent'
        graded_diagnosis = '潜伏趋势（静态尚可但趋势已现），提前干预为佳'
    else:
        classification = 'normal'
        graded_diagnosis = '大致正常'

    return {
        'delta_F_coupled': delta_coupled,
        'delta_F_static': delta_F_static,
        'delta_F_trend': delta_trend,
        'classification': classification,
        'per_dim': per_dim,
        'graded_diagnosis': graded_diagnosis,
        'n_beats': n_beats,
        'chronic_dim_count': chronic_dims,
    }


def aggregate_wuxing_beats(wuxing_list: List[Dict]) -> np.ndarray:
    """多搏五形向量聚合（中位数聚合，抗异常搏动干扰）"""
    arr = np.array([[b[k] for k in WUXING_KEYS] for b in wuxing_list])
    return np.median(arr, axis=0)

# ============================================================
# Layer 3c：几何五形分析法
# ============================================================

def analyze_position_geometric(ecg: np.ndarray, fs: float, name: str = "",
                                return_raw_wuxing: bool = False
                                ) -> Tuple[Dict, float, float,
                                           Optional[List[Dict]],
                                           Optional[List[Dict]]]:
    """
    完整分析单路 ECG 信号的**几何五形**。

    流程：
      ECG → 预处理 → R峰检测 → 提取原始beat → 几何五段拆解
      → 每搏五形向量(Layer 1) → 多搏趋势分析(Layer 2)
      → 双层耦合 → 输出

    返回：
        (coupled_result, hr_bpm, sqi, wuxing_list, raw_beats)
          coupled_result: couple_static_trend() 的返回
          wuxing_list: 每搏五形向量列表
          raw_beats: 原始 beat 数据列表
    """
    clean = preprocess_ecg(ecg, fs)
    r_peaks = detect_r_peaks(clean, fs)
    sqi = compute_sqi(ecg, fs, r_peaks)

    if sqi < 0.5 or len(r_peaks) < 5:
        return {'delta_F_coupled': np.zeros(5), 'delta_F_static': np.zeros(5),
                'delta_F_trend': np.zeros(5), 'classification': 'insufficient',
                'graded_diagnosis': '信号质量不足或心搏太少'}, 0.0, sqi, None, None

    # 提取原始 beats（从原始信号，避免带通滤波抹平幅值差异）
    raw_beats = extract_raw_beats(ecg, r_peaks, fs)
    if len(raw_beats) < 3:
        return {'delta_F_coupled': np.zeros(5), 'delta_F_static': np.zeros(5),
                'delta_F_trend': np.zeros(5), 'classification': 'insufficient',
                'graded_diagnosis': '心搏数不足'}, 0.0, sqi, None, None

    # ── Layer 1：每搏的五形几何向量 ──
    wuxing_list = [extract_single_beat_wuxing(b, fs) for b in raw_beats]

    # 静态聚合向量（中位数聚合）
    delta_F_static = aggregate_wuxing_beats(wuxing_list)

    # ── Layer 2：多搏趋势分析 ──
    trend_result = compute_wuxing_trends(wuxing_list)

    # ── 双层耦合 ──
    coupled_result = couple_static_trend(delta_F_static, trend_result)

    # 心率
    rr_intervals = np.diff(r_peaks) / fs
    hr = 60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0

    if return_raw_wuxing:
        return coupled_result, hr, sqi, wuxing_list, raw_beats
    else:
        return coupled_result, hr, sqi, None, None


# ============================================================
# 主入口
# ============================================================

def spum_pulse_diagnosis(ecg_signal: np.ndarray,
                         fs: float = 360.0, verbose: bool = False) -> Dict:
    """
    SPUM 五形脉诊 v3.0 — 主入口（单路 ECG 信号 → 几何五形分析）

    参数：
        ecg_signal: 单路 ECG 信号，≥30s @ ≥250Hz
        fs: 采样率 (Hz)
        verbose: 是否返回原始数据

    返回：
        完整报告
    """
    # ── 几何五形分析（一次调用完成所有分析）───
    result, hr, sqi, wuxing_list, raw_beats = analyze_position_geometric(
        ecg_signal, fs, return_raw_wuxing=True)

    if sqi < 0.5:
        return {"error": f"信号质量不足 (SQI={sqi:.2f})", "sqi": sqi}

    coupled = result['delta_F_coupled']

    report = {
        "信号质量 SQI": round(sqi, 3),
        "心率(bpm)": round(hr, 1),
        "五形几何分析（双层耦合）": {
            "静态向量 (Layer 1)": {
                "S_木(上升爬坡)": round(float(result['delta_F_static'][0]), 3),
                "S_火(峰值幅值)": round(float(result['delta_F_static'][1]), 3),
                "S_土(平台承载)": round(float(result['delta_F_static'][2]), 3),
                "S_金(回落收敛)": round(float(result['delta_F_static'][3]), 3),
                "S_水(谷底静息)": round(float(result['delta_F_static'][4]), 3),
            },
            "趋势修正 (Layer 2)": {
                "趋势向量": [round(float(t), 4) for t in result['delta_F_trend']],
                "参与搏数": result['n_beats'],
            },
            "耦合融合向量 ΔF": {
                "S_木": round(float(coupled[0]), 3),
                "S_火": round(float(coupled[1]), 3),
                "S_土": round(float(coupled[2]), 3),
                "S_金": round(float(coupled[3]), 3),
                "S_水": round(float(coupled[4]), 3),
            },
            "诊断分类": result['classification'],
            "诊断建议": result['graded_diagnosis'],
            "每维详情": result['per_dim'],
        },
    }

    # ─── 辨证解码 ───
    syndrome_result = decode_syndrome(coupled, {}, {},
                                      baseline=NORMAL_BASELINE)
    report["辨证诊断"] = {
        "主证": syndrome_result.get("主证", {}).get("证型", "未明确"),
        "置信度": syndrome_result.get("主证", {}).get("置信度", 0),
        "兼证": [s["证型"] for s in syndrome_result.get("兼证", [])],
        "组合模式": syndrome_result.get("组合证型", []),
        "综合摘要": syndrome_result.get("综合摘要", ""),
    }

    if verbose:
        report["_原始五形向量序列"] = wuxing_list

    return report


# ============================================================
# 测试与验证
# ============================================================

def generate_position_ecg(duration_sec: float = 30.0, fs: float = 360.0,
                          hr_bpm: float = 72.0,
                          mode: str = 'normal') -> np.ndarray:
    """
    生成单路模拟 ECG，支持六品质调控

    mode: normal/hard/soft/thick/thin/urgent/slow
    """
    t = np.arange(0, duration_sec, 1/fs)
    rr_interval = 60.0 / hr_bpm

    if mode == 'hard':
        qrs_width = 0.04; amp_factor = 1.0; t_wave_amp = 0.2
    elif mode == 'soft':
        qrs_width = 0.12; amp_factor = 0.8; t_wave_amp = 0.4
    elif mode == 'thick':
        qrs_width = 0.08; amp_factor = 1.8; t_wave_amp = 0.5
    elif mode == 'thin':
        qrs_width = 0.08; amp_factor = 0.3; t_wave_amp = 0.15
    elif mode == 'urgent':
        qrs_width = 0.06; amp_factor = 1.2; t_wave_amp = 0.25
        rr_interval *= 0.65
    elif mode == 'slow':
        qrs_width = 0.10; amp_factor = 0.9; t_wave_amp = 0.35
        rr_interval *= 1.5
    else:
        qrs_width = 0.08; amp_factor = 1.0; t_wave_amp = 0.3

    signal_ecg = np.zeros_like(t)

    for i in range(int(duration_sec / rr_interval) + 1):
        r_time = i * rr_interval
        r_idx = int(r_time * fs)
        if r_idx >= len(t): break

        # QRS
        qrs_env = np.exp(-((t[r_idx:] - t[r_idx]) ** 2) / (2 * qrs_width ** 2))
        qrs_wave = np.sin(2 * np.pi * (t[r_idx:] - t[r_idx]) / qrs_width * 2)
        qrs_wave = qrs_wave * qrs_env

        # T 波
        t_offset = 0.25
        t_idx = r_idx + int(t_offset * fs)
        if t_idx < len(t):
            t_width = 0.12
            t_wave = t_wave_amp * np.exp(-((t[t_idx:] - t[t_idx]) ** 2) / (2 * t_width ** 2))
            end = min(len(signal_ecg), t_idx + len(t_wave))
            signal_ecg[t_idx:end] += t_wave[:end - t_idx]

        # P 波
        p_offset = -0.18
        p_idx = r_idx + int(p_offset * fs)
        if 0 <= p_idx < len(t):
            p_width = 0.06
            p_wave = 0.12 * np.exp(-((t[p_idx:] - t[p_idx]) ** 2) / (2 * p_width ** 2))
            end = min(len(signal_ecg), p_idx + len(p_wave))
            signal_ecg[p_idx:end] += p_wave[:end - p_idx]

        end = min(len(signal_ecg), r_idx + len(qrs_wave))
        signal_ecg[r_idx:end] += qrs_wave[:end - r_idx] * amp_factor

    noise = np.random.randn(len(t)) * 0.02
    baseline = 0.005 * np.sin(2 * np.pi * 0.25 * t)
    return signal_ecg + noise + baseline


def self_test():
    print("=" * 70)
    print("SPUM 五形脉诊 v3.0 — 几何五形分析 自测")
    print("=" * 70)

    tests = [
        ("1. 正常波形", 'normal', 72),
        ("2. 偏硬（高尖 QRS）", 'hard', 72),
        ("3. 偏软（低平 QRS）", 'soft', 72),
        ("4. 偏粗（高幅值）", 'thick', 72),
        ("5. 偏细（低幅值）", 'thin', 72),
        ("6. 偏急（高心率）", 'urgent', 100),
        ("7. 偏缓（低心率）", 'slow', 50),
    ]

    for test_name, mode, hr in tests:
        print(f"\n{'─' * 50}")
        ecg = generate_position_ecg(30, 360, hr, mode)
        result = spum_pulse_diagnosis(ecg, fs=360)

        if "error" in result:
            print(f"  [{test_name}] ❌ {result['error']}")
            continue

        geom = result['五形几何分析（双层耦合）']
        df = geom['耦合融合向量 ΔF']
        print(f"  [{test_name}]  SQI={result['信号质量 SQI']:.2f}  HR={result['心率(bpm)']:.0f}")
        print(f"    分类: {geom['诊断分类']}  |  {geom['诊断建议']}")
        print(f"    ΔF: 木={df['S_木']:+.3f} 火={df['S_火']:+.3f} "
              f"土={df['S_土']:+.3f} 金={df['S_金']:+.3f} 水={df['S_水']:+.3f}")

    print(f"\n{'=' * 70}")
    print("自测完成")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        import pandas as pd
        filepath = sys.argv[2]
        fs = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--fs" else 360.0
        df = pd.read_csv(filepath)
        ecg_col = [c for c in df.columns if 'ecg' in c.lower() or 'i' in c.lower()][0]
        ecg = df[ecg_col].values
        result = spum_pulse_diagnosis(ecg, fs=fs)
        import json
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        self_test()
