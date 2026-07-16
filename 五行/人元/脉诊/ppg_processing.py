#!/usr/bin/env python3
"""
PPG 信号处理模块 — 预处理、搏动检测、特征提取、SQI 评估
========================================================

实现自 SPUM-脉诊-PPG形态学算法.md §2–§3 的技术方案。

v1.1 / 2026-07-15 — 优化版：滤波系数缓存、梯度复用、SQI 加速
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Optional, Tuple, Callable


# ═══════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════
DEFAULT_FS = 250.0
BAND_LOW = 0.5
BAND_HIGH = 15.0
FILTER_ORDER = 4
MIN_BPM = 40
MAX_BPM = 200
MIN_BEAT_DIST = 0.3        # 秒
MAX_BEAT_DIST = 1.8
SQI_GOOD = 0.75
SQI_FAIR = 0.50


# ═══════════════════════════════════════════════════════════
# 滤波系数缓存（避免每帧重算）
# ═══════════════════════════════════════════════════════════

_filter_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

def _get_filter(fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """缓存 Butterworth 带通滤波器系数。"""
    if fs not in _filter_cache:
        nyq = 0.5 * fs
        _filter_cache[fs] = signal.butter(FILTER_ORDER,
                                           [BAND_LOW / nyq, BAND_HIGH / nyq],
                                           btype='band')
    return _filter_cache[fs]


# ═══════════════════════════════════════════════════════════
# 预处理
# ═══════════════════════════════════════════════════════════

def preprocess_ppg(ppg: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """预处理：带通滤波 + 基线校正 + 标准化。"""
    if len(ppg) < int(fs * 2):
        raise ValueError(f"信号过短 ({len(ppg)}), 至少 {int(fs * 2)}")

    b, a = _get_filter(fs)
    filtered = signal.filtfilt(b, a, ppg)

    # 中值滤波去基线
    bw = int(1.5 * fs) | 1
    corrected = filtered - signal.medfilt(filtered, kernel_size=bw)

    # 标准化
    return (corrected - np.mean(corrected)) / (np.std(corrected) + 1e-8)


# ═══════════════════════════════════════════════════════════
# 搏动检测
# ═══════════════════════════════════════════════════════════

def detect_peaks(ppg: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """自适应阈值 PPG 主波峰值检测。"""
    n = len(ppg)
    if n < int(fs):
        return np.array([], dtype=np.int64)

    sig_range = np.percentile(ppg, 95) - np.percentile(ppg, 5)
    min_dist = int(MIN_BEAT_DIST * fs)

    peaks, props = signal.find_peaks(
        ppg, distance=min_dist,
        height=np.percentile(ppg, 50),
        prominence=0.3 * sig_range,
        width=int(0.02 * fs),
    )

    # 峰值不足时放宽阈值重试
    expected = n / fs / (MIN_BEAT_DIST + MAX_BEAT_DIST) * 2
    if len(peaks) < max(2, int(expected * 0.3)):
        peaks, _ = signal.find_peaks(ppg, distance=min_dist,
                                      prominence=0.15 * sig_range)

    # MAD 后过滤异常峰值
    if len(peaks) > 3:
        h = ppg[peaks]
        mad = np.median(np.abs(h - np.median(h)))
        peaks = peaks[np.abs(h - np.median(h)) < 3.5 * max(mad, 0.1)]

    return peaks


def segment_beats(ppg: np.ndarray, peaks: np.ndarray,
                  fs: float = DEFAULT_FS) -> List[Dict]:
    """按搏动周期分割信号。"""
    beats = []
    for i in range(len(peaks) - 1):
        p0, p1 = peaks[i], peaks[i + 1]
        sr = max(0, p0 - int(0.2 * fs))
        onset = sr + np.argmin(ppg[sr:p0 + 1])

        end_search = min(p1, p0 + int(0.5 * fs))
        offset = p0 + np.argmin(ppg[p0:end_search]) if p1 - p0 > int(0.3 * fs) else p0 + int((p1 - p0) * 0.85)
        offset = max(offset, p0 + int(0.15 * fs))
        offset = min(offset, p1)

        sig = ppg[onset:offset + 1]
        if len(sig) >= int(0.15 * fs):
            beats.append(dict(start_idx=onset, end_idx=offset,
                              peak_idx=p0, peak_val=float(ppg[p0]),
                              signal=sig.copy()))
    return beats


# ═══════════════════════════════════════════════════════════
# 特征点检测（梯度一次计算，多处复用）
# ═══════════════════════════════════════════════════════════

def _compute_gradients(sig: np.ndarray, fs: float):
    """一次性计算一阶/二阶梯度。"""
    g1 = np.gradient(sig) * fs
    g2 = np.gradient(g1) * fs
    return g1, g2


def detect_ppg_landmarks(beat_signal: np.ndarray, fs: float = DEFAULT_FS,
                         _grad1: np.ndarray = None,
                         _grad2: np.ndarray = None) -> Dict:
    """
    检测单个 PPG 搏动周期内的关键特征点。
    可传入预计算的梯度以复用。
    """
    n = len(beat_signal)
    if n < int(0.15 * fs):
        return {'error': 'beat too short'}

    onset = 0
    peak = int(np.argmax(beat_signal))
    a1 = beat_signal[peak] - beat_signal[onset]

    # 梯度（复用或计算）
    g1 = _grad1 if _grad1 is not None else np.gradient(beat_signal) * fs
    g2 = _grad2 if _grad2 is not None else np.gradient(g1) * fs

    # 拐点：二阶导过零
    inflection = onset
    for i in range(onset + 1, peak):
        if g2[i - 1] <= 0 < g2[i]:
            inflection = i
            break

    # 重搏波切迹 + 峰值
    ss = peak + int(0.1 * fs)
    if ss >= n:
        notch, dp, a2 = peak, peak, 0.0
    else:
        zc = np.where(np.diff(np.sign(g1[ss:])) > 0)[0]
        notch = ss + int(zc[0]) if len(zc) > 0 else ss + int(np.argmin(beat_signal[ss:]))
        pr = beat_signal[notch:min(n, notch + int(0.3 * fs))]
        dp = notch + int(np.argmax(pr)) if len(pr) > 0 else notch
        a2 = beat_signal[dp] - beat_signal[notch]

    return dict(onset=onset, peak=peak, inflection=inflection,
                dicrotic_notch=notch, dicrotic_peak=dp, end=n - 1,
                A1=float(a1), A2=float(max(a2, 0)),
                A1_A2_ratio=float(a1 / max(a2, 1e-10)),
                peak_val=float(beat_signal[peak]),
                notch_val=float(beat_signal[notch]))


# ═══════════════════════════════════════════════════════════
# 单周期特征提取
# ═══════════════════════════════════════════════════════════

def extract_beat_features(beat_signal: np.ndarray, fs: float = DEFAULT_FS,
                          _grad1: np.ndarray = None,
                          _grad2: np.ndarray = None) -> Dict:
    """从单个 PPG 搏动周期提取完整特征。"""
    lm = detect_ppg_landmarks(beat_signal, fs, _grad1, _grad2)
    if 'error' in lm:
        return lm

    o, p, n_idx, e = lm['onset'], lm['peak'], lm['dicrotic_notch'], lm['end']
    g1 = _grad1 if _grad1 is not None else np.gradient(beat_signal) * fs
    g2 = _grad2 if _grad2 is not None else np.gradient(g1) * fs

    t_rise = (p - o) / fs
    t_total = (e - o) / fs
    t_notch = (n_idx - o) / fs

    # 面积
    sys_seg = beat_signal[o:n_idx + 1]
    dia_seg = beat_signal[n_idx:e + 1]
    area_sys = float(np.trapezoid(sys_seg)) if len(sys_seg) > 1 else 0.0
    area_dia = float(np.trapezoid(dia_seg)) if len(dia_seg) > 1 else 0.0
    area_total = float(np.trapezoid(beat_signal))

    # 形态
    mu, std = np.mean(beat_signal), np.std(beat_signal)
    skew = float(np.mean((beat_signal - mu)**3) / (std**3 + 1e-10))
    kurt = float(np.mean((beat_signal - mu)**4) / (std**4 + 1e-10))

    return dict(
        A1=lm['A1'], A2=lm['A2'], A1_A2_ratio=lm['A1_A2_ratio'],
        AIx=lm['A2'] / max(lm['A1'], 1e-10),
        T_rise=t_rise, T_total=t_total, T_notch=t_notch,
        T_rise_ratio=t_rise / max(t_total, 1e-6),
        dPdt_max=float(np.max(g1[o:p + 1])) if p > o else 0.0,
        dPdt_min=float(np.min(g1[p:e + 1])) if e > p else 0.0,
        grad2_max=float(np.max(g2)), grad2_min=float(np.min(g2)),
        Area_sys=area_sys, Area_dia=area_dia, Area_total=area_total,
        Area_ratio=area_sys / max(area_total, 1e-10),
        skewness=skew, kurtosis=kurt,
        inflection_point_ratio=(lm['inflection'] - o) / max(p - o, 1),
        peak_val=lm['peak_val'], notch_val=lm['notch_val'],
    )


# ═══════════════════════════════════════════════════════════
# SQI 评估（优化：不遍历完整特征提取）
# ═══════════════════════════════════════════════════════════

def _extract_a1_fast(signal: np.ndarray) -> float:
    """快速提取 A1（仅用于 SQI 形态稳定性评估）。"""
    return float(np.max(signal) - signal[0])


def compute_ppg_sqi(ppg: np.ndarray, peaks: np.ndarray,
                    beats: List[Dict], fs: float = DEFAULT_FS) -> float:
    """PPG 信号质量指数 SQI ∈ [0, 1]。"""
    if len(peaks) < 2:
        return 0.0

    ac_dc = min(np.std(ppg) / max(np.mean(np.abs(ppg)), 1e-6), 1.0)

    rri = np.diff(peaks) / fs
    cv_rri = np.std(rri) / max(np.mean(rri), 1e-6)
    beat_reg = float(np.exp(-cv_rri))

    # SNR
    if len(beats) > 2:
        sigs = [b['signal'] for b in beats]
        min_len = min(len(s) for s in sigs)
        aligned = np.array([s[:min_len] for s in sigs])
        mean_beat = np.mean(aligned, axis=0)
        noise_power = np.mean(np.std(aligned, axis=0)**2)
        sig_power = np.var(mean_beat)
        snr = 10 * np.log10(sig_power / max(noise_power, 1e-10))
        snr_norm = min(max((snr - 5) / 20, 0), 1.0)
    else:
        snr_norm = 0.3

    # 形态稳定性（快速 A1 提取）
    a1s = [_extract_a1_fast(b['signal']) for b in beats]
    if a1s:
        cv_a1 = np.std(a1s) / max(np.mean(np.abs(a1s)), 1e-6)
        morph_stable = float(np.exp(-cv_a1))
    else:
        morph_stable = 0.0

    return float(np.clip(0.35 * ac_dc + 0.25 * beat_reg
                         + 0.20 * snr_norm + 0.20 * morph_stable,
                         0.0, 1.0))


def sqi_grade(sqi: float) -> Tuple[str, str]:
    if sqi >= SQI_GOOD:
        return "优秀", "全特征提取，高置信度"
    if sqi >= SQI_FAIR:
        return "可用", "特征提取，降置信度输出"
    return "差", "仅提取基本特征" if sqi >= 0.25 else ("无效", "重新放置传感器")


# ═══════════════════════════════════════════════════════════
# 六品质连续谱映射
# ═══════════════════════════════════════════════════════════

def six_qualities_from_features(features: Dict) -> Dict[str, float]:
    """聚合 PPG 特征 → 六品质连续谱 ∈ [0,1]。"""
    if 'error' in features:
        return {k: 0.5 for k in
                ['coarse_score', 'hard_score', 'rate_score',
                 'smooth_score', 'depth_score', 'strength_score']}

    a1 = features.get('A1_mean', 0.5)
    aix = features.get('AIx_mean', 0.4)
    dpmax = features.get('dPdt_max_mean', 1.0)
    a1_cv = features.get('A1_cv', 5.0)
    aix_cv = features.get('AIx_cv', 5.0)
    area_sys = features.get('Area_sys_mean', 0.3)
    area_total = features.get('Area_total_mean', 0.5)
    hr = features.get('HR_mean', 72.0)
    si = features.get('SI_mean', 8.0)
    ptt = features.get('PTT_mean', 0)

    return {
        'coarse_score': float(np.clip(0.6 * (a1 / 0.8) + 0.4 * (area_sys / 0.5), 0, 1)),
        'hard_score': float(np.clip(0.7 * ((si - 5) / 15) + 0.3 * (aix / 0.7), 0, 1)),
        'rate_score': float(np.clip((hr - 40) / 80, 0, 1)),
        'smooth_score': float(np.clip(1.0 - np.sqrt(a1_cv**2 + aix_cv**2) / (np.sqrt(2) * 20), 0, 1)),
        'depth_score': float(np.clip((300 - ptt) / 200, 0, 1)) if ptt > 0 else 0.5,
        'strength_score': float(np.clip(0.5 * (dpmax / 2.0) + 0.5 * (area_total / 0.6), 0, 1)),
    }


# ═══════════════════════════════════════════════════════════
# 血管硬度指数 SI
# ═══════════════════════════════════════════════════════════

def compute_si(beats_features: List[Dict], fs: float,
               patient_height_m: float = 1.70) -> float:
    """血管硬度指数 SI = 身高 / ΔT。"""
    dts = [f.get('T_notch', 0) - f.get('T_rise', 0) for f in beats_features
           if 'error' not in f and 0.1 < f.get('T_notch', 0) - f.get('T_rise', 0) < 0.5]
    if not dts:
        return 8.0
    return patient_height_m / max(np.mean(dts), 1e-6)


# ═══════════════════════════════════════════════════════════
# 测试信号生成（单一定义，供外部模块复用）
# ═══════════════════════════════════════════════════════════

def generate_test_ppg(fs: float = 250.0, duration: float = 10.0,
                      bpm: float = 72.0) -> np.ndarray:
    """生成模拟 PPG 信号用于测试。"""
    t = np.arange(0, duration, 1.0 / fs)
    bi = 60.0 / bpm
    bl = int(bi * fs)
    rise, fall = int(0.12 * fs), int(0.3 * fs)

    single = np.zeros(bl)
    for i in range(bl):
        if i < rise:
            single[i] = 1 - np.exp(-3 * i / rise)
        elif i < rise + fall:
            single[i] = np.exp(-2 * (i - rise) / fall)
        else:
            single[i] = 0.05

    sig = np.tile(single, int(np.ceil(duration / bi)))[:len(t)]
    sig += np.random.normal(0, 0.03, len(sig))
    sig += 0.1 * np.sin(2 * np.pi * 0.2 * t[:len(sig)])
    return sig


# ═══════════════════════════════════════════════════════════
# PPGProcessor 类
# ═══════════════════════════════════════════════════════════

class PPGProcessor:
    """PPG 信号处理器（预处理 → 搏动检测 → 特征 → SQI → 六品质）。"""

    def __init__(self, fs: float = DEFAULT_FS):
        self.fs = fs

    def process(self, ppg_signal: np.ndarray,
                patient_height_m: float = 1.70,
                return_beats: bool = False) -> Dict:
        """完整处理流水线。"""
        clean = preprocess_ppg(ppg_signal, self.fs)
        peaks = detect_peaks(clean, self.fs)
        if len(peaks) < 2:
            return {'error': f'搏动检测失败({len(peaks)})', 'ppg_clean': clean, 'peaks': peaks}

        beats = segment_beats(clean, peaks, self.fs)
        if len(beats) < 2:
            return {'error': f'有效搏动不足({len(beats)})', 'ppg_clean': clean, 'peaks': peaks}

        # 特征提取（梯度复用）
        bfs = []
        for b in beats:
            g1, g2 = _compute_gradients(b['signal'], self.fs)
            feat = extract_beat_features(b['signal'], self.fs, g1, g2)
            if 'error' not in feat:
                bfs.append(feat)

        if len(bfs) < 2:
            return {'error': f'特征提取失败({len(bfs)})', 'ppg_clean': clean, 'peaks': peaks}

        features = self._aggregate_features(bfs)
        features['SI_mean'] = compute_si(bfs, self.fs, patient_height_m)

        # 心率
        rri = np.diff(peaks) / self.fs
        hr = 60.0 / np.mean(rri) if np.mean(rri) > 0 else 0
        features['HR_mean'] = hr
        features['HRV'] = float(np.std(rri) / max(np.mean(rri), 1e-6) * 100)

        sqi = compute_ppg_sqi(clean, peaks, beats, self.fs)
        sq_label, sq_advice = sqi_grade(sqi)
        sq_dict = six_qualities_from_features(features)

        result = dict(features=features, six_qualities=sq_dict,
                       sqi=sqi, sqi_grade=sq_label, sqi_advice=sq_advice,
                       peaks=peaks, n_beats=len(bfs), hr_bpm=hr,
                       ppg_clean=clean)
        if return_beats:
            result['beats'] = beats
            result['beat_features'] = bfs
        return result

    @staticmethod
    def _aggregate_features(beat_features: List[Dict]) -> Dict:
        """聚合特征为统计量。"""
        keys = beat_features[0].keys()
        agg = {}
        for k in keys:
            vals = [b[k] for b in beat_features]
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            agg[f'{k}_mean'] = float(mean_v)
            agg[f'{k}_std'] = float(std_v)
            agg[f'{k}_cv'] = float(std_v / max(abs(mean_v), 1e-10) * 100)
        return agg


# ═══════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PPG 信号处理模块 — 自测")
    print("=" * 60)

    fs = 250.0
    sig = generate_test_ppg(fs, duration=15.0, bpm=72)
    print(f"\n模拟 PPG: {len(sig)} 样本 @ {fs} Hz, 72 BPM")

    proc = PPGProcessor(fs=fs)
    r = proc.process(sig, return_beats=True)

    if 'error' in r:
        print(f"失败: {r['error']}")
        return

    print(f"\n  搏动: {r['n_beats']} | HR: {r['hr_bpm']:.1f} BPM | SQI: {r['sqi']:.3f} ({r['sqi_grade']})")
    print(f"\n  六品质:")
    for k, v in r['six_qualities'].items():
        bar = '█' * int(v * 20) + '░' * int((1 - v) * 20)
        print(f"    {k:20s}: {v:.3f}  {bar}")

    feat = r['features']
    for k in ['A1_mean', 'AIx_mean', 'T_rise_ratio_mean', 'dPdt_max_mean', 'SI_mean']:
        print(f"    {k:25s}: {feat.get(k, 0):.4f}")

    print(f"\n✅ 自测完成")


if __name__ == '__main__':
    main()
