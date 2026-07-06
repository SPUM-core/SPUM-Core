"""
青囊闻诊 · 声域分型分析器
==========================
基于 SPUM 五形关系学，从语音录音中提取声学特征，
自动分类为五类声域体质（角木/徵火/宫土/商金/羽水）。

依赖安装：
    pip install librosa numpy imageio-ffmpeg

使用：
    python voice_analyzer.py <录音文件.m4a> [录音文件2.m4a ...]
"""

import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────
# 五类声域参考模板（来自青囊agent.md §4.3）
# ──────────────────────────────────────────────
VOICE_TYPES = {
    "角木": {
        "label": "角音·木型体质",
        "description": "声细尖高、音位偏高、气息上扬、穿透力强",
        "delta_S": {"S_木": "↑↑", "S_火": "↔", "S_土": "↓", "S_金": "↓", "S_水": "↔"},
        "thresholds": {
            "f0_mean": (180, 999),
            "f0_std": (35, 999),
            "spectral_centroid": (2200, 9999),
            "spectral_rolloff": (4000, 9999),
            "zcr": (0.08, 1.0),
        },
    },
    "徵火": {
        "label": "徵音·火型体质",
        "description": "声洪亮上扬、清亮外放、语速偏快",
        "delta_S": {"S_木": "↔", "S_火": "↑↑", "S_土": "↔", "S_金": "↓", "S_水": "↓"},
        "thresholds": {
            "f0_mean": (160, 240),
            "f0_std": (40, 999),
            "spectral_centroid": (2000, 2800),
            "spectral_rolloff": (3500, 5000),
            "zcr": (0.07, 0.15),
        },
    },
    "宫土": {
        "label": "宫音·土型体质",
        "description": "声浑厚憨直、居中低音、沉稳不飘",
        "delta_S": {"S_木": "↔", "S_火": "↔", "S_土": "↑", "S_金": "↔", "S_水": "↔"},
        "thresholds": {
            "f0_mean": (100, 170),
            "f0_std": (20, 50),
            "spectral_centroid": (1200, 2100),
            "spectral_rolloff": (2300, 4000),
            "zcr": (0.04, 0.09),
        },
    },
    "商金": {
        "label": "商音·金型体质",
        "description": "声清冷干净、收敛发紧、克制压抑",
        "delta_S": {"S_木": "↓", "S_火": "↓", "S_土": "↔", "S_金": "↑↑", "S_水": "↔"},
        "thresholds": {
            "f0_mean": (140, 210),
            "f0_std": (15, 35),
            "spectral_centroid": (1800, 2500),
            "spectral_rolloff": (3000, 4200),
            "zcr": (0.04, 0.08),
        },
    },
    "羽水": {
        "label": "羽音·水型体质",
        "description": "声低沉粗旷、音区下沉、低沉缓慢",
        "delta_S": {"S_木": "↓", "S_火": "↓", "S_土": "↔", "S_金": "↔", "S_水": "↑↑"},
        "thresholds": {
            "f0_mean": (60, 130),
            "f0_std": (10, 30),
            "spectral_centroid": (800, 1700),
            "spectral_rolloff": (1800, 3000),
            "zcr": (0.02, 0.05),
        },
    },
}


def load_and_convert(audio_path, sr=16000):
    """
    加载音频文件。
    支持 .m4a/.mp4/.aac（自动用 imageio_ffmpeg 转换为临时 WAV）。
    下采样到 16kHz，只取中间 15 秒语音段（降速+降内存）。
    """
    import librosa
    import subprocess

    ext = os.path.splitext(audio_path)[1].lower()
    temp_wav = None
    if ext in (".m4a", ".mp4", ".aac", ".mp3"):
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            temp_wav = audio_path + "_temp.wav"
            subprocess.run(
                [ffmpeg, "-y", "-i", audio_path, temp_wav],
                capture_output=True, check=True, timeout=120,
            )
            audio_path = temp_wav
        except Exception as e:
            print(f"[警告] 音频转换失败: {e}")
            raise

    try:
        y, sr_orig = librosa.load(audio_path, sr=sr)
    except Exception as e:
        raise RuntimeError(f"无法加载音频文件: {e}")

    if temp_wav and os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except OSError:
            pass

    # 去除首尾静音段
    y, _ = librosa.effects.trim(y, top_db=20)
    duration = len(y) / sr

    # 只取中间 ≤15 秒语音段加速处理
    max_len = 15 * sr
    if len(y) > max_len:
        start = (len(y) - max_len) // 2
        y = y[start:start + max_len]
        duration = len(y) / sr

    return y, sr, duration


def extract_features_fast(y, sr):
    """快速提取语音声学特征（避免慢速 pyin）。"""
    import librosa

    # ── 基频：用 piptrack（比 pyin 快 10-100 倍） ──
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=65, fmax=1047)
    pitch_frames = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        mag = magnitudes[idx, t]
        if mag > 0.5:
            pitch_frames.append(pitches[idx, t])
    pitch_frames = np.array(pitch_frames)

    if len(pitch_frames) > 0:
        f0_mean = float(np.mean(pitch_frames))
        f0_std = float(np.std(pitch_frames))
        f0_med = float(np.median(pitch_frames))

        f0_harmonic_check = f0_mean
        if f0_mean > 220:
            f0_harmonic_check = f0_mean / 2.0
    else:
        f0_mean, f0_std, f0_med = 0.0, 0.0, 0.0
        f0_harmonic_check = 0.0

    # ── 谱特征 ──
    n_fft = 2048
    hop_length = 512

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    centroid_mean = float(np.mean(centroid))

    rolloff = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85
    )[0]
    rolloff_mean = float(np.mean(rolloff))

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # RMS 能量
    rms = float(np.mean(librosa.feature.rms(y=y)[0]))

    features = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_median": f0_med,
        "n_pitch_frames": len(pitch_frames),
        "spectral_centroid": centroid_mean,
        "spectral_rolloff": rolloff_mean,
        "zcr": zcr_mean,
        "mfcc_1": float(mfcc_mean[1]),
        "mfcc_2": float(mfcc_mean[2]),
        "rms_energy": rms,
        "duration": len(y) / sr,
    }

    # 双模式分类：原始 F0 和泛音校正 F0
    raw_type, raw_scores = classify_voice(features)
    if f0_harmonic_check != f0_mean:
        features_harmonic = dict(features)
        features_harmonic["f0_mean"] = f0_harmonic_check
        features_harmonic["f0_std"] = f0_std / 2.0
        features_harmonic["f0_median"] = f0_med / 2.0
        harm_type, harm_scores = classify_voice(features_harmonic)
        if raw_scores[raw_type]["score"] >= harm_scores[harm_type]["score"]:
            features["harmonic_mode"] = "原始基频"
        else:
            features["harmonic_mode"] = "泛音校正（÷2）"
            features["f0_mean"] = f0_harmonic_check
            features["f0_std"] = f0_std / 2.0
            features["f0_median"] = f0_med / 2.0
            best_type, all_scores = harm_type, harm_scores
            return features, best_type, all_scores
    else:
        features["harmonic_mode"] = "原始基频"

    best_type, all_scores = raw_type, raw_scores
    return features, best_type, all_scores


def classify_voice(features):
    """基于特征 + 距离评分进行五型分类。"""
    scores = {}
    for vtype, template in VOICE_TYPES.items():
        thr = template["thresholds"]
        score = 0.0
        matches = 0
        n_features = 0

        checks = [
            ("f0_mean", "f0_mean"),
            ("f0_std", "f0_std"),
            ("spectral_centroid", "spectral_centroid"),
            ("spectral_rolloff", "spectral_rolloff"),
            ("zcr", "zcr"),
        ]
        for feat_key, thr_key in checks:
            n_features += 1
            val = features[feat_key]
            lo, hi = thr[thr_key]
            if lo <= val <= hi:
                score += 1.0
                matches += 1
            else:
                dist = min(abs(val - lo), abs(val - hi))
                penalty = -0.2 * min(dist / (hi - lo + 1e-6), 3.0)
                score += penalty

        # MFCC-1 辅助修正
        m1 = features["mfcc_1"]
        if vtype == "商金" and m1 < -5:
            score += 0.5
        elif vtype == "宫土" and -2 < m1 < 8:
            score += 0.3
        elif vtype == "羽水" and m1 > 3:
            score += 0.3

        scores[vtype] = {
            "score": round(score, 2),
            "match_rate": f"{matches}/{n_features}",
        }

    best = max(scores, key=lambda k: scores[k]["score"])
    return best, scores


def extract_five_tone_bands(y, sr):
    """提取五音频段能量分布（青囊agent.md §4.6 协议）"""
    import librosa

    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bands = {
        "宫(土)": (130, 260),
        "商(金)": (260, 390),
        "角(木)": (390, 520),
        "徵(火)": (520, 780),
        "羽(水)": (780, 1040),
    }
    total = np.sum(D[(freqs >= 130) & (freqs < 1040)])
    if total == 0:
        return {k: 0.0 for k in bands}
    ratios = {}
    for name, (lo, hi) in bands.items():
        ratios[name] = float(np.sum(D[(freqs >= lo) & (freqs < hi)]) / total * 100)
    return ratios


def generate_s_vector_from_bands(band_ratios, rms_energy):
    """根据五音频段能量生成 S 向量闻诊判定（§4.6 步骤5）"""
    s_vector = {}
    # 基准分布：各 20%
    mapping = {
        "宫(土)": "S_土",
        "商(金)": "S_金",
        "角(木)": "S_木",
        "徵(火)": "S_火",
        "羽(水)": "S_水",
    }
    for band, name in mapping.items():
        ratio = band_ratios.get(band, 0) / 100.0
        dev = (ratio - 0.20) / 0.20 * 100
        if dev > 15:
            s_vector[name] = "↑↑"
        elif dev > 5:
            s_vector[name] = "↑"
        elif dev < -15:
            s_vector[name] = "↓↓"
        elif dev < -5:
            s_vector[name] = "↓"
        else:
            s_vector[name] = "↔"

    # 虚阳外浮检测（§4.6 步骤4）
    fire_ratio = band_ratios.get("徵(火)", 0)
    if fire_ratio > 28 and rms_energy < 0.02:
        s_vector["S_火"] = "↓↓↓(浮越)"

    return s_vector


def analyze(audio_path):
    """主分析流程——返回结构化结果。"""
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"文件不存在: {audio_path}")

    y, sr, duration = load_and_convert(audio_path)
    if duration < 1.0:
        import logging
        logging.warning(f"录音过短 ({duration:.1f}s)")

    features, best_type, all_scores = extract_features_fast(y, sr)
    band_ratios = extract_five_tone_bands(y, sr)
    s_vector = generate_s_vector_from_bands(band_ratios, features.get("rms_energy", 0))
    vtype = VOICE_TYPES[best_type]

    result = {
        "file": os.path.basename(audio_path),
        "duration_seconds": round(features["duration"], 1),
        "rms_energy": features["rms_energy"],
        "f0_mean_hz": round(features["f0_mean"], 1),
        "f0_median_hz": round(features["f0_median"], 1),
        "spectral_centroid_hz": round(features["spectral_centroid"], 0),
        "spectral_rolloff_hz": round(features["spectral_rolloff"], 0),
        "zcr": round(features["zcr"], 4),
        "harmonic_mode": features.get("harmonic_mode", "原始基频"),
        "voice_type": best_type,
        "voice_label": vtype["label"],
        "voice_description": vtype["description"],
        "delta_S": vtype["delta_S"],
        "scores": all_scores,
        "five_tone_bands": {k: round(v, 2) for k, v in band_ratios.items()},
        "s_vector_auscultation": s_vector,
    }
    return result


# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python voice_analyzer.py <录音文件.m4a> [录音文件2.m4a ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        result = analyze(path)
        print(f"\n{'='*62}")
        print(f"  青囊闻诊 · 声域分型分析报告")
        print(f"{'='*62}")
        print(f"  音频：{result['file']}")
        print(f"\n── 声学特征 ──")
        print(f"  基频模式  {result['harmonic_mode']}")
        print(f"  基频均值  {result['f0_mean_hz']} Hz")
        print(f"  基频中值  {result['f0_median_hz']} Hz")
        print(f"  谱质心    {result['spectral_centroid_hz']} Hz")
        print(f"  谱滚降    {result['spectral_rolloff_hz']} Hz")
        print(f"  过零率    {result['zcr']}")
        print(f"  RMS能量   {result['rms_energy']}")
        print(f"\n── 分型判定 ──")
        print(f"  {result['voice_label']} — {result['voice_description']}")
        print(f"\n── 五音频段能量 ──")
        for band, val in result['five_tone_bands'].items():
            print(f"  {band}: {val}%")
        print(f"\n── 闻诊 S 向量 ──")
        for k, v in result['s_vector_auscultation'].items():
            print(f"  {k} = {v}")
        print(f"\n{'='*62}")
