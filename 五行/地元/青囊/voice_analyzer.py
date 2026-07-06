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
    if ext in (".m4a", ".mp4", ".aac"):
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
            print(f"[警告] m4a 转换失败: {e}")
            sys.exit(1)

    try:
        y, sr_orig = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"[错误] 无法加载音频文件: {e}")
        sys.exit(1)

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
    # 取每个帧的最大幅度对应的频率
    pitch_frames = []
    for t in range(pitches.shape[1]):
        idx = magnitudes[:, t].argmax()
        mag = magnitudes[idx, t]
        if mag > 0.5:  # 能量阈值过滤噪音
            pitch_frames.append(pitches[idx, t])
    pitch_frames = np.array(pitch_frames)

    if len(pitch_frames) > 0:
        f0_mean = float(np.mean(pitch_frames))
        f0_std = float(np.std(pitch_frames))
        f0_med = float(np.median(pitch_frames))

        # 泛音校验：如果 F0_mean > 220 Hz（高于成年男性典型上限），
        # 尝试 ÷2 降八度再匹配（piptrack 可能抓到第一泛音）
        f0_harmonic_check = f0_mean
        if f0_mean > 220:
            f0_harmonic_check = f0_mean / 2.0
    else:
        f0_mean, f0_std, f0_med = 0.0, 0.0, 0.0
        f0_harmonic_check = 0.0

    # ── 谱特征 ──
    # 用 2048 窗口减少计算量
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

    # MFCC：5 帧步长加速
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    mfcc_mean = np.mean(mfcc, axis=1)

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
    }

    # 双模式分类：原始 F0 和泛音校正 F0
    raw_type, raw_scores = classify_voice(features)
    if f0_harmonic_check != f0_mean:
        features_harmonic = dict(features)
        features_harmonic["f0_mean"] = f0_harmonic_check
        features_harmonic["f0_std"] = f0_std / 2.0
        features_harmonic["f0_median"] = f0_med / 2.0
        harm_type, harm_scores = classify_voice(features_harmonic)
        # 取得分更高的模式
        if raw_scores[raw_type]["score"] >= harm_scores[harm_type]["score"]:
            features["harmonic_mode"] = "原始基频"
        else:
            features["harmonic_mode"] = "泛音校正（÷2）"
            features["f0_mean"] = f0_harmonic_check
            features["f0_std"] = f0_std / 2.0
            features["f0_median"] = f0_med / 2.0
            # 重新用校正值分类
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


def print_report(audio_path, features, best_type, all_scores):
    """打印终端报告。"""
    vtype = VOICE_TYPES[best_type]

    print("=" * 62)
    print("  青囊闻诊 · 声域分型分析报告")
    print("=" * 62)
    print(f"  音频：{os.path.basename(audio_path)}")
    print()

    print("── 声学特征 ──")
    harm = features.get("harmonic_mode", "")
    print(f"  基频模式  {'  ' + harm if harm else ''}")
    print(f"  基频均值    F0_mean  = {features['f0_mean']:.1f} Hz")
    print(f"  泛音校验    F0/2     = {features['f0_mean']/2:.1f} Hz（如 >220 Hz 可能为泛音）")
    print(f"  基频标准差  F0_std   = {features['f0_std']:.1f} Hz")
    print(f"  基频中位数  F0_med   = {features['f0_median']:.1f} Hz")
    print(f"  有效帧数              = {features.get('n_pitch_frames', 0)}")
    print(f"  谱质心均值            = {features['spectral_centroid']:.0f} Hz")
    print(f"  谱滚降点              = {features['spectral_rolloff']:.0f} Hz")
    print(f"  过零率                = {features['zcr']:.4f}")
    print(f"  MFCC-1                = {features['mfcc_1']:.2f}")
    print()

    print("── 分型判定 ──")
    print(f"  ► {vtype['label']}")
    print(f"    特征：{vtype['description']}")
    print()

    print("── 匹配得分 ──")
    ranking = sorted(all_scores.items(), key=lambda x: x[1]["score"], reverse=True)
    for vt, sc in ranking:
        bar = "█" * max(1, int(sc["score"] * 5))
        print(f"  {vt:4s}  {sc['score']:5.2f}  {bar}  ({sc['match_rate']})")
    print()

    print("── SPUM ΔS 向量 ──")
    ds = vtype["delta_S"]
    vec = ", ".join(f"{k}={v}" for k, v in ds.items())
    print(f"  ΔS = ({vec})")
    print()

    print("── 体质倾向与治疗倾向 ──")
    _print_treatment_notes(best_type)
    print()

    print("─" * 62)
    print("  ⚠️  声域分型反映先天体质底色。")
    print("     临床辨证需结合四诊合参。")
    print("=" * 62)


def _print_treatment_notes(vtype):
    """各类型治疗倾向。"""
    notes = {
        "角木": (
            "  木气偏旺，肝木疏泄过亢。\n"
            "  治疗：增强金气收敛、加固中土制约。\n"
            "  方剂参考：香砂六君子固土，陈皮木香调木。"
        ),
        "徵火": (
            "  火气亢盛，心阳浮越不降。\n"
            "  治疗：滋水降火、助金收敛。\n"
            "  方剂参考：二至丸/增液汤滋水，沙参/麦冬助金。"
        ),
        "宫土": (
            "  土气充足，中焦承载之力强健。\n"
            "  治疗：无需大幅矫正，随证微调即可。\n"
            "  方剂参考：平胃散类温和运化。"
        ),
        "商金": (
            "  金气收敛太过，木气生发受阻。\n"
            "  治疗：宣发疏泄、助火升发。\n"
            "  方剂参考：桔梗/升麻助宣发，桂枝/干姜升火。"
        ),
        "羽水": (
            "  水气潜藏过盛，命门火弱。\n"
            "  治疗：益火温土、升提气机。\n"
            "  方剂参考：党参/黄芪升火温土，升麻提升金气。"
        ),
    }
    print(notes.get(vtype, ""))


def analyze(audio_path):
    """主分析流程。"""
    if not os.path.exists(audio_path):
        print(f"[错误] 文件不存在: {audio_path}")
        sys.exit(1)

    print(f"[加载] {audio_path}")
    y, sr, duration = load_and_convert(audio_path)

    if duration < 1.0:
        print(f"[警告] 录音过短 ({duration:.1f}s)，建议 ≥10s。")
    print(f"[时长] {duration:.1f}s")

    features, best_type, all_scores = extract_features_fast(y, sr)
    print_report(audio_path, features, best_type, all_scores)
    return best_type, features


# ──────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python voice_analyzer.py <录音文件.m4a> [录音文件2.m4a ...]")
        print("示例：python voice_analyzer.py 录音.m4a")
        sys.exit(1)
    for path in sys.argv[1:]:
        analyze(path)
        print()
