#!/usr/bin/env python3
"""
PPG → 五形 ΔF 桥接模块
========================

将 PPG 六品质连续谱映射为五形 ΔF 向量，无缝接入现有 SPUM 辨证系统。

输入：PPG 六品质 (six_qualities, ∈[0,1], 0.5=正常)
输出：ΔF = [S_木, S_火, S_土, S_金, S_水] (∈[-1,1], 0=正常)

支持两种运行模式：
  1. PPG-only（独立模式）— 仅用 Pulsesensor 输出 ΔF
  2. PPG+ECG（增强模式）— ECG 为主，PPG 校正六品质精度

用法：
  # PPG-only 模式
  from ppg_to_wuxing_bridge import PpgToWuxingBridge
  bridge = PpgToWuxingBridge()
  delta_F = bridge.six_qualities_to_deltaF(six_qualities)
  diagnosis = bridge.decode(six_qualities)

  # 完整流水线（从原始 PPG 信号到辨证）
  result = bridge.pipeline(ppg_signal, fs=250.0)

依赖：
  pip install numpy scipy

v1.0 / 2026-07-15
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from ppg_processing import PPGProcessor, six_qualities_from_features, generate_test_ppg
from syndrome_decoder import decode_syndrome, NORMAL_BASELINE


# ============================================================
# 六品质 → ΔF 维度映射权重
# ============================================================
# 每个六品质维度对五形 ΔF 各分量的贡献权重。
#
# 映射逻辑：
#   ΔF[i] = Σ( w_ji × (sq_j - 0.5) × 2 )
#   其中 sq_j ∈ [0,1], 0.5 = 正常
#   乘以 2 将 [0.5, 1.0] 映射为 [0, 1], [0, 0.5] 映射为 [-1, 0]

QUALITY_TO_DELTAF_WEIGHTS = {
    'coarse_score':    [ 0.0,  0.4,  0.3,  0.0,  0.0],  # 粗↔细 → 火↑土↑
    'hard_score':      [ 0.5,  0.0,  0.0,  0.4,  0.0],  # 软↔硬 → 木↑金↑
    'rate_score':      [ 0.0,  0.4,  0.0,  0.0,  0.0],  # 缓↔急 → 火↑
    'smooth_score':    [ 0.0,  0.0,  0.0,  0.0,  0.6],  # 滑↔涩 → 水↑
    'depth_score':     [ 0.0,  0.2,  0.0,  0.0,  0.0],  # 浮↔沉 → 火(方向)
    'strength_score':  [ 0.0,  0.3,  0.3,  0.0,  0.0],  # 有力↔无力 → 火↑土↑
}

DIMENSION_NAMES = ['S_木', 'S_火', 'S_土', 'S_金', 'S_水']


# ============================================================
# 六品质 → ΔF 转换
# ============================================================

def six_qualities_to_deltaF(six_qualities: Dict[str, float]) -> List[float]:
    """
    将 PPG 六品质连续谱转换为五形 ΔF 向量。

    参数：
        six_qualities: {
            'coarse_score': float ∈ [0,1],  # 粗↔细
            'hard_score':   float ∈ [0,1],  # 软↔硬
            'rate_score':   float ∈ [0,1],  # 缓↔急
            'smooth_score': float ∈ [0,1],  # 滑↔涩
            'depth_score':  float ∈ [0,1],  # 浮↔沉
            'strength_score': float ∈ [0,1], # 有力↔无力
        }

    返回：
        delta_F: List[float, 5]  # [S_木, S_火, S_土, S_金, S_水]
                 每个分量 ∈ [-1, 1], 0 = 正常
    """
    delta_F = np.zeros(5, dtype=np.float64)

    for quality_name, value in six_qualities.items():
        if quality_name not in QUALITY_TO_DELTAF_WEIGHTS:
            continue

        # 转换: [0,1] → [-1,1], 0.5 → 0
        deviation = (value - 0.5) * 2.0
        # 裁剪到有效范围
        deviation = np.clip(deviation, -1.0, 1.0)

        # 加权分配到五形维度
        weights = QUALITY_TO_DELTAF_WEIGHTS[quality_name]
        for i in range(5):
            delta_F[i] += deviation * weights[i]

    # 裁剪到 [-1, 1]
    delta_F = np.clip(delta_F, -1.0, 1.0)

    return delta_F.tolist()


# ============================================================
# ΔF → 症状解读
# ============================================================

def describe_deltaF(delta_F: List[float]) -> List[str]:
    """
    将 ΔF 向量翻译为中文语义描述。

    参数：
        delta_F: [S_木, S_火, S_土, S_金, S_水]

    返回：
        descriptions: 偏离方向描述列表
    """
    names = ['木(肝胆/弹性)', '火(心/脉势)', '土(脾/容量)',
             '金(肺/回弹)', '水(肾/传导)']
    descriptions = []

    for i, (name, val) in enumerate(zip(names, delta_F)):
        if val > 0.2:
            descriptions.append(f"{name} ↑ ({val:+.2f})")
        elif val < -0.2:
            descriptions.append(f"{name} ↓ ({val:+.2f})")

    if not descriptions:
        descriptions.append("五形均在正常范围")

    return descriptions


# ============================================================
# PPG-only 流水线
# ============================================================

class PpgToWuxingBridge:
    """
    PPG→五形桥接器。

    将 PPG 信号处理 + 六品质提取 + ΔF 映射 + 辨证解码
    整合为一步调用。
    """

    def __init__(self, fs: float = 250.0, patient_height_m: float = 1.70):
        """
        参数：
            fs: 采样率 (Hz)
            patient_height_m: 身高（米），用于 SI 计算
        """
        self.fs = fs
        self.patient_height_m = patient_height_m
        self._processor = PPGProcessor(fs=fs)

    def six_qualities_to_deltaF(self, six_qualities: Dict[str, float]) -> List[float]:
        """六品质 → ΔF 向量。"""
        return six_qualities_to_deltaF(six_qualities)

    def decode(self, six_qualities: Dict[str, float]) -> Dict:
        """
        六品质 → 辨证解码。

        返回：
            {
                'delta_F': [S_木, ..., S_水],
                'syndrome_main': {...},   # 主证 dict
                'syndrome_sub': [...],    # 兼证列表
                'syndrome_combo': [...],  # 组合证型
                'summary': str,          # 综合摘要
                'descriptions': [...],   # 五形语义描述
            }
        """
        delta_F = self.six_qualities_to_deltaF(six_qualities)
        sd = decode_syndrome(delta_F)
        descriptions = describe_deltaF(delta_F)

        return {
            'delta_F': delta_F,
            'syndrome_main': sd.get('主证', {}),
            'syndrome_sub': sd.get('兼证', []),
            'syndrome_combo': sd.get('组合证型', []),
            'summary': sd.get('综合摘要', ''),
            'descriptions': descriptions,
        }

    def pipeline(self, ppg_signal: np.ndarray,
                 return_raw: bool = False) -> Dict:
        """
        完整流水线：PPG 原始信号 → 辨证结果。

        处理步骤：
          1. PPG 预处理 + 搏动检测 + 特征提取
          2. 六品质连续谱计算
          3. 六品质 → ΔF 向量转换
          4. ΔF → 辨证解码

        参数：
            ppg_signal: 原始 PPG 信号 (1D array)
            return_raw: 是否返回中间处理结果

        返回：
            {
                'delta_F': [float x 5],
                'six_qualities': {...},
                'syndromes': [...],
                'descriptions': [...],
                'sqi': float,
                'sqi_grade': str,
                'hr_bpm': float,
                # 以下仅在 return_raw=True 时返回
                'features': {...},
                'ppg_clean': array,
                'peaks': array,
            }
        """
        # 1. PPG 处理
        proc_result = self._processor.process(
            ppg_signal,
            patient_height_m=self.patient_height_m,
            return_beats=False,
        )

        if 'error' in proc_result:
            return {
                'error': proc_result['error'],
                'delta_F': [0.0, 0.0, 0.0, 0.0, 0.0],
                'six_qualities': {k: 0.5 for k in
                    ['coarse_score', 'hard_score', 'rate_score',
                     'smooth_score', 'depth_score', 'strength_score']},
                'syndromes': [],
                'sqi': 0.0,
                'sqi_grade': '无效',
                'hr_bpm': 0.0,
            }

        # 2. 六品质
        six_qualities = proc_result['six_qualities']

        # 3. ΔF + 辨证
        diagnosis = self.decode(six_qualities)

        # 4. 组装结果
        result = {
            'delta_F': diagnosis['delta_F'],
            'six_qualities': six_qualities,
            'syndrome_main': diagnosis['syndrome_main'],
            'syndrome_sub': diagnosis['syndrome_sub'],
            'syndrome_combo': diagnosis['syndrome_combo'],
            'summary': diagnosis['summary'],
            'descriptions': diagnosis['descriptions'],
            'sqi': proc_result['sqi'],
            'sqi_grade': proc_result['sqi_grade'],
            'hr_bpm': proc_result['hr_bpm'],
            'n_beats': proc_result['n_beats'],
        }

        if return_raw:
            result['features'] = proc_result['features']
            result['ppg_clean'] = proc_result['ppg_clean']
            result['peaks'] = proc_result['peaks']

        return result


# ============================================================
# ECG + PPG 联合模式（增强）
# ============================================================

def enhance_ecg_deltaF(
    ecg_delta_F: List[float],
    ppg_six_qualities: Dict[str, float],
    ppg_confidence: float = 0.8,
) -> Tuple[List[float], Dict[str, float]]:
    """
    用 PPG 六品质增强 / 修正 ECG ΔF 向量。

    当 ECG 和 PPG 同步采集时，PPG 可以提供 ECG 无法直接测量的信息：
      - 外周血管阻力 (AIx) → 修正 S_木, S_金
      - 血管弹性 (SI, T_rise) → 修正 S_木
      - 搏动间变异度 → 修正 S_水

    参数：
        ecg_delta_F: ECG 导出的 ΔF 向量 [S_木, S_火, S_土, S_金, S_水]
        ppg_six_qualities: PPG 六品质字典
        ppg_confidence: PPG 校正置信度 ∈ [0, 1]

    返回：
        (enhanced_delta_F, correction_details)
    """
    ppg_delta_F = six_qualities_to_deltaF(ppg_six_qualities)

    # 各维度的 PPG 增强权重
    # ECG 对 S_火 (脉率)、S_土 (ST段) 更准确
    # PPG 对 S_木 (弹性)、S_水 (传导) 提供额外信息
    enhancement_weights = [0.3, 0.1, 0.1, 0.3, 0.3]

    enhanced = []
    corrections = {}
    for i in range(5):
        w = enhancement_weights[i] * ppg_confidence
        combined = ecg_delta_F[i] * (1 - w) + ppg_delta_F[i] * w
        combined = np.clip(combined, -1.0, 1.0)
        enhanced.append(float(combined))
        corrections[DIMENSION_NAMES[i]] = {
            'ecg': float(ecg_delta_F[i]),
            'ppg': float(ppg_delta_F[i]),
            'combined': float(combined),
            'weight': w,
        }

    return enhanced, corrections


# ============================================================
# CLI 自测
# ============================================================

# generate_test_ppg 从 ppg_processing 导入复用


def main():
    print("=" * 60)
    print("PPG → 五形 ΔF 桥接模块 — 自测")
    print("=" * 60)

    fs = 250.0

    # ── 测试 1: 六品质 → ΔF ──
    print("\n[测试 1] 六品质 → ΔF 映射")
    test_sq = {
        'coarse_score': 0.7,     # 偏粗
        'hard_score': 0.6,       # 偏硬
        'rate_score': 0.55,      # 略数
        'smooth_score': 0.5,     # 正常
        'depth_score': 0.5,      # 正常（无 ECG）
        'strength_score': 0.65,  # 偏有力
    }
    delta_F = six_qualities_to_deltaF(test_sq)
    print(f"  六品质: {test_sq}")
    print(f"  ΔF:     [{', '.join(f'{v:+.3f}' for v in delta_F)}]")
    desc = describe_deltaF(delta_F)
    print(f"  解读:   {'; '.join(desc)}")

    # ── 测试 2: 完整流水线 ──
    print("\n[测试 2] PPG 完整流水线（模拟信号 @ 72 BPM）")
    sig = generate_test_ppg(fs, duration=15.0, bpm=72)

    bridge = PpgToWuxingBridge(fs=fs)
    result = bridge.pipeline(sig, return_raw=True)

    if 'error' in result:
        print(f"  流水线失败: {result['error']}")
    else:
        print(f"  SQI:     {result['sqi']:.3f} ({result['sqi_grade']})")
        print(f"  心率:    {result['hr_bpm']:.1f} BPM")
        print(f"  搏动数:  {result['n_beats']}")
        print(f"  ΔF:      [{', '.join(f'{v:+.3f}' for v in result['delta_F'])}]")
        print(f"  解读:    {'; '.join(result['descriptions'])}")

        if result['syndrome_main']:
            main = result['syndrome_main']
            print(f"  主证:    {main.get('证型', main.get('name', '?'))} "
                  f"(置信度: {main.get('置信度', 0):.2f})")
        if result['syndrome_sub']:
            for sub in result['syndrome_sub'][:2]:
                print(f"  兼证:    {sub.get('证型', sub.get('name', '?'))} "
                      f"(置信度: {sub.get('置信度', 0):.3f})")
        if result['syndrome_combo']:
            print(f"  组合:    {'; '.join(result['syndrome_combo'])}")
        print(f"  摘要:    {result['summary']}")

        print(f"\n  六品质:")
        for k, v in result['six_qualities'].items():
            bar = '█' * int(v * 20) + '░' * int((1 - v) * 20)
            print(f"    {k:20s}: {v:.3f}  {bar}")

    # ── 测试 3: ECG+PPG 联合增强 ──
    print("\n[测试 3] ECG+PPG 联合增强（模拟）")
    ecg_delta = [0.1, 0.3, -0.1, -0.2, 0.0]
    ppg_sq = test_sq
    enhanced, details = enhance_ecg_deltaF(ecg_delta, ppg_sq, ppg_confidence=0.8)
    print(f"  ECG ΔF:  [{', '.join(f'{v:+.3f}' for v in ecg_delta)}]")
    print(f"  PPG ΔF:  [{', '.join(f'{v:+.3f}' for v in six_qualities_to_deltaF(ppg_sq))}]")
    print(f"  增强 ΔF: [{', '.join(f'{v:+.3f}' for v in enhanced)}]")

    print(f"\n✅ 自测完成")


if __name__ == '__main__':
    main()
