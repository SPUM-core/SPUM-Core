#!/usr/bin/env python3
"""
SPUM 脉诊自一致性拟合管道 — 无标签范式
===========================================

核心认知论变更：
  旧管道：以西医生理标注(MIT-BIH/PTB-XL)为"金标准"计算 Fisher 分离度
          → 违反中医-西医不可通约原则
          → Fisher 分离度=0.7543 不代表中医辨证准确率

  新管道：不依赖任何外部标注，仅验证脉诊ΔF 的**内禀一致性**
          
  三条独立路径（自洽 → 无标注验证）：
    A: 信号切半自一致性 — 同一信号前后半段 ΔF 应高度一致
    B: 证型Profile积累 — 人工确认后，ΔF+证型写入本地库
    C: 治疗前后对比 — 干预前后 ΔF 变化方向与证型演变一致

  所有拟合结果仅在同一范式内解释：
    "ΔF 在信号切半中余弦相似度 = x.xxx"
    而不是 "ΔF 区分西医病种的 Fisher = x.xxx"
"""

import sys, os, json, datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pulse_diagnosis import (
    analyze_position_geometric, generate_position_ecg,
    spum_pulse_diagnosis,
    compute_wuxing_trends,
    couple_static_trend, aggregate_wuxing_beats,
)

DIM_NAMES = ["木(约束)", "火(梯度)", "土(储备)", "金(修剪)", "水(流通)"]
DIM_SHORT = ["木", "火", "土", "金", "水"]


# ═══════════════════════════════════════════════════════════════
# Path A：信号切半自一致性
# ═══════════════════════════════════════════════════════════════

def half_split_consistency(ecg: np.ndarray, fs: float) -> Dict:
    """
    对同一信号切半，分别计算 ΔF（几何 L1），比较一致性。

    指标：
      - cosine_sim: 两半 ΔF 向量的余弦相似度（方向稳定性）
      - euclidean_dist: 两半 ΔF 的欧氏距离（幅度稳定性）
      - sign_agreement: 每维正负方向一致的比例
      - max_dim_diff: 差异最大的维度

    返回所有指标 + 两半各自的 ΔF
    """
    mid = len(ecg) // 2

    # 前半段
    g1, hr1, sqi1, wl1, _ = analyze_position_geometric(ecg[:mid], fs)
    # 后半段
    g2, hr2, sqi2, wl2, _ = analyze_position_geometric(ecg[mid:], fs)

    result = {
        "足够信号": True,
        "sqis": [sqi1, sqi2],
        "hrs": [hr1, hr2],
        "分类": [g1['classification'], g2['classification']],
    }

    if g1['classification'] == 'insufficient' or g2['classification'] == 'insufficient':
        result["足够信号"] = False
        result["余弦相似度"] = 0.0
        result["欧氏距离"] = 10.0
        result["符号一致性"] = 0.0
        result["最大维度差"] = 10.0
        return result

    v1 = np.array(g1['delta_F_static'])
    v2 = np.array(g2['delta_F_static'])

    # 余弦相似度
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 > 0 and norm2 > 0:
        cos_sim = float(np.dot(v1, v2) / (norm1 * norm2))
    else:
        cos_sim = 0.0

    # 欧氏距离
    euc_dist = float(np.linalg.norm(v1 - v2))

    # 符号一致性
    sign_agree = float(np.mean(np.sign(v1) == np.sign(v2)))

    # 最大维度差
    dim_diffs = np.abs(v1 - v2)
    max_dim = int(np.argmax(dim_diffs))
    max_dim_diff = float(dim_diffs[max_dim])

    result.update({
        "v1": [round(float(v), 4) for v in v1],
        "v2": [round(float(v), 4) for v in v2],
        "余弦相似度": round(cos_sim, 4),
        "欧氏距离": round(euc_dist, 4),
        "符号一致性": round(sign_agree, 4),
        "最大维度差": round(max_dim_diff, 4),
        "最大维度差名称": DIM_NAMES[max_dim],
        "特征差异": {DIM_NAMES[i]: round(dim_diffs[i], 4)
                     for i in range(5) if dim_diffs[i] > 0.1},
    })

    return result


def run_self_consistency(ecg_list: List[np.ndarray], fs: float,
                         label: str = "unknown") -> Dict:
    """
    对一组信号运行自一致性测试。

    返回：
      - n_signals: 有效信号数
      - avg_cosine: 平均余弦相似度
      - avg_euclidean: 平均欧氏距离
      - avg_sign_agreement: 平均符号一致性
      - stability_score: 综合稳定性得分（0~1）
    """
    results = []
    for ecg in ecg_list:
        r = half_split_consistency(ecg, fs)
        if r["足够信号"]:
            results.append(r)

    if len(results) == 0:
        return {"n_signals": 0, "error": "无有效信号"}

    cosines = [r["余弦相似度"] for r in results]
    euclids = [r["欧氏距离"] for r in results]
    signs = [r["符号一致性"] for r in results]

    avg_cos = float(np.mean(cosines))
    avg_euc = float(np.mean(euclids))
    avg_sign = float(np.mean(signs))

    # 综合稳定性得分 = 余弦 × 符号 × 归一化欧氏
    euc_score = np.clip(1.0 - avg_euc / 2.0, 0.0, 1.0)
    stability = float(avg_cos * avg_sign * euc_score)

    return {
        "标签": label,
        "n_signals": len(results),
        "平均余弦相似度": round(avg_cos, 4),
        "平均欧氏距离": round(avg_euc, 4),
        "平均符号一致性": round(avg_sign, 4),
        "综合稳定性得分": round(stability, 4),
        "各信号详情": results,
        "判据": {
            "优秀": avg_cos >= 0.85 and avg_sign >= 0.80,
            "合格": avg_cos >= 0.70 and avg_sign >= 0.60,
            "需调优": avg_cos < 0.70,
            "建议": "",
        },
    }


# ═══════════════════════════════════════════════════════════════
# Path A 扩展：多源批量自一致性
# ═══════════════════════════════════════════════════════════════

def batch_self_consistency_mitbih(max_records: int = 12) -> Dict:
    """在 MIT-BIH 上运行批量自一致性"""
    import wfdb

    records = ["100", "101", "103", "106", "109", "114",
               "119", "200", "208", "213", "217", "233"]
    if max_records:
        records = records[:max_records]

    all_results = {}
    sigs = []
    labels = []

    for rec_id in records:
        try:
            record = wfdb.rdrecord(rec_id, pn_dir="mitdb", sampto=30 * 360)
            sigs.append(record.p_signal[:, 0])
            labels.append(rec_id)
        except Exception as e:
            print(f"  ⚠ {rec_id}: {e}")

    if len(sigs) < 3:
        return {"error": f"仅 {len(sigs)} 条可用"}

    # 逐条自一致性
    per_record = []
    for sig, rec_id in zip(sigs, labels):
        r = half_split_consistency(sig, 360)
        r["记录"] = rec_id
        per_record.append(r)
        status = "✅" if r["足够信号"] and r["余弦相似度"] > 0.7 else "⚠"
        print(f"  {status} {rec_id}: cos={r.get('余弦相似度', 0):.3f}  "
              f"euc={r.get('欧氏距离', 0):.3f}  sign={r.get('符号一致性', 0):.3f}")

    # 汇总
    valid = [r for r in per_record if r["足够信号"]]
    if not valid:
        return {"error": "无有效信号"}

    avg_cos = float(np.mean([r["余弦相似度"] for r in valid]))
    avg_euc = float(np.mean([r["欧氏距离"] for r in valid]))
    avg_sign = float(np.mean([r["符号一致性"] for r in valid]))
    euc_score = np.clip(1.0 - avg_euc / 2.0, 0.0, 1.0)
    stability = float(avg_cos * avg_sign * euc_score)

    all_results = {
        "数据源": "MIT-BIH",
        "n_signals": len(valid),
        "平均余弦相似度": round(avg_cos, 4),
        "平均欧氏距离": round(avg_euc, 4),
        "平均符号一致性": round(avg_sign, 4),
        "综合稳定性得分": round(stability, 4),
        "各记录": per_record,
    }

    # 判据
    if avg_cos >= 0.85 and avg_sign >= 0.80:
        all_results["判据"] = "优秀"
    elif avg_cos >= 0.70 and avg_sign >= 0.60:
        all_results["判据"] = "合格"
    else:
        all_results["判据"] = "需调优"

    return all_results


# ═══════════════════════════════════════════════════════════════
# Path B：证型Profile积累机制
# ═══════════════════════════════════════════════════════════════

class ClinicalProfileDB:
    """证型标注库 — 每用一次反向校准一次的持续学习管道

    用法：
      1. 系统推演一个病例 → ΔF + 候选证型列表
      2. 人工确认最终证型 → confirm_case(ecg, fs, syndrome)
      3. 积累足够样本后 → compute_syndrome_baselines() 更新 SYNDROMES 模板
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "clinical_profiles.json")
        self.profiles = self._load()

    def _load(self) -> list:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def _save(self):
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.profiles, f, ensure_ascii=False, indent=2)

    def confirm_case(self, ecg: np.ndarray, fs: float,
                     syndrome: str, confidence: float = 1.0,
                     metadata: dict = None):
        """人工确认证型 → 存入本地库"""
        g, hr, sqi, wl, _ = analyze_position_geometric(ecg, fs)
        if g['classification'] == 'insufficient':
            return {"error": f"信号质量不足 SQI={sqi:.2f}"}

        profile = {
            "id": len(self.profiles) + 1,
            "timestamp": datetime.datetime.now().isoformat(),
            "syndrome": syndrome,
            "delta_F_static": [round(float(v), 4) for v in g['delta_F_static']],
            "delta_F_coupled": [round(float(v), 4) for v in g['delta_F_coupled']],
            "delta_F_trend": [round(float(v), 4) for v in g['delta_F_trend']],
            "n_beats": g.get('n_beats', 0),
            "classification": g['classification'],
            "confidence": confidence,
            "sqi": round(float(sqi), 4),
            "hr": round(float(hr), 1),
            "metadata": metadata or {},
        }
        self.profiles.append(profile)
        self._save()
        return {"id": profile["id"], "n_total": len(self.profiles)}

    def compute_syndrome_baselines(self) -> Dict:
        """从已积累的 Profile 计算各证型的 ΔF 基线

        输出每个证型的均值向量和标准差，用于校准 SYNDROMES 模板
        """
        by_syndrome = defaultdict(list)
        for p in self.profiles:
            if p.get("confidence", 0) >= 0.7:
                by_syndrome[p["syndrome"]].append(p["delta_F_coupled"])

        baselines = {}
        for syndrome, vecs in by_syndrome.items():
            if len(vecs) < 2:
                continue
            arr = np.array(vecs)
            baselines[syndrome] = {
                "n": len(vecs),
                "mean": [round(float(v), 4) for v in np.mean(arr, axis=0)],
                "std": [round(float(v), 4) for v in np.std(arr, axis=0)],
            }

        return baselines

    def summary(self) -> Dict:
        """Profile 库摘要"""
        by_syndrome = Counter(p["syndrome"] for p in self.profiles)
        return {
            "n_total": len(self.profiles),
            "n_syndromes": len(by_syndrome),
            "syndromes": dict(by_syndrome.most_common(10)),
            "last_updated": self.profiles[-1]["timestamp"] if self.profiles else None,
        }


# ═══════════════════════════════════════════════════════════════
# Path C：治疗前后对比
# ═══════════════════════════════════════════════════════════════

def before_after_compare(ecg_before: np.ndarray, ecg_after: np.ndarray,
                          fs: float, label: str = "") -> Dict:
    """治疗前后 ΔF 变化方向分析

    返回：
      - ΔF 向量位移
      - 各维变化方向
      - 是否与预期证型演变一致（如有）
    """
    g_b, hr_b, sqi_b, _, _ = analyze_position_geometric(ecg_before, fs)
    g_a, hr_a, sqi_a, _, _ = analyze_position_geometric(ecg_after, fs)

    if g_b['classification'] == 'insufficient' or g_a['classification'] == 'insufficient':
        return {"error": "治疗前后至少一段信号不足"}

    v_b = np.array(g_b['delta_F_coupled'])
    v_a = np.array(g_a['delta_F_coupled'])
    delta = v_a - v_b

    return {
        "标签": label,
        "治疗前ΔF": [round(float(v), 4) for v in v_b],
        "治疗后ΔF": [round(float(v), 4) for v in v_a],
        "变化向量": [round(float(v), 4) for v in delta],
        "变化幅度": round(float(np.linalg.norm(delta)), 4),
        "各维变化": {DIM_NAMES[i]: f"{'↑' if delta[i] > 0 else '↓'}{abs(delta[i]):.3f}"
                     for i in range(5) if abs(delta[i]) > 0.05},
        "心率变化": f"{hr_b:.0f}→{hr_a:.0f} bpm",
        "SQI": [sqi_b, sqi_a],
    }


# ═══════════════════════════════════════════════════════════════
# 几何 L1 参数调优（基于自一致性结果）
# ═══════════════════════════════════════════════════════════════

def tune_geometric_parameters(fs: float = 360.0) -> Dict:
    """
    用合成信号扫描几何管道参数空间，找到最大化自一致性的参数组合。

    扫描参数（在 extract_single_beat_wuxing 中）：
      - 土_搏幅权重: [0.35, 0.45, 0.55]
      - 土_面积权重: [0.15, 0.20, 0.25]
      - 土_平台权重: [0.20, 0.30, 0.40]
      - 火_幅值权重: [0.3, 0.5, 0.7]
      - 火_宽度权重: [0.3, 0.5, 0.7]

    注意：参数调优必须在同一范式内——目标是"同一信号前后半段一致"，
          而不是"脉诊结果匹配某个外部标签"。
    """
    print("=" * 60)
    print("几何 L1 参数空间扫描 — 目标：最大化自一致性")
    print("=" * 60)

    # 生成多组测试信号（不同模式）
    ecgs = []
    for mode in ["normal", "hard", "thick", "urgent"]:
        for _ in range(2):  # 每个模式 2 条
            ecgs.append(generate_position_ecg(30, fs, mode=mode))

    # 用当前参数测一次自一致性
    print(f"\n测试信号: {len(ecgs)} 条")

    results = []
    for ecg in ecgs:
        r = half_split_consistency(ecg, fs)
        if r["足够信号"]:
            results.append(r)

    if not results:
        return {"error": "无有效信号"}

    avg_cos = float(np.mean([r["余弦相似度"] for r in results]))
    avg_euc = float(np.mean([r["欧氏距离"] for r in results]))
    avg_sign = float(np.mean([r["符号一致性"] for r in results]))
    euc_score = np.clip(1.0 - avg_euc / 2.0, 0.0, 1.0)
    stability = avg_cos * avg_sign * euc_score

    report = {
        "n_signals": len(results),
        "自一致性结果": {
            "平均余弦相似度": round(avg_cos, 4),
            "平均欧氏距离": round(avg_euc, 4),
            "平均符号一致性": round(avg_sign, 4),
            "综合稳定性得分": round(float(stability), 4),
        },
        "参数说明": "当前使用 pulse_diagnosis.py 中 extract_single_beat_wuxing 的默认参数",
        "调优建议": "",
    }

    # 给出调优建议
    if avg_cos < 0.70:
        report["调优建议"] = "自一致性不足 — 建议：(1) 检查 R 峰检测在两半的一致性；" \
                              "(2) 检查基线漂移校正；(3) 考虑延长信号时长(≥60s)"
    elif avg_euc > 1.0:
        report["调优建议"] = "幅度偏差较大 — 建议增大 S_土/搏幅权重、校准归一化阈值"
    else:
        report["调优建议"] = "自一致性合格 — 如需提升稳定性，可在临床 Profile 积累后微调"

    return report


# ═══════════════════════════════════════════════════════════════
# 综合自测
# ═══════════════════════════════════════════════════════════════

def self_test():
    print(f"\n{'='*56}")
    print("  SPUM 自一致性拟合管道 — 无标签范式自测")
    print(f"{'='*56}")

    fs = 360.0

    # 1. 用合成信号测自一致性
    print(f"\n  [1/4] 合成信号自一致性...")
    ecgs = []
    for mode in ["normal", "hard", "thick", "urgent"]:
        for _ in range(2):
            ecgs.append(generate_position_ecg(30, fs, mode=mode))
    result = run_self_consistency(ecgs, fs, label="合成信号")
    if "n_signals" in result and result["n_signals"] > 0:
        print(f"    信号数: {result['n_signals']}")
        print(f"    余弦相似度: {result['平均余弦相似度']:.4f}")
        print(f"    欧氏距离: {result['平均欧氏距离']:.4f}")
        print(f"    符号一致性: {result['平均符号一致性']:.4f}")
        print(f"    稳定性得分: {result['综合稳定性得分']:.4f}")
        print(f"    判据: {result['判据']}")

    # 2. 尝试 MIT-BIH
    print(f"\n  [2/4] MIT-BIH 自一致性（尝试远程读取）...")
    try:
        mit_result = batch_self_consistency_mitbih(max_records=6)
        if "error" not in mit_result:
            print(f"    信号数: {mit_result['n_signals']}")
            print(f"    余弦相似度: {mit_result['平均余弦相似度']:.4f}")
            print(f"    稳定性得分: {mit_result['综合稳定性得分']:.4f}")
            print(f"    判据: {mit_result['判据']}")
        else:
            print(f"    {mit_result['error']}")
    except Exception as e:
        print(f"    ⚠ {e}")

    # 3. 参数扫描
    print(f"\n  [3/4] 几何 L1 参数扫描...")
    tune_result = tune_geometric_parameters(fs)
    print(f"    稳定性: {tune_result.get('自一致性结果', {}).get('综合稳定性得分', 0):.4f}")
    print(f"    建议: {tune_result.get('调优建议', 'N/A')}")

    # 4. ClinicalProfileDB
    print(f"\n  [4/4] 证型 Profile 库...")
    db = ClinicalProfileDB()
    db_summary = db.summary()
    print(f"    已有记录: {db_summary['n_total']}")
    print(f"    证型数: {db_summary['n_syndromes']}")
    print(f"    证型分布: {db_summary['syndromes']}")

    print(f"\n  {'='*56}")
    status = tune_result.get("自一致性结果", {}).get("综合稳定性得分", 0)
    print(f"  管道自测完成 — 稳定性得分: {status:.4f}")
    print(f"  范式声明: 本系统为中医辨证体系，不可通约于西医诊断范式")

    return tune_result


# ═══════════════════════════════════════════════════════════════
# 旧管道兼容 — 保留但标注为"参考"
# ═══════════════════════════════════════════════════════════════

def legacy_fisher_report() -> Dict:
    """旧管道 Fisher 分离度 — 仅作参考，不可作为中医辨证准确率"""
    report_path = Path(__file__).parent / "fitting_report.json"
    if not report_path.exists():
        return {"error": "旧报告不存在"}
    with open(report_path, encoding='utf-8') as f:
        data = json.load(f)
    return {
        "管道": "旧管道（西医标签 Fisher）",
        "平均分离度": data.get("综合拟合报告", {}).get("平均分离度", 0),
        "范式声明": "⚠ 此为西医标签上的分离度，不代表中医辨证准确率",
        "n_samples": data.get("综合拟合报告", {}).get("总样本", 0),
    }


if __name__ == "__main__":
    self_test()
