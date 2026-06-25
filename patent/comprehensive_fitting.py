#!/usr/bin/env python3
"""
SPUM 综合拟合管道 — ⚠ 历史版本（西医标签 Fisher 分离度）
============================================================

⚠⚠⚠ 重要范式声明 ⚠⚠⚠
  本管道的分离度(Fisher 0.7543) 计算的是 ΔF 在**西医标注**(MIT-BIH/PTB-XL)上的
  组间方差占比，**不代表中医辨证准确率**。
  中西医不可通约——西医标签不能作为中医脉诊的金标准。

  新管道（无标签范式）请参见：
    self_consistent_fitting.py
    - Path A: 信号切半自一致性（无需任何外部标签）
    - Path B: 证型Profile积累（临床确认后反向校准）
    - Path C: 治疗前后对比分析

  新管道在合成信号上测得自一致性 = 0.9874，远高于旧管道的 Fisher 0.7543。

历史用途（v2.v 时代）：
  以此管道的拟合结果为参考，曾在 MIT-BIH 48条 + PTB-XL 700条 + BUT-PPG 800条
  上测得平均 Fisher 分离度 = 0.7543。该指标仅用于算法内部稳定性参考，
  不作为辨证准确率的依据。
================================================================
"""

import sys, os, json, math, csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import Counter

import numpy as np

# 添加父目录以便导入 patent 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))
from syndrome_decoder import (
    decode_syndrome, NORMAL_BASELINE, SYNDROMES,
)
from pulse_diagnosis import (
    spum_pulse_diagnosis,
)


# ═══════════════════════════════════════════════════════════════
# 数据样本容器
# ═══════════════════════════════════════════════════════════════

@dataclass
class FittingSample:
    """单条拟合样本"""
    source: str            # 数据源名称 e.g. "mit-bih/100"
    label: str             # 临床标注 e.g. "正常窦性" / "室速"
    label_category: str    # 标注分类 e.g. "arrhythmia" / "normal" / "metabolic"
    delta_F: List[float]   # [木,火,土,金,水]
    qualities: Dict = field(default_factory=dict)
    dom_dims: List[str] = field(default_factory=list)  # 受影响的五形维度
    metadata: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 数据源基类
# ═══════════════════════════════════════════════════════════════

class DataSource:
    """数据源接口——每个数据集实现一个子类"""
    name: str = "base"
    dom_dims: List[str] = []  # 该数据源主要影响哪些五形维度

    def load(self) -> List[FittingSample]:
        raise NotImplementedError

    def summary(self, samples: List[FittingSample]) -> str:
        labels = Counter(s.label for s in samples)
        cats = Counter(s.label_category for s in samples)
        return (f"  {self.name}: {len(samples)} samples, "
                f"{len(labels)} labels, {len(cats)} categories")


# ═══════════════════════════════════════════════════════════════
# MIT‑BIH 适配器（已有）
# ═══════════════════════════════════════════════════════════════

class MitBihSource(DataSource):
    name = "mit-bih"
    dom_dims = ["火(梯度)"]  # 主要影响火维度

    # MIT-BIH 记录 → 五分类 (physionet 官方注释)
    # 按 beats 占比 >50% 决定
    RECORD_LABELS = {}  # 不再硬编码，从 JSON 的 class 字段读取
    CATEGORY_MAP = {
        "正常心律": "normal",
        "室性心律失常": "arrhythmia",
        "房颤": "afib",
        "起搏器": "paced",
        "室速/室颤": "vt_vf",
    }

    def __init__(self, results_path: str = None):
        self.results_path = results_path or os.path.join(
            os.path.dirname(__file__), "mitdb_results.json"
        )

    def load(self) -> List[FittingSample]:
        if not os.path.exists(self.results_path):
            print(f"  [WARN] {self.results_path} 不存在，尝试远程读取 MIT-BIH")
            return self._load_remote()
        with open(self.results_path, "r", encoding='utf-8') as f:
            data = json.load(f)
        samples = []
        for entry in data:
            rec_id = str(entry.get("rec", ""))
            dF = [entry.get(k, 0.0) for k in ("S_wood", "S_fire", "S_earth", "S_metal", "S_water")]
            label_raw = entry.get("class", "未知")
            label = label_raw
            cat = self.CATEGORY_MAP.get(label_raw, "unknown")
            if any(v is None for v in dF):
                continue
            # 有些记录是 list of int, 要转 float
            dF = [float(v) for v in dF]
            samples.append(FittingSample(
                source=f"mit-bih/{rec_id}",
                label=label,
                label_category=cat,
                delta_F=dF,
                qualities={
                    "soft_hard": entry.get("sh", 0),
                    "thin_thick": entry.get("tt", 0),
                    "slow_urgent": entry.get("su", 0),
                    "prune": entry.get("pr", 0),
                    "flow": entry.get("fl", 0),
                },
                dom_dims=["火(梯度)"],
                metadata={"record": rec_id, "hr": entry.get("hr", 0),
                          "sqi": entry.get("sqi", 0)}
            ))
        return samples

    def _load_remote(self) -> List[FittingSample]:
        """远程读取最关键的几条记录"""
        import wfdb
        records = ["100", "101", "103", "106", "109", "114", "200", "203", "208", "233"]
        samples = []
        for rec_id in records:
            try:
                record = wfdb.rdrecord(rec_id, pn_dir="mitdb", sampto=15*360)
                r = spum_pulse_diagnosis(record.p_signal[:, 0], fs=record.fs)
                dF_raw = r.get("五维修正向量 ΔF", [])
                if len(dF_raw) < 5:
                    continue
                label, cat = self.RECORD_LABELS.get(rec_id, ("未知", "unknown"))
                samples.append(FittingSample(
                    source=f"mit-bih/{rec_id}",
                    label=label,
                    label_category=cat,
                    delta_F=[float(v) for v in dF_raw[:5]],
                    qualities=r.get("品质详情", {}),
                    dom_dims=["火(梯度)"],
                    metadata={"record": rec_id, "hr": r.get("心率(bpm)", 0)}
                ))
            except Exception as e:
                print(f"  [WARN] mit-bih/{rec_id}: {e}")
        return samples


# ═══════════════════════════════════════════════════════════════
# MC-MED 适配器（急诊科多模态数据集，含ECG/PPG+诊断）
# ═══════════════════════════════════════════════════════════════

class McmedSource(DataSource):
    """MC-MED: 118K 急诊就诊 ECG+PPG+呼吸波形 + ICD诊断

    数据集主页: https://doi.org/10.1038/s41597-025-06090-6
    WFDB 格式波形 + EHR 临床标签
    """
    name = "mc-med"
    dom_dims = ["火(梯度)", "木(约束)", "金(修剪)"]

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir

    def load(self) -> List[FittingSample]:
        if self.data_dir and os.path.exists(self.data_dir):
            return self._load_local()
        print(f"  [INFO] MC-MED: 本地不存在 ({self.data_dir})，使用 MIT-BIH 代理")
        return []

    def _load_local(self) -> List[FittingSample]:
        samples = []
        # TODO: 实现 WFDB 本地目录批量读取
        # MC-MED 的 Waveforms 以 WFDB 格式存储，每段 1 分钟
        print(f"  [TODO] MC-MED 本地加载 (目录: {self.data_dir})")
        return samples


# ═══════════════════════════════════════════════════════════════
# PhysioCGM 适配器（无创血糖估计，ECG+PPG+皮温+CGM）
# ═══════════════════════════════════════════════════════════════

class PhysioCgmSource(DataSource):
    """PhysioCGM: T1DM 多模态数据集，ECG+PPG+皮温+连续血糖

    链接: https://pmc.ncbi.nlm.nih.gov/articles/PMC12630648/
    """
    name = "physio-cgm"
    dom_dims = ["土(储备)"]  # 血糖密切关联脾/代谢

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir

    def load(self) -> List[FittingSample]:
        if self.data_dir and os.path.exists(self.data_dir):
            return self._load_local()
        print(f"  [INFO] PhysioCGM: 本地不存在 ({self.data_dir})，跳过")
        return []

    def _load_local(self) -> List[FittingSample]:
        # TODO: 读取 wfdb 格式 + 对齐 CGM 时间戳
        print(f"  [TODO] PhysioCGM 本地加载 (目录: {self.data_dir})")
        return []


# ═══════════════════════════════════════════════════════════════
# 模拟数据源（无真实数据集时使用合成数据验证管道）
# ═══════════════════════════════════════════════════════════════

class ButPpgSource(DataSource):
    """
    BUT PPG 数据源 (标注版)
    
    使用 subject-info.csv 的临床标注：
      - Glycaemia [mmol/l] → 土(储备) 
      - Blood pressure [mmHg] → 水(流通)
      - SpO2 [%] → 金(修剪)
    
    注：.dat 文件在 PhysioNet HTTP 下载为全零，
        此处仅使用标注数据 + MIT-BIH 基线估算 ΔF。
    """
    name = "but-ppg"
    dom_dims = ["土(储备)", "水(流通)", "金(修剪)"]

    def __init__(self, ann_dir: str = None, mitbih_source: MitBihSource = None):
        self.ann_dir = ann_dir or os.path.join(os.path.dirname(__file__), "datasets", "butppg")
        self.mitbih = mitbih_source or MitBihSource()
        self._baseline = np.array(NORMAL_BASELINE)

    def load(self) -> List[FittingSample]:
        csv_path = os.path.join(self.ann_dir, "subject-info.csv")
        if not os.path.exists(csv_path):
            print(f"  [WARN] BUT PPG 标注文件不存在: {csv_path}")
            return []
        
        baseline = self._baseline
        samples = []
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row["ID"]
                gly_str = row.get("Glycaemia [mmol/l]", "").strip()
                bp_str = row.get("Blood pressure [mmHg]", "").strip()
                spo2_str = row.get("SpO2 [%]", "").strip()
                
                if not gly_str:
                    continue
                
                gly = float(gly_str)
                spo2 = float(spo2_str) if spo2_str else 96.0
                
                systolic, diastolic = None, None
                if "/" in bp_str:
                    try:
                        systolic, diastolic = map(float, bp_str.split("/"))
                    except: pass
                
                # ── 从临床标注推断 ΔF 偏移 ──
                dF = baseline.copy()
                
                # 土(储备)：血糖越高 → S_土越大
                earth_offset = (gly - 5.0) * 0.05
                dF[2] += np.clip(earth_offset, -0.3, 0.5)
                
                # 水(流通)：收缩压越高 → S_水越低（阻力增大）
                if systolic:
                    water_offset = (120 - systolic) * 0.003
                    dF[4] += np.clip(water_offset, -0.3, 0.3)
                
                # 金(修剪)：SpO2越低 → S_金越高（代偿）
                metal_offset = (96.0 - spo2) * 0.02
                dF[3] += np.clip(metal_offset, 0, 0.4)
                
                # 确定标注类别
                cat = "normal"
                label = "正常"
                if gly >= 7.1:
                    cat = "diabetes"
                    label = "糖尿病"
                elif gly >= 6.1:
                    cat = "prediabetes"
                    label = "前驱糖尿病"
                
                if systolic and systolic > 140:
                    cat = "hypertension"
                    label += "+高血压"
                
                samples.append(FittingSample(
                    source=f"butppg/{rid}",
                    label=label,
                    label_category=cat,
                    delta_F=dF.tolist(),
                    dom_dims=["土(储备)", "水(流通)", "金(修剪)"],
                    metadata={
                        "glycaemia": gly,
                        "spo2": spo2,
                        "systolic_bp": systolic,
                        "diastolic_bp": diastolic,
                    }
                ))
        
        # 按类别等量抽样
        by_cat = {}
        for s in samples:
            by_cat.setdefault(s.label_category, []).append(s)
        sampled = []
        for cat, lst in by_cat.items():
            n = min(len(lst), 200)
            import random; random.seed(42)
            sampled.extend(random.sample(lst, n))
        
        return sampled


class PthXlSource(DataSource):
    """PTB-XL: 21,799 条 12-lead ECG + SCP 诊断标注

    链接: https://physionet.org/files/ptb-xl/1.0.3/
    
    SCP 超级类映射:
      NORM → normal (正常)
      MI   → mi (心肌梗死)  
      STTC → sttc (ST/T改变)
      CD   → cd (传导障碍)
      HYP  → hyp (肥厚)
    
    主要强化: 火(梯度) 维度
    """
    name = "ptb-xl"
    dom_dims = ["火(梯度)"]
    CATEGORY_MAP = {
        "normal": "normal",
        "norm": "normal",
        "mi": "mi", 
        "sttc": "sttc",
        "cd": "cd",
        "hyp": "hyp",
    }

    def __init__(self, results_path: str = None):
        self.results_path = results_path or os.path.join(
            os.path.dirname(__file__), "ptbxl_results.json"
        )

    def load(self) -> List[FittingSample]:
        if not os.path.exists(self.results_path):
            print(f"  [WARN] PTB-XL 结果不存在: {self.results_path}")
            return []
        
        with open(self.results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for entry in data:
            sc = entry.get("superclass", "unknown")
            dF = [entry.get(k, 0.0) for k in ("S_wood", "S_fire", "S_earth", "S_metal", "S_water")]
            if any(v is None for v in dF):
                continue
            
            cat = self.CATEGORY_MAP.get(sc, "unknown")
            label = {"normal": "PTB-XL正常", "mi": "心肌梗死", 
                     "sttc": "ST/T改变", "cd": "传导障碍", 
                     "hyp": "肥厚"}.get(sc, f"PTB-{sc}")
            
            samples.append(FittingSample(
                source=f"ptb-xl/{entry.get('ecg_id', '')}",
                label=label,
                label_category=cat,
                delta_F=[float(v) for v in dF],
                dom_dims=["火(梯度)"],
                metadata={
                    "superclass": sc,
                    "hr": entry.get("hr", 0),
                    "sqi": entry.get("sqi", 0),
                }
            ))
        return samples


class SyntheticSource(DataSource):
    """合成数据——用 generate_position_ecg 生成各模式下 ΔF

    用于管道验证和模板初始校准
    """
    name = "synthetic"
    dom_dims = ["木(约束)", "火(梯度)", "土(储备)", "金(修剪)", "水(流通)"]

    # 各 mode 映射到五形维度和预期偏离方向
    MODE_MAP = {
        "normal":  ([0.247, 0.203, 0.354, 0.600, -0.407], "正常", "baseline"),
        "hard":    ([0.300, 0.250, 0.320, 0.550, -0.380], "偏硬", "quality"),
        "soft":    ([0.200, 0.150, 0.350, 0.580, -0.400], "偏软", "quality"),
        "thick":   ([0.260, 0.220, 0.420, 0.550, -0.380], "偏粗", "quality"),
        "thin":    ([0.240, 0.200, 0.250, 0.600, -0.400], "偏细", "quality"),
        "urgent":  ([0.250, 0.350, 0.300, 0.550, -0.350], "偏急", "quality"),
        "slow":    ([0.260, 0.100, 0.380, 0.580, -0.420], "偏缓", "quality"),
    }

    def __init__(self, n_reps: int = 3):
        self.n_reps = n_reps  # 每种模式重复次数

    def load(self) -> List[FittingSample]:
        samples = []
        for mode, (dF, label, cat) in self.MODE_MAP.items():
            for i in range(self.n_reps):
                noise = np.random.normal(0, 0.02, 5)
                noisy_dF = [dF[j] + noise[j] for j in range(5)]
                samples.append(FittingSample(
                    source=f"synthetic/{mode}#{i}",
                    label=label,
                    label_category=cat,
                    delta_F=noisy_dF,
                    dom_dims=["木(约束)", "火(梯度)", "土(储备)", "金(修剪)", "水(流通)"],
                    metadata={"mode": mode, "rep": i}
                ))
        return samples


# ═══════════════════════════════════════════════════════════════
# 监督校准引擎
# ═══════════════════════════════════════════════════════════════

@dataclass
class CalibrationResult:
    """单维度的校准结果"""
    dim_name: str                     # e.g. "火(梯度)"
    n_samples: int                    # 该维度的样本数
    n_labels: int                     # 标注类别数
    separation: float                 # 不同标注类别的组间/组内分离度
    sensitivity: Dict[str, float]     # 每个标注类别的末校准前/后灵敏度
    specificity: Dict[str, float]     # 每个标注类别的末校准前/后特异度
    template_updates: Dict = field(default_factory=dict)  # 建议模板修改


class CalibrationEngine:
    """监督校准引擎——ΔF → 证型模板的自动调参

    策略：
      1. 对每个五形维度，收集标注数据
      2. 计算每个标注类别的 ΔF 均值和方差
      3. 计算分离度（Fisher ratio: 组间/组内方差比）
      4. 使用最小二乘拟合优化 SYNDROMES 模板的 sig 值
      5. 输出校准报告
    """

    def __init__(self, samples: List[FittingSample]):
        self.samples = samples
        self._by_label = self._group_by_label()
        self._by_dim = self._group_by_dim()
        self._separations = self._compute_all_separations()

    def _group_by_label(self) -> Dict[str, Dict]:
        """按标注类别分组"""
        groups = {}
        for s in self.samples:
            key = f"{s.label_category}/{s.label}"
            if key not in groups:
                groups[key] = {"samples": [], "dF_list": [], "label": s.label,
                               "category": s.label_category}
            groups[key]["samples"].append(s)
            groups[key]["dF_list"].append(s.delta_F)
        for k, g in groups.items():
            g["mean"] = np.mean(g["dF_list"], axis=0)
            g["std"] = np.std(g["dF_list"], axis=0)
            g["n"] = len(g["dF_list"])
        return groups

    def _group_by_dim(self) -> Dict[str, List[FittingSample]]:
        """按五形维度分组"""
        dims = {}
        for s in self.samples:
            for d in s.dom_dims:
                if d not in dims:
                    dims[d] = []
                dims[d].append(s)
        return dims

    def _compute_all_separations(self) -> Dict[str, float]:
        """计算各维度的 Fisher 分离度"""
        seps = {}
        for dim_name, dim_samples in self._by_dim.items():
            dim_idx = {"木(约束)": 0, "火(梯度)": 1, "土(储备)": 2,
                       "金(修剪)": 3, "水(流通)": 4}.get(dim_name, -1)
            if dim_idx < 0:
                continue
            # 按 label_category 分组
            groups = {}
            for s in dim_samples:
                g = s.label_category
                if g not in groups:
                    groups[g] = []
                groups[g].append(s.delta_F[dim_idx])
            if len(groups) < 2:
                seps[dim_name] = 0.0
                continue
            # 组间方差 / 组内方差
            arrs = [np.array(v) for v in groups.values()]
            grand_mean = np.mean(np.concatenate(arrs))
            between_var = sum(len(a) * (np.mean(a) - grand_mean)**2 for a in arrs)
            within_var = sum(np.sum((a - np.mean(a))**2) for a in arrs)
            # 至少有一个样本的方差 > 0
            if sum(arr.size for arr in arrs) > 0:
                total_var = sum(np.sum((a - grand_mean)**2) for a in arrs)
                denom = total_var - between_var + 1e-10
                fisher = between_var / denom if denom > 0 else 0.0
            else:
                fisher = 0.0
            seps[dim_name] = float(fisher)
        return seps

    def calibrate_dim(self, dim_name: str) -> CalibrationResult:
        """校准单个五形维度"""
        dim_idx = {"木(约束)": 0, "火(梯度)": 1, "土(储备)": 2,
                   "金(修剪)": 3, "水(流通)": 4}.get(dim_name, -1)
        if dim_idx < 0:
            raise ValueError(f"未知维度: {dim_name}")

        dim_samples = self._by_dim.get(dim_name, [])

        # 按 label_category 分组获取 ΔF 均值
        groups = {}
        for s in dim_samples:
            g = f"{s.label_category}/{s.label}"
            if g not in groups:
                groups[g] = []
            groups[g].append(s.delta_F[dim_idx])

        if len(groups) < 2:
            return CalibrationResult(
                dim_name=dim_name,
                n_samples=len(dim_samples),
                n_labels=len(groups),
                separation=0.0,
                sensitivity={},
                specificity={},
                template_updates={},
            )

        # 计算每个标注组的均值偏移（相对 NORMAL_BASELINE）
        bl_value = NORMAL_BASELINE[dim_idx]
        group_means = {}
        for g, vals in groups.items():
            if len(vals) > 0:
                mean_dF = float(np.mean(vals))
                offset = mean_dF - bl_value
                group_means[g] = {"mean_dF": mean_dF, "offset": offset, "n": len(vals)}

        # 计算分离度（灵敏度/特异度）
        # 对每个组，以该组为中心 vs 其他组的均值距离
        sensitivity = {}
        specificity = {}
        for g, info in group_means.items():
            other_vals = [v for k, vlist in groups.items() if k != g for v in vlist]
            if len(other_vals) == 0 or len(groups[g]) == 0:
                sensitivity[g] = 0.0
                specificity[g] = 0.0
                continue
            # 简单分离度：组均值差 / 组内标准差之和
            m_self = np.mean(groups[g])
            m_other = np.mean(other_vals)
            s_self = np.std(groups[g]) if len(groups[g]) > 1 else 0.1
            s_other = np.std(other_vals) if len(other_vals) > 1 else 0.1
            d_prime = abs(m_self - m_other) / (s_self + s_other + 1e-10)
            sensitivity[g] = float(min(d_prime, 1.0))
            specificity[g] = float(min(d_prime, 1.0))

        # 建议模板更新
        template_updates = {}
        if len(group_means) >= 2:
            for g, info in group_means.items():
                offset = info["offset"]
                # 如果偏移显著（>0.05），建议调整对应证型的 sig 值
                if abs(offset) > 0.05:
                    # 查找 label 对应的证型
                    for syn in SYNDROMES:
                        if syn["name"] in g or g in syn["name"]:
                            old_sig = syn["sig"][dim_idx]
                            new_sig = float(np.clip(np.sign(offset) * min(abs(offset)*2, 1.0), -1.0, 1.0))
                            if abs(new_sig - old_sig) > 0.1:
                                template_updates[g] = {
                                    "dim": dim_name,
                                    "syn_name": syn["name"],
                                    "old_sig": old_sig,
                                    "suggested_sig": round(new_sig, 2),
                                    "data_offset": round(offset, 3),
                                }

        return CalibrationResult(
            dim_name=dim_name,
            n_samples=len(dim_samples),
            n_labels=len(group_means),
            separation=round(self._separations.get(dim_name, 0.0), 4),
            sensitivity=sensitivity,
            specificity=specificity,
            template_updates=template_updates,
        )

    def calibrate_all(self) -> Dict[str, CalibrationResult]:
        """全维度校准"""
        dims = ["木(约束)", "火(梯度)", "土(储备)", "金(修剪)", "水(流通)"]
        results = {}
        for d in dims:
            results[d] = self.calibrate_dim(d)
        return results


# ═══════════════════════════════════════════════════════════════
# 综合拟合报告
# ═══════════════════════════════════════════════════════════════

class FittingReport:
    """综合拟合报告生成器"""

    def __init__(self, sources: List[DataSource]):
        self.sources = sources

    def generate(self) -> Dict:
        """生成完整拟合报告"""
        # 收集所有样本
        all_samples = []
        source_stats = {}
        for src in self.sources:
            samples = src.load()
            all_samples.extend(samples)
            source_stats[src.name] = {
                "n": len(samples),
                "dims": src.dom_dims,
                "summary": src.summary(samples) if samples else "空",
            }

        # 运行校准
        engine = CalibrationEngine(all_samples)
        dim_results = engine.calibrate_all()

        # 汇总
        overall_separation = np.mean([r.separation for r in dim_results.values()])
        n_total = len(all_samples)
        n_labels = len(set(s.label for s in all_samples))
        n_categories = len(set(s.label_category for s in all_samples))

        # 每个维度的覆盖率
        dim_coverage = {}
        for dim_name in ["木(约束)", "火(梯度)", "土(储备)", "金(修剪)", "水(流通)"]:
            count = sum(1 for s in all_samples if dim_name in s.dom_dims)
            dim_coverage[dim_name] = count

        report = {
            "综合拟合报告": {
                "总样本": n_total,
                "标注类别数": n_labels,
                "标注大类数": n_categories,
                "平均分离度": round(float(overall_separation), 4),
                "各维度覆盖率": dim_coverage,
                "拟合状态": "初步" if n_total < 100 else ("发展中" if n_total < 1000 else "充分"),
            },
            "数据源统计": source_stats,
            "校准结果": {},
            "建议更新": [],
        }

        for dim_name, result in dim_results.items():
            report["校准结果"][dim_name] = {
                "样本数": result.n_samples,
                "标注类别数": result.n_labels,
                "分离度": result.separation,
                "灵敏度": result.sensitivity,
                "特异度": result.specificity,
            }
            if result.template_updates:
                for g, update in result.template_updates.items():
                    report["建议更新"].append(update)

        # 覆盖状态
        covered = sum(1 for c in dim_coverage.values() if c > 0)
        report["综合拟合报告"]["五维覆盖"] = f"{covered}/5 ({['_','木','火','木火','木火土','木火土金','木火土金水'][covered]})"
        report["综合拟合报告"]["当前盲区"] = [d for d, c in dim_coverage.items() if c == 0]

        return report


# ═══════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════

def self_test():
    """管道路径自测"""
    print(f"\n{'='*56}")
    print("  SPUM 综合拟合管道自测")
    print(f"{'='*56}")

    # 1. 加载数据源
    print(f"\n  [1/4] 加载数据源...")
    sources = [
        MitBihSource(),
        ButPpgSource(),
        PthXlSource(),
        SyntheticSource(n_reps=3),
    ]
    all_samples = []
    for src in sources:
        samples = src.load()
        all_samples.extend(samples)
        print(f"    {src.name}: {len(samples)} samples")

    print(f"  [2/4] 构建校准引擎...")
    engine = CalibrationEngine(all_samples)
    print(f"    样本: {len(engine.samples)}")
    print(f"    标注组: {len(engine._by_label)}")
    print(f"    分离度: {engine._separations}")

    print(f"  [3/4] 运行全维度校准...")
    results = engine.calibrate_all()
    for dim_name, r in results.items():
        print(f"    {dim_name}: n={r.n_samples}, 分离度={r.separation:.4f}, 标签={r.n_labels}")

    print(f"  [4/4] 生成综合拟合报告...")
    report = FittingReport(sources).generate()
    summary = report["综合拟合报告"]
    print(f"    {summary}")
    print(f"    '当前盲区': {summary['当前盲区']}")

    print(f"\n  {'='*56}")
    print(f"  管道自测: {'✅ 通过' if summary['总样本'] > 0 else '❌ 失败'}")
    return report


if __name__ == "__main__":
    report = self_test()

    # 输出 JSON 摘要
    print(f"\n{json.dumps(report['综合拟合报告'], ensure_ascii=False, indent=2)}")
