#!/usr/bin/env python3
"""
大规模自一致性拟合管道
=======================
无标签范式 — 处理 40000+ 脉波信号，输出自一致性综合报告。
数据来源：PTB-XL (1455) + MIT-BIH (34) + BUT-PPG (10) + 大规模模拟

Output: fitting_report_large.json
"""
import sys, os, json, time, csv, numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pulse_diagnosis import generate_position_ecg
from self_consistent_fitting import half_split_consistency

# ══════════════════════════════════════════════════════════════
# 1. 读取真实数据
# ══════════════════════════════════════════════════════════════

def read_ptbxl_local(limit=None):
    """读取本地 PTB-XL 文件"""
    ecg_dir = os.path.join(os.path.dirname(__file__), "datasets", "ptbxl", "ecg")
    hea_files = sorted([f for f in os.listdir(ecg_dir) if f.endswith(".hea")])
    if limit: hea_files = hea_files[:limit]
    signals, meta = [], []
    for h in hea_files:
        try:
            dat = h.replace(".hea", ".dat")
            with open(os.path.join(ecg_dir, h)) as f:
                lines = f.readlines()
            hdr = lines[0].strip().split()
            n_sig, fs, n_samples = int(hdr[1]), float(hdr[2]), int(hdr[3])
            g = float(lines[1].split()[2].split("(")[0])
            b = float(lines[1].split()[2].split("(")[1].split(")")[0])
            raw = np.fromfile(os.path.join(ecg_dir, dat), dtype=np.int16)
            sig = raw.reshape(-1, n_sig)[:n_samples, 0]
            sig_mv = (sig.astype(np.float64) - b) / g if g != 0 else sig.astype(np.float64) / 1000.0
            if np.std(sig_mv) < 0.01: continue
            signals.append((sig_mv, fs, {"source": "ptbxl", "file": h.replace(".hea", "")}))
        except Exception:
            continue
    return signals

def read_mitbih_local(limit=None):
    mitdb = os.path.join(os.path.dirname(__file__), "..", "data", "mitdb")
    hea_files = sorted([f for f in os.listdir(mitdb) if f.endswith(".hea")])
    if limit: hea_files = hea_files[:limit]
    signals = []
    for h in hea_files:
        try:
            dat = h.replace(".hea", ".dat")
            with open(os.path.join(mitdb, h)) as f:
                lines = f.readlines()
            hdr = lines[0].strip().split()
            n_sig, fs = int(hdr[1]), float(hdr[2])
            raw = np.fromfile(os.path.join(mitdb, dat), dtype=np.int16)
            sig = raw.reshape(-1, n_sig)[:int(fs*60), 0].astype(np.float64) / 200.0
            if np.std(sig) < 0.01: continue
            signals.append((sig[:int(10*fs)], fs, {"source": "mitdb", "file": h.replace(".hea", "")}))
        except Exception:
            continue
    return signals

def read_butppg_local(limit=None):
    but_dir = os.path.join(os.path.dirname(__file__), "datasets", "butppg", "records")
    subs = sorted([d for d in os.listdir(but_dir) if os.path.isdir(os.path.join(but_dir, d))])
    if limit: subs = subs[:limit]
    signals = []
    for s in subs:
        sd = os.path.join(but_dir, s)
        ecg_files = sorted([f for f in os.listdir(sd) if f.endswith("_ECG.dat")])
        for ef in ecg_files:
            try:
                hea_file = ef.replace(".dat", ".hea")
                with open(os.path.join(sd, hea_file)) as f:
                    hdr_line = f.readline().strip().split()
                fs = float(hdr_line[2]) if len(hdr_line) > 2 else 1000.0
                raw = np.fromfile(os.path.join(sd, ef), dtype=np.int16)
                sig = raw.astype(np.float64) / 1000.0
                if np.std(sig) < 0.01: continue
                signals.append((sig[:int(10*fs)], fs, {"source": "butppg", "file": ef.replace(".dat", "")}))
            except Exception:
                continue
    return signals

# ══════════════════════════════════════════════════════════════
# 2. 流式处理
# ══════════════════════════════════════════════════════════════

DIM_NAMES = ["S_木", "S_火", "S_土", "S_金", "S_水"]

def process_signal(sig, fs, meta):
    """处理单个信号，返回 half-split consistency 结果"""
    r = half_split_consistency(sig, fs)
    r["meta"] = meta
    return r

def generate_report(all_results):
    """从全部结果生成综合报告"""
    valid = [r for r in all_results if r["足够信号"]]
    invalid = [r for r in all_results if not r["足够信号"]]
    n = len(valid)
    if n == 0: return {"error": "无有效信号"}

    cosines = np.array([r["余弦相似度"] for r in valid])
    euclids  = np.array([r["欧氏距离"] for r in valid])
    signs    = np.array([r["符号一致性"] for r in valid])
    v1_all   = np.array([r["v1"] for r in valid])
    v2_all   = np.array([r["v2"] for r in valid])
    v_avg    = (v1_all + v2_all) / 2
    stabilities = cosines * signs * np.clip(1 - euclids / 2, 0, 1)

    by_source = defaultdict(list)
    for r in valid: by_source[r["meta"].get("source", "unknown")].append(r)

    by_mode = defaultdict(list)
    for r in valid:
        if r["meta"].get("source") == "synthetic":
            by_mode[r["meta"].get("mode", "unknown")].append(r)

    report = {
        "报告类型": "自一致性拟合（无标签范式）",
        "生成时间": datetime.now().isoformat(),
        "范式声明": "本系统采用无标签自一致性校准范式。所有统计量反映管道输出的内禀稳定性，不涉及任何西医诊断标签。",
        "样本统计": {
            "总样本数": len(all_results),
            "有效信号": n,
            "无效信号(信号不足)": len(invalid),
            "有效率": f"{n/len(all_results)*100:.1f}%" if all_results else "0%",
        },
        "数据源分布": dict(sorted(by_source.items())),
        "自一致性指标": {
            "综合稳定性(平均)":   f"{float(np.mean(stabilities)):.4f}",
            "综合稳定性(中位数)": f"{float(np.median(stabilities)):.4f}",
            "综合稳定性(标准差)": f"{float(np.std(stabilities)):.4f}",
            "平均余弦相似度":   f"{float(np.mean(cosines)):.4f}",
            "余弦标准差":       f"{float(np.std(cosines)):.4f}",
            "平均欧氏距离":     f"{float(np.mean(euclids)):.4f}",
            "欧氏标准差":       f"{float(np.std(euclids)):.4f}",
            "平均符号一致性":   f"{float(np.mean(signs)):.4f}",
            "符号标准差":       f"{float(np.std(signs)):.4f}",
            "稳定性≥0.85占比":  f"{float(np.mean(stabilities>=0.85))*100:.1f}%",
            "稳定性≥0.90占比":  f"{float(np.mean(stabilities>=0.90))*100:.1f}%",
            "稳定性≥0.95占比":  f"{float(np.mean(stabilities>=0.95))*100:.1f}%",
            "稳定性≥0.99占比":  f"{float(np.mean(stabilities>=0.99))*100:.1f}%",
        },
        "ΔF分布(全体)": {
            DIM_NAMES[i]: {
                "均值": f"{float(np.mean(v_avg[:,i])):.4f}",
                "标准差": f"{float(np.std(v_avg[:,i])):.4f}",
                "中位数": f"{float(np.median(v_avg[:,i])):.4f}",
                "范围": f"[{float(np.min(v_avg[:,i])):.3f}, {float(np.max(v_avg[:,i])):.3f}]",
            } for i in range(5)
        },
        "ΔF维度相关性": {
            f"{DIM_NAMES[i]}-{DIM_NAMES[j]}": f"{float(np.corrcoef(v_avg[:,i], v_avg[:,j])[0,1]):.4f}"
            for i in range(5) for j in range(i+1, 5)
        },
        "按数据源稳定性": {
            src: f"{float(np.mean([rr['余弦相似度']*rr['符号一致性']*np.clip(1-rr['欧氏距离']/2,0,1) for rr in vs])):.4f}"
            for src, vs in sorted(by_source.items())
        },
    }

    if len(by_mode) > 0:
        report["合成数据按模式稳定性"] = {
            f"mode={mode}": {
                "计数": len(vs),
                "稳定性": f"{float(np.mean([rr['余弦相似度']*rr['符号一致性']*np.clip(1-rr['欧氏距离']/2,0,1) for rr in vs])):.4f}",
            } for mode, vs in sorted(by_mode.items())
        }

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    report["稳定性分位数"] = {f"P{p}": f"{float(np.percentile(stabilities, p)):.4f}" for p in pcts}

    worst_idx = np.argsort(stabilities)[:10]
    report["稳定性最低10个样本"] = [
        {"来源": valid[i]["meta"].get("source","?"), "模式": str(valid[i]["meta"].get("mode","")), "稳定性": f"{float(stabilities[i]):.4f}"}
        for i in worst_idx
    ]

    return report


# ══════════════════════════════════════════════════════════════
# 3. 主流程（流式：边生成边处理）
# ══════════════════════════════════════════════════════════════

def main(target_total=42000):
    print(f"{'='*60}")
    print(f"  大规模自一致性拟合")
    print(f"  目标样本数: {target_total}")
    print(f"  范式: 无标签自一致性")
    print(f"{'='*60}")
    t_start = time.time()

    all_results = []

    # ─── 3a. 处理真实数据 ───
    print("\n[1/4] 读取并处理真实数据...")
    t0 = time.time()
    readers = [("PTB-XL", read_ptbxl_local()),
               ("MIT-BIH", read_mitbih_local()),
               ("BUT-PPG", read_butppg_local())]

    for name, sigs in readers:
        for sig, fs, meta in sigs:
            r = process_signal(sig, fs, meta)
            all_results.append(r)
        print(f"  {name}: {len(sigs)} 条 ({time.time()-t0:.1f}s)")
    real_count = len(all_results)
    print(f"  真实数据合计: {real_count} 条")

    # ─── 3b. 流式生成+处理模拟数据 ───
    need = max(0, target_total - real_count)
    print(f"\n[2/4] 流式生成+处理模拟数据 (需补 {need} 条)...")

    modes = ["normal", "hard", "soft", "thick", "thin", "urgent", "slow"]
    durations = [10, 15, 30]
    hrs = [60, 72, 96]
    combos = [(m, d, h) for m in modes for d in durations for h in hrs]
    n_combo = max(1, need // len(combos))
    extra = need - n_combo * len(combos)

    np.random.seed(42)
    produced = 0
    t_gen = time.time()

    for i, (mode, dur, hr) in enumerate(combos):
        batch_size = n_combo + (1 if i < extra else 0)
        for _ in range(batch_size):
            ecg = generate_position_ecg(dur, 360, hr_bpm=hr, mode=mode)
            ecg += np.random.randn(len(ecg)) * 0.02
            r = process_signal(ecg, 360.0, {
                "source": "synthetic", "mode": mode, "dur": dur, "hr": hr
            })
            all_results.append(r)
            produced += 1
            if produced >= need: break
        if produced >= need: break
        if (i+1) % 10 == 0:
            elapsed = time.time() - t_gen
            rate = produced / elapsed if elapsed > 0 else 0
            print(f"  synthetic: {produced}/{need} ({produced/need*100:.0f}%) @ {rate:.0f}/s", flush=True)

    gen_time = time.time() - t_gen
    print(f"  模拟数据处理完成: {produced} 条 in {gen_time:.0f}s ({produced/gen_time:.0f}/s)")

    # ─── 3c. 生成报告 ───
    print(f"\n[3/4] 综合报告...")
    report = generate_report(all_results)

    out_path = os.path.join(os.path.dirname(__file__), "fitting_report_large.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  报告保存: {out_path}")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  拟合完成!")
    print(f"  总时间: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  总样本数: {len(all_results)}")
    print(f"  有效率: {report['样本统计']['有效率']}")
    print(f"  综合稳定性: {report['自一致性指标']['综合稳定性(平均)']} ± {report['自一致性指标']['综合稳定性(标准差)']}")
    print(f"  (中位数: {report['自一致性指标']['综合稳定性(中位数)']})")
    print(f"  稳定性≥0.85: {report['自一致性指标']['稳定性≥0.85占比']}")
    print(f"  稳定性≥0.95: {report['自一致性指标']['稳定性≥0.95占比']}")
    print(f"{'='*60}")
    print(f"\n{'─'*60}")
    print(f"  摘要")
    print(f"{'─'*60}")
    ds = report['数据源分布']
    print(f"  数据源: " + ", ".join(f"{k}={v}" for k, v in sorted(ds.items())))
    print(f"  平均余弦相似度: {report['自一致性指标']['平均余弦相似度']}")
    print(f"  平均欧氏距离: {report['自一致性指标']['平均欧氏距离']}")
    print(f"  平均符号一致性: {report['自一致性指标']['平均符号一致性']}")
    print(f"{'─'*60}")

    return report

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=42000, help="目标样本数")
    args = parser.parse_args()
    main(args.target)
