#!/usr/bin/env python3
"""分析已下载的 PTB-XL ECG 文件 (v2 — 已验证)"""
import sys, os, csv, json, numpy as np
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pulse_diagnosis import spum_pulse_diagnosis

DEST = os.path.join(os.path.dirname(__file__), "datasets", "ptbxl")
ECG_DIR = os.path.join(DEST, "ecg")
OUTPUT = os.path.join(os.path.dirname(__file__), "ptbxl_results.json")

# Build SCP map
scp_map = {}
scp_path = os.path.join(DEST, "scp_statements.csv")
with open(scp_path, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        code = row.get('', '').strip()
        cls = row.get('diagnostic_class', '').strip()
        if cls in ("NORM", "MI", "STTC", "CD", "HYP"):
            name = {"NORM":"normal","MI":"mi","STTC":"sttc","CD":"cd","HYP":"hyp"}[cls]
            scp_map[code] = name

# Build filename→(superclass, ecg_id) map
fn_map = {}
csv_path = os.path.join(DEST, "ptbxl_database.csv")
with open(csv_path, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        fn = row.get("filename_lr", "")
        key = fn.replace("records100/", "").replace("/", "_")
        scp_str = row.get("scp_codes", "")
        best = "unknown"
        if scp_str and scp_str != "{}":
            try:
                codes = json.loads(scp_str.replace("'", '"'))
                cs = Counter()
                for c, v in codes.items():
                    sc = scp_map.get(c, "other")
                    cs[sc] += float(v)
                if cs:
                    best = cs.most_common(1)[0][0]
            except: pass
        fn_map[key] = {"superclass": best, "ecg_id": int(row["ecg_id"])}

# Analyze each .hea file
hea_files = [f for f in sorted(os.listdir(ECG_DIR)) if f.endswith('.hea')]
print(f"Found {len(hea_files)} .hea files", flush=True)

results = []
for i, h in enumerate(hea_files):
    key = h.replace('.hea', '').replace('records100_', '', 1)
    dat = h.replace('.hea', '.dat')
    dat_path = os.path.join(ECG_DIR, dat)
    
    if not os.path.exists(dat_path):
        continue
    
    info = fn_map.get(key)
    if not info:
        continue
    
    try:
        with open(os.path.join(ECG_DIR, h)) as f:
            lines = f.readlines()
        
        hdr = lines[0].strip().split()
        n_sig, fs, n_samples = int(hdr[1]), float(hdr[2]), int(hdr[3])
        
        # Gain from first lead
        g = float(lines[1].split()[2].split('(')[0])
        b = float(lines[1].split()[2].split('(')[1].split(')')[0])
        raw = np.fromfile(dat_path, dtype=np.int16)
        expected = int(n_samples * n_sig)
        if len(raw) < expected:
            raw = np.pad(raw, (0, expected - len(raw)))
        sig = raw.reshape(-1, n_sig)[:n_samples, 0]
        # PTB-XL gain unit is /mV (not /uV like BUT PPG)
        if g != 0:
            sig_mv = (sig.astype(np.float64) - b) / g  # convert ADC → mV
        else:
            sig_mv = sig.astype(np.float64) / 1000.0
        
        # Skip constant signals
        if np.std(sig_mv) < 0.01:
            continue
        
        r = spum_pulse_diagnosis(sig_mv, fs=fs)
        dF_raw = r.get("五维修正向量 ΔF", {})
        if isinstance(dF_raw, dict):
            keys_order = ["S_木(约束)", "S_火(梯度)", "S_土(储备)", "S_金(修剪)", "S_水(流通)"]
            dF = [float(dF_raw.get(k, 0)) for k in keys_order]
        elif isinstance(dF_raw, list):
            dF = [float(v) for v in dF_raw]
        else:
            dF = []
        
        if len(dF) >= 5:
            results.append({
                "ecg_id": info["ecg_id"],
                "superclass": info["superclass"],
                "S_wood": float(dF[0]),
                "S_fire": float(dF[1]),
                "S_earth": float(dF[2]),
                "S_metal": float(dF[3]),
                "S_water": float(dF[4]),
                "hr": int(r.get("心率(bpm)", 0)),
                "sqi": float(r.get("信号质量 SQI", 0)),
            })
    except Exception as e:
        pass
    
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(hea_files)} ({len(results)} OK)", flush=True)

print(f"\nAnalyzed: {len(results)} records", flush=True)

cat_counts = Counter(r["superclass"] for r in results)
print(f"Classes: {dict(cat_counts)}", flush=True)

for sc in sorted(cat_counts):
    items = [r for r in results if r["superclass"] == sc]
    if items:
        m = np.mean([[r["S_wood"], r["S_fire"], r["S_earth"], r["S_metal"], r["S_water"]] for r in items], axis=0)
        print(f"  {sc} (n={len(items)}): ΔF=[{m[0]:.3f},{m[1]:.3f},{m[2]:.3f},{m[3]:.3f},{m[4]:.3f}]", flush=True)

with open(OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {OUTPUT}", flush=True)
