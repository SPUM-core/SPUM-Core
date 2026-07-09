"""Benchmark + data survey for large-scale fitting"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from pulse_diagnosis import generate_position_ecg
from self_consistent_fitting import half_split_consistency, run_self_consistency

# 1. Benchmark: synthetic generation + half-split
t0 = time.time()
ecgs = [generate_position_ecg(30, 360, hr_bpm=72, mode='normal') for _ in range(10)]
print(f"10x gen: {time.time()-t0:.2f}s")

t0 = time.time()
for ecg in ecgs:
    r = half_split_consistency(ecg, 360)
print(f"10x half_split: {time.time()-t0:.2f}s")

t0 = time.time()
r = run_self_consistency(ecgs, 360)
print(f"10x batch: stability={r['综合稳定性得分']:.4f} in {time.time()-t0:.2f}s")

# 2. PTB-XL local survey
DEST = os.path.join(os.path.dirname(__file__), "datasets", "ptbxl")
ECG_DIR = os.path.join(DEST, "ecg")
if os.path.exists(ECG_DIR):
    hea_files = [f for f in os.listdir(ECG_DIR) if f.endswith(".hea")]
    print(f"\nPTB-XL local: {len(hea_files)} .hea files")
    # Check a few headers
    for h in hea_files[:3]:
        with open(os.path.join(ECG_DIR, h)) as f:
            print(f"  {h}: {f.readline().strip()}")
else:
    print("\nPTB-XL local: NOT FOUND")

# 3. MIT-BIH local survey
MITDB = os.path.join(os.path.dirname(__file__), "..", "data", "mitdb")
if os.path.exists(MITDB):
    hea = [f for f in os.listdir(MITDB) if f.endswith(".hea")]
    print(f"MIT-BIH local: {len(hea)} .hea files")
    for h in hea[:3]:
        with open(os.path.join(MITDB, h)) as f:
            print(f"  {h}: {f.readline().strip()}")
else:
    print("MIT-BIH local: NOT FOUND")

# 4. BUT-PPG local survey
BUT = os.path.join(os.path.dirname(__file__), "datasets", "butppg", "records")
if os.path.exists(BUT):
    subs = [d for d in os.listdir(BUT) if os.path.isdir(os.path.join(BUT, d))]
    ecg_files = []
    for s in subs:
        sd = os.path.join(BUT, s)
        ecg_files.extend([f for f in os.listdir(sd) if f.endswith("_ECG.dat")])
    print(f"BUT-PPG local: {len(subs)} subjects, {len(ecg_files)} ECG files")
else:
    print("BUT-PPG local: NOT FOUND")

# 5. Speed estimate for 40k
per_sample = (time.time() - t0) / 10
print(f"\nPer-sample time: {per_sample:.3f}s")
print(f"Estimated 40000 samples: {40000 * per_sample / 60:.1f} min")
