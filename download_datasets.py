#!/usr/bin/env python3
"""
SPUM 脉诊 — 公开数据集下载脚本
===============================
下载用于帧间动力学验证的公开 ECG/PPG 数据集。

依赖: pip install wfdb

用法:
  python download_datasets.py                   # 下载全部
  python download_datasets.py --list            # 只列出
  python download_datasets.py --senssmarttech   # 只下某个
"""

import os, sys, argparse, urllib.request
from pathlib import Path

DATA_DIR = Path("f:/spum-core/data")
PHYSIONET_BASE = "https://physionet.org/files"

datasets = {
    "senssmarttech": {
        "name": "SensSmartTech",
        "url": f"{PHYSIONET_BASE}/senssmarttech/1.0.0/",
        "s3_url": "s3://physionet-open/senssmarttech/1.0.0/",
        "size_gb": 1.5,
        "why": "ECG+PPG+PCG, 静息→活动后HR转换, 最适合动力学验证",
    },
    "wesad": {
        "name": "WESAD (Wearable Stress and Affect Detection)",
        "url": "http://archive.ics.uci.edu/ml/machine-learning-databases/00465/WESAD.zip",
        "size_gb": 2.1,
        "why": "baseline→stress→amusement 状态标签, ECG+PPG+呼吸",
    },
    "mitdb": {
        "name": "MIT-BIH Arrhythmia Database",
        "url": f"{PHYSIONET_BASE}/mitdb/1.0.0/",
        "s3_url": "s3://physionet-open/mitdb/1.0.0/",
        "size_gb": 0.3,
        "why": "48条ECG, 金标准, 360Hz, 含专家标注",
    },
    "nsrdb": {
        "name": "MIT-BIH Normal Sinus Rhythm",
        "url": f"{PHYSIONET_BASE}/nsrdb/1.0.0/",
        "s3_url": "s3://physionet-open/nsrdb/1.0.0/",
        "size_gb": 0.6,
        "why": "18健康人24h ECG, 健康基线",
    },
}


def download_physionet(name, url, dest):
    """从 PhysioNet S3 下载 WFDB 格式数据集"""
    import wfdb
    dest = DATA_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    ds = datasets[name]

    print(f"\n{'='*60}")
    print(f"下载: {ds['name']}")
    print(f"  目标: {dest}")
    print(f"  大小: ~{ds['size_gb']}GB")
    print(f"  用途: {ds['why']}")
    print(f"{'='*60}")

    path_part = url.replace("https://physionet.org/files/", "").rstrip("/")
    s3_base = f"https://physionet-open.s3.amazonaws.com/{path_part}"

    records = wfdb.io.get_record_list(name)
    total = len(records)
    print(f"  找到 {total} 个记录, 开始下载 (每文件 120s 超时)...")

    done, skip, fail = 0, 0, 0
    fail_streak = 0
    for i, rec in enumerate(records):
        parts = rec.split("/")
        rec_name = parts[-1]
        toplevel = parts[0] if len(parts) > 1 else None

        if toplevel:
            url_prefix = f"{s3_base}/{rec}/{rec_name}"
            local_dir = dest / toplevel / rec_name
        else:
            url_prefix = f"{s3_base}/{rec_name}"
            local_dir = dest
        local_dir.mkdir(parents=True, exist_ok=True)

        for ext in ('hea', 'dat'):
            fname = f"{rec_name}.{ext}"
            fp = local_dir / fname
            if fp.exists() and fp.stat().st_size > 0:
                skip += 1; continue
            try:
                resp = urllib.request.urlopen(f"{url_prefix}.{ext}", timeout=120)
                fp.write_bytes(resp.read())
                done += 1; fail_streak = 0
            except Exception:
                fail += 1; fail_streak += 1

        sys.stdout.write(f"\r  进度: {i+1}/{total}  OK={done} 跳过={skip} 失败={fail}")
        sys.stdout.flush()

        # 连续 10 个失败 → 网络问题, 给提示后退出
        if fail_streak >= 10:
            print(f"\n  ⚠️  连续 {fail_streak} 个文件下载失败 (网络问题)")
            break

    print(f"\n  ✅ 完成: 成功 {done}, 跳过 {skip}, 失败 {fail}")

    # 如果失败太多, 给出备选
    if fail > total * 0.3 and ds.get("s3_url"):
        print(f"\n  ⚡ 网络下载不稳定. 建议用 AWS CLI 完整下载:")
        print(f"     aws s3 sync --no-sign-request {ds['s3_url']} {dest}")


def download_wesad(dest):
    """下载 WESAD zip + 解压"""
    dest = DATA_DIR / "wesad"
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "wesad.zip"

    print(f"\n{'='*60}")
    print(f"下载: WESAD")
    print(f"  目标: {dest}")
    print(f"{'='*60}")

    if zip_path.exists():
        print("  ⚠️  zip 已存在, 跳过下载")
    else:
        url = datasets["wesad"]["url"]
        print(f"  下载 {url}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("OK")
        except Exception as e:
            print(f"FAIL: {e}")
            return

    import zipfile
    print("  解压...", end=" ", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    print("OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    for k in datasets:
        parser.add_argument(f"--{k}", action="store_true")
    args = parser.parse_args()

    if args.list:
        for k, ds in datasets.items():
            print(f"\n[{k}] {ds['name']}  ~{ds['size_gb']}GB\n  {ds['why']}\n  {ds['url']}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"数据目录: {DATA_DIR}")

    targets = [k for k in datasets if getattr(args, k.replace("-", "_"), False)]
    if not targets:
        targets = list(datasets.keys())

    for key in targets:
        if key == "wesad":
            download_wesad(DATA_DIR / key)
        else:
            download_physionet(key, datasets[key]["url"], DATA_DIR / key)

if __name__ == "__main__":
    main()
