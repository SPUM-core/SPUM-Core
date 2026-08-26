#!/usr/bin/env python3
"""
脉诊一键触发入口 — 供 AI agent 调用
=====================================
用法：
  python pulse_diagnosis_cli.py --patient 胡运涛                  # 自动检测串口
  python pulse_diagnosis_cli.py --patient 胡运涛 --port COM5      # 指定串口
  python pulse_diagnosis_cli.py --patient 胡运涛 --simulate       # 模拟模式（无硬件）
  python pulse_diagnosis_cli.py --patient 胡运涛 --duration 120   # 采集120秒

AI agent 收到"进行脉诊"指令时的标准应答：
  1. 告知患者佩戴腕带式 PPG 传感器
  2. 待患者确认后运行本脚本
  3. 将输出的诊断结果写入病历

v1.6 / 2026-07-16 — 指检合并到采集窗口，串口只开一次"""

import sys, os

# 强制 UTF-8 编码输出，防止 GBK 终端 UnicodeEncodeError
if sys.stdout.encoding is None or sys.stdout.encoding.upper() not in ('UTF-8', 'UTF8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse, time, glob, shutil
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# 病历保存到用户本地（不入仓库）：~ / 青囊病历
CASE_ROOT = os.path.join(os.path.expanduser("~"), "青囊病历")


def find_patient_dir(name: str) -> str:
    """在病历目录下查找或创建患者文件夹。"""
    direct = os.path.join(CASE_ROOT, name)
    if os.path.isdir(direct):
        return direct
    for d in os.listdir(CASE_ROOT):
        if name in d:
            return os.path.join(CASE_ROOT, d)
    os.makedirs(direct, exist_ok=True)
    return direct


def simulate_acquisition(patient_dir: str, tag: str, duration: int):
    """生成模拟 PPG 数据（用于测试管线 / AI 演示）。"""
    from ppg_processing import generate_test_ppg
    fs = 125
    n = fs * duration

    sig = generate_test_ppg(fs, duration, 72)
    sig = sig * 300 + 512  # 映射到 ADC 范围

    np.save(os.path.join(PROJECT_ROOT, f"{tag}_smooth.npy"), sig.astype(np.float32))
    np.save(os.path.join(PROJECT_ROOT, f"{tag}_raw.npy"), sig.astype(np.float32))
    np.save(os.path.join(PROJECT_ROOT, f"{tag}_hrv.npy"), np.full(60, 42, dtype=np.float32))
    print(f"  → 模拟数据已生成 ({duration}s)")
    return True


def archive_data(patient_dir: str, tag: str) -> str:
    """归档采集数据到病历目录。"""
    pulse_dir = os.path.join(patient_dir, "脉诊")
    os.makedirs(pulse_dir, exist_ok=True)
    for f in glob.glob(os.path.join(PROJECT_ROOT, f"{tag}_*")):
        shutil.copy2(f, os.path.join(pulse_dir, os.path.basename(f)))
    return pulse_dir


def cleanup_temp_files(tag: str):
    """分析完成后删除项目根目录的临时采集文件（已归档至病历）。"""
    removed = 0
    for f in sorted(glob.glob(os.path.join(PROJECT_ROOT, f"{tag}_*"))):
        try:
            os.remove(f)
            removed += 1
        except Exception as e:
            print(f"  [清理] 删除失败: {os.path.basename(f)} — {e}")
    if removed:
        print(f"  → 已清理 {removed} 个临时文件")


def run_pipeline(tag: str, pulse_dir: str, fs: int = 125):
    """运行 PPG → 五形 ΔF → 辨证 管线。"""
    sys.path.insert(0, PROJECT_ROOT)
    from ppg_to_wuxing_bridge import PpgToWuxingBridge

    sig_path = os.path.join(PROJECT_ROOT, f"{tag}_smooth.npy")
    if not os.path.exists(sig_path):
        print(f"[错误] 波形文件不存在: {sig_path}")
        return None

    bridge = PpgToWuxingBridge(fs=fs)
    return bridge.pipeline(np.load(sig_path), return_raw=True)


def save_report(result: dict, patient_name: str, tag: str, pulse_dir: str) -> str:
    """将脉诊报告写入病历。"""
    df = result['delta_F']
    sq = result['six_qualities']
    lines = [
        f"## PPG 脉诊报告 — {patient_name} — {tag}",
        "",
        f"- 心率：{result['hr_bpm']:.1f} BPM | 搏动数：{result['n_beats']} | "
        f"质量：{result['sqi']:.3f} ({result['sqi_grade']})",
        f"- 六品质：粗{sq['coarse_score']:.2f} 硬{sq['hard_score']:.2f} "
        f"缓{sq['rate_score']:.2f} 滑{sq['smooth_score']:.2f} "
        f"浮{sq['depth_score']:.2f} 有力{sq['strength_score']:.2f}",
        f"- 五形 ΔF：木{df[0]:+.2f} 火{df[1]:+.2f} 土{df[2]:+.2f} "
        f"金{df[3]:+.2f} 水{df[4]:+.2f}",
        f"- 辨证摘要：{result['summary']}",
        "",
    ]
    for d in result['descriptions']:
        lines.append(f"  - {d}")

    report_path = os.path.join(pulse_dir, f"{tag}_pulse_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  → 报告已写入: {report_path}")
    return report_path


def format_diagnosis(result: dict) -> str:
    """格式化诊断结果。"""
    df = result['delta_F']
    names = ['木(弹性)', '火(脉势)', '土(容量)', '金(回弹)', '水(传导)']
    bars = []
    for n, v in zip(names, df):
        bar = '■' * min(int(abs(v) * 10), 10) + '□' * max(0, 10 - min(int(abs(v) * 10), 10))
        arrow = '↑' if v > 0.1 else ('↓' if v < -0.1 else '→')
        bars.append(f"  {n}: {v:+.2f} {arrow}  {bar}")

    return (
        "═══════════════════════════════════════\n"
        f"  脉诊结果 — {result.get('patient', '?')}\n"
        "═══════════════════════════════════════\n"
        f"  心率: {result['hr_bpm']:.1f} BPM\n"
        f"  搏动数: {result['n_beats']} / {result.get('duration_s', 60):.0f}s\n"
        f"  质量: {result['sqi']:.3f} ({result['sqi_grade']})\n"
        "─────────────────────────────────────\n"
        + "\n".join(bars) + "\n"
        "─────────────────────────────────────\n"
        + "\n".join(f"  • {d}" for d in result['descriptions']) + "\n"
        "─────────────────────────────────────\n"
        f"  辨证: {result['summary']}\n"
        "═══════════════════════════════════════\n"
    )


def main():
    parser = argparse.ArgumentParser(description="SPUM PPG 脉诊一键入口")
    parser.add_argument("--patient", "-p", required=True)
    parser.add_argument("--port", help="串口 (默认自动检测)")
    parser.add_argument("--duration", "-d", type=int, default=60)
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--terminal", action="store_true",
                        help="强制终端模式（无 GUI 弹窗）")
    parser.add_argument("--finger-check", action="store_true", default=True,
                        help="采集前检测手指（默认开启）")
    parser.add_argument("--no-finger-check", action="store_false", dest="finger_check",
                        help="跳过手指检测")
    args = parser.parse_args()

    # ── 串口检测 ──
    if not args.simulate:
        from ppg_acquisition import resolve_port
        port = resolve_port(args.port)
        if not port:
            print("[错误] 未检测到串口。请使用 --port 指定或确认硬件已连接。")
            sys.exit(1)
        print(f"[串口] {port}")

    tag = time.strftime("rec_%Y%m%d_%H%M%S")
    patient_dir = find_patient_dir(args.patient)

    print(f"\n{'='*55}\n  脉诊启动 — {args.patient}\n  病历目录: {patient_dir}\n{'='*55}")

    # ── 阈值：期望最少样本数 ──
    from ppg_acquisition import MIN_EXPECTED_SAMPLES
    min_samples = int(MIN_EXPECTED_SAMPLES * (args.duration / 60.0))

    # 采集
    if args.simulate:
        print("\n[提示] 模拟模式：生成合成 PPG 数据...")
        simulate_acquisition(patient_dir, tag, args.duration)
    else:
        from acquisition_window import AcquisitionWindow
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            print(f"\n[采集] 第 {attempt}/{max_retries} 次...")
            win = AcquisitionWindow(
                port=port, duration=args.duration,
                tag=tag, patient=args.patient,
                finger_check=args.finger_check,
                terminal=args.terminal
            )
            ok, prefix = win.run()

            if not ok:
                print(f"[采集] 第 {attempt} 次采集失败。")
                cleanup_temp_files(tag)
                if attempt < max_retries:
                    print("[采集] 等待 3 秒后重试...")
                    time.sleep(3.0)
                continue

            # 检查实际采集样本数
            smooth_path = f"{prefix}_smooth.npy"
            if os.path.exists(smooth_path):
                actual = len(np.load(smooth_path))
                if actual >= min_samples:
                    break  # 成功
                else:
                    print(f"[采集] 第 {attempt} 次样本不足: {actual} < {min_samples}")
            else:
                print(f"[采集] 第 {attempt} 次未生成波形文件")

            cleanup_temp_files(tag)
            if attempt < max_retries:
                print("[采集] 等待 3 秒后重试...")
                time.sleep(3.0)
                tag = time.strftime("rec_%Y%m%d_%H%M%S")  # 新 tag 防覆盖
        else:
            print("[错误] 采集持续失败，终止。")
            cleanup_temp_files(tag)
            sys.exit(1)

    # 归档 + 分析
    print("\n[提示] 正在处理数据...")
    pulse_dir = archive_data(patient_dir, tag)
    result = run_pipeline(tag, pulse_dir)
    if result is None:
        cleanup_temp_files(tag)
        sys.exit(1)

    # 报告
    result['patient'] = args.patient
    result['duration_s'] = args.duration
    report_path = save_report(result, args.patient, tag, pulse_dir)

    print("\n" + format_diagnosis(result))
    print(f"  完整报告: {report_path}")

    # 清理临时文件（已归档至病历），包括模拟模式生成的文件
    cleanup_temp_files(tag)

    print(f"\n{'='*55}\n  脉诊完成。如需更新病历，请告知。\n{'='*55}\n")


if __name__ == "__main__":
    main()
