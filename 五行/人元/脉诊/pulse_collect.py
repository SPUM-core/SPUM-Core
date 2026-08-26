#!/usr/bin/env python3
"""
脉诊采集 — AI 一键调用接口

AI 用法:
  python pulse_collect.py --patient 胡运涛           # 自动检测串口
  python pulse_collect.py --patient 胡运涛 --port COM5  # 指定串口

返回（stdout 末行）:
  OK | <报告路径>
  FAIL | <错误描述>

双击等同: 脉诊一键采集.bat
"""

import sys, os, time

# 强制 UTF-8
if sys.stdout.encoding is None or sys.stdout.encoding.upper() not in ('UTF-8', 'UTF8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import argparse

# 确保能导入同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _ok(msg: str):
    print(f"\nOK | {msg}")
    return 0


def _fail(msg: str):
    print(f"\nFAIL | {msg}")
    return 1


def main():
    parser = argparse.ArgumentParser(description="脉诊一键采集 (AI 接口)")
    parser.add_argument("--patient", "-p", required=True, help="患者姓名")
    parser.add_argument("--port", help="串口 (默认自动检测)")
    parser.add_argument("--duration", "-d", type=int, default=60, help="采集时长(秒)")
    parser.add_argument("--no-finger-check", action="store_true", help="跳过手指检测")
    args = parser.parse_args()

    # ── 串口检测 ──
    from ppg_acquisition import resolve_port
    port = resolve_port(args.port)
    if not port:
        return _fail("未检测到串口。请使用 --port 指定串口，或确认硬件已连接。")
    print(f"[串口] {port}")

    tag = time.strftime("rec_%Y%m%d_%H%M%S")

    # ── 采集（含手指检测，串口只开一次） ──
    from acquisition_window import AcquisitionWindow
    win = AcquisitionWindow(
        port=port, duration=args.duration,
        tag=tag, patient=args.patient,
        finger_check=not args.no_finger_check
    )
    ok, prefix = win.run()

    if not ok:
        return _fail("采集失败")

    smooth_path = f"{prefix}_smooth.npy"
    if not os.path.exists(smooth_path):
        return _fail("波形文件未生成")

    # ── 阶段2：归档 ──
    from pulse_diagnosis_cli import archive_data, run_pipeline, save_report, format_diagnosis, CASE_ROOT

    patient_dir = os.path.join(CASE_ROOT, args.patient)
    pulse_dir = archive_data(patient_dir, tag)

    # ── 阶段3：分析 ──
    result = run_pipeline(tag, pulse_dir)
    if result is None:
        return _fail("脉象分析失败")

    result['patient'] = args.patient
    result['duration_s'] = args.duration
    report_path = save_report(result, args.patient, tag, pulse_dir)

    # ── 输出 ──
    print("\n" + format_diagnosis(result))
    print(f"  报告: {report_path}")

    # 清理临时文件
    from pulse_diagnosis_cli import cleanup_temp_files
    cleanup_temp_files(tag)

    return _ok(report_path)


if __name__ == "__main__":
    sys.exit(main())
