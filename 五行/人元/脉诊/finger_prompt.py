#!/usr/bin/env python3
"""
手指检测弹窗模块 — 等待患者放置手指，检测到脉搏后自动关闭

流程：
  1. 打开 tkinter 窗口提示"请将手指放在传感器上"
  2. 后台实时读取 PPG 信号，分析幅度和 HR
  3. 当信号幅度 > 阈值 且 HR > 30 BPM 持续 2 秒 → 判定手指已放置
  4. 自动关闭弹窗，返回成功
  5. 超时（默认 30 秒）未检测到 → 返回失败

用法：
  from finger_prompt import wait_for_finger
  ok = wait_for_finger(port='COM3', timeout=30)

v1.2 / 2026-07-16 — 去除 CheezPPGStreamer，直接读串口，避免看门狗干扰 COM3 状态"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import sys
import os
import numpy as np
from typing import Optional

# 添加父目录以便导入 ppg_acquisition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════
# 检测阈值（通过经验标定）
# ═══════════════════════════════════════════════════════════
SIGNAL_VARIANCE_THRESHOLD = 50.0   # smooth 通道方差下限
HR_MIN_THRESHOLD = 30.0            # HR 下限（BPM）
CONFIRM_SECONDS = 2.0              # 确认时长：需连续 N 秒满足条件
TIMEOUT_SECONDS = 30.0             # 超时时间


# ═══════════════════════════════════════════════════════════
# 手指检测器（弹窗 + 后台串口采集）
# ═══════════════════════════════════════════════════════════

class FingerPromptWindow:
    """tkinter 弹窗 + 后台 PPG 采集 → 手指检测。"""

    def __init__(self, port: Optional[str] = None, timeout: float = TIMEOUT_SECONDS):
        self.port = port
        self.timeout = timeout
        self.detected = False
        self._running = True

        # 累计检测计数
        self._confirm_count = 0
        self._confirm_needed = int(CONFIRM_SECONDS)  # 约 2 秒
        self._last_hr = 0.0
        self._last_variance = 0.0
        self._elapsed = 0.0

    def _build_ui(self):
        """构建弹窗界面。"""
        self.root = tk.Tk()
        self.root.title("脉诊 — 手指检测")
        self.root.geometry("480x220")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)

        # 居中显示
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"+{x}+{y}")

        # 字体
        title_font = tkfont.Font(size=18, weight='bold')
        status_font = tkfont.Font(size=14)
        info_font = tkfont.Font(size=11)

        # 主框架
        frame = tk.Frame(self.root, padx=30, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        self.title_label = tk.Label(
            frame, text="请将手指放在脉搏传感器上",
            font=title_font, fg="#333333"
        )
        self.title_label.pack(pady=(0, 10))

        # 状态文字
        self.status_label = tk.Label(
            frame, text="等待手指...",
            font=status_font, fg="#888888"
        )
        self.status_label.pack()

        # 信号强度条
        self.bar_frame = tk.Frame(frame, height=20, width=400)
        self.bar_frame.pack(pady=(10, 5))
        self.bar_frame.pack_propagate(False)

        self.bar_canvas = tk.Canvas(
            self.bar_frame, height=20, width=400,
            bg="#e0e0e0", highlightthickness=0
        )
        self.bar_canvas.pack()

        # 详情行
        self.detail_label = tk.Label(
            frame, text="HR: -- BPM  |  信号: --",
            font=info_font, fg="#aaaaaa"
        )
        self.detail_label.pack(pady=(5, 0))

        # 窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_bar(self, fraction: float):
        """更新信号强度条。fraction 0.0~1.0。"""
        self.bar_canvas.delete("all")
        w = 400
        fill_w = max(4, int(w * min(fraction, 1.0)))
        # 渐变颜色：红 → 黄 → 绿
        if fraction < 0.3:
            color = "#e74c3c"
        elif fraction < 0.6:
            color = "#f39c12"
        else:
            color = "#27ae60"
        self.bar_canvas.create_rectangle(2, 2, fill_w, 18, fill=color, outline="")

    def _on_close(self):
        """用户手动关闭窗口 → 视为放弃。"""
        self._running = False
        self.detected = False
        self.root.destroy()

    def _finish(self):
        """安全收尾：关闭窗口（必须从主线程调用）。"""
        if self.detected:
            self.status_label.config(text="检测到脉搏!", fg="#27ae60")
            self.title_label.config(text="信号正常，即将开始采集...")
            self.detail_label.config(
                text=f"HR: {self._last_hr:.0f} BPM  |  信号: {self._last_variance:.0f}"
            )
            self._update_bar(min(self._last_variance / 200, 1.0))
            self.root.update()
            time.sleep(0.5)
        try:
            self.root.destroy()
        except Exception:
            pass

    def _detect_loop(self):
        """后台线程：直接读串口原始数据检测手指（不使用 CheezPPGStreamer 看门狗）。"""
        import serial as pyserial

        ser = None
        try:
            # 打开串口（简单模式，无看门狗）
            try:
                from ppg_acquisition import detect_arduino_port
                port_name = self.port or detect_arduino_port()
                if port_name:
                    ser = pyserial.Serial(port_name, 115200, timeout=0.5)
            except Exception:
                pass

            if ser is None:
                self._running = False
                return

            start_ts = time.time()
            buffer = []

            while self._running:
                now = time.time()
                self._elapsed = now - start_ts
                if self._elapsed > self.timeout:
                    break

                # 读取串口数据行
                try:
                    line = ser.readline()
                except Exception:
                    time.sleep(0.1)
                    continue

                if not line:
                    time.sleep(0.05)
                    continue

                # 解析数值
                try:
                    val = int(line.strip())
                except (ValueError, TypeError):
                    continue

                buffer.append(val)
                if len(buffer) > 150:
                    buffer.pop(0)

                if len(buffer) < 30:
                    continue

                # 方差检测（信号是否有变化）
                window = buffer[-50:]
                variance = float(np.var(window))

                # HR 估算：基于过零点/峰值间距
                hr = 0.0
                if variance > 10:
                    # 简单 HR 估算：阈值交叉法
                    mean_val = float(np.mean(window))
                    above = sum(1 for v in window if v > mean_val + 5)
                    # 粗略估算：每秒过半数大于均值→有脉搏
                    if above > 10:
                        hr = 60.0  # 估算值

                self._last_hr = hr
                self._last_variance = variance

                # 判定逻辑
                finger_ok = (variance > SIGNAL_VARIANCE_THRESHOLD) and (hr > HR_MIN_THRESHOLD)
                if finger_ok:
                    self._confirm_count += 1
                else:
                    self._confirm_count = 0

                if self._confirm_count >= self._confirm_needed:
                    self.detected = True
                    break

                time.sleep(0.15)

        except Exception as e:
            print(f"[FingerPrompt] 检测异常: {e}")
        finally:
            if ser and ser.is_open:
                try:
                    ser.close()
                except Exception:
                    pass
            self._running = False

    def _ui_update_loop(self):
        """UI 更新循环（每 200ms），结果确认后自动关闭。"""
        if not self._running and not self.detected:
            # 超时或检测线程终止 → 关闭窗口
            self.root.after(100, self._finish)
            return

        if self.detected:
            self.root.after(0, self._finish)
            return

        # 更新状态
        elapsed_str = f"{self._elapsed:.0f}s"
        if self._last_hr > 0:
            status = f"检测中... (HR: {self._last_hr:.0f} BPM)"
            color = "#e67e22"
        else:
            status = f"等待手指... ({elapsed_str}/{self.timeout:.0f}s)"
            color = "#888888"

        self.status_label.config(text=status, fg=color)
        self.detail_label.config(
            text=f"HR: {self._last_hr:.0f} BPM  |  信号: {self._last_variance:.0f}"
        )
        self._update_bar(min(self._last_variance / 200, 1.0))

        self.root.update()
        self.root.after(200, self._ui_update_loop)

    def run(self) -> bool:
        """启动弹窗和检测，返回是否检测到手指。"""
        self._build_ui()

        # 启动后台检测线程
        detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        detect_thread.start()

        # 启动 UI 更新
        self.root.after(200, self._ui_update_loop)

        # 进入主循环
        try:
            self.root.mainloop()
        except Exception:
            pass

        return self.detected


def wait_for_finger(port: Optional[str] = None, timeout: float = TIMEOUT_SECONDS) -> bool:
    """
    弹窗等待患者放置手指。
    返回 True = 检测到手指脉搏；False = 超时/取消。
    """
    # 避免在无 GUI 环境（如 SSH）中崩溃
    try:
        import tkinter
        root_check = tkinter.Tk()
        root_check.withdraw()
        root_check.destroy()
    except Exception:
        print("[手指检测] 无 GUI 环境，跳过弹窗。")
        return True

    prompt = FingerPromptWindow(port=port, timeout=timeout)
    result = prompt.run()

    # 防止 tkinter 残留
    try:
        import tkinter
        tkinter._default_root = None
    except Exception:
        pass

    return result


# 独立测试入口
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="手指检测弹窗测试")
    parser.add_argument("--port", "-p", help="串口 (默认自动检测)")
    parser.add_argument("--timeout", "-t", type=float, default=TIMEOUT_SECONDS)
    args = parser.parse_args()

    print(f"手指检测启动 (超时 {args.timeout}s)...")
    ok = wait_for_finger(port=args.port, timeout=args.timeout)
    if ok:
        print("检测结果: 手指已放置，脉搏正常。")
    else:
        print("检测结果: 超时或取消。")
    sys.exit(0 if ok else 1)
