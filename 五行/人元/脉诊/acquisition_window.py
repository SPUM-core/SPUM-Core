#!/usr/bin/env python3
"""
采集进度弹窗 — 手指检测 + PPG 采集一体化

流程：
  1. 打开 tkinter 窗口（串口只开一次）
  2. 阶段① 手指检测：显示"请放手指"，监测信号直到检测到脉搏
  3. 阶段② PPG 采集：实时显示 HR/进度/信号质量
  4. 采集完成 → 自动关闭窗口 → 返回保存路径

用法：
  from acquisition_window import AcquisitionWindow
  win = AcquisitionWindow(port='COM3', duration=60, tag='rec_001')
  ok, save_prefix = win.run()

v2.1 / 2026-07-17 — 线程独立保存 + 常量来自共享模块"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ppg_acquisition import (
    SIGNAL_VARIANCE_THRESHOLD, HR_MIN_THRESHOLD,
    CONFIRM_SECONDS, FINGER_TIMEOUT,
)


class AcquisitionWindow:
    """
    一体化采集窗口：手指检测 → PPG 采集。
    run() 阻塞直到完成 → 返回 (ok, save_prefix)。
    """

    def __init__(self, port: str, duration: int, tag: str,
                 patient: str = "", baud: int = 115200,
                 finger_check: bool = True,
                 terminal: bool = False):
        self.port = port
        self.duration = duration
        self.tag = tag
        self.patient = patient
        self.baud = baud
        self.finger_check = finger_check
        self.terminal = terminal

        self._streamer = None
        self._running = True
        self._success = False
        self._save_prefix = ""

        # 实时数据
        self._hr = 0.0
        self._elapsed = 0.0
        self._progress = 0.0
        self._samples = 0
        self._signal_bar = 0.0

        # 阶段控制
        self._phase = "finger"         # "finger" → "acquisition" → "done"
        self._finger_ok = False        # 手指是否已检测到
        self._finger_timeout = FINGER_TIMEOUT
        self._finger_start = 0.0
        self._acq_start = 0.0

        # 指检确认计数
        self._confirm_count = 0
        self._confirm_needed = int(CONFIRM_SECONDS)

        # 线程安全结束信号
        self._thread_done = threading.Event()
        self._thread_success = False
        self._thread_prefix = ""

    def _save_dir(self) -> str:
        """数据保存目录（acquisition_window.py 所在目录）。"""
        return os.path.dirname(os.path.abspath(__file__))

    def _make_save_path(self) -> str:
        """构造保存路径前缀。"""
        return os.path.join(self._save_dir(), self.tag)

    # ── UI ──────────────────────────────────────────────────

    def _build_ui(self):
        self.root = tk.Tk()
        self.root.title("脉诊 — 信号采集")
        self.root.geometry("500x290")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)
        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"+{(sw-w)//2}+{(sh-h)//2}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        frame = tk.Frame(self.root, padx=25, pady=16)
        frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_font = tkfont.Font(size=16, weight='bold')
        title = f"脉诊采集 — {self.patient}" if self.patient else "脉诊采集"
        tk.Label(frame, text=title, font=title_font,
                 fg="#333333").pack(pady=(0, 6))

        # 大提示（手指检测阶段显示的引导文字）
        prompt_font = tkfont.Font(size=14)
        self.prompt_label = tk.Label(
            frame, text="", font=prompt_font, fg="#e67e22"
        )
        self.prompt_label.pack(pady=(0, 2))

        # HR 大数字
        hr_font = tkfont.Font(size=36, weight='bold')
        self.hr_label = tk.Label(frame, text="-- BPM",
                                 font=hr_font, fg="#2c3e50")
        self.hr_label.pack(pady=(0, 2))

        # 进度条
        bar_frame = tk.Frame(frame, height=22, width=450)
        bar_frame.pack(pady=(4, 2))
        bar_frame.pack_propagate(False)
        self.bar_canvas = tk.Canvas(bar_frame, height=22, width=450,
                                    bg="#ecf0f1", highlightthickness=0)
        self.bar_canvas.pack()
        self.progress_text = self.bar_canvas.create_text(
            225, 11, text="", fill="#555555",
            font=tkfont.Font(size=10, weight='bold')
        )

        # 状态
        status_font = tkfont.Font(size=12)
        self.status_label = tk.Label(
            frame, text="正在连接传感器...",
            font=status_font, fg="#7f8c8d"
        )
        self.status_label.pack(pady=(2, 2))

        # 详细指标
        info_font = tkfont.Font(size=10)
        self.info_label = tk.Label(
            frame, text="", font=info_font, fg="#95a5a6"
        )
        self.info_label.pack()

    def _update_bar(self, fraction: float, label: str = ""):
        self.bar_canvas.delete("bar")
        w = 450
        fill_w = max(6, int(w * min(fraction, 1.0)))
        if fraction < 0.4:
            color = "#e74c3c"
        elif fraction < 0.7:
            color = "#f39c12"
        else:
            color = "#27ae60"
        self.bar_canvas.create_rectangle(
            2, 2, fill_w, 20, fill=color, outline="", tags="bar"
        )
        self.bar_canvas.itemconfig(self.progress_text, text=label)

    def _on_close(self):
        self._running = False
        self._success = False
        if self._streamer:
            try:
                self._streamer.stop()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _finish(self, success: bool):
        self._success = success
        self._running = False

        if success and self._streamer:
            try:
                self._streamer.stop()
                prefix = self._make_save_path()
                self._streamer.save(prefix)
                self._save_prefix = prefix
            except Exception as e:
                print(f"[AcquisitionWindow] 保存失败: {e}")
                self._success = False

        if success:
            self.status_label.config(
                text=f"采集完成! {self._samples} 样本 ({self.duration}s)",
                fg="#27ae60"
            )
        else:
            self.status_label.config(text="采集已中止", fg="#e74c3c")
        self.root.update()
        time.sleep(0.6)
        try:
            self.root.destroy()
        except Exception:
            pass

    # ── 后台线程 ────────────────────────────────────────────

    def _acquisition_thread(self):
        """后台线程：指检 → 采集（串口只开一次）。独立保存数据，不依赖 Tk mainloop。"""
        from ppg_acquisition import CheezPPGStreamer

        try:
            streamer = CheezPPGStreamer(port=self.port, baud=self.baud)
            self._streamer = streamer

            # 采集总时长 = finger_timeout + duration
            total_duration = self.duration
            if self.finger_check:
                total_duration += self._finger_timeout

            if not streamer.start(duration=total_duration):
                self._thread_success = False
                self._thread_done.set()
                return

            start_ts = time.time()
            self._finger_start = start_ts

            while self._running and streamer._running:
                now = time.time()
                elapsed = now - start_ts

                # ── 阶段①：手指检测 ──
                if self._phase == "finger":
                    finger_elapsed = now - self._finger_start
                    smooth = streamer.get_channel('smooth')
                    hr = streamer._latest.get('HR', 0.0)

                    # 信号幅度
                    if len(smooth) > 10:
                        window = smooth[-min(125, len(smooth)):]
                        variance = float(np.var(window))
                        self._signal_bar = min(variance / 200, 1.0)
                    else:
                        variance = 0.0
                        self._signal_bar = 0.0

                    self._hr = hr
                    self._elapsed = finger_elapsed

                    # 判定
                    finger_ok = (variance > SIGNAL_VARIANCE_THRESHOLD) and (hr > HR_MIN_THRESHOLD)
                    if finger_ok:
                        self._confirm_count += 1
                    else:
                        self._confirm_count = 0

                    if self._confirm_count >= self._confirm_needed:
                        self._finger_ok = True
                        self._phase = "acquisition"
                        self._acq_start = now
                        self._confirm_count = 0

                    # 超时：跳过指检进入采集
                    if finger_elapsed > self._finger_timeout:
                        self._phase = "acquisition"
                        self._acq_start = now

                    time.sleep(0.15)
                    continue

                # ── 阶段②：主采集 ──
                acq_elapsed = now - self._acq_start
                self._elapsed = acq_elapsed
                self._progress = acq_elapsed / self.duration
                self._hr = streamer._latest.get('HR', 0.0)
                self._samples = streamer._total

                smooth = streamer.get_channel('smooth')
                if len(smooth) > 10:
                    window = smooth[-min(125, len(smooth)):]
                    var = float(np.var(window))
                    self._signal_bar = min(var / 200, 1.0)
                else:
                    self._signal_bar = 0.0

                if acq_elapsed >= self.duration:
                    break

                time.sleep(0.1)

            # 采集结束 — 独立保存，不依赖 Tk
            self._running = False
            streamer.stop()
            prefix = self._make_save_path()
            streamer.save(prefix)
            self._thread_prefix = prefix
            self._thread_success = True
            self._thread_done.set()

            print(f"[AcquisitionWindow] 采集完成: {streamer._total} 样本 → {prefix}")

        except Exception as e:
            print(f"[AcquisitionWindow] 采集异常: {e}")
            self._thread_success = False
            self._thread_done.set()

    # ── UI 更新 ────────────────────────────────────────────

    def _ui_update_loop(self):
        if not self._running:
            return

        # 根据阶段显示不同内容
        if self._phase == "finger":
            remaining = max(0, self._finger_timeout - self._elapsed)
            self.prompt_label.config(
                text="请将手指放在脉搏传感器上"
            )
            self.hr_label.config(
                text=f"{self._hr:.0f} BPM" if self._hr > 0 else "-- BPM"
            )
            self._update_bar(
                self._elapsed / self._finger_timeout,
                f"等待手指... {remaining:.0f}s"
            )
            self.status_label.config(
                text="检测脉搏信号...",
                fg="#e67e22"
            )
            self.info_label.config(
                text=f"HR: {self._hr:.0f} BPM  |  信号: {self._signal_bar * 100:.0f}%"
            )

        elif self._phase == "acquisition":
            self.prompt_label.config(text="")
            remaining = max(0, self.duration - self._elapsed)
            hr_text = f"{self._hr:.0f} BPM" if self._hr > 0 else "-- BPM"
            self.hr_label.config(text=hr_text)
            self._update_bar(
                self._progress,
                f"{int(self._progress * 100)}% / {self.duration}s"
            )
            self.status_label.config(
                text=f"采集中... 剩余 {remaining:.0f}s",
                fg="#2c3e50"
            )
            sig_text = f"{self._signal_bar * 100:.0f}%"
            sqi = "--"
            if self._hr > 0 and self._signal_bar > 0.2:
                sqi = "良好" if self._signal_bar > 0.5 else "可用"
            self.info_label.config(
                text=f"样本: {self._samples}  |  信号: {sig_text}  |  质量: {sqi}"
            )

        self.root.update()
        self.root.after(200, self._ui_update_loop)

    # ── 入口 ───────────────────────────────────────────────

    def run(self):
        """启动窗口（阻塞）。返回 (success, save_prefix)。无论 Tk 是否崩溃都等待线程完成。"""
        # 强制终端模式
        if self.terminal:
            print("[采集] 终端模式（--terminal）。")
            return self._fallback_terminal()

        # GUI 检查
        try:
            root_check = tk.Tk()
            root_check.withdraw()
            root_check.destroy()
        except Exception:
            print("[采集] 无 GUI 环境，回退到终端模式。")
            return self._fallback_terminal()

        # 启动采集线程（先启动，确保串口立即打开）
        acq_thread = threading.Thread(target=self._acquisition_thread,
                                       daemon=True)
        acq_thread.start()

        # 尝试启动 GUI（可能因 Tk 主题问题提前崩溃）
        try:
            self._build_ui()
            self.root.after(200, self._ui_update_loop)
            self.root.mainloop()
        except Exception as e:
            print(f"[采集] GUI 异常（不影响后台采集）: {e}")

        # mainloop 退出后等线程完成为止（最多 +60s 余量）
        self._thread_done.wait(timeout=self.duration + 60)

        # 取线程结果
        if self._thread_success:
            self._success = True
            self._save_prefix = self._thread_prefix
            print(f"[AcquisitionWindow] 采集完成（线程模式）: {self._save_prefix}")

        return self._success, self._save_prefix

    def _fallback_terminal(self):
        """无 GUI 时的终端回退。"""
        from ppg_acquisition import CheezPPGStreamer

        print("\n[采集] 终端模式...")
        try:
            streamer = CheezPPGStreamer(port=self.port, baud=self.baud)
            if not streamer.start(duration=self.duration):
                print("[采集] 连接失败。")
                return False, ""

            print(f"[采集] 开始 {self.duration}s ...")
            while streamer._running:
                time.sleep(1.0)
                elapsed = time.time() - streamer.start_time
                if elapsed >= self.duration:
                    break
                hr = streamer._latest.get('HR', 0)
                pct = int(elapsed / self.duration * 100)
                print(f"  {pct}% | HR: {hr:.0f} BPM | 样本: {streamer._total}")

            streamer.stop()
            prefix = self._make_save_path()
            streamer.save(prefix)
            print(f"[采集] 完成: {streamer._total} 样本")
            return True, prefix
        except Exception as e:
            print(f"[采集] 异常: {e}")
            return False, ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="采集窗口测试")
    parser.add_argument("--port", "-p", required=True)
    parser.add_argument("--duration", "-d", type=int, default=30)
    parser.add_argument("--patient", default="测试")
    parser.add_argument("--no-finger-check", action="store_true")
    args = parser.parse_args()

    tag = time.strftime("rec_%Y%m%d_%H%M%S_test")
    win = AcquisitionWindow(
        port=args.port, duration=args.duration,
        tag=tag, patient=args.patient,
        finger_check=not args.no_finger_check
    )
    ok, prefix = win.run()
    print(f"\n结果: {'成功' if ok else '失败'} | 前缀: {prefix}")
    sys.exit(0 if ok else 1)
