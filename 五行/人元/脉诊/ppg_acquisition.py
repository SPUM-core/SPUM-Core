#!/usr/bin/env python3
"""
PPG 串口采集模块 — 单通道 / CheezPPG 6通道 实时数据流
====================================================

支持两种固件模式：
  - simple:  单通道 PPG (250Hz)，Pulsesensor 兼容
  - cheez:   6通道 CSV (125Hz)，CheezPPG 腕戴式

用法：
  # CLI 采集
  python ppg_acquisition.py --cheez -d 60 -s rec_001

  # 作为模块
  from ppg_acquisition import CheezPPGStreamer
  with CheezPPGStreamer(port='COM5') as streamer:
      streamer.start(duration=30)
      raw = streamer.get_channel('raw')

v1.2 / 2026-07-16 — 硬件稳定性优化：ASCII状态栏 + 串口看门狗 + 编码兼容"""

import serial
import serial.tools.list_ports
import numpy as np
import time
import threading
import os
import argparse
from typing import Optional, List, Dict, Tuple


# ═══════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════
DEFAULT_BAUD = 115200
DEFAULT_BUFFER_SECONDS = 60
CHEEZ_RATE = 125
CHEEZ_CHANNELS = ['raw', 'smooth', 'filtered', 'peak', 'HR', 'HRV']


# ═══════════════════════════════════════════════════════════
# 串口工具函数
# ═══════════════════════════════════════════════════════════

def list_ports() -> List[str]:
    """列出所有可用串口。"""
    return [p.device for p in serial.tools.list_ports.comports()]


def detect_arduino_port() -> Optional[str]:
    """自动检测 Arduino 串口（按关键词匹配）。"""
    keywords = ['arduino', 'ch340', 'ch341', 'ftdi', 'cp210', 'usb serial']
    for p in serial.tools.list_ports.comports():
        desc = p.description.lower()
        if p.vid and p.pid:
            desc += str(p.vid).lower() + str(p.pid).lower()
        for kw in keywords:
            if kw in desc:
                return p.device
    ports = list_ports()
    return ports[0] if ports else None


# ═══════════════════════════════════════════════════════════
# 基类：共享串口/线程/缓冲区逻辑
# ═══════════════════════════════════════════════════════════

class _BaseStreamer:
    """
    PPG 采集器基类。
    子类只需实现 _parse_line(line) → Optional[Dict[str,float]]。
    """

    def __init__(self, port: Optional[str], baud: int,
                 buffer_seconds: int, expected_rate: float, n_channels: int,
                 channel_names: List[str]):
        self.port = port
        self.baud = baud
        self.expected_rate = expected_rate
        self.n_channels = n_channels
        self.channel_names = channel_names

        # 环形缓冲区
        self.buffer_size = int(buffer_seconds * expected_rate)
        self._buf = {name: np.zeros(self.buffer_size, dtype=np.float64)
                     for name in channel_names}
        self._latest = {name: 0.0 for name in channel_names}
        self._write_idx = 0
        self._total = 0

        # 串口 + 线程
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # 统计
        self.start_time = 0.0
        self._last_ts = 0.0
        self._intervals: List[float] = []
        self.dropouts = 0
        self._sec_counter = 0
        self.current_rate = 0.0

        # 看门狗：追踪最后收到数据的时间（秒）
        self._last_data_time = 0.0
        self._watchdog_triggered = False

    # ── 解析（子类实现） ──────────────────────────────────

    def _parse_line(self, decoded: str) -> Optional[Dict[str, float]]:
        """解析一行为 {channel: value}。子类实现。"""
        raise NotImplementedError

    def _format_status(self) -> str:
        """格式化每秒状态行（仅 ASCII 字符，兼容 GBK 终端）。"""
        hr = self._latest.get('HR', self._latest.get('smooth', 0))
        extra = f" HR:{hr:3.0f}" if 'HR' in self.channel_names else ""
        n = min(int(self.current_rate / 5), 25)
        bar = '#' * n + '.' * max(0, 25 - n)
        return f"\r[{self.tag}] {self.current_rate:5.1f} Hz |{extra} 总计 {self._total} | {bar}"

    # ── 串口管理 ──────────────────────────────────────────

    def connect(self) -> bool:
        port = self.port or detect_arduino_port()
        if not port:
            print(f"[{self.tag}] 错误：未检测到串口")
            return False
        try:
            self._serial = serial.Serial(port, self.baud, timeout=0.5)
            time.sleep(2.0)  # 给 Arduino 重启留足时间
            self._serial.reset_input_buffer()
            print(f"[{self.tag}] 已连接到 {port} @ {self.baud} baud")
            self.port = port
            return True
        except serial.SerialException as e:
            print(f"[{self.tag}] 连接失败: {e}")
            return False

    def disconnect(self):
        if self._serial and self._serial.is_open:
            self._serial.close()
            print(f"[{self.tag}] 已断开 {self.port}")

    # ── 采集控制 ──────────────────────────────────────────

    def start(self, duration: Optional[float] = None) -> bool:
        if self._running:
            return True
        if not self._serial or not self._serial.is_open:
            if not self.connect():
                return False
        self._running = True
        self._thread = threading.Thread(target=self._read_loop,
                                        args=(duration,), daemon=True)
        self._thread.start()
        print(f"\n[{self.tag}] 开始采集 {f'{duration:.0f}s' if duration else '持续'}...")
        return True

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.disconnect()
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"\n[{self.tag}] 采集结束: {elapsed:.1f}s, {self._total} 样本")

    def _safe_print(self, msg: str, **kwargs):
        """安全打印，捕获编码异常防止杀死采集线程。"""
        try:
            print(msg, **kwargs)
        except (UnicodeEncodeError, OSError):
            try:
                print(msg.encode('ascii', errors='replace').decode('ascii'), **kwargs)
            except Exception:
                pass

    def _reset_serial(self) -> bool:
        """软复位串口连接：关闭 → 等待 → 重开。"""
        self._safe_print(f"\n[{self.tag}] 看门狗触发：尝试复位串口...")
        try:
            if self._serial and self._serial.is_open:
                self._serial.close()
            time.sleep(1.0)
            port = self.port
            self._serial = serial.Serial(port, self.baud, timeout=0.5)
            self._serial.setDTR(False)
            time.sleep(0.5)
            self._serial.setDTR(True)
            time.sleep(1.0)
            self._serial.reset_input_buffer()
            self._safe_print(f"[{self.tag}] 串口复位成功: {port}")
            self._watchdog_triggered = False
            return True
        except Exception as e:
            self._safe_print(f"[{self.tag}] 串口复位失败: {e}")
            return False

    def _read_loop(self, duration: Optional[float] = None):
        self.start_time = time.time()
        self._last_ts = self.start_time
        self._last_data_time = self.start_time
        end_time = None if duration is None else self.start_time + duration
        self._sec_counter = self.start_time

        while self._running:
            now = time.time()
            if end_time and now >= end_time:
                break
            if not self._serial or not self._serial.is_open:
                self.dropouts += 1
                time.sleep(0.05)
                continue

            # ── 看门狗：超过 5 秒无数据 → 尝试复位 ──
            if self._total > 0 and (now - self._last_data_time) > 5.0:
                if not self._watchdog_triggered:
                    self._watchdog_triggered = True
                    self._safe_print(f"\n[{self.tag}] 看门狗：{now - self._last_data_time:.0f}s 无数据，尝试复位...")
                    self._reset_serial()
                time.sleep(0.1)
                continue

            try:
                raw = self._serial.readline()
            except Exception:
                self.dropouts += 1
                time.sleep(0.01)
                continue
            if not raw:
                continue

            try:
                decoded = raw.decode('utf-8', errors='ignore').strip()
            except Exception:
                continue
            if not decoded or decoded.startswith('###'):
                if decoded.startswith('###') and 'STATS' in decoded:
                    self._safe_print(f"\r[{self.tag}] {decoded}", end='', flush=True)
                continue

            parsed = self._parse_line(decoded)
            if parsed is None:
                continue

            for name in self.channel_names:
                self._buf[name][self._write_idx] = parsed[name]
            self._latest = parsed
            self._write_idx = (self._write_idx + 1) % self.buffer_size
            self._total += 1

            # 更新最后数据时间
            now = time.time()
            self._last_data_time = now
            self._watchdog_triggered = False

            # 采样间隔统计
            dt = (now - self._last_ts) * 1000
            if 0 < dt < 100:
                self._intervals.append(dt)
                if len(self._intervals) > self.expected_rate:
                    self._intervals.pop(0)
            self._last_ts = now

            # 每秒输出
            if now - self._sec_counter >= 1.0:
                if len(self._intervals) > 1:
                    avg = np.mean(self._intervals[-int(self.expected_rate):])
                    self.current_rate = 1000.0 / avg if avg > 0 else 0
                else:
                    self.current_rate = 0
                self._safe_print(self._format_status(), end='', flush=True)
                self._sec_counter = now

    # ── 数据访问 ──────────────────────────────────────────

    def get_channel(self, name: str) -> np.ndarray:
        if name not in self.channel_names:
            raise ValueError(f"通道 {name} 不在 {self.channel_names} 中")
        buf = self._buf[name]
        if self._total < self.buffer_size:
            return buf[:self._write_idx].copy()
        return np.concatenate([buf[self._write_idx:], buf[:self._write_idx]])

    def get_latest(self, n: int = 500) -> Dict[str, np.ndarray]:
        return {name: (ch[-n:] if len(ch) >= n else ch)
                for name, ch in self.get_buffer_dict().items()}

    def get_buffer_dict(self) -> Dict[str, np.ndarray]:
        return {name: self.get_channel(name) for name in self.channel_names}

    def stats(self) -> dict:
        d = self.get_channel(self.channel_names[0])
        hr, hrv = self._latest.get('HR', 0), self._latest.get('HRV', 0)
        return {
            'port': self.port, 'total': self._total,
            'duration_s': self._total / self.expected_rate,
            'rate_hz': self.current_rate, 'dropouts': self.dropouts,
            'hr_bpm': hr, 'hrv_ms': hrv,
            'mean': float(np.mean(d)), 'std': float(np.std(d)),
        }

    # ── 保存 ──────────────────────────────────────────────

    def save(self, prefix: str):
        for name in self.channel_names:
            np.save(f"{prefix}_{name}.npy", self.get_channel(name))
        combined = np.column_stack([self.get_channel(n) for n in self.channel_names])
        np.savetxt(f"{prefix}_all.csv", combined, delimiter=',',
                   header=','.join(self.channel_names), comments='')
        print(f"[{self.tag}] 已保存: {prefix}_*.npy + all.csv")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()


# ═══════════════════════════════════════════════════════════
# PPGStreamer：单通道 (Pulsesensor 兼容, 250Hz)
# ═══════════════════════════════════════════════════════════

class PPGStreamer(_BaseStreamer):
    tag = 'PPG'

    def __init__(self, port=None, baud=DEFAULT_BAUD,
                 buffer_seconds=DEFAULT_BUFFER_SECONDS):
        super().__init__(port, baud, buffer_seconds, 250, 1, ['adc'])

    def _parse_line(self, decoded: str) -> Optional[Dict[str, float]]:
        try:
            return {'adc': float(decoded)}
        except ValueError:
            return None

    def _format_status(self) -> str:
        n = min(int(self.current_rate / 10), 25)
        bar = '#' * n + '.' * max(0, 25 - n)
        return f"\r[PPG] {self.current_rate:5.1f} Hz | 总计 {self._total} | {bar}"


# ═══════════════════════════════════════════════════════════
# CheezPPGStreamer：6通道 (CheezPPG腕戴式, 125Hz)
# ═══════════════════════════════════════════════════════════

class CheezPPGStreamer(_BaseStreamer):
    tag = 'CheezPPG'

    def __init__(self, port=None, baud=DEFAULT_BAUD,
                 buffer_seconds=DEFAULT_BUFFER_SECONDS):
        super().__init__(port, baud, buffer_seconds, CHEEZ_RATE,
                         6, CHEEZ_CHANNELS)

    def _parse_line(self, decoded: str) -> Optional[Dict[str, float]]:
        parts = decoded.split(',')
        if len(parts) != 6:
            return None
        try:
            return dict(zip(CHEEZ_CHANNELS, [float(p) for p in parts]))
        except ValueError:
            return None

    def get_hr_hrv(self) -> Tuple[float, float]:
        return self._latest.get('HR', 0), self._latest.get('HRV', 0)

    def get_peak_buffer(self) -> np.ndarray:
        return self.get_channel('peak')

    def stats(self) -> dict:
        s = super().stats()
        s['hr_bpm'] = self._latest.get('HR', 0)
        s['hrv_sdnn_ms'] = self._latest.get('HRV', 0)
        return s


# ═══════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='PPG 串口采集')
    parser.add_argument('--port', '-p', help='串口 (默认自动检测)')
    parser.add_argument('--baud', '-b', type=int, default=DEFAULT_BAUD)
    parser.add_argument('--duration', '-d', type=float, default=30.0)
    parser.add_argument('--save', '-s', help='保存前缀')
    parser.add_argument('--buffer', type=int, default=DEFAULT_BUFFER_SECONDS)
    parser.add_argument('--list', '-l', action='store_true', help='列出串口')
    parser.add_argument('--cheez', action='store_true', help='CheezPPG 6通道模式')

    args = parser.parse_args()

    if args.list:
        print("可用串口:", list_ports())
        return

    Cls = CheezPPGStreamer if args.cheez else PPGStreamer
    with Cls(port=args.port, baud=args.baud, buffer_seconds=args.buffer) as s:
        if not s.start(duration=args.duration):
            return
        if s._thread:
            s._thread.join()

        if args.save:
            s.save(args.save)

        st = s.stats()
        print(f"\n采集摘要:\n  端口: {st['port']}\n  时长: {st['duration_s']:.1f}s"
              f"\n  样本数: {st['total']}\n  速率: {st['rate_hz']:.1f} Hz"
              + (f"\n  HR: {st['hr_bpm']:.0f} BPM" if 'hr_bpm' in st else ""))


if __name__ == '__main__':
    main()
