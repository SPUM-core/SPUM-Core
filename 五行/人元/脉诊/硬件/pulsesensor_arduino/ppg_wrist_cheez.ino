/*
 * ppg_wrist_cheez.ino
 * ====================
 * 启思 PPG 腕戴式脉搏传感器 — CheezPPG 库固件
 *
 * 适用硬件：
 *   Cheez PPG Wrist Pulse Sensor Module (XH2.54-4P)
 *   含腕带 + 蓝牙套件版本通用
 *
 * 接线（XH2.54-4P）：
 *   引脚 1 (GND)  → GND
 *   引脚 2 (VCC)  → 5V
 *   引脚 3 (AO)   → A0 (模拟输出)
 *   引脚 4 (DO)   → （可选，数字方波输出）
 *
 * 依赖库：
 *   CheezPPG 库 (安装方法: Arduino IDE → 工具 → 管理库 → 搜索 "CheezPPG")
 *   或手动安装: https://makerselectronics.com/wp-content/uploads/2025/09/src.zip
 *
 * 串口输出格式 (CSV, 115200 baud, 125 Hz):
 *   raw, smooth, filtered, peak, HR, HRV(SDNN)
 *
 *   各通道说明：
 *     raw     — 原始 ADC 值 (0–1023)
 *     smooth  — 滑动平均平滑后的 PPG
 *     filtered — 带通滤波后的 PPG (0.5–5 Hz)
 *     peak    — 搏动检测标记 (0 或 1)
 *     HR      — 心率 (BPM)
 *     HRV     — HRV SDNN (ms)
 *
 * 用途：
 *   在 Serialplot 中选 6 通道模式，查看所有通道波形
 *   或通过 ppg_acquisition.py 的 CheezPPG 模式解析
 *
 * v1.0 / 2026-07-15
 */

#include "CheezPPG.h"

// ============================================================
// 引脚与参数
// ============================================================
const int PPG_PIN = A0;           // PPG 模拟输入
const int SAMPLE_RATE = 125;      // CheezPPG 库固定 125 Hz
const int BAUD_RATE = 115200;     // 串口波特率
const int WEAR_THRESHOLD = 80;    // 佩戴检测阈值（-1 关闭）

// ============================================================
// 蓝牙模块控制 (可选)
// ============================================================
// 若使用蓝牙套件 (HC-05/HC-06/BLE), 连接:
//   蓝牙 TX → Arduino RX (Pin 2)
//   蓝牙 RX → Arduino TX (Pin 3)
// 通过 SoftwareSerial 转发数据

// #define USE_BLUETOOTH
#ifdef USE_BLUETOOTH
  #include <SoftwareSerial.h>
  SoftwareSerial btSerial(2, 3);  // RX, TX
#endif

// ============================================================
// CheezPPG 实例
// ============================================================
CheezPPG ppg(PPG_PIN, SAMPLE_RATE);

// ============================================================
// 统计输出定时
// ============================================================
unsigned long lastStatsTime = 0;
const unsigned long STATS_INTERVAL_MS = 5000;  // 每 5s 输出一次统计摘要

void setup() {
  Serial.begin(BAUD_RATE);

  // 佩戴检测阈值
  //   80 = 默认值，可根据实际佩戴情况调整
  //   若不需要佩戴检测，设为 -1
  ppg.setWearThreshold(WEAR_THRESHOLD);

  // 等待串口稳定
  delay(200);

  // 启动标记 — Serialplot 或采集脚本可据此识别
  Serial.println("### CHEEZ_PPG_WRIST_START ###");
  Serial.print("### SAMPLE_RATE:"); Serial.print(SAMPLE_RATE);
  Serial.print(" CHANNELS:6"); Serial.println(" ###");

#ifdef USE_BLUETOOTH
  btSerial.begin(BAUD_RATE);
  btSerial.println("### BT_CHEEZ_PPG_START ###");
#endif

  lastStatsTime = millis();
}

void loop() {
  // CheezPPG 库处理 (125 Hz 定时)
  if (ppg.checkSampleInterval()) {
    ppg.ppgProcess();

    // ─── 6 通道 CSV 输出 ───
    // 格式: raw,smooth,filtered,peak,HR,HRV
    Serial.print((int)ppg.getRawPPG());      // 0: 原始 ADC
    Serial.print(",");
    Serial.print((int)ppg.getAvgPPG());      // 1: 平滑 PPG
    Serial.print(",");
    Serial.print((int)ppg.getFilterPPG());   // 2: 带通滤波 PPG
    Serial.print(",");
    Serial.print((int)ppg.getPpgPeak());     // 3: 搏动标记 (0/1)
    Serial.print(",");
    Serial.print((int)ppg.getPpgHr());       // 4: 心率 (BPM)
    Serial.print(",");
    Serial.println((int)ppg.getPpgHrv());    // 5: HRV SDNN (ms)

#ifdef USE_BLUETOOTH
    btSerial.print((int)ppg.getRawPPG());
    btSerial.print(",");
    btSerial.print((int)ppg.getAvgPPG());
    btSerial.print(",");
    btSerial.print((int)ppg.getFilterPPG());
    btSerial.print(",");
    btSerial.print((int)ppg.getPpgPeak());
    btSerial.print(",");
    btSerial.print((int)ppg.getPpgHr());
    btSerial.print(",");
    btSerial.println((int)ppg.getPpgHrv());
#endif
  }

  // ─── 定期统计输出 ───
  unsigned long now = millis();
  if (now - lastStatsTime >= STATS_INTERVAL_MS) {
    lastStatsTime = now;
    Serial.print("### STATS HR:");
    Serial.print((int)ppg.getPpgHr());
    Serial.print(" HRV:");
    Serial.print((int)ppg.getPpgHrv());
    Serial.print(" PEAK:");
    Serial.print((int)ppg.getPpgPeak());
    Serial.println(" ###");
  }

  // ─── 处理串口命令 ───
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    switch (cmd) {
      case 'r':  // 重置（库无重置 API，重新上电即可）
        Serial.println("### RESET: power cycle to reset ###");
        break;
      case 's':  // 状态查询
        Serial.print("### CHEEZ_PPG @");
        Serial.print(SAMPLE_RATE);
        Serial.print("Hz HR:");
        Serial.print((int)ppg.getPpgHr());
        Serial.print(" HRV:");
        Serial.print((int)ppg.getPpgHrv());
        Serial.println(" ###");
        break;
      case 't':  // 切换佩戴检测阈值
        Serial.println("### TOGGLE_WEAR_THRESHOLD not supported in lib ###");
        break;
    }
  }
}
