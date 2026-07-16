/*
 * pulsesensor_arduino.ino
 * =======================
 * Pulsesensor (PPG) Arduino 固件
 *
 * 接线：
 *   Pulsesensor OUT → A0
 *   Pulsesensor VCC → 3.3V 或 5V
 *   Pulsesensor GND → GND
 *
 * 串口输出格式（ASCII，Serialplot 兼容）：
 *   <ADC_raw>\n               — 原始 ADC 值 (0–1023)
 *
 * 使用 Serialplot 实时查看波形：
 *   1. 选择串口，波特率 115200
 *   2. Data Format → ASCII
 *   3. 即可看到实时 PPG 波形
 *
 * 采样率：250 Hz（每 4ms 一个样本）
 * 适用于 0.5–15 Hz 的 PPG 信号频带（满足奈奎斯特）
 *
 * v1.0 / 2026-07-15
 */

// ============================================================
// 引脚定义
// ============================================================
const int PPG_PIN = A0;          // Pulsesensor 模拟输入
const int LED_PIN = LED_BUILTIN; // 心跳指示 LED

// ============================================================
// 采样参数
// ============================================================
const float SAMPLE_RATE_HZ = 250.0;           // 采样率 250 Hz
const unsigned long INTERVAL_US = 4000;        // 采样间隔 4ms
const int BAUD_RATE = 115200;                  // 串口波特率

// ============================================================
// 信号处理参数
// ============================================================
const int MOVING_WINDOW = 3;   // 移动平均窗口大小（3 点轻度平滑）
const int BEAT_THRESHOLD = 50; // 搏动检测阈值（ADC 变化量）
const int MIN_BEAT_INTERVAL = 180; // 最小搏动间隔（ms），约 333 BPM 上限

// ============================================================
// 状态变量
// ============================================================
unsigned long lastSampleTime = 0;

// 搏动检测状态
int lastRawValue = 0;
bool lastBeatState = false;
unsigned long lastBeatTime = 0;
int beatCount = 0;
float heartRateBPM = 0.0;

// 移动平均缓冲区
int rawBuffer[MOVING_WINDOW];
int bufferIndex = 0;

void setup() {
    Serial.begin(BAUD_RATE);
    pinMode(LED_PIN, OUTPUT);

    // 初始化缓冲区
    for (int i = 0; i < MOVING_WINDOW; i++) {
        rawBuffer[i] = analogRead(PPG_PIN);
    }

    lastRawValue = rawBuffer[0];
    lastSampleTime = micros();

    // 启动信号 — Serialplot 配置完成后可看到此标记
    delay(100);
    Serial.println("### PULSESENSOR_PPG_START ###");
}

void loop() {
    unsigned long now = micros();

    // ── 精确采样时序（不阻塞） ──
    if (now - lastSampleTime >= INTERVAL_US) {
        lastSampleTime = now;

        // 1. 读取 ADC
        int raw = analogRead(PPG_PIN);

        // 2. 移动平均滤波（轻度平滑，去除高频噪声）
        rawBuffer[bufferIndex] = raw;
        bufferIndex = (bufferIndex + 1) % MOVING_WINDOW;
        int filtered = 0;
        for (int i = 0; i < MOVING_WINDOW; i++) {
            filtered += rawBuffer[i];
        }
        filtered /= MOVING_WINDOW;

        // 3. 输出（Serialplot 兼容格式）
        Serial.println(filtered);

        // 4. 简单搏动检测（用于 LED 指示，非正式分析）
        int delta = filtered - lastRawValue;
        bool isBeat = (delta > BEAT_THRESHOLD && !lastBeatState);

        if (isBeat) {
            unsigned long now_ms = millis();
            if (now_ms - lastBeatTime > MIN_BEAT_INTERVAL) {
                // 有效搏动 — 闪烁 LED
                digitalWrite(LED_PIN, HIGH);
                beatCount++;
                if (lastBeatTime > 0) {
                    float interval_ms = (now_ms - lastBeatTime);
                    heartRateBPM = 60000.0 / interval_ms;
                }
                lastBeatTime = now_ms;
            }
            lastBeatState = true;
        }

        // 检测下降沿 — 重置搏动状态
        if (delta < -BEAT_THRESHOLD && lastBeatState) {
            lastBeatState = false;
            digitalWrite(LED_PIN, LOW);
        }

        lastRawValue = filtered;
    }

    // ── 串口命令处理（非阻塞） ──
    if (Serial.available() > 0) {
        char cmd = Serial.read();
        switch (cmd) {
            case 'b':  // 查询心率
                Serial.print("### BPM: ");
                Serial.print(heartRateBPM);
                Serial.println(" ###");
                break;
            case 'r':  // 重置搏动计数
                beatCount = 0;
                heartRateBPM = 0.0;
                Serial.println("### BEAT_RESET ###");
                break;
            case 's':  // 状态查询
                Serial.print("### SAMPLE_RATE:");
                Serial.print(SAMPLE_RATE_HZ);
                Serial.print(" BEATS:");
                Serial.print(beatCount);
                Serial.print(" BPM:");
                Serial.print(heartRateBPM);
                Serial.println(" ###");
                break;
        }
    }
}
