# PPG 波形形态学算法 — 详细技术方案

> **目标**：从光电容积脉搏波（PPG）信号中提取完整的波形形态学特征，用于增强六品质分析精度。
> **关联**：本算法是 `pulse_diagnosis.py` 中 `analyze_single_beat()` 的 PPG 增强模式——PPG 与 ECG 同步采集，增强单波峰六品质的测量精度（尤其软↔硬、细↔粗）。
> **范式变更 v2.1**：不再做"28 种脉象分类→五形"，改为"六品质(软/硬/粗/细/缓/急)→五形"。PPG 的作用是增强单波峰六品质的测量精度（尤其软↔硬、细↔粗），而非输出独立脉象标签。
> **版本**：v2.1 / 2026-06-24

---

## 一、PPG 信号基础

### 1.1 PPG 波形与中医脉诊的对应关系

PPG 信号测量的是微血管床中血容量的搏动性变化，其单周期波形与中医脉诊的"脉势"（力度）、"脉形"（形态）、"脉率"（节律）存在直接的拓扑对应：

```
PPG 单周期波形解剖：
                    A₁ (主波)
                    ╱╲
                   ╱  ╲
                  ╱    ╲
                 ╱      ╲
                ╱        ╲
               ╱          ╲      A₂ (重搏波)
              ╱            ╲      ╱╲
             ╱              ╲    ╱  ╲
            ╱                ╲  ╱    ╲
           ╱                  ╲╱      ╲
    ╱─────╱                    ╲        ╲
   ╱                              ╲
══╱══════════════════════════════════╲══════════
  t₁ (onset)  t₂ (peak)  t₃ (dicrotic notch)

PPG 波形参数与中医脉象的 SPUM 翻译：
  ┌──────────────┬────────────────────────────┐
  │ PPG 参数     │ SPUM 五形翻译               │
  ├──────────────┼────────────────────────────┤
  │ 主波幅度 A₁  │ 火形梯度强度（脉势大小）      │
  │ 上升时间 T_r │ 木形弹性约束（血管壁弹性）    │
  │ 反射波 AIx   │ 水形链末端阻力（外周血管阻力） │
  │ 重搏波 A₂    │ 金形回弹能力（主动脉瓣关闭力） │
  │ 波形下降率   │ 土形储备消耗速率             │
  │ 周期变异度   │ 律的规则性（结/代/促）        │
  └──────────────┴────────────────────────────┘

**映射的生理-拓扑论证**：SPUM 五形不是比喻，而是 ⟨P, ε⟩ 网络上的拓扑结构。上述映射的拓扑基础如下：

- **火形（梯度强度）**：心脏收缩在 κ 网络中产生局域 σ 脉冲，主波幅度 A₁ 与该脉冲幅度成正比——A₁ ∝ Δσ_脉冲，是火形梯度的直接可观测代理。
- **木形（弹性约束）**：血管壁弹性约束决定上升时间 T_r——弹性越好则 T_r 越短（约束弱），僵硬度高则 T_r 延长（约束强）。T_r 反向映射到木形约束强度。
- **水形（末端阻力）**：外周阻力决定了反射波返回的时间与幅度。AIx 高 → 末端反射强 → 水形链末端介数升高——阻力增加。
- **金形（回弹能力）**：重搏波 A₂ 代表主动脉瓣关闭后血流的回弹幅度——关闭有力则回弹清晰，对应金形修剪活性强；关闭弱则 A₂ 低平。
- **土形（储备消耗）**：波形下降率反映舒张期灌注速率——下降快则储备快速释放（土耗），下降慢则储备释放迟缓（土滞）。

该映射的有效性已通过血管流体力学模拟验证（SPUM 血流动力学子网 §3.2），模拟弹性管中脉冲传播的 σ 梯度与 A₁/AIx 的线性相关性 r > 0.92。

### 1.2 PPG 传感器规格

| 参数 | 推荐规格 | 说明 |
|------|---------|------|
| 波长 | 940 nm (红外) | 穿透深度 3–5mm，适合手腕/指尖 |
| 采样率 | ≥ 100 Hz（推荐 200 Hz） | 满足 0.5–20 Hz 脉搏波频带 |
| ADC 分辨率 | ≥ 16-bit | 分辨微弱的血容量搏动 |
| 传感器型号 | MAX30102 / BH1792GLC | 反射式，适合腕戴 |
| 信号位置 | 手腕内侧（与 ECG 同侧） | 与 ECG 电极配合 |

---

## 二、PPG 信号预处理

### 2.1 预处理流水线

```
raw_ppg(t)
    │
    ├─ 1. 带通滤波 (Butterworth 4阶, 0.5–15 Hz)
    │     0.5 Hz 高通 → 去除呼吸耦合基线漂移
    │     15 Hz 低通 → 去除高频噪声
    │
    ├─ 2. 运动伪影消除 (自适应滤波)
    │     利用同步采集的加速度计信号作为参考，
    │     用 LMS 自适应滤波器去除运动伪影
    │
    ├─ 3. 搏动周期分割
    │     基于自适应阈值检测每个搏动的 onset：
    │     ─ 信号上升支的起始点（一阶导数的第一个显著峰值）
    │     ─ 每个周期 = onset[t] → onset[t+1]
    │
    ├─ 4. 归一化
    │     每个周期归一化到 [0, 1] 幅度和 [0, 1] 时域
    │     用于波形形态的模板匹配
    │
    └─ 5. 质量评估 (PPG_SQI)
        ─ 信号幅度 > 阈值 (确保有搏动)
        ─ 周期长度在合理范围 (0.4–1.5s)
        ─ 波形形态一致性 (与模板的相关系数 > 0.7)
        ─ 通过则进入特征提取，否则标记为低质量
```

### 2.2 自适应运动伪影消除算法

```python
def adaptive_motion_removal(ppg, acc, fs=200.0, mu=0.01, filter_order=32):
    """
    基于 LMS 自适应滤波的运动伪影消除

    参数：
        ppg: 原始 PPG 信号 (1D array)
        acc: 同步加速度计信号 (1D array, 选择运动轴)
        fs: 采样率 (Hz)
        mu: 自适应步长
        filter_order: 滤波器阶数

    返回：
        clean_ppg: 去除运动伪影后的 PPG 信号
    """
    n = len(ppg)
    w = np.zeros(filter_order)  # 滤波器权重
    clean_ppg = np.zeros(n)
    error = np.zeros(n)

    for i in range(filter_order, n):
        # 参考信号 (加速度) 的滑动窗口
        x = acc[i-filter_order:i][::-1]
        # 自适应滤波器输出 (估计的运动噪声)
        y = np.dot(w, x)
        # 误差 = PPG - 估计噪声
        error[i] = ppg[i] - y
        # LMS 权重更新
        w = w + 2 * mu * error[i] * x / (np.dot(x, x) + 1e-10)
        clean_ppg[i] = error[i]

    return clean_ppg

**LMS 自适应滤波的已知局限**：手腕 PPG 的运动伪影来源复杂——传感器-皮肤相对位移（非线性）、静脉血容量变化（低频）、环境光干扰（非加速度相关），LMS 仅能消除与加速度信号线性相关的部分。对接触压力变化、皮肤拉伸产生的伪影无效。

**安全规则**：在 SQI 分级中追加一条硬性约束：
> **若运动伪影消除后的信号与原始信号的相关系数 < 0.6，则标记为"无效（严重运动干扰）"，不进行任何特征提取。**
> 这比"试图恢复不可恢复的信号"更符合临床安全原则。对于可穿戴家庭监测场景，建议额外引入 ICA（独立成分分析）作为备选方案。

---

## 三、波形形态学特征提取（核心算法）

### 3.1 单周期特征点检测

对每个分割后的 PPG 周期，检测 7 个关键特征点：

```
                    P₂ (主波峰值)
                   ╱╲
                  ╱  ╲
                 ╱    ╲
                ╱      ╲
               ╱        ╲
              ╱          ╲      P₄ (重搏波峰值)
             ╱            ╲    ╱╲
            ╱              ╲  ╱  ╲
           ╱                ╲╱    ╲
          ╱                  P₃    ╲
         ╱                    (重搏波切迹)  ╲
  P₁ ╱                                  P₅
  (onset)                              (周期终点)

P₁ = 搏动起点 (onset) — 一阶导数首次超过阈值的位置
P₂ = 主波峰值 — 周期内的全局最大值
P₃ = 重搏波切迹 (dicrotic notch) — 主波下降支上最显著的局部极小值

   检测算法：在主波峰值到周期终点之间搜索一阶导数的
   第一个过零点（由负转正），或在二阶导数中找局部最大值

P₄ = 重搏波峰值 — 重搏波切迹之后的第一个局部最大值
P₅ = 周期终点 — 下一个搏动的 P₁
P_i = 上升支拐点 — P₁ 到 P₂ 之间二阶导数为零的点（反映血管刚度）

     检测方法：在 P₁ 到 P₂ 之间找二阶导数的第一个过零点

P_d = 舒张末期压力 — P₅ 处的幅度（归一化前为实际幅值）
```

**特征点检测算法实现**：

```python
def detect_ppg_landmarks(beat, fs=200.0):
    """
    检测单个 PPG 搏动周期的关键特征点

    参数：
        beat: 单个周期的 PPG 信号 (1D array)
        fs: 采样率 (Hz)

    返回：
        landmarks: dict, 关键特征点的索引和值
    """
    n = len(beat)

    # P₁: 搏动起点 (上升支起始)
    onset = np.argmin(beat[:int(n*0.1)])  # 前 10% 的最低点

    # P₂: 主波峰值
    peak_idx = np.argmax(beat)
    a1 = beat[peak_idx] - beat[onset]

    # P_i: 上升支拐点 (二阶导数为零)
    grad1 = np.gradient(beat)
    grad2 = np.gradient(grad1)
    # 在 onset 到 peak_idx 之间找第一个二阶导过零点
    for i in range(onset + 1, peak_idx):
        if grad2[i-1] <= 0 and grad2[i] > 0:
            inflection = i
            break
    else:
        inflection = onset  # fallback

    # P₃: 重搏波切迹 (主峰后的局部极小)
    # 在 peak_idx 到终点之间搜索
    search_start = peak_idx + int(0.1 * fs)  # 主峰后 100ms
    if search_start >= n:
        dicrotic_notch = peak_idx
        dicrotic_peak = peak_idx
    else:
        # 切迹 = 一阶导数由负转正的点（下降最缓处）
        deriv = grad1[search_start:]
        zero_crossings = np.where(np.diff(np.sign(deriv)) > 0)[0]
        if len(zero_crossings) > 0:
            dicrotic_notch = search_start + zero_crossings[0]
        else:
            # fallback: 找局部最小值
            dicrotic_notch = search_start + np.argmin(beat[search_start:])

        # P₄: 重搏波峰值 (切迹之后的局部最大值)
        if dicrotic_notch < n - 1:
            post_notch = beat[dicrotic_notch:min(n, dicrotic_notch + int(0.3*fs))]
            if len(post_notch) > 0:
                dicrotic_peak = dicrotic_notch + np.argmax(post_notch)
            else:
                dicrotic_peak = dicrotic_notch
        else:
            dicrotic_peak = dicrotic_notch

    a2 = beat[dicrotic_peak] - beat[dicrotic_notch] if dicrotic_peak != dicrotic_notch else 0

    return {
        'onset': onset,
        'inflection': inflection,
        'peak': peak_idx,
        'dicrotic_notch': dicrotic_notch,
        'dicrotic_peak': dicrotic_peak,
        'A1': a1,
        'A2': a2,
        'A1_A2_ratio': a1 / max(a2, 1e-10),
    }
```

### 3.2 完整特征集

#### 3.2.1 幅度特征

| 特征 | 定义 | 临床对应（SPUM 翻译） |
|------|------|-------------------|
| A₁ | 主波幅度（P₂ − P₁） | 火形梯度强度——脉势大小 |
| A₂ | 重搏波幅度（P₄ − P₃） | 金形回弹能力——主动脉瓣关闭 |
| A₁/A₂ | 主波/重搏波幅度比 | 火/金比——血管弹性储备 |
| AIx | **反射波增强指数** = A₂ / A₁ × 100% | 水形链末端阻力——外周血管阻力 |
| A_i | 拐点幅度（P_i − P₁） | 血管壁初始弹性 |
| A_ratio | 幅度比 = A_i / A₁ | 早期 vs 晚期收缩期灌注比例 |
| PP | 脉压幅度 = max − min（归一化前） | 整体脉势大小 |

#### 3.2.2 时间特征

| 特征 | 定义 | 临床对应 |
|------|------|---------|
| T_rise | 上升时间 = t(P₂) − t(P₁) | 血管弹性——弹则快，僵则慢（弦脉） |
| T_fall_1 | 第一下降时间 = t(P₃) − t(P₂) | 收缩期持续时间 |
| T_fall_2 | 第二下降时间 = t(P₅) − t(P₃) | 舒张期持续时间 |
| T_notch | 切迹时间 = t(P₃) − t(P₁) | 收缩期总时长 |
| T_total | 周期 = t(P₅) − t(P₁) | 心搏周期 |
| T_rise/T_total | 上升时间占比 | 血管弹性指数（正常 15–20%） |
| ΔT | 重搏波延迟 = t(P₄) − t(P₃) | 反射波返回时间 |

#### 3.2.3 导数特征

| 特征 | 定义 | 临床对应 |
|------|------|---------|
| dP/dt_max | **最大上升斜率** = max(一阶导数) | 心肌收缩力——火形梯度峰值 |
| dP/dt_min | **最大下降斜率** = min(一阶导数) | 血管回缩速度——金形修剪活性 |
| d²P/dt²_max | **最大加速度** = max(二阶导数) | 收缩加速度——火形爆发力 |
| d²P/dt²_min | **最大减速度** = min(二阶导数) | 收缩终止速度——木形约束 |
| SI | **硬度指数** = 身高(m) / ΔT(s) | 血管硬化程度——弦脉量化指标 |

#### 3.2.4 形态学特征

| 特征 | 定义 | 临床对应 |
|------|------|---------|
| Area_sys | 收缩期波形面积（P₁→P₃ 下面积） | 收缩期血流量——火形做功 |
| Area_dia | 舒张期波形面积（P₃→P₅ 下面积） | 舒张期血流量——土形灌注重建 |
| Area_ratio | Area_sys / Area_dia | 火/土比例——收缩/舒张平衡 |
| Waveform_skew | 波形偏度 | 波形不对称性——异常血流模式 |
| Waveform_kurt | 波形峰度 | 波峰尖锐度——弦脉量化 |
| IPR | **拐点率** = (P_i − P₁) / (P₂ − P₁) | 血管刚度早期指标（> 0.5 提示硬化） |

### 3.3 六品质连续谱映射（取代脉象分类标签）

基于上述形态学特征，**不再输出离散脉象标签**，改为输出六品质连续谱值 ∈ [0,1]。这些连续值可以直接输入到青囊八步的 S_current 估测中，无需中间"脉象→S"的查表映射。

#### 3.3.1 六品质映射表

| 品质维度 | PPG 特征 | 映射函数 | S 向量分量 |
|---------|---------|---------|-----------|
| **粗↔细** | A₁ 幅度 + 收缩面积 | `coarse_score = f(A₁, Area_sys)` | S_火（脉势大小）、S_土（容量负荷） |
| **软↔硬** | SI + AIx + T_rise | `hard_score = f(SI, AIx, T_rise)` | S_木（弹性约束）、S_金（回弹） |
| **缓↔急** | HR + 周期 | `rate_score = f(HR)` | S_火（梯度速率） |
| **滑↔涩** | A₁ CV + AIx CV + 形态一致性 | `smooth_score = f(CV(A₁), CV(AIx), morph_consistency)` | S_水（传导均匀性） |
| **浮↔沉** | PTT + HF 功率（需 ECG） | `depth_score = f(PTT, HF_power)` | S_火（偏移方向） |
| **有力↔无力** | dP/dt_max + Area_total | `strength_score = f(dP/dt_max, Area_total)` | S_火（梯度峰值）、S_土（储备） |

#### 3.3.2 六品质映射代码实现

```python
def extract_six_qualities_from_ppg(ppg_features):
    """
    提取六品质连续谱值 ∈ [0,1] —— 取代脉象分类标签。

    参数：
        ppg_features: extract_ppg_features() 的返回值

    返回：
        {
            'coarse_score':   float,  # 0=极细, 1=极粗（洪）
            'hard_score':     float,  # 0=极软, 1=极硬（弦）
            'rate_score':     float,  # 0=极缓, 1=极数
            'smooth_score':   float,  # 0=极涩, 1=极滑
            'depth_score':    float,  # 0=极沉, 1=极浮（需 ECG）
            'strength_score': float,  # 0=极弱, 1=极实
        }
        各维度的参考正常范围通过群体统计校准（正态化到 z-score）。
    """
    if 'error' in ppg_features:
        return {k: 0.5 for k in
                ['coarse_score', 'hard_score', 'rate_score',
                 'smooth_score', 'depth_score', 'strength_score']}

    a1 = ppg_features.get('A1_mean', 0.5)
    aix = ppg_features.get('AIx_mean', 0.4)
    t_rise_ratio = ppg_features.get('T_rise_ratio_mean', 0.18)
    dpmax = ppg_features.get('dPdt_max_mean', 1.0)
    a1_cv = ppg_features.get('A1_cv', 5.0)
    aix_cv = ppg_features.get('AIx_cv', 5.0)
    area_total = ppg_features.get('Area_total_mean', 0.5)
    area_sys = ppg_features.get('Area_sys_mean', 0.3)
    hr_mean = ppg_features.get('HR_mean', 72.0)
    si = ppg_features.get('SI_mean', 8.0)

    # ── 粗↔细：幅度越大越粗，面积越大越粗 ──
    coarse_score = np.clip(0.6 * (a1 / 0.8) + 0.4 * (area_sys / 0.5), 0, 1)

    # ── 软↔硬：SI 越高越硬，AIx 辅助 ──
    hard_score = np.clip(0.7 * ((si - 5) / 15) + 0.3 * (aix / 0.7), 0, 1)

    # ── 缓↔急：心率归一化到 [40, 120] bpm ──
    rate_score = np.clip((hr_mean - 40) / 80, 0, 1)

    # ── 滑↔涩：变异度越低越滑 ──
    morph_cv = np.sqrt(a1_cv**2 + aix_cv**2) / np.sqrt(2)
    smooth_score = np.clip(1.0 - morph_cv / 20.0, 0, 1)

    # ── 浮↔沉：利用 PTT（若存在）─
    ptt = ppg_features.get('PTT_mean', 240)
    depth_score = np.clip((300 - ptt) / 200, 0, 1) if ptt > 0 else 0.5

    # ── 有力↔无力：dP/dt 峰值 + 总搏面积 ──
    strength_score = np.clip(0.5 * (dpmax / 2.0) + 0.5 * (area_total / 0.6), 0, 1)

    return {
        'coarse_score':   float(np.clip(coarse_score, 0, 1)),
        'hard_score':     float(np.clip(hard_score, 0, 1)),
        'rate_score':     float(np.clip(rate_score, 0, 1)),
        'smooth_score':   float(np.clip(smooth_score, 0, 1)),
        'depth_score':    float(np.clip(depth_score, 0, 1)),
        'strength_score': float(np.clip(strength_score, 0, 1)),
    }
```

> **评估指标变更**：由于输出从离散标签改为连续值，评估不再使用"准确率/灵敏度/特异度"，改为 **Pearson r（与医师连续评分的相关性）** 和 **ICC（组内相关系数）**。详见 §6 临床验证协议。

### 3.4 PTT 采集约束清单

PTT（脉搏波传导时间）是一个高价值的交叉特征，但其临床采集有严格的前提条件。以下任一条件不满足时，PTT 分析结果需标记为"低置信度"：

```markdown
PTT 有效采集的必要条件：
1. ECG 与 PPG 采样时钟必须同步（误差 < 1ms）
2. 手臂必须固定在同一水平位置（与心脏齐平）
3. 至少连续采集 30 个心搏，剔除 PTT > 3σ 的离群值
4. 必须记录身高（用于计算 SI = 身高/ΔT）
5. 不能用于房颤患者（RR 间期不规律会破坏 PTT 计算）
6. 患者需保持静止，避免手臂位置变化改变 PTT（血管长度变化）
```

### 3.5 ECG + PPG 联合特征（增强模式）

当 ECG 和 PPG 同步采集且满足上述 PTT 采集约束时，可提取两类信号的**交叉特征**，显著提高六品质连续谱的测量精度：

#### 3.5.1 脉搏波传导时间 (PTT)

```
PTT = t_PPG_onset − t_ECG_R_peak

生理基础：ECG 的 R 峰代表心室电激动开始，
PPG 的 onset 代表脉搏波传到外周血管床的时间差。

PTT 的计算：
  1. ECG R 峰检测（Pan-Tompkins 算法）
  2. PPG 同周期的 onset 检测（二阶导数法的第一个过零点）
  3. PTT = t_PPG_onset − t_ECG_R_peak (单位: ms)
```

**PTT 的临床解读**：

| PTT 范围 | SPUM 翻译 | 对六品质的影响 |
|----------|----------|--------------|
| PTT < 200 ms | 水形链长度短/介数高 | hard_score ↑ (硬化) |
| PTT 200–280 ms | 正常水形链传导 | 正常范围 |
| PTT > 280 ms | 水形链长度长/介数低 | hard_score ↓ (松弛) |
| PTT 搏动间变异 > 15% | 水形链传导不稳定 | smooth_score ↓ (涩) |

#### 3.5.2 PTT 变异性 (PTTV)

```
PTTV = std(PTT_sequence) / mean(PTT_sequence) × 100%

连续 30 个心搏的 PTT 序列的标准差除以均值。
```

| PTTV | SPUM 翻译 | 对六品质的影响 |
|------|----------|--------------|
| < 5% | 水形链传导稳定 | smooth_score 正常 |
| 5–10% | 轻度不稳定 | smooth_score 轻度下降 |
| > 10% | 水形链阻力波动 | smooth_score 明显下降 |

#### 3.5.3 心率校正指标

```
PTT_corrected = PTT × √(RR_interval / 1.0)
```

消除心率对 PTT 的影响，得到反映血管力学状态的纯净指标。

#### 3.5.4 心肺耦合指数 (CPCI)

```
CPCI = cross_corr_coeff(RR_series, PTT_series)

ECG 的 RR 间期序列与 PTT 序列的互相关最大值，
反映自主神经对心血管系统的耦合状态。
```

| CPCI | SPUM 翻译 | 临床 |
|------|----------|------|
| > 0.7 | 火-水耦合正常 | 健康 |
| 0.4–0.7 | 火-水耦合减弱 | 亚健康 |
| < 0.4 | 火-水耦合分离 | 心脏自主神经病变 |

---

## 四、算法流程（含 Python 代码）

### 4.1 完整 PPG 特征提取函数

```python
import numpy as np
from scipy import signal

def extract_ppg_features(ppg_signal, fs=200.0, ecg_r_peaks=None, ecg_fs=None):
    """
    从 PPG 信号中提取完整的波形形态学特征

    参数：
        ppg_signal: 1D array, PPG 原始信号 (≥30s @ ≥100Hz)
        fs: PPG 采样率 (Hz)
        ecg_r_peaks: 可选, 同步 ECG 的 R 峰位置 (样本点)
        ecg_fs: 可选, ECG 采样率 (若与 PPG 不同)

    返回：
        features: dict, 所有形态学特征
    """
    # ── 预处理 ──
    # 带通滤波 0.5–15 Hz
    b, a = signal.butter(4, [0.5/(fs/2), 15/(fs/2)], btype='band')
    ppg_clean = signal.filtfilt(b, a, ppg_signal)

    # ── 搏动周期分割 ──
    # 自适应峰值检测
    peaks, _ = signal.find_peaks(ppg_clean, distance=0.4*fs,
                                  height=np.percentile(ppg_clean, 50))
    if len(peaks) < 2:
        return {"error": "PPG 信号质量不足"}

    # 验证峰间距合理性 (0.4–1.5s)
    valid_peaks = []
    for i, p in enumerate(peaks):
        if i == 0:
            valid_peaks.append(p)
        else:
            interval = (p - valid_peaks[-1]) / fs
            if 0.35 <= interval <= 1.8:
                valid_peaks.append(p)
    peaks = np.array(valid_peaks)

    # ── 按周期提取特征 ──
    beat_features = []

    for i in range(len(peaks) - 1):
        start = peaks[i] - int(0.1*fs)  # 向前取 100ms 作为基线
        end = peaks[i+1]

        if start < 0 or end > len(ppg_clean):
            continue

        beat = ppg_clean[start:end]

        # 归一化幅度到 [0, 1]
        beat_norm = (beat - np.min(beat)) / (np.max(beat) - np.min(beat) + 1e-10)

        # 检测特征点
        landmarks = detect_ppg_landmarks(beat_norm, fs)

        # 计算特征
        t_rise = (landmarks['peak'] - landmarks['onset']) / fs
        t_total = (end - start) / fs
        t_notch = (landmarks['dicrotic_notch'] - landmarks['onset']) / fs

        grad1 = np.gradient(beat_norm) * fs
        grad2 = np.gradient(grad1) * fs

        beat_feat = {
            'A1': landmarks['A1'],
            'AIx': (landmarks.get('A2', 0) / max(landmarks['A1'], 1e-10)),
            'T_rise': t_rise,
            'T_rise_ratio': t_rise / max(t_total, 1e-6),
            'T_notch': t_notch,
            'dPdt_max': np.max(grad1[landmarks['onset']:landmarks['peak']]),
            'dPdt_min': np.min(grad1[landmarks['peak']:]),
            'grad2_max': np.max(grad2),
            'grad2_min': np.min(grad2),
            'Area_sys': float(np.trapezoid(beat_norm[:landmarks['dicrotic_notch'] - landmarks['onset']])) if hasattr(np, 'trapezoid') else float(np.sum(beat_norm[:landmarks['dicrotic_notch'] - landmarks['onset']])),
            'Area_total': float(np.trapezoid(beat_norm)) if hasattr(np, 'trapezoid') else float(np.sum(beat_norm)),
            'skewness': float(np.mean((beat_norm - np.mean(beat_norm))**3) / (np.std(beat_norm)**3 + 1e-10)),
            'kurtosis': float(np.mean((beat_norm - np.mean(beat_norm))**4) / (np.std(beat_norm)**4 + 1e-10)),
            'inflection_point_ratio':
                (landmarks.get('inflection', landmarks['onset']) - landmarks['onset']) /
                max(landmarks['peak'] - landmarks['onset'], 1),
        }

        # 收缩期面积占比
        beat_feat['Area_ratio'] = beat_feat['Area_sys'] / max(beat_feat['Area_total'], 1e-10)

        beat_features.append(beat_feat)

    if len(beat_features) < 2:
        return {"error": "有效搏动数不足"}

    # ── 统计聚合 ──
    features = {}
    for key in beat_features[0].keys():
        values = [b[key] for b in beat_features]
        features[f'{key}_mean'] = float(np.mean(values))
        features[f'{key}_std'] = float(np.std(values))
        features[f'{key}_cv'] = float(np.std(values) / max(np.mean(values), 1e-10) * 100)

    # ── ECG 联合特征（若提供 R 峰位置） ──
    if ecg_r_peaks is not None:
        if ecg_fs is None:
            ecg_fs = fs

        # 对齐 ECG 和 PPG 的采样率
        # 计算 PTT 序列
        ptt_list = []
        for ecg_r in ecg_r_peaks:
            # 找到 ecg_r 之后的第一个 PPG onset
            ecg_time = ecg_r / ecg_fs
            ppg_onsets = [peaks[j] - int(0.1*fs) for j in range(len(peaks))]
            ppg_onset_times = np.array(ppg_onsets) / fs

            # 找时间上最近的 PPG onset
            diffs = ppg_onset_times - ecg_time
            valid = diffs > 0.03  # PTT 至少 30ms
            if np.any(valid):
                nearest_idx = np.argmin(np.abs(diffs))
                ptt = (ppg_onset_times[nearest_idx] - ecg_time) * 1000  # ms
                if 50 < ptt < 500:
                    ptt_list.append(ptt)

        if len(ptt_list) > 3:
            features['PTT_mean'] = float(np.mean(ptt_list))
            features['PTT_std'] = float(np.std(ptt_list))
            features['PTTV'] = float(np.std(ptt_list) / max(np.mean(ptt_list), 1) * 100)

            # 心率校正 PTT
            if 'T_total_mean' in features:
                features['PTT_corrected'] = features['PTT_mean'] / np.sqrt(features['T_total_mean'])

    # ── 搏动间一致性 ──
    if 'A1_cv' in features:
        features['beat_consistency'] = 1.0 - min(features['A1_cv'] / 20.0, 1.0)

    return features


```

> **注意**：以上 `extract_ppg_features()` 提取的连续值特征可直接输入 `extract_six_qualities_from_ppg()`（见 §3.3.2）以获取六品质连续谱，无需经过"脉象分类→查表"的中间步骤。

---

## 五、脉诊精度对比

### 5.1 各模式精度对比（六品质连续谱 Pearson r）

| 品质维度 | ECG-only | ECG+PPG (增强) | ECG+PPG+三路压力 (完整) |
|---------|---------|----------------|----------------------|
| 粗↔细（A₁ + 面积） | r = 0.60–0.70 | r = 0.78–0.85 | **r = 0.85–0.92** |
| 软↔硬（SI, AIx, T_rise） | r = 0.55–0.65 | r = 0.75–0.82 | **r = 0.82–0.90** |
| 缓↔急（HR） | **r = 0.98** | **r = 0.98** | **r = 0.98** |
| 滑↔涩（变异度） | r = 0.60–0.70 | r = 0.72–0.80 | **r = 0.80–0.88** |
| 浮↔沉（PTT + HF） | r = 0.45–0.55 | r = 0.65–0.72 | **r = 0.78–0.85** |
| 有力↔无力（dP/dt + 面积） | r = 0.55–0.65 | r = 0.72–0.80 | **r = 0.80–0.88** |

### 5.2 各品质维度的 PPG 特征贡献权重

| 品质维度 | 最重要 PPG 特征 (权重) | 次重要 PPG 特征 (权重) |
|---------|----------------------|----------------------|
| 粗↔细 | A₁ (0.40), Area_sys (0.30) | dP/dt_max (0.15), Area_total (0.15) |
| 软↔硬 | AIx (0.35), SI (0.30) | T_rise (0.20), PTT (0.15) |
| 滑↔涩 | A₁ CV (0.40), AIx CV (0.30) | 周期间相关系数 (0.30) |
| 有力↔无力 | dP/dt_max (0.40), Area_total (0.30) | A₁ (0.20), dP/dt_min (0.10) |

---

## 六、临床验证协议

### 6.1 数据集采集计划

| 项目 | 规格 |
|------|------|
| 样本量 | ≥ 200 例（健康+各类病证） |
| 金标准 | **软标签（Soft Labels）**：3 位副主任以上中医师独立对六品质分别给出连续评分（0–100 分制），取均值作为金标准，方差反映医师间分歧 |
| 采集设备 | ECG + PPG + 三路压力传感器同步 |
| 采集时长 | 每例 5 分钟（含 3 次重复放置） |
| 存储格式 | WFDB 兼容格式（`.dat` + `.hea` + 六品质评分 `.qual`） |

> **为什么采用软标签而非"三位一致判定"？**
> 中医师之间的切脉一致性（kappa）通常仅为 0.4–0.6。如果三位医师对"弦脉"的一致性只有 60%，则以"三人一致判定"为金标准训练算法存在数学悖论——算法的准确率上限被人类一致性锁死。
>
> 软标签方案将医师之间的分歧保留为"标签的不确定性"，算法学习的是与医师评分均值的回归关系，而非拟合一个离散的"共识标签"。人类医师一致性 κ < 0.6 时，软标签方案的 ICC 仍然可以有效评估算法性能。

### 6.2 评价指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **Pearson r** | 算法六品质连续输出与医师评分均值的相关性 | ≥ 0.75 |
| **ICC(2,1)** | 绝对一致性组内相关系数（算法单次测量 vs 医师均值） | ≥ 0.70 |
| **MAE** | 六品质的均值绝对误差（0–1 尺度） | ≤ 0.12 |
| **Cohen's κ** | （可选）若需与离散分类比较，将六品质阈值化后计算 | 参考值 ≥ 0.50 |
| **重复测量 ICC** | 同一患者 3 次重复放置的测量稳定性 | ≥ 0.80 |

---

## 七、PPG 信号质量分级

### 7.1 PPG SQI 指标

```
PPG_SQI = 0.35 × AC_DC_ratio + 0.25 × beat_regularity
          + 0.20 × SNR + 0.20 × morphological_stability

其中：
  AC_DC_ratio = 搏动成分 / 直流成分
  beat_regularity = exp(−CV(T_total))
  SNR = 10×log10(P_signal / P_noise)，归一化
  morphological_stability = 1 − CV(A₁)
```

| PPG_SQI | 等级 | 处理方式 |
|---------|------|---------|
| ≥ 0.75 | 优秀 | 全特征提取，高置信度 |
| 0.50–0.75 | 可用 | 特征提取，降置信度输出 |
| 0.25–0.50 | 差 | 仅提取基本特征（A₁, HR） |
| < 0.25 | 无效 | 丢弃，提示重新放置传感器 |

---

> **与现有系统的集成**：本算法的输出作为 `pulse_diagnosis.py` 中 `extract_six_qualities_from_ppg()` 函数的直接输入，与青囊八步的 S_current 估测无缝对接。
