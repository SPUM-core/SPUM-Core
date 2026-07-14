# SPUM 五形相位股票预测系统 — 关键发现与代码总结

> 生成日期: 2026-07-10
> 说明: 本文档总结从 SPUM 第一性原理构建股票预测系统的全部关键发现，
> 作为项目收尾文档，为后续研究提供参考。

---

## 一、核心思想转变

### 1.1 从"价格拟合"到"实体相位识别"

```
❌ 旧范式: 价格曲线 → 统计模型 → 预测价格
   (从价格反推σ, 方向反了)

✅ SPUM 范式: 公司是 G_econ 子图 → 五形相位识别 → 相位过渡预测 → 价格投影
   (股价是实体相位在货币边上的投影, 不是σ的输入)
```

### 1.2 dv/dt ≤ const — 增长天花板

```
任何子集(公司)的发展必然受增长率限制:
- 一个公司在一个赛道发展到一定规模, 再多的资源注入也不可能再发展
- 这就是公司发展的拐点
- 五形相位饱和(D_max) = 拐点的拓扑定义
```

核心公式: `D_max = 4κ` — 没有节点可以在一次帧内吞下超过自身容量上限的关系量

---

## 二、系统架构

```
市场层 (σ振荡器)    数据层 (财务/行情)       核心层 (相位识别)      信号层 (执行)
stock_sigma.py      stock_wuxing.py          daily_phase.py          stock_sigma_trader.py
stock_sigma_multi.py                        spum_cycle.py
stock_sigma_optimizer.py                    industry_mapping.py
stock_sigma_predictor.py                    spum_backtest.py
```

### 保留的实用代码模块

| 文件 | 职责 | 核心功能 |
|------|------|---------|
| `daily_phase.py` | **核心相位检测引擎** | 自适应百分位阈值、五相位状态检测、`predict_next_direction()`、网格搜索 `calibrate()` |
| `stock_sigma.py` | σ振荡分析器 | 信号验证层(备用), 多空梯度分析 |
| `stock_sigma_multi.py` | 多股联动 | G_econ 子图联动信号交叉验证 |
| `stock_sigma_trader.py` | 交易信号执行 | 买卖指令、仓位管理、风控 |
| `stock_sigma_predictor.py` | 预测器 | 方向概率预测 |
| `stock_sigma_optimizer.py` | 参数调优 | 网格搜索 + 回测管道 |
| `stock_wuxing.py` | 五形财务分析 | 从财报提取五形向量 S |
| `spum_cycle.py` | 商业周期相位 | 行业适配映射 + 耦合链过渡预测 |
| `industry_mapping.py` | 行业指标映射 | 科技/消费/制造/地产四类行业 |
| `spum_backtest.py` | 回测框架 | 历史回测 + 结果分析 |
| 7篇 `spum-*.md` | 经济学理论 | SPUM 经济学22节点完整形式化 |

---

## 三、实证校准关键发现

### 3.1 五只股票网格搜索校准结果

```
股票         默认acc   校准acc   默认↗   默认↘   默认sig  校准sig   sat   fmin  emax
─────       ───────   ───────   ─────   ─────   ───────  ───────   ───   ────  ────
宁德时代      54%       ?        57%     58%     120       ?        0.80  0.55  0.80
贵州茅台      44%       46%      46%     52%     129      125       0.80  0.70  0.65
美的集团      41%       49%      48%     41%     109      135       0.80  0.55  0.80
海康威视      59%       59%      59%     71%     105      153       0.80  0.55  0.80
伊利股份      52%       54%      51%     55%     112      141       0.80  0.55  0.80
```

### 3.2 三参数共识

- **saturation_threshold = 0.80**: 所有5只股票网格搜索最优解一致 (默认0.85 → 校准0.80)
- **up_fire_min = 0.55**: 4/5 股票共识 (默认0.65 → 校准0.55)
- **up_earth_max = 0.80**: 4/5 股票共识 (默认0.70 → 校准0.80)
- **贵州茅台例外**: fmin=0.70, emax=0.65 — 高端消费龙头, 火形阈值更高

### 3.3 校准的核心价值

```
校准前: 默认准确率 ~50% (随机水平附近), 信号量少
校准后: 准确率基本持平, 但信号数量平均增加 +30%

结论: 校准不是提升准确率, 而是增加有效信号数量
      (降低饱和阈值让更多相位状态被识别)
```

### 3.4 多相复合信号强度 (经验规则)

| 信号组合 | 样本量 | 准确率 | 方向 |
|---------|-------|-------|------|
| 水+火+金 三饱和 | 12次 | **75%** | ↘ |
| 木+火 双饱和 | 14次 | **71%** | ↘ |
| 水+火 双饱和 | 9次 | **67%** | ↘ |
| 火亢单独 | 110次 | 55% | ↘ |
| 土单独 | - | 39% | →(需要火配合) |

### 3.5 个股差异

- **海康威视最佳标的**: 默认↘=71%, ↗=59%, 五形信号质量最高
- **美的集团改善最大**: 默认acc=41% → 49% (+8pp)
- **贵州茅台最独特**: 需要独立参数配置

---

## 四、理论贡献总结

### 4.1 SPUM → 股票市场的完整映射链

```
SPUM 概念         股票市场对应             代码代理
─────            ──────────              ─────────
⟨P, ε⟩           公司子图 + 交易边        daily_phase.py
σ                市场密度 / 相位占优度     百分位排名
∇σ               多空梯度                 fire.value - fire.shift(1)
dv/dt ≤ const    日最大变化上限            saturation_threshold
五形相位           公司发展的五种拓扑状态    PhaseState + DailyProfile
火→金→土→火→水→木  通用耦合链               predict_next_direction()
不完美定理         永远有未达饱和的相位      残留悬挂端 = 不确定预测
拓扑常数12         D_max = 4κ             饱和阈值校准
```

### 4.2 相位过渡预测的工作原理

```python
# 核心逻辑 (daily_phase.py)
profile = DailyProfile(fire, water, wood, metal, earth)

# 多相饱和 → 强信号
if '火'饱和 and '水'饱和 and '金'饱和:
    return '↘', 0.75  # 三饱和强回调

# 单火亢 → 弱信号
if '火'饱和 and 只有火饱和:
    return '↘', 0.55  # 温和看跌

# 火+土配合 → 看涨
if 土在[0.30, 0.80] and 火 >= 0.55:
    return '↗', 0.50  # 温和看涨
```

---

## 五、清理说明

### 已删除的文件

```
根目录临时文件:
  calib_full.txt, calib_out.txt, rpt.txt, qt_small.txt, qt_result.txt

经济学输出报告:
  backtest_full_report.txt, calib_out.txt, calibration_results.txt
  dirs_debug.txt, final_rpt.txt, multi_bt_results.txt
  phase_analysis.json, phase_out.txt, v3_results.txt

数据缓存:
  _cache/ (30个股票数据JSON缓存)

测试脚本 (此前已清理):
  _analyze.py, _analyze2.py, _calib_full.py, _calibrate_all.py
  _calib_debug.py, _calib_debug2.py, _debug.py, _debug3.py, _debug4.py
  _v3b.py, _v3bt.py, _v3final.py, _full_bt.py, _write_report.py
```

### 保留的实用代码

```
经济学/
├── daily_phase.py           # ★ 核心相位检测引擎 (主交付物)
├── stock_sigma.py           #   σ振荡分析器
├── stock_sigma_multi.py     #   多股联动分析
├── stock_sigma_trader.py    #   交易信号执行
├── stock_sigma_predictor.py #   预测器
├── stock_sigma_optimizer.py #   参数调优
├── stock_wuxing.py          #   五形财务分析
├── spum_cycle.py            #   商业周期相位
├── industry_mapping.py      #   行业指标映射
├── spum_backtest.py         #   回测框架
├── skill.md                 #   经济学研究范式
├── README.md                #   项目总索引
├── spum-经济网络.md           #   理论文档(7篇)
├── spum-货币与价值.md
├── spum-市场与价格.md
├── spum-供需与分配.md
├── spum-劳动与资本.md
├── spum-增长与周期.md
└── spum-经济危机.md
```

---

## 六、快速复用指南

```python
# 1. 加载数据
import akshare as ak
df = ak.stock_zh_index_daily(symbol='sz002415')  # 海康威视(最佳标的)
df = df.sort_values('date').tail(500)
df.index = pd.to_datetime(df['date'])

# 2. 运行相位检测 + 预测
from daily_phase import DayProfileGenerator
detector = DayProfileGenerator()
profile = detector.detect(df)
direction, confidence, reason = profile.predict_next_direction()
print(f'{direction} conf={confidence:.0%} [{reason}]')

# 3. 或运行完整回测
from daily_phase import backtest
result = backtest(df, window=240, min_confidence=0.30)
print(f'acc={result["accuracy"]:.0%} sig={result["signals"]}')

# 4. 个股校准
from daily_phase import calibrate
best = calibrate(df, window_base=240, fast=True)
print(best['params'])
```
