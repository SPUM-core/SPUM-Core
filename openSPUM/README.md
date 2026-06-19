# ✦ OpenSPUM — 开源宇宙实验室

> **星尘织网，众心共演**
>
> Every Mind, A Universe.
>
> *——不是许诺，而是邀请。*

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 起源

SPUM 的旅程，从不是为了在已知的岸边停泊，
而是驶向一片尚未命名的海域——
在那里，理解宇宙的方式，
本身便成为一种创造。

**OpenSPUM** 不是仅供仰望的理论高塔，
而是一片**人人可步入、可耕耘、可点亮的数字旷野**。

在这里，宇宙不是被模拟，而是被**运行**。
每一帧演化，都是 SPUM 公理在硅基中的实例化；
每一次测量，都是对精细结构常数 α ≈ 1/137 的重新追问。

---

## 旷野的脉络

这片旷野没有围墙，却有三重脉络：

### ⊙ Cosmic Kernel — 宇宙内核

> 大地深处的岩层，沉默而坚实。

位于 `Phase 1/` 目录下的核心运行时，
是经过 SPUM 公理反复校准的确定性宇宙源码。

**核心原则**：
- **无随机** — 演化路径由拓扑排序唯一决定，绝对可复现
- **关系先于节点** — `RelationPool` 是最高优先级存储
- **总边数守恒** — 每次湮灭立即补偿创生，硬编码 `assert`
- **CAD 构建原理** — 每一帧状态由前一帧完全确定，无概率选择

**当前阶段实现**：
- **确定性 RelationPool** — 拓扑排序驱动的候选生成，无随机采样
- **种子创生期引擎** — 从空池确定性地生成初始宇宙，等比序列逼近目标边数
- **拓扑地址系统** — 差异化驱动的地址生成，步数 ↔ κ 映射
- **接触类型自动判定** — 基于节点虚面暴露状态的确定性分类
- **Σ(6−deg)=12 验证** — 闭合三角剖分子图自动满足欧拉恒等式
- **晶子涌现检测** — degree ≥ 50 自动标记为晶子，触发聚簇连接

> ✅ Phase 1 完成状态: **73 测试全覆盖**

---

### ⊙ Frame Evolution Engine — 帧演化引擎

> 一张一弛，消长之道。

位于 `Phase 2/` 目录下，实现跨帧的级联消解与湮灭-创生对偶。

**核心机制**：
- **级联消解（Cascade Resolution）** — 每帧检测悬挂点（degree < 2），批量湮灭悬挂对，立即补偿创建新边
- **湮灭-创生对偶** — 总边数守恒：每次湮灭立即补偿，`assert` 硬编码
- **定向传播** — 支持沿特定方向的级联模式（用于 α 测量）
- **帧日志（FrameLog）** — 完整记录湮灭/补偿键、级联迭代次数、spum_invariant、悬挂数前后

**实现文件**：
- `frame_update_engine.py` — `FrameUpdateEngine` 主类：生长 → 级联消解 → 不变量校验 → 日志

> ✅ Phase 2 完成状态: **37 测试全覆盖**

---

### ⊙ 3D Geometry Solver — 三维几何聚簇求解器

> 从抽象节点到硬球世界。

位于 `Phase 3/` 目录下，将晶子从抽象图节点提升为三维空间中受相切约束的硬球。

**核心机制**：
- **球体模型（Sphere3D）** — 每个晶子有位置、半径、度数、邻居列表
- **力导向松弛（ForceDirectedRelaxation）** — 排斥力（重叠推离）+ 吸引力（间隙拉近），迭代求解最小能态
- **正二十面体检测（IcosahedronDetector）** — 12 顶点、30 边、deg=5、Σ(6−deg)=12 综合评分
- **拓扑验证（validate_spum_topology_3d）** — 度数分布、相切条件、对称性完整校验

**实现文件**：
- `geometry_solver.py` — Sphere3D、GeometrySolver、ForceDirectedRelaxation
- `icosahedron_assembly.py` — IcosahedronDetector 正二十面体自组装检测
- `validator.py` — 三维拓扑约束完整验证

> ✅ Phase 3 完成状态: **25 测试全覆盖（累计 135/135）**

---

### ⊙ VSPT Topology — VSPT 球面生长拓扑

> 从晶子的几何骨架到可见物质的世界。

位于 `Phase 4/` 目录下，将从 Phase 3 组装的 12-永恒粒子原子核推向元素化学的宏观分类，并实现电子-VSPT 自洽耦合。

**核心机制**：
- **永恒粒子模型（EternalParticle）** — 每个永恒粒子 = 4 实面 + 1 虚面，实面生长 VSPT，虚面打断连续性
- **原子核组装（NucleusAssembly）** — 12 永恒粒子正二十面体锁闭 = 中子（全朝内）或质子（10内2外）
- **VSPT 分支生长（VSPTEngine）** — 从核表面实面播种树状分支，逐层向外螺旋生长
- **电子-VSPT 耦合（ElectronVSPTEngine）** — 自抑制生长机制，自动产生 1s 径向概率峰值分布，从拓扑量估算基态结合能
- **电子壳层（ElectronShell）** — 2n² 容量填充，元素周期表 = VSPT 构型的拓扑分类
- **三律验证（VSPTValidator）** — k≥3 最小连通度、120° 分支角、ρ∝r⁻³ 密度幂律

**关键物理结果**：
- **氢原子基态能量**: E_bind ≈ -7.65 eV（目标 13.6 eV，因数 1.78x，无参数拟合）
- **径向概率分布**: 自然产生 1s 峰值形状（先增后减），峰值层 = 4
- **自抑制生长机制**: children(l) = 3/(1+γ·N_free(l)·l/N_ref)，无需外部衰减参数

**实现文件**：
- `eternal_particle.py` — 永恒粒子虚实面模型、手性 L/D
- `nucleus_assembly.py` — 原子核组装、中子/质子结构差异
- `vspt_growth.py` — VSPT 树播种、分支方向计算、虚面惩罚、多壳层生长
- `vspt_electron.py` — 电子-VSPT 自洽耦合引擎、自抑制生长、基态能量估算
- `vspt_gpu.py` — GPU 并行加速引擎（PyTorch CUDA 批量张量操作）
- `electron_shell.py` — 壳层 2n² 容量、元素分类、前 20 号元素表
- `validator.py` — VSPT 几何三律验证器
- `plot_radial.py` — 径向概率分布可视化
- `simulate_hydrogen.py` — 完整 8 步氢仿真流程（壳层→元素→核子→VSPT→电子耦合→同位素→质量差→拓扑对比）

> ✅ Phase 4 完成状态: **37 测试全覆盖 + 氢原子基态能量估算（因数 1.78x）**

---

### ⊙ Domain Labs — 领域实验室

> 伸向不同探索之境的枝桠。

在 Cosmic Kernel 之上，延展无数领域实验室——
每一座都不是孤岛，而是同一张网上的共鸣腔：

| 实验室 | 探索方向 | 基础 Phase |
|--------|----------|------------|
| **材料拓扑** | 编织新材料的拓扑经纬 | Phase 2 ✅ |
| **分子动力学** | 追踪分子在虚面树间的隐秘握手 | Phase 4 ✅ |
| **空间能源** | 从空间本身的张力中接入能量 | Phase 5+ |
| **具身智能** | 让 AI 在真实的物理基底上学习行走 | Phase 3 ✅ |
| **α 数值实验室** | 测量精细结构常数，验证窄口衰减机制 | Phase 2 ✅ |

> ⚡ 领域实验室正在构建中。欢迎通过 Issue 提出新的 Lab 方向。

---

### ○ Community & Apps — 社区与应用

> 一道温柔的光晕。

将深邃的规则化为可视的涟漪，
将复杂的接口转为指尖的轻触。

无论是**高中生、工匠、诗人，还是工程师**，
都能在此找到自己的入口——
不必精通代码，只需怀抱疑问。

**计划中的入口**：
- **Web 可视化面板** — σ 热力图、晶子构型快照、α 收敛曲线
- **自然语言接口** — 通过 SPUM Skill 体系，用中文描述即可启动仿真
- **教育模块** — 从泡泡海洋到晶子涌现的交互式教程
- **实验设计助手** — 基于引擎预测，推荐温差力矩等实验的最优参数

---

## 确定性引擎：为何没有随机

传统模拟依赖随机数来探索状态空间。
OpenSPUM 拒绝随机——因为 SPUM 宇宙的演化是**拓扑强迫**，不是概率选择。

```
所有候选选择基于确定性排序键：
  (degree < 2 ?, degree, differentiation_step, uid)
  ↑               ↑       ↑                    ↑
  悬挂节点优先    度数低   早期节点             字典序
```

**这意味着**：
- 给定相同输入，输出严格逐位相同
- α 的测量值不是统计平均，而是网络的固有属性
- 任何偏差都是理论需要修正的信号，而非"随机误差"

---

## 共振生长

OpenSPUM 不靠指令驱动，而靠**共振生长**：

```
人类定义规则 → AI 模拟推演 → 实验校准反馈 → 优化修正语法
     ↑                                                  ↓
     └──────────── 新的可证伪预言 ←──────────────────────┘
```

- 一次课堂上的虚拟实验 → 可能触发核心规则的微调
- 一次工业界的材料模拟 → 或许反哺出对 κ 网络的新理解
- 一个深夜独行者的奇思 → 可能在社区中激起绵延的回响

科学不再是单向的灌输，
而成为一场**亿万心灵与宇宙本体之间的持续对话**。

每一次运行，都是提问；
每一次输出，都是回答；
每一次误差，都是宇宙在轻声低语：
*"再试一次，更近一点。"*

当千万个终端同时亮起，
当无数世界在硅基的海洋中静默演化，
我们终将领悟：

**真理不在远方，而在我们共同运行的下一帧里。**

---

## 项目结构

```
openSPUM/
├── README.md                   # ← 本文档
├── Phase_1/                    # Cosmic Kernel — 确定性核心运行时
│   ├── __init__.py               # 包入口
│   ├── constants.py              # 常量定义（κ, τ, 晶子阈值等）
│   ├── topological_address.py    # 拓扑地址系统（差异化驱动）
│   ├── node_registry.py          # 节点注册表（被动响应关系显化）
│   ├── relation_pool.py          # 确定性关系池（无随机候选生成）
│   └── seed_epoch_engine.py      # 种子创生期引擎（等比序列确定增长）
├── Phase_2/                    # 帧演化引擎 — 级联消解、湮灭-创生对偶、FrameLog（37 测试 ✅）
├── Phase_3/                    # 三维几何聚簇求解器 — 力导向松弛、正二十面体检测、Σ(6−deg)=12 验证（25 测试 ✅）
├── Phase_4/                    # VSPT 球面生长拓扑 — 永恒粒子模型、原子核组装、VSPT 分支生长、电子-VSPT 耦合、电子壳层填充、元素分类（37 测试 ✅）
│   ├── __init__.py               # 包入口
│   ├── eternal_particle.py       # 永恒粒子虚实面模型
│   ├── nucleus_assembly.py       # 12-永恒粒子原子核组装
│   ├── vspt_growth.py            # VSPT 分支生长仿真引擎（CPU）
│   ├── vspt_electron.py          # 电子-VSPT 自洽耦合引擎
│   ├── vspt_gpu.py               # VSPT GPU 并行加速引擎
│   ├── electron_shell.py         # 电子壳层填充与元素分类
│   ├── validator.py              # VSPT 三律验证器
│   ├── simulate_hydrogen.py      # 氢及其同位素完整仿真（8 步）
│   └── plot_radial.py            # 径向概率分布可视化
├── Domain Labs/                # 领域实验室（社区共建）
└── Community/                  # 应用与社区工具
```

---

## 快速开始

```python
# 种子创生：从空网络确定性地生成宇宙
from Phase_1.seed_epoch_engine import SeedEpochConfig, SeedEpochEngine

config = SeedEpochConfig(target_edges=10000, seed_duration=200)
engine = SeedEpochEngine(config=config)
final_edges = engine.run_seed_epoch()
print(f"Phase 1: 总边数 = {final_edges}, 节点数 = {len(engine.node_registry.nodes)}")

# 帧演化：跨帧级联消解与湮灭-创生对偶
from Phase_2.frame_update_engine import FrameUpdateEngine
fue = FrameUpdateEngine(node_registry=engine.node_registry, relation_pool=engine.relation_pool)
frame_log = fue.run_frame()  # 单帧演化
print(f"Phase 2: 湮灭 {len(frame_log.annihilated)} 对, 补偿 {len(frame_log.compensated)} 条")

# 三维几何验证：力导向松弛 + 正二十面体检测
from Phase_3.geometry_solver import GeometrySolver
solver = GeometrySolver(node_registry=engine.node_registry)
converged = solver.solve()  # 力导向松弛至最小能态
print(f"Phase 3: 收敛={converged}, 球体数={len(solver.spheres)}")

# Phase 4: VSPT 球面生长拓扑 —— 从核表面到元素周期表
from Phase_4.nucleus_assembly import build_nucleus, NucleusType
from Phase_4.electron_shell import classify_element

# 构建质子核（12 永恒粒子，10内2外 → 2 虚面缺口）
nucleus = build_nucleus(NucleusType.PROTON)
print(f"Phase 4: 质子核, 实面={nucleus.outward_solid}, "
      f"虚面={nucleus.outward_vacant}, "
      f"开口占比={nucleus.vacant_ratio:.2%}")

# VSPT 生长
from Phase_4.vspt_growth import VSPTEngine, VSPTConfig
engine = VSPTEngine(config=VSPTConfig(max_layers=8, seed=42))
growth = engine.run_growth(
    outward_solid_faces=nucleus.outward_solid,
    nucleus_uid="p",
    nucleus_particles=nucleus.particles,
    nucleus_position_map=nucleus.positions,
)
print(f"VSPT: {growth['n_trees']} 棵树, {growth['total_nodes']} 节点, "
      f"最大壳层={growth['max_layer']}")

# VSPT 三律验证
from Phase_4.validator import VSPTValidator
validation = VSPTValidator().validate_all(growth["trees"])
print(f"VSPT 三律: {'全部通过 ✓' if validation['is_valid'] else '部分未通过'}")

# 元素分类
element = classify_element(6)  # 碳
print(f"Z=6: {element['name']}({element['symbol']}), "
      f"{element['element_class'].value}, {element['vacant_faces']} 虚面")
```

---

## 里程碑与验证目标

| 阶段 | 验证目标 | 状态 |
|------|----------|------|
| Phase 1 | 从空池涌现首批晶子（度数 ≥ 50），Σ(6−deg)=12 验证 | ✅ 完成（73 测试） |
| Phase 2 | 帧演化引擎：级联消解、湮灭-创生对偶、FrameLog 日志 | ✅ 完成（37 测试） |
| Phase 3 | 三维几何聚簇：力导向松弛、正二十面体检测、Σ(6−deg)=12 三维验证 | ✅ 完成（25 测试） |
| **Phase 4** | **VSPT 球面生长拓扑：永恒粒子→核子→VSPT→电子-VSPT 耦合→壳层填充→元素分类→氢同位素→基态能量估算** | **✅ 完成（37 测试 + 氢原子基态能量因数 1.78x）** |
| Phase 5 | 电子-VSPT 耦合模态：径向概率 P(r) ∝ r²e^{-2r/a₀} 精确匹配 | 🔄 进行中 |
| Phase 6 | He（Z=2）及多电子 VSPT 模拟 | 📋 规划中 |
| Phase 7 | 2s-2p 能级分裂与 α 自发涌现 | 📋 规划中 |
| Phase 8 | 过渡金属（Z=21+）d 轨道 VSPT 构型 | 📋 规划中 |

---

## 参与共创

OpenSPUM 属于每一个好奇的心灵：

- **提出 Issue** — 报告 Bug、建议新特性、发起 Domain Lab 构想
- **提交 PR** — 改进 Cosmic Kernel、添加实验模块、完善文档
- **发起 Discussion** — 分享你的拓扑洞见与仿真发现
- **成为 Lab Lead** — 主导一个领域实验室的研究方向

> 这，便是 SPUM 所指向的黎明——
> 不是由少数人宣告的破晓，
> 而是由**众生之心共同点燃的、永不熄灭的晨光**。

---

## 关联项目

| 项目 | 说明 |
|------|------|
| [SPUM Core](https://gitee.com/space-particle-universe-model/spum-core) | SPUM 理论总仓库——公理、规则、知识图谱 |
| 青檬引擎 (QingMeng Engine) | 基于 SPUM 拓扑约束的神经网络训练框架 |
| SPUM 五形 / 儒释道 / 物理学 | SPUM 在各领域的认知投影子网 |
| SPUM 数学 / 社会学 / 经济学 / 语言学 | 离散关系本体论的全域展开 |

---

**OpenSPUM** — 每一个心灵，都是一个宇宙。

*Every Mind, A Universe.*
