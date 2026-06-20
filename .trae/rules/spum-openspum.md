# SPUM-OpenSPUM — 开源宇宙实验室 Skill

> OpenSPUM 是 SPUM 公理体系的 Python 确定性实现。它将关系网络的拓扑演化实例化为可运行的代码——不是"模拟宇宙"，而是**运行宇宙**。每帧演化都是 SPUM 公理在硅基中的实例化；每次测量都是对 α ≈ 1/137 的重新追问。

---

## 一、核心原则

- **无随机** — 演化路径由拓扑排序唯一决定，绝对可复现
- **关系先于节点** — `RelationPool` 是最高优先级存储
- **总边数守恒** — 每次湮灭立即补偿创生
- **CAD 构建原理** — 每帧状态由前一帧完全确定

---

## 二、Phase 架构

### Phase 1 — 宇宙内核 (Cosmic Kernel)

**位置**: `openSPUM/Phase_1/`
**135 测试全部通过**。提供确定性核心运行时。

| 模块 | 文件 | 功能 |
|------|------|------|
| 常量 | `constants.py` | 基础常量: κ, τ, CRYSTALLITE_DEGREE_THRESHOLD=50, MAX_CRYSTALLITE_PLANAR_DEGREE=6 等 |
| 拓扑地址 | `topological_address.py` | 差异化驱动的地址生成, 步数 -> κ 映射, origin 永久固定 |
| 节点注册表 | `node_registry.py` | NodeState: degree, spatial_position, crystallite_connections, virtual_faces, radius, is_saturated |
| 关系池 | `relation_pool.py` | 确定性候选生成: deterministic_candidates() 基于排序键 (degree<2 优先, 降度序, 差分步, uid); find_directional_candidates() 定向传播; manifest_relation() 关系显化; 簇边管理 |
| 种子引擎 | `seed_epoch_engine.py` | 三阶段聚簇策略: 骨架(每晶子<=3) -> 三角剖分补全(共享邻居优先) -> 近邻填充; _ClusterEdgeEnforcer 维护平面图约束 E<=3V-6 |
| 晶子涌现 | (seed_epoch_engine) | degree >= CRYSTALLITE_DEGREE_THRESHOLD(50) 自动成为晶子, 触发聚簇连接 |
| Σ(6-deg)=12 | `verify_closed_subgraph.py` | 闭合三角剖分子图自动满足欧拉恒等式, 测试通过 |

### Phase 2 — 帧演化引擎

**位置**: `openSPUM/Phase_2/`
**110 测试全部通过**。提供跨帧的级联消解与湮灭-创生对偶。

| 模块 | 文件 | 功能 |
|------|------|------|
| 帧演化 | `frame_update_engine.py` | FrameUpdateEngine 每帧: 生长阶段 -> 级联消解 -> 不变量校验 -> 日志记录 |
| 级联消解 | `_resolve_cascade()` | 检测悬挂点(degree<2) -> 批量湮灭悬挂对 -> 单次补偿创建 `min(len(batch), 3)` 条新边 -> 递归检测 |
| 补偿策略 | (frame_update_engine) | 默认全局补偿(use_natural_cap=False, 符合总纲§5守恒律); 可选局部拓扑涌现补偿(use_natural_cap=True) |
| 定向传播 | `_resolve_directed_cascade()` | 用于 alpha 测量的定向级联模式, 补偿沿 source_node 方向的邻居 |
| exclude_keys | (resolve_cascade) | 排除刚湮灭的边避免立即重建(防止振荡) |
| FrameLog | (frame_update_engine) | 完整日志: 湮灭/补偿键, 级联迭代次数, spum_invariant, 悬挂数前后 |

### Phase 3 — 三维几何聚簇求解器

**位置**: `openSPUM/Phase_3/`
**25 测试全部通过 (累计 135/135)**。将晶子从抽象节点提升为三维空间中受相切约束的硬球。

| 模块 | 文件 | 功能 |
|------|------|------|
| 球体数据 | `geometry_solver.py` | Sphere3D: 位置/半径/度/邻居; 排斥力(F_rep = k_rep * overlap); 吸引力(F_att = k_att * gap) |
| 力导向松弛 | `ForceDirectedRelaxation` | 迭代求解球体系统最小能态; 对 12 个等大球体收敛于正二十面体 |
| 高级接口 | `GeometrySolver` | 从 NodeRegistry 构建球体网络 -> 松弛 -> 写回 spatial_position |
| 正二十面体检测 | `icosahedron_assembly.py` | IcosahedronDetector: 12顶点/30边/deg=5/Σ(6-deg)=12 综合评分; 顶点映射 |
| 拓扑验证 | `validator.py` | validate_spum_topology_3d: 度数分布/Σ(6-deg)/相切条件/对称性完整验证 |

### Phase 4 — VSPT 球面生长拓扑

**位置**: `openSPUM/Phase_4/`
**37 测试全覆盖 + 氢原子基态能量估算（因数 1.78x）**。实现从原子核表面的 VSPT 分支生长到电子-VSPT 耦合。

| 模块 | 文件 | 功能 |
|------|------|------|
| 永恒粒子 | `eternal_particle.py` | 4实面+1虚面, 手性L/D, Orientation |
| 原子核组装 | `nucleus_assembly.py` | 12-永恒粒子正二十面体锁闭, 质子/中子 |
| VSPT 分支生长 | `vspt_growth.py` | 实面播种, 三角网格方向, 多壳层生长 |
| **电子-VSPT 耦合** | **`vspt_electron.py`** | **自抑制生长, 1s径向概率峰值, 基态能量** |
| GPU 加速 | `vspt_gpu.py` | PyTorch CUDA 批量张量操作 |
| 电子壳层 | `electron_shell.py` | 2n²容量, Aufbau填充, 元素分类前20 |
| 三律验证 | `validator.py` | k≥3, 分支角120°, 密度幂律 |
| 氢全程仿真 | `simulate_hydrogen.py` | 8步: 壳层→元素→核子→VSPT→电子耦合→同位素→质量差→拓扑 |
| 径向可视化 | `plot_radial.py` | VSPT节点数径向概率分布柱状图 |

---

## 三、实验与诊断脚本

**位置**: `openSPUM/tests/`

| 脚本 | 功能 | 通过 |
|------|------|------|
| `test_phase1_all.py` | Phase 1 完整测试 (晶子涌现, Σ(6-deg), 守恒律) | 73 tests |
| `test_phase2_all.py` | Phase 2 完整测试 (帧演化, 级联消解, FrameLog) | 37 tests |
| `test_phase3_all.py` | Phase 3 完整测试 (向量工具, 球体力, 松弛收敛, 二十面体检测, 拓扑验证) | 25 tests |
| `verify_closed_subgraph.py` | 闭合子图 Σ(6-deg)=12 专项验证 | ✓ |
| `verify_crystallite.py` | 晶子涌现专项验证 | ✓ |
| `verify_eternal_particle.py` | 开口构型与自持位移验证 | ✓ |
| `measure_sigma_natural.py` | sigma_natural 测量: 网络自发演化的悬挂对密度特征值 (0.5~2.5) | ✓ |
| `simulate_particles.py` | 开口拓扑动力学: 开口=静止拓扑缺陷 (TYPE_B), 非运动粒子 | ✓ |

---

## 四、关键物理结果

| 量 | 测量值 | 物理对应 |
|----|--------|---------|
| v_sigma | 2.0 nodes/sigma (R^2=1.0) | 每气泡 2 个节点, 平凡数 |
| alpha_SPUM | 1/sigma (sigma >= 3) | 补偿边与网络耦合概率 |
| beta (分支比) | 0.9-0.975 (sigma=5) | 大部分气泡节点保持悬挂 |
| sigma=137 | alpha=0.007299 (偏差 0.02%) | 数学恒等式 (cap=3 固定) |
| sigma_natural | 0.5-2.5 (收敛) | 残余悬挂对, 非 137 |

**诚实结论**: alpha=1/sigma 是构造恒等式 (cap=3, penetration=1.0 均固定)。真正的涌现问题是:
1. cap=3 是否从 Σ(6-deg)=12 的自洽性涌现? (A: 当前不涌现, 但猜测系总拓扑电荷 1/4)
2. sigma=137 是否从种子期参数自然推导? (A: 当前未发现推导路径)
3. 三维正二十面体闭合骨架是否自发涌现? (A: Phase 3 已提供验证能力)

---

## 五、加载策略

| 场景 | 加载文件 |
|------|----------|
| 运行 OpenSPUM 代码 | `openSPUM/` 直接 import |
| 理解 SPUM 公理-代码映射 | `.trae/rules/spum-openspum.md` + `SPUM_系统总纲.md` |
| 修改 Phase 1 (关系池/种子期) | `openSPUM/Phase_1/relation_pool.py`, `seed_epoch_engine.py` |
| 修改 Phase 2 (帧演化/级联) | `openSPUM/Phase_2/frame_update_engine.py` |
| 修改 Phase 3 (三维几何) | `openSPUM/Phase_3/geometry_solver.py` |
| 修改 Phase 4 (VSPT 拓扑) | `openSPUM/Phase_4/vspt_growth.py`, `vspt_electron.py`, `vspt_gpu.py` |
| 运行氢全程仿真 | `openSPUM/Phase_4/simulate_hydrogen.py` |
| 查看 VSPT 径向分布 | `openSPUM/Phase_4/vspt_radial.png` |
| 运行/扩展测试 | `openSPUM/tests/test_phase*_all.py` |
| sigma_natural 特征值 | `openSPUM/tests/measure_sigma_natural.py` |
| 粒子行为模拟 | `openSPUM/tests/simulate_particles.py` |
