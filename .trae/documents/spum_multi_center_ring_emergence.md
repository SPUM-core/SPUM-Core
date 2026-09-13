# 多中心分布式初态：从帧内创生涌现 12 晶子闭环

## Context

SPUM 的"12 晶子正二十面体闭环"是理论推导**输出**（[gpu_engine.py L80-81](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/Phase_0/gpu_engine.py#L80-L82)"不作为输入种子构型"）。但当前引擎只有单中心 star 初态：创生围绕唯一中心向心填充，晶子（deg≥42）全部聚拢成一个超密堆大团簇（195 边互连），**永不形成恰好 12 晶子闭环**。

此前我用 `ForceDirectedRelaxation` 独立随机松弛撞出正二十面体——这是把推导结果当输入，是第二代物理做法，用户否决。用户要求**晶子密堆从帧内创生**。

已完成代码检查确认：
- **创生 [step1_create](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/Phase_0/frame_kernels.py#L372-L424) 是全粒子通用**（空间哈希三体缝隙，无中心依赖）；只有 [step1b_gap_fill](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/Phase_0/frame_kernels.py#L548-L680) 和 [step2_connect](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/Phase_0/frame_kernels.py#L700-L826) 的 `star_mode` 依赖单一中心 `_find_center`。
- [step2_connect](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/Phase_0/frame_kernels.py#L773-L826) 的**非 star CPU 分支不存在**——`star_mode=False` 且 `pregrowth_knn=0` 时直接返回 0，唯一连接在 star 分支。多中心必须新建按簇连接逻辑。
- 检测器 [detect_12_crystallite_ring](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/universe/emergence.py#L95-L185) 是**全局判定**：`crys_idx.size<12` 直接跳。多簇各自闭环时全局混合 → 永不 formed。必须改为**连通分量级**检测。

本计划：让引擎支持多中心分布式初态，晶子分散成多个独立簇，每簇局部演化，争取**帧内涌现**恰好 12 晶子闭环。

## 方案（多中心分布式初态）

### 1. 多中心初态几何（[gpu_engine.py](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/Phase_0/gpu_engine.py)）

- EngineConfig 增 `n_centers:int`（建议 6~10，**≠12 防作弊**）、`surface_per_center:int`（M，建议 12~20）。
- 布点：C 个中心放单位球面 `_fibonacci_sphere(C)`（非正二十面体，防作弊边界）× 簇壳半径 R。每中心用 `_star_surface(M)` 缩放挂 M 个内部 coda。
- 簇间距 > 2×(簇中心 r + 表面 r)×安全系数，保证跨簇不相切、不被全局空间哈希创生染指。
- UID：`cent_{I:04d}`、`surf_{I}_{j:04d}`（I=簇号）。每中心 deg=M、coda deg=1，`add_connection()` 立最小度≥2。
- 新增 `_init_multi_center()`；`_initialize_seeds()` 增 `multi_center` 分支。

### 2. 创生泛化（[frame_kernels.py](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/Phase_0/frame_kernels.py)）

- 最小改动优先：多中心下 `step1_create` 已是通用三体缝隙创生（无中心依赖），**提高其 `max_checks`**（如 1e6）以吞掉本会跳过转交 `step1b` 的载荷。`step1b` 仅在 `step1_create` 返回 0 才被调。
- 备选：泛化 `_find_center` 为多中心列表，`step1b` 按"最近中心管辖"逐簇 Apollonian。优先级低于前者，本次优先做 `max_checks`。

### 3. 连接泛化（[frame_kernels.py](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/Phase_0/frame_kernels.py#L700-L826)）

- `step2_connect` 增 multi_center 分支：为每个活跃粒子按"最近 `cent_`"赋 cluster_id，逐簇做类似 star 分支的切向连接（capped deg≤42、每帧 k_max=6），仅连同簇成员。
- `run_frame` 的 `star_mode` 判定扩展为 `seed_geometry in {"star","multi_center"}`（[gpu_engine.py L166](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/Phase_0/gpu_engine.py#L166)）。

### 4. 观测接入连通分量级（[emergence.py](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/universe/emergence.py#L95-L185) / [simulator.py](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/universe/simulator.py#L132-L145)）

- 新增 `_crystallite_components()`：从晶子诱导子图拆解连通分量；对每个分量做 12 节点/30 边/度5/Σ=12 判定。任一满足即该簇涌现闭环，返回 `{ring_formed, frame, cluster_id, component_uids}`。
- `simulator._update_milestones` 增 `ring_emergence_events` 列表，记录 `(frame, cluster_id)`；`UniverseReport.summary_lines` 加一行"多簇涌现事件"。

### 5. 诚实边界（写入代码注释，防伪加载）

纯多中心仍用**无差别缝隙创生**——每簇本质是缩小的单中心，原"过饱和聚成 195 边团簇"会在每簇局部重演，命中"恰好 12"是**时机窗口**而非保证。诚实表述为"提升帧内涌现窗口/增加抽样看涌现"。若不足，备选是"局部闭合创生抑制"（对达 12 项链的闭簇禁止其内部创生），**本次仅列为风险预案，不纳入**。

## 决定：删除聚合通道（2026-09-11）

用户否决此前用 `ForceDirectedRelaxation` 做独立随机松弛撞出正二十面体的做法（把推导结果当输入、绕开帧演化）。**晶子密堆必须从帧内创生**。

据此采取"仅删聚合通道 + 诚实报告"：
- 删除 `universe/ring_emergence.py`、`universe/demo_ring.py` 与 `simulator.run_aggregation_phase()`。
- 报告不再声明"闭环聚合通道 7/8 涌现正二十面体"。
- 实测（多中心 8×12，25 帧，纯帧内）：晶子涌现 174，但 `12晶子闭环 = None`（Σ(6−deg) 漂至 −8794，缝隙创生过饱和越界）。报告如实呈现该负结果。
- 12 晶子闭环的调试验证改由合成构型测试 U4 承担（`detect_12_crystallite_ring` 的等价性验证），帧内真实闭环仍为未达成目标。

## 验证

- [demo_universe.py](file:///c:/Users/macotai/Desktop/工作/spum-core/openSPUM/universe/demo_universe.py) 增参数 `--seed_geometry multi_center --n_centers 8 --surface_per_center 12`，跑 30-60 帧。
- 检查 `ring_emergence_events` 是否出现 `(frame, cluster_id)`。
- 对照每簇晶子数直方图确认"某簇恰达 12"的帧。
- 打印跨簇粒子数确认隔离未穿透。
- 回归：`tests/test_universe.py` 既有 U1-U6 仍通过（star/sequential 路径不动）。

## 涉及文件

- `openSPUM/Phase_0/gpu_engine.py` — EngineConfig 字段、`_init_multi_center`、`_initialize_seeds` 分支、`run_frame` star_mode 判定
- `openSPUM/Phase_0/frame_kernels.py` — `step1_create` max_checks、`step2_connect` 按簇连接分支、`_fibonacci_sphere`
- `openSPUM/universe/emergence.py` — `_crystallite_components` 连通分量级闭环检测
- `openSPUM/universe/simulator.py` — 里程碑 `ring_emergence_events`、summary 行
- `openSPUM/universe/demo_universe.py` — CLI 参数