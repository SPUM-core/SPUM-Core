# CUDA 空间粒子引擎 — 每线程 = 一个 ⟨P, ε⟩ 节点

> **核心命题**：一个 CUDA 线程就是一个空间粒子。
> 线程不"计算"粒子——线程**是**粒子。它读取自己的邻居数，计算自己的体量，感知自己周围的缝隙。

---

## 一、哲学立场：GPU 线程 = 空间粒子本体

```
旧视角                        新视角（您纠正的）
───────────────────────────── ─────────────────────────────
GPU 加速已有的 CPU 算法        GPU 线程本身是 ⟨P, ε⟩ 节点
每个节点是结构体/对象           每个线程是一个空间粒子
度数由中央注册表维护            度数 = 线程自己 count 邻居
级联消解是算法步骤             没有级联——就一帧，干净结束
湮灭=删除边+补偿创建           删除=回归潜在状态，做下一帧的创生元
随机数种子驱动                  球体密堆几何=唯一确定性来源
```

您的纠正的深刻之处：**CUDA 的 SIMT 模型不是"加速器"——它是 SPUM 本体论的物理实现**。

- 每个线程有自己的寄存器（= 空间体量）
- 每个线程读自己的邻居（= 度数）
- 线程之间通过 shared memory 通信（= 边）
- 线程可以"休眠"（= 回到潜在状态）
- 没有线程管理中央协调器——这就是 ⟨P, ε⟩

---

## 二、帧结构（与总纲完全一致）

```
一帧 = 5 步，无级联，无湮灭-补偿

┌─────────────────────────────────────────────────────────────┐
│  第 1 步: 创生                                              │
│  gap_detect_kernel → 在球体密堆缝隙中确定新节点位置         │
│  缝隙来自上一帧的体积变化 + 潜在池中的可用节点              │
├─────────────────────────────────────────────────────────────┤
│  第 2 步: 连接                                              │
│  connect_kernel → 新节点接触已有节点 → 边显化                │
│  相切即连接——不需要配对算法                                  │
├─────────────────────────────────────────────────────────────┤
│  第 3 步: 变化体积                                          │
│  volume_reconfigure_kernel → 每个线程重读 degree → 计算半径 │
│  radius = degree × κ (最小差异尺度)                         │
│  体积变化 = 半径增量/减量                                    │
├─────────────────────────────────────────────────────────────┤
│  第 4 步: 判断悬挂边                                        │
│  dangling_check_kernel → degree < 2 的节点标记为悬挂         │
├─────────────────────────────────────────────────────────────┤
│  第 5 步: 删除悬挂边                                        │
│  dangling_purge_kernel → 悬挂节点失去坐标，回归潜在池        │
│  它的边被删除，但它作为"创生元"进入下一帧                    │
└─────────────────────────────────────────────────────────────┘
```

### 与总纲对比

总纲（§4）：
> 一个完整帧 = 创生 → 连接 → 变化体积 → 判断悬挂边 → 删除悬挂边

本实现：**完全一致，顺序完全一致，无额外步骤**。

之前的代码（Phase 2）的错误：
- 在"删除悬挂边"之后添加了"补偿创建"——**多余**
- 在"删除悬挂边"之后添加了"级联迭代"——**多余**
- 将"创生"替换为随机候选 + 配额——**错误**

---

## 三、每线程 = 一个空间粒子

### 3.1 线程布局

```cuda
// ─── 每个 CUDA 线程就是一个空间粒子 ───
// blockDim.x = 256, gridDim.x 取决于总粒子数

__global__ void particle_frame_kernel() {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;  // 粒子 ID

    // ========== 线程局部状态（寄存器） ==========
    uint32_t degree     = read_neighbor_count(pid);    // 我连接了几个邻居？
    float    radius     = degree * KAPPA;               // 我的空间体量
    float3   my_pos     = read_position(pid);           // 我的位置
    float    gap_radius = compute_gap(my_pos, radius);  // 我周围的最大缝隙半径
    uint32_t latent     = is_latent(pid);               // 我在潜在池中吗？

    // ========== 第 1 步：创生（从缝隙） ==========
    if (latent && gap_radius > MIN_GAP) {
        // 我是潜在粒子，且周围有足够的缝隙 → 重生
        // 缝隙位置 = 我周围空洞的几何中心
        float3 new_pos = find_gap_center(my_pos, radius, gap_radius);
        activate_particle(pid, new_pos);
        latent = 0;
        // 重置度数：刚活化的粒子从裸开始
        degree = 0;
    }

    // ========== 第 2 步：连接 ==========
    // 检查我周围是否有其他粒子与我相切
    // 切距 = radius + other.radius
    // 实际距离 = distance(my_pos, other_pos)
    // if |切距 - 实际距离| < ε → 它们相切 → 应有边
    uint32_t neighbor_count = 0;
    for (int other = warp_start; other < warp_end; other += warpSize) {
        if (other == pid || is_latent(other)) continue;
        float3 other_pos = read_position(other);
        float  other_r   = read_radius(other);
        float  dist      = distance(my_pos, other_pos);
        float  tangent   = radius + other_r;
        if (fabs(dist - tangent) < GEOMETRIC_TOLERANCE) {
            // 相切！→ 建立或确认边
            ensure_edge(pid, other);
            neighbor_count++;
        }
    }

    // ========== 第 3 步：变化体积 ==========
    // 我的度数变了 → 我的半径变了
    // 半径变化导致我的边界移动 → 产生或消除缝隙
    degree = neighbor_count;
    radius = degree * KAPPA;
    write_radius(pid, radius);  // 广播给其他线程

    // ========== 第 4 步：判断悬挂 ==========
    int is_dangling = (degree < 2) ? 1 : 0;

    // ========== 第 5 步：删除悬挂 ==========
    if (is_dangling) {
        delete_all_edges(pid);        // 清除我的所有边
        set_latent(pid);              // → 回归潜在状态
        // 我不再占据空间坐标
        // 但我不消失——我进入潜在池，作为下一帧的创生元
    }
}
```

### 3.2 线程如何"读取邻居"

**没有中央邻接表。每个线程通过 shared memory 的 warp-level 协作来自知邻居数。**

```cuda
// warp 级邻居计数
// 一个 warp = 32 线程 = 32 个空间粒子
// 检查 warp 内两两相切

__shared__ float3 s_pos[256];    // block 内所有粒子的位置
__shared__ float  s_rad[256];    // block 内所有粒子的半径
__shared__ uint32_t s_deg[256];  // block 内所有粒子的度数

// 每个线程写自己的状态到 shared memory
s_pos[threadIdx.x] = my_pos;
s_rad[threadIdx.x] = radius;
__syncthreads();

// 然后 warp 内协作扫描
// 每个线程只检查 warp 中的其他线程
uint32_t local_deg = 0;
uint32_t lane = threadIdx.x & 0x1f;  // warp 内索引
uint32_t warp_id = threadIdx.x >> 5; // warp 编号

for (int i = 0; i < 32; i++) {
    if (i == lane) continue;
    int other_idx = (warp_id << 5) + i;  // 同 warp 内的其他线程
    if (other_idx >= blockDim.x) break;

    float3 op = s_pos[other_idx];
    float  or = s_rad[other_idx];
    float  d  = distance(my_pos, op);

    if (fabs(d - (radius + or)) < GEOMETRIC_TOLERANCE) {
        local_deg++;
    }
}
// warp 级归约 → global memory 写入边
```

**关键**：这不需要全局的邻接表字典。相切关系是**几何的直接结果**，不是数据结构的维护产物。

---

## 四、缝隙(创生元)检测 — 完全确定性几何

### 4.1 缝隙的拓扑定义

三个两两相切的球体中间必然存在一个空隙。这个空隙的大小由三个球体的半径决定。

```
已知:
    球 A (r_a, pos_a)
    球 B (r_b, pos_b)
    球 C (r_c, pos_c)
    且 d_ab = r_a + r_b, d_bc = r_b + r_c, d_ca = r_c + r_a (相切)

求: 三个球体的共切球(与三者都相切)的半径 r_gap
```

这是 **Soddy 圆定理** 的 3D 推广——笛卡尔定理：

$$(k_1 + k_2 + k_3 + k_4)^2 = 2(k_1^2 + k_2^2 + k_3^2 + k_4^2)$$

其中 $k_i = 1/r_i$ 为曲率。对三个已知球体，解 $k_4$ 得新球的曲率：

$$k_4 = k_1 + k_2 + k_3 \pm 2\sqrt{k_1k_2 + k_2k_3 + k_3k_1}$$

**正号**对应包覆外球（大球包裹三者），**负号**对应内切球（三者之间的缝隙）。我们取负号。

### 4.2 GPU 上的缝隙检测

```cuda
__global__ void gap_detect_kernel() {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (is_latent(pid)) return;  // 潜在粒子不参与缝隙检测

    float3 my_pos = read_position(pid);
    float  my_r   = read_radius(pid);

    // 扫描本地邻居，找三体组
    // 每三体组计算一个缝隙球
    __shared__ float3 s_pos[256];
    __shared__ float  s_rad[256];
    s_pos[threadIdx.x] = my_pos;
    s_rad[threadIdx.x] = my_r;
    __syncthreads();

    int gap_count = 0;
    for (int j = 0; j < blockDim.x && gap_count < MAX_GAPS; j++) {
        float3 pj = s_pos[j];
        float  rj = s_rad[j];
        if (j == threadIdx.x || is_latent_block(j)) continue;
        if (fabs(distance(my_pos, pj) - (my_r + rj)) >= TOLERANCE) continue;

        for (int k = j + 1; k < blockDim.x && gap_count < MAX_GAPS; k++) {
            float3 pk = s_pos[k];
            float  rk = s_rad[k];
            if (k == threadIdx.x || is_latent_block(k)) continue;
            if (fabs(distance(pj, pk) - (rj + rk)) >= TOLERANCE) continue;
            if (fabs(distance(my_pos, pk) - (my_r + rk)) >= TOLERANCE) continue;

            // 三体两两相切 → 它们中间有缝隙
            float k1 = 1.0f / my_r;
            float k2 = 1.0f / rj;
            float k3 = 1.0f / rk;

            // 笛卡尔定理: 缝隙球的曲率
            float sum_kk = k1*k2 + k2*k3 + k3*k1;
            float k_gap = k1 + k2 + k3 - 2.0f * sqrtf(sum_kk);

            if (k_gap > 0) {
                // 缝隙球可用 → 记录缝隙位置
                float3 gap_pos = compute_gap_center(my_pos, pj, pk, my_r, rj, rk, k_gap);
                write_gap(gap_count++, pid, gap_pos, 1.0f / k_gap);
            }
        }
    }

    if (gap_count > 0) {
        atomicAdd(g_total_gaps, gap_count);
    }
}
```

### 4.3 缝隙大小的决定因素

**大球产生大缝隙，小球产生小缝隙。**

```
三个半径均为 r 的等大球体，其内切缝隙球半径为:
    k_gap = 3/r - 2√(3/r²) = (3 - 2√3)/r ≈ -0.464/r
    → r_gap = |1/k_gap| ≈ 2.155r

即: 等大球间的缝隙 ≈ 2.155 × 球半径
```

**推论**：度数越高的节点（半径越大），其周围的缝隙越大 → 下一帧在这个缝隙中诞生的节点体积也越大。这是"体积膨胀→缝隙扩大→更多生长"的正反馈，也是为何晶子（deg=50）会成为超稳定节点。

---

## 五、潜在池（Latent Pool）

### 5.1 定义

```cuda
// 潜在池 = 全局 memory 中的环形缓冲区
// 存放"悬挂→被删除"的粒子

struct LatentPool {
    uint32_t count;              // 当前潜在粒子数
    uint32_t slots[MAX_PARTICLES]; // 粒子 ID（索引到全局节点数组）
    // 潜在粒子的状态:
    //   degree = 0, 无边, 无坐标
    //   但保留了 uid（唯一标识不消失）
    //   保留了"创生权重"（基于其最后一帧的度数）
};

// 帧 5 步完成后的潜在池变化:
// Step 5（删除悬挂） → 池增加 dangling 粒子
// Step 1（创生）     → 池减少被活化的粒子
// 池的净变化 = dangling_in - activated_out
```

### 5.2 潜在粒子如何成为创生元

1. 粒子在帧 N 被删除（degree < 2 → 回归潜在）
2. 它在潜在池中保留其历史和 uid（标识不消失）
3. 帧 N+1、Step 1 中，gap_detect_kernel 扫描：
   - 检测到缝隙 → 需要一个新粒子填充
   - 优先从潜在池取用（而不是创建全新粒子）
   - 取用时，潜在粒子的 uid 被重新激活 → 保持标识连续性
4. 如果潜在池为空，才创建全新粒子（uid = 新分化）

**这实现了"边数守恒"的拓扑精神——不是通过立即补偿，而是通过物质循环：**
```
活性粒子 → 悬挂 → 回到潜在 → 缝隙活化 → 活性粒子
```

---

## 六、SPUM 无随机 — 确定性保证

当前代码中所有 `random.Random` 的使用都被移除：

| 位置 | 原用途 | 替换方案 |
|------|--------|---------|
| `self_evolution.py` | Thomson 模拟退火初态 | 几何确定性初态（球面均匀分布不用随机——用斐波那契球体算法） |
| `geometry_solver.py` | 随机初态位置 | 同上，确定性球面分布 |
| `vspt_growth.py` | VSPT 分支种子 | 三角网格的确定性方向选择 |
| `vspt_electron.py` | 子节点存活随机 | 纯 3-分支几何生长 |

**斐波那契球面算法**（确定性球面均匀分布）：
```cuda
// 替代 random 初态
float phi = (1.0f + sqrtf(5.0f)) / 2.0f;  // 黄金比例

float3 fibonacci_sphere(int index, int total) {
    float y = 1.0f - (2.0f * index + 1.0f) / total;
    float r = sqrtf(1.0f - y * y);
    float theta = 2.0f * M_PI / phi * index;
    return (float3){r * cosf(theta), y, r * sinf(theta)};
}
```

---

## 七、GPU 全局内存布局

```cuda
// ─── 设备内存 ───
// SoA 布局: 每个字段连续存放

// 活性粒子数组 (最大 MAX_PARTICLES)
float*   d_pos_x;          // [MAX_PARTICLES] 位置 x
float*   d_pos_y;          // [MAX_PARTICLES] 位置 y
float*   d_pos_z;          // [MAX_PARTICLES] 位置 z
float*   d_radius;         // [MAX_PARTICLES] 半径 = degree × κ
uint32_t d_degree[MAX_PARTICLES];         // 度数
uint32_t d_is_active[MAX_PARTICLES];      // 0=潜在, 1=活性
char     d_uid_str[MAX_PARTICLES][16];    // UID 字符串

// 潜在池
uint32_t d_latent_pool[MAX_PARTICLES];    // 池中粒子 ID
uint32_t d_latent_count;                  // 池大小

// 帧快照 (每帧写入环形缓冲区)
struct FrameSnapshot {
    uint32_t active_count;                  // 当前活性粒子数
    uint32_t latent_count;                  // 当前潜在粒子数
    uint32_t degree_hist[51];               // 度分布
    uint32_t total_gaps;                    // 本帧检测到的缝隙数
    uint32_t newly_activated;               // 本帧从潜在活化的粒子数
    uint32_t newly_dangling;                // 本帧新悬挂的粒子数
    uint32_t crystallite_count;             // degree ≥ 50 的粒子数
};
```

### 7.1 为何用 SoA 而非 AoS

```
SoA (选择):                      AoS (放弃):
d_pos_x[0..255] 连续 → 合并访问   struct P { x,y,z,deg,... } → 跨步
d_pos_y[0..255] 连续 → 合并访问   一次读 pos_x 带上 7 个无关字段
d_radius[0..255] 连续 → 合并访问   cache line 利用率低
```

每个 kernel 通常只读写一两个字段。例如 `dangling_check_kernel` 只需要 `d_degree`——SoA 保证一次 coalesced memory read 读到 128 字节全部有用。

---

## 八、完整帧的 CUDA kernel 流

```cuda
// ===== 一帧 = 5 个 kernel =====

// Kernel 1: 创生
gap_detect_kernel<<<grid, block>>>(
    d_pos_x, d_pos_y, d_pos_z, d_radius, d_is_active,
    d_latent_pool, &d_latent_count,
    d_gap_positions, &d_total_gaps
);

// Kernel 2: 连接
// 每个活性线程检查 warp 内邻居的相切关系 → 更新 degree
connect_kernel<<<grid, block>>>(
    d_pos_x, d_pos_y, d_pos_z, d_radius,
    d_degree, d_is_active
);

// Kernel 3: 变化体积
// degree → radius 映射 (线性)
volume_kernel<<<grid, block>>>(
    d_degree, d_radius, d_is_active
);

// Kernel 4: 判断悬挂
dangling_check_kernel<<<grid, block>>>(
    d_degree, d_is_dangling, d_is_active
);

// Kernel 5: 删除悬挂
// 悬挂 → 回归潜在池
dangling_purge_kernel<<<grid, block>>>(
    d_is_dangling, d_is_active,
    d_latent_pool, &d_latent_count,
    d_pos_x, d_pos_y, d_pos_z,
    d_degree  // 清零
);

// 快照 (规约 kernel)
snapshot_kernel<<<grid, block>>>(
    d_degree, d_is_active, d_latent_count,
    d_total_gaps, d_frame_snapshots, frame_idx
);

// ← 这里的同步点就是 GPU→CPU 传输点
// 只传输 FrameSnapshot (约 200 bytes)
cudaMemcpy(&host_snapshot, d_frame_snapshots[frame_idx],
           sizeof(FrameSnapshot), cudaMemcpyDeviceToHost);

// CPU 元数据解读
metadata = decode_snapshot(host_snapshot);
log.info(f"帧 {frame_idx}: 活性={metadata.active_count}, "
         f"潜在={metadata.latent_count}, "
         f"晶子={metadata.crystallite_count}");
```

---

## 九、加速比估算（vs 当前 CPU）

| 操作 | 当前 CPU | GPU 实现 | 加速比 |
|------|---------|---------|-------|
| 一帧完整演化 (10⁵ 节点) | ~50ms (Phase 1 帧 + 悬挂扫描) | 5 kernel × 0.05ms = 0.25ms | **~200x** |
| 缝隙检测 (每帧) | N/A (当前无此操作) | 1 kernel × 0.1ms | **新功能** |
| 度分布统计 | O(V) Python loop | warp-level 归约, ~0.005ms | **~100x** |
| 潜在池管理 | N/A | O(1) 原子操作 | **新功能** |
| GPU→CPU 传输 | — | 200 bytes × PCIe ≈ 0.001ms | **可忽略** |

**关键优势**：GPU 帧时间 (0.25ms) ≪ CPU 帧时间 (50ms) → 可以在 GPU 跑完 **200 帧**的时间内 CPU 跑 1 帧。这就让相空间采样成为可能——GPU 产生大量帧演化轨迹，CPU 选择性解读。

---

## 十、与总纲的完全一致性检查

| 总纲 § | 陈述 | 本实现 |
|--------|------|--------|
| §1 | 关系第一性，节点从关系中诞生 | 线程从相切关系中获得度数 |
| §2 | 度数决定体量 | radius = degree × κ |
| §2 | 唯一标识 | uid 在线程生命周期不消失，即使回潜在池 |
| §4 | 一帧 = 5 步 | 5 kernel，步骤一致、顺序一致 |
| §5 | 边数守恒 | 通过潜在池循环实现，非立即补偿 |
| §8 | 不完美定理 | dangling 必然存在，每帧都有 |
| §9 | Σ(6-deg)=12 | 快照校验 |
| §18 | dv/dt ≤ const | 每帧度变化有限 |

> 您之前的代码（Phase 2 的级联消解）违背了 §4——在一帧中插入了多余操作。
> 本设计严格遵循 §4 的 5 步帧结构。

---

## 十一、与现有代码的集成路径

### 阶段 1: 最小 GPU 帧引擎

```python
# Phase_0/gpu_engine.py
class SPUMCUDAParticleEngine:
    def __init__(self, max_particles=65536, seed_geometry="fibonacci"):
        self.max_particles = max_particles
        # 分配 GPU 内存 (SoA layout)
        # 初始化粒子位置 (斐波那契球面, 无随机)

    def run_frame(self) -> FrameSnapshot:
        """一帧 = 5 kernel launch → 快照传输到 CPU"""
        pass

    def get_snapshot(self) -> FrameSnapshot:
        pass

    # 与 Phase_2 的 FrameUpdateEngine 接口兼容:
    @property
    def node_registry(self):
        """返回模拟的 NodeRegistry 对象 (CPU 端解读)"""
        pass
```

### 阶段 2: CPU 元数据层

```python
# 利用现有 FrameLog 结构:
from Phase_2.frame_update_engine import FrameLog

class CUDAMetadataDecoder:
    def decode(self, raw: FrameSnapshot) -> FrameLog:
        return FrameLog(
            frame_number=raw.frame_idx,
            node_count=raw.active_count,
            edge_count=sum(raw.degree_hist) // 2,
            spum_invariant=6 * raw.active_count - sum(raw.degree_hist),
            dangling_before=raw.newly_dangling,
            crystallite_count=raw.crystallite_count,
            # cascade_iters = 0 (无级联!)
            # edges_annihilated = 0 (无湮灭!)
        )
```

### 阶段 3: 现有 Phase 废弃

| 当前模块 | 状态 | 说明 |
|---------|------|------|
| `Phase_1/seed_epoch_engine.py` | **废弃** | 配额增长被缝隙生长替代 |
| `Phase_1/relation_pool.py` | **废弃** | 邻接表被 GPU 几何邻居计算替代 |
| `Phase_1/node_registry.py` | **保留** | 仅用作 CPU 端解读的模拟接口 |
| `Phase_2/frame_update_engine.py` | **废弃** | 级联消解被 GPU 帧的 5 步替代 |
| `Phase_3/geometry_solver.py` | **保留** | 正二十面体组装部分可复用 |
| `Phase_4/*` | **保留** | 上层解读不受影响 |

> **文件版本**: v2 (2026-06-19)
> **前版本**: `CUDA_并行架构设计.md` (v1, 有级联错误, 已废弃)
