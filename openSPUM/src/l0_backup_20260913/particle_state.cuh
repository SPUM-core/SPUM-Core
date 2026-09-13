// ============================================================================
// particle_state.cuh — L0 层粒子状态（SoA）+ 排他性几何原语
//
// 架构基准: openSPUM/docs/L0L1L2_架构设计.md §2–§4
//
// 核心设计:
//   - 每个 CUDA 线程管理一个粒子的局部状态，禁止全局遍历
//   - 邻接关系用固定槽位表存储：nb_idx / nb_dir
//   - 排他性 = 球面角坐标下的不可入性（角距 ≥ 角半径之和）
//   - 代码中不出现 12 / 42 / 50 等度数魔数；饱和度由 r_min + 立体角涌现
// ============================================================================

#pragma once

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// 常量（全部是内存/几何约束，无拓扑魔数）
// ---------------------------------------------------------------------------

// 邻接槽位数：显存分配上界，不是度数逻辑上限。
// 逻辑上限由 §3.4 的立体角链条涌现（等大球 + r_min → 自然饱和）。
// 64 覆盖尺寸悬殊场景，且为 2 的幂（对齐）。
#define SPUM_MAX_NB 64

// 最小可创生粒子半径。创生有最小体积成本：缝隙可容球小于此值则不活化。
// 对应球体积 V_min ≈ (4/3)π·0.415³ ≈ 0.30。
// 与等大球构型联立 → 20 面缝隙 r_max=0.258 < 0.415 → 度 12 自然饱和（涌现）。
#define SPUM_R_MIN 0.415f

// 排他性/相切判定容差（相对值）
#define SPUM_TOL 0.02f

// 球面滑动单步最大角步长（弧度），防止振荡
#define SPUM_SLIDE_MAX 0.2f

// 空槽位标记
#define SPUM_EMPTY -1

// ---------------------------------------------------------------------------
// 粒子状态（Structure of Arrays，全部驻留设备内存）
// ---------------------------------------------------------------------------
struct ParticlesSoA {
    float3* pos;         // [N]  位置（L1 投影坐标，由 nb_dir 重构；帧循环内只读）
    float*  radius;      // [N]  半径（K1 由 degree 更新）
    int*    nb_count;    // [N]  已占邻接槽位数（= degree，单一事实源）
    int*    nb_idx;      // [N×SPUM_MAX_NB] 邻居粒子索引，SPUM_EMPTY=空槽
    float2* nb_dir;      // [N×SPUM_MAX_NB] 邻居在本粒子球面上的方向 (θ,φ)
    bool*   active;      // [N]  活性（false = 潜在池）
    int*    n_active;    // [1]  活性粒子计数（设备侧，atomic 维护）
    int*    lock;        // [N]  行自旋锁（0=自由）：K3/K5 的"终审+占槽"临界区
    int     capacity;    // N（槽位总量）
};

// ---------------------------------------------------------------------------
// 请求队列（K2/K4 产出，L1 去重后由 K3/写边阶段消费）
// ---------------------------------------------------------------------------

// 创生请求：A 类缝隙由三个互切球定义（中心 i + 邻居对 j,k）。
// 三元组排序 (a<b<c) 作为 L1 去重主键——同一缝隙必被三方各自报告；
// 每方报告 ± 两侧对偶洞位，L1 组内按 pos 聚类得到独立缝隙位（文档 §4.3）。
//
// 三个 party 两两相邻（三角形）是**局部可解性**要求，不是相切要求：
// 只有三方互为邻居，K3 中每个 party 才能在自己的行里用两条已知角距
// 解析求出新球方向（两圆交点）。相邻性由 K2 的 jk_adj 过滤保证。
struct CreationRequest {
    int     a, b, c;     // 定义缝隙的三个粒子索引（升序）
    float3  pos;         // 新球位置（请求发起方局部计算）
    float   radius;      // 新球半径（v1 等大创生：= 发起方半径）
};

// 连接请求：两个已存在粒子的 V⁺（二度邻居因体积增长而新相切）。
// (i<j) 作为去重/仲裁键。
struct EdgeRequest {
    int i, j;
};

struct RequestQueues {
    CreationRequest* cre;   // [cap_cre]
    int*             cre_count;  // [1]
    int              cre_cap;

    EdgeRequest*     edge;  // [cap_edge]
    int*             edge_count; // [1]
    int              edge_cap;
};

// ---------------------------------------------------------------------------
// 几何原语（设备内联）
// ---------------------------------------------------------------------------

// (θ,φ) → 单位向量
__device__ __forceinline__ float3 sph2cart(float theta, float phi) {
    float st = sinf(theta);
    return make_float3(st * cosf(phi), st * sinf(phi), cosf(theta));
}

// 单位向量 → (θ,φ)
__device__ __forceinline__ float2 cart2sph(float3 v) {
    float n = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (n < 1e-12f) return make_float2(0.0f, 0.0f);
    float inv = 1.0f / n;
    float theta = acosf(fminf(fmaxf(v.z * inv, -1.0f), 1.0f));
    float phi   = atan2f(v.y, v.x);
    return make_float2(theta, phi);
}

// 球面大圆角距
__device__ __forceinline__ float angular_distance(float2 u, float2 v) {
    float c = sinf(u.x) * sinf(v.x) * cosf(u.y - v.y)
            + cosf(u.x) * cosf(v.x);
    return acosf(fminf(fmaxf(c, -1.0f), 1.0f));
}

// 角半径：与 i（半径 r_self）相切的 j（半径 r_other）在 i 球面上占据的
// 圆顶半角 α = asin(r_other / (r_self + r_other))
__device__ __forceinline__ float angular_radius(float r_self, float r_other) {
    float s = r_other / (r_self + r_other);
    return asinf(fminf(fmaxf(s, -1.0f), 1.0f));
}

// 相邻接触角 —— 排他性判据的统一入口【θ* 对照实验开关】
//
// 求 i 的两个邻居 j,k（各自与 i 相切）互不穿透所需的最小角距。
//   精确解（三角余弦定理，两球心距 ≥ r_j+r_k）：
//       cos θ* = [r_i² + r_i(r_j+r_k) − r_j·r_k] / [(r_i+r_j)(r_i+r_k)]
//   α 近似（默认，SPUM_EXACT_THETA 未定义）：
//       θ ≈ α_j + α_k = asin(r_j/(r_i+r_j)) + asin(r_k/(r_i+r_k))
//
// 等大球时两者恒等（= 60°）。尺寸悬殊时 α_j+α_k **偏大**（保守、充分非必要），
// 实测最大偏高约 6.6°——"饱和上界 12"可能是该保守性的人为产物，故设本开关做对照。
//
// 同一公式也给出"新球 n 与 i,j 同时相切"时 n 相对 j 的方向角，故 K2/K3 的洞位
// 解算复用本函数——保证"接纳判据"与"摆放位置"使用同一套几何（否则接纳用宽松
// 阈值、摆放用保守阈值，新球将不与 j,k 真正相切，留下系统性缝隙）。
__device__ __forceinline__ float contact_angle(float r_i, float r_j, float r_k) {
#ifdef SPUM_EXACT_THETA
    float num = r_i*r_i + r_i*(r_j + r_k) - r_j*r_k;
    float den = (r_i + r_j) * (r_i + r_k);
    return acosf(fminf(fmaxf(num / den, -1.0f), 1.0f));
#else
    return angular_radius(r_i, r_j) + angular_radius(r_i, r_k);
#endif
}

// 度数 → 几何半径 —— 【未推导的自由设计杠杆 + 单点入口】
//
// 本体约束（用户 2026-09-11 修正）：度数与半径**不是**直接的函数对应；
// 目前唯一已知的方向性约束是"度数越大 → 半径越大"（单调递增），
// 具体 r=f(deg) 的形式**尚未推导**。故此处不写死任何公式，只留单点入口。
//
// 当前取值 r≡1.0（单位球）是**中性占位**，不是"已推导的律"：
// 8 帧正四面体种子实测（L0 nb_dir 判据）：
//   半径律                     dev违例（帧8）   最差比值
//   等大 r≡1.0                     0            1.000   ← 全帧零违反
//   r = 1 + 0.01·deg              36            0.956
//   r = cbrt(0.75(1+deg)/π)      313            0.801
//
// 障碍（非本体结论，是实现层的已知困难）：边建立后两端 nb_dir 是**冻结**的
// 局部角坐标；任何非恒定半径律都会改变角半径 α = asin(r_j/(r_i+r_j))，
// 而冻结角位不跟随，于是排他性（角距 ≥ α_j+α_k）被破坏。要让 r 随 deg 变化
// 且零违反，必须配套一个"邻球重排"机制（重解接触图下的角位）——这属于尚未
// 解决的问题。实测 L1 帧末全局重排（host.global_realign）因接触图过约束
// 不收敛，反而制造违例，故暂不作为默认路径。
//
// 换言之：等大球不是"唯一自洽几何"的结论，而是"在重排机制缺失下的可用占位"。
// 设计文档 §3.4 的 12 涌现推导与断言 V2-U3 均以 r≡1 为暂定前提。
__device__ __forceinline__ float degree_radius(int deg) {
#ifdef SPUM_RADIUS_MONO
    // 单调递增占位律（仅用于"邻球重排"实验；r=f(deg) 的真实形式未推导）
    return 1.0f + SPUM_RADIUS_MONO * (float)deg;
#else
    (void)deg;
    return 1.0f;   // 占位：单位球。r=f(deg) 待推导，替换此处即可（需配套重排）
#endif
}

// A 类缝隙创生的出生度数（K3 一次性建立与三方 party 的 3 条边）。
#define SPUM_BIRTH_DEG 3

// 悬挂判定阈值（K6：nb_count < SPUM_DANGLING_MIN → dangling）。
// 公理本身只要求最小度数 ≥ 2（deg < 2 才是逻辑不自洽）；默认值 3 是"闭合三角
// 剖分"这一更强的工程下限（文档 §7）。种子探测需要把它降回 2，以观察 deg-2
// 边界环能否存活，故此处做成可被 host 侧 #define 覆盖的默认值。
#ifndef SPUM_DANGLING_MIN
#define SPUM_DANGLING_MIN 3
#endif

// ---------------------------------------------------------------------------
// 排他性检查（§3.2 排他性定理）
//
// 粒子 i 考虑在方向 dir_new 接纳半径 r_new 的新邻居：
// 对每个已占槽位邻居 j，要求 angular_distance(dir_new, dir_j) ≥ contact_angle(r_i, r_j, r_new)。
// 全部通过 → 该方向未被占据（排他性满足）。
// 纯角度比较，不访问其他粒子的 pos —— O(deg_i) 局部操作。
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool exclusivity_pass(
    const ParticlesSoA p, int i, float2 dir_new, float r_new)
{
    float r_i   = p.radius[i];
    int   n     = p.nb_count[i];
    const int*    slot_idx = p.nb_idx + (size_t)i * SPUM_MAX_NB;
    const float2* slot_dir = p.nb_dir + (size_t)i * SPUM_MAX_NB;

    for (int s = 0; s < n; ++s) {
        int j = slot_idx[s];
        if (j < 0) continue;
        float req  = contact_angle(r_i, p.radius[j], r_new);
        float dist = angular_distance(dir_new, slot_dir[s]);
        if (dist < req * (1.0f - SPUM_TOL))
            return false;   // 方向已被占据：排他性拒绝
    }
    return true;
}

// ---------------------------------------------------------------------------
// 行锁（K3/K5 终审临界区）
//
// 终审+占槽必须原子：两个并发请求若同时通过终审再各自占槽，
// 方向可能互相重叠（排他性形同虚设）。每行一把自旋锁；
// 任何线程同一时刻至多持一把锁（K3 逐 party 申请即放），无死锁。
// atomicCAS/atomicExch 是 relaxed 语义，配对 __threadfence 保证
// 行内数据在持锁期间的读写可见性。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void row_lock(ParticlesSoA p, int i) {
    while (atomicCAS(&p.lock[i], 0, 1) != 0) {}
    __threadfence();
}
__device__ __forceinline__ void row_unlock(ParticlesSoA p, int i) {
    __threadfence();
    atomicExch(&p.lock[i], 0);
}

// ---------------------------------------------------------------------------
// 缝隙几何（§4）
// ---------------------------------------------------------------------------

// 两球面圆交解：球面上求方向使 angdist(·,vu)=a 且 angdist(·,vv)=b。
// 有解时写出两个镜像解 s1/s2（z=0 时两解相同），返回 true。
// 这就是"与两球同时相切的第三球心方向"的解析解。
__device__ __forceinline__ bool solve_tangent_dirs(
    float3 vu, float3 vv, float a, float b, float3& s1, float3& s2)
{
    float cv = fminf(fmaxf(vu.x*vv.x + vu.y*vv.y + vu.z*vv.z, -1.0f), 1.0f);
    float d = acosf(cv);
    if (d > a + b || d < fabsf(a - b)) return false;   // 无解
    float ca = cosf(a), cb = cosf(b);
    float denom = 1.0f - cv * cv;
    if (denom < 1e-12f) return false;                  // vu ∥ vv 退化
    float x = (ca - cv * cb) / denom;
    float y = (cb - cv * ca) / denom;
    float z2 = 1.0f - (x*x + y*y + 2.0f*x*y*cv);
    if (z2 < 0.0f) return false;
    float z = sqrtf(z2);
    float3 nx = make_float3(vu.y*vv.z - vu.z*vv.y,
                            vu.z*vv.x - vu.x*vv.z,
                            vu.x*vv.y - vu.y*vv.x);
    float nn = sqrtf(nx.x*nx.x + nx.y*nx.y + nx.z*nx.z);
    if (nn < 1e-12f) return false;
    nx.x /= nn; nx.y /= nn; nx.z /= nn;
    float3 base = make_float3(x*vu.x + y*vv.x, x*vu.y + y*vv.y, x*vu.z + y*vv.z);
    s1 = make_float3(base.x + z*nx.x, base.y + z*nx.y, base.z + z*nx.z);
    s2 = make_float3(base.x - z*nx.x, base.y - z*nx.y, base.z - z*nx.z);
    return true;
}

// 缝隙面中心方向：三邻居方向的归一化和（近等边球面三角形的内切/外接
// 圆心近似）。返回方向与到三顶点的最小角距 θ_min。
__device__ __forceinline__ float2 gap_center_dir(
    float2 dj, float2 dk, float2 dl, float& theta_min)
{
    float3 vj = sph2cart(dj.x, dj.y);
    float3 vk = sph2cart(dk.x, dk.y);
    float3 vl = sph2cart(dl.x, dl.y);
    float3 c = make_float3(vj.x + vk.x + vl.x,
                           vj.y + vk.y + vl.y,
                           vj.z + vk.z + vl.z);
    float2 dc = cart2sph(c);
    float t1 = angular_distance(dc, dj);
    float t2 = angular_distance(dc, dk);
    float t3 = angular_distance(dc, dl);
    theta_min = fminf(t1, fminf(t2, t3));
    return dc;
}

// 缝隙可容新球半径（笛卡尔精确解，文档 §3.4）
//   R  = 中心粒子半径, rn = 邻居半径（等大近似取三邻居均值）,
//   θ  = 面中心到邻居的角距
//   r_new = R·(R+rn)·(1−cosθ) / ( (R+rn)·cosθ − R + rn )
// 返回 ≤ 0 表示无解（几何上缝隙不可容）。
__device__ __forceinline__ float gap_spawn_radius(float R, float rn, float theta) {
    float B = R + rn;
    float c = cosf(theta);
    float denom = B * c - R + rn;
    if (denom < 1e-8f) return -1.0f;
    return R * B * (1.0f - c) / denom;
}

// ---------------------------------------------------------------------------
// 邻接槽位操作（供 K3/K4 使用）
//
// 注意：不提供"立即移除"原语。跨行清除与并发 claim 存在竞态，
// 所有行内条目的移除统一由帧末 K8a/K8b 完成（见 frame_kernels.cu 文件头）。
// ---------------------------------------------------------------------------

// 在粒子 i 的邻接行抢占一个空槽，写入邻居 j 及其方向。
// 多线程可能同时向 i 的行写不同邻居 → atomicCAS 抢槽。
// 返回 true = 写入成功。
__device__ __forceinline__ bool nb_slot_claim(
    ParticlesSoA p, int i, int j, float2 dir)
{
    int*    slot_idx = p.nb_idx + (size_t)i * SPUM_MAX_NB;
    float2* slot_dir = p.nb_dir + (size_t)i * SPUM_MAX_NB;
    // 已存在检查（幂等）
    int n = p.nb_count[i];
    for (int s = 0; s < n; ++s)
        if (slot_idx[s] == j) return true;
    // 抢空槽
    for (int s = 0; s < SPUM_MAX_NB; ++s) {
        if (atomicCAS(&slot_idx[s], SPUM_EMPTY, j) == SPUM_EMPTY) {
            slot_dir[s] = dir;
            atomicAdd(&p.nb_count[i], 1);
            return true;
        }
    }
    return false;  // 槽位满（内存上界，非逻辑上限）
}
