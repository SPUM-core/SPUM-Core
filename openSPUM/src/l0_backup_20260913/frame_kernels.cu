// ============================================================================
// frame_kernels.cu — L0 层帧内核 K1–K7
//
// 架构基准: openSPUM/docs/L0L1L2_架构设计.md §5
//
// 一帧 = K1..K7 顺序执行；kernel 之间是 L1 同步点。
// 每个 kernel 内部完全并行：线程 i 只读写粒子 i 自己的行，
// 只读其一度/二度邻居的状态 —— 无全局遍历，无上帝视角。
//
// 状态通过 __device__ 全局 g_p / g_q 传递（CuPy RawModule 无法按 ABI
// 传结构体参数）；由 k_init_state / k_init_queues 一次性写入指针表。
//
//   K1 update_volume      degree → radius
//   K2 detect_gaps        邻居对 A 类缝隙扫描 → 创生请求（atomicAdd 入队）
//   -- L1: 创生请求去重（三元组分组 + pos 聚类）+ 预分配新粒子索引 --
//   K3 spawn              写入新粒子 + 3 条边（双方槽位，排他性终审，回滚）
//   K4 connect_existing   二度邻居相切 → 连接请求
//   -- L1: 连接请求去重（二元组键）--
//   K5 commit_edges       写边（双方排他性终审）
//   K5b impenetrability   球面滑动（纯角坐标，只写自己的行）
//   K6 mark_dangling      nb_count < 3 → 标记
//   K7 purge              被标记者失活并清行；邻居清除指向它的槽位（只删一层）
//   K8a check_mutuality   只读扫描对偶性（j 失活 / j 的行不含 i）→ 掩码
//   K8b compact_mask      按掩码独占压缩自己的行 + 尾部槽位清零
//
// 为什么需要 K8：K3 回滚 / K5 失败清理若立即 nb_slot_remove，其
// swap-compress 读取的 nb_count 可能落后于并发 claim 的 atomicAdd，
// 会把并发写入的合法条目挤出计数窗口（互指违例）。故帧内不做跨行
// 清除，所有对偶性修复集中到帧末 K8（K8a 全行只读、K8b 只写己行，
// 均无跨线程写冲突）。
// ============================================================================

// 注：host.py 加载时将 particle_state.cuh 内容前置拼接（nvrtc 对中文
// include 路径不兼容）。独立编译时请改为 #include "particle_state.cuh"。

// ---------------------------------------------------------------------------
// 设备全局状态（host 初始化一次，帧循环内指针不变）
// ---------------------------------------------------------------------------
__device__ ParticlesSoA g_p;
__device__ RequestQueues g_q;

extern "C" __global__ void k_init_state(float3* pos, float* radius, int* nb_count,
                             int* nb_idx, float2* nb_dir, bool* active,
                             int* n_active, int* lock, int capacity)
{
    g_p.pos = pos; g_p.radius = radius; g_p.nb_count = nb_count;
    g_p.nb_idx = nb_idx; g_p.nb_dir = nb_dir; g_p.active = active;
    g_p.n_active = n_active; g_p.lock = lock; g_p.capacity = capacity;
}

extern "C" __global__ void k_init_queues(CreationRequest* cre, int* cre_count, int cre_cap,
                              EdgeRequest* edge, int* edge_count, int edge_cap)
{
    g_q.cre = cre; g_q.cre_count = cre_count; g_q.cre_cap = cre_cap;
    g_q.edge = edge; g_q.edge_count = edge_count; g_q.edge_cap = edge_cap;
}

// ============================================================================
// K1 变化体积：degree → radius
// 通过 degree_radius() 单点取半径律（该律未推导，见 particle_state.cuh）。
// ============================================================================
extern "C" __global__ void k_update_volume()
{
    ParticlesSoA p = g_p;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.capacity || !p.active[i]) return;

    p.radius[i] = degree_radius(p.nb_count[i]);
}

// ============================================================================
// K2 缝隙检测（A 类边缝隙，创生 V⁺ 候选）
//
// 线程 i 枚举邻居对 (j,k)：若 j,k 之间可容一个与 i 等大的新球
// （球面解析解存在），则 jk 大圆两侧各得到一个对偶洞位方向。
// 每侧做排他性初筛，通过则发创生请求。
//
// 洞位解析解（文档 §4.2）：
//   v_new = x·v_j + y·v_k + z·(v_j×v_k)/|v_j×v_k|
//   x,y 由 angdist(v_new,v_j)=α_j+α_new, angdist(v_new,v_k)=α_k+α_new 解出
//   z = ±√(1−|x·v_j+y·v_k|²)  —— z²<0 即无解（jk 间塞不下新球）
//
// 同一缝隙被 i,j,k 三方各报一次（± 两侧），L1 三元组分组 + pos 聚类去重。
// O(deg_i²) 局部操作。B 类面洞（deg=4 诞生）恒不活化（文档 §4.2），不实现。
// ============================================================================
extern "C" __global__ void k_detect_gaps()
{
    ParticlesSoA p = g_p;
    RequestQueues q = g_q;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.capacity || !p.active[i]) return;

    int n = p.nb_count[i];
    if (n < 2) return;

    float r_i = p.radius[i];
    // 新粒子出生度数恒为 3 → 其几何半径 = degree_radius(3)（不是 r_i）。
    // 用出生半径解洞位，才能与下一帧 K1 重置后的实际半径自洽。
    float r_new = degree_radius(SPUM_BIRTH_DEG);

    const int*    my_idx = p.nb_idx + (size_t)i * SPUM_MAX_NB;
    const float2* my_dir = p.nb_dir + (size_t)i * SPUM_MAX_NB;

    // 寄存器快照本行
    int    nb[SPUM_MAX_NB];
    float2 nd[SPUM_MAX_NB];
    for (int s = 0; s < n; ++s) { nb[s] = my_idx[s]; nd[s] = my_dir[s]; }

    for (int u = 0; u < n; ++u) {
        int j = nb[u];
        if (j < 0 || !p.active[j]) continue;
        float3 vj = sph2cart(nd[u].x, nd[u].y);

        for (int v = u + 1; v < n; ++v) {
            int k = nb[v];
            if (k < 0 || !p.active[k]) continue;

            // A 类缝隙 = 面缝隙：j,k 必须互为邻居（三球两两相切才成洞）。
            // 此条件同时保证 K3 能在每个 party 的局部球面坐标系解析求解。
            bool jk_adj = false;
            {
                int nj = p.nb_count[j];
                const int* jrow = p.nb_idx + (size_t)j * SPUM_MAX_NB;
                for (int t = 0; t < nj; ++t)
                    if (jrow[t] == k) { jk_adj = true; break; }
            }
            if (!jk_adj) continue;

            float3 vk = sph2cart(nd[v].x, nd[v].y);

            float a = contact_angle(r_i, p.radius[j], r_new);  // v_new 到 v_j 的目标角距
            float b = contact_angle(r_i, p.radius[k], r_new);  // v_new 到 v_k 的目标角距

            float3 s1, s2;
            if (!solve_tangent_dirs(vj, vk, a, b, s1, s2)) continue;

            for (int sol = 0; sol < 2; ++sol) {
                float3 vn = sol == 0 ? s1 : s2;
                float2 dir_new = cart2sph(vn);

                // 排他性初筛：洞位不得被已占邻居覆盖
                if (!exclusivity_pass(p, i, dir_new, r_new)) continue;

                int slot = atomicAdd(q.cre_count, 1);
                if (slot >= q.cre_cap) { atomicSub(q.cre_count, 1); return; }

                // 键 = 排序三元组
                int key[3] = { i, j, k };
                for (int s1 = 1; s1 < 3; ++s1) {
                    int vv = key[s1]; int s2 = s1 - 1;
                    while (s2 >= 0 && key[s2] > vv) { key[s2+1] = key[s2]; --s2; }
                    key[s2+1] = vv;
                }

                CreationRequest req;
                req.a = key[0]; req.b = key[1]; req.c = key[2];
                float dist = r_i + r_new;
                req.pos = make_float3(p.pos[i].x + dist * vn.x,
                                      p.pos[i].y + dist * vn.y,
                                      p.pos[i].z + dist * vn.z);
                req.radius = r_new;
                q.cre[slot] = req;
            }
        }
    }
}

// ============================================================================
// K3 创生写入（每线程处理一个已去重的创生请求）
//
// 请求携带 L1 预分配的新粒子索引 new_idx。
// 写入：新粒子 pos/radius/active + 与三方 (a,b,c) 的 3 条边。
// 边写入 = 双方槽位互写（atomicCAS 抢槽），写入前各方排他性终审。
// 任一方失败 → 回滚：新粒子失活，已写槽位清除。
// ============================================================================
extern "C" __global__ void k_spawn(const CreationRequest* reqs,
                        const int* new_indices, int n_reqs, int* dbg)
{
    ParticlesSoA p = g_p;
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_reqs) return;

    int ni = new_indices[r];
    if (ni < 0 || ni >= p.capacity) return;

    CreationRequest req = reqs[r];
    int parties[3] = { req.a, req.b, req.c };

    // 初始化新粒子行（槽位已被 L1 从潜在池划出，这里写内容）
    p.pos[ni]    = req.pos;
    p.radius[ni] = req.radius;
    p.nb_count[ni] = 0;
    int*    ni_idx = p.nb_idx + (size_t)ni * SPUM_MAX_NB;
    float2* ni_dir = p.nb_dir + (size_t)ni * SPUM_MAX_NB;
    for (int s = 0; s < SPUM_MAX_NB; ++s) ni_idx[s] = SPUM_EMPTY;

    // 对三方：局部方向解算 + 行锁内排他性终审 + 双方槽位互写。
    //
    // dir_g 不由 pos 差回算（pos 是 L1 投影，多路径重构有漂移），而是在 g
    // 自己的行里，用与另外两个 party 的角距解析求解（两圆交点）。
    //
    // 关键：所有 nb_dir 都是**同一全局笛卡尔帧**下的球坐标（cart2sph/
    // sph2cart 的公共约定）。因此 g 解出的"g→ni"方向可以直接取负得到
    // "ni→g"方向——两行天然互为反向量（antipodal），跨粒子帧严格一致。
    // （此前改用余弦定理在新粒子自己的"规范框架"里构造邻接夹角，等于把
    //  ni 的行旋转到一个任意朝向，破坏了全局帧 → 全图互指偏差达 80°，
    //  是 V 停在 16 的真正根因。）
    int n_written = 0;
    bool ok = true;
    float3 dir_to_ni[3];      // 各 party 眼中 g→ni 的方向（全局帧）

    for (int t = 0; t < 3 && ok; ++t) {
        int g = parties[t];
        int u = parties[(t + 1) % 3], v = parties[(t + 2) % 3];

        row_lock(p, g);

        bool pass = false;
        int  reason = 1;              // 1=party 失活 2=u/v 不在行 3=solve 无解
                                      // 4=排他性失败 5=槽位失败
        float2 dir_g = make_float2(0.0f, 0.0f);

        if (p.active[g]) {
            // 在 g 的行内取另外两个 party 的局部方向（A 类面缝隙 ⇒ 必在）
            int n_g = p.nb_count[g];
            const int*    g_idx = p.nb_idx + (size_t)g * SPUM_MAX_NB;
            const float2* g_dir = p.nb_dir + (size_t)g * SPUM_MAX_NB;
            float2 du = make_float2(-1.0f, 0.0f), dv = make_float2(-1.0f, 0.0f);
            for (int s = 0; s < n_g; ++s) {
                if (g_idx[s] == u) du = g_dir[s];
                else if (g_idx[s] == v) dv = g_dir[s];
            }
            if (du.x >= 0.0f && dv.x >= 0.0f) {
                reason = 3;
                float r_g = p.radius[g];
                float a_u = contact_angle(r_g, p.radius[u], req.radius);
                float a_v = contact_angle(r_g, p.radius[v], req.radius);
                float3 s1, s2;
                if (solve_tangent_dirs(sph2cart(du.x, du.y),
                                       sph2cart(dv.x, dv.y), a_u, a_v,
                                       s1, s2)) {
                    // 镜像选边：用请求携带的 pos 作为参考方向（全局帧一致，
                    // pos 仅用于在两侧对偶洞位中定位本请求所指的那一侧）。
                    float3 ref = make_float3(req.pos.x - p.pos[g].x,
                                             req.pos.y - p.pos[g].y,
                                             req.pos.z - p.pos[g].z);
                    float dot1 = ref.x*s1.x + ref.y*s1.y + ref.z*s1.z;
                    float dot2 = ref.x*s2.x + ref.y*s2.y + ref.z*s2.z;
                    bool s1_sel = (dot1 >= dot2);
                    float3 chosen = s1_sel ? s1 : s2;
                    dir_g = cart2sph(chosen);

                    // 排他性终审 + 占槽（行锁内原子完成）
                    reason = 4;
                    if (exclusivity_pass(p, g, dir_g, req.radius)) {
                        reason = 5;
                        if (nb_slot_claim(p, g, ni, dir_g)) {
                            pass = true; reason = 0;
                            dir_to_ni[t] = chosen;   // 记录全局帧方向
                        }
                    } else if (dbg) {
                        float3 other = s1_sel ? s2 : s1;
                        if (exclusivity_pass(p, g, cart2sph(other), req.radius))
                            atomicAdd(&dbg[18], 1);
                        atomicAdd(&dbg[19], 1);
                    }
                }
            } else {
                reason = 2;
            }
        }

        row_unlock(p, g);
        if (!pass) {
            if (dbg) atomicAdd(&dbg[t * 5 + reason], 1);
            ok = false; break;
        }
        ++n_written;
    }

    if (ok && n_written == 3) {
        // 新粒子行 = 各 party 方向的负向量（全局帧取负 ⇒ ni→g = −(g→ni)）。
        for (int t = 0; t < 3; ++t) {
            float3 d = dir_to_ni[t];
            ni_idx[t] = parties[t];
            ni_dir[t] = cart2sph(make_float3(-d.x, -d.y, -d.z));
        }
        p.nb_count[ni] = 3;              // A 类缝隙创生 = 3 条边同时建立
        p.active[ni] = true;
        atomicAdd(p.n_active, 1);
        if (dbg) atomicAdd(&dbg[16], 1);
    } else {
        // 回滚：不在此清除对方槽位。swap-compress 式的立即清除与并发
        // claim 存在竞态（见文件头 K8 说明）；对方行中指向 ni 的残留
        // 条目是"指向失活粒子的保守占位"，由帧末 K8 统一清扫。
        p.active[ni] = false;
        if (dbg) atomicAdd(&dbg[17], 1);
    }
}

// ============================================================================
// K4 二度邻居相切发现（补充 V⁺）
//
// 线程 i 遍历邻居 j 的邻居 k（二度邻居）：
//   - k ≠ i 且未与 i 连接（k>i 保证每对只报一次）
//   - 体积增长后新进入相切：|pos_i − pos_k| ≈ r_i + r_k
//   - i 侧排他性初筛
// pos 使用帧开始时 L1 重构的坐标（帧内无中间状态）。
// 访问跳度 ≤ 二度邻居（文档 §11-U6 局部性断言的实现边界）。
// ============================================================================
extern "C" __global__ void k_connect_existing()
{
    ParticlesSoA p = g_p;
    RequestQueues q = g_q;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.capacity || !p.active[i]) return;

    int n = p.nb_count[i];
    if (n < 1) return;

    float r_i = p.radius[i];
    float3 pos_i = p.pos[i];
    const int* my_idx = p.nb_idx + (size_t)i * SPUM_MAX_NB;

    for (int s = 0; s < n; ++s) {
        int j = my_idx[s];
        if (j < 0 || !p.active[j]) continue;

        int nj = p.nb_count[j];
        const int* j_idx = p.nb_idx + (size_t)j * SPUM_MAX_NB;

        for (int t = 0; t < nj; ++t) {
            int k = j_idx[t];
            if (k <= i || !p.active[k]) continue;

            bool already = false;
            for (int u = 0; u < n; ++u)
                if (my_idx[u] == k) { already = true; break; }
            if (already) continue;

            float3 dvec = make_float3(p.pos[k].x - pos_i.x,
                                      p.pos[k].y - pos_i.y,
                                      p.pos[k].z - pos_i.z);
            float d = sqrtf(dvec.x*dvec.x + dvec.y*dvec.y + dvec.z*dvec.z);
            float rsum = r_i + p.radius[k];
            if (fabsf(d - rsum) > SPUM_TOL * rsum) continue;

            float2 dir_i = cart2sph(dvec);
            if (!exclusivity_pass(p, i, dir_i, p.radius[k])) continue;

            int slot = atomicAdd(q.edge_count, 1);
            if (slot >= q.edge_cap) { atomicSub(q.edge_count, 1); return; }
            EdgeRequest req; req.i = i; req.j = k;
            q.edge[slot] = req;
        }
    }
}

// ============================================================================
// K5 连接写入（每线程处理一个已去重的连接请求）
//
// 双方排他性终审（帧内 K3 可能已改变邻居布局），通过后双方槽位互写。
// 每侧的"终审+占槽"在行锁内原子完成；同一时刻至多持一把锁，无死锁。
// 第二侧失败不清除第一侧，残留半边由 K8 统一修复。
// 注意：K5 的 dir 只能来自 pos 差（双方无共享局部框架），投影漂移由
// K5b 滑动在后续帧消解——这是 L1 投影与 L0 角坐标的固有张力。
// ============================================================================
extern "C" __global__ void k_commit_edges(const EdgeRequest* reqs, int n_reqs)
{
    ParticlesSoA p = g_p;
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_reqs) return;

    int i = reqs[r].i, j = reqs[r].j;
    if (!p.active[i] || !p.active[j]) return;

    float3 dvec = make_float3(p.pos[j].x - p.pos[i].x,
                              p.pos[j].y - p.pos[i].y,
                              p.pos[j].z - p.pos[i].z);
    float2 dir_i = cart2sph(dvec);
    float2 dir_j = cart2sph(make_float3(-dvec.x, -dvec.y, -dvec.z));

    row_lock(p, i);
    bool ok = exclusivity_pass(p, i, dir_i, p.radius[j])
           && nb_slot_claim(p, i, j, dir_i);
    row_unlock(p, i);
    if (!ok) return;

    row_lock(p, j);
    if (exclusivity_pass(p, j, dir_j, p.radius[i]))
        nb_slot_claim(p, j, i, dir_j);
    row_unlock(p, j);
}

// ============================================================================
// K5b 不可入性球面滑动（逻辑等价旧架构 step3b，纯角坐标）
//
// 线程 i 检查每对邻居 (j,k) 在 i 球面上的角距：
//   角距 < contact_angle(r_i,r_j,r_k) → 沿大圆弧各推 δ = min(SPUM_SLIDE_MAX, 缺口/2)
// 只写自己的 nb_dir 行（无跨线程冲突）。各粒子独立调整自己视角，
// 多路径不一致由 L1 投影层调和（文档 §8）。
// ============================================================================
extern "C" __global__ void k_impenetrability(int max_iter)
{
    ParticlesSoA p = g_p;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.capacity || !p.active[i]) return;

    int n = p.nb_count[i];
    if (n < 2) return;

    float r_i = p.radius[i];
    float2* my_dir = p.nb_dir + (size_t)i * SPUM_MAX_NB;
    const int* my_idx = p.nb_idx + (size_t)i * SPUM_MAX_NB;

    for (int it = 0; it < max_iter; ++it) {
        bool pushed = false;
        for (int a = 0; a < n; ++a) {
            int ja = my_idx[a];
            if (ja < 0) continue;
            for (int b = a + 1; b < n; ++b) {
                int jb = my_idx[b];
                if (jb < 0) continue;
                float dist = angular_distance(my_dir[a], my_dir[b]);
                float req  = contact_angle(r_i, p.radius[ja], p.radius[jb]);
                if (dist < req * (1.0f - SPUM_TOL)) {
                    float delta = fminf(SPUM_SLIDE_MAX, (req - dist) * 0.5f);
                    float3 va = sph2cart(my_dir[a].x, my_dir[a].y);
                    float3 vb = sph2cart(my_dir[b].x, my_dir[b].y);
                    float3 ax = make_float3(
                        va.y*vb.z - va.z*vb.y,
                        va.z*vb.x - va.x*vb.z,
                        va.x*vb.y - va.y*vb.x);
                    float an = sqrtf(ax.x*ax.x + ax.y*ax.y + ax.z*ax.z);
                    if (an < 1e-12f) continue;
                    ax.x /= an; ax.y /= an; ax.z /= an;
                    // 方向约定：轴 n = va×vb，则绕 n 把 va 转 +δ 会移向 vb。
                    // 要"推开"必须 va 转 −δ、vb 转 +δ（此前符号相反，导致
                    // 滑动把邻居对越挤越近，是本轮违例爆发的根因）。
                    float cd = cosf(delta), sd = sinf(delta);
                    float dot_a = ax.x*va.x + ax.y*va.y + ax.z*va.z;
                    float3 va2 = make_float3(
                        va.x*cd - (ax.y*va.z - ax.z*va.y)*sd + ax.x*dot_a*(1-cd),
                        va.y*cd - (ax.z*va.x - ax.x*va.z)*sd + ax.y*dot_a*(1-cd),
                        va.z*cd - (ax.x*va.y - ax.y*va.x)*sd + ax.z*dot_a*(1-cd));
                    float dot_b = ax.x*vb.x + ax.y*vb.y + ax.z*vb.z;
                    float3 vb2 = make_float3(
                        vb.x*cd + (ax.y*vb.z - ax.z*vb.y)*sd + ax.x*dot_b*(1-cd),
                        vb.y*cd + (ax.z*vb.x - ax.x*vb.z)*sd + ax.y*dot_b*(1-cd),
                        vb.z*cd + (ax.x*vb.y - ax.y*vb.x)*sd + ax.z*dot_b*(1-cd));
                    my_dir[a] = cart2sph(va2);
                    my_dir[b] = cart2sph(vb2);
                    pushed = true;
                }
            }
        }
        if (!pushed) break;
    }
}

// ============================================================================
// K6 悬挂标记：nb_count < SPUM_DANGLING_MIN → dangling（纯局部判定）
// 默认阈值 3（闭合三角剖分下限）；种子探测可经 host 侧 #define 降为 2
// （公理下限：deg<2 才逻辑不自洽），以观察 deg-2 边界环能否存活。
// ============================================================================
extern "C" __global__ void k_mark_dangling(bool* dangling)
{
    ParticlesSoA p = g_p;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.capacity) return;
    dangling[i] = (p.active[i] && p.nb_count[i] < SPUM_DANGLING_MIN);
}

// ============================================================================
// K7 删除悬挂（只删一层——不完美定理的实现）
//
// 线程 i：
//   (a) 若自己被标记：清空邻接行，失活
//   (b) 否则扫描自己的槽位，移除指向被标记邻居的条目
// dangling[] 是 K6 的稳定输出（本 kernel 内只读），无跨线程写冲突。
// 禁止级联：本帧新产生的 nb_count=2 由下一帧 K6 发现。
// ============================================================================
extern "C" __global__ void k_purge(const bool* dangling)
{
    ParticlesSoA p = g_p;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.capacity || !p.active[i]) return;

    int*    my_idx = p.nb_idx + (size_t)i * SPUM_MAX_NB;
    float2* my_dir = p.nb_dir + (size_t)i * SPUM_MAX_NB;

    if (dangling[i]) {
        for (int s = 0; s < SPUM_MAX_NB; ++s) my_idx[s] = SPUM_EMPTY;
        p.nb_count[i] = 0;
        p.active[i] = false;
        atomicSub(p.n_active, 1);
        return;
    }

    int n = p.nb_count[i];
    for (int s = n - 1; s >= 0; --s) {
        int j = my_idx[s];
        if (j >= 0 && dangling[j]) {
            int last = n - 1;
            my_idx[s] = my_idx[last];
            my_dir[s] = my_dir[last];
            my_idx[last] = SPUM_EMPTY;
            --n;
        }
    }
    p.nb_count[i] = n;
}

// ============================================================================
// K8a 关系对偶检查（只读，产出逐槽保留/丢弃掩码）
//
// 关系确认的对偶性：i 的行里有 j，当且仅当 j 的行里有 i。
// 帧内并发写边可能留下两类残迹（见文件头 K8 说明）：
//   (a) j 已失活（K3 回滚残留 / K7 删除）；
//   (b) j 活性但 j 的行内不含 i（K5 第二侧失败 / 竞态挤出）。
// 本 kernel 全部行数据只读（掩码是私有输出），无跨线程写冲突。
// 掩码仅对 [0, nb_count[i]) 槽位有效，其余槽位由 K8b 无条件清零。
// ============================================================================
extern "C" __global__ void k_check_mutuality(unsigned char* mask)
{
    ParticlesSoA p = g_p;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.capacity) return;

    unsigned char* my_mask = mask + (size_t)i * SPUM_MAX_NB;
    if (!p.active[i]) return;                // 失活行由 K8b 直接清空

    const int* my_idx = p.nb_idx + (size_t)i * SPUM_MAX_NB;
    int n = p.nb_count[i];

    for (int s = 0; s < n; ++s) {
        int j = my_idx[s];
        bool keep = false;
        if (j >= 0 && p.active[j]) {
            // 对偶确认：j 的行内必须含 i
            const int* j_idx = p.nb_idx + (size_t)j * SPUM_MAX_NB;
            int nj = p.nb_count[j];
            for (int t = 0; t < nj; ++t) {
                if (j_idx[t] == i) { keep = true; break; }
            }
        }
        my_mask[s] = keep ? 1 : 0;
    }
}

// ============================================================================
// K8b 掩码压缩（每线程独占自己的行，只写己行）
//
// 按 K8a 掩码压缩 [0, nb_count[i])，随后把尾部槽位全部置 SPUM_EMPTY
// ——同时回收竞态在计数窗口外留下的幻影占用槽（非 EMPTY 且不可见）。
// 失活行整行清空并归零计数（供 L1 回收槽位到潜在池）。
// ============================================================================
extern "C" __global__ void k_compact_mask(const unsigned char* mask)
{
    ParticlesSoA p = g_p;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p.capacity) return;

    int*    my_idx = p.nb_idx + (size_t)i * SPUM_MAX_NB;
    float2* my_dir = p.nb_dir + (size_t)i * SPUM_MAX_NB;
    const unsigned char* my_mask = mask + (size_t)i * SPUM_MAX_NB;

    int m = 0;
    if (p.active[i]) {
        int n = p.nb_count[i];
        for (int s = 0; s < n; ++s) {
            if (my_mask[s]) {
                my_idx[m] = my_idx[s];
                my_dir[m] = my_dir[s];
                ++m;
            }
        }
    }
    for (int s = m; s < SPUM_MAX_NB; ++s) my_idx[s] = SPUM_EMPTY;
    p.nb_count[i] = m;
}

