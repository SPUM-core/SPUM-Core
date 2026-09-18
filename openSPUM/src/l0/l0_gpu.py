# -*- coding: utf-8 -*-
"""l0_gpu.py — L0 决策层的 GPU 并行实现（阶段三）。

架构基准: docs/L0L1L2_Architecture.md §5（GPU 映射）
对偶实现: l0_core.py（CPU 内核，本文件的正确性基准）

搬运范围
--------
  第一增量（决策段，唯一完全可并行的部分）：
      propose（候选生成）→ priority（优先级键）
  第二增量（提交段的**判定**）：
      §3.4 resolve（局部消解：按节点分组、组内按 key 排序、截断到 cap−deg 预算）
      §3.5 commit 交集判据（候选存活 ⟺ 每个角点都接受它）
      §4.3 prune 标记（度数更新 deg_{t+1} = deg_t + cones_at − breaks_at，再标 deg < dmin）
  第三增量（写操作）：
      cone_* / break_edge / remove_vertex 合成为**一次性 CSR 重建**（`rebuild_rings`）——
      见「写操作为什么可以一次性重建」。L0 整帧自此在 GPU 闭环判定 + 写入。
  第四增量（采纳动力学，2026-09-13，见 §7.4 / §7 分岔 N2）：
      `vminus="dense"`（非对称湮灭：断两端皆负曲率的过密边）与 `vplus`（创生抑制）
      接入 GPU 全链 —— 决策段补两道过滤（`fv_len ≤ cap` 容量守卫 + `δ > 0`），
      写操作段的湮灭预算改**显式 `budget`** 入参（`None ⇒ 断全部 plan`，即 dense/none
      语义），`frame_gpu` 按 `core.vminus` 分派 plan（`_annihilation_plan`）。
      **注**：`fv_len ≤ cap` 不是新规则，而是 CPU 异常 X1 修复在 GPU 侧的对应缺陷
      —— 此前 GPU 同样无新点容量校验，只是基线动力学在 4 帧内造不出 k > cap 的洞
      而未被触发。基线路径（`dangling`）行为逐位不变。

  仍留在 CPU 的两件事（均为**固有串行**或**接口开销**，非可并行决策）：
      ① §3.2 湮灭预算的贪心扫描（`_plan_breaks` / `_plan_dense_breaks` 与 conserved
         下的逐条预算截断）；
      ② 帧末把重建后的 CSR 物化成 RotNet（`_to_rotnet`，CPU 内核的既有接口）。
  ①只在 `pairing="conserved"` 或 `vminus="dense"` 下才进入热路径（默认 `none` + `dangling` 时零成本）。

  边界判据：决策段只读 S_t、不改状态 ⇒ 输出是 S_t 的纯函数 ⇒ 可与 CPU 逐位比对。

第二增量的两个关键简化（已实测确立）
------------------------------------
  ① **commit 内 cone_* 永不失败**：候选来自帧首快照，而 cone 只改写自己那个面的
     内部定向（插入 w 只变动 succ_i(k)/succ_j(i)/succ_k(j)），不会破坏快照里任何
     其它面 —— 因为「被破坏」要求两个不同面共用同一条 dart，不可能。
     实测 24 组配置（2 种子 × 2 drive × 2 pairing × 3 cap × 有/无穿刺）× 6 帧，
     cone_* 返回 None 的次数恒为 0。故 commit 无需逐条重新终审。
  ② 于是 commit 只剩「交集判据」这一步可并行的判定；只有 §3.2 的预算扫描串行。

关键简化（与 CPU 的等价性依据）
-------------------------------
  CPU 的 propose() 按 faces() 的迭代顺序 append 候选；但该顺序**不影响结果**：
      resolve() 与 commit() 都对候选按 key 排序；
      而 key =（类型, ±slack, sorted(corners)）在三角剖分上唯一
      （两个不同的面不可能共享同一角点集，否则违背流形性）。
  ⇒ GPU 只需产出「候选集 + 正确的 key」，**不必复刻 CPU 的迭代顺序**。
  验证时按排序后的键多重集比对，而非按列表顺序。

数据布局（§5.1 边中心）
-----------------------
  旋转系统展平为 CSR。全部下标都是**位置下标**（ids 中的序号），不是原 id：
      ids     [V]    活跃节点原 id，升序
      rot_ptr [V+1]  rot_ptr[i+1] = rot_ptr[i] + deg(i)
      rot_idx [2E]   展平的环序邻居表
      deg     [V]
  dart = 有向边 (v, w)，flat 索引 = rot_ptr[v] + (w 在 v 环中的位置)。
  反向 dart：rev[d] = dart (target, owner) 的 flat 索引；dart_next[d] = 见下。

用法
----
  python l0_gpu.py selftest=1
  python l0_gpu.py verify=1 seed=icosa nframes=4 cap=12 drive=saturate
  python l0_gpu.py commit=1 seed=icosa nframes=4 cap=12 pairing=none
  python l0_gpu.py write=1  seed=icosa nframes=4 cap=12 pairing=none
  python l0_gpu.py verify=1 seed=icosa nframes=4 cap=12 vminus=dense vplus=any
                              （第四增量：采纳动力学，另可加 kappa=…）
  python l0_gpu.py bench=1 seed=icosa nframes=4            （CPU/GPU 决策段耗时对比）
"""

import sys

import cupy as cp
import numpy as np

CREATION = "V+"
ANNIHILATION = "V-"
KIND_FACE = 0
KIND_HOLE = 1
KIND_DEGEN = -1


# ============================================================
# §5.1 数据布局：旋转系统 → CSR
# ============================================================
def build_csr(rot):
    """旋转系统（dict[int, list[int]]）→ CSR。返回 (ids, rot_ptr, rot_idx, deg)。

    输入是 Python dict（宿主侧），故此处允许用 numpy 做一次性的展平；
    输出的四个数组全部在 GPU 上，后续所有算子都在 GPU 上完成。
    关键：rot_idx 存**位置下标**，需用 searchsorted 把原 id 映射到 ids 中的序号。
    """
    ids_np = np.array(sorted(rot), dtype=np.int64)
    V = int(ids_np.size)
    deg_np = np.array([len(rot[int(v)]) for v in ids_np], dtype=np.int64)
    rot_ptr_np = np.zeros(V + 1, dtype=np.int64)
    np.cumsum(deg_np, out=rot_ptr_np[1:])

    if V:
        flat = np.concatenate([np.asarray(rot[int(v)], dtype=np.int64)
                               for v in ids_np])
        pos = np.searchsorted(ids_np, flat)
        if not np.array_equal(ids_np[pos], flat):
            raise ValueError("rot 含不在 ids 中的邻居 id（环序表不自洽）")
    else:
        pos = np.zeros(0, dtype=np.int64)

    return (cp.asarray(ids_np, dtype=cp.int32),
            cp.asarray(rot_ptr_np, dtype=cp.int32),
            cp.asarray(pos, dtype=cp.int32),
            cp.asarray(deg_np, dtype=cp.int32))


def owner_of_dart(rot_ptr):
    """每个 dart 的 owner（位置下标）。

    owner[d] = 满足 rot_ptr[i] <= d < rot_ptr[i+1] 的 i。
    用 side='right' 的 searchsorted 再减 1，即「<= d 的元素个数 - 1」。
    """
    nd = int(rot_ptr[-1])
    d = cp.arange(nd, dtype=cp.int32)
    return (cp.searchsorted(rot_ptr, d, side='right') - 1).astype(cp.int32)


def reverse_dart(rot_ptr, rot_idx):
    """每个 dart 的反向 dart 的 flat 索引。

    dart d = (owner, target)；rev[d] 满足 rot_idx[rev[d]] == owner 且在 target 环内。
    做法：把 (owner, target) 打包成单调键后 argsort，再用 (target, owner) 的键
    searchsorted 定位 —— 全程两个排序/查找算子，无逐元素循环。
    键打包 owner*V + target 无碰撞：owner, target ∈ [0, V)。
    """
    V = int(rot_ptr.size) - 1
    owner = owner_of_dart(rot_ptr).astype(cp.int64)
    target = rot_idx.astype(cp.int64)
    key = owner * V + target
    order = cp.argsort(key)
    skey = key[order]
    pos = cp.searchsorted(skey, target * V + owner)
    return order[pos].astype(cp.int32)


def dart_next_index(rot_ptr, rot_idx, rev):
    """dart_next(d) 的 flat 索引。

    d = (owner, target)；rev[d] 是 dart (target, owner)，它在 target 环中的位置
    p = rev[d] - rot_ptr[target]；于是 succ(target, owner) 的位置是 (p+1) % deg(target)，
    该 dart 的 flat 索引 = rot_ptr[target] + (p+1) % deg(target)。
    回绕由取模自然给出（此处不需要 where 分支）。
    """
    target = rot_idx.astype(cp.int64)
    pos = rev.astype(cp.int64) - rot_ptr[target].astype(cp.int64)
    deg = (rot_ptr[1:] - rot_ptr[:-1]).astype(cp.int64)
    nxt = (pos + 1) % deg[target]
    return (rot_ptr[target] + nxt).astype(cp.int32)


# ============================================================
# §5.2 并行原语：面枚举
# ============================================================
def enumerate_faces(rot_ptr, rot_idx, rev, dn=None):
    """枚举全部 dart 环（面）。返回 (face_id, n_faces, face_rep)。

    需要每个 dart 所属环的**最小 dart 索引** rep[d]，用指针跳跃 O(log) 轮：
        rep ← min(rep, rep[nxt]);  nxt ← nxt[nxt]
    第 k 轮后 rep[d] = d 起沿环 2^k 步内的最小 dart；取 k = bit_length(2E) 即覆盖整环。
    """
    nd = int(rot_ptr[-1])
    if dn is None:
        dn = dart_next_index(rot_ptr, rot_idx, rev)
    rep = cp.arange(nd, dtype=cp.int32)
    nxt = dn
    for _ in range(max(1, nd.bit_length())):
        rep = cp.minimum(rep, rep[nxt])
        nxt = nxt[nxt]
    face_rep = cp.unique(rep)                    # 已是升序
    n_faces = int(face_rep.size)
    face_id = cp.searchsorted(face_rep, rep).astype(cp.int32)
    return face_id, n_faces, face_rep


def face_vertices(rot_ptr, rot_idx, rev, face_rep, dn=None):
    """把每个面的顶点环按 dart 顺序展开成 CSR。返回 (fv_ptr, fv_idx, fv_len)。

    环长用「走到回到起点」的步数（按轮向量化，避免逐面 while）；
    顶点取该 dart 的 owner。两趟都只是「按轮推进 + 掩码」，无逐元素循环。
    """
    if dn is None:
        dn = dart_next_index(rot_ptr, rot_idx, rev)
    n_faces = int(face_rep.size)
    if n_faces == 0:
        z = cp.zeros(0, dtype=cp.int32)
        return cp.zeros(1, dtype=cp.int32), z, z

    # 第一趟：环长
    cur = face_rep.copy()
    fv_len = cp.zeros(n_faces, dtype=cp.int32)
    live = cp.ones(n_faces, dtype=bool)
    for _ in range(int(rot_ptr[-1])):
        cur = dn[cur]
        fv_len += live.astype(cp.int32)
        live &= (cur != face_rep)
        if not bool(live.any()):
            break

    fv_ptr = cp.zeros(n_faces + 1, dtype=cp.int32)
    fv_ptr[1:] = cp.cumsum(fv_len).astype(cp.int32)
    m = int(fv_ptr[-1])

    # 第二趟：写顶点（owner）
    owner = owner_of_dart(rot_ptr)
    fv_idx = cp.zeros(m, dtype=cp.int32)
    if m:
        cur = face_rep.copy()
        base = fv_ptr[:-1]
        for step in range(int(fv_len.max())):
            sel = step < fv_len                      # 只有还没走完的面才写
            if not bool(sel.any()):
                break
            fv_idx[(base + step)[sel]] = owner[cur[sel]]
            cur = dn[cur]
    return fv_ptr, fv_idx, fv_len


def classify_faces(fv_ptr, fv_idx, fv_len):
    """面分类。返回 (kind, n_face, n_hole)。kind: 0=真三角面, 1=洞, -1=退化。

    「顶点两两不同」用全局排序后的相邻去重计数实现（不用逐面 Python set）：
        distinct[face] = 该面内不同顶点数
    """
    n_faces = int(fv_len.size)
    m = int(fv_idx.size)
    kind = cp.full(n_faces, KIND_DEGEN, dtype=cp.int32)
    if m == 0:
        return kind, 0, 0
    fid = (cp.searchsorted(fv_ptr, cp.arange(m, dtype=cp.int32), side="right")
           - 1).astype(cp.int32)
    key = (fid.astype(cp.int64) * (int(fv_idx.max()) + 1)
           + fv_idx.astype(cp.int64)).astype(cp.int64)
    key.sort()
    flag = cp.ones(m, dtype=cp.int32)
    flag[1:] = (key[1:] != key[:-1]).astype(cp.int32)
    distinct = cp.bincount(fid, weights=flag.astype(cp.float64),
                           minlength=n_faces).astype(cp.int32)

    kind[(fv_len == 3) & (distinct == 3)] = KIND_FACE
    kind[(fv_len >= 4) & (distinct == fv_len)] = KIND_HOLE
    return kind, int((kind == KIND_FACE).sum()), int((kind == KIND_HOLE).sum())


# ============================================================
# §3.1/§3.3 候选生成与优先级键（复刻 l0_core.priority）
# ============================================================
def _slack_of(pos, deg, cap):
    """**按槽位/按单点**的 slack 贡献 = cap − deg(v)。

    CPU 的 slack = Σ_{v ∈ set(corners)} (cap − deg(v))。
    面内角点已互不重复（kind != -1 保证），故 set 大小 = 角点数；
    于是「按候选汇总」由调用方用 bincount 完成，本函数只出单项，不自行求和。
    """
    return int(cap) - deg[pos].astype(cp.int64)


def candidates_and_keys(rot_ptr, rot_idx, rev, ids, deg,
                        fv_ptr, fv_idx, fv_len, kind, cap, dmin, drive,
                        vplus="any", kappa=6, dn=None):
    """创生候选 + 优先级键。

    返回 (cand_ptr, cand_idx, cand_len, key1, key2, key3)：
        cand_ptr [n_cand+1] / cand_idx [Σ角点] / cand_len [n_cand]  角点 CSR
        key1 恒 0（0 = 创生，与 CPU 的 ctype 编码一致）
        key2 = sign * slack，sign = +1 if drive == "saturate" else −1
        key3 [n_cand, W] = sorted(原 id 角点)，不足 W 列补 −1
            —— 补 −1 使「定宽整数行字典序」等价于「Python 变长 tuple 比较」
               （短 tuple 是长 tuple 前缀时更小，−1 小于一切原 id，同序）。

    两道过滤（与 CPU `propose()` 对齐，顺序无关）：
        · `fv_len ≤ cap` —— §2.2 容量约束（L0-A5）。新点度数 = 3（face）/ k（hole），
          而 §3.4 局部消解只约束**旧角点**预算，故新点度数必须在此独立校验
          （异常 X1：`cone_hole` 锥化 k 边洞会造出 deg=k 的超容点）。
        · `vplus="gap"` —— 只保留正角亏 δ = Σ_{角点}(κ − deg) > 0 的候选。
    """
    n_faces = int(kind.size)
    sel = cp.where(kind != KIND_DEGEN)[0].astype(cp.int32)
    if n_faces and int(fv_idx.size):
        fid = (cp.searchsorted(fv_ptr, cp.arange(int(fv_idx.size), dtype=cp.int32),
                               side="right") - 1).astype(cp.int32)
    else:
        fid = cp.zeros(0, dtype=cp.int32)

    if int(sel.size):
        sel = sel[fv_len[sel] <= int(cap)]
    if vplus == "gap" and int(sel.size):
        # kind != DEGEN ⇒ 面内角点互不重复 ⇒ 逐角点求和 == 对 set 求和
        dgap = cp.bincount(
            fid, weights=(cp.float64(kappa) - deg[fv_idx].astype(cp.float64)),
            minlength=n_faces)
        sel = sel[dgap[sel] > 0]

    n_cand = int(sel.size)
    if n_cand == 0:
        z = cp.zeros(0, dtype=cp.int32)
        return z, z, z, z, z, cp.zeros((0, 0), dtype=cp.int32)

    cand_len = fv_len[sel].astype(cp.int32)
    cand_ptr = cp.zeros(n_cand + 1, dtype=cp.int32)
    cand_ptr[1:] = cp.cumsum(cand_len).astype(cp.int32)
    m = int(cand_ptr[-1])

    # 角点 CSR：把「被选中面的槽位」搬到候选自己的偏移上
    keep = cp.zeros(n_faces, dtype=cp.int32)
    keep[sel] = cp.arange(n_cand, dtype=cp.int32)
    face_sel = cp.zeros(n_faces, dtype=bool)
    face_sel[sel] = True
    slots = cp.where(face_sel[fid])[0]
    cand_idx = cp.zeros(m, dtype=cp.int32)
    cand_idx[cand_ptr[keep[fid[slots]]] + (slots - fv_ptr[fid[slots]])] = fv_idx[slots]

    # key2：slack
    cslot = (cp.searchsorted(cand_ptr, cp.arange(m, dtype=cp.int32),
                             side="right") - 1).astype(cp.int32)
    slack = cp.bincount(cslot,
                        weights=_slack_of(cand_idx, deg, cap).astype(cp.float64),
                        minlength=n_cand).astype(cp.int64)
    sign = 1 if drive == "saturate" else -1
    key2 = (sign * slack).astype(cp.int32)

    # key3：每个候选的角点按**原 id** 升序，写进该候选的列
    # 排序键 = (cslot, cid)：打包成单调整数即可（每个槽位的 (cslot, 原 id) 唯一，
    # 无并列 ⇒ argsort 结果确定）。不用 cp.lexsort —— 其签名与 numpy 不同。
    cid = ids[cand_idx].astype(cp.int64)
    pack = cslot.astype(cp.int64) * (int(ids.max()) + 1) + cid
    order = cp.argsort(pack)
    W = int(cand_len.max())
    key3 = cp.full((n_cand, W), -1, dtype=cp.int32)
    rank = (cp.arange(m, dtype=cp.int32) - cand_ptr[cslot[order]])
    key3[cslot[order], rank] = cid[order].astype(cp.int32)

    key1 = cp.zeros(n_cand, dtype=cp.int32)
    return cand_ptr, cand_idx, cand_len, key1, key2, key3


def annihilation_candidates(ids, deg, cap, dmin, drive):
    """湮灭候选（悬挂/孤立节点）。返回 (node_pos, key2)。

    key1 恒 1（1 = 湮灭）；key3 = (原 id,)，故不必单独输出。
    """
    pos = cp.where(deg < int(dmin))[0].astype(cp.int32)
    sign = 1 if drive == "saturate" else -1
    key2 = (sign * _slack_of(pos, deg, cap)).astype(cp.int32)
    return pos, key2


# ============================================================
# §3.4/§3.5/§4.3 提交段判定（阶段三第二增量）
#   全部向量化：只用 argsort / lexsort / searchsorted / bincount / cumsum 等
#   O(1) 次内核启动的算子，**无任何逐元素 Python 循环**。
# ============================================================
def rank_creation(key1, key2, key3):
    """创生候选按 key 升序的位次。返回 rank [n] int32。

    CPU 的 key = (0, sign*slack, tuple(sorted(corners)))，即按 (key1, key2, key3 行)
    字典序。用 cp.lexsort 做定宽行字典序：**最后一行为主键**（与 numpy 一致），
    故行序为 [key3 末列, …, key3 首列, key2, key1]。
    key3 的补位 −1 小于一切原 id ⇒ 与 Python 变长 tuple 比较同序。
    注：CuPy 的 lexsort 只接受 **2-D 数组**（不接受 tuple）。
    """
    n = int(key2.size)
    if n == 0:
        return cp.zeros(0, dtype=cp.int32)
    W = int(key3.shape[1])
    rows = [key3[:, j] for j in range(W - 1, -1, -1)]
    rows.append(key2)
    rows.append(key1)
    order = cp.lexsort(cp.stack(rows))
    rank = cp.empty(n, dtype=cp.int32)
    rank[order] = cp.arange(n, dtype=cp.int32)
    return rank


def incidence(cand_ptr, cand_idx):
    """候选-角点 CSR → (inc_c, inc_v)：每个槽位属于哪个候选、角点是谁。

    inc_c[m] = 槽位 m 的候选下标（cand_ptr 的分段展开）。
    searchsorted 返回「插入位」，即段号 + 1，故减 1。
    """
    m = int(cand_idx.size)
    if m == 0:
        z = cp.zeros(0, dtype=cp.int32)
        return z, z
    inc_c = (cp.searchsorted(cand_ptr, cp.arange(m, dtype=cp.int32),
                             side="right") - 1).astype(cp.int32)
    return inc_c, cand_idx.astype(cp.int32)


def resolve_accept(inc_c, inc_v, rank, deg, cap, n_cand):
    """§3.4 局部消解：每个角点是否接受与之相关的候选。

    accept[v] = v 的候选表按 rank 升序的前 budget_v 个，budget_v = cap − deg(v)。
    做法：把 (节点, 组内序) 打包成单调整数 key = inc_v * n_cand + rank[inc_c]
    （每个槽位唯一 ⇒ argsort 结果确定），排序后即「按节点分组、组内按 rank 升序」；
    组内位置 = 全局行序 − 该组起始行（前缀和右移一位）。
    返回 (acc, sv, order, inc_c_row)：acc 为行序的接受标记，inc_c_row = inc_c[order]。
    """
    m = int(inc_v.size)
    if m == 0:
        z = cp.zeros(0, dtype=cp.int32)
        return cp.zeros(0, dtype=bool), z, z, cp.zeros(0, dtype=cp.int32)
    nv = int(deg.size)
    pack = inc_v.astype(cp.int64) * int(n_cand) + rank[inc_c].astype(cp.int64)
    order = cp.argsort(pack)
    sv = inc_v[order].astype(cp.int32)
    cnt = cp.bincount(sv.astype(cp.int64), minlength=nv).astype(cp.int64)
    # 组起始 = 严格小于该节点的节点数 = 计数前缀和右移一位
    grp_start = cp.concatenate([cp.zeros(1, dtype=cp.int64), cp.cumsum(cnt)[:-1]])
    gpos = cp.arange(m, dtype=cp.int64) - grp_start[sv.astype(cp.int64)]
    acc = gpos < (int(cap) - deg[sv].astype(cp.int64))
    return acc, sv, order, inc_c[order].astype(cp.int32)


def intersection(inc_c_row, acc, cand_len, n_cand):
    """§3.5 交集提交判据：候选存活 ⟺ 它的每个角点都接受它。

    按候选汇总被接受的入射数，与角点数相等即存活。
    """
    if int(cand_len.size) == 0:
        return cp.zeros(0, dtype=bool)
    cnt = cp.bincount(inc_c_row.astype(cp.int64),
                      weights=acc.astype(cp.float64),
                      minlength=int(n_cand)).astype(cp.int64)
    return cnt == cand_len.astype(cp.int64)


def survivors(passes, rank):
    """通过交集的候选，按 rank 升序（= CPU commit 的提交顺序）。返回 sel [m] int32。"""
    sel = cp.where(passes)[0].astype(cp.int32)
    if int(sel.size) == 0:
        return sel
    return sel[cp.argsort(rank[sel])].astype(cp.int32)


def slots_of(sel, cand_ptr, cand_len):
    """sel 中各候选的角点槽位（cand_idx 中的下标），按 sel 顺序拼接。

    ragged range：把每个候选的 [cand_ptr[c], cand_ptr[c]+len_c) 展开成一维。
    seg = searchsorted(包含式前缀和, idx, 'right') 恰给出 idx 所属的段号。
    """
    if int(sel.size) == 0:
        return cp.zeros(0, dtype=cp.int32)
    rep = cand_len[sel].astype(cp.int64)
    total = int(rep.sum())
    if total == 0:
        return cp.zeros(0, dtype=cp.int32)
    idx = cp.arange(total, dtype=cp.int64)
    cum = cp.cumsum(rep)
    seg = cp.searchsorted(cum, idx, side="right")
    base = cum - rep
    return (cand_ptr[sel].astype(cp.int64)[seg] + (idx - base[seg])).astype(cp.int32)


def degree_next(deg, corner_nodes, break_u=None, break_v=None, dmin=3):
    """§4.3 修剪标记：先算下一帧度数，再标出 deg < dmin 的位置。

    deg_{t+1}(v) = deg_t(v) + (以 v 为角点的锥化次数) − (v 的入射断边数)。
    corner_nodes / break_u / break_v 均为**位置下标**。
    """
    nv = int(deg.size)
    dn = deg.astype(cp.int64)
    if int(corner_nodes.size):
        dn = dn + cp.bincount(corner_nodes.astype(cp.int64),
                              minlength=nv).astype(cp.int64)
    for arr in (break_u, break_v):
        if arr is not None and int(arr.size):
            dn = dn - cp.bincount(arr.astype(cp.int64), minlength=nv).astype(cp.int64)
    return dn, cp.where(dn < int(dmin))[0].astype(cp.int32)


def gpu_commit_stage(rot, cap=64, dmin=3, drive="slack", vplus="any", kappa=6):
    """提交段判定总入口。返回 = 决策段 dict + {rank, sel, slots}。

    sel 为「通过交集、按 rank 升序」的候选下标；角点用 cand_idx[slots_of(sel)] 取
    （**保环序**：slots 沿 cand_ptr 分段连续，故 cone_* 的定向正确）。
    §3.2 的预算截断不在此处（它依赖湮灭规划，属固有串行，见 frame_gpu）。
    """
    g = gpu_decision(rot, cap=cap, dmin=dmin, drive=drive, vplus=vplus, kappa=kappa)
    n_cand = int(g["cand_len"].size)
    rank = rank_creation(g["key1"], g["key2"], g["key3"])
    inc_c, inc_v = incidence(g["cand_ptr"], g["cand_idx"])
    acc, _sv, _order, inc_c_row = resolve_accept(
        inc_c, inc_v, rank, g["deg"], cap, n_cand)
    passes = intersection(inc_c_row, acc, g["cand_len"], n_cand)
    sel = survivors(passes, rank)
    slots = slots_of(sel, g["cand_ptr"], g["cand_len"])
    g.update({"rank": rank, "sel": sel, "slots": slots})
    return g


# ============================================================
# §5.3 写操作（阶段三第三增量）：cone / break / remove → 一次性 CSR 重建
#   CPU 的写操作只是三种局部环序编辑：插入、删除、新建/丢弃整行。
#   本帧的插入锚点 dart 两两不同（每个 dart 恰属一个面，而候选 = 面），
#   故插入与锚点一一对应、无冲突 ⇒ 整帧写入可完全并行地一次重建。
#   全部向量化：只用 argsort / searchsorted / bincount / cumsum 等 O(1) 次启动。
# ============================================================
def ring_pos_of_dart(rot_ptr, rot_idx, q_owner, q_member):
    """一批有序对 (q_owner, q_member)（位置下标）→ 它们的 flat dart 索引 [n]。

    把 (owner, member) 打包成单调整数键后 argsort，再用查询键 searchsorted 定位。
    键无碰撞（owner, member ∈ [0, V)），故结果确定。约定查询对必然存在。
    """
    V = int(rot_ptr.size) - 1
    owner = owner_of_dart(rot_ptr).astype(cp.int64)
    key = owner * V + rot_idx.astype(cp.int64)
    order = cp.argsort(key)
    skey = key[order]
    q = (q_owner.astype(cp.int64) * V + q_member.astype(cp.int64))
    idx = cp.searchsorted(skey, q)          # 返回 int64，索引前先降位
    return order[idx].astype(cp.int32)


def rebuild_rings(rot_ptr, rot_idx, deg, nid0, dead_pos, broken,
                  cr_ptr, cr_idx, cr_len, dmin):
    """把本帧「已提交的创生 + 断边 + 删点」一次性写成新的 CSR。

    返回 (rot_ptr2, rot_idx2, deg2, nid2)；ids2 由调用方拼（存活旧 ids ‖ 新 id）。
    m = cr_len.size = **全部**已提交候选数（id 一律被消耗，即使随后被删）。

    依据（见 §5.3 与规范 spec3）：
      * 候选 i 的角点保环序 c[0..k-1] ⇒ 写操作 = 对每个 t，在 owner=c[t] 的环序里
        **紧跟锚点 c[(t-1) mod k] 之后**插入新元胞；新行 = [c0] + reversed(c[1:])。
        face 与 hole 用**同一条规则**（cone_tri_face 的 rot[w]=[i,k,j] 亦然）。
      * 每条旧 dart (p,m) 的保留判据：keep = (p 存活) ∧ (m 存活) ∧ (该 dart 未断)。
      * 每条插入记录的发射判据：emit = (新元胞存活 ⟺ cr_len ≥ dmin) ∧ (owner 存活)。
        锚点是否存活**不**参与判据 —— CPU 是「先插入、后删点」，锚点被删后新元胞
        恰好落在原锚点位置，故 emit 不查锚点才是逐位一致的。
      * 新行滤掉已死角点，行长 = cr_len − 该候选的已死角点数。
    """
    V = int(deg.size)
    nd_tot = int(rot_ptr[-1])
    m = int(cr_len.size)
    M = int(cr_idx.size)

    # --- S1 存活掩码 ---
    alive = cp.ones(V, dtype=bool)
    if int(dead_pos.size):
        alive[dead_pos.astype(cp.int64)] = False
    no_alive = int(alive.sum())
    new_alive = cr_len >= int(dmin)
    opos = (cp.cumsum(alive.astype(cp.int32)) - 1).astype(cp.int32)
    newrank = (cp.cumsum(new_alive.astype(cp.int32)) - 1).astype(cp.int32)

    # --- S2 旧 dart 的保留掩码（长度 2E）---
    owner = owner_of_dart(rot_ptr)
    member = rot_idx.astype(cp.int32)
    own64 = owner.astype(cp.int64)
    keep = alive[own64] & alive[member.astype(cp.int64)]
    if int(broken.size):
        bu = broken[:, 0].astype(cp.int64)
        bv = broken[:, 1].astype(cp.int64)
        bkey = cp.sort(cp.concatenate([bu * V + bv, bv * V + bu]))
        dkey = own64 * V + member.astype(cp.int64)
        pos = cp.minimum(cp.searchsorted(bkey, dkey), bkey.size - 1)
        keep &= bkey[pos] != dkey

    # --- S3 插入记录（由 cr_* 展开，无逐候选循环）---
    ins_flag = cp.zeros(nd_tot, dtype=bool)
    ins_val = cp.zeros(nd_tot, dtype=cp.int32)
    if M:
        slot = cp.arange(M, dtype=cp.int32)
        inc_c = (cp.searchsorted(cr_ptr, slot, side="right") - 1).astype(cp.int32)
        base = cr_ptr[inc_c]
        t = slot - base
        klen = cr_len[inc_c]
        prev = base + ((t - 1 + klen) % klen)          # 锚点所在槽位
        o_owner = cr_idx[slot].astype(cp.int32)
        o_anch = cr_idx[prev].astype(cp.int32)
        emit = new_alive[inc_c] & alive[o_owner.astype(cp.int64)]
        if int(emit.sum()):
            flat = ring_pos_of_dart(rot_ptr, rot_idx, o_owner[emit], o_anch[emit])
            ins_flag[flat] = True                      # 锚点两两不同 ⇒ 下标唯一
            ins_val[flat] = no_alive + newrank[inc_c[emit]]

    # --- S4 每行长度 ---
    kcnt = cp.bincount(owner[keep].astype(cp.int64), minlength=V).astype(cp.int32)
    icnt = cp.bincount(owner[ins_flag].astype(cp.int64), minlength=V).astype(cp.int32)
    olen = cp.where(alive, kcnt + icnt, cp.zeros_like(kcnt)).astype(cp.int32)
    if m:
        dead_corner = cp.bincount(inc_c[~alive[cr_idx.astype(cp.int64)]],
                                  minlength=m).astype(cp.int32)
        nlen = cp.where(new_alive, cr_len - dead_corner,
                        cp.zeros_like(cr_len)).astype(cp.int32)
    else:
        nlen = cp.zeros(0, dtype=cp.int32)

    # --- S5 输出分段起点与总长 ---
    olen_alive = olen[alive]
    nlen_alive = nlen[new_alive]
    base_new = int(olen_alive.sum())
    tot = base_new + int(nlen_alive.sum())
    start_old = cp.concatenate([cp.zeros(1, cp.int32),
                                cp.cumsum(olen)[:-1]]).astype(cp.int32)
    if m:
        start_new = (base_new + cp.concatenate(
            [cp.zeros(1, cp.int32), cp.cumsum(nlen)[:-1]])).astype(cp.int32)
    else:
        start_new = cp.zeros(0, dtype=cp.int32)
    lens = cp.concatenate([olen_alive, nlen_alive]).astype(cp.int32)
    rot_ptr2 = cp.concatenate([cp.zeros(1, cp.int32),
                               cp.cumsum(lens)]).astype(cp.int32)

    # --- S6 行内槽位（段内前缀和；段起点由 rot_ptr 给出）---
    cs = cp.concatenate([cp.zeros(1, cp.int64), cp.cumsum(keep.astype(cp.int64))])
    csi = cp.concatenate([cp.zeros(1, cp.int64), cp.cumsum(ins_flag.astype(cp.int64))])
    seg = rot_ptr[:-1].astype(cp.int64)
    kpre = cs[:-1] - cs[seg[own64]]
    ipre = csi[:-1] - csi[seg[own64]]
    k64 = keep.astype(cp.int64)
    so64 = start_old.astype(cp.int64)[own64]
    slot_keep = so64 + kpre + ipre
    slot_ins = so64 + kpre + k64 + ipre

    # --- S7 写入旧节点内容 ---
    out = cp.full(max(tot, 1), cp.int32(-1), cp.int32)
    if int(keep.sum()):
        out[slot_keep[keep]] = opos[member[keep].astype(cp.int64)].astype(cp.int32)
    if int(ins_flag.sum()):
        out[slot_ins[ins_flag]] = ins_val[ins_flag]

    # --- S8 写入新节点内容（模板 [c0] + reversed(c1..)，滤死角点）---
    if M:
        lens_c = cr_len.astype(cp.int64)
        cum = cp.cumsum(lens_c)
        idx8 = cp.arange(M, dtype=cp.int64)
        segc = cp.searchsorted(cum, idx8, side="right")
        j = idx8 - (cum - lens_c)[segc]                # 模板行内序号
        col = cp.where(j == 0, j, lens_c[segc] - j)    # 模板序 → 候选内槽位偏移
        src = cr_ptr[segc].astype(cp.int64) + col
        corner = cr_idx[src].astype(cp.int64)
        val = opos[corner]
        ok = alive[corner] & new_alive[segc]
        okey = segc * (int(lens_c.max()) + 1) + j      # 组内唯一 ⇒ 排序即模板序
        ord_ = cp.argsort(okey)
        s_seg, s_val, s_ok = segc[ord_], val[ord_], ok[ord_]
        exc = cp.concatenate([cp.zeros(1, cp.int64),
                              cp.cumsum(s_ok.astype(cp.int64))])[:-1]
        gcnt = cp.bincount(s_seg[s_ok].astype(cp.int64), minlength=m)
        gstart = cp.concatenate([cp.zeros(1, cp.int64), cp.cumsum(gcnt)[:-1]])
        rank_in_row = exc - gstart[s_seg]              # 过滤后行内序号
        tgt = start_new.astype(cp.int64)[s_seg] + rank_in_row
        if int(s_ok.sum()):
            out[tgt[s_ok]] = s_val[s_ok]

    return rot_ptr2, out[:tot], lens, int(nid0) + m


def _to_rotnet(ids2, rot_ptr2, rot_idx2, nid2):
    """重建后的 CSR → RotNet（宿主侧物化，CPU 内核的既有接口）。

    这里是唯一不可免的 O(V) 宿主开销：把列式 CSR 摊回 dict[int, list[int]]，
    供 l0_core 的只读查询 / 审计使用（帧调度本身不依赖它）。
    """
    from combinatorial_proto import RotNet
    ids_l = cp.asnumpy(ids2).tolist()
    ptr = cp.asnumpy(rot_ptr2).tolist()
    idx = cp.asnumpy(rot_idx2).tolist()
    rot = {int(v): [int(ids_l[j]) for j in idx[ptr[i]:ptr[i + 1]]]
           for i, v in enumerate(ids_l)}
    net = RotNet(rot)
    net.nid = int(nid2)
    return net


def gpu_write_stage(net, cap, dmin, drive, plan=(), pairing="none",
                    vplus="any", kappa=6, budget=None):
    """写操作段总入口：GPU 判定 + GPU 一次性重建。纯函数（不改 net）。

    plan 为 §3.2 的湮灭规划（原始 id 对）。`budget` 显式给出本帧创生预算：
        budget 为 None  ⇒ 不配平，断**全部** plan（净 ΔE ≠ 0）；
        budget 为整数  ⇒ 创生逐条对预算截断，只断 plan[:spent]（ΣΔE = 0）。
    未显式传入时按 pairing 推导（conserved ⇒ len(plan)），故既有基线路径不变。
    返回 dict：CSR 四元组 + 事件计数。唯一留在宿主的可并行工作是 §3.2 的
    逐条预算截断（固有串行扫描），见模块头「留在 CPU 的两件事」①。
    """
    if budget is None and pairing == "conserved":
        budget = len(plan)
    st = gpu_commit_stage(net.rot, cap=cap, dmin=dmin, drive=drive,
                          vplus=vplus, kappa=kappa)
    ids_np = cp.asnumpy(st["ids"])
    n_sel = int(st["sel"].size)

    # ① 预算截断：按 rank 升序逐条判 spent+cost>budget ⇒ 跳过（不是 break）
    lens = cp.asnumpy(st["cand_len"][st["sel"]]) if n_sel else np.zeros(0, np.int64)
    sel_keep = np.ones(int(lens.size), dtype=bool)
    spent = 0
    for i in range(int(lens.size)):
        k = int(lens[i])
        if budget is not None and spent + k > budget:
            sel_keep[i] = False
        else:
            spent += k
    sel = st["sel"][cp.asarray(sel_keep)] if n_sel else st["sel"]
    m = int(sel.size)

    # 已提交候选的角点 CSR（提交序 = rank 升序 = 新 id 序）
    cr_len = st["cand_len"][sel].astype(cp.int32)
    cr_ptr = cp.zeros(m + 1, dtype=cp.int32)
    if m:
        cr_ptr[1:] = cp.cumsum(cr_len).astype(cp.int32)
    cr_idx = (st["cand_idx"][slots_of(sel, st["cand_ptr"], st["cand_len"])]
              if m else cp.zeros(0, dtype=cp.int32)).astype(cp.int32)

    # ② 断边（位置下标对）与 §4.3 修剪标记
    pl = list(plan[:spent]) if budget is not None else list(plan)
    if pl:
        pa = np.asarray(pl, dtype=np.int64)
        bu = ids_np.searchsorted(pa[:, 0]).astype(np.int32)
        bv = ids_np.searchsorted(pa[:, 1]).astype(np.int32)
        broken_pos = cp.asarray(np.stack([bu, bv], axis=1), dtype=cp.int32)
    else:
        broken_pos = cp.zeros((0, 2), dtype=cp.int32)
    _, dead_pos = degree_next(st["deg"], cr_idx, cp.asarray(bu) if pl else None,
                              cp.asarray(bv) if pl else None, dmin)

    rot_ptr2, rot_idx2, deg2, nid2 = rebuild_rings(
        st["rot_ptr"], st["rot_idx"], st["deg"], net.nid, dead_pos,
        broken_pos, cr_ptr, cr_idx, cr_len, dmin)

    # ③ ids2 = 存活旧 ids（升序）‖ 存活新 id（创建序）
    V = int(st["deg"].size)
    alive = cp.ones(V, dtype=bool)
    if int(dead_pos.size):
        alive[dead_pos.astype(cp.int64)] = False
    new_alive = cr_len >= int(dmin)
    ids2 = cp.concatenate([st["ids"][alive],
                           (int(net.nid) + cp.arange(m, dtype=cp.int32))[new_alive]])
    return {"ids2": ids2, "rot_ptr2": rot_ptr2, "rot_idx2": rot_idx2,
            "deg2": deg2, "nid2": nid2,
            "born": m, "spent": spent, "broken": len(pl),
            "dead": int(dead_pos.size) + int((~new_alive).sum())}


def gpu_decision(rot, cap=64, dmin=3, drive="slack", vplus="any", kappa=6):
    """决策段总入口：旋转系统 → (候选, 键, 计数)。

    纯函数，只读输入；返回 dict 便于与 CPU 版比对。
    """
    ids, rot_ptr, rot_idx, deg = build_csr(rot)
    rev = reverse_dart(rot_ptr, rot_idx)
    dn = dart_next_index(rot_ptr, rot_idx, rev)
    face_id, n_faces, face_rep = enumerate_faces(rot_ptr, rot_idx, rev, dn)
    fv_ptr, fv_idx, fv_len = face_vertices(rot_ptr, rot_idx, rev, face_rep, dn)
    kind, n_face, n_hole = classify_faces(fv_ptr, fv_idx, fv_len)
    cand = candidates_and_keys(rot_ptr, rot_idx, rev, ids, deg, fv_ptr, fv_idx,
                               fv_len, kind, cap, dmin, drive, vplus, kappa, dn)
    node_pos, node_key2 = annihilation_candidates(ids, deg, cap, dmin, drive)
    return {"ids": ids, "rot_ptr": rot_ptr, "rot_idx": rot_idx, "deg": deg,
            "rev": rev, "dn": dn, "face_id": face_id, "n_faces": n_faces,
            "face_rep": face_rep, "fv_ptr": fv_ptr, "fv_idx": fv_idx,
            "fv_len": fv_len, "kind": kind, "n_face": n_face, "n_hole": n_hole,
            "cand_ptr": cand[0], "cand_idx": cand[1], "cand_len": cand[2],
            "key1": cand[3], "key2": cand[4], "key3": cand[5],
            "node_pos": node_pos, "node_key2": node_key2}


# ============================================================
# 帧驱动：GPU 决策 + CPU 提交
# ============================================================
def gpu_propose(rot, cap, dmin, drive, vplus="any", kappa=6):
    """GPU 决策段 → CPU 候选表（与 l0_core.propose() 的输出格式完全同形）。

    角点必须**保序**：commit 里 `cone_tri_face(*corners)` 依赖面的环序定向，
    所以取 cand_idx（源自 fv_idx，是面的环序），不能用 key3（已排序）。
    此处的 Python 循环只跨「候选数」量级，不跨 dart/节点，是决策段与提交段的
    边界开销；后续增量若要消掉它，需把提交段也搬上 GPU。
    """
    g = gpu_decision(rot, cap=cap, dmin=dmin, drive=drive, vplus=vplus, kappa=kappa)
    ids = cp.asnumpy(g["ids"]).tolist()
    cptr = cp.asnumpy(g["cand_ptr"]).tolist()
    cidx = cp.asnumpy(g["cand_idx"]).tolist()
    key2 = cp.asnumpy(g["key2"]).tolist()
    key3 = cp.asnumpy(g["key3"])
    out = []
    for i in range(len(cptr) - 1):
        corners = tuple(ids[cidx[j]] for j in range(cptr[i], cptr[i + 1]))
        out.append({"type": CREATION,
                    "kind": "face" if len(corners) == 3 else "hole",
                    "corners": corners,
                    "key": (0, int(key2[i]),
                            tuple(int(x) for x in key3[i] if x >= 0))})
    npos = cp.asnumpy(g["node_pos"]).tolist()
    nkey2 = cp.asnumpy(g["node_key2"]).tolist()
    for i, p in enumerate(npos):
        v = ids[p]
        out.append({"type": ANNIHILATION, "kind": "node", "corners": (v,),
                    "key": (1, int(nkey2[i]), (v,))})
    return out


def _annihilation_plan(core):
    """§3.2 湮灭规划分派：(plan, budget)。

    frame_gpu 与写操作核验共用同一函数，避免「GPU 路径」与「核验路径」各自分派而漂移。
    dense（§7.4 采纳）：断两端皆负曲率的过密边；conserved 下再把 len(plan) 当创生预算。
    """
    if core.vminus == "dense":
        plan = core._plan_dense_breaks(core.net)
        return plan, (len(plan) if core.pairing == "conserved" else None)
    if core.pairing == "conserved":
        plan = core._plan_breaks(core.net)
        return plan, len(plan)
    return (), None


def frame_gpu(core):
    """用 GPU 整帧路径驱动一帧。语义与 L0Core.frame() 逐位一致。

    分工：候选生成 / 优先级 / 局部消解 / 交集提交 / 修剪标记 / **写操作** 全在 GPU；
    宿主侧只剩帧调度、§3.2 的湮灭规划（paired 时）与帧末的 CSR → RotNet 物化。
    写操作不必留在宿主：新 id 的消耗顺序 = 提交序（rank 升序），而重建时已把
    已提交候选按提交序排成 cr_*，故 id 分配天然与 CPU 一致。
    """
    net = core.net
    e0 = net.E()
    plan, budget = _annihilation_plan(core)
    st = gpu_write_stage(net, core.cap, core.dmin, core.drive, plan,
                         core.pairing, vplus=core.vplus, kappa=core.kappa,
                         budget=budget)
    core.net = _to_rotnet(st["ids2"], st["rot_ptr2"], st["rot_idx2"], st["nid2"])
    core.t += 1
    return {"t": core.t, "born": st["born"], "dead": st["dead"],
            "broken": st["broken"], "spent": st["spent"],
            "dE": core.net.E() - e0}


# ============================================================
# 与 CPU 内核的逐位一致性核验
# ============================================================
def _cpu_keys(core, cap, drive):
    """CPU 版的候选多重集：[(type, key), ...]（type: 0 创生 / 1 湮灭）。"""
    out = []
    for c in core.propose():
        t = 0 if c["type"] == CREATION else 1
        out.append((t, c["key"]))
    return sorted(out)


def _gpu_keys(g):
    """GPU 版的候选多重集，键的构造与 CPU 完全同形。"""
    out = []
    n_cand = int(g["key3"].shape[0])
    if n_cand:
        k1 = cp.asnumpy(g["key1"]).tolist()
        k2 = cp.asnumpy(g["key2"]).tolist()
        k3 = cp.asnumpy(g["key3"])
        for i in range(n_cand):
            row = tuple(int(x) for x in k3[i] if x >= 0)
            out.append((int(k1[i]), (int(k1[i]), int(k2[i]), row)))
    n_node = int(g["node_pos"].size)
    if n_node:
        ids = cp.asnumpy(g["ids"]).tolist()
        pos = cp.asnumpy(g["node_pos"]).tolist()
        nk2 = cp.asnumpy(g["node_key2"]).tolist()
        for i in range(n_node):
            v = int(ids[pos[i]])
            out.append((1, (1, int(nk2[i]), (v,))))
    return sorted(out)


def verify_against_cpu(core, cap, dmin, drive, label=""):
    """逐位比对 GPU 决策段与 CPU l0_core.propose()。返回 (是否一致, 说明)。"""
    cpu = _cpu_keys(core, cap, drive)
    gpu = _gpu_keys(gpu_decision(core.net.rot, cap=cap, dmin=dmin, drive=drive,
                                 vplus=getattr(core, "vplus", "any"),
                                 kappa=getattr(core, "kappa", 6)))
    if cpu == gpu:
        return True, f"{label} 候选多重集一致（{len(cpu)} 条）"
    only_cpu = [x for x in cpu if x not in gpu]
    only_gpu = [x for x in gpu if x not in cpu]
    return False, (f"{label} 不一致 CPU={len(cpu)} GPU={len(gpu)}；"
                   f"仅 CPU 有 {len(only_cpu)} 条 {only_cpu[:3]}；"
                   f"仅 GPU 有 {len(only_gpu)} 条 {only_gpu[:3]}")


def _cpu_commit_expect(core):
    """CPU 版的「通过 §3.5 交集」候选有序角点表（按 key 升序 = 提交顺序）。"""
    cands = core.propose()
    accept = core.resolve(cands)
    out = []
    for c in sorted((c for c in cands if c["type"] == CREATION),
                    key=lambda c: c["key"]):
        if all(id(c) in accept.get(v, ()) for v in set(c["corners"])):
            out.append(tuple(c["corners"]))
    return out


def verify_commit_against_cpu(core, cap, dmin, drive, label=""):
    """逐位比对 GPU 提交段判定（resolve + 交集）与 CPU。返回 (是否一致, 说明)。"""
    exp = _cpu_commit_expect(core)
    st = gpu_commit_stage(core.net.rot, cap=cap, dmin=dmin, drive=drive,
                          vplus=getattr(core, "vplus", "any"),
                          kappa=getattr(core, "kappa", 6))
    ids = cp.asnumpy(st["ids"])
    sel = cp.asnumpy(st["sel"]).tolist()
    slen = cp.asnumpy(st["cand_len"][st["sel"]]).tolist() if sel else []
    spos = cp.asnumpy(st["cand_idx"][st["slots"]]).tolist() if sel else []
    got, off = [], 0
    for k in slen:
        got.append(tuple(int(ids[p]) for p in spos[off:off + k]))
        off += k
    if got == exp:
        return True, f"{label} 提交集与顺序一致（{len(exp)} 条）"
    return False, (f"{label} 不一致 CPU={len(exp)} GPU={len(got)}；"
                   f"CPU 前 3 {exp[:3]}；GPU 前 3 {got[:3]}")


def verify_write_against_cpu(core, cap, dmin, drive, label=""):
    """逐位比对 GPU 写操作段（rebuild）与 CPU _step 产出的下一状态。"""
    plan, budget = _annihilation_plan(core)
    nx_cpu, born_c, dead_c, broken_c, spent_c = core._step(core.propose())
    st = gpu_write_stage(core.net, cap, dmin, drive, plan, core.pairing,
                         vplus=core.vplus, kappa=core.kappa, budget=budget)
    nx_gpu = _to_rotnet(st["ids2"], st["rot_ptr2"], st["rot_idx2"], st["nid2"])

    a = {v: tuple(r) for v, r in sorted(nx_cpu.rot.items())}
    b = {v: tuple(r) for v, r in sorted(nx_gpu.rot.items())}
    cnt = (len(born_c), len(dead_c), len(broken_c), spent_c)
    gcnt = (st["born"], st["dead"], st["broken"], st["spent"])
    if a == b and nx_cpu.nid == nx_gpu.nid and cnt == gcnt:
        return True, (f"{label} 写操作一致（V={nx_gpu.V()} E={nx_gpu.E()} "
                      f"nid={nx_gpu.nid} born/dead/broken/spent={gcnt}）")
    if a != b:
        keys = sorted(set(a) | set(b))
        bad = next((v for v in keys if a.get(v) != b.get(v)), None)
        return False, (f"{label} 环序不同；首个差异 id={bad} "
                       f"CPU={a.get(bad)} GPU={b.get(bad)}")
    if nx_cpu.nid != nx_gpu.nid:
        return False, f"{label} nid 不同 CPU={nx_cpu.nid} GPU={nx_gpu.nid}"
    return False, f"{label} 事件计数不同 CPU={cnt} GPU={gcnt}"


def _fx_tetra_csr():
    """夹具用的 tetra CSR（规范 spec3 §4 公共布局）。"""
    return (cp.asarray([0, 3, 6, 9, 12], dtype=cp.int32),
            cp.asarray([1, 3, 2, 0, 2, 3, 1, 0, 3, 0, 1, 2], dtype=cp.int32),
            cp.asarray([3, 3, 3, 3], dtype=cp.int32))


def verify_fixtures():
    """规范 spec3 §4 的 5 组最小夹具（期望值全部手工推导）。返回 (ok, rows)。"""
    rp, ri, dg = _fx_tetra_csr()
    base_ptr = [0, 3, 6, 9, 12]
    base_idx = [1, 3, 2, 0, 2, 3, 1, 0, 3, 0, 1, 2]
    z_pair = cp.zeros((0, 2), dtype=cp.int32)
    z_i = cp.zeros(0, dtype=cp.int32)
    cr_ptr, cr_idx, cr_len = (cp.asarray([0, 4], dtype=cp.int32),
                              cp.asarray([0, 1, 2], dtype=cp.int32),
                              cp.asarray([3], dtype=cp.int32))
    cases = [
        # 名称, (nid0, dead_pos, broken, cr_ptr, cr_idx, cr_len, dmin), 期望
        ("A 面锥化", (4, z_i, z_pair, cr_ptr, cr_idx, cr_len, 3),
         [0, 4, 8, 12, 15, 18],
         [1, 3, 2, 4, 0, 4, 2, 3, 1, 4, 0, 3, 0, 1, 2, 0, 2, 1],
         [4, 4, 4, 3, 3], 5),
        ("B 面锥化+断边(0,3)", (4, z_i, cp.asarray([[0, 3]], cp.int32),
                              cr_ptr, cr_idx, cr_len, 3),
         [0, 3, 7, 11, 13, 16],
         [1, 2, 4, 0, 4, 2, 3, 1, 4, 0, 3, 1, 2, 0, 2, 1],
         [3, 4, 4, 2, 3], 5),
        ("C 新元胞必删(dmin=4)", (4, z_i, z_pair, cr_ptr, cr_idx, cr_len, 4),
         base_ptr, base_idx, [3, 3, 3, 3], 5),
        ("D 旧节点3被删", (4, cp.asarray([3], cp.int32), z_pair,
                         cr_ptr, cr_idx, cr_len, 3),
         [0, 3, 6, 9, 12], [1, 2, 3, 0, 3, 2, 1, 3, 0, 0, 2, 1],
         [3, 3, 3, 3], 5),
        ("E 空输入边界", (7, z_i, z_pair, cp.asarray([0], cp.int32), z_i, z_i, 3),
         base_ptr, base_idx, [3, 3, 3, 3], 7),
    ]
    ok, rows = True, []
    for name, args, wptr, widx, wdeg, wnid in cases:
        got = rebuild_rings(rp, ri, dg, *args)
        gptr = cp.asnumpy(got[0]).tolist()
        gidx = cp.asnumpy(got[1]).tolist()
        gdeg = cp.asnumpy(got[2]).tolist()
        good = (gptr == wptr and gidx == widx and gdeg == wdeg
                and int(got[3]) == wnid)
        ok &= good
        rows.append(f"  [{'OK' if good else 'FAIL'}] 夹具 {name}"
                    + ("" if good else f"\n        ptr={gptr} 期望={wptr}"
                       f"\n        idx={gidx} 期望={widx}"
                       f"\n        deg={gdeg}/{wnid} 期望={wdeg}/{wnid}"))
    return ok, rows


def verify_frame_path(seed="icosa", nframes=4, cap=64, dmin=3, drive="slack",
                      pairing="none", label="", puncture=False,
                      vminus="dangling", vplus="any", kappa=6):
    """GPU 驱动帧 vs 纯 CPU 帧：逐帧 state_hash 必须相同。

    两个独立内核从同一种子出发，一个走 L0Core.frame()（纯 CPU），
    一个走 frame_gpu()（GPU 判定 + GPU 重建），逐帧比对状态键与事件计数。
    """
    from combinatorial_proto import SEEDS, RotNet
    from l0_core import L0Core

    mk = lambda: L0Core(RotNet(SEEDS[seed]()), cap=cap, dmin=dmin,
                        drive=drive, pairing=pairing,
                        vminus=vminus, vplus=vplus, kappa=kappa)
    a, b = mk(), mk()
    if puncture:
        if a.puncture() != b.puncture():
            return False, f"{label} 穿刺结果不同（不应发生）"
    for t in range(nframes):
        if a.state_hash() != b.state_hash():
            return False, f"{label} 第 {t} 帧起点不一致（不应发生）"
        ra = a.frame()
        rb = frame_gpu(b)
        if ra != rb:
            return False, f"{label} 第 {t + 1} 帧事件计数不同 CPU={ra} GPU={rb}"
        if a.state_hash() != b.state_hash():
            return False, f"{label} 第 {t + 1} 帧 state_hash 不同"
    return True, (f"{label} {nframes} 帧状态键逐帧一致"
                  f"（V={a.net.V()} E={a.net.E()} hash={a.state_hash()}）")


def verify_structural(g):
    """不变量自检（与 CPU 无关，纯结构）。返回 (是否全通过, 行列表)。"""
    nd = int(g["rot_ptr"][-1])
    d = cp.arange(nd, dtype=cp.int32)
    rows, ok = [], True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        rows.append(f"  [{'OK' if cond else 'FAIL'}] {name}")

    chk("rev[rev[d]] == d", cp.all(g["rev"][g["rev"]] == d))
    chk("face_id 沿 dn 不变", cp.all(g["face_id"][g["dn"]] == g["face_id"]))
    chk("rep 是环内最小 dart",
        cp.all(g["face_rep"][g["face_id"]] == cp.minimum(
            g["face_rep"][g["face_id"]],
            g["face_rep"][g["face_id"]][g["dn"]])))
    chk(f"Σ环长 == 2E（{int(g['fv_len'].sum())} == {nd}）",
        int(g["fv_len"].sum()) == nd)
    # 每条无向边的两个 dart 必须落在两个不同的面里（L0-A1）
    e0 = g["rev"][d]
    chk("无领结（rev 的 face_id 互异）", cp.all(g["face_id"] != g["face_id"][e0]))
    chk("Euler χ = V − E + F", int(g["ids"].size) - nd // 2 + g["n_faces"] == 2)
    return ok, rows


# ============================================================
# 自检
# ============================================================
def _seeds():
    """构造种子（**每次调用都新建**，绝不复用缓存对象）。

    注意：`RotNet(dict)` 只浅拷贝外层 dict，值（环序 list）是与入参共享的；
    而 `puncture()` / `remove_vertex` / `cone_*` 都会**原地改写**这些 list。
    若返回缓存对象，一次穿刺就会污染此后所有拿到的种子（实测：紧跟 punc=1 的
    配置会在 t=0 就报「0 is not in list」）。故此处逐行复制。
    """
    from combinatorial_proto import RotNet, SEEDS
    return {n: RotNet({v: list(r) for v, r in SEEDS[n]().items()})
            for n in ("tetra", "icosa")}


def selftest():
    from l0_core import L0Core

    print("=" * 78)
    print("[自检] L0 GPU 决策层：结构不变量 / 与 CPU 逐位一致 / 多帧漂移")
    print("=" * 78)
    ok = True

    for name, want in (("tetra", (4, 6, 4)), ("icosa", (12, 30, 20))):
        rot = _seeds()[name].rot
        g = gpu_decision(rot, cap=64, dmin=3, drive="slack")
        V, E, F = int(g["ids"].size), int(g["rot_ptr"][-1]) // 2, g["n_faces"]
        got = (V, E, F)
        flag = got == want
        ok &= flag
        print(f"  [{'OK' if flag else 'FAIL'}] {name:6s} (V,E,F)={got} 期望 {want} "
              f"| 三角={g['n_face']} 洞={g['n_hole']}")
        sok, rows = verify_structural(g)
        ok &= sok
        for r in rows:
            print("   " + r)

    # 与 CPU 逐位一致：多种子 × 多配置 × 多帧（帧推进以 CPU 为准，GPU 只核对决策段）
    print("-" * 78)
    for name in ("tetra", "icosa"):
        for drive in ("slack", "saturate"):
            for cap, dmin in ((64, 3), (12, 3), (5, 3)):
                core = L0Core(type(_seeds()[name])(_seeds()[name].rot),
                              cap=cap, dmin=dmin, drive=drive)
                tag = f"{name} cap={cap} drive={drive}"
                good, msg = verify_against_cpu(core, cap, dmin, drive, f"t=0 {tag}")
                ok &= good
                print(f"  [{'OK' if good else 'FAIL'}] {msg}")
                for t in range(1, 4):
                    core.frame()
                    good, msg = verify_against_cpu(core, cap, dmin, drive,
                                                   f"t={t} {tag}")
                    ok &= good
                    if not good:
                        print(f"  [FAIL] {msg}")
                        break
                if ok:
                    print(f"  [OK] {tag} 4 帧决策段全部逐位一致")

    # 第二增量：提交段判定（§3.4 消解 + §3.5 交集）与 CPU 逐位一致
    print("-" * 78)
    print("  第二增量：GPU 提交段判定（resolve + 交集） vs CPU resolve/commit 筛选")
    for name in ("tetra", "icosa"):
        for drive in ("slack", "saturate"):
            for cap, punc in ((64, False), (12, False), (12, True), (5, False)):
                core = L0Core(type(_seeds()[name])(_seeds()[name].rot),
                              cap=cap, dmin=3, drive=drive)
                if punc:
                    core.puncture()
                tag = f"{name} cap={cap} drive={drive} punc={int(punc)}"
                for t in range(4):
                    good, msg = verify_commit_against_cpu(
                        core, cap, 3, drive, f"t={t} {tag}")
                    ok &= good
                    if not good:
                        print(f"  [FAIL] {msg}")
                        break
                    core.frame()
                else:
                    print(f"  [OK] {tag} 4 帧提交集与顺序全部一致")

    # 第三增量：写操作（一次性 CSR 重建）—— 先过规范夹具，再与 CPU 逐位比对
    print("-" * 78)
    print("  第三增量：GPU 写操作（rebuild_rings） vs CPU _step")
    fok, rows = verify_fixtures()
    ok &= fok
    for r in rows:
        print(" " + r)
    for name in ("tetra", "icosa"):
        for drive in ("slack", "saturate"):
            for cap, dmin, pairing, punc in (
                    (64, 3, "none", False), (12, 3, "none", False),
                    (12, 3, "conserved", False), (12, 4, "none", False),
                    (5, 3, "none", False), (12, 3, "none", True)):
                core = L0Core(type(_seeds()[name])(_seeds()[name].rot),
                              cap=cap, dmin=dmin, drive=drive, pairing=pairing)
                if punc:
                    core.puncture()
                tag = f"{name} cap={cap} dmin={dmin} {drive}/{pairing} punc={int(punc)}"
                for t in range(4):
                    good, msg = verify_write_against_cpu(
                        core, cap, dmin, drive, f"t={t} {tag}")
                    ok &= good
                    if not good:
                        print(f"  [FAIL] {msg}")
                        break
                    core.frame()
                else:
                    print(f"  [OK] {tag} 4 帧写操作全部逐位一致")

    # 第四增量：采纳动力学（vminus=dense / vplus）搬上 GPU —— 端到端逐位一致
    print("-" * 78)
    print("  第四增量：GPU 采纳动力学（vminus=dense / vplus） vs 纯 CPU")
    for seed in ("tetra", "icosa"):
        for drive in ("slack", "saturate"):
            for cap, dmin, pairing, vminus, vplus in (
                    (12, 3, "none", "dense", "any"),
                    (12, 3, "conserved", "dense", "any"),
                    (12, 3, "none", "dense", "gap"),
                    (64, 3, "none", "dense", "any"),
                    (12, 4, "none", "dense", "any")):
                tag = (f"{seed} cap={cap} dmin={dmin} {drive}/{pairing} "
                       f"{vminus}/{vplus}")
                good, msg = verify_frame_path(
                    seed, 4, cap, dmin, drive, pairing, tag, puncture=False,
                    vminus=vminus, vplus=vplus)
                ok &= good
                print(f"  [{'OK' if good else 'FAIL'}] {msg}")

    # X1 守护（GPU 侧）：采纳动力学长跑，两路 maxdeg 必须恒 ≤ cap
    # （异常 X1 = cone_hole 锥化 k 边洞造出 deg=k 超容点；dense 会造出大洞）
    for cap, nf in ((12, 10), (64, 5)):
        s1, s2 = _seeds()["icosa"], _seeds()["icosa"]
        a = L0Core(s1, cap=cap, dmin=3, drive="slack", vminus="dense", vplus="any")
        b = L0Core(s2, cap=cap, dmin=3, drive="slack", vminus="dense", vplus="any")
        capok, det = True, True
        for _ in range(nf):
            a.frame()
            frame_gpu(b)
            capok &= max(a.degrees()) <= cap and max(b.degrees()) <= cap
            det &= a.state_hash() == b.state_hash()
        ok &= capok and det
        print(f"  [{'OK' if capok and det else 'FAIL'}] X1 守护 cap={cap} {nf} 帧："
              f"maxdeg≤cap（CPU∧GPU）={capok} | GPU≡CPU={det} | V={a.net.V()}")

    # 端到端：GPU 整帧（判定 + 重建） vs 纯 CPU 帧，逐帧状态键一致
    print("-" * 78)
    print("  端到端：GPU 判定 + GPU 重建（frame_gpu） vs 纯 CPU（L0Core.frame）")
    for seed in ("tetra", "icosa"):
        for drive in ("slack", "saturate"):
            for cap, dmin, pairing, punc in (
                    (64, 3, "none", False), (12, 3, "none", False),
                    (12, 3, "conserved", False), (12, 4, "none", False),
                    (12, 3, "none", True), (12, 3, "conserved", True)):
                tag = f"{seed} cap={cap} dmin={dmin} {drive}/{pairing} punc={int(punc)}"
                good, msg = verify_frame_path(seed, 4, cap, dmin, drive, pairing,
                                              tag, puncture=punc)
                ok &= good
                print(f"  [{'OK' if good else 'FAIL'}] {msg}")

    print("-" * 78)
    print(f"自检总体: {'全部通过' if ok else '存在失败项'}")
    return ok


def bench(seed="icosa", nframes=4, cap=64):
    """决策段 CPU / GPU 耗时对比（GPU 含传输，故小规模时不占优，属预期）。"""
    import time

    from l0_core import L0Core
    net = type(_seeds()[seed])(_seeds()[seed].rot)
    core = L0Core(net, cap=cap)
    for _ in range(nframes):
        core.frame()
    rot = core.net.rot

    t0 = time.perf_counter()
    for _ in range(20):
        core.propose()
    t_cpu = (time.perf_counter() - t0) / 20

    gpu_decision(rot, cap=cap)          # 预热（含驱动/JIT）
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        gpu_decision(rot, cap=cap)
    cp.cuda.Stream.null.synchronize()
    t_gpu = (time.perf_counter() - t0) / 20

    V, E = core.net.V(), core.net.E()
    print(f"[bench] seed={seed} {nframes} 帧后 V={V} E={E} 2E={2 * E}")
    print(f"  决策段 CPU propose(): {t_cpu * 1e3:8.3f} ms")
    print(f"  决策段 GPU 全链    : {t_gpu * 1e3:8.3f} ms  （含 H2D/D2H）")
    print(f"  倍率 GPU/CPU       : {t_gpu / t_cpu:8.2f}×")


def main():
    kv = {}
    for a in sys.argv[1:]:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k] = v
    if kv.get("selftest", "0") == "1":
        selftest()
    elif kv.get("bench", "0") == "1":
        bench(kv.get("seed", "icosa"), int(kv.get("nframes", 4)),
              int(kv.get("cap", 64)))
    elif kv.get("verify", "0") == "1":
        from l0_core import L0Core
        seed = kv.get("seed", "icosa")
        cap = int(kv.get("cap", 64))
        dmin = int(kv.get("dmin", 3))
        drive = kv.get("drive", "slack")
        vminus = kv.get("vminus", "dangling")
        vplus = kv.get("vplus", "any")
        kappa = int(kv.get("kappa", 6))
        core = L0Core(type(_seeds()[seed])(_seeds()[seed].rot),
                      cap=cap, dmin=dmin, drive=drive,
                      vminus=vminus, vplus=vplus, kappa=kappa)
        print("=" * 78)
        print(f"[GPU 决策层核验] seed={seed} cap={cap} dmin={dmin} drive={drive} "
              f"vminus={vminus} vplus={vplus} κ={kappa}")
        print("=" * 78)
        allok = True
        for t in range(int(kv.get("nframes", 4)) + 1):
            good, msg = verify_against_cpu(core, cap, dmin, drive, f"t={t}")
            allok &= good
            print(f"  [{'OK' if good else 'FAIL'}] {msg}")
            if t < int(kv.get("nframes", 4)):
                core.frame()
        print("-" * 78)
        print(f"核验结论: {'逐帧一致' if allok else '存在不一致'}")
    elif kv.get("commit", "0") == "1":
        from l0_core import L0Core
        seed = kv.get("seed", "icosa")
        cap = int(kv.get("cap", 12))
        dmin = int(kv.get("dmin", 3))
        drive = kv.get("drive", "slack")
        pairing = kv.get("pairing", "none")
        nframes = int(kv.get("nframes", 4))
        vminus = kv.get("vminus", "dangling")
        vplus = kv.get("vplus", "any")
        kappa = int(kv.get("kappa", 6))
        s = _seeds()[seed]
        core = L0Core(type(s)(s.rot), cap=cap, dmin=dmin, drive=drive,
                      pairing=pairing, vminus=vminus, vplus=vplus, kappa=kappa)
        print("=" * 78)
        print(f"[GPU 提交段核验] seed={seed} cap={cap} dmin={dmin} "
              f"drive={drive} pairing={pairing} vminus={vminus} vplus={vplus}")
        print("=" * 78)
        allok = True
        for t in range(nframes + 1):
            good, msg = verify_commit_against_cpu(core, cap, dmin, drive, f"t={t}")
            allok &= good
            print(f"  [{'OK' if good else 'FAIL'}] {msg}")
            if t < nframes:
                core.frame()
        good, msg = verify_frame_path(seed, nframes, cap, dmin, drive, pairing)
        allok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {msg}")
        print("-" * 78)
        print(f"核验结论: {'提交段与端到端均逐位一致' if allok else '存在不一致'}")
    elif kv.get("write", "0") == "1":
        from l0_core import L0Core
        seed = kv.get("seed", "icosa")
        cap = int(kv.get("cap", 12))
        dmin = int(kv.get("dmin", 3))
        drive = kv.get("drive", "slack")
        pairing = kv.get("pairing", "none")
        nframes = int(kv.get("nframes", 4))
        vminus = kv.get("vminus", "dangling")
        vplus = kv.get("vplus", "any")
        kappa = int(kv.get("kappa", 6))
        s = _seeds()[seed]
        core = L0Core(type(s)(s.rot), cap=cap, dmin=dmin, drive=drive,
                      pairing=pairing, vminus=vminus, vplus=vplus, kappa=kappa)
        print("=" * 78)
        print(f"[GPU 写操作段核验] seed={seed} cap={cap} dmin={dmin} "
              f"drive={drive} pairing={pairing} vminus={vminus} vplus={vplus}")
        print("=" * 78)
        allok = True
        fok, rows = verify_fixtures()
        allok &= fok
        for r in rows:
            print(" " + r)
        for t in range(nframes):
            good, msg = verify_write_against_cpu(core, cap, dmin, drive, f"t={t}")
            allok &= good
            print(f"  [{'OK' if good else 'FAIL'}] {msg}")
            core.frame()
        good, msg = verify_frame_path(seed, nframes, cap, dmin, drive, pairing)
        allok &= good
        print(f"  [{'OK' if good else 'FAIL'}] {msg}")
        print("-" * 78)
        print(f"核验结论: {'写操作段与端到端均逐位一致' if allok else '存在不一致'}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
