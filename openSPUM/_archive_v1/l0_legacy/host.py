"""
host.py — L1 层：帧调度器与仲裁簿记

架构基准: openSPUM/docs/L0L1L2_Architecture.md §5/§6/§8

职责（只做仲裁与簿记，不做物理决策）:
    1. 设备缓冲区管理（SoA 分配、潜在池空闲链表）
    2. 帧调度：K1..K7 kernel 序列 + 同步点
    3. 请求去重：创生三元组分组 + pos 聚类；连接二元组去重
       （去重后字典序执行，保证帧级确定性 U7）
    4. 坐标投影：nb_dir 角坐标 → pos 笛卡尔（BFS 重构，每帧末）
    5. 快照输出（供 L2 只读观测）

状态传递：ParticlesSoA / RequestQueues 经 __device__ 全局 g_p / g_q，
由 k_init_state / k_init_queues 在初始化时一次性写入指针表
（CuPy RawModule 无法按 ABI 值传结构体参数）。
"""

from pathlib import Path

import cupy as cp
import numpy as np

_KERNEL_DIR = Path(__file__).resolve().parent
_SPUM_MAX_NB = 64

# 悬挂判定阈值（文档 §7：三维三角剖分闭合下限）
DANGLING_DEGREE = 3

_CRE_DTYPE = np.dtype([
    ("a", np.int32), ("b", np.int32), ("c", np.int32),
    ("pos", np.float32, (3,)), ("radius", np.float32),
])
_EDGE_DTYPE = np.dtype([("i", np.int32), ("j", np.int32)])


def _gbk_safe(s: str) -> str:
    """CuPy 用系统默认编码（Windows 下为 GBK）写临时源文件，源码里的
    非 GBK 字符（如 ³ ⁺）会抛 UnicodeEncodeError。这些字符只出现在注释，
    直接降级为 '?' 即可，不影响编译语义。"""
    return s.encode("gbk", errors="replace").decode("gbk")


# 相邻接触角（Python 侧镜像，与 particle_state.cuh::contact_angle 同步）。
# 顶点 v（半径 rv）的两个邻居（半径 rn, rm）互不穿透所需的最小角距。
def _contact_angle_alpha(rv: float, rn: float, rm: float) -> float:
    return (np.arcsin(np.clip(rn / (rv + rn), -1.0, 1.0))
            + np.arcsin(np.clip(rm / (rv + rm), -1.0, 1.0)))


def _contact_angle_exact(rv: float, rn: float, rm: float) -> float:
    num = rv * rv + rv * (rn + rm) - rn * rm
    den = (rv + rn) * (rv + rm)
    return np.arccos(np.clip(num / den, -1.0, 1.0))


def _load_module(radius_slope: float | None = None,
                 exact_theta: bool = False,
                 dangling_min: int = DANGLING_DEGREE) -> cp.RawModule:
    # nvrtc 对中文 include 路径不兼容 → 宿主侧拼接 cuh + cu 为单一源码
    hdr = (_KERNEL_DIR / "particle_state.cuh").read_text(encoding="utf-8")
    cu = (_KERNEL_DIR / "frame_kernels.cu").read_text(encoding="utf-8")
    # 半径律杠杆：None → 单位球占位；给定斜率 → 单调递增占位律（实验用）
    # θ* 开关：定义 SPUM_EXACT_THETA → 排他性判据用精确接触角而非 α_j+α_k
    # 悬挂阈值：默认 3；种子探测可传 2（公理下限）以观察 deg-2 边界存活
    prefix = ""
    if radius_slope is not None:
        prefix += f"#define SPUM_RADIUS_MONO {float(radius_slope)!r}f\n"
    if exact_theta:
        prefix += "#define SPUM_EXACT_THETA\n"
    if int(dangling_min) != DANGLING_DEGREE:
        prefix += f"#define SPUM_DANGLING_MIN {int(dangling_min)}\n"
    src = _gbk_safe(prefix + hdr + "\n" + cu)
    return cp.RawModule(code=src, backend="nvrtc",
                        options=("-std=c++17", "--use_fast_math"))


class FrameScheduler:
    """L1 帧调度器。管理一个固定容量的粒子槽位池。"""

    def __init__(self, capacity: int = 65536, cre_cap: int = 1 << 16,
                 edge_cap: int = 1 << 16, radius_slope: float | None = None,
                 exact_theta: bool = False, dangling_min: int = DANGLING_DEGREE):
        self.capacity = capacity
        self.radius_slope = radius_slope
        self.exact_theta = exact_theta
        self.dangling_min = dangling_min
        self.mod = _load_module(radius_slope, exact_theta, dangling_min)

        f3 = cp.float32
        self.pos       = cp.zeros((capacity, 3), dtype=f3)
        self.radius    = cp.zeros(capacity, dtype=f3)
        self.nb_count  = cp.zeros(capacity, dtype=cp.int32)
        self.nb_idx    = cp.full((capacity, _SPUM_MAX_NB), -1, dtype=cp.int32)
        self.nb_dir    = cp.zeros((capacity, _SPUM_MAX_NB, 2), dtype=f3)
        self.active    = cp.zeros(capacity, dtype=cp.bool_)
        self.n_active  = cp.zeros(1, dtype=cp.int32)
        self.dangling  = cp.zeros(capacity, dtype=cp.bool_)
        self.mask      = cp.zeros((capacity, _SPUM_MAX_NB), dtype=cp.uint8)
        self.lock      = cp.zeros(capacity, dtype=cp.int32)
        self.dbg       = cp.zeros(20, dtype=cp.int32)   # K3 回滚归因计数

        # 请求队列
        self.cre = cp.zeros(cre_cap, dtype=_CRE_DTYPE)
        self.cre_count = cp.zeros(1, dtype=cp.int32)
        self.edge = cp.zeros(edge_cap, dtype=_EDGE_DTYPE)
        self.edge_count = cp.zeros(1, dtype=cp.int32)
        self.cre_cap = cre_cap
        self.edge_cap = edge_cap

        self.free_slots = list(range(capacity - 1, -1, -1))  # 潜在池
        self.frame_number = 0
        self.last_stats: dict = {}

        # 一次性写入设备全局指针表
        self._launch("k_init_state", 1, (
            self.pos.data.ptr, self.radius.data.ptr, self.nb_count.data.ptr,
            self.nb_idx.data.ptr, self.nb_dir.data.ptr, self.active.data.ptr,
            self.n_active.data.ptr, self.lock.data.ptr, np.int32(capacity)))
        self._launch("k_init_queues", 1, (
            self.cre.data.ptr, self.cre_count.data.ptr, np.int32(cre_cap),
            self.edge.data.ptr, self.edge_count.data.ptr, np.int32(edge_cap)))

    # ------------------------------------------------------------------
    def _launch(self, name: str, n_threads: int, args: tuple):
        if n_threads <= 0:
            return
        block = 256
        grid = (n_threads + block - 1) // block
        self.mod.get_function(name)((grid,), (block,), args)

    # ------------------------------------------------------------------
    # L1 初始化簿记（仅初始化路径允许全局视角）
    # ------------------------------------------------------------------
    def spawn_initial(self, positions: np.ndarray,
                      edges: list[tuple[int, int]], radius0: float = 1.0):
        """写入初始粒子与初始边。初始化是 L1 簿记，直接双方槽位互写。"""
        n = len(positions)
        assert n <= len(self.free_slots)
        idx = [self.free_slots.pop() for _ in range(n)]
        self.pos[idx] = cp.asarray(positions, dtype=cp.float32)
        self.radius[idx] = radius0
        self.active[idx] = True
        self.n_active += n

        nb_idx_h = self.nb_idx.get()
        nb_dir_h = self.nb_dir.get()
        nb_cnt_h = self.nb_count.get()
        pos_h = positions.astype(np.float64)
        for (a, b) in edges:
            ia, ib = idx[a], idx[b]
            for src, dst in ((ia, ib), (ib, ia)):
                dvec = pos_h[dst] - pos_h[src]
                dvec /= np.linalg.norm(dvec)
                theta = float(np.arccos(np.clip(dvec[2], -1, 1)))
                phi = float(np.arctan2(dvec[1], dvec[0]))
                s = nb_cnt_h[src]
                nb_idx_h[src, s] = dst
                nb_dir_h[src, s] = (theta, phi)
                nb_cnt_h[src] += 1
        self.nb_idx.set(nb_idx_h)
        self.nb_dir.set(nb_dir_h)
        self.nb_count.set(nb_cnt_h)
        return idx

    # ------------------------------------------------------------------
    # L1 去重与仲裁
    # ------------------------------------------------------------------
    def _dedup_creations(self, cluster_tol: float = 0.5):
        """创生请求去重（文档 §4.3）：
        1. 按排序三元组分组（同一缝隙被 i,j,k 三方各报一次，± 两侧）
        2. 组内按 pos 聚类（两侧对偶 = 两个独立缝隙位）
        3. 每簇取首份（确定性：组内顺序已按帧内入队序，外部排序后稳定）
        返回 (去重请求, 预分配新粒子索引)。
        """
        n = int(self.cre_count.get()[0])
        if n == 0:
            return np.empty(0, dtype=_CRE_DTYPE), np.empty(0, dtype=np.int32)
        reqs = self.cre[:n].get()
        order = np.lexsort((reqs["c"], reqs["b"], reqs["a"]))
        reqs = reqs[order]

        out: list[np.void] = []
        i0 = 0
        while i0 < len(reqs):
            # 同三元组分组
            i1 = i0
            while (i1 < len(reqs)
                   and reqs["a"][i1] == reqs["a"][i0]
                   and reqs["b"][i1] == reqs["b"][i0]
                   and reqs["c"][i1] == reqs["c"][i0]):
                i1 += 1
            group = reqs[i0:i1]
            # 组内 pos 聚类
            reps: list[int] = []
            for m in range(len(group)):
                p_m = group["pos"][m]
                dup = False
                for r_i in reps:
                    if np.linalg.norm(p_m - group["pos"][r_i]) < cluster_tol:
                        dup = True
                        break
                if not dup:
                    reps.append(m)
            for r_i in reps:
                out.append(group[r_i])
            i0 = i1

        if not out:
            return np.empty(0, dtype=_CRE_DTYPE), np.empty(0, dtype=np.int32)
        dedup = np.array(out, dtype=_CRE_DTYPE)
        n_new = min(len(dedup), len(self.free_slots))
        dedup = dedup[:n_new]
        new_idx = np.array([self.free_slots.pop() for _ in range(n_new)],
                           dtype=np.int32)
        return dedup, new_idx

    def _dedup_edges(self) -> np.ndarray:
        n = int(self.edge_count.get()[0])
        if n == 0:
            return np.empty(0, dtype=_EDGE_DTYPE)
        reqs = self.edge[:n].get()
        order = np.lexsort((reqs["j"], reqs["i"]))
        reqs = reqs[order]
        keep = np.ones(len(reqs), dtype=bool)
        keep[1:] = ~((reqs["i"][1:] == reqs["i"][:-1])
                     & (reqs["j"][1:] == reqs["j"][:-1]))
        return reqs[keep]

    # ------------------------------------------------------------------
    # L1 投影：nb_dir 角坐标 → pos 笛卡尔（BFS 重构，文档 §8）
    # ------------------------------------------------------------------
    def reproject_positions(self, anchor: int | None = None):
        from collections import deque

        active = self.active.get()
        nb_idx = self.nb_idx.get()
        nb_dir = self.nb_dir.get()
        nb_cnt = self.nb_count.get()
        radius = self.radius.get()
        pos = self.pos.get()

        ids = np.where(active)[0]
        if len(ids) == 0:
            return
        if anchor is None or not active[anchor]:
            anchor = int(ids[0])

        new_pos = {anchor: pos[anchor].astype(np.float64)}
        dq = deque([anchor])
        while dq:
            i = dq.popleft()
            for s in range(nb_cnt[i]):
                j = int(nb_idx[i, s])
                if j < 0 or not active[j] or j in new_pos:
                    continue
                th = float(nb_dir[i, s, 0]); ph = float(nb_dir[i, s, 1])
                d = float(radius[i]) + float(radius[j])
                new_pos[j] = new_pos[i] + d * np.array(
                    [np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                     np.cos(th)])
                dq.append(j)
        for i, v in new_pos.items():
            pos[i] = v.astype(np.float32)
        self.pos.set(pos)

    # ------------------------------------------------------------------
    # L1 纯角域邻球重排：`r=f(deg)` 与「排他性零违反」之间的桥
    #
    # 目标：**只改 nb_dir（球面角坐标），不动 pos**，使每个粒子球面上的邻居
    # 方向满足排他性——任意邻居对 (j,k) 的角距 ≥ α_j + α_k，其中
    #   α_j = asin(r_j / (r_i + r_j))   （邻居 j 的球在 i 球面上的角半径）
    #
    # 动机（§12.2–§12.3）：半径律 r=f(deg) 尚未推导，只知单调递增。一旦半径
    # 随度数变化，α 就变，而边建立时冻结的角位不跟随 → 排他性残余违例。
    # 本步负责把冻结的角位重新解成一组自洽角坐标，使半径律重新成为可实验的杠杆。
    #
    # 表示：每条边 e=(i,j) 只维护一个全局帧球面方向 u_e（单位向量）。
    #   i 行写 cart2sph(u_e)，j 行写 cart2sph(−u_e)
    #   ⇒ 互指 u_ij = −u_ji 由构造成立（偏差恒 0），无需额外校验。
    #
    # 求解方式（**不是**物理动力学）：把每个违反约束看成球面上的角度缺口，
    # 用迭代投影把 u_e、u_f 沿大圆各修正 deficit/2；累加归一化、迭代至收敛。
    #
    # 概念澄清（用户 2026-09-11）：这里**不是**"度数变大把别人推开"——那是
    # 背景/力驱动思维的残留。SPUM 无背景容器，邻居的相对角位由关系本身的一致
    # 性确定，不存在"需要被推开的东西"。本步只是在 L1 投影坐标里**重解一组自洽
    # 角坐标**，迭代是数值求解手段，不是本体动力学。
    # 程序终究是 L1 模拟：坐标背景无法根除，故此处把背景当"投影装置"用，
    # 而不让它反过来驱动设计（不发明力、不发明推动）。
    # ------------------------------------------------------------------
    def angular_realign(self, iters: int = 60, step: float = 0.3,
                        tol: float = 1e-4, verbose: bool = False) -> float:
        act = self.active.get()
        nb_idx = self.nb_idx.get()
        nb_cnt = self.nb_count.get()
        rad = self.radius.get().astype(np.float64)
        nb_dir = self.nb_dir.get()

        ids = np.where(act)[0]
        n = len(ids)
        if n < 2:
            return 0.0
        lid = np.full(act.shape[0], -1, dtype=np.int64)
        lid[ids] = np.arange(n)
        r = rad[ids]
        cnt = nb_cnt[ids].astype(np.int64)
        nbr = nb_idx[ids]

        # 边表（局部 a<b）+ 两端各自槽位
        ekey: dict[tuple[int, int], int] = {}
        ei, ej, si, sj = [], [], [], []
        for a in range(n):
            for s in range(int(cnt[a])):
                b = int(nbr[a, s])
                if b < 0 or not act[b]:
                    continue
                lb = int(lid[b])
                if lb < 0:
                    continue
                key = (a, lb) if a < lb else (lb, a)
                e = ekey.get(key)
                if e is None:
                    e = len(ei)
                    ekey[key] = e
                    ei.append(key[0]); ej.append(key[1])
                    si.append(-1); sj.append(-1)
                if a == key[0]:
                    si[e] = s
                else:
                    sj[e] = s
        if not ei:
            return 0.0
        ei = np.asarray(ei); ej = np.asarray(ej)
        si = np.asarray(si); sj = np.asarray(sj)

        def _sph2cart(t, p):
            st = np.sin(t)
            return np.stack([st * np.cos(p), st * np.sin(p), np.cos(t)], axis=1)

        # 种子方向：两端已存角位取平均（一致时 u_ij − u_ji = 2·u_ij）
        u_i = _sph2cart(nb_dir[ids[ei], si, 0].astype(np.float64),
                        nb_dir[ids[ei], si, 1].astype(np.float64))
        u_j = _sph2cart(nb_dir[ids[ej], sj, 0].astype(np.float64),
                        nb_dir[ids[ej], sj, 1].astype(np.float64))
        U = u_i - u_j
        nrm = np.linalg.norm(U, axis=1)
        U[nrm < 1e-6] = u_i[nrm < 1e-6]
        nrm = np.linalg.norm(U, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        U = U / nrm

        # 每个粒子的邻居对约束
        # 注意：在顶点 v 上，边 e 的实际方向是 s·U[e]（s=+1 若 v 是 ei，−1 若是 ej）
        inc: list[list[tuple[int, float, float]]] = [[] for _ in range(n)]
        for e in range(len(ei)):
            inc[int(ei[e])].append((e, float(r[ej[e]]), 1.0))
            inc[int(ej[e])].append((e, float(r[ei[e]]), -1.0))
        pe, pf, req, se, sf = [], [], [], [], []
        _req_fn = _contact_angle_exact if self.exact_theta else _contact_angle_alpha
        for v in range(n):
            lst = inc[v]
            L = len(lst)
            if L < 2:
                continue
            rv = float(r[v])
            for p in range(L):
                for q in range(p + 1, L):
                    pe.append(lst[p][0]); pf.append(lst[q][0])
                    se.append(lst[p][2]); sf.append(lst[q][2])
                    req.append(_req_fn(rv, lst[p][1], lst[q][1]))
        if not pe:
            return 0.0
        pe = np.asarray(pe); pf = np.asarray(pf)
        se = np.asarray(se, dtype=np.float64)
        sf = np.asarray(sf, dtype=np.float64)
        req = np.asarray(req)

        acc = np.zeros_like(U)
        cnt_e = np.zeros(len(U))
        resid = 0.0
        for _ in range(iters):
            a = se[:, None] * U[pe]
            b = sf[:, None] * U[pf]
            cr = np.cross(a, b)
            nn = np.linalg.norm(cr, axis=1)
            ang = np.arccos(np.clip((a * b).sum(1), -1.0, 1.0))
            deficit = req - ang
            resid = float(deficit.max()) if len(deficit) else 0.0
            m = (deficit > tol) & (nn > 1e-9)
            if not m.any():
                break
            nh = cr[m] / nn[m][:, None]
            aa = a[m]; bb = b[m]
            dl = np.minimum(step * 0.5 * deficit[m], 0.3)[:, None]
            cd = np.cos(dl); sd = np.sin(dl)
            da = aa * (cd - 1.0) - np.cross(nh, aa) * sd
            db = bb * (cd - 1.0) + np.cross(nh, bb) * sd
            acc[:] = 0.0
            cnt_e[:] = 0.0
            np.add.at(acc, pe[m], se[m][:, None] * da)
            np.add.at(acc, pf[m], sf[m][:, None] * db)
            np.add.at(cnt_e, pe[m], 1.0)
            np.add.at(cnt_e, pf[m], 1.0)
            U = U + acc / np.maximum(cnt_e, 1.0)[:, None]
            un = np.linalg.norm(U, axis=1, keepdims=True)
            un[un < 1e-12] = 1.0
            U = U / un

        if verbose:
            print(f"    [angular_realign] 最大角亏残差 = {resid:.4f} rad")

        # 回写：i 行 = u_e，j 行 = −u_e
        th_i = np.arccos(np.clip(U[:, 2], -1.0, 1.0))
        ph_i = np.arctan2(U[:, 1], U[:, 0])
        N = -U
        th_j = np.arccos(np.clip(N[:, 2], -1.0, 1.0))
        ph_j = np.arctan2(N[:, 1], N[:, 0])
        gbi = ids[ei]; gbj = ids[ej]
        gi = si >= 0; gj = sj >= 0
        nb_dir[gbi[gi], si[gi], 0] = th_i[gi].astype(np.float32)
        nb_dir[gbi[gi], si[gi], 1] = ph_i[gi].astype(np.float32)
        nb_dir[gbj[gj], sj[gj], 0] = th_j[gj].astype(np.float32)
        nb_dir[gbj[gj], sj[gj], 1] = ph_j[gj].astype(np.float32)
        self.nb_dir.set(nb_dir)
        return resid

    # ------------------------------------------------------------------
    # L1 帧末全局重排（方案 C）：几何自洽优先于"纯局部"
    #
    # V = 1 + deg 使几何半径随度数增长。边一旦建立，其两条 nb_dir 是**冻结**
    # 的局部角坐标；当邻居半径增长后，冻结角距不再满足 α_j+α_k，排他性
    # 必然残余违例（V2-U5）。K5b 只写己行的滑动无法消解——它会破坏互指
    # 一致性（u_ij ≠ −u_ji）。
    #
    # 本步在帧末（拓扑确定后）求一个全局嵌入 p 满足：
    #   (1) 每条边相切：|p_i − p_j| = r_i + r_j
    #   (2) 任意两粒子不重叠：|p_i − p_j| ≥ r_i + r_j
    # 再由 p 反推每行的 nb_dir（同一全局帧 ⇒ 两行天然互为反向量）。
    #
    # 数学依据：若 |p_j − p_k| = r_j + r_k（j,k 同为 i 的邻居且相切），则
    # 由余弦定理 ∠jik (i 球面上的角距) 恒等于 α_j + α_k。故 (2) 成立
    # ⇒ 排他性零违反（阈值处恰好相等）。
    #
    # 这是 L1 的全局操作，违背"每单元只关心自己状态"，是用户明确选择的
    # 本体权衡：保留 r=f(deg) 的几何语义。
    # ------------------------------------------------------------------
    def global_realign(self, iters: int = 300, step: float = 0.5,
                       tol: float = 1e-4):
        act = self.active.get()
        nb_idx = self.nb_idx.get()
        nb_cnt = self.nb_count.get()
        rad = self.radius.get().astype(np.float64)
        pos = self.pos.get().astype(np.float64)

        ids = np.where(act)[0]
        n = len(ids)
        if n < 2:
            return
        lid = np.full(act.shape[0], -1, dtype=np.int64)
        lid[ids] = np.arange(n)
        r = rad[ids]
        p = pos[ids].copy()
        cnt = nb_cnt[ids].astype(np.int64)
        nbr = nb_idx[ids]

        # 边表（局部索引，a<b 去重）
        ea_l, eb_l = [], []
        for a in range(n):
            for s in range(int(cnt[a])):
                b = int(nbr[a, s])
                if b < 0 or not act[b]:
                    continue
                lb = int(lid[b])
                if lb < 0 or lb <= a:
                    continue
                ea_l.append(a)
                eb_l.append(lb)
        ea = np.asarray(ea_l, dtype=np.int64)
        eb = np.asarray(eb_l, dtype=np.int64)
        rest = r[ea] + r[eb] if len(ea) else np.empty(0)

        for _ in range(iters):
            # (1) 边相切：把边长拉回 r_i + r_j
            if len(ea):
                d = p[ea] - p[eb]
                L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
                corr = ((L - rest) / L * (step * 0.5))[:, None] * d
                np.add.at(p, ea, -corr)
                np.add.at(p, eb, corr)
            # (2) 全对不重叠：重叠对沿连线推开
            d2 = p[:, None, :] - p[None, :, :]
            L2 = np.sqrt((d2 * d2).sum(axis=2))
            np.fill_diagonal(L2, np.inf)
            mind = r[:, None] + r[None, :]
            viol = L2 < mind
            if viol.any():
                c2 = np.where(
                    viol,
                    (L2 - mind) / np.where(viol, L2, 1.0) * (step * 0.5),
                    0.0)
                p -= (c2[:, :, None] * d2).sum(axis=1)
            # 收敛判据：边长方差 + 最大重叠量
            e_err = 0.0
            if len(ea):
                e_err = float(np.max(np.abs(
                    np.linalg.norm(p[ea] - p[eb], axis=1) - rest)))
            o_err = 0.0
            if viol.any():
                o_err = float(np.max(np.where(viol, mind - L2, 0.0)))
            if e_err < tol and o_err < tol:
                break

        # 由全局嵌入反推 nb_dir（全局帧 ⇒ 两行互为反向量）
        new_dir = self.nb_dir.get()
        new_pos = pos.copy()
        new_pos[ids] = p
        for a in range(n):
            i = int(ids[a])
            for s in range(int(cnt[a])):
                j = int(nbr[a, s])
                if j < 0:
                    continue
                lj = int(lid[j])
                if lj < 0:
                    continue
                v = p[lj] - p[a]
                nv = float(np.linalg.norm(v))
                if nv < 1e-12:
                    continue
                theta = float(np.arccos(np.clip(v[2] / nv, -1, 1)))
                phi = float(np.arctan2(v[1], v[0]))
                new_dir[i, s] = (theta, phi)
        self.pos.set(new_pos.astype(np.float32))
        self.nb_dir.set(new_dir)

    # ------------------------------------------------------------------
    # 帧调度（文档 §5 的 kernel 序列）
    # ------------------------------------------------------------------
    def run_frame(self, slide_iters: int = 0, reproject: bool = True,
                  realign_iters: int = 0, angular_iters: int = 0):
        if int(self.n_active.get()[0]) == 0:
            return

        # K1 变化体积
        self._launch("k_update_volume", self.capacity, ())

        # K2 缝隙检测 → 创生请求
        self.cre_count.fill(0)
        self._launch("k_detect_gaps", self.capacity, ())
        n_cre_raw = int(self.cre_count.get()[0])

        # L1 去重 + 预分配 → K3 创生写入
        reqs, new_idx = self._dedup_creations()
        self.dbg.fill(0)
        if len(reqs) > 0:
            self.cre[:len(reqs)] = cp.asarray(reqs)
            self._launch("k_spawn", len(reqs),
                         (self.cre.data.ptr, cp.asarray(new_idx).data.ptr,
                          np.int32(len(reqs)), self.dbg.data.ptr))

        # K4 二度邻居相切 → 连接请求
        self.edge_count.fill(0)
        self._launch("k_connect_existing", self.capacity, ())

        # L1 去重 → K5 写边
        ereqs = self._dedup_edges()
        if len(ereqs) > 0:
            self.edge[:len(ereqs)] = cp.asarray(ereqs)
            self._launch("k_commit_edges", len(ereqs),
                         (self.edge.data.ptr, np.int32(len(ereqs))))

        self.last_stats = {"cre_raw": n_cre_raw, "cre": len(reqs),
                           "edge_raw": int(self.edge_count.get()[0]),
                           "edge": len(ereqs),
                           "dbg": self.dbg.get().tolist()}

        # K5b 不可入性球面滑动
        self._launch("k_impenetrability", self.capacity,
                     (np.int32(slide_iters),))

        # K6 悬挂标记 → K7 删除（只删一层，无级联）
        self._launch("k_mark_dangling", self.capacity,
                     (self.dangling.data.ptr,))
        self._launch("k_purge", self.capacity, (self.dangling.data.ptr,))

        # K8 关系对偶修复（帧内并发写边的残迹集中簿记）：
        # K8a 全行只读扫对偶性 → 掩码；K8b 独占压缩己行 + 尾部清零
        self._launch("k_check_mutuality", self.capacity,
                     (self.mask.data.ptr,))
        self._launch("k_compact_mask", self.capacity,
                     (self.mask.data.ptr,))

        # L1 回收失活槽位到潜在池
        act = self.active.get()
        in_pool = set(self.free_slots)
        for i in np.where(~act)[0]:
            if int(i) not in in_pool:
                self.free_slots.append(int(i))

        # L1 帧末先做纯角域邻球重排（若启用）：只改 nb_dir，解开冻结角位
        if angular_iters > 0:
            self.angular_realign(iters=angular_iters, verbose=True)

        # L1 帧末全局重排（方案 C）：求满足「边相切 + 全对不重叠」的全局
        # 嵌入，并反推 nb_dir。取代旧的 BFS 投影重构——pos 成为一致嵌入，
        # 下一帧 K4 相切检查与 L2 观测都基于它。
        # realign_iters=0 → 退回旧 BFS 投影（仅用于对照实验，隔离重排的影响）
        if reproject:
            if realign_iters > 0:
                self.global_realign(iters=realign_iters)
            else:
                self.reproject_positions()

        self.frame_number += 1

    def run(self, n_frames: int, **kw):
        for _ in range(n_frames):
            self.run_frame(**kw)

    # ------------------------------------------------------------------
    # 快照（供 L2 只读观测）
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        act = self.active.get()
        return {
            "frame": self.frame_number,
            "V": int(self.n_active.get()[0]),
            "active": act,
            "pos": self.pos.get()[act],
            "radius": self.radius.get()[act],
            "nb_count": self.nb_count.get()[act],
            "nb_idx": self.nb_idx.get()[act],
            "nb_dir": self.nb_dir.get()[act],
        }
