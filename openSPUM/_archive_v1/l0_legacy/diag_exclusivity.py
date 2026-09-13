"""排他性违例归因诊断：区分 L0 设备侧(nb_dir)违例 与 L1 投影(pos)漂移。

逐帧输出：
  - dev_viol: 用 nb_dir 角坐标直接计算的排他性违例数（L0 真实状态）
  - pos_viol: 用 pos 投影坐标计算的违例数（含 BFS 重构漂移）
  - overlap:  球体空间重叠对数（|pi-pj| < (ri+rj)(1-tol) 的邻居对）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from host import (FrameScheduler, _contact_angle_alpha, _contact_angle_exact)


def tetra_seed():
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                 dtype=np.float64)
    v /= np.linalg.norm(v[0])
    v *= np.sqrt(6.0) / 2.0
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return v, edges


def count_viol(sch, tol=0.02, dump=False, exact=False):
    act = sch.active.get()
    nb_idx = sch.nb_idx.get()
    nb_cnt = sch.nb_count.get()
    nb_dir = sch.nb_dir.get()
    rad = sch.radius.get()
    pos = sch.pos.get()

    dev_viol = pos_viol = overlap = 0
    worst = 1.0
    gap_max = 0.0        # max(α_j+α_k − θ*) 全图，量化两判据的真实差异
    for i in np.where(act)[0]:
        n = nb_cnt[i]
        ri = rad[i]
        # 设备侧：nb_dir 角坐标
        dirs = []
        pdirs = []
        for s in range(n):
            j = int(nb_idx[i, s])
            if j < 0 or not act[j]:
                continue
            dirs.append((nb_dir[i, s], rad[j], j))
            dvec = pos[j] - pos[i]
            nrm = np.linalg.norm(dvec)
            if nrm > 1e-9:
                pdirs.append(dvec / nrm)
            if nrm < (ri + rad[j]) * (1 - tol):
                overlap += 1
        for a in range(len(dirs)):
            for b in range(a + 1, len(dirs)):
                da, ra, ja = dirs[a]
                db, rb, jb = dirs[b]
                cosd = (np.sin(da[0]) * np.sin(db[0]) * np.cos(da[1] - db[1])
                        + np.cos(da[0]) * np.cos(db[0]))
                dist = np.arccos(np.clip(cosd, -1, 1))
                req = (_contact_angle_exact(ri, ra, rb) if exact
                       else _contact_angle_alpha(ri, ra, rb))
                gap_max = max(gap_max,
                              np.degrees(_contact_angle_alpha(ri, ra, rb)
                                         - _contact_angle_exact(ri, ra, rb)))
                ratio = dist / req
                worst = min(worst, ratio)
                if dist < req * (1 - tol):
                    dev_viol += 1
                    if dump:
                        print(f"  违例: i={i}(deg={n},r={ri:.3f}) 邻居对 "
                              f"({ja},r={ra:.3f})-({jb},r={rb:.3f}) "
                              f"角距={np.degrees(dist):.2f}° "
                              f"需求={np.degrees(req):.2f}° 比值={ratio:.3f}")
        for a in range(len(pdirs)):
            for b in range(a + 1, len(pdirs)):
                cosd = np.clip(pdirs[a] @ pdirs[b], -1, 1)
                dist = np.arccos(cosd)
                # pos 侧无法精确取 α（半径属于对方球面），用等大近似 α=π/6
                if dist < (np.pi / 6) * 2 * (1 - 0.15):
                    pos_viol += 1
    return dev_viol, pos_viol, overlap // 2, worst, gap_max


def antipodality_error(sch):
    """每条边 (i,j) 的方向应满足 u_ij = -u_ji（同一边从两端看到的方向相反）。
    返回 (最大偏差角°, 平均偏差角°)。偏差大 ⇒ 各粒子局部角坐标系已漂移失配。"""
    act = sch.active.get()
    nb_idx = sch.nb_idx.get()
    nb_cnt = sch.nb_count.get()
    nb_dir = sch.nb_dir.get()

    def vec(th, ph):
        return np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])

    errs = []
    for i in np.where(act)[0]:
        for s in range(nb_cnt[i]):
            j = int(nb_idx[i, s])
            if j < 0 or not act[j] or j <= i:
                continue
            # 在 j 的行里找 i
            for t in range(nb_cnt[j]):
                if int(nb_idx[j, t]) == i:
                    u = vec(nb_dir[i, s, 0], nb_dir[i, s, 1])
                    v = vec(nb_dir[j, t, 0], nb_dir[j, t, 1])
                    c = np.clip(-(u @ v), -1, 1)     # 理想 u·v = -1
                    errs.append(np.degrees(np.arccos(c)))
                    break
    if not errs:
        return 0.0, 0.0
    return float(np.max(errs)), float(np.mean(errs))


def main():
    import sys as _sys
    slide = int(_sys.argv[1]) if len(_sys.argv) > 1 else 0
    nframes = int(_sys.argv[2]) if len(_sys.argv) > 2 else 8
    realign = int(_sys.argv[3]) if len(_sys.argv) > 3 else 0
    angular = int(_sys.argv[4]) if len(_sys.argv) > 4 else 0
    slope = float(_sys.argv[5]) if len(_sys.argv) > 5 else None
    exact = bool(int(_sys.argv[6])) if len(_sys.argv) > 6 else False
    pos, edges = tetra_seed()
    sch = FrameScheduler(capacity=4096, radius_slope=slope, exact_theta=exact)
    sch.spawn_initial(pos, edges, radius0=1.0)
    law = "r≡1(单位球)" if slope is None else f"r=1+{slope}·deg"
    crit = "θ*(精确)" if exact else "α_j+α_k(近似)"
    print(f"[slide_iters={slide} realign_iters={realign} angular_iters={angular} "
          f"半径律={law} 判据={crit}]")
    print(f"{'帧':>3} {'V':>5} {'degmax':>7} {'dev违例':>8} {'pos违例':>8} "
          f"{'重叠对':>6} {'最差比值':>8} {'判据差°':>8} {'互指偏差°':>9} "
          f"{'缝隙原/去重':>11} {'连接原/去重':>11} {'K3成功/回滚':>11}")
    for f in range(nframes):
        sch.run_frame(slide_iters=slide, realign_iters=realign,
                      angular_iters=angular)
        dv, pv, ov, worst, gap = count_viol(sch, dump=(sch.frame_number == 6),
                                            exact=exact)
        act = sch.active.get()
        degmax = int(sch.nb_count.get()[act].max()) if act.any() else 0
        amax, amean = antipodality_error(sch)
        st = sch.last_stats
        d = st["dbg"]
        # d[t*5+reason]: t=0..2, reason 1..5；d[16]=成功 d[17]=回滚
        r_break = " | ".join(
            f"t{t}:活{d[t*5+1]},缺{d[t*5+2]},解{d[t*5+3]},斥{d[t*5+4]},槽{d[t*5+5]}"
            for t in range(3))
        print(f"{sch.frame_number:>3} {int(sch.n_active.get()[0]):>5} "
              f"{degmax:>7} "
              f"{dv:>8} {pv:>8} {ov:>6} {worst:>8.3f} {gap:>8.3f} "
              f"{amax:>4.1f}/{amean:<4.1f} "
              f"{st['cre_raw']:>5}/{st['cre']:<5} "
              f"{st['edge_raw']:>5}/{st['edge']:<5} "
              f"{d[16]:>5}/{d[17]:<5}  斥总{d[19]}可换{d[18]}  {r_break}")


if __name__ == "__main__":
    main()
