"""
冒烟测试：L0/L1 新架构最小验证

场景：正四面体种子（4 个等大球互切，K4 完全图），跑 3 帧。

预期（手工推导）:
    帧 0: K2 每粒子 3 对相切邻居 → 每对 1 个空侧洞位
          → 4 个面各 1 个 A 类缝隙 → 创生 4 球（deg=3 诞生）
          → V: 4 → 8，无悬挂
    后续帧: 表面继续生长，V 单调不减
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from host import FrameScheduler


def tetrahedron_seed():
    """4 个半径 1 的等大球互切（正四面体，中心距 = 2）。
    边长 a=2 的正四面体外接球半径 R = a√6/4 = √6/2 ≈ 1.2247。"""
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                 dtype=np.float64)
    v /= np.linalg.norm(v[0])          # 归一化到单位球
    v *= np.sqrt(6.0) / 2.0            # 外接球半径 → 中心距恰好 2
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # K4
    return v, edges


def main():
    pos, edges = tetrahedron_seed()
    sch = FrameScheduler(capacity=1024)
    idx = sch.spawn_initial(pos, edges, radius0=1.0)
    print(f"种子: idx={idx}  V={int(sch.n_active.get()[0])}")

    v_traj = []
    for f in range(3):
        sch.run_frame()
        snap = sch.snapshot()
        v_traj.append(snap["V"])
        deg = snap["nb_count"]
        n_dangling = int(sch.dangling.get().sum())
        print(f"帧 {snap['frame']}: V={snap['V']}  "
              f"deg 分布 min={deg.min() if len(deg) else '-'} "
              f"max={deg.max() if len(deg) else '-'} "
              f"mean={deg.mean():.2f}  悬挂标记={n_dangling}")

    # 一致性快检：邻接互指（i→j 蕴含 j 活性 且 j→i）
    act = snap["active"]
    nb_idx = sch.nb_idx.get()
    nb_cnt = sch.nb_count.get()
    bad_inactive = 0   # 指向失活粒子
    bad_oneway = 0     # j 活性但 j 的行不含 i
    for i in np.where(act)[0]:
        for s in range(nb_cnt[i]):
            j = int(nb_idx[i, s])
            if j < 0 or not act[j]:
                bad_inactive += 1
                continue
            found = False
            for t in range(nb_cnt[j]):
                if int(nb_idx[j, t]) == i:
                    found = True
                    break
            if not found:
                bad_oneway += 1
    bad = bad_inactive + bad_oneway
    print(f"邻接互指违例: {bad}  (指向失活={bad_inactive}, 单向边={bad_oneway})")
    print("PASS" if v_traj[0] == 8 and bad == 0 else "CHECK")


if __name__ == "__main__":
    main()
