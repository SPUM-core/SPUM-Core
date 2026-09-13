"""最小复现：校验 nb_dir 数据 + Python 复刻 K2 公式。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from host import FrameScheduler
from smoke_test import tetrahedron_seed


def sph2cart(theta, phi):
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


def main():
    pos, edges = tetrahedron_seed()
    sch = FrameScheduler(capacity=64)
    sch.spawn_initial(pos, edges, radius0=1.0)

    nb_idx = sch.nb_idx.get()
    nb_dir = sch.nb_dir.get()
    print("粒子 0 的邻居槽位:", nb_idx[0, :3])
    print("粒子 0 的 nb_dir:", nb_dir[0, :3])

    # 理论方向（从 0 看 1,2,3）
    for j in range(1, 4):
        dvec = pos[j] - pos[0]
        dvec /= np.linalg.norm(dvec)
        print(f"  理论 dir(0->{j}) = {dvec}")

    # Python 复刻 K2：i=0, 邻居对 (1,2)
    vj = sph2cart(*nb_dir[0, 0])
    vk = sph2cart(*nb_dir[0, 1])
    print("sph2cart(nb_dir[0,0]) =", vj)
    print("sph2cart(nb_dir[0,1]) =", vk)

    r_i = 1.0
    r_new = r_i
    a_new = np.arcsin(r_new / (r_i + r_new))
    a_j = a_k = np.arcsin(1.0 / 2.0)
    a = a_j + a_new
    b = a_k + a_new
    cv = np.clip(vj @ vk, -1, 1)
    d_jk = np.arccos(cv)
    print(f"d_jk={np.degrees(d_jk):.2f}°  a=b={np.degrees(a):.2f}°")
    ca, cb = np.cos(a), np.cos(b)
    denom = 1 - cv * cv
    x = (ca - cv * cb) / denom
    y = (cb - cv * ca) / denom
    w2 = x * x + y * y + 2 * x * y * cv
    z2 = 1 - w2
    z = np.sqrt(max(z2, 0))
    nx = np.cross(vj, vk)
    nx /= np.linalg.norm(nx)
    base = x * vj + y * vk
    for sign in (+1, -1):
        vn = base + sign * z * nx
        p_new = pos[0] + (r_i + r_new) * vn
        print(f"  sign={sign:+d} vn={vn.round(4)} |vn|={np.linalg.norm(vn):.4f}"
              f" pos={p_new.round(4)}")


if __name__ == "__main__":
    main()
