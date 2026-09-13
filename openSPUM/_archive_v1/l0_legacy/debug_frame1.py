"""分步调试帧 1：定位创生回滚环节。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import cupy as cp

from host import FrameScheduler
from smoke_test import tetrahedron_seed


def main():
    pos, edges = tetrahedron_seed()
    sch = FrameScheduler(capacity=1024)
    sch.spawn_initial(pos, edges, radius0=1.0)

    # K1
    sch._launch("k_update_volume", sch.capacity, ())
    cp.cuda.Device().synchronize()
    print("K1 后 radius:", sch.radius.get()[:4])

    # K2
    sch.cre_count.fill(0)
    sch._launch("k_detect_gaps", sch.capacity, ())
    cp.cuda.Device().synchronize()
    n_cre = int(sch.cre_count.get()[0])
    print(f"K2 后 cre_count = {n_cre}")
    if n_cre:
        reqs = sch.cre[:n_cre].get()
        for r in reqs:
            print(f"  req ({r['a']},{r['b']},{r['c']}) "
                  f"pos=({r['pos'][0]:.3f},{r['pos'][1]:.3f},{r['pos'][2]:.3f}) "
                  f"r={r['radius']:.3f}")

    # L1 去重
    dedup, new_idx = sch._dedup_creations()
    print(f"去重后 {len(dedup)} 个, new_idx={new_idx}")

    # K3
    if len(dedup) > 0:
        sch.cre[:len(dedup)] = cp.asarray(dedup)
        sch._launch("k_spawn", len(dedup),
                    (sch.cre.data.ptr, cp.asarray(new_idx).data.ptr,
                     np.int32(len(dedup))))
        cp.cuda.Device().synchronize()
        print(f"K3 后 n_active = {int(sch.n_active.get()[0])}")
        print(f"新粒子 active: {sch.active.get()[new_idx]}")
        print(f"原 4 粒子 deg: {sch.nb_count.get()[:4]}")
        print(f"新粒子 deg: {sch.nb_count.get()[new_idx]}")


if __name__ == "__main__":
    main()
