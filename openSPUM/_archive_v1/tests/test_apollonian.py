"""Test Apollonian step1b_gap_fill: center + 12 ring → recursive filling."""
import sys, os, time, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Phase_0.frame_kernels import _fibonacci_sphere, step3_volume, enforce_tangency, step1_create, step1b_gap_fill, step2_connect, step3b_enforce_impenetrability, step4_dangling, step5_purge, reincarnate_to_minimum
from Phase_0.particle_array import ParticleArray

p = ParticleArray(max_n=5000)

p.add_particle((0,0,0), uid='cent_0000', degree=0, initial_volume=1.0, initial_degree=0)
dirs = _fibonacci_sphere(12)
for i in range(12):
    p.add_particle(tuple(dirs[i].astype(float)*1.0), uid=f'surf_{i:04d}', degree=0, initial_volume=1.0, initial_degree=0)
    p.add_connection(0, 1+i)
    if i > 0:
        p.add_connection(1+i, 1+(i-1))
p.add_connection(1, 12)

print(f"Init: V={p.active_count()} Σ={p.spum_invariant()}", flush=True)

sigmas = []
for frame in range(1, 101):
    t0 = time.time()
    step3_volume(p)
    enforce_tangency(p)
    c1 = step1_create(p, max_checks=100000)
    c1b = step1b_gap_fill(p, max_per_frame=30)
    c2 = step2_connect(p, star_mode=True)
    c3b = step3b_enforce_impenetrability(p, allow_disconnect=True)
    d = step4_dangling(p)
    dc = int(np.sum(d))
    c5 = step5_purge(p, d)
    rn = reincarnate_to_minimum(p, min_active=0)
    dt = time.time() - t0
    s = p.spum_invariant()
    sigmas.append(s)
    
    if frame <= 20 or frame % 10 == 0:
        print(f"F{frame:>2}: V={p.active_count():>3} c={c1+c1b:>2} e={c2:>3} ov={c3b:>3} d={dc:>2} p={c5:>2} Σ={s:>4} {dt:.2f}s", flush=True)
    
    if p.active_count() < 3:
        print("COLLAPSE", flush=True)
        break
    if p.active_count() > 500:
        print(f"Reached V=500 at frame {frame}", flush=True)
        break

print(f"\nFinal: V={p.active_count()} Σ={p.spum_invariant()}")
print(f"Σ range: [{min(sigmas)}, {max(sigmas)}] mean={np.mean(sigmas):.0f}")
