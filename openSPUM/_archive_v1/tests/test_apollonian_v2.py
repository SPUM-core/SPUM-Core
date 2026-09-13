"""Bootstrap: dense first shell, then let Apollonian step1b take over."""
import sys, os, time, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Phase_0.frame_kernels import _fibonacci_sphere, step3_volume, enforce_tangency, step1_create, step1b_gap_fill, step2_connect, step3b_enforce_impenetrability, step4_dangling, step5_purge, reincarnate_to_minimum
from Phase_0.particle_array import ParticleArray

p = ParticleArray(max_n=10000)

# Center + 12 seeds
p.add_particle((0,0,0), uid='cent_0000', degree=0, initial_volume=1.0, initial_degree=0)
dirs = _fibonacci_sphere(12)
for i in range(12):
    p.add_particle(tuple(dirs[i].astype(float)*1.0), uid=f'surf_{i:04d}', degree=0, initial_volume=1.0, initial_degree=0)
    p.add_connection(0, 1+i)

# Phase 1: bootstrap — dense first shell via 3-body gap detection
# Run step1_create with min_active=0 so it fills all detected gaps
print("Phase 1: Bootstrap dense shell...", flush=True)
for frame in range(1, 51):
    step3_volume(p)
    enforce_tangency(p)
    c1 = step1_create(p, max_checks=100000)  # 3-body gaps on surface
    c1b = step1b_gap_fill(p, max_per_frame=50)  # Apollonian (will be 0 until tangencies exist)
    c2 = step2_connect(p, star_mode=True)
    c3b = step3b_enforce_impenetrability(p, allow_disconnect=True)
    d = step4_dangling(p)
    dc = int(np.sum(d))
    c5 = step5_purge(p, d)
    
    if c1 == 0 and c1b > 0:
        print(f"  Frame {frame}: step1b activated! V={p.active_count()} Σ={p.spum_invariant()}", flush=True)
    
    if dc > 0:
        pass
    
    if p.active_count() > 500:
        print(f"  Bootstrap done at frame {frame}, V={p.active_count()}", flush=True)
        break
    if frame % 10 == 0:
        print(f"  Frame {frame}: V={p.active_count()} Σ={p.spum_invariant()} c1={c1} c1b={c1b}", flush=True)

print(f"\nBootstrap result: V={p.active_count()} Σ={p.spum_invariant()}", flush=True)

# Phase 2: full evolution
print("\nPhase 2: Evolution...", flush=True)
sigmas = [p.spum_invariant()]
for frame in range(1, 101):
    t0 = time.time()
    step3_volume(p)
    enforce_tangency(p)
    c1 = step1_create(p, max_checks=100000)
    c1b = step1b_gap_fill(p, max_per_frame=50)
    c2 = step2_connect(p, star_mode=True)
    c3b = step3b_enforce_impenetrability(p, allow_disconnect=True)
    d = step4_dangling(p)
    dc = int(np.sum(d))
    c5 = step5_purge(p, d)
    dt = time.time() - t0
    s = p.spum_invariant()
    sigmas.append(s)
    
    if frame <= 10 or frame % 20 == 0:
        print(f"F{frame:>2}: V={p.active_count():>3} c={c1+c1b:>2} e={c2:>3} ov={c3b:>3} d={dc:>2} Σ={s:>4} {dt:.2f}s", flush=True)
    
    if p.active_count() < 3:
        print("COLLAPSE", flush=True)
        break
    if p.active_count() > 1000:
        print(f"V=1000 at frame {frame}", flush=True)
        break

print(f"\nFinal: V={p.active_count()} Σ={p.spum_invariant()}")
print(f"Σ range: [{min(sigmas)}, {max(sigmas)}]")
