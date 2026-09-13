"""Test PyTorch GPU frame kernels vs CPU."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch

from Phase_0.particle_array import ParticleArray
from Phase_0.frame_kernels import (
    _fibonacci_sphere, _find_center, run_full_frame,
    step3_volume, step2_connect, enforce_tangency,
    step4_dangling, step5_purge, step1b_gap_fill,
    step3b_enforce_impenetrability, reincarnate_to_minimum
)
from Phase_0.gpu_frame_kernels import (
    torch_step3_volume, torch_surface_adjacency,
    torch_detect_overlaps, torch_enforce_tangency,
    torch_run_full_frame
)
from Phase_0.gpu_particle_array import TorchParticleArray

print("=== PyTorch GPU Frame Kernels Test ===", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)

# ── Init CPU reference ──
cpu = ParticleArray(max_n=5000)
cpu.add_particle(pos=(0,0,0), uid="cent_0000", degree=0,
                 initial_volume=1.0, initial_degree=0)
dirs = _fibonacci_sphere(12)
for i in range(12):
    cpu.add_particle(pos=tuple(dirs[i].astype(float) * 1.0),
                     uid=f"surf_{i:04d}", degree=0,
                     initial_volume=1.0, initial_degree=0)
    cpu.add_connection(0, 1+i)
print(f"CPU init: V={cpu.active_count()} Σ={cpu.spum_invariant()}", flush=True)

# ── Test 1: step3_volume ──
print("\n--- Test 1: step3_volume ---", flush=True)
gpu = TorchParticleArray(max_n=5000)
for i in range(cpu.N):
    if cpu.active[i]:
        gpu.add_particle(pos=tuple(cpu.pos[i]), uid=str(cpu.uid[i]),
                         degree=int(cpu.degree[i]),
                         initial_volume=float(cpu.initial_volume[i]),
                         initial_degree=int(cpu.initial_degree[i]))
for a, b in cpu.connections:
    if cpu.active[a] and cpu.active[b]:
        gpu.add_connection(a, b)

# Run step3 on both
step3_volume(cpu)
torch_step3_volume(gpu)

cpu_r = cpu.radius
gpu_r = gpu.radius.cpu().numpy()
match = np.allclose(cpu_r[cpu.active], gpu_r[cpu.active], atol=1e-6)
print(f"  Radius match: {match}", flush=True)

# ── Test 2: Surface adjacency ──
print("\n--- Test 2: Surface adjacency ---", flush=True)
act_idx = np.where(cpu.active)[0]
surf_idx = [i for i in act_idx if i != 0]
pairs_gpu = torch_surface_adjacency(gpu.pos, gpu.radius, surf_idx, 0)
print(f"  GPU surface pairs: {len(pairs_gpu)}", flush=True)

# ── Test 3: Overlap detection ──
print("\n--- Test 3: Overlap detection ---", flush=True)
overlaps = torch_detect_overlaps(gpu)
print(f"  GPU overlaps: {len(overlaps)}", flush=True)

# ── Test 4: Full frame ──
print("\n--- Test 4: Single frame (no purge) ---", flush=True)
cpu2 = ParticleArray(max_n=5000)
cpu2.add_particle(pos=(0,0,0), uid="cent_0000", degree=0,
                  initial_volume=1.0, initial_degree=0)
for i in range(12):
    cpu2.add_particle(pos=tuple(dirs[i].astype(float) * 1.0),
                      uid=f"surf_{i:04d}", degree=0,
                      initial_volume=1.0, initial_degree=0)
    cpu2.add_connection(0, 1+i)

gpu2 = TorchParticleArray(max_n=5000)
for i in range(cpu2.N):
    if cpu2.active[i]:
        gpu2.add_particle(pos=tuple(cpu2.pos[i]), uid=str(cpu2.uid[i]),
                          degree=int(cpu2.degree[i]),
                          initial_volume=1.0, initial_degree=0)
for a, b in cpu2.connections:
    gpu2.add_connection(a, b)

# CPU
t0 = time.time()
result_cpu = run_full_frame(cpu2, star_mode=True, allow_disconnect=False,
                            no_purge=True, gap_fill_per_frame=30)
t_cpu = time.time() - t0

# GPU
t0 = time.time()
# Manually run steps to avoid circular imports
# Step 3: volume
torch_step3_volume(gpu2)
# Step 3c: tangency
torch_enforce_tangency(gpu2)
# Step 1b: gap fill (CPU, need numpy arrays)
# For this test, skip gap fill — just measure GPU part
# Step 2: surface adjacency
center_idx = 0
surf_pairs = torch_surface_adjacency(gpu2.pos, gpu2.radius, surf_idx, center_idx)
for i, j in surf_pairs:
    gpu2.add_connection(i, j)
# Step 3b: overlap detection
overlaps_gpu = torch_detect_overlaps(gpu2)
for i, j in overlaps_gpu:
    if gpu2.active[i].item() and gpu2.active[j].item():
        d = float(torch.norm(gpu2.pos[i] - gpu2.pos[j]).item())
        min_d = gpu2.radius[i].item() + gpu2.radius[j].item()
        if d < min_d - 0.01 and gpu2.connected(i, j):
            gpu2.remove_connection(i, j)

t_gpu = time.time() - t0

print(f"  CPU: {t_cpu:.3f}s  V={cpu2.active_count()} Σ={cpu2.spum_invariant()}", flush=True)
print(f"  GPU: {t_gpu:.3f}s  V={gpu2.active_count()} Σ={gpu2.spum_invariant()}", flush=True)
if t_gpu > 0:
    print(f"  Speedup: {t_cpu/t_gpu:.1f}x", flush=True)

# ── Test 5: Scale test ──
print("\n--- Test 5: Scale test (1000 particles) ---", flush=True)
N = 1000
gpu3 = TorchParticleArray(max_n=5000)
gpu3.add_particle(pos=(0,0,0), uid="cent_0000", degree=0,
                  initial_volume=1.0, initial_degree=0)
r0 = (3.0/(4.0*np.pi))**(1.0/3.0)
for i in range(N-1):
    theta = np.random.rand() * 2 * np.pi
    phi = np.arccos(2*np.random.rand() - 1)
    x = np.sin(phi) * np.cos(theta) * 1.0
    y = np.sin(phi) * np.sin(theta) * 1.0
    z = np.cos(phi) * 1.0
    gpu3.add_particle(pos=(x,y,z), uid=f"p_{i:04d}", degree=0,
                      initial_volume=1.0, initial_degree=0)
    gpu3.add_connection(0, i+1)

# Time the GPU ops
torch.cuda.synchronize()
t0 = time.time()

# step3 volume
torch_step3_volume(gpu3)
torch.cuda.synchronize()
t3 = time.time() - t0

# surface adjacency
surf_idx_full = list(range(1, N))
pairs = torch_surface_adjacency(gpu3.pos, gpu3.radius, surf_idx_full, 0)
torch.cuda.synchronize()
t2 = time.time() - t0 - t3

# overlap detection
overlaps = torch_detect_overlaps(gpu3)
torch.cuda.synchronize()
t3b = time.time() - t0 - t3 - t2

print(f"  N={N}")
print(f"  step3_volume: {t3:.3f}s", flush=True)
print(f"  surface_adj: {t2:.3f}s ({len(pairs)} pairs)", flush=True)
print(f"  overlap_detect: {t3b:.3f}s ({len(overlaps)} overlaps)", flush=True)

# ── Test 6: 10k particles (overlap only) ──
print("\n--- Test 6: 10k overlap only ---", flush=True)
N10k = 10000
gpu10k = TorchParticleArray(max_n=20000)
gpu10k.add_particle(pos=(0,0,0), uid="cent_0000", degree=0,
                    initial_volume=1.0, initial_degree=0)
for i in range(N10k - 1):
    theta = np.random.rand() * 2 * np.pi
    phi = np.arccos(2*np.random.rand() - 1)
    gpu10k.add_particle(pos=(np.sin(phi)*np.cos(theta),
                              np.sin(phi)*np.sin(theta),
                              np.cos(phi)),
                        uid=f"p_{i:04d}", degree=1,
                        initial_volume=1.0, initial_degree=0)
    gpu10k.add_connection(0, i+1)

torch.cuda.synchronize()
t0 = time.time()
overlaps10k = torch_detect_overlaps(gpu10k)
torch.cuda.synchronize()
t10k = time.time() - t0
print(f"  N={N10k}: overlap detection {t10k:.3f}s ({len(overlaps10k)} pairs)", flush=True)

print("\nDone!", flush=True)
