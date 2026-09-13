"""GPU 后端部署测试 — 后端探测 / 强制选择 / 严格语义等价。

运行 (需在 openSPUM 根目录):
    python tests/test_gpu_backend.py

覆盖:
    1. gpu_backend: resolve/active_backend/cuda_available 探测与强制选择
    2. 严格语义等价: backend='cpu' 与 backend='cuda' 逐帧逐指标完全一致
       (V, E, Σ(6-deg), 晶子数, 悬挂) — 证明 GPU 只加速 O(N²) 相切扫描,
       不改变任何帧语义。
    3. GPU 热区加速可行性微基准 (触发 CUDA cdist, 不比对数值)。
"""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from Phase_0.gpu_backend import (
    resolve, active_backend, cuda_available, device_name,
)
from Phase_0.gpu_engine import SPUMEngine, EngineConfig

PASS = 0
FAIL = 0
SKIP = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name} {detail}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def test_backend_resolution():
    print("\n=== 1. 后端探测与选择 ===")
    cuda = cuda_available()
    check(f"CUDA 可用 ({device_name()})", isinstance(cuda, bool))
    check("active_backend ∈ {'cuda','cpu'}", active_backend() in ('cuda', 'cpu'))
    check("resolve('cpu')=='cpu'", resolve('cpu') == 'cpu')
    check("resolve('auto')=active_backend", resolve('auto') == active_backend())
    if cuda:
        check("resolve('cuda')=='cuda'", resolve('cuda') == 'cuda')
    else:
        # 无 GPU: 强制 cuda 应抛错
        try:
            resolve('cuda')
            check("resolve('cuda') 无GPU应抛错", False)
        except RuntimeError:
            check("resolve('cuda') 无GPU抛错", True)


def run_backend(backend, frames=8):
    cfg = EngineConfig(seed_geometry="star", n_surface=42,
                       pre_growth_frames=5, backend=backend,
                       gap_enabled=True, verbose=False)
    e = SPUMEngine(cfg)
    rows = []
    for _ in range(frames):
        snap = e.run_frame()
        edge = (6 * snap.active_count - snap.spum_invariant) // 2
        rows.append((snap.active_count, edge, snap.spum_invariant,
                     snap.crystallite_count, snap.dangling_count))
    return rows


def test_gpu_cpu_equivalence():
    print("\n=== 2. GPU == CPU 严格语义等价 ===")
    if active_backend() != 'cuda':
        global SKIP
        SKIP += 1
        print("  [SKIP] 无 CUDA, 跳过 GPU/CPU 对照")
        return
    frames = 8
    cpu = run_backend('cpu', frames)
    gpu = run_backend('cuda', frames)
    all_ok = True
    print(f"  {'帧':>3} | {'V':>5} | {'E':>6} | {'Σ':>8} | {'晶':>4} | {'悬':>4}")
    for f in range(frames):
        row_match = tuple(cpu[f]) == tuple(gpu[f])
        all_ok = all_ok and row_match
        print(f"  {f+1:>3} | {cpu[f][0]:>5} | {cpu[f][1]:>6} | {cpu[f][2]:>8} "
              f"| {cpu[f][3]:>4} | {cpu[f][4]:>4}  {'OK' if row_match else 'X'}")
    check("8 帧 V/E/Σ/晶子/悬挂 全部一致", all_ok)


def test_cuda_hot_region():
    print("\n=== 3. CUDA 热区 (相切扫描) 微基准 ===")
    if active_backend() != 'cuda':
        SKIP += 1
        print("  [SKIP] 无 CUDA")
        return
    from Phase_0.gpu_accel import tangent_pairs_gpu
    rng = np.random.default_rng(0)
    for n in (500, 2000):
        pos = rng.normal(size=(n, 3)).astype(np.float32)
        radius = np.full(n, 1.0, dtype=np.float32)
        active = np.ones(n, dtype=bool)
        t0 = time.perf_counter()
        pairs = tangent_pairs_gpu(pos, radius, active)
        dt = time.perf_counter() - t0
        print(f"  N={n:>5}: tangent_pairs_gpu (CUDA) {dt*1000:.1f} ms, "
              f"{len(pairs)} 候选对")
    check("CUDA 相切扫描可执行", True)


if __name__ == "__main__":
    test_backend_resolution()
    test_gpu_cpu_equivalence()
    test_cuda_hot_region()

    print(f"\n=== 结果: PASS={PASS} FAIL={FAIL} SKIP={SKIP} ===")
    sys.exit(1 if FAIL else 0)