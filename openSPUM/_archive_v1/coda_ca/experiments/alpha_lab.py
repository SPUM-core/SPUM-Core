"""
α 数值实验 — 用纯 CA 规则测量精细结构常数。

SPUM 假说：α ≈ 1/137 不是外部常数，而是 CA 规则
稳态下悬挂边比例的自然涌现值。

测量方法：
    α = n_dangling / n_active

在足够大的 CA 中，悬挂边比例在稳态下收敛到
一个固定值——这就是 SPUM 版本的"精细结构常数"。

运行:
    python -m openSPUM.coda_ca.experiments.alpha_lab
"""

import sys
import math
from pathlib import Path

# 确保可以导入 coda_ca
_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from coda_ca import Simulator


def measure_alpha(sim: Simulator, settle_frames: int = 50,
                  measure_frames: int = 50) -> float:
    """测量 α = 悬挂边比例。

    步骤：
        1. 等待演化稳定（settle_frames 帧）
        2. 在 measure_frames 帧中采样
        3. 取平均值

    Returns:
        α 估计值（悬挂边/活跃节点）
    """
    # 等待稳定
    sim.step_n(settle_frames)

    # 采样
    alphas = []
    histories = sim.step_n(measure_frames)
    for snap in histories:
        n_active = snap.n_active
        n_dangling = snap.n_dangling
        if n_active > 0:
            alphas.append(n_dangling / n_active)

    if not alphas:
        return 0.0
    return sum(alphas) / len(alphas)


def run_experiment(n_codas: int = 2000,
                   rule: str = "mean",
                   init: str = "icosahedron",
                   trials: int = 3) -> dict:
    """运行 α 测量实验。"""
    print(f"\n{'='*60}")
    print(f"实验: n={n_codas}, rule={rule}, init={init}")
    print(f"{'='*60}")

    results = []
    for t in range(trials):
        sim = Simulator(n_codas=n_codas, rule_name=rule)
        if init == "icosahedron":
            sim.initialize_icosahedron()
        elif init == "star":
            sim.initialize_star(n_surface=min(50, n_codas - 1))
        elif init == "random":
            sim.initialize_random_graph(edge_density=0.03)
        elif init == "sequence":
            sim.initialize_from_sequence(n_active=min(200, n_codas))

        alpha = measure_alpha(sim, settle_frames=100, measure_frames=100)
        summary = sim.summary()
        results.append(alpha)

        print(f"  试验 {t+1}: α ≈ {alpha:.6f}  "
              f"(active={summary['active']}, "
              f"dangling={sim.history[-1].n_dangling}, "
              f"edges={summary['edges']})")

    avg = sum(results) / len(results)
    std = math.sqrt(sum((a - avg) ** 2 for a in results) / len(results))

    print(f"\n  结果: α = {avg:.6f} ± {std:.6f}")
    print(f"  倒数 1/α = {1/avg:.2f}" if avg > 0 else "  α=0")

    return {
        "rule": rule,
        "init": init,
        "n_codas": n_codas,
        "alpha_mean": avg,
        "alpha_std": std,
        "alpha_inv": 1.0 / avg if avg > 0 else float('inf'),
        "trials": trials,
    }


def compare_rules():
    """比较不同规则下的 α 值。"""
    print("\n\n========== 规则比较实验 ==========")
    for rule in ["mean", "median", "gradient"]:
        run_experiment(n_codas=2000, rule=rule, trials=1)


def compare_init():
    """比较不同初态下的 α 值。"""
    print("\n\n========== 初态比较实验 ==========")
    for init in ["icosahedron", "star", "sequence"]:
        run_experiment(n_codas=1000, rule="mean", init=init, trials=1)


def scan_n():
    """扫描 n 对 α 的影响。"""
    print("\n\n========== 规模扫描实验 ==========")
    for n in [100, 500, 1000, 2000, 5000]:
        run_experiment(n_codas=n, rule="mean", trials=1)


if __name__ == "__main__":
    print("SPUM CA — α 精细结构常数测量实验")
    print("=" * 60)
    print("假说: α = n_dangling / n_active 在稳态下收敛")

    # 默认实验
    result = run_experiment(n_codas=2000, rule="mean", init="icosahedron",
                            trials=3)
    print(f"\n1/α ≈ {result['alpha_inv']:.2f}  "
          f"(预期物理 α⁻¹ ≈ 137)")

    # 扫描
    scan_n()
