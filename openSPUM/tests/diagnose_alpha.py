"""
cap=3 涌现诊断 — 局部拓扑决定补偿边数

评审摘要:
    用户指出 α=1/σ 是构造恒等式，非物理预测。
    cap=3 硬编码，σ=137 后验选择。

本实验:
    1. 测量每个湮灭事件局部邻域中的可用节点数
       → "自然 cap" 分布
    2. 若自然 cap 聚类于 3，则 cap=3 是涌现的而非构造的
    3. 扫描网络参数，寻找 σ 的"自然特征值"

运行: python openSPUM/tests/diagnose_alpha.py
"""

import sys
import math
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from Phase_1.seed_epoch_engine import SeedEpochConfig, SeedEpochEngine
from Phase_1.topological_address import TopologicalAddress
from Phase_2.frame_update_engine import (
    FrameUpdateConfig, FrameUpdateEngine, FrameLog,
)
from Phase_1.relation_pool import RelationPool
from Phase_1.node_registry import NodeRegistry

def measure_natural_sigma(
    target_edges_list: List[int],
    n_samples: int = 5,
) -> Dict:
    """扫描不同网络大小，寻找产生"临界响应"的 σ。

    定义: 使级联迭代次数首次超过 1 的最小 σ。
    这代表扰动开始产生跨帧传播的阈值。
    如果此阈值 ≈ 137，则是真正的物理预测。
    """
    results = {}
    for te in target_edges_list:
        thresholds = []
        for _ in range(n_samples):
            eng = SeedEpochEngine(config=SeedEpochConfig(te, max(50, te//20)))
            eng.run_seed_epoch()
            evo = FrameUpdateEngine(
                eng.relation_pool, eng.node_registry,
                FrameUpdateConfig(growth_per_frame=0, max_cascade_iterations=100, enable_cascade=True),
            )
            evo.run_frames(5)

            # 二分查找临界 σ
            lo, hi = 1, 200
            threshold = hi
            while lo <= hi:
                mid = (lo + hi) // 2
                # 重建网络
                eng2 = SeedEpochEngine(config=SeedEpochConfig(te, max(50, te//20)))
                eng2.run_seed_epoch()
                evo2 = FrameUpdateEngine(
                    eng2.relation_pool, eng2.node_registry,
                    FrameUpdateConfig(growth_per_frame=0, max_cascade_iterations=100, enable_cascade=True),
                )
                evo2.run_frames(5)

                # 注入 mid 个气泡
                origin = TopologicalAddress.origin()
                for _ in range(mid):
                    a = TopologicalAddress.differentiate_from(origin)
                    b = TopologicalAddress.differentiate_from(origin)
                    evo2.relation_pool.manifest_relation(a, b, evo2.current_frame, evo2.node_registry)
                evo2.current_frame += 1

                log = evo2.run_frame()
                if log.cascade_iters > 1:
                    threshold = mid
                    hi = mid - 1
                else:
                    lo = mid + 1

            thresholds.append(threshold)

        results[te] = {
            "thresholds": thresholds,
            "mean": sum(thresholds)/len(thresholds) if thresholds else None,
            "min": min(thresholds) if thresholds else None,
            "max": max(thresholds) if thresholds else None,
        }

    return results


# ============================================================
# 实验 C: 拓扑解释 cap=3
# ============================================================

def topo_derive_cap3() -> str:
    """拓扑推导 cap=3 的理论框架。

    在 SPUM 三角剖分网络中:
        - Σ(6-deg(v)) = 12 (闭合球面的欧拉恒等式)
        - 每个节点的"平面度"定义为 6-deg(v)
        - 一个悬挂对 (deg=1,1) 湮灭后产生两个 deg=0 节点
        - 局部拓扑亏损变化: 每个 deg=0 贡献 6, 合计 12
        - 每条补偿边连接两个 deg<2 节点, 各增加 1 度
        - 单条补偿边的"修复能力" = (6-(d+1)) - (6-d) = -1 每端点
        - 总修复量 = 2 每边
        - 所需边数 = 局部亏损变化 / 每边修复量

    正二十面体论证:
        正二十面体每个顶点 5 个三角形汇聚:
        - 平面角 = 5 × 60° = 300°
        - 缺陷角 = 360° - 300° = 60° = π/3
        - 总缺陷 = 12 × π/3 = 4π (球面)
        - 每个这样的节点 degree=5, deficit=1 (6-5=1)
        - 12 个这样的节点正好贡献 Σ(6-deg)=12

    当湮灭一个悬挂对:
        - 两个节点从 deg=1 变为 deg=0
        - 局部 deficit 从 (6-1)×2=10 变为 (6-0)×2=12
        - 增加 2 单位亏损
        - 需要补偿边恢复局部平坦

    单个补偿边连接两个可用节点:
        - 两端各减 1 单位亏损
        - 共修复 2 单位
        - 故 1 条补偿边足以修复一个悬挂对

    为什么 cap=3 (不是 1)?
        - 批量湮灭涉及多个悬挂对
        - 每个批次最多 `len(batch)` 个边被湮灭
        - 但局部可用节点有限 (2-hop 内 degree<2 的节点数)
        - cap=3 等价于 "每个批处理最多修复 3 个本地缺陷"
        - 3 = Σ(6-deg)/4 = 12/4 — 总拓扑电荷的 1/4

    但这是后验合理化, 不是先验涌现。

    真正的涌现测试:
        如果用局部邻域中的可用节点数代替 cap=3,
        这个数是否自然 ≈ 3?
    """
    return ("cap=3 是 Σ(6-deg)=12 的 1/4, 但在代码中硬编码。"
            "涌现测试见实验 A。")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print(f"{'=' * 72}")
    print(f"  SPUM α 验证 — 诚实评估与涌现诊断")
    print(f"{'=' * 72}")

    print(f"""
  用户评审结论 (完全接受):
  1. alpha = 1/sigma 是构造恒等式 (cap=3, penetration=1.0 均固定)
  2. cap=3 硬编码, 非涌现
  3. sigma=137 后验选择, 非先验预测
  4. SPUM 未独立推导任何 QED 现象

  以下是纠正性实验。
""")

    # === 实验 A: 自然 cap 分布 — 局部拓扑涌现 ===
    print(f"{'─' * 72}")
    print(f"  实验 A: 自然 cap 分布 — 从局部拓扑涌现的补偿边数")
    print(f"{'─' * 72}")
    print()
    print(f"  方法: 注入 σ 气泡, 运行级联消解, 记录实际补偿边数")
    print(f"   (use_natural_cap=True: 由 _compute_natural_cap 决定)")
    print(f"  如果自然 cap 在不同网络规模下收敛于同一值,")
    print(f"  则该值是拓扑涌现的物理常数。")
    print()

    test_configs = [
        ("1K", SeedEpochConfig(2000, 100)),
        ("10K", SeedEpochConfig(20000, 300)),
        ("100K", SeedEpochConfig(100000, 500)),
    ]

    sigma_values = [5, 10, 20]

    for label, cfg in test_configs:
        for sigma in sigma_values:
            caps = []
            for s in range(5):
                eng = SeedEpochEngine(config=cfg)
                eng.run_seed_epoch()
                evo = FrameUpdateEngine(
                    eng.relation_pool, eng.node_registry,
                    FrameUpdateConfig(
                        growth_per_frame=0,
                        max_cascade_iterations=100,
                        enable_cascade=True,
                        use_natural_cap=True,
                    ),
                )
                evo.run_frames(5)

                origin = TopologicalAddress.origin()
                for _ in range(sigma):
                    a = TopologicalAddress.differentiate_from(origin)
                    b = TopologicalAddress.differentiate_from(origin)
                    evo.relation_pool.manifest_relation(a, b, evo.current_frame, evo.node_registry)
                evo.current_frame += 1

                log = evo.run_frame()
                caps.append(len(log.compensated_keys))

            mean_cap = sum(caps) / len(caps)
            hist = {}
            for c in caps:
                hist[c] = hist.get(c, 0) + 1
            hist_str = ", ".join(f"{k}:{v}" for k, v in sorted(hist.items()))
            print(f"  {label:>5s} sigma={sigma:>2d}: "
                  f"cap={mean_cap:.1f} (n={len(caps)}) 分布={{{hist_str}}}")

    print()
    print(f"  预期 (如果涌现): cap 在不同网络规模下保持稳定")
    print(f"  预期 (如果构造): cap 随 sigma 线性增长")

    # === 诊断 cap=3 的拓扑来源 ===
    print(f"{'─' * 72}")
    print(f"  拓扑诊断: cap=3 的数学结构")
    print(f"{'─' * 72}")
    print()
    print(f"  Σ(6-deg) = 12 (闭合三角剖分球面欧拉恒等式)")
    print(f"  悬挂对湮灭: 局部 deficit 增加 2 (±0)")  
    print(f"  单条补偿边: 修复 2 单位 deficit")
    print(f"  1 对 1 湮灭: 刚好 1 条边够用")
    print(f"  批量湮灭: min(batch_size, 3) -> 不是拓扑推导")
    print()
    print(f"  cap=3 的来源: 代码中硬编码的 min(len(batch), 3)")
    print(f"  这个 3 没有拓扑涌现证明。")

    # === 实验 B 弃用: 级联迭代阈值与 σ 无直接关系 ===

    # === 当前 alpha 的本质 ===
    print(f"\n{'─' * 72}")
    print(f"  当前 alpha 的本质: 最终判断")
    print(f"{'─' * 72}")
    print(f"""
  当前 alpha_SPUM = 0.007299 (sigma=137):

  alpha 定义: (cap x penetration) / (sigma x cap)
           = (3 x 1.0) / (137 x 3)
           = 1/137

  cap=3:         硬编码, 无涌现证明
  penetration=1: 全部补偿边耦合到网络 (排序键构造所致)
  sigma=137:     后验选择 (1/137 ≈ 物理值)

  结论: alpha=1/137 是构造恒等式, 非物理预测。
  与精细结构常数的同名关系不意味着物理等价。

  要成为真正的物理预测, SPUM 需要:
  1. cap=3 从 Σ(6-deg)=12 涌现 (实验 A 未验证)
  2. sigma=137 从种子期参数推导 (实验 B 未验证)
  3. 独立计算 QED 现象 (能级、磁矩)
""")

    # 清理诊断脚本
    print(f"{'=' * 72}")
    print(f"  诊断完成")
    print(f"{'=' * 72}")
