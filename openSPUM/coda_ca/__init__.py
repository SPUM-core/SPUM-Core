"""
Coda CA — SPUM 元胞自动机引擎。

本体论宣言：
    节点不存储任何属性，只存储一个整数：度数。
    节点的全部"知识"来自与邻居的通信。
    没有坐标、没有半径、没有全局调度器。
    宇宙 = 一条局部规则重复执行。

使用方式：
    from coda_ca import Simulator

    sim = Simulator(n_codas=1000, rule_name="mean")
    sim.initialize_icosahedron()
    for _ in range(100):
        sim.step()
    print(sim.summary())
"""

from .core.coda import Coda
from .core.rules import (
    rule_mean_tracking,
    rule_median_convergence,
    rule_gradient_driven,
    RULES,
    get_rule,
    KAPPA,
    CRYSTALLITE_DEGREE_THRESHOLD,
)
from .core.simulator import Simulator, FrameSnapshot
from .projection.geometric import (
    GeometricProjector,
    TemporalProjector,
    ChemicalProjector,
)

__all__ = [
    "Coda",
    "Simulator",
    "FrameSnapshot",
    "rule_mean_tracking",
    "rule_median_convergence",
    "rule_gradient_driven",
    "RULES", "get_rule",
    "KAPPA", "CRYSTALLITE_DEGREE_THRESHOLD",
    "GeometricProjector",
    "TemporalProjector",
    "ChemicalProjector",
]
