"""青檬引擎 · 公理内核层 —— 理论不可变。

core 层硬编码 SPUM 公理（11 条不变量契约），不随领域/业务变化。
本层禁止动态修改——共识底线。

模块:
    axioms     12 晶子原语 · Σ(6−deg)=12 · σ · dv/dt ≤ const · 级联消解
    state      五步帧状态机（创生→连接→变化体积→判断悬挂→删除）
    consensus  11 条不变量校验器（所有推理路径的强制关卡）
"""

from .axioms import Core, Constants, CORE_SIZE, ICOSAHEDRON_EDGES
from .state import StateEvolutionEngine, FrameSnapshot, CREATION_MAX_PER_FRAME
from .consensus import (
    ConsensusValidator,
    ConsensusReport,
    ConsensusFailure,
    ConsensusViolationError,
)

__all__ = [
    "Core",
    "Constants",
    "CORE_SIZE",
    "ICOSAHEDRON_EDGES",
    "StateEvolutionEngine",
    "FrameSnapshot",
    "CREATION_MAX_PER_FRAME",
    "ConsensusValidator",
    "ConsensusReport",
    "ConsensusFailure",
    "ConsensusViolationError",
]
