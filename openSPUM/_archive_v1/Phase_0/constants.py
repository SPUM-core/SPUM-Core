"""Phase 0 常量 — 与 Phase 1 对齐 + GPU 专用常量。

设计原则：
    GPU 常量从 Phase 1 导入，不重复定义。
    只有 GPU 专用的参数（如 MAX_PARTICLES, MIN_GAP_RATIO）在此定义。
"""

# GPU 粒子数组最大容量
MAX_PARTICLES: int = 65536

# 最小缝隙比率：缝隙半径 / 粒子半径 低于此值则忽略
MIN_GAP_RATIO: float = 1.5

# 导入 Phase 1 的 SPUM 基础常量
from Phase_1.constants import (
    KAPPA,                    # 1.0 — 最小差异尺度
    TAU,                      # 1   — 离散帧时间单位
    CRYSTALLITE_DEGREE_THRESHOLD,  # 42 — 晶子度数阈值 (T5 稳定解)
    GEOMETRIC_TOLERANCE,      # 0.1 — 相切判定容差
    CRYSTALLITE_CAPACITY,     # 16π ≈ 50.3 (T8 容量极致，仅几何参考)
)
