"""
诱导坐标嵌入包：从 ⟨P, ε⟩ 离散关系网络到 ℝ³ 的内生嵌入。

严格遵循 L0.5_induced_metric.md 的理论定义：
  1. 12-晶子种子初始化（正二十面体顶点）
  2. BFS传播 + 球面码方向规则
  3. σ调节步长
  4. 多邻居冲突加权质心解决
"""

from .icosahedron_seed import (
    generate_icosahedron_seed,
    get_spherical_code_directions,
)

from .embedding import (
    InducedMetricEmbedding,
)

from .uniform_limit import (
    uniform_limit_metric,
    prove_r_inverse_square,
    run_verification,
)

__all__ = [
    'generate_icosahedron_seed',
    'get_spherical_code_directions',
    'InducedMetricEmbedding',
    'uniform_limit_metric',
    'prove_r_inverse_square',
    'run_verification',
]
