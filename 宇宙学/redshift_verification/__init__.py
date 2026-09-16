"""
SPUM 红移验证包

验证 SPUM 宇宙学的核心预言：同一星系团内，红移是 σ 差异的本征读数，
而非退行速度。

**口径（2026-09-16 统一）**：本包使用 `z = σ_source/σ_obs - 1` 作为
**局域两端退化读数**（仅适用于簇内两端近似）。红移的**通式**是光子
数据链的累积读数 `z ∝ Σ_k (σ(l_k)−σ₀)/σ₀`——光子途中不被改造，逐跳
追加 σ 记录。详见 SPUM2610.md §9.1.3 与 .trae/rules/spum-vocabulary.md §三第 6 条。

依赖: numpy, scipy, matplotlib, astropy, astroquery（可选）
"""

from .model import SPUMRedshiftModel, sigma_from_z, z_from_sigma
from .density import SigmaEstimator
from .cluster import GalaxyCluster, ClusterGalaxy

__version__ = "0.1.0"
