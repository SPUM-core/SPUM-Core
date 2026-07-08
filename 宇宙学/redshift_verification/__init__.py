"""
SPUM 红移验证包

验证 SPUM 宇宙学的核心预言：红移是 σ 差异的本征读数，
而非退行速度。z = σ_source/σ_obs - 1。

依赖: numpy, scipy, matplotlib, astropy, astroquery（可选）
"""

from .model import SPUMRedshiftModel, sigma_from_z, z_from_sigma
from .density import SigmaEstimator
from .cluster import GalaxyCluster, ClusterGalaxy

__version__ = "0.1.0"
