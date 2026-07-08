"""
星系团数据结构。

每个星系团 = 一个 SPUM 星系子网的观测投影。
"""

import numpy as np
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClusterGalaxy:
    """单个星系的数据。"""
    objid: str = ""            # SDSS 对象 ID
    ra: float = 0.0             # 赤经（度）
    dec: float = 0.0            # 赤纬（度）
    z: float = 0.0              # 光谱红移
    z_err: float = 0.0          # 红移误差
    mag_r: float = -99.0        # r 波段星等（Petrosian）
    petroR50: float = -99.0     # 半光半径（角秒）
    morphology: str = ""         # 形态分类: "E"=早型, "S"=晚型, "S0", "Irr"
    concentration: float = -99.0 # 聚集指数 C = R90/R50（SDSS 形态代理）


@dataclass
class GalaxyCluster:
    """星系团 = SPUM 星系子网。

    拓扑参数（SPUM 核心指标）：
    - sigma_mean / sigma_std : σ 场的中心值和弥散
    - sigma_gradient : 核心-外围 σ 梯度
    - cluster_z_range : 红移范围（核心 vs 外围）
    """
    name: str
    ra_center: float = 0.0
    dec_center: float = 0.0
    z_mean: float = 0.0
    radius_deg: float = 0.5

    # 成员星系
    galaxies: List[ClusterGalaxy] = field(default_factory=list)

    # SPUM 诊断
    sigma_mean: float = 0.0
    sigma_std: float = 0.0
    sigma_gradient: float = 0.0  # Δσ/Δr
    z_gradient: float = 0.0      # Δz/Δr
    z_sigma_correlation: float = 0.0  # Spearman ρ
    p_value: float = 1.0

    @property
    def n_members(self) -> int:
        return len(self.galaxies)

    @property
    def redshifts(self) -> np.ndarray:
        return np.array([g.z for g in self.galaxies])

    @property
    def ra_list(self) -> np.ndarray:
        return np.array([g.ra for g in self.galaxies])

    @property
    def dec_list(self) -> np.ndarray:
        return np.array([g.dec for g in self.galaxies])

    @property
    def projected_radii(self) -> np.ndarray:
        """计算每个星系距团中心的投影角距离（度）。"""
        from density import angular_separation
        return angular_separation(
            self.ra_center, self.dec_center,
            self.ra_list, self.dec_list
        )

    @property
    def morphologies(self) -> List[str]:
        return [g.morphology for g in self.galaxies]

    @property
    def concentrations(self) -> np.ndarray:
        return np.array([g.concentration for g in self.galaxies])

    @property
    def redshifts_early(self) -> np.ndarray:
        """早型星系（E/S0）的红移数组。"""
        return np.array([g.z for g in self.galaxies
                         if g.morphology in ("E", "S0")])

    @property
    def redshifts_late(self) -> np.ndarray:
        """晚型星系（Scd/Irr）的红移数组。"""
        return np.array([g.z for g in self.galaxies
                         if g.morphology in ("S", "Irr")])

    @property
    def n_early(self) -> int:
        return len(self.redshifts_early)

    @property
    def n_late(self) -> int:
        return len(self.redshifts_late)

    @property
    def early_fraction(self) -> float:
        return self.n_early / max(self.n_members, 1)

    @property
    def morphology_labels(self) -> np.ndarray:
        """返回每个星系的形态标签（0=晚型, 1=早型）。"""
        return np.array([
            1 if g.morphology in ("E", "S0") else 0
            for g in self.galaxies
        ])

    def bin_by_morphology(self, sigma_norm: np.ndarray) -> dict:
        """按形态分类返回 σ 和 z 统计。"""
        early_sigma, late_sigma = [], []
        early_z, late_z = [], []
        for g, s in zip(self.galaxies, sigma_norm):
            if g.morphology in ("E", "S0"):
                early_sigma.append(s)
                early_z.append(g.z)
            else:
                late_sigma.append(s)
                late_z.append(g.z)
        return {
            "early": {"sigma": np.array(early_sigma), "z": np.array(early_z)},
            "late": {"sigma": np.array(late_sigma), "z": np.array(late_z)},
        }

    def summary(self) -> str:
        return (
            f"GalaxyCluster({self.name}, "
            f"z={self.z_mean:.4f}, "
            f"N={self.n_members}, "
            f"E={self.n_early}/{self.n_late}, "
            f"σ_μ={self.sigma_mean:.3f}, "
            f"σ_σ={self.sigma_std:.3f}, "
            f"∇z={self.z_gradient:.5f}, "
            f"ρ(z,σ)={self.z_sigma_correlation:.3f}, "
            f"p={self.p_value:.4f})"
        )
