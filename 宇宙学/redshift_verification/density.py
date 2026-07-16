"""
局部 σ（环境密度）估计器。

σ 的观测代理：局部星系面密度 Σ_N（第 N 近邻投影密度）。

在 SPUM 框架中，σ = |P|/|ε| 是子网的空间密度。观测上，
高 σ 区 = 星系密集区（星系团核心），低 σ 区 = 稀疏区（外围）。
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Optional

# 宇宙学常数
DEG2RAD = np.pi / 180.0


def angular_separation(ra1: np.ndarray, dec1: np.ndarray,
                       ra2: np.ndarray, dec2: np.ndarray) -> np.ndarray:
    """
    计算两点间的角距离（度）。
    使用 Haversine 公式。
    """
    d_ra = np.radians(ra2 - ra1)
    d_dec = np.radians(dec2 - dec1)
    lat1 = np.radians(dec1)
    lat2 = np.radians(dec2)

    a = np.sin(d_dec / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(d_ra / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))))


def comoving_distance(z: float, H0: float = 70.0,
                      Omega_m: float = 0.3,
                      Omega_lambda: float = 0.7) -> float:
    """
    计算红移 z 对应的共动距离（Mpc）。
    使用 ΛCDM 模型——仅用于坐标转换，不涉及物理假设。
    """
    from scipy.integrate import quad
    c = 299792.458  # km/s
    E_inv = lambda z: 1.0 / np.sqrt(
        Omega_m * (1 + z) ** 3 + Omega_lambda
    )
    integral = quad(E_inv, 0, z)[0]
    return c / H0 * integral


class SigmaEstimator:
    """
    σ 估计器：从星系位置计算局部环境密度。

    将 SPUM 的 σ = |P|/|ε| 映射为天文学的
    局部面密度 Σ_N = N / (π · d_N²)。

    在 N 固定时，高 Σ_N = 高 σ = 帧周期更长的区域。
    """

    def __init__(self, n_nearest: int = 5,
                 H0: float = 70.0,
                 Omega_m: float = 0.3,
                 Omega_lambda: float = 0.7):
        """
        参数
        ----------
        n_nearest : 第 N 近邻数（默认 5）
        """
        self.n_nearest = n_nearest
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_lambda = Omega_lambda

    def compute_surface_density(self, ra: np.ndarray, dec: np.ndarray,
                                z: np.ndarray) -> np.ndarray:
        """
        计算每个星系的局部面密度 Σ_N。

        步骤：
        1. 将 (RA, Dec, z) 转换为 3D 笛卡尔坐标
        2. 用 KDTree 找第 N 近邻
        3. Σ_N = N / (π · d_N²)

        参数
        ----------
        ra, dec : 赤经、赤纬（度）
        z : 光谱红移

        返回
        -------
        sigma_local : 每个星系的局部面密度（任意单位，相对值有意义）
        """
        n = len(ra)
        # 转换为 3D 笛卡尔坐标（单位球面上）
        ra_r = np.radians(ra)
        dec_r = np.radians(dec)
        x = np.cos(dec_r) * np.cos(ra_r)
        y = np.cos(dec_r) * np.sin(ra_r)
        z_coord = np.sin(dec_r)

        coords_3d = np.column_stack([x, y, z_coord])

        # 使用 KDTree
        tree = KDTree(coords_3d)
        # 查询第 n_nearest 近邻的距离
        # k = n_nearest + 1 因为最近的是自身（距离 0）
        k = min(self.n_nearest + 1, n)
        distances, indices = tree.query(coords_3d, k=k)

        # 第 N 近邻的角距离（弧度）
        d_N_rad = distances[:, -1]  # 第 n_nearest 近邻的距离
        # 确保非零
        d_N_rad = np.maximum(d_N_rad, 1e-10)

        # 面密度 Σ = N / (π · d_N²)
        sigma_local = self.n_nearest / (np.pi * d_N_rad ** 2)

        return sigma_local

    def compute_sky_density(self, ra: np.ndarray, dec: np.ndarray,
                            z: np.ndarray) -> np.ndarray:
        """
        计算投影面密度（仅使用 RA/Dec，不考虑 z）。

        对于深度较小的星系团（z < 0.05），投影密度
        是 σ 的良好近似。
        """
        n = len(ra)
        # 将 RA/Dec 转换为弧度
        ra_r = np.radians(ra)
        dec_r = np.radians(dec)
        x = np.cos(dec_r) * np.cos(ra_r)
        y = np.cos(dec_r) * np.sin(ra_r)
        z_c = np.sin(dec_r)

        coords = np.column_stack([x, y, z_c])
        tree = KDTree(coords)
        k = min(self.n_nearest + 1, n)
        distances, _ = tree.query(coords, k=k)

        d_N_rad = np.maximum(distances[:, -1], 1e-10)
        sigma = self.n_nearest / (np.pi * d_N_rad ** 2)
        return sigma

    def estimate_sigma_field(self, ra: np.ndarray, dec: np.ndarray,
                              z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计每个星系的 σ 值及其归一化形式。

        返回
        -------
        sigma_raw : 原始面密度
        sigma_norm : 归一化到 [0, 1] 的 σ（用于跨簇比较）
        """
        sigma_raw = self.compute_sky_density(ra, dec, z)
        # 对数归一化（面密度通常是对数分布的）
        log_sigma = np.log10(np.maximum(sigma_raw, 1e-10))
        sigma_norm = (log_sigma - log_sigma.min()) / \
                     max(log_sigma.max() - log_sigma.min(), 1e-10)
        return sigma_raw, sigma_norm

    def compute_z_sigma_correlation(self, ra: np.ndarray, dec: np.ndarray,
                                     z: np.ndarray) -> dict:
        """
        计算红移 z 与局部 σ 之间的 Spearman 秩相关。

        这是 SPUM 预言的直接检验：
        如果 z ~ σ（本征频率差），则 ρ_spearman > 0。
        """
        from scipy import stats

        sigma_raw, sigma_norm = self.estimate_sigma_field(ra, dec, z)

        # Spearman 秩相关
        rho, p_value = stats.spearmanr(z, sigma_norm)

        # Pearson 线性相关
        r_pearson, p_pearson = stats.pearsonr(z, sigma_norm)

        return {
            "spearman_rho": rho,
            "spearman_p": p_value,
            "pearson_r": r_pearson,
            "pearson_p": p_pearson,
            "n_galaxies": len(z),
            "sigma_range": (float(sigma_norm.min()), float(sigma_norm.max())),
            "z_range": (float(z.min()), float(z.max())),
        }
