"""
SPUM 红移验证 — 配置参数
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Config:
    """全局配置"""

    # ── 宇宙学参数 ──
    H0: float = 70.0           # 哈勃常数 (km/s/Mpc)，仅用于距离转换
    Omega_m: float = 0.3       # 物质密度参数，仅用于距离转换
    Omega_lambda: float = 0.7  # 暗能量密度参数，仅用于距离转换

    # ── 密度估计 ──
    n_nearest: int = 5         # 第 N 近邻用于局部密度估计
    z_max_cluster: float = 0.15  # 星系团最大红移（SDSS 光谱完备性限制）
    max_radial_distance: float = 2.0  # 最大投影半径 (Mpc)

    # ── 簇选取 ──
    min_galaxies_per_cluster: int = 30  # 每个星系团最少成员数
    redshift_bins: int = 20     # 红移直方图 bin 数

    # ── SDSS 查询 ──
    sdss_dr: str = "DR18"
    sdss_query_timeout: int = 60  # 秒

    # ── 已知星系团（备用，API 不可用时使用） ──
    known_clusters: List[dict] = field(default_factory=lambda: [
        {
            "name": "Coma",
            "ra": 194.95, "dec": 27.98,
            "z": 0.0231,
            "n_members_expected": 200,
            "radius_deg": 0.5,
        },
        {
            "name": "Abell1367",
            "ra": 176.18, "dec": 19.71,
            "z": 0.0220,
            "n_members_expected": 150,
            "radius_deg": 0.4,
        },
        {
            "name": "Abell2199",
            "ra": 247.38, "dec": 39.55,
            "z": 0.0302,
            "n_members_expected": 100,
            "radius_deg": 0.3,
        },
        {
            "name": "Virgo",
            "ra": 186.75, "dec": 12.72,
            "z": 0.0036,
            "n_members_expected": 300,
            "radius_deg": 1.0,
        },
        {
            "name": "Abell1656_Coma_extended",
            "ra": 194.95, "dec": 27.98,
            "z": 0.0231,
            "n_members_expected": 400,
            "radius_deg": 1.0,
        },
    ])

    # ── 输出 ──
    output_dir: str = "results"
    figure_format: str = "png"
    figure_dpi: int = 150


# 全局配置实例
config = Config()
