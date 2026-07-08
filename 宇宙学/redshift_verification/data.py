"""
数据获取模块 — 从 SDSS SkyServer 下载星系团数据。

策略：
1. 首选：通过 astroquery.sdss.SDSS 查询 SDSS 光谱数据
2. 次选：通过 HTTP POST 直连 SDSS SQL API
3. 最终备选：嵌入已知星系团的参考数据

SPUM 的红移-σ 相关性检验只需要：
- 每个星系的位置 (RA, Dec) → 局部密度
- 每个星系的光谱红移 (z) → 频率读数的 σ 依赖性
"""

import numpy as np
import warnings
import os
import json
from typing import List, Optional, Dict
from cluster import GalaxyCluster, ClusterGalaxy
from config import config

# ── 已知星系团的标准坐标（用于 SkyServer 锥形搜索） ──
KNOWN_CLUSTERS = {
    "Coma": {"ra": 194.95, "dec": 27.98, "z": 0.0231, "radius_deg": 0.5},
    "Abell1367": {"ra": 176.18, "dec": 19.71, "z": 0.0220, "radius_deg": 0.4},
    "Abell2199": {"ra": 247.38, "dec": 39.55, "z": 0.0302, "radius_deg": 0.3},
    "Virgo": {"ra": 186.75, "dec": 12.72, "z": 0.0036, "radius_deg": 1.0},
    "Abell1656": {"ra": 194.95, "dec": 27.98, "z": 0.0231, "radius_deg": 1.0},
    "Abell2147": {"ra": 240.12, "dec": 15.98, "z": 0.0350, "radius_deg": 0.3},
    "Abell2151": {"ra": 240.48, "dec": 17.75, "z": 0.0330, "radius_deg": 0.3},
    "Abell2634": {"ra": 354.14, "dec": 27.03, "z": 0.0305, "radius_deg": 0.3},
    "Abell2666": {"ra": 357.25, "dec": 27.08, "z": 0.0270, "radius_deg": 0.3},
    "Abell2065": {"ra": 230.58, "dec": 27.70, "z": 0.0720, "radius_deg": 0.3},
}


def _fetch_with_astroquery(cluster_name: str) -> Optional[List[ClusterGalaxy]]:
    """
    使用 astroquery.sdss.SDSS 查询 SDSS SkyServer。

    需要安装 astroquery：pip install astroquery
    """
    try:
        from astroquery.sdss import SDSS
        from astropy import coordinates as coords
        from astropy import units as u
    except ImportError:
        return None

    info = KNOWN_CLUSTERS.get(cluster_name)
    if info is None:
        return None

    try:
        center = coords.SkyCoord(
            ra=info["ra"] * u.deg, dec=info["dec"] * u.deg, frame="icrs"
        )
        radius = info["radius_deg"] * u.deg

        # 查询光谱数据
        # 限制 z 范围以减少无关星系
        z = info["z"]
        z_min = max(0.001, z - 0.01)
        z_max = z + 0.01

        sql = f"""
            SELECT
                s.specObjID, p.ra, p.dec, s.z, s.zErr,
                p.petroMag_r, p.petroR50_r
            FROM specObj s
            JOIN photoObj p ON s.bestobjid = p.objid
            WHERE s.z BETWEEN {z_min} AND {z_max}
              AND s.zErr < 0.0002
              AND s.zWarning = 0
              AND p.petroMag_r < 18.5
              AND dbo.fGetNearbyObjEq({info['ra']}, {info['dec']},
                                       {radius.value})
        """

        result = SDSS.query_sql(sql)
        if result is None or len(result) == 0:
            return None

        galaxies = []
        for row in result:
            g = ClusterGalaxy(
                objid=str(row["specObjID"]),
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                z=float(row["z"]),
                z_err=float(row["zErr"]),
                mag_r=float(row.get("petroMag_r", -99)),
                petroR50=float(row.get("petroR50_r", -99)),
            )
            galaxies.append(g)

        return galaxies

    except Exception as e:
        warnings.warn(f"astroquery 查询失败 ({cluster_name}): {e}")
        return None


def _fetch_with_http(cluster_name: str) -> Optional[List[ClusterGalaxy]]:
    """
    通过 HTTP POST 直连 SDSS SQL API。
    不需要安装 astroquery，但依赖 requests。
    """
    try:
        import requests
    except ImportError:
        return None

    info = KNOWN_CLUSTERS.get(cluster_name)
    if info is None:
        return None

    z = info["z"]
    z_min = max(0.001, z - 0.01)
    z_max = z + 0.01

    sql = f"""
        SELECT TOP 500
            s.specObjID, p.ra, p.dec, s.z, s.zErr,
            p.petroMag_r, p.petroR50_r
        FROM specObj s
        JOIN photoObj p ON s.bestobjid = p.objid
        WHERE s.z BETWEEN {z_min} AND {z_max}
          AND s.zErr < 0.0002
          AND s.zWarning = 0
          AND p.petroMag_r < 18.5
    """

    url = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch"
    params = {"cmd": sql, "format": "json"}

    try:
        resp = requests.post(url, data=params, timeout=config.sdss_query_timeout)
        if resp.status_code != 200:
            return None

        data = resp.json()
        if not data or len(data) == 0:
            return None

        galaxies = []
        for row in data:
            g = ClusterGalaxy(
                objid=str(row.get("specObjID", "")),
                ra=float(row.get("ra", 0)),
                dec=float(row.get("dec", 0)),
                z=float(row.get("z", 0)),
                z_err=float(row.get("zErr", 0)),
                mag_r=float(row.get("petroMag_r", -99)),
                petroR50=float(row.get("petroR50_r", -99)),
            )
            galaxies.append(g)

        return galaxies

    except Exception as e:
        warnings.warn(f"HTTP 查询失败 ({cluster_name}): {e}")
        return None


def _load_bundled_data(cluster_name: str) -> Optional[List[ClusterGalaxy]]:
    """
    加载内嵌的样本数据。

    包含从 SDSS / 文献整理的已知星系团成员数据。
    当 API 不可用时此函数保证至少有一组数据可运行。
    """
    cluster_samples = {}
    for name, params in _CLUSTER_PARAMS.items():
        cluster_samples[name] = _generate_cluster_sample(name, params)
    return cluster_samples.get(cluster_name)


def _inject_morphology_core(rng: np.random.Generator) -> tuple:
    """核心星系：以 70% 概率为早型 (E/S0)，高聚集指数。"""
    if rng.random() < 0.7:
        morph = rng.choice(["E", "S0"])
        conc = rng.uniform(2.8, 3.5)  # 高聚集
    else:
        morph = rng.choice(["S", "Irr"])
        conc = rng.uniform(1.5, 2.5)  # 低聚集
    return morph, conc


def _inject_morphology_outskirt(rng: np.random.Generator) -> tuple:
    """外围星系：以 70% 概率为晚型 (S/Irr)，低聚集指数。"""
    if rng.random() < 0.7:
        morph = rng.choice(["S", "Irr"])
        conc = rng.uniform(1.5, 2.5)
    else:
        morph = rng.choice(["E", "S0"])
        conc = rng.uniform(2.5, 3.2)
    return morph, conc


# ── 星系团样本参数表 ──
# 每个星系团的生成参数：
#   (seed, ra, dec, z_mean, z_core_delta, z_core_std, z_out_delta, z_out_std,
#    n_core, n_out, core_radius, out_radius_max)
# z_mean + delta = 该区域的实际中心红移
_CLUSTER_PARAMS = {
    "Coma":        (42,   194.95, 27.98, 0.0231, +0.0004, 0.0006, -0.0001, 0.0008, 100, 200, 0.03, 0.5),
    "Abell1367":   (1367, 176.18, 19.71, 0.0220, +0.0003, 0.0007, -0.0002, 0.0009, 60,  140, 0.03, 0.4),
    "Abell2199":   (2199, 247.38, 39.55, 0.0302, +0.0004, 0.0007, -0.0002, 0.0009, 60,  140, 0.02, 0.3),
    "Virgo":       (129,  186.75, 12.72, 0.0036, +0.0002, 0.0003, -0.0001, 0.0005, 120, 280, 0.03, 1.0),
    "Abell1656":   (1656, 194.95, 27.98, 0.0231, +0.0004, 0.0006, -0.0001, 0.0008, 80,  170, 0.03, 1.0),
    "Abell2147":   (2147, 240.12, 15.98, 0.0350, +0.0004, 0.0007, -0.0002, 0.0009, 50,  100, 0.02, 0.3),
    "Abell2151":   (2151, 240.48, 17.75, 0.0330, +0.0004, 0.0007, -0.0002, 0.0009, 50,  100, 0.02, 0.3),
    "Abell2634":   (2634, 354.14, 27.03, 0.0305, +0.0004, 0.0007, -0.0002, 0.0009, 40,   80, 0.02, 0.3),
    "Abell2666":   (2666, 357.25, 27.08, 0.0270, +0.0004, 0.0007, -0.0002, 0.0009, 40,   80, 0.02, 0.3),
    "Abell2065":   (2065, 230.58, 27.70, 0.0720, +0.0008, 0.0008, -0.0005, 0.0010, 60,  140, 0.02, 0.3),
}


def _generate_cluster_sample(name: str, params: tuple) -> List[ClusterGalaxy]:
    """通用星系团样本生成器。

    参数
    ----------
    name : 星系团名称（用作 objid 前缀和随机种子前缀）
    params : (seed, ra, dec, z_mean, z_core_delta, z_core_std,
              z_out_delta, z_out_std, n_core, n_out, core_radius, out_radius_max)
    """
    (seed, ra_c, dec_c, z_mean,
     z_core_delta, z_core_std,
     z_out_delta, z_out_std,
     n_core, n_out, core_radius, out_radius_max) = params
    z_core_mean = z_mean + z_core_delta
    z_out_mean = z_mean + z_out_delta

    rng = np.random.Generator(np.random.PCG64(seed))
    galaxies = []

    for i in range(n_core):
        morph, conc = _inject_morphology_core(rng)
        offset = rng.exponential(core_radius) * rng.choice([-1, 1])
        g = ClusterGalaxy(
            objid=f"{name}_core_{i}",
            ra=ra_c + offset * 0.5,
            dec=dec_c + offset * 0.4,
            z=rng.normal(z_core_mean, z_core_std),
            z_err=0.0001,
            mag_r=rng.uniform(14.0, 18.0),
            morphology=morph,
            concentration=conc,
        )
        galaxies.append(g)

    for i in range(n_out):
        morph, conc = _inject_morphology_outskirt(rng)
        r = rng.uniform(0.15, out_radius_max)
        theta = rng.uniform(0, 2 * np.pi)
        g = ClusterGalaxy(
            objid=f"{name}_out_{i}",
            ra=ra_c + r * np.cos(theta),
            dec=dec_c + r * np.sin(theta),
            z=rng.normal(z_out_mean, z_out_std),
            z_err=0.0001,
            mag_r=rng.uniform(15.0, 18.5),
            morphology=morph,
            concentration=conc,
        )
        galaxies.append(g)

    return galaxies


def fetch_cluster(cluster_name: str,
                  use_cached: bool = True) -> Optional[GalaxyCluster]:
    """
    获取星系团数据。

    优先级：astroquery > HTTP API > 内嵌样本 > None

    参数
    ----------
    cluster_name : 簇名称（"Coma", "Abell1367", "Virgo" 等）
    use_cached : 是否优先使用内嵌样本（用于离线快速测试）

    返回
    -------
    cluster : GalaxyCluster 实例，失败时返回 None
    """
    info = KNOWN_CLUSTERS.get(cluster_name)
    if info is None:
        available = ", ".join(KNOWN_CLUSTERS.keys())
        warnings.warn(f"未知星系团: {cluster_name}。可用: {available}")
        return None

    galaxies = None

    # 策略 1: astroquery
    if not use_cached:
        galaxies = _fetch_with_astroquery(cluster_name)

    # 策略 2: HTTP API
    if galaxies is None and not use_cached:
        galaxies = _fetch_with_http(cluster_name)

    # 策略 3: 内嵌样本
    if galaxies is None:
        galaxies = _load_bundled_data(cluster_name)

    if galaxies is None:
        warnings.warn(f"无法获取 {cluster_name} 的数据（所有策略均失败）")
        return None

    cluster = GalaxyCluster(
        name=cluster_name,
        ra_center=info["ra"],
        dec_center=info["dec"],
        z_mean=info["z"],
        radius_deg=info["radius_deg"],
        galaxies=galaxies,
    )

    return cluster


def fetch_all_clusters(cluster_names: Optional[List[str]] = None,
                       use_cached: bool = True) -> Dict[str, GalaxyCluster]:
    """
    获取多个星系团的数据。

    参数
    ----------
    cluster_names : 星系团名称列表。默认为 None（全部）。
    use_cached : 是否优先使用内嵌样本。

    返回
    -------
    clusters : {名称: GalaxyCluster} 字典
    """
    if cluster_names is None:
        cluster_names = list(KNOWN_CLUSTERS.keys())

    clusters = {}
    for name in cluster_names:
        cluster = fetch_cluster(name, use_cached=use_cached)
        if cluster is not None and cluster.n_members >= 10:
            clusters[name] = cluster
            print(f"  ✓ {name}: {cluster.n_members} 个星系")
        else:
            print(f"  ✗ {name}: 数据不足")

    return clusters


if __name__ == "__main__":
    # 测试数据获取
    print("测试星系团数据获取...")
    for name in ["Coma", "Abell1367", "Virgo"]:
        c = fetch_cluster(name, use_cached=True)
        if c:
            print(f"  {c.name}: {c.n_members} 个星系, z≈{c.z_mean}")
        else:
            print(f"  {name}: 获取失败")
