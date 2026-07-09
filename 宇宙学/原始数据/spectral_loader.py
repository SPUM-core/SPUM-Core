"""
SPUM 宇宙学 — 原始光谱加载器

从 SDSS SkyServer 加载原始光谱数据，过滤所有 ΛCDM pipeline 输出。
仅保留：flux 阵列、wavelength 阵列、RA/Dec 指向、MJD/plate/fiberID。

不读取：z, z_err, class, subclass, veldisp, snMedian 等。

用法：
    from spectral_loader import load_spectra_by_coords
    
    # 以 Coma 星系团为中心，半径 0.5 度
    spectra = load_spectra_by_coords(ra=194.95, dec=27.98, radius_deg=0.5)
    
    # 获取指纹向量
    fingerprints = build_fingerprints(spectra, n_bins=100)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings


# ── 数据类（不含任何 ΛCDM 字段） ──

class RawSpectrum:
    """一条原始光谱，仅包含仪器直接输出和位置信息。"""
    
    def __init__(self, flux: np.ndarray, wavelength: np.ndarray,
                 ra: float, dec: float,
                 mjd: int, plate: int, fiber: int,
                 spectrum_id: str = ""):
        self.flux = flux
        self.wavelength = wavelength
        self.ra = ra
        self.dec = dec
        self.mjd = mjd
        self.plate = plate
        self.fiber = fiber
        self.spectrum_id = spectrum_id or f"{plate}-{mjd}-{fiber}"
    
    @property
    def n_pixels(self) -> int:
        return len(self.flux)
    
    @property
    def wavelength_range(self) -> Tuple[float, float]:
        return (float(self.wavelength[0]), float(self.wavelength[-1]))


# ── SDSS 加载器 ──

def load_spectra_by_coords(ra: float, dec: float, radius_deg: float = 0.5,
                           max_spectra: int = 500,
                           use_mock: bool = False) -> List[RawSpectrum]:
    """
    在给定 RA/Dec 坐标周围加载原始光谱。
    
    参数
    ----------
    ra, dec : 指向中心坐标（度）
    radius_deg : 搜索半径（度）
    max_spectra : 最大加载数
    use_mock : 是否使用内嵌模拟数据（用于离线测试）
    
    返回
    -------
    spectra : List[RawSpectrum] — 仅含原始光谱数据，不含 z
    """
    if not use_mock:
        return _load_sdss_spectra(ra, dec, radius_deg, max_spectra)
    else:
        return _generate_mock_spectra(ra, dec, radius_deg, max_spectra)


def _load_sdss_spectra(ra: float, dec: float, radius_deg: float,
                       max_spectra: int) -> List[RawSpectrum]:
    """
    从 SDSS SkyServer 加载原始光谱。
    
    使用 astroquery.sdss，但拒绝读取任何 z 相关字段。
    仅读取：flux, wavelength, ra, dec, mjd, plate, fiberID
    """
    try:
        from astroquery.sdss import SDSS
        SDSS.TIMEOUT = 120
    except ImportError:
        warnings.warn("astroquery 未安装，降级到模拟数据")
        return _generate_mock_spectra(ra, dec, radius_deg, max_spectra)
    
    from astropy.io import fits
    from astropy import units as u
    
    # 查询光谱 — 只拉取位置信息，不请求 z
    # 使用 SDSS 的 specObj 查询但只获取身份信息
    coord_ra_deg = ra
    coord_dec_deg = dec
    radius_arcmin = min(radius_deg * 60.0, 2.9)  # SDSS SkyServer 限制 < 3 角分
    
    try:
        from astropy.coordinates import SkyCoord
        # 先用 query_region 获取光谱元数据（plate, mjd, fiberID）
        coords = SkyCoord(ra=coord_ra_deg, dec=coord_dec_deg, 
                          unit=(u.deg, u.deg), frame='icrs')
        tab = SDSS.query_region(coords, radius=radius_arcmin * u.arcmin,
                                spectro=True, timeout=60)
        if tab is None or len(tab) == 0:
            warnings.warn("SDSS 查询未返回结果")
            return []
        
        # 限制数量
        if len(tab) > max_spectra:
            tab = tab[:max_spectra]
        
        spectra = []
        for row in tab:
            try:
                plate = int(row['plate'])
                mjd = int(row['mjd'])
                fiber = int(row['fiberID'])
                
                spectra_raw = SDSS.get_spectra(
                    plate=plate, mjd=mjd, fiberID=fiber,
                    timeout=60,
                )
                
                if spectra_raw is None or len(spectra_raw) == 0:
                    continue
                
                # astroquery 版本差异处理
                sp = spectra_raw[0]
                if isinstance(sp, tuple) and len(sp) >= 2:
                    hdu = sp[0]
                    meta = sp[1] if isinstance(sp[1], dict) else {}
                else:
                    hdu = sp
                    meta = {}
                
                # ── DR17 FITS 格式 ──
                # ext 0 (PRIMARY): header only (COEFF0, COEFF1, RA, DEC), data=None
                # ext 1 (COADD):   BinTable, 列: flux, loglam, ivar, andmask, ...
                try:
                    if (len(hdu) > 1 and 
                        hdu[1].header.get('XTENSION') == 'BINTABLE' and
                        hdu[1].data is not None):
                        # DR17+ 格式
                        coadd = hdu[1].data
                        flux = coadd['flux']
                        loglam = coadd['loglam']      # 已经是 log10(波长)
                        wavelength = 10 ** loglam
                    else:
                        # 旧格式: PRIMARY 有数据
                        flux = hdu[0].data
                        loglam_hdr = hdu[0].header.get('COEFF0', None)
                        loglam_delta = hdu[0].header.get('COEFF1', None)
                        if loglam_hdr is None or loglam_delta is None:
                            continue
                        n_pix = len(flux)
                        loglam = loglam_hdr + loglam_delta * np.arange(n_pix)
                        wavelength = 10 ** loglam
                except (KeyError, IndexError, TypeError) as e:
                    continue
                
                # 位置信息（PRIMARY header 中）
                obj_ra = hdu[0].header.get('RA', float(row.get('ra', ra)))
                obj_dec = hdu[0].header.get('DEC', float(row.get('dec', dec)))
                
                spec = RawSpectrum(
                    flux=flux.astype(np.float64),
                    wavelength=wavelength.astype(np.float64),
                    ra=float(obj_ra),
                    dec=float(obj_dec),
                    mjd=mjd,
                    plate=plate,
                    fiber=fiber,
                )
                spectra.append(spec)
                
            except Exception as e:
                continue
        
        print(f"      成功加载 {len(spectra)}/{len(tab)} 条光谱")
        return spectra
        
    except Exception as e:
        warnings.warn(f"SDSS 查询失败: {e}")
        return []


# ── 模拟光谱生成（离线测试用） ──

def _generate_mock_spectra(ra_center: float, dec_center: float,
                            radius_deg: float, n: int) -> List[RawSpectrum]:
    """
    生成模拟光谱用于离线测试。
    
    模拟包含真实的光谱多样性：
    - 连续谱斜率变化（幂律指数 -1.2 ~ -2.5）
    - 发射线强度变化（按光谱类型分组）
    - 吸收特征（CaII H&K, NaI D）
    - 空间梯度：核心偏红 + 核心谱型收敛
    - 噪声水平按信噪比变化
    """
    base_rng = np.random.Generator(np.random.PCG64(42))
    spectra = []
    
    # 波长网格（SDSS 范围）
    lam = np.linspace(3800, 9200, 4000)
    
    # 谱线参数（实验室波长，Å）
    emission_lines = {
        'Hβ':      (4861.3, (0.2, 1.5)),
        '[OIII]':  (4958.9, (0.1, 0.8)),
        '[OIII]2': (5006.8, (0.3, 2.0)),
        '[NII]':   (6548.1, (0.1, 0.6)),
        'Hα':      (6562.8, (0.5, 3.0)),
        '[NII]2':  (6583.5, (0.2, 1.2)),
        '[SII]':   (6716.4, (0.1, 0.5)),
        '[SII]2':  (6730.8, (0.1, 0.5)),
    }
    # 吸收线
    absorption_lines = {
        'CaII_K': (3933.7, (0.3, 0.7)),
        'CaII_H': (3968.5, (0.2, 0.5)),
        'NaI_D':  (5890.0, (0.1, 0.4)),
        'NaI_D2': (5895.6, (0.1, 0.4)),
    }
    
    for i in range(n):
        rng = np.random.Generator(np.random.PCG64(42 + i * 7))
        
        # 位置 (均匀分布盘)
        r = radius_deg * np.sqrt(rng.uniform())
        theta = rng.uniform(0, 2 * np.pi)
        ra = ra_center + r * np.cos(theta) * 0.8
        dec = dec_center + r * np.sin(theta) * 0.6
        r_norm = r / radius_deg  # [0, 1]，0=核心
        
        # ── 连续谱 ──
        # 核心：偏红（幂律指数小，红端强）
        # 外围：偏蓝（幂律指数大，蓝端强）
        power_idx = -1.2 + r_norm * 1.3  # 核心 -1.2, 外围 -2.5
        power_idx += rng.uniform(-0.3, 0.3)  # 个体差异
        continuum = (lam / 5000) ** power_idx
        
        # 调光强度（总通量随半径略降）
        brightness = 1.0 + rng.uniform(-0.3, 0.3) * (1.0 - r_norm * 0.5)
        continuum *= brightness
        
        # ── 发射线 ──
        # 核心：Hα 强，[OIII] 强（类 AGN 型）
        # 外围：Hα 强，[OIII] 弱（类恒星形成型）
        emission = np.zeros_like(lam)
        redshift_factor = 1.0 + (1.0 - r_norm) * 2e-4  # 核心偏红
        
        for line_name, (lam0, (strength_min, strength_max)) in emission_lines.items():
            lam_shifted = lam0 * redshift_factor
            
            # 核心外围强度梯度
            if 'OIII' in line_name or 'SII' in line_name:
                # 高激发线 → 核心更强
                strength_factor = 1.0 + (1.0 - r_norm) * 1.0
            else:
                # 低激发线 → 均匀
                strength_factor = 1.0
            
            strength = rng.uniform(strength_min, strength_max) * strength_factor
            sigma = 2.0 + rng.uniform(0.5, 2.0)
            gauss = strength * np.exp(-0.5 * ((lam - lam_shifted) / sigma) ** 2)
            emission += gauss
        
        # ── 吸收线 ──
        absorption = np.ones_like(lam)
        for line_name, (lam0, (depth_min, depth_max)) in absorption_lines.items():
            lam_shifted = lam0 * redshift_factor
            depth = rng.uniform(depth_min, depth_max)
            sigma = 1.5 + rng.uniform(0.5, 1.5)
            absorption -= depth * np.exp(-0.5 * ((lam - lam_shifted) / sigma) ** 2)
        
        # ── 合成 ──
        flux = continuum * absorption + emission
        
        # 噪声（信噪比在核心高，外围低）
        snr = rng.uniform(15, 40) * (1.0 + (1.0 - r_norm) * 0.3)
        noise_level = np.mean(flux) / snr
        flux += rng.normal(0, noise_level, len(lam))
        flux = np.maximum(flux, 0)
        
        spec = RawSpectrum(
            flux=flux,
            wavelength=lam.copy(),
            ra=ra, dec=dec,
            mjd=58000 + i // 100, plate=i // 50, fiber=i,
            spectrum_id=f"mock_comA_{i}",
        )
        spectra.append(spec)
    
    return spectra


# ── 指纹向量构建 ──

def build_fingerprints(spectra: List[RawSpectrum],
                        n_bins: int = 100,
                        lam_min: float = 3850.0,
                        lam_max: float = 9150.0) -> Dict:
    """
    从原始光谱构建指纹向量数据集。
    
    步骤：
    1. 统一波长网格
    2. 插值每条光谱到统一网格
    3. 总通量归一化（消除亮度差异）
    4. 谱段 bin 降维
    
    参数
    ----------
    spectra : 原始光谱列表
    n_bins : 降维目标维数
    lam_min, lam_max : 波长范围
    
    返回
    -------
    dataset : {
        "metadata": [{"ra", "dec", "mjd", "plate", "fiber", "id"}],
        "fingerprints": np.ndarray (n_spectra, n_bins),
        "wavelength_grid": 1D array (统一波长网格),
        "bin_edges": 波长 bin 边界,
    }
    不含任何红移相关的字段。
    """
    if not spectra:
        return {"metadata": [], "fingerprints": np.array([]),
                "wavelength_grid": np.array([]), "bin_edges": np.array([])}
    
    # 统一波长网格
    lam_grid = np.linspace(lam_min, lam_max, 5000)
    
    # 谱段 bin
    bin_edges = np.linspace(lam_min, lam_max, n_bins + 1)
    
    # 插值并归一化
    n_spectra = len(spectra)
    fingerprints = np.zeros((n_spectra, n_bins))
    metadata = []
    
    for i, spec in enumerate(spectra):
        # 插值到统一网格
        flux_interp = np.interp(lam_grid, spec.wavelength, spec.flux,
                                 left=0.0, right=0.0)
        
        # 总通量归一化（保留谱型，消除亮度）
        total_flux = np.trapezoid(flux_interp, lam_grid)
        if total_flux > 0:
            flux_norm = flux_interp / total_flux
        else:
            flux_norm = flux_interp
        
        # 谱段 bin 降维
        for b in range(n_bins):
            mask = (lam_grid >= bin_edges[b]) & (lam_grid < bin_edges[b + 1])
            if np.any(mask):
                fingerprints[i, b] = np.mean(flux_norm[mask])
            else:
                fingerprints[i, b] = 0.0
        
        # 元数据（不含 z）
        metadata.append({
            "ra": spec.ra,
            "dec": spec.dec,
            "mjd": spec.mjd,
            "plate": spec.plate,
            "fiber": spec.fiber,
            "id": spec.spectrum_id,
        })
    
    return {
        "metadata": metadata,
        "fingerprints": fingerprints,
        "wavelength_grid": lam_grid,
        "bin_edges": bin_edges,
    }


# ── 双通道指纹构建（σ 独立约束） ──

def build_dual_fingerprints(spectra: List[RawSpectrum],
                              lam_min: float = 3800.0,
                              lam_max: float = 9200.0) -> Dict:
    """
    从原始光谱构建两组独立的指纹向量。
    
    双通道设计（关键约束—详见 地基/光谱指纹管道.md）：
      σ ← 指纹_A（连续谱）  ← 不含窄线信息
      谱偏移 ← 指纹_B（全谱） ← 含窄线信息
    
    参数
    ----------
    spectra : 原始光谱列表
    lam_min, lam_max : 波长范围
    
    返回
    -------
    dataset : {
        "metadata": [{"ra", "dec", "mjd", "plate", "fiber", "id"}],
        "fingerprint_continuum": (n, 50)  → σ 计算,
        "fingerprint_full": (n, 200)      → 偏移测量,
        "continuum_smoothed": [(n, 5400)],
        "wavelength_grid": 1D array,
    }
    """
    if not spectra:
        return {"metadata": [], "fingerprint_continuum": np.array([]),
                "fingerprint_full": np.array([])}

    # 统一波长网格 (Δλ = 1Å)
    lam_grid = np.linspace(lam_min, lam_max, int(lam_max - lam_min))
    dlam = lam_grid[1] - lam_grid[0]
    
    # 连续谱 bin 边界 (50 bins, each ~108Å)
    n_bins_cont = 50
    bin_edges_cont = np.linspace(lam_min, lam_max, n_bins_cont + 1)
    
    # 全谱 bin 边界 (200 bins, each ~27Å)
    n_bins_full = 200
    bin_edges_full = np.linspace(lam_min, lam_max, n_bins_full + 1)
    
    n_spectra = len(spectra)
    fp_continuum = np.zeros((n_spectra, n_bins_cont))
    fp_full = np.zeros((n_spectra, n_bins_full))
    metadata = []
    continuum_smoothed = []
    
    for i, spec in enumerate(spectra):
        # 插值到统一网格
        flux_interp = np.interp(lam_grid, spec.wavelength, spec.flux,
                                 left=0.0, right=0.0)
        
        # ── 指纹_A：连续谱（用于 σ） ──
        # 高斯平滑 σ=150Å 抹平窄线
        sigma_pixels = 150.0 / dlam  # dlam ≈ 1Å/像素
        try:
            from scipy.ndimage import gaussian_filter1d
            flux_cont = gaussian_filter1d(flux_interp, sigma=sigma_pixels,
                                           mode='constant', cval=0.0)
        except ImportError:
            # 回退：简单盒状平滑
            kernel_width = int(sigma_pixels * 3)
            kernel = np.ones(kernel_width) / kernel_width
            flux_cont = np.convolve(flux_interp, kernel, mode='same')
        
        # 归一化（保留连续谱形状，消除亮度差异）
        total_cont = np.trapezoid(flux_cont, lam_grid)
        if total_cont > 0:
            flux_cont_norm = flux_cont / total_cont
        else:
            flux_cont_norm = flux_cont
        
        # 谱段 bin
        for b in range(n_bins_cont):
            mask = (lam_grid >= bin_edges_cont[b]) & (lam_grid < bin_edges_cont[b + 1])
            if np.any(mask):
                fp_continuum[i, b] = np.mean(flux_cont_norm[mask])
        
        # ── 指纹_B：全谱（用于谱线偏移测量） ──
        total_full = np.trapezoid(flux_interp, lam_grid)
        if total_full > 0:
            flux_full_norm = flux_interp / total_full
        else:
            flux_full_norm = flux_interp
        
        for b in range(n_bins_full):
            mask = (lam_grid >= bin_edges_full[b]) & (lam_grid < bin_edges_full[b + 1])
            if np.any(mask):
                fp_full[i, b] = np.mean(flux_full_norm[mask])
        
        metadata.append({
            "ra": spec.ra, "dec": spec.dec,
            "mjd": spec.mjd, "plate": spec.plate,
            "fiber": spec.fiber, "id": spec.spectrum_id,
        })
        continuum_smoothed.append(flux_cont)
    
    return {
        "metadata": metadata,
        "fingerprint_continuum": fp_continuum,   # (n, 50)
        "fingerprint_full": fp_full,              # (n, 200)
        "continuum_smoothed": continuum_smoothed,
        "wavelength_grid": lam_grid,
    }


# ── 自检 ──

def self_check():
    """运行自检：验证加载器不包含任何 z 相关字段。"""
    print("=" * 50)
    print("SPUM 原始光谱加载器 — 自检")
    print("=" * 50)
    
    # 检查 RawSpectrum 的字段
    forbidden = ['z', 'redshift', 'veldisp', 'velocity']
    fields = [f for f in dir(RawSpectrum) if not f.startswith('_')]
    violations = [f for f in fields if any(v in f.lower() for v in forbidden)]
    
    if violations:
        print(f"⚠ 违反地基原则的字段: {violations}")
    else:
        print("✅ RawSpectrum 不含 ΛCDM 字段")
    
    # 测试加载
    print("\n测试模拟数据加载...")
    mock = load_spectra_by_coords(194.95, 27.98, 0.5, max_spectra=50, use_mock=True)
    print(f"  加载 {len(mock)} 条模拟光谱")
    
    fp = build_fingerprints(mock, n_bins=100)
    print(f"  指纹矩阵形状: {fp['fingerprints'].shape}")
    
    # 验证输出不包含 z
    for key in fp:
        if 'z' in key.lower() or 'redshift' in key.lower():
            print(f"⚠ 指纹输出包含 ΛCDM 字段: {key}")
    
    print("✅ 自检完成")
    return True


if __name__ == "__main__":
    self_check()
