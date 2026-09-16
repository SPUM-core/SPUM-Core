"""
SPUM 红移模型：红移的 σ 归约（**局域退化读数**）

核心公式：z = σ_source / σ_obs - 1

**适用范围（2026-09-16 口径统一）**：本式是**局域两端退化读数**，
只在"同一星系团内、两端 σ 差异即全部信息"的近似下使用，
**不是红移的通式**。红移的通式是**光子数据链的累积读数**：

    z ∝ Σ_k (σ(l_k) − σ₀)/σ₀   ⟺   ∫_γ (σ − σ₀)/σ₀ dl

即光子发射时封装完毕、途中不被改造，每跳向链上追加一条局部 σ 记录，
接收端读出整条链与本地标准帧时钟比对。把两端值之比当作通式，会在
质量场中（σ ∝ 1/r²）给出错误的 z ∝ 1/r²（观测为 1/r）。
判定依据：SPUM2610.md §9.1.3；术语基准：.trae/rules/spum-vocabulary.md §三第 6 条。

其中 σ = |P|/|ε| 是子网的空间密度。
在高 σ 区（密集连接），帧周期更长，本征频率更低 → 红移更大。

扩展模型包含非线性形式（幂律、指数、多项式），
用于逼近 f(σ) 映射函数的真正形式。
"""

import numpy as np
from typing import Union, Optional, Callable


def z_from_sigma(sigma_source: Union[float, np.ndarray],
                  sigma_obs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    从 σ 计算红移（线性模型）。

    SPUM 核心关系：红移 = σ 之比减 1。
    """
    return sigma_source / sigma_obs - 1.0


def sigma_from_z(z: Union[float, np.ndarray],
                  sigma_obs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    从红移反推源 σ。
    """
    return sigma_obs * (1.0 + z)


class SPUMRedshiftModel:
    """
    SPUM 红移模型。

    将星系团视为 ⟨P, ε⟩ 子网，其内部 σ 分布决定成员星系的
    本征频率。观测到的红移差异 = σ 差异的直接读数。

    核心假定：
    - 原子振荡频率 f ∝ 1/σ（帧周期反比于 σ）
    - 局域退化读数：z = f(σ_obs)/f(σ_source) - 1 = σ_source/σ_obs - 1
      （**仅限局域两端近似**；通式是链上累积跳数差 z ∝ Σ_k (σ(l_k)−σ₀)/σ₀）
    - 同一星系团内 σ 的差异仅来自局部环境密度

    一阶近似假设 f(σ) ∝ 1/σ 是最简形式。严格形式需从
    帧动力学推导（待推导 — SPUM2610.md §9.2）。
    """

    def __init__(self, sigma_obs: float = 1.0):
        self.sigma_obs = sigma_obs

    # ── 线性预测 ──

    def predict_z(self, sigma_source: np.ndarray) -> np.ndarray:
        """给定源 σ，预测红移 z（线性模型）。"""
        return z_from_sigma(sigma_source, self.sigma_obs)

    def predict_sigma(self, z: np.ndarray) -> np.ndarray:
        """给定观测红移 z，反推源 σ。"""
        return sigma_from_z(z, self.sigma_obs)

    def redshift_gradient(self, sigma_core: float,
                          sigma_outskirt: float) -> float:
        """计算星系团核心与外围之间的红移差。"""
        return float(self.predict_z(sigma_core)
                     - self.predict_z(sigma_outskirt))

    def sigma_field_from_z(self, z_map: np.ndarray) -> np.ndarray:
        """从红移场重建 σ 场。"""
        return self.predict_sigma(z_map)

    # ── 线性拟合 ──

    def fit_linear(self, sigma: np.ndarray, z: np.ndarray) -> dict:
        """线性模型: z = a * σ + b。"""
        from scipy import stats
        slope, intercept, r_val, p_val, std_err = stats.linregress(sigma, z)
        residuals = z - (slope * sigma + intercept)
        n = len(z)
        aic = n * np.log(np.mean(residuals ** 2)) + 2 * 2
        bic = n * np.log(np.mean(residuals ** 2)) + 2 * np.log(n)
        return {
            "model": "linear",
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_val ** 2,
            "p_value": p_val,
            "std_err": std_err,
            "aic": aic,
            "bic": bic,
            "n_points": n,
        }

    # ── 非线性拟合 ──

    def fit_power_law(self, sigma: np.ndarray, z: np.ndarray) -> dict:
        """
        幂律模型: z = a * σ^α + b。

        使用 curve_fit 拟合三个参数 (a, α, b)。
        当 z 与 σ 的关系偏离线性时捕获非线性行为。
        """
        from scipy.optimize import curve_fit

        def _power(s, a, alpha, b):
            return a * (s ** alpha) + b

        # 初始猜测：a=1, α=1, b=0（退化为线性）
        try:
            popt, pcov = curve_fit(
                _power, sigma, z,
                p0=[1.0, 1.0, 0.0],
                maxfev=10000
            )
            z_pred = _power(sigma, *popt)
            residuals = z - z_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((z - np.mean(z)) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)
            n = len(z)
            mse = ss_res / n
            aic = n * np.log(mse) + 2 * 3
            bic = n * np.log(mse) + 3 * np.log(n)
            return {
                "model": "power_law",
                "a": popt[0],
                "alpha": popt[1],
                "b": popt[2],
                "r_squared": r_squared,
                "aic": aic,
                "bic": bic,
                "n_points": n,
            }
        except Exception as e:
            return {"model": "power_law", "error": str(e),
                    "r_squared": -np.inf, "aic": np.inf, "bic": np.inf}

    def fit_exponential(self, sigma: np.ndarray, z: np.ndarray) -> dict:
        """
        指数模型: z = exp(a * σ + b) - 1。

        相当于对数空间中的线性关系：
        ln(z+1) = a * σ + b
        """
        from scipy.optimize import curve_fit

        def _exp(s, a, b):
            return np.exp(a * s + b) - 1.0

        try:
            # 初始猜测: a=1, b=0
            popt, pcov = curve_fit(
                _exp, sigma, z,
                p0=[1.0, 0.0],
                maxfev=10000
            )
            z_pred = _exp(sigma, *popt)
            residuals = z - z_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((z - np.mean(z)) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)
            n = len(z)
            mse = ss_res / n
            aic = n * np.log(mse) + 2 * 2
            bic = n * np.log(mse) + 2 * np.log(n)
            return {
                "model": "exponential",
                "a": popt[0],
                "b": popt[1],
                "r_squared": r_squared,
                "aic": aic,
                "bic": bic,
                "n_points": n,
            }
        except Exception as e:
            return {"model": "exponential", "error": str(e),
                    "r_squared": -np.inf, "aic": np.inf, "bic": np.inf}

    def fit_quadratic(self, sigma: np.ndarray, z: np.ndarray) -> dict:
        """
        二次多项式模型: z = a₀ + a₁σ + a₂σ²。

        捕获 z-σ 关系中的曲率。
        """
        from scipy.optimize import curve_fit

        def _quad(s, a0, a1, a2):
            return a0 + a1 * s + a2 * s ** 2

        try:
            popt, _ = curve_fit(
                _quad, sigma, z,
                p0=[0.0, 1.0, 0.0],
                maxfev=10000
            )
            z_pred = _quad(sigma, *popt)
            residuals = z - z_pred
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((z - np.mean(z)) ** 2)
            r_squared = 1 - ss_res / max(ss_tot, 1e-10)
            n = len(z)
            mse = ss_res / n
            aic = n * np.log(mse) + 2 * 3
            bic = n * np.log(mse) + 3 * np.log(n)
            return {
                "model": "quadratic",
                "a0": popt[0],
                "a1": popt[1],
                "a2": popt[2],
                "r_squared": r_squared,
                "aic": aic,
                "bic": bic,
                "n_points": n,
            }
        except Exception as e:
            return {"model": "quadratic", "error": str(e),
                    "r_squared": -np.inf, "aic": np.inf, "bic": np.inf}

    # ── 模型比较 ──

    def compare_models(self, sigma: np.ndarray, z: np.ndarray) -> dict:
        """
        拟合所有模型并比较。

        返回每个模型的 R²、AIC、BIC，
        以及最佳模型的判定。
        """
        models = {
            "linear": self.fit_linear(sigma, z),
            "power_law": self.fit_power_law(sigma, z),
            "exponential": self.fit_exponential(sigma, z),
            "quadratic": self.fit_quadratic(sigma, z),
        }

        # 选出最佳模型（AIC 最低）
        valid = {k: v for k, v in models.items()
                 if v.get("aic", np.inf) < np.inf}
        if valid:
            best = min(valid, key=lambda k: valid[k]["aic"])
            models["best"] = {
                "name": best,
                "aic": valid[best]["aic"],
                "r_squared": valid[best]["r_squared"],
            }
        else:
            models["best"] = {"name": "none", "aic": np.inf,
                              "r_squared": -np.inf}

        return models

    def fit_sigma_from_cluster(self, redshifts: np.ndarray,
                                sigma_norm: np.ndarray) -> dict:
        """
        调用 compare_models 完成拟合比较。
        兼容旧版接口名称。
        """
        return self.compare_models(sigma_norm, redshifts)
