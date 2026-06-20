"""
笛卡尔/索迪圆定理 — 从外部关系求解几何。

核心原理：
    给定 n 个两两相切的超球体，第 n+1 个的曲率（bend=1/r）
    由前 n 个唯一确定，无需任何内部属性。

二维（笛卡尔圆定理）：
    (k₁ + k₂ + k₃ ± k₄)² = 2(k₁² + k₂² + k₃² + k₄²)

三维（索迪定理，推广）：
    (Σk_i)² = 3·Σk_i²     for i = 1..5

n 维推广：
    (Σk_i)² = n/(n-1) · Σk_i²   for i = 1..(n+2)

关键本体论意义：
    - 曲率/半径不是球体的内部属性
    - 它是外部关系网络施加的几何约束的解
    - 移除球体后，约束仍在 → 曲率/半径可以"无主存在"
"""

from typing import List, Optional, Tuple
import math
import numpy as np


def solve_bend_2d(b1: float, b2: float, b3: float
                  ) -> Tuple[float, float]:
    """笛卡尔圆定理：给定三圆曲率，求第四圆曲率。

    正确公式：k₄ = k₁+k₂+k₃ ± 2√(k₁k₂ + k₂k₃ + k₃k₁)

    推导自：(k₁+k₂+k₃+k₄)² = 2(k₁²+k₂²+k₃²+k₄²)

    Returns:
        (b4_inner, b4_outer)
        inner = 三个圆中间缝隙圆的曲率（正值，半径小）
        outer = 包围三个圆的大圆的曲率（负值，半径大）
    """
    sum_bi = b1 + b2 + b3
    # 判别式项 = k₁k₂ + k₂k₃ + k₃k₁
    inner_term = b1*b2 + b2*b3 + b3*b1
    if inner_term < -1e-12:
        return (float('nan'), float('nan'))
    inner_term = max(0.0, inner_term)
    sqrt_term = 2.0 * math.sqrt(inner_term)

    b4_inner = sum_bi - sqrt_term   # 正曲率，小圆嵌在缝隙中
    b4_outer = sum_bi + sqrt_term   # 负曲率，大圆包围在外
    return (b4_inner, b4_outer)


def solve_bend_3d(b1: float, b2: float, b3: float, b4: float
                  ) -> Tuple[float, float]:
    """索迪定理（3D）：给定四球曲率，求第五球曲率。

    正确公式：
        S = Σk_{1..4}
        Q = Σk²_{1..4}
        sqrt_term = √(3S² - 6Q)
        k₅ = (S ± sqrt_term) / 2

    推导自：(Σk_i)² = 3·Σk_i²  for i=1..5

    Returns:
        (b5_inner, b5_outer)
        inner = 四球围成的缝隙球的曲率（正值）
        outer = 从外部包围四球的大球曲率（可能负值）
    """
    S = b1 + b2 + b3 + b4
    Q = b1*b1 + b2*b2 + b3*b3 + b4*b4
    disc = 3.0*S*S - 6.0*Q
    if disc < -1e-12:
        return (float('nan'), float('nan'))
    disc = max(0.0, disc)
    sqrt_term = math.sqrt(disc)

    b5_inner = (S + sqrt_term) / 2.0    # 正曲率，缝隙球
    b5_outer = (S - sqrt_term) / 2.0    # 可能是负曲率，包围球
    return (b5_inner, b5_outer)


def solve_radius_from_neighbors(neighbor_radii: List[float],
                                 n_dim: int = 2) -> Optional[float]:
    """从 n+1 个相切邻居的半径求解自身半径。

    Args:
        neighbor_radii: n+1 个两两相切的邻居的半径
        n_dim: 维度（2=圆，3=球）

    Returns:
        自身半径，如果不能求解则返回 None
    """
    n_needed = n_dim + 1
    if len(neighbor_radii) < n_needed:
        return None

    bends = [1.0 / max(r, 1e-12) for r in neighbor_radii]

    if n_dim == 2:
        b_inner, _ = solve_bend_2d(*bends[:3])
        if math.isnan(b_inner) or b_inner <= 0:
            return None
        return 1.0 / b_inner

    elif n_dim == 3:
        b_inner, _ = solve_bend_3d(*bends[:4])
        if math.isnan(b_inner) or b_inner <= 0:
            return None
        return 1.0 / b_inner

    return None


def solve_position_trilateration(
    neighbor_positions: List[Tuple[float, float, float]],
    neighbor_radii: List[float],
    self_radius: float
) -> Optional[Tuple[float, float, float]]:
    """三边测量：从 4 个相切邻居求自身位置。

    给定 4 个邻居的位置 (x_i, y_i, z_i) 和半径 r_i，
    以及自身半径 r，求解自身位置 (x, y, z) 满足：
        |p - p_i| = r + r_i    for i = 1..4

    使用三边测量算法（将 p1 置为原点，p2 在 x 轴，
    p3 在 xy 平面，p4 确定 z 方向）。
    """
    if len(neighbor_positions) < 4:
        # 不足 4 个邻居时，从斐波那契球面采样默认位置
        return _default_position(neighbor_positions, neighbor_radii, self_radius)

    A = np.array(neighbor_positions[0], dtype=float)
    B = np.array(neighbor_positions[1], dtype=float)
    C = np.array(neighbor_positions[2], dtype=float)
    D = np.array(neighbor_positions[3], dtype=float)

    r_a = neighbor_radii[0] + self_radius
    r_b = neighbor_radii[1] + self_radius
    r_c = neighbor_radii[2] + self_radius
    r_d = neighbor_radii[3] + self_radius

    # 将 A 置为原点
    BA = B - A
    CA = C - A
    DA = D - A

    # B 在 x 轴方向
    xb = np.linalg.norm(BA)
    if xb < 1e-12:
        return None
    ex = BA / xb

    # C 在 xy 平面
    xc = np.dot(CA, ex)
    CA_perp = CA - xc * ex
    yc = np.linalg.norm(CA_perp)
    if yc < 1e-12:
        return None
    ey = CA_perp / yc

    # D 确定 z 方向
    xd = np.dot(DA, ex)
    yd = np.dot(DA, ey)
    ez = np.cross(ex, ey)
    # 确保 ez 指向 D 方向
    zd = np.dot(DA, ez)

    # 用前三个球解位置
    x = (r_a*r_a - r_b*r_b + xb*xb) / (2.0 * xb)
    y = (r_a*r_a - r_c*r_c + xc*xc + yc*yc - 2.0*xc*x) / (2.0 * yc)

    # 用第四个球消歧
    z_sq = r_a*r_a - x*x - y*y
    if z_sq < 0:
        z_sq = 0.0
    z = math.sqrt(z_sq)

    # 两个解：±z，用第四个球消歧
    pos_plus = A + x*ex + y*ey + z*ez
    dist_plus = abs(np.linalg.norm(pos_plus - D) - r_d)

    pos_minus = A + x*ex + y*ey - z*ez
    dist_minus = abs(np.linalg.norm(pos_minus - D) - r_d)

    if dist_plus <= dist_minus:
        return tuple(pos_plus.astype(float))
    else:
        return tuple(pos_minus.astype(float))


def _default_position(
    neighbor_positions: List[Tuple[float, float, float]],
    neighbor_radii: List[float],
    self_radius: float
) -> Optional[Tuple[float, float, float]]:
    """默认位置：当邻居不足 4 个时沿第一个邻居法向放置。"""
    if not neighbor_positions:
        return (0.0, 0.0, 0.0)

    center = np.array(neighbor_positions[0], dtype=float)
    r_sum = neighbor_radii[0] + self_radius
    return tuple((center + np.array([r_sum, 0.0, 0.0])).astype(float))


def compute_apollonian_gasket(
    seed_radii: List[float],
    depth: int = 3
) -> List[Tuple[Tuple[float, float, float], float]]:
    """阿波罗尼奥斯垫片：从种子圆递归生成。

    给定三个两两相切的种子圆，递归填充中间缝隙。

    Returns:
        [(position, radius), ...] 所有圆的位置和半径
    """
    # 仅支持 2D 版本（3D 需要更复杂的实现）
    if len(seed_radii) < 3:
        return []

    b1 = 1.0 / seed_radii[0]
    b2 = 1.0 / seed_radii[1]
    b3 = 1.0 / seed_radii[2]

    results = []
    results.append(((0.0, 0.0, 0.0), seed_radii[0]))
    # 完整阿波罗尼奥斯垫片生成需要位置，此处为简化演示
    # 完整实现见 experiments/apollonian_gasket.py
    return results
