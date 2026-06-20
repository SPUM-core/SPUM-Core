"""
VSPT 三律验证器 — 确认 VSPT 生长满足 SPUM 几何三律

VSPT 几何三律（元素化学/SPUM-VSPT.md §3.3）：
    1. k≥3（最小连通度）：每个分支节点至少被 3 个邻接关系锚定
    2. 分支角 ≈ 120°：分支角来自正二十面体二面角
    3. ρ∝r⁻³（密度幂律）：节点数密度沿径向 r⁻³ 衰减

诚实性说明：
    Law 1 (k≥3)：当前 VSPT 纯三角网格生长默认满足 k≥3。
                 容差 0.1（90%）较为合理，边缘节点因横向连接不足可能有少量违规。
    Law 2 (120°)：由三角网格几何精确实现，应 100% 通过。
    Law 3 (ρ∝r⁻³)：当前 VSPT 为纯 3-分支三角网格生长（无随机存活衰减），
                    节点数随壳层指数增长，密度幂律自然不成立。
                    此律是预期不能通过的——它标志 VSPT 需要电子耦合等衰减机制。
    
    验证器设计原则：容差反映对 SPUM 几何约束的置信度，而非反向适配模拟输出。
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Phase_4.vspt_growth import VSPTNode, VSPTTree, VSPTEngine


# ============================================================
# Law 1: k≥3 最小连通度
# ============================================================

def validate_k3_law(
    trees: Dict[str, VSPTTree],
    min_k: int = 3,
    tolerance: float = 0.2,
) -> Dict:
    """验证 VSPT 节点度数是否满足 k≥3。

    Law 1（SPUM-VSPT.md §3.3）：
        每个分支节点至少被 3 个邻接关系锚定，满足局部拓扑刚性。

    Args:
        trees:     VSPT 树字典
        min_k:     最小度数（默认 3）
        tolerance: 允许的未达标节点比例

    Returns:
        {is_valid, degree_distribution, mean_degree,
         met_ratio, violations, ...}
    """
    all_nodes: List[VSPTNode] = []
    for tree in trees.values():
        all_nodes.extend(tree.nodes.values())

    if not all_nodes:
        return {"is_valid": False, "n_nodes": 0, "message": "无节点"}

    # 度数分布（仅统计非叶子节点：至少有一个子节点的分支节点）
    # Law 1 的"分支节点"指实际参与分岔的节点，不是末端叶子
    interior_nodes = [n for n in all_nodes if n.layer > 0 and len(n.children) > 0]
    if not interior_nodes:
        return {"is_valid": True, "n_nodes": 0, "mean_degree": 0,
                "message": "仅根节点，无法验证 Law 1"}

    degree_dist = {}
    for n in interior_nodes:
        degree_dist[n.uid] = n.degree

    # 统计
    values = list(degree_dist.values())
    mean_deg = sum(values) / len(values) if values else 0
    max_deg = max(values) if values else 0
    min_deg = min(values) if values else 0

    met = sum(1 for d in values if d >= min_k)
    met_ratio = met / len(values) if values else 0

    violations = [uid for uid, d in degree_dist.items() if d < min_k]

    return {
        "is_valid": met_ratio >= (1.0 - tolerance),
        "n_nodes": len(interior_nodes),
        "mean_degree": round(mean_deg, 3),
        "max_degree": max_deg,
        "min_degree": min_deg,
        f"k≥{min_k}_ratio": round(met_ratio, 4),
        "violations": violations[:20],  # 前 20 个违规
        "degree_distribution": {
            str(k): sum(1 for d in values if d == k)
            for k in sorted(set(values))
        },
        "law_name": f"k≥{min_k}",
        "message": (
            f"通过 ✓ (k≥{min_k} 比例 = {met_ratio:.1%})"
            if met_ratio >= (1.0 - tolerance)
            else f"未通过 (k≥{min_k} 比例 = {met_ratio:.1%} < {1.0-tolerance:.0%})"
        ),
    }


# ============================================================
# Law 2: 分支角 120°
# ============================================================

def validate_120_angle(
    trees: Dict[str, VSPTTree],
    target_angle: float = 120.0,
    angle_tolerance: float = 5.0,
    min_pass_ratio: float = 0.8,
) -> Dict:
    """验证 VSPT 分支角 ≈ 120°。

    Law 2（SPUM-VSPT.md §3.3）：
        分支角锁定为正二十面体二面角 ≈ 120°，
        球面拓扑 ∫κg dA = 4π 的唯一稳定解。

    Args:
        trees:           VSPT 树字典
        target_angle:    目标分支角（度，默认 120°）
        angle_tolerance: 允许偏差（度，默认 ±5°）
        min_pass_ratio:  在容差内的最小比例（默认 0.8）

    Returns:
        {is_valid, mean_angle, std_angle, angle_distribution, ...}
    """
    angles = []
    node_positions = {}

    for tree in trees.values():
        for uid, node in tree.nodes.items():
            node_positions[uid] = node.position

    for tree in trees.values():
        for uid, node in tree.nodes.items():
            # 跳过根节点和叶子节点（没有足够子节点）
            if len(node.children) < 2:
                continue

            # 取前两个子节点，计算它们之间的角度（从父节点视角）
            c1 = tree.nodes.get(node.children[0])
            c2 = tree.nodes.get(node.children[1])
            if c1 is None or c2 is None:
                continue

            p_pos = node.position
            # 父节点的径向法线（球面法线 = 位置方向）
            p_r = math.sqrt(p_pos[0]**2 + p_pos[1]**2 + p_pos[2]**2)
            if p_r < 1e-12:
                continue
            normal = (p_pos[0]/p_r, p_pos[1]/p_r, p_pos[2]/p_r)

            # 子→父向量
            v1 = tuple(c1.position[i] - p_pos[i] for i in range(3))
            v2 = tuple(c2.position[i] - p_pos[i] for i in range(3))

            # 投影到切平面（去除径向分量）
            dot1 = v1[0]*normal[0] + v1[1]*normal[1] + v1[2]*normal[2]
            dot2 = v2[0]*normal[0] + v2[1]*normal[1] + v2[2]*normal[2]
            v1p = (v1[0] - dot1*normal[0], v1[1] - dot1*normal[1], v1[2] - dot1*normal[2])
            v2p = (v2[0] - dot2*normal[0], v2[1] - dot2*normal[1], v2[2] - dot2*normal[2])

            n1 = math.sqrt(v1p[0]**2 + v1p[1]**2 + v1p[2]**2)
            n2 = math.sqrt(v2p[0]**2 + v2p[1]**2 + v2p[2]**2)

            if n1 < 1e-12 or n2 < 1e-12:
                continue

            cos_angle = (v1p[0]*v2p[0] + v1p[1]*v2p[1] + v1p[2]*v2p[2]) / (n1 * n2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_deg = math.degrees(math.acos(cos_angle))
            angles.append(angle_deg)

    if not angles:
        return {"is_valid": False, "n_angles": 0, "message": "无足够节点计算分支角"}

    mean_angle = sum(angles) / len(angles)
    variance = sum((a - mean_angle) ** 2 for a in angles) / len(angles)
    std_angle = math.sqrt(variance)

    within_tolerance = sum(
        1 for a in angles
        if abs(a - target_angle) <= angle_tolerance
    )
    pass_ratio = within_tolerance / len(angles)

    return {
        "is_valid": pass_ratio >= min_pass_ratio,
        "n_angles": len(angles),
        "mean_angle": round(mean_angle, 2),
        "std_angle": round(std_angle, 2),
        "target_angle": target_angle,
        "tolerance": angle_tolerance,
        "min_pass_ratio": min_pass_ratio,
        "within_tolerance": within_tolerance,
        "pass_ratio": round(pass_ratio, 4),
        "min_angle": round(min(angles), 2),
        "max_angle": round(max(angles), 2),
        "law_name": "分支角 120°",
        "message": (
            f"通过 ✓ (均值 {mean_angle:.1f}° ± {std_angle:.1f}°, "
            f"{pass_ratio:.0%} 在容差内)"
            if pass_ratio >= min_pass_ratio
            else f"未通过 (均值 {mean_angle:.1f}°, 仅 {pass_ratio:.0%} 在容差内)"
        ),
    }


# ============================================================
# Law 3: ρ ∝ r⁻³ 密度幂律
# ============================================================

def validate_density_power_law(
    trees: Dict[str, VSPTTree],
    target_exponent: float = -3.0,
    exponent_tolerance: float = 0.5,
) -> Dict:
    """验证 VSPT 节点密度沿径向 ρ∝r⁻³ 衰减。

    Law 3（SPUM-VSPT.md §3.3）：
        从中心点源向外生长的分形树状网络在球对称约束下的唯一自洽解。

    Args:
        trees:              VSPT 树字典
        target_exponent:    目标幂指数（默认 -3）
        exponent_tolerance: 允许的指数偏差（默认 ±0.5）

    Returns:
        {is_valid, fitted_exponent, r_squared, layer_density, ...}
    """
    # 按壳层统计节点数和平均半径
    layer_data = {}  # layer -> (count, avg_radius, min_radius, max_radius)

    for tree in trees.values():
        for node in tree.nodes.values():
            layer = node.layer
            r = node.radius
            if layer not in layer_data:
                layer_data[layer] = {"count": 0, "r_sum": 0.0}
            layer_data[layer]["count"] += 1
            layer_data[layer]["r_sum"] += r

    if len(layer_data) < 3:
        return {"is_valid": False, "n_layers": len(layer_data),
                "message": "需要至少 3 个壳层进行幂律拟合"}

    # 构建拟合数据
    layers = sorted(layer_data.keys())
    radii = []
    densities = []
    for l in layers:
        d = layer_data[l]
        avg_r = d["r_sum"] / d["count"]
        # 壳密度 = 该层节点数 / 该层球壳体积
        # 球壳体积 ≈ 4πr²·Δr，Δr ≈ 步长（相邻壳层间距）
        # 简化：用 4πr² 做面积归一化（假设 Δr 恒定）
        # 这样密度单位实际是"节点数/单位球面积"，趋势等价
        shell_area = 4.0 * math.pi * (avg_r ** 2) if avg_r > 1e-12 else 1.0
        density = d["count"] / shell_area
        radii.append(avg_r)
        densities.append(density)

    # 线性拟合 log(ρ) = a * log(r) + b → 指数 = a
    log_radii = [math.log(r) for r in radii]
    log_densities = [math.log(max(d, 1e-30)) for d in densities]

    n = len(log_radii)
    mean_x = sum(log_radii) / n
    mean_y = sum(log_densities) / n

    # 最小二乘
    num = sum((log_radii[i] - mean_x) * (log_densities[i] - mean_y) for i in range(n))
    den = sum((log_radii[i] - mean_x) ** 2 for i in range(n))
    exponent = num / den if den != 0 else 0.0
    intercept = mean_y - exponent * mean_x

    # R²
    ss_res = sum(
        (log_densities[i] - (exponent * log_radii[i] + intercept)) ** 2
        for i in range(n)
    )
    ss_tot = sum((log_densities[i] - mean_y) ** 2 for i in range(n))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-30 else 0.0

    exponent_diff = abs(exponent - target_exponent)

    return {
        "is_valid": exponent_diff <= exponent_tolerance,
        "n_layers": n,
        "fitted_exponent": round(exponent, 4),
        "target_exponent": target_exponent,
        "deviation": round(exponent_diff, 4),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "layer_density": {
            str(l): {
                "avg_radius": round(layer_data[l]["r_sum"] / layer_data[l]["count"], 4),
                "count": layer_data[l]["count"],
            }
            for l in layers
        },
        "law_name": "ρ ∝ r⁻³",
        "message": (
            f"通过 ✓ (拟合指数 {exponent:.2f}, R²={r_squared:.3f})"
            if exponent_diff <= exponent_tolerance
            else f"未通过 (拟合指数 {exponent:.2f}, 偏差 {exponent_diff:.2f})"
        ),
    }


# ============================================================
# 完整验证器
# ============================================================

@dataclass
class VSPTValidator:
    """VSPT 三律完整验证器。"""

    k3_tolerance: float = 0.1        # k≥3: 允许 ≤10% 节点不达标
    angle_tolerance: float = 5.0     # 120°: 允许 ±5° 偏差
    exponent_tolerance: float = 0.5   # ρ∝r⁻³: 允许 ±0.5（此律当前预期不通过）

    def validate_all(
        self,
        trees: Dict[str, VSPTTree],
    ) -> Dict:
        """执行 VSPT 三律完整验证。

        Returns:
            {law1, law2, law3, is_valid, summary}
        """
        law1 = validate_k3_law(trees, tolerance=self.k3_tolerance)
        law2 = validate_120_angle(trees, angle_tolerance=self.angle_tolerance)
        law3 = validate_density_power_law(
            trees, exponent_tolerance=self.exponent_tolerance
        )

        all_valid = law1["is_valid"] and law2["is_valid"] and law3["is_valid"]

        return {
            "law1_k3": law1,
            "law2_120": law2,
            "law3_density": law3,
            "is_valid": all_valid,
            "summary": (
                "VSPT 几何三律全部通过 ✓"
                if all_valid
                else f"VSPT 三律未全部通过: "
                     f"{'k≥3 ' if not law1['is_valid'] else ''}"
                     f"{'分支角 ' if not law2['is_valid'] else ''}"
                     f"{'密度 ' if not law3['is_valid'] else ''}"
            ),
        }
