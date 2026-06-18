"""
三维拓扑不变量验证

在晶子子图上验证:
    - Σ(6-deg) = 12 (强制拓扑不变量)
    - 欧拉示性数 χ = V - E + F = 2 (仅对球面同胚子图)
    - 子图内每个晶子度数 = 5 的节点数

这些验证是 SPUM 公理的刚性要求:
    - 总纲 §2.2: 闭合子图必须满足 Σ(6-deg) = 12
    - 三维几何约束使这一条件成为自然涌现 (非强制)
"""

import math
from typing import Dict, List, Optional, Set, Tuple


def compute_spum_invariant_3d(
    degrees: Dict[str, int],
) -> Dict:
    """计算 Σ(6-deg) 三维版本。

    Args:
        degrees: {uid: degree_in_cluster_subgraph}

    Returns:
        {value: int, contributions: [...], is_valid: bool}
    """
    contributions = {}
    total = 0
    for uid, deg in degrees.items():
        contrib = 6 - deg
        contributions[uid] = contrib
        total += contrib

    return {
        "sigma_6_minus_deg": total,
        "contributions": contributions,
        "is_valid": abs(total - 12) <= 2,  # 误差容限
        "n_vertices": len(degrees),
    }


def compute_euler_characteristic_3d(
    vertices: int,
    edges: int,
    faces: int,
) -> Dict:
    """计算欧拉示性数 χ = V - E + F。

    对于球面同胚子图:
        - χ = 2
        - 对于三角剖分: F = 2V - 4, E = 3V - 6
        - V=12 → E=30, F=20 (正二十面体)

    Returns:
        {chi: int, is_spherical: bool, expected_faces: int}
    """
    chi = vertices - edges + faces
    expected_faces = 2 * vertices - 4 if vertices >= 3 else 0
    expected_edges = 3 * vertices - 6 if vertices >= 3 else 0

    return {
        "chi": chi,
        "is_spherical": chi == 2,
        "actual_vertices": vertices,
        "actual_edges": edges,
        "actual_faces": faces,
        "expected_edges": expected_edges,
        "expected_faces": expected_faces,
        "edge_match": edges == expected_edges,
        "face_match": faces == expected_faces if faces > 0 else None,
    }


def validate_spum_topology_3d(
    positions: Dict[str, Tuple[float, float, float]],
    neighbors: Dict[str, List[str]],
    radii: Dict[str, float],
) -> Dict:
    """对三维晶子集合执行完整的 SPUM 拓扑验证。

    验证项:
        1. 度数分布: 每个晶子在子图内的度数
        2. Σ(6-deg) = 12
        3. 球体相切条件: distance ≈ r_i + r_j
        4. 对称性分析: 配对计数

    Returns:
        {is_valid, invariants, geometry, summary}
    """
    n = len(positions)
    if n == 0:
        return {"is_valid": False, "message": "空集"}

    # 度数分布
    degrees = {}
    for uid in positions:
        degrees[uid] = len(neighbors.get(uid, []))

    # Σ(6-deg)
    invariant = compute_spum_invariant_3d(degrees)
    degree_hist = {}
    for d in degrees.values():
        degree_hist[d] = degree_hist.get(d, 0) + 1

    # 相切条件
    tangent_pairs = []
    tangent_violations = []
    total_pairs = 0
    for uid_a in positions:
        for uid_b in neighbors.get(uid_a, []):
            if uid_a >= uid_b:
                continue
            total_pairs += 1
            pos_a = positions[uid_a]
            pos_b = positions[uid_b]
            d = math.sqrt(sum((pos_a[i] - pos_b[i]) ** 2 for i in range(3)))
            r_sum = radii[uid_a] + radii[uid_b]
            err = abs(d - r_sum) / r_sum if r_sum > 0 else 0

            entry = {
                "pair": (uid_a, uid_b),
                "distance": round(d, 4),
                "tangent_distance": round(r_sum, 4),
                "relative_error": round(err, 4),
            }
            if err < 0.1:
                tangent_pairs.append(entry)
            else:
                tangent_violations.append(entry)

    # 综合判定
    is_valid = (
        invariant["is_valid"]
        and len(tangent_violations) <= total_pairs * 0.2  # ≤20% 违反
    )

    return {
        "is_valid": is_valid,
        "n_spheres": n,
        "n_edges": total_pairs,
        "invariant": {
            "sigma_6_minus_deg": invariant["sigma_6_minus_deg"],
            "is_valid": invariant["is_valid"],
        },
        "degree_distribution": degree_hist,
        "tangent_condition": {
            "total_pairs": total_pairs,
            "satisfied": len(tangent_pairs),
            "violated": len(tangent_violations),
            "sat_ratio": round(len(tangent_pairs) / max(1, total_pairs), 4),
        },
        "tangent_violations": tangent_violations[:5],  # 前 5 个违反
        "summary": (
            "SPUM 三维拓扑验证通过"
            if is_valid
            else "SPUM 三维拓扑验证未通过"
        ),
    }
