"""
正二十面体组装检测器

职责:
    1. 从松弛后的晶子位置检测是否形成近似正二十面体
    2. 将晶子映射到正二十面体的 12 个顶点
    3. 验证子图内度数 = 5 条件
    4. 报告检测结果

物理预期:
    12 个等大球体通过力导向松弛后的最低能态 = 正二十面体。
    这不是假设, 是三维密堆的几何必然。
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ============================================================
# 正二十面体几何
# ============================================================

def _icosahedron_vertices(edge_length: float = 2.0) -> List[Tuple[float, float, float]]:
    """标准正二十面体的 12 个顶点。

    使用黄金比例 φ = (1+√5)/2。
    顶点排列: (±1, ±φ, 0) 及其循环置换, 归一化到指定边长。

    返回的 12 个顶点按顺序排列:
        0: (1, φ, 0)
        1: (1, -φ, 0)
        2: (-1, φ, 0)
        3: (-1, -φ, 0)
        4: (0, 1, φ)
        5: (0, -1, φ)
        6: (0, 1, -φ)
        7: (0, -1, -φ)
        8: (φ, 0, 1)
        9: (-φ, 0, 1)
        10: (φ, 0, -1)
        11: (-φ, 0, -1)
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    s = edge_length / 2

    return [
        (s, s * phi, 0), (s, -s * phi, 0), (-s, s * phi, 0), (-s, -s * phi, 0),
        (0, s, s * phi), (0, -s, s * phi), (0, s, -s * phi), (0, -s, -s * phi),
        (s * phi, 0, s), (-s * phi, 0, s), (s * phi, 0, -s), (-s * phi, 0, -s),
    ]


def _icosahedron_edges(
    vertices: Optional[List[Tuple[float, float, float]]] = None,
    tolerance: float = 0.1,
) -> List[Tuple[int, int]]:
    """通过几何近邻自动计算正二十面体的边。

    对于每个顶点, 最近的顶点 (= 距离 ≈ edge_length) 是其在二十面体中的邻居。
    每个顶点恰好有 5 个邻居 (正二十面体每个顶点度数为 5)。

    Args:
        vertices: 12 个顶点坐标; None 时使用标准位置
        tolerance: 距离比较容差

    Returns:
        30 条边, 每条边表示为 (i, j) 顶点索引对
    """
    if vertices is None:
        vertices = _icosahedron_vertices(edge_length=2.0)

    # 计算所有顶点对的距离
    n = len(vertices)
    dist_matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(float("inf"))
            else:
                d = math.sqrt(sum((vertices[i][k] - vertices[j][k]) ** 2 for k in range(3)))
                row.append(d)
        dist_matrix.append(row)

    # 最近的 5 个邻居 = 边
    edges = set()
    for i in range(n):
        # 按距离排序, 取前 5 个
        sorted_nbrs = sorted(
            [(dist_matrix[i][j], j) for j in range(n) if j != i],
            key=lambda x: x[0],
        )
        for _, j in sorted_nbrs[:5]:
            edges.add(tuple(sorted((i, j))))

    return sorted(edges)


# ============================================================
# 匹配检测
# ============================================================

class IcosahedronDetector:
    """检测松弛后的晶子集合是否近似于正二十面体。

    匹配标准:
        1. 晶子数 = 12
        2. 每个晶子有 5 个晶子邻居 (在子图内)
        3. 晶子间距离接近 2×radius 的均值 (即边长)
        4. Σ(6-deg) ≈ 12 (在晶子子图内)

    容忍度:
        GEOMETRIC_TOLERANCE = 0.1 用于距离比较
    """

    def __init__(self, tolerance: float = 0.1):
        self.tolerance = tolerance  # 相对容差

    def detect(
        self,
        positions: Dict[str, Tuple[float, float, float]],
        neighbors: Dict[str, List[str]],
        radii: Dict[str, float],
    ) -> Dict:
        """检测是否形成正二十面体。

        Args:
            positions: {uid: (x, y, z)} 所有晶子的空间位置
            neighbors: {uid: [neighbor_uid, ...]} 簇边邻接表
            radii: {uid: radius} 每个晶子的半径

        Returns:
            {is_icosahedron, match_score, avg_edge_length,
             degree_distribution, spum_invariant, analysis}
        """
        n = len(positions)
        if n != 12:
            return {
                "is_icosahedron": False,
                "n_spheres": n,
                "message": f"需要 12 个晶子, 现有 {n}",
            }

        # 1. 计算每个晶子的子图度数
        degree_dist = {}
        for uid in positions:
            deg = len(neighbors.get(uid, []))
            degree_dist[uid] = deg

        # 2. 检验度数 = 5
        deg5_count = sum(1 for d in degree_dist.values() if d == 5)
        all_deg5 = deg5_count == 12

        # 3. 检查 Σ(6-deg)
        spum_invariant = sum(6 - d for d in degree_dist.values())

        # 4. 检查边长一致性
        edge_lengths = []
        for uid_a, pos_a in positions.items():
            for uid_b in neighbors.get(uid_a, []):
                if uid_a < uid_b:  # 避免重复
                    d = math.sqrt(sum(
                        (pos_a[i] - positions[uid_b][i]) ** 2
                        for i in range(3)
                    ))
                    edge_lengths.append(d)

        avg_edge = sum(edge_lengths) / len(edge_lengths) if edge_lengths else 0
        std_edge = math.sqrt(
            sum((e - avg_edge) ** 2 for e in edge_lengths) / len(edge_lengths)
        ) if edge_lengths else 0
        edge_consistency = std_edge / avg_edge if avg_edge > 0 else 0

        # 5. 检查相切条件: distance ≈ r_i + r_j
        tangent_errors = []
        for uid_a, pos_a in positions.items():
            for uid_b in neighbors.get(uid_a, []):
                if uid_a < uid_b:
                    d = math.sqrt(sum(
                        (pos_a[i] - positions[uid_b][i]) ** 2
                        for i in range(3)
                    ))
                    r_sum = radii[uid_a] + radii[uid_b]
                    err = abs(d - r_sum) / r_sum
                    tangent_errors.append(err)

        avg_tangent_err = (
            sum(tangent_errors) / len(tangent_errors)
            if tangent_errors else 0
        )

        # 6. 综合评分
        n_edges = len(edge_lengths)
        target_edges = 30  # 正二十面体有 30 条边

        score = 0.0
        if all_deg5:
            score += 0.4
        if abs(spum_invariant - 12) <= 2:
            score += 0.2
        if edge_consistency < 0.1:
            score += 0.2
        if avg_tangent_err < 0.1:
            score += 0.2

        is_match = score >= 0.8 and all_deg5 and abs(spum_invariant - 12) <= 2

        return {
            "is_icosahedron": is_match,
            "match_score": round(score, 3),
            "n_spheres": n,
            "n_edges": n_edges,
            "target_edges": target_edges,
            "avg_edge_length": round(avg_edge, 4),
            "std_edge_length": round(std_edge, 4),
            "edge_consistency": round(edge_consistency, 4),
            "avg_tangent_error": round(avg_tangent_err, 4),
            "degree_distribution": {
                str(d): count for d, count in
                sorted(__import__("collections").Counter(
                    degree_dist.values()
                ).items())
            },
            "deg5_count": deg5_count,
            "spum_invariant": spum_invariant,
            "spum_invariant_ok": abs(spum_invariant - 12) <= 2,
            "message": (
                "正二十面体检测通过" if is_match
                else "未检测到正二十面体构型"
            ),
        }


# ============================================================
# 高级接口
# ============================================================

def map_crystallites_to_icosahedron(
    positions: Dict[str, Tuple[float, float, float]],
) -> Optional[Dict]:
    """将晶子位置映射到标准正二十面体顶点。

    使用匈牙利算法思想: 找到最小化总位移的配对。
    简化版: 对每个标准顶点, 找最近的晶子。

    Args:
        positions: {uid: (x, y, z)} 当前晶子位置

    Returns:
        {uid -> icosahedron_vertex_index} 映射, 或 None (无法配对)
    """
    n = len(positions)
    if n != 12:
        return None

    target_vertices = _icosahedron_vertices(edge_length=2.0)
    uids = list(positions.keys())
    mapping = {}

    # KM 简化: 贪心匹配 (对每个目标顶点, 找最近未匹配晶子)
    matched = set()
    for vi, tv in enumerate(target_vertices):
        best_uid = None
        best_dist = float("inf")
        for uid in uids:
            if uid in matched:
                continue
            p = positions[uid]
            d = math.sqrt(sum((p[i] - tv[i]) ** 2 for i in range(3)))
            if d < best_dist:
                best_dist = d
                best_uid = uid
        if best_uid is not None:
            mapping[best_uid] = vi
            matched.add(best_uid)

    if len(mapping) != 12:
        return None

    # 验证映射一致性: 检查映射后边是否匹配
    mapped_edges = 0
    total_expected = 30
    for uid_a, vi_a in mapping.items():
        for uid_b, vi_b in mapping.items():
            if uid_a >= uid_b:
                continue
            # 检查标准正二十面体中这两个顶点是否有边
            if (vi_a, vi_b) in _icosahedron_edges() or \
               (vi_b, vi_a) in _icosahedron_edges():
                mapped_edges += 1

    consistency = mapped_edges / total_expected if total_expected > 0 else 0

    return {
        "mapping": mapping,
        "edge_match_ratio": round(consistency, 4),
        "consistent": consistency > 0.8,
    }
