"""
12-晶子种子生成与球面码方向。

12-晶子闭环 = 拓扑常数12的强制解（欧拉恒等式 Σ(6-deg(v)) = 12）。
晶子半径 = kappa，作为基本长度单位。
"""

import numpy as np
from typing import List, Tuple, Optional


def generate_icosahedron_seed(kappa: float = 1.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    生成正二十面体的12个顶点作为嵌入种子。

    12-晶子闭环 = 拓扑常数12的强制解。
    晶子半径 = kappa。

    Args:
        kappa: 晶子半径（基本长度单位）

    Returns:
        vertices: (12, 3) array of vertex positions
        edges: list of (i, j) tuples connecting adjacent vertices
    """
    # 黄金比例
    phi = (1 + np.sqrt(5)) / 2

    # 正二十面体12个顶点
    # 每组坐标取全部符号组合得到4个点，3组共12个点
    vertices = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            vertices.append((0, s1, s2 * phi))
            vertices.append((s1, s2 * phi, 0))
            vertices.append((s2 * phi, 0, s1))

    vertices = np.array(vertices, dtype=float)

    # 归一化到半径 kappa
    norms = np.linalg.norm(vertices, axis=1)
    vertices = vertices * (kappa / norms[:, np.newaxis])

    # 确定邻接边（距离接近最小距离的即为相邻顶点）
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            dists.append(np.linalg.norm(vertices[i] - vertices[j]))
    min_dist = min(dists)

    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(np.linalg.norm(vertices[i] - vertices[j]) - min_dist) < 0.1 * min_dist:
                edges.append((i, j))

    return vertices, edges


def get_spherical_code_directions(d: int, rng_seed: Optional[int] = None) -> np.ndarray:
    """
    返回d个球面码方向（最大化最小角间距）。

    标准解：
    d=12 -> 正二十面体顶点
    d=8  -> 正方体顶点
    d=6  -> 正八面体顶点
    d=4  -> 正四面体顶点
    d=3  -> 等边三角形（大圆上120°）
    d=2  -> 对径点（180°）
    d=1  -> [1, 0, 0]
    d>12 -> 近似均匀随机（使用Fibonacci球面算法）

    Args:
        d: 需要产生的方向数
        rng_seed: 随机种子（d>12时有用）

    Returns:
        directions: (d, 3) array of unit vectors
    """
    if d == 1:
        return np.array([[1.0, 0.0, 0.0]])

    if d == 2:
        return np.array([[1.0, 0.0, 0.0],
                         [-1.0, 0.0, 0.0]])

    if d == 3:
        # 大圆上120°等间距
        angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
        return np.column_stack([np.cos(angles), np.sin(angles), np.zeros(3)])

    if d == 4:
        # 正四面体顶点
        verts = np.array([[1, 1, 1],
                          [1, -1, -1],
                          [-1, 1, -1],
                          [-1, -1, 1]], dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    if d == 6:
        # 正八面体顶点
        return np.array([[1, 0, 0],
                         [-1, 0, 0],
                         [0, 1, 0],
                         [0, -1, 0],
                         [0, 0, 1],
                         [0, 0, -1]], dtype=float)

    if d == 8:
        # 正方体顶点
        verts = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                for s3 in [-1, 1]:
                    verts.append([s1, s2, s3])
        verts = np.array(verts, dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    if d == 12:
        # 正二十面体顶点
        phi = (1 + np.sqrt(5)) / 2
        verts = []
        for s1 in [-1, 1]:
            for s2 in [-1, 1]:
                verts.append((0, s1, s2 * phi))
                verts.append((s1, s2 * phi, 0))
                verts.append((s2 * phi, 0, s1))
        verts = np.array(verts, dtype=float)
        norms = np.linalg.norm(verts, axis=1)
        return verts / norms[:, np.newaxis]

    # d > 12: Fibonacci球面算法（近似均匀分布）
    rng = np.random.RandomState(rng_seed)
    directions = np.zeros((d, 3), dtype=float)

    # 使用Fibonacci球面点集
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(d):
        theta = np.arccos(1 - 2 * (i + 0.5) / d)
        phi_angle = 2 * np.pi * i / golden_ratio
        directions[i] = [
            np.sin(theta) * np.cos(phi_angle),
            np.sin(theta) * np.sin(phi_angle),
            np.cos(theta)
        ]

    return directions
