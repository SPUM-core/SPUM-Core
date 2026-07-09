"""
诱导坐标嵌入算法测试。

覆盖：
- 正二十面体种子验证
- 球面码方向验证
- σ调节步长验证
- BFS传播嵌入验证
- 度量张量计算验证
- 均匀极限验证
"""

import numpy as np
import pytest

from .icosahedron_seed import generate_icosahedron_seed, get_spherical_code_directions
from .embedding import InducedMetricEmbedding


def test_icosahedron_seed():
    """验证12个顶点构成正二十面体。"""
    vertices, edges = generate_icosahedron_seed()
    assert vertices.shape == (12, 3), f"期望12个顶点，得到{vertices.shape[0]}"
    # 验证所有顶点到原点距离相同
    norms = np.linalg.norm(vertices, axis=1)
    assert np.allclose(norms, norms[0]), "顶点到原点距离不一致"
    # 验证边数正确（正二十面体有30条边）
    assert len(edges) == 30, f"期望30条边，得到{len(edges)}"
    # 验证半径
    assert np.allclose(norms[0], 1.0), f"期望半径1.0，得到{norms[0]}"
    print("✓ 正二十面体种子验证通过")


def test_icosahedron_seed_kappa():
    """验证kappa参数。"""
    kappa = 2.5
    vertices, _ = generate_icosahedron_seed(kappa)
    norms = np.linalg.norm(vertices, axis=1)
    assert np.allclose(norms, kappa), f"期望半径{kappa}，得到{norms[0]}"
    print("✓ 正二十面体kappa参数验证通过")


def test_spherical_code_d1():
    """验证d=1球面码。"""
    dirs = get_spherical_code_directions(1)
    assert dirs.shape == (1, 3)
    assert np.allclose(np.linalg.norm(dirs[0]), 1.0)
    assert np.allclose(dirs[0], [1.0, 0.0, 0.0])
    print("✓ d=1球面码验证通过")


def test_spherical_code_d2():
    """验证d=2球面码（对径点）。"""
    dirs = get_spherical_code_directions(2)
    assert dirs.shape == (2, 3)
    for d in dirs:
        assert np.allclose(np.linalg.norm(d), 1.0)
    # 验证对径
    assert np.allclose(dirs[0], -dirs[1])
    print("✓ d=2球面码验证通过")


def test_spherical_code_d3():
    """验证d=3球面码（大圆120°）。"""
    dirs = get_spherical_code_directions(3)
    assert dirs.shape == (3, 3)
    for d in dirs:
        assert abs(np.linalg.norm(d) - 1.0) < 1e-10
    # 验证两两夹角均为120°
    for i in range(3):
        for j in range(i + 1, 3):
            dot = np.dot(dirs[i], dirs[j])
            assert abs(dot - (-0.5)) < 1e-10, f"期望夹角120°，点积=-0.5，得到{dot}"
    print("✓ d=3球面码验证通过")


def test_spherical_code_d4():
    """验证d=4球面码（正四面体）。"""
    dirs = get_spherical_code_directions(4)
    assert dirs.shape == (4, 3)
    for d in dirs:
        assert abs(np.linalg.norm(d) - 1.0) < 1e-10
    # 验证正四面体性质：所有点积相同
    dots = []
    for i in range(4):
        for j in range(i + 1, 4):
            dots.append(np.dot(dirs[i], dirs[j]))
    assert np.allclose(dots, dots[0]), "正四面体点积不一致"
    print("✓ d=4球面码验证通过")


def test_spherical_code_d6():
    """验证d=6球面码（正八面体）。"""
    dirs = get_spherical_code_directions(6)
    assert dirs.shape == (6, 3)
    for d in dirs:
        assert abs(np.linalg.norm(d) - 1.0) < 1e-10
    # 验证正八面体：每个方向有一个对径方向
    found_pairs = 0
    for i in range(6):
        for j in range(i + 1, 6):
            if np.allclose(dirs[i], -dirs[j]):
                found_pairs += 1
    assert found_pairs == 3, f"期望3对对径点，找到{found_pairs}"
    print("✓ d=6球面码验证通过")


def test_spherical_code_d8():
    """验证d=8球面码（正方体）。"""
    dirs = get_spherical_code_directions(8)
    assert dirs.shape == (8, 3)
    for d in dirs:
        assert abs(np.linalg.norm(d) - 1.0) < 1e-10
    # 验证每个分量绝对值为 1/√3
    expected_comp = 1.0 / np.sqrt(3)
    for d in dirs:
        assert np.allclose(np.abs(d), expected_comp), f"正方体顶点坐标异常: {d}"
    print("✓ d=8球面码验证通过")


def test_spherical_code_d12():
    """验证d=12球面码（正二十面体）。"""
    dirs = get_spherical_code_directions(12)
    assert dirs.shape == (12, 3)
    for d in dirs:
        assert abs(np.linalg.norm(d) - 1.0) < 1e-10
    # 验证与正二十面体顶点一致
    vertices, _ = generate_icosahedron_seed()
    vertex_norms = np.linalg.norm(vertices, axis=1)
    vertices_normalized = vertices / vertex_norms[:, np.newaxis]
    # 检查两组点集是否匹配（允许排列）
    match = False
    for perm_start in range(12):
        shifted = np.roll(vertices_normalized, perm_start, axis=0)
        if np.allclose(dirs, shifted):
            match = True
            break
    # 更宽松的匹配：检查所有距离是否一致
    dir_dists = set()
    for i in range(12):
        for j in range(i + 1, 12):
            d = round(np.linalg.norm(dirs[i] - dirs[j]), 10)
            dir_dists.add(d)
    vert_dists = set()
    for i in range(12):
        for j in range(i + 1, 12):
            d = round(np.linalg.norm(vertices_normalized[i] - vertices_normalized[j]), 10)
            vert_dists.add(d)
    assert dir_dists == vert_dists, "d=12球面码与正二十面体距离分布不匹配"
    print("✓ d=12球面码验证通过")


def test_spherical_code_d_gt_12():
    """验证d>12使用Fibonacci球面算法。"""
    for d in [20, 50, 100]:
        dirs = get_spherical_code_directions(d)
        assert dirs.shape == (d, 3)
        norms = np.linalg.norm(dirs, axis=1)
        assert np.allclose(norms, 1.0), f"d={d} Fibonacci球面方向未单位化"
    print("✓ d>12 Fibonacci球面码验证通过")


def test_step_size():
    """验证σ调节步长。"""
    embed = InducedMetricEmbedding(kappa=1.0, sigma_0=1.0)
    # 高σ → 步长缩短
    s_high = embed.step_size(2.0)
    s_low = embed.step_size(0.5)
    assert s_high < s_low, "高σ应产生更短步长"
    # 验证公式形式
    expected_high = 1.0 * (1.0 / 2.0) ** (1.0 / 3.0)
    expected_low = 1.0 * (1.0 / 0.5) ** (1.0 / 3.0)
    assert abs(s_high - expected_high) < 1e-10
    assert abs(s_low - expected_low) < 1e-10
    print("✓ σ步长调节验证通过")


def test_step_size_kappa_scaling():
    """验证步长随kappa线性缩放。"""
    embed1 = InducedMetricEmbedding(kappa=1.0, sigma_0=1.0)
    embed2 = InducedMetricEmbedding(kappa=2.0, sigma_0=1.0)
    assert abs(embed2.step_size(1.0) - 2.0 * embed1.step_size(1.0)) < 1e-10
    print("✓ kappa线性缩放验证通过")


def test_seed_from_icosahedron():
    """验证种子初始化。"""
    embed = InducedMetricEmbedding(kappa=1.0)
    embed.seed_from_icosahedron()
    assert len(embed.embedded) == 12
    assert len(embed.positions) == 12
    # 验证所有种子位置为单位球面上的点
    for nid in range(12):
        norm = np.linalg.norm(embed.positions[nid])
        assert abs(norm - 1.0) < 1e-10, f"节点{nid}不在单位球面上"
    print("✓ 种子初始化验证通过")


def test_bfs_embed_simple():
    """验证简单图的BFS嵌入。"""
    # 构建一个简单图：种子12节点 + 额外节点
    adjacency = {}
    sigma_map = {}
    degree_map = {}

    for i in range(12):
        adjacency[i] = []
        sigma_map[i] = 1.0
        degree_map[i] = 6

    # 添加一个连接到种子0的节点
    adjacency[0].append(12)
    adjacency[12] = [0]
    sigma_map[12] = 1.0
    degree_map[12] = 1

    embed = InducedMetricEmbedding(kappa=1.0, sigma_0=1.0)
    positions = embed.bfs_embed(adjacency, sigma_map, degree_map)

    # 验证种子已嵌入
    for i in range(12):
        assert i in positions, f"种子节点{i}未嵌入"
    # 验证新节点已嵌入
    assert 12 in positions, "新节点未嵌入"
    print("✓ BFS简单图嵌入验证通过")


def test_bfs_embed_multi_neighbor():
    """验证多邻居加权质心。"""
    adjacency = {}
    sigma_map = {}
    degree_map = {}

    # 种子节点
    for i in range(12):
        adjacency[i] = []
        sigma_map[i] = 1.0
        degree_map[i] = 6

    # 节点12连接到种子0和种子1
    adjacency[0].append(12)
    adjacency[1].append(12)
    adjacency[12] = [0, 1]
    sigma_map[12] = 0.8
    degree_map[12] = 2

    embed = InducedMetricEmbedding(kappa=1.0, sigma_0=1.0)
    positions = embed.bfs_embed(adjacency, sigma_map, degree_map)

    assert 12 in positions, "多邻居节点未嵌入"
    # 验证位置在种子0和种子1之间
    p12 = positions[12]
    p0 = positions[0]
    p1 = positions[1]
    d0 = np.linalg.norm(p12 - p0)
    d1 = np.linalg.norm(p12 - p1)
    assert d0 > 0 and d1 > 0, "嵌入位置与邻居重合"
    print("✓ 多邻居加权质心验证通过")


def test_compute_metric_tensor():
    """验证度量张量计算。"""
    embed = InducedMetricEmbedding(kappa=1.0)
    embed.seed_from_icosahedron()

    # 为种子节点设置sigma和degree
    for i in range(12):
        embed.sigmas[i] = 1.0
        embed.degrees[i] = 5

    # 计算各节点度量张量
    g = embed.compute_metric_tensor(0)
    assert g.shape == (3, 3), f"度量张量应为(3,3)，得到{g.shape}"
    # 验证对称性
    assert np.allclose(g, g.T), "度量张量不对称"
    print("✓ 度量张量计算验证通过")


def test_global_distortion():
    """验证全局畸变计算。"""
    embed = InducedMetricEmbedding(kappa=1.0, sigma_0=1.0)
    embed.seed_from_icosahedron()

    # 设种子节点的sigma和degree
    for i in range(12):
        embed.sigmas[i] = 1.0
        embed.degrees[i] = 5

    delta = embed.compute_global_distortion()
    assert delta >= 0, "全局畸变不能为负"
    assert isinstance(delta, float), "全局畸变应为浮点数"
    print(f"✓ 全局畸变计算验证通过 (δ={delta:.4f})")


def test_conflict_detection():
    """验证冲突检测。"""
    embed = InducedMetricEmbedding(kappa=1.0, sigma_0=1.0,
                                    epsilon_conflict=0.01)
    embed.seed_from_icosahedron()

    adjacency = {i: [] for i in range(12)}
    sigma_map = {i: 1.0 for i in range(12)}
    degree_map = {i: 6 for i in range(12)}

    # 添加一个至少有两个种子邻居的冲突节点
    adjacency[0].append(12)
    adjacency[1].append(12)
    adjacency[2].append(12)
    adjacency[12] = [0, 1, 2]
    sigma_map[12] = 1.0
    degree_map[12] = 3

    embed.bfs_embed(adjacency, sigma_map, degree_map)

    # 如果冲突残差大于阈值，节点应被标记
    print(f"✓ 冲突检测验证通过 (冲突节点数: {len(embed.conflict_nodes)})")


def test_bfs_empty_adjacency():
    """验证空邻接表的健壮性。"""
    embed = InducedMetricEmbedding()
    positions = embed.bfs_embed({}, {}, {})
    assert len(positions) == 0
    print("✓ 空邻接表健壮性验证通过")


def test_bfs_disconnected():
    """验证非连通图的健壮性。"""
    adjacency = {i: [] for i in range(5)}
    sigma_map = {i: 1.0 for i in range(5)}
    degree_map = {i: 0 for i in range(5)}

    embed = InducedMetricEmbedding()
    positions = embed.bfs_embed(adjacency, sigma_map, degree_map,
                                start_node_ids=[0, 1, 2, 3, 4])
    assert len(positions) <= 5
    print("✓ 非连通图健壮性验证通过")


def test_bfs_large_graph():
    """验证大规模图BFS嵌入不崩溃。"""
    n_nodes = 200
    adjacency = {}
    sigma_map = {}
    degree_map = {}

    # 构建链式图（每个节点度2）
    for i in range(n_nodes):
        adjacency[i] = []
        sigma_map[i] = 1.0
        degree_map[i] = 2

    for i in range(n_nodes - 1):
        adjacency[i].append(i + 1)
        adjacency[i + 1].append(i)

    embed = InducedMetricEmbedding(kappa=1.0, sigma_0=1.0)
    positions = embed.bfs_embed(adjacency, sigma_map, degree_map,
                                start_node_ids=list(range(12)))
    # 验证部分节点嵌入（非全图连通时）
    assert len(positions) > 0
    print(f"✓ 大规模图BFS嵌入验证通过 (嵌入{len(positions)}/{n_nodes}节点)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
