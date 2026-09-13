"""
Phase 3 测试套件 — 三维几何聚簇求解器

测试内容:
    1. 基本 3D 向量工具
    2. Sphere3D 数据结构与力计算
    3. 力导向松弛收敛性
    4. 标准正二十面体检测
    5. Σ(6-deg)=12 拓扑验证
    6. 从已有网络整合
"""

import sys
import math
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest

from Phase_3.geometry_solver import (
    Sphere3D, ForceDirectedRelaxation, GeometrySolver,
    vec_add, vec_sub, vec_scale, vec_norm, vec_normalize, vec_distance,
    tangent_distance,
)
from Phase_3.icosahedron_assembly import IcosahedronDetector, map_crystallites_to_icosahedron
from Phase_3.validator import compute_spum_invariant_3d, validate_spum_topology_3d


# ============================================================
# 测试 1: 基本 3D 向量工具
# ============================================================

class TestVectorTools:
    def test_vec_add(self):
        assert vec_add((1, 2, 3), (4, 5, 6)) == (5, 7, 9)

    def test_vec_sub(self):
        assert vec_sub((5, 7, 9), (1, 2, 3)) == (4, 5, 6)

    def test_vec_scale(self):
        assert vec_scale((1, 2, 3), 2) == (2, 4, 6)

    def test_vec_norm(self):
        assert abs(vec_norm((3, 4, 0)) - 5.0) < 1e-12
        assert abs(vec_norm((1, 0, 0)) - 1.0) < 1e-12

    def test_vec_normalize(self):
        v = vec_normalize((3, 4, 0))
        assert abs(vec_norm(v) - 1.0) < 1e-12
        assert abs(v[0] - 0.6) < 1e-12
        assert abs(v[1] - 0.8) < 1e-12

    def test_vec_distance(self):
        assert abs(vec_distance((0, 0, 0), (3, 4, 0)) - 5.0) < 1e-12

    def test_tangent_distance(self):
        assert abs(tangent_distance(2.0, 3.0) - 5.0) < 1e-12

    def test_normalize_zero(self):
        v = vec_normalize((0, 0, 0))
        assert v == (0, 0, 0)  # 零向量不变


# ============================================================
# 测试 2: Sphere3D
# ============================================================

class TestSphere3D:
    def test_basic(self):
        s = Sphere3D(uid="t1", degree=10, position=(0, 0, 0), radius=2.0)
        assert s.uid == "t1"
        assert s.diameter == 4.0

    def test_tangent_distance(self):
        a = Sphere3D("a", 0, (0, 0, 0), 2.0)
        b = Sphere3D("b", 0, (5, 0, 0), 3.0)
        assert abs(a.tangent_distance(b) - 5.0) < 1e-12

    def test_overlap_positive(self):
        """球体重叠时 overlap > 0"""
        a = Sphere3D("a", 0, (0, 0, 0), 2.0)
        b = Sphere3D("b", 0, (3.0, 0, 0), 2.0)
        assert a.overlap_with(b) > 0  # 重叠

    def test_overlap_negative(self):
        """球体分离时 overlap < 0"""
        a = Sphere3D("a", 0, (0, 0, 0), 2.0)
        b = Sphere3D("b", 0, (10.0, 0, 0), 2.0)
        assert a.overlap_with(b) < 0  # 分离

    def test_overlap_tangent(self):
        """球体相切时 overlap ≈ 0"""
        a = Sphere3D("a", 0, (0, 0, 0), 2.0)
        b = Sphere3D("b", 0, (4.0, 0, 0), 2.0)
        assert abs(a.overlap_with(b)) < 1e-12  # 相切

    def test_force_repulsive(self):
        """重叠球体产生排斥力。"""
        a = Sphere3D("a", 0, (0, 0, 0), 2.0)
        b = Sphere3D("b", 0, (1.0, 0, 0), 2.0)
        f = a.force_to(b, k_rep=10.0)
        # 力方向: 指向远离 b (即 -x 方向)
        assert f[0] < 0  # 排斥力沿 -x

    def test_force_attractive(self):
        """连接的分离球体产生吸引力。"""
        a = Sphere3D("a", 0, (0, 0, 0), 2.0, neighbors=["b"])
        b = Sphere3D("b", 0, (10.0, 0, 0), 2.0, neighbors=["a"])
        f = a.force_to(b, k_att=1.0)
        assert f[0] > 0  # 吸引力沿 +x


# ============================================================
# 测试 3: 力导向松弛
# ============================================================

class TestForceDirectedRelaxation:
    def test_convergence_two_spheres(self):
        """两个重叠球体应推开至相切距离。"""
        spheres = {
            "a": Sphere3D("a", 0, (0, 0, 0), 2.0),
            "b": Sphere3D("b", 0, (1.0, 0, 0), 2.0),
        }
        solver = ForceDirectedRelaxation(spheres, k_rep=20.0, dt=0.005)
        result = solver.solve()
        final_d = vec_distance(spheres["a"].position, spheres["b"].position)
        assert abs(final_d - 4.0) < 0.1  # 收敛到相切距离

    def test_three_spheres_cluster(self):
        """三个球体应形成三角形。"""
        spheres = {
            "a": Sphere3D("a", 0, (0, 0, 0), 1.0),
            "b": Sphere3D("b", 0, (2.0, 0, 0), 1.0, neighbors=["a"]),
            "c": Sphere3D("c", 0, (0, 2.0, 0), 1.0, neighbors=["a"]),
        }
        # 只固定 a
        spheres["a"].fixed = True
        solver = ForceDirectedRelaxation(spheres, k_rep=10.0, damping=0.5, dt=0.01)
        result = solver.solve()
        # b 和 c 应该靠近 a
        d_ab = vec_distance(spheres["a"].position, spheres["b"].position)
        d_ac = vec_distance(spheres["a"].position, spheres["c"].position)
        assert d_ab < 3.0
        assert d_ac < 3.0

    def test_energy_decreases(self):
        """松弛过程中总能量应单调减少。"""
        spheres = {
            "a": Sphere3D("a", 0, (0, 0, 0), 2.0),
            "b": Sphere3D("b", 0, (0.5, 0, 0), 2.0),
        }
        solver = ForceDirectedRelaxation(spheres, k_rep=20.0, dt=0.001)
        initial_energy = solver.total_energy
        solver.solve()
        assert solver.total_energy < initial_energy


# ============================================================
# 测试 4: 正二十面体检测
# ============================================================

class TestIcosahedronDetector:
    def test_standard_icosahedron(self):
        """标准正二十面体应被检测器识别。"""
        detector = IcosahedronDetector(tolerance=0.1)

        # 从标准顶点构造
        from Phase_3.geometry_solver import GeometrySolver
        vertices = GeometrySolver.compute_icosahedron_positions(edge_length=2.0)

        positions = {f"v{i}": v for i, v in enumerate(vertices)}

        # 标准正二十面体的边 (30 条)
        from Phase_3.icosahedron_assembly import _icosahedron_edges
        edges = _icosahedron_edges()
        neighbors = {f"v{i}": [] for i in range(12)}
        for i, j in edges:
            neighbors[f"v{i}"].append(f"v{j}")
            neighbors[f"v{j}"].append(f"v{i}")

        radii = {f"v{i}": 1.0 for i in range(12)}

        result = detector.detect(positions, neighbors, radii)
        assert result["is_icosahedron"], f"标准正二十面体应被识别: {result['message']}"
        assert result["deg5_count"] == 12
        assert abs(result["spum_invariant"] - 12) <= 2

    def test_non_icosahedron_rejected(self):
        """非正二十面体构型应被拒绝。"""
        detector = IcosahedronDetector()
        # 8 个晶子 (不够)
        positions = {f"v{i}": (float(i), 0, 0) for i in range(8)}
        neighbors = {f"v{i}": [] for i in range(8)}
        radii = {f"v{i}": 1.0 for i in range(8)}
        result = detector.detect(positions, neighbors, radii)
        assert not result["is_icosahedron"]
        assert result["n_spheres"] == 8

    def test_spum_invariant_12_spheres_correct(self):
        """12 个 deg-5 的晶子应满足 Σ(6-deg)=12。"""
        degs = {f"v{i}": 5 for i in range(12)}
        inv = compute_spum_invariant_3d(degs)
        assert inv["sigma_6_minus_deg"] == 12
        assert inv["is_valid"]

    def test_spum_invariant_bad(self):
        """错误的度数分布应不满足不变量。"""
        degs = {f"v{i}": 4 for i in range(12)}  # 全部 degree-4
        inv = compute_spum_invariant_3d(degs)
        assert inv["sigma_6_minus_deg"] == 24
        assert not inv["is_valid"]


# ============================================================
# 测试 5: 映射
# ============================================================

class TestIcosahedronMapping:
    def test_perfect_mapping(self):
        """标准顶点应完美映射。"""
        from Phase_3.geometry_solver import GeometrySolver
        vertices = GeometrySolver.compute_icosahedron_positions(edge_length=2.0)
        positions = {f"v{i}": v for i, v in enumerate(vertices)}
        result = map_crystallites_to_icosahedron(positions)
        assert result is not None
        assert result["consistent"]
        assert result["edge_match_ratio"] > 0.8

    def test_mapping_small_set(self):
        """少于 12 个点时返回 None。"""
        positions = {"a": (0, 0, 0), "b": (1, 0, 0)}
        result = map_crystallites_to_icosahedron(positions)
        assert result is None


# ============================================================
# 测试 6: 完整验证流水线
# ============================================================

class TestFullValidation:
    def test_perfect_icosahedron_validation(self):
        """标准正二十面体应通过全部验证。"""
        from Phase_3.geometry_solver import GeometrySolver
        from Phase_3.icosahedron_assembly import _icosahedron_edges

        vertices = GeometrySolver.compute_icosahedron_positions(edge_length=2.0)
        positions = {f"v{i}": v for i, v in enumerate(vertices)}

        edges = _icosahedron_edges()
        neighbors = {f"v{i}": [] for i in range(12)}
        for i, j in edges:
            neighbors[f"v{i}"].append(f"v{j}")
            neighbors[f"v{j}"].append(f"v{i}")

        radii = {f"v{i}": 1.0 for i in range(12)}

        result = validate_spum_topology_3d(positions, neighbors, radii)
        assert result["is_valid"]
        assert result["invariant"]["sigma_6_minus_deg"] == 12
        assert result["invariant"]["is_valid"]
        assert result["tangent_condition"]["violated"] == 0


# ============================================================
# 入口 (直接运行)
# ============================================================

if __name__ == "__main__":
    print(f"{'=' * 72}")
    print(f"  Phase 3 测试: 三维几何聚簇求解器")
    print(f"{'=' * 72}")

    # 运行简洁版测试
    pytest.main([__file__, "-v", "--tb=short", "-q"])
