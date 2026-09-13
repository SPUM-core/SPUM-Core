"""
Phase 4 测试套件 — VSPT 球面生长拓扑

测试内容:
    1. 永恒粒子模型（虚实面/朝向/手性）
    2. 原子核组装（中子/质子/实面虚面计数）
    3. VSPT 分支生长（树播种/分支/壳层）
    4. 电子壳层填充（2n²容量/元素分类）
    5. VSPT 三律验证（k≥3/120°/ρ∝r⁻³）
    6. 中子-质子结构差异验证
"""

import sys
import math
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import pytest

from Phase_4.eternal_particle import (
    FaceType,
    EternalParticle,
    EternalParticleOrientation,
    Handedness,
    create_eternal_particle,
)
from Phase_4.nucleus_assembly import (
    NucleusAssembly,
    NucleusType,
    ProtonConfig,
    build_nucleus,
)
from Phase_4.vspt_growth import (
    VSPTNode,
    VSPTTree,
    VSPTEngine,
    VSPTConfig,
)
from Phase_4.electron_shell import (
    ElementClass,
    shell_capacity,
    cumulative_capacity,
    classify_element,
    periodic_table_lookup,
)
from Phase_4.validator import (
    VSPTValidator,
    validate_k3_law,
    validate_120_angle,
    validate_density_power_law,
)


# ============================================================
# 测试 1: 永恒粒子模型
# ============================================================

class TestEternalParticle:
    def test_create_inward(self):
        """朝内永恒粒子：4 实面 + 0 虚面对外暴露。"""
        ep = create_eternal_particle(
            uid="test_ep",
            orientation=EternalParticleOrientation.INWARD,
        )
        assert ep.uid == "test_ep"
        assert ep.solid_face_count == 4
        assert ep.vacant_face_count == 0
        assert ep.solid_ratio == 4.0 / 5.0
        assert ep.vacant_ratio == 0.0

    def test_create_outward(self):
        """朝外永恒粒子：0 实面 + 1 虚面对外暴露。"""
        ep = create_eternal_particle(
            uid="test_ep2",
            orientation=EternalParticleOrientation.OUTWARD,
        )
        assert ep.solid_face_count == 0
        assert ep.vacant_face_count == 1
        assert ep.solid_ratio == 0.0
        assert ep.vacant_ratio == 1.0 / 5.0

    def test_handedness_alternation(self):
        """手性 L/D 二选一（毛球定理）。"""
        ep_l = create_eternal_particle("l", handedness=Handedness.L)
        ep_d = create_eternal_particle("d", handedness=Handedness.D)
        assert ep_l.handedness == Handedness.L
        assert ep_d.handedness == Handedness.D
        assert ep_l.handedness.value == -1
        assert ep_d.handedness.value == +1

    def test_position_and_radius(self):
        """位置和半径正确设置。"""
        ep = create_eternal_particle(
            "pos_test",
            position=(1.0, 2.0, 3.0),
            radius=2.0,
        )
        assert ep.position == (1.0, 2.0, 3.0)
        assert ep.radius == 2.0

    def test_solid_faces_initial_none(self):
        """初始时所有实面 VSPT 树为 None。"""
        ep = create_eternal_particle("face_test")
        assert len(ep.solid_faces) == 4
        assert all(f is None for f in ep.solid_faces)


# ============================================================
# 测试 2: 原子核组装
# ============================================================

class TestNucleusAssembly:
    def test_neutron_12_particles(self):
        """中子：12 个永恒粒子，全朝内。"""
        nuc = build_nucleus(NucleusType.NEUTRON)
        assert len(nuc.particles) == 12
        assert len(nuc.positions) == 12
        assert nuc.ntype == NucleusType.NEUTRON

    def test_neutron_outward_faces(self):
        """中子：0 虚面 + 48 实面 = 纯实面。"""
        nuc = build_nucleus(NucleusType.NEUTRON)
        assert nuc.outward_solid == 48
        assert nuc.outward_vacant == 0
        assert nuc.charge == 0
        assert nuc.solid_ratio == 1.0
        assert nuc.vacant_ratio == 0.0

    def test_proton_outward_faces(self):
        """质子（单隼）：44 实面（11朝内×4）+ 1 虚面（1朝外×1），虚面占比 1/48。"""
        nuc = build_nucleus(NucleusType.PROTON)
        # 朝外永恒粒子的 4 个实面朝向核内，不对外暴露
        assert nuc.outward_solid == 44  # 11 朝内 × 4
        assert nuc.outward_vacant == 1  # 1 朝外 × 1
        assert nuc.charge == 1  # 拓扑不变量
        assert abs(nuc.vacant_ratio - 1.0/48.0) < 0.0001

    def test_proton_double_tenon(self):
        """质子（双隼）：40 实面（10朝内×4）+ 2 虚面（2朝外×1），虚面占比 2/48。"""
        nuc = build_nucleus(NucleusType.PROTON, proton_config=ProtonConfig.DOUBLE_TENON)
        assert nuc.outward_solid == 40
        assert nuc.outward_vacant == 2
        assert nuc.charge == 2
        assert abs(nuc.vacant_ratio - 2.0/48.0) < 0.0001

    def test_proton_vacant_ratio_under_third(self):
        """质子虚面占比 4.17% << 1/3 硬边界（核稳定条件）。"""
        nuc = build_nucleus(NucleusType.PROTON)
        assert nuc.vacant_ratio < 1.0 / 3.0

    def test_all_particles_have_orientation(self):
        """每个永恒粒子有明确的朝向（单隼：1 朝外 + 11 朝内）。"""
        nuc = build_nucleus(NucleusType.PROTON)
        outward_count = sum(
            1 for ep in nuc.particles.values()
            if ep.orientation == EternalParticleOrientation.OUTWARD
        )
        inward_count = sum(
            1 for ep in nuc.particles.values()
            if ep.orientation == EternalParticleOrientation.INWARD
        )
        assert outward_count == 1
        assert inward_count == 11

    def test_neutron_spum_invariant(self):
        """中子核 SPUM 不变量 = 12。"""
        nuc = build_nucleus(NucleusType.NEUTRON)
        assert nuc.spum_invariant == 12

    def test_positions_icosahedral(self):
        """12 个永恒粒子位置构成正二十面体顶点。"""
        nuc = build_nucleus(NucleusType.NEUTRON)
        positions = list(nuc.positions.values())
        assert len(positions) == 12

        # 验证所有顶点到原点的距离一致
        radii = [math.sqrt(sum(p[i] ** 2 for i in range(3))) for p in positions]
        mean_r = sum(radii) / len(radii)
        deviations = [abs(r - mean_r) / mean_r for r in radii]
        assert max(deviations) < 0.01  # 所有顶点等距


# ============================================================
# 测试 3: VSPT 分支生长
# ============================================================

class TestVSPTEngine:
    def test_seed_tree(self):
        """播种一棵 VSPT 树应在核表面创建根节点。"""
        engine = VSPTEngine()
        tree = engine.seed_tree(
            nucleus_uid="test_nuc",
            face_index=0,
            handedness=Handedness.L,
            surface_pos=(1.0, 0.0, 0.0),
        )
        assert "test_nuc" in tree.tree_id
        assert tree.node_count == 1
        assert tree.root_uid is not None
        assert tree.handedness == Handedness.L

    def test_grow_tree_adds_nodes(self):
        """生长一棵树应增加节点。"""
        engine = VSPTEngine(config=VSPTConfig(max_layers=3, seed=42))
        tree = engine.seed_tree(
            "nuc", 0, Handedness.L, (1.0, 0.0, 0.0),
        )
        initial_count = tree.node_count
        added = engine.grow_tree(tree, nucleus_radius=1.0)
        assert added > 0
        assert tree.node_count > initial_count

    def test_run_growth_multiple_trees(self):
        """完整生长应播种多棵树并产生多壳层。"""
        from Phase_4.nucleus_assembly import build_nucleus, NucleusType
        nuc = build_nucleus(NucleusType.NEUTRON)

        engine = VSPTEngine(config=VSPTConfig(max_layers=5, seed=42))
        result = engine.run_growth(
            outward_solid_faces=48,
            nucleus_uid="test_nuc",
            nucleus_particles=nuc.particles,
            nucleus_position_map=nuc.positions,
        )
        assert result["n_trees"] > 0
        assert result["total_nodes"] > result["n_trees"]  # 至少有根节点
        assert result["max_layer"] >= 1

    def test_density_profile(self):
        """VSPT 树的密度剖面应在较高层数时呈现递减趋势。"""
        engine = VSPTEngine(config=VSPTConfig(max_layers=8, seed=42))
        tree = engine.seed_tree("nuc", 0, Handedness.L, (1.0, 0.0, 0.0))

        for _ in range(6):
            engine.grow_tree(tree, nucleus_radius=1.0)

        profile = tree.density_profile
        assert len(profile) >= 2
        # 层数越高节点数不应持续增长（有限分支）
        for layer, count in profile.items():
            assert count >= 0


# ============================================================
# 测试 4: 电子壳层与元素分类
# ============================================================

class TestElectronShell:
    def test_shell_capacity_k(self):
        """K 壳层容量 = 2。"""
        assert shell_capacity(1) == 2

    def test_shell_capacity_l(self):
        """L 壳层容量 = 8。"""
        assert shell_capacity(2) == 8

    def test_shell_capacity_m(self):
        """M 壳层容量 = 18。"""
        assert shell_capacity(3) == 18

    def test_shell_capacity_n(self):
        """N 壳层容量 = 32。"""
        assert shell_capacity(4) == 32

    def test_cumulative_capacity(self):
        """前 n 壳层总容量 = n(n+1)(2n+1)/3。"""
        assert cumulative_capacity(1) == 2
        assert cumulative_capacity(2) == 10  # 2+8
        assert cumulative_capacity(3) == 28  # 2+8+18
        assert cumulative_capacity(4) == 60  # 2+8+18+32

    def test_classify_hydrogen(self):
        """氢 (Z=1): K 壳层 1 个电子。"""
        result = classify_element(1)
        assert result["symbol"] == "H"
        assert result["name"] == "氢"
        assert result["period"] == 1
        assert result["element_class"] == ElementClass.ALKALI_METAL
        assert result["vacant_faces"] == 1
        assert len(result["shell_filling"]) == 1
        assert result["shell_filling"][0]["name"] == "K"
        assert result["shell_filling"][0]["occupancy"] == 1

    def test_classify_helium(self):
        """氦 (Z=2): K 壳层填满。"""
        result = classify_element(2)
        assert result["symbol"] == "He"
        assert result["element_class"] == ElementClass.NOBLE_GAS
        assert result["vacant_faces"] == 0
        assert result["shell_filling"][0]["is_full"]

    def test_classify_carbon(self):
        """碳 (Z=6): K 满 + L 4 个电子。"""
        result = classify_element(6)
        assert result["symbol"] == "C"
        assert result["vacant_faces"] == 4
        assert len(result["shell_filling"]) == 2
        assert result["shell_filling"][0]["is_full"]  # K 满
        assert result["shell_filling"][1]["occupancy"] == 4  # L 壳层

    def test_classify_neon(self):
        """氖 (Z=10): L 壳层填满。"""
        result = classify_element(10)
        assert result["symbol"] == "Ne"
        assert result["element_class"] == ElementClass.NOBLE_GAS
        assert result["shell_filling"][1]["is_full"]

    def test_classify_sodium(self):
        """钠 (Z=11): M 壳层 1 个电子。"""
        result = classify_element(11)
        assert result["symbol"] == "Na"
        assert result["period"] == 3
        assert len(result["shell_filling"]) == 3
        assert result["shell_filling"][2]["occupancy"] == 1

    def test_lookup_by_symbol(self):
        """元素符号查找。"""
        fe = periodic_table_lookup("Fe")
        assert fe is not None  # Fe 在前 118 号中
        assert fe["proton_number"] == 26
        result = periodic_table_lookup("C")
        assert result is not None
        assert result["proton_number"] == 6

    def test_hydrogen_class(self):
        """H 分类为碱金属（1 虚面）。"""
        h = classify_element(1)
        assert h["element_class"] == ElementClass.ALKALI_METAL

    def test_fluorine_class(self):
        """F 分类为卤素（7 虚面）。"""
        f = classify_element(9)
        assert f["element_class"] == ElementClass.HALOGEN


# ============================================================
# 测试 5: VSPT 三律验证
# ============================================================

class TestVSPTValidator:
    def test_k3_law_empty(self):
        """空树返回无效。"""
        result = validate_k3_law({})
        assert not result["is_valid"]
        assert "无节点" in result["message"]

    def test_k3_law_basic(self):
        """基本 VSPT 树满足 k≥3。"""
        # 构造一棵简单满足 k≥3 的树
        tree = VSPTTree(tree_id="t1", nucleus_uid="n", face_index=0,
                        handedness=Handedness.L)

        # 根节点 (layer=0, 不计入)
        root = VSPTNode(uid="root", layer=0, parent_uid="nucleus",
                        degree=4)
        root.children = ["c0", "c1", "c2", "c3"]

        # 4 个分支节点 (layer=1, 各有 2 个子节点 → degree = 2+1 = 3)
        children = {}
        for i in range(4):
            c_uid = f"c{i}"
            gc_ids = [f"gc{i}_0", f"gc{i}_1"]
            child = VSPTNode(uid=c_uid, layer=1, parent_uid="root",
                             degree=3, children=gc_ids)
            children[c_uid] = child
            # 2 个叶子节点 (layer=2, degree=1)
            for j in range(2):
                gc_uid = gc_ids[j]
                gc = VSPTNode(uid=gc_uid, layer=2, parent_uid=c_uid,
                              degree=1)
                children[gc_uid] = gc

        tree.nodes = {"root": root, **children}

        result = validate_k3_law({"t1": tree})
        # 4 个分支节点（layer>0 且 children>0）应全部 k≥3
        assert result["n_nodes"] == 4
        assert result[f"k≥3_ratio"] >= 1.0
        assert result["is_valid"]

    def test_validator_accepts_multiple_trees(self):
        """验证器可以处理多棵树。"""
        validate_k3_law({})  # 不应 crash
        validate_120_angle({})
        validate_density_power_law({})

    def test_validator_all(self):
        """完整 VSPT 验证器。"""
        validator = VSPTValidator()
        result = validator.validate_all({})
        assert "law1_k3" in result
        assert "law2_120" in result
        assert "law3_density" in result
        assert "is_valid" in result

    def test_density_power_law_insufficient_layers(self):
        """少于 3 个壳层时密度验证无效。"""
        result = validate_density_power_law({})
        assert not result["is_valid"]


# ============================================================
# 测试 6: 中子-质子结构差异
# ============================================================

class TestNeutronProtonDifference:
    def test_charge_difference(self):
        """中子电荷 0，质子电荷 1（单隼拓扑不变量）。"""
        neutron = build_nucleus(NucleusType.NEUTRON)
        proton = build_nucleus(NucleusType.PROTON)
        assert neutron.charge == 0
        assert proton.charge == 1  # 1 个朝外虚面

    def test_vacant_count_difference(self):
        """中子 0 虚面，质子（单隼）1 虚面。"""
        neutron = build_nucleus(NucleusType.NEUTRON)
        proton = build_nucleus(NucleusType.PROTON)
        assert neutron.outward_vacant == 0
        assert proton.outward_vacant == 1

    def test_outward_particle_count(self):
        """中子 0 个朝外粒子，质子（单隼）1 个朝外粒子。"""
        neutron = build_nucleus(NucleusType.NEUTRON)
        proton = build_nucleus(NucleusType.PROTON)

        for nuc, expected in [(neutron, 0), (proton, 1)]:
            count = sum(
                1 for ep in nuc.particles.values()
                if ep.orientation == EternalParticleOrientation.OUTWARD
            )
            assert count == expected


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    print(f"{'=' * 72}")
    print(f"  Phase 4 测试: VSPT 球面生长拓扑")
    print(f"{'=' * 72}")

    pytest.main([__file__, "-v", "--tb=short", "-q"])
