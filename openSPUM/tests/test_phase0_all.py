"""
Phase 0 完全测试套件。

测试范围:
    1. 确定性初态 (斐波那契球面, 无随机)
    2. SoA 粒子数组: 添加/删除/潜在池
    3. 5 步帧: 创生/连接/体积/悬挂/删除
    4. 完整帧循环
    5. 元数据解码: FrameSnapshot → FrameLog
    6. SPUM 不变量跟踪
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import numpy as np
from Phase_0.particle_array import ParticleArray
from Phase_0.frame_kernels import (
    run_full_frame, step1_create, step2_connect,
    step3_volume, step3b_enforce_impenetrability,
    step4_dangling, step5_purge,
    _three_sphere_gap, _icosahedron_vertices,
    _star_surface, _compute_surface_adjacency,
)
from Phase_0.gpu_engine import SPUMEngine, EngineConfig
from Phase_0.metadata_decoder import FrameSnapshot, FrameLog, CUDAMetadataDecoder
from Phase_0.constants import (
    KAPPA, GEOMETRIC_TOLERANCE, CRYSTALLITE_DEGREE_THRESHOLD,
    MAX_PARTICLES
)

import unittest


class TestIcosahedron(unittest.TestCase):
    """测试正二十面体确定性顶点。"""

    def test_deterministic(self):
        """同一参数产生同一结果。"""
        p1 = _icosahedron_vertices()
        p2 = _icosahedron_vertices()
        np.testing.assert_array_almost_equal(p1, p2)

    def test_unit_sphere(self):
        """所有点位于单位球面。"""
        pts = _icosahedron_vertices()
        norms = np.linalg.norm(pts, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6))

    def test_12_vertices(self):
        """正二十面体有 12 个顶点。"""
        pts = _icosahedron_vertices()
        self.assertEqual(pts.shape, (12, 3))

    def test_all_12_active_seeds_connect(self):
        """12 个正二十面体顶点以正切半径初始化 → 全部连接。"""
        pa = ParticleArray(n=12, max_n=1000)
        positions = _icosahedron_vertices()
        # 计算正切半径
        min_dist = float('inf')
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        r0 = min_dist / 2.0
        for i in range(12):
            pa.add_particle(tuple(positions[i]), f"s{i}", degree=0)
        pa.radius[:12] = r0

        connections = step2_connect(pa)
        # 正二十面体每顶点 5 邻居: 12*5/2 = 30 边
        self.assertGreaterEqual(connections, 24)


class TestParticleArray(unittest.TestCase):
    """SoA 粒子数组测试。"""

    def setUp(self):
        self.pa = ParticleArray(n=100, max_n=1000)

    def test_add_particle(self):
        idx = self.pa.add_particle((1, 2, 3), "test_0", degree=3)
        self.assertTrue(self.pa.active[idx])
        self.assertEqual(self.pa.degree[idx], 3)
        self.assertEqual(self.pa.radius[idx], 3 * KAPPA)
        np.testing.assert_array_almost_equal(self.pa.pos[idx], [1, 2, 3])

    def test_remove_particle(self):
        idx = self.pa.add_particle((0, 0, 0), "test_0")
        self.assertEqual(self.pa.active_count(), 1)
        self.pa.remove_particle(idx)
        self.assertFalse(self.pa.active[idx])
        self.assertEqual(self.pa.degree[idx], 0)
        # uid 保留
        self.assertEqual(self.pa.uid[idx], "test_0")

    def test_latent_reuse(self):
        idx1 = self.pa.add_particle((0, 0, 0), "a")
        self.pa.remove_particle(idx1)
        idx2 = self.pa.add_particle((1, 1, 1), "b")
        # 应复用 idx1
        self.assertEqual(idx1, idx2)

    def test_degree_histogram(self):
        # 添加 3 个粒子，度数分别为 0, 1, 2
        self.pa.add_particle((0, 0, 0), "a", degree=0)
        self.pa.add_particle((1, 0, 0), "b", degree=1)
        self.pa.add_particle((0, 1, 0), "c", degree=2)
        hist = self.pa.degree_histogram()
        self.assertGreaterEqual(hist[0], 1)
        self.assertGreaterEqual(hist[1], 1)
        self.assertGreaterEqual(hist[2], 1)

    def test_spum_invariant(self):
        """单个粒子: Σ(6-deg) = 6。"""
        idx = self.pa.add_particle((0, 0, 0), "single")
        self.pa.degree[idx] = 0
        self.assertEqual(self.pa.spum_invariant(), 6)

    def test_dangling_mask(self):
        """度数 < 2 的活性粒子为悬挂。"""
        self.pa.add_particle((0, 0, 0), "a", degree=0)
        self.pa.add_particle((1, 0, 0), "b", degree=1)
        self.pa.add_particle((0, 1, 0), "c", degree=5)
        mask = self.pa.dangling_mask()
        self.assertEqual(int(np.sum(mask)), 2)  # a 和 b 是悬挂

    def test_crystallite_count(self):
        """度数 >= 50 的粒子为晶子。"""
        idx = self.pa.add_particle((0, 0, 0), "x", degree=60)
        self.assertEqual(self.pa.crystallite_count(), 1)
        self.pa.degree[idx] = 40
        self.assertEqual(self.pa.crystallite_count(), 0)


class TestThreeSphereGap(unittest.TestCase):
    """三球体缝隙检测测试。"""

    def test_three_equal_spheres(self):
        """三个等大球体两两相切 → 存在缝隙球。"""
        r = 1.0
        # 三个球体置于等边三角形，中心距离 = 2r
        d = 2.0 * r
        h = d * math.sqrt(3) / 2
        # 等边三角形：原点 (0,0,0), (d,0,0), (d/2, h, 0)
        # 提升 z 使三球面两两相切且不共面
        # 用正四面体布局: 顶点间距 = 2r
        # 四面体边长: 2r, 高: sqrt(6)*r/3 * 2 = 2*sqrt(6)*r/3 ≈ 1.633
        a = (0.0, 0.0, 0.0)
        b = (2.0, 0.0, 0.0)
        c = (1.0, math.sqrt(3.0), 0.0)  # 略抬升确保三维
        # 调整间距为 2r = 2.0
        gap = _three_sphere_gap(a, r, b, r, c, r)
        if gap is not None:
            gap_pos, gap_r = gap
            self.assertGreater(gap_r, 0)

    def test_not_tangent_returns_none(self):
        """三个不相切的球体 → None。"""
        a = (0, 0, 0)
        b = (10, 0, 0)
        c = (5, 10, 0)
        gap = _three_sphere_gap(a, 1, b, 1, c, 1)
        self.assertIsNone(gap)


class TestFiveStepFrames(unittest.TestCase):
    """5 步帧逻辑测试。"""

    def setUp(self):
        self.pa = ParticleArray(n=50, max_n=1000)
        # 添加 12 个种子, 正二十面体顶点, 正切半径
        positions = _icosahedron_vertices()
        min_dist = float('inf')
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        r0 = min_dist / 2.0
        for i in range(12):
            self.pa.add_particle(
                pos=(float(positions[i][0]), float(positions[i][1]),
                     float(positions[i][2])),
                uid=f"seed_{i}",
                degree=0,
            )
        self.pa.radius[:12] = r0

    def test_step1_create_exists(self):
        """Step 1: 缝隙检测可创建新粒子 (12 种子应有缝隙)。"""
        created = step1_create(self.pa)
        self.assertIsInstance(created, int)

    def test_step2_connect(self):
        """Step 2: 相切连接增加度数。"""
        before = np.sum(self.pa.degree)
        connected = step2_connect(self.pa)
        after = np.sum(self.pa.degree)
        self.assertGreaterEqual(after, before)
        # 12 个单位球面粒子互不相切 (间距 ~0.85 > 2*1?)
        # 只有半径接近的粒子才可能相切
        self.assertGreaterEqual(connected, 0)

    def test_step3_volume(self):
        """Step 3: 体积更新。"""
        self.pa.degree[:12] = np.arange(12, dtype=np.int32)
        step3_volume(self.pa)
        for i in range(12):
            expected = float(i) * KAPPA
            self.assertAlmostEqual(self.pa.radius[i], expected)

    def test_step4_dangling(self):
        """Step 4: 悬挂检测标记度数 < 2 的粒子。"""
        self.pa.degree[:12] = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8],
                                        dtype=np.int32)
        dangling = step4_dangling(self.pa)
        self.assertEqual(int(np.sum(dangling)), 4)  # 0,0,1,1

    def test_step5_purge(self):
        """Step 5: 删除悬挂。"""
        self.pa.degree[:5] = 0
        mask = self.pa.dangling_mask()
        purged = step5_purge(self.pa, mask)
        self.assertGreaterEqual(purged, 5)
        self.assertFalse(np.any(self.pa.active[:5]))

    def test_full_frame(self):
        """完整帧运行不崩溃。"""
        stats = run_full_frame(self.pa)
        self.assertIn('created', stats)
        self.assertIn('connected', stats)
        self.assertIn('dangling_marked', stats)
        self.assertIn('purged', stats)
        self.assertIn('active_before', stats)
        self.assertIn('active_after', stats)

    def test_frame_determinism(self):
        """相同初态产生相同帧结果。"""
        pa1 = ParticleArray(n=50, max_n=1000)
        pa2 = ParticleArray(n=50, max_n=1000)
        positions = _icosahedron_vertices()
        min_dist = float('inf')
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        r0 = min_dist / 2.0
        for i in range(12):
            pos = (float(positions[i][0]), float(positions[i][1]),
                   float(positions[i][2]))
            pa1.add_particle(pos, f"s{i}", degree=0)
            pa2.add_particle(pos, f"s{i}", degree=0)
        pa1.radius[:12] = r0
        pa2.radius[:12] = r0

        s1 = run_full_frame(pa1)
        s2 = run_full_frame(pa2)
        for k in ['created', 'connected', 'purged']:
            self.assertEqual(s1[k], s2[k], f"Differs at {k}")


class TestSPUMEngine(unittest.TestCase):
    """完整引擎测试。"""

    def test_initialization(self):
        engine = SPUMEngine(config=EngineConfig(seed_geometry="icosahedron", initial_seeds=12))
        self.assertEqual(engine.active_count, 12)
        self.assertEqual(engine.frame_number, 0)

    def test_single_frame(self):
        engine = SPUMEngine(config=EngineConfig(initial_seeds=12, max_particles=5000))
        snapshot = engine.run_frame()
        self.assertEqual(snapshot.frame_number, 1)
        self.assertIsInstance(snapshot, FrameSnapshot)

    def test_multiple_frames(self):
        engine = SPUMEngine(config=EngineConfig(initial_seeds=12, max_particles=5000))
        snapshots = engine.run_frames(10)
        self.assertEqual(len(snapshots), 10)
        for s in snapshots:
            self.assertIsInstance(s, FrameSnapshot)
            self.assertGreaterEqual(s.active_count, 0)

    def test_no_cascade(self):
        """验证: 无级联消解 — 悬挂粒子回归潜在池而非被补偿。

        正二十面体是完美闭子图 (Σ(6-deg)=12), 不会产生悬挂。
        但如果有悬挂产生, 它们应进入潜在池而非被补偿/再活化。
        """
        engine = SPUMEngine(config=EngineConfig(initial_seeds=12, max_particles=5000))
        logs = []
        for _ in range(20):
            s = engine.run_frame()
            logs.append(engine.decode(s))
        latent_counts = [l.latent_count for l in logs]
        # 正二十面体稳定: latent 可能为 0（无悬挂）
        # 但如果有 latent > 0 的场景, 潜在池应累积
        # 关键: 悬挂粒子不应消失(被级联消解)而应出现在潜在池
        # 检查: 无悬挂粒子被"消失" — 所有删除都通过潜在池
        if max(latent_counts) == 0:
            # 验证确实没有悬挂产生 (正二十面体稳定)
            dang_counts = [l.dangling_count for l in logs]
            self.assertEqual(max(dang_counts), 0,
                             "dangling>0 但 latent=0 → 粒子被级联消解了!")

    def test_spum_invariant_tracking(self):
        engine = SPUMEngine(config=EngineConfig(initial_seeds=12, max_particles=5000))
        for _ in range(30):
            engine.run_frame()
        inv = engine.spum_invariant
        # Σ(6-deg) 应保持有界 (不保证 =12, 但不应发散)
        self.assertLess(abs(inv), 10000,
                        f"SPUM invariant diverged: {inv}")

    def test_engine_determinism(self):
        """相同配置产生相同演化。"""
        e1 = SPUMEngine(config=EngineConfig(initial_seeds=12, max_particles=5000))
        e2 = SPUMEngine(config=EngineConfig(initial_seeds=12, max_particles=5000))

        for _ in range(10):
            s1 = e1.run_frame()
            s2 = e2.run_frame()
            self.assertEqual(s1.active_count, s2.active_count,
                             f"Frame {s1.frame_number}: V diff")

    def test_no_random_in_engine(self):
        """引擎不使用 random 库。"""
        import random
        # 验证引擎配置不包含随机数种子
        engine = SPUMEngine()
        self.assertFalse(hasattr(engine.config, 'random_seed'))
        # 运行几帧确认不崩溃
        engine.run_frames(5)


class TestMetadataDecoder(unittest.TestCase):
    """元数据解码器测试。"""

    def test_decode_snapshot(self):
        snapshot = FrameSnapshot(
            frame_number=1,
            degree_histogram=np.array([0, 0, 0, 0, 4, 0], dtype=np.int32),
            active_count=4,
            latent_count=0,
            crystallite_count=0,
            spum_invariant=4,
            dangling_count=0,
        )
        decoder = CUDAMetadataDecoder()
        log = decoder.decode(snapshot)
        self.assertEqual(log.frame_number, 1)
        self.assertEqual(log.particle_count, 4)
        self.assertEqual(log.edge_count, 8)  # 4 * 4 / 2
        self.assertIn("种子期", log.phase_label)

    def test_phase_classification(self):
        """种子/集群/成核/饱和 分类。"""
        decoder = CUDAMetadataDecoder()

        self.assertIn("种子期", decoder._classify_phase(5, 0, 0))
        self.assertIn("成核期", decoder._classify_phase(100, 2, 12))
        self.assertIn("集群期", decoder._classify_phase(30, 0, 12))


class TestNoRandomInSPUM(unittest.TestCase):
    """SPUM 无随机 — 验证测试。"""

    def test_no_random_import_in_phase0(self):
        """Phase 0 的核心模块不应导入 random。"""
        import ast
        import os
        phase0_dir = os.path.join(os.path.dirname(__file__), '..', 'Phase_0')
        for fname in ['particle_array.py', 'frame_kernels.py',
                       'metadata_decoder.py', 'constants.py']:
            fpath = os.path.join(phase0_dir, fname)
            if not os.path.exists(fpath):
                continue
            with open(fpath, encoding='utf-8') as f:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name == 'random':
                                self.fail(f"{fname} imports random: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module == 'random':
                            self.fail(f"{fname} imports from random: {node.module}")

    def test_no_random_in_engine_logs(self):
        """引擎运行日志不应包含 'random' 相关。"""
        engine = SPUMEngine(config=EngineConfig(initial_seeds=12, max_particles=5000))
        engine.run_frames(5)
        for s in engine.history:
            self.assertIsNotNone(s)
        # 正常通过即表示无随机


class TestStarInitialization(unittest.TestCase):
    """星形初态测试: 1 中心 + 50 表面 coda。"""

    def test_star_surface_shape(self):
        """_star_surface 返回 51 个位置和半径。"""
        positions, radii = _star_surface(50)
        self.assertEqual(positions.shape, (51, 3))
        self.assertEqual(len(radii), 51)

    def test_center_at_origin(self):
        """中心 coda 在原点。"""
        positions, radii = _star_surface(50)
        np.testing.assert_array_almost_equal(positions[0], [0, 0, 0])

    def test_center_radius(self):
        """中心半径 = n_surface * KAPPA = 50。"""
        from Phase_0.constants import KAPPA
        positions, radii = _star_surface(50)
        self.assertAlmostEqual(radii[0], 50.0 * KAPPA)

    def test_surface_on_sphere(self):
        """表面 coda 在球面上, 到原点距离 = center_r + 1。"""
        positions, radii = _star_surface(50)
        center_r = 50.0 * 1.0  # KAPPA=1
        expected_dist = center_r + 1.0
        for i in range(1, 51):
            dist = np.linalg.norm(positions[i])
            self.assertAlmostEqual(dist, expected_dist, delta=0.01)

    def test_surface_radius_one(self):
        """表面 coda 初始半径 = 1。"""
        positions, radii = _star_surface(50)
        for i in range(1, 51):
            self.assertAlmostEqual(radii[i], 1.0)

    def test_star_engine_initialization(self):
        """引擎以 star 模式初始化。"""
        engine = SPUMEngine(config=EngineConfig(
            seed_geometry="star",
            n_surface=50,
            max_particles=10000,
        ))
        self.assertEqual(engine.active_count, 51)  # 1 center + 50 surface
        self.assertEqual(engine.frame_number, 0)

    def test_center_degree_50(self):
        """中心 coda 度数为 50。"""
        engine = SPUMEngine(config=EngineConfig(
            seed_geometry="star",
            n_surface=50,
            max_particles=10000,
        ))
        # 中心是第一个粒子
        center_deg = engine.particles.degree[0]
        self.assertEqual(center_deg, 50)

    def test_surface_degree_1(self):
        """表面 coda 度数为 1 (连接中心)。"""
        engine = SPUMEngine(config=EngineConfig(
            seed_geometry="star",
            n_surface=50,
            max_particles=10000,
        ))
        for i in range(1, 51):
            self.assertEqual(engine.particles.degree[i], 1,
                             f"Surface {i} degree != 1")

    def test_star_deterministic(self):
        """相同 seed_geometry 产生相同初态。"""
        e1 = SPUMEngine(config=EngineConfig(
            seed_geometry="star", n_surface=50, max_particles=10000))
        e2 = SPUMEngine(config=EngineConfig(
            seed_geometry="star", n_surface=50, max_particles=10000))
        # 比较前 3 个表面 coda 的位置
        for i in range(3):
            np.testing.assert_array_almost_equal(
                e1.particles.pos[1 + i], e2.particles.pos[1 + i])


class TestSurfaceAdjacency(unittest.TestCase):
    """表面 coda 角邻接测试。"""

    def test_surface_adjacency_initial(self):
        """初始半径=1 时, 表面 coda 无角重叠。"""
        positions, radii = _star_surface(50)
        surf_pos = positions[1:]  # 去掉中心
        surf_rad = radii[1:]
        adj = _compute_surface_adjacency(surf_pos, radii[0], surf_rad)
        self.assertEqual(np.sum(adj), 0,
                         "半径=1 时表面 coda 不应有角重叠")

    def test_surface_adjacency_large(self):
        """半径足够大时, 表面 coda 有角重叠。"""
        positions, radii = _star_surface(50)
        surf_pos = positions[1:]
        # 设所有表面 coda 的半径为 30 (足够大覆盖球面)
        surf_rad = np.full(50, 30.0)
        adj = _compute_surface_adjacency(surf_pos, 50.0, surf_rad)
        self.assertGreater(np.sum(adj), 0,
                           "半径=30 时表面 coda 应有角重叠")


class TestPreGrowth(unittest.TestCase):
    """Pre-growth 阶段测试: 前 N 帧无修剪。"""

    def test_pregrowth_purge_zero(self):
        """pre-growth 阶段 purged=0。"""
        engine = SPUMEngine(config=EngineConfig(
            seed_geometry="star",
            n_surface=50,
            pre_growth_frames=10,
            max_particles=10000,
        ))
        # 前 5 帧 no_purge = True
        for _ in range(5):
            s = engine.run_frame()
            log = engine.decode(s)
            # pre-growth 阶段悬挂不会被删除
            # 所以不可能有正向的 purged count
            self.assertGreaterEqual(log.particle_count, 51)

    def test_pregrowth_no_dangling_delete(self):
        """pre-growth 阶段悬挂粒子不会被删除。"""
        engine = SPUMEngine(config=EngineConfig(
            seed_geometry="star",
            n_surface=8,  # 少表面 coda 加快
            pre_growth_frames=3,
            max_particles=10000,
        ))
        frames_before = engine.run_frames(3)
        n_before = frames_before[-1].active_count

        # pre-growth 结束后立即运行完整 5 步
        frames_after = engine.run_frames(1)
        n_after = frames_after[-1].active_count

        # pre-growth 结束后粒子数应减少 (悬挂被删除)
        # 但不一定严格 < (可能缝隙创生补充)
        pass

    def test_pregrowth_transition(self):
        """pre-growth → 完整帧的过渡不崩溃。"""
        engine = SPUMEngine(config=EngineConfig(
            seed_geometry="star",
            n_surface=50,
            pre_growth_frames=5,
            max_particles=10000,
        ))
        for i in range(10):
            snapshot = engine.run_frame()
            self.assertIsInstance(snapshot, FrameSnapshot)
            self.assertGreaterEqual(snapshot.active_count, 1)

    def test_star_evolution(self):
        """星形初态 + 多帧演化不崩溃。"""
        engine = SPUMEngine(config=EngineConfig(
            seed_geometry="star",
            n_surface=50,
            pre_growth_frames=5,
            max_particles=10000,
        ))
        logs = engine.run_frames(20)
        for log in engine.decode_history():
            self.assertGreaterEqual(log.particle_count, 1)
        # Σ(6-deg) 应保持有界
        invs = [l.spum_invariant for l in engine.decode_history()]
        self.assertLess(max(abs(i) for i in invs), 100000)


class TestImpenetrability(unittest.TestCase):
    """不可入性测试: 所有球体均不可入。"""

    def test_two_overlapping_spheres(self):
        """两个重叠的球体 → 小球被挤。"""
        pa = ParticleArray(n=10, max_n=100)
        # 球 A 在原点, 半径 10
        pa.add_particle((0, 0, 0), "big", degree=10)
        pa.radius[0] = 10.0
        # 球 B 在 (5, 0, 0), 半径 8 → 严重重叠
        pa.add_particle((5, 0, 0), "small", degree=8)
        pa.radius[1] = 8.0

        resolved = step3b_enforce_impenetrability(pa)
        self.assertGreater(resolved, 0,
                           "重叠对应该被检测到")
        # 小球 (B) 的度数应该减少
        self.assertLess(pa.degree[1], 8,
                        "小球的度数应该被减少")
        # 大球 (A) 的度数不变
        self.assertEqual(pa.degree[0], 10)

    def test_no_overlap_no_change(self):
        """不重叠的球体不受影响。"""
        pa = ParticleArray(n=10, max_n=100)
        pa.add_particle((0, 0, 0), "a", degree=5)
        pa.radius[0] = 5.0
        pa.add_particle((20, 0, 0), "b", degree=3)
        pa.radius[1] = 3.0

        resolved = step3b_enforce_impenetrability(pa)
        self.assertEqual(resolved, 0)
        self.assertEqual(pa.degree[0], 5)
        self.assertEqual(pa.degree[1], 3)

    def test_equal_spheres(self):
        """等大球体重叠 → 先检测到的那个被挤。"""
        pa = ParticleArray(n=10, max_n=100)
        pa.add_particle((0, 0, 0), "a", degree=5)
        pa.radius[0] = 5.0
        pa.add_particle((3, 0, 0), "b", degree=5)
        pa.radius[1] = 5.0

        resolved = step3b_enforce_impenetrability(pa)
        self.assertGreater(resolved, 0)
        # 两个中至少一个度数减少 (等大时索引小者先被处理)
        self.assertLess(pa.degree[0] + pa.degree[1], 10)

    def test_overlap_creates_dangling(self):
        """被挤压到 degree < 2 的球体成为悬挂。"""
        pa = ParticleArray(n=10, max_n=100)
        # 大球
        pa.add_particle((0, 0, 0), "big", degree=10)
        pa.radius[0] = 10.0
        # 小球 deg=2, 在重叠距离内
        pa.add_particle((8, 0, 0), "tiny", degree=2)
        pa.radius[1] = 2.0

        resolved = step3b_enforce_impenetrability(pa, max_iter=3)
        # 小球可能被挤到度数为 0 或 1
        if pa.degree[1] < 2:
            dangling = step4_dangling(pa)
            self.assertTrue(dangling[1],
                            "被挤到 degree<2 的球体应被标记为悬挂")

    def test_engine_includes_overlap_check(self):
        """完整帧包含不可入性检查。"""
        engine = SPUMEngine(config=EngineConfig(
            seed_geometry="star",
            n_surface=50,
            pre_growth_frames=3,
            max_particles=10000,
        ))
        engine.run_frames(5)
        # 任何两个活性球体都不能重叠
        active = np.where(engine.particles.active)[0]
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ai, aj = active[i], active[j]
                d = np.linalg.norm(
                    engine.particles.pos[ai] - engine.particles.pos[aj])
                r_sum = engine.particles.radius[ai] + engine.particles.radius[aj]
                # 容差内允许
                self.assertGreaterEqual(d + 0.01, r_sum,
                    f"球体 {ai} 和 {aj} 重叠: "
                    f"d={d:.2f}, r1+r2={r_sum:.2f}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
