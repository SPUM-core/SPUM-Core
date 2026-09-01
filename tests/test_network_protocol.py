"""
SPUM 网络协议层测试 — network/protocol
========================================

覆盖:
    GraphStore — canonical 加载 / 边验证 / 轨迹读写 / 合并
    TrajectoryEncoder — 路径编码 / 持久化 / 解码 / 分叉 / diff
    SessionManifest — 会话元数据
    protocol_integration — load_env / write_trajectory

所有测试使用临时目录，不触碰仓库内真实快照。

运行:
    python -m unittest tests.test_network_protocol -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from network.protocol.graph_store import (
    GraphStore, TrajectoryEdge, is_valid_node_id, VALID_EDGE_TYPES,
)
from network.protocol.trajectory_encoder import TrajectoryEncoder, Trajectory

CANONICAL_LINES = [
    "N001 | N002 | derives_from | 差异是存在自身的第一刻痕",
    "N002 | N003 | derives_from | 差异需要边界来界定和稳定",
    "N005 | N006 | derives_from | 开放的关系链必须闭合",
    "N013 | N009 | drives | 不完美是下一帧创生事件的启动信号",
    "N014 | N015 | refines | 晶子是空间粒子的一种特殊饱和形态",
]


def make_env_dir():
    """创建临时目录，写入规范 edges.txt（base/network/edges.txt 结构）。"""
    tmp = tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    network_dir = base / "network"
    network_dir.mkdir()
    (network_dir / "edges.txt").write_text(
        "\n".join(CANONICAL_LINES) + "\n", encoding="utf-8"
    )
    return tmp, base


class TestNodeIdValidation(unittest.TestCase):
    """节点 ID 格式验证。"""

    def test_valid_ids(self):
        for nid in ["N001", "N022", "SOC-001", "ECON-002", "LING-003",
                    "FI-001", "AGT-001", "VSPT-001", "R1", "R9"]:
            self.assertTrue(is_valid_node_id(nid), f"应合法: {nid}")

    def test_invalid_ids(self):
        for nid in ["001", "N01", "N", "abc", "SOC001", "N00a", "N-", "-001"]:
            self.assertFalse(is_valid_node_id(nid), f"应非法: {nid}")


class TestGraphStore(unittest.TestCase):
    """边集存储。"""

    def setUp(self):
        self.tmp, self.base = make_env_dir()
        self.store = GraphStore(self.base / "network")

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_canonical(self):
        edges = self.store.load_canonical()
        self.assertGreaterEqual(len(edges), len(CANONICAL_LINES))
        self.assertIn("N001|N002|derives_from", edges)

    def test_load_canonical_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as empty:
            store = GraphStore(empty)
            with self.assertRaises(FileNotFoundError):
                store.load_canonical()

    def test_validate_edge_canonical(self):
        self.store.load_canonical()
        valid, reason = self.store.validate_edge("N001", "N002", "derives_from")
        self.assertTrue(valid)
        self.assertIn("canonical", reason)

    def test_validate_edge_novel(self):
        self.store.load_canonical()
        valid, reason = self.store.validate_edge("N001", "N013", "drives")
        self.assertTrue(valid)
        self.assertEqual(reason, "novel")

    def test_validate_edge_invalid_type(self):
        self.store.load_canonical()
        valid, reason = self.store.validate_edge("N001", "N002", "magic")
        self.assertFalse(valid)
        self.assertIn("非法边类型", reason)

    def test_write_and_read_trajectory(self):
        self.store.load_canonical()
        edge = TrajectoryEdge(
            source="N001", target="N002", edge_type="derives_from",
            session_id="S-TEST-001", trajectory_id="T-TEST-001",
            timestamp="2026-08-31T00:00:00Z", confidence=1.0,
        )
        path = self.store.write_trajectory("T-TEST-001", [edge])
        self.assertTrue(Path(path).exists())
        edges = self.store.read_trajectory("T-TEST-001")
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].source, "N001")
        self.assertEqual(edges[0].target, "N002")

    def test_read_missing_trajectory_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.store.read_trajectory("T-NOT-EXIST")

    def test_merge_all(self):
        self.store.load_canonical()
        e1 = TrajectoryEdge("N001", "N002", "derives_from", "S-1",
                            "T-A", "2026-08-31T00:00:00Z")
        e2 = TrajectoryEdge("N002", "N003", "derives_from", "S-2",
                            "T-B", "2026-08-31T00:00:01Z")
        self.store.write_trajectory("T-A", [e1])
        self.store.write_trajectory("T-B", [e2])
        merged_path = self.store.merge_all()
        self.assertTrue(Path(merged_path).exists())
        self.assertEqual(self.store.get_merged_edge_count(), 2)

    def test_stats(self):
        self.store.load_canonical()
        stats = self.store.stats()
        self.assertEqual(stats["trajectory_count"], 0)
        self.assertGreaterEqual(stats["canonical_edges"], len(CANONICAL_LINES))

    def test_trajectory_edge_roundtrip(self):
        e = TrajectoryEdge("N001", "N002", "derives_from", "S-1",
                           "T-1", "2026-08-31T00:00:00Z", 0.9)
        line = e.to_line()
        restored = TrajectoryEdge.from_line(line)
        self.assertEqual(restored, e)

    def test_trajectory_edge_from_bad_line(self):
        self.assertIsNone(TrajectoryEdge.from_line("garbage"))
        self.assertIsNone(TrajectoryEdge.from_line("A|B"))


class TestTrajectoryEncoder(unittest.TestCase):
    """轨迹编码器。"""

    def setUp(self):
        self.tmp, self.base = make_env_dir()
        self.store = GraphStore(self.base / "network")
        self.encoder = TrajectoryEncoder(self.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_encode_path(self):
        traj = self.encoder.encode_path(
            path=["N001", "N002", "N003"],
            session_id="S-TEST-001",
            model="gpt-4o",
            origin="测试",
        )
        self.assertEqual(traj.node_count(), 3)
        self.assertEqual(traj.edge_count(), 2)
        # 边类型从 canonical 推断
        self.assertEqual(traj.edges[0].edge_type, "derives_from")

    def test_encode_path_signature(self):
        t1 = self.encoder.encode_path(["N001", "N002"], "S-1", "m", "o")
        t2 = self.encoder.encode_path(["N001", "N002"], "S-1", "m", "o")
        self.assertEqual(t1.signature, t2.signature)

    def test_persist_and_decode(self):
        traj = self.encoder.encode_path(
            path=["N001", "N002", "N003"],
            session_id="S-TEST-002",
            model="gpt-4o",
            origin="持久化测试",
        )
        path = self.encoder.persist(traj)
        self.assertTrue(Path(path).exists())
        decoded = self.encoder.decode(traj.trajectory_id)
        self.assertEqual(decoded.path, traj.path)
        self.assertEqual(decoded.model, "gpt-4o")
        self.assertEqual(decoded.origin, "持久化测试")

    def test_diff(self):
        t1 = self.encoder.encode_path(["N001", "N002"], "S-1", "m", "o")
        t2 = self.encoder.encode_path(["N001", "N005"], "S-1", "m", "o")
        d = self.encoder.diff(t1, t2)
        self.assertIn("N002", d["nodes_only_in_first"])
        self.assertIn("N005", d["nodes_only_in_second"])

    def test_fork(self):
        parent = self.encoder.encode_path(["N001", "N002"], "S-1", "m", "o")
        child = self.encoder.fork(parent, ["N001", "N002", "N003"], "继续推理")
        self.assertEqual(child.metadata.get("parent_trajectory"),
                         parent.trajectory_id)
        self.assertIn("forked_from", child.origin)

    def test_encode_edges(self):
        e = TrajectoryEdge("N001", "N002", "derives_from", "S-1",
                           "T-1", "2026-08-31T00:00:00Z")
        traj = self.encoder.encode_edges([e], "S-1", "m", "o")
        self.assertEqual(traj.edge_count(), 1)


class TestProtocolIntegration(unittest.TestCase):
    """单函数入口。"""

    def setUp(self):
        self.tmp, self.base = make_env_dir()
        import network.protocol.protocol_integration as pi
        self.pi = pi
        self._old_cache = pi._env_cache
        pi._env_cache = None

    def tearDown(self):
        self.pi._env_cache = self._old_cache
        self.tmp.cleanup()

    def test_load_env_with_base_dir(self):
        env = self.pi.load_env(base_dir=self.base)
        self.assertIsNotNone(env.store)
        self.assertIn("canonical_edges", env.stats())
        self.assertIn("manifests", env.stats())

    def test_load_env_auto_detect(self):
        """自动检测应定位到真实仓库根目录。"""
        env = self.pi.load_env()
        self.assertTrue(env.base_dir.exists())
        self.assertTrue((env.base_dir / "network" / "edges.txt").exists())

    def test_write_trajectory_full_flow(self):
        """完整回写流程——在临时环境中验证，不污染真实快照。"""
        env = self.pi.load_env(base_dir=self.base)
        self.pi._env_cache = env  # write_trajectory 内部调用 load_env() 走缓存
        result = self.pi.write_trajectory(
            path=["N001", "N002", "N003"],
            model="test-model",
            origin="单元测试",
        )
        self.assertIn("trajectory_id", result)
        self.assertIn("session_id", result)
        self.assertEqual(result["edge_count"], 2)
        self.assertEqual(result["edge_validation"][0]["status"], "canonical")
        # 轨迹文件应写入临时目录而非真实仓库
        self.assertIn(self.base.name, result["snap_file"])


if __name__ == "__main__":
    unittest.main()
