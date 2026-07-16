"""
帧内握手引理 + 欧拉示性数 — 公理 4 的数学验证工具

握手引理:
    Σ deg(v) = 2|ε|
    任何无向图中，所有节点的度数之和 = 边数的 2 倍。
    在 SPUM 中每帧独立验证——上一帧的结论不保证下一帧成立。

欧拉示性数（离散高斯-博内）:
    χ = |V| - |ε| + |F|
    对球面拓扑闭合子图，χ = 2。
    Σ(6 − deg(v)) = 12  (Σ(6−deg) = 6χ(M) 的球面特例)

隔离原则:
    这些公式是 SPUM-图论的内部定理，从 N001-N013 推导。
    不直接引用经典图论教材中的证明——
    经典证明可能引入全局坐标系或预设性质。
"""

from typing import Dict, Set, Optional
from .graph import FrameGraph


class HandshakingVerifier:
    """帧内握手引理 + 欧拉示性数验证器。

    用法:
        verifier = HandshakingVerifier()
        verifier.verify_frame(graph)  # 返回验证报告
    """

    @staticmethod
    def verify_handshaking(graph: FrameGraph) -> dict:
        """验证帧内握手引理: Σ deg(v) = 2|ε|

        Returns:
            dict: {"passed": bool, "sum_deg": int, "expected": int, "delta": int}
        """
        sum_deg = sum(len(adj) for adj in graph._neighbors.values())
        expected = 2 * graph.edge_count
        passed = sum_deg == expected
        return {
            "passed": passed,
            "sum_deg": sum_deg,
            "expected": expected,
            "delta": sum_deg - expected,
        }

    @staticmethod
    def euler_characteristic(
        graph: FrameGraph,
        faces: Optional[int] = None
    ) -> dict:
        """计算帧内欧拉示性数: χ = |V| - |ε| + |F|

        对于 L0 图论，面数 F 是三角剖分的属性——
        如果未提供 faces，仅返回 V - E 部分。

        Returns:
            dict: {"v": int, "e": int, "f": int|None, "chi": int|None}
        """
        v = graph.node_count
        e = graph.edge_count
        chi = None if faces is None else v - e + faces
        return {"v": v, "e": e, "f": faces, "chi": chi}

    @staticmethod
    def angle_deficit_sum(graph: FrameGraph) -> dict:
        """计算 Σ(6 − deg(v)) —— 角度亏损总和。

        对闭合球面子图（χ=2），此值必须 = 12。
        这是 L0 拓扑常数 12 的帧内验证公式。

        Returns:
            dict: {"sum_6_minus_deg": int, "degrees": dict}
        """
        degrees = graph.degrees()
        total = sum(6 - d for d in degrees.values())
        return {
            "sum_6_minus_deg": total,
            "degrees": degrees,
        }


def euler_characteristic(
    vertex_count: int,
    edge_count: int,
    face_count: Optional[int] = None
) -> Optional[int]:
    """便捷函数: 计算欧拉示性数。"""
    if face_count is None:
        return None
    return vertex_count - edge_count + face_count
