"""
SPUM API — 统一 AI 调用入口
==============================

将全部公理、帧引擎、跨域桥接、推理工具聚合为单一入口。
AI 只需 `from spum import SPUM` 即可调用全部能力。

典型用法:

    from spum import SPUM
    spum = SPUM()

    # 1. 公理原语
    spum.add_edge("A", "B")       # 公理1: 关系第一性
    spum.degree("A")              # 查询度数
    spum.dangling_density()       # 公理2: 悬挂端密度 delta
    spum.angle_deficit_sum()      # 公理4: Sigma(6-deg) = 12
    spum.check_imperfection()     # 公理5: 不完美检查

    # 2. 帧演化
    spum.new_frame()
    spum.v_plus("X", "Y")
    spum.v_minus("X", "Z")
    spum.commit()

    # 3. 跨域映射
    spum.map_concept("physics", "gravity")
    spum.map_concept("rushidao", "dao")

    # 4. 推理
    spum.derive("N005", "N013")
    spum.multi_path_validate(...)
    spum.review_text("...")

    # 5. 信息
    print(spum.summary())
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict

# 本包模块
from .axioms import Axioms, Constants
from .frame import FrameEngine, FrameType, FrameState
from .domain import DomainBridge, ConceptMapping
from .reason import (
    ReasoningEngine, DerivationChain, MultiPathResult,
    ReviewResult, ReviewViolation, ReviewVerdict,
)


class SPUM:
    """SPUM 统一 API 入口。

    所有 SPUM 能力通过此类的单一接口暴露给 AI。
    内部委托给专用引擎（公理/帧/域/推理）。

    用法:
        spum = SPUM()

        # 直接操作公理
        spum.add_edge("A", "B")
        print(spum.delta)

        # 帧演化
        spum.new_frame()
        spum.v_plus("X", "Y")
        spum.commit()

        # 推理
        spum.derive("N001", "N020")
        spum.map_concept("physics", "gravity")
    """

    def __init__(self):
        # 核心引擎
        self._axioms = Axioms()
        self._frame = FrameEngine()
        self._domain = DomainBridge()
        self._reason = ReasoningEngine()

        # 当前图状态
        self._neighbors: Dict[str, Set[str]] = defaultdict(set)
        self._edge_count: int = 0

    # ============================================================
    # 属性: 引擎访问
    # ============================================================

    @property
    def axioms(self) -> Axioms:
        """公理引擎 — 5 条核心公理 + 派生操作。"""
        return self._axioms

    @property
    def frame(self) -> FrameEngine:
        """帧引擎 — 演化帧 + 推理帧。"""
        return self._frame

    @property
    def domain(self) -> DomainBridge:
        """跨域桥接 — 7 领域概念映射。"""
        return self._domain

    @property
    def reason(self) -> ReasoningEngine:
        """推理引擎 — 推导链 + 评审 + 多路径锁定。"""
        return self._reason

    @property
    def constants(self) -> Constants:
        """SPUM 派生常数。"""
        return self._axioms.constants

    # ============================================================
    # 公理 1: 关系第一性 — 图操作
    # ============================================================

    def add_edge(self, u: str, v: str) -> dict:
        """添加一条边（公理1: 关系定义存在）。

        Args:
            u: 节点名
            v: 节点名

        Returns:
            {"event": "V+", "u": u, "v": v, "new_nodes": [...]}
        """
        new_nodes = []
        for node in (u, v):
            if node not in self._neighbors:
                new_nodes.append(node)

        self._axioms.axiom1.add_edge(u, v, self._neighbors, [self._edge_count])
        self._edge_count = sum(len(adj) for adj in self._neighbors.values()) // 2

        return {"event": "V+", "u": u, "v": v, "new_nodes": new_nodes}

    def remove_edge(self, u: str, v: str) -> dict:
        """删除一条边。

        Returns:
            {"event": "V-", "u": u, "v": v, "removed_nodes": [...]}
        """
        self._axioms.axiom1.remove_edge(u, v, self._neighbors, [self._edge_count])
        self._edge_count = sum(len(adj) for adj in self._neighbors.values()) // 2

        removed = []
        for node in (u, v):
            if node not in self._neighbors:
                removed.append(node)

        return {"event": "V-", "u": u, "v": v, "removed_nodes": removed}

    def degree(self, node: str) -> int:
        """节点度数。"""
        return self._axioms.axiom1.degree(node, self._neighbors)

    def has_node(self, node: str) -> bool:
        """节点是否存在。"""
        return self._axioms.axiom1.has_node(node, self._neighbors)

    @property
    def nodes(self) -> List[str]:
        """所有活跃节点列表。"""
        return list(self._axioms.axiom1.nodes(self._neighbors))

    @property
    def node_count(self) -> int:
        """活跃节点数。"""
        return len(self._neighbors)

    @property
    def edge_count(self) -> int:
        """边数。"""
        return self._edge_count

    # ============================================================
    # 公理 2 and 5: 悬挂端与不完美
    # ============================================================

    @property
    def dangling_nodes(self) -> List[str]:
        """悬挂端节点列表（度数 < 2）。"""
        return sorted(self._axioms.axiom2.dangling_nodes(self._neighbors))

    @property
    def dangling_count(self) -> int:
        """悬挂端数量。"""
        return len(self.dangling_nodes)

    @property
    def delta(self) -> float:
        """悬挂端密度 delta。"""
        return self._axioms.axiom2.dangling_density(self._neighbors)

    def is_dangling(self, node: str) -> bool:
        """节点是否为悬挂端。"""
        return self._axioms.axiom2.is_dangling(node, self._neighbors)

    @property
    def is_closed(self) -> bool:
        """当前图是否闭合（无悬挂端）。"""
        return self.dangling_count == 0

    def check_imperfection(self) -> dict:
        """检查边缘不完美（公理5）。"""
        return self._axioms.axiom5.check_edge_imperfection(
            self._neighbors, self._edge_count
        )

    # ============================================================
    # 公理 4: 拓扑守恒
    # ============================================================

    @property
    def handshaking(self) -> dict:
        """握手引理验证: Sigma deg(v) = 2|E|。"""
        return self._axioms.axiom4.handshaking(self._neighbors, self._edge_count)

    @property
    def angle_deficit_sum(self) -> dict:
        """角度亏损总和: Sigma(6-deg(v))。球面闭合子图应为 12。"""
        return self._axioms.axiom4.angle_deficit_sum(self._neighbors)

    @property
    def sigma(self) -> float:
        """空间密度 sigma = |P|/|E|。"""
        return self._axioms.sigma(self.node_count, self.edge_count)

    @property
    def avg_degree(self) -> float:
        """平均度数 <deg> = 2/sigma。"""
        return self._axioms.avg_degree(self.node_count, self.edge_count)

    @property
    def opening_ratio(self) -> float:
        """开口占比 = |悬挂端|/|节点|。应 <= 1/3。"""
        return self._axioms.axiom4.opening_ratio(self.dangling_count, self.node_count)

    @property
    def opening_ratio_satisfied(self) -> bool:
        """开口占比是否满足 <= 1/3。"""
        return self._axioms.axiom4.opening_ratio_satisfied(self.opening_ratio)

    # ============================================================
    # 公理 3: 帧演化
    # ============================================================

    def new_frame(self) -> int:
        """开始新演化帧。返回当前帧号。"""
        self._frame.new_evolution_frame()
        return self._frame.frame_id

    def v_plus(self, u: str, v: str) -> dict:
        """在当前帧中执行 V+ 操作。"""
        result = self._frame.v_plus(u, v)
        # 同步到主图
        self._axioms.axiom1.add_edge(u, v, self._neighbors, [self._edge_count])
        self._edge_count = sum(len(adj) for adj in self._neighbors.values()) // 2
        return result

    def v_minus(self, u: str, v: str) -> dict:
        """在当前帧中执行 V- 操作。"""
        result = self._frame.v_minus(u, v)
        # 同步到主图
        self._axioms.axiom1.remove_edge(u, v, self._neighbors, [self._edge_count])
        self._edge_count = sum(len(adj) for adj in self._neighbors.values()) // 2
        return result

    def commit_frame(self) -> dict:
        """结束当前帧。返回帧快照。"""
        frame_state = self._frame.commit()
        return frame_state.to_dict()

    @property
    def frame_summary(self) -> str:
        """帧序列摘要。"""
        return self._frame.summary()

    @property
    def last_frame(self) -> Optional[dict]:
        """最近一帧的快照。"""
        f = self._frame.last_frame()
        return f.to_dict() if f else None

    # ============================================================
    # 跨域概念映射
    # ============================================================

    def map_concept(self, domain: str, concept: str) -> Optional[dict]:
        """正向映射: 领域概念 -> SPUM 归约。

        Args:
            domain: 领域名
            concept: 概念名

        Returns:
            dict with spum_reduction, axioms, formula or None
        """
        result = self._domain.map(domain, concept)
        if result is None:
            return None
        return {
            "domain": result.domain,
            "concept": result.concept,
            "spum_reduction": result.spum_reduction,
            "axioms": result.axioms,
            "formula": result.formula,
            "path": result.path,
            "derived_from": result.derived_from,
        }

    def reverse_map(self, spum_primitive: str) -> List[dict]:
        """反向映射: SPUM 原语 -> 所有使用了它的领域概念。"""
        return [
            {"domain": m.domain, "concept": m.concept,
             "spum_reduction": m.spum_reduction[:100]}
            for m in self._domain.reverse_map(spum_primitive)
        ]

    def search_concepts(self, keyword: str) -> List[dict]:
        """全文搜索所有概念映射。"""
        return [
            {"domain": m.domain, "concept": m.concept}
            for m in self._domain.search(keyword)
        ]

    def list_domains(self) -> List[str]:
        """列出所有领域。"""
        return self._domain.list_domains()

    def list_concepts(self, domain: str) -> List[str]:
        """列出某领域的所有概念。"""
        return self._domain.list_concepts(domain)

    def cross_domain_chain(self, concept1: str, concept2: str) -> Optional[dict]:
        """寻找两个概念间通过 <P, E> 的联系路径。"""
        return self._domain.cross_domain_chain(concept1, concept2)

    @property
    def domain_summary(self) -> str:
        """跨域桥接汇总。"""
        return self._domain.summary()

    # ============================================================
    # 推理
    # ============================================================

    def derive(self, source: str, target: str,
               max_depth: int = 10) -> dict:
        """推导: 从 source 到 target 的推理路径。

        Args:
            source: 起始节点 ID (如 "N005")
            target: 目标节点 ID (如 "N013")
            max_depth: 最大搜索深度

        Returns:
            dict with path, steps, is_complete
        """
        chain = self._reason.derive(source, target, max_depth)
        return chain.to_dict()

    def all_paths(self, source: str, target: str,
                  max_paths: int = 3) -> List[dict]:
        """多路径推导: 找到所有独立路径。"""
        chains = self._reason.all_paths(source, target, max_paths)
        return [c.to_dict() for c in chains]

    def multi_path_validate(self, conclusion: str,
                            paths: Dict[str, bool]) -> dict:
        """多路径锁定验证。仅当所有路径都确认时结论被锁定。

        Args:
            conclusion: 待验证的结论
            paths: {路径名: 是否通过}

        Returns:
            dict with confidence, locked, paths
        """
        result = self._reason.multi_path_validate(conclusion, paths)
        return {
            "conclusion": result.conclusion,
            "paths": result.paths,
            "confidence": result.confidence,
            "locked": result.locked,
            "independent_paths": result.independent_paths,
            "confirmed_paths": result.confirmed_paths,
        }

    def review(self, text: str = "",
               A: int = 85, B: int = 80, C: int = 85, D: int = 80,
               LE: int = 0) -> dict:
        """执行四维评审。若提供 text 则自动评分。

        Args:
            text: 待评审文本（可选，提供后自动估算分数）
            A/B/C/D: 手动分数（text为空时使用）
            LE: 梯子依赖

        Returns:
            dict with A/B/C/D/LE scores, verdict, violations
        """
        if text:
            result = self._reason.auto_review_from_text(text)
        else:
            result = self._reason.review(A=A, B=B, C=C, D=D, LE=LE)
        return {
            "A": result.A,
            "B": result.B,
            "C": result.C,
            "D": result.D,
            "LE": result.LE,
            "verdict": result.verdict().value,
            "report": result.format_report(),
            "violations": [{"severity": v.severity, "dimension": v.dimension,
                            "description": v.description}
                           for v in result.violations],
        }

    def forward_chain(self, premises: List[str],
                      goal: str = "") -> dict:
        """前向推理: 从前提出发，沿推导边推出新结论。"""
        return self._reason.forward_chain(premises, goal)

    def backward_chain(self, goal: str) -> dict:
        """后向推理: 从目标反向搜索所需前提。"""
        return self._reason.backward_chain(goal)

    def query_node(self, node_id: str) -> Optional[dict]:
        """查询核心知识图谱中的节点定义和连接。"""
        return self._reason.query_node(node_id)

    def query_edge(self, source: str, target: str) -> Optional[dict]:
        """查询两个核心节点之间的推导边。"""
        return self._reason.query_edge(source, target)

    def verify_derivation(self) -> dict:
        """验证知识图谱推导完整性。"""
        return self._reason.verify_derivation_completeness()

    # ============================================================
    # 信息
    # ============================================================

    def summary(self) -> str:
        """SPUM 引擎整体状态摘要。"""
        try:
            from .reason import CORE_EDGES as ce
            n_edges = len(ce)
        except Exception:
            n_edges = 0
        return (
            f"SPUM API — AI 之家\n"
            f"====================\n"
            f"公理: 5 条核心公理 + 5 条派生操作\n"
            f"常数: kappa={self.constants.KAPPA}, tau={self.constants.TAU}, "
            f"c={self.constants.C}\n"
            f"图状态: {self.node_count} 节点, {self.edge_count} 边, "
            f"delta={self.delta:.4f}, sigma={self.sigma:.4f}\n"
            f"知识图谱: 22 核心节点 + {n_edges} 推导边\n"
            f"跨域: {len(self._domain.list_domains())} 领域\n"
            f"帧序列: 当前帧 {self._frame.frame_id}\n"
            f"闭合: {'是' if self.is_closed else '否'}"
        )

    def __repr__(self) -> str:
        return (f"SPUM(nodes={self.node_count}, edges={self.edge_count}, "
                f"sigma={self.sigma:.3f}, delta={self.delta:.3f})")
