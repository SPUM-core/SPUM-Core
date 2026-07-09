"""
SPUM 推理引擎 — 推导链 + 多路径锁定 + 评审
=============================================

AI 自主推演的核心工具:
    1. 推导链 (DerivationChain): 从源节点到目标节点的推理路径
    2. 多路径锁定 (MultiPathLock): 从多个独立路径验证同一结论
    3. 评审 (Review): A/B/C/D/LE 四维评分
    4. 前向/后向推理 (Forward/Backward Chaining)

用法:
    from spum.reason import ReasoningEngine
    engine = ReasoningEngine()

    # 推导
    chain = engine.derive("N005", "N006")
    print(chain.path)

    # 多路径锁定
    result = engine.multi_path_validate(
        conclusion="星系旋转曲线可以不用暗物质解释",
        paths=["子图协动拟合", "σ梯度预测", "边缘泄露衰减"],
    )

    # 评审
    review = engine.review(A=85, B=72, C=90, D=78, LE=1)
    print(review.verdict)

    # 自主推理
    result = engine.forward_chain(
        premises=["N001 宇宙存在", "N002 差异第一性"],
        goal="推导引力本质",
    )
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict

# 依赖本包的其他模块
from .axioms import Axioms
from .domain import DomainBridge


# ══════════════════════════════════════════════════════════════════════
# 知识图谱节点定义（内联，不依赖 network/nodes.txt 文件）
# ══════════════════════════════════════════════════════════════════════

# SPUM 核心知识图谱: 22 个核心节点定义
CORE_NODES: Dict[str, str] = {
    "N001": "宇宙存在",
    "N002": "差异",
    "N003": "边界",
    "N004": "节点/空间粒子(κ‑particle)",
    "N005": "边/连接",
    "N006": "闭合网络",
    "N007": "最小度数≥2",
    "N008": "总边数守恒",
    "N009": "创生事件 V⁺",
    "N010": "体积/存在权重",
    "N011": "悬挂边删除",
    "N012": "离散帧",
    "N013": "不完美",
    "N014": "晶子(Crystallite)",
    "N015": "分形约束生长",
    "N016": "拓扑常数12",
    "N017": "π的降级",
    "N018": "永恒粒子",
    "N019": "几何发生学",
    "N020": "运动与引力",
    "N021": "拓扑认知",
    "N022": "SPUM-图论",
}

# 核心推导边 (source, target, type, description)
CORE_EDGES: List[Tuple[str, str, str, str]] = [
    ("N001", "N002", "requires", "宇宙存在需要差异"),
    ("N002", "N003", "derives_from", "差异产生边界"),
    ("N003", "N004", "derives_from", "边界定义节点/空间粒子"),
    ("N004", "N005", "derives_from", "节点产生边/连接"),
    ("N005", "N006", "derives_from", "连接形成闭合网络"),
    ("N006", "N007", "requires", "闭合网络要求最小度数≥2"),
    ("N007", "N008", "derives_from", "度数约束导出总边数守恒"),
    ("N009", "N008", "explains", "创生事件受守恒约束"),
    ("N004", "N010", "derives_from", "节点度数定义体积/存在权重"),
    ("N011", "N013", "drives", "悬挂边删除驱动不完美"),
    ("N012", "N013", "drives", "离散帧结构导致不完美残留"),
    ("N013", "N009", "drives", "不完美驱动下一帧创生事件"),
    ("N014", "N015", "refines", "晶子通过分形约束生长"),
    ("N015", "N016", "derives_from", "分形约束导出拓扑常数12"),
    ("N016", "N017", "explains", "拓扑常数12解释π的降级"),
    ("N014", "N018", "derives_from", "晶子组合形成永恒粒子"),
    ("N018", "N020", "explains", "永恒粒子解释运动与引力"),
    ("N015", "N020", "explains", "分形生长与引力"),
    ("N010", "N020", "explains", "体积权重与引力"),
    ("N004", "N019", "derives_from", "空间粒子定义几何发生学"),
    ("N005", "N021", "derives_from", "连接定义拓扑认知"),
    ("N007", "N021", "refines", "最小度数解释锚点稳定性"),
    ("N011", "N021", "refines", "悬挂边删除在认知中的表现"),
    ("N013", "N021", "refines", "不完美在认知中的表现"),
    ("N020", "N021", "refines", "运动对认知的影响"),
    ("N005", "N022", "derives_from", "边定义SPUM-图论"),
    ("N012", "N022", "refines", "离散帧定义帧快照"),
    ("N008", "N022", "refines", "总边数守恒定义完美极限"),
    ("N013", "N022", "refines", "不完美定义悬挂端不可消除"),
]


# ══════════════════════════════════════════════════════════════════════
# 推导链
# ══════════════════════════════════════════════════════════════════════

@dataclass
class DerivationStep:
    """推导步骤。"""
    source: str
    target: str
    edge_type: str
    description: str
    axioms_used: List[str] = field(default_factory=list)


@dataclass
class DerivationChain:
    """完整推导链。"""
    source: str
    target: str
    steps: List[DerivationStep] = field(default_factory=list)
    is_complete: bool = False
    axioms_used: Set[str] = field(default_factory=set)

    @property
    def path_str(self) -> str:
        if not self.steps:
            return f"{self.source} → (无路径) → {self.target}"
        parts = [self.steps[0].source]
        for step in self.steps:
            parts.append(f" --[{step.edge_type}]--> {step.target}")
        return "".join(parts)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "is_complete": self.is_complete,
            "path": self.path_str,
            "steps": [
                {"source": s.source, "target": s.target,
                 "edge_type": s.edge_type, "description": s.description}
                for s in self.steps
            ],
            "axioms_used": sorted(self.axioms_used),
        }


# ══════════════════════════════════════════════════════════════════════
# 多路径锁定结果
# ══════════════════════════════════════════════════════════════════════

@dataclass
class MultiPathResult:
    """多路径验证结果。"""
    conclusion: str
    paths: Dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.0
    locked: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def independent_paths(self) -> int:
        return len(self.paths)

    @property
    def confirmed_paths(self) -> int:
        return sum(1 for v in self.paths.values() if v)


# ══════════════════════════════════════════════════════════════════════
# 评审系统
# ══════════════════════════════════════════════════════════════════════

class ReviewVerdict(Enum):
    PASS = "通过"
    WARN = "警告"
    ROLLBACK = "回滚"


@dataclass
class ReviewViolation:
    """单条违规记录。"""
    severity: str  # FATAL/SEVERE/WARNING/REMINDER
    dimension: str  # A/B/C/D
    description: str
    location: str = ""


@dataclass
class ReviewResult:
    """四维评审结果。

    A 架构纯净度: SPUM 范式一致性
    B 推导严密性: 逻辑链完整性
    C 归约完备度: 是否归约到 ⟨P, ε⟩
    D 方法论透明度: 认知层级标记
    LE 梯子依赖: 0(无) ~ 3(严重)
    """
    A: int  # 0-100
    B: int  # 0-100
    C: int  # 0-100
    D: int  # 0-100
    LE: int  # 0-3
    violations: List[ReviewViolation] = field(default_factory=list)
    frame_id: int = 0

    def verdict(self) -> ReviewVerdict:
        dims = {"A": self.A, "B": self.B, "C": self.C, "D": self.D}
        for dim, score in dims.items():
            if score < 50:
                return ReviewVerdict.ROLLBACK
        if self.LE >= 2:
            return ReviewVerdict.ROLLBACK
        for dim, score in dims.items():
            if score < 70:
                return ReviewVerdict.WARN
        return ReviewVerdict.PASS

    def failing_dimensions(self) -> List[str]:
        result = []
        for dim, score in [("A", self.A), ("B", self.B),
                           ("C", self.C), ("D", self.D)]:
            if score < 50:
                result.append(dim)
        if self.LE >= 2:
            result.append("LE")
        return result

    def format_report(self) -> str:
        n = self.frame_id
        v = self.verdict()
        lines = [
            f"REVIEW REPORT 帧{n}" if n else "REVIEW REPORT",
            f"A 架构纯净度：{self.A}/100",
            f"B 推导严密性：{self.B}/100",
            f"C 归约完备度：{self.C}/100",
            f"D 方法论透明度：{self.D}/100",
            f"LE 梯子依赖：{self.LE}",
            f"判定：{v.value}",
        ]
        if self.violations:
            for vl in self.violations[:5]:
                lines.append(f"  [{vl.severity}] [{vl.dimension}] {vl.description}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# 推理引擎
# ══════════════════════════════════════════════════════════════════════

class ReasoningEngine:
    """SPUM 推理引擎 — 推导链 + 多路径锁定 + 评审 + 自主推理。

    用法:
        engine = ReasoningEngine()

        # 推导链
        chain = engine.derive("N005", "N013")

        # 多路径锁定
        result = engine.multi_path_validate(
            conclusion="X",
            paths={"路径A": True, "路径B": True},
        )

        # 评审
        review = engine.review(A=85, B=72, C=90, D=78, LE=1)

        # 自主推理
        result = engine.forward_chain(["N001"], "推导新结论")
    """

    def __init__(self):
        self._axioms = Axioms()
        self._domain = DomainBridge()

        # 构建邻接表
        self._graph: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        for s, t, et, desc in CORE_EDGES:
            self._graph[s].add((t, et, desc))
            self._graph[t].add((s, "inverse_of", f"反向: {desc}"))

    # ── 属性 ───────────────────────────────────────────────────

    @property
    def nodes(self) -> Dict[str, str]:
        return dict(CORE_NODES)

    @property
    def edges(self) -> List[Tuple[str, str, str, str]]:
        return list(CORE_EDGES)

    @property
    def axioms(self) -> Axioms:
        return self._axioms

    @property
    def domain(self) -> DomainBridge:
        return self._domain

    # ── 推导链 ─────────────────────────────────────────────────

    def derive(self, source: str, target: str,
               max_depth: int = 10) -> DerivationChain:
        """BFS 搜索从 source 到 target 的推导路径。

        Args:
            source: 起始节点 ID (如 "N005")
            target: 目标节点 ID (如 "N013")
            max_depth: 最大搜索深度

        Returns:
            DerivationChain: 包含推导路径
        """
        if source not in CORE_NODES:
            return DerivationChain(source=source, target=target,
                                   is_complete=False,
                                   steps=[])
        if target not in CORE_NODES:
            return DerivationChain(source=source, target=target,
                                   is_complete=False,
                                   steps=[])

        # BFS
        visited = {source}
        queue = [(source, [])]  # (node, path_steps)

        while queue:
            current, path = queue.pop(0)
            if current == target:
                axioms_used = set()
                for step in path:
                    if step.edge_type == "derives_from":
                        axioms_used.add("axiom1")
                    elif step.edge_type == "requires":
                        axioms_used.add("axiom4")
                    elif step.edge_type == "drives":
                        axioms_used.add("axiom5")
                    elif step.edge_type == "refines":
                        axioms_used.add("axiom3")
                    elif step.edge_type == "explains":
                        axioms_used.add("derived.sigma")
                return DerivationChain(
                    source=source, target=target,
                    steps=path, is_complete=True,
                    axioms_used=axioms_used,
                )

            if len(path) >= max_depth:
                continue

            for neighbor, edge_type, desc in self._graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    step = DerivationStep(
                        source=current, target=neighbor,
                        edge_type=edge_type, description=desc,
                    )
                    queue.append((neighbor, path + [step]))

        return DerivationChain(source=source, target=target,
                               is_complete=False,
                               steps=[])

    def all_paths(self, source: str, target: str,
                  max_paths: int = 3,
                  max_depth: int = 8) -> List[DerivationChain]:
        """找到从 source 到 target 的所有独立路径。

        Args:
            source: 起始节点
            target: 目标节点
            max_paths: 最多返回的路径数
            max_depth: 最大深度

        Returns:
            推导链列表（按长度排序）
        """
        if source not in CORE_NODES or target not in CORE_NODES:
            return []

        results = []
        # DFS 搜索所有路径
        def dfs(current: str, target: str, visited: Set[str],
                path: List[DerivationStep], depth: int):
            if len(results) >= max_paths:
                return
            if depth > max_depth:
                return
            if current == target:
                axioms_used = set()
                for step in path:
                    if step.edge_type == "derives_from":
                        axioms_used.add("axiom1")
                    elif step.edge_type == "requires":
                        axioms_used.add("axiom4")
                    elif step.edge_type == "drives":
                        axioms_used.add("axiom5")
                results.append(DerivationChain(
                    source=source, target=target,
                    steps=list(path), is_complete=True,
                    axioms_used=axioms_used,
                ))
                return

            for neighbor, edge_type, desc in self._graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    step = DerivationStep(
                        source=current, target=neighbor,
                        edge_type=edge_type, description=desc,
                    )
                    path.append(step)
                    dfs(neighbor, target, visited, path, depth + 1)
                    path.pop()
                    visited.discard(neighbor)

        dfs(source, target, {source}, [], 0)
        results.sort(key=lambda c: len(c.steps))
        return results

    # ── 多路径锁定 ─────────────────────────────────────────────

    def multi_path_validate(
        self, conclusion: str,
        paths: Dict[str, bool],
        notes: Optional[List[str]] = None,
    ) -> MultiPathResult:
        """多路径锁定验证。

        从多个独立路径验证同一结论。
        仅当所有路径都确认时，结论被"锁定"。

        Args:
            conclusion: 待验证的结论
            paths: {路径名: 是否通过} 字典
            notes: 额外备注

        Returns:
            MultiPathResult
        """
        total = len(paths)
        confirmed = sum(1 for v in paths.values() if v)
        locked = total >= 2 and confirmed == total
        confidence = confirmed / max(total, 1)

        return MultiPathResult(
            conclusion=conclusion,
            paths=paths,
            confidence=confidence,
            locked=locked,
            notes=notes or [],
        )

    # ── 评审 ───────────────────────────────────────────────────

    def review(self, A: int = 85, B: int = 80, C: int = 85, D: int = 80,
               LE: int = 0, violations: Optional[List[ReviewViolation]] = None,
               frame_id: int = 0) -> ReviewResult:
        """执行四维评审。

        参数同 ReviewResult。
        """
        return ReviewResult(
            A=A, B=B, C=C, D=D, LE=LE,
            violations=violations or [],
            frame_id=frame_id,
        )

    def auto_review_from_text(self, text: str) -> ReviewResult:
        """从文本自动估算评审分数（启发式）。

        根据文本特征自动评分:
        - 是否使用 ⟨P,ε⟩/σ/V⁺ 等原语
        - 是否有推导路径
        - 是否有旧范式词汇（"力""能量""场"）
        - 是否有认知层级标记

        Args:
            text: 待评审查的文本

        Returns:
            ReviewResult
        """
        score = {"A": 50, "B": 50, "C": 50, "D": 50, "LE": 2}
        violations = []

        # A 架构纯净度
        spum_terms = ["⟨P, ε⟩", "σ", "V⁺", "V⁻", "δ", "Δμ",
                      "空间粒子", "帧", "悬挂", "拓扑"]
        spum_count = sum(1 for t in spum_terms if t in text)
        score["A"] = min(100, 50 + spum_count * 10)
        if spum_count < 3:
            violations.append(ReviewViolation(
                "WARNING", "A", f"SPUM原语使用较少({spum_count}/9)"))

        # B 推导严密性
        old_terms = ["力", "能量", "场", "连续", "弯曲时空", "大爆炸"]
        old_count = sum(1 for t in old_terms if t in text)
        score["B"] = max(0, min(100, 80 - old_count * 15))
        if old_count > 0:
            violations.append(ReviewViolation(
                "SEVERE", "B", f"使用了{old_count}个旧范式词汇: "
                f"{[t for t in old_terms if t in text]}"))

        # C 归约完备度
        has_reduction = any(t in text for t in ["= ε", "= ∇σ", "= V⁺",
                                                  "还原为", "归约"])
        if has_reduction:
            score["C"] = 70
        else:
            violations.append(ReviewViolation(
                "WARNING", "C", "未明确归约到 ⟨P, ε⟩"))

        # D 方法论透明度
        has_layer = any(t in text for t in ["L0", "L1", "认知投影", "本体"])
        score["D"] = 70 if has_layer else 50
        if not has_layer:
            violations.append(ReviewViolation(
                "WARNING", "D", "未标记认知层级"))

        # LE
        if old_count >= 3:
            score["LE"] = 3
        elif old_count >= 1:
            score["LE"] = 1
        else:
            score["LE"] = 0

        return ReviewResult(
            A=score["A"], B=score["B"], C=score["C"], D=score["D"],
            LE=score["LE"], violations=violations,
        )

    # ── 图查询 ─────────────────────────────────────────────────

    def query_node(self, node_id: str) -> Optional[dict]:
        """查询节点的定义和出边。

        Args:
            node_id: 节点 ID (如 "N005")

        Returns:
            {"id": ..., "name": ..., "connections": [...], "edges": [...]}
        """
        if node_id not in CORE_NODES:
            return None

        connections = []
        for neighbor, edge_type, desc in self._graph.get(node_id, set()):
            connections.append({
                "target": neighbor,
                "target_name": CORE_NODES.get(neighbor, neighbor),
                "edge_type": edge_type,
                "description": desc,
            })

        return {
            "id": node_id,
            "name": CORE_NODES[node_id],
            "connections": connections,
            "derivation_count": len(connections),
        }

    def query_edge(self, source: str, target: str) -> Optional[dict]:
        """查询两个节点之间的边。"""
        if source not in CORE_NODES or target not in CORE_NODES:
            return None
        for neighbor, edge_type, desc in self._graph.get(source, set()):
            if neighbor == target:
                return {
                    "source": source,
                    "source_name": CORE_NODES[source],
                    "target": target,
                    "target_name": CORE_NODES[target],
                    "edge_type": edge_type,
                    "description": desc,
                }
        return None

    def node_list(self) -> List[dict]:
        """返回所有节点列表。"""
        return [
            {"id": nid, "name": name}
            for nid, name in sorted(CORE_NODES.items())
        ]

    # ── 自主推理 ───────────────────────────────────────────────

    def forward_chain(self, premises: List[str],
                      goal: str = "",
                      max_steps: int = 10) -> dict:
        """前向推理: 从前提出发，沿推导边推出新结论。

        Args:
            premises: 初始前提列表（节点 ID 或陈述）
            goal: 推理目标描述（可选）
            max_steps: 最大推理步数

        Returns:
            {"derived": [...], "chain": [...], "goal_reached": bool}
        """
        derived = set()
        chain = []
        queue = list(premises)
        goal_reached = False

        for _ in range(max_steps):
            if not queue:
                break
            current = queue.pop(0)
            if current not in CORE_NODES:
                continue

            derived.add(current)

            for neighbor, edge_type, desc in self._graph.get(current, set()):
                if neighbor not in derived:
                    derived.add(neighbor)
                    queue.append(neighbor)
                    chain.append({
                        "from": current,
                        "from_name": CORE_NODES.get(current, current),
                        "to": neighbor,
                        "to_name": CORE_NODES.get(neighbor, neighbor),
                        "edge_type": edge_type,
                        "description": desc,
                    })
                    if goal and goal in CORE_NODES.get(neighbor, ""):
                        goal_reached = True

        return {
            "premises": premises,
            "goal": goal,
            "derived": sorted(derived),
            "derived_count": len(derived),
            "chain": chain,
            "chain_length": len(chain),
            "goal_reached": goal_reached,
        }

    def backward_chain(self, goal: str,
                       max_depth: int = 10) -> dict:
        """后向推理: 从目标反向搜索所需前提。

        Args:
            goal: 目标节点 ID 或关键概念
            max_depth: 最大深度

        Returns:
            {"goal": ..., "required_premises": [...], "chain": [...]}
        """
        # 找到匹配的节点
        target_node = None
        for nid, name in CORE_NODES.items():
            if goal in nid or goal in name:
                target_node = nid
                break

        if target_node is None:
            return {"goal": goal, "required_premises": [], "chain": [],
                    "note": "未在核心图谱中找到匹配节点"}

        # 反向 BFS: 寻找指向 target_node 的所有路径
        reverse_graph: Dict[str, Set[Tuple[str, str, str]]] = defaultdict(set)
        for s, t, et, desc in CORE_EDGES:
            reverse_graph[t].add((s, et, desc))

        premises = []
        chain = []
        visited = {target_node}
        queue = [(target_node, None, None, None)]  # (node, from_node, edge_type, desc)

        for _ in range(max_depth):
            if not queue:
                break
            current, from_n, et, desc = queue.pop(0)
            if from_n is not None:
                chain.insert(0, {
                    "from": from_n,
                    "from_name": CORE_NODES.get(from_n, from_n),
                    "to": current,
                    "to_name": CORE_NODES.get(current, current),
                    "edge_type": et,
                    "description": desc,
                })
                premises.append(from_n)

            for neighbor, edge_type, description in reverse_graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current, edge_type, description))

        return {
            "goal": goal,
            "target_node": target_node,
            "required_premises": premises,
            "premise_count": len(premises),
            "chain": chain,
            "chain_length": len(chain),
        }

    # ── 完整性检查 ─────────────────────────────────────────────

    def verify_derivation_completeness(self) -> dict:
        """检查知识图谱推导的完整性。

        返回:
            - 所有节点的入度/出度
            - 孤立节点（无入边也无出边）
            - 无法从 N001 到达的节点
        """
        out_degree = defaultdict(int)
        in_degree = defaultdict(int)
        for s, t, et, desc in CORE_EDGES:
            out_degree[s] += 1
            in_degree[t] += 1

        # 从 N001 BFS
        reachable = set()
        queue = ["N001"]
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            for n, et, d in self._graph.get(current, set()):
                if n not in reachable:
                    queue.append(n)

        all_nodes = set(CORE_NODES.keys())
        unreachable = all_nodes - reachable

        return {
            "total_nodes": len(CORE_NODES),
            "total_edges": len(CORE_EDGES),
            "nodes_without_incoming": [n for n in all_nodes if in_degree.get(n, 0) == 0],
            "nodes_without_outgoing": [n for n in all_nodes if out_degree.get(n, 0) == 0],
            "reachable_from_N001": len(reachable),
            "unreachable_from_N001": sorted(unreachable),
            "is_fully_connected": len(unreachable) == 0,
        }

    def summary(self) -> str:
        return (
            f"SPUM 推理引擎 | {len(CORE_NODES)} 节点 | {len(CORE_EDGES)} 推导边 | "
            f"22 核心节点 | 7 领域跨域桥接 | 四维评审 A/B/C/D/LE"
        )
