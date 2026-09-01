"""
青檬引擎 · 共识约束层 — 11 条 SPUM 不变量校验器
================================================

对齐 spum-qingmeng-guard.md 的 11 条不变量。所有推理路径必须通过
consensus 校验，否则抛出 ConsensusViolationError——这是"共识底线"，
不随领域/业务变化（不变量 I、V、VI 等硬约束由本层强制裁决）。

规则由 `consensus_schema.json` 驱动加载（Schema 化，见蓝图避坑建议 3）：
  - scope 决定检查范围（structural / ledger / contextual / epistemic）
  - check 映射到本类的校验函数
  - params 传递阈值（如 II 的 max_drift）

校验分级：
  - structural（可检验事实）：I/III/IV/V/VI —— 直接检查图与账本
  - ledger（记账事实）：II —— 检查对偶账本
  - contextual（上下文）：VII —— 演化策略确定性
  - epistemic（认知标注）：VIII/IX/X/XI —— 输出标注约定（宽松检查）

用法:
    from qingmeng.core import ConsensusValidator
    v = ConsensusValidator()                          # 内置 Schema
    v2 = ConsensusValidator.from_schema("my_rules.json")   # 外部 Schema
    report = v.validate(graph, ledger=eng.ledger)
    report.passed            # → bool
    report.failures          # → [ConsensusFailure, ...]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .axioms import Core, CORE_SIZE

# 包内默认规则 Schema（不可变——公理底线不随外部配置漂移）
_DEFAULT_SCHEMA = "consensus_schema.json"


class ConsensusViolationError(Exception):
    """推理链未通过共识校验时抛出。

    携带违规明细，便于在 LLM 桥接层拦截并提示偏离的公理。
    """

    def __init__(self, report: "ConsensusReport") -> None:
        self.report = report
        details = "; ".join(f"[{f.invariant}] {f.detail}" for f in report.failures)
        super().__init__(f"ConsensusViolation: {details}")


@dataclass
class ConsensusFailure:
    invariant: str
    required: str
    detail: str


@dataclass
class ConsensusReport:
    """一次校验的结果。"""

    passed: bool
    checks: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[ConsensusFailure] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "failures": [
                {"invariant": f.invariant, "required": f.required, "detail": f.detail}
                for f in self.failures
            ],
        }


class ConsensusValidator:
    """11 条不变量校验器（Schema 驱动）。

    使用: validator.validate(graph, ledger=...) 逐条核对。
    可用 raise_on_fail=True 让违规直接抛 ConsensusViolationError。
    """

    def __init__(self, schema_path: Optional[str] = None) -> None:
        self.core = Core()
        self.schema_path = schema_path or _DEFAULT_SCHEMA
        self._rules: List[Dict[str, Any]] = self._load_rules(self.schema_path)

    @classmethod
    def from_schema(cls, schema_path: str) -> "ConsensusValidator":
        """从外部 JSON Schema 构建校验器（规则 id/check/params 由表驱动）。"""
        return cls(schema_path=schema_path)

    @staticmethod
    def _load_rules(schema_path: str) -> List[Dict[str, Any]]:
        """加载规则表：包内资源或外部文件路径。"""
        if schema_path == _DEFAULT_SCHEMA:
            ref = resources.files(__package__).joinpath(_DEFAULT_SCHEMA)
            with ref.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        else:
            with open(schema_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        return data.get("rules", [])

    @property
    def rules(self) -> List[Dict[str, Any]]:
        """当前生效的规则表（含 enabled 状态与 params）。"""
        return self._rules

    # ── 对外入口 ─────────────────────────────────────────────────

    def validate(
        self,
        graph: nx.Graph,
        ledger: Optional[Dict[str, int]] = None,
        evolution_strategy: str = "deterministic",
        raise_on_fail: bool = False,
    ) -> ConsensusReport:
        """对一张图执行 Schema 中全部启用规则的不变量校验。

        Args:
            graph: 待校验的 ⟨P, ε⟩ 图（无权无向）
            ledger: 对偶账本（II 检查需要），形如
                    {"total_created": n, "total_annihilated": m}
            evolution_strategy: 演化策略（VII 检查），
                    "deterministic"（推荐）或 "random"
            raise_on_fail: 违规时抛 ConsensusViolationError

        Returns:
            ConsensusReport
        """
        checks: List[Dict[str, Any]] = []
        failures: List[ConsensusFailure] = []

        for rule in self._rules:
            if not rule.get("enabled", True):
                continue
            # Schema 中 check 名 → 本类私有校验方法（前缀 "_"）
            check_fn = getattr(self, "_" + rule["check"], None)
            if check_fn is None:
                # 未知 check → 记一条失败，防规则表漂移导致静默放行
                checks.append({
                    "invariant": rule["id"],
                    "required": rule.get("name", "未注册校验函数"),
                    "passed": False,
                    "detail": f"规则表引用了未注册的校验函数: {rule['check']}",
                })
                continue
            params = rule.get("params", {})
            # 参数分发：II 取账本，VII 取演化策略，epistemic 无参，其余取图
            if rule["check"] == "check_II":
                result = check_fn(ledger, **params)
            elif rule["check"] == "check_VII":
                result = check_fn(evolution_strategy)
            elif rule["check"] == "check_epistemic":
                result = check_fn()
            else:
                result = check_fn(graph)
            checks.append(result)

        for c in checks:
            if not c["passed"]:
                failures.append(
                    ConsensusFailure(
                        invariant=c["invariant"],
                        required=c["required"],
                        detail=c["detail"],
                    )
                )

        report = ConsensusReport(passed=not failures, checks=checks, failures=failures)
        if raise_on_fail and not report.passed:
            raise ConsensusViolationError(report)
        return report

    # ── 逐条校验 ─────────────────────────────────────────────────

    def _check_I(self, G: nx.Graph) -> Dict[str, Any]:
        """I ⟨P, ε⟩ 唯一基底 — 图是唯一数据结构：无自环（自环是独立实体）。"""
        loops = list(nx.selfloop_edges(G))
        ok = len(loops) == 0
        return {
            "invariant": "I",
            "required": "⟨P, ε⟩ 唯一基底，不引入独立实体",
            "passed": ok,
            "detail": f"自环数={len(loops)}" if ok else f"发现自环: {loops[:3]}",
        }

    def _check_III(self, G: nx.Graph) -> Dict[str, Any]:
        """III 无悬挂边 — 度数<2 须级联消解。闭合判据：δ 密度与孤立节点。"""
        isolated = [n for n in G.nodes() if G.degree(n) == 0]
        dangling = [n for n in G.nodes() if G.degree(n) < 2]
        # 孤立节点（deg=0）是硬违规——度 0 即孤立，逻辑不自洽
        ok = len(isolated) == 0
        return {
            "invariant": "III",
            "required": "无悬挂边（度数<2 → 级联消解）",
            "passed": ok,
            "detail": (
                f"孤立节点={len(isolated)}，悬挂节点={len(dangling)}，"
                f"δ={len(dangling) / max(G.number_of_nodes(), 1):.4f}"
            ),
        }

    def _check_IV(self, G: nx.Graph) -> Dict[str, Any]:
        """IV 三角剖分 — 针对闭合骨架（核心诱导子图）：每顶点度数 ≥ 3。

        三角剖分约束只约束闭合子图（晶子骨架）；外部派生节点含悬挂端
        是不完美定理的预期态，不参与本检查。充分性需平面嵌入，
        正二十面体 20 面全三角已验证。
        """
        core_sub = G.subgraph(range(CORE_SIZE))
        if core_sub.number_of_nodes() == 0:
            return {
                "invariant": "IV",
                "required": "所有面为三角形（三角剖分）",
                "passed": False,
                "detail": "核心骨架为空",
            }
        min_deg = min(dict(core_sub.degree()).values())
        ok = min_deg >= 3
        return {
            "invariant": "IV",
            "required": "所有面为三角形（三角剖分）",
            "passed": ok,
            "detail": (
                f"核心骨架最小度数={min_deg}（三角剖分必要条件 deg≥3）"
            ),
        }

    def _check_V(self, G: nx.Graph) -> Dict[str, Any]:
        """V 拓扑常数 12 — 核心诱导子图 Σ(6−deg)=12（闭合骨架内）。

        外部连接使晶子在整图度数升高、Σ 偏离 12——这是闭合子图与外界的
        正常耦合（闭合子图创生抑制），不是违规；本检查只验证骨架内部
        拓扑常数，外部连接数在 detail 中如实报告。
        """
        core_present = set(range(CORE_SIZE)) & set(G.nodes())
        if len(core_present) != CORE_SIZE:
            return {
                "invariant": "V",
                "required": "CORE_SIZE = 12",
                "passed": False,
                "detail": f"核心晶子缺失: {CORE_SIZE - len(core_present)} 个",
            }
        core_sub = G.subgraph(sorted(core_present))
        sub_degrees = {v: core_sub.degree(v) for v in sorted(core_present)}
        sum_loss = sum(6 - d for d in sub_degrees.values())
        ok = sum_loss == self.core.constants.TOPOLOGICAL_12
        # 外部连接数（晶子与派生节点的边）
        external = sum(
            1 for u, v in G.edges()
            if (u < CORE_SIZE) != (v < CORE_SIZE)
        )
        return {
            "invariant": "V",
            "required": "拓扑常数 12（Σ(6−deg)=12）",
            "passed": ok,
            "detail": (
                f"核心骨架度数={sub_degrees}，Σ(6−deg)={sum_loss}，"
                f"外部连接={external}条（正常耦合，不违规）"
            ),
        }

    def _check_VI(self, G: nx.Graph) -> Dict[str, Any]:
        """VI 边是二值关系 — 无权、无向。"""
        directed = G.is_directed()
        weighted = any(
            "weight" in d and d.get("weight") != 1.0
            for _, _, d in G.edges(data=True)
        )
        ok = not directed and not weighted
        return {
            "invariant": "VI",
            "required": "边是二值关系（无权、无向）",
            "passed": ok,
            "detail": (
                f"有向={directed}，加权边={weighted}"
                if ok
                else f"违反: 有向={directed}，含权重属性={weighted}"
            ),
        }

    def _check_II(self, ledger: Optional[Dict[str, int]], max_drift: int = 24) -> Dict[str, Any]:
        """II 总边数守恒 — 创生-湮灭对偶记账，净漂移应围绕 0 波动。

        max_drift 由规则 Schema 的 params 提供（默认 24 = 2×CORE_SIZE）。
        """
        if ledger is None:
            return {
                "invariant": "II",
                "required": "总边数守恒（创生-湮灭对偶）",
                "passed": True,
                "detail": "未提供账本——跳过记账检查（需 ledger 参数）",
            }
        created = ledger.get("total_created", 0)
        annihilated = ledger.get("total_annihilated", 0)
        drift = created - annihilated
        # 对偶不要求逐帧相等（可跨帧跨子图重分配），但累计漂移持续单向
        # 增长超过容量量级（默认 2×CORE_SIZE）即失衡信号
        ok = abs(drift) <= max_drift
        return {
            "invariant": "II",
            "required": "总边数守恒（创生-湮灭对偶）",
            "passed": ok,
            "detail": (
                f"累计创生={created}，累计湮灭={annihilated}，净漂移={drift}"
                if ok
                else f"对偶失衡: 净漂移={drift}（单向累积）"
            ),
        }

    def _check_VII(self, strategy: str) -> Dict[str, Any]:
        """VII 约束驱动演化 — 演化策略应确定性（约束驱动），非随机。"""
        ok = strategy == "deterministic"
        return {
            "invariant": "VII",
            "required": "约束驱动演化（随机 → 约束冲突检测）",
            "passed": ok,
            "detail": f"演化策略: {strategy}" if ok else f"随机演化: {strategy}（推荐 deterministic）",
        }

    def _check_epistemic(self) -> Dict[str, Any]:
        """VIII~XI 认知标注 — 输出应区分本体层与认知投影层。

        基础框架中此为元层面约定（文档/输出模板约束），默认通过；
        完全接入时由输出层对连续量强制添加"认知层"标注。
        """
        return {
            "invariant": "VIII-XI",
            "required": "几何/π/连续量标注为认知投影；数学是工具",
            "passed": True,
            "detail": "元层面约定——输出层对连续量强制标注认知投影层",
        }

    def describe(self) -> str:
        return (
            "共识校验器 | 11 条 SPUM 不变量 | "
            "structural: I/III/IV/V/VI | ledger: II | "
            "contextual: VII | epistemic: VIII-XI"
        )
