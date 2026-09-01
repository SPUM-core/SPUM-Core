"""
Φ 可计算形式 — 关系闭环度计算器（L0.5 元语法层）

SPUM_认知投影论.md 第七章 Φ = (C_自洽 · S_稳态) / R_冗余 的可执行实现。

逻辑树拆解（模拟 `逻辑树.md` 的分层归约）：

    Φ（范式闭环比）
    ├── C_自洽（自洽度 = 1 − |MFAS| / |ε_c|）
    │   ├── 输入：核心概念弧集 ε_c（有向：概念→被依赖概念）
    │   ├── 计算：MFAS 最小反馈弧集（NP-hard → 确定性贪婪环分解近似）
    │   └── 语义：矛盾环（循环论证）的最低删除成本
    ├── S_稳态（稳态度 = 回归计数比例）
    │   ├── 输入：扰动集 D = {删去任意一条核心边}
    │   ├── 计算：确定性回归计数（删边后矛盾环结构不变 ⟹ 扰动被吸收）
    │   └── 语义：删掉一条边不破坏矛盾环结构的比例 = 抗干扰强度
    └── R_冗余（冗余度 = 补丁边数）
        ├── 输入：范式补丁记录（本轮、例外假设、特设边）
        ├── 计算：补丁边计数
        └── 语义：不完美定理保证 R ≥ 1，Φ 恒为正数无除零

借鉴 spum-review.md / src/core/review.py 的评分机制：
    - PhiReport.format_report()：与 REVIEW REPORT 兼容的维度化报告
    - ClosureScore.level()：§7.3 四层级评级（阈值判定，阈值可配置）
    - 违规明细（矛盾环、非环边、补丁）逐条列出，可审计

术语纪律：本模块的"概率/比例"均为有限样本上的计数比例，
不是概率本体化（spum-anti-pattern 模式 4）。

用法：
    # 模式 A：自动计算（给有向弧集 + 补丁边集）
    from spum_graph.closure import ClosureScore
    score = ClosureScore(
        arcs={("A","B"),("B","C"),("C","A"),("C","D")},
        patch_edges={"E_ad_hoc"},
    )
    print(score.report.format_report())

    # 模式 B：直接输入三参量（§7.5 演示模式）
    from spum_graph.closure import PhiReport
    report = PhiReport(C=0.85, S=0.6, R=12, epsilon_size=40, mfas_size=6)
    print(report.format_report())
"""

import os
import random
import warnings

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ── 基础图工具 ────────────────────────────────────────────────────────

Arc = Tuple[str, str]  # (u, v)：有向弧 u → v（u 依赖/蕴含 v）


def _adjacency(arcs: Set[Arc]) -> Dict[str, Set[str]]:
    """弧集 → 邻接表 {u: {v}}。"""
    adj: Dict[str, Set[str]] = {}
    for u, v in arcs:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set())
    return adj


def find_directed_cycle(arcs: Set[Arc]) -> Optional[List[str]]:
    """在弧集中找任意一个有向环（DFS），无环返回 None。

    确定性：按字典序遍历，结果可复现。返回环的顶点序列（含首尾闭合）。
    """
    adj = _adjacency(arcs)
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in adj}
    stack: List[Tuple[str, Optional[str]]] = []  # (顶点, 父顶点)
    parent: Dict[str, str] = {}

    for start in sorted(adj):
        if color[start] != WHITE:
            continue
        color[start] = GRAY
        stack.append((start, None))
        while stack:
            u, _ = stack[-1]
            advanced = False
            for v in sorted(adj[u]):
                if color[v] == GRAY:
                    # 找到环：从 u 沿父链回溯到 v
                    cycle = [v]
                    cur = u
                    while cur != v:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(v)
                    return cycle[::-1]
                if color[v] == WHITE:
                    color[v] = GRAY
                    parent[v] = u
                    stack.append((v, u))
                    advanced = True
                    break
            if not advanced:
                color[u] = BLACK
                stack.pop()
    return None


def greedy_mfas(arcs: Set[Arc]) -> Set[Arc]:
    """最小反馈弧集（MFAS）贪婪近似——矛盾环的最低删除成本。

    NP-hard（Karp 1972）→ 确定性启发式：反复找一个有向环，
    在环上"入度最大的顶点"的候选入边中删除字典序最小的一条
    （启发式：优先消除承载最多入矛盾的顶点）。结果**跨进程可复现**
    （不依赖 set 遍历顺序，见 victim 选择）；不承诺全局最优，
    承诺"删除后无矛盾环"（有效集）。

    确定性保证：find_directed_cycle 按字典序遍历；indeg 为纯计数
    （与遍历顺序无关）；target 以 (indeg, 顶点) 字典序 tiebreak；
    victim 从候选集取 min —— 四个环节均不依赖字符串哈希序。
    """
    remaining = set(arcs)
    fas: Set[Arc] = set()

    while True:
        cycle = find_directed_cycle(remaining)
        if cycle is None:
            break
        # 环内入度（仅统计两端都在环上的边，与剩余弧集遍历顺序无关）
        indeg = {n: 0 for n in cycle}
        for u, v in remaining:
            if u in indeg and v in indeg:
                indeg[v] += 1
        target = max(indeg, key=lambda n: (indeg[n], n))  # 入度最大的顶点
        # 候选 = target 在环上的全部入边（环保证非空）；min → 字典序确定
        candidates = [(u, v) for u, v in remaining
                      if v == target and u in cycle]
        victim = min(candidates)
        fas.add(victim)
        remaining.discard(victim)

    return fas


# ── MFAS 误差自检（7.1：随机重启采样）──────────────────────────────

def greedy_mfas_randomized(arcs: Set[Arc], rng: "random.Random") -> Set[Arc]:
    """随机化变体：每轮从当前环上随机删除一条边。

    与 greedy_mfas 的差异仅在选边策略——保留"反复找环并删除"的骨架，
    选边由 rng 决定。用于 estimate_mfas 的多重采样。
    """
    remaining = set(arcs)
    fas: Set[Arc] = set()

    while True:
        cycle = find_directed_cycle(remaining)
        if cycle is None:
            break
        # 环上边：顶点序列的相邻对（首尾闭合，全部是实际边）
        edges_on_cycle = [
            (cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)
        ]
        victim = edges_on_cycle[rng.randrange(len(edges_on_cycle))]
        fas.add(victim)
        remaining.discard(victim)

    return fas


@dataclass
class MfasEstimate:
    """MFAS 近似的误差自检结果。

    随机重启采样的 size 分布：最优 / 平均 / 最差 / 标准差。
    术语纪律：该分布是**近似精度的认知投影**（对全局最优解的不完备
    信息），不是本体概率——不改变"删除后无矛盾环"的确定性质（反模式 4）。
    """
    restarts: int
    sizes: List[int]
    best: Set[Arc]
    seed: int = 0
    deterministic_size: Optional[int] = None

    @property
    def best_size(self) -> int:
        return min(self.sizes)

    @property
    def avg_size(self) -> float:
        return sum(self.sizes) / len(self.sizes)

    @property
    def worst_size(self) -> int:
        return max(self.sizes)

    @property
    def std(self) -> float:
        """样本标准差（n−1）。n < 2 时无定义 → 0.0。"""
        n = len(self.sizes)
        if n < 2:
            return 0.0
        avg = self.avg_size
        return (sum((s - avg) ** 2 for s in self.sizes) / (n - 1)) ** 0.5

    def format_report(self) -> str:
        lines = [
            "MFAS ESTIMATE（随机重启采样）",
            f"确定性贪婪：{self.deterministic_size} 条" if self.deterministic_size is not None else "确定性贪婪：—",
            f"采样（{self.restarts} 重启, seed={self.seed}）："
            f"最优 {self.best_size} | 平均 {self.avg_size:.2f} | "
            f"最差 {self.worst_size} | 标准差 {self.std:.2f}",
        ]
        if self.deterministic_size is not None and self.best_size <= self.deterministic_size:
            lines.append("结论：确定性结果 ≥ 采样最优——近似精度可信（采样区间覆盖）")
        return "\n".join(lines)


def estimate_mfas(
    arcs: Set[Arc],
    restarts: int = 32,
    seed: int = 0,
) -> MfasEstimate:
    """MFAS 误差自检：多重随机重启采样，报告最优/平均/最差/标准差。

    仍不保证全局最优（NP-hard），但给出近似精度的置信区间：
        - best_size ≤ 确定性 greedy_mfas 的 size 时，精度可信；
        - std 小说明近似对选边策略不敏感（稳定）；
        - std 大说明图结构对删边选择敏感，应增大 restarts 或改用
          显式人工标注的 MFAS（ClosureScore 的 mfas 参数）。

    seed 固定保证可复现——同图同 seed 必得同分布。
    """
    rng = random.Random(seed)
    sizes: List[int] = []
    best: Optional[Set[Arc]] = None
    for _ in range(restarts):
        fas = greedy_mfas_randomized(arcs, rng)
        sizes.append(len(fas))
        if best is None or len(fas) < len(best):
            best = fas
    return MfasEstimate(
        restarts=restarts,
        sizes=sizes,
        best=best or set(),
        seed=seed,
        deterministic_size=len(greedy_mfas(arcs)),
    )


def _scc_tarjan(arcs: Set[Arc]) -> Dict[str, int]:
    """Tarjan 强连通分量 → {顶点: 分量ID}。

    迭代实现（显式工作栈，避免深链图触发递归限制）。
    确定性：按字典序启动 DFS，分量 ID 按完成顺序分配，结果可复现。
    """
    adj = _adjacency(arcs)
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    comp: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    counter = 0
    comp_id = 0

    for start in sorted(adj):
        if start in index:
            continue
        work: List[Tuple[str, int]] = [(start, 0)]  # (顶点, 下一邻接下标)
        index[start] = lowlink[start] = counter
        counter += 1
        stack.append(start)
        on_stack.add(start)
        while work:
            v, i = work[-1]
            neighbors = sorted(adj.get(v, ()))
            if i < len(neighbors):
                w = neighbors[i]
                work[-1] = (v, i + 1)
                if w not in index:
                    index[w] = lowlink[w] = counter
                    counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, 0))
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])
            else:
                # 子树处理完毕：根节点出栈成一个分量
                if lowlink[v] == index[v]:
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp[w] = comp_id
                        if w == v:
                            break
                    comp_id += 1
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[v])
    return comp


def cyclic_edges(arcs: Set[Arc]) -> Set[Arc]:
    """位于至少一个有向环上的边集。

    判定：e=(u,v) 位于某环上 ⟺ u、v 属于同一强连通分量（SCC）——
    e 自身给出 u→v，v 可达 u 则闭合为环（含自环 u→u）。
    一次 Tarjan SCC 线性扫描 O(V+E) 得到全部环边，替代逐边 BFS 的
    O(E·(V+E))，为 F.4 规划的"批量 Φ 排名"铺路。
    确定性：SCC 划分与遍历顺序无关，结果可复现。
    """
    comp = _scc_tarjan(arcs)
    return {e for e in arcs if comp.get(e[0]) == comp.get(e[1])}


# ── 三参量 ────────────────────────────────────────────────────────────

def self_consistency(arcs: Set[Arc], mfas: Optional[Set[Arc]] = None) -> float:
    """自洽度 C = 1 − |MFAS| / |ε_c|。

    未提供 MFAS 时自动调用 greedy_mfas 近似。
    C ∈ [0, 1]；C = 1 ⟺ 无矛盾环。
    空集返回 1.0（平凡自洽）——但"空集/单环 → C=S=1 无代表力"的
    空洞自洽防护由 ClosureScore 自动模式负责（见其 _compute_auto）。
    """
    n = len(arcs)
    if n == 0:
        return 1.0
    fas = mfas if mfas is not None else greedy_mfas(arcs)
    return 1.0 - len(fas) / n


def steadiness(arcs: Set[Arc]) -> float:
    """稳态度 S = 非矛盾环边比例（单边扰动回归计数比例）。

    ⚠ 代理口径声明（与定义 7.3 动力学语义的差距，评审点 1.2）：
    定义 7.3 的 S 是动力学语义——扰动 d ∈ D 后经权重演化 F: W→W
    的确定性回归计数；本实现是**静态图代理**：D = {删去任意一条
    核心边 e}，回归判定退化为"删 e 后矛盾环结构不变 ⟺ e 不位于
    任何有向环上"，故 S = |{e ∈ ε_c : e 非环边}| / |ε_c|。

    代理的性质与局限（须在解读 Φ 时知悉）：
    - **S ≤ C 恒成立**：MFAS ⊆ 环边集 ⟹ |MFAS| ≤ |cyclic_edges|
      ⟹ C ≥ S。Φ = (C·S)/R 的分子是对同一环结构信号的两次计数，
      信息冗余——"三参量独立判据"的宣称在本口径下被削弱；
    - 无环图 S=1 是"无环可破坏"的平凡值，不是动力学意义上的
      "一切扰动均被吸引子吸收"。
    真实动力学回归（权重演化采样）见 F.4 路线图，尚未实现。
    """
    n = len(arcs)
    if n == 0:
        return 1.0
    return 1.0 - len(cyclic_edges(arcs)) / n


def redundancy(patch_edges: Set[Arc]) -> int:
    """冗余度 R = |补丁边集|。

    补丁边 = 本轮、例外假设、特设边。不完美定理（N013）保证
    任何认知范式每帧必残留补丁 → R ≥ 1 → Φ 无除零。
    R = 0 的范式不存在（"完全自洽无例外"是旧范式的幻想）——
    补丁边集为空时显式拒绝，避免除零。
    """
    r = len(patch_edges)
    if r == 0:
        raise ValueError(
            "不完美定理（N013）禁止 R=0：任何认知范式每帧必残留补丁，"
            "补丁边集不能为空（'完全自洽无例外'的完美体系不存在）"
        )
    return r


# ── Φ 聚合与评级（借鉴 spum-review 评分机制）────────────────────────

@dataclass
class PhiConfig:
    """四层级评级阈值（演示校准值，可按知识域调整）。"""
    PHI_OPEN: float = 0.02    # Φ < 此值 → 开放雏形 / 半闭环叙事
    PHI_MATURE: float = 0.5   # Φ ≥ 此值 → 成熟体系候选
    R_HIGH: int = 20          # 补丁冗余爆炸阈值
    R_LOW: int = 5            # 成熟体系的补丁上限


@dataclass
class PhiReport:
    """Φ 判据维度化报告（兼容 REVIEW REPORT 风格）。

    借鉴 src/core/review.py 的 ReviewResult 设计：
    A/B/C/D/LE 维度报告 + 阈值判定 + 违规明细。
    对应 Φ 的 C/S/R 三参量 + 明细。
    """
    C: float              # 自洽度
    S: float              # 稳态度
    R: int                # 冗余度（补丁边数）
    epsilon_size: int = 0  # 核心弧集大小 |ε_c|
    mfas_size: int = 0     # |MFAS|
    cyclic_count: int = 0  # 矛盾环边数
    patch_edges: List[Arc] = field(default_factory=list)
    config: PhiConfig = field(default_factory=PhiConfig)

    def __post_init__(self) -> None:
        """参量范围校验（模式 A 自动计算必在范围内；模式 B 人工输入可越界）。"""
        if not 0.0 <= self.C <= 1.0:
            raise ValueError(
                f"C 自洽度必须 ∈ [0, 1]：收到 {self.C}。"
                "C = 1 − |MFAS|/|ε_c| 是比例，越界即输入错误。"
            )
        if not 0.0 <= self.S <= 1.0:
            raise ValueError(
                f"S 稳态度必须 ∈ [0, 1]：收到 {self.S}。"
                "S 是确定性计数比例，越界即输入错误。"
            )
        if self.R < 1:
            raise ValueError(
                f"不完美定理（N013）禁止 R < 1：收到 R={self.R}。"
                "任何认知范式每帧必残留补丁，'完全自洽无例外'的体系不存在。"
            )

    @property
    def phi(self) -> float:
        """关系闭环度 Φ = (C · S) / R。R ≥ 1（不完美定理）→ 无除零。"""
        if self.R < 1:  # 双保险：构造期已校验，此处防字段被直接修改
            raise ValueError(
                f"不完美定理（N013）禁止 R < 1：收到 R={self.R}。"
                "任何认知范式每帧必残留补丁，'完全自洽无例外'的体系不存在。"
            )
        return (self.C * self.S) / self.R

    def level(self) -> str:
        """§7.3 四层级评级（阈值判定，借鉴 review.py 的 verdict() 结构）。"""
        cfg = self.config
        phi = self.phi
        if self.R >= cfg.R_HIGH:              # 冗余爆炸
            return "开放雏形" if phi < cfg.PHI_OPEN else "半闭环叙事"
        if phi < cfg.PHI_OPEN:
            return "开放雏形"
        if phi < 0.1:
            return "半闭环叙事"
        if phi >= cfg.PHI_MATURE and self.R <= cfg.R_LOW:
            return "成熟体系"
        return "准自洽理论"

    def _level_note(self) -> str:
        notes = {
            "开放雏形": "C 低、S 低、R 极高 — Φ → 0",
            "半闭环叙事": "局部自洽、矛盾环多、S 低、R 高 — Φ 低",
            "准自洽理论": "高 C、中等 S、R 中 — Φ 中高",
            "成熟体系": "高 C、高 S、低 R — Φ 最大化",
        }
        return notes.get(self.level(), "")

    def format_report(self) -> str:
        """生成与 spum-review 兼容的维度化报告。"""
        lines = [
            "CLOSURE REPORT 关系闭环度 Φ",
            f"C 自洽度：{self.C:.3f}/1.000",
            f"S 稳态度：{self.S:.3f}/1.000",
            f"R 冗余度：{self.R} 条补丁",
            f"Φ = (C·S)/R：{self.phi:.4f}",
            f"判定：{self.level()}（{self._level_note()}）",
        ]
        if self.epsilon_size:
            lines.append(
                f"明细：|ε_c|={self.epsilon_size}  |MFAS|={self.mfas_size}"
                f"  矛盾环边={self.cyclic_count}"
            )
        if self.patch_edges:
            # 补丁可表示为概念弧 (u,v) 或单概念名——统一安全格式化
            def _fmt(patch):
                if isinstance(patch, tuple) and len(patch) == 2:
                    return f"{patch[0]}→{patch[1]}"
                return str(patch)

            shown = ", ".join(_fmt(p) for p in list(self.patch_edges)[:6])
            tail = "…" if len(self.patch_edges) > 6 else ""
            lines.append(f"补丁明细：{shown}{tail}")
        return "\n".join(lines)


@dataclass
class ClosureScore:
    """Φ 计算器：从核心弧集 + 补丁边集自动计算三参量。

    模式 A（自动）：ClosureScore(arcs=..., patch_edges=...)
    模式 B（直接）：ClosureScore(C=0.85, S=0.6, R=12, epsilon_size=40, mfas_size=6)
    """
    arcs: Optional[Set[Arc]] = None
    patch_edges: Optional[Set[Arc]] = None
    mfas: Optional[Set[Arc]] = None      # 可选：显式提供 MFAS（人工标注）
    C: Optional[float] = None            # 可选：直接给参量（跳过自动计算）
    S: Optional[float] = None
    R: Optional[int] = None
    epsilon_size: int = 0
    mfas_size: int = 0
    config: PhiConfig = field(default_factory=PhiConfig)

    def __post_init__(self) -> None:
        if self.C is None or self.S is None or self.R is None:
            if self.arcs is None:
                raise ValueError("自动模式需提供 arcs；或直接提供 C/S/R")
            self._compute_auto()
        self.report = PhiReport(
            C=self.C, S=self.S, R=self.R,
            epsilon_size=self.epsilon_size, mfas_size=self.mfas_size,
            cyclic_count=len(cyclic_edges(self.arcs or set())),
            patch_edges=sorted(self.patch_edges or set()),
            config=self.config,
        )

    def _compute_auto(self) -> None:
        arcs = set(self.arcs or set())
        if not arcs:
            raise ValueError(
                "自动模式收到空弧集：C=S=1 的'空洞自洽'无代表力（§7.5.1 退化防护）。"
                "请提供非空核心弧集，或改用直接参量模式（C/S/R）。"
            )
        self.epsilon_size = len(arcs)
        fas = self.mfas if self.mfas is not None else greedy_mfas(arcs)
        self.mfas_size = len(fas)
        self.C = self_consistency(arcs, fas)
        self.S = steadiness(arcs)
        self.R = redundancy(self.patch_edges or set())

    def format_report(self) -> str:
        return self.report.format_report()


# ── 知识图谱抽取（7.2：network/nodes.txt + edges.txt → Arc 集）─────

def parse_arc_lines(lines) -> Set[Arc]:
    """解析知识图谱边行 → 弧集。

    格式（与 network/edges.txt 一致）：
        `A | B | type | 描述`（可省略 type/描述，至少 2 列）
    跳过空行与 `#` 注释行；格式异常的行（少于 2 列或端点为空）跳过
    并发出 UserWarning——静默丢弃会掩盖数据问题。方向保留为源 → 目标。
    """
    arcs: Set[Arc] = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        u, v = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
        if u and v:
            arcs.add((u, v))
        else:
            warnings.warn(
                f"跳过无法解析的边行：{line!r}（需 ≥2 列且源/目标非空）",
                UserWarning,
                stacklevel=2,
            )
    return arcs


def load_knowledge_edges(path) -> Set[Arc]:
    """从知识图谱边文件读取推导弧集（UTF-8）。"""
    with open(path, encoding="utf-8") as f:
        return parse_arc_lines(f)


# ── 代表性边集自举（COG-006 / §7.5.1）──────────────────────────────

def _deg_in(arcs: Set[Arc], node: str) -> int:
    return sum(1 for u, v in arcs if u == node or v == node)


def _covers(arcs: Set[Arc], cover_nodes: Set[str], min_degree: int) -> bool:
    """覆盖约束：每个 cover 节点在 ε 中至少出现 min_degree 次。"""
    for node in cover_nodes:
        if _deg_in(arcs, node) < min_degree:
            return False
    return True


def _greedy_coverage_start(
    candidates: Set[Arc], cover_nodes: Set[str], min_degree: int
) -> Set[Arc]:
    """构造满足覆盖约束的初始骨架（集合覆盖近似）。

    每轮选"覆盖最多未达标节点"的边，直至全部 cover 节点达标或候选耗尽。
    """
    selected: Set[Arc] = set()
    while True:
        unmet = {n for n in cover_nodes if _deg_in(selected, n) < min_degree}
        if not unmet:
            break
        best_edge, best_gain = None, -1
        for e in sorted(candidates - selected):
            gain = sum(1 for n in (e[0], e[1]) if n in unmet)
            if gain > best_gain:
                best_gain, best_edge = gain, e
        if best_edge is None or best_gain <= 0:
            break
        selected.add(best_edge)
    return selected


@dataclass
class BootstrapResult:
    """COG-006 代表性边集自举结果。"""
    best_arcs: Set[Arc]
    coverage: Set[str]
    missing: Set[str]
    phi: float
    C: float
    S: float
    R: int
    evaluated: int
    exhaustive: bool

    def format_report(self) -> str:
        lines = [
            "BOOTSTRAP REPORT 代表性边集自举（COG-006）",
            f"评估 {self.evaluated} 次（{'穷举' if self.exhaustive else '贪心+局部搜索'}）",
            f"覆盖 {len(self.coverage)} 节点 / 缺失 {sorted(self.missing) or '—'}",
            f"ε* = {len(self.best_arcs)} 条边  Φ={self.phi:.4f}"
            f"（C={self.C:.3f}, S={self.S:.3f}, R={self.R}）",
        ]
        return "\n".join(lines)


def bootstrap_selection(
    candidates: Set[Arc],
    cover_nodes: Set[str],
    patch_edges: Optional[Set] = None,
    config: Optional[PhiConfig] = None,
    exhaustive: bool = False,
    min_degree: int = 2,
) -> BootstrapResult:
    """代表性边集自举（§7.5.1 / COG-006）。

    ε* = argmax Φ(ε)  s.t.  ε 覆盖 cover_nodes（每节点 deg ≥ min_degree）

    - exhaustive=True：穷举全部子集（仅小候选集可用，评估 2^|S| 次）；
    - exhaustive=False（默认）：集合覆盖骨架起步 + 贪心局部搜索；
    - patch_edges 缺省时自动补一个占位补丁，保证 R ≥ 1（不完美定理 N013）。

    术语纪律：Φ 在此处是序数（比较准则），自举是固定点而非循环论证；
    不承诺全局最优（NP-hard 变体），穷举模式仅在测试/小图上提供精确基准。
    """
    cfg = config or PhiConfig()
    patches = set(patch_edges) if patch_edges is not None else {"_bootstrap_patch_"}
    candidates = set(candidates)
    evaluated = 0

    def phi_of(arcs: Set[Arc]):
        nonlocal evaluated
        evaluated += 1
        if not arcs:
            return 0.0, 1.0, 1.0, len(patches)
        report = PhiReport(
            C=self_consistency(arcs),
            S=steadiness(arcs),
            R=redundancy(patches),
            cyclic_count=len(cyclic_edges(arcs)),
            config=cfg,
        )
        return report.phi, report.C, report.S, report.R

    if exhaustive:
        best_arcs: Set[Arc] = set()
        # 软约束：优先最大化达标节点数，其次最大化 Φ。覆盖约束不可达
        # （如链图 A 只能度 1）时仍返回最佳尽力集，missing 如实报告——
        # 与贪心模式一致，保证 Φ > 0（不完美定理 N013，R ≥ 1 强制）。
        # 空集默认 (Φ=0, C=S=1)，R 由补丁决定。
        best_key = (0, 0.0)  # (达标节点数, Φ)
        best_score = (0.0, 1.0, 1.0, len(patches))
        cand_list = sorted(candidates)
        n = len(cand_list)
        for mask in range(1 << n):
            arcs = {cand_list[i] for i in range(n) if (mask >> i) & 1}
            score = phi_of(arcs)
            sat = sum(1 for node in cover_nodes
                      if _deg_in(arcs, node) >= min_degree)
            if (sat, score[0]) > best_key:
                best_key = (sat, score[0])
                best_score, best_arcs = score, arcs
        phi, C, S, R = best_score
    else:
        arcs = _greedy_coverage_start(candidates, cover_nodes, min_degree)
        current = phi_of(arcs)
        improved = True
        while improved:
            improved = False
            best_add, best_score = None, None
            for e in sorted(candidates - arcs):
                score = phi_of(arcs | {e})
                if best_score is None or score[0] > best_score[0]:
                    best_add, best_score = e, score
            if best_score is not None and best_score[0] > current[0]:
                arcs = arcs | {best_add}
                current = best_score
                improved = True
        phi, C, S, R = current
        best_arcs = arcs

    covered = {v for e in best_arcs for v in e}
    # missing = 未达骨架度（min_degree）的 cover 节点——覆盖但度不足也算缺失
    missing = {n for n in cover_nodes if _deg_in(best_arcs, n) < min_degree}
    return BootstrapResult(
        best_arcs=best_arcs,
        coverage=covered,
        missing=missing,
        phi=phi, C=C, S=S, R=R,
        evaluated=evaluated,
        exhaustive=exhaustive,
    )


# ── 演示：§7.5 三个真实知识体系 ──────────────────────────────────────

DEMO_SYSTEMS = {
    "托勒密地心说（经院后期）": dict(C=0.850, S=0.6, R=12, epsilon_size=40, mfas_size=6),
    "哥白尼日心说（早期）":     dict(C=0.914, S=0.5, R=4, epsilon_size=35, mfas_size=3),
    "牛顿力学（19世纪末）":     dict(C=0.980, S=0.9, R=3, epsilon_size=50, mfas_size=1),
    "广义相对论（1916后）":     dict(C=0.982, S=0.95, R=1, epsilon_size=55, mfas_size=1),
    "古典体液说（晚期）":       dict(C=0.867, S=0.4, R=8, epsilon_size=30, mfas_size=4),
    "现代病原微生物学":         dict(C=0.967, S=0.9, R=2, epsilon_size=60, mfas_size=2),
}


def run_demo(edges_path: Optional[str] = None) -> str:
    """演示输出。

    段一（模式 B）：§7.5 六体系**人工估算快照**——复现文章表格数值，
    非自动测量（F.4 方法声明：真实计算需完整边集与 MFAS 近似）。
    段二（模式 A，可选）：传入 edges_path 时从知识图谱自动抽取弧集
    → 自动计算三参量，端到端打通 F.4 "以测量为准" 的承诺——
    演示由此从"公式复现器"升级为"独立验证器"。
    """
    out = ["=" * 60, "Φ 可计算形式 — 演示", "=" * 60,
           "§7.5 六体系（模式 B：人工估算快照，非自动测量）"]
    for name, params in DEMO_SYSTEMS.items():
        rep = PhiReport(
            C=params["C"], S=params["S"], R=params["R"],
            epsilon_size=params["epsilon_size"],
            mfas_size=params["mfas_size"],
        )
        out.append("")
        out.append(f"◆ {name}")
        out.append(rep.format_report())
    if edges_path is not None and os.path.exists(edges_path):
        arcs = load_knowledge_edges(edges_path)
        out.append("")
        out.append("=" * 60)
        out.append("知识图谱实况（模式 A：自动抽取 → 自动计算）")
        out.append("=" * 60)
        whole = ClosureScore(arcs=arcs, patch_edges={"（图谱临时补丁）"})
        out.append(whole.format_report())
        out.append("")
        out.append("模块子图 Φ（按节点前缀划分，自动计算）：")
        for prefix, label in (("COG", "COG 认知投影论"), ("N", "N 核心推导链")):
            sub = {e for e in arcs
                   if e[0].startswith(prefix) and e[1].startswith(prefix)}
            if not sub:
                continue
            ssub = ClosureScore(arcs=sub, patch_edges={"（子图补丁）"})
            out.append(
                f"  {label}：{len(sub)} 边  Φ={ssub.report.phi:.4f}"
                f"（C={ssub.report.C:.3f}, S={ssub.report.S:.3f}, R={ssub.report.R}）"
            )
    return "\n".join(out)


# ── 跃迁动力学预报接口占位（F.4 路线图：R(t) > R_crit）─────────────

def forecast_transition(
    patch_history: List[int],
    r_crit: Optional[int] = None,
) -> bool:
    """R(t) 跃迁预报（F.4 路线图占位接口）。

    输入：补丁冗余的帧序列 R(t0..tn)（每帧末的补丁边计数）；
    输出：是否预报下一帧 R 突破 r_crit（缺省取 PhiConfig.R_HIGH = 20）。

    当前为**占位实现**——线性外推末帧趋势（ΔR = R(tn) − R(tn−1)）；
    未接入权重动力学采样（F: W→W）的真实跃迁预报。动力学采样与
    R_crit 判据见 SPUM_认知投影论.md F.4（"补丁冗余 R(t) 的时序采样，
    接入跃迁动力学预报"），本函数使该路线图在代码中可见、可演进。
    """
    if len(patch_history) < 2:
        return False
    drift = patch_history[-1] - patch_history[-2]
    return patch_history[-1] + drift > (
        r_crit if r_crit is not None else PhiConfig().R_HIGH
    )


if __name__ == "__main__":
    print(run_demo())
